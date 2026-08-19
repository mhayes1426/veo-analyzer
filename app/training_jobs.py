from __future__ import annotations

import json
import importlib.util
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .annotations import annotation_frame_path, extract_annotation_frame
from .config import Settings
from .dataset import recording_split, yolo_label
from .db import Database
from .media import resolve_known_media


ALLOWED_MODELS = {"yolo11n.pt", "yolo11s.pt", "yolo11m.pt"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _metrics(result, prefix: str = "") -> dict[str, float]:
    values = getattr(result, "results_dict", {}) or {}
    metrics: dict[str, float] = {}
    for key, value in values.items():
        try:
            metrics[f"{prefix}{key}"] = float(value)
        except (TypeError, ValueError):
            continue
    return metrics


def training_runtime_available() -> bool:
    return importlib.util.find_spec("ultralytics") is not None


def launch_training(job_id: str, settings: Settings) -> None:
    job_dir = settings.config_root / "training" / "jobs" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    log = (job_dir / "training.log").open("ab", buffering=0)
    environment = os.environ.copy()
    environment["YOLO_CONFIG_DIR"] = str(settings.config_root / "ultralytics")
    subprocess.Popen(
        [sys.executable, "-m", "app.training_jobs", job_id],
        stdout=log,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
        env=environment,
    )


def _update(db: Database, job_id: str, **values) -> None:
    if not values:
        return
    assignments = ", ".join(f"{key}=?" for key in values)
    with db.transaction() as connection:
        connection.execute(
            f"UPDATE training_jobs SET {assignments}, updated_at=CURRENT_TIMESTAMP WHERE job_id=?",
            (*values.values(), job_id),
        )


def _reviewed_frames(db: Database) -> list[dict]:
    with db.connect() as connection:
        frames = connection.execute(
            """SELECT f.*, e.event_id, r.recording_id, r.media_path, r.size_bytes, r.mtime_ns, r.availability
               FROM annotation_frames f JOIN events e ON e.event_id=f.event_id
               JOIN recordings r ON r.recording_id=e.recording_id
               WHERE f.review_status='reviewed' AND r.availability='present'
               ORDER BY r.recording_id, f.event_id, f.frame_time_seconds"""
        ).fetchall()
        boxes = connection.execute("SELECT * FROM annotation_boxes ORDER BY frame_id, created_at").fetchall()
    by_frame: dict[str, list[dict]] = {}
    for box in boxes:
        by_frame.setdefault(box["frame_id"], []).append(dict(box))
    return [dict(row) | {"boxes": by_frame.get(row["frame_id"], [])} for row in frames]


def prepare_dataset(db: Database, settings: Settings, job_id: str) -> tuple[Path, dict[str, int]]:
    job_dir = settings.config_root / "training" / "jobs" / job_id
    dataset_dir = job_dir / "dataset"
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    for split in ("train", "val", "test"):
        (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    counts = {"train": 0, "val": 0, "test": 0}
    for row in _reviewed_frames(db):
        source = resolve_known_media(settings.media_root, row["media_path"])
        stat = source.stat()
        if stat.st_size != row["size_bytes"] or stat.st_mtime_ns != row["mtime_ns"]:
            continue
        cached = annotation_frame_path(settings.config_root, row["event_id"], row["frame_index"])
        if not cached.exists():
            extract_annotation_frame(source, cached, row["frame_time_seconds"])
        split = recording_split(row["recording_id"])
        stem = row["frame_id"]
        shutil.copy2(cached, dataset_dir / "images" / split / f"{stem}.jpg")
        labels = "\n".join(yolo_label(box) for box in row["boxes"])
        (dataset_dir / "labels" / split / f"{stem}.txt").write_text(labels + ("\n" if labels else ""))
        counts[split] += 1
    (dataset_dir / "data.yaml").write_text(
        f"path: {dataset_dir}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: basketball\n  1: hoop\n"
    )
    return dataset_dir / "data.yaml", counts


def run_job(job_id: str) -> None:
    settings = Settings.from_env()
    db = Database(settings.db_path)
    try:
        with db.connect() as connection:
            job = connection.execute("SELECT * FROM training_jobs WHERE job_id=?", (job_id,)).fetchone()
        if not job:
            raise RuntimeError("Training job was not found")
        _update(db, job_id, status="preparing", started_at=_now())
        data_yaml, counts = prepare_dataset(db, settings, job_id)
        total = sum(counts.values())
        _update(db, job_id, frame_count=total)
        if counts["train"] == 0 or counts["val"] == 0:
            raise RuntimeError(
                "Training requires reviewed frames in both train and validation splits. Annotate more recordings and try again."
            )

        from ultralytics import YOLO, settings as yolo_settings

        yolo_settings.update({"sync": False})
        model_path = Path("/opt/models") / job["model_name"]
        model = YOLO(str(model_path if model_path.exists() else job["model_name"]))

        def epoch_finished(trainer) -> None:
            current = int(trainer.epoch) + 1
            progress = min(99.0, current / int(job["epochs"]) * 100)
            _update(db, job_id, current_epoch=current, progress_percent=progress)

        model.add_callback("on_train_epoch_end", epoch_finished)
        _update(db, job_id, status="running")
        result = model.train(
            data=str(data_yaml),
            epochs=int(job["epochs"]),
            imgsz=int(job["image_size"]),
            batch=int(job["batch_size"]),
            device=job["device"],
            project=job["output_dir"],
            name="run",
            exist_ok=True,
            plots=True,
            verbose=True,
        )
        best = Path(job["output_dir"]) / "run" / "weights" / "best.pt"
        metrics = _metrics(result, "val/")
        if counts["test"] and best.exists():
            test_result = YOLO(str(best)).val(
                data=str(data_yaml), split="test", device=job["device"],
                project=job["output_dir"], name="test", exist_ok=True, plots=True,
            )
            metrics.update(_metrics(test_result, "test/"))
        _update(
            db,
            job_id,
            status="completed",
            progress_percent=100,
            current_epoch=int(job["epochs"]),
            model_path=str(best) if best.exists() else None,
            metrics_json=json.dumps(metrics),
            completed_at=_now(),
        )
    except Exception as exc:
        _update(db, job_id, status="failed", error_message=str(exc)[:1000], completed_at=_now())
        raise


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python -m app.training_jobs JOB_ID")
    run_job(sys.argv[1])
