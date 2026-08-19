from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .config import Settings
from .db import Database
from .media import resolve_known_media


class AnalysisCancelled(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def inference_runtime_available() -> bool:
    try:
        import importlib.util
        return importlib.util.find_spec("ultralytics") is not None
    except (ImportError, ValueError):
        return False


def launch_analysis(job_id: str, settings: Settings) -> None:
    job_dir = settings.config_root / "analysis" / "jobs" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    log = (job_dir / "analysis.log").open("ab", buffering=0)
    environment = os.environ.copy()
    environment["YOLO_CONFIG_DIR"] = str(settings.config_root / "ultralytics")
    subprocess.Popen(
        [sys.executable, "-m", "app.analysis_jobs", job_id], stdout=log, stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL, start_new_session=True, env=environment,
    )


def _update(db: Database, job_id: str, **values) -> None:
    assignments = ", ".join(f"{key}=?" for key in values)
    with db.transaction() as connection:
        connection.execute(
            f"UPDATE analysis_jobs SET {assignments}, updated_at=CURRENT_TIMESTAMP WHERE job_id=?",
            (*values.values(), job_id),
        )


def _cancelled(db: Database, job_id: str) -> bool:
    with db.connect() as connection:
        row = connection.execute(
            "SELECT status,cancel_requested FROM analysis_jobs WHERE job_id=?", (job_id,),
        ).fetchone()
    return not row or bool(row["cancel_requested"]) or row["status"] != "running"


def frame_times(start: float, end: float, interval: float) -> list[float]:
    count = int((end - start) / interval) + 1
    values = [round(start + index * interval, 3) for index in range(count)]
    if values[-1] < end - 0.001:
        values.append(round(end, 3))
    return values


def _extract_frame(source: Path, output: Path, time_seconds: float) -> None:
    subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{time_seconds:.3f}",
         "-i", str(source), "-frames:v", "1", "-vf", "scale=1280:-2", "-q:v", "3", str(output)],
        check=True, capture_output=True, timeout=90,
    )


def _extract_chunk(source: Path, directory: Path, start: float, end: float, interval: float) -> list[tuple[Path, float]]:
    directory.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{start:.3f}",
         "-t", f"{max(0.001, end - start):.3f}", "-i", str(source),
         "-vf", f"fps={1 / interval:.8f},scale=1280:-2", "-q:v", "3", str(directory / "raw-%06d.jpg")],
        check=True, capture_output=True, timeout=max(90, int(end - start) * 3),
    )
    paths = sorted(directory.glob("raw-*.jpg"))
    return [(path, round(start + index * interval, 3)) for index, path in enumerate(paths)]


def _detections(result) -> list[dict]:
    names = result.names
    detections = []
    if result.boxes is None:
        return detections
    for box in result.boxes:
        class_id = int(box.cls[0].item())
        xyxy = [float(value) for value in box.xyxyn[0].tolist()]
        detections.append({
            "class": str(names[class_id]), "confidence": float(box.conf[0].item()),
            "x1": xyxy[0], "y1": xyxy[1], "x2": xyxy[2], "y2": xyxy[3],
        })
    return detections


def _center(box: dict) -> tuple[float, float]:
    return (box["x1"] + box["x2"]) / 2, (box["y1"] + box["y2"]) / 2


def rim_observations(frames: list[dict]) -> list[dict]:
    observations = []
    for frame in frames:
        hoops = [box for box in frame["detections"] if box["class"] in {"hoop", "rim"}]
        balls = [box for box in frame["detections"] if box["class"] == "basketball"]
        for ball in balls:
            ball_x, ball_y = _center(ball)
            choices = []
            for hoop in hoops:
                hoop_x, hoop_y = _center(hoop)
                width = max(0.001, hoop["x2"] - hoop["x1"])
                height = max(0.001, hoop["y2"] - hoop["y1"])
                dx, dy = (ball_x - hoop_x) / width, (ball_y - hoop_y) / height
                if abs(dx) <= 1.5 and -5 <= dy <= 5:
                    choices.append((abs(dx) + abs(dy) * .08, hoop, dx, dy, hoop_x, hoop_y))
            if choices:
                _, hoop, dx, dy, hoop_x, hoop_y = min(choices, key=lambda item: item[0])
                observations.append({
                    "time_seconds": frame["time_seconds"], "dx_rim_widths": round(dx, 4),
                    "dy_rim_heights": round(dy, 4), "hoop_x": hoop_x, "hoop_y": hoop_y,
                    "ball_confidence": ball["confidence"], "hoop_confidence": hoop["confidence"],
                })
    return observations


def frame_explanation(frame: dict) -> dict:
    observations = rim_observations([frame])
    if not observations:
        return {"is_candidate": False, "state": "no_aligned_pair",
                "reason": "No basketball and rim pair was close enough in this frame."}
    observation = min(observations, key=lambda item: abs(item["dx_rim_widths"]) + abs(item["dy_rim_heights"]))
    if observation["dy_rim_heights"] <= -.25 and abs(observation["dx_rim_widths"]) <= .8:
        state = "above_rim"
        reason = "Ball is above and horizontally aligned with the rim; waiting for a below-rim observation."
    elif observation["dy_rim_heights"] >= .35 and abs(observation["dx_rim_widths"]) <= .8:
        state = "below_rim"
        reason = "Ball is below and horizontally aligned with the rim; an earlier above-rim observation is required."
    else:
        state = "near_rim"
        reason = "Ball is near the rim but does not yet satisfy the above or below crossing boundary."
    return {"is_candidate": False, "state": state, "reason": reason, "observation": observation}


def made_basket_candidates(frames: list[dict], crossing_window_seconds: float) -> list[dict]:
    observations = rim_observations(frames)
    candidates = []
    for index, above in enumerate(observations):
        if above["dy_rim_heights"] > -.25 or abs(above["dx_rim_widths"]) > .8:
            continue
        for below in observations[index + 1:]:
            elapsed = below["time_seconds"] - above["time_seconds"]
            if elapsed <= 0:
                continue
            if elapsed > crossing_window_seconds:
                break
            if below["dy_rim_heights"] < .35 or abs(below["dx_rim_widths"]) > .8:
                continue
            if abs(below["hoop_x"] - above["hoop_x"]) > .25 or abs(below["hoop_y"] - above["hoop_y"]) > .2:
                continue
            span = below["dy_rim_heights"] - above["dy_rim_heights"]
            fraction = min(1.0, max(0.0, -above["dy_rim_heights"] / span))
            event_time = above["time_seconds"] + elapsed * fraction
            detector_confidence = min(above["ball_confidence"], above["hoop_confidence"],
                                      below["ball_confidence"], below["hoop_confidence"])
            alignment = max(0.0, 1 - (abs(above["dx_rim_widths"]) + abs(below["dx_rim_widths"])) / 2)
            confidence = round(min(1.0, detector_confidence * (.75 + .25 * alignment)), 4)
            candidates.append({
                "is_candidate": True, "event_time_seconds": round(event_time, 3), "confidence": confidence,
                "above": above, "below": below,
                "reason": "Ball moved from above the rim to below it while horizontally aligned.",
            })
            break
    deduplicated = []
    for candidate in sorted(candidates, key=lambda item: item["confidence"], reverse=True):
        if not any(abs(candidate["event_time_seconds"] - kept["event_time_seconds"]) < 3 for kept in deduplicated):
            deduplicated.append(candidate)
    return sorted(deduplicated, key=lambda item: item["event_time_seconds"])


def _create_candidate_events(db: Database, settings: Settings, job, candidates: list[dict]) -> list[tuple[str, dict]]:
    created = []
    model_version = f"{job['model_name']}:{job['training_job_id'][:8]}"
    with db.transaction() as connection:
        lineage = connection.execute(
            "SELECT job_id FROM analysis_jobs WHERE recording_id=? AND training_job_id=?",
            (job["recording_id"], job["training_job_id"]),
        ).fetchall()
        lineage_ids = [row["job_id"] for row in lineage]
        if lineage_ids:
            placeholders = ",".join("?" for _ in lineage_ids)
            connection.execute(
                f"""DELETE FROM events WHERE source='model' AND review_status='candidate'
                    AND analysis_job_id IN ({placeholders}) AND event_time_seconds BETWEEN ? AND ?""",
                (*lineage_ids, job["start_seconds"], job["end_seconds"]),
            )
        for candidate in candidates:
            event_time = candidate["event_time_seconds"]
            protected = connection.execute(
                """SELECT event_id FROM events WHERE recording_id=? AND event_type='made_basket'
                   AND ABS(event_time_seconds-?) < 3 AND (source!='model' OR review_status!='candidate') LIMIT 1""",
                (job["recording_id"], event_time),
            ).fetchone()
            duplicate = connection.execute(
                """SELECT event_id FROM events WHERE recording_id=? AND event_type='made_basket'
                   AND ABS(event_time_seconds-?) < 3 LIMIT 1""", (job["recording_id"], event_time),
            ).fetchone()
            if protected or duplicate:
                continue
            event_id = str(uuid.uuid4())
            play_from = max(0.0, event_time - settings.default_before_seconds)
            duration = float(job["duration_seconds"] or job["end_seconds"])
            play_until = min(duration, event_time + settings.default_after_seconds)
            connection.execute(
                """INSERT INTO events
                   (event_id,recording_id,event_type,event_time_seconds,play_from_seconds,play_until_seconds,
                    confidence,review_status,source,model_version,analysis_job_id)
                   VALUES (?,?, 'made_basket', ?,?,?,?,'candidate','model',?,?)""",
                (event_id, job["recording_id"], event_time, play_from, play_until, candidate["confidence"],
                 model_version, job["job_id"]),
            )
            created.append((event_id, candidate))
    return created


def run_job(job_id: str) -> None:
    settings = Settings.from_env()
    db = Database(settings.db_path)
    job_root = settings.config_root / "analysis" / "jobs" / job_id
    try:
        with db.connect() as connection:
            job = connection.execute(
                """SELECT a.*, r.media_path, r.size_bytes, r.mtime_ns, r.availability,
                          r.duration_seconds, t.model_path, t.model_name
                   FROM analysis_jobs a JOIN recordings r ON r.recording_id=a.recording_id
                   JOIN training_jobs t ON t.job_id=a.training_job_id WHERE a.job_id=?""", (job_id,),
            ).fetchone()
        if not job:
            raise RuntimeError("Analysis job was not found")
        if job["availability"] != "present":
            raise RuntimeError("The selected recording is unavailable")
        source = resolve_known_media(settings.media_root, job["media_path"])
        stat = source.stat()
        if stat.st_size != job["size_bytes"] or stat.st_mtime_ns != job["mtime_ns"]:
            raise RuntimeError("The recording changed after it was cataloged; rescan before analysis")
        model_path = Path(job["model_path"] or "").resolve()
        training_root = (settings.config_root / "training" / "jobs" / job["training_job_id"]).resolve()
        if not model_path.is_relative_to(training_root) or not model_path.is_file():
            raise RuntimeError("The selected trained model is unavailable")

        from ultralytics import YOLO, settings as yolo_settings

        yolo_settings.update({"sync": False})
        model = YOLO(str(model_path))
        output_dir = Path(job["output_dir"])
        frames_dir = output_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        expected_frames = len(frame_times(job["start_seconds"], job["end_seconds"], job["sample_interval_seconds"]))
        _update(db, job_id, status="running", started_at=_now())
        total_detections = 0
        processed = 0
        frame_records = []
        result_records = []
        cursor = float(job["start_seconds"])
        chunk_number = 0
        while cursor < float(job["end_seconds"]) - .001:
            if _cancelled(db, job_id):
                raise AnalysisCancelled("Stopped by user")
            chunk_end = min(float(job["end_seconds"]), cursor + 60)
            chunk_dir = output_dir / f"chunk-{chunk_number:05d}"
            extracted = _extract_chunk(source, chunk_dir, cursor, chunk_end, float(job["sample_interval_seconds"]))
            if not extracted:
                raise RuntimeError(f"No video frames could be extracted near {cursor:.1f} seconds")
            predictions = model.predict(
                source=[str(raw) for raw, _ in extracted], conf=job["confidence_threshold"],
                device="0", verbose=False, stream=True,
            )
            for (raw, time_seconds), result in zip(extracted, predictions):
                detected = _detections(result)
                frame_record = {"time_seconds": time_seconds, "detections": detected}
                frame_records.append(frame_record)
                total_detections += len(detected)
                near_rim = bool(rim_observations([frame_record]))
                if job["mode"] == "calibration" or near_rim:
                    preview = frames_dir / f"result-{processed:07d}.jpg"
                    result.save(filename=str(preview))
                    result_id = str(uuid.uuid4())
                    with db.transaction() as connection:
                        connection.execute(
                            """INSERT INTO analysis_results
                               (result_id,job_id,frame_time_seconds,image_path,detections_json,detection_count,
                                explanation_json) VALUES (?,?,?,?,?,?,?)""",
                            (result_id, job_id, time_seconds, str(preview), json.dumps(detected), len(detected),
                             json.dumps(frame_explanation(frame_record))),
                        )
                    result_records.append({"result_id": result_id, "time_seconds": time_seconds})
                processed += 1
                if processed == 1 or processed % 10 == 0:
                    if _cancelled(db, job_id):
                        raise AnalysisCancelled("Stopped by user")
                    _update(
                        db, job_id, processed_frames=processed, detection_count=total_detections,
                        progress_percent=min(99, round(processed / max(1, expected_frames) * 100, 2)),
                    )
            shutil.rmtree(chunk_dir, ignore_errors=True)
            cursor = chunk_end
            chunk_number += 1

        if _cancelled(db, job_id):
            raise AnalysisCancelled("Stopped by user")
        candidates = made_basket_candidates(frame_records, float(job["crossing_window_seconds"]))
        created = _create_candidate_events(db, settings, job, candidates) if job["mode"] == "full_game" else []
        created_by_time = {candidate["event_time_seconds"]: event_id for event_id, candidate in created}
        with db.transaction() as connection:
            for candidate in candidates:
                if not result_records:
                    continue
                evidence = min(result_records, key=lambda item: abs(item["time_seconds"] - candidate["below"]["time_seconds"]))
                connection.execute(
                    """UPDATE analysis_results SET explanation_json=?,candidate_event_id=? WHERE result_id=?""",
                    (json.dumps(candidate), created_by_time.get(candidate["event_time_seconds"]), evidence["result_id"]),
                )
        _update(
            db, job_id, status="completed", progress_percent=100, processed_frames=processed,
            detection_count=total_detections, candidate_count=len(created) if job["mode"] == "full_game" else len(candidates),
            completed_at=_now(),
        )
    except AnalysisCancelled as exc:
        _update(db, job_id, status="failed", error_message=str(exc), completed_at=_now())
    except Exception as exc:
        _update(db, job_id, status="failed", error_message=str(exc)[:1000], completed_at=_now())
        raise
    finally:
        for chunk_dir in job_root.glob("chunk-*"):
            shutil.rmtree(chunk_dir, ignore_errors=True)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python -m app.analysis_jobs JOB_ID")
    run_job(sys.argv[1])
