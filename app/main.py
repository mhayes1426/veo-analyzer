from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import uuid
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, model_validator

from . import __version__
from .annotations import FRAME_OFFSETS, annotation_frame_path, detailed_frame_times, extract_annotation_frame
from .analysis_jobs import inference_runtime_available, launch_analysis
from .catalog import scan_media
from .config import Settings
from .db import Database
from .dataset import quality_report, recording_split, yolo_label
from .gpu import gpu_diagnostics
from .media import resolve_known_media, stream_media
from .thumbnails import generate_thumbnail, thumbnail_path
from .training_jobs import ALLOWED_MODELS, launch_training, training_runtime_available

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("veo-analyzer")

BASE_DIR = Path(__file__).parent
settings = Settings.from_env()
db = Database(settings.db_path)
templates = Jinja2Templates(directory=BASE_DIR / "templates")


def row_dict(row):
    return dict(row) if row else None


async def catalog_loop() -> None:
    while True:
        try:
            count = await asyncio.to_thread(scan_media, db, settings.media_root, settings.stable_age_seconds)
            logger.info("Media catalog scan completed: %s recordings", count)
        except Exception:
            logger.exception("Media catalog scan failed")
        await asyncio.sleep(settings.scan_interval_seconds)


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings.validate()
    db.initialize()
    with db.transaction() as connection:
        connection.execute(
            """UPDATE training_jobs SET status='failed', error_message='Analyzer restarted during training',
               completed_at=CURRENT_TIMESTAMP, updated_at=CURRENT_TIMESTAMP
               WHERE status IN ('queued','preparing','running')"""
        )
        connection.execute(
            """UPDATE analysis_jobs SET status='failed', error_message='Analyzer restarted during analysis',
               completed_at=CURRENT_TIMESTAMP, updated_at=CURRENT_TIMESTAMP
               WHERE status IN ('queued','running')"""
        )
    task = asyncio.create_task(catalog_loop())
    yield
    task.cancel()


app = FastAPI(title="Veo Analyzer", version=__version__, lifespan=lifespan)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


class EventInput(BaseModel):
    event_type: str = Field(default="made_basket", pattern=r"^[a-z][a-z0-9_]{0,31}$")
    event_time_seconds: float = Field(ge=0)
    play_from_seconds: float = Field(ge=0)
    play_until_seconds: float = Field(ge=0)
    review_status: str = Field(default="approved", pattern=r"^(candidate|approved|rejected)$")
    sequence_outcome: str | None = Field(default=None, pattern=r"^(made|missed|uncertain)$")

    @model_validator(mode="after")
    def valid_window(self):
        if not self.play_from_seconds <= self.event_time_seconds <= self.play_until_seconds:
            raise ValueError("Playback window must contain the event time")
        return self


class BoxInput(BaseModel):
    object_class: str = Field(pattern=r"^(basketball|hoop)$")
    x_center: float = Field(ge=0, le=1)
    y_center: float = Field(ge=0, le=1)
    width: float = Field(gt=0, le=1)
    height: float = Field(gt=0, le=1)
    occluded: bool = False

    @model_validator(mode="after")
    def inside_frame(self):
        if self.x_center - self.width / 2 < 0 or self.x_center + self.width / 2 > 1:
            raise ValueError("Box exceeds frame horizontally")
        if self.y_center - self.height / 2 < 0 or self.y_center + self.height / 2 > 1:
            raise ValueError("Box exceeds frame vertically")
        return self


class FrameBoxesInput(BaseModel):
    boxes: list[BoxInput] = Field(max_length=100)
    review_status: str = Field(default="reviewed", pattern=r"^(pending|reviewed|skipped)$")


class DetailedFramesInput(BaseModel):
    center_time_seconds: float = Field(ge=0)


class SequenceOutcomeInput(BaseModel):
    sequence_outcome: str = Field(pattern=r"^(made|missed|uncertain)$")


class AnnotationSizeInput(BaseModel):
    object_class: str = Field(pattern=r"^(basketball|hoop)$")
    width: float = Field(gt=0, le=1)
    height: float = Field(gt=0, le=1)


class TrainingJobInput(BaseModel):
    model_name: str = Field(default="yolo11n.pt", pattern=r"^yolo11(n|s|m)\.pt$")
    epochs: int = Field(default=50, ge=1, le=500)
    image_size: int = Field(default=640, ge=320, le=1280, multiple_of=32)
    batch_size: int = Field(default=8, ge=-1, le=128)

    @model_validator(mode="after")
    def valid_training_options(self):
        if self.model_name not in ALLOWED_MODELS:
            raise ValueError("Unsupported model")
        if self.batch_size == 0:
            raise ValueError("Batch size cannot be zero")
        return self


class AnalysisJobInput(BaseModel):
    recording_id: str = Field(min_length=1, max_length=100)
    training_job_id: str = Field(min_length=1, max_length=100)
    start_seconds: float = Field(default=0, ge=0)
    end_seconds: float = Field(default=30, gt=0)
    sample_interval_seconds: float = Field(default=0.5, ge=0.1, le=10)
    confidence_threshold: float = Field(default=0.25, ge=0.01, le=1)
    mode: str = Field(default="calibration", pattern=r"^(calibration|full_game)$")
    crossing_window_seconds: float = Field(default=1.0, ge=0.3, le=2.0)

    @model_validator(mode="after")
    def valid_range(self):
        if self.end_seconds <= self.start_seconds:
            raise ValueError("End time must be after start time")
        if self.mode == "calibration" and self.end_seconds - self.start_seconds > 600:
            raise ValueError("Detector tests are limited to 10 minutes")
        return self


def training_job_dict(row) -> dict:
    result = {key: row[key] for key in (
        "job_id", "status", "model_name", "epochs", "image_size", "batch_size", "device",
        "progress_percent", "current_epoch", "frame_count", "error_message", "created_at",
        "started_at", "completed_at", "updated_at",
    )}
    result["metrics"] = json.loads(row["metrics_json"] or "{}")
    result["model_available"] = bool(row["model_path"] and Path(row["model_path"]).is_file())
    return result


def analysis_job_dict(row) -> dict:
    result = {key: row[key] for key in (
        "job_id", "recording_id", "training_job_id", "status", "mode", "start_seconds", "end_seconds",
        "sample_interval_seconds", "confidence_threshold", "progress_percent", "processed_frames",
        "crossing_window_seconds", "detection_count", "candidate_count", "cancel_requested", "error_message", "created_at",
        "started_at", "completed_at", "updated_at",
    )}
    result["recording_title"] = row["recording_title"] if "recording_title" in row.keys() else None
    return result


def get_recording(recording_id: str):
    with db.connect() as connection:
        row = connection.execute("SELECT * FROM recordings WHERE recording_id=?", (recording_id,)).fetchone()
    if not row:
        raise HTTPException(404, "Recording not found")
    return row


def get_event(event_id: str):
    with db.connect() as connection:
        row = connection.execute("SELECT * FROM events WHERE event_id=?", (event_id,)).fetchone()
    if not row:
        raise HTTPException(404, "Event not found")
    return row


def ensure_annotation_frames(event_id: str):
    event = get_event(event_id)
    recording = get_recording(event["recording_id"])
    duration = recording["duration_seconds"] or event["play_until_seconds"]
    with db.transaction() as connection:
        for index, offset in enumerate(FRAME_OFFSETS):
            frame_time = max(0.0, min(float(duration), event["event_time_seconds"] + offset))
            connection.execute(
                """INSERT OR IGNORE INTO annotation_frames
                   (frame_id, event_id, frame_time_seconds, frame_index) VALUES (?, ?, ?, ?)""",
                (str(uuid.uuid4()), event_id, frame_time, index),
            )
        return connection.execute(
            "SELECT * FROM annotation_frames WHERE event_id=? ORDER BY frame_time_seconds, frame_index", (event_id,)
        ).fetchall()


def dataset_frames() -> list[dict]:
    with db.connect() as connection:
        frames = connection.execute(
            """SELECT f.*, e.sequence_outcome, e.event_type, e.event_time_seconds,
                      r.recording_id, r.media_path, r.size_bytes, r.mtime_ns, r.availability
               FROM annotation_frames f JOIN events e ON e.event_id=f.event_id
               JOIN recordings r ON r.recording_id=e.recording_id
               ORDER BY r.recording_id, f.event_id, f.frame_time_seconds"""
        ).fetchall()
        boxes = connection.execute(
            "SELECT * FROM annotation_boxes ORDER BY frame_id, created_at"
        ).fetchall()
    boxes_by_frame: dict[str, list[dict]] = {}
    for box in boxes:
        boxes_by_frame.setdefault(box["frame_id"], []).append(dict(box))
    return [dict(row) | {"boxes": boxes_by_frame.get(row["frame_id"], [])} for row in frames]


def annotation_queue(recording_id: str) -> list[dict]:
    with db.connect() as connection:
        rows = connection.execute(
            """SELECT e.*,
                      (SELECT COUNT(*) FROM annotation_frames f WHERE f.event_id=e.event_id) AS frame_count,
                      (SELECT COUNT(*) FROM annotation_frames f WHERE f.event_id=e.event_id
                       AND f.review_status='pending') AS pending_count,
                      (SELECT COUNT(*) FROM annotation_frames f WHERE f.event_id=e.event_id
                       AND f.review_status='reviewed') AS reviewed_count,
                      (SELECT COUNT(*) FROM annotation_frames f WHERE f.event_id=e.event_id
                       AND f.review_status='skipped') AS skipped_count
               FROM events e WHERE e.recording_id=? ORDER BY e.event_time_seconds""",
            (recording_id,),
        ).fetchall()
    queue = []
    for row in rows:
        item = dict(row)
        frame_count = item["frame_count"]
        basket_event = item["event_type"] in {"basket_attempt", "made_basket"}
        if frame_count and item["skipped_count"] == frame_count:
            state = "unusable"
        elif not frame_count or (item["pending_count"] == frame_count and not item["reviewed_count"] and not item["skipped_count"]):
            state = "not_started"
        elif not item["pending_count"] and (not basket_event or item["sequence_outcome"] != "uncertain"):
            state = "complete"
        else:
            state = "in_progress"
        item["annotation_state"] = state
        queue.append(item)
    return queue


@app.get("/health")
def health():
    with db.connect() as connection:
        connection.execute("SELECT 1").fetchone()
    return {"status": "ok", "version": __version__}


@app.get("/training", response_class=HTMLResponse)
def training_page(request: Request):
    frames = dataset_frames()
    report = quality_report(frames)
    diagnostics = gpu_diagnostics()
    split_counts = {"train": 0, "val": 0, "test": 0}
    for row in frames:
        if row["review_status"] == "reviewed" and row["availability"] == "present":
            split_counts[recording_split(row["recording_id"])] += 1
    with db.connect() as connection:
        jobs = connection.execute("SELECT * FROM training_jobs ORDER BY created_at DESC LIMIT 20").fetchall()
    return templates.TemplateResponse(request, "training.html", {
        "report": report,
        "gpu": diagnostics,
        "jobs": [training_job_dict(row) for row in jobs],
        "training_available": training_runtime_available(),
        "split_counts": split_counts,
    })


@app.get("/api/system/gpu")
def gpu_status():
    return gpu_diagnostics()


@app.get("/api/training/quality")
def training_quality():
    return quality_report(dataset_frames())


@app.get("/api/training/jobs")
def list_training_jobs():
    with db.connect() as connection:
        rows = connection.execute("SELECT * FROM training_jobs ORDER BY created_at DESC LIMIT 50").fetchall()
    return {"jobs": [training_job_dict(row) for row in rows], "training_available": training_runtime_available()}


@app.post("/api/training/jobs", status_code=202)
def create_training_job(payload: TrainingJobInput):
    if not training_runtime_available():
        raise HTTPException(409, "Training is available only in the NVIDIA GPU image")
    if not gpu_diagnostics().get("cuda_available"):
        raise HTTPException(409, "CUDA is not available to PyTorch")
    split_counts = {"train": 0, "val": 0, "test": 0}
    for frame in dataset_frames():
        if frame["review_status"] == "reviewed" and frame["availability"] == "present":
            split_counts[recording_split(frame["recording_id"])] += 1
    if split_counts["train"] == 0 or split_counts["val"] == 0:
        raise HTTPException(409, "Reviewed frames are required in both training and validation splits")
    with db.transaction() as connection:
        active = connection.execute(
            "SELECT job_id FROM training_jobs WHERE status IN ('queued','preparing','running') LIMIT 1"
        ).fetchone()
        active_analysis = connection.execute(
            "SELECT job_id FROM analysis_jobs WHERE status IN ('queued','running') LIMIT 1"
        ).fetchone()
        if active or active_analysis:
            raise HTTPException(409, "The GPU is already running another training or analysis job")
        job_id = str(uuid.uuid4())
        output_dir = settings.config_root / "training" / "jobs" / job_id
        connection.execute(
            """INSERT INTO training_jobs
               (job_id,status,model_name,epochs,image_size,batch_size,device,output_dir)
               VALUES (?, 'queued', ?, ?, ?, ?, '0', ?)""",
            (job_id, payload.model_name, payload.epochs, payload.image_size, payload.batch_size, str(output_dir)),
        )
        row = connection.execute("SELECT * FROM training_jobs WHERE job_id=?", (job_id,)).fetchone()
    try:
        launch_training(job_id, settings)
    except OSError as exc:
        with db.transaction() as connection:
            connection.execute(
                """UPDATE training_jobs SET status='failed', error_message=?, completed_at=CURRENT_TIMESTAMP,
                   updated_at=CURRENT_TIMESTAMP WHERE job_id=?""",
                (str(exc)[:1000], job_id),
            )
        raise HTTPException(503, "Could not launch training worker") from exc
    return training_job_dict(row)


@app.get("/api/training/jobs/{job_id}")
def get_training_job(job_id: str):
    with db.connect() as connection:
        row = connection.execute("SELECT * FROM training_jobs WHERE job_id=?", (job_id,)).fetchone()
    if not row:
        raise HTTPException(404, "Training job not found")
    return training_job_dict(row)


@app.get("/api/training/jobs/{job_id}/model")
def download_training_model(job_id: str):
    with db.connect() as connection:
        row = connection.execute("SELECT * FROM training_jobs WHERE job_id=?", (job_id,)).fetchone()
    if not row or not row["model_path"]:
        raise HTTPException(404, "Trained model not found")
    root = (settings.config_root / "training" / "jobs" / job_id).resolve()
    model_path = Path(row["model_path"]).resolve()
    if not model_path.is_relative_to(root) or not model_path.is_file():
        raise HTTPException(404, "Trained model not found")
    return FileResponse(model_path, filename=f"veo-{job_id[:8]}-{row['model_name']}")


@app.get("/api/training/jobs/{job_id}/log")
def training_job_log(job_id: str):
    with db.connect() as connection:
        row = connection.execute("SELECT job_id FROM training_jobs WHERE job_id=?", (job_id,)).fetchone()
    if not row:
        raise HTTPException(404, "Training job not found")
    path = settings.config_root / "training" / "jobs" / job_id / "training.log"
    return {"log": path.read_text(errors="replace")[-20000:] if path.is_file() else "Training has not produced output yet."}


@app.get("/analyze", response_class=HTMLResponse)
def analysis_page(request: Request):
    with db.connect() as connection:
        recordings = connection.execute(
            "SELECT recording_id,title,duration_seconds FROM recordings WHERE availability='present' ORDER BY created_at DESC"
        ).fetchall()
        models = connection.execute(
            """SELECT job_id,model_name,completed_at FROM training_jobs
               WHERE status='completed' AND model_path IS NOT NULL ORDER BY completed_at DESC"""
        ).fetchall()
        jobs = connection.execute(
            """SELECT a.*,r.title AS recording_title FROM analysis_jobs a
               JOIN recordings r ON r.recording_id=a.recording_id ORDER BY a.created_at DESC LIMIT 30"""
        ).fetchall()
    return templates.TemplateResponse(request, "analyze.html", {
        "recordings": recordings, "models": models, "jobs": [analysis_job_dict(row) for row in jobs],
        "inference_available": inference_runtime_available(), "gpu": gpu_diagnostics(),
    })


@app.get("/api/analysis/jobs")
def list_analysis_jobs():
    with db.connect() as connection:
        rows = connection.execute(
            """SELECT a.*,r.title AS recording_title FROM analysis_jobs a
               JOIN recordings r ON r.recording_id=a.recording_id ORDER BY a.created_at DESC LIMIT 50"""
        ).fetchall()
    return {"jobs": [analysis_job_dict(row) for row in rows]}


@app.post("/api/analysis/jobs", status_code=202)
def create_analysis_job(payload: AnalysisJobInput):
    if not inference_runtime_available():
        raise HTTPException(409, "Detector testing is available only in the NVIDIA GPU image")
    if not gpu_diagnostics().get("cuda_available"):
        raise HTTPException(409, "CUDA is not available to PyTorch")
    recording = get_recording(payload.recording_id)
    if recording["availability"] != "present":
        raise HTTPException(409, "The selected recording is unavailable")
    if recording["duration_seconds"] is not None and payload.end_seconds > recording["duration_seconds"] + 0.01:
        raise HTTPException(422, "End time exceeds the recording duration")
    with db.transaction() as connection:
        model = connection.execute(
            """SELECT * FROM training_jobs WHERE job_id=? AND status='completed'
               AND model_path IS NOT NULL""", (payload.training_job_id,),
        ).fetchone()
        if not model:
            raise HTTPException(404, "Completed trained model not found")
        model_path = Path(model["model_path"]).resolve()
        model_root = (settings.config_root / "training" / "jobs" / payload.training_job_id).resolve()
        if not model_path.is_relative_to(model_root) or not model_path.is_file():
            raise HTTPException(404, "Trained model file not found")
        active_analysis = connection.execute(
            "SELECT job_id FROM analysis_jobs WHERE status IN ('queued','running') LIMIT 1"
        ).fetchone()
        active_training = connection.execute(
            "SELECT job_id FROM training_jobs WHERE status IN ('queued','preparing','running') LIMIT 1"
        ).fetchone()
        if active_analysis or active_training:
            raise HTTPException(409, "The GPU is already running another training or analysis job")
        job_id = str(uuid.uuid4())
        output_dir = settings.config_root / "analysis" / "jobs" / job_id
        connection.execute(
            """INSERT INTO analysis_jobs
               (job_id,recording_id,training_job_id,status,mode,start_seconds,end_seconds,
                sample_interval_seconds,confidence_threshold,crossing_window_seconds,output_dir)
               VALUES (?,?,?,'queued',?,?,?,?,?,?,?)""",
            (job_id, payload.recording_id, payload.training_job_id, payload.mode, payload.start_seconds,
             payload.end_seconds, payload.sample_interval_seconds, payload.confidence_threshold,
             payload.crossing_window_seconds, str(output_dir)),
        )
        row = connection.execute("SELECT * FROM analysis_jobs WHERE job_id=?", (job_id,)).fetchone()
    try:
        launch_analysis(job_id, settings)
    except OSError as exc:
        with db.transaction() as connection:
            connection.execute(
                """UPDATE analysis_jobs SET status='failed',error_message=?,completed_at=CURRENT_TIMESTAMP,
                   updated_at=CURRENT_TIMESTAMP WHERE job_id=?""", (str(exc)[:1000], job_id),
            )
        raise HTTPException(503, "Could not launch detector test") from exc
    return analysis_job_dict(row)


@app.get("/api/analysis/jobs/{job_id}")
def get_analysis_job(job_id: str):
    with db.connect() as connection:
        row = connection.execute(
            """SELECT a.*,r.title AS recording_title FROM analysis_jobs a
               JOIN recordings r ON r.recording_id=a.recording_id WHERE a.job_id=?""", (job_id,),
        ).fetchone()
        results = connection.execute(
            """SELECT result_id,frame_time_seconds,detections_json,detection_count,
                      explanation_json,candidate_event_id
               FROM analysis_results WHERE job_id=? ORDER BY frame_time_seconds""", (job_id,),
        ).fetchall()
    if not row:
        raise HTTPException(404, "Analysis job not found")
    payload = analysis_job_dict(row)
    payload["results"] = []
    for result in results:
        item = dict(result)
        item["detections"] = json.loads(item.pop("detections_json"))
        explanation = item.pop("explanation_json")
        item["explanation"] = json.loads(explanation) if explanation else None
        item["image_url"] = f"/api/analysis/results/{item['result_id']}/image"
        payload["results"].append(item)
    return payload


@app.get("/api/analysis/jobs/{job_id}/log")
def analysis_job_log(job_id: str):
    with db.connect() as connection:
        row = connection.execute("SELECT job_id FROM analysis_jobs WHERE job_id=?", (job_id,)).fetchone()
    if not row:
        raise HTTPException(404, "Analysis job not found")
    path = settings.config_root / "analysis" / "jobs" / job_id / "analysis.log"
    return {"log": path.read_text(errors="replace")[-20000:] if path.is_file() else "Analysis has not produced output yet."}


@app.post("/api/analysis/jobs/{job_id}/cancel", status_code=202)
def cancel_analysis_job(job_id: str):
    with db.transaction() as connection:
        row = connection.execute("SELECT status FROM analysis_jobs WHERE job_id=?", (job_id,)).fetchone()
        if not row:
            raise HTTPException(404, "Analysis job not found")
        if row["status"] not in {"queued", "running"}:
            raise HTTPException(409, "Analysis job is no longer active")
        connection.execute(
            "UPDATE analysis_jobs SET cancel_requested=1,updated_at=CURRENT_TIMESTAMP WHERE job_id=?", (job_id,),
        )
    return {"job_id": job_id, "stop_requested": True}


@app.get("/api/analysis/results/{result_id}/image")
def analysis_result_image(result_id: str):
    with db.connect() as connection:
        row = connection.execute(
            "SELECT result_id,job_id,image_path FROM analysis_results WHERE result_id=?", (result_id,),
        ).fetchone()
    if not row:
        raise HTTPException(404, "Analysis result not found")
    root = (settings.config_root / "analysis" / "jobs" / row["job_id"]).resolve()
    image_path = Path(row["image_path"]).resolve()
    if not image_path.is_relative_to(root) or not image_path.is_file():
        raise HTTPException(404, "Analysis result image not found")
    return FileResponse(image_path, media_type="image/jpeg", headers={"Cache-Control": "private, no-store"})


@app.get("/api/training/export/yolo")
def export_yolo_dataset():
    frames = dataset_frames()
    reviewed = [row for row in frames if row["review_status"] == "reviewed" and row["availability"] == "present"]
    if not reviewed:
        raise HTTPException(409, "No reviewed annotation frames are available")
    export_dir = settings.export_root / "datasets"
    export_dir.mkdir(parents=True, exist_ok=True)
    output = export_dir / "veo-yolo-dataset.zip"
    temporary = output.with_suffix(".zip.tmp")
    manifest = {"schema_version": 1, "classes": ["basketball", "hoop"], "frames": []}
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("data.yaml", "path: .\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: basketball\n  1: hoop\n")
        for row in reviewed:
            source = resolve_known_media(settings.media_root, row["media_path"])
            stat = source.stat()
            if stat.st_size != row["size_bytes"] or stat.st_mtime_ns != row["mtime_ns"]:
                continue
            image_path = annotation_frame_path(settings.config_root, row["event_id"], row["frame_index"])
            try:
                extract_annotation_frame(source, image_path, row["frame_time_seconds"])
            except (OSError, subprocess.SubprocessError):
                logger.exception("Dataset frame extraction failed for %s", row["frame_id"])
                continue
            split = recording_split(row["recording_id"])
            stem = row["frame_id"]
            archive.write(image_path, f"images/{split}/{stem}.jpg")
            labels = "\n".join(yolo_label(box) for box in row["boxes"])
            archive.writestr(f"labels/{split}/{stem}.txt", labels + ("\n" if labels else ""))
            manifest["frames"].append({
                "frame_id": row["frame_id"], "recording_id": row["recording_id"], "event_id": row["event_id"],
                "frame_time_seconds": row["frame_time_seconds"], "split": split,
                "event_type": row["event_type"], "sequence_outcome": row["sequence_outcome"],
                "box_count": len(row["boxes"]),
                "source_fingerprint": {"size_bytes": row["size_bytes"], "mtime_ns": row["mtime_ns"]},
            })
        archive.writestr("manifest.json", json.dumps(manifest, indent=2) + "\n")
        archive.writestr("quality-report.json", json.dumps(quality_report(frames), indent=2) + "\n")
    temporary.replace(output)
    return FileResponse(output, filename="veo-yolo-dataset.zip", media_type="application/zip")


@app.get("/", response_class=HTMLResponse)
def library(request: Request):
    with db.connect() as connection:
        recordings = connection.execute(
            """SELECT r.*,
                      (SELECT COUNT(*) FROM events e WHERE e.recording_id=r.recording_id) AS event_count,
                      (SELECT COUNT(*) FROM events e WHERE e.recording_id=r.recording_id
                       AND EXISTS (SELECT 1 FROM annotation_frames f WHERE f.event_id=e.event_id
                                   AND f.review_status='reviewed')) AS annotated_event_count,
                      (SELECT COUNT(*) FROM events e WHERE e.recording_id=r.recording_id
                       AND NOT (EXISTS (SELECT 1 FROM annotation_frames f WHERE f.event_id=e.event_id)
                                AND NOT EXISTS (SELECT 1 FROM annotation_frames f WHERE f.event_id=e.event_id
                                                AND f.review_status='pending')
                                AND (e.event_type NOT IN ('basket_attempt','made_basket')
                                     OR e.sequence_outcome!='uncertain'
                                     OR NOT EXISTS (SELECT 1 FROM annotation_frames f WHERE f.event_id=e.event_id
                                                    AND f.review_status!='skipped')))) AS unannotated_event_count
               FROM recordings r ORDER BY r.created_at DESC"""
        ).fetchall()
    return templates.TemplateResponse(request, "library.html", {"recordings": recordings, "version": __version__})


@app.post("/api/scan")
async def scan_now():
    count = await asyncio.to_thread(scan_media, db, settings.media_root, settings.stable_age_seconds)
    return {"recordings_seen": count}


@app.get("/recordings/{recording_id}", response_class=HTMLResponse)
def recording_page(request: Request, recording_id: str):
    recording = get_recording(recording_id)
    with db.connect() as connection:
        events = connection.execute(
            "SELECT * FROM events WHERE recording_id=? ORDER BY event_time_seconds", (recording_id,)
        ).fetchall()
    queue = annotation_queue(recording_id)
    remaining = [item for item in queue if item["annotation_state"] not in {"complete", "unusable"}]
    return templates.TemplateResponse(
        request,
        "recording.html",
        {
            "recording": recording,
            "events_json": json.dumps([dict(row) for row in events]),
            "before": settings.default_before_seconds,
            "after": settings.default_after_seconds,
            "annotation_remaining": len(remaining),
            "annotation_start_event_id": (remaining[0] if remaining else queue[0] if queue else {}).get("event_id"),
        },
    )


@app.get("/events/{event_id}/annotate", response_class=HTMLResponse)
def annotation_page(request: Request, event_id: str):
    event = get_event(event_id)
    recording = get_recording(event["recording_id"])
    frames = ensure_annotation_frames(event_id)
    with db.connect() as connection:
        boxes = connection.execute(
            """SELECT b.* FROM annotation_boxes b JOIN annotation_frames f ON f.frame_id=b.frame_id
               WHERE f.event_id=? ORDER BY f.frame_index, b.created_at""",
            (event_id,),
        ).fetchall()
        preset_rows = connection.execute(
            "SELECT object_class, width, height FROM annotation_size_presets WHERE recording_id=?",
            (recording["recording_id"],),
        ).fetchall()
        model_rows = connection.execute(
            """SELECT job_id,model_name,completed_at FROM training_jobs
               WHERE status='completed' AND model_path IS NOT NULL ORDER BY completed_at DESC"""
        ).fetchall()
    boxes_by_frame: dict[str, list[dict]] = {row["frame_id"]: [] for row in frames}
    for box in boxes:
        boxes_by_frame[box["frame_id"]].append(dict(box))
    frame_data = [dict(row) | {"boxes": boxes_by_frame[row["frame_id"]]} for row in frames]
    queue = annotation_queue(recording["recording_id"])
    queue_index = next(index for index, item in enumerate(queue) if item["event_id"] == event_id)
    previous_event = queue[(queue_index - 1) % len(queue)] if len(queue) > 1 else None
    next_event = queue[(queue_index + 1) % len(queue)] if len(queue) > 1 else None
    next_pending = next(
        (item for item in queue[queue_index + 1:] + queue[:queue_index]
         if item["annotation_state"] not in {"complete", "unusable"}),
        None,
    )
    return templates.TemplateResponse(
        request,
        "annotate.html",
        {"event": event, "recording": recording, "frames_json": json.dumps(frame_data),
         "size_presets_json": json.dumps({row["object_class"]: {"width": row["width"], "height": row["height"]} for row in preset_rows}),
         "annotation_queue": queue, "previous_event": previous_event, "next_event": next_event,
         "next_pending": next_pending, "models": model_rows,
         "model_test_available": inference_runtime_available()},
    )


@app.put("/api/recordings/{recording_id}/annotation-size")
def save_annotation_size(recording_id: str, payload: AnnotationSizeInput):
    get_recording(recording_id)
    with db.transaction() as connection:
        connection.execute(
            """INSERT INTO annotation_size_presets (recording_id, object_class, width, height)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(recording_id, object_class) DO UPDATE SET
               width=excluded.width, height=excluded.height, updated_at=CURRENT_TIMESTAMP""",
            (recording_id, payload.object_class, payload.width, payload.height),
        )
    return {"recording_id": recording_id, **payload.model_dump()}


@app.delete("/api/recordings/{recording_id}/annotation-size/{object_class}")
def reset_annotation_size(recording_id: str, object_class: str):
    get_recording(recording_id)
    if object_class not in {"basketball", "hoop"}:
        raise HTTPException(422, "Unknown annotation class")
    with db.transaction() as connection:
        connection.execute(
            "DELETE FROM annotation_size_presets WHERE recording_id=? AND object_class=?",
            (recording_id, object_class),
        )
    return {"recording_id": recording_id, "object_class": object_class, "reset": True}


@app.get("/api/recordings/{recording_id}/media")
def media(request: Request, recording_id: str):
    recording = get_recording(recording_id)
    if recording["availability"] != "present":
        raise HTTPException(404, "Recording is unavailable")
    path = resolve_known_media(settings.media_root, recording["media_path"])
    stat = path.stat()
    if stat.st_size != recording["size_bytes"] or stat.st_mtime_ns != recording["mtime_ns"]:
        raise HTTPException(409, "Recording changed; run a new scan")
    return stream_media(request, path)


@app.get("/api/recordings/{recording_id}/thumbnail")
def thumbnail(recording_id: str):
    recording = get_recording(recording_id)
    if recording["availability"] != "present":
        raise HTTPException(404, "Recording is unavailable")
    source = resolve_known_media(settings.media_root, recording["media_path"])
    stat = source.stat()
    if stat.st_size != recording["size_bytes"] or stat.st_mtime_ns != recording["mtime_ns"]:
        raise HTTPException(409, "Recording changed; run a new scan")
    output = thumbnail_path(
        settings.config_root,
        recording_id,
        recording["size_bytes"],
        recording["mtime_ns"],
    )
    try:
        generate_thumbnail(source, output, recording["duration_seconds"])
    except (OSError, subprocess.SubprocessError):
        logger.exception("Thumbnail generation failed for recording %s", recording_id)
        raise HTTPException(503, "Thumbnail generation failed") from None
    return FileResponse(output, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=86400"})


@app.get("/api/annotation-frames/{frame_id}/image")
def annotation_frame_image(frame_id: str):
    with db.connect() as connection:
        frame = connection.execute(
            """SELECT f.*, e.recording_id, r.media_path, r.size_bytes, r.mtime_ns, r.availability
               FROM annotation_frames f JOIN events e ON e.event_id=f.event_id
               JOIN recordings r ON r.recording_id=e.recording_id WHERE f.frame_id=?""",
            (frame_id,),
        ).fetchone()
    if not frame or frame["availability"] != "present":
        raise HTTPException(404, "Annotation frame not found")
    source = resolve_known_media(settings.media_root, frame["media_path"])
    stat = source.stat()
    if stat.st_size != frame["size_bytes"] or stat.st_mtime_ns != frame["mtime_ns"]:
        raise HTTPException(409, "Recording changed; run a new scan")
    output = annotation_frame_path(settings.config_root, frame["event_id"], frame["frame_index"])
    try:
        extract_annotation_frame(source, output, frame["frame_time_seconds"])
    except (OSError, subprocess.SubprocessError):
        logger.exception("Annotation frame extraction failed for frame %s", frame_id)
        raise HTTPException(503, "Frame extraction failed") from None
    return FileResponse(output, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=86400"})


@app.put("/api/annotation-frames/{frame_id}/boxes")
def save_annotation_boxes(frame_id: str, payload: FrameBoxesInput):
    with db.transaction() as connection:
        frame = connection.execute("SELECT * FROM annotation_frames WHERE frame_id=?", (frame_id,)).fetchone()
        if not frame:
            raise HTTPException(404, "Annotation frame not found")
        connection.execute("DELETE FROM annotation_boxes WHERE frame_id=?", (frame_id,))
        for box in payload.boxes:
            connection.execute(
                """INSERT INTO annotation_boxes
                   (box_id, frame_id, object_class, x_center, y_center, width, height, occluded)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (str(uuid.uuid4()), frame_id, box.object_class, box.x_center, box.y_center,
                 box.width, box.height, int(box.occluded)),
            )
        connection.execute(
            "UPDATE annotation_frames SET review_status=? WHERE frame_id=?",
            (payload.review_status, frame_id),
        )
    return {"frame_id": frame_id, "saved": len(payload.boxes), "review_status": payload.review_status}


@app.put("/api/events/{event_id}/outcome")
def update_sequence_outcome(event_id: str, payload: SequenceOutcomeInput):
    get_event(event_id)
    with db.transaction() as connection:
        connection.execute(
            "UPDATE events SET sequence_outcome=?, updated_at=CURRENT_TIMESTAMP WHERE event_id=?",
            (payload.sequence_outcome, event_id),
        )
    return {"event_id": event_id, "sequence_outcome": payload.sequence_outcome}


@app.post("/api/events/{event_id}/detailed-frames", status_code=201)
def create_detailed_frames(event_id: str, payload: DetailedFramesInput):
    event = get_event(event_id)
    recording = get_recording(event["recording_id"])
    duration = float(recording["duration_seconds"] or event["play_until_seconds"])
    center = min(duration, payload.center_time_seconds)
    candidate_times = detailed_frame_times(center, duration)
    created = 0
    detail_group_id = str(uuid.uuid4())
    with db.transaction() as connection:
        existing = connection.execute(
            "SELECT frame_time_seconds FROM annotation_frames WHERE event_id=?", (event_id,)
        ).fetchall()
        existing_times = [float(row["frame_time_seconds"]) for row in existing]
        maximum = connection.execute(
            "SELECT MAX(frame_index) AS value FROM annotation_frames WHERE event_id=? AND frame_index>=1000",
            (event_id,),
        ).fetchone()["value"]
        next_index = max(1000, (maximum or 999) + 1)
        for frame_time in candidate_times:
            if any(abs(frame_time - saved_time) < 0.001 for saved_time in existing_times):
                continue
            connection.execute(
                """INSERT INTO annotation_frames
                   (frame_id, event_id, frame_time_seconds, frame_index, detail_group_id)
                   VALUES (?, ?, ?, ?, ?)""",
                (str(uuid.uuid4()), event_id, frame_time, next_index, detail_group_id),
            )
            existing_times.append(frame_time)
            next_index += 1
            created += 1
    return {"created": created, "center_time_seconds": center, "detail_group_id": detail_group_id if created else None}


@app.delete("/api/annotation-frames/{frame_id}/detail-group")
def remove_detailed_frame_group(frame_id: str):
    with db.transaction() as connection:
        selected = connection.execute(
            "SELECT * FROM annotation_frames WHERE frame_id=?", (frame_id,)
        ).fetchone()
        if not selected:
            raise HTTPException(404, "Annotation frame not found")
        if selected["frame_index"] < 1000:
            raise HTTPException(409, "Original broad frames cannot be removed")
        if selected["detail_group_id"]:
            frames = connection.execute(
                "SELECT * FROM annotation_frames WHERE event_id=? AND detail_group_id=?",
                (selected["event_id"], selected["detail_group_id"]),
            ).fetchall()
        else:
            frames = connection.execute(
                """SELECT * FROM annotation_frames
                   WHERE event_id=? AND frame_index>=1000 AND detail_group_id IS NULL""",
                (selected["event_id"],),
            ).fetchall()
        for removable in frames:
            connection.execute("DELETE FROM annotation_frames WHERE frame_id=?", (removable["frame_id"],))
    for removable in frames:
        annotation_frame_path(
            settings.config_root, removable["event_id"], removable["frame_index"]
        ).unlink(missing_ok=True)
    return {"removed": len(frames)}


@app.get("/api/events/{event_id}/annotations/export")
def export_annotations(event_id: str):
    event = get_event(event_id)
    recording = get_recording(event["recording_id"])
    ensure_annotation_frames(event_id)
    with db.connect() as connection:
        frames = connection.execute(
            "SELECT * FROM annotation_frames WHERE event_id=? ORDER BY frame_time_seconds, frame_index", (event_id,)
        ).fetchall()
        boxes = connection.execute(
            """SELECT b.* FROM annotation_boxes b JOIN annotation_frames f ON f.frame_id=b.frame_id
               WHERE f.event_id=? ORDER BY f.frame_index, b.created_at""", (event_id,)
        ).fetchall()
    boxes_by_frame: dict[str, list[dict]] = {row["frame_id"]: [] for row in frames}
    for box in boxes:
        boxes_by_frame[box["frame_id"]].append({
            "class": box["object_class"], "x_center": box["x_center"], "y_center": box["y_center"],
            "width": box["width"], "height": box["height"], "occluded": bool(box["occluded"]),
        })
    document = {
        "schema_version": 1,
        "recording": {"recording_id": recording["recording_id"], "size_bytes": recording["size_bytes"]},
        "event": {"event_id": event_id, "event_type": event["event_type"], "event_time_seconds": event["event_time_seconds"],
                  "sequence_outcome": event["sequence_outcome"]},
        "frames": [
            {"frame_index": row["frame_index"], "frame_time_seconds": row["frame_time_seconds"],
             "review_status": row["review_status"],
             "boxes": boxes_by_frame[row["frame_id"]]} for row in frames
        ],
    }
    export_dir = settings.export_root / recording["recording_id"] / "annotations"
    export_dir.mkdir(parents=True, exist_ok=True)
    output = export_dir / f"{event_id}.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    return FileResponse(output, filename=f"{event_id}-annotations.json", media_type="application/json")


@app.get("/api/recordings/{recording_id}/events")
def list_events(recording_id: str):
    get_recording(recording_id)
    with db.connect() as connection:
        rows = connection.execute(
            "SELECT * FROM events WHERE recording_id=? ORDER BY event_time_seconds", (recording_id,)
        ).fetchall()
    return [dict(row) for row in rows]


@app.post("/api/recordings/{recording_id}/events", status_code=201)
def create_event(recording_id: str, event: EventInput):
    recording = get_recording(recording_id)
    duration = recording["duration_seconds"]
    if duration is not None and event.play_until_seconds > duration + 0.01:
        raise HTTPException(422, "Playback window exceeds recording duration")
    event_id = str(uuid.uuid4())
    sequence_outcome = event.sequence_outcome or ("made" if event.event_type == "made_basket" else "uncertain")
    with db.transaction() as connection:
        connection.execute(
            """INSERT INTO events
                (event_id, recording_id, event_type, event_time_seconds, play_from_seconds,
                 play_until_seconds, review_status, source, reviewed_at, sequence_outcome)
                VALUES (?, ?, ?, ?, ?, ?, 'approved', 'manual', CURRENT_TIMESTAMP, ?)""",
            (event_id, recording_id, event.event_type, event.event_time_seconds,
             event.play_from_seconds, event.play_until_seconds, sequence_outcome),
        )
    return row_dict(get_event(event_id))


@app.put("/api/events/{event_id}")
def update_event(event_id: str, event: EventInput):
    existing = get_event(event_id)
    recording = get_recording(existing["recording_id"])
    if recording["duration_seconds"] is not None and event.play_until_seconds > recording["duration_seconds"] + 0.01:
        raise HTTPException(422, "Playback window exceeds recording duration")
    content_changed = (
        event.event_type != existing["event_type"]
        or abs(event.event_time_seconds - existing["event_time_seconds"]) > .001
        or abs(event.play_from_seconds - existing["play_from_seconds"]) > .001
        or abs(event.play_until_seconds - existing["play_until_seconds"]) > .001
    )
    source = "corrected" if existing["source"] == "model" and content_changed else existing["source"]
    sequence_outcome = event.sequence_outcome or existing["sequence_outcome"]
    review_status = "approved" if existing["source"] == "manual" else event.review_status
    with db.transaction() as connection:
        connection.execute(
            """UPDATE events SET event_type=?, event_time_seconds=?, play_from_seconds=?,
               play_until_seconds=?, review_status=?, source=?, sequence_outcome=?, updated_at=CURRENT_TIMESTAMP,
               reviewed_at=CASE WHEN ?='candidate' THEN NULL ELSE CURRENT_TIMESTAMP END
               WHERE event_id=?""",
            (event.event_type, event.event_time_seconds, event.play_from_seconds,
             event.play_until_seconds, review_status, source, sequence_outcome,
             review_status, event_id),
        )
    return row_dict(get_event(event_id))


@app.delete("/api/events/{event_id}", status_code=204)
def delete_event(event_id: str):
    existing = get_event(event_id)
    if existing["source"] == "model":
        raise HTTPException(409, "Model events must be rejected, not deleted")
    with db.transaction() as connection:
        connection.execute("DELETE FROM events WHERE event_id=?", (event_id,))


@app.delete("/api/recordings/{recording_id}/events")
def delete_recording_events(recording_id: str):
    get_recording(recording_id)
    with db.transaction() as connection:
        count = connection.execute(
            "SELECT COUNT(*) FROM events WHERE recording_id=?", (recording_id,)
        ).fetchone()[0]
        connection.execute("DELETE FROM events WHERE recording_id=?", (recording_id,))
    return {"recording_id": recording_id, "deleted_count": count}


@app.get("/api/recordings/{recording_id}/export/json")
def export_json(recording_id: str):
    recording = get_recording(recording_id)
    with db.connect() as connection:
        events = connection.execute(
            "SELECT * FROM events WHERE recording_id=? ORDER BY event_time_seconds", (recording_id,)
        ).fetchall()
    export_dir = settings.export_root / recording_id
    export_dir.mkdir(parents=True, exist_ok=True)
    output = export_dir / "chapters.json"
    temporary = output.with_suffix(".json.tmp")
    document = {
        "schema_version": 1,
        "recording": {
            "recording_id": recording_id,
            "title": recording["title"],
            "duration_seconds": recording["duration_seconds"],
            "size_bytes": recording["size_bytes"],
        },
        "events": [dict(row) for row in events],
    }
    temporary.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    return FileResponse(output, filename=f"{recording_id}-chapters.json", media_type="application/json")
