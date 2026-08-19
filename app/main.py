from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, model_validator

from . import __version__
from .annotations import FRAME_OFFSETS, annotation_frame_path, detailed_frame_times, extract_annotation_frame
from .catalog import scan_media
from .config import Settings
from .db import Database
from .media import resolve_known_media, stream_media
from .thumbnails import generate_thumbnail, thumbnail_path

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


@app.get("/health")
def health():
    with db.connect() as connection:
        connection.execute("SELECT 1").fetchone()
    return {"status": "ok", "version": __version__}


@app.get("/", response_class=HTMLResponse)
def library(request: Request):
    with db.connect() as connection:
        recordings = connection.execute(
            """SELECT r.*, COUNT(e.event_id) AS event_count
               FROM recordings r LEFT JOIN events e ON e.recording_id=r.recording_id
               GROUP BY r.recording_id ORDER BY r.created_at DESC"""
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
    return templates.TemplateResponse(
        request,
        "recording.html",
        {
            "recording": recording,
            "events_json": json.dumps([dict(row) for row in events]),
            "before": settings.default_before_seconds,
            "after": settings.default_after_seconds,
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
    boxes_by_frame: dict[str, list[dict]] = {row["frame_id"]: [] for row in frames}
    for box in boxes:
        boxes_by_frame[box["frame_id"]].append(dict(box))
    frame_data = [dict(row) | {"boxes": boxes_by_frame[row["frame_id"]]} for row in frames]
    return templates.TemplateResponse(
        request,
        "annotate.html",
        {"event": event, "recording": recording, "frames_json": json.dumps(frame_data)},
    )


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
    reviewed_at = "CURRENT_TIMESTAMP" if event.review_status != "candidate" else "NULL"
    with db.transaction() as connection:
        connection.execute(
            f"""INSERT INTO events
                (event_id, recording_id, event_type, event_time_seconds, play_from_seconds,
                 play_until_seconds, review_status, source, reviewed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'manual', {reviewed_at})""",
            (event_id, recording_id, event.event_type, event.event_time_seconds,
             event.play_from_seconds, event.play_until_seconds, event.review_status),
        )
    return row_dict(get_event(event_id))


@app.put("/api/events/{event_id}")
def update_event(event_id: str, event: EventInput):
    existing = get_event(event_id)
    recording = get_recording(existing["recording_id"])
    if recording["duration_seconds"] is not None and event.play_until_seconds > recording["duration_seconds"] + 0.01:
        raise HTTPException(422, "Playback window exceeds recording duration")
    source = "corrected" if existing["source"] == "model" else existing["source"]
    with db.transaction() as connection:
        connection.execute(
            """UPDATE events SET event_type=?, event_time_seconds=?, play_from_seconds=?,
               play_until_seconds=?, review_status=?, source=?, updated_at=CURRENT_TIMESTAMP,
               reviewed_at=CASE WHEN ?='candidate' THEN NULL ELSE CURRENT_TIMESTAMP END
               WHERE event_id=?""",
            (event.event_type, event.event_time_seconds, event.play_from_seconds,
             event.play_until_seconds, event.review_status, source, event.review_status, event_id),
        )
    return row_dict(get_event(event_id))


@app.delete("/api/events/{event_id}", status_code=204)
def delete_event(event_id: str):
    existing = get_event(event_id)
    if existing["source"] == "model":
        raise HTTPException(409, "Model events must be rejected, not deleted")
    with db.transaction() as connection:
        connection.execute("DELETE FROM events WHERE event_id=?", (event_id,))


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
