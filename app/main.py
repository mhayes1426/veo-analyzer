from __future__ import annotations

import asyncio
import json
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, model_validator

from . import __version__
from .catalog import scan_media
from .config import Settings
from .db import Database
from .media import resolve_known_media, stream_media

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

