from __future__ import annotations

import json
import subprocess
import time
import uuid
from pathlib import Path

from .db import Database


MEDIA_EXTENSIONS = {".mp4", ".m4v", ".mov", ".mkv"}


def probe_duration(path: Path) -> float | None:
    try:
        completed = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "json", str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        value = json.loads(completed.stdout)["format"].get("duration")
        return float(value) if value is not None else None
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, json.JSONDecodeError):
        return None


def scan_media(db: Database, media_root: Path, stable_age_seconds: int) -> int:
    if not media_root.is_dir():
        return 0

    now = time.time()
    found: set[str] = set()
    count = 0
    for path in media_root.rglob("*"):
        try:
            if path.suffix.lower() not in MEDIA_EXTENSIONS or not path.is_file() or path.is_symlink():
                continue
            stat = path.stat()
            if now - stat.st_mtime < stable_age_seconds:
                continue
            resolved = path.resolve(strict=True)
            relative = resolved.relative_to(media_root).as_posix()
        except (OSError, ValueError):
            continue

        found.add(relative)
        with db.transaction() as connection:
            row = connection.execute(
                "SELECT recording_id, size_bytes, mtime_ns FROM recordings WHERE media_path = ?",
                (relative,),
            ).fetchone()
            if row and row["size_bytes"] == stat.st_size and row["mtime_ns"] == stat.st_mtime_ns:
                connection.execute(
                    "UPDATE recordings SET availability='present', last_seen_at=CURRENT_TIMESTAMP WHERE recording_id=?",
                    (row["recording_id"],),
                )
            else:
                duration = probe_duration(resolved)
                if duration is None:
                    continue
                if row:
                    connection.execute(
                        """UPDATE recordings SET title=?, size_bytes=?, mtime_ns=?, duration_seconds=?,
                           availability='present', updated_at=CURRENT_TIMESTAMP, last_seen_at=CURRENT_TIMESTAMP
                           WHERE recording_id=?""",
                        (path.stem, stat.st_size, stat.st_mtime_ns, duration, row["recording_id"]),
                    )
                else:
                    connection.execute(
                        """INSERT INTO recordings
                           (recording_id, media_path, title, size_bytes, mtime_ns, duration_seconds)
                           VALUES (?, ?, ?, ?, ?, ?)""",
                        (str(uuid.uuid4()), relative, path.stem, stat.st_size, stat.st_mtime_ns, duration),
                    )
            count += 1

    with db.transaction() as connection:
        rows = connection.execute("SELECT recording_id, media_path FROM recordings").fetchall()
        for row in rows:
            if row["media_path"] not in found:
                connection.execute(
                    "UPDATE recordings SET availability='missing', updated_at=CURRENT_TIMESTAMP WHERE recording_id=?",
                    (row["recording_id"],),
                )
    return count

