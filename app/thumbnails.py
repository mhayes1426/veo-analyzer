from __future__ import annotations

import subprocess
from pathlib import Path


def thumbnail_path(config_root: Path, recording_id: str, size_bytes: int, mtime_ns: int) -> Path:
    cache = config_root / "thumbnails"
    cache.mkdir(parents=True, exist_ok=True)
    return cache / f"{recording_id}-{size_bytes}-{mtime_ns}.jpg"


def generate_thumbnail(source: Path, output: Path, duration_seconds: float | None) -> Path:
    if output.is_file():
        return output

    seek_seconds = max(1.0, min(60.0, (duration_seconds or 30.0) * 0.08))
    temporary = output.with_suffix(".jpg.tmp")
    try:
        subprocess.run(
            [
                "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                "-ss", f"{seek_seconds:.3f}", "-i", str(source),
                "-frames:v", "1", "-vf", "scale=640:-2", "-q:v", "4",
                "-f", "image2", str(temporary),
            ],
            check=True,
            capture_output=True,
            timeout=90,
        )
        temporary.replace(output)
    except (OSError, subprocess.SubprocessError):
        temporary.unlink(missing_ok=True)
        raise
    return output

