from __future__ import annotations

import subprocess
from pathlib import Path


FRAME_OFFSETS = (-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0)


def annotation_frame_path(config_root: Path, event_id: str, frame_index: int) -> Path:
    directory = config_root / "annotation-frames" / event_id
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"frame-{frame_index:02d}.jpg"


def detailed_frame_times(center_seconds: float, duration_seconds: float) -> list[float]:
    center = max(0.0, min(duration_seconds, center_seconds))
    return [max(0.0, min(duration_seconds, center + offset / 10)) for offset in range(-5, 6)]


def extract_annotation_frame(source: Path, output: Path, time_seconds: float) -> Path:
    if output.is_file():
        return output
    temporary = output.with_suffix(".jpg.tmp")
    try:
        subprocess.run(
            [
                "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                "-ss", f"{time_seconds:.3f}", "-i", str(source), "-frames:v", "1",
                "-vf", "scale=1280:-2", "-q:v", "3", "-f", "image2", str(temporary),
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


def clamp_box(x1: float, y1: float, x2: float, y2: float) -> dict[str, float]:
    left, right = sorted((max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))))
    top, bottom = sorted((max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))))
    return {
        "x_center": (left + right) / 2,
        "y_center": (top + bottom) / 2,
        "width": right - left,
        "height": bottom - top,
    }
