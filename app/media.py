from __future__ import annotations

import re
from pathlib import Path

from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse


RANGE = re.compile(r"^bytes=(\d*)-(\d*)$")


def resolve_known_media(media_root: Path, relative_path: str) -> Path:
    if not relative_path or Path(relative_path).is_absolute():
        raise HTTPException(404)
    try:
        path = (media_root / relative_path).resolve(strict=True)
        path.relative_to(media_root)
    except (OSError, ValueError):
        raise HTTPException(404) from None
    if not path.is_file():
        raise HTTPException(404)
    return path


def _parse_range(value: str | None, size: int) -> tuple[int, int, int]:
    if not value:
        return 0, size - 1, 200
    match = RANGE.fullmatch(value.strip())
    if not match or "," in value:
        raise HTTPException(416, headers={"Content-Range": f"bytes */{size}"})
    first, last = match.groups()
    if not first and not last:
        raise HTTPException(416, headers={"Content-Range": f"bytes */{size}"})
    if not first:
        length = int(last)
        if length <= 0:
            raise HTTPException(416, headers={"Content-Range": f"bytes */{size}"})
        start = max(0, size - length)
        end = size - 1
    else:
        start = int(first)
        end = min(int(last), size - 1) if last else size - 1
    if start >= size or end < start:
        raise HTTPException(416, headers={"Content-Range": f"bytes */{size}"})
    return start, end, 206


def stream_media(request: Request, path: Path) -> StreamingResponse:
    size = path.stat().st_size
    start, end, status = _parse_range(request.headers.get("range"), size)
    length = end - start + 1

    def content():
        remaining = length
        with path.open("rb") as handle:
            handle.seek(start)
            while remaining:
                chunk = handle.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    headers = {
        "Accept-Ranges": "bytes",
        "Content-Length": str(length),
        "Cache-Control": "private, no-store",
    }
    if status == 206:
        headers["Content-Range"] = f"bytes {start}-{end}/{size}"
    content_type = "video/mp4" if path.suffix.lower() in {".mp4", ".m4v"} else "video/x-matroska"
    return StreamingResponse(content(), status_code=status, headers=headers, media_type=content_type)

