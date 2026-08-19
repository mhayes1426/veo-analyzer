from pathlib import Path

import pytest
from fastapi import HTTPException

from app.media import _parse_range, resolve_known_media


def test_resolves_known_file(roots):
    media, _, _ = roots
    game = media / "game.mp4"
    game.write_bytes(b"video")
    assert resolve_known_media(media.resolve(), "game.mp4") == game.resolve()


def test_rejects_escape(roots):
    media, _, _ = roots
    outside = media.parent / "secret.mp4"
    outside.write_bytes(b"secret")
    with pytest.raises(HTTPException) as error:
        resolve_known_media(media.resolve(), "../secret.mp4")
    assert error.value.status_code == 404


def test_rejects_symlink_escape(roots):
    media, _, _ = roots
    outside = media.parent / "secret.mp4"
    outside.write_bytes(b"secret")
    (media / "link.mp4").symlink_to(outside)
    with pytest.raises(HTTPException):
        resolve_known_media(media.resolve(), "link.mp4")


@pytest.mark.parametrize(("header", "expected"), [
    (None, (0, 99, 200)),
    ("bytes=0-9", (0, 9, 206)),
    ("bytes=90-", (90, 99, 206)),
    ("bytes=-10", (90, 99, 206)),
])
def test_ranges(header, expected):
    assert _parse_range(header, 100) == expected


@pytest.mark.parametrize("header", ["bytes=", "bytes=100-101", "bytes=20-10", "bytes=0-1,4-5"])
def test_bad_ranges(header):
    with pytest.raises(HTTPException) as error:
        _parse_range(header, 100)
    assert error.value.status_code == 416

