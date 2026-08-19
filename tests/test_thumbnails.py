from pathlib import Path

from app.thumbnails import thumbnail_path


def test_thumbnail_cache_key_changes_with_source(roots):
    _, config, _ = roots
    first = thumbnail_path(config, "recording-id", 100, 200)
    second = thumbnail_path(config, "recording-id", 101, 200)
    assert first != second
    assert first.parent == config / "thumbnails"
    assert first.name == "recording-id-100-200.jpg"
