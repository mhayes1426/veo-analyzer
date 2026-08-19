from __future__ import annotations

import hashlib
from collections import Counter
from typing import Iterable, Mapping


CLASS_IDS = {"basketball": 0, "hoop": 1}


def recording_split(recording_id: str) -> str:
    bucket = int(hashlib.sha256(recording_id.encode()).hexdigest()[:8], 16) % 10
    return "train" if bucket < 7 else "val" if bucket < 9 else "test"


def yolo_label(box: Mapping[str, object]) -> str:
    return " ".join(
        [str(CLASS_IDS[str(box["object_class"])])] +
        [f"{float(box[key]):.8f}" for key in ("x_center", "y_center", "width", "height")]
    )


def quality_report(frames: Iterable[Mapping[str, object]]) -> dict:
    rows = list(frames)
    statuses = Counter(str(row["review_status"]) for row in rows)
    warnings: list[dict[str, str]] = []
    for row in rows:
        boxes = list(row.get("boxes", []))
        classes = {str(box["object_class"]) for box in boxes}
        if row["review_status"] == "reviewed" and "hoop" not in classes:
            warnings.append({"code": "missing_hoop", "frame_id": str(row["frame_id"]), "message": "Reviewed frame has no hoop box."})
        if row["review_status"] == "reviewed" and not boxes:
            warnings.append({"code": "reviewed_empty", "frame_id": str(row["frame_id"]), "message": "Reviewed frame is intentionally empty; verify the scoring hoop is not visible."})
    outcomes_by_event = {
        str(row.get("event_id", row["frame_id"])): str(row["sequence_outcome"]) for row in rows
    }
    outcomes = Counter(outcomes_by_event.values())
    return {
        "total_frames": len(rows),
        "reviewed": statuses["reviewed"],
        "pending": statuses["pending"],
        "skipped": statuses["skipped"],
        "exportable": statuses["reviewed"],
        "outcomes": dict(outcomes),
        "warnings": warnings,
    }
