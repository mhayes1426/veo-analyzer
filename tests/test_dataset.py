from app.dataset import quality_report, recording_split, yolo_label


def test_split_is_stable_by_recording():
    assert recording_split("recording-a") == recording_split("recording-a")
    assert recording_split("recording-a") in {"train", "val", "test"}


def test_yolo_label_uses_expected_class_order():
    assert yolo_label({"object_class": "basketball", "x_center": .5, "y_center": .4, "width": .1, "height": .2}) == "0 0.50000000 0.40000000 0.10000000 0.20000000"
    assert yolo_label({"object_class": "hoop", "x_center": .5, "y_center": .4, "width": .1, "height": .2}).startswith("1 ")


def test_quality_report_distinguishes_review_states():
    frames = [
        {"frame_id": "a", "review_status": "reviewed", "sequence_outcome": "made", "boxes": [{"object_class": "hoop"}]},
        {"frame_id": "b", "review_status": "pending", "sequence_outcome": "made", "boxes": []},
        {"frame_id": "c", "review_status": "skipped", "sequence_outcome": "missed", "boxes": []},
    ]
    report = quality_report(frames)
    assert report["exportable"] == 1
    assert report["pending"] == 1
    assert report["skipped"] == 1
    assert report["warnings"] == []


def test_quality_report_warns_on_reviewed_empty_frame():
    report = quality_report([{"frame_id": "a", "review_status": "reviewed", "sequence_outcome": "uncertain", "boxes": []}])
    assert {warning["code"] for warning in report["warnings"]} == {"missing_hoop", "reviewed_empty"}
