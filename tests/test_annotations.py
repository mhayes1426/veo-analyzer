import pytest

from app.annotations import FRAME_OFFSETS, clamp_box, detailed_frame_times


def test_frame_offsets_center_on_event():
    assert len(FRAME_OFFSETS) == 9
    assert FRAME_OFFSETS[0] == -2.0
    assert FRAME_OFFSETS[4] == 0.0
    assert FRAME_OFFSETS[-1] == 2.0


def test_clamp_box_normalizes_reversed_points():
    box = clamp_box(0.8, 0.7, 0.2, 0.1)
    assert box == pytest.approx({"x_center": 0.5, "y_center": 0.4, "width": 0.6, "height": 0.6})


def test_clamp_box_stays_in_frame():
    box = clamp_box(-1, -1, 2, 2)
    assert box == {"x_center": 0.5, "y_center": 0.5, "width": 1.0, "height": 1.0}


def test_detailed_frames_are_tenth_second_samples():
    assert detailed_frame_times(2.54, 10) == pytest.approx([
        2.04, 2.14, 2.24, 2.34, 2.44, 2.54, 2.64, 2.74, 2.84, 2.94, 3.04,
    ])


def test_detailed_frames_clamp_to_media_bounds():
    times = detailed_frame_times(0.1, 1.0)
    assert min(times) == 0
    assert max(times) == pytest.approx(0.6)
