import uuid

from app.analysis_jobs import (_create_candidate_events, _detections, frame_explanation, frame_times,
                               made_basket_candidates, rim_observations)
from app.config import Settings
from app.db import Database


class Scalar:
    def __init__(self, value): self.value = value
    def item(self): return self.value


class Vector:
    def __init__(self, values): self.values = values
    def tolist(self): return self.values


class Box:
    cls = [Scalar(1)]
    conf = [Scalar(.91)]
    xyxyn = [Vector([.1, .2, .3, .4])]


class Result:
    names = {0: "basketball", 1: "hoop"}
    boxes = [Box()]


def test_frame_times_include_end_without_exceeding_it():
    assert frame_times(10, 11, .5) == [10, 10.5, 11]
    assert frame_times(10, 10.8, .5) == [10, 10.5, 10.8]


def test_detections_are_privacy_safe_normalized_values():
    assert _detections(Result()) == [{
        "class": "hoop", "confidence": .91, "x1": .1, "y1": .2, "x2": .3, "y2": .4,
    }]


def detection(object_class, x1, y1, x2, y2, confidence=.9):
    return {"class": object_class, "confidence": confidence, "x1": x1, "y1": y1, "x2": x2, "y2": y2}


def trajectory(ball_x, ball_y_values, start=10):
    frames = []
    for index, ball_y in enumerate(ball_y_values):
        frames.append({
            "time_seconds": start + index * .2,
            "detections": [
                detection("hoop", .4, .4, .5, .44),
                detection("basketball", ball_x - .01, ball_y - .01, ball_x + .01, ball_y + .01),
            ],
        })
    return frames


def test_rim_crossing_creates_explainable_made_basket_candidate():
    frames = trajectory(.45, [.37, .39, .43, .46])
    candidates = made_basket_candidates(frames, 1.0)
    assert len(candidates) == 1
    assert 10 < candidates[0]["event_time_seconds"] < 10.6
    assert candidates[0]["confidence"] > .6
    assert candidates[0]["above"]["dy_rim_heights"] < 0
    assert candidates[0]["below"]["dy_rim_heights"] > 0


def test_ball_outside_rim_does_not_create_candidate():
    frames = trajectory(.62, [.37, .39, .43, .46])
    assert rim_observations(frames) == []
    assert made_basket_candidates(frames, 1.0) == []


def test_calibration_frame_explains_above_rim_state():
    explanation = frame_explanation(trajectory(.45, [.37])[0])
    assert explanation["state"] == "above_rim"
    assert "waiting" in explanation["reason"]


def test_crossings_within_three_seconds_are_deduplicated():
    frames = trajectory(.45, [.37, .46], 10) + trajectory(.45, [.37, .46], 11)
    assert len(made_basket_candidates(frames, 1.0)) == 1


def test_reanalysis_replaces_only_unreviewed_lineage_candidates(roots):
    media, config, exports = roots
    database = Database(config / "analyzer.db")
    database.initialize()
    recording_id, training_id, first_job, second_job = (str(uuid.uuid4()) for _ in range(4))
    settings = Settings(media, config, exports, config / "analyzer.db", 300, 0, 8, 5)
    with database.transaction() as connection:
        connection.execute(
            """INSERT INTO recordings
               (recording_id,media_path,title,size_bytes,mtime_ns,duration_seconds) VALUES (?,?,?,?,?,?)""",
            (recording_id, "game.mp4", "Game", 1, 1, 100),
        )
        connection.execute(
            """INSERT INTO training_jobs
               (job_id,status,model_name,epochs,image_size,batch_size,device,output_dir)
               VALUES (?,?,?,?,?,?,?,?)""",
            (training_id, "completed", "yolo11n.pt", 1, 640, 8, "0", str(config / "training")),
        )
        for job_id in (first_job, second_job):
            connection.execute(
                """INSERT INTO analysis_jobs
                   (job_id,recording_id,training_job_id,status,mode,start_seconds,end_seconds,
                    sample_interval_seconds,confidence_threshold,output_dir)
                   VALUES (?,?,?,'completed','full_game',0,100,.2,.25,?)""",
                (job_id, recording_id, training_id, str(config / "analysis" / job_id)),
            )
        connection.execute(
            """INSERT INTO events
               (event_id,recording_id,event_type,event_time_seconds,play_from_seconds,play_until_seconds,
                review_status,source) VALUES (?,?,'made_basket',10,2,15,'approved','manual')""",
            (str(uuid.uuid4()), recording_id),
        )
    candidates = [
        {"event_time_seconds": 10, "confidence": .9},
        {"event_time_seconds": 20, "confidence": .8},
    ]
    job = {"job_id": first_job, "recording_id": recording_id, "training_job_id": training_id,
           "model_name": "yolo11n.pt", "start_seconds": 0, "end_seconds": 100, "duration_seconds": 100}
    created = _create_candidate_events(database, settings, job, candidates)
    assert len(created) == 1
    job["job_id"] = second_job
    created = _create_candidate_events(database, settings, job, [candidates[1]])
    assert len(created) == 1
    with database.connect() as connection:
        rows = connection.execute("SELECT * FROM events ORDER BY event_time_seconds").fetchall()
    assert len(rows) == 2
    assert rows[0]["review_status"] == "approved"
    assert rows[1]["analysis_job_id"] == second_job
