import pytest

from app.db import Database


def test_schema_initializes(roots):
    _, config, _ = roots
    db = Database(config / "analyzer.db")
    db.initialize()
    with db.connect() as connection:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"recordings", "events", "annotation_frames", "annotation_boxes", "annotation_size_presets",
            "training_jobs", "analysis_jobs", "analysis_results"} <= tables
    with db.connect() as connection:
        event_columns = {row[1] for row in connection.execute("PRAGMA table_info(events)")}
        frame_columns = {row[1] for row in connection.execute("PRAGMA table_info(annotation_frames)")}
        analysis_columns = {row[1] for row in connection.execute("PRAGMA table_info(analysis_jobs)")}
        result_columns = {row[1] for row in connection.execute("PRAGMA table_info(analysis_results)")}
    assert "sequence_outcome" in event_columns
    assert "analysis_job_id" in event_columns
    assert {"review_status", "detail_group_id"} <= frame_columns
    assert {"mode", "crossing_window_seconds", "candidate_count", "noisy_frame_count", "quality_status",
            "cancel_requested"} <= analysis_columns
    assert {"explanation_json", "candidate_event_id"} <= result_columns


def test_annotation_box_constraints(roots):
    import sqlite3
    import uuid

    _, config, _ = roots
    db = Database(config / "analyzer.db")
    db.initialize()
    with db.transaction() as connection:
        recording_id, event_id, frame_id = (str(uuid.uuid4()) for _ in range(3))
        connection.execute(
            "INSERT INTO recordings (recording_id,media_path,title,size_bytes,mtime_ns) VALUES (?,?,?,?,?)",
            (recording_id, "game.mp4", "Game", 1, 1),
        )
        connection.execute(
            """INSERT INTO events
               (event_id,recording_id,event_type,event_time_seconds,play_from_seconds,play_until_seconds,review_status,source)
               VALUES (?,?,?,?,?,?,?,?)""",
            (event_id, recording_id, "made_basket", 10, 8, 12, "approved", "manual"),
        )
        connection.execute(
            "INSERT INTO annotation_frames (frame_id,event_id,frame_time_seconds,frame_index) VALUES (?,?,?,?)",
            (frame_id, event_id, 10, 0),
        )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """INSERT INTO annotation_boxes
                   (box_id,frame_id,object_class,x_center,y_center,width,height) VALUES (?,?,?,?,?,?,?)""",
                (str(uuid.uuid4()), frame_id, "player", .5, .5, .1, .1),
            )


def test_annotation_size_presets_are_per_recording_and_class(roots):
    import sqlite3
    import uuid

    _, config, _ = roots
    db = Database(config / "analyzer.db")
    db.initialize()
    recording_id = str(uuid.uuid4())
    with db.transaction() as connection:
        connection.execute(
            "INSERT INTO recordings (recording_id,media_path,title,size_bytes,mtime_ns) VALUES (?,?,?,?,?)",
            (recording_id, "game.mp4", "Game", 1, 1),
        )
        connection.execute(
            "INSERT INTO annotation_size_presets (recording_id,object_class,width,height) VALUES (?,?,?,?)",
            (recording_id, "basketball", .03, .04),
        )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO annotation_size_presets (recording_id,object_class,width,height) VALUES (?,?,?,?)",
                (recording_id, "player", .1, .1),
            )


def test_training_job_constraints(roots):
    import sqlite3
    import uuid

    _, config, _ = roots
    db = Database(config / "analyzer.db")
    db.initialize()
    with db.transaction() as connection:
        connection.execute(
            """INSERT INTO training_jobs
               (job_id,status,model_name,epochs,image_size,batch_size,device,output_dir)
               VALUES (?,?,?,?,?,?,?,?)""",
            (str(uuid.uuid4()), "queued", "yolo11n.pt", 50, 640, 8, "0", str(config / "training")),
        )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """INSERT INTO training_jobs
                   (job_id,status,model_name,epochs,image_size,batch_size,device,output_dir)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (str(uuid.uuid4()), "unknown", "yolo11n.pt", 50, 640, 8, "0", str(config / "bad")),
            )


def test_analysis_job_constraints_and_relationships(roots):
    import sqlite3
    import uuid

    _, config, _ = roots
    db = Database(config / "analyzer.db")
    db.initialize()
    recording_id, training_id = str(uuid.uuid4()), str(uuid.uuid4())
    with db.transaction() as connection:
        connection.execute(
            "INSERT INTO recordings (recording_id,media_path,title,size_bytes,mtime_ns) VALUES (?,?,?,?,?)",
            (recording_id, "game.mp4", "Game", 1, 1),
        )
        connection.execute(
            """INSERT INTO training_jobs
               (job_id,status,model_name,epochs,image_size,batch_size,device,output_dir)
               VALUES (?,?,?,?,?,?,?,?)""",
            (training_id, "completed", "yolo11n.pt", 1, 640, 8, "0", str(config / "training")),
        )
        connection.execute(
            """INSERT INTO analysis_jobs
               (job_id,recording_id,training_job_id,status,start_seconds,end_seconds,
                sample_interval_seconds,confidence_threshold,output_dir)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (str(uuid.uuid4()), recording_id, training_id, "queued", 0, 30, .5, .25, str(config / "analysis")),
        )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """INSERT INTO analysis_jobs
                   (job_id,recording_id,training_job_id,status,start_seconds,end_seconds,
                    sample_interval_seconds,confidence_threshold,output_dir)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (str(uuid.uuid4()), recording_id, training_id, "queued", 30, 10, .5, .25, str(config / "bad")),
            )
