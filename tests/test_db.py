import pytest

from app.db import Database


def test_schema_initializes(roots):
    _, config, _ = roots
    db = Database(config / "analyzer.db")
    db.initialize()
    with db.connect() as connection:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"recordings", "events", "annotation_frames", "annotation_boxes"} <= tables
    with db.connect() as connection:
        event_columns = {row[1] for row in connection.execute("PRAGMA table_info(events)")}
        frame_columns = {row[1] for row in connection.execute("PRAGMA table_info(annotation_frames)")}
    assert "sequence_outcome" in event_columns
    assert {"review_status", "detail_group_id"} <= frame_columns


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
