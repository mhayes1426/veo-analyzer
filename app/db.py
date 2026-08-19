from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


SCHEMA = """
CREATE TABLE IF NOT EXISTS recordings (
    recording_id TEXT PRIMARY KEY,
    media_path TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    mtime_ns INTEGER NOT NULL,
    duration_seconds REAL,
    availability TEXT NOT NULL DEFAULT 'present',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_seen_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY,
    recording_id TEXT NOT NULL REFERENCES recordings(recording_id) ON DELETE CASCADE,
    event_type TEXT NOT NULL,
    event_time_seconds REAL NOT NULL CHECK(event_time_seconds >= 0),
    play_from_seconds REAL NOT NULL CHECK(play_from_seconds >= 0),
    play_until_seconds REAL NOT NULL,
    confidence REAL CHECK(confidence IS NULL OR (confidence >= 0 AND confidence <= 1)),
    review_status TEXT NOT NULL CHECK(review_status IN ('candidate','approved','rejected')),
    source TEXT NOT NULL CHECK(source IN ('manual','model','corrected')),
    model_version TEXT,
    analysis_job_id TEXT REFERENCES analysis_jobs(job_id) ON DELETE SET NULL,
    sequence_outcome TEXT NOT NULL DEFAULT 'uncertain' CHECK(sequence_outcome IN ('made','missed','uncertain')),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    reviewed_at TEXT,
    CHECK(play_from_seconds <= event_time_seconds),
    CHECK(event_time_seconds <= play_until_seconds)
);

CREATE INDEX IF NOT EXISTS idx_events_recording_time
ON events(recording_id, event_time_seconds);

CREATE TABLE IF NOT EXISTS annotation_frames (
    frame_id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL REFERENCES events(event_id) ON DELETE CASCADE,
    frame_time_seconds REAL NOT NULL CHECK(frame_time_seconds >= 0),
    frame_index INTEGER NOT NULL,
    review_status TEXT NOT NULL DEFAULT 'pending' CHECK(review_status IN ('pending','reviewed','skipped')),
    detail_group_id TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(event_id, frame_index)
);

CREATE TABLE IF NOT EXISTS annotation_boxes (
    box_id TEXT PRIMARY KEY,
    frame_id TEXT NOT NULL REFERENCES annotation_frames(frame_id) ON DELETE CASCADE,
    object_class TEXT NOT NULL CHECK(object_class IN ('basketball','hoop')),
    x_center REAL NOT NULL CHECK(x_center >= 0 AND x_center <= 1),
    y_center REAL NOT NULL CHECK(y_center >= 0 AND y_center <= 1),
    width REAL NOT NULL CHECK(width > 0 AND width <= 1),
    height REAL NOT NULL CHECK(height > 0 AND height <= 1),
    occluded INTEGER NOT NULL DEFAULT 0 CHECK(occluded IN (0,1)),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_annotation_frames_event
ON annotation_frames(event_id, frame_index);

CREATE INDEX IF NOT EXISTS idx_annotation_boxes_frame
ON annotation_boxes(frame_id);

CREATE TABLE IF NOT EXISTS annotation_size_presets (
    recording_id TEXT NOT NULL REFERENCES recordings(recording_id) ON DELETE CASCADE,
    object_class TEXT NOT NULL CHECK(object_class IN ('basketball','hoop')),
    width REAL NOT NULL CHECK(width > 0 AND width <= 1),
    height REAL NOT NULL CHECK(height > 0 AND height <= 1),
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (recording_id, object_class)
);

CREATE TABLE IF NOT EXISTS training_jobs (
    job_id TEXT PRIMARY KEY,
    status TEXT NOT NULL CHECK(status IN ('queued','preparing','running','completed','failed')),
    model_name TEXT NOT NULL,
    epochs INTEGER NOT NULL CHECK(epochs >= 1),
    image_size INTEGER NOT NULL CHECK(image_size >= 320),
    batch_size INTEGER NOT NULL,
    device TEXT NOT NULL,
    progress_percent REAL NOT NULL DEFAULT 0 CHECK(progress_percent >= 0 AND progress_percent <= 100),
    current_epoch INTEGER NOT NULL DEFAULT 0,
    frame_count INTEGER NOT NULL DEFAULT 0,
    output_dir TEXT NOT NULL,
    model_path TEXT,
    metrics_json TEXT,
    error_message TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    started_at TEXT,
    completed_at TEXT,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_training_jobs_created
ON training_jobs(created_at DESC);

CREATE TABLE IF NOT EXISTS analysis_jobs (
    job_id TEXT PRIMARY KEY,
    recording_id TEXT NOT NULL REFERENCES recordings(recording_id) ON DELETE CASCADE,
    training_job_id TEXT NOT NULL REFERENCES training_jobs(job_id) ON DELETE RESTRICT,
    status TEXT NOT NULL CHECK(status IN ('queued','running','completed','failed')),
    mode TEXT NOT NULL DEFAULT 'calibration' CHECK(mode IN ('calibration','full_game')),
    start_seconds REAL NOT NULL CHECK(start_seconds >= 0),
    end_seconds REAL NOT NULL CHECK(end_seconds > start_seconds),
    sample_interval_seconds REAL NOT NULL CHECK(sample_interval_seconds >= 0.1),
    confidence_threshold REAL NOT NULL CHECK(confidence_threshold >= 0 AND confidence_threshold <= 1),
    crossing_window_seconds REAL NOT NULL DEFAULT 1.0 CHECK(crossing_window_seconds >= 0.3 AND crossing_window_seconds <= 2.0),
    progress_percent REAL NOT NULL DEFAULT 0 CHECK(progress_percent >= 0 AND progress_percent <= 100),
    processed_frames INTEGER NOT NULL DEFAULT 0,
    detection_count INTEGER NOT NULL DEFAULT 0,
    candidate_count INTEGER NOT NULL DEFAULT 0,
    cancel_requested INTEGER NOT NULL DEFAULT 0 CHECK(cancel_requested IN (0,1)),
    output_dir TEXT NOT NULL,
    error_message TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    started_at TEXT,
    completed_at TEXT,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_analysis_jobs_created
ON analysis_jobs(created_at DESC);

CREATE TABLE IF NOT EXISTS analysis_results (
    result_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES analysis_jobs(job_id) ON DELETE CASCADE,
    frame_time_seconds REAL NOT NULL CHECK(frame_time_seconds >= 0),
    image_path TEXT NOT NULL,
    detections_json TEXT NOT NULL,
    explanation_json TEXT,
    candidate_event_id TEXT REFERENCES events(event_id) ON DELETE SET NULL,
    detection_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_analysis_results_job_time
ON analysis_results(job_id, frame_time_seconds);
"""


class Database:
    def __init__(self, path: Path):
        self.path = path

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(SCHEMA)
            self._migrate_columns(connection)

    @staticmethod
    def _migrate_columns(connection: sqlite3.Connection) -> None:
        event_columns = {row[1] for row in connection.execute("PRAGMA table_info(events)")}
        if "sequence_outcome" not in event_columns:
            connection.execute(
                """ALTER TABLE events ADD COLUMN sequence_outcome TEXT NOT NULL DEFAULT 'uncertain'
                   CHECK(sequence_outcome IN ('made','missed','uncertain'))"""
            )
        if "analysis_job_id" not in event_columns:
            connection.execute("ALTER TABLE events ADD COLUMN analysis_job_id TEXT")
        frame_columns = {row[1] for row in connection.execute("PRAGMA table_info(annotation_frames)")}
        if "review_status" not in frame_columns:
            connection.execute(
                """ALTER TABLE annotation_frames ADD COLUMN review_status TEXT NOT NULL DEFAULT 'pending'
                   CHECK(review_status IN ('pending','reviewed','skipped'))"""
            )
        if "detail_group_id" not in frame_columns:
            connection.execute("ALTER TABLE annotation_frames ADD COLUMN detail_group_id TEXT")
        analysis_columns = {row[1] for row in connection.execute("PRAGMA table_info(analysis_jobs)")}
        if "mode" not in analysis_columns:
            connection.execute("ALTER TABLE analysis_jobs ADD COLUMN mode TEXT NOT NULL DEFAULT 'calibration'")
        if "crossing_window_seconds" not in analysis_columns:
            connection.execute(
                "ALTER TABLE analysis_jobs ADD COLUMN crossing_window_seconds REAL NOT NULL DEFAULT 1.0"
            )
        if "candidate_count" not in analysis_columns:
            connection.execute("ALTER TABLE analysis_jobs ADD COLUMN candidate_count INTEGER NOT NULL DEFAULT 0")
        if "cancel_requested" not in analysis_columns:
            connection.execute("ALTER TABLE analysis_jobs ADD COLUMN cancel_requested INTEGER NOT NULL DEFAULT 0")
        result_columns = {row[1] for row in connection.execute("PRAGMA table_info(analysis_results)")}
        if "explanation_json" not in result_columns:
            connection.execute("ALTER TABLE analysis_results ADD COLUMN explanation_json TEXT")
        if "candidate_event_id" not in result_columns:
            connection.execute("ALTER TABLE analysis_results ADD COLUMN candidate_event_id TEXT")

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self.connect()
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
