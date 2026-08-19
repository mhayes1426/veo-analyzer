import uuid

import pytest

from app.db import Database
from app.training_jobs import ALLOWED_MODELS, _metrics, run_job


class FakeResults:
    results_dict = {"precision": 0.91, "recall": "0.82", "ignored": object()}


def test_training_models_are_explicitly_allowlisted():
    assert ALLOWED_MODELS == {"yolo11n.pt", "yolo11s.pt", "yolo11m.pt"}


def test_metrics_are_serializable_and_prefixed():
    assert _metrics(FakeResults(), "test/") == {"test/precision": 0.91, "test/recall": 0.82}


def test_worker_records_actionable_failure_when_splits_are_empty(roots, monkeypatch):
    media, config, exports = roots
    database = Database(config / "analyzer.db")
    database.initialize()
    job_id = str(uuid.uuid4())
    with database.transaction() as connection:
        connection.execute(
            """INSERT INTO training_jobs
               (job_id,status,model_name,epochs,image_size,batch_size,device,output_dir)
               VALUES (?, 'queued', 'yolo11n.pt', 1, 640, 8, '0', ?)""",
            (job_id, str(config / "training" / "jobs" / job_id)),
        )
    monkeypatch.setenv("MEDIA_ROOT", str(media))
    monkeypatch.setenv("CONFIG_ROOT", str(config))
    monkeypatch.setenv("EXPORT_ROOT", str(exports))
    monkeypatch.setenv("ANALYZER_DB", str(config / "analyzer.db"))
    with pytest.raises(RuntimeError, match="train and validation"):
        run_job(job_id)
    with database.connect() as connection:
        job = connection.execute("SELECT * FROM training_jobs WHERE job_id=?", (job_id,)).fetchone()
    assert job["status"] == "failed"
    assert "more recordings" in job["error_message"]
