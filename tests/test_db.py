from app.db import Database


def test_schema_initializes(roots):
    _, config, _ = roots
    db = Database(config / "analyzer.db")
    db.initialize()
    with db.connect() as connection:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"recordings", "events"} <= tables

