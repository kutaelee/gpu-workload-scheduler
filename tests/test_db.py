from pathlib import Path

import psycopg

from gpuq.db import Database


def test_wait_and_migrate_retries_operational_errors(monkeypatch):
    database = Database("postgresql://unused")
    attempts = 0

    def migrate(_path):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise psycopg.OperationalError("database is starting")

    monkeypatch.setattr(database, "migrate", migrate)
    monkeypatch.setattr("gpuq.db.time.sleep", lambda _seconds: None)
    database.wait_and_migrate([Path("unused.sql")], timeout_seconds=30)
    assert attempts == 3
