from __future__ import annotations

import json
import time
import uuid
from datetime import datetime
from pathlib import Path

import psycopg
from psycopg.rows import dict_row


class Database:
    def __init__(self, database_url: str):
        self.database_url = database_url

    def connect(self):
        return psycopg.connect(
            self.database_url,
            row_factory=dict_row,
            connect_timeout=5,
        )

    def migrate(self, sql_path: Path) -> None:
        with self.connect() as conn:
            conn.execute(sql_path.read_text(encoding="utf-8"))

    def wait_and_migrate(
        self,
        sql_paths: list[Path],
        *,
        timeout_seconds: int = 900,
        retry_seconds: int = 5,
    ) -> None:
        deadline = time.monotonic() + timeout_seconds
        while True:
            try:
                for sql_path in sql_paths:
                    self.migrate(sql_path)
                return
            except psycopg.OperationalError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(retry_seconds)

    def orphan_interrupted_jobs(self) -> int:
        with self.connect() as conn:
            cursor = conn.execute(
                """
                UPDATE jobs
                SET status = 'orphaned',
                    finished_at = now(),
                    error = 'Scheduler restarted while job was recorded as running'
                WHERE status = 'running'
                """
            )
            return cursor.rowcount

    def submit_job(
        self,
        *,
        agent_name: str,
        workload_key: str,
        argv: list[str],
        cwd: str,
        requested_vram_mb: int,
        estimated_seconds: int,
        priority: int,
        max_runtime_seconds: int,
    ) -> dict:
        job_id = uuid.uuid4()
        with self.connect() as conn:
            row = conn.execute(
                """
                INSERT INTO jobs (
                    id, agent_name, workload_key, argv, cwd,
                    requested_vram_mb, estimated_seconds, priority,
                    max_runtime_seconds, status
                )
                VALUES (%s, %s, %s, %s::jsonb, %s, %s, %s, %s, %s, 'queued')
                RETURNING *
                """,
                (
                    job_id,
                    agent_name,
                    workload_key,
                    json.dumps(argv),
                    cwd,
                    requested_vram_mb,
                    estimated_seconds,
                    priority,
                    max_runtime_seconds,
                ),
            ).fetchone()
        return normalize_row(row)

    def queue_candidates(self, window_minutes: int) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                WITH recent AS (
                    SELECT
                        agent_name,
                        COALESCE(SUM(
                            requested_vram_mb *
                            EXTRACT(EPOCH FROM (
                                COALESCE(finished_at, now()) - started_at
                            ))
                        ), 0)::bigint AS recent_vram_seconds
                    FROM jobs
                    WHERE started_at >= now() - (%s * interval '1 minute')
                      AND started_at IS NOT NULL
                    GROUP BY agent_name
                )
                SELECT j.*, COALESCE(r.recent_vram_seconds, 0) AS recent_vram_seconds
                FROM jobs j
                LEFT JOIN recent r USING (agent_name)
                WHERE j.status = 'queued'
                ORDER BY j.manual_rank ASC NULLS LAST, j.submitted_at ASC
                """,
                (window_minutes,),
            ).fetchall()
        return [normalize_row(row) for row in rows]

    def active_jobs(self) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM jobs WHERE status = 'running' ORDER BY started_at"
            ).fetchall()
        return [normalize_row(row) for row in rows]

    def get_job(self, job_id: str) -> dict | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = %s", (job_id,)).fetchone()
        return normalize_row(row) if row else None

    def list_jobs(self, completed_limit: int = 30) -> dict:
        with self.connect() as conn:
            queued = conn.execute(
                """
                SELECT * FROM jobs
                WHERE status = 'queued'
                ORDER BY manual_rank ASC NULLS LAST, submitted_at ASC
                """
            ).fetchall()
            active = conn.execute(
                "SELECT * FROM jobs WHERE status = 'running' ORDER BY started_at"
            ).fetchall()
            completed = conn.execute(
                """
                SELECT * FROM jobs
                WHERE status NOT IN ('queued', 'running')
                ORDER BY finished_at DESC NULLS LAST
                LIMIT %s
                """,
                (completed_limit,),
            ).fetchall()
        return {
            "queued": [normalize_row(row) for row in queued],
            "active": [normalize_row(row) for row in active],
            "completed": [normalize_row(row) for row in completed],
        }

    def workload_estimate(self, workload_key: str) -> dict | None:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT
                    percentile_cont(0.5) WITHIN GROUP (
                        ORDER BY EXTRACT(EPOCH FROM (finished_at - started_at))
                    )::integer AS p50_seconds,
                    percentile_cont(0.9) WITHIN GROUP (
                        ORDER BY EXTRACT(EPOCH FROM (finished_at - started_at))
                    )::integer AS p90_seconds,
                    MAX(requested_vram_mb)::integer AS max_requested_vram_mb,
                    COUNT(*)::integer AS samples
                FROM jobs
                WHERE workload_key = %s
                  AND status = 'succeeded'
                  AND started_at IS NOT NULL
                  AND finished_at IS NOT NULL
                """,
                (workload_key,),
            ).fetchone()
        if not row or not row["samples"]:
            return None
        return normalize_row(row)

    def mark_running(
        self,
        job_id: str,
        *,
        pid: int,
        log_path: str,
        scheduling_note: str,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status = 'running', started_at = now(), pid = %s,
                    log_path = %s, scheduling_note = %s
                WHERE id = %s AND status = 'queued'
                """,
                (pid, log_path, scheduling_note, job_id),
            )

    def mark_finished(
        self,
        job_id: str,
        *,
        status: str,
        exit_code: int | None,
        error: str | None = None,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status = %s, finished_at = now(), exit_code = %s, error = %s
                WHERE id = %s
                """,
                (status, exit_code, error, job_id),
            )

    def update_peak(self, job_id: str, peak_total_gpu_used_mb: int) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET peak_total_gpu_used_mb = GREATEST(
                    COALESCE(peak_total_gpu_used_mb, 0), %s
                )
                WHERE id = %s
                """,
                (peak_total_gpu_used_mb, job_id),
            )

    def request_cancel(self, job_id: str) -> bool:
        with self.connect() as conn:
            cursor = conn.execute(
                """
                UPDATE jobs
                SET cancel_requested = true,
                    status = CASE WHEN status = 'queued' THEN 'canceled' ELSE status END,
                    finished_at = CASE WHEN status = 'queued' THEN now() ELSE finished_at END
                WHERE id = %s AND status IN ('queued', 'running')
                """,
                (job_id,),
            )
            return cursor.rowcount == 1

    def reorder_queued_jobs(self, job_ids: list[str]) -> list[dict]:
        """Persist one complete, user-chosen queued order atomically.

        The request must include every currently queued job exactly once.  This
        turns a stale drag/drop view into a conflict rather than silently
        reordering a different queue after another client has changed it.
        """
        if not job_ids or len(job_ids) != len(set(job_ids)):
            raise ValueError("job_ids must contain unique queued job IDs")
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT id::text AS id FROM jobs WHERE status = 'queued' FOR UPDATE"
            ).fetchall()
            queued_ids = {row["id"] for row in rows}
            if queued_ids != set(job_ids) or len(rows) != len(job_ids):
                raise ValueError("queue changed; refresh and try again")
            for rank, job_id in enumerate(job_ids, start=1):
                conn.execute(
                    """
                    UPDATE jobs
                    SET manual_rank = %s
                    WHERE id = %s AND status = 'queued'
                    """,
                    (rank, job_id),
                )
            ordered = conn.execute(
                """
                SELECT * FROM jobs
                WHERE status = 'queued'
                ORDER BY manual_rank ASC, submitted_at ASC
                """
            ).fetchall()
        return [normalize_row(row) for row in ordered]

    def set_state(self, key: str, value: dict) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO scheduler_state (key, value, updated_at)
                VALUES (%s, %s::jsonb, now())
                ON CONFLICT (key) DO UPDATE
                SET value = EXCLUDED.value, updated_at = now()
                """,
                (key, json.dumps(value)),
            )


def normalize_row(row):
    if row is None:
        return None
    result = dict(row)
    for key, value in tuple(result.items()):
        if isinstance(value, (datetime, uuid.UUID)):
            result[key] = value.isoformat() if isinstance(value, datetime) else str(value)
    return result
