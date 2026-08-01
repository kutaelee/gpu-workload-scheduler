CREATE TABLE IF NOT EXISTS jobs (
    id uuid PRIMARY KEY,
    agent_name text NOT NULL,
    workload_key text NOT NULL,
    argv jsonb NOT NULL,
    cwd text NOT NULL,
    requested_vram_mb integer NOT NULL CHECK (requested_vram_mb > 0),
    estimated_seconds integer NOT NULL CHECK (estimated_seconds > 0),
    priority integer NOT NULL CHECK (priority BETWEEN 0 AND 100),
    max_runtime_seconds integer NOT NULL CHECK (max_runtime_seconds > 0),
    status text NOT NULL CHECK (status IN (
        'queued', 'running', 'succeeded', 'failed', 'canceled', 'timed_out', 'orphaned'
    )),
    submitted_at timestamptz NOT NULL DEFAULT now(),
    started_at timestamptz,
    finished_at timestamptz,
    pid integer,
    exit_code integer,
    log_path text,
    error text,
    cancel_requested boolean NOT NULL DEFAULT false,
    peak_total_gpu_used_mb integer,
    scheduling_note text
);

CREATE INDEX IF NOT EXISTS jobs_queue_idx
    ON jobs (status, priority DESC, submitted_at ASC);

CREATE INDEX IF NOT EXISTS jobs_recent_usage_idx
    ON jobs (agent_name, finished_at)
    WHERE status IN ('succeeded', 'failed', 'canceled', 'timed_out');

CREATE TABLE IF NOT EXISTS scheduler_state (
    key text PRIMARY KEY,
    value jsonb NOT NULL,
    updated_at timestamptz NOT NULL DEFAULT now()
);
