ALTER TABLE jobs
    ADD COLUMN IF NOT EXISTS manual_rank integer;

CREATE INDEX IF NOT EXISTS jobs_manual_queue_idx
    ON jobs (status, manual_rank ASC NULLS LAST, submitted_at ASC);
