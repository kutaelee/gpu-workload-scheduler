from datetime import datetime, timedelta, timezone

from gpuq.policy import ActiveReservation, Candidate, choose_candidate, effective_score


NOW = datetime(2026, 7, 24, tzinfo=timezone.utc)


def job(job_id, vram, eta, priority=50, age=0, usage=0, manual_rank=None):
    return Candidate(job_id, priority, NOW - timedelta(minutes=age), vram, eta, usage, manual_rank)


def test_age_prevents_starvation():
    old = job("old", 1000, 60, priority=40, age=60)
    new = job("new", 1000, 60, priority=50)
    assert effective_score(old, total_vram_mb=32000, fairness_window_seconds=3600, now=NOW) > effective_score(
        new, total_vram_mb=32000, fairness_window_seconds=3600, now=NOW
    )


def test_head_runs_when_it_fits():
    selected, reason = choose_candidate(
        [job("head", 12000, 600, priority=80), job("small", 2000, 20)],
        [],
        total_vram_mb=32000,
        baseline_used_mb=4000,
        safety_vram_mb=2000,
        observed_free_mb=28000,
        max_parallel_jobs=2,
        fairness_window_seconds=3600,
        now=NOW,
    )
    assert selected.id == "head"
    assert reason == "head-fits"


def test_safe_backfill_only_before_expected_release():
    active = [ActiveReservation(20000, NOW - timedelta(seconds=40), 100)]
    queued = [job("head", 12000, 600, priority=80), job("short", 4000, 30), job("long", 4000, 90)]
    selected, reason = choose_candidate(
        queued,
        active,
        total_vram_mb=32000,
        baseline_used_mb=0,
        safety_vram_mb=2000,
        observed_free_mb=10000,
        max_parallel_jobs=2,
        fairness_window_seconds=3600,
        now=NOW,
    )
    assert selected.id == "short"
    assert reason.startswith("safe-backfill")


def test_manual_queue_position_precedes_policy_score_when_it_fits():
    selected, reason = choose_candidate(
        [
            job("policy-head", 1000, 60, priority=100),
            job("manual-head", 1000, 60, priority=1, manual_rank=1),
        ],
        [],
        total_vram_mb=32000,
        baseline_used_mb=4000,
        safety_vram_mb=2000,
        observed_free_mb=28000,
        max_parallel_jobs=2,
        fairness_window_seconds=3600,
        now=NOW,
    )
    assert selected.id == "manual-head"
    assert reason == "head-fits"


def test_manual_head_still_allows_safe_backfill():
    active = [ActiveReservation(20000, NOW - timedelta(seconds=40), 100)]
    selected, reason = choose_candidate(
        [
            job("manual-head", 12000, 600, priority=1, manual_rank=1),
            job("short", 4000, 30, priority=1, manual_rank=2),
        ],
        active,
        total_vram_mb=32000,
        baseline_used_mb=0,
        safety_vram_mb=2000,
        observed_free_mb=10000,
        max_parallel_jobs=2,
        fairness_window_seconds=3600,
        now=NOW,
    )
    assert selected.id == "short"
    assert reason == "safe-backfill-before-head:manual-head"
