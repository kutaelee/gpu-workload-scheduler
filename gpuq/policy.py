from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class Candidate:
    id: str
    priority: int
    submitted_at: datetime
    requested_vram_mb: int
    estimated_seconds: int
    recent_vram_seconds: int
    manual_rank: int | None = None


@dataclass(frozen=True)
class ActiveReservation:
    requested_vram_mb: int
    started_at: datetime
    estimated_seconds: int


def effective_score(
    candidate: Candidate,
    *,
    total_vram_mb: int,
    fairness_window_seconds: int,
    now: datetime | None = None,
) -> float:
    now = now or datetime.now(timezone.utc)
    submitted = candidate.submitted_at
    if submitted.tzinfo is None:
        submitted = submitted.replace(tzinfo=timezone.utc)
    age_minutes = max(0.0, (now - submitted).total_seconds() / 60.0)
    age_bonus = min(20.0, age_minutes / 3.0)
    window_capacity = max(1, total_vram_mb * fairness_window_seconds)
    recent_share = candidate.recent_vram_seconds / window_capacity
    fairness_penalty = min(20.0, recent_share * 40.0)
    return float(candidate.priority) + age_bonus - fairness_penalty


def choose_candidate(
    queued: list[Candidate],
    active: list[ActiveReservation],
    *,
    total_vram_mb: int,
    baseline_used_mb: int,
    safety_vram_mb: int,
    observed_free_mb: int,
    max_parallel_jobs: int,
    fairness_window_seconds: int,
    now: datetime | None = None,
) -> tuple[Candidate | None, str]:
    now = now or datetime.now(timezone.utc)
    if not queued:
        return None, "queue-empty"
    if len(active) >= max_parallel_jobs:
        return None, "parallel-limit"

    active_reserved = sum(item.requested_vram_mb for item in active)
    reservation_free = total_vram_mb - baseline_used_mb - safety_vram_mb - active_reserved
    observed_guarded_free = observed_free_mb - safety_vram_mb
    fit_mb = max(0, min(reservation_free, observed_guarded_free))

    ranked = sorted(
        queued,
        key=lambda job: (
            job.manual_rank is None,
            job.manual_rank if job.manual_rank is not None else 0,
            effective_score(
                job,
                total_vram_mb=total_vram_mb,
                fairness_window_seconds=fairness_window_seconds,
                now=now,
            ),
            -job.estimated_seconds,
            -job.submitted_at.timestamp(),
        ),
        reverse=False,
    )
    # Jobs without a manually set position retain the fairness/priority policy.
    # Reverse only that tail so an explicit user order always stays ahead of it.
    manually_ranked = [job for job in ranked if job.manual_rank is not None]
    policy_ranked = sorted(
        (job for job in ranked if job.manual_rank is None),
        key=lambda job: (
            effective_score(
                job,
                total_vram_mb=total_vram_mb,
                fairness_window_seconds=fairness_window_seconds,
                now=now,
            ),
            -job.estimated_seconds,
            -job.submitted_at.timestamp(),
        ),
        reverse=True,
    )
    ranked = manually_ranked + policy_ranked
    head = ranked[0]
    if head.requested_vram_mb <= fit_mb:
        return head, "head-fits"

    if not active:
        return None, f"head-waits-for-vram:{head.requested_vram_mb}>{fit_mb}"

    remaining = []
    for item in active:
        started = item.started_at
        if started.tzinfo is None:
            started = started.replace(tzinfo=timezone.utc)
        elapsed = max(0.0, (now - started).total_seconds())
        remaining.append(max(0.0, item.estimated_seconds - elapsed))
    earliest_release = min(remaining, default=0.0)

    for candidate in ranked[1:]:
        if (
            candidate.requested_vram_mb <= fit_mb
            and candidate.estimated_seconds <= earliest_release
        ):
            return candidate, f"safe-backfill-before-head:{head.id}"
    return None, f"head-blocks-backfill:{head.requested_vram_mb}>{fit_mb}"
