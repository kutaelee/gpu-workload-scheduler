from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from fnmatch import fnmatchcase
from pathlib import Path

from .config import REPO_ROOT, Config
from .db import Database
from .external import (
    inspect_ollama_container,
    inspect_ollama_host,
    stop_ollama_container,
    stop_ollama_host,
)
from .gpu import (
    GpuProcess,
    GpuTelemetry,
    is_fatal_gpu_error,
    read_gpu,
    read_gpu_processes,
)
from .policy import ActiveReservation, Candidate, choose_candidate, effective_score
from .safety import (
    fatal_gpu_events_since_boot,
    read_fault_state,
    runtime_provenance,
    windows_boot_epoch,
    write_fault_state,
)


@dataclass
class RunningProcess:
    process: subprocess.Popen
    log_handle: object
    started_monotonic: float
    max_runtime_seconds: int
    workload_key: str
    argv: tuple[str, ...] = ()
    termination_status: str | None = None
    cancel_signal_sent_at: float | None = None
    terminate_sent_at: float | None = None
    cleanup_attempted: bool = False
    cleanup_error: str | None = None
    baseline_used_mb: int = 0


@dataclass
class HighLoadTransitionGate:
    job_ids: tuple[str, ...]
    not_before_monotonic: float
    baseline_used_mb: int
    stable_samples: int = 0
    process_stable_scans: int = 0
    last_process_generation: int = 0
    last_process_signature: tuple[tuple[int, str], ...] = ()
    reason: str = "minimum-cooldown"


class Scheduler:
    def __init__(self, config: Config, database: Database):
        self.config = config
        self.database = database
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._loop, name="gpu-scheduler", daemon=True)
        self.running: dict[str, RunningProcess] = {}
        self.lock = threading.Lock()
        self.telemetry: GpuTelemetry | None = None
        self.baseline_used_mb = 0
        self.last_decision = "starting"
        self.last_error: str | None = None
        self.gpu_processes: list[GpuProcess] = []
        self.process_scan_error: str | None = None
        self.last_process_scan_monotonic = 0.0
        self.gpu_process_scan_generation = 0
        self.last_external_scan_monotonic = 0.0
        self.external_workloads: list[dict] = []
        self.admission_not_before_monotonic = 0.0
        self.high_load_transition_gate: HighLoadTransitionGate | None = None
        self.no_touch_until_monotonic = 0.0
        self.next_gpu_probe_monotonic = 0.0
        self.transient_failure_backoff_index = 0
        self.boot_epoch = windows_boot_epoch()
        self.fault_state_path = REPO_ROOT / ".runtime" / "gpu-fault-state.json"
        self.transition_state_path = (
            REPO_ROOT / ".runtime" / "gpu-transition-state.json"
        )
        self.workload_boundary_state_path = (
            REPO_ROOT / ".runtime" / "gpu-workload-reboot-boundary.json"
        )
        self.persisted_fault = read_fault_state(self.fault_state_path)
        self.gpu_fault_status = "starting"
        if self.persisted_fault and self.persisted_fault.get("status") == "reboot_required":
            self.gpu_fault_status = (
                "reboot_required"
                if self.persisted_fault.get("windows_boot_epoch") == self.boot_epoch
                else "rearm_required"
            )
        persisted_boundary = read_fault_state(self.workload_boundary_state_path)
        self.workload_reboot_boundary = (
            persisted_boundary
            if persisted_boundary
            and persisted_boundary.get("status") in {"active", "armed"}
            and persisted_boundary.get("windows_boot_epoch") == self.boot_epoch
            else None
        )
        self.provenance = runtime_provenance(REPO_ROOT, config.config_path)
        persisted_transition = read_fault_state(self.transition_state_path)
        if (
            persisted_transition
            and persisted_transition.get("status") == "active"
            and persisted_transition.get("windows_boot_epoch") == self.boot_epoch
        ):
            remaining = max(
                0.0,
                float(persisted_transition.get("no_touch_until_unix", 0.0))
                - time.time(),
            )
            not_before = time.monotonic() + remaining
            self.high_load_transition_gate = HighLoadTransitionGate(
                job_ids=tuple(persisted_transition.get("job_ids") or ()),
                not_before_monotonic=not_before,
                baseline_used_mb=max(
                    0, int(persisted_transition.get("baseline_used_mb") or 0)
                ),
            )
            self.no_touch_until_monotonic = not_before
            self.next_gpu_probe_monotonic = not_before
            self.admission_not_before_monotonic = not_before
        self.gpu_healthy = False
        self.gpu_health_consecutive_failures = 0
        self.gpu_health_recovery_successes = 0
        self.gpu_health_last_failure_at: str | None = None
        self.gpu_health_last_recovered_at: str | None = None
        self._last_gpu_health_log_signature: str | None = None
        self._last_gpu_telemetry_log_monotonic = 0.0

    def start(self) -> None:
        if not self.config.gpu_telemetry_enabled:
            self.gpu_fault_status = "telemetry_disabled"
            self.last_decision = "telemetry-disabled"
            self.thread.start()
            return
        if self.gpu_fault_status in {"reboot_required", "rearm_required"}:
            self.last_decision = f"gpu-{self.gpu_fault_status}"
            self.thread.start()
            return
        if self.workload_reboot_boundary is not None:
            self.last_decision = "workload-reboot-boundary-required"
            self.thread.start()
            return
        try:
            telemetry = read_gpu()
        except Exception as exc:
            self._record_gpu_failure(exc)
        else:
            self.telemetry = telemetry
            self.baseline_used_mb = telemetry.used_mb
            self.gpu_healthy = True
            self.gpu_fault_status = "healthy"
            self._refresh_gpu_processes(force=True)
            self._refresh_external_workloads(force=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=10)

    def snapshot(self) -> dict:
        with self.lock:
            last_known_telemetry = self.telemetry.to_dict() if self.telemetry else None
            telemetry = last_known_telemetry if self.gpu_healthy else None
            managed_pids = {record.process.pid for record in self.running.values()}
            unmanaged_processes = [
                process.to_dict()
                for process in self.gpu_processes
                if process.pid not in managed_pids
            ]
            return {
                "gpu": telemetry,
                "last_known_gpu": last_known_telemetry,
                "gpu_health": {
                    "status": (
                        self.gpu_fault_status
                        if self.gpu_fault_status
                        in {"reboot_required", "rearm_required", "telemetry_disabled"}
                        else (
                            "healthy"
                            if self.gpu_healthy
                            else (
                                "recovering"
                                if self.gpu_health_recovery_successes
                                else "blocked"
                            )
                        )
                    ),
                    "consecutive_failures": self.gpu_health_consecutive_failures,
                    "recovery_successes": self.gpu_health_recovery_successes,
                    "recovery_samples_required": self.config.gpu_health_recovery_samples,
                    "last_failure_at": self.gpu_health_last_failure_at,
                    "last_recovered_at": self.gpu_health_last_recovered_at,
                    "fault_latched": self.gpu_fault_status
                    in {"reboot_required", "rearm_required"},
                    "manual_rearm_required": self.gpu_fault_status
                    in {"reboot_required", "rearm_required"},
                },
                "admission_ready": self.admission_ready(),
                "workload_reboot_boundary": self.workload_reboot_boundary,
                "runtime_provenance": dict(self.provenance),
                "baseline_used_mb": self.baseline_used_mb,
                "safety_vram_mb": self.config.safety_vram_mb,
                "fairness_window_minutes": self.config.fairness_window_minutes,
                "max_parallel_jobs": self.config.max_parallel_jobs,
                "managed_running": len(self.running),
                "last_decision": self.last_decision,
                "last_error": self.last_error,
                "unmanaged_gpu_processes": unmanaged_processes,
                "unmanaged_gpu_process_count": len(unmanaged_processes),
                "gpu_process_scan_error": self.process_scan_error,
                "external_workloads": list(self.external_workloads),
                "post_job_cooldown_remaining_seconds": max(
                    0.0, self.admission_not_before_monotonic - time.monotonic()
                ),
                "transition_gate": self._transition_gate_snapshot(),
                "no_touch_remaining_seconds": max(
                    0.0, self.no_touch_until_monotonic - time.monotonic()
                ),
            }

    def admission_ready(self) -> bool:
        return bool(
            self.config.gpu_telemetry_enabled
            and self.gpu_healthy
            and self.gpu_fault_status == "healthy"
            and self.workload_reboot_boundary is None
            and self.high_load_transition_gate is None
            and time.monotonic() >= self.admission_not_before_monotonic
            and time.monotonic() >= self.no_touch_until_monotonic
        )

    def scores_for(self, queued: list[dict]) -> dict[str, float]:
        if not queued:
            return {}
        telemetry = self.telemetry
        if telemetry is None:
            return {}
        now = datetime.now(timezone.utc)
        return {
            row["id"]: round(
                effective_score(
                    _candidate(row),
                    total_vram_mb=telemetry.total_mb,
                    fairness_window_seconds=self.config.fairness_window_minutes * 60,
                    now=now,
                ),
                2,
            )
            for row in queued
        }

    def _transition_gate_snapshot(self) -> dict | None:
        gate = self.high_load_transition_gate
        if gate is None:
            return None
        return {
            "job_ids": list(gate.job_ids),
            "minimum_cooldown_remaining_seconds": max(
                0.0, gate.not_before_monotonic - time.monotonic()
            ),
            "baseline_used_mb": gate.baseline_used_mb,
            "vram_limit_mb": (
                gate.baseline_used_mb
                + self.config.post_high_load_vram_tolerance_mb
            ),
            "stable_samples": gate.stable_samples,
            "stable_samples_required": self.config.post_high_load_stable_samples,
            "process_stable_scans": gate.process_stable_scans,
            "process_stable_scans_required": (
                self.config.post_high_load_process_stable_scans
            ),
            "reason": gate.reason,
        }

    def _loop(self) -> None:
        while not self.stop_event.is_set():
            # Process lifecycle state is host-local and must be checked before any
            # NVIDIA query.  Otherwise a process that exited just after the prior
            # iteration receives one extra nvidia-smi touch before its no-touch or
            # reboot boundary is armed.
            try:
                terminal_transition = self._reap_and_cancel(None)
                if terminal_transition:
                    with self.lock:
                        self.last_decision = "post-job-no-touch"
                    self.database.set_state("runtime", self.snapshot())
                    self.stop_event.wait(self.config.poll_seconds)
                    continue
            except Exception as exc:
                # Reaping/persistence failure is safety relevant.  Do not probe the
                # GPU until a later loop proves lifecycle state can be processed.
                with self.lock:
                    self.last_error = f"{type(exc).__name__}: {exc}"[:1000]
                    self.last_decision = "lifecycle-reap-failed-no-gpu-probe"
                self.stop_event.wait(self.config.poll_seconds)
                continue

            now_monotonic = time.monotonic()
            probe_block_reason = self._gpu_probe_block_reason(now_monotonic)

            if probe_block_reason is not None:
                try:
                    self._reap_and_cancel(None)
                    with self.lock:
                        self.last_decision = probe_block_reason
                    self.database.set_state("runtime", self.snapshot())
                except Exception as exc:
                    with self.lock:
                        self.last_error = f"{type(exc).__name__}: {exc}"[:1000]
                self.stop_event.wait(self.config.poll_seconds)
                continue

            try:
                telemetry = read_gpu()
            except Exception as exc:
                self._record_gpu_failure(exc)
                self._reap_and_cancel(None)
                try:
                    self.database.set_state("runtime", self.snapshot())
                except Exception as state_exc:
                    with self.lock:
                        self.last_error = (
                            f"{self.last_error}; state persistence failed: "
                            f"{type(state_exc).__name__}: {state_exc}"
                        )[:1000]
                self.stop_event.wait(self.config.poll_seconds)
                continue

            try:
                recovered = self._record_gpu_success(telemetry)
                self._log_gpu_telemetry(telemetry)
                terminal_transition = self._reap_and_cancel(telemetry)
                if self.gpu_healthy and not terminal_transition:
                    gate = self.high_load_transition_gate
                    self._refresh_gpu_processes(
                        force=bool(
                            gate is not None
                            and time.monotonic() >= gate.not_before_monotonic
                        )
                    )
                    self._refresh_external_workloads()
                active_rows = self.database.active_jobs()
                if not active_rows and self.high_load_transition_gate is None:
                    self.baseline_used_mb = telemetry.used_mb
                if terminal_transition:
                    with self.lock:
                        self.last_decision = "post-job-no-touch"
                elif not self.gpu_healthy:
                    if self.gpu_fault_status == "transient_failure":
                        self.next_gpu_probe_monotonic = (
                            time.monotonic() + 5.0
                        )
                    with self.lock:
                        self.last_decision = (
                            "gpu-health-recovering:"
                            f"{self.gpu_health_recovery_successes}/"
                            f"{self.config.gpu_health_recovery_samples}"
                        )
                elif not self._evaluate_high_load_transition_gate(telemetry):
                    self.next_gpu_probe_monotonic = (
                        time.monotonic()
                        + self.config.post_high_load_probe_interval_seconds
                    )
                    with self.lock:
                        gate = self.high_load_transition_gate
                        self.last_decision = (
                            f"post-transition:{gate.reason}"
                            if gate is not None
                            else "post-transition"
                        )
                else:
                    self.next_gpu_probe_monotonic = 0.0
                    if recovered:
                        self._pause_admission()
                    self._schedule_once(telemetry, active_rows)
                self.database.set_state("runtime", self.snapshot())
            except Exception as exc:
                with self.lock:
                    self.last_error = f"{type(exc).__name__}: {exc}"[:1000]
            self.stop_event.wait(self.config.poll_seconds)

    def _gpu_probe_block_reason(self, now_monotonic: float) -> str | None:
        if not self.config.gpu_telemetry_enabled:
            return "telemetry-disabled"
        if self.gpu_fault_status in {"reboot_required", "rearm_required"}:
            return f"gpu-{self.gpu_fault_status}"
        if self.workload_reboot_boundary is not None:
            return "workload-reboot-boundary-required"
        if now_monotonic < self.no_touch_until_monotonic:
            return (
                "post-high-load-no-touch:"
                f"{self.no_touch_until_monotonic - now_monotonic:.1f}s"
            )
        if now_monotonic < self.next_gpu_probe_monotonic:
            return (
                "gpu-probe-backoff:"
                f"{self.next_gpu_probe_monotonic - now_monotonic:.1f}s"
            )
        return None

    def _record_gpu_failure(self, exc: Exception) -> None:
        now = datetime.now(timezone.utc).isoformat()
        error = f"{type(exc).__name__}: {exc}"[:1000]
        with self.lock:
            protected_workload_running = any(
                self._requires_reboot_boundary(record.workload_key)
                for record in self.running.values()
            )
        # A timeout while a reboot-boundary workload is active is not a normal
        # observability miss: retrying the NVIDIA stack was part of the prior TDR
        # escalation sequence.  Latch immediately and let Windows reboot define
        # the next safe probe boundary.
        fatal = is_fatal_gpu_error(exc) or (
            protected_workload_running
            and isinstance(exc, subprocess.TimeoutExpired)
        )
        with self.lock:
            self.gpu_healthy = False
            self.gpu_health_consecutive_failures += 1
            self.gpu_health_recovery_successes = 0
            self.gpu_health_last_failure_at = now
            self.gpu_fault_status = (
                "reboot_required" if fatal else "transient_failure"
            )
            self.last_decision = (
                "gpu-reboot-required" if fatal else "gpu-health-blocked"
            )
            self.last_error = error
            if self.high_load_transition_gate is not None:
                self.high_load_transition_gate.stable_samples = 0
                self.high_load_transition_gate.process_stable_scans = 0
                self.high_load_transition_gate.reason = "gpu-health-blocked"
        if fatal:
            fault = {
                "schema_version": 1,
                "status": "reboot_required",
                "latched_at": now,
                "windows_boot_epoch": self.boot_epoch,
                "reason": error,
                "runtime_provenance": self.provenance,
            }
            write_fault_state(self.fault_state_path, fault)
            self.persisted_fault = fault
            self.next_gpu_probe_monotonic = float("inf")
        else:
            delays = (5.0, 15.0, 60.0)
            delay = delays[min(self.transient_failure_backoff_index, 2)]
            self.transient_failure_backoff_index = min(
                self.transient_failure_backoff_index + 1, 2
            )
            self.next_gpu_probe_monotonic = time.monotonic() + delay
        self._log_gpu_health_event(self.gpu_fault_status, error)

    def _record_gpu_success(self, telemetry: GpuTelemetry) -> bool:
        recovered = False
        with self.lock:
            self.telemetry = telemetry
            if self.gpu_healthy:
                self.last_error = None
                return False
            self.gpu_health_recovery_successes += 1
            if (
                self.gpu_health_recovery_successes
                >= self.config.gpu_health_recovery_samples
            ):
                self.gpu_healthy = True
                self.gpu_fault_status = "healthy"
                self.transient_failure_backoff_index = 0
                self.gpu_health_consecutive_failures = 0
                self.gpu_health_recovery_successes = 0
                self.gpu_health_last_recovered_at = datetime.now(
                    timezone.utc
                ).isoformat()
                self.last_error = None
                recovered = True
        if recovered:
            self._log_gpu_health_event("recovered", None)
        return recovered

    def _log_gpu_health_event(self, status: str, error: str | None) -> None:
        signature = f"{status}:{error or ''}"
        if signature == self._last_gpu_health_log_signature:
            return
        self._last_gpu_health_log_signature = signature
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "error": error,
            "consecutive_failures": self.gpu_health_consecutive_failures,
            "runtime_provenance": self.provenance,
        }
        try:
            self.config.log_root.mkdir(parents=True, exist_ok=True)
            with (self.config.log_root / "gpu-health.jsonl").open(
                "a", encoding="utf-8"
            ) as handle:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        except Exception:
            # Health logging must never bypass the admission circuit breaker.
            pass

    def _log_gpu_telemetry(self, telemetry: GpuTelemetry) -> None:
        now_monotonic = time.monotonic()
        if (
            now_monotonic - self._last_gpu_telemetry_log_monotonic
            < self.config.gpu_telemetry_log_interval_seconds
        ):
            return
        self._last_gpu_telemetry_log_monotonic = now_monotonic
        now = datetime.now(timezone.utc)
        event = {
            "timestamp": now.isoformat(),
            "git_commit": self.provenance.get("git_commit"),
            "git_branch": self.provenance.get("git_branch"),
            "dirty_worktree": self.provenance.get("dirty_worktree"),
            "config_sha256": self.provenance.get("config_sha256"),
            "service_started_at": self.provenance.get("service_started_at"),
            "windows_boot_epoch": self.provenance.get("windows_boot_epoch"),
            **telemetry.to_dict(),
        }
        try:
            self.config.log_root.mkdir(parents=True, exist_ok=True)
            path = self.config.log_root / f"gpu-telemetry-{now:%Y%m%d}.jsonl"
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        except Exception:
            # Observability must not interfere with admission or job reaping.
            pass

    def _refresh_gpu_processes(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if (
            not force
            and now - self.last_process_scan_monotonic
            < self.config.process_scan_interval_seconds
        ):
            return
        # This is deliberately periodic rather than part of every scheduling
        # decision. `nvidia-smi` process accounting is observational only and
        # must not become a CPU hot loop.
        try:
            processes = read_gpu_processes()
        except Exception as exc:
            if is_fatal_gpu_error(exc):
                self._record_gpu_failure(exc)
            with self.lock:
                self.process_scan_error = f"{type(exc).__name__}: {exc}"
                self.last_process_scan_monotonic = now
            return
        with self.lock:
            self.gpu_processes = processes
            self.process_scan_error = None
            self.last_process_scan_monotonic = now
            self.gpu_process_scan_generation += 1

    def _refresh_external_workloads(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if (
            not force
            and now - self.last_external_scan_monotonic
            < self.config.process_scan_interval_seconds
        ):
            return
        workloads: list[dict] = []
        for key, config in self.config.external_workloads.items():
            if config["kind"] == "ollama_container":
                try:
                    workloads.append(
                        inspect_ollama_container(
                            config["container"],
                            key=key,
                            label=config["label"],
                        )
                    )
                except Exception as exc:
                    workloads.append(
                        {
                            "key": key,
                            "label": config["label"],
                            "kind": config["kind"],
                            "container": config["container"],
                            "state": "error",
                            "models": [],
                            "can_stop": False,
                            "error": f"{type(exc).__name__}: {exc}"[:500],
                        }
                    )
            elif config["kind"] == "ollama_host":
                try:
                    workloads.append(
                        inspect_ollama_host(
                            config["executable"],
                            key=key,
                            label=config["label"],
                        )
                    )
                except Exception as exc:
                    workloads.append(
                        {
                            "key": key,
                            "label": config["label"],
                            "kind": config["kind"],
                            "target": "127.0.0.1:11434",
                            "state": "error",
                            "models": [],
                            "can_stop": False,
                            "error": f"{type(exc).__name__}: {exc}"[:500],
                        }
                    )
        with self.lock:
            self.external_workloads = workloads
            self.last_external_scan_monotonic = now

    def stop_external_workload(self, key: str) -> dict | None:
        config = self.config.external_workloads.get(key)
        if config is None:
            return None
        if config["kind"] == "ollama_container":
            result = stop_ollama_container(
                config["container"],
                key=key,
                label=config["label"],
            )
        elif config["kind"] == "ollama_host":
            result = stop_ollama_host(
                config["executable"],
                key=key,
                label=config["label"],
            )
        else:
            raise ValueError("unsupported external workload kind")
        self._refresh_external_workloads(force=True)
        return result

    def rearm_gpu(self) -> dict:
        if self.gpu_fault_status not in {"reboot_required", "rearm_required"}:
            return {"rearmed": False, "reason": "no-fatal-latch"}
        latched_boot = (self.persisted_fault or {}).get("windows_boot_epoch")
        if latched_boot == self.boot_epoch:
            return {"rearmed": False, "reason": "windows-reboot-required"}
        try:
            event_count = fatal_gpu_events_since_boot()
        except Exception as exc:
            return {
                "rearmed": False,
                "reason": "windows-event-check-failed",
                "error": f"{type(exc).__name__}: {exc}"[:500],
            }
        if event_count:
            return {
                "rearmed": False,
                "reason": "current-boot-nvlddmkm-errors",
                "event_count": event_count,
            }
        try:
            telemetry = read_gpu()
        except Exception as exc:
            if is_fatal_gpu_error(exc):
                self._record_gpu_failure(exc)
            return {
                "rearmed": False,
                "reason": "gpu-health-check-failed",
                "error": f"{type(exc).__name__}: {exc}"[:500],
            }
        cleared = {
            "schema_version": 1,
            "status": "cleared",
            "rearmed_at": datetime.now(timezone.utc).isoformat(),
            "windows_boot_epoch": self.boot_epoch,
            "prior_fault": self.persisted_fault,
        }
        write_fault_state(self.fault_state_path, cleared)
        with self.lock:
            self.persisted_fault = cleared
            self.telemetry = telemetry
            self.baseline_used_mb = telemetry.used_mb
            self.gpu_healthy = True
            self.gpu_fault_status = "healthy"
            self.gpu_health_consecutive_failures = 0
            self.gpu_health_recovery_successes = 0
            self.last_error = None
            self.last_decision = "gpu-manually-rearmed"
            self.next_gpu_probe_monotonic = 0.0
        return {"rearmed": True, "gpu": telemetry.to_dict()}

    def _reap_and_cancel(self, telemetry: GpuTelemetry | None) -> bool:
        terminal_transition = False
        with self.lock:
            running_items = list(self.running.items())
        for job_id, record in running_items:
            job = self.database.get_job(job_id)
            if job and telemetry is not None:
                self.database.update_peak(job_id, telemetry.used_mb)
            exit_code = record.process.poll()
            if exit_code is not None:
                status = record.termination_status or (
                    "succeeded" if exit_code == 0 else "failed"
                )
                cleanup_error = None
                if status != "succeeded":
                    cleanup_error = self._run_cleanup_once(record)
                error = None
                if record.termination_status:
                    error = f"Job {status} by scheduler"
                if cleanup_error:
                    error = (
                        f"{error}; cleanup failed: {cleanup_error}"
                        if error
                        else f"Cleanup failed after job {status}: {cleanup_error}"
                    )
                self.database.mark_finished(
                    job_id,
                    status=status,
                    exit_code=exit_code,
                    error=error,
                )
                self._close_record(job_id)
                high_capacity = self._is_high_load_terminal_job(
                    job, record, telemetry
                )
                self._arm_high_load_transition_gate(
                    job_id,
                    record,
                    no_touch_seconds=(
                        self.config.post_high_load_no_touch_seconds
                        if high_capacity
                        else self.config.post_job_no_touch_seconds
                    ),
                )
                if self._requires_reboot_boundary(record.workload_key):
                    self._arm_workload_reboot_boundary(job_id, record, status)
                terminal_transition = True
                continue
            if job and job.get("cancel_requested"):
                self._request_termination(job_id, record, "canceled")
                continue
            if time.monotonic() - record.started_monotonic > record.max_runtime_seconds:
                self._request_termination(job_id, record, "timed_out")
        return terminal_transition

    def _requires_reboot_boundary(self, workload_key: str) -> bool:
        if workload_key in self.config.reboot_boundary_workloads:
            return True
        return any(
            fnmatchcase(workload_key, pattern)
            for pattern in self.config.reboot_boundary_workload_patterns
        )

    def _arm_workload_reboot_boundary(
        self,
        job_id: str,
        record: RunningProcess,
        status: str,
    ) -> None:
        """Block all further GPU access until Windows has rebooted.

        This is deliberately a persistent admission/probe latch rather than an
        automatic ``wsl --terminate``.  A shared WSL distro can contain unrelated
        long-lived processes, so the scheduler must not destroy that state as an
        implicit post-job side effect.
        """
        boundary = {
            "schema_version": 1,
            "status": "active",
            "latched_at": datetime.now(timezone.utc).isoformat(),
            "windows_boot_epoch": self.boot_epoch,
            "job_id": job_id,
            "workload_key": record.workload_key,
            "terminal_status": status,
            "reason": "configured-one-gpu-workload-per-windows-boot",
            "runtime_provenance": self.provenance,
        }
        with self.lock:
            self.workload_reboot_boundary = boundary
            self.last_decision = "workload-reboot-boundary-required"
            self.next_gpu_probe_monotonic = float("inf")
        write_fault_state(self.workload_boundary_state_path, boundary)

    def _arm_workload_boundary_before_launch(self, job: dict) -> dict | None:
        """Persist a diagnostic boundary before a high-risk process is started.

        The in-memory boundary remains clear while the current daemon owns the
        process, allowing bounded telemetry and lifecycle handling.  If the daemon
        exits, a new instance treats the persisted ``armed`` state as an active
        same-boot boundary and cannot start or probe another workload.
        """
        workload_key = str(job["workload_key"])
        if not self._requires_reboot_boundary(workload_key):
            return None
        boundary = {
            "schema_version": 1,
            "status": "armed",
            "armed_at": datetime.now(timezone.utc).isoformat(),
            "windows_boot_epoch": self.boot_epoch,
            "job_id": job["id"],
            "workload_key": workload_key,
            "reason": "configured-one-gpu-workload-per-windows-boot",
            "runtime_provenance": self.provenance,
        }
        write_fault_state(self.workload_boundary_state_path, boundary)
        return boundary

    def _clear_unlaunched_workload_boundary(self, boundary: dict) -> None:
        write_fault_state(
            self.workload_boundary_state_path,
            {
                **boundary,
                "status": "cleared-without-launch",
                "cleared_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    def _request_termination(
        self, job_id: str, record: RunningProcess, status: str
    ) -> None:
        now = time.monotonic()
        if record.termination_status is None:
            record.termination_status = status
            record.cancel_signal_sent_at = now
            try:
                if os.name == "nt":
                    record.process.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    record.process.terminate()
            except Exception as exc:
                with self.lock:
                    self.last_error = (
                        f"Could not request cooperative stop for {job_id}: "
                        f"{type(exc).__name__}: {exc}"
                    )
            return

        if record.process.poll() is not None:
            return
        if record.cancel_signal_sent_at is None:
            record.cancel_signal_sent_at = now
        if now - record.cancel_signal_sent_at < self.config.cancel_grace_seconds:
            return

        is_wsl = bool(record.argv) and Path(record.argv[0]).name.lower() in {
            "wsl",
            "wsl.exe",
        }
        has_cleanup = record.workload_key in self.config.cleanup_commands
        if is_wsl and not self.config.wsl_force_terminate and not has_cleanup:
            with self.lock:
                self.last_error = (
                    f"WSL job {job_id} ignored cooperative cancellation; "
                    "refusing to kill only the Windows wrapper without an "
                    "allowlisted Linux cleanup command"
                )
            return

        if has_cleanup:
            self._run_cleanup_once(record)
        if record.terminate_sent_at is None:
            try:
                record.process.terminate()
                record.terminate_sent_at = now
            except Exception as exc:
                with self.lock:
                    self.last_error = (
                        f"Could not terminate managed job {job_id}: "
                        f"{type(exc).__name__}: {exc}"
                    )
            return

        if now - record.terminate_sent_at < self.config.terminate_grace_seconds:
            return
        try:
            record.process.kill()
        except Exception as exc:
            with self.lock:
                self.last_error = (
                    f"Could not kill managed job {job_id}: "
                    f"{type(exc).__name__}: {exc}"
                )

    def _run_cleanup_once(self, record: RunningProcess) -> str | None:
        if record.cleanup_attempted:
            return record.cleanup_error
        record.cleanup_attempted = True
        record.cleanup_error = self._run_cleanup(record)
        return record.cleanup_error

    def _run_cleanup(self, record: RunningProcess) -> str | None:
        argv = self.config.cleanup_commands.get(record.workload_key)
        if not argv:
            return None
        try:
            completed = subprocess.run(
                list(argv),
                stdin=subprocess.DEVNULL,
                stdout=record.log_handle,
                stderr=subprocess.STDOUT,
                shell=False,
                timeout=60,
                check=False,
                env=os.environ.copy(),
            )
        except Exception as exc:
            return f"{type(exc).__name__}: {exc}"[:500]
        return None if completed.returncode == 0 else f"exit_code={completed.returncode}"

    def _close_record(self, job_id: str) -> None:
        with self.lock:
            record = self.running.pop(job_id, None)
        if record:
            record.log_handle.close()

    def _is_high_load_terminal_job(
        self,
        job: dict | None,
        record: RunningProcess,
        telemetry: GpuTelemetry | None,
    ) -> bool:
        peak_used_mb = int((job or {}).get("peak_total_gpu_used_mb") or 0)
        if telemetry is not None:
            peak_used_mb = max(peak_used_mb, telemetry.used_mb)
        return peak_used_mb >= self.config.high_load_min_peak_used_mb

    def _arm_high_load_transition_gate(
        self,
        job_id: str,
        record: RunningProcess,
        *,
        no_touch_seconds: float | None = None,
    ) -> None:
        if no_touch_seconds is None:
            no_touch_seconds = self.config.post_high_load_no_touch_seconds
        not_before = (
            time.monotonic() + no_touch_seconds
        )
        baseline_used_mb = max(0, record.baseline_used_mb)
        with self.lock:
            existing = self.high_load_transition_gate
            if existing is not None:
                job_ids = tuple(dict.fromkeys((*existing.job_ids, job_id)))
                not_before = max(not_before, existing.not_before_monotonic)
                baseline_used_mb = min(
                    baseline_used_mb, existing.baseline_used_mb
                )
            else:
                job_ids = (job_id,)
            self.high_load_transition_gate = HighLoadTransitionGate(
                job_ids=job_ids,
                not_before_monotonic=not_before,
                baseline_used_mb=baseline_used_mb,
                last_process_generation=self.gpu_process_scan_generation,
            )
            self.admission_not_before_monotonic = max(
                self.admission_not_before_monotonic, not_before
            )
            self.no_touch_until_monotonic = max(
                self.no_touch_until_monotonic, not_before
            )
            self.next_gpu_probe_monotonic = max(
                self.next_gpu_probe_monotonic, not_before
            )
            self._write_transition_state(
                {
                    "schema_version": 1,
                    "status": "active",
                    "job_ids": list(job_ids),
                    "windows_boot_epoch": self.boot_epoch,
                    "no_touch_until_unix": time.time()
                    + max(0.0, not_before - time.monotonic()),
                    "baseline_used_mb": baseline_used_mb,
                }
            )

    def _evaluate_high_load_transition_gate(
        self, telemetry: GpuTelemetry
    ) -> bool:
        gate = self.high_load_transition_gate
        if gate is None:
            return True
        now = time.monotonic()
        if now < gate.not_before_monotonic:
            gate.stable_samples = 0
            gate.process_stable_scans = 0
            gate.reason = "minimum-cooldown"
            return False
        if not self.gpu_healthy:
            gate.stable_samples = 0
            gate.process_stable_scans = 0
            gate.reason = "gpu-health-blocked"
            return False
        if self.process_scan_error is not None:
            gate.stable_samples = 0
            gate.process_stable_scans = 0
            gate.reason = "gpu-process-scan-error"
            return False

        process_generation = self.gpu_process_scan_generation
        if process_generation > gate.last_process_generation:
            signature = tuple(
                sorted(
                    (process.pid, process.process_name)
                    for process in self.gpu_processes
                )
            )
            if signature == gate.last_process_signature:
                gate.process_stable_scans += 1
            else:
                gate.last_process_signature = signature
                gate.process_stable_scans = 1
            gate.last_process_generation = process_generation

        vram_limit_mb = (
            gate.baseline_used_mb
            + self.config.post_high_load_vram_tolerance_mb
        )
        if telemetry.used_mb > vram_limit_mb:
            gate.stable_samples = 0
            gate.reason = (
                f"vram-not-idle:{telemetry.used_mb}>{vram_limit_mb}MB"
            )
            return False
        if (
            telemetry.utilization_percent
            > self.config.post_high_load_max_idle_utilization_percent
        ):
            gate.stable_samples = 0
            gate.reason = (
                "gpu-not-idle:"
                f"{telemetry.utilization_percent}>"
                f"{self.config.post_high_load_max_idle_utilization_percent}%"
            )
            return False

        gate.stable_samples += 1
        if (
            gate.stable_samples < self.config.post_high_load_stable_samples
        ):
            gate.reason = (
                "telemetry-stabilizing:"
                f"{gate.stable_samples}/"
                f"{self.config.post_high_load_stable_samples}"
            )
            return False
        if (
            gate.process_stable_scans
            < self.config.post_high_load_process_stable_scans
        ):
            gate.reason = (
                "process-census-stabilizing:"
                f"{gate.process_stable_scans}/"
                f"{self.config.post_high_load_process_stable_scans}"
            )
            return False

        self.high_load_transition_gate = None
        self.admission_not_before_monotonic = 0.0
        self.no_touch_until_monotonic = 0.0
        self.baseline_used_mb = telemetry.used_mb
        self._write_transition_state(
            {
                "schema_version": 1,
                "status": "cleared",
                "cleared_at": datetime.now(timezone.utc).isoformat(),
                "windows_boot_epoch": self.boot_epoch,
            }
        )
        return True

    def _write_transition_state(self, value: dict) -> None:
        path = getattr(self, "transition_state_path", None)
        if path is not None:
            write_fault_state(path, value)

    def _pause_admission(self, seconds: float | None = None) -> None:
        if seconds is None:
            seconds = self.config.post_job_cooldown_seconds
        self.admission_not_before_monotonic = max(
            self.admission_not_before_monotonic,
            time.monotonic() + seconds,
        )

    def _schedule_once(self, telemetry: GpuTelemetry, active_rows: list[dict]) -> None:
        if not self.gpu_healthy:
            with self.lock:
                self.last_decision = "gpu-health-blocked"
            return
        if self.high_load_transition_gate is not None:
            with self.lock:
                self.last_decision = "post-transition"
            return
        cooldown_remaining = self.admission_not_before_monotonic - time.monotonic()
        if cooldown_remaining > 0:
            with self.lock:
                self.last_decision = f"gpu-recovery-pause:{cooldown_remaining:.1f}s"
            return
        queued_rows = self.database.queue_candidates(self.config.fairness_window_minutes)
        queued = [_candidate(row) for row in queued_rows]
        active = [
            ActiveReservation(
                requested_vram_mb=row["requested_vram_mb"],
                started_at=datetime.fromisoformat(row["started_at"]),
                estimated_seconds=row["estimated_seconds"],
            )
            for row in active_rows
        ]
        candidate, reason = choose_candidate(
            queued,
            active,
            total_vram_mb=telemetry.total_mb,
            baseline_used_mb=self.baseline_used_mb,
            safety_vram_mb=self.config.safety_vram_mb,
            observed_free_mb=telemetry.free_mb,
            max_parallel_jobs=self.config.max_parallel_jobs,
            fairness_window_seconds=self.config.fairness_window_minutes * 60,
        )
        with self.lock:
            self.last_decision = reason
        if candidate is None:
            return
        row = next(item for item in queued_rows if item["id"] == candidate.id)
        argv = row["argv"]
        if isinstance(argv, str):
            argv = json.loads(argv)
        is_wsl = bool(argv) and Path(argv[0]).name.lower() in {"wsl", "wsl.exe"}
        if is_wsl:
            conflicts = [
                process
                for process in self.gpu_processes
                if "comfyui" in process.process_name.lower()
                or process.process_name.lower().endswith("python.exe")
            ]
            if conflicts:
                with self.lock:
                    self.last_decision = "wsl-blocked-by-windows-cuda-context"
                return
        self._launch(row, reason)

    def _launch(self, job: dict, reason: str) -> None:
        cwd = Path(job["cwd"])
        if not cwd.is_dir():
            self.database.mark_finished(
                job["id"],
                status="failed",
                exit_code=None,
                error=f"Working directory does not exist: {cwd}",
            )
            return
        argv = job["argv"]
        if isinstance(argv, str):
            argv = json.loads(argv)
        self.config.log_root.mkdir(parents=True, exist_ok=True)
        log_path = self.config.log_root / f"{job['id']}.log"
        log_handle = log_path.open("ab", buffering=0)
        creationflags = 0
        if os.name == "nt":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
        prepared_boundary = None
        try:
            prepared_boundary = self._arm_workload_boundary_before_launch(job)
            process = subprocess.Popen(
                argv,
                cwd=str(cwd),
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                shell=False,
                creationflags=creationflags,
                env=os.environ.copy(),
            )
        except Exception as exc:
            log_handle.close()
            if prepared_boundary is not None:
                try:
                    self._clear_unlaunched_workload_boundary(prepared_boundary)
                except Exception as boundary_error:
                    # No child was launched, but inability to clear the persisted
                    # arm must still fail closed in the current daemon.
                    failed_boundary = {
                        **prepared_boundary,
                        "status": "active",
                        "reason": "failed-to-clear-prelaunch-boundary",
                    }
                    with self.lock:
                        self.workload_reboot_boundary = failed_boundary
                        self.last_decision = "workload-reboot-boundary-required"
                        self.next_gpu_probe_monotonic = float("inf")
                    self.last_error = (
                        "Failed to clear prelaunch boundary: "
                        f"{type(boundary_error).__name__}: {boundary_error}"
                    )[:1000]
            self.database.mark_finished(
                job["id"],
                status="failed",
                exit_code=None,
                error=f"Launch failed: {type(exc).__name__}: {exc}",
            )
            return
        record = RunningProcess(
            process=process,
            log_handle=log_handle,
            started_monotonic=time.monotonic(),
            max_runtime_seconds=job["max_runtime_seconds"],
            workload_key=job["workload_key"],
            argv=tuple(argv),
            baseline_used_mb=self.baseline_used_mb,
        )
        with self.lock:
            self.running[job["id"]] = record
        self.database.mark_running(
            job["id"],
            pid=process.pid,
            log_path=str(log_path),
            scheduling_note=reason,
        )


def _candidate(row: dict) -> Candidate:
    return Candidate(
        id=row["id"],
        priority=row["priority"],
        submitted_at=datetime.fromisoformat(row["submitted_at"]),
        requested_vram_mb=row["requested_vram_mb"],
        estimated_seconds=row["estimated_seconds"],
        recent_vram_seconds=int(row.get("recent_vram_seconds", 0)),
        manual_rank=row.get("manual_rank"),
    )
