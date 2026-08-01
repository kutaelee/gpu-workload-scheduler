from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .config import Config
from .db import Database
from .external import (
    inspect_ollama_container,
    inspect_ollama_host,
    stop_ollama_container,
    stop_ollama_host,
)
from .gpu import GpuProcess, GpuTelemetry, read_gpu, read_gpu_processes
from .policy import ActiveReservation, Candidate, choose_candidate, effective_score


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
        self.last_external_scan_monotonic = 0.0
        self.external_workloads: list[dict] = []
        self.admission_not_before_monotonic = 0.0

    def start(self) -> None:
        telemetry = read_gpu()
        self.telemetry = telemetry
        self.baseline_used_mb = telemetry.used_mb
        self._refresh_gpu_processes(force=True)
        self._refresh_external_workloads(force=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=10)

    def snapshot(self) -> dict:
        with self.lock:
            telemetry = self.telemetry.to_dict() if self.telemetry else None
            managed_pids = {record.process.pid for record in self.running.values()}
            unmanaged_processes = [
                process.to_dict()
                for process in self.gpu_processes
                if process.pid not in managed_pids
            ]
            return {
                "gpu": telemetry,
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
            }

    def scores_for(self, queued: list[dict]) -> dict[str, float]:
        telemetry = self.telemetry or read_gpu()
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

    def _loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                telemetry = read_gpu()
                with self.lock:
                    self.telemetry = telemetry
                    self.last_error = None
                self._refresh_gpu_processes()
                self._refresh_external_workloads()
                terminal_transition = self._reap_and_cancel(telemetry)
                active_rows = self.database.active_jobs()
                if not active_rows:
                    self.baseline_used_mb = telemetry.used_mb
                if terminal_transition:
                    with self.lock:
                        self.last_decision = "post-job-cooldown"
                else:
                    self._schedule_once(telemetry, active_rows)
                self.database.set_state("runtime", self.snapshot())
            except Exception as exc:
                with self.lock:
                    self.last_error = f"{type(exc).__name__}: {exc}"
            self.stop_event.wait(self.config.poll_seconds)

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
            with self.lock:
                self.process_scan_error = f"{type(exc).__name__}: {exc}"
                self.last_process_scan_monotonic = now
            return
        with self.lock:
            self.gpu_processes = processes
            self.process_scan_error = None
            self.last_process_scan_monotonic = now

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

    def _reap_and_cancel(self, telemetry: GpuTelemetry) -> bool:
        terminal_transition = False
        with self.lock:
            running_items = list(self.running.items())
        for job_id, record in running_items:
            job = self.database.get_job(job_id)
            if job:
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
                self._pause_admission()
                terminal_transition = True
                continue
            if job and job.get("cancel_requested"):
                self._request_termination(job_id, record, "canceled")
                continue
            if time.monotonic() - record.started_monotonic > record.max_runtime_seconds:
                self._request_termination(job_id, record, "timed_out")
        return terminal_transition

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

    def _pause_admission(self) -> None:
        self.admission_not_before_monotonic = max(
            self.admission_not_before_monotonic,
            time.monotonic() + self.config.post_job_cooldown_seconds,
        )

    def _schedule_once(self, telemetry: GpuTelemetry, active_rows: list[dict]) -> None:
        cooldown_remaining = self.admission_not_before_monotonic - time.monotonic()
        if cooldown_remaining > 0:
            with self.lock:
                self.last_decision = f"post-job-cooldown:{cooldown_remaining:.1f}s"
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
        try:
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
