from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / ".runtime" / "config.json"


@dataclass(frozen=True)
class Config:
    config_path: Path
    database_url: str
    api_token: str
    api_host: str
    api_port: int
    log_root: Path
    safety_vram_mb: int
    fairness_window_minutes: int
    max_parallel_jobs: int
    gpu_telemetry_enabled: bool
    poll_seconds: float
    process_scan_interval_seconds: float
    cancel_grace_seconds: float
    terminate_grace_seconds: float
    post_job_cooldown_seconds: float
    post_job_no_touch_seconds: float
    post_high_load_no_touch_seconds: float
    post_high_load_probe_interval_seconds: float
    post_high_load_stable_samples: int
    post_high_load_process_stable_scans: int
    post_high_load_vram_tolerance_mb: int
    post_high_load_max_idle_utilization_percent: int
    high_load_min_peak_used_mb: int
    gpu_health_recovery_samples: int
    gpu_telemetry_log_interval_seconds: float
    wsl_force_terminate: bool
    reboot_boundary_workloads: frozenset[str] = field(default_factory=frozenset)
    cleanup_commands: dict[str, tuple[str, ...]] = field(default_factory=dict)
    external_workloads: dict[str, dict[str, str]] = field(default_factory=dict)

    @classmethod
    def load(cls) -> Config:
        path = Path(os.environ.get("GPUQ_CONFIG", DEFAULT_CONFIG_PATH))
        raw = json.loads(path.read_text(encoding="utf-8"))
        cleanup_commands: dict[str, tuple[str, ...]] = {}
        for workload, argv in (raw.get("cleanup_commands") or {}).items():
            if (
                isinstance(workload, str)
                and workload
                and isinstance(argv, list)
                and argv
                and all(isinstance(item, str) and item for item in argv)
            ):
                cleanup_commands[workload[:120]] = tuple(argv)
        external_workloads: dict[str, dict[str, str]] = {}
        for key, value in (raw.get("external_workloads") or {}).items():
            if not isinstance(key, str) or not re.fullmatch(
                r"[a-z0-9][a-z0-9._-]{0,119}", key
            ):
                continue
            if not isinstance(value, dict):
                continue
            label = value.get("label")
            if not isinstance(label, str) or not label.strip():
                continue
            if value.get("kind") == "ollama_container":
                container = value.get("container")
                if not isinstance(container, str) or not re.fullmatch(
                    r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", container
                ):
                    continue
                external_workloads[key] = {
                    "kind": "ollama_container",
                    "container": container,
                    "label": label.strip()[:120],
                }
            elif value.get("kind") == "ollama_host":
                executable = value.get("executable")
                executable_path = (
                    Path(executable) if isinstance(executable, str) else None
                )
                if (
                    executable_path is None
                    or not executable_path.is_absolute()
                    or executable_path.name.lower() != "ollama.exe"
                ):
                    continue
                external_workloads[key] = {
                    "kind": "ollama_host",
                    "executable": str(executable_path),
                    "label": label.strip()[:120],
                }
        reboot_boundary_workloads = frozenset(
            workload
            for workload in (raw.get("reboot_boundary_workloads") or [])
            if isinstance(workload, str)
            and re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,119}", workload)
        )
        return cls(
            config_path=path,
            database_url=raw["database_url"],
            api_token=raw["api_token"],
            api_host=raw.get("api_host", "127.0.0.1"),
            api_port=int(raw.get("api_port", 8790)),
            log_root=Path(raw.get("log_root", r"E:\Data\GpuScheduler\Logs")),
            safety_vram_mb=int(raw.get("safety_vram_mb", 2048)),
            fairness_window_minutes=int(raw.get("fairness_window_minutes", 60)),
            max_parallel_jobs=max(1, int(raw.get("max_parallel_jobs", 1))),
            gpu_telemetry_enabled=bool(raw.get("gpu_telemetry_enabled", True)),
            poll_seconds=float(raw.get("poll_seconds", 2.0)),
            process_scan_interval_seconds=max(
                30.0, float(raw.get("process_scan_interval_seconds", 30.0))
            ),
            cancel_grace_seconds=float(raw.get("cancel_grace_seconds", 30.0)),
            terminate_grace_seconds=float(raw.get("terminate_grace_seconds", 10.0)),
            post_job_cooldown_seconds=float(raw.get("post_job_cooldown_seconds", 2.0)),
            post_job_no_touch_seconds=max(
                15.0, float(raw.get("post_job_no_touch_seconds", 15.0))
            ),
            post_high_load_no_touch_seconds=max(
                20.0, float(raw.get("post_high_load_no_touch_seconds", 20.0))
            ),
            post_high_load_probe_interval_seconds=max(
                5.0,
                float(raw.get("post_high_load_probe_interval_seconds", 5.0)),
            ),
            post_high_load_stable_samples=max(
                3, int(raw.get("post_high_load_stable_samples", 3))
            ),
            post_high_load_process_stable_scans=max(
                1, int(raw.get("post_high_load_process_stable_scans", 2))
            ),
            post_high_load_vram_tolerance_mb=max(
                0, int(raw.get("post_high_load_vram_tolerance_mb", 4096))
            ),
            post_high_load_max_idle_utilization_percent=max(
                0,
                min(
                    100,
                    int(raw.get("post_high_load_max_idle_utilization_percent", 5)),
                ),
            ),
            high_load_min_peak_used_mb=int(
                raw.get("high_load_min_peak_used_mb", 24576)
            ),
            gpu_health_recovery_samples=max(
                1, int(raw.get("gpu_health_recovery_samples", 3))
            ),
            gpu_telemetry_log_interval_seconds=max(
                2.0, float(raw.get("gpu_telemetry_log_interval_seconds", 10.0))
            ),
            wsl_force_terminate=bool(raw.get("wsl_force_terminate", False)),
            reboot_boundary_workloads=reboot_boundary_workloads,
            cleanup_commands=cleanup_commands,
            external_workloads=external_workloads,
        )
