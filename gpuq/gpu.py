from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class GpuTelemetry:
    name: str
    total_mb: int
    used_mb: int
    free_mb: int
    utilization_percent: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class GpuProcess:
    """A process observed by the NVIDIA driver, without command-line data."""

    pid: int
    process_name: str
    used_mb: int | None

    def to_dict(self) -> dict:
        return asdict(self)


def read_gpu() -> GpuTelemetry:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
        creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
    )
    line = result.stdout.strip().splitlines()[0]
    name, total, used, free, utilization = [part.strip() for part in line.split(",", 4)]
    return GpuTelemetry(
        name=name,
        total_mb=int(total),
        used_mb=int(used),
        free_mb=int(free),
        utilization_percent=int(utilization),
    )


def read_gpu_processes() -> list[GpuProcess]:
    """Return a bounded NVIDIA process census for attribution only.

    Windows WDDM commonly reports ``[N/A]`` for per-process memory.  Preserve
    that uncertainty instead of inventing an allocation value, and never
    collect command lines or environment values into the scheduler API.
    """
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
        creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
    )
    processes: list[GpuProcess] = []
    for line in result.stdout.splitlines()[:128]:
        parts = [part.strip() for part in line.split(",", 2)]
        if len(parts) != 3:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        try:
            used_mb = int(parts[2])
        except ValueError:
            used_mb = None
        processes.append(
            GpuProcess(pid=pid, process_name=parts[1][:240], used_mb=used_mb)
        )
    return processes
