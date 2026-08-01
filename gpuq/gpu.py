from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass


class GpuTelemetryError(RuntimeError):
    """Bounded, secret-free NVIDIA telemetry failure details."""

    def __init__(self, message: str, *, returncode: int | None = None):
        super().__init__(message[:1000])
        self.returncode = returncode


@dataclass(frozen=True)
class GpuTelemetry:
    name: str
    total_mb: int
    used_mb: int
    free_mb: int
    utilization_percent: int
    temperature_c: int | None = None
    power_draw_w: float | None = None
    power_limit_w: float | None = None
    graphics_clock_mhz: int | None = None
    memory_clock_mhz: int | None = None
    pcie_generation: int | None = None
    pcie_width: int | None = None
    driver_version: str | None = None

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
            "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,"
            "temperature.gpu,power.draw,power.limit,clocks.gr,clocks.mem,"
            "pcie.link.gen.current,pcie.link.width.current,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
        creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "no diagnostic output").strip()
        raise GpuTelemetryError(
            f"nvidia-smi telemetry failed with exit code {result.returncode}: {detail}",
            returncode=result.returncode,
        )
    lines = result.stdout.strip().splitlines()
    if not lines:
        raise GpuTelemetryError("nvidia-smi telemetry returned no GPU rows")
    try:
        parts = [part.strip() for part in lines[0].split(",")]
        if len(parts) != 13:
            raise ValueError(f"expected 13 columns, got {len(parts)}")
        (
            name,
            total,
            used,
            free,
            utilization,
            temperature,
            power_draw,
            power_limit,
            graphics_clock,
            memory_clock,
            pcie_generation,
            pcie_width,
            driver_version,
        ) = parts
        return GpuTelemetry(
            name=name,
            total_mb=int(total),
            used_mb=int(used),
            free_mb=int(free),
            utilization_percent=int(utilization),
            temperature_c=_optional_int(temperature),
            power_draw_w=_optional_float(power_draw),
            power_limit_w=_optional_float(power_limit),
            graphics_clock_mhz=_optional_int(graphics_clock),
            memory_clock_mhz=_optional_int(memory_clock),
            pcie_generation=_optional_int(pcie_generation),
            pcie_width=_optional_int(pcie_width),
            driver_version=driver_version or None,
        )
    except (TypeError, ValueError) as exc:
        raise GpuTelemetryError(
            f"nvidia-smi telemetry row could not be parsed: {lines[0][:500]}"
        ) from exc


def _optional_int(value: str) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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
