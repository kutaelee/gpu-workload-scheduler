from types import SimpleNamespace

import pytest

from gpuq.gpu import GpuTelemetryError, read_gpu


def test_read_gpu_parses_diagnostic_fields(monkeypatch):
    monkeypatch.setattr(
        "gpuq.gpu.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=(
                "NVIDIA GeForce RTX 5090, 32607, 9327, 22861, 5, 63, "
                "449.50, 450.00, 2407, 14001, 5, 16, 610.62\n"
            ),
            stderr="",
        ),
    )

    telemetry = read_gpu()

    assert telemetry.temperature_c == 63
    assert telemetry.power_draw_w == 449.5
    assert telemetry.pcie_width == 16
    assert telemetry.driver_version == "610.62"


def test_read_gpu_preserves_bounded_failure_detail(monkeypatch):
    monkeypatch.setattr(
        "gpuq.gpu.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=6,
            stdout="",
            stderr="GPU is lost. Reboot the system to recover this GPU",
        ),
    )

    with pytest.raises(GpuTelemetryError, match="GPU is lost"):
        read_gpu()
