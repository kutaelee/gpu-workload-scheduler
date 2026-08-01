from types import SimpleNamespace

import pytest

from gpuq.gpu import GpuTelemetryError, read_gpu, read_gpu_processes


def test_process_census_preserves_unknown_wddm_memory(monkeypatch):
    def fake_run(*_args, **_kwargs):
        return SimpleNamespace(
            returncode=0,
            stdout="""25360, C:\\Python\\python.exe, [N/A]\n4242, worker.exe, 512\ninvalid\n""",
            stderr="",
        )

    monkeypatch.setattr("gpuq.gpu.subprocess.run", fake_run)

    processes = read_gpu_processes()

    assert [process.pid for process in processes] == [25360, 4242]
    assert processes[0].used_mb is None
    assert processes[1].used_mb == 512
    assert "command" not in processes[0].to_dict()


def test_gpu_lost_exit_is_classified_as_fatal(monkeypatch):
    monkeypatch.setattr(
        "gpuq.gpu.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=6,
            stdout="",
            stderr="Unable to determine the device handle: GPU is lost. Reboot required.",
        ),
    )

    with pytest.raises(GpuTelemetryError) as caught:
        read_gpu()

    assert caught.value.fatal is True
    assert caught.value.returncode == 6
