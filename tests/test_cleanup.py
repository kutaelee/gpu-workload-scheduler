from io import BytesIO
from types import SimpleNamespace

from gpuq.scheduler import RunningProcess, Scheduler


def test_cleanup_uses_only_operator_configured_workload_argv(monkeypatch):
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("gpuq.scheduler.subprocess.run", fake_run)
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.config = SimpleNamespace(
        cleanup_commands={
            "portal-ollama": ("wsl.exe", "--", "sh", "cleanup.sh", "ollama-generation")
        }
    )
    record = RunningProcess(
        process=SimpleNamespace(),
        log_handle=BytesIO(),
        started_monotonic=0,
        max_runtime_seconds=60,
        workload_key="portal-ollama",
        argv=("wsl.exe", "--", "sh", "cleanup.sh"),
    )

    assert scheduler._run_cleanup(record) is None
    assert calls[0][0] == [
        "wsl.exe",
        "--",
        "sh",
        "cleanup.sh",
        "ollama-generation",
    ]
    assert calls[0][1]["shell"] is False

    record.workload_key = "unmanaged-workload"
    assert scheduler._run_cleanup(record) is None
    assert len(calls) == 1
