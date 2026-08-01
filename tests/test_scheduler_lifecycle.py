from io import BytesIO
from types import SimpleNamespace

from gpuq.gpu import GpuTelemetry
from gpuq.scheduler import RunningProcess, Scheduler


class FakeProcess:
    def __init__(self, exit_code=None):
        self.exit_code = exit_code
        self.signals = []
        self.terminated = 0
        self.killed = 0

    def poll(self):
        return self.exit_code

    def send_signal(self, value):
        self.signals.append(value)

    def terminate(self):
        self.terminated += 1

    def kill(self):
        self.killed += 1


def make_scheduler(monkeypatch, *, wsl_force_terminate=False, cleanup_commands=None):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.lock = __import__("threading").Lock()
    scheduler.running = {}
    scheduler.last_error = None
    scheduler.admission_not_before_monotonic = 0.0
    scheduler.config = SimpleNamespace(
        cancel_grace_seconds=30.0,
        terminate_grace_seconds=10.0,
        post_job_cooldown_seconds=2.0,
        wsl_force_terminate=wsl_force_terminate,
        cleanup_commands=cleanup_commands or {},
    )
    clock = SimpleNamespace(value=100.0)
    monkeypatch.setattr("gpuq.scheduler.time.monotonic", lambda: clock.value)
    return scheduler, clock


def make_record(process, argv=("python.exe", "job.py")):
    return RunningProcess(
        process=process,
        log_handle=BytesIO(),
        started_monotonic=0.0,
        max_runtime_seconds=600,
        workload_key="training-job",
        argv=argv,
    )


def test_wsl_cancel_refuses_wrapper_only_force_kill(monkeypatch):
    scheduler, clock = make_scheduler(monkeypatch)
    process = FakeProcess()
    record = make_record(
        process,
        ("wsl.exe", "-d", "Ubuntu", "--", "python", "job.py"),
    )

    scheduler._request_termination("job-1", record, "canceled")
    assert process.signals

    clock.value += 31
    scheduler._request_termination("job-1", record, "canceled")
    assert process.terminated == 0
    assert process.killed == 0
    assert "refusing to kill only the Windows wrapper" in scheduler.last_error


def test_native_cancel_escalates_after_bounded_grace(monkeypatch):
    scheduler, clock = make_scheduler(monkeypatch)
    process = FakeProcess()
    record = make_record(process)

    scheduler._request_termination("job-2", record, "canceled")
    clock.value += 31
    scheduler._request_termination("job-2", record, "canceled")
    assert process.terminated == 1

    clock.value += 11
    scheduler._request_termination("job-2", record, "canceled")
    assert process.killed == 1


def test_terminal_failure_runs_cleanup_once_and_pauses_admission(monkeypatch):
    scheduler, _clock = make_scheduler(
        monkeypatch,
        cleanup_commands={"training-job": ("cleanup.exe", "job-3")},
    )
    process = FakeProcess(exit_code=1)
    record = make_record(process)
    scheduler.running = {"job-3": record}
    calls = []
    scheduler.database = SimpleNamespace(
        get_job=lambda _job_id: {"cancel_requested": False},
        update_peak=lambda *_args: None,
        mark_finished=lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        "gpuq.scheduler.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0),
    )

    transitioned = scheduler._reap_and_cancel(
        GpuTelemetry("GPU", 32000, 1000, 31000, 0)
    )

    assert transitioned is True
    assert record.cleanup_attempted is True
    assert calls[0][1]["status"] == "failed"
    assert scheduler.admission_not_before_monotonic == 102.0
