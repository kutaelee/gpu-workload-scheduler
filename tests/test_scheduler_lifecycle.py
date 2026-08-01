from io import BytesIO
import json
import time
from types import SimpleNamespace

from gpuq.gpu import GpuProcess, GpuTelemetry, GpuTelemetryError
from gpuq.scheduler import RunningProcess, Scheduler


class FakeProcess:
    def __init__(self, exit_code=None):
        self.pid = 4242
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
    scheduler.high_load_transition_gate = None
    scheduler.transition_state_path = None
    scheduler.workload_boundary_state_path = None
    scheduler.workload_reboot_boundary = None
    scheduler.boot_epoch = 123456
    scheduler.no_touch_until_monotonic = 0.0
    scheduler.next_gpu_probe_monotonic = 0.0
    scheduler.baseline_used_mb = 1000
    scheduler.gpu_healthy = True
    scheduler.gpu_fault_status = "healthy"
    scheduler.gpu_health_consecutive_failures = 0
    scheduler.gpu_health_recovery_successes = 0
    scheduler.gpu_health_last_failure_at = None
    scheduler.gpu_health_last_recovered_at = None
    scheduler.gpu_processes = []
    scheduler.process_scan_error = None
    scheduler.gpu_process_scan_generation = 0
    scheduler.config = SimpleNamespace(
        gpu_telemetry_enabled=True,
        poll_seconds=2.0,
        cancel_grace_seconds=30.0,
        terminate_grace_seconds=10.0,
        post_job_cooldown_seconds=2.0,
        post_job_no_touch_seconds=15.0,
        post_high_load_no_touch_seconds=20.0,
        post_high_load_probe_interval_seconds=5.0,
        post_high_load_stable_samples=3,
        post_high_load_process_stable_scans=2,
        post_high_load_vram_tolerance_mb=4096,
        post_high_load_max_idle_utilization_percent=5,
        high_load_min_peak_used_mb=24576,
        gpu_health_recovery_samples=3,
        gpu_telemetry_log_interval_seconds=10.0,
        wsl_force_terminate=wsl_force_terminate,
        reboot_boundary_workloads=frozenset(),
        reboot_boundary_workload_patterns=(),
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
        baseline_used_mb=1000,
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
    assert scheduler.admission_not_before_monotonic == 115.0
    assert scheduler.high_load_transition_gate is not None


def test_terminal_process_is_reaped_without_gpu_telemetry(monkeypatch):
    scheduler, _clock = make_scheduler(monkeypatch)
    process = FakeProcess(exit_code=1)
    record = make_record(process)
    scheduler.running = {"job-lost-gpu": record}
    calls = []
    scheduler.database = SimpleNamespace(
        get_job=lambda _job_id: {
            "cancel_requested": True,
            "peak_total_gpu_used_mb": 9000,
        },
        update_peak=lambda *_args: (_ for _ in ()).throw(
            AssertionError("peak must not update from missing telemetry")
        ),
        mark_finished=lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    transitioned = scheduler._reap_and_cancel(None)

    assert transitioned is True
    assert calls[0][1]["status"] == "failed"
    assert scheduler.running == {}


def test_scheduler_reaps_terminal_process_before_any_gpu_probe(monkeypatch):
    scheduler, _clock = make_scheduler(monkeypatch)
    scheduler.running = {"job-terminal": make_record(FakeProcess(exit_code=0))}
    scheduler.database = SimpleNamespace(
        get_job=lambda _job_id: {
            "cancel_requested": False,
            "peak_total_gpu_used_mb": 9000,
        },
        update_peak=lambda *_args: None,
        mark_finished=lambda *_args, **_kwargs: None,
        set_state=lambda *_args, **_kwargs: None,
    )
    scheduler.snapshot = lambda: {}

    class OneIterationStop:
        stopped = False

        def is_set(self):
            return self.stopped

        def wait(self, _seconds):
            self.stopped = True

    scheduler.stop_event = OneIterationStop()
    monkeypatch.setattr(
        "gpuq.scheduler.read_gpu",
        lambda: (_ for _ in ()).throw(
            AssertionError("terminal transition must precede GPU probe")
        ),
    )

    scheduler._loop()

    assert scheduler.running == {}
    assert scheduler.last_decision == "post-job-no-touch"


def test_configured_workload_latches_same_boot_reboot_boundary(
    monkeypatch, tmp_path
):
    scheduler, clock = make_scheduler(monkeypatch)
    scheduler.workload_boundary_state_path = tmp_path / "gpu-workload-boundary.json"
    scheduler.provenance = {"git_commit": "test"}
    scheduler.config.reboot_boundary_workloads = frozenset({"training-job"})
    process = FakeProcess(exit_code=0)
    scheduler.running = {"job-boundary": make_record(process)}
    scheduler.database = SimpleNamespace(
        get_job=lambda _job_id: {
            "cancel_requested": False,
            "peak_total_gpu_used_mb": 20000,
        },
        update_peak=lambda *_args: None,
        mark_finished=lambda *_args, **_kwargs: None,
    )

    scheduler._reap_and_cancel(GpuTelemetry("GPU", 32000, 1000, 31000, 0))

    state = json.loads(
        scheduler.workload_boundary_state_path.read_text(encoding="utf-8")
    )
    assert state["workload_key"] == "training-job"
    assert state["windows_boot_epoch"] == 123456
    assert scheduler.workload_reboot_boundary == state
    assert scheduler.admission_ready() is False
    assert (
        scheduler._gpu_probe_block_reason(clock.value)
        == "workload-reboot-boundary-required"
    )


def test_pattern_matched_workload_latches_same_boot_reboot_boundary(
    monkeypatch, tmp_path
):
    scheduler, _clock = make_scheduler(monkeypatch)
    scheduler.workload_boundary_state_path = tmp_path / "gpu-workload-boundary.json"
    scheduler.provenance = {"git_commit": "test"}
    scheduler.config.reboot_boundary_workload_patterns = (
        "wedding-*klein9b*qfloat8*",
    )
    process = FakeProcess(exit_code=0)
    record = make_record(process)
    record.workload_key = "wedding-v289-klein9b-groom-r64-qfloat8-lora3000"
    scheduler.running = {"job-pattern-boundary": record}
    scheduler.database = SimpleNamespace(
        get_job=lambda _job_id: {
            "cancel_requested": False,
            "peak_total_gpu_used_mb": 20000,
        },
        update_peak=lambda *_args: None,
        mark_finished=lambda *_args, **_kwargs: None,
    )

    scheduler._reap_and_cancel(GpuTelemetry("GPU", 32000, 1000, 31000, 0))

    assert scheduler.workload_reboot_boundary is not None
    assert scheduler.workload_reboot_boundary["workload_key"] == record.workload_key


def test_reboot_boundary_pattern_does_not_match_unrelated_workload(monkeypatch):
    scheduler, _clock = make_scheduler(monkeypatch)
    scheduler.config.reboot_boundary_workload_patterns = (
        "wedding-*klein9b*qfloat8*",
    )

    assert scheduler._requires_reboot_boundary("comfyui-prompt") is False


def test_runtime_alone_does_not_trigger_high_capacity_transition(monkeypatch):
    scheduler, clock = make_scheduler(monkeypatch)
    process = FakeProcess(exit_code=0)
    record = make_record(process)
    record.started_monotonic = clock.value - 2000
    scheduler.running = {"job-long": record}
    scheduler.database = SimpleNamespace(
        get_job=lambda _job_id: {
            "cancel_requested": False,
            "peak_total_gpu_used_mb": 19000,
        },
        update_peak=lambda *_args: None,
        mark_finished=lambda *_args, **_kwargs: None,
    )

    scheduler._reap_and_cancel(GpuTelemetry("GPU", 32000, 1000, 31000, 0))

    assert scheduler.admission_not_before_monotonic == 115.0
    assert scheduler.high_load_transition_gate is not None
    assert scheduler.high_load_transition_gate.job_ids == ("job-long",)


def test_high_load_gate_requires_post_cooldown_telemetry_and_process_stability(
    monkeypatch,
):
    scheduler, clock = make_scheduler(monkeypatch)
    record = make_record(FakeProcess(exit_code=0))
    scheduler._arm_high_load_transition_gate("job-long", record)
    telemetry = GpuTelemetry("GPU", 32000, 2000, 30000, 2)

    assert scheduler._evaluate_high_load_transition_gate(telemetry) is False
    assert scheduler.high_load_transition_gate.reason == "minimum-cooldown"

    clock.value = 121.0
    scheduler.gpu_processes = [GpuProcess(42, "comfyui.exe", 1000)]
    scheduler.gpu_process_scan_generation = 1
    assert scheduler._evaluate_high_load_transition_gate(telemetry) is False

    scheduler.gpu_process_scan_generation = 2
    assert scheduler._evaluate_high_load_transition_gate(telemetry) is False
    assert scheduler._evaluate_high_load_transition_gate(telemetry) is True
    assert scheduler.high_load_transition_gate is None
    assert scheduler.baseline_used_mb == 2000


def test_high_load_gate_resets_when_vram_is_not_idle(monkeypatch):
    scheduler, clock = make_scheduler(monkeypatch)
    record = make_record(FakeProcess(exit_code=0))
    scheduler._arm_high_load_transition_gate("job-long", record)
    clock.value = 121.0
    scheduler.gpu_process_scan_generation = 1

    busy = GpuTelemetry("GPU", 32000, 6000, 26000, 2)
    assert scheduler._evaluate_high_load_transition_gate(busy) is False
    assert scheduler.high_load_transition_gate.stable_samples == 0
    assert scheduler.high_load_transition_gate.reason.startswith("vram-not-idle")


def test_fatal_gpu_loss_is_persistently_latched(monkeypatch, tmp_path):
    scheduler, _clock = make_scheduler(monkeypatch)
    scheduler.fault_state_path = tmp_path / "gpu-fault-state.json"
    scheduler.boot_epoch = 123456
    scheduler.provenance = {"git_commit": "test"}
    scheduler.persisted_fault = None
    scheduler.transient_failure_backoff_index = 0
    scheduler._last_gpu_health_log_signature = None
    scheduler.config.log_root = tmp_path

    scheduler._record_gpu_failure(
        GpuTelemetryError("GPU is lost; reboot required", returncode=6, fatal=True)
    )

    state = json.loads(scheduler.fault_state_path.read_text(encoding="utf-8"))
    assert scheduler.gpu_fault_status == "reboot_required"
    assert scheduler.gpu_healthy is False
    assert state["status"] == "reboot_required"
    assert state["windows_boot_epoch"] == 123456
    assert scheduler.next_gpu_probe_monotonic == float("inf")
    monkeypatch.setattr(
        "gpuq.scheduler.read_gpu",
        lambda: (_ for _ in ()).throw(AssertionError("must not probe same boot")),
    )
    assert scheduler.rearm_gpu() == {
        "rearmed": False,
        "reason": "windows-reboot-required",
    }


def test_gpu_query_timeout_is_fatal_during_reboot_boundary_workload(
    monkeypatch, tmp_path
):
    scheduler, _clock = make_scheduler(monkeypatch)
    scheduler.fault_state_path = tmp_path / "gpu-fault-state.json"
    scheduler.provenance = {"git_commit": "test"}
    scheduler.persisted_fault = None
    scheduler.transient_failure_backoff_index = 0
    scheduler._last_gpu_health_log_signature = None
    scheduler.config.log_root = tmp_path
    scheduler.config.reboot_boundary_workloads = frozenset({"training-job"})
    scheduler.running = {"job-protected": make_record(FakeProcess())}

    scheduler._record_gpu_failure(
        __import__("subprocess").TimeoutExpired("nvidia-smi", 10)
    )

    state = json.loads(scheduler.fault_state_path.read_text(encoding="utf-8"))
    assert state["status"] == "reboot_required"
    assert scheduler.gpu_fault_status == "reboot_required"
    assert scheduler.next_gpu_probe_monotonic == float("inf")


def test_boundary_write_failure_still_latches_current_daemon(monkeypatch, tmp_path):
    scheduler, _clock = make_scheduler(monkeypatch)
    scheduler.workload_boundary_state_path = tmp_path / "gpu-workload-boundary.json"
    scheduler.provenance = {"git_commit": "test"}
    record = make_record(FakeProcess(exit_code=0))
    monkeypatch.setattr(
        "gpuq.scheduler.write_fault_state",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )

    try:
        scheduler._arm_workload_reboot_boundary(
            "job-boundary-write-failed", record, "succeeded"
        )
    except OSError:
        pass
    else:
        raise AssertionError("state write failure must propagate")

    assert scheduler.workload_reboot_boundary is not None
    assert scheduler.workload_reboot_boundary["status"] == "active"
    assert scheduler.next_gpu_probe_monotonic == float("inf")


def test_no_touch_window_blocks_gpu_probe(monkeypatch):
    scheduler, clock = make_scheduler(monkeypatch)
    scheduler.no_touch_until_monotonic = clock.value + 15.0

    reason = scheduler._gpu_probe_block_reason(clock.value)

    assert reason == "post-high-load-no-touch:15.0s"


def test_same_boot_fatal_latch_prevents_startup_gpu_probe(monkeypatch, tmp_path):
    runtime = tmp_path / ".runtime"
    runtime.mkdir()
    fault = runtime / "gpu-fault-state.json"
    fault.write_text(
        json.dumps(
            {
                "status": "reboot_required",
                "windows_boot_epoch": 123456,
                "reason": "GPU is lost",
            }
        ),
        encoding="utf-8",
    )
    config_path = runtime / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    config = SimpleNamespace(
        config_path=config_path,
        gpu_telemetry_enabled=True,
    )
    monkeypatch.setattr("gpuq.scheduler.REPO_ROOT", tmp_path)
    monkeypatch.setattr("gpuq.scheduler.windows_boot_epoch", lambda: 123456)
    monkeypatch.setattr("gpuq.scheduler.runtime_provenance", lambda *_args: {})
    monkeypatch.setattr(
        "gpuq.scheduler.read_gpu",
        lambda: (_ for _ in ()).throw(AssertionError("must not probe GPU")),
    )
    scheduler = Scheduler(config, SimpleNamespace())
    started = []
    scheduler.thread = SimpleNamespace(start=lambda: started.append(True))

    scheduler.start()

    assert started == [True]
    assert scheduler.gpu_fault_status == "reboot_required"
    assert scheduler.gpu_healthy is False


def test_same_boot_workload_boundary_prevents_startup_gpu_probe(
    monkeypatch, tmp_path
):
    runtime = tmp_path / ".runtime"
    runtime.mkdir()
    (runtime / "gpu-workload-reboot-boundary.json").write_text(
        json.dumps(
            {
                "status": "active",
                "windows_boot_epoch": 123456,
                "workload_key": "training-job",
            }
        ),
        encoding="utf-8",
    )
    config_path = runtime / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    config = SimpleNamespace(
        config_path=config_path,
        gpu_telemetry_enabled=True,
    )
    monkeypatch.setattr("gpuq.scheduler.REPO_ROOT", tmp_path)
    monkeypatch.setattr("gpuq.scheduler.windows_boot_epoch", lambda: 123456)
    monkeypatch.setattr("gpuq.scheduler.runtime_provenance", lambda *_args: {})
    monkeypatch.setattr(
        "gpuq.scheduler.read_gpu",
        lambda: (_ for _ in ()).throw(AssertionError("must not probe GPU")),
    )
    scheduler = Scheduler(config, SimpleNamespace())
    started = []
    scheduler.thread = SimpleNamespace(start=lambda: started.append(True))

    scheduler.start()

    assert started == [True]
    assert scheduler.workload_reboot_boundary is not None
    assert scheduler.last_decision == "workload-reboot-boundary-required"


def test_same_boot_armed_workload_boundary_prevents_startup_gpu_probe(
    monkeypatch, tmp_path
):
    runtime = tmp_path / ".runtime"
    runtime.mkdir()
    (runtime / "gpu-workload-reboot-boundary.json").write_text(
        json.dumps(
            {
                "status": "armed",
                "windows_boot_epoch": 123456,
                "workload_key": "training-job",
            }
        ),
        encoding="utf-8",
    )
    config_path = runtime / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    config = SimpleNamespace(
        config_path=config_path,
        gpu_telemetry_enabled=True,
    )
    monkeypatch.setattr("gpuq.scheduler.REPO_ROOT", tmp_path)
    monkeypatch.setattr("gpuq.scheduler.windows_boot_epoch", lambda: 123456)
    monkeypatch.setattr("gpuq.scheduler.runtime_provenance", lambda *_args: {})
    monkeypatch.setattr(
        "gpuq.scheduler.read_gpu",
        lambda: (_ for _ in ()).throw(AssertionError("must not probe GPU")),
    )
    scheduler = Scheduler(config, SimpleNamespace())
    started = []
    scheduler.thread = SimpleNamespace(start=lambda: started.append(True))

    scheduler.start()

    assert started == [True]
    assert scheduler.workload_reboot_boundary["status"] == "armed"
    assert scheduler.last_decision == "workload-reboot-boundary-required"


def test_reboot_boundary_is_persisted_before_process_launch(monkeypatch, tmp_path):
    scheduler, _clock = make_scheduler(monkeypatch)
    scheduler.workload_boundary_state_path = tmp_path / "gpu-workload-boundary.json"
    scheduler.provenance = {"git_commit": "test"}
    scheduler.config.reboot_boundary_workloads = frozenset({"training-job"})
    observed = []

    def fake_popen(*_args, **_kwargs):
        observed.append(
            json.loads(
                scheduler.workload_boundary_state_path.read_text(encoding="utf-8")
            )
        )
        return FakeProcess()

    monkeypatch.setattr("gpuq.scheduler.subprocess.Popen", fake_popen)
    scheduler.database = SimpleNamespace(mark_running=lambda *_args, **_kwargs: None)
    scheduler.config.log_root = tmp_path
    job = {
        "id": "job-prearmed",
        "cwd": str(tmp_path),
        "argv": ["worker.exe"],
        "max_runtime_seconds": 600,
        "workload_key": "training-job",
    }

    scheduler._launch(job, "head-fits")

    assert observed[0]["status"] == "armed"
    assert observed[0]["job_id"] == "job-prearmed"
    assert "job-prearmed" in scheduler.running


def test_same_boot_no_touch_transition_survives_daemon_restart(
    monkeypatch, tmp_path
):
    runtime = tmp_path / ".runtime"
    runtime.mkdir()
    (runtime / "gpu-transition-state.json").write_text(
        json.dumps(
            {
                "status": "active",
                "windows_boot_epoch": 123456,
                "job_ids": ["job-prior"],
                "no_touch_until_unix": time.time() + 15.0,
                "baseline_used_mb": 2048,
            }
        ),
        encoding="utf-8",
    )
    config_path = runtime / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr("gpuq.scheduler.REPO_ROOT", tmp_path)
    monkeypatch.setattr("gpuq.scheduler.windows_boot_epoch", lambda: 123456)
    monkeypatch.setattr("gpuq.scheduler.runtime_provenance", lambda *_args: {})

    scheduler = Scheduler(
        SimpleNamespace(config_path=config_path), SimpleNamespace()
    )

    assert scheduler.high_load_transition_gate is not None
    assert scheduler.high_load_transition_gate.job_ids == ("job-prior",)
    assert scheduler.high_load_transition_gate.baseline_used_mb == 2048
    assert scheduler.no_touch_until_monotonic > time.monotonic()
