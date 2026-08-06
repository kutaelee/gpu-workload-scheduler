from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_daemon_waits_for_docker_without_a_boot_deadline():
    source = (ROOT / "scripts" / "Start-Daemon.ps1").read_text(encoding="utf-8-sig")

    assert "while ($true)" in source
    assert "no startup deadline" in source
    assert "attempt $attempt/90" not in source
    assert "exit 10" not in source


def test_scheduled_task_recovers_a_missed_logon_start():
    source = (ROOT / "scripts" / "Register-Task.ps1").read_text(encoding="utf-8-sig")

    assert "-StartWhenAvailable" in source
    assert "-ExecutionTimeLimit ([TimeSpan]::Zero)" in source
