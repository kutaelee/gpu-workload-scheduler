from types import SimpleNamespace

from gpuq.external import (
    inspect_ollama_container,
    inspect_ollama_host,
    stop_ollama_container,
    stop_ollama_host,
)


def completed(returncode=0, stdout="", stderr=""):
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


def test_ollama_probe_reports_loaded_models_without_inventing_vram(monkeypatch):
    calls = []
    responses = iter(
        [
            completed(stdout="true\n"),
            completed(
                stdout=(
                    "NAME  ID  SIZE  PROCESSOR  CONTEXT  UNTIL\n"
                    "qwen3-embedding:0.6b  ac6da0df  2.4 GB  100% CPU  4096  24 hours\n"
                )
            ),
        ]
    )

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return next(responses)

    monkeypatch.setattr("gpuq.external.subprocess.run", fake_run)
    status = inspect_ollama_container(
        "portal-ollama", key="portal-embedding", label="Portal embedding"
    )

    assert status["state"] == "active"
    assert status["models"][0]["name"] == "qwen3-embedding:0.6b"
    assert status["models"][0]["context"] == "4096"
    assert status["models"][0]["until"] == "24 hours"
    assert "used_mb" not in status["models"][0]
    assert all(call[1]["shell"] is False for call in calls)


def test_ollama_stop_uses_only_discovered_model_names(monkeypatch):
    calls = []
    responses = iter(
        [
            completed(stdout="true\n"),
            completed(
                stdout=(
                    "NAME  ID  SIZE  PROCESSOR  CONTEXT  UNTIL\n"
                    "model:1  abc  4 GB  100% GPU  4096  1h\n"
                )
            ),
            completed(),
            completed(stdout="true\n"),
            completed(stdout="NAME  ID  SIZE  PROCESSOR  CONTEXT  UNTIL\n"),
        ]
    )

    def fake_run(argv, **kwargs):
        calls.append(argv)
        return next(responses)

    monkeypatch.setattr("gpuq.external.subprocess.run", fake_run)
    status = stop_ollama_container(
        "portal-ollama", key="portal-embedding", label="Portal embedding"
    )

    assert ["docker", "exec", "portal-ollama", "ollama", "stop", "model:1"] in calls
    assert status["state"] == "idle"


def test_windows_ollama_probe_uses_only_configured_executable(monkeypatch):
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return completed(
            stdout=(
                "NAME  ID  SIZE  PROCESSOR  CONTEXT  UNTIL\n"
                "gemma4:12b  4eb23ef1  8.4 GB  100% GPU  32768  4 minutes\n"
            )
        )

    monkeypatch.setattr("gpuq.external.subprocess.run", fake_run)
    status = inspect_ollama_host(
        r"C:\Ollama\ollama.exe",
        key="windows-ollama-generation",
        label="Workstation Ollama",
    )

    assert calls[0][0] == [r"C:\Ollama\ollama.exe", "ps"]
    assert calls[0][1]["shell"] is False
    assert status["state"] == "active"
    assert status["models"][0]["name"] == "gemma4:12b"
    assert status["models"][0]["processor"] == "100% GPU"


def test_windows_ollama_stop_uses_discovered_model_name(monkeypatch):
    calls = []
    responses = iter(
        [
            completed(
                stdout=(
                    "NAME  ID  SIZE  PROCESSOR  CONTEXT  UNTIL\n"
                    "gemma4:12b  4eb23ef1  8.4 GB  100% GPU  32768  4 minutes\n"
                )
            ),
            completed(),
            completed(stdout="NAME  ID  SIZE  PROCESSOR  CONTEXT  UNTIL\n"),
        ]
    )

    def fake_run(argv, **kwargs):
        calls.append(argv)
        return next(responses)

    monkeypatch.setattr("gpuq.external.subprocess.run", fake_run)
    status = stop_ollama_host(
        r"C:\Ollama\ollama.exe",
        key="windows-ollama-generation",
        label="Workstation Ollama",
    )

    assert [r"C:\Ollama\ollama.exe", "stop", "gemma4:12b"] in calls
    assert status["state"] == "idle"
