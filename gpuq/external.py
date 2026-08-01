from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass

CREATE_NO_WINDOW = (
    subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
)


@dataclass(frozen=True)
class OllamaModel:
    name: str
    model_id: str
    size: str
    processor: str
    context: str
    until: str

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "model_id": self.model_id,
            "size": self.size,
            "processor": self.processor,
            "context": self.context,
            "until": self.until,
        }


def _run(argv: list[str], *, timeout: float = 10) -> subprocess.CompletedProcess:
    return subprocess.run(
        argv,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
        shell=False,
        creationflags=CREATE_NO_WINDOW,
    )


def _parse_ollama_models(output: str) -> list[OllamaModel]:
    models: list[OllamaModel] = []
    for raw_line in output.splitlines()[1:17]:
        columns = re.split(r"\s{2,}", raw_line.strip())
        if len(columns) < 5:
            continue
        has_context_column = len(columns) >= 6
        models.append(
            OllamaModel(
                name=columns[0][:160],
                model_id=columns[1][:80],
                size=columns[2][:40],
                processor=columns[3][:40],
                context=columns[4][:40] if has_context_column else "",
                until=" · ".join(columns[5:] if has_context_column else columns[4:])[
                    :120
                ],
            )
        )
    return models


def inspect_ollama_container(container: str, *, key: str, label: str) -> dict:
    inspected = _run(
        ["docker", "inspect", "--format", "{{.State.Running}}", container]
    )
    if inspected.returncode != 0 or inspected.stdout.strip().lower() != "true":
        return {
            "key": key,
            "label": label,
            "kind": "ollama_container",
            "container": container,
            "state": "stopped",
            "models": [],
            "can_stop": False,
            "error": None,
        }
    result = _run(["docker", "exec", container, "ollama", "ps"])
    if result.returncode != 0:
        return {
            "key": key,
            "label": label,
            "kind": "ollama_container",
            "container": container,
            "state": "error",
            "models": [],
            "can_stop": False,
            "error": (result.stderr or result.stdout).strip()[:500],
        }
    models = _parse_ollama_models(result.stdout)
    return {
        "key": key,
        "label": label,
        "kind": "ollama_container",
        "container": container,
        "state": "active" if models else "idle",
        "models": [model.to_dict() for model in models],
        "can_stop": bool(models),
        "error": None,
    }


def stop_ollama_container(container: str, *, key: str, label: str) -> dict:
    status = inspect_ollama_container(container, key=key, label=label)
    if status["state"] not in {"active", "idle"}:
        return status
    errors: list[str] = []
    for model in status["models"]:
        result = _run(
            ["docker", "exec", container, "ollama", "stop", model["name"]],
            timeout=30,
        )
        if result.returncode != 0:
            errors.append(
                f"{model['name']}: {(result.stderr or result.stdout).strip()[:200]}"
            )
    refreshed = inspect_ollama_container(container, key=key, label=label)
    if errors:
        refreshed["state"] = "error"
        refreshed["error"] = "; ".join(errors)[:500]
    return refreshed


def inspect_ollama_host(executable: str, *, key: str, label: str) -> dict:
    result = _run([executable, "ps"])
    if result.returncode != 0:
        return {
            "key": key,
            "label": label,
            "kind": "ollama_host",
            "target": "127.0.0.1:11434",
            "state": "error",
            "models": [],
            "can_stop": False,
            "error": (result.stderr or result.stdout).strip()[:500],
        }
    models = _parse_ollama_models(result.stdout)
    return {
        "key": key,
        "label": label,
        "kind": "ollama_host",
        "target": "127.0.0.1:11434",
        "state": "active" if models else "idle",
        "models": [model.to_dict() for model in models],
        "can_stop": bool(models),
        "error": None,
    }


def stop_ollama_host(executable: str, *, key: str, label: str) -> dict:
    status = inspect_ollama_host(executable, key=key, label=label)
    if status["state"] not in {"active", "idle"}:
        return status
    errors: list[str] = []
    for model in status["models"]:
        result = _run([executable, "stop", model["name"]], timeout=30)
        if result.returncode != 0:
            errors.append(
                f"{model['name']}: {(result.stderr or result.stdout).strip()[:200]}"
            )
    refreshed = inspect_ollama_host(executable, key=key, label=label)
    if errors:
        refreshed["state"] = "error"
        refreshed["error"] = "; ".join(errors)[:500]
    return refreshed
