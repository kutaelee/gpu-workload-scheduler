from __future__ import annotations

import ctypes
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def windows_boot_epoch() -> int:
    """Return a stable minute-granularity identifier for the current boot."""
    if os.name != "nt":
        return 0
    uptime_seconds = ctypes.windll.kernel32.GetTickCount64() / 1000.0
    boot_time = time.time() - uptime_seconds
    return int(round(boot_time / 60.0) * 60)


def read_fault_state(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, dict) else None


def write_fault_state(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def fatal_gpu_events_since_boot() -> int:
    """Count current-boot nvlddmkm 14/153 events; fail closed on errors."""
    if os.name != "nt":
        return 0
    command = (
        "$start=(Get-Date).AddMilliseconds(-[Environment]::TickCount64);"
        "$events=@(Get-WinEvent -FilterHashtable "
        "@{LogName='System';ProviderName='nvlddmkm';Id=14,153;StartTime=$start} "
        "-ErrorAction SilentlyContinue);$events.Count"
    )
    completed = subprocess.run(
        [
            os.path.join(
                os.environ.get("SystemRoot", r"C:\Windows"),
                "System32",
                "WindowsPowerShell",
                "v1.0",
                "powershell.exe",
            ),
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            command,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        creationflags=(
            subprocess.CREATE_NO_WINDOW
            if hasattr(subprocess, "CREATE_NO_WINDOW")
            else 0
        ),
    )
    if completed.returncode != 0:
        raise RuntimeError("Windows GPU event query failed")
    try:
        return int(completed.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError) as exc:
        raise RuntimeError("Windows GPU event query returned an invalid count") from exc


def runtime_provenance(repo_root: Path, config_path: Path) -> dict:
    def git(*args: str) -> str | None:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
            creationflags=(
                subprocess.CREATE_NO_WINDOW
                if hasattr(subprocess, "CREATE_NO_WINDOW")
                else 0
            ),
        )
        value = completed.stdout.strip()
        return value if completed.returncode == 0 else None

    try:
        config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    except OSError:
        config_sha256 = None
    status = git("status", "--porcelain")
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_branch": git("branch", "--show-current"),
        "dirty_worktree": bool(status) if status is not None else None,
        "config_sha256": config_sha256,
        "service_started_at": datetime.now(timezone.utc).isoformat(),
        "windows_boot_epoch": windows_boot_epoch(),
    }
