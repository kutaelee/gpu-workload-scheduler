from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

from .config import Config


def request(method: str, path: str, body: dict | None = None) -> dict:
    config = Config.load()
    payload = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {"Accept": "application/json"}
    if payload is not None:
        headers["Content-Type"] = "application/json"
        headers["X-GPUQ-Token"] = config.api_token
    req = urllib.request.Request(
        f"http://{config.api_host}:{config.api_port}{path}",
        data=payload,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"GPU queue returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"GPU queue is unavailable: {exc.reason}") from exc


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="gpuq", description="RTX workload reservation queue")
    commands = root.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="queue a command for GPU execution")
    run.add_argument("--vram", type=int, required=True, metavar="MIB")
    run.add_argument("--eta", type=int, required=True, metavar="SECONDS")
    run.add_argument("--priority", type=int, default=50, choices=range(0, 101), metavar="0..100")
    run.add_argument("--max-runtime", type=int, default=None, metavar="SECONDS")
    run.add_argument("--agent", default=os.environ.get("CODEX_THREAD_ID", "interactive"))
    run.add_argument("--workload", default="generic")
    run.add_argument("--cwd", default=os.environ.get("GPUQ_CALLER_CWD", os.getcwd()))
    run.add_argument("argv", nargs=argparse.REMAINDER)

    commands.add_parser("status", help="show scheduler and queue state")
    cancel = commands.add_parser("cancel", help="cancel a queued or running job")
    cancel.add_argument("job_id")
    wait = commands.add_parser("wait", help="wait for a job and return its exit code")
    wait.add_argument("job_id")
    wait.add_argument("--poll", type=float, default=2.0)
    estimate = commands.add_parser("estimate", help="show learned duration estimate")
    estimate.add_argument("workload_key")
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "run":
        argv = args.argv
        if argv and argv[0] == "--":
            argv = argv[1:]
        if not argv:
            raise SystemExit("Command is required after --")
        cwd = str(Path(args.cwd).resolve())
        body = {
            "argv": argv,
            "cwd": cwd,
            "requested_vram_mb": args.vram,
            "estimated_seconds": args.eta,
            "priority": args.priority,
            "max_runtime_seconds": args.max_runtime or max(args.eta * 3, 600),
            "agent_name": args.agent,
            "workload_key": args.workload,
        }
        result = request("POST", "/api/jobs", body)
        print(result["id"])
        return
    if args.command == "status":
        print(json.dumps(request("GET", "/api/status"), ensure_ascii=False, indent=2))
        return
    if args.command == "cancel":
        print(json.dumps(request("POST", f"/api/jobs/{args.job_id}/cancel", {}), indent=2))
        return
    if args.command == "estimate":
        print(json.dumps(request("GET", f"/api/estimate/{args.workload_key}"), indent=2))
        return
    if args.command == "wait":
        import time

        while True:
            job = request("GET", f"/api/jobs/{args.job_id}")
            status = job["status"]
            if status not in {"queued", "running"}:
                print(json.dumps(job, ensure_ascii=False, indent=2))
                if status == "succeeded":
                    raise SystemExit(0)
                raise SystemExit(job.get("exit_code") or 1)
            time.sleep(max(0.2, args.poll))


if __name__ == "__main__":
    main()
