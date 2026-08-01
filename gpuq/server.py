from __future__ import annotations

import json
import os
import signal
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from .config import REPO_ROOT, Config
from .db import Database
from .scheduler import Scheduler


class ApiServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, handler, *, config, database, scheduler):
        super().__init__(address, handler)
        self.config = config
        self.database = database
        self.scheduler = scheduler


class Handler(BaseHTTPRequestHandler):
    server: ApiServer

    def log_message(self, format, *args):
        return

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            return self._serve_file(REPO_ROOT / "static" / "index.html", "text/html; charset=utf-8")
        if path == "/api/health":
            return self._json({"ok": True, "scheduler": self.server.scheduler.snapshot()})
        if path == "/api/status":
            jobs = self.server.database.list_jobs()
            scores = self.server.scheduler.scores_for(jobs["queued"])
            for job in jobs["queued"]:
                job["effective_score"] = scores.get(job["id"])
            return self._json(
                {
                    "runtime": self.server.scheduler.snapshot(),
                    "jobs": jobs,
                }
            )
        if path.startswith("/api/jobs/"):
            job_id = path.removeprefix("/api/jobs/")
            job = self.server.database.get_job(job_id)
            return self._json(job or {"error": "not-found"}, 200 if job else 404)
        if path.startswith("/api/estimate/"):
            key = path.removeprefix("/api/estimate/")
            estimate = self.server.database.workload_estimate(key)
            return self._json(estimate or {"samples": 0})
        self._json({"error": "not-found"}, 404)

    def do_OPTIONS(self):
        origin = self.headers.get("Origin", "")
        if origin not in {"http://localhost:3010", "http://127.0.0.1:3010"}:
            return self._json({"error": "origin-not-allowed"}, 403)
        self.send_response(HTTPStatus.NO_CONTENT)
        self._cors_headers(origin)
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Max-Age", "600")
        self.end_headers()

    def do_POST(self):
        if not self._authorized():
            return self._json({"error": "unauthorized"}, 401)
        path = urlparse(self.path).path
        if path == "/api/jobs":
            return self._submit()
        if path == "/api/jobs/reorder":
            return self._reorder()
        if path == "/api/shutdown":
            if self.server.scheduler.snapshot()["managed_running"]:
                return self._json({"error": "managed-jobs-running"}, 409)
            threading.Thread(target=self.server.shutdown, daemon=True).start()
            return self._json({"ok": True})
        if path.startswith("/api/jobs/") and path.endswith("/cancel"):
            job_id = path.removeprefix("/api/jobs/").removesuffix("/cancel").rstrip("/")
            changed = self.server.database.request_cancel(job_id)
            return self._json({"ok": changed}, 200 if changed else 409)
        if path.startswith("/api/external-workloads/") and path.endswith("/stop"):
            key = (
                path.removeprefix("/api/external-workloads/")
                .removesuffix("/stop")
                .rstrip("/")
            )
            try:
                result = self.server.scheduler.stop_external_workload(key)
            except ValueError as exc:
                return self._json({"error": str(exc)}, 400)
            if result is None:
                return self._json({"error": "not-found"}, 404)
            return self._json(
                {"ok": result.get("state") != "error", "workload": result},
                200 if result.get("state") != "error" else 409,
            )
        self._json({"error": "not-found"}, 404)

    def _reorder(self):
        try:
            body = self._read_json()
            job_ids = body["job_ids"]
            if (
                not isinstance(job_ids, list)
                or not job_ids
                or len(job_ids) > 1000
                or not all(isinstance(job_id, str) for job_id in job_ids)
            ):
                raise ValueError("job_ids must be a non-empty string array of at most 1000 items")
            ordered = self.server.database.reorder_queued_jobs(job_ids)
            return self._json({"ok": True, "queued": ordered})
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            status = 409 if "queue changed" in str(exc) else 400
            self._json({"error": str(exc)}, status)

    def _submit(self):
        try:
            body = self._read_json()
            argv = body["argv"]
            if (
                not isinstance(argv, list)
                or not argv
                or not all(isinstance(x, str) and x for x in argv)
            ):
                raise ValueError("argv must be a non-empty string array")
            requested = int(body["requested_vram_mb"])
            estimated = int(body["estimated_seconds"])
            priority = int(body.get("priority", 50))
            max_runtime = int(body.get("max_runtime_seconds", max(estimated * 3, 600)))
            agent = str(body.get("agent_name", "anonymous")).strip()[:80]
            workload = str(body.get("workload_key", "generic")).strip()[:120]
            cwd = str(body["cwd"])
            if not (0 <= priority <= 100):
                raise ValueError("priority must be between 0 and 100")
            telemetry = self.server.scheduler.telemetry
            if telemetry and requested > telemetry.total_mb - self.server.config.safety_vram_mb:
                raise ValueError("requested VRAM exceeds schedulable GPU capacity")
            if requested <= 0 or estimated <= 0 or max_runtime <= 0:
                raise ValueError("VRAM and time values must be positive")
            job = self.server.database.submit_job(
                agent_name=agent or "anonymous",
                workload_key=workload or "generic",
                argv=argv,
                cwd=cwd,
                requested_vram_mb=requested,
                estimated_seconds=estimated,
                priority=priority,
                max_runtime_seconds=max_runtime,
            )
            self._json(job, 201)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self._json({"error": str(exc)}, 400)

    def _authorized(self) -> bool:
        return self.headers.get("X-GPUQ-Token", "") == self.server.config.api_token

    def _read_json(self):
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > 1_000_000:
            raise ValueError("invalid content length")
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _json(self, value, status=200):
        payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self._cors_headers(self.headers.get("Origin", ""))
        self.end_headers()
        self.wfile.write(payload)

    def _serve_file(self, path: Path, content_type: str):
        payload = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-cache")
        self._cors_headers(self.headers.get("Origin", ""))
        self.end_headers()
        self.wfile.write(payload)

    def _cors_headers(self, origin: str):
        if origin in {"http://localhost:3010", "http://127.0.0.1:3010"}:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")


def main() -> None:
    config = Config.load()
    database = Database(config.database_url)
    # The Windows launcher verifies the host port before creating this process.
    # Keep this bounded so a pathological connection failure returns to the
    # supervisor instead of leaving a scheduled task apparently "running" for
    # fifteen minutes with no API listener.
    database.wait_and_migrate(
        sorted((REPO_ROOT / "sql").glob("*.sql")),
        timeout_seconds=60,
        retry_seconds=5,
    )
    scheduler = Scheduler(config, database)
    server = ApiServer(
        (config.api_host, config.api_port),
        Handler,
        config=config,
        database=database,
        scheduler=scheduler,
    )
    database.orphan_interrupted_jobs()
    scheduler.start()
    runtime_root = REPO_ROOT / ".runtime"
    runtime_root.mkdir(parents=True, exist_ok=True)
    pid_path = runtime_root / "server.pid"
    pid_path.write_text(str(os.getpid()), encoding="ascii")
    stopping = threading.Event()

    def stop_handler(*_):
        if stopping.is_set():
            return
        stopping.set()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        scheduler.stop()
        server.server_close()
        try:
            if pid_path.read_text(encoding="ascii").strip() == str(os.getpid()):
                pid_path.unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
