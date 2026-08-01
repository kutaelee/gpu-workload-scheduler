import json
import threading
from http.client import HTTPConnection
from types import SimpleNamespace

from gpuq.server import ApiServer, Handler


class FakeScheduler:
    def __init__(self):
        self.stopped_external: list[str] = []

    def snapshot(self):
        return {"managed_running": 0}

    def scores_for(self, _queued):
        return {}

    def stop_external_workload(self, key):
        self.stopped_external.append(key)
        if key != "portal-ollama":
            return None
        return {"key": key, "state": "idle", "models": []}


class FakeDatabase:
    def __init__(self):
        self.cancelled: list[str] = []
        self.reordered: list[str] = []

    def request_cancel(self, job_id: str) -> bool:
        self.cancelled.append(job_id)
        return True

    def reorder_queued_jobs(self, job_ids: list[str]) -> list[dict]:
        self.reordered = job_ids
        return [
            {"id": job_id, "manual_rank": rank}
            for rank, job_id in enumerate(job_ids, start=1)
        ]


def post(server: ApiServer, path: str, payload: dict, token: str | None):
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json", "Content-Length": str(len(body))}
    if token is not None:
        headers["X-GPUQ-Token"] = token
    connection = HTTPConnection(*server.server_address, timeout=2)
    try:
        connection.request("POST", path, body=body, headers=headers)
        response = connection.getresponse()
        return response.status, json.loads(response.read())
    finally:
        connection.close()


def test_authenticated_control_endpoints_are_bounded_to_cancel_and_complete_reorder():
    database = FakeDatabase()
    scheduler = FakeScheduler()
    server = ApiServer(
        ("127.0.0.1", 0),
        Handler,
        config=SimpleNamespace(api_token="test-token"),
        database=database,
        scheduler=scheduler,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        assert post(server, "/api/jobs/one/cancel", {}, None)[0] == 401

        status, cancel = post(server, "/api/jobs/one/cancel", {}, "test-token")
        assert status == 200
        assert cancel == {"ok": True}
        assert database.cancelled == ["one"]

        status, reordered = post(
            server,
            "/api/jobs/reorder",
            {"job_ids": ["first", "second"]},
            "test-token",
        )
        assert status == 200
        assert reordered["ok"] is True
        assert [item["manual_rank"] for item in reordered["queued"]] == [1, 2]
        assert database.reordered == ["first", "second"]

        assert post(
            server,
            "/api/external-workloads/portal-ollama/stop",
            {},
            None,
        )[0] == 401
        status, stopped = post(
            server,
            "/api/external-workloads/portal-ollama/stop",
            {},
            "test-token",
        )
        assert status == 200
        assert stopped["ok"] is True
        assert scheduler.stopped_external == ["portal-ollama"]
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()
