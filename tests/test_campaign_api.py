import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from wmloop.control.campaign_api import CampaignStore, _Handler


def _server(tmp_path: Path):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.store = CampaignStore(tmp_path / "state")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _request(server, method, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(f"http://127.0.0.1:{server.server_port}{path}", data=data, method=method)
    if data is not None:
        request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_campaign_lifecycle_and_idempotent_create(tmp_path: Path):
    server, thread = _server(tmp_path)
    try:
        payload = {"campaign_id": "demo-1", "goal": "improve action following", "model": "synthetic"}
        status, created = _request(server, "POST", "/v1/campaigns", payload)
        assert status == 201 and created["status"] == "created"
        assert _request(server, "POST", "/v1/campaigns", payload)[1] == created
        assert _request(server, "POST", "/v1/campaigns/demo-1/confirm", {})[1]["status"] == "confirmed"
        assert _request(server, "POST", "/v1/campaigns/demo-1/reproduce", {})[1]["status"] == "queued"
        assert _request(server, "POST", "/v1/campaigns/demo-1/cancel", {})[1]["status"] == "cancelled"
        assert _request(server, "GET", "/v1/campaigns/demo-1")[1]["status"] == "cancelled"
    finally:
        server.shutdown()
        thread.join(timeout=2)


def test_campaign_rejects_invalid_requests(tmp_path: Path):
    server, thread = _server(tmp_path)
    try:
        assert _request(server, "POST", "/v1/campaigns", {}) == (400, {"error": "GOAL_REQUIRED"})
        assert _request(server, "GET", "/v1/campaigns/missing")[0] == 404
    finally:
        server.shutdown()
        thread.join(timeout=2)


def test_confirm_with_execution_contract_creates_durable_dispatch(tmp_path: Path):
    server, thread = _server(tmp_path)
    try:
        payload = {
            "campaign_id": "queued-1",
            "goal": "run bounded verification",
            "execution": {
                "kind": "evolution",
                "repo_root": "/share/project/model",
                "output_root": "/share/project/runs/queued-1",
                "state_root": "/share/project/state/queued-1",
                "evaluator_contract": "/share/project/evaluator.json",
                "total_budget_gpu_hours": 0.1,
            },
        }
        assert _request(server, "POST", "/v1/campaigns", payload)[0] == 201
        status, queued = _request(server, "POST", "/v1/campaigns/queued-1/confirm", {})
        assert status == 202
        assert queued["status"] == "queued"
        dispatch = tmp_path / "state" / "dispatch" / "pending" / "queued-1.json"
        assert dispatch.is_file()
        assert json.loads(dispatch.read_text())["state"] == "pending"
    finally:
        server.shutdown()
        thread.join(timeout=2)
