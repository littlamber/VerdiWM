import json
import sys
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from wmloop.control.campaign_api import CampaignStore, _Handler


def _execution(name: str):
    return {
        "kind": "evolution",
        "repo_root": "/share/project/model",
        "output_root": f"/share/project/runs/{name}",
        "state_root": f"/share/project/state/{name}",
        "evaluator_contract": "/share/project/evaluator.json",
        "total_budget_gpu_hours": 0.1,
    }


def _payload(campaign_id: str, goal: str, *, execution=None):
    return {
        "campaign_id": campaign_id,
        "goal": goal,
        "model": "/share/project/model",
        "dataset": "/share/project/data",
        "budget": "0.1gpu-hour",
        "execution": execution or _execution(campaign_id),
    }


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
        payload = _payload("demo-1", "improve action following")
        status, created = _request(server, "POST", "/v1/campaigns", payload)
        assert status == 201 and created["status"] == "created"
        assert _request(server, "POST", "/v1/campaigns", payload)[1] == created
        assert _request(server, "POST", "/v1/campaigns/demo-1/confirm", {})[1]["status"] == "queued"
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
        assert _request(server, "POST", "/v1/campaigns", {"goal": "x"}) == (
            400,
            {"error": "MODEL_REQUIRED"},
        )
        assert _request(
            server,
            "POST",
            "/v1/campaigns",
            {"goal": "x", "model": "m", "dataset": "d"},
        ) == (400, {"error": "BUDGET_INVALID"})
        assert _request(server, "GET", "/v1/campaigns/missing")[0] == 404
    finally:
        server.shutdown()
        thread.join(timeout=2)


def test_campaign_list_filters_status(tmp_path: Path):
    server, thread = _server(tmp_path)
    try:
        _request(server, "POST", "/v1/campaigns", _payload("a", "one"))
        _request(server, "POST", "/v1/campaigns", _payload("b", "two"))
        _request(server, "POST", "/v1/campaigns/a/confirm", {})
        status, payload = _request(server, "GET", "/v1/campaigns?status=queued")
        assert status == 200
        assert [item["campaign_id"] for item in payload["items"]] == ["a"]
    finally:
        server.shutdown()
        thread.join(timeout=2)


def test_confirm_with_execution_contract_creates_durable_dispatch(tmp_path: Path):
    server, thread = _server(tmp_path)
    try:
        payload = {
            "campaign_id": "queued-1",
            "goal": "run bounded verification",
            "model": "/share/project/model",
            "dataset": "/share/project/data",
            "budget": 0.1,
            "execution": _execution("queued-1"),
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


def test_campaign_compiles_four_field_request_into_pipeline(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    model = tmp_path / "model"
    data = tmp_path / "data"
    (model / "models").mkdir(parents=True)
    (model / "scripts").mkdir()
    (model / "models" / "ctrl_world.py").write_text("# marker\n")
    (model / "scripts" / "rollout_replay_traj.py").write_text("# marker\n")
    (model / "asset.bin").write_bytes(b"asset")
    data.mkdir()
    profile = tmp_path / "profile.json"
    profile.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "verdiwm-adapter-profile",
                "profile_id": "test-profile",
                "aliases": ["test"],
                "model_family": "ctrl_world",
                "capability_level": "L2",
                "execution_kind": "pipeline",
                "repo_markers": [
                    "models/ctrl_world.py",
                    "scripts/rollout_replay_traj.py",
                ],
                "goal_keywords": ["predict"],
                "evaluator_contract": "configs/onboarding/ctrl_world_predictive_probe_evaluator_v2.json",
                "probe_contract": None,
                "constitution_freeze": "configs/constitution/ctrl_world_predictive_quality_pilot_v2.freeze.json",
                "runtime_candidates": [sys.executable],
                "asset_bindings": [
                    {"parameter": "--asset", "candidates": ["{model}/asset.bin"]}
                ],
                "probe_imports": False,
            }
        )
    )
    store = CampaignStore(tmp_path / "state" / "campaigns")

    created = store.create(
        {
            "campaign_id": "compiled-1",
            "model": str(model),
            "dataset": str(data),
            "goal": "predict longer",
            "budget": "2gpu-hours",
            "adapter_profile_path": str(profile),
        }
    )

    assert created["adapter_profile"] == "test-profile"
    assert created["execution"]["kind"] == "pipeline"
    assert created["execution"]["budget_total_gpu_hours"] == 2.0
    queued = store.confirm("compiled-1")
    assert queued["status"] == "queued"
    assert Path(queued["dispatch_ref"]).is_file()


def test_reproduction_gets_isolated_execution_and_durable_dispatch(tmp_path: Path):
    store = CampaignStore(tmp_path / "campaigns")
    source = store.create(_payload("source-1", "verify"))
    store.confirm(source["campaign_id"])

    reproduced = store.reproduce(source["campaign_id"])

    assert reproduced["status"] == "queued"
    assert reproduced["parent_campaign_id"] == "source-1"
    assert reproduced["execution"]["output_root"] != source["execution"]["output_root"]
    assert reproduced["campaign_id"] in reproduced["execution"]["output_root"]
    assert Path(reproduced["dispatch_ref"]).is_file()
