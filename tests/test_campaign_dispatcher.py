import json
from pathlib import Path

from wmloop.control.campaign_api import CampaignStore
from wmloop.control.campaign_dispatcher import DispatcherOptions, run_dispatcher


def test_dispatcher_consumes_pending_manifest_and_settles_campaign(tmp_path: Path):
    api_root = tmp_path / "campaigns"
    store = CampaignStore(api_root)
    record = store.create(
        {
            "campaign_id": "dispatch-1",
            "goal": "verify",
            "execution": {
                "kind": "evolution",
                "repo_root": "/share/project/model",
                "output_root": "/share/project/runs/dispatch-1",
                "state_root": "/share/project/state/dispatch-1",
                "evaluator_contract": "/share/project/evaluator.json",
                "total_budget_gpu_hours": 0.1,
            },
        }
    )
    assert store.confirm(record["campaign_id"])["status"] == "queued"

    def runner(execution):
        assert execution["kind"] == "evolution"
        return {"verdict": "PASS", "state": "completed"}

    result = run_dispatcher(
        DispatcherOptions(state_root=api_root, max_cycles=1),
        runner=runner,
    )
    assert result["settled_campaign_ids"] == ["dispatch-1"]
    assert store.get("dispatch-1")["status"] == "completed"
    assert (api_root / "dispatch" / "completed" / "dispatch-1.json").is_file()
