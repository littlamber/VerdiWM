from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from wmloop.experiments.cpbe_stage_runner import publish_static_offline_receipts


def test_publishes_hash_bound_static_and_offline_receipts() -> None:
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        probe_id = "fixture_cpbe_probe"
        module_path = root / "wmloop/diagnose/probes" / f"{probe_id}.py"
        test_path = root / "tests" / f"test_{probe_id}.py"
        descriptor_path = root / "configs/probes/staging" / f"{probe_id}.json"
        work_order_path = root / "plan/work-orders" / f"{probe_id}.json"
        for parent in (module_path.parent, test_path.parent, descriptor_path.parent, work_order_path.parent):
            parent.mkdir(parents=True, exist_ok=True)

        program = {
            "probe_id": probe_id,
            "signal_source": "fixture",
            "hook_type": "H2",
            "spatial_mask": "global",
            "temporal_basis": "latest",
            "contrast_operator": "signed",
            "dose_schedule": [-0.1, 0.0, 0.1],
            "aggregation": "mean",
            "invariants": ["same_seed"],
            "required_capabilities": [],
            "estimated_gpu_hours": 0.01,
            "origin": "residual",
            "parent_probe_ids": ["parent"],
            "rationale": "fixture",
            "diagnostic_only": True,
            "reversible": True,
        }
        order = {
            "artifact_type": "verdiwm-cpbe-probe-work-order",
            "probe_id": probe_id,
            "role": "diagnostic",
            "verdict_exposure_allowed": False,
            "program": program,
        }
        plan = {
            "artifact_type": "verdiwm-cpbe-plan",
            "state": "ready",
            "experiment_id": "fixture",
            "selected_work_orders": [order],
        }
        (root / "plan/cpbe-plan.json").write_text(json.dumps(plan), encoding="utf-8")
        work_order_path.write_text(json.dumps(order), encoding="utf-8")
        descriptor_path.write_text(
            json.dumps(
                {
                    "probe_id": probe_id,
                    "role": "diagnostic",
                    "verdict_exposure_allowed": False,
                    "program": program,
                    "module": f"wmloop.diagnose.probes.{probe_id}",
                    "callable": "measure",
                    "admission_state": {"static": "implemented", "offline": "fixture_test_available"},
                }
            ),
            encoding="utf-8",
        )
        module_path.write_text("def measure():\n    return {'verdict_exposure_allowed': False}\n", encoding="utf-8")
        test_path.write_text("def test_fixture():\n    assert True\n", encoding="utf-8")

        manifest = publish_static_offline_receipts(
            plan_path=root / "plan/cpbe-plan.json",
            repo_root=root,
            output_root=root / "receipts",
        )
        assert manifest["state"] == "ready"
        receipts = [
            json.loads(line)
            for line in (root / "receipts/cpbe-stage-receipts.jsonl").read_text().splitlines()
        ]
        assert [receipt["stage"] for receipt in receipts] == ["static", "offline"]
        assert all(receipt["passed"] for receipt in receipts)
        for receipt in receipts:
            for artifact in receipt["evidence_artifacts"]:
                path = root / "receipts" / artifact["path"]
                assert path.stat().st_size == artifact["size_bytes"]
