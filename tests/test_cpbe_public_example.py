from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from wmloop.diagnose.diagnostic_probe_materialization_prompt import (
    run_diagnostic_probe_materialization_prompt_batch,
)


ROOT = Path(__file__).resolve().parents[1]


def test_cpbe_work_order_feeds_guarded_materialization_prompt() -> None:
    """Prompt materialization needs a contract fixture, not a past campaign bundle."""

    with TemporaryDirectory() as temporary:
        fixture_root = Path(temporary) / "fixture"
        work_orders = fixture_root / "work-orders"
        work_orders.mkdir(parents=True)
        probe_id = "fixture_retrieved_boundary_phase"
        (work_orders / "fixture.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "artifact_type": "verdiwm-cpbe-probe-work-order",
                    "role": "diagnostic",
                    "verdict_exposure_allowed": False,
                    "probe_id": probe_id,
                    "environment": "fixture_target",
                    "signature": "long_horizon_motion_aliasing",
                    "priority": "P0",
                    "signal_contract": "diagnostic-only fixture",
                    "allowed_mutation_paths": ["wmloop/diagnose/probes/fixture.py"],
                    "forbidden_surfaces": ["verdict_evidence"],
                    "admission_gates": ["offline_fixture_test_passed"],
                }
            ),
            encoding="utf-8",
        )
        manifest_path = fixture_root / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "artifact_type": "verdiwm-cpbe-plan-manifest",
                    "state": "ready",
                    "probe_work_order_paths": {probe_id: "work-orders/fixture.json"},
                }
            ),
            encoding="utf-8",
        )

        manifest = run_diagnostic_probe_materialization_prompt_batch(
            repo_root=ROOT,
            failure_signature_bank_manifest=manifest_path,
            probe_ids=(probe_id,),
            output_root=Path(temporary) / "prompts",
        )

        assert manifest["prompt_count"] == 1
        prompt = Path(manifest["records"][0]["prompt_path"]).read_text(encoding="utf-8")
        assert probe_id in prompt
        assert "diagnostic-only fixture" in prompt
        assert "verdict_evidence" in prompt
