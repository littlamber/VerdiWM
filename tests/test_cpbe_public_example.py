from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from wmloop.diagnose.diagnostic_probe_materialization_prompt import (
    run_diagnostic_probe_materialization_prompt_batch,
)


ROOT = Path(__file__).resolve().parents[1]


def _verify_manifest(bundle: Path) -> None:
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    for relative, uri in manifest["cas_refs"].items():
        payload = (bundle / relative).read_bytes()
        assert uri == f"cas://sha256/{hashlib.sha256(payload).hexdigest()}"


def test_cpbe_algorithm_smoke_is_explicitly_synthetic_and_uses_four_sources() -> None:
    bundle = ROOT / "examples/cpbe_algorithm_smoke_v1"
    _verify_manifest(bundle)
    report = json.loads((bundle / "cpbe-plan.json").read_text(encoding="utf-8"))
    assert report["evidence_class"] == "synthetic_fixture"
    assert report["claim_boundary"].startswith("Synthetic algorithm fixture only.")
    assert {row["origin"] for row in report["ranking"]} == {"residual", "mutation", "retrieval", "llm"}
    assert report["candidate_generation"]["selected_count"] == 3
    assert all(not work_order["verdict_exposure_allowed"] for work_order in report["selected_work_orders"])


def test_cpbe_algorithm_smoke_settlement_exercises_three_terminal_paths() -> None:
    bundle = ROOT / "examples/cpbe_algorithm_smoke_settlement_v1"
    _verify_manifest(bundle)
    report = json.loads((bundle / "cpbe-settlement.json").read_text(encoding="utf-8"))
    assert report["evidence_class"] == "synthetic_fixture"
    assert report["state"] == "settled"
    assert report["admitted_count"] == 1
    assert {row["state"] for row in report["candidates"]} == {
        "settled_admitted",
        "eliminated_canary",
        "eliminated_expanded",
    }


def test_cpbe_work_order_feeds_guarded_materialization_prompt() -> None:
    source = ROOT / "examples/cpbe_algorithm_smoke_v1/manifest.json"
    with TemporaryDirectory() as temporary:
        manifest = run_diagnostic_probe_materialization_prompt_batch(
            repo_root=ROOT,
            failure_signature_bank_manifest=source,
            probe_ids=("fixture_retrieved_boundary_phase",),
            output_root=Path(temporary) / "prompts",
        )
        assert manifest["prompt_count"] == 1
        prompt = Path(manifest["records"][0]["prompt_path"]).read_text(encoding="utf-8")
        assert "fixture_retrieved_boundary_phase" in prompt
        assert "diagnostic-only" in prompt
        assert "verdict_evidence" in prompt
