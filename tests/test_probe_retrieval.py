from __future__ import annotations

import json
from pathlib import Path

from wmloop.archive.store import ArchiveStore, BaselineRecord, ContentAddressedStore, SettledTrialRecord
from wmloop.diagnose.probe_campaign import _build_plan, _validate_probe_result
from wmloop.retrieve.index import index_probe_experience, retrieve_probe_experiences
from wmloop.retrieve.literature import LiteratureRecord, stage_literature_results
from wmloop.propose.scheduler import InterventionCell


def test_probe_plan_preserves_runtime_placeholders() -> None:
    contract = {
        "probe_id": "diagnostic-v1",
        "objective": "Measure a bounded failure signature before candidate search.",
        "hypothesis": "The model exposes a long horizon drift signature.",
        "selection_reason": "A cheap paired probe is required before retrieval.",
        "falsification_criterion": "Missing signature output blocks the campaign.",
        "command": ["/env/python", "probe.py", "{scratch_dir}/result.json"],
        "working_directory": ".",
        "allowed_gpu_indices": [0],
        "estimated_gpu_hours": 0.01,
        "total_budget_gpu_hours": 0.02,
        "timeout_seconds": 30,
        "gpu_wait_seconds": 0,
        "sample_interval_seconds": 0.2,
        "result_path": "result.json",
        "artifacts": ["result.json"],
        "metric_gates": [{"metric": "probe_ready", "role": "primary", "operator": "gte", "threshold": 1}],
        "environment": {},
        "cleanup_policy": "archive_then_delete",
    }
    report = {
        "repo_root": "/tmp/model",
        "runtime": {"selected_python": "/env/python"},
        "connector": {"asset_bindings": []},
    }
    plan = _build_plan(contract, report=report, admission={"receipt_path": "r", "receipt_sha256": "a" * 64, "onboarding_report_sha256": "b" * 64})
    assert plan["command"][-1] == "{scratch_dir}/result.json"
    assert plan["stage"] == "screen"
    assert plan["environment"]["VERDIWM_PROBE_RESULT_PATH"] == "{scratch_dir}/result.json"


def test_probe_result_requires_failure_signatures() -> None:
    result = {
        "schema_version": 1,
        "artifact_type": "verdiwm-auto-experiment-result",
        "state": "ready",
        "probe_id": "diagnostic-v1",
        "model_family": "ctrl-world",
        "runtime_capability": "predictive-video",
        "failure_signatures": ["horizon_drift"],
        "response_chart": {"axes": ["action_scale"]},
    }
    contract = {"probe_id": "diagnostic-v1", "failure_signature_fields": ["failure_signatures", "response_chart"]}
    _validate_probe_result(result, contract)


def test_retrieval_returns_only_settled_cas_bound_rows(tmp_path: Path) -> None:
    cas = ContentAddressedStore(tmp_path / "cas-root")
    archive = ArchiveStore(tmp_path / "archive.db")
    result = {
        "schema_version": 1,
        "artifact_type": "verdiwm-diagnostic-probe-result",
        "state": "ready",
        "probe_id": "probe-v1",
        "model_family": "ctrl-world",
        "runtime_capability": "predictive-video",
        "failure_signatures": ["horizon_drift", "action_binding"],
        "metrics": {"probe_score": 0.8},
    }
    result_ref = cas.put_bytes(json.dumps(result).encode(), media_type="application/json").uri
    verdict_ref = cas.put_bytes(b"{}", media_type="application/json").uri
    context_ref = cas.put_bytes(b"{}", media_type="application/json").uri
    receipt_core = {
        "schema_version": 1,
        "artifact_type": "verdiwm-auto-experiment-receipt",
        "settlement_state": "settled",
        "archive_trial_id": "trial-probe-v1",
        "stage": "screen",
        "verdict": {"verdict": "PASS"},
    }
    receipt_ref = cas.put_bytes(json.dumps(receipt_core).encode(), media_type="application/json").uri
    receipt_hash = receipt_ref.rsplit("/", 1)[-1]
    receipt = {
        **receipt_core,
        "trial_id": "probe-v1",
        "receipt_ref": receipt_ref,
        "receipt_hash": receipt_hash,
        "artifact_refs": {"result.json": result_ref},
        "failure_context_ref": context_ref,
        "verdict_ref": verdict_ref,
    }
    archive.record_settled_trial(
        SettledTrialRecord(
            trial_id="trial-probe-v1",
            proposal_id="trial-probe-v1",
            goal_id="probe-v1",
            library_version="test",
            failure_context_ref=context_ref,
            verdict_ref=verdict_ref,
            receipt_ref=receipt_ref,
            gpu_hours=0.01,
            hypothesis_hash="a" * 64,
            impl_diff_hash="b" * 64,
            evaluator_hash="c" * 64,
            settlement_state="settled",
            receipt_hash=receipt_hash,
            exploratory=True,
        )
    )
    inserted = index_probe_experience(
        database_path=tmp_path / "retrieval.db",
        result=result,
        receipt=receipt,
        archive_db=tmp_path / "archive.db",
        cas_root=tmp_path / "cas-root",
        asset_fingerprint="d" * 64,
    )
    assert inserted == 2
    rows = retrieve_probe_experiences(
        database_path=tmp_path / "retrieval.db",
        model_family="ctrl-world",
        runtime_capability="predictive-video",
        failure_signatures=["horizon_drift"],
        archive_db=tmp_path / "archive.db",
        cas_root=tmp_path / "cas-root",
    )
    assert len(rows) == 1
    assert rows[0].receipt_ref == receipt_ref
    assert not retrieve_probe_experiences(
        database_path=tmp_path / "retrieval.db",
        model_family="ctrl-world",
        runtime_capability="predictive-video",
        failure_signatures=["horizon_drift"],
        archive_db=tmp_path / "archive.db",
        cas_root=tmp_path / "cas-root",
        exclude_archive_trial_id="trial-probe-v1",
    )


def test_literature_results_are_staged_shadow_only(tmp_path: Path) -> None:
    rows = stage_literature_results(
        [LiteratureRecord("2401.12345", "A method", "A bounded method for drift.", "https://arxiv.org/pdf/2401.12345", "2024")],
        staging_root=tmp_path / "staged",
        query="world model drift",
    )
    assert rows[0]["state"] == "staged"
    payload = json.loads(Path(str(rows[0]["path"])).read_text(encoding="utf-8"))
    assert payload["proposed_manifest"]["execution_authority"] == "shadow_only"
