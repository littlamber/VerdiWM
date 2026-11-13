"""Recover diagnostic routing only from archived evidence."""

import hashlib
import json
import sqlite3

import pytest

from wmloop.archive.store import ArchiveStore, ContentAddressedStore, SettledTrialRecord
from wmloop.diagnose import probe_campaign
from wmloop.retrieve.index import retrieve_probe_experiences
from wmloop.storage import canonical_bytes


@pytest.fixture
def recovered(tmp_path, monkeypatch):
    sidecar = tmp_path / "sidecar"
    sidecar.mkdir()
    conformance = tmp_path / "conformance"
    conformance.mkdir()
    report = b'{}'
    (sidecar / "onboarding-report.json").write_bytes(report)
    admission = {"onboarding_report_sha256": hashlib.sha256(report).hexdigest()}
    contract = {"probe_id": "probe", "model_family": "source", "runtime_capability": "video", "failure_signature_fields": []}
    monkeypatch.setattr(probe_campaign, "_load_contract", lambda path: contract)
    monkeypatch.setattr(probe_campaign, "admission_from_manifest", lambda path: admission)
    monkeypatch.setattr(probe_campaign, "_build_plan", lambda *args, campaign_id=None, **kwargs: {"trial_id": campaign_id or "trial"})
    out = tmp_path / "out"
    receipt_path = out / "run" / "receipts" / "trial.json"
    receipt_path.parent.mkdir(parents=True)
    input_hash = probe_campaign._sha256(probe_campaign._canonical_json({"contract": contract, "admission": admission, "workspace": str(tmp_path)}))
    (out / "manifest.json").write_text(json.dumps({"input_hash": input_hash, "failure_signatures": ["forged"]}))
    cas = ContentAddressedStore(tmp_path / "cas")
    result = {"schema_version": 1, "artifact_type": "verdiwm-diagnostic-probe-result", "state": "ready", "probe_id": "probe", "model_family": "source", "runtime_capability": "video", "failure_signatures": ["history"], "primitive": "anchor", "metrics": {"primary": 0.2}, "claim_boundary": "Diagnostic routing only; no formal effect claim."}
    result_ref = cas.put_bytes(canonical_bytes(result), media_type="application/json")
    receipt = {"trial_id": "trial", "archive_trial_id": "trial", "settlement_state": "settled", "stage": "screen", "verdict": {"verdict": "PASS"}, "artifact_refs": {"result.json": result_ref.uri}}
    ref = cas.put_bytes(canonical_bytes(receipt), media_type="application/json")
    receipt.update(receipt_ref=ref.uri, receipt_hash=ref.sha256, cleanup={"state": "completed"})
    receipt_path.write_text(json.dumps(receipt))
    archive_db = tmp_path / "archive.db"
    ArchiveStore(archive_db).record_settled_trial(SettledTrialRecord(trial_id="trial", proposal_id="proposal", goal_id="goal", library_version="v1", failure_context_ref=result_ref.uri, verdict_ref=result_ref.uri, receipt_ref=ref.uri, gpu_hours=0.0, hypothesis_hash="a" * 64, impl_diff_hash="b" * 64, evaluator_hash="c" * 64, settlement_state="settled", receipt_hash=ref.sha256))
    calls = []
    def recover(**kwargs):
        calls.append(kwargs)
        return {"receipt_path": str(receipt_path), "verdict": "PASS"}
    monkeypatch.setattr(probe_campaign, "run_auto_experiment", recover)
    kwargs = dict(contract_path=tmp_path / "contract.json", sidecar_root=sidecar, conformance_root=conformance, output_root=out, workspace_root=tmp_path, archive_db=archive_db, cas_root=tmp_path / "cas", lock_root=tmp_path / "locks", retrieval_db=tmp_path / "new-index.db")
    return kwargs, calls, receipt_path, ref.path


def test_recovery_reconstructs_routing_and_indexes_new_database(recovered):
    kwargs, calls, _, _ = recovered
    manifest = probe_campaign.run_diagnostic_probe(**kwargs)
    assert manifest["failure_signatures"] == ["history"]
    assert len(calls) == 1
    matches = retrieve_probe_experiences(database_path=kwargs["retrieval_db"], archive_db=kwargs["archive_db"], cas_root=kwargs["cas_root"], model_family="source", runtime_capability="video", failure_signatures=("history",))
    assert len(matches) == 1


@pytest.mark.parametrize("damage", ["receipt_missing", "receipt_changed", "cas_missing", "archive_binding_missing", "campaign_changed"])
def test_recovery_refuses_missing_or_changed_evidence_without_execution(recovered, damage):
    kwargs, calls, receipt_path, cas_path = recovered
    if damage == "receipt_missing":
        receipt_path.unlink()
    elif damage == "receipt_changed":
        receipt = json.loads(receipt_path.read_text())
        receipt["stage"] = "confirm"
        receipt_path.write_text(json.dumps(receipt))
    elif damage == "cas_missing":
        cas_path.unlink()
    elif damage == "archive_binding_missing":
        with sqlite3.connect(kwargs["archive_db"]) as connection:
            connection.execute("DELETE FROM receipts")
    else:
        kwargs["campaign_id"] = "another-campaign"
    with pytest.raises(probe_campaign.DiagnosticProbeError, match="DIAGNOSTIC_PROBE_RECOVERY_"):
        probe_campaign.run_diagnostic_probe(**kwargs)
    assert not calls
    assert not kwargs["retrieval_db"].exists()


@pytest.mark.parametrize("linked_path", ["output", "output_parent", "receipt_parent", "plan"])
def test_recovery_rejects_symlink_paths_before_execution(recovered, tmp_path, linked_path):
    kwargs, calls, receipt_path, _ = recovered
    if linked_path == "output":
        original = kwargs["output_root"]
        real = tmp_path / "real-output"
        original.rename(real)
        original.symlink_to(real, target_is_directory=True)
    elif linked_path == "output_parent":
        link = tmp_path / "linked-parent"
        link.symlink_to(tmp_path, target_is_directory=True)
        kwargs["output_root"] = link / "out"
    elif linked_path == "receipt_parent":
        original = receipt_path.parent
        real = original.with_name("real-receipts")
        original.rename(real)
        original.symlink_to(real, target_is_directory=True)
    else:
        real = tmp_path / "external-plan.json"
        real.write_text(json.dumps({"trial_id": "trial"}))
        (kwargs["output_root"] / "probe-plan.json").symlink_to(real)
    with pytest.raises(probe_campaign.DiagnosticProbeError):
        probe_campaign.run_diagnostic_probe(**kwargs)
    assert not calls
    assert not kwargs["retrieval_db"].exists()
