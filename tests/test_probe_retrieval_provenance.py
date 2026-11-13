"""CPU-only integration of retrieval with real temporary Archive and CAS stores."""

from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
import sqlite3

import pytest

from wmloop.archive.store import ArchiveInvariantError, ArchiveStore, ContentAddressedStore, SettledTrialRecord
from wmloop.retrieve.index import (
    ProbeRetrievalError,
    index_probe_experience,
    recheck_probe_experiences,
    retrieve_probe_experiences,
)
from wmloop.storage import canonical_bytes, canonical_json_bytes


@pytest.fixture
def evidence(tmp_path: Path) -> dict[str, object]:
    cas = ContentAddressedStore(tmp_path / "cas")
    archive_db = tmp_path / "archive.db"
    archive = ArchiveStore(archive_db)
    result = {
        "schema_version": 1, "artifact_type": "verdiwm-diagnostic-probe-result",
        "state": "ready", "claim_boundary": "Diagnostic routing only; no formal effect claim.",
        "probe_id": "probe", "model_family": "source", "runtime_capability": "video",
        "failure_signatures": ["history"], "primitive": "anchor", "metrics": {"primary": 0.2},
    }
    result_ref = cas.put_bytes(canonical_bytes(result), media_type="application/json")
    receipt = {
        "archive_trial_id": "trial", "settlement_state": "settled", "stage": "confirm",
        "verdict": {"verdict": "PASS"}, "artifact_refs": {"result.json": result_ref.uri},
    }
    receipt_ref = cas.put_bytes(canonical_bytes(receipt), media_type="application/json")
    receipt.update(receipt_ref=receipt_ref.uri, receipt_hash=receipt_ref.sha256)
    archive.record_settled_trial(SettledTrialRecord(
        trial_id="trial", proposal_id="proposal", goal_id="goal", library_version="v1",
        failure_context_ref=result_ref.uri, verdict_ref=result_ref.uri, receipt_ref=receipt_ref.uri,
        gpu_hours=0.0, hypothesis_hash="a" * 64, impl_diff_hash="b" * 64,
        evaluator_hash="c" * 64, settlement_state="settled", receipt_hash=receipt_ref.sha256,
    ))
    return dict(database_path=tmp_path / "index.db", result=result, receipt=receipt,
                archive_db=archive_db, cas_root=tmp_path / "cas")


def _retrieve(evidence: dict[str, object]) -> tuple:
    return retrieve_probe_experiences(
        **{key: evidence[key] for key in ("database_path", "archive_db", "cas_root")},
        model_family="source", runtime_capability="video", failure_signatures=("history",),
    )


def test_receipted_index_is_idempotent_and_retrievable(evidence: dict[str, object]) -> None:
    assert index_probe_experience(**evidence) == 1
    assert index_probe_experience(**evidence) == 0
    assert _retrieve(evidence)[0].metric_outcome == 0.2
    # The historical minimal receipt contract still supports general reuse.
    assert recheck_probe_experiences(
        [row.to_dict() for row in _retrieve(evidence)],
        archive_db=evidence["archive_db"], cas_root=evidence["cas_root"],
    ) == _retrieve(evidence)


def _settle_receipt(evidence: dict[str, object], receipt: dict[str, object]) -> None:
    cas = ContentAddressedStore(evidence["cas_root"])
    stored = cas.put_bytes(canonical_bytes(receipt), media_type="application/json")
    ArchiveStore(evidence["archive_db"]).record_settled_trial(SettledTrialRecord(
        trial_id=receipt["archive_trial_id"], proposal_id=receipt["archive_trial_id"],
        goal_id="goal", library_version="v1",
        failure_context_ref=receipt.get("failure_context_ref", receipt["artifact_refs"]["result.json"]),
        verdict_ref=receipt.get("verdict_ref", receipt["artifact_refs"]["result.json"]),
        receipt_ref=stored.uri, gpu_hours=0.0, hypothesis_hash="a" * 64,
        impl_diff_hash="b" * 64, evaluator_hash="c" * 64,
        settlement_state="settled", receipt_hash=stored.sha256,
    ))
    evidence["receipt"] = dict(receipt, receipt_ref=stored.uri, receipt_hash=stored.sha256)


@pytest.fixture
def complete_evidence(evidence: dict[str, object]) -> dict[str, object]:
    cas = ContentAddressedStore(evidence["cas_root"])
    receipt = deepcopy(evidence["receipt"])
    receipt.pop("receipt_ref")
    receipt.pop("receipt_hash")
    receipt.update(
        artifact_type="verdiwm-auto-experiment-receipt", archive_trial_id="complete-trial",
        verdict_ref=cas.put_bytes(canonical_bytes(receipt["verdict"]), media_type="application/json").uri,
        failure_context_ref=cas.put_bytes(canonical_bytes({"probe": "probe"}), media_type="application/json").uri,
        support_refs={"gpu_sampling": cas.put_bytes(b"gpu sampling", media_type="text/plain").uri},
    )
    receipt["artifact_refs"]["auxiliary.txt"] = cas.put_bytes(b"support artifact", media_type="text/plain").uri
    _settle_receipt(evidence, receipt)
    return evidence


def test_complete_older_auto_receipt_without_conformance_is_reusable(complete_evidence: dict[str, object]) -> None:
    assert index_probe_experience(**complete_evidence) == 1
    rows = _retrieve(complete_evidence)
    assert len(rows) == 1 and rows[0].asset_fingerprint is None
    assert recheck_probe_experiences(
        [row.to_dict() for row in rows], archive_db=complete_evidence["archive_db"],
        cas_root=complete_evidence["cas_root"],
    ) == rows


@pytest.mark.parametrize("field", ["verdict_ref", "failure_context_ref", "gpu_sampling", "auxiliary.txt"])
@pytest.mark.parametrize("damage", ["missing", "tampered"])
def test_required_evidence_damage_blocks_index_and_both_queries(
    complete_evidence: dict[str, object], field: str, damage: str,
) -> None:
    assert index_probe_experience(**complete_evidence) == 1
    matches = [row.to_dict() for row in _retrieve(complete_evidence)]
    assert matches
    receipt = complete_evidence["receipt"]
    if field == "gpu_sampling":
        ref = receipt["support_refs"][field]
    elif field == "auxiliary.txt":
        ref = receipt["artifact_refs"][field]
    else:
        ref = receipt[field]
    digest = ref.rsplit("/", 1)[-1]
    member = complete_evidence["cas_root"] / "cas" / digest[:2] / digest
    if damage == "missing":
        member.unlink()
    else:
        member.write_bytes(b"changed CAS evidence")
    assert _retrieve(complete_evidence) == ()
    assert recheck_probe_experiences(
        matches, archive_db=complete_evidence["archive_db"], cas_root=complete_evidence["cas_root"],
    ) == ()
    new_index = complete_evidence["database_path"].with_name("new-index.db")
    with pytest.raises(ProbeRetrievalError, match="PROBE_RETRIEVAL_EVIDENCE_INVALID"):
        index_probe_experience(**dict(complete_evidence, database_path=new_index))
    assert not new_index.exists()


@pytest.mark.parametrize("field", ["verdict_ref", "failure_context_ref", "support_refs"])
def test_auto_receipt_missing_required_field_cannot_use_legacy_compatibility(
    complete_evidence: dict[str, object], field: str,
) -> None:
    index_probe_experience(**complete_evidence)
    matches = [row.to_dict() for row in _retrieve(complete_evidence)]
    receipt = deepcopy(complete_evidence["receipt"])
    receipt.pop("receipt_ref")
    receipt.pop("receipt_hash")
    receipt.pop(field)
    receipt["archive_trial_id"] = "missing-field-trial"
    _settle_receipt(complete_evidence, receipt)
    _assert_invalid_receipt_rejected(complete_evidence, matches, "INVALID")


@pytest.mark.parametrize("field,document,reason", [
    ("verdict_ref", {"verdict": "VOID"}, "MISMATCH"),
    ("verdict_ref", [], "INVALID"),
    ("failure_context_ref", [], "INVALID"),
])
def test_receipt_json_evidence_requires_object_and_matching_verdict(
    complete_evidence: dict[str, object], field: str, document: object, reason: str,
) -> None:
    index_probe_experience(**complete_evidence)
    matches = [row.to_dict() for row in _retrieve(complete_evidence)]
    receipt = deepcopy(complete_evidence["receipt"])
    receipt.pop("receipt_ref")
    receipt.pop("receipt_hash")
    cas = ContentAddressedStore(complete_evidence["cas_root"])
    receipt[field] = cas.put_bytes(canonical_bytes(document), media_type="application/json").uri
    receipt["archive_trial_id"] = "invalid-document-trial"
    _settle_receipt(complete_evidence, receipt)
    _assert_invalid_receipt_rejected(complete_evidence, matches, reason)


def _assert_invalid_receipt_rejected(
    evidence: dict[str, object], matches: list[dict[str, object]], reason: str,
) -> None:
    with pytest.raises(ProbeRetrievalError, match=f"PROBE_RETRIEVAL_EVIDENCE_{reason}"):
        index_probe_experience(**evidence)
    # Simulate a projection written before the new read-side guard existed.
    receipt = evidence["receipt"]
    binding = {key: receipt[key] for key in ("archive_trial_id", "receipt_ref", "receipt_hash")}
    with sqlite3.connect(evidence["database_path"]) as connection:
        connection.execute(
            "UPDATE experiences SET archive_trial_id=?, receipt_ref=?, receipt_hash=?",
            tuple(binding.values()),
        )
    assert _retrieve(evidence) == ()
    assert recheck_probe_experiences(
        [dict(row, **binding) for row in matches],
        archive_db=evidence["archive_db"], cas_root=evidence["cas_root"],
    ) == ()


@pytest.mark.parametrize("field,value", [
    ("metrics", {"primary": 999.0}), ("failure_signatures", ["action"]),
    ("model_family", "other"), ("primitive", "other"),
])
def test_same_probe_id_cannot_hide_changed_result(evidence: dict[str, object], field: str, value: object) -> None:
    evidence["result"] = dict(evidence["result"], **{field: value})
    with pytest.raises(ProbeRetrievalError, match="RESULT_CONTENT_MISMATCH"):
        index_probe_experience(**evidence)
    assert not evidence["database_path"].exists()


def test_receipt_verdict_cannot_be_rewritten(evidence: dict[str, object]) -> None:
    evidence["receipt"] = dict(evidence["receipt"], verdict={"verdict": "VOID"})
    with pytest.raises(ProbeRetrievalError, match="RECEIPT_CONTENT_MISMATCH"):
        index_probe_experience(**evidence)


def test_unbound_result_cannot_be_substituted(evidence: dict[str, object]) -> None:
    cas = ContentAddressedStore(evidence["cas_root"])
    result = dict(evidence["result"], metrics={"primary": 999.0})
    forged = cas.put_bytes(canonical_bytes(result), media_type="application/json")
    evidence["result"] = result
    evidence["receipt"] = dict(evidence["receipt"], artifact_refs={"result.json": forged.uri})
    with pytest.raises(ProbeRetrievalError, match="RECEIPT_CONTENT_MISMATCH"):
        index_probe_experience(**evidence)


@pytest.mark.parametrize("field,value", [
    ("metric_outcome", 999.0), ("verdict", "VOID"), ("primitive", "other"), ("stage", "screen"),
])
def test_tampered_projection_cannot_affect_retrieval(evidence: dict[str, object], field: str, value: object) -> None:
    index_probe_experience(**evidence)
    # Column names come only from this fixed parametrization, never user input.
    with sqlite3.connect(evidence["database_path"]) as connection:
        connection.execute(f"UPDATE experiences SET {field} = ?", (value,))
    assert _retrieve(evidence) == ()


def test_missing_archive_receipt_removes_projection(evidence: dict[str, object]) -> None:
    index_probe_experience(**evidence)
    with sqlite3.connect(evidence["archive_db"]) as connection:
        connection.execute("DELETE FROM receipts")
    assert _retrieve(evidence) == ()


def test_tampered_cas_result_cannot_be_indexed_or_retrieved(evidence: dict[str, object]) -> None:
    index_probe_experience(**evidence)
    cas = ContentAddressedStore(evidence["cas_root"])
    result_ref = cas.put_bytes(canonical_bytes(evidence["result"]), media_type="application/json")
    result_ref.path.write_bytes(canonical_bytes(dict(evidence["result"], metrics={"primary": 999.0})))
    assert _retrieve(evidence) == ()
    with pytest.raises(ArchiveInvariantError, match="CAS_DIGEST_MISMATCH"):
        index_probe_experience(**evidence)


def test_invalid_higher_ranked_projection_does_not_hide_valid_memory(evidence: dict[str, object]) -> None:
    index_probe_experience(**evidence)
    with sqlite3.connect(evidence["database_path"]) as connection:
        connection.execute(
            """INSERT INTO experiences(
                probe_id, model_family, runtime_capability, failure_signature,
                asset_fingerprint, primitive, stage, verdict, metric_outcome,
                archive_trial_id, receipt_ref, result_ref, receipt_hash
            ) SELECT probe_id, model_family, runtime_capability, failure_signature,
                asset_fingerprint, primitive, stage, verdict, 999.0,
                'missing-trial', receipt_ref, result_ref, receipt_hash FROM experiences"""
        )
    matches = retrieve_probe_experiences(
        **{key: evidence[key] for key in ("database_path", "archive_db", "cas_root")},
        model_family="source", runtime_capability="video", failure_signatures=("history",),
        limit=1,
    )
    assert [row.archive_trial_id for row in matches] == ["trial"]


def test_different_cas_receipt_with_same_trial_is_not_archive_authority(evidence: dict[str, object]) -> None:
    receipt = deepcopy(evidence["receipt"])
    receipt.pop("receipt_ref")
    receipt.pop("receipt_hash")
    receipt["stage"] = "screen"
    cas = ContentAddressedStore(evidence["cas_root"])
    forged = cas.put_bytes(canonical_bytes(receipt), media_type="application/json")
    receipt.update(receipt_ref=forged.uri, receipt_hash=forged.sha256)
    evidence["receipt"] = receipt
    with pytest.raises(ProbeRetrievalError, match="ARCHIVE_RECEIPT_MISMATCH"):
        index_probe_experience(**evidence)


def test_asset_identity_is_derived_from_receipted_conformance(evidence: dict[str, object]) -> None:
    cas = ContentAddressedStore(evidence["cas_root"])
    fingerprint = "d" * 64
    conformance = {
        "schema_version": 1, "artifact_type": "wmloop-model-conformance-receipt",
        "state": "settled", "verdict": "PASS", "optimization_launch_allowed": True,
        "repo_root": "/model", "sidecar_root": "/sidecar",
        "onboarding_report_sha256": "a" * 64, "model_capability_ir_sha256": "b" * 64,
        "model_capability_semantic_digest": "c" * 64, "source_revision": {},
        "source_tree_revision": {}, "runtime_python": None, "evaluator_contract_sha256": None,
        "asset_bindings": [{"parameter": "--checkpoint", "kind": "checkpoint",
                            "resolved_path": "/model/weights", "onboarding_fingerprint": fingerprint,
                            "observed_fingerprint": fingerprint, "state": "bound"}],
        "checks": [{"name": "assets", "status": "pass", "detail": "verified"}],
        "side_effects": {"source_modified": False, "dependency_install_started": False,
                         "model_import_executed": False, "gpu_execution_started": False},
        "claim_boundary": "Conformance does not prove model quality.",
    }
    ref = cas.put_bytes(canonical_bytes(conformance), media_type="application/json")
    receipt = dict(evidence["receipt"])
    receipt.pop("receipt_ref")
    receipt.pop("receipt_hash")
    receipt["archive_trial_id"] = "bound-trial"
    receipt["support_refs"] = {"onboarding_conformance": ref.uri}
    stored = cas.put_bytes(canonical_bytes(receipt), media_type="application/json")
    receipt.update(receipt_ref=stored.uri, receipt_hash=stored.sha256)
    ArchiveStore(evidence["archive_db"]).record_settled_trial(SettledTrialRecord(
        trial_id="bound-trial", proposal_id="bound-proposal", goal_id="goal", library_version="v1",
        failure_context_ref=receipt["artifact_refs"]["result.json"],
        verdict_ref=receipt["artifact_refs"]["result.json"], receipt_ref=stored.uri,
        gpu_hours=0.0, hypothesis_hash="a" * 64, impl_diff_hash="b" * 64,
        evaluator_hash="c" * 64, settlement_state="settled", receipt_hash=stored.sha256,
    ))
    evidence["receipt"] = receipt
    expected = hashlib.sha256(canonical_json_bytes([fingerprint])).hexdigest()
    with pytest.raises(ProbeRetrievalError, match="ASSET_FINGERPRINT_MISMATCH"):
        index_probe_experience(**evidence, asset_fingerprint="forged")
    assert index_probe_experience(**evidence, asset_fingerprint=expected) == 1
    assert _retrieve(evidence)[0].asset_fingerprint == expected
    with sqlite3.connect(evidence["database_path"]) as connection:
        connection.execute("UPDATE experiences SET asset_fingerprint = 'forged'")
    assert _retrieve(evidence) == ()
