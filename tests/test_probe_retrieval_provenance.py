"""CPU-only integration of retrieval with real temporary Archive and CAS stores."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sqlite3

import pytest

from wmloop.archive.store import ArchiveStore, ContentAddressedStore, SettledTrialRecord
from wmloop.retrieve.index import ProbeRetrievalError, index_probe_experience, retrieve_probe_experiences
from wmloop.storage import canonical_bytes


@pytest.fixture
def evidence(tmp_path: Path) -> dict[str, object]:
    cas = ContentAddressedStore(tmp_path / "cas")
    archive_db = tmp_path / "archive.db"
    archive = ArchiveStore(archive_db)
    result = {
        "schema_version": 1, "artifact_type": "verdiwm-diagnostic-probe-result",
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
