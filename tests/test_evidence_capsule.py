from __future__ import annotations

import json
from pathlib import Path

import pytest

from wmloop.execute.autonomous_pipeline import _runtime_retrieval_manifest
from wmloop.retrieve.evidence_capsule import (
    EvidenceCapsuleError,
    build_evidence_capsule,
    write_evidence_capsule,
)


def _match(signature: str, verdict: str, metric: float, trial: str) -> dict[str, object]:
    return {
        "failure_signature": signature,
        "verdict": verdict,
        "metric_outcome": metric,
        "primitive": "anchor_repair",
        "archive_trial_id": trial,
        "receipt_ref": f"cas://receipt/{trial}",
        "result_ref": f"cas://result/{trial}",
        "receipt_hash": "a" * 64,
    }


def test_capsule_is_bounded_and_deterministic() -> None:
    probe = {
        "model_family": "ctrl-world",
        "runtime_capability": "predictive-video",
        "failure_signatures": ["action_binding", "horizon_drift", "action_binding"],
    }
    matches = [
        _match("horizon_drift", "VOID", 0.99, "trial-z"),
        _match("action_binding", "PASS", 0.2, "trial-a"),
        _match("action_binding", "PASS", 0.1, "trial-b"),
    ]
    first = build_evidence_capsule(probe=probe, matches=matches, max_evidence=2)
    second = build_evidence_capsule(probe=probe, matches=matches, max_evidence=2)
    assert first == second
    assert first["route"] == "reuse_settled"
    assert first["query"]["failure_signatures"] == ["action_binding", "horizon_drift"]
    assert len(first["selected_evidence"]) == 2
    assert first["selected_evidence"][0]["evidence"]["archive_trial_id"] == "trial-a"


def test_capsule_distinguishes_cold_start_and_no_diagnostic() -> None:
    assert build_evidence_capsule(probe=None)["route"] == "no_diagnostic"
    assert build_evidence_capsule(
        probe={"failure_signatures": ["drift"]}, matches=[]
    )["route"] == "cold_start"


def test_capsule_rejects_unbound_rows_and_writes_atomically(tmp_path: Path) -> None:
    with pytest.raises(EvidenceCapsuleError, match="MATCH_INVALID"):
        build_evidence_capsule(
            probe={"failure_signatures": ["drift"]},
            matches=[{"failure_signature": "drift"}],
        )
    capsule = build_evidence_capsule(probe=None)
    path = write_evidence_capsule(tmp_path / "retrieval" / "capsule.json", capsule)
    assert json.loads(path.read_text(encoding="utf-8")) == capsule


def test_pipeline_manifest_uses_capsule_without_repeating_matches() -> None:
    capsule = build_evidence_capsule(
        probe={"model_family": "m", "failure_signatures": ["drift"]},
        matches=[_match("drift", "PASS", 0.5, "trial-a")],
    )
    projected = _runtime_retrieval_manifest(
        {
            "state": "matched",
            "matches": [_match("drift", "PASS", 0.5, "trial-a")],
            "capsule": capsule,
            "capsule_path": "/tmp/evidence-capsule.json",
            "index_path": "/tmp/retrieval.db",
        }
    )
    assert "matches" not in projected
    assert projected["match_count"] == 1
    assert projected["capsule_path"] == "/tmp/evidence-capsule.json"
