from __future__ import annotations

import json
from pathlib import Path

import pytest

from wmloop.evidence_capsule import (
    EvidenceCapsuleError,
    export_evidence_capsule,
    import_evidence_capsule,
    validate_evidence_capsule,
)


def _receipt(path: Path) -> Path:
    payload = {
        "artifact_type": "verdiwm-wan22-droid-closed-loop-receipt",
        "state": "verified",
        "checkpoint_path": "/private/checkpoints/candidate.pt",
        "dataset_path": "/private/data/droid",
        "runtime_path": "/private/runs/42",
        "command": ["python", "train.py"],
        "environment": {"CUDA_VISIBLE_DEVICES": "0"},
        "model_family": "wan2.2",
        "evidence": {
            "capability_class": "action-conditioned-video",
            "irg_coordinates": {"temporal": 0.4, "action": -0.2},
            "failure_signatures": ["subject_consistency"],
            "probe_lineage": ["probe-1"],
            "intervention_semantics": {"dose": 0.2},
            "evaluator_sha256": "a" * 64,
            "effect_estimate": {"mean": 0.1},
            "uncertainty": {"ci95": [0.0, 0.2]},
            "protected_metrics": {"smoothness": {"delta": -0.1}},
            "anti_conditions": ["short horizon"],
            "raw_video": "/private/videos/episode.mp4",
            "cas_ref": "cas://sha256/" + "b" * 64,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_export_removes_private_runtime_fields_and_validates(tmp_path: Path) -> None:
    source = _receipt(tmp_path / "receipt.json")
    output = tmp_path / "public" / "capsule.json"
    capsule = export_evidence_capsule(source, output, publisher="lab", license="Apache-2.0")
    assert output.is_file()
    assert validate_evidence_capsule(output) == capsule
    text = output.read_text(encoding="utf-8")
    assert "checkpoint_path" not in text
    assert "raw_video" not in text
    assert capsule["state"] == "verified"
    assert capsule["evidence"]["model_family"] == "wan2.2"
    assert capsule["evidence"]["capability_class"] == "action-conditioned-video"


def test_import_indexes_capsule_as_prior_only(tmp_path: Path) -> None:
    source = _receipt(tmp_path / "receipt.json")
    capsule_path = tmp_path / "capsule.json"
    export_evidence_capsule(source, capsule_path, publisher="lab", license="Apache-2.0")
    result = import_evidence_capsule(capsule_path, tmp_path / "index")
    receipt = result["receipt"]
    assert receipt["artifact_type"] == "verdiwm-evidence-capsule-import-receipt"
    assert receipt["routing_authority"] == "prior_only"
    assert receipt["target_side_verdict"] is None
    assert receipt["trial_authorized"] is False
    index = json.loads(Path(result["index_path"]).read_text(encoding="utf-8"))
    assert index["routing_authority"] == "prior_only"
    assert len(index["capsules"]) == 1


def test_validate_rejects_private_absolute_path_and_bad_hash() -> None:
    base = {
        "schema_version": 1,
        "artifact_type": "verdiwm-evidence-capsule",
        "capsule_id": "capsule-test",
        "state": "exploratory",
        "publisher": "lab",
        "license": "Apache-2.0",
        "source_artifact_type": "receipt",
        "source_sha256": "a" * 64,
        "evidence": {"model_family": "m"},
        "relations": [],
        "claim_boundary": "This capsule is a portable evidence projection and routing prior, not a target-side verdict or execution authorization.",
    }
    bad_path = dict(base, evidence={"private": "/etc/passwd"})
    with pytest.raises(EvidenceCapsuleError, match="PRIVATE"):
        validate_evidence_capsule(bad_path)
    bad_hash = dict(base, source_sha256="not-a-hash")
    with pytest.raises(EvidenceCapsuleError, match="HASH"):
        validate_evidence_capsule(bad_hash)
