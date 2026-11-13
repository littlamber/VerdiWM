"""One contract governs diagnostic producers, routing and retrieval."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from wmloop.archive.store import ContentAddressedStore
from wmloop.contracts import ContractValidationError, validate_document
from wmloop.diagnose import external_video_probe, probe_campaign
from wmloop.retrieve import index
from wmloop.storage import canonical_bytes


@pytest.fixture
def result() -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_type": "verdiwm-diagnostic-probe-result",
        "state": "ready",
        "probe_id": "probe",
        "model_family": "source",
        "runtime_capability": "video",
        "failure_signatures": ["history"],
        "metrics": {"primary": 0.2},
        "claim_boundary": "Diagnostic routing only; no formal effect claim.",
    }


def _contract() -> dict[str, object]:
    return {
        "probe_id": "probe", "model_family": "source", "runtime_capability": "video",
        "failure_signature_fields": ["failure_signatures", "metrics"],
    }


def _reject_every_boundary(result: dict[str, object], tmp_path: Path) -> None:
    with pytest.raises(ContractValidationError):
        validate_document("diagnostic_probe_result", result)
    with pytest.raises(probe_campaign.DiagnosticProbeError, match="RESULT_CONTRACT_INVALID"):
        probe_campaign._validate_probe_result(result, _contract())
    with pytest.raises(index.ProbeRetrievalError, match="RESULT_CONTRACT_INVALID"):
        index.index_probe_experience(
            database_path=tmp_path / "index.db", result=result, receipt={},
            archive_db=tmp_path / "archive.db", cas_root=tmp_path / "cas",
        )
    with pytest.raises(external_video_probe.ExternalVideoProbeError, match="RESULT_INVALID"):
        external_video_probe._write_json(tmp_path / "result.json", result)
    assert not (tmp_path / "index.db").exists()
    assert not (tmp_path / "result.json").exists()


@pytest.mark.parametrize("metric", [float("nan"), float("inf"), -float("inf"), True, False, "0.2", None, 10**400])
def test_invalid_metrics_fail_all_boundaries(result, tmp_path, metric):
    result["metrics"] = {"primary": metric}
    _reject_every_boundary(result, tmp_path)


@pytest.mark.parametrize("field", [
    "schema_version", "artifact_type", "state", "probe_id", "model_family",
    "runtime_capability", "failure_signatures", "metrics", "claim_boundary",
])
def test_missing_fields_fail_all_boundaries(result, tmp_path, field):
    result.pop(field)
    _reject_every_boundary(result, tmp_path)


@pytest.mark.parametrize("field,value", [
    ("schema_version", True), ("schema_version", 2), ("state", "partial"),
    ("artifact_type", "verdiwm-auto-experiment-result"),
    ("probe_id", ""), ("model_family", False), ("runtime_capability", ""),
    ("failure_signatures", []), ("failure_signatures", ["history", "history"]),
    ("failure_signatures", [True]), ("metrics", {}), ("metrics", {"": 1.0}),
    ("unknown_field", "undeclared"), ("primitive", False),
    ("video", {"path": "rollout.mp4", "layout": "vertical", "unknown": 1}),
])
def test_invalid_shapes_and_unknown_fields_fail_all_boundaries(result, tmp_path, field, value):
    result[field] = value
    _reject_every_boundary(result, tmp_path)


def test_valid_result_survives_cas_and_both_consumers(result, tmp_path):
    cas = ContentAddressedStore(tmp_path / "cas")
    ref = cas.put_bytes(canonical_bytes(result), media_type="application/json")
    restored = json.loads(cas.read_bytes(ref.uri))
    validate_document("diagnostic_probe_result", restored)
    probe_campaign._validate_probe_result(restored, _contract())
    index._validate_result(restored)


@pytest.mark.parametrize("field", ["probe_id", "model_family", "runtime_capability"])
def test_routing_requires_contract_identity(result, field):
    result[field] = "different"
    with pytest.raises(probe_campaign.DiagnosticProbeError, match="ID_OR_STATE_INVALID"):
        probe_campaign._validate_probe_result(result, _contract())


def test_invalid_state_is_a_valid_failure_artifact_but_not_retrieval_evidence(result):
    result.update(state="invalid", failure_signatures=[], error="WORKLOAD_FAILED")
    validate_document("diagnostic_probe_result", result)
    with pytest.raises(probe_campaign.DiagnosticProbeError, match="ID_OR_STATE_INVALID"):
        probe_campaign._validate_probe_result(result, _contract())
    with pytest.raises(index.ProbeRetrievalError, match="RESULT_NOT_READY"):
        index._validate_result(result)


@pytest.mark.parametrize("exit_code", [0, 3])
def test_external_video_producer_emits_schema_valid_success_and_failure(tmp_path, monkeypatch, exit_code):
    (tmp_path / "rollout.mp4").touch()
    monkeypatch.setenv("VERDIWM_PROBE_ID", "probe")
    monkeypatch.setattr(external_video_probe.subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(args, exit_code))
    monkeypatch.setattr(external_video_probe, "_read_video", lambda *args, **kwargs: np.zeros((2, 4, 2, 3), dtype=np.uint8))
    status = external_video_probe.run_external_video_probe(
        command=["declared-model-command"], scratch_root=tmp_path,
        video_pattern="rollout.mp4", output_path="diagnostic.json",
        model_family="source", runtime_capability="video",
    )
    output = json.loads((tmp_path / "diagnostic.json").read_text())
    validate_document("diagnostic_probe_result", output)
    assert status == exit_code
    assert output["state"] == ("ready" if exit_code == 0 else "invalid")


def test_legacy_diagnostic_output_contract_remains_readable():
    legacy = {
        "schema_version": 1, "artifact_type": "wmloop-diagnostic-probe-output",
        "probe_id": "legacy", "role": "diagnostic", "environment": "test",
        "signature": "history", "state": "measured", "metrics": {"score": 0.2},
        "flags": [], "evidence_refs": [], "verdict_exposure_allowed": False,
        "limitations": ["Diagnostic only."],
    }
    validate_document("diagnostic_probe_output", deepcopy(legacy))
    with pytest.raises(ContractValidationError):
        validate_document("diagnostic_probe_result", legacy)
