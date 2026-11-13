"""Diagnostic execution must use the report admitted by conformance."""

import hashlib
import json

import pytest

from wmloop.diagnose import probe_campaign


def test_probe_rejects_replaced_report_before_execution(tmp_path, monkeypatch):
    sidecar = tmp_path / "sidecar"
    sidecar.mkdir()
    conformance = tmp_path / "conformance"
    conformance.mkdir()
    original = json.dumps({"repo_root": "original"}).encode()
    (sidecar / "onboarding-report.json").write_text(json.dumps({"repo_root": "replacement"}))
    monkeypatch.setattr(probe_campaign, "_load_contract", lambda path: {})
    monkeypatch.setattr(probe_campaign, "admission_from_manifest", lambda path: {
        "onboarding_report_sha256": hashlib.sha256(original).hexdigest(),
    })
    monkeypatch.setattr(probe_campaign, "run_auto_experiment", lambda **kwargs: pytest.fail("unexpected execution"))

    with pytest.raises(probe_campaign.DiagnosticProbeError, match="ONBOARDING_REPORT_HASH_MISMATCH"):
        probe_campaign.run_diagnostic_probe(
            contract_path=tmp_path / "contract.json", sidecar_root=sidecar,
            conformance_root=conformance, output_root=tmp_path / "out",
            workspace_root=tmp_path, archive_db=tmp_path / "archive.db",
            cas_root=tmp_path / "cas", lock_root=tmp_path / "locks",
        )
    assert not (tmp_path / "out").exists()
