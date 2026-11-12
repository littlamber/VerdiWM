from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import json

import pytest

from scripts.evaluate_psnr_ssim_smoke import verify_receipt
from wmloop.archive.store import ContentAddressedStore
from wmloop.evaluate import local_maturity_audit as audit
from wmloop.evaluate.local_maturity_audit import LocalMaturityAuditError, _tree_digest, run_local_maturity_audit


def test_source_digest_ignores_outputs_but_detects_code_and_configuration_changes(tmp_path):
    source = tmp_path / "wmloop"
    source.mkdir()
    code = source / "entry.py"
    code.write_text("VERSION = 1\n")
    baseline = _tree_digest(tmp_path)
    for name in ("artifacts", ".venv", "wmloop/__pycache__", ".pytest_cache"):
        directory = tmp_path / name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "generated").write_text("temporary")
    assert _tree_digest(tmp_path) == baseline
    code.write_text("VERSION = 2\n")
    changed = _tree_digest(tmp_path)
    assert changed != baseline
    (tmp_path / "pyproject.toml").write_text('[project]\nname="fixture"\n')
    assert _tree_digest(tmp_path) != changed


def test_missing_assets_leave_no_output_directory(tmp_path):
    repo = tmp_path / "source"
    repo.mkdir()
    output = tmp_path / "new-parent" / "audit"
    with pytest.raises(LocalMaturityAuditError, match="LOCAL_AUDIT_ASSETS_MISSING"):
        run_local_maturity_audit(repo_root=repo, output_root=output)
    assert not output.parent.exists()


@pytest.mark.parametrize("linked_parent", [False, True])
def test_output_links_are_rejected_before_writing(tmp_path, linked_parent):
    actual = tmp_path / "actual"
    actual.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(actual, target_is_directory=True)
    output = alias / "audit" if linked_parent else alias
    with pytest.raises(LocalMaturityAuditError, match="LOCAL_AUDIT_OUTPUT_SYMLINK"):
        run_local_maturity_audit(repo_root=Path(__file__).resolve().parents[1], output_root=output)
    assert list(actual.iterdir()) == []


@pytest.mark.parametrize("failure", [LocalMaturityAuditError("FIXTURE_EVALUATOR_FAILED"), KeyboardInterrupt()])
def test_evaluator_failure_removes_only_owned_output(tmp_path, monkeypatch, failure):
    repo = tmp_path / "source"
    script = repo / "scripts/evaluate_psnr_ssim_smoke.py"
    contract = repo / "configs/evaluators/wan22_droid_psnr_ssim_smoke_v1.json"
    for path in (script, contract):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("test fixture")
    output = tmp_path / "audit"
    sibling = tmp_path / "keep"
    sibling.write_text("existing user file")
    def fail(*args, **kwargs):
        raise failure
    monkeypatch.setattr(audit, "_invoke", fail)
    with pytest.raises(type(failure)):
        run_local_maturity_audit(repo_root=repo, output_root=output)
    assert not output.exists()
    assert sibling.read_text() == "existing user file"
    output.mkdir()
    (output / "keep").write_text("existing output")
    with pytest.raises(LocalMaturityAuditError, match="LOCAL_AUDIT_OUTPUT_EXISTS"):
        run_local_maturity_audit(repo_root=repo, output_root=output)
    assert (output / "keep").read_text() == "existing output"


def test_evaluator_timeout_reaps_child(tmp_path, monkeypatch):
    import os
    import sys

    script = tmp_path / "scripts" / "evaluator.py"
    script.parent.mkdir()
    pid_file = tmp_path / "child.pid"
    script.write_text("import os, sys, time\nfrom pathlib import Path\nPath(sys.argv[1]).write_text(str(os.getpid()))\ntime.sleep(30)\n")
    monkeypatch.setattr(audit, "EVALUATOR_TIMEOUT_SECONDS", 1)
    with pytest.raises(LocalMaturityAuditError, match="LOCAL_AUDIT_EVALUATOR_TIMEOUT"):
        audit._run_evaluator([sys.executable, str(script), str(pid_file)], script, tmp_path)
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_file.read_text()), 0)


@pytest.mark.parametrize("raw", ["[]", "null", "23"])
def test_evaluator_output_requires_object(raw):
    with pytest.raises(LocalMaturityAuditError, match="LOCAL_AUDIT_OUTPUT_OBJECT_REQUIRED"):
        audit._response_object(raw)


@pytest.mark.parametrize("missing_source", [True, False])
def test_cli_reports_unavailable_paths_without_traceback(tmp_path, capsys, missing_source):
    from wmloop.cli import main

    source = tmp_path / "missing" if missing_source else Path(__file__).resolve().parents[1]
    occupied = tmp_path / "file"
    occupied.write_text("keep")
    result = main(["audit", "local-smoke", "--repo-root", str(source), "--output-root", str(occupied / "output")])
    assert result == 2
    error = capsys.readouterr().err
    assert "LOCAL_AUDIT_IO_ERROR" in error
    assert "Traceback" not in error
    assert occupied.read_text() == "keep"


def test_report_manifest_is_published_only_after_all_reports(tmp_path, monkeypatch):
    report = {"state": "ready", "operational_state": "ready", "evaluator_state": "ready", "receipt_state": "ready", "knowledge_state": "ready", "gates": [], "claim_boundary": "fixture", "audit_id": "fixture"}
    written = []
    real_write = audit.atomic_write
    def fail_markdown(path, payload):
        written.append(path.name)
        if path.suffix == ".md":
            raise OSError("fixture disk failure")
        real_write(path, payload)
    monkeypatch.setattr(audit, "atomic_write", fail_markdown)
    with pytest.raises(OSError, match="fixture disk failure"):
        audit._write_report(tmp_path, report)
    assert not (tmp_path / "manifest.json").exists()
    assert written == ["local-maturity-audit.json", "local-maturity-audit.md"]
    monkeypatch.setattr(audit, "atomic_write", real_write)
    audit._write_report(tmp_path, report)
    assert json.loads((tmp_path / "manifest.json").read_text())["state"] == "ready"


class LocalMaturityAuditTests(unittest.TestCase):
    def test_cpu_local_smoke_is_ready_and_claim_bounded(self) -> None:
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "audit"
            report = run_local_maturity_audit(repo_root=Path(__file__).resolve().parents[1], output_root=output)
            self.assertEqual(report["state"], "ready")
            self.assertEqual(report["operational_state"], "ready")
            self.assertEqual(report["evaluator_state"], "ready")
            self.assertEqual(report["receipt_state"], "ready")
            self.assertEqual(report["knowledge_state"], "ready")
            self.assertEqual(report["blocked_gate_count"], 0)
            self.assertFalse(report["side_effects"]["gpu_execution_started"])
            self.assertIn("does not establish", report["claim_boundary"])
            self.assertTrue((Path(temporary) / "audit" / "local-maturity-audit.json").is_file())
            cas = ContentAddressedStore(output / "cas")
            receipt = cas.read_bytes(report["artifacts"]["receipt_ref"])
            knowledge = cas.read_bytes(report["artifacts"]["knowledge_ref"])
            self.assertEqual(json.loads(knowledge)["artifact_type"], "verdiwm-knowledge-lifecycle")
            evidence = json.loads(receipt)
            for binding in evidence["input_artifacts"].values():
                self.assertTrue(Path(binding["path"]).exists())
            self.assertEqual(
                verify_receipt(output / "evidence" / "perfect-output" / "psnr-ssim-receipt.json")["state"],
                "ready",
            )
            self.assertTrue((output / "inputs").is_dir())


if __name__ == "__main__":
    unittest.main()
