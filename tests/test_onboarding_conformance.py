from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest

from wmloop.control.onboarding import OnboardingOptions, run_onboarding
from wmloop.control.onboarding_conformance import (
    ConformanceOptions,
    ModelConformanceError,
    run_conformance,
)


def test_conformance_passes_without_mutating_source(tmp_path: Path) -> None:
    repo = _model_repo(tmp_path / "model")
    evaluator = _evaluator_contract(tmp_path / "evaluator.json")
    before = _snapshot(repo)
    sidecar = tmp_path / "sidecar"
    run_onboarding(
        OnboardingOptions(
            repo_root=repo,
            output_root=sidecar,
            runtime_python=Path(sys.executable),
            evaluator_contract=evaluator,
            probe_imports=False,
        )
    )

    output = tmp_path / "conformance"
    manifest = run_conformance(
        ConformanceOptions(sidecar_root=sidecar, output_root=output)
    )

    assert manifest["verdict"] == "PASS"
    assert manifest["optimization_launch_allowed"] is True
    assert _snapshot(repo) == before
    receipt = json.loads(
        (output / "conformance-receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["side_effects"]["gpu_execution_started"] is False
    assert receipt["side_effects"]["source_modified"] is False
    assert receipt["source_tree_revision"]["kind"] == "source_tree_sha256"
    assert receipt["asset_bindings"][0]["state"] == "bound"
    assert all(check["status"] == "pass" for check in receipt["checks"])


def test_transitive_import_failure_is_a_durable_blocker(tmp_path: Path) -> None:
    repo = _model_repo(tmp_path / "model")
    (repo / "broken_import.py").write_text(
        "raise ImportError('incompatible dependency')\n", encoding="utf-8"
    )
    evaluator = _evaluator_contract(
        tmp_path / "evaluator.json", imports=["broken_import"]
    )
    sidecar = tmp_path / "sidecar"
    run_onboarding(
        OnboardingOptions(
            repo_root=repo,
            output_root=sidecar,
            runtime_python=Path(sys.executable),
            evaluator_contract=evaluator,
            probe_imports=False,
        )
    )

    output = tmp_path / "conformance"
    manifest = run_conformance(
        ConformanceOptions(sidecar_root=sidecar, output_root=output)
    )

    assert manifest["verdict"] == "BLOCKED"
    receipt = json.loads(
        (output / "conformance-receipt.json").read_text(encoding="utf-8")
    )
    import_check = next(
        check for check in receipt["checks"] if check["name"] == "module_import_00"
    )
    assert import_check["status"] == "fail"
    assert "incompatible dependency" in import_check["detail"]
    assert receipt["optimization_launch_allowed"] is False


def test_tampered_onboarding_report_is_rejected(tmp_path: Path) -> None:
    repo = _model_repo(tmp_path / "model")
    sidecar = tmp_path / "sidecar"
    run_onboarding(
        OnboardingOptions(
            repo_root=repo,
            output_root=sidecar,
            runtime_python=Path(sys.executable),
            evaluator_contract=_evaluator_contract(tmp_path / "evaluator.json"),
            probe_imports=False,
        )
    )
    report_path = sidecar / "onboarding-report.json"
    report_path.write_text(
        report_path.read_text(encoding="utf-8") + " ", encoding="utf-8"
    )

    with pytest.raises(ModelConformanceError, match="REPORT_HASH_MISMATCH"):
        run_conformance(
            ConformanceOptions(
                sidecar_root=sidecar, output_root=tmp_path / "conformance"
            )
        )


def test_conformance_resume_verifies_source_and_receipt(tmp_path: Path) -> None:
    repo = _model_repo(tmp_path / "model")
    sidecar = tmp_path / "sidecar"
    run_onboarding(
        OnboardingOptions(
            repo_root=repo,
            output_root=sidecar,
            runtime_python=Path(sys.executable),
            evaluator_contract=_evaluator_contract(tmp_path / "evaluator.json"),
            probe_imports=False,
        )
    )
    options = ConformanceOptions(
        sidecar_root=sidecar, output_root=tmp_path / "conformance"
    )
    first = run_conformance(options)
    second = run_conformance(options)
    assert first == second

    receipt_path = tmp_path / "conformance" / "conformance-receipt.json"
    receipt_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ModelConformanceError, match="RECEIPT_HASH_MISMATCH"):
        run_conformance(options)


def test_output_inside_source_is_rejected(tmp_path: Path) -> None:
    repo = _model_repo(tmp_path / "model")
    sidecar = tmp_path / "sidecar"
    run_onboarding(
        OnboardingOptions(
            repo_root=repo,
            output_root=sidecar,
            runtime_python=Path(sys.executable),
            evaluator_contract=_evaluator_contract(tmp_path / "evaluator.json"),
            probe_imports=False,
        )
    )
    with pytest.raises(ModelConformanceError, match="OUTPUT_INSIDE_SOURCE"):
        run_conformance(
            ConformanceOptions(sidecar_root=sidecar, output_root=repo / "conformance")
        )


def test_asset_drift_before_conformance_is_a_durable_blocker(tmp_path: Path) -> None:
    repo = _model_repo(tmp_path / "model")
    sidecar = tmp_path / "sidecar"
    run_onboarding(
        OnboardingOptions(
            repo_root=repo,
            output_root=sidecar,
            runtime_python=Path(sys.executable),
            evaluator_contract=_evaluator_contract(tmp_path / "evaluator.json"),
            probe_imports=False,
        )
    )
    (repo / "checkpoint.pt").write_bytes(b"changed-checkpoint")

    output = tmp_path / "conformance"
    manifest = run_conformance(
        ConformanceOptions(sidecar_root=sidecar, output_root=output)
    )

    assert manifest["verdict"] == "BLOCKED"
    receipt = json.loads(
        (output / "conformance-receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["asset_bindings"][0]["state"] == "drifted"
    check = next(item for item in receipt["checks"] if item["name"] == "asset_bindings")
    assert check["status"] == "fail"


def _model_repo(repo: Path) -> Path:
    (repo / "scripts").mkdir(parents=True)
    (repo / "requirements.txt").write_text("\n", encoding="utf-8")
    (repo / "checkpoint.pt").write_bytes(b"checkpoint")
    (repo / "scripts" / "eval_rollout.py").write_text(
        "import argparse\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--ckpt-path')\n"
        "if __name__ == '__main__':\n"
        "    parser.parse_args()\n",
        encoding="utf-8",
    )
    return repo


def _evaluator_contract(path: Path, *, imports: list[str] | None = None) -> Path:
    path.write_text(
        json.dumps(
            {
                "evaluator_id": "synthetic_eval_v1",
                "command": [
                    "{python}",
                    "scripts/eval_rollout.py",
                    "--ckpt-path",
                    "{asset:--ckpt-path}",
                ],
                "input_artifacts": ["checkpoint"],
                "output_artifacts": ["result.json"],
                "metrics": ["success_rate"],
                "verifier": "synthetic_receipt_v1",
                "conformance_imports": imports or ["json"],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def _snapshot(repo: Path) -> dict[str, str]:
    return {
        path.relative_to(repo).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(repo.rglob("*"))
        if path.is_file()
    }
