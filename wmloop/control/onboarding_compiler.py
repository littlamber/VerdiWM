"""Compile a passing onboarding admission into a scheduler-ready queue."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

from wmloop.contracts import ContractValidationError, validate_document
from wmloop.control.onboarding_admission import (
    OnboardingAdmissionError,
    admission_from_manifest,
    verify_onboarding_admission,
)
from wmloop.execute.experiment_scheduler import (
    ExperimentSchedulerError,
    _validate_batch_semantics,
    plan_candidate_batch,
)


_PLACEHOLDER = re.compile(
    r"\{(?:python|verdiwm_python|repo_root|asset:--[A-Za-z0-9_-]+)\}"
)
_RUNTIME_PLACEHOLDERS = {
    "{scratch_dir}",
    "{workspace_root}",
    "{output_root}",
    "{gpu_index}",
    "{gpu_uuid}",
}


class OnboardingCompilerError(RuntimeError):
    """A passing admission could not be compiled safely."""


def compile_and_plan(
    *,
    sidecar_root: Path,
    conformance_root: Path,
    output_root: Path,
) -> dict[str, object]:
    """Materialize a frozen template and produce its deterministic queue."""

    sidecar = Path(sidecar_root).expanduser().resolve()
    conformance = Path(conformance_root).expanduser().resolve()
    destination = Path(output_root).expanduser().resolve()
    sidecar_manifest = _load_json(
        sidecar / "manifest.json", "ONBOARDING_COMPILER_SIDECAR_MANIFEST_INVALID"
    )
    report_path = sidecar / "onboarding-report.json"
    report_bytes = _read_bound_file(
        report_path,
        str(sidecar_manifest.get("report_sha256", "")),
        "ONBOARDING_COMPILER_REPORT_HASH_MISMATCH",
    )
    report = _decode_object(report_bytes, "ONBOARDING_COMPILER_REPORT_INVALID")
    repo = Path(str(report.get("repo_root", ""))).resolve()
    _validate_output(destination, repo=repo, sidecar=sidecar, conformance=conformance)

    admission = admission_from_manifest(conformance)
    try:
        receipt = verify_onboarding_admission(admission, expected_repo_root=repo)
    except OnboardingAdmissionError as exc:
        raise OnboardingCompilerError(
            f"ONBOARDING_COMPILER_ADMISSION_INVALID:{exc}"
        ) from exc
    evaluator = report.get("evaluator_contract")
    if not isinstance(evaluator, Mapping) or evaluator.get("state") != "ready":
        raise OnboardingCompilerError("ONBOARDING_COMPILER_EVALUATOR_NOT_READY")
    template_path = Path(str(evaluator.get("scheduler_template_path", ""))).resolve()
    template_sha256 = str(evaluator.get("scheduler_template_sha256", ""))
    template_bytes = _read_bound_file(
        template_path,
        template_sha256,
        "ONBOARDING_COMPILER_TEMPLATE_HASH_MISMATCH",
    )
    template = _decode_object(template_bytes, "ONBOARDING_COMPILER_TEMPLATE_INVALID")
    input_hash = _sha256(
        report_bytes + b"\0" + _canonical_json(receipt) + b"\0" + template_bytes
    )

    if destination.exists() or destination.is_symlink():
        manifest = _resume(destination, input_hash=input_hash)
        plan_candidate_batch(
            batch_path=destination / "candidate-batch.json",
            output_root=destination / "queue",
            workspace_root=repo,
        )
        return manifest

    values = _materialization_values(report, repo=repo)
    batch = _materialize(template, values)
    if not isinstance(batch, dict):
        raise OnboardingCompilerError("ONBOARDING_COMPILER_TEMPLATE_OBJECT_REQUIRED")
    batch["onboarding_admission"] = admission
    try:
        validate_document("auto_experiment_candidate_batch", batch)
        _validate_batch_semantics(batch, workspace_root=repo)
    except (ContractValidationError, ExperimentSchedulerError) as exc:
        raise OnboardingCompilerError(
            f"ONBOARDING_COMPILER_BATCH_INVALID:{exc}"
        ) from exc

    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=str(destination.parent))
    )
    try:
        _write_json(temporary / "candidate-batch.json", batch)
        _write_json(temporary / "onboarding-admission.json", admission)
        manifest = {
            "schema_version": 1,
            "artifact_type": "wmloop-onboarded-candidate-compilation",
            "state": "ready",
            "input_hash": input_hash,
            "repo_root": str(repo),
            "candidate_batch_path": str(destination / "candidate-batch.json"),
            "candidate_batch_sha256": _sha256(_canonical_json(batch)),
            "admission_receipt_path": admission["receipt_path"],
            "admission_receipt_sha256": admission["receipt_sha256"],
            "queue_path": str(destination / "queue" / "queue.json"),
            "optimization_launch_allowed": True,
        }
        _write_json(temporary / "manifest.json", manifest)
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    plan_candidate_batch(
        batch_path=destination / "candidate-batch.json",
        output_root=destination / "queue",
        workspace_root=repo,
    )
    return manifest


def _materialization_values(
    report: Mapping[str, object], *, repo: Path
) -> dict[str, str]:
    runtime = report.get("runtime")
    connector = report.get("connector")
    if not isinstance(runtime, Mapping) or not isinstance(connector, Mapping):
        raise OnboardingCompilerError("ONBOARDING_COMPILER_CONNECTOR_INVALID")
    selected_python = runtime.get("selected_python")
    if not isinstance(selected_python, str) or not Path(selected_python).is_file():
        raise OnboardingCompilerError("ONBOARDING_COMPILER_RUNTIME_INVALID")
    values = {
        "{python}": selected_python,
        "{verdiwm_python}": str(Path(__import__("sys").executable).absolute()),
        "{repo_root}": str(repo),
    }
    asset_rows = connector.get("asset_bindings")
    if not isinstance(asset_rows, list):
        raise OnboardingCompilerError("ONBOARDING_COMPILER_ASSETS_INVALID")
    for row in asset_rows:
        if not isinstance(row, Mapping) or row.get("state") != "discovered":
            continue
        parameter = str(row.get("parameter", ""))
        resolved_path = row.get("resolved_path")
        if parameter and isinstance(resolved_path, str):
            values[f"{{asset:{parameter}}}"] = resolved_path
    return values


def _materialize(value: object, values: Mapping[str, str]) -> object:
    if isinstance(value, dict):
        return {str(key): _materialize(child, values) for key, child in value.items()}
    if isinstance(value, list):
        return [_materialize(child, values) for child in value]
    if not isinstance(value, str):
        return value
    materialized = value
    for placeholder in _PLACEHOLDER.findall(value):
        replacement = values.get(placeholder)
        if replacement is None:
            raise OnboardingCompilerError(
                f"ONBOARDING_COMPILER_PLACEHOLDER_UNBOUND:{placeholder}"
            )
        materialized = materialized.replace(placeholder, replacement)
    remaining = set(re.findall(r"\{[^{}]+\}", materialized))
    if remaining - _RUNTIME_PLACEHOLDERS:
        raise OnboardingCompilerError("ONBOARDING_COMPILER_PLACEHOLDER_INVALID")
    return materialized


def _resume(destination: Path, *, input_hash: str) -> dict[str, object]:
    if destination.is_symlink() or not destination.is_dir():
        raise OnboardingCompilerError("ONBOARDING_COMPILER_OUTPUT_INVALID")
    manifest = _load_json(
        destination / "manifest.json", "ONBOARDING_COMPILER_MANIFEST_INVALID"
    )
    if manifest.get("input_hash") != input_hash:
        raise OnboardingCompilerError("ONBOARDING_COMPILER_INPUT_MISMATCH")
    batch_path = destination / "candidate-batch.json"
    if _sha256(batch_path.read_bytes()) != manifest.get("candidate_batch_sha256"):
        raise OnboardingCompilerError("ONBOARDING_COMPILER_BATCH_HASH_MISMATCH")
    return manifest


def _validate_output(
    destination: Path, *, repo: Path, sidecar: Path, conformance: Path
) -> None:
    for root, code in (
        (repo, "ONBOARDING_COMPILER_OUTPUT_INSIDE_SOURCE"),
        (sidecar, "ONBOARDING_COMPILER_OUTPUT_INSIDE_SIDECAR"),
        (conformance, "ONBOARDING_COMPILER_OUTPUT_INSIDE_CONFORMANCE"),
    ):
        if destination == root or root in destination.parents:
            raise OnboardingCompilerError(code)


def _read_bound_file(path: Path, expected_sha256: str, code: str) -> bytes:
    if not path.is_file() or path.is_symlink():
        raise OnboardingCompilerError(code)
    payload = path.read_bytes()
    if _sha256(payload) != expected_sha256:
        raise OnboardingCompilerError(code)
    return payload


def _decode_object(payload: bytes, code: str) -> dict[str, object]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OnboardingCompilerError(code) from exc
    if not isinstance(value, dict):
        raise OnboardingCompilerError(code)
    return value


def _load_json(path: Path, code: str) -> dict[str, object]:
    try:
        return _decode_object(path.read_bytes(), code)
    except OSError as exc:
        raise OnboardingCompilerError(code) from exc


def _canonical_json(payload: object) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.write_bytes(_canonical_json(payload))


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sidecar-root", type=Path, required=True)
    parser.add_argument("--conformance-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        manifest = compile_and_plan(
            sidecar_root=args.sidecar_root,
            conformance_root=args.conformance_root,
            output_root=args.output_root,
        )
    except (OnboardingCompilerError, OnboardingAdmissionError) as exc:
        print(str(exc), file=__import__("sys").stderr)
        return 2
    print(json.dumps(manifest, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
