"""Run or resume onboarding through settled GPU experiment execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from wmloop.contracts import ContractValidationError, validate_document
from wmloop.control.onboarding import (
    OnboardingError,
    OnboardingOptions,
    compute_asset_fingerprint,
    compute_source_revision,
    compute_source_tree_revision,
    run_onboarding,
)
from wmloop.control.onboarding_compiler import (
    OnboardingCompilerError,
    compile_and_plan,
)
from wmloop.control.onboarding_conformance import (
    ConformanceOptions,
    ModelConformanceError,
    run_conformance,
)
from wmloop.execute.auto_experiment import AutoExperimentError
from wmloop.execute.experiment_scheduler import (
    ExperimentSchedulerError,
    run_selected_queue,
)
from wmloop.execute.gpu_lease import GpuLeaseError


class AutonomousPipelineError(RuntimeError):
    """The immutable pipeline inputs or output root are invalid."""


@dataclass(frozen=True)
class AutonomousPipelineOptions:
    """Inputs for one resumable external-model experiment transaction."""

    repo_root: Path
    output_root: Path
    evaluator_contract: Path
    runtime_python: Path | None = None
    asset_bindings: tuple[tuple[str, Path], ...] = ()
    probe_imports: bool = True
    max_files: int = 20_000
    conformance_timeout_seconds: float = 30.0
    archive_db: Path | None = None
    cas_root: Path | None = None
    lock_root: Path = Path("/tmp/verdiwm-gpu-leases")
    budget_db: Path | None = None


def run_autonomous_pipeline(
    options: AutonomousPipelineOptions,
) -> dict[str, object]:
    """Run or resume the declared closed loop and return its durable manifest."""

    repo = Path(options.repo_root).expanduser().resolve()
    destination = Path(options.output_root).expanduser().resolve()
    evaluator = Path(options.evaluator_contract).expanduser().resolve()
    if not repo.is_dir() or repo.is_symlink():
        raise AutonomousPipelineError("AUTONOMOUS_PIPELINE_REPOSITORY_INVALID")
    if not evaluator.is_file() or evaluator.is_symlink():
        raise AutonomousPipelineError("AUTONOMOUS_PIPELINE_EVALUATOR_INVALID")
    if _overlaps(destination, repo):
        raise AutonomousPipelineError("AUTONOMOUS_PIPELINE_OUTPUT_OVERLAPS_SOURCE")

    archive_db = _resolved(options.archive_db, destination / "archive.db")
    cas_root = _resolved(options.cas_root, destination / "cas")
    lock_root = Path(options.lock_root).expanduser().resolve()
    budget_db = _resolved(
        options.budget_db, destination / "compiled" / "queue" / "budget.db"
    )
    for output in (archive_db, cas_root, lock_root, budget_db):
        if _overlaps(output, repo):
            raise AutonomousPipelineError(
                "AUTONOMOUS_PIPELINE_PERSISTENT_OUTPUT_OVERLAPS_SOURCE"
            )

    input_document = _input_document(
        options,
        repo=repo,
        evaluator=evaluator,
        archive_db=archive_db,
        cas_root=cas_root,
        lock_root=lock_root,
        budget_db=budget_db,
    )
    input_hash = _sha256(_canonical_json(input_document))
    _bind_output_root(destination, input_document=input_document, input_hash=input_hash)

    sidecar = destination / "onboarding"
    conformance = destination / "conformance"
    compiled = destination / "compiled"
    stage = "onboarding"
    try:
        if sidecar.exists() or sidecar.is_symlink():
            onboarding_manifest = _load_json(
                sidecar / "manifest.json", "AUTONOMOUS_PIPELINE_ONBOARDING_INVALID"
            )
        else:
            onboarding_manifest = run_onboarding(
                OnboardingOptions(
                    repo_root=repo,
                    output_root=sidecar,
                    runtime_python=options.runtime_python,
                    evaluator_contract=evaluator,
                    asset_bindings=options.asset_bindings,
                    probe_imports=options.probe_imports,
                    max_files=options.max_files,
                )
            )
        if onboarding_manifest.get("state") != "ready_for_conformance_smoke":
            return _settle_pipeline(
                destination,
                input_hash=input_hash,
                state="blocked",
                verdict="BLOCKED",
                blocked_stage="onboarding",
            )

        stage = "conformance"
        conformance_manifest = run_conformance(
            ConformanceOptions(
                sidecar_root=sidecar,
                output_root=conformance,
                timeout_seconds=options.conformance_timeout_seconds,
            )
        )
        if conformance_manifest.get("verdict") != "PASS":
            return _settle_pipeline(
                destination,
                input_hash=input_hash,
                state="blocked",
                verdict="BLOCKED",
                blocked_stage="conformance",
            )

        stage = "compilation"
        compilation_manifest = compile_and_plan(
            sidecar_root=sidecar,
            conformance_root=conformance,
            output_root=compiled,
        )
        queue_path = Path(str(compilation_manifest["queue_path"]))

        stage = "execution"
        execution = run_selected_queue(
            queue_path=queue_path,
            workspace_root=repo,
            archive_db=archive_db,
            cas_root=cas_root,
            lock_root=lock_root,
            budget_db=budget_db,
        )
        candidate_states = execution.get("candidate_states")
        passed = (
            isinstance(candidate_states, Mapping)
            and bool(candidate_states)
            and all(value == "completed" for value in candidate_states.values())
        )
        return _settle_pipeline(
            destination,
            input_hash=input_hash,
            state="settled" if passed else "blocked",
            verdict="PASS" if passed else "BLOCKED",
            blocked_stage=None if passed else "execution",
            candidate_states=(
                dict(candidate_states) if isinstance(candidate_states, Mapping) else {}
            ),
        )
    except Exception as exc:
        _write_json_atomic(
            destination / "pipeline-manifest.json",
            _manifest(
                destination,
                input_hash=input_hash,
                state="interrupted",
                verdict=None,
                blocked_stage=stage,
                error={"type": type(exc).__name__, "message": str(exc)[:500]},
            ),
        )
        raise


def _input_document(
    options: AutonomousPipelineOptions,
    *,
    repo: Path,
    evaluator: Path,
    archive_db: Path,
    cas_root: Path,
    lock_root: Path,
    budget_db: Path,
) -> dict[str, object]:
    bindings: list[dict[str, str]] = []
    parameters: set[str] = set()
    for raw_parameter, raw_path in options.asset_bindings:
        parameter = (
            raw_parameter if raw_parameter.startswith("--") else f"--{raw_parameter}"
        )
        if parameter in parameters:
            raise AutonomousPipelineError(
                f"AUTONOMOUS_PIPELINE_ASSET_DUPLICATE:{parameter}"
            )
        parameters.add(parameter)
        path = Path(raw_path).expanduser().resolve()
        try:
            fingerprint = compute_asset_fingerprint(path)
        except OnboardingError as exc:
            raise AutonomousPipelineError(
                f"AUTONOMOUS_PIPELINE_ASSET_INVALID:{parameter}"
            ) from exc
        bindings.append(
            {
                "parameter": parameter,
                "path": str(path),
                "fingerprint": fingerprint,
            }
        )
    runtime = (
        str(Path(options.runtime_python).expanduser().resolve())
        if options.runtime_python is not None
        else None
    )
    return {
        "schema_version": 1,
        "artifact_type": "verdiwm-autonomous-pipeline-input",
        "repo_root": str(repo),
        "source_revision": compute_source_revision(repo, max_files=options.max_files),
        "source_tree_revision": compute_source_tree_revision(
            repo, max_files=options.max_files
        ),
        "runtime_python": runtime,
        "evaluator_contract": str(evaluator),
        "evaluator_sha256": _sha256(evaluator.read_bytes()),
        "asset_bindings": sorted(bindings, key=lambda row: row["parameter"]),
        "probe_imports": options.probe_imports,
        "max_files": options.max_files,
        "conformance_timeout_seconds": options.conformance_timeout_seconds,
        "archive_db": str(archive_db),
        "cas_root": str(cas_root),
        "lock_root": str(lock_root),
        "budget_db": str(budget_db),
    }


def _bind_output_root(
    destination: Path,
    *,
    input_document: Mapping[str, object],
    input_hash: str,
) -> None:
    lock_path = destination / "pipeline-input.lock.json"
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or not destination.is_dir():
            raise AutonomousPipelineError("AUTONOMOUS_PIPELINE_OUTPUT_INVALID")
        lock = _load_json(lock_path, "AUTONOMOUS_PIPELINE_INPUT_LOCK_INVALID")
        if lock.get("input_hash") != input_hash or lock.get("input") != input_document:
            raise AutonomousPipelineError("AUTONOMOUS_PIPELINE_INPUT_MISMATCH")
        return
    destination.mkdir(mode=0o700, parents=True)
    _write_json_atomic(
        lock_path,
        {
            "schema_version": 1,
            "artifact_type": "verdiwm-autonomous-pipeline-input-lock",
            "input_hash": input_hash,
            "input": dict(input_document),
        },
    )


def _settle_pipeline(
    destination: Path,
    *,
    input_hash: str,
    state: str,
    verdict: str,
    blocked_stage: str | None,
    candidate_states: Mapping[str, object] | None = None,
) -> dict[str, object]:
    manifest = _manifest(
        destination,
        input_hash=input_hash,
        state=state,
        verdict=verdict,
        blocked_stage=blocked_stage,
        candidate_states=candidate_states,
    )
    _write_json_atomic(destination / "pipeline-manifest.json", manifest)
    return manifest


def _manifest(
    destination: Path,
    *,
    input_hash: str,
    state: str,
    verdict: str | None,
    blocked_stage: str | None,
    candidate_states: Mapping[str, object] | None = None,
    error: Mapping[str, str] | None = None,
) -> dict[str, object]:
    manifest = {
        "schema_version": 1,
        "artifact_type": "verdiwm-autonomous-pipeline-manifest",
        "state": state,
        "verdict": verdict,
        "input_hash": input_hash,
        "blocked_stage": blocked_stage,
        "candidate_states": dict(candidate_states or {}),
        "paths": {
            "onboarding": str(destination / "onboarding" / "manifest.json"),
            "conformance": str(destination / "conformance" / "manifest.json"),
            "compilation": str(destination / "compiled" / "manifest.json"),
            "queue_execution": str(
                destination / "compiled" / "queue" / "execution.json"
            ),
        },
        "error": dict(error) if error is not None else None,
        "claim_boundary": "A PASS proves the declared evaluator completed and settled under its frozen runtime contract; model-quality promotion remains evaluator-specific.",
    }
    try:
        validate_document("autonomous_pipeline_manifest", manifest)
    except ContractValidationError as exc:
        raise AutonomousPipelineError("AUTONOMOUS_PIPELINE_MANIFEST_INVALID") from exc
    return manifest


def _resolved(value: Path | None, default: Path) -> Path:
    return Path(value if value is not None else default).expanduser().resolve()


def _overlaps(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def _load_json(path: Path, code: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AutonomousPipelineError(code) from exc
    if not isinstance(value, dict):
        raise AutonomousPipelineError(code)
    return value


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_json_atomic(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(_canonical_json(value))
        os.replace(temporary, path)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def _parse_asset(value: str) -> tuple[str, Path]:
    parameter, separator, path = value.partition("=")
    if not separator or not parameter.strip() or not path.strip():
        raise AutonomousPipelineError("AUTONOMOUS_PIPELINE_ASSET_ARGUMENT_INVALID")
    return parameter.strip(), Path(path.strip())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--evaluator-contract", type=Path, required=True)
    parser.add_argument("--runtime-python", type=Path)
    parser.add_argument("--asset", action="append", default=[], metavar="PARAM=PATH")
    parser.add_argument("--no-import-probe", action="store_true")
    parser.add_argument("--max-files", type=int, default=20_000)
    parser.add_argument("--conformance-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--archive-db", type=Path)
    parser.add_argument("--cas-root", type=Path)
    parser.add_argument(
        "--lock-root", type=Path, default=Path("/tmp/verdiwm-gpu-leases")
    )
    parser.add_argument("--budget-db", type=Path)
    args = parser.parse_args(argv)
    try:
        manifest = run_autonomous_pipeline(
            AutonomousPipelineOptions(
                repo_root=args.repo_root,
                output_root=args.output_root,
                evaluator_contract=args.evaluator_contract,
                runtime_python=args.runtime_python,
                asset_bindings=tuple(_parse_asset(value) for value in args.asset),
                probe_imports=not args.no_import_probe,
                max_files=args.max_files,
                conformance_timeout_seconds=args.conformance_timeout_seconds,
                archive_db=args.archive_db,
                cas_root=args.cas_root,
                lock_root=args.lock_root,
                budget_db=args.budget_db,
            )
        )
    except (
        AutonomousPipelineError,
        AutoExperimentError,
        ExperimentSchedulerError,
        GpuLeaseError,
        ModelConformanceError,
        OnboardingCompilerError,
        OnboardingError,
    ) as exc:
        print(str(exc), file=__import__("sys").stderr)
        return 2
    print(json.dumps(manifest, ensure_ascii=True, sort_keys=True))
    return 0 if manifest.get("verdict") == "PASS" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
