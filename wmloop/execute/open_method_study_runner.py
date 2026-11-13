"""Execute a compiled four-arm open-method study with frozen verification.

Generated candidate processes may produce artifacts, but only a separately
bound verifier can produce evaluation rows. Both phases are exploratory until verify-layer settlement and archive binding are supported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
import statistics
import sys
import time

from wmloop.archive.store import ContentAddressedStore
from wmloop.contracts import ContractValidationError, validate_document
from wmloop.control.method_calibration import run_method_calibration
from wmloop.execute.gpu_lease import GpuLeaseManager
from wmloop.execute.open_method_runtime import load_compilation, run_candidate_operation, run_local_command, resolve_runtime_python
from wmloop.process_lifecycle import LocalCommandSignals
from wmloop.runtime_env import default_lock_root
from wmloop.storage import atomic_write, canonical_bytes, checked_path, exclusive_file_lock, sha256_bytes


ROLES = ("baseline", "source_only", "target_only", "combined")
PHASES = ("selection", "confirmation")


class OpenMethodStudyExecutionError(RuntimeError):
    """A study input or frozen execution invariant failed closed."""


def artifact_digest(path: Path) -> str:
    """Hash one regular file or a symlink-free directory tree."""

    source = checked_path(path, code="OPEN_EXECUTION_ARTIFACT_PATH_INVALID", error=OpenMethodStudyExecutionError)
    if source.is_symlink():
        raise OpenMethodStudyExecutionError("OPEN_EXECUTION_ARTIFACT_SYMLINK_FORBIDDEN")
    if source.is_file():
        return _file_digest(source)
    if not source.is_dir():
        raise OpenMethodStudyExecutionError("OPEN_EXECUTION_ARTIFACT_MISSING")
    rows: list[dict[str, str]] = []
    for child in sorted(source.rglob("*")):
        if child.is_symlink():
            raise OpenMethodStudyExecutionError("OPEN_EXECUTION_ARTIFACT_SYMLINK_FORBIDDEN")
        if child.is_file():
            rows.append({"relative_path": child.relative_to(source).as_posix(), "sha256": _file_digest(child)})
    if not rows:
        raise OpenMethodStudyExecutionError("OPEN_EXECUTION_ARTIFACT_EMPTY")
    return hashlib.sha256(canonical_bytes(rows)).hexdigest()


def build_open_method_verifier(
    *,
    verifier_root: Path,
    output_path: Path,
    command: Sequence[str],
    implementation_files: Sequence[str],
    primary_metrics: Sequence[str],
    protected_metrics: Sequence[str],
    metric_policy: Mapping[str, Mapping[str, object]],
    required_validity_gates: Sequence[str],
    effect_context: Mapping[str, object],
    composition_operator: str = "sequential",
) -> dict[str, object]:
    """Freeze an evaluator command together with its implementation bytes."""

    root = checked_path(verifier_root, code="OPEN_VERIFIER_ROOT_INVALID", error=OpenMethodStudyExecutionError)
    destination = checked_path(output_path, code="OPEN_VERIFIER_OUTPUT_INVALID", error=OpenMethodStudyExecutionError)
    # Implementation paths are interpreted relative to the verifier document
    # at execution time. Reject a spec that would be frozen against a different
    # directory and then fail (or bind different files) when loaded.
    if destination.parent != root:
        raise OpenMethodStudyExecutionError("OPEN_VERIFIER_OUTPUT_ROOT_MISMATCH")
    if destination.exists() or destination.is_symlink():
        raise OpenMethodStudyExecutionError("OPEN_VERIFIER_OUTPUT_EXISTS")
    files: list[dict[str, str]] = []
    for relative in implementation_files:
        pure = _relative_path(relative, "OPEN_VERIFIER_IMPLEMENTATION_PATH_INVALID")
        source = checked_path(root.joinpath(*pure.parts), code="OPEN_VERIFIER_IMPLEMENTATION_PATH_INVALID", error=OpenMethodStudyExecutionError)
        if not source.is_file():
            raise OpenMethodStudyExecutionError("OPEN_VERIFIER_IMPLEMENTATION_MISSING")
        files.append({"relative_path": str(pure), "sha256": _file_digest(source)})
    document: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "verdiwm-open-method-verifier",
        "command": _safe_command(command, "OPEN_VERIFIER_COMMAND_INVALID"),
        "implementation_files": files,
        "primary_metrics": list(primary_metrics),
        "protected_metrics": list(protected_metrics),
        "metric_policy": {str(name): dict(policy) for name, policy in metric_policy.items()},
        "required_validity_gates": list(required_validity_gates),
        "effect_context": dict(effect_context),
        "composition_operator": composition_operator,
        "claim_boundary": "This frozen verifier evaluates candidate artifacts but grants no publication or promotion authority.",
    }
    document["verifier_id"] = "open-verifier-" + _digest(document)[:24]
    _validate_verifier(document, verifier_path=destination, check_bytes=False)
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock = checked_path(destination.with_name(f".{destination.name}.lock"), code="OPEN_VERIFIER_OUTPUT_INVALID", error=OpenMethodStudyExecutionError)
    with exclusive_file_lock(lock):
        if destination.exists() or destination.is_symlink():
            raise OpenMethodStudyExecutionError("OPEN_VERIFIER_OUTPUT_EXISTS")
        atomic_write(destination, canonical_bytes(document) + b"\n")
    return document


def execute_open_method_study(
    *,
    study_root: Path,
    checkpoint: Path,
    train_split: Path,
    selection_split: Path,
    confirmation_split: Path,
    verifier: Path,
    output_root: Path,
    runtime_python: Path = Path(sys.executable),
    gpu_indices: Sequence[int] = (),
    lock_root: Path | None = None,
    max_parallel: int = 1,
    calibration_timeout_seconds: float = 300.0,
    gpu_wait_seconds: float = 60.0,
    lease_manager: GpuLeaseManager | None = None,
) -> dict[str, object]:
    """Run calibration, paired trials, frozen evaluation, and local settlement."""

    lock_root = lock_root if lock_root is not None else default_lock_root()
    started = time.monotonic()
    study_path = checked_path(study_root, code="OPEN_EXECUTION_STUDY_PATH_INVALID", error=OpenMethodStudyExecutionError)
    destination = checked_path(output_root, code="OPEN_EXECUTION_OUTPUT_INVALID", error=OpenMethodStudyExecutionError)
    if destination.exists() or destination.is_symlink():
        raise OpenMethodStudyExecutionError("OPEN_EXECUTION_OUTPUT_EXISTS")
    if isinstance(max_parallel, bool) or not isinstance(max_parallel, int) or max_parallel < 1:
        raise OpenMethodStudyExecutionError("OPEN_EXECUTION_PARALLELISM_INVALID")
    indices = _gpu_indices(gpu_indices)
    if not indices:
        raise OpenMethodStudyExecutionError("OPEN_EXECUTION_GPUS_REQUIRED")
    if max_parallel > len(indices):
        raise OpenMethodStudyExecutionError("OPEN_EXECUTION_PARALLELISM_EXCEEDS_GPUS")
    try:
        runtime = resolve_runtime_python(runtime_python)
    except Exception as exc:
        raise OpenMethodStudyExecutionError("OPEN_EXECUTION_RUNTIME_INVALID") from exc

    study = _load_study(study_path)
    paths = {
        "checkpoint": checked_path(checkpoint, code="OPEN_EXECUTION_CHECKPOINT_INVALID", error=OpenMethodStudyExecutionError),
        "train_split": checked_path(train_split, code="OPEN_EXECUTION_TRAIN_SPLIT_INVALID", error=OpenMethodStudyExecutionError),
        "selection_split": checked_path(selection_split, code="OPEN_EXECUTION_SELECTION_SPLIT_INVALID", error=OpenMethodStudyExecutionError),
        "confirmation_split": checked_path(confirmation_split, code="OPEN_EXECUTION_CONFIRMATION_SPLIT_INVALID", error=OpenMethodStudyExecutionError),
        "verifier": checked_path(verifier, code="OPEN_EXECUTION_VERIFIER_INVALID", error=OpenMethodStudyExecutionError),
    }
    verifier_spec = _read_json(paths["verifier"])
    _validate_verifier(verifier_spec, verifier_path=paths["verifier"])
    input_binding = _verify_bound_inputs(study, paths, verifier_spec)
    split_validation = _validate_disjoint_splits(paths)
    if verifier_spec["primary_metrics"] != study["primary_metrics"] or verifier_spec["protected_metrics"] != study["protected_metrics"]:
        raise OpenMethodStudyExecutionError("OPEN_EXECUTION_VERIFIER_METRIC_BINDING_MISMATCH")

    arm_methods: dict[str, dict[str, object]] = {}
    for arm in study["arms"]:
        role = str(arm["role"])
        method, _execution, _plan = load_compilation(study_path / str(arm["bundle_path"]))
        if method["method_id"] != arm["method_id"] or method.get("study_role") != role:
            raise OpenMethodStudyExecutionError("OPEN_EXECUTION_ARM_BINDING_MISMATCH")
        arm_methods[role] = method

    destination.mkdir(mode=0o700, parents=True)
    calibrations = _run_calibrations(
        study_path=study_path,
        study=study,
        output_root=destination / "calibration",
        timeout_seconds=calibration_timeout_seconds,
        runtime_python=runtime,
    )
    calibration_ok = all(row["state"] == "passed" for row in calibrations)
    manager = lease_manager or GpuLeaseManager(lock_root=lock_root)
    trials: list[dict[str, object]] = []
    def run_phase(phase: str, seeds: Sequence[int]) -> list[dict[str, object]]:
        work = [(arm, int(seed)) for arm in study["arms"] for seed in seeds]
        phase_trials: list[dict[str, object]] = []
        workers = min(max_parallel, len(work), len(indices))
        with LocalCommandSignals() as signal_scope, ThreadPoolExecutor(max_workers=workers, thread_name_prefix="verdiwm-open-study") as pool:
            futures = {
                pool.submit(
                    _run_trial,
                    study_path=study_path, study=study, arm=arm,
                    method=arm_methods[str(arm["role"])], seed=seed, phase=phase,
                    paths=paths, verifier_spec=verifier_spec,
                    output_root=destination / "trials" / phase / str(arm["role"]) / f"seed-{seed}",
                    runtime_python=runtime, gpu_indices=indices, lease_manager=manager,
                    gpu_wait_seconds=gpu_wait_seconds, signal_scope=signal_scope,
                ): (str(arm["role"]), seed)
                for arm, seed in work
            }
            for future in as_completed(futures):
                role, seed = futures[future]
                try:
                    phase_trials.append(future.result())
                except Exception as exc:
                    failure = _failed_trial(role, seed, exc)
                    trial_root = destination / "trials" / phase / role / f"seed-{seed}"
                    trial_root.mkdir(mode=0o700, parents=True, exist_ok=True)
                    _write_json(trial_root / "trial.json", failure)
                    phase_trials.append(failure)
        phase_trials.sort(key=lambda row: (ROLES.index(str(row["role"])), int(row["seed"])))
        return phase_trials

    if calibration_ok:
        trials.extend(run_phase("selection", study["paired_seeds"]))
    selection = _settle_phase(study, trials, verifier_spec, phase="selection", expected_units=set(_read_json(paths["selection_split"])["episode_ids"]))
    selection_cost = sum(float(row.get("gpu_seconds", 0.0)) for row in trials) / 3600.0
    if (calibration_ok and selection["state"] == "settled"
            and all(row["state"] == "passed" for row in trials)
            and selection_cost <= float(study["budget_gpu_hours"])):
        trials.extend(run_phase("confirmation", study["confirmation_seeds"]))
    confirmation = _settle_phase(study, trials, verifier_spec, phase="confirmation", expected_units=set(_read_json(paths["confirmation_split"])["episode_ids"]))
    actual_gpu_hours = sum(float(row.get("gpu_seconds", 0.0)) for row in trials) / 3600.0
    budget_ok = actual_gpu_hours <= float(study["budget_gpu_hours"])

    if not budget_ok:
        selection = {"state": "abstained", "phase": "selection", "reason": "ACTUAL_GPU_BUDGET_EXCEEDED", "actual_gpu_hours": actual_gpu_hours, "budget_gpu_hours": study["budget_gpu_hours"]}
        confirmation = {"state": "abstained", "phase": "confirmation", "reason": "ACTUAL_GPU_BUDGET_EXCEEDED", "actual_gpu_hours": actual_gpu_hours, "budget_gpu_hours": study["budget_gpu_hours"]}
    cas = ContentAddressedStore(destination / "evidence")
    evidence_index = _archive_raw_files(destination, cas)
    frozen_inputs = _archive_frozen_inputs(study_path, study, paths, verifier_spec, cas)
    frozen_inputs_ref = cas.put_bytes(canonical_bytes(frozen_inputs), media_type="application/json")
    # Bind the raw file mapping into the execution receipt. The on-disk index
    # remains a convenience view and must not itself be treated as authority.
    raw_manifest_ref = cas.put_bytes(canonical_bytes(evidence_index), media_type="application/json")
    knowledge = _write_knowledge()
    state = "failed" if not calibration_ok or not budget_ok else ("completed" if confirmation["state"] == "settled" else "abstained")
    public_trials = [{key: value for key, value in row.items() if key != "_evaluation_documents"} for row in trials]
    receipt: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "verdiwm-open-method-study-execution",
        "study_id": study["study_id"],
        "raw_evidence_manifest_ref": raw_manifest_ref.uri,
        "frozen_inputs_manifest_ref": frozen_inputs_ref.uri,
        "state": state,
        "input_binding": input_binding,
        "split_validation": split_validation,
        "calibrations": calibrations,
        "trials": public_trials,
        "selection_summary": selection,
        "confirmation_settlement": confirmation,
        "knowledge": knowledge,
        "resource_usage": {
            "gpu_indices": list(indices),
            "gpu_seconds": sum(float(row.get("gpu_seconds", 0.0)) for row in trials),
            "gpu_hours": actual_gpu_hours,
            "budget_state": "within_bound" if budget_ok else "exceeded",
            "budget_gpu_hours": study["budget_gpu_hours"],
            "estimated_gpu_hours": study["estimated_total_gpu_hours"],
            "wall_seconds": time.monotonic() - started,
        },
        "authority": {"local_evidence": bool(knowledge["effect_record_count"]), "community_projection": False, "promotion": False},
        "claim_boundary": "This receipt establishes only target-local results under the bound verifier. Community publication and model promotion require separate policy decisions.",
    }
    identity = {key: value for key, value in receipt.items() if key not in {"execution_id", "resource_usage"}}
    receipt["execution_id"] = "open-execution-" + _digest(identity)[:24]
    try:
        validate_document("open_method_study_execution", receipt)
    except ContractValidationError as exc:
        raise OpenMethodStudyExecutionError(f"OPEN_EXECUTION_RECEIPT_INVALID:{exc}") from exc
    # Do not publish a successful receipt until the bound inputs and study
    # have been checked again after execution. Partial outputs may remain for
    # diagnosis, but they must not look like a settled execution.
    _verify_bound_inputs(study, paths, verifier_spec)
    if _load_study(study_path) != study:
        raise OpenMethodStudyExecutionError("OPEN_EXECUTION_STUDY_CHANGED")
    _write_json(destination / "execution.json", receipt)
    final_ref = cas.put_bytes((destination / "execution.json").read_bytes(), media_type="application/json")
    _write_json(destination / "evidence-index.json", {"raw": evidence_index, "execution_receipt": final_ref.uri})
    return receipt


def _load_study(root: Path) -> dict[str, object]:
    manifest = checked_path(root / "study.json", code="OPEN_EXECUTION_STUDY_PATH_INVALID", error=OpenMethodStudyExecutionError)
    study = _read_json(manifest)
    if study.get("artifact_type") != "verdiwm-open-method-study" or study.get("state") != "ready_for_calibration":
        raise OpenMethodStudyExecutionError("OPEN_EXECUTION_STUDY_NOT_READY")
    if study.get("study_id", "").startswith("open-study-") is False:
        raise OpenMethodStudyExecutionError("OPEN_EXECUTION_STUDY_ID_INVALID")
    identity = {key: value for key, value in study.items() if key != "study_id"}
    if study["study_id"] != "open-study-" + _digest(identity)[:24]:
        raise OpenMethodStudyExecutionError("OPEN_EXECUTION_STUDY_ID_MISMATCH")
    arms = study.get("arms")
    if not isinstance(arms, list) or {row.get("role") for row in arms if isinstance(row, Mapping)} != set(ROLES):
        raise OpenMethodStudyExecutionError("OPEN_EXECUTION_ARMS_INVALID")
    expected = study.get("file_sha256")
    if not isinstance(expected, Mapping):
        raise OpenMethodStudyExecutionError("OPEN_EXECUTION_STUDY_FILES_MISSING")
    actual = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise OpenMethodStudyExecutionError("OPEN_EXECUTION_STUDY_SYMLINK")
        if path == manifest:
            continue
        if path.is_file():
            actual[path.relative_to(root).as_posix()] = _file_digest(path)
        elif not path.is_dir():
            raise OpenMethodStudyExecutionError("OPEN_EXECUTION_STUDY_TYPE_INVALID")
    if dict(expected) != actual:
        raise OpenMethodStudyExecutionError("OPEN_EXECUTION_STUDY_CHANGED")
    if not study.get("paired_seeds") or len(set(study["paired_seeds"])) != len(study["paired_seeds"]):
        raise OpenMethodStudyExecutionError("OPEN_EXECUTION_SEEDS_INVALID")
    confirmation = study.get("confirmation_seeds")
    seeds = study["paired_seeds"]
    if (not isinstance(confirmation, list) or len(confirmation) < 2
            or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in [*seeds, *confirmation])
            or len(set(confirmation)) != len(confirmation) or set(seeds) & set(confirmation)):
        raise OpenMethodStudyExecutionError("OPEN_EXECUTION_CONFIRMATION_SEEDS_INVALID")
    try:
        costs = [row["estimated_gpu_hours_per_seed"] for row in arms]
        budget = study["budget_gpu_hours"]
        estimate = study["estimated_total_gpu_hours"]
        if (any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
                for value in [*costs, budget, estimate])
                or not math.isclose(estimate, sum(costs) * (len(seeds) + len(confirmation)), rel_tol=1e-12)
                or estimate > budget):
            raise ValueError
    except (KeyError, TypeError, ValueError):
        raise OpenMethodStudyExecutionError("OPEN_EXECUTION_BUDGET_INVALID") from None
    return study


def _verify_bound_inputs(study: Mapping[str, object], paths: Mapping[str, Path], verifier: Mapping[str, object]) -> dict[str, object]:
    binding = study.get("experiment_binding")
    if not isinstance(binding, Mapping):
        raise OpenMethodStudyExecutionError("OPEN_EXECUTION_EXPERIMENT_BINDING_MISSING")
    mapping = {
        "checkpoint_digest": artifact_digest(paths["checkpoint"]),
        "train_split_digest": artifact_digest(paths["train_split"]),
        "selection_split_digest": artifact_digest(paths["selection_split"]),
        "confirmation_split_digest": artifact_digest(paths["confirmation_split"]),
        "verifier_digest": _file_digest(paths["verifier"]),
    }
    if dict(binding) != mapping:
        raise OpenMethodStudyExecutionError("OPEN_EXECUTION_INPUT_BINDING_MISMATCH")
    files = verifier.get("implementation_files", [])
    verifier_root = paths["verifier"].parent
    for row in files:
        pure = _relative_path(str(row["relative_path"]), "OPEN_EXECUTION_VERIFIER_PATH_INVALID")
        candidate = checked_path(verifier_root / pure, code="OPEN_EXECUTION_VERIFIER_PATH_INVALID", error=OpenMethodStudyExecutionError)
        if not candidate.is_file() or _file_digest(candidate) != row["sha256"]:
            raise OpenMethodStudyExecutionError("OPEN_EXECUTION_VERIFIER_CHANGED")
    return mapping


def _validate_disjoint_splits(paths: Mapping[str, Path]) -> dict[str, object]:
    manifests: dict[str, set[str] | None] = {}
    for name in ("train_split", "selection_split", "confirmation_split"):
        path = paths[name]
        ids: set[str] | None = None
        try:
            payload = _read_json(path)
            raw = payload.get("episode_ids") if isinstance(payload, Mapping) else None
            if (isinstance(raw, list) and raw
                    and all(isinstance(value, str) and value.strip() for value in raw)
                    and len(set(raw)) == len(raw)):
                ids = set(raw)
        except (OSError, ValueError, OpenMethodStudyExecutionError):
            ids = None
        manifests[name] = ids
    known = {name: sorted(value) for name, value in manifests.items() if value is not None}
    if len(known) != 3:
        raise OpenMethodStudyExecutionError("OPEN_EXECUTION_SPLIT_DISJOINTNESS_UNPROVEN")
    names = list(known)
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            if set(known[first]) & set(known[second]):
                raise OpenMethodStudyExecutionError("OPEN_EXECUTION_SPLIT_EPISODE_OVERLAP")
    return {"state": "verified", "episode_counts": {name: len(ids) for name, ids in known.items()}, "overlaps": []}


def _run_calibrations(*, study_path: Path, study: Mapping[str, object], output_root: Path, timeout_seconds: float, runtime_python: Path) -> list[dict[str, object]]:
    rows = []
    for arm in study["arms"]:
        role = str(arm["role"])
        destination = output_root / role
        try:
            result = run_method_calibration(
                compilation_root=study_path / str(arm["bundle_path"]),
                output_root=destination,
                timeout_seconds=timeout_seconds,
                runtime_python=runtime_python,
            )
            rows.append({"role": role, "method_id": arm["method_id"], **result})
        except Exception as exc:
            rows.append({"role": role, "method_id": arm["method_id"], "state": "failed", "error": f"{type(exc).__name__}:{exc}"})
    return rows


def _run_trial(*, study_path: Path, study: Mapping[str, object], arm: Mapping[str, object], method: Mapping[str, object], seed: int, phase: str, paths: Mapping[str, Path], verifier_spec: Mapping[str, object], output_root: Path, runtime_python: Path, gpu_indices: Sequence[int], lease_manager: GpuLeaseManager, gpu_wait_seconds: float, signal_scope: LocalCommandSignals | None = None) -> dict[str, object]:
    if signal_scope is not None:
        signal_scope.check()
    role = str(arm["role"])
    output_root.mkdir(mode=0o700, parents=True, exist_ok=False)
    lease = lease_manager.acquire(gpu_indices, wait_seconds=gpu_wait_seconds)
    started = time.monotonic()
    operations: list[dict[str, object]] = []
    documents: list[dict[str, object]] = []
    try:
        common = {
            "study_id": study["study_id"], "method_id": method["method_id"], "role": role, "seed": seed,
            "checkpoint": str(paths["checkpoint"]), "train_split": str(paths["train_split"]),
            "verifier_id": verifier_spec["verifier_id"],
        }
        mode = str(method["training"]["mode"])
        operation_count = 2 if mode == "training" else 1
        operation_timeout = float(arm["estimated_gpu_hours_per_seed"]) * 3600.0 / operation_count
        if mode == "training":
            train = run_candidate_operation(
                compilation_root=study_path / str(arm["bundle_path"]), operation="train", input_document={**common, "phase": "train"},
                output_root=output_root / "train", timeout_seconds=operation_timeout,
                runtime_python=runtime_python, gpu_devices=(lease.index,), signal_scope=signal_scope,
            )
            operations.append(train)
        if not operations or operations[-1]["state"] == "passed":
            split_key = "selection_split" if phase == "selection" else "confirmation_split"
            infer = run_candidate_operation(
                compilation_root=study_path / str(arm["bundle_path"]), operation="infer", input_document={**common, "phase": phase, "split": str(paths[split_key]), "prior_artifacts": [str(output_root / "train" / "artifacts")] if mode == "training" else []},
                output_root=output_root / phase, timeout_seconds=operation_timeout,
                runtime_python=runtime_python, gpu_devices=(lease.index,), signal_scope=signal_scope,
            )
            operations.append(infer)
            if infer["state"] == "passed":
                eval_doc = _run_frozen_verifier(verifier_spec=verifier_spec, verifier_root=paths["verifier"].parent, candidate_output=Path(str(infer["artifacts"])), input_document={**common, "phase": phase, "split": str(paths[split_key]), "artifacts": str(infer["artifacts"])}, output_root=output_root / f"verify-{phase}", runtime_python=runtime_python, signal_scope=signal_scope)
                documents.append(eval_doc)
        operation_seconds = sum(float(row.get("duration_seconds", 0.0)) for row in operations)
        # The device remains leased during verification and orchestration too.
        # Charge the full lease rather than undercounting only child operations.
        lease_seconds = time.monotonic() - started
        charged_seconds = max(operation_seconds, lease_seconds)
        within_budget = charged_seconds <= float(arm["estimated_gpu_hours_per_seed"]) * 3600.0
        evaluations_valid = _evaluation_gates_valid(documents, verifier_spec)
        state = (
            "passed"
            if len(documents) == 1
            and within_budget
            and all(row.get("state") == "passed" for row in operations)
            and evaluations_valid
            else "failed"
        )
        result = {"role": role, "method_id": method["method_id"], "seed": seed, "state": state, "gpu_lease": lease.to_document(), "gpu_seconds": charged_seconds, "lease_seconds": lease_seconds, "budget_state": "within_bound" if within_budget else "exceeded", "operations": operations, "evaluations": documents, "_evaluation_documents": documents}
    except Exception as exc:
        # Failed attempts still consume the lease and must count toward the
        # study budget. Keep their partial operations for diagnosis.
        lease_seconds = time.monotonic() - started
        result = {"role": role, "method_id": method["method_id"], "seed": seed,
                  "state": "failed", "error": f"{type(exc).__name__}:{exc}",
                  "gpu_lease": lease.to_document(), "gpu_seconds": lease_seconds,
                  "lease_seconds": lease_seconds, "budget_state": "exceeded" if lease_seconds > float(arm["estimated_gpu_hours_per_seed"]) * 3600.0 else "within_bound",
                  "operations": operations, "evaluations": documents, "_evaluation_documents": documents}
    finally:
        lease.release()
    _write_json(output_root / "trial.json", {key: value for key, value in result.items() if key != "_evaluation_documents"})
    return result


def _evaluation_gates_valid(
    documents: Sequence[Mapping[str, object]],
    verifier_spec: Mapping[str, object],
) -> bool:
    """Require every evaluator receipt gate before marking a trial passed.

    Settlement already applies these gates, but the per-trial state is also
    consumed by budget and recovery summaries.  Keeping an invalid evaluator
    receipt in a ``passed`` trial would make those operational projections
    claim success even though the phase must abstain.
    """

    if len(documents) != 1:
        return False
    required = set(verifier_spec["required_validity_gates"]) | {
        "verifier_process",
        "evaluation_schema",
        "identity_binding",
    }
    return all(
        all(document.get("validity_gates", {}).get(gate) is True for gate in required)
        for document in documents
    )


def _run_frozen_verifier(*, verifier_spec: Mapping[str, object], verifier_root: Path, candidate_output: Path, input_document: Mapping[str, object], output_root: Path, runtime_python: Path, signal_scope: LocalCommandSignals | None = None) -> dict[str, object]:
    output_root.mkdir(mode=0o700, parents=True, exist_ok=False)
    manifest = output_root / "input.json"
    _write_json(manifest, {**input_document, "candidate_output": str(candidate_output)})
    command = list(verifier_spec["command"])
    receipt = run_local_command(command=command, cwd=verifier_root, output_root=output_root / "process", timeout_seconds=3600.0, runtime_python=runtime_python, gpu_devices=(), signal_scope=signal_scope, replacements={"candidate_root": candidate_output, "input_manifest": manifest, "output_root": output_root / "result", "world_size": 1, "rank": 0})
    result_path = output_root / "result" / "evaluation.json"
    if receipt["state"] != "passed" or not result_path.is_file():
        return {"schema_version": 1, "artifact_type": "verdiwm-open-method-evaluation", "state": "failed", "validity_gates": {"verifier_process": False}, "error": receipt.get("error") or "EVALUATION_OUTPUT_MISSING"}
    payload = _read_json(result_path)
    payload.setdefault("schema_version", 1)
    payload.setdefault("artifact_type", "verdiwm-open-method-evaluation")
    payload["validity_gates"] = {**dict(payload.get("validity_gates", {})), "verifier_process": True}
    payload.setdefault("claim_boundary", "Verifier output is a paired trial observation; it is not a settled effect claim.")
    try:
        validate_document("open_method_evaluation", payload)
    except ContractValidationError:
        payload["validity_gates"]["evaluation_schema"] = False
        payload["state"] = "invalid"
    else:
        payload["validity_gates"]["evaluation_schema"] = True
    expected = {key: input_document[key] for key in ("study_id", "method_id", "role", "phase", "seed", "verifier_id")}
    if any(payload.get(key) != value for key, value in expected.items()):
        payload["validity_gates"]["identity_binding"] = False
        payload["state"] = "invalid"
    else:
        payload["validity_gates"]["identity_binding"] = True
    return payload


def _settle_phase(study: Mapping[str, object], trials: Sequence[Mapping[str, object]], verifier: Mapping[str, object], *, phase: str, expected_units: set[str]) -> dict[str, object]:
    seeds = [int(seed) for seed in study["paired_seeds" if phase == "selection" else "confirmation_seeds"]]
    required = {(role, seed) for role in ROLES for seed in seeds}
    observations: dict[tuple[str, int], Mapping[str, object]] = {}
    seen: set[tuple[str, int]] = set()
    for trial in trials:
        key = (str(trial["role"]), int(trial["seed"]))
        if key not in required:
            continue
        if key in seen:
            return {"state": "abstained", "phase": phase, "reason": "DUPLICATE_PAIRED_TRIAL"}
        seen.add(key)
        if trial.get("state") != "passed":
            return {"state": "abstained", "phase": phase, "reason": "PAIRED_TRIAL_FAILED"}
        for evaluation in trial.get("evaluations", []):
            if evaluation.get("phase") == phase:
                if key in observations:
                    return {"state": "abstained", "phase": phase, "reason": "DUPLICATE_PAIRED_OBSERVATION"}
                observations[key] = evaluation
    complete = required <= set(observations)
    if not complete:
        return {"state": "abstained", "phase": phase, "reason": "INCOMPLETE_PAIRED_OBSERVATIONS", "observed": len(observations), "required": len(required)}
    required_gates = set(verifier["required_validity_gates"]) | {"verifier_process", "evaluation_schema", "identity_binding"}
    if any(not all(doc.get("validity_gates", {}).get(gate) is True for gate in required_gates) for doc in observations.values()):
        return {"state": "abstained", "phase": phase, "reason": "VALIDITY_GATE_FAILED"}
    unit_maps: dict[tuple[str, int], dict[str, Mapping[str, object]]] = {}
    for key, document in observations.items():
        rows = document["unit_metrics"]
        mapped = {str(row["unit_id"]): row["metrics"] for row in rows}
        if len(mapped) != len(rows):
            return {"state": "abstained", "phase": phase, "reason": "DUPLICATE_UNIT_ID"}
        unit_maps[key] = mapped
    common_units = set.intersection(*(set(value) for value in unit_maps.values()))
    if len(common_units) < 2 or any(set(value) != common_units for value in unit_maps.values()):
        return {"state": "abstained", "phase": phase, "reason": "PAIRED_UNIT_MISMATCH"}
    if common_units != expected_units:
        return {"state": "abstained", "phase": phase, "reason": "SPLIT_EPISODE_COVERAGE_MISMATCH"}
    metric_rows: dict[str, object] = {}
    for metric in [*verifier["primary_metrics"], *verifier["protected_metrics"]]:
        policy = verifier["metric_policy"][metric]
        sign = 1.0 if policy["direction"] == "maximize" else -1.0
        per_role = {role: [] for role in ROLES}
        effects = {role: [] for role in ROLES[1:]}
        interactions = []
        for unit_id in sorted(common_units):
            role_values: dict[str, float] = {}
            for role in ROLES:
                seed_values = []
                for seed in seeds:
                    value = unit_maps[(role, seed)][unit_id].get(metric)
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                        return {"state": "abstained", "phase": phase, "reason": "METRIC_VALUE_INVALID", "metric": metric}
                    seed_values.append(sign * float(value))
                role_values[role] = statistics.mean(seed_values)
                per_role[role].append(role_values[role])
            for role in ROLES[1:]:
                effects[role].append(role_values[role] - role_values["baseline"])
            interactions.append(role_values["combined"] - role_values["source_only"] - role_values["target_only"] + role_values["baseline"])
        metric_rows[metric] = {"means": {role: statistics.mean(values) for role, values in per_role.items()}, "effects_vs_baseline": {role: _estimate(values) for role, values in effects.items()}, "interaction": _estimate(interactions), "unit_count": len(common_units), "direction_normalized": True}
    primary = str(verifier["primary_metrics"][0])
    result = {"state": "settled", "phase": phase, "primary_metric": primary, "metrics": metric_rows, "paired_units": len(common_units), "paired_seeds": len(seeds), "exploratory": True}
    if phase == "confirmation":
        # Independent observations still require verify-layer formal settlement
        # and settled, CAS-bound archive evidence before they can enter memory.
        result.update(state="abstained", reason="FORMAL_VERIFICATION_REQUIRED")
    return result


def _write_knowledge() -> dict[str, object]:
    """Fail closed until independent formal settlement and archive binding exist."""

    return {"state": "abstained", "effect_record_count": 0, "relation_ref": None}


def _archive_raw_files(root: Path, cas: ContentAddressedStore) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_relative_to(root / "evidence" / "cas"):
            continue
        if path.is_symlink():
            raise OpenMethodStudyExecutionError("OPEN_EXECUTION_EVIDENCE_SYMLINK")
        if path == root / "evidence-index.json":
            continue
        if path.is_file():
            ref = cas.put_file(path, media_type="application/json" if path.suffix == ".json" else "application/octet-stream")
            result[path.relative_to(root).as_posix()] = ref.uri
        elif not path.is_dir():
            raise OpenMethodStudyExecutionError("OPEN_EXECUTION_EVIDENCE_TYPE_INVALID")
    return result


def _archive_frozen_inputs(study_root: Path, study: Mapping[str, object],
                           paths: Mapping[str, Path], verifier: Mapping[str, object],
                           cas: ContentAddressedStore) -> dict[str, object]:
    """Persist reviewable frozen inputs without copying checkpoint weights."""
    study_files = _archive_raw_files(study_root, cas)
    expected_files = study["file_sha256"]
    if set(study_files) != set(expected_files) | {"study.json"}:
        raise OpenMethodStudyExecutionError("OPEN_EXECUTION_STUDY_CHANGED")
    if any(study_files[name] != f"cas://sha256/{digest}" for name, digest in expected_files.items()):
        raise OpenMethodStudyExecutionError("OPEN_EXECUTION_STUDY_CHANGED")
    if json.loads(cas.read_bytes(study_files["study.json"])) != study:
        raise OpenMethodStudyExecutionError("OPEN_EXECUTION_STUDY_CHANGED")

    inputs = {}
    binding = study["experiment_binding"]
    for name in ("train_split", "selection_split", "confirmation_split", "verifier"):
        ref = cas.put_file(paths[name], media_type="application/json")
        if ref.sha256 != binding[f"{name}_digest"]:
            raise OpenMethodStudyExecutionError("OPEN_EXECUTION_INPUT_BINDING_MISMATCH")
        inputs[name] = ref.uri
    implementation = {}
    for row in verifier["implementation_files"]:
        name = row["relative_path"]
        path = checked_path(paths["verifier"].parent / _relative_path(name, "OPEN_VERIFIER_PATH_INVALID"),
                            code="OPEN_VERIFIER_PATH_INVALID", error=OpenMethodStudyExecutionError)
        ref = cas.put_file(path, media_type="application/octet-stream")
        if ref.sha256 != row["sha256"]:
            raise OpenMethodStudyExecutionError("OPEN_EXECUTION_VERIFIER_CHANGED")
        implementation[name] = ref.uri
    return {"study_files": study_files, "inputs": inputs, "verifier_implementation": implementation,
            "checkpoint_digest": binding["checkpoint_digest"]}


def _validate_verifier(document: Mapping[str, object], *, verifier_path: Path, check_bytes: bool = True) -> None:
    try:
        validate_document("open_method_verifier", document)
    except ContractValidationError as exc:
        raise OpenMethodStudyExecutionError(f"OPEN_VERIFIER_SCHEMA_INVALID:{exc}") from exc
    if document["verifier_id"] != "open-verifier-" + _digest({key: value for key, value in document.items() if key != "verifier_id"})[:24]:
        raise OpenMethodStudyExecutionError("OPEN_VERIFIER_ID_MISMATCH")
    metrics = [*document["primary_metrics"], *document["protected_metrics"]]
    if len(metrics) != len(set(metrics)) or set(document["metric_policy"]) != set(metrics):
        raise OpenMethodStudyExecutionError("OPEN_VERIFIER_METRIC_POLICY_MISMATCH")
    placeholders = ("{candidate_root}", "{input_manifest}", "{output_root}", "{world_size}", "{rank}")
    for token in document["command"]:
        stripped = token
        for placeholder in placeholders:
            stripped = stripped.replace(placeholder, "placeholder")
        if "{" in stripped or "}" in stripped or stripped.startswith("/") or ".." in PurePosixPath(stripped).parts:
            raise OpenMethodStudyExecutionError("OPEN_VERIFIER_COMMAND_UNSAFE")
    if check_bytes:
        root = verifier_path.parent
        for row in document["implementation_files"]:
            path = checked_path(root / _relative_path(row["relative_path"], "OPEN_VERIFIER_PATH_INVALID"), code="OPEN_VERIFIER_PATH_INVALID", error=OpenMethodStudyExecutionError)
            if not path.is_file() or _file_digest(path) != row["sha256"]:
                raise OpenMethodStudyExecutionError("OPEN_EXECUTION_VERIFIER_CHANGED")


def _verifier_body(**kwargs: object) -> dict[str, object]:
    return {"schema_version": 1, "artifact_type": "verdiwm-open-method-verifier", **kwargs, "claim_boundary": "This frozen verifier evaluates candidate artifacts but grants no publication or promotion authority."}


def _safe_command(value: Sequence[str], error: str) -> list[str]:
    if not isinstance(value, (list, tuple)) or not value or any(not isinstance(token, str) or not token for token in value):
        raise OpenMethodStudyExecutionError(error)
    for token in value:
        if any(char in token for char in ("\x00", "\n", "\r", ";", "&", "|", "`")):
            raise OpenMethodStudyExecutionError(error)
    return list(value)


def _relative_path(value: str, error: str) -> PurePosixPath:
    pure = PurePosixPath(value)
    if not value or pure.is_absolute() or ".." in pure.parts or str(pure) != value:
        raise OpenMethodStudyExecutionError(error)
    return pure


def _gpu_indices(values: Sequence[int]) -> tuple[int, ...]:
    result = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value in result:
            raise OpenMethodStudyExecutionError("OPEN_EXECUTION_GPU_INDEX_INVALID")
        result.append(value)
    return tuple(result)


def _failed_trial(role: str, seed: int, exc: BaseException) -> dict[str, object]:
    return {"role": role, "seed": seed, "state": "failed", "error": f"{type(exc).__name__}:{exc}", "gpu_lease": None, "gpu_seconds": 0.0, "operations": [], "evaluations": []}


def _estimate(values: Sequence[float]) -> dict[str, float]:
    mean = statistics.mean(values)
    se = statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return {"mean": mean, "standard_error": se, "lower_bound": mean - 1.96 * se, "upper_bound": mean + 1.96 * se}


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise OpenMethodStudyExecutionError(f"OPEN_EXECUTION_JSON_INVALID:{path.name}") from exc
    if not isinstance(value, dict):
        raise OpenMethodStudyExecutionError("OPEN_EXECUTION_JSON_OBJECT_REQUIRED")
    return value


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    atomic_write(path, canonical_bytes(value) + b"\n")


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value: object) -> str:
    return sha256_bytes(canonical_bytes(value))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-split", type=Path, required=True)
    parser.add_argument("--selection-split", type=Path, required=True)
    parser.add_argument("--confirmation-split", type=Path, required=True)
    parser.add_argument("--verifier", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--runtime-python",
        type=Path,
        default=Path(sys.executable),
        help="candidate model environment Python (defaults to this interpreter)",
    )
    parser.add_argument(
        "--lock-root",
        type=Path,
        default=default_lock_root(),
        help="shared GPU lease directory",
    )
    parser.add_argument(
        "--calibration-timeout-seconds",
        type=float,
        default=300.0,
        help="per-method CPU calibration timeout",
    )
    parser.add_argument(
        "--gpu-wait-seconds",
        type=float,
        default=60.0,
        help="maximum wait for a leased GPU per trial",
    )
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--max-parallel", type=int, default=1)
    args = parser.parse_args()
    receipt = execute_open_method_study(
        study_root=args.study,
        checkpoint=args.checkpoint,
        train_split=args.train_split,
        selection_split=args.selection_split,
        confirmation_split=args.confirmation_split,
        verifier=args.verifier,
        output_root=args.output,
        runtime_python=args.runtime_python,
        gpu_indices=tuple(
            int(value) for value in args.gpus.split(",") if value.strip()
        ),
        lock_root=args.lock_root,
        max_parallel=args.max_parallel,
        calibration_timeout_seconds=args.calibration_timeout_seconds,
        gpu_wait_seconds=args.gpu_wait_seconds,
    )
    print(json.dumps({"execution_id": receipt["execution_id"], "state": receipt["state"], "output": str(args.output)}, ensure_ascii=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
