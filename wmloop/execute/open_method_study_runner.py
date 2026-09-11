"""Execute a compiled four-arm open-method study with frozen verification.

Generated candidate processes may produce artifacts, but only a separately
bound verifier can produce evaluation rows. Selection is exploratory; only a
complete paired confirmation result may enter local effect memory.
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
from wmloop.geometry.mechanism_relations import build_mechanism_relation, classify_interaction
from wmloop.geometry.memory import EffectContext, EffectMemory, EffectRecord
from wmloop.storage import atomic_write, canonical_bytes, checked_path


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
    if destination.exists() or destination.is_symlink():
        raise OpenMethodStudyExecutionError("OPEN_VERIFIER_OUTPUT_EXISTS")
    files: list[dict[str, str]] = []
    for relative in implementation_files:
        pure = _relative_path(relative, "OPEN_VERIFIER_IMPLEMENTATION_PATH_INVALID")
        source = root.joinpath(*pure.parts)
        if not source.is_file() or source.is_symlink():
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
    lock_root: Path = Path("/tmp/verdiwm-gpu-leases"),
    max_parallel: int = 1,
    calibration_timeout_seconds: float = 300.0,
    gpu_wait_seconds: float = 60.0,
    lease_manager: GpuLeaseManager | None = None,
) -> dict[str, object]:
    """Run calibration, paired trials, frozen evaluation, and local settlement."""

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
    if calibration_ok:
        work = [(arm, int(seed)) for arm in study["arms"] for seed in study["paired_seeds"]]
        workers = min(max_parallel, len(work), len(indices))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="verdiwm-open-study") as pool:
            futures = {
                pool.submit(
                    _run_trial,
                    study_path=study_path,
                    study=study,
                    arm=arm,
                    method=arm_methods[str(arm["role"])],
                    seed=seed,
                    paths=paths,
                    verifier_spec=verifier_spec,
                    output_root=destination / "trials" / str(arm["role"]) / f"seed-{seed}",
                    runtime_python=runtime,
                    gpu_indices=indices,
                    lease_manager=manager,
                    gpu_wait_seconds=gpu_wait_seconds,
                ): (str(arm["role"]), seed)
                for arm, seed in work
            }
            for future in as_completed(futures):
                role, seed = futures[future]
                try:
                    trials.append(future.result())
                except BaseException as exc:
                    failure = _failed_trial(role, seed, exc)
                    trial_root = destination / "trials" / role / f"seed-{seed}"
                    trial_root.mkdir(mode=0o700, parents=True, exist_ok=True)
                    _write_json(trial_root / "trial.json", failure)
                    trials.append(failure)
    trials.sort(key=lambda row: (ROLES.index(str(row["role"])), int(row["seed"])))

    actual_gpu_hours = sum(float(row.get("gpu_seconds", 0.0)) for row in trials) / 3600.0
    budget_ok = actual_gpu_hours <= float(study["budget_gpu_hours"])

    selection = _settle_phase(study, trials, verifier_spec, phase="selection")
    confirmation = _settle_phase(study, trials, verifier_spec, phase="confirmation")
    if not budget_ok:
        selection = {"state": "abstained", "phase": "selection", "reason": "ACTUAL_GPU_BUDGET_EXCEEDED", "actual_gpu_hours": actual_gpu_hours, "budget_gpu_hours": study["budget_gpu_hours"]}
        confirmation = {"state": "abstained", "phase": "confirmation", "reason": "ACTUAL_GPU_BUDGET_EXCEEDED", "actual_gpu_hours": actual_gpu_hours, "budget_gpu_hours": study["budget_gpu_hours"]}
    cas = ContentAddressedStore(destination / "evidence")
    evidence_index = _archive_raw_files(destination, cas)
    knowledge = _write_knowledge(
        output_root=destination / "knowledge",
        study=study,
        arm_methods=arm_methods,
        settlement=confirmation,
        verifier_spec=verifier_spec,
        evidence_index=evidence_index,
    )
    state = "failed" if not calibration_ok or not budget_ok else ("completed" if confirmation["state"] == "settled" else "abstained")
    public_trials = [{key: value for key, value in row.items() if key != "_evaluation_documents"} for row in trials]
    receipt: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "verdiwm-open-method-study-execution",
        "study_id": study["study_id"],
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
    _write_json(destination / "execution.json", receipt)
    final_ref = cas.put_bytes((destination / "execution.json").read_bytes(), media_type="application/json")
    _write_json(destination / "evidence-index.json", {"raw": evidence_index, "execution_receipt": final_ref.uri})
    _verify_bound_inputs(study, paths, verifier_spec)
    _load_study(study_path)
    return receipt


def _load_study(root: Path) -> dict[str, object]:
    study = _read_json(root / "study.json")
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
    actual = {
        path.relative_to(root).as_posix(): _file_digest(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "study.json"
    }
    if dict(expected) != actual:
        raise OpenMethodStudyExecutionError("OPEN_EXECUTION_STUDY_CHANGED")
    if not study.get("paired_seeds") or len(set(study["paired_seeds"])) != len(study["paired_seeds"]):
        raise OpenMethodStudyExecutionError("OPEN_EXECUTION_SEEDS_INVALID")
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
        candidate = verifier_root / pure
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
            if isinstance(raw, list) and all(isinstance(value, str) and value for value in raw):
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
        except BaseException as exc:
            rows.append({"role": role, "method_id": arm["method_id"], "state": "failed", "error": f"{type(exc).__name__}:{exc}"})
    return rows


def _run_trial(*, study_path: Path, study: Mapping[str, object], arm: Mapping[str, object], method: Mapping[str, object], seed: int, paths: Mapping[str, Path], verifier_spec: Mapping[str, object], output_root: Path, runtime_python: Path, gpu_indices: Sequence[int], lease_manager: GpuLeaseManager, gpu_wait_seconds: float) -> dict[str, object]:
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
            "selection_split": str(paths["selection_split"]), "confirmation_split": str(paths["confirmation_split"]),
            "verifier_id": verifier_spec["verifier_id"],
        }
        mode = str(method["training"]["mode"])
        operation_count = 3 if mode == "training" else 2
        operation_timeout = float(arm["estimated_gpu_hours_per_seed"]) * 3600.0 / operation_count
        if mode == "training":
            train = run_candidate_operation(
                compilation_root=study_path / str(arm["bundle_path"]), operation="train", input_document={**common, "phase": "train"},
                output_root=output_root / "train", timeout_seconds=operation_timeout,
                runtime_python=runtime_python, gpu_devices=(lease.index,),
            )
            operations.append(train)
        for phase in PHASES:
            split_key = "selection_split" if phase == "selection" else "confirmation_split"
            infer = run_candidate_operation(
                compilation_root=study_path / str(arm["bundle_path"]), operation="infer", input_document={**common, "phase": phase, "split": str(paths[split_key]), "prior_artifacts": [str(output_root / "train" / "artifacts")] if mode == "training" else []},
                output_root=output_root / phase, timeout_seconds=operation_timeout,
                runtime_python=runtime_python, gpu_devices=(lease.index,),
            )
            operations.append(infer)
            if infer["state"] != "passed":
                break
            eval_doc = _run_frozen_verifier(verifier_spec=verifier_spec, verifier_root=paths["verifier"].parent, candidate_output=Path(str(infer["artifacts"])), input_document={**common, "phase": phase, "split": str(paths[split_key]), "artifacts": str(infer["artifacts"])}, output_root=output_root / f"verify-{phase}", runtime_python=runtime_python)
            documents.append(eval_doc)
        operation_seconds = sum(float(row.get("duration_seconds", 0.0)) for row in operations)
        within_budget = operation_seconds <= float(arm["estimated_gpu_hours_per_seed"]) * 3600.0
        state = "passed" if len(documents) == 2 and within_budget and all(row.get("state") == "passed" for row in operations) and all(doc.get("validity_gates", {}).get("verifier_process", False) for doc in documents) else "failed"
        result = {"role": role, "method_id": method["method_id"], "seed": seed, "state": state, "gpu_lease": lease.to_document(), "gpu_seconds": operation_seconds, "lease_seconds": time.monotonic() - started, "budget_state": "within_bound" if within_budget else "exceeded", "operations": operations, "evaluations": documents, "_evaluation_documents": documents}
    finally:
        lease.release()
    _write_json(output_root / "trial.json", {key: value for key, value in result.items() if key != "_evaluation_documents"})
    return result


def _run_frozen_verifier(*, verifier_spec: Mapping[str, object], verifier_root: Path, candidate_output: Path, input_document: Mapping[str, object], output_root: Path, runtime_python: Path) -> dict[str, object]:
    output_root.mkdir(mode=0o700, parents=True, exist_ok=False)
    manifest = output_root / "input.json"
    _write_json(manifest, {**input_document, "candidate_output": str(candidate_output)})
    command = list(verifier_spec["command"])
    receipt = run_local_command(command=command, cwd=verifier_root, output_root=output_root / "process", timeout_seconds=3600.0, runtime_python=runtime_python, gpu_devices=(), replacements={"candidate_root": candidate_output, "input_manifest": manifest, "output_root": output_root / "result", "world_size": 1, "rank": 0})
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


def _settle_phase(study: Mapping[str, object], trials: Sequence[Mapping[str, object]], verifier: Mapping[str, object], *, phase: str) -> dict[str, object]:
    observations: dict[tuple[str, int], Mapping[str, object]] = {}
    for trial in trials:
        for evaluation in trial.get("evaluations", []):
            if evaluation.get("phase") == phase:
                observations[(str(trial["role"]), int(trial["seed"]))] = evaluation
    seeds = [int(seed) for seed in study["paired_seeds"]]
    required = {(role, seed) for role in ROLES for seed in seeds}
    complete = required <= set(observations)
    if not complete:
        return {"state": "abstained", "phase": phase, "reason": "INCOMPLETE_PAIRED_OBSERVATIONS", "observed": len(observations), "required": len(required)}
    required_gates = set(verifier["required_validity_gates"]) | {"verifier_process", "evaluation_schema", "identity_binding"}
    if any(not all(bool(doc.get("validity_gates", {}).get(gate)) for gate in required_gates) for doc in observations.values()):
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
    return {"state": "settled", "phase": phase, "primary_metric": primary, "metrics": metric_rows, "paired_units": len(common_units), "paired_seeds": len(seeds), "exploratory": phase == "selection"}


def _write_knowledge(*, output_root: Path, study: Mapping[str, object], arm_methods: Mapping[str, Mapping[str, object]], settlement: Mapping[str, object], verifier_spec: Mapping[str, object], evidence_index: Mapping[str, str]) -> dict[str, object]:
    output_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if settlement.get("state") != "settled":
        return {"state": "abstained", "effect_record_count": 0, "relation_ref": None}
    primary = str(settlement["primary_metric"])
    metric_rows = settlement["metrics"]
    primary_row = metric_rows[primary]
    means = primary_row["means"]
    replications = int(settlement["paired_units"])
    context_spec = verifier_spec["effect_context"]
    context = EffectContext(campaign_id=str(study["study_id"]), horizons=tuple(context_spec["horizons"]), **{key: str(context_spec[key]) for key in ("backbone_family", "capability_class", "goal_schema", "outcome_schema", "chart_id", "data_regime")})
    memory = EffectMemory()
    refs = tuple(evidence_index.values()) or ("cas://unavailable",)
    records = []
    for role in ROLES[1:]:
        estimate = primary_row["effects_vs_baseline"][role]
        threshold = float(verifier_spec["metric_policy"][primary]["minimum_effect"])
        protected_ok = all(
            float(metric_rows[metric]["effects_vs_baseline"][role]["lower_bound"])
            >= -float(verifier_spec["metric_policy"][metric]["maximum_regression"])
            for metric in verifier_spec["protected_metrics"]
        )
        gates = {"frozen_verifier": True, "paired_confirmation": True, "split_disjoint": True, "protected_metrics": protected_ok}
        if float(estimate["lower_bound"]) > threshold and protected_ok and replications >= 2:
            status = "confirmed"
        elif float(estimate["upper_bound"]) < -threshold:
            status = "rejected"
        else:
            status = "null"
        record = EffectRecord(record_id=f"{study['study_id']}-{role}", primitive=str(arm_methods[role]["method_id"]), context=context, status=status, mean_effect=float(estimate["mean"]), standard_error=float(estimate["standard_error"]), lower_bound=float(estimate["lower_bound"]), goal_threshold=threshold, validity_gates=gates, replication_count=replications, evidence_refs=refs, notes=("Generated open-method confirmation result; inspect paired unit receipts before reuse.",))
        memory.add(record)
        records.append(record.to_dict())
    interaction = primary_row["interaction"]
    relation_type = classify_interaction(baseline=float(means["baseline"]), source=float(means["source_only"]), target=float(means["target_only"]), combined=float(means["combined"]), uncertainty=1.96 * float(interaction["standard_error"]))[0]
    if relation_type == "abstained":
        relation_type = "conditional_compatibility"
    relation = build_mechanism_relation(source_mechanism_id=str(arm_methods["source_only"]["method_id"]), target_mechanism_id=str(arm_methods["target_only"]["method_id"]), relation_type=relation_type, composition_operator=str(verifier_spec["composition_operator"]), baseline_effect=float(means["baseline"]), source_effect=float(means["source_only"]), target_effect=float(means["target_only"]), combined_effect=float(means["combined"]), uncertainty=1.96 * float(interaction["standard_error"]), replication_count=replications, required_ablations=["source_only", "target_only", "baseline"], evidence_refs=refs, claim_scope="target_local_confirmation", verification_state="candidate", validity_gates={"frozen_verifier": True, "paired_confirmation": True}, notes=("Relation remains candidate until policy-level scientific review.",))
    _write_json(output_root / "effect-records.json", {"records": records})
    _write_json(output_root / "mechanism-relation.json", relation)
    return {"state": "settled", "effect_record_count": len(records), "effect_records": str(output_root / "effect-records.json"), "relation_ref": str(output_root / "mechanism-relation.json")}


def _archive_raw_files(root: Path, cas: ContentAddressedStore) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name not in {"evidence-index.json"} and "/evidence/cas/" not in path.as_posix():
            ref = cas.put_bytes(path.read_bytes(), media_type="application/json" if path.suffix == ".json" else "application/octet-stream")
            result[path.relative_to(root).as_posix()] = ref.uri
    return result


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
            path = root / _relative_path(row["relative_path"], "OPEN_VERIFIER_PATH_INVALID")
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
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-split", type=Path, required=True)
    parser.add_argument("--selection-split", type=Path, required=True)
    parser.add_argument("--confirmation-split", type=Path, required=True)
    parser.add_argument("--verifier", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--max-parallel", type=int, default=1)
    args = parser.parse_args()
    receipt = execute_open_method_study(study_root=args.study, checkpoint=args.checkpoint, train_split=args.train_split, selection_split=args.selection_split, confirmation_split=args.confirmation_split, verifier=args.verifier, output_root=args.output, gpu_indices=tuple(int(value) for value in args.gpus.split(",") if value.strip()), max_parallel=args.max_parallel)
    print(json.dumps({"execution_id": receipt["execution_id"], "state": receipt["state"], "output": str(args.output)}, ensure_ascii=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
