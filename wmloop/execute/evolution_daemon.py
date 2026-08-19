"""Continuously materialize and execute immutable experiment candidate batches.

An evolution daemon is a controller around the existing diagnostic-first
pipeline.  It never edits a model checkout and never reuses a settled trial
identity.  Every iteration receives its own input lock, probe contract,
evaluator contract, candidate batch, pipeline output, and daemon state.  GPU
contention is delegated to ``verdiwm-run-daemon``; scientific failures and
budget exhaustion are persisted as terminal controller outcomes.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import signal
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from wmloop.contracts import ContractValidationError, validate_document
from wmloop.control.onboarding import compute_source_tree_revision
from wmloop.execute.autonomous_pipeline import AutonomousPipelineOptions
from wmloop.execute.pipeline_daemon import (
    PipelineDaemonOptions,
    run_pipeline_daemon,
)


class EvolutionDaemonError(RuntimeError):
    """The evolution controller could not preserve its durable contract."""


EvolutionRunner = Callable[[PipelineDaemonOptions], Mapping[str, object]]
SleepFunction = Callable[[float], None]

_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_INTERACT_NUM = re.compile(r"(setattr\(self,['\"]interact_num['\"],)\d+([,)])")
_INFERENCE_STEPS = re.compile(
    r"(setattr\(self,['\"]num_inference_steps['\"],)\d+([,)])"
)
_STRATEGIES = ("horizon_extension", "inference_budget", "control_replay")


@dataclass(frozen=True)
class EvolutionDaemonOptions:
    """Configuration for a resumable multi-iteration evolution controller."""

    repo_root: Path
    output_root: Path
    state_root: Path
    evaluator_contract: Path
    probe_contract: Path | None = None
    candidate_catalog: Path | None = None
    settlement_manifest: Path | None = None
    runtime_python: Path | None = None
    asset_bindings: tuple[tuple[str, Path], ...] = ()
    probe_imports: bool = True
    max_files: int = 20_000
    conformance_timeout_seconds: float = 30.0
    archive_db: Path | None = None
    cas_root: Path | None = None
    lock_root: Path = Path("/tmp/verdiwm-gpu-leases")
    budget_db: Path | None = None
    total_budget_gpu_hours: float = 1.0
    budget_max_trial_gpu_hours: float = 120.0
    budget_high_trial_limit: int = 2
    budget_require_high_cost_approval: bool = True
    retrieval_db: Path | None = None
    literature_query: str | None = None
    literature_max_results: int = 8
    literature_timeout_seconds: float = 10.0
    poll_seconds: float = 60.0
    max_iterations: int = 0  # zero means long-running until a stop condition.
    max_failures: int = 3
    max_no_information: int = 3
    batch_size: int = 1
    inner_max_cycles: int = 1_440
    inner_max_attempts: int = 3


def run_evolution_daemon(
    options: EvolutionDaemonOptions,
    *,
    pipeline_runner: EvolutionRunner | None = None,
    sleeper: SleepFunction = time.sleep,
) -> dict[str, object]:
    """Run iterations until a policy stop, budget stop, or explicit signal."""

    _validate_options(options)
    destination = Path(options.output_root).expanduser().resolve()
    state_root = Path(options.state_root).expanduser().resolve()
    input_document = _input_document(options)
    input_hash = _sha256(_canonical_json(input_document))
    _bind_root(state_root, input_document=input_document, input_hash=input_hash)
    _acquire_lock(state_root)
    stop_requested = False

    def request_stop(_signal: int, _frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True

    previous_handlers: dict[int, Any] = {}
    state: dict[str, object] | None = None
    runner = pipeline_runner or run_pipeline_daemon
    try:
        previous_handlers = _install_signal_handlers(request_stop)
        state = _load_state(state_root, input_hash=input_hash, options=options)
        if state.get("state") in {"completed", "blocked"} or state.get(
            "stop_reason"
        ) == "GLOBAL_BUDGET_EXHAUSTED":
            return _manifest(state)
        state["state"] = "running"
        state["stop_reason"] = None
        _write_json(state_root / "status.json", state)

        while not stop_requested:
            _verify_frozen_inputs(options, expected=input_document)
            resume_current = state.get("last_outcome") in {
                "stopped",
                "exhausted",
            } and isinstance(state.get("current_iteration_path"), str)
            iteration = int(state["iteration"]) if resume_current else int(
                state["iteration"]
            ) + 1
            if options.max_iterations and iteration > options.max_iterations:
                state["state"] = "completed"
                state["stop_reason"] = "MAX_ITERATIONS_REACHED"
                break
            if int(state["failure_count"]) >= options.max_failures:
                state["state"] = "blocked"
                state["stop_reason"] = "CONSECUTIVE_FAILURE_LIMIT"
                break
            if int(state["no_information_count"]) >= options.max_no_information:
                state["state"] = "completed"
                state["stop_reason"] = "NO_NEW_INFORMATION_LIMIT"
                break

            iteration_root = destination / "iterations" / f"iteration-{iteration:06d}"
            materialized = _materialize_iteration(
                options,
                iteration=iteration,
                iteration_root=iteration_root,
                previous_probe_signatures=_previous_signatures(state),
            )
            literature_query = _iteration_literature_query(
                options,
                strategy=str(materialized["strategy"]),
                failure_signatures=_previous_signatures(state),
            )
            pipeline_options = AutonomousPipelineOptions(
                repo_root=options.repo_root,
                output_root=iteration_root / "pipeline",
                evaluator_contract=materialized["evaluator_contract"],
                runtime_python=options.runtime_python,
                asset_bindings=options.asset_bindings,
                probe_imports=options.probe_imports,
                max_files=options.max_files,
                conformance_timeout_seconds=options.conformance_timeout_seconds,
                archive_db=options.archive_db,
                cas_root=options.cas_root,
                lock_root=options.lock_root,
                budget_db=options.budget_db,
                budget_total_gpu_hours=options.total_budget_gpu_hours,
                budget_max_trial_gpu_hours=options.budget_max_trial_gpu_hours,
                budget_high_trial_limit=options.budget_high_trial_limit,
                budget_require_high_cost_approval=(
                    options.budget_require_high_cost_approval
                ),
                probe_contract=materialized["probe_contract"],
                candidate_catalog=options.candidate_catalog,
                settlement_manifest=options.settlement_manifest,
                retrieval_db=options.retrieval_db,
                literature_query=literature_query,
                literature_max_results=options.literature_max_results,
                literature_timeout_seconds=options.literature_timeout_seconds,
            )
            daemon_options = PipelineDaemonOptions(
                pipeline=pipeline_options,
                state_root=iteration_root / "daemon",
                poll_seconds=options.poll_seconds,
                max_cycles=_effective_inner_max_cycles(
                    iteration_root, options.inner_max_cycles
                ),
                max_attempts=options.inner_max_attempts,
            )
            cycle_record: dict[str, object] = {
                "iteration": iteration,
                "started_at": _utc_now(),
                "strategy": materialized["strategy"],
                "literature_query": literature_query,
                "candidate_batch_sha256": materialized["candidate_batch_sha256"],
            }
            try:
                result = dict(runner(daemon_options))
                outcome = _record_iteration_result(state, result, iteration_root)
                cycle_record.update(outcome)
            except Exception as exc:
                state["failure_count"] = int(state["failure_count"]) + 1
                error = {"type": type(exc).__name__, "message": str(exc)[:500]}
                state["last_outcome"] = "error"
                state["last_error"] = error
                cycle_record.update({"outcome": "error", "error": error})
                if int(state["failure_count"]) >= options.max_failures:
                    state["state"] = "blocked"
            state["iteration"] = iteration
            cycle_record["finished_at"] = _utc_now()
            _write_json(
                state_root / "iterations" / f"iteration-{iteration:06d}.json",
                cycle_record,
            )
            _write_json(state_root / "status.json", state)
            if state.get("state") in {
                "blocked",
                "completed",
                "exhausted",
                "stopped",
            }:
                break
            if not stop_requested:
                sleeper(options.poll_seconds)

        if stop_requested and state.get("state") == "running":
            state["state"] = "stopped"
            state["stop_reason"] = "SIGNAL_RECEIVED"
        _write_json(state_root / "status.json", state)
        return _manifest(state)
    except Exception as exc:
        if state is not None:
            state["state"] = "interrupted"
            state["last_error"] = {"type": type(exc).__name__, "message": str(exc)[:500]}
            _write_json(state_root / "status.json", state)
        raise
    finally:
        _restore_signal_handlers(previous_handlers)
        _release_lock(state_root)


def _record_iteration_result(
    state: dict[str, object], result: Mapping[str, object], iteration_root: Path
) -> dict[str, object]:
    result_state = str(result.get("state") or "")
    verdict = result.get("pipeline_verdict")
    if result_state == "completed" and verdict == "PASS":
        state["deferral_count"] = int(state["deferral_count"]) + _nonnegative_int(
            result.get("deferral_count")
        )
        state["failure_count"] = 0
        state["last_outcome"] = "completed"
        state["last_error"] = None
        signatures = _pipeline_signatures(
            iteration_root / "pipeline" / "pipeline-manifest.json"
        )
        signature_hash = _sha256(_canonical_json(signatures))
        if state.get("last_signature_hash") == signature_hash:
            state["no_information_count"] = int(state["no_information_count"]) + 1
        else:
            state["no_information_count"] = 0
        state["last_signature_hash"] = signature_hash
        state["last_signatures"] = signatures
        state["settled_iterations"] = int(state["settled_iterations"]) + 1
        state["current_iteration_path"] = str(iteration_root)
        return {
            "outcome": "completed",
            "pipeline_verdict": "PASS",
            "failure_signatures": signatures,
            "iteration_path": str(iteration_root),
        }
    if result_state in {"exhausted", "stopped", "interrupted"}:
        state["last_outcome"] = result_state
        state["state"] = "exhausted" if result_state == "exhausted" else "stopped"
        state["current_iteration_path"] = str(iteration_root)
        return {"outcome": result_state, "pipeline_verdict": verdict}
    if "GLOBAL_BUDGET_EXHAUSTED" in json.dumps(result, ensure_ascii=True):
        state["deferral_count"] = int(state["deferral_count"]) + _nonnegative_int(
            result.get("deferral_count")
        )
        state["state"] = "exhausted"
        state["last_outcome"] = "budget_exhausted"
        state["stop_reason"] = "GLOBAL_BUDGET_EXHAUSTED"
        state["current_iteration_path"] = str(iteration_root)
        return {"outcome": "budget_exhausted", "pipeline_verdict": verdict}
    state["failure_count"] = int(state["failure_count"]) + 1
    state["deferral_count"] = int(state["deferral_count"]) + _nonnegative_int(
        result.get("deferral_count")
    )
    state["last_outcome"] = "blocked"
    state["last_error"] = {
        "type": "PIPELINE_BLOCKED",
        "message": str(
            result.get("blocked_stage")
            or result.get("last_error")
            or "pipeline blocked"
        ),
    }
    state["current_iteration_path"] = str(iteration_root)
    return {
        "outcome": "blocked",
        "pipeline_verdict": verdict,
        "blocked_stage": result.get("blocked_stage"),
    }


def _materialize_iteration(
    options: EvolutionDaemonOptions,
    *,
    iteration: int,
    iteration_root: Path,
    previous_probe_signatures: Sequence[str],
) -> dict[str, object]:
    input_root = iteration_root / "inputs"
    lock_path = input_root / "input.lock.json"
    if input_root.exists() or input_root.is_symlink():
        if input_root.is_symlink() or not lock_path.is_file():
            raise EvolutionDaemonError("EVOLUTION_ITERATION_INPUT_INVALID")
        lock = _load_json(lock_path)
        if lock.get("iteration") != iteration:
            raise EvolutionDaemonError("EVOLUTION_ITERATION_INPUT_MISMATCH")
        return lock["materialized"]  # type: ignore[return-value]

    evaluator = _load_json(Path(options.evaluator_contract).resolve(strict=True))
    template_value = evaluator.get("scheduler_template")
    if not isinstance(template_value, str) or not template_value:
        raise EvolutionDaemonError("EVOLUTION_EVALUATOR_TEMPLATE_INVALID")
    template_path = Path(template_value)
    if not template_path.is_absolute():
        template_path = Path(options.evaluator_contract).resolve().parent / template_path
    template = _load_json(template_path.resolve(strict=True))
    probe = (
        _load_json(Path(options.probe_contract).resolve(strict=True))
        if options.probe_contract is not None
        else None
    )
    strategy = _STRATEGIES[(iteration - 1) % len(_STRATEGIES)]
    suffix = f"iteration-{iteration:06d}"
    campaign_id = _new_id(str(template.get("campaign_id", "verdiwm-campaign")), suffix)
    probe_id = (
        _new_id(str(probe.get("probe_id", "probe")) + "-" + suffix, "")
        if probe
        else None
    )
    if probe is not None:
        probe["probe_id"] = probe_id
    batch = copy.deepcopy(template)
    _require_confirmation_ladder(batch)
    batch["campaign_id"] = campaign_id
    batch["max_selected_candidates"] = min(
        int(batch["max_selected_candidates"]), options.batch_size
    )
    batch["selection_reason"] = (
        f"{batch['selection_reason']} Evolution {suffix} uses controlled "
        f"strategy {strategy}."
    )
    batch["objective"] = (
        f"{batch['objective']} Evolution {suffix} evaluates the next "
        "immutable candidate batch."
    )
    for index, candidate in enumerate(batch.get("candidates", []), start=1):
        if not isinstance(candidate, dict):
            continue
        candidate["candidate_id"] = _new_id(
            str(candidate.get("candidate_id", f"candidate-{index}")), suffix
        )
        candidate["selection_reason"] = (
            f"{candidate['selection_reason']} Strategy: {strategy}; "
            f"immutable iteration {iteration}."
        )
        keys = candidate.setdefault("retrieval_keys", {})
        if isinstance(keys, dict) and previous_probe_signatures:
            declared = list(keys.get("failure_signatures", []))
            keys["failure_signatures"] = sorted(
                {
                    str(value)
                    for value in [*declared, *previous_probe_signatures]
                }
            )
        for stage in candidate.get("stages", []):
            if not isinstance(stage, dict):
                continue
            stage_name = str(stage.get("stage") or "")
            environment = stage.setdefault("environment", {})
            if isinstance(environment, dict) and probe_id is not None:
                environment["VERDIWM_PROBE_ID"] = probe_id
            if strategy == "horizon_extension":
                stage_floor = {"screen": 2, "gate": 4, "confirm": 8}[stage_name]
                stage["command"] = _replace_command_value(
                    stage.get("command", []),
                    _INTERACT_NUM,
                    min(12, stage_floor + iteration - 1),
                )
            elif strategy == "inference_budget":
                stage_floor = {"screen": 2, "gate": 4, "confirm": 6}[stage_name]
                stage["command"] = _replace_command_value(
                    stage.get("command", []),
                    _INFERENCE_STEPS,
                    min(12, stage_floor + 2 * (iteration - 1)),
                )
    validate_document("auto_experiment_candidate_batch", batch)
    if probe is not None:
        validate_document("diagnostic_probe_contract", probe)

    temporary = (
        iteration_root.parent
        / f".{iteration_root.name}.{uuid.uuid4().hex}.tmp"
    )
    (temporary / "inputs").mkdir(mode=0o700, parents=True)
    generated_evaluator = copy.deepcopy(evaluator)
    generated_evaluator["evaluator_id"] = _new_id(
        str(evaluator.get("evaluator_id", "evaluator")), suffix
    )
    generated_evaluator["scheduler_template"] = "candidate-batch.json"
    _write_json(temporary / "inputs" / "evaluator.json", generated_evaluator)
    _write_json(temporary / "inputs" / "candidate-batch.json", batch)
    if probe is not None:
        _write_json(temporary / "inputs" / "probe.json", probe)
    materialized = {
        "iteration": iteration,
        "strategy": strategy,
        "campaign_id": campaign_id,
        "probe_id": probe_id,
        "evaluator_contract": str(iteration_root / "inputs" / "evaluator.json"),
        "probe_contract": (
            str(iteration_root / "inputs" / "probe.json")
            if probe is not None
            else None
        ),
        "candidate_batch": str(iteration_root / "inputs" / "candidate-batch.json"),
        "candidate_batch_sha256": _sha256(_canonical_json(batch)),
    }
    materialized["input_hash"] = _sha256(_canonical_json(materialized))
    _write_json(
        temporary / "inputs" / "input.lock.json",
        {
            "schema_version": 1,
            "artifact_type": "verdiwm-evolution-iteration-input-lock",
            "iteration": iteration,
            "materialized": materialized,
        },
    )
    iteration_root.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.replace(temporary, iteration_root)
    return materialized


def _require_confirmation_ladder(batch: Mapping[str, object]) -> None:
    """Reject unattended evolution that can never reach formal evidence."""

    candidates = batch.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise EvolutionDaemonError("EVOLUTION_CANDIDATES_INVALID")
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise EvolutionDaemonError("EVOLUTION_CANDIDATES_INVALID")
        stages = candidate.get("stages")
        if not isinstance(stages, list):
            raise EvolutionDaemonError("EVOLUTION_CONFIRMATION_LADDER_REQUIRED")
        names = [
            str(stage.get("stage"))
            for stage in stages
            if isinstance(stage, Mapping)
        ]
        if names != ["screen", "gate", "confirm"]:
            raise EvolutionDaemonError("EVOLUTION_CONFIRMATION_LADDER_REQUIRED")


def _iteration_literature_query(
    options: EvolutionDaemonOptions,
    *,
    strategy: str,
    failure_signatures: Sequence[str],
) -> str:
    """Create a fresh external-search view for every immutable iteration."""

    base = (options.literature_query or "world model optimization").strip()
    terms = [base, strategy.replace("_", " ")]
    terms.extend(
        signature.replace("_", " ")
        for signature in failure_signatures
        if signature.strip()
    )
    return " ".join(dict.fromkeys(term for term in terms if term))[:512]


def _replace_command_value(
    command: object, pattern: re.Pattern[str], value: int
) -> list[object]:
    if not isinstance(command, list):
        return command  # type: ignore[return-value]
    return [
        pattern.sub(
            lambda match: f"{match.group(1)}{value}{match.group(2)}", token
        )
        if isinstance(token, str)
        else token
        for token in command
    ]


def _previous_signatures(state: Mapping[str, object]) -> list[str]:
    values = state.get("last_signatures", [])
    if not isinstance(values, list):
        return []
    return sorted(
        {str(value) for value in values if isinstance(value, str) and value}
    )


def _pipeline_signatures(path: Path) -> list[str]:
    if not path.is_file() or path.is_symlink():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    probe = payload.get("diagnostic_probe") if isinstance(payload, dict) else None
    values = probe.get("failure_signatures", []) if isinstance(probe, dict) else []
    if not isinstance(values, list):
        return []
    return sorted(
        {str(value) for value in values if isinstance(value, str) and value}
    )


def _effective_inner_max_cycles(iteration_root: Path, requested: int) -> int:
    status_path = iteration_root / "daemon" / "status.json"
    if not status_path.is_file() or status_path.is_symlink():
        return requested
    status = _load_json(status_path)
    cycle = _nonnegative_int(status.get("cycle"))
    if status.get("state") == "exhausted" and cycle >= requested:
        return cycle + requested
    return max(requested, cycle + 1)


def _validate_options(options: EvolutionDaemonOptions) -> None:
    repo = Path(options.repo_root).expanduser().resolve()
    evaluator = Path(options.evaluator_contract).expanduser().resolve()
    state = Path(options.state_root).expanduser().resolve()
    output = Path(options.output_root).expanduser().resolve()
    if not repo.is_dir() or repo.is_symlink():
        raise EvolutionDaemonError("EVOLUTION_REPOSITORY_INVALID")
    if not evaluator.is_file() or evaluator.is_symlink():
        raise EvolutionDaemonError("EVOLUTION_EVALUATOR_INVALID")
    if options.probe_contract is not None:
        probe = Path(options.probe_contract).expanduser().resolve()
        if not probe.is_file() or probe.is_symlink():
            raise EvolutionDaemonError("EVOLUTION_PROBE_CONTRACT_INVALID")
    if options.candidate_catalog is not None:
        catalog = Path(options.candidate_catalog).expanduser().resolve()
        if not catalog.is_file() or catalog.is_symlink():
            raise EvolutionDaemonError("EVOLUTION_CANDIDATE_CATALOG_INVALID")
    if options.settlement_manifest is not None:
        settlement = Path(options.settlement_manifest).expanduser().resolve()
        if not settlement.is_file() or settlement.is_symlink():
            raise EvolutionDaemonError("EVOLUTION_SETTLEMENT_MANIFEST_INVALID")
    if state == output or state in output.parents or output in state.parents:
        raise EvolutionDaemonError("EVOLUTION_STATE_OUTPUT_OVERLAP")
    if (
        repo in state.parents
        or state in repo.parents
        or repo in output.parents
        or output in repo.parents
    ):
        raise EvolutionDaemonError("EVOLUTION_REPOSITORY_OUTPUT_OVERLAP")
    if (
        not math.isfinite(options.total_budget_gpu_hours)
        or options.total_budget_gpu_hours <= 0
    ):
        raise EvolutionDaemonError("EVOLUTION_BUDGET_INVALID")
    if (
        options.max_iterations < 0
        or options.max_failures < 1
        or options.max_no_information < 1
        or options.batch_size < 1
        or options.inner_max_cycles < 1
        or options.inner_max_attempts < 1
        or not math.isfinite(options.budget_max_trial_gpu_hours)
        or options.budget_max_trial_gpu_hours <= 0
        or options.budget_high_trial_limit < 0
    ):
        raise EvolutionDaemonError("EVOLUTION_ARGUMENT_INVALID")


def _input_document(options: EvolutionDaemonOptions) -> dict[str, object]:
    def digest(path: Path) -> str:
        return _sha256(path.resolve(strict=True).read_bytes())

    evaluator_path = Path(options.evaluator_contract).resolve(strict=True)
    evaluator = _load_json(evaluator_path)
    template_value = evaluator.get("scheduler_template")
    if not isinstance(template_value, str) or not template_value:
        raise EvolutionDaemonError("EVOLUTION_EVALUATOR_TEMPLATE_INVALID")
    template_path = Path(template_value)
    if not template_path.is_absolute():
        template_path = evaluator_path.parent / template_path
    template_path = template_path.resolve(strict=True)
    return {
        "schema_version": 1,
        "artifact_type": "verdiwm-evolution-daemon-input",
        "repo_root": str(Path(options.repo_root).resolve()),
        "source_tree_revision": compute_source_tree_revision(
            Path(options.repo_root).resolve(), max_files=options.max_files
        ),
        "output_root": str(Path(options.output_root).resolve()),
        "evaluator_contract": str(evaluator_path),
        "evaluator_sha256": digest(evaluator_path),
        "candidate_template": str(template_path),
        "candidate_template_sha256": digest(template_path),
        "probe_contract": (
            str(Path(options.probe_contract).resolve())
            if options.probe_contract is not None
            else None
        ),
        "candidate_catalog": (
            str(Path(options.candidate_catalog).resolve())
            if options.candidate_catalog is not None
            else None
        ),
        "candidate_catalog_sha256": (
            digest(Path(options.candidate_catalog))
            if options.candidate_catalog is not None
            else None
        ),
        "settlement_manifest": (
            str(Path(options.settlement_manifest).resolve())
            if options.settlement_manifest is not None
            else None
        ),
        "settlement_manifest_sha256": (
            digest(Path(options.settlement_manifest))
            if options.settlement_manifest is not None
            else None
        ),
        "probe_contract_sha256": (
            digest(Path(options.probe_contract))
            if options.probe_contract is not None
            else None
        ),
        "asset_bindings": sorted(
            [
                {"parameter": key, "path": str(Path(value).resolve())}
                for key, value in options.asset_bindings
            ],
            key=lambda row: row["parameter"],
        ),
        "archive_db": (
            str(Path(options.archive_db).resolve())
            if options.archive_db is not None
            else None
        ),
        "cas_root": (
            str(Path(options.cas_root).resolve())
            if options.cas_root is not None
            else None
        ),
        "budget_db": (
            str(Path(options.budget_db).resolve())
            if options.budget_db is not None
            else None
        ),
        "retrieval_db": (
            str(Path(options.retrieval_db).resolve())
            if options.retrieval_db is not None
            else None
        ),
        "runtime_python": (
            str(Path(options.runtime_python).resolve())
            if options.runtime_python is not None
            else None
        ),
        "probe_imports": options.probe_imports,
        "max_files": options.max_files,
        "conformance_timeout_seconds": options.conformance_timeout_seconds,
        "lock_root": str(Path(options.lock_root).resolve()),
        "literature_query": options.literature_query,
        "literature_max_results": options.literature_max_results,
        "literature_timeout_seconds": options.literature_timeout_seconds,
        "total_budget_gpu_hours": options.total_budget_gpu_hours,
        **_nondefault_resource_policy(options),
        "poll_seconds": options.poll_seconds,
        "max_iterations": options.max_iterations,
        "max_failures": options.max_failures,
        "max_no_information": options.max_no_information,
        "batch_size": options.batch_size,
        "inner_max_attempts": options.inner_max_attempts,
    }


def _nondefault_resource_policy(
    options: EvolutionDaemonOptions,
) -> dict[str, object]:
    if (
        options.budget_max_trial_gpu_hours == 120.0
        and options.budget_high_trial_limit == 2
        and options.budget_require_high_cost_approval
    ):
        return {}
    return {
        "resource_policy": {
            "max_trial_gpu_hours": options.budget_max_trial_gpu_hours,
            "high_trial_limit": options.budget_high_trial_limit,
            "require_high_cost_approval": (
                options.budget_require_high_cost_approval
            ),
        }
    }


def _verify_frozen_inputs(
    options: EvolutionDaemonOptions, *, expected: Mapping[str, object]
) -> None:
    current = _input_document(options)
    if current != dict(expected):
        raise EvolutionDaemonError("EVOLUTION_FROZEN_INPUT_DRIFT")


def _bind_root(root: Path, *, input_document: Mapping[str, object], input_hash: str) -> None:
    lock_path = root / "input.lock.json"
    if root.exists() or root.is_symlink():
        if root.is_symlink() or not root.is_dir():
            raise EvolutionDaemonError("EVOLUTION_STATE_ROOT_INVALID")
        lock = _load_json(lock_path)
        if lock.get("input_hash") != input_hash or lock.get("input") != dict(
            input_document
        ):
            raise EvolutionDaemonError("EVOLUTION_INPUT_MISMATCH")
        return
    (root / "iterations").mkdir(mode=0o700, parents=True)
    _write_json(
        lock_path,
        {
            "schema_version": 1,
            "artifact_type": "verdiwm-evolution-daemon-input-lock",
            "input_hash": input_hash,
            "input": dict(input_document),
        },
    )


def _load_state(
    root: Path, *, input_hash: str, options: EvolutionDaemonOptions
) -> dict[str, object]:
    path = root / "status.json"
    if path.is_file() and not path.is_symlink():
        state = _load_json(path)
        if state.get("input_hash") != input_hash:
            raise EvolutionDaemonError("EVOLUTION_STATUS_INPUT_MISMATCH")
        return state
    state = {
        "schema_version": 1,
        "artifact_type": "verdiwm-evolution-daemon-manifest",
        "state": "running",
        "input_hash": input_hash,
        "output_root": str(Path(options.output_root).resolve()),
        "state_root": str(Path(options.state_root).resolve()),
        "iteration": 0,
        "max_iterations": options.max_iterations,
        "poll_seconds": options.poll_seconds,
        "total_budget_gpu_hours": options.total_budget_gpu_hours,
        "resource_policy": {
            "max_trial_gpu_hours": options.budget_max_trial_gpu_hours,
            "high_trial_limit": options.budget_high_trial_limit,
            "require_high_cost_approval": (
                options.budget_require_high_cost_approval
            ),
        },
        "max_failures": options.max_failures,
        "max_no_information": options.max_no_information,
        "failure_count": 0,
        "no_information_count": 0,
        "settled_iterations": 0,
        "deferral_count": 0,
        "last_outcome": None,
        "last_error": None,
        "last_signatures": [],
        "last_signature_hash": None,
        "current_iteration_path": None,
        "stop_reason": None,
        "claim_boundary": (
            "Iterations are exploratory and immutable; a PASS is execution "
            "evidence, not automatic model-quality promotion."
        ),
    }
    _write_json(path, state)
    return state


def _manifest(state: Mapping[str, object]) -> dict[str, object]:
    result = dict(state)
    try:
        validate_document("pipeline_evolution_manifest", result)
    except ContractValidationError as exc:
        raise EvolutionDaemonError("EVOLUTION_MANIFEST_INVALID") from exc
    return result


def _new_id(base: str, suffix: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", base).strip("-.") or "candidate"
    result = f"{value}-{suffix}" if suffix else value
    if len(result) > 128:
        result = result[:128].rstrip("-.")
    if not _ID.fullmatch(result):
        raise EvolutionDaemonError("EVOLUTION_ID_INVALID")
    return result


def _nonnegative_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


def _load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvolutionDaemonError("EVOLUTION_JSON_INVALID") from exc
    if not isinstance(value, dict):
        raise EvolutionDaemonError("EVOLUTION_JSON_OBJECT_REQUIRED")
    return value


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(_canonical_json(value))
        os.replace(temporary, path)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _acquire_lock(root: Path) -> None:
    path = root / "daemon.lock"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        existing = _load_json(path)
        pid = existing.get("pid")
        if isinstance(pid, int) and pid > 0 and _pid_alive(pid):
            raise EvolutionDaemonError("EVOLUTION_ALREADY_RUNNING")
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        try:
            descriptor = os.open(
                path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
            )
        except FileExistsError as exc:
            raise EvolutionDaemonError("EVOLUTION_ALREADY_RUNNING") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {"pid": os.getpid(), "created_at": _utc_now()}, sort_keys=True
            )
            + "\n"
        )


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _release_lock(root: Path) -> None:
    path = root / "daemon.lock"
    try:
        payload = _load_json(path)
        if payload.get("pid") == os.getpid():
            path.unlink()
    except (OSError, EvolutionDaemonError):
        return


def _install_signal_handlers(handler: Callable[[int, object], None]) -> dict[int, Any]:
    previous: dict[int, Any] = {}
    for signum in (signal.SIGTERM, signal.SIGINT):
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, handler)
    return previous


def _restore_signal_handlers(previous: Mapping[int, Any]) -> None:
    for signum, handler in previous.items():
        signal.signal(signum, handler)


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--evaluator-contract", type=Path, required=True)
    parser.add_argument("--probe-contract", type=Path)
    parser.add_argument("--candidate-catalog", type=Path)
    parser.add_argument("--settlement-manifest", type=Path)
    parser.add_argument("--runtime-python", type=Path)
    parser.add_argument("--asset", action="append", default=[], metavar="PARAM=PATH")
    parser.add_argument("--no-import-probe", action="store_true")
    parser.add_argument("--archive-db", type=Path)
    parser.add_argument("--cas-root", type=Path)
    parser.add_argument("--lock-root", type=Path, default=Path("/tmp/verdiwm-gpu-leases"))
    parser.add_argument("--budget-db", type=Path)
    parser.add_argument("--total-budget-gpu-hours", type=float, required=True)
    parser.add_argument("--budget-max-trial-gpu-hours", type=float, default=120.0)
    parser.add_argument("--budget-high-trial-limit", type=int, default=2)
    parser.add_argument("--auto-approve-high-cost", action="store_true")
    parser.add_argument("--retrieval-db", type=Path)
    parser.add_argument("--literature-query")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--max-iterations", type=int, default=0)
    parser.add_argument("--max-failures", type=int, default=3)
    parser.add_argument("--max-no-information", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--inner-max-cycles", type=int, default=1_440)
    parser.add_argument("--inner-max-attempts", type=int, default=3)
    args = parser.parse_args(argv)
    try:
        manifest = run_evolution_daemon(
            EvolutionDaemonOptions(
                repo_root=args.repo_root,
                output_root=args.output_root,
                state_root=args.state_root,
                evaluator_contract=args.evaluator_contract,
                probe_contract=args.probe_contract,
                candidate_catalog=args.candidate_catalog,
                settlement_manifest=args.settlement_manifest,
                runtime_python=args.runtime_python,
                asset_bindings=tuple(_parse_asset(value) for value in args.asset),
                probe_imports=not args.no_import_probe,
                archive_db=args.archive_db,
                cas_root=args.cas_root,
                lock_root=args.lock_root,
                budget_db=args.budget_db,
                total_budget_gpu_hours=args.total_budget_gpu_hours,
                budget_max_trial_gpu_hours=args.budget_max_trial_gpu_hours,
                budget_high_trial_limit=args.budget_high_trial_limit,
                budget_require_high_cost_approval=(
                    not args.auto_approve_high_cost
                ),
                retrieval_db=args.retrieval_db,
                literature_query=args.literature_query,
                poll_seconds=args.poll_seconds,
                max_iterations=args.max_iterations,
                max_failures=args.max_failures,
                max_no_information=args.max_no_information,
                batch_size=args.batch_size,
                inner_max_cycles=args.inner_max_cycles,
                inner_max_attempts=args.inner_max_attempts,
            )
        )
    except EvolutionDaemonError as exc:
        print(str(exc), file=__import__("sys").stderr)
        return 2
    print(json.dumps(manifest, ensure_ascii=True, sort_keys=True))
    return 0


def _parse_asset(value: str) -> tuple[str, Path]:
    parameter, separator, path = value.partition("=")
    if not separator or not parameter.strip() or not path.strip():
        raise EvolutionDaemonError("EVOLUTION_ASSET_ARGUMENT_INVALID")
    return parameter.strip(), Path(path.strip())


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
