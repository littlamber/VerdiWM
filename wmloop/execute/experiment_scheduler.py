"""Deterministic candidate selection and progressive-fidelity execution.

The scheduler owns admission order, not scientific claims. It selects a
bounded set of pre-registered candidates using a transparent utility score,
then runs each candidate through ``screen -> gate -> confirm`` only when the
previous stage returns ``PASS``. The generic auto-experiment runner remains
the sole process/GPU/receipt boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from wmloop.contracts import ContractValidationError, validate_document
from wmloop.execute.auto_experiment import (
    AutoExperimentError,
    _validate_plan_semantics,
    _write_json_atomic,
    run_auto_experiment,
)


class ExperimentSchedulerError(RuntimeError):
    """A candidate batch or progressive-fidelity transition is invalid."""


_STAGE_ORDER = {"screen": 0, "gate": 1, "confirm": 2}
_SCORE_FIELDS = ("expected_gain", "uncertainty", "information_gain", "novelty")
_SCORE_WEIGHTS = (
    "expected_gain_weight",
    "uncertainty_weight",
    "information_gain_weight",
    "novelty_weight",
    "cost_weight",
)


def plan_candidate_batch(
    *,
    batch_path: Path,
    output_root: Path,
    workspace_root: Path,
) -> dict[str, object]:
    """Validate, rank, budget-truncate, and materialize a candidate queue."""

    batch_source = Path(batch_path).resolve(strict=True)
    workspace = Path(workspace_root).resolve(strict=True)
    batch = _load_batch(batch_source, workspace_root=workspace)
    batch_sha256 = _sha256_bytes(batch_source.read_bytes())
    destination = Path(output_root).resolve()
    lock_path = destination / "batch.lock.json"
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or not lock_path.is_file():
            raise ExperimentSchedulerError("EXPERIMENT_SCHEDULER_OUTPUT_ROOT_UNBOUND")
        lock = _load_json_object(lock_path, "EXPERIMENT_SCHEDULER_LOCK_INVALID")
        if lock.get("batch_sha256") != batch_sha256:
            raise ExperimentSchedulerError("EXPERIMENT_SCHEDULER_BATCH_MISMATCH")
        return _load_json_object(destination / "queue.json", "EXPERIMENT_SCHEDULER_QUEUE_INVALID")

    destination.mkdir(mode=0o700, parents=True)
    (destination / "plans").mkdir(mode=0o700)
    ranked = _rank_candidates(batch)
    selected, deferred = _budget_select(
        ranked,
        total_budget_gpu_hours=float(batch["total_budget_gpu_hours"]),
        max_selected_candidates=int(batch["max_selected_candidates"]),
    )
    selected_records: list[dict[str, object]] = []
    for row in selected:
        candidate = row["candidate"]
        stage_records = []
        for stage in candidate["stages"]:
            plan = _stage_plan(batch=batch, candidate=candidate, stage=stage)
            filename = f"{plan['trial_id']}.json"
            plan_path = destination / "plans" / filename
            _write_json_atomic(plan_path, plan)
            stage_records.append(
                {
                    "stage": stage["stage"],
                    "plan_path": f"plans/{filename}",
                    "plan_sha256": _sha256_bytes(plan_path.read_bytes()),
                    "estimated_gpu_hours": float(stage["estimated_gpu_hours"]),
                }
            )
        selected_records.append(
            {
                "candidate_id": candidate["candidate_id"],
                "rank": row["rank"],
                "score": row["score"],
                "screen_gpu_hours": row["screen_gpu_hours"],
                "max_ladder_gpu_hours": sum(float(stage["estimated_gpu_hours"]) for stage in candidate["stages"]),
                "stages": stage_records,
            }
        )
    queue = {
        "schema_version": 1,
        "artifact_type": "verdiwm-auto-experiment-queue",
        "state": "ready",
        "campaign_id": batch["campaign_id"],
        "batch_sha256": batch_sha256,
        "objective": batch["objective"],
        "selection_reason": batch["selection_reason"],
        "falsification_criterion": batch["falsification_criterion"],
        "total_budget_gpu_hours": float(batch["total_budget_gpu_hours"]),
        "scoring": dict(batch["scoring"]),
        "ranked_candidate_count": len(ranked),
        "selected": selected_records,
        "deferred": deferred,
        "claim_boundary": "Queue order is an exploratory resource decision. It is not evidence of model quality or an optimization-memory update.",
    }
    _write_json_atomic(
        lock_path,
        {
            "schema_version": 1,
            "artifact_type": "verdiwm-auto-experiment-batch-lock",
            "campaign_id": batch["campaign_id"],
            "batch_sha256": batch_sha256,
        },
    )
    _write_json_atomic(destination / "queue.json", queue)
    _write_json_atomic(destination / "manifest.json", _queue_manifest(queue, destination))
    _write_markdown(destination / "queue.md", queue)
    return queue


def run_selected_queue(
    *,
    queue_path: Path,
    workspace_root: Path,
    archive_db: Path,
    cas_root: Path,
    lock_root: Path,
    budget_db: Path | None = None,
) -> dict[str, object]:
    """Run selected candidates sequentially and promote only on ``PASS``."""

    queue_source = Path(queue_path).resolve(strict=True)
    queue = _load_json_object(queue_source, "EXPERIMENT_SCHEDULER_QUEUE_INVALID")
    if queue.get("artifact_type") != "verdiwm-auto-experiment-queue" or queue.get("state") != "ready":
        raise ExperimentSchedulerError("EXPERIMENT_SCHEDULER_QUEUE_NOT_READY")
    root = queue_source.parent
    execution_path = root / "execution.json"
    execution = _load_optional_json(execution_path)
    shared_budget = Path(budget_db).resolve() if budget_db is not None else root / "budget.db"
    if execution and execution.get("budget_db") != str(shared_budget):
        raise ExperimentSchedulerError("EXPERIMENT_SCHEDULER_BUDGET_PATH_MISMATCH")
    results = execution.get("results", {}) if isinstance(execution.get("results"), Mapping) else {}
    results = dict(results)
    candidate_states: dict[str, str] = {}
    for selected in queue["selected"]:
        candidate_id = str(selected["candidate_id"])
        candidate_state = "completed"
        for stage_record in selected["stages"]:
            stage = str(stage_record["stage"])
            result_key = f"{candidate_id}:{stage}"
            previous = results.get(result_key)
            if isinstance(previous, Mapping) and previous.get("state") in {"completed", "blocked"}:
                if previous.get("state") == "blocked":
                    candidate_state = "blocked"
                    break
                continue
            plan_path = _resolve_inside(root, str(stage_record["plan_path"]), "EXPERIMENT_SCHEDULER_PLAN_PATH_INVALID")
            if _sha256_bytes(plan_path.read_bytes()) != stage_record.get("plan_sha256"):
                raise ExperimentSchedulerError("EXPERIMENT_SCHEDULER_PLAN_HASH_MISMATCH")
            run_root = root / "runs" / candidate_id / stage
            try:
                manifest = run_auto_experiment(
                    plan_path=plan_path,
                    output_root=run_root,
                    workspace_root=workspace_root,
                    archive_db=archive_db,
                    cas_root=cas_root,
                    lock_root=lock_root,
                    budget_db=shared_budget,
                )
            except Exception as exc:
                results[result_key] = {
                    "state": "error",
                    "error": {"type": type(exc).__name__, "message": str(exc)[:500]},
                }
                _write_json_atomic(execution_path, _execution_document(queue, results, shared_budget))
                raise
            verdict = manifest.get("verdict")
            passed = verdict == "PASS"
            results[result_key] = {
                "state": "completed" if passed else "blocked",
                "stage": stage,
                "verdict": verdict,
                "evidence_level": manifest.get("evidence_level"),
                "manifest_path": str(run_root / "manifest.json"),
            }
            _write_json_atomic(execution_path, _execution_document(queue, results, shared_budget))
            if not passed:
                candidate_state = "blocked"
                break
        candidate_states[candidate_id] = candidate_state
    document = _execution_document(queue, results, shared_budget)
    document["candidate_states"] = candidate_states
    _write_json_atomic(execution_path, document)
    return document


def _load_batch(path: Path, *, workspace_root: Path) -> dict[str, object]:
    try:
        batch = _load_json_object(path, "EXPERIMENT_SCHEDULER_BATCH_INVALID")
        validate_document("auto_experiment_candidate_batch", batch, root=workspace_root)
    except (ContractValidationError, AutoExperimentError) as exc:
        raise ExperimentSchedulerError(f"EXPERIMENT_SCHEDULER_BATCH_INVALID:{exc}") from exc
    _validate_batch_semantics(batch, workspace_root=workspace_root)
    return batch


def _validate_batch_semantics(batch: Mapping[str, object], *, workspace_root: Path) -> None:
    if not math.isfinite(float(batch["total_budget_gpu_hours"])):
        raise ExperimentSchedulerError("EXPERIMENT_SCHEDULER_BUDGET_INVALID")
    scoring = batch["scoring"]
    if not isinstance(scoring, Mapping) or not any(float(scoring[key]) > 0 for key in _SCORE_WEIGHTS):
        raise ExperimentSchedulerError("EXPERIMENT_SCHEDULER_SCORING_ZERO")
    candidates = batch["candidates"]
    if not isinstance(candidates, list):
        raise ExperimentSchedulerError("EXPERIMENT_SCHEDULER_CANDIDATES_INVALID")
    ids: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise ExperimentSchedulerError("EXPERIMENT_SCHEDULER_CANDIDATE_INVALID")
        candidate_id = str(candidate["candidate_id"])
        if candidate_id in ids:
            raise ExperimentSchedulerError("EXPERIMENT_SCHEDULER_CANDIDATE_DUPLICATE")
        ids.add(candidate_id)
        stages = candidate["stages"]
        if not isinstance(stages, list):
            raise ExperimentSchedulerError("EXPERIMENT_SCHEDULER_STAGES_INVALID")
        names = [str(stage["stage"]) for stage in stages if isinstance(stage, Mapping)]
        if names != sorted(names, key=lambda name: _STAGE_ORDER.get(name, 99)) or names[0:1] != ["screen"]:
            raise ExperimentSchedulerError("EXPERIMENT_SCHEDULER_STAGE_ORDER_INVALID")
        if len(set(names)) != len(names) or names not in (["screen"], ["screen", "gate"], ["screen", "gate", "confirm"]):
            raise ExperimentSchedulerError("EXPERIMENT_SCHEDULER_STAGE_LADDER_INVALID")
        if sum(float(stage["estimated_gpu_hours"]) for stage in stages) > float(batch["total_budget_gpu_hours"]):
            raise ExperimentSchedulerError("EXPERIMENT_SCHEDULER_CANDIDATE_LADDER_EXCEEDS_BUDGET")
        for stage in stages:
            if not isinstance(stage, Mapping):
                raise ExperimentSchedulerError("EXPERIMENT_SCHEDULER_STAGE_INVALID")
            plan = _stage_plan(batch=batch, candidate=candidate, stage=stage)
            try:
                _validate_plan_semantics(plan, workspace_root=workspace_root)
            except AutoExperimentError as exc:
                raise ExperimentSchedulerError(f"EXPERIMENT_SCHEDULER_STAGE_INVALID:{exc}") from exc


def _rank_candidates(batch: Mapping[str, object]) -> list[dict[str, object]]:
    scoring = batch["scoring"]
    rows = []
    for candidate in batch["candidates"]:
        stage = candidate["stages"][0]
        screen_hours = float(stage["estimated_gpu_hours"])
        score = sum(
            float(scoring[f"{field}_weight"]) * float(candidate[field])
            for field in _SCORE_FIELDS
        ) - float(scoring["cost_weight"]) * screen_hours
        rows.append(
            {
                "candidate": candidate,
                "score": _round(score),
                "screen_gpu_hours": screen_hours,
            }
        )
    rows.sort(key=lambda row: (-float(row["score"]), str(row["candidate"]["candidate_id"])))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def _budget_select(
    ranked: Sequence[Mapping[str, object]], *, total_budget_gpu_hours: float, max_selected_candidates: int
) -> tuple[list[Mapping[str, object]], list[dict[str, object]]]:
    selected: list[Mapping[str, object]] = []
    deferred: list[dict[str, object]] = []
    spent = 0.0
    for row in ranked:
        candidate_id = str(row["candidate"]["candidate_id"])
        hours = float(row["screen_gpu_hours"])
        if len(selected) >= max_selected_candidates:
            reason = "MAX_SELECTED_CANDIDATES"
        elif spent + hours > total_budget_gpu_hours + 1e-12:
            reason = "SCREEN_BUDGET_EXCEEDED"
        else:
            selected.append(row)
            spent += hours
            continue
        deferred.append(
            {
                "candidate_id": candidate_id,
                "rank": row["rank"],
                "score": row["score"],
                "screen_gpu_hours": hours,
                "reason": reason,
            }
        )
    return selected, deferred


def _stage_plan(
    *, batch: Mapping[str, object], candidate: Mapping[str, object], stage: Mapping[str, object]
) -> dict[str, object]:
    candidate_id = str(candidate["candidate_id"])
    stage_name = str(stage["stage"])
    trial_id = _trial_id(candidate_id, stage_name)
    plan = {
        "schema_version": 1,
        "artifact_type": "verdiwm-auto-experiment-plan",
        "campaign_id": batch["campaign_id"],
        "trial_id": trial_id,
        "objective": batch["objective"],
        "hypothesis": candidate["hypothesis"],
        "selection_reason": f"{batch['selection_reason']} Candidate: {candidate['selection_reason']}",
        "falsification_criterion": candidate["falsification_criterion"],
        "stage": stage_name,
        "total_budget_gpu_hours": batch["total_budget_gpu_hours"],
    }
    plan.update({key: stage[key] for key in (
        "command", "working_directory", "allowed_gpu_indices", "estimated_gpu_hours",
        "timeout_seconds", "gpu_wait_seconds", "sample_interval_seconds", "result_path",
        "artifacts", "metric_gates", "environment", "cleanup_policy",
    )})
    return plan


def _trial_id(candidate_id: str, stage: str) -> str:
    value = f"auto-{candidate_id}-{stage}"
    if len(value) <= 128:
        return value
    digest = _sha256_bytes(value.encode("utf-8"))[:12]
    return f"auto-{candidate_id[:108]}-{digest}"


def _queue_manifest(queue: Mapping[str, object], destination: Path) -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_type": "verdiwm-auto-experiment-queue-manifest",
        "state": queue["state"],
        "campaign_id": queue["campaign_id"],
        "batch_sha256": queue["batch_sha256"],
        "ranked_candidate_count": queue["ranked_candidate_count"],
        "selected_candidate_count": len(queue["selected"]),
        "deferred_candidate_count": len(queue["deferred"]),
        "queue_path": str(destination / "queue.json"),
    }


def _execution_document(
    queue: Mapping[str, object], results: Mapping[str, object], budget_db: Path
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_type": "verdiwm-auto-experiment-queue-execution",
        "state": "ready",
        "campaign_id": queue["campaign_id"],
        "batch_sha256": queue["batch_sha256"],
        "budget_db": str(budget_db),
        "results": dict(results),
    }


def _write_markdown(path: Path, queue: Mapping[str, object]) -> None:
    lines = [
        "# VerdiWM Auto-Experiment Queue",
        "",
        f"- campaign: `{queue['campaign_id']}`",
        f"- ranked candidates: `{queue['ranked_candidate_count']}`",
        f"- selected for screen: `{len(queue['selected'])}`",
        f"- deferred: `{len(queue['deferred'])}`",
        "",
        "| Rank | Candidate | Score | Screen GPU hours | Stages |",
        "|---:|---|---:|---:|---|",
    ]
    for row in queue["selected"]:
        lines.append(
            f"| {row['rank']} | `{row['candidate_id']}` | {float(row['score']):.6f} | {float(row['screen_gpu_hours']):.4f} | "
            + " -> ".join(str(stage["stage"]) for stage in row["stages"])
            + " |"
        )
    lines.extend(("", "Deferred candidates are retained with a machine-readable reason; no GPU work is implied by this queue."))
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def _load_optional_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8")) if path.is_file() and not path.is_symlink() else {}
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _resolve_inside(root: Path, value: str, code: str) -> Path:
    resolved = (root / value).resolve() if not Path(value).is_absolute() else Path(value).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ExperimentSchedulerError(code) from exc
    return resolved


def _load_json_object(path: Path, code: str) -> dict[str, object]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentSchedulerError(code) from exc
    if not isinstance(value, dict):
        raise ExperimentSchedulerError(code)
    return value


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _round(value: float) -> float:
    return float(f"{value:.12g}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan_parser = commands.add_parser("plan", help="rank candidates and write a bounded queue")
    plan_parser.add_argument("--batch", type=Path, required=True)
    plan_parser.add_argument("--output-root", type=Path, required=True)
    plan_parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    run_parser = commands.add_parser("run", help="run selected candidates with PASS-only promotion")
    run_parser.add_argument("--queue", type=Path, required=True)
    run_parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    run_parser.add_argument("--archive-db", type=Path, required=True)
    run_parser.add_argument("--cas-root", type=Path, required=True)
    run_parser.add_argument("--lock-root", type=Path, default=Path("/tmp/verdiwm-gpu-leases"))
    run_parser.add_argument("--budget-db", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            output = plan_candidate_batch(
                batch_path=args.batch,
                output_root=args.output_root,
                workspace_root=args.workspace_root,
            )
        else:
            output = run_selected_queue(
                queue_path=args.queue,
                workspace_root=args.workspace_root,
                archive_db=args.archive_db,
                cas_root=args.cas_root,
                lock_root=args.lock_root,
                budget_db=args.budget_db,
            )
        print(json.dumps(output, ensure_ascii=True, sort_keys=True))
        return 0
    except (ExperimentSchedulerError, AutoExperimentError) as exc:
        print(str(exc), file=__import__("sys").stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
