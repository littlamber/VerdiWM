"""Training-step budget, checkpoint-ladder, and monitor contracts."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import shutil
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


class TrainingMonitorPolicyError(RuntimeError):
    """The training monitor policy input or evidence is invalid."""


DEFAULT_CONFIRMATION_STEPS = 1000
DEFAULT_CHECKPOINT_EVAL_STEPS = (512, 800, 1000)
DEFAULT_TRAIN_STEP_CAP = DEFAULT_CONFIRMATION_STEPS
DEFAULT_MAX_CONSECUTIVE_REGRESSIONS = 2
DEFAULT_SEQ_LEN = 37
DEFAULT_TRAJECTORY_LENGTH = 64
DEFAULT_TRAIN_SIZE = 32
DEFAULT_BATCH_SIZE = 16


def checkpoint_eval_ladder(
    train_steps: int,
    *,
    requested_steps: Sequence[int] = DEFAULT_CHECKPOINT_EVAL_STEPS,
) -> tuple[int, ...]:
    """Return relative training steps that should be evaluated on held-out rollouts."""

    steps = _positive_int(train_steps, "TRAINING_MONITOR_TRAIN_STEPS_INVALID")
    parsed = tuple(sorted({_positive_int(step, "TRAINING_MONITOR_LADDER_INVALID") for step in requested_steps}))
    if not parsed:
        raise TrainingMonitorPolicyError("TRAINING_MONITOR_LADDER_INVALID")
    selected = [step for step in parsed if step <= steps]
    if not selected or selected[-1] != steps:
        selected.append(steps)
    return tuple(dict.fromkeys(selected))


def epoch_conversion(
    *,
    train_steps: int,
    batch_size: int = DEFAULT_BATCH_SIZE,
    train_size: int = DEFAULT_TRAIN_SIZE,
    trajectory_length: int = DEFAULT_TRAJECTORY_LENGTH,
    seq_len: int = DEFAULT_SEQ_LEN,
) -> dict[str, object]:
    """Convert update steps into dataset-pass counts for the current ACWM windowing."""

    steps = _positive_int(train_steps, "TRAINING_MONITOR_TRAIN_STEPS_INVALID")
    batch = _positive_int(batch_size, "TRAINING_MONITOR_BATCH_SIZE_INVALID")
    trajectories = _positive_int(train_size, "TRAINING_MONITOR_TRAIN_SIZE_INVALID")
    horizon = _positive_int(trajectory_length, "TRAINING_MONITOR_TRAJECTORY_LENGTH_INVALID")
    sequence = _positive_int(seq_len, "TRAINING_MONITOR_SEQ_LEN_INVALID")
    if sequence > horizon:
        raise TrainingMonitorPolicyError("TRAINING_MONITOR_SEQ_LEN_INVALID")
    windows_per_trajectory = horizon - sequence + 1
    sliding_windows = trajectories * windows_per_trajectory
    steps_per_epoch = max(1, math.ceil(sliding_windows / batch))
    return {
        "train_steps": steps,
        "train_size_trajectories": trajectories,
        "trajectory_length": horizon,
        "seq_len": sequence,
        "windows_per_trajectory": windows_per_trajectory,
        "sliding_windows": sliding_windows,
        "batch_size": batch,
        "steps_per_epoch": steps_per_epoch,
        "epochs": steps / steps_per_epoch,
    }


def training_monitor_policy_document(
    *,
    train_steps: int = DEFAULT_TRAIN_STEP_CAP,
    batch_size: int = DEFAULT_BATCH_SIZE,
    train_size: int = DEFAULT_TRAIN_SIZE,
    trajectory_length: int = DEFAULT_TRAJECTORY_LENGTH,
    seq_len: int = DEFAULT_SEQ_LEN,
    requested_ladder_steps: Sequence[int] = DEFAULT_CHECKPOINT_EVAL_STEPS,
    max_default_train_steps: int = DEFAULT_TRAIN_STEP_CAP,
    allow_extended_confirmation: bool = False,
) -> dict[str, object]:
    """Build the machine-readable monitor contract used by future GPU trials."""

    cap = _positive_int(max_default_train_steps, "TRAINING_MONITOR_CAP_INVALID")
    steps = _positive_int(train_steps, "TRAINING_MONITOR_TRAIN_STEPS_INVALID")
    ladder = checkpoint_eval_ladder(steps, requested_steps=requested_ladder_steps)
    conversion = epoch_conversion(
        train_steps=steps,
        batch_size=batch_size,
        train_size=train_size,
        trajectory_length=trajectory_length,
        seq_len=seq_len,
    )
    extended = steps > cap
    blocked = extended and not allow_extended_confirmation
    return {
        "schema_version": 1,
        "artifact_type": "wmloop-training-monitor-policy",
        "state": "needs_explicit_extended_confirmation" if blocked else "ready",
        "generated_at": _now(),
        "train_steps": steps,
        "max_default_train_steps": cap,
        "allow_extended_confirmation": bool(allow_extended_confirmation),
        "budget_profile": _budget_profile(steps, cap=cap),
        "checkpoint_eval_ladder": list(ladder),
        "checkpoint_eval_policy": {
            "evaluate_each_checkpoint_with_frozen_heldout_rollout": True,
            "select_best_checkpoint_by": "heldout_candidate_primary_metric",
            "tie_breakers": [
                "higher_paired_delta_primary_metric",
                "earlier_checkpoint_step",
            ],
            "final_checkpoint_is_not_automatically_best": True,
            "max_consecutive_regressions": DEFAULT_MAX_CONSECUTIVE_REGRESSIONS,
            "extension_requires_latest_checkpoint_to_pass": True,
        },
        "epoch_conversion": conversion,
        "monitor_metrics": {
            "train_health": [
                "train/loss",
                "primitive_aux_loss",
                "grad_norm",
                "learning_rate",
                "seconds_per_step",
                "oom_or_nan_count",
            ],
            "heldout_rollout": [
                "psnr_by_horizon",
                "ssim_by_horizon",
                "masked_mse_by_horizon",
                "horizon_auc_primary_metric",
                "per_frame_psnr_decay",
            ],
            "action_conditioning": [
                "action_conditioned_psnr",
                "no_action_psnr",
                "no_action_delta_psnr",
                "inverse_dynamics_accuracy_per_frame",
                "inverse_dynamics_r2",
            ],
            "primitive_specific": [
                "primitive_sidecar_contract_present",
                "runtime_hook_invoked",
                "bounded_intervention_rate",
            ],
        },
        "overfit_alarm_rules": [
            {
                "name": "train_down_heldout_flat",
                "condition": "train_loss_decreases while heldout_primary_metric does not improve at two consecutive ladder points",
                "action": "stop_promoting_final_checkpoint; pick best heldout checkpoint or reject",
            },
            {
                "name": "short_horizon_up_long_horizon_down",
                "condition": "horizon16 improves while longest supported horizon regresses",
                "action": "mark primitive as horizon-overfit for this environment",
            },
            {
                "name": "action_signal_collapse",
                "condition": "action-conditioned advantage or inverse-dynamics confidence decreases",
                "action": "fail action-following gate even if visual PSNR rises",
            },
            {
                "name": "ind_ood_split",
                "condition": "InD heldout improves while OoD/probe metric regresses",
                "action": "downgrade from formal claim to environment-local canary",
            },
        ],
        "blockers": ["train_steps_exceeds_default_cap_without_extended_confirmation"] if blocked else [],
        "claim_boundary": (
            "1k is the default staged-confirmation cap; longer runs require an explicit best-checkpoint review."
            if extended
            else "This budget is inside the default 1k staged-confirmation cap."
        ),
    }


def select_best_checkpoint(
    records: Sequence[Mapping[str, object]],
    *,
    primary_metric: str = "ladder_auc_psnr_envmax",
    require_complete: bool = True,
    require_action_gate: bool = True,
) -> dict[str, object]:
    """Select the best checkpoint from already-evaluated ladder records."""

    if not records:
        raise TrainingMonitorPolicyError("TRAINING_MONITOR_SELECTION_RECORDS_EMPTY")
    annotated = [_annotate_checkpoint_record(record, primary_metric=primary_metric) for record in records]
    eligible = [
        item
        for item in annotated
        if _record_complete(item, require_complete=require_complete)
        and _record_action_gate_passed(item, require_action_gate=require_action_gate)
        and math.isfinite(float(item["selection_score"]))
    ]
    if not eligible:
        return {
            "schema_version": 1,
            "artifact_type": "wmloop-checkpoint-ladder-selection",
            "state": "no_eligible_checkpoint",
            "primary_metric": primary_metric,
            "require_complete": require_complete,
            "require_action_gate": require_action_gate,
            "records": annotated,
            "best_checkpoint": None,
        }
    best = max(
        eligible,
        key=lambda item: (
            float(item["selection_score"]),
            float(item.get("delta_primary_metric", float("-inf"))),
            -int(item["checkpoint_step"]),
        ),
    )
    return {
        "schema_version": 1,
        "artifact_type": "wmloop-checkpoint-ladder-selection",
        "state": "ready",
        "primary_metric": primary_metric,
        "require_complete": require_complete,
        "require_action_gate": require_action_gate,
        "selection_rule": "max heldout candidate primary metric; tie by paired delta; tie by earlier checkpoint",
        "records": annotated,
        "best_checkpoint": best,
    }


def select_best_official_checkpoint(
    records: Sequence[Mapping[str, object]],
    *,
    max_consecutive_regressions: int = DEFAULT_MAX_CONSECUTIVE_REGRESSIONS,
) -> dict[str, object]:
    """Select the best official-gate checkpoint and decide whether budget may extend."""

    if not records:
        raise TrainingMonitorPolicyError("TRAINING_MONITOR_SELECTION_RECORDS_EMPTY")
    regression_limit = _positive_int(
        max_consecutive_regressions,
        "TRAINING_MONITOR_REGRESSION_LIMIT_INVALID",
    )
    annotated = sorted((_annotate_official_record(record) for record in records), key=lambda item: int(item["checkpoint_step"]))
    eligible = [item for item in annotated if item["official_gate_passed"] is True]
    best = (
        max(
            eligible,
            key=lambda item: (
                float(item["candidate_psnr"]),
                float(item["candidate_ssim"]),
                -float(item["candidate_mse"]),
                -float(item["candidate_masked_mse"]),
                -int(item["checkpoint_step"]),
            ),
        )
        if eligible
        else None
    )
    running_best_psnr = float("-inf")
    consecutive_regressions = 0
    max_observed_consecutive_regressions = 0
    for item in annotated:
        passed = item["official_gate_passed"] is True
        psnr = float(item["candidate_psnr"])
        if passed and psnr >= running_best_psnr:
            running_best_psnr = psnr
            consecutive_regressions = 0
            item["regressed_from_running_best"] = False
        else:
            consecutive_regressions += 1
            item["regressed_from_running_best"] = True
        item["consecutive_regressions"] = consecutive_regressions
        max_observed_consecutive_regressions = max(max_observed_consecutive_regressions, consecutive_regressions)
    stop_requested = max_observed_consecutive_regressions >= regression_limit
    latest_passed = annotated[-1]["official_gate_passed"] is True
    return {
        "schema_version": 1,
        "artifact_type": "wmloop-official-checkpoint-ladder-selection",
        "state": "ready" if best is not None else "no_eligible_checkpoint",
        "selection_rule": "best passing official gate by candidate PSNR, SSIM, inverse MSE, inverse masked-MSE, then earlier step",
        "max_consecutive_regressions": regression_limit,
        "max_observed_consecutive_regressions": max_observed_consecutive_regressions,
        "stop_requested": stop_requested,
        "extension_allowed": bool(best is not None and latest_passed and not stop_requested),
        "records": annotated,
        "best_checkpoint": best,
    }


def _annotate_official_record(record: Mapping[str, object]) -> dict[str, object]:
    step = _positive_int(record.get("checkpoint_step"), "TRAINING_MONITOR_SELECTION_RECORD_INVALID")
    gate = record.get("official_quality_gate")
    if not isinstance(gate, Mapping):
        raise TrainingMonitorPolicyError("TRAINING_MONITOR_OFFICIAL_GATE_INVALID")
    candidate = gate.get("candidate")
    if not isinstance(candidate, Mapping):
        raise TrainingMonitorPolicyError("TRAINING_MONITOR_OFFICIAL_GATE_INVALID")
    return {
        **dict(record),
        "checkpoint_step": step,
        "official_gate_passed": gate.get("pass") is True,
        "candidate_psnr": _metric(candidate, "psnr"),
        "candidate_ssim": _metric(candidate, "ssim"),
        "candidate_mse": _metric(candidate, "mse"),
        "candidate_masked_mse": _metric(candidate, "masked_mse"),
    }


def write_policy_bundle(*, policy: Mapping[str, object], output_root: Path) -> dict[str, object]:
    """Write a JSON/Markdown/manifest bundle for a monitor policy document."""

    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise TrainingMonitorPolicyError("TRAINING_MONITOR_OUTPUT_EXISTS")
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    try:
        temporary.mkdir(mode=0o700, parents=True)
        _write_json_atomic(temporary / "training-monitor-policy.json", policy)
        (temporary / "training-monitor-policy.md").write_text(_render_markdown(policy), encoding="utf-8")
        manifest = {
            "schema_version": 1,
            "artifact_type": "wmloop-training-monitor-policy-manifest",
            "state": policy["state"],
            "output_root": str(destination),
            "report_path": str(destination / "training-monitor-policy.json"),
            "markdown_path": str(destination / "training-monitor-policy.md"),
            "train_steps": policy["train_steps"],
            "checkpoint_eval_ladder": policy["checkpoint_eval_ladder"],
            "budget_profile": policy["budget_profile"],
            "blockers": policy["blockers"],
        }
        _write_json_atomic(temporary / "manifest.json", manifest)
        os.replace(temporary, destination)
        return manifest
    except Exception:
        if temporary.exists() or temporary.is_symlink():
            shutil.rmtree(temporary, ignore_errors=True)
        raise


def _annotate_checkpoint_record(record: Mapping[str, object], *, primary_metric: str) -> dict[str, object]:
    checkpoint_step = _positive_int(record.get("checkpoint_step"), "TRAINING_MONITOR_SELECTION_RECORD_INVALID")
    candidate = _metric(record, f"candidate_{primary_metric}", "candidate_primary_metric", "candidate_auc_psnr_16_64")
    delta = _metric(record, f"delta_{primary_metric}", "delta_primary_metric", "delta_auc_psnr_16_64", required=False)
    complete = _bool_metric(record, "required_horizons_complete", "evidence_complete", default=False)
    action = _action_gate(record)
    reasons: list[str] = []
    if not complete:
        reasons.append("required_horizons_incomplete")
    if not action["passed"]:
        reasons.append("action_gate_failed_or_missing")
    return {
        **dict(record),
        "checkpoint_step": checkpoint_step,
        "candidate_primary_metric": candidate,
        "delta_primary_metric": delta,
        "selection_score": candidate,
        "required_horizons_complete": complete,
        "action_gate": action,
        "eligibility_reasons": reasons,
    }


def _record_complete(record: Mapping[str, object], *, require_complete: bool) -> bool:
    return not require_complete or bool(record.get("required_horizons_complete"))


def _record_action_gate_passed(record: Mapping[str, object], *, require_action_gate: bool) -> bool:
    if not require_action_gate:
        return True
    action = record.get("action_gate")
    return isinstance(action, Mapping) and action.get("passed") is True


def _metric(record: Mapping[str, object], *keys: str, required: bool = True) -> float:
    spaces = [record]
    evaluation = record.get("evaluation")
    if isinstance(evaluation, Mapping):
        spaces.insert(0, evaluation)
    for space in spaces:
        for key in keys:
            value = space.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
                return float(value)
    if required:
        raise TrainingMonitorPolicyError("TRAINING_MONITOR_SELECTION_RECORD_INVALID")
    return float("-inf")


def _bool_metric(record: Mapping[str, object], *keys: str, default: bool) -> bool:
    spaces = [record]
    evaluation = record.get("evaluation")
    if isinstance(evaluation, Mapping):
        spaces.insert(0, evaluation)
    for space in spaces:
        for key in keys:
            value = space.get(key)
            if isinstance(value, bool):
                return value
    return default


def _action_gate(record: Mapping[str, object]) -> dict[str, object]:
    evaluation = record.get("evaluation")
    action = evaluation.get("action_following") if isinstance(evaluation, Mapping) else record.get("action_following")
    if not isinstance(action, Mapping):
        return {"present": False, "passed": False, "observed": None, "threshold": None}
    observed = action.get("gate_observed", action.get("action_following_observed"))
    threshold = action.get("gate_threshold", action.get("action_following_threshold"))
    if threshold is None:
        return {"present": True, "passed": True, "observed": observed, "threshold": None}
    if not isinstance(observed, (int, float)) or isinstance(observed, bool):
        return {"present": True, "passed": False, "observed": observed, "threshold": threshold}
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        return {"present": True, "passed": False, "observed": observed, "threshold": threshold}
    return {
        "present": True,
        "passed": float(observed) >= float(threshold),
        "observed": float(observed),
        "threshold": float(threshold),
    }


def _budget_profile(train_steps: int, *, cap: int) -> str:
    if train_steps < 512:
        return "startup_health_only"
    if train_steps <= 512:
        return "rapid_screen_minimum_signal"
    if train_steps <= cap:
        return "staged_confirmation"
    return "extended_confirmation_requires_reason"


def _render_markdown(policy: Mapping[str, object]) -> str:
    conversion = policy["epoch_conversion"]
    if not isinstance(conversion, Mapping):
        raise TrainingMonitorPolicyError("TRAINING_MONITOR_POLICY_INVALID")
    lines = [
        "# Training Monitor Policy",
        "",
        f"State: `{policy['state']}`",
        f"Budget profile: `{policy['budget_profile']}`",
        f"Train steps: `{policy['train_steps']}`",
        f"Default cap: `{policy['max_default_train_steps']}`",
        f"Checkpoint eval ladder: `{policy['checkpoint_eval_ladder']}`",
        f"Steps per epoch: `{conversion['steps_per_epoch']}`",
        f"Epochs at requested steps: `{conversion['epochs']}`",
        "",
        "## Selection Rule",
        "",
        "- Evaluate every ladder checkpoint with the frozen held-out rollout protocol.",
        "- Pick the best held-out candidate primary metric, not the final checkpoint by default.",
        "- Break ties with paired delta, then the earlier checkpoint.",
        "",
        "## Monitor Metrics",
        "",
    ]
    metrics = policy["monitor_metrics"]
    if isinstance(metrics, Mapping):
        for group, names in metrics.items():
            joined = ", ".join(str(name) for name in names) if isinstance(names, list) else str(names)
            lines.append(f"- `{group}`: {joined}")
    blockers = policy.get("blockers")
    if isinstance(blockers, list) and blockers:
        lines.extend(["", "## Blockers", ""])
        lines.extend(f"- {item}" for item in blockers)
    lines.append("")
    return "\n".join(lines)


def _positive_int(value: object, code: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise TrainingMonitorPolicyError(code)
    return value


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def _now() -> str:
    return dt.datetime.now(tz=dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="write a training monitor policy bundle")
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--train-steps", type=int, default=DEFAULT_TRAIN_STEP_CAP)
    run.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    run.add_argument("--train-size", type=int, default=DEFAULT_TRAIN_SIZE)
    run.add_argument("--trajectory-length", type=int, default=DEFAULT_TRAJECTORY_LENGTH)
    run.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    run.add_argument("--max-default-train-steps", type=int, default=DEFAULT_TRAIN_STEP_CAP)
    run.add_argument("--checkpoint-eval-steps", type=int, nargs="+", default=list(DEFAULT_CHECKPOINT_EVAL_STEPS))
    run.add_argument("--allow-extended-confirmation", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "run":
        policy = training_monitor_policy_document(
            train_steps=args.train_steps,
            batch_size=args.batch_size,
            train_size=args.train_size,
            trajectory_length=args.trajectory_length,
            seq_len=args.seq_len,
            requested_ladder_steps=tuple(args.checkpoint_eval_steps),
            max_default_train_steps=args.max_default_train_steps,
            allow_extended_confirmation=args.allow_extended_confirmation,
        )
        manifest = write_policy_bundle(policy=policy, output_root=args.output_root)
        print(json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
