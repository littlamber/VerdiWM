"""Deterministic training-scale planning for world-model experiments.

The planner keeps dataset size, episode diversity, update count, and held-out
checkpoint cadence in one receipt.  It does not launch training or silently
select a larger budget when a stage is underpowered.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from wmloop.contracts import ContractValidationError, validate_document


class TrainingScaleError(ValueError):
    """The dataset or scale policy cannot support a safe training plan."""


STAGE_POLICIES: dict[str, dict[str, float | int]] = {
    "smoke": {
        "dataset_fraction": 0.02,
        "target_epochs": 0.02,
        "min_steps": 8,
        "max_steps": 32,
        "min_episodes": 1,
        "seed_count": 1,
    },
    "screen": {
        "dataset_fraction": 0.10,
        "target_epochs": 0.25,
        "min_steps": 64,
        "max_steps": 512,
        "min_episodes": 4,
        "seed_count": 1,
    },
    "pilot": {
        "dataset_fraction": 0.50,
        "target_epochs": 2.0,
        "min_steps": 256,
        "max_steps": 4096,
        "min_episodes": 8,
        "seed_count": 3,
    },
    "confirm": {
        "dataset_fraction": 1.0,
        "target_epochs": 5.0,
        "min_steps": 1024,
        "max_steps": 20000,
        "min_episodes": 16,
        "seed_count": 3,
    },
}


def build_training_scale_plan(
    *,
    train_manifest: Path,
    val_manifest: Path | None = None,
    stage: str = "screen",
    batch_size: int = 1,
    gradient_accumulation: int = 1,
    world_size: int = 1,
    sequence_length: int | None = None,
    requested_seed_count: int | None = None,
    training_profile: Mapping[str, Any] | None = None,
    root: Path | None = None,
) -> dict[str, object]:
    """Build a scale plan from sample manifests without touching the GPU."""

    policy = _resolve_policy(stage, training_profile)
    batch = _positive_int(batch_size, "TRAINING_SCALE_BATCH_SIZE_INVALID")
    accumulation = _positive_int(
        gradient_accumulation, "TRAINING_SCALE_GRADIENT_ACCUMULATION_INVALID"
    )
    workers = _positive_int(world_size, "TRAINING_SCALE_WORLD_SIZE_INVALID")
    if sequence_length is not None:
        _positive_int(sequence_length, "TRAINING_SCALE_SEQUENCE_LENGTH_INVALID")

    train = _load_samples(Path(train_manifest), "TRAINING_SCALE_TRAIN_MANIFEST_INVALID")
    validation = (
        _load_samples(Path(val_manifest), "TRAINING_SCALE_VAL_MANIFEST_INVALID")
        if val_manifest is not None
        else []
    )
    train_count = len(train)
    val_count = len(validation)
    train_episodes = _episode_ids(train)
    val_episodes = _episode_ids(validation)

    fraction = float(policy["dataset_fraction"])
    selected_count = min(train_count, max(1, math.ceil(train_count * fraction)))
    selected = _ranked_subset(train, selected_count)
    selected_episodes = _episode_ids(selected)
    minimum_episodes = int(policy["min_episodes"])
    diversity_state = "pass" if len(selected_episodes) >= minimum_episodes else "blocked"
    effective_batch = batch * accumulation * workers
    steps_per_epoch = int(policy.get("steps_per_epoch") or max(1, math.ceil(selected_count / effective_batch)))
    target_steps = math.ceil(steps_per_epoch * float(policy["target_epochs"]))
    planned_steps = min(
        int(policy["max_steps"]),
        max(int(policy["min_steps"]), target_steps),
    )
    checkpoint_steps = _checkpoint_steps(
        planned_steps,
        fractions=(
            tuple(float(value) for value in policy["checkpoint_ladder"])
            if policy.get("checkpoint_ladder")
            else (0.25, 0.5, 0.75, 1.0)
        ),
    )
    requested_seeds = int(policy["seed_count"] if requested_seed_count is None else requested_seed_count)
    if requested_seeds < 1:
        raise TrainingScaleError("TRAINING_SCALE_SEED_COUNT_INVALID")

    state = "ready" if diversity_state == "pass" else "blocked"
    blockers = []
    if diversity_state == "blocked":
        blockers.append(
            "TRAINING_SCALE_EPISODE_DIVERSITY_INSUFFICIENT"
            f":have={len(selected_episodes)}:need={minimum_episodes}"
        )
    if not validation:
        blockers.append("TRAINING_SCALE_VALIDATION_MANIFEST_MISSING")
        state = "blocked"

    manifest_paths = {
        "train": str(Path(train_manifest).expanduser().resolve()),
        "validation": (
            str(Path(val_manifest).expanduser().resolve())
            if val_manifest is not None
            else None
        ),
    }
    plan: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "verdiwm-training-scale-plan",
        "state": state,
        "stage": stage,
        "dataset": {
            "manifest_paths": manifest_paths,
            "train_examples": train_count,
            "validation_examples": val_count,
            "train_episode_count": len(train_episodes),
            "validation_episode_count": len(val_episodes),
            "selected_train_examples": selected_count,
            "selected_train_episode_count": len(selected_episodes),
            "selection_policy": "sha256_ranked_sample_fraction_v1",
            "selection_fraction": fraction,
            "sequence_length": sequence_length,
            "train_manifest_sha256": _sha256(Path(train_manifest)),
            "validation_manifest_sha256": (
                _sha256(Path(val_manifest)) if val_manifest is not None else None
            ),
        },
        "parallelism": {
            "batch_size": batch,
            "gradient_accumulation": accumulation,
            "world_size": workers,
            "effective_batch_size": effective_batch,
        },
        "updates": {
            "steps_per_epoch": steps_per_epoch,
            "target_epochs": float(policy["target_epochs"]),
            "planned_steps": planned_steps,
            "realized_epochs": planned_steps / steps_per_epoch,
            "checkpoint_eval_steps": checkpoint_steps,
        },
        "replication": {
            "requested_seed_count": requested_seeds,
            "seed_policy": "freeze_before_launch_and_reuse_for_all_arms",
        },
        "stopping_policy": {
            "evaluate_each_checkpoint_on_heldout": True,
            "max_consecutive_heldout_regressions": 2,
            "select_final_checkpoint_by": "heldout_primary_metric_then_earlier_step",
            "never_extend_budget_without_new_manifest": True,
            "early_stop": policy.get("early_stop") or "two_consecutive_heldout_regressions",
        },
        "blockers": blockers,
        "rationale": (
            "Stage budgets are derived from deterministic sample counts and effective batch size. "
            "Episode diversity is a hard gate because repeated windows from one episode are not "
            "independent world-model evidence."
        ),
    }
    if training_profile is not None:
        plan["training_profile"] = {
            "recipe_id": training_profile.get("recipe_id"),
            "status": training_profile.get("status"),
            "evidence_tier": training_profile.get("evidence_tier"),
            "planner_policy": training_profile.get("planner_policy"),
        }
    try:
        validate_document("training_scale_plan", plan, root=root)
    except ContractValidationError as exc:
        raise TrainingScaleError(f"TRAINING_SCALE_SCHEMA_INVALID:{exc}") from exc
    return plan


def write_training_scale_plan(plan: Mapping[str, object], output: Path) -> None:
    """Write a canonical plan atomically so a resumed run never sees a partial file."""

    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    payload = json.dumps(plan, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=str(destination.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _load_samples(path: Path, error_code: str) -> list[Mapping[str, Any]]:
    try:
        payload = json.loads(path.expanduser().resolve(strict=True).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingScaleError(error_code) from exc
    if isinstance(payload, Mapping):
        payload = payload.get("samples", payload.get("records"))
    if not isinstance(payload, list) or not payload:
        raise TrainingScaleError(error_code)
    if not all(isinstance(row, Mapping) for row in payload):
        raise TrainingScaleError(error_code)
    return [row for row in payload if isinstance(row, Mapping)]


def _episode_ids(samples: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    values = {str(row["episode_id"]) for row in samples if "episode_id" in row}
    return tuple(sorted(values))


def _ranked_subset(
    samples: Sequence[Mapping[str, Any]], count: int
) -> list[Mapping[str, Any]]:
    ranked = sorted(
        enumerate(samples),
        key=lambda item: hashlib.sha256(
            (
                json.dumps(item[1], sort_keys=True, separators=(",", ":"), ensure_ascii=True)
                + f"\0{item[0]}"
            ).encode("utf-8")
        ).hexdigest(),
    )
    return [row for _, row in ranked[:count]]


def _checkpoint_steps(
    steps: int, *, fractions: Sequence[float] = (0.25, 0.5, 0.75, 1.0)
) -> list[int]:
    points = {max(1, math.ceil(steps * fraction)) for fraction in fractions}
    return sorted(points)


def _resolve_policy(
    stage: str, training_profile: Mapping[str, Any] | None
) -> dict[str, Any]:
    base = STAGE_POLICIES.get(stage)
    if base is None:
        raise TrainingScaleError("TRAINING_SCALE_STAGE_INVALID")
    policy: dict[str, Any] = dict(base)
    if training_profile is None:
        return policy
    status = training_profile.get("status")
    if (
        status not in {"local_validated", "reusable_optimization_memory"}
        or training_profile.get("planner_policy") != "admitted"
    ):
        raise TrainingScaleError("TRAINING_SCALE_EXTERNAL_RECIPE_NOT_ADMITTED")
    overrides = training_profile.get("planning", {})
    if not isinstance(overrides, Mapping):
        raise TrainingScaleError("TRAINING_SCALE_PROFILE_INVALID")
    for key in (
        "dataset_fraction",
        "target_epochs",
        "min_steps",
        "max_steps",
        "min_episodes",
        "steps_per_epoch",
        "early_stop",
        "checkpoint_ladder",
    ):
        if key in overrides and overrides[key] is not None:
            policy[key] = overrides[key]
    fraction = policy.get("dataset_fraction")
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0 < fraction <= 1:
        raise TrainingScaleError("TRAINING_SCALE_PROFILE_DATASET_FRACTION_INVALID")
    for key in ("min_steps", "max_steps", "min_episodes", "steps_per_epoch"):
        value = policy.get(key)
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
            raise TrainingScaleError(f"TRAINING_SCALE_PROFILE_{key.upper()}_INVALID")
    if int(policy["min_steps"]) > int(policy["max_steps"]):
        raise TrainingScaleError("TRAINING_SCALE_PROFILE_STEP_RANGE_INVALID")
    target_epochs = policy.get("target_epochs")
    if isinstance(target_epochs, bool) or not isinstance(target_epochs, (int, float)) or target_epochs <= 0:
        raise TrainingScaleError("TRAINING_SCALE_PROFILE_TARGET_EPOCHS_INVALID")
    ladder = policy.get("checkpoint_ladder")
    if ladder is not None:
        if not isinstance(ladder, Sequence) or isinstance(ladder, (str, bytes)) or not ladder:
            raise TrainingScaleError("TRAINING_SCALE_PROFILE_CHECKPOINT_LADDER_INVALID")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value <= 1 for value in ladder):
            raise TrainingScaleError("TRAINING_SCALE_PROFILE_CHECKPOINT_LADDER_INVALID")
    return policy


def _positive_int(value: object, code: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise TrainingScaleError(code)
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.expanduser().resolve(strict=True).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
