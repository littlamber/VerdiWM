"""Deterministic, model-neutral training-scale and stage planning.

The planner keeps dataset size, episode diversity, update count, and held-out
checkpoint cadence in one receipt.  It does not launch training or silently
select a larger budget when a stage is underpowered.  A probe is explicitly
runtime-only; a formal stage always receives a long-training contract.
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
from wmloop.experiments.job_spec import effective_batch_size
from wmloop.storage import checked_path


class TrainingScaleError(ValueError):
    """The dataset or scale policy cannot support a safe training plan."""


STAGE_POLICIES: dict[str, dict[str, float | int]] = {
    "engineering_diagnostic": {
        "dataset_fraction": 1.0,
        "target_epochs": 0.02,
        "min_steps": 1,
        "max_steps": 8,
        "min_episodes": 1,
        "seed_count": 1,
    },
    "probe": {
        "dataset_fraction": 0.02,
        "target_epochs": 0.02,
        "min_steps": 1,
        "max_steps": 8,
        "min_episodes": 1,
        "seed_count": 1,
    },
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

_TRAINING_STAGE_ORDER = ("probe", "screen", "pilot", "confirm")
_CURRENT_STAGE_ALIASES = {"smoke": "probe"}
_FORMAL_STAGES = frozenset(("pilot", "confirm"))


def build_training_ladder(
    *,
    train_manifest: Path,
    val_manifest: Path,
    current_stage: str = "probe",
    target_stage: str = "confirm",
    batch_size: int = 1,
    gradient_accumulation: int = 1,
    world_size: int = 1,
    sequence_length: int | None = None,
    requested_seed_count: int | None = None,
    seed: int = 0,
    training_profile: Mapping[str, Any] | None = None,
    root: Path | None = None,
) -> dict[str, object]:
    """Plan every required upgrade from a runtime probe to a formal stage.

    This is deliberately a planning-only control-plane operation.  The
    returned ladder is consumed by a model adapter, while the stage semantics
    remain independent of model names, optimizer APIs, and checkpoint formats.
    A completed negative screen is retained as diagnostic evidence and does
    not veto construction of the formal plan.
    """

    if (
        current_stage not in set(_TRAINING_STAGE_ORDER) | set(_CURRENT_STAGE_ALIASES)
        or target_stage not in _TRAINING_STAGE_ORDER
    ):
        raise TrainingScaleError("TRAINING_SCALE_STAGE_INVALID")
    current_index = _TRAINING_STAGE_ORDER.index(
        _CURRENT_STAGE_ALIASES.get(current_stage, current_stage)
    )
    target_index = _TRAINING_STAGE_ORDER.index(target_stage)
    if target_index <= current_index:
        raise TrainingScaleError("TRAINING_SCALE_STAGE_TRANSITION_NOT_FORWARD")

    plans: list[dict[str, object]] = []
    transitions: list[dict[str, object]] = []
    for index in range(current_index + 1, target_index + 1):
        stage = _TRAINING_STAGE_ORDER[index]
        plan = build_training_scale_plan(
            train_manifest=train_manifest,
            val_manifest=val_manifest,
            stage=stage,
            batch_size=batch_size,
            gradient_accumulation=gradient_accumulation,
            world_size=world_size,
            sequence_length=sequence_length,
            requested_seed_count=requested_seed_count,
            seed=seed,
            training_profile=training_profile,
            root=root,
        )
        plans.append(plan)
        transitions.append(
            {
                "from_stage": (
                    current_stage
                    if index == current_index + 1
                    else _TRAINING_STAGE_ORDER[index - 1]
                ),
                "to_stage": stage,
                "automatic": True,
                "requires_prior_terminal_receipt": True,
                "screen_failure_veto": False,
                "claim_boundary": (
                    "The prior stage only authorizes the next declared runtime step; "
                    "it does not establish quality or promotion."
                ),
            }
        )

    blockers = [
        f"{stage}:{blocker}"
        for plan in plans
        for stage in [str(plan["stage"])]
        for blocker in plan["blockers"]
    ]
    state = "ready" if not blockers else "blocked"
    ladder = {
        "schema_version": 1,
        "artifact_type": "verdiwm-training-stage-ladder",
        "state": state,
        "current_stage": current_stage,
        "target_stage": target_stage,
        "automatic_upgrade": True,
        "plans": plans,
        "transitions": transitions,
        "blockers": blockers,
        "claim_boundary": (
            "A training ladder is an executable-scale admission receipt, not a "
            "quality result. Probe and screen evidence cannot be promoted; only "
            "the declared formal stage may enter a frozen quality gate."
        ),
    }
    try:
        validate_document("training_stage_ladder", ladder, root=root)
    except ContractValidationError as exc:
        raise TrainingScaleError(f"TRAINING_SCALE_LADDER_SCHEMA_INVALID:{exc}") from exc
    return ladder


def build_training_scale_plan(
    *,
    train_manifest: Path | None = None,
    val_manifest: Path | None = None,
    dataset_root: Path | None = None,
    stage: str = "screen",
    batch_size: int = 1,
    gradient_accumulation: int = 1,
    world_size: int = 1,
    sequence_length: int | None = None,
    requested_seed_count: int | None = None,
    seed: int = 0,
    training_profile: Mapping[str, Any] | None = None,
    root: Path | None = None,
) -> dict[str, object]:
    """Build a scale plan from sample manifests without touching the GPU."""

    train_path, validation_path, native_agibot = _resolve_dataset_inputs(
        train_manifest=train_manifest,
        val_manifest=val_manifest,
        dataset_root=dataset_root,
    )
    if native_agibot and stage != "engineering_diagnostic":
        raise TrainingScaleError(
            "TRAINING_SCALE_AGIBOT_NATIVE_INDEX_ENGINEERING_DIAGNOSTIC_ONLY"
        )

    policy = _resolve_policy(stage, training_profile)
    batch = _positive_int(batch_size, "TRAINING_SCALE_BATCH_SIZE_INVALID")
    accumulation = _positive_int(
        gradient_accumulation, "TRAINING_SCALE_GRADIENT_ACCUMULATION_INVALID"
    )
    workers = _positive_int(world_size, "TRAINING_SCALE_WORLD_SIZE_INVALID")
    if sequence_length is not None:
        _positive_int(sequence_length, "TRAINING_SCALE_SEQUENCE_LENGTH_INVALID")

    train_summary = (
        _summarize_agibot_index(
            train_path,
            expected_split="train",
            error_code="TRAINING_SCALE_TRAIN_MANIFEST_INVALID",
        )
        if native_agibot
        else None
    )
    validation_summary = (
        _summarize_agibot_index(
            validation_path,
            expected_split="val",
            error_code="TRAINING_SCALE_VAL_MANIFEST_INVALID",
        )
        if native_agibot and validation_path is not None
        else None
    )
    train = (
        []
        if native_agibot
        else _load_samples(train_path, "TRAINING_SCALE_TRAIN_MANIFEST_INVALID")
    )
    validation = (
        []
        if native_agibot or validation_path is None
        else _load_samples(validation_path, "TRAINING_SCALE_VAL_MANIFEST_INVALID")
    )
    train_count = train_summary["count"] if train_summary else len(train)
    val_count = validation_summary["count"] if validation_summary else len(validation)
    train_episodes = (
        train_summary["episode_ids"] if train_summary else _episode_ids(train)
    )
    val_episodes = (
        validation_summary["episode_ids"]
        if validation_summary
        else _episode_ids(validation)
    )
    episode_overlap = sorted(set(train_episodes) & set(val_episodes))

    fraction = float(policy["dataset_fraction"])
    selected_count = min(train_count, max(1, math.ceil(train_count * fraction)))
    if native_agibot:
        if selected_count != train_count or train_summary is None:
            raise TrainingScaleError("TRAINING_SCALE_AGIBOT_SELECTION_BOUNDARY_INVALID")
        selected_indices: list[int] = []
        selected_episodes = train_episodes
        selected_record_ids = list(train_summary["record_ids"])
        selected_record_ids_binding = _compact_id_binding(selected_record_ids)
        selected_episode_ids_binding = _compact_id_binding(list(selected_episodes))
    else:
        selected_indices = _episode_balanced_indices(train, selected_count)
        selected = [train[index] for index in selected_indices]
        selected_episodes = _episode_ids(selected)
        selected_record_ids = [
            _training_record_id(index, train[index]) for index in selected_indices
        ]
        selected_record_ids_binding = None
        selected_episode_ids_binding = None
    minimum_episodes = int(policy["min_episodes"])
    diversity_state = "pass" if len(selected_episodes) >= minimum_episodes else "blocked"
    effective_batch = effective_batch_size(batch, accumulation, workers)
    if native_agibot:
        # Native episode indexes do not define training windows. Keep this
        # diagnostic strictly bounded by optimizer updates until the frozen
        # Agibot loader supplies the authoritative window count.
        steps_per_epoch = None
        target_epochs = None
        target_steps = int(policy["min_steps"])
    else:
        steps_per_epoch = int(policy.get("steps_per_epoch") or max(1, math.ceil(selected_count / effective_batch)))
        target_epochs = float(policy["target_epochs"])
        target_steps = math.ceil(steps_per_epoch * target_epochs)
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
    if stage == "engineering_diagnostic" and requested_seeds != 1:
        raise TrainingScaleError(
            "TRAINING_SCALE_ENGINEERING_DIAGNOSTIC_SINGLE_SEED_REQUIRED"
        )
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 or seed + requested_seeds > 2**63:
        raise TrainingScaleError("TRAINING_SCALE_SEED_INVALID")

    state = "ready" if diversity_state == "pass" else "blocked"
    blockers = []
    if diversity_state == "blocked":
        blockers.append(
            "TRAINING_SCALE_EPISODE_DIVERSITY_INSUFFICIENT"
            f":have={len(selected_episodes)}:need={minimum_episodes}"
        )
    if val_count == 0:
        blockers.append("TRAINING_SCALE_VALIDATION_MANIFEST_MISSING")
        state = "blocked"
    if episode_overlap:
        blockers.append(
            "TRAINING_SCALE_TRAIN_VALIDATION_EPISODE_OVERLAP:"
            + ",".join(episode_overlap[:8])
        )
        state = "blocked"

    manifest_paths = {
        "train": str(train_path),
        "validation": str(validation_path) if validation_path is not None else None,
    }
    if len(selected_record_ids) != selected_count:
        raise TrainingScaleError("TRAINING_SCALE_SELECTED_RECORD_COUNT_MISMATCH")
    if stage == "engineering_diagnostic" and (
        selected_count != train_count
        or selected_episodes != train_episodes
        or planned_steps > 8
        or requested_seeds != 1
    ):
        raise TrainingScaleError(
            "TRAINING_SCALE_ENGINEERING_DIAGNOSTIC_BOUNDARY_VIOLATION"
        )
    plan: dict[str, object] = {
        "schema_version": 3,
        "artifact_type": "verdiwm-training-scale-plan",
        "state": state,
        "stage": stage,
        "training_mode": "probe" if stage in {"engineering_diagnostic", "probe", "smoke"} else "long",
        "evidence_class": (
            "runtime_only"
            if stage in {"engineering_diagnostic", "probe", "smoke"}
            else "diagnostic"
            if stage == "screen"
            else "formal"
        ),
        "quality_eligible": stage in _FORMAL_STAGES,
        "dataset": {
            "input_format": (
                "agibot_native_index_jsonl"
                if native_agibot
                else "sample_manifest_json"
            ),
            "manifest_paths": manifest_paths,
            "train_examples": train_count,
            "validation_examples": val_count,
            "train_episode_count": len(train_episodes),
            "validation_episode_count": len(val_episodes),
            "selected_train_examples": selected_count,
            "selected_train_episode_count": len(selected_episodes),
            "train_validation_episode_overlap": episode_overlap,
            "validation_episode_disjoint": not episode_overlap and val_count > 0,
            "count_unit": (
                "agibot_episode_index_record"
                if native_agibot
                else "sample_manifest_record"
            ),
            "selection_policy": "sha256_episode_balanced_sample_fraction_v2",
            "selection_fraction": fraction,
            "sequence_length": sequence_length,
            "train_manifest_sha256": _sha256(train_path),
            "validation_manifest_sha256": (
                _sha256(validation_path) if validation_path is not None else None
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
            "target_epochs": target_epochs,
            "planned_steps": planned_steps,
            "realized_epochs": (
                planned_steps / steps_per_epoch
                if steps_per_epoch is not None
                else None
            ),
            "checkpoint_eval_steps": checkpoint_steps,
        },
        "replication": {
            "requested_seed_count": requested_seeds,
            "seeds": list(range(seed, seed + requested_seeds)),
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
            (
                "Engineering diagnostic only: full declared input support is bound separately "
                "by the train/validation episode-index digests and episode UIDs. Native indexes "
                "do not define loader windows, so epoch estimates are omitted. This run plans "
                f"{planned_steps} optimizer step(s). The {int(policy['max_steps'])}-step "
                "value is the policy cap, not extra work added to this plan. Loss and checkpoint execution do not "
                "establish model improvement."
                if stage == "engineering_diagnostic"
                else "Stage budgets are derived from deterministic sample counts and effective batch size. "
                "Episode diversity is a hard gate because repeated windows from one episode are not "
                "independent world-model evidence."
            )
        ),
    }
    if native_agibot:
        plan["dataset"]["selected_train_record_ids_binding"] = selected_record_ids_binding  # type: ignore[index]
        plan["dataset"]["selected_train_episode_ids_binding"] = selected_episode_ids_binding  # type: ignore[index]
    else:
        plan["dataset"]["selected_train_record_ids"] = selected_record_ids  # type: ignore[index]
        plan["dataset"]["selected_train_episode_ids"] = list(selected_episodes)  # type: ignore[index]
    if native_agibot:
        if train_summary is None or validation_summary is None:
            raise TrainingScaleError("TRAINING_SCALE_AGIBOT_DOMAIN_SUPPORT_EMPTY")
        domain_ids = sorted(
            set(train_summary["domain_ids"])
            | set(validation_summary["domain_ids"])
        )
        if not domain_ids:
            raise TrainingScaleError("TRAINING_SCALE_AGIBOT_DOMAIN_SUPPORT_EMPTY")
        plan["dataset"]["domain_support"] = {  # type: ignore[index]
            "source": "agibot_native_index_v1",
            "domain_ids": domain_ids,
            "required_action_num_domains": domain_ids[-1] + 1,
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
    validate_training_scale_checkpoint_schedule(plan)
    if len(plan["replication"]["seeds"]) != requested_seeds:  # type: ignore[index]
        raise TrainingScaleError("TRAINING_SCALE_SEED_COUNT_MISMATCH")
    return plan


def validate_training_scale_checkpoint_schedule(
    plan: Mapping[str, object],
) -> None:
    """Require a bounded, ordered schedule that evaluates the final update."""

    updates = plan.get("updates")
    if not isinstance(updates, Mapping):
        raise TrainingScaleError("TRAINING_SCALE_CHECKPOINT_SCHEDULE_INVALID")
    planned_steps = updates.get("planned_steps")
    checkpoint_steps = updates.get("checkpoint_eval_steps")
    if (
        isinstance(planned_steps, bool)
        or not isinstance(planned_steps, int)
        or planned_steps < 1
        or not isinstance(checkpoint_steps, list)
        or not checkpoint_steps
        or any(
            isinstance(step, bool)
            or not isinstance(step, int)
            or step < 1
            or step > planned_steps
            for step in checkpoint_steps
        )
        or checkpoint_steps != sorted(set(checkpoint_steps))
        or checkpoint_steps[-1] != planned_steps
    ):
        raise TrainingScaleError("TRAINING_SCALE_CHECKPOINT_SCHEDULE_INVALID")


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


def _resolve_dataset_inputs(
    *,
    train_manifest: Path | None,
    val_manifest: Path | None,
    dataset_root: Path | None,
) -> tuple[Path, Path | None, bool]:
    if dataset_root is not None:
        if train_manifest is not None or val_manifest is not None:
            raise TrainingScaleError("TRAINING_SCALE_DATASET_INPUTS_CONFLICT")
        root = checked_path(
            Path(dataset_root),
            code="TRAINING_SCALE_AGIBOT_DATASET_ROOT_INVALID",
            error=TrainingScaleError,
        )
        if not root.is_dir():
            raise TrainingScaleError("TRAINING_SCALE_AGIBOT_DATASET_ROOT_INVALID")
        train_path = checked_path(
            root / "train.index.jsonl",
            code="TRAINING_SCALE_TRAIN_MANIFEST_INVALID",
            error=TrainingScaleError,
        )
        validation_path = checked_path(
            root / "val.index.jsonl",
            code="TRAINING_SCALE_VAL_MANIFEST_INVALID",
            error=TrainingScaleError,
        )
        return train_path, validation_path, True

    if train_manifest is None:
        raise TrainingScaleError("TRAINING_SCALE_TRAIN_MANIFEST_REQUIRED")
    train_path = checked_path(
        Path(train_manifest),
        code="TRAINING_SCALE_TRAIN_MANIFEST_INVALID",
        error=TrainingScaleError,
    )
    validation_path = (
        checked_path(
            Path(val_manifest),
            code="TRAINING_SCALE_VAL_MANIFEST_INVALID",
            error=TrainingScaleError,
        )
        if val_manifest is not None
        else None
    )
    native_agibot = train_path.name.endswith(".index.jsonl") or (
        validation_path is not None
        and validation_path.name.endswith(".index.jsonl")
    )
    if native_agibot and validation_path is None:
        raise TrainingScaleError("TRAINING_SCALE_VAL_MANIFEST_REQUIRED")
    if native_agibot and not (
        train_path.name == "train.index.jsonl"
        and validation_path is not None
        and validation_path.name == "val.index.jsonl"
    ):
        raise TrainingScaleError("TRAINING_SCALE_AGIBOT_INDEX_NAMES_INVALID")
    return train_path, validation_path, native_agibot


def _load_samples(
    path: Path,
    error_code: str,
) -> list[Mapping[str, Any]]:
    resolved = checked_path(path, code=error_code, error=TrainingScaleError)
    if resolved.name.endswith(".index.jsonl"):
        raise TrainingScaleError(
            "TRAINING_SCALE_AGIBOT_INDEX_REQUIRES_NATIVE_SUMMARY"
        )
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except OSError as exc:
        raise TrainingScaleError(error_code) from exc
    except json.JSONDecodeError as exc:
        raise TrainingScaleError(error_code) from exc
    if isinstance(payload, Mapping):
        payload = payload.get("samples", payload.get("records"))
    if not isinstance(payload, list) or not payload:
        raise TrainingScaleError(error_code)
    if not all(isinstance(row, Mapping) for row in payload):
        raise TrainingScaleError(error_code)
    return [row for row in payload if isinstance(row, Mapping)]


def _summarize_agibot_index(
    path: Path,
    *,
    expected_split: str | None,
    error_code: str,
) -> dict[str, Any]:
    if expected_split not in {"train", "val"}:
        raise TrainingScaleError(error_code)
    resolved = checked_path(path, code=error_code, error=TrainingScaleError)
    seen_episode_uids: set[str] = set()
    domain_ids: set[int] = set()
    record_ids: list[str] = []
    row_count = 0
    try:
        with resolved.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise TrainingScaleError(f"{error_code}:BLANK_LINE:{line_number}")
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise TrainingScaleError(
                        f"{error_code}:INVALID_JSONL:{line_number}"
                    ) from exc
                if not isinstance(row, Mapping):
                    raise TrainingScaleError(f"{error_code}:ROW_INVALID:{line_number}")
                episode_uid = row.get("episode_uid")
                episode_id = row.get("episode_id")
                domain = row.get("domain")
                domain_id = (
                    domain.get("domain_id")
                    if isinstance(domain, Mapping)
                    else None
                )
                if (
                    row.get("dataset") != "agibot"
                    or row.get("split") != expected_split
                    or not isinstance(episode_uid, str)
                    or not episode_uid.strip()
                    or episode_uid != episode_uid.strip()
                    or isinstance(episode_id, bool)
                    or not isinstance(episode_id, int)
                    or episode_id < 0
                    or not isinstance(domain, Mapping)
                    or domain.get("dataset") != "agibot"
                    or isinstance(domain_id, bool)
                    or not isinstance(domain_id, int)
                    or domain_id < 0
                ):
                    raise TrainingScaleError(
                        f"{error_code}:AGIBOT_ROW_CONTRACT_INVALID:{line_number}"
                    )
                if episode_uid in seen_episode_uids:
                    raise TrainingScaleError(
                        "TRAINING_SCALE_AGIBOT_EPISODE_UID_DUPLICATE"
                    )
                seen_episode_uids.add(episode_uid)
                domain_ids.add(domain_id)
                record_ids.append(_training_record_id(row_count, row))
                row_count += 1
    except OSError as exc:
        raise TrainingScaleError(error_code) from exc
    if row_count == 0:
        raise TrainingScaleError(error_code)
    return {
        "count": row_count,
        "episode_ids": tuple(sorted(seen_episode_uids)),
        "record_ids": tuple(record_ids),
        "domain_ids": tuple(sorted(domain_ids)),
    }


def _episode_ids(samples: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    values = set()
    for row in samples:
        episode_id = row.get("episode_id")
        if not isinstance(episode_id, str) or not episode_id.strip():
            raise TrainingScaleError("TRAINING_SCALE_EPISODE_ID_MISSING")
        values.add(episode_id.strip())
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


def _episode_balanced_subset(
    samples: Sequence[Mapping[str, Any]], count: int
) -> list[Mapping[str, Any]]:
    """Select deterministic episode coverage before repeated windows."""

    return [samples[index] for index in _episode_balanced_indices(samples, count)]


def _episode_balanced_indices(
    samples: Sequence[Mapping[str, Any]], count: int
) -> list[int]:
    """Return selected manifest positions, preserving the existing selection order."""

    ranked = sorted(
        enumerate(samples),
        key=lambda item: hashlib.sha256(
            (
                str(item[1]["episode_id"])
                + "\0"
                + json.dumps(item[1], sort_keys=True, separators=(",", ":"), ensure_ascii=True)
                + f"\0{item[0]}"
            ).encode("utf-8")
        ).hexdigest(),
    )
    by_episode: dict[str, list[int]] = {}
    for index, row in ranked:
        by_episode.setdefault(str(row["episode_id"]), []).append(index)
    episode_order = sorted(
        by_episode,
        key=lambda episode: hashlib.sha256(episode.encode("utf-8")).hexdigest(),
    )
    selected: list[int] = []
    round_index = 0
    while len(selected) < count:
        added = False
        for episode in episode_order:
            rows = by_episode[episode]
            if round_index < len(rows):
                selected.append(rows[round_index])
                added = True
                if len(selected) == count:
                    break
        if not added:
            break
        round_index += 1
    return selected


def _training_record_id(index: int, sample: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        sample, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"train-record-{index + 1:08d}-{digest}"


def _compact_id_binding(values: Sequence[str]) -> dict[str, object]:
    """Bind an ordered identity list without copying it into a large artifact."""

    serialized = ("\n".join(values) + "\n").encode("utf-8")
    return {"count": len(values), "sha256": hashlib.sha256(serialized).hexdigest()}


def _checkpoint_steps(
    steps: int, *, fractions: Sequence[float] = (0.25, 0.5, 0.75, 1.0)
) -> list[int]:
    points = {max(1, math.ceil(steps * fraction)) for fraction in fractions}
    points.add(steps)
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
    if stage == "engineering_diagnostic" and (
        float(policy["dataset_fraction"]) != 1.0
        or int(policy["min_steps"]) < 1
        or int(policy["min_steps"]) > 8
        or int(policy["max_steps"]) < 1
        or int(policy["max_steps"]) > 8
        or int(policy["seed_count"]) != 1
    ):
        raise TrainingScaleError(
            "TRAINING_SCALE_ENGINEERING_DIAGNOSTIC_POLICY_EXCEEDS_BOUND"
        )
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
