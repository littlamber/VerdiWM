"""Training and frozen evaluation smoke components: planning."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping

from wmloop.execute.training_smoke.errors import OrchestratorTrainingEvalSmokeError
from wmloop.execute.training_smoke.metrics import (
    _environment_spec,
    _max_observed_horizon,
    _vendor_environment_name,
)
from wmloop.execute.training_smoke.runtime import _load_yaml_with_pyyaml, _safe_run_name
from wmloop.primitives.registry import PrimitiveRegistry
from wmloop.propose.generator import CURRENT_LIBRARY_VERSION


class _TrainingEvalSmokeProposalClient:
    def __init__(
        self,
        *,
        environment: str,
        primitive: str,
        params: Mapping[str, object],
        metric_name: str,
        budget_estimate_gpu_hours: float | None = None,
        trial_arm: str | None = None,
        trial_seed: int | None = None,
    ) -> None:
        self._environment = environment
        self._primitive = primitive
        self._params = dict(params)
        self._metric_name = metric_name
        self._budget_estimate_gpu_hours = budget_estimate_gpu_hours
        self._trial_arm = trial_arm
        self._trial_seed = trial_seed

    def complete(self, prompt: str) -> str:
        packet = json.loads(prompt)
        failure = packet["failure_report"]
        allowed = {item["name"]: item for item in packet["allowed_primitives"]}
        manifest = allowed.get(self._primitive)
        if manifest is None:
            raise OrchestratorTrainingEvalSmokeError(
                f"M3_TRAINING_EVAL_SMOKE_PRIMITIVE_UNAVAILABLE:{self._primitive}"
            )
        proposal = {
            "proposal_id": self._proposal_id(),
            "round": int(failure["round"]),
            "env": failure["env"],
            "goal_id": failure["goal_id"],
            "based_on_failure": failure["dominant_failure"],
            "interventions": [
                {
                    "layer": manifest["layer"],
                    "primitive": self._primitive,
                    "params": self._params,
                }
            ],
            "falsifiable_prediction": {
                "metric": self._metric_name,
                "horizon": _max_observed_horizon(failure),
                "split": "accept",
                "min_relative_gain": 0.01,
            },
            "budget_estimate_gpu_hours": float(
                self._budget_estimate_gpu_hours
                if self._budget_estimate_gpu_hours is not None
                else manifest["estimated_gpu_hours"]
            ),
            "rationale_ref": f"training_eval_m3_smoke#{failure['dominant_failure']}->{self._primitive}",
            "library_version": packet.get("library_version", CURRENT_LIBRARY_VERSION),
        }
        return json.dumps(
            proposal,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )

    def _proposal_id(self) -> str:
        suffix = ""
        if self._trial_arm is not None or self._trial_seed is not None:
            arm = _safe_run_name(self._trial_arm or "unlabeled")
            seed = "unknown" if self._trial_seed is None else str(self._trial_seed)
            suffix = f"-{arm}-s{seed}"
        return f"m3-{_safe_run_name(self._environment)}-training-eval-smoke-{self._primitive}{suffix}-r1"


def _training_eval_budget_estimate_gpu_hours(
    *,
    registry: PrimitiveRegistry,
    primitive: str,
    train_steps: int,
    replication_count: int,
    horizon_count: int,
    action_following_probe_count: int,
) -> float:
    """Estimate the whole training+eval trial, not only the selected primitive hook."""

    if (
        train_steps < 1
        or replication_count < 1
        or horizon_count < 1
        or action_following_probe_count < 0
    ):
        raise OrchestratorTrainingEvalSmokeError(
            "M3_TRAINING_EVAL_SMOKE_BUDGET_ESTIMATE_INVALID"
        )
    manifest_estimate = float(registry.manifest(primitive).estimated_gpu_hours)
    training_hours = max(0.25, train_steps / 1024.0)
    probe_count = 2 * replication_count + action_following_probe_count
    eval_hours = probe_count * max(0.05, 0.01 * horizon_count)
    overhead_hours = 0.25
    estimated = max(manifest_estimate, training_hours + eval_hours + overhead_hours)
    goal_cap = 48.0
    if estimated > goal_cap:
        estimated = goal_cap
    return round(estimated, 3)


def _official_resume_training_config(
    *,
    vendor_root: Path,
    checkpoint_root: Path,
    environment: str = "push_cube",
    run_name: str,
    total_steps: int,
    batch_size: int = 1,
    val_batch_size: int | None = None,
    train_size: int = 1,
    num_workers: int = 0,
    learning_rate: float | None = None,
    train_steps: int = 1,
) -> dict[str, object]:
    if (
        batch_size < 1
        or (val_batch_size is not None and val_batch_size < 1)
        or train_size < 1
        or num_workers < 0
    ):
        raise OrchestratorTrainingEvalSmokeError(
            "M3_TRAINING_EVAL_SMOKE_ARGUMENT_INVALID"
        )
    if train_steps < 1:
        raise OrchestratorTrainingEvalSmokeError(
            "M3_TRAINING_EVAL_SMOKE_ARGUMENT_INVALID"
        )
    if learning_rate is not None and (
        not math.isfinite(float(learning_rate)) or learning_rate <= 0.0
    ):
        raise OrchestratorTrainingEvalSmokeError(
            "M3_TRAINING_EVAL_SMOKE_ARGUMENT_INVALID"
        )
    _environment_spec(environment)
    base = _load_yaml_with_pyyaml(
        Path(vendor_root)
        / "configs"
        / "envs"
        / f"{_vendor_environment_name(environment)}.yaml"
    )
    model_type = str(base.get("model_type") or "")
    if model_type:
        model_spec = _load_yaml_with_pyyaml(
            Path(vendor_root) / "configs" / "model" / f"{model_type}.yaml"
        )
        model_config = dict(base.get("model_config") or {})
        model_config.update(dict(model_spec.get("model_config") or {}))
        base["model_config"] = model_config
    model_config = dict(base["model_config"])  # type: ignore[index]
    model_config["vae_config"] = [
        str(Path(checkpoint_root).resolve() / "Wan2.1_VAE.pth")
    ]
    model_config["use_flash_attn"] = False
    dataset = dict(base["dataset"])  # type: ignore[index]
    dataset.update(
        {
            "train_size": int(train_size),
            "ind_test_size": 1,
            "ood_test_size": 1,
            "test_cuts": 1,
            "cache_size": 2,
        }
    )
    training = dict(base["training"])  # type: ignore[index]
    training.update(
        {
            "batch_size": int(batch_size),
            "val_batch_size": int(
                val_batch_size if val_batch_size is not None else batch_size
            ),
            **(
                {"learning_rate": float(learning_rate)}
                if learning_rate is not None
                else {}
            ),
            "num_epochs": max(
                1,
                int(base.get("training", {}).get("num_epochs", 1))
                if isinstance(base.get("training"), Mapping)
                else 1,
            ),
            "total_steps": int(total_steps),
            "num_workers": int(num_workers),
            "log_freq": 1,
            "val_freq": 999999,
            "checkpoint_freq": _checkpoint_frequency(train_steps),
            "gen_mode": "parallel",
            "inference_steps": 1,
        }
    )
    return {
        **base,
        "model_config": model_config,
        "dataset": dataset,
        "training": training,
        "wandb": {"project": "wmloop-training-eval-smoke", "run_name": run_name},
        "distributed": {"use_fsdp": False},
    }


def _checkpoint_frequency(train_steps: int) -> int:
    """Keep smoke runs debuggable without making long confirmation runs I/O-bound."""

    if train_steps <= 32:
        return 1
    if train_steps <= 250:
        return 25
    if train_steps <= 1000:
        return 100
    return 500


def _bind_primitive_training_config(
    config: Mapping[str, object],
    *,
    primitive: str,
    params: Mapping[str, object],
) -> dict[str, object]:
    """Make config-facing primitive intent explicit before the runtime hook runs."""

    output = dict(config)
    model_config = dict(output.get("model_config") or {})
    # This key is owned exclusively by latent_motion_prior. Scrub any stale
    # value before binding the currently reviewed intervention.
    model_config.pop("wmloop_latent_motion_prior_weight", None)
    if primitive == "latent_motion_prior":
        value = params.get("weight")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise OrchestratorTrainingEvalSmokeError(
                "M3_TRAINING_EVAL_SMOKE_LATENT_MOTION_WEIGHT_INVALID"
            )
        latent_weight = float(value)
        if (
            not math.isfinite(latent_weight)
            or latent_weight <= 0.0
            or latent_weight > 4.0
        ):
            raise OrchestratorTrainingEvalSmokeError(
                "M3_TRAINING_EVAL_SMOKE_LATENT_MOTION_WEIGHT_INVALID"
            )
        model_config["wmloop_latent_motion_prior_weight"] = latent_weight
    if primitive == "motion_region_reweight":
        value = params.get("weight")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise OrchestratorTrainingEvalSmokeError(
                "M3_TRAINING_EVAL_SMOKE_MOTION_REGION_WEIGHT_INVALID"
            )
        gamma = float(value)
        if not math.isfinite(gamma) or gamma <= 0.0 or gamma > 4.0:
            raise OrchestratorTrainingEvalSmokeError(
                "M3_TRAINING_EVAL_SMOKE_MOTION_REGION_WEIGHT_INVALID"
            )
        model_config["motion_weighting_gamma"] = gamma
    output["model_config"] = model_config
    return output


def _proposal_params(
    *,
    primitive: str,
    weight: float,
    action_balance_blend: float = 0.5,
    action_balance_max_gain: float = 4.0,
    history_noise: float,
    keep_tokens: int = 2,
    frontier_weight: float = 1.0,
    event_weight: float = 4.0,
    event_quantile: float = 0.75,
    event_visual_blend: float = 0.7,
    next_forcing_chunks: int = 2,
    next_forcing_steps: int = 1,
    next_forcing_lr: float = 0.00001,
    reward_weight: float = 0.5,
    inv_dyn_steps: int = 1,
    inv_dyn_lr: float = 0.00001,
    memory_slots: int = 16,
    memory_weight: float = 0.2,
    anchor_every: int = 8,
    anchor_weight: float = 0.2,
    guidance_start: float = 1.0,
    guidance_end: float = 1.5,
    wmsd_teacher_ema: float = 0.9,
    wmsd_steps: int = 1,
    wmsd_lr: float = 0.00001,
    self_forcing_rollout_horizon: int = 4,
    self_forcing_steps: int = 1,
    self_forcing_lr: float = 0.00001,
) -> dict[str, object]:
    if primitive in {"latent_motion_prior", "motion_region_reweight"}:
        if (
            not math.isfinite(float(weight))
            or weight <= 0.0
            or (primitive == "motion_region_reweight" and weight > 4.0)
        ):
            raise OrchestratorTrainingEvalSmokeError(
                "M3_TRAINING_EVAL_SMOKE_WEIGHT_INVALID"
            )
        return {"weight": weight}
    if primitive == "dino_rep_injection":
        if not math.isfinite(float(weight)) or weight <= 0.0 or weight > 1.0:
            raise OrchestratorTrainingEvalSmokeError(
                "M3_TRAINING_EVAL_SMOKE_DINO_WEIGHT_INVALID"
            )
        return {"injection_weight": float(weight)}
    if primitive == "action_contrastive_finetune":
        if not math.isfinite(float(weight)) or weight <= 0.0 or weight > 1.0:
            raise OrchestratorTrainingEvalSmokeError(
                "M3_TRAINING_EVAL_SMOKE_ACTION_CONTRASTIVE_WEIGHT_INVALID"
            )
        return {"weight": float(weight)}
    if primitive == "action_dimension_balancing":
        if (
            not math.isfinite(float(action_balance_blend))
            or action_balance_blend <= 0.0
            or action_balance_blend > 1.0
        ):
            raise OrchestratorTrainingEvalSmokeError(
                "M3_TRAINING_EVAL_SMOKE_ACTION_BALANCE_BLEND_INVALID"
            )
        if (
            not math.isfinite(float(action_balance_max_gain))
            or action_balance_max_gain < 1.0
            or action_balance_max_gain > 8.0
        ):
            raise OrchestratorTrainingEvalSmokeError(
                "M3_TRAINING_EVAL_SMOKE_ACTION_BALANCE_MAX_GAIN_INVALID"
            )
        return {
            "blend": float(action_balance_blend),
            "max_gain": float(action_balance_max_gain),
        }
    if primitive == "history_noise_schedule":
        if (
            not math.isfinite(float(history_noise))
            or history_noise < 0.0
            or history_noise > 1.0
        ):
            raise OrchestratorTrainingEvalSmokeError(
                "M3_TRAINING_EVAL_SMOKE_HISTORY_NOISE_INVALID"
            )
        return {"history_noise": float(history_noise)}
    if primitive == "drift_token_trim":
        if keep_tokens < 1:
            raise OrchestratorTrainingEvalSmokeError(
                "M3_TRAINING_EVAL_SMOKE_KEEP_TOKENS_INVALID"
            )
        return {"keep_tokens": int(keep_tokens)}
    if primitive == "mixture_reweight":
        if not math.isfinite(float(frontier_weight)) or frontier_weight < 0.0:
            raise OrchestratorTrainingEvalSmokeError(
                "M3_TRAINING_EVAL_SMOKE_FRONTIER_WEIGHT_INVALID"
            )
        return {"frontier_weight": float(frontier_weight)}
    if primitive == "event_window_reweight":
        if (
            not math.isfinite(float(event_weight))
            or event_weight <= 0.0
            or event_weight > 16.0
        ):
            raise OrchestratorTrainingEvalSmokeError(
                "M3_TRAINING_EVAL_SMOKE_EVENT_WEIGHT_INVALID"
            )
        if (
            not math.isfinite(float(event_quantile))
            or event_quantile < 0.5
            or event_quantile > 0.95
        ):
            raise OrchestratorTrainingEvalSmokeError(
                "M3_TRAINING_EVAL_SMOKE_EVENT_QUANTILE_INVALID"
            )
        if (
            not math.isfinite(float(event_visual_blend))
            or event_visual_blend < 0.0
            or event_visual_blend > 1.0
        ):
            raise OrchestratorTrainingEvalSmokeError(
                "M3_TRAINING_EVAL_SMOKE_EVENT_VISUAL_BLEND_INVALID"
            )
        return {
            "event_weight": float(event_weight),
            "event_quantile": float(event_quantile),
            "visual_motion_blend": float(event_visual_blend),
        }
    if primitive == "next_forcing":
        if next_forcing_chunks < 2 or next_forcing_chunks > 8:
            raise OrchestratorTrainingEvalSmokeError(
                "M3_TRAINING_EVAL_SMOKE_NEXT_FORCING_CHUNKS_INVALID"
            )
        if next_forcing_steps < 1:
            raise OrchestratorTrainingEvalSmokeError(
                "M3_TRAINING_EVAL_SMOKE_NEXT_FORCING_STEPS_INVALID"
            )
        if not math.isfinite(float(next_forcing_lr)) or next_forcing_lr <= 0.0:
            raise OrchestratorTrainingEvalSmokeError(
                "M3_TRAINING_EVAL_SMOKE_NEXT_FORCING_LR_INVALID"
            )
        return {
            "chunks": int(next_forcing_chunks),
            "steps": int(next_forcing_steps),
            "lr": float(next_forcing_lr),
        }
    if primitive == "inv_dyn_reward_finetune":
        if not math.isfinite(float(reward_weight)) or reward_weight <= 0.0:
            raise OrchestratorTrainingEvalSmokeError(
                "M3_TRAINING_EVAL_SMOKE_INV_DYN_REWARD_WEIGHT_INVALID"
            )
        if inv_dyn_steps < 1:
            raise OrchestratorTrainingEvalSmokeError(
                "M3_TRAINING_EVAL_SMOKE_INV_DYN_STEPS_INVALID"
            )
        if not math.isfinite(float(inv_dyn_lr)) or inv_dyn_lr <= 0.0:
            raise OrchestratorTrainingEvalSmokeError(
                "M3_TRAINING_EVAL_SMOKE_INV_DYN_LR_INVALID"
            )
        return {
            "reward_weight": float(reward_weight),
            "steps": int(inv_dyn_steps),
            "lr": float(inv_dyn_lr),
        }
    if primitive == "latent_spatial_memory":
        if memory_slots < 1 or memory_slots > 128:
            raise OrchestratorTrainingEvalSmokeError(
                "M3_TRAINING_EVAL_SMOKE_MEMORY_SLOTS_INVALID"
            )
        if (
            not math.isfinite(float(memory_weight))
            or memory_weight <= 0.0
            or memory_weight > 1.0
        ):
            raise OrchestratorTrainingEvalSmokeError(
                "M3_TRAINING_EVAL_SMOKE_MEMORY_WEIGHT_INVALID"
            )
        return {
            "memory_slots": int(memory_slots),
            "memory_weight": float(memory_weight),
        }
    if primitive == "first_frame_anchor":
        if anchor_every < 4 or anchor_every > 32:
            raise OrchestratorTrainingEvalSmokeError(
                "M3_TRAINING_EVAL_SMOKE_ANCHOR_EVERY_INVALID"
            )
        if (
            not math.isfinite(float(anchor_weight))
            or anchor_weight < 0.0
            or anchor_weight > 1.0
        ):
            raise OrchestratorTrainingEvalSmokeError(
                "M3_TRAINING_EVAL_SMOKE_ANCHOR_WEIGHT_INVALID"
            )
        return {
            "anchor_every": int(anchor_every),
            "anchor_weight": float(anchor_weight),
        }
    if primitive == "cfg_guidance_schedule":
        if not math.isfinite(float(guidance_start)) or guidance_start < 0.0:
            raise OrchestratorTrainingEvalSmokeError(
                "M3_TRAINING_EVAL_SMOKE_GUIDANCE_START_INVALID"
            )
        if not math.isfinite(float(guidance_end)) or guidance_end < 0.0:
            raise OrchestratorTrainingEvalSmokeError(
                "M3_TRAINING_EVAL_SMOKE_GUIDANCE_END_INVALID"
            )
        return {
            "guidance_start": float(guidance_start),
            "guidance_end": float(guidance_end),
        }
    if primitive == "wmsd_self_distill":
        if (
            not math.isfinite(float(wmsd_teacher_ema))
            or wmsd_teacher_ema <= 0.0
            or wmsd_teacher_ema > 1.0
        ):
            raise OrchestratorTrainingEvalSmokeError(
                "M3_TRAINING_EVAL_SMOKE_WMSD_TEACHER_EMA_INVALID"
            )
        if wmsd_steps < 1:
            raise OrchestratorTrainingEvalSmokeError(
                "M3_TRAINING_EVAL_SMOKE_WMSD_STEPS_INVALID"
            )
        if not math.isfinite(float(wmsd_lr)) or wmsd_lr <= 0.0:
            raise OrchestratorTrainingEvalSmokeError(
                "M3_TRAINING_EVAL_SMOKE_WMSD_LR_INVALID"
            )
        return {
            "teacher_ema": float(wmsd_teacher_ema),
            "steps": int(wmsd_steps),
            "lr": float(wmsd_lr),
        }
    if primitive == "self_forcing_finetune":
        if self_forcing_rollout_horizon < 2:
            raise OrchestratorTrainingEvalSmokeError(
                "M3_TRAINING_EVAL_SMOKE_SELF_FORCING_HORIZON_INVALID"
            )
        if self_forcing_steps < 1:
            raise OrchestratorTrainingEvalSmokeError(
                "M3_TRAINING_EVAL_SMOKE_SELF_FORCING_STEPS_INVALID"
            )
        if not math.isfinite(float(self_forcing_lr)) or self_forcing_lr <= 0.0:
            raise OrchestratorTrainingEvalSmokeError(
                "M3_TRAINING_EVAL_SMOKE_SELF_FORCING_LR_INVALID"
            )
        return {
            "rollout_horizon": int(self_forcing_rollout_horizon),
            "steps": int(self_forcing_steps),
            "lr": float(self_forcing_lr),
        }
    raise OrchestratorTrainingEvalSmokeError(
        f"M3_TRAINING_EVAL_SMOKE_PRIMITIVE_UNSUPPORTED:{primitive}"
    )
