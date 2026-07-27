from __future__ import annotations

import difflib
import json
import math
from pathlib import Path
from typing import Mapping


def apply(repo_worktree: str, params: Mapping[str, object]) -> str:
    worktree = Path(repo_worktree)
    rollout_horizon, steps, lr = _validate_params(params)
    trainer_path = worktree / "acwm" / "trainer" / "train_dynamics.py"
    original = trainer_path.read_text(encoding="utf-8")
    patched = _patch_trainer(original)
    sidecar = _sidecar_payload(rollout_horizon=rollout_horizon, steps=steps, lr=lr)
    return "\n".join(
        part.rstrip("\n")
        for part in (
            _unified_diff("acwm/trainer/train_dynamics.py", original, patched),
            _optional_new_file_diff(
                worktree,
                "acwm/wmloop_hooks/__init__.py",
                '"""Runtime hooks materialized by reviewed wm-loop primitive templates."""\n',
            ),
            _new_file_diff("acwm/wmloop_hooks/self_forcing_finetune.py", _HOOK_MODULE),
            _new_file_diff(
                "wmloop_interventions/self_forcing_finetune.json",
                json.dumps(sidecar, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            ),
        )
        if part.strip()
    ) + "\n"


def _validate_params(params: Mapping[str, object]) -> tuple[int, int, float]:
    horizon_value = params.get("rollout_horizon")
    steps_value = params.get("steps")
    lr_value = params.get("lr")
    if not isinstance(horizon_value, int) or isinstance(horizon_value, bool) or horizon_value < 2:
        raise ValueError("SELF_FORCING_ROLLOUT_HORIZON_INVALID")
    if not isinstance(steps_value, int) or isinstance(steps_value, bool) or steps_value < 1:
        raise ValueError("SELF_FORCING_STEPS_INVALID")
    if not isinstance(lr_value, (int, float)) or isinstance(lr_value, bool):
        raise ValueError("SELF_FORCING_LR_INVALID")
    lr = float(lr_value)
    if not math.isfinite(lr) or lr <= 0.0:
        raise ValueError("SELF_FORCING_LR_INVALID")
    return int(horizon_value), int(steps_value), lr


def _patch_trainer(original: str) -> str:
    import_marker = "from acwm.utils.visualization import visualize_layout\n"
    import_replacement = (
        "from acwm.utils.visualization import visualize_layout\n"
        "from acwm.wmloop_hooks.self_forcing_finetune import apply_self_forcing_loss\n"
    )
    clean_loss_marker = (
        "            loss = model.module.training_loss(z, action) if hasattr(model, 'module') else model.training_loss(z, action)\n"
        "            loss.backward()\n"
    )
    clean_loss_replacement = (
        "            wmloop_primitive_metrics = {}\n"
        "            wmloop_base_model = model.module if hasattr(model, 'module') else model\n"
        "            loss = wmloop_base_model.training_loss(z, action)\n"
        "            loss, wmloop_self_forcing_metrics = apply_self_forcing_loss(loss, wmloop_base_model, z, action)\n"
        "            wmloop_primitive_metrics.update(wmloop_self_forcing_metrics)\n"
        "            loss.backward()\n"
    )
    existing_metrics_backward_marker = "            loss.backward()\n"
    existing_metrics_backward_replacement = (
        "            wmloop_base_model = model.module if hasattr(model, 'module') else model\n"
        "            loss, wmloop_self_forcing_metrics = apply_self_forcing_loss(loss, wmloop_base_model, z, action)\n"
        "            wmloop_primitive_metrics.update(wmloop_self_forcing_metrics)\n"
        "            loss.backward()\n"
    )
    clean_log_marker = (
        "                    wandb.log({\n"
        "                        \"train/loss\": loss.item(),\n"
        "                        \"train/grad_norm\": grad_norm,\n"
        "                        \"train/epoch\": epoch,\n"
        "                        \"time/data_loading\": data_time,\n"
        "                        \"time/training_step\": train_step_time,\n"
        "                        \"time/seconds_per_step\": step_time_total,\n"
        "                    }, step=step)\n"
    )
    clean_log_replacement = (
        "                    train_metrics = {\n"
        "                        \"train/loss\": loss.item(),\n"
        "                        \"train/grad_norm\": grad_norm,\n"
        "                        \"train/epoch\": epoch,\n"
        "                        \"time/data_loading\": data_time,\n"
        "                        \"time/training_step\": train_step_time,\n"
        "                        \"time/seconds_per_step\": step_time_total,\n"
        "                    }\n"
        "                    train_metrics.update(wmloop_primitive_metrics)\n"
        "                    wandb.log(train_metrics, step=step)\n"
    )
    patched = original
    if import_marker not in patched:
        raise ValueError("SELF_FORCING_IMPORT_ANCHOR_MISSING")
    patched = patched.replace(import_marker, import_replacement, 1)
    if clean_loss_marker in patched:
        patched = patched.replace(clean_loss_marker, clean_loss_replacement, 1)
    elif "wmloop_primitive_metrics" in patched and existing_metrics_backward_marker in patched:
        patched = patched.replace(existing_metrics_backward_marker, existing_metrics_backward_replacement, 1)
    else:
        raise ValueError("SELF_FORCING_LOSS_ANCHOR_MISSING")
    if clean_log_marker in patched:
        patched = patched.replace(clean_log_marker, clean_log_replacement, 1)
    elif "wandb.log(train_metrics, step=step)" not in patched:
        raise ValueError("SELF_FORCING_LOG_ANCHOR_MISSING")
    return patched


def _sidecar_payload(*, rollout_horizon: int, steps: int, lr: float) -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_type": "wmloop-materialized-primitive-smoke",
        "primitive": "self_forcing_finetune",
        "layer": "L3",
        "hook": "H3",
        "materialization_state": "acwm_runtime_hook_smoke",
        "params": {"rollout_horizon": rollout_horizon, "steps": steps, "lr": lr},
        "runtime_hook": {
            "module": "acwm.wmloop_hooks.self_forcing_finetune",
            "function": "apply_self_forcing_loss",
            "trainer_patch": "acwm/trainer/train_dynamics.py",
        },
        "runtime_hook_paths": [
            "acwm/trainer/train_dynamics.py",
            "acwm/wmloop_hooks/self_forcing_finetune.py",
        ],
        "intent_to_code_contract": {
            "method_intent": (
                "Reduce train-inference mismatch by training the dynamics model against a bounded latent rollout "
                "that is produced by the current model under the same action prefix."
            ),
            "runtime_behavior": (
                "Adds an H3 auxiliary loss after the base ACWM training_loss. The hook performs one differentiable "
                "self-forced denoising pass over a latent prefix and penalizes deviation from the held-out latent target."
            ),
            "declared_proxy": (
                "Implements a latent-prefix self-forcing proxy because ACWM-Phys exposes differentiable DiT and "
                "scheduler objects in training, but not a pixel rollout loss inside train_dynamics."
            ),
            "not_claimed": [
                "does not reproduce the full Self-Forcing paper algorithm",
                "does not decode generated pixels inside the training loop",
                "does not change ACWM evaluator files or held-out splits",
                "does not prove train-inference mismatch improvement until GPU smoke and paired eval pass",
            ],
        },
        "notes": [
            "Adds a real H3 train-objective hook instead of a sidecar-only request.",
            "The rollout_horizon parameter is consumed by the hook; steps and lr remain orchestrator training-budget metadata.",
            "The hook is trial-worktree only and does not modify frozen evaluator files.",
        ],
    }


def _unified_diff(path: str, original: str, patched: str) -> str:
    if original == patched:
        raise ValueError("SELF_FORCING_PATCH_EMPTY")
    return "".join(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            patched.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
        )
    )


def _new_file_diff(path: str, content: str) -> str:
    lines = content.splitlines()
    return "\n".join(
        [
            f"diff --git a/{path} b/{path}",
            "new file mode 100644",
            "--- /dev/null",
            f"+++ b/{path}",
            f"@@ -0,0 +1,{len(lines)} @@",
            *(f"+{line}" for line in lines),
            "",
        ]
    )


def _optional_new_file_diff(worktree: Path, path: str, content: str) -> str:
    if (worktree / path).exists():
        return ""
    return _new_file_diff(path, content)


_HOOK_MODULE = '''from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping

import torch


_SIDECAR = Path("wmloop_interventions/self_forcing_finetune.json")


@dataclass(frozen=True)
class SelfForcingConfig:
    rollout_horizon: int
    steps: int
    lr: float


def apply_self_forcing_loss(
    base_loss: torch.Tensor,
    model: object,
    latents: torch.Tensor,
    actions: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Add a bounded latent self-forcing objective to the ACWM training path."""

    config = _configured_params()
    if latents.ndim != 5:
        raise RuntimeError("SELF_FORCING_LATENT_SHAPE_INVALID")
    if actions.ndim != 3:
        raise RuntimeError("SELF_FORCING_ACTION_SHAPE_INVALID")
    if latents.shape[1] < 2:
        return base_loss, _zero_metrics(config)

    inner_model = getattr(model, "model", None)
    scheduler = getattr(model, "scheduler", None)
    if not callable(inner_model):
        raise RuntimeError("SELF_FORCING_INNER_MODEL_MISSING")
    _validate_scheduler(scheduler)

    latent_frames = _latent_frames_for_rollout_horizon(
        model,
        rollout_horizon=config.rollout_horizon,
        available_frames=latents.shape[1],
    )
    if latent_frames < 2:
        return base_loss, _zero_metrics(config)

    prefix = latents[:, :latent_frames].float()
    action_prefix = _slice_actions_for_latent_window(model, actions, latent_frames=latent_frames)
    t_values = _self_forcing_timesteps(scheduler, prefix)
    noisy_prefix, _noise = scheduler.add_independent_noise(prefix, t_values)
    noisy_prefix = noisy_prefix.clone()
    noisy_prefix[:, 0] = prefix[:, 0]

    v_pred = inner_model(noisy_prefix.to(dtype=latents.dtype), t_values, action_prefix)
    forced_prefix = scheduler.step(v_pred.float(), t_values, noisy_prefix.float(), to_final=True)
    forced_prefix = forced_prefix.clone()
    forced_prefix[:, 0] = prefix[:, 0]

    aux_loss = torch.nn.functional.mse_loss(forced_prefix[:, 1:], prefix.detach()[:, 1:])
    aux_weight = 1.0 / float(max(1, latent_frames - 1))
    total = base_loss + base_loss.new_tensor(aux_weight) * aux_loss.to(dtype=base_loss.dtype)
    return total, {
        "train/wmloop_self_forcing_aux_loss": float(aux_loss.detach().cpu()),
        "train/wmloop_self_forcing_aux_weight": float(aux_weight),
        "train/wmloop_self_forcing_latent_frames": float(latent_frames),
        "train/wmloop_self_forcing_rollout_horizon": float(config.rollout_horizon),
        "train/wmloop_self_forcing_lr": float(config.lr),
    }


def _zero_metrics(config: SelfForcingConfig) -> dict[str, float]:
    return {
        "train/wmloop_self_forcing_aux_loss": 0.0,
        "train/wmloop_self_forcing_aux_weight": 0.0,
        "train/wmloop_self_forcing_latent_frames": 0.0,
        "train/wmloop_self_forcing_rollout_horizon": float(config.rollout_horizon),
        "train/wmloop_self_forcing_lr": float(config.lr),
    }


def _latent_frames_for_rollout_horizon(model: object, *, rollout_horizon: int, available_frames: int) -> int:
    compress_rate = _action_compress_rate(model)
    latent_frames = ((int(rollout_horizon) - 1) // compress_rate) + 1
    return max(2, min(int(available_frames), int(latent_frames)))


def _action_compress_rate(model: object) -> int:
    nested = getattr(model, "model", None)
    for candidate in (nested, model):
        value = getattr(candidate, "action_compress_rate", None)
        if isinstance(value, int) and value >= 1:
            return value
    model_config = getattr(model, "model_config", None)
    if isinstance(model_config, Mapping):
        value = model_config.get("action_compress_rate")
        if isinstance(value, int) and value >= 1:
            return value
    return 4


def _slice_actions_for_latent_window(model: object, actions: torch.Tensor, *, latent_frames: int) -> torch.Tensor:
    compress_rate = _action_compress_rate(model)
    requested = compress_rate * max(0, latent_frames - 1) + 1
    length = max(1, min(actions.shape[1], requested))
    return actions[:, :length]


def _self_forcing_timesteps(scheduler: object, prefix: torch.Tensor) -> torch.Tensor:
    timesteps = getattr(scheduler, "timesteps")
    if not isinstance(timesteps, torch.Tensor) or timesteps.ndim != 1 or timesteps.numel() < 1:
        raise RuntimeError("SELF_FORCING_TIMESTEPS_INVALID")
    index = min(max(0, int(timesteps.numel()) // 2), int(timesteps.numel()) - 1)
    t_value = timesteps[index].to(device=prefix.device)
    t_values = torch.full((prefix.shape[0], prefix.shape[1]), t_value, device=prefix.device, dtype=timesteps.dtype)
    t_values[:, 0] = torch.zeros((), device=prefix.device, dtype=timesteps.dtype)
    return t_values


def _validate_scheduler(scheduler: object) -> None:
    if scheduler is None:
        raise RuntimeError("SELF_FORCING_SCHEDULER_MISSING")
    if not callable(getattr(scheduler, "add_independent_noise", None)):
        raise RuntimeError("SELF_FORCING_SCHEDULER_ADD_NOISE_MISSING")
    if not callable(getattr(scheduler, "step", None)):
        raise RuntimeError("SELF_FORCING_SCHEDULER_STEP_MISSING")
    timesteps = getattr(scheduler, "timesteps", None)
    if not isinstance(timesteps, torch.Tensor) or timesteps.ndim != 1 or timesteps.numel() < 1:
        raise RuntimeError("SELF_FORCING_TIMESTEPS_INVALID")
    sigmas = getattr(scheduler, "sigmas", None)
    if not isinstance(sigmas, torch.Tensor) or sigmas.ndim != 1 or sigmas.numel() != timesteps.numel():
        raise RuntimeError("SELF_FORCING_SIGMAS_INVALID")


@lru_cache(maxsize=1)
def _configured_params() -> SelfForcingConfig:
    payload = _load_sidecar()
    params = payload.get("params")
    if not isinstance(params, Mapping):
        raise RuntimeError("SELF_FORCING_PARAMS_INVALID")
    horizon = params.get("rollout_horizon")
    steps = params.get("steps")
    lr = params.get("lr")
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 2:
        raise RuntimeError("SELF_FORCING_ROLLOUT_HORIZON_INVALID")
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
        raise RuntimeError("SELF_FORCING_STEPS_INVALID")
    if not isinstance(lr, (int, float)) or isinstance(lr, bool):
        raise RuntimeError("SELF_FORCING_LR_INVALID")
    lr_value = float(lr)
    if not math.isfinite(lr_value) or lr_value <= 0.0:
        raise RuntimeError("SELF_FORCING_LR_INVALID")
    return SelfForcingConfig(rollout_horizon=int(horizon), steps=int(steps), lr=lr_value)


def _load_sidecar() -> Mapping[str, object]:
    try:
        payload = json.loads(_SIDECAR.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("SELF_FORCING_SIDECAR_UNREADABLE") from exc
    if not isinstance(payload, Mapping):
        raise RuntimeError("SELF_FORCING_SIDECAR_INVALID")
    if (
        payload.get("artifact_type") != "wmloop-materialized-primitive-smoke"
        or payload.get("primitive") != "self_forcing_finetune"
        or payload.get("hook") != "H3"
        or payload.get("materialization_state") != "acwm_runtime_hook_smoke"
    ):
        raise RuntimeError("SELF_FORCING_SIDECAR_INVALID")
    return payload
'''
