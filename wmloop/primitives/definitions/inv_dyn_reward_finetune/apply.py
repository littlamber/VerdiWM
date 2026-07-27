from __future__ import annotations

import difflib
import json
import math
from pathlib import Path
from typing import Mapping


def apply(repo_worktree: str, params: Mapping[str, object]) -> str:
    worktree = Path(repo_worktree)
    reward_weight, steps, lr = _validate_params(params)
    trainer_path = worktree / "acwm" / "trainer" / "train_dynamics.py"
    original = trainer_path.read_text(encoding="utf-8")
    patched = _patch_trainer(original)
    sidecar = _sidecar_payload(reward_weight=reward_weight, steps=steps, lr=lr)
    return "\n".join(
        part.rstrip("\n")
        for part in (
            _unified_diff("acwm/trainer/train_dynamics.py", original, patched),
            _optional_new_file_diff(
                worktree,
                "acwm/wmloop_hooks/__init__.py",
                '"""Runtime hooks materialized by reviewed wm-loop primitive templates."""\n',
            ),
            _new_file_diff("acwm/wmloop_hooks/inv_dyn_reward_finetune.py", _HOOK_MODULE),
            _new_file_diff(
                "wmloop_interventions/inv_dyn_reward_finetune.json",
                json.dumps(sidecar, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            ),
        )
        if part.strip()
    ) + "\n"


def _validate_params(params: Mapping[str, object]) -> tuple[float, int, float]:
    reward_value = params.get("reward_weight")
    steps_value = params.get("steps")
    lr_value = params.get("lr")
    if not isinstance(reward_value, (int, float)) or isinstance(reward_value, bool):
        raise ValueError("INV_DYN_REWARD_WEIGHT_INVALID")
    reward_weight = float(reward_value)
    if not math.isfinite(reward_weight) or reward_weight <= 0.0:
        raise ValueError("INV_DYN_REWARD_WEIGHT_INVALID")
    if not isinstance(steps_value, int) or isinstance(steps_value, bool) or steps_value < 1:
        raise ValueError("INV_DYN_STEPS_INVALID")
    if not isinstance(lr_value, (int, float)) or isinstance(lr_value, bool):
        raise ValueError("INV_DYN_LR_INVALID")
    lr = float(lr_value)
    if not math.isfinite(lr) or lr <= 0.0:
        raise ValueError("INV_DYN_LR_INVALID")
    return reward_weight, int(steps_value), lr


def _patch_trainer(original: str) -> str:
    import_marker = "from acwm.utils.visualization import visualize_layout\n"
    import_replacement = (
        "from acwm.utils.visualization import visualize_layout\n"
        "from acwm.wmloop_hooks.inv_dyn_reward_finetune import apply_inv_dyn_reward_loss\n"
    )
    loss_marker = (
        "            loss = model.module.training_loss(z, action) if hasattr(model, 'module') else model.training_loss(z, action)\n"
    )
    loss_replacement = (
        "            loss = model.module.training_loss(z, action) if hasattr(model, 'module') else model.training_loss(z, action)\n"
        "            wmloop_primitive_metrics = {}\n"
        "            loss, wmloop_inv_dyn_metrics = apply_inv_dyn_reward_loss(loss, z, action)\n"
        "            wmloop_primitive_metrics.update(wmloop_inv_dyn_metrics)\n"
    )
    log_marker = (
        "                    wandb.log({\n"
        "                        \"train/loss\": loss.item(),\n"
        "                        \"train/grad_norm\": grad_norm,\n"
        "                        \"train/epoch\": epoch,\n"
        "                        \"time/data_loading\": data_time,\n"
        "                        \"time/training_step\": train_step_time,\n"
        "                        \"time/seconds_per_step\": step_time_total,\n"
        "                    }, step=step)\n"
    )
    log_replacement = (
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
        raise ValueError("INV_DYN_IMPORT_ANCHOR_MISSING")
    patched = patched.replace(import_marker, import_replacement, 1)
    if loss_marker not in patched:
        raise ValueError("INV_DYN_LOSS_ANCHOR_MISSING")
    patched = patched.replace(loss_marker, loss_replacement, 1)
    if log_marker in patched:
        patched = patched.replace(log_marker, log_replacement, 1)
    elif "train_metrics.update(wmloop_primitive_metrics)" not in patched:
        raise ValueError("INV_DYN_LOG_ANCHOR_MISSING")
    return patched


def _sidecar_payload(*, reward_weight: float, steps: int, lr: float) -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_type": "wmloop-materialized-primitive-smoke",
        "primitive": "inv_dyn_reward_finetune",
        "layer": "L3",
        "hook": "H3",
        "materialization_state": "acwm_runtime_hook_smoke",
        "params": {"reward_weight": reward_weight, "steps": steps, "lr": lr},
        "runtime_hook": {
            "module": "acwm.wmloop_hooks.inv_dyn_reward_finetune",
            "function": "apply_inv_dyn_reward_loss",
            "trainer_patch": "acwm/trainer/train_dynamics.py",
        },
        "runtime_hook_paths": [
            "acwm/trainer/train_dynamics.py",
            "acwm/wmloop_hooks/inv_dyn_reward_finetune.py",
        ],
        "intent_to_code_contract": {
            "method_intent": "Improve action binding by making latent transitions predictive of action labels.",
            "runtime_behavior": (
                "Adds an H3 auxiliary inverse-dynamics reward loss immediately after ACWM training_loss. "
                "The hook fits a detached ridge inverse head per batch and backpropagates the action-prediction "
                "error through latent deltas."
            ),
            "declared_proxy": (
                "Uses an online linear inverse-dynamics head over spatially pooled latent deltas as the executable "
                "proxy for RLIR-style inverse-dynamics reward fine-tuning."
            ),
            "not_claimed": [
                "does not add a persistent inverse-dynamics checkpointed head",
                "does not change ACWM evaluator files or held-out splits",
                "does not prove action-binding improvement until the inverse-dynamics confidence gate and paired eval pass",
            ],
        },
        "notes": [
            "Adds a real train-objective hook instead of a sidecar-only request.",
            "The lr and steps parameters are recorded for scheduler/orchestrator policy; this smoke hook consumes reward_weight.",
            "The hook is trial-worktree only and does not modify frozen evaluator files.",
        ],
    }


def _unified_diff(path: str, original: str, patched: str) -> str:
    if original == patched:
        raise ValueError("INV_DYN_PATCH_EMPTY")
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
from functools import lru_cache
from pathlib import Path
from typing import Mapping

import torch


_SIDECAR = Path("wmloop_interventions/inv_dyn_reward_finetune.json")


def apply_inv_dyn_reward_loss(
    base_loss: torch.Tensor,
    latents: torch.Tensor,
    actions: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Add an online inverse-dynamics auxiliary loss over latent deltas."""

    weight = _configured_weight()
    if latents.ndim != 5:
        raise RuntimeError("INV_DYN_LATENT_SHAPE_INVALID")
    if actions.ndim != 3:
        raise RuntimeError("INV_DYN_ACTION_SHAPE_INVALID")
    if latents.shape[1] < 2 or actions.shape[1] < 1:
        return base_loss, {
            "train/wmloop_inv_dyn_reward_loss": 0.0,
            "train/wmloop_inv_dyn_reward_weight": float(weight),
            "train/wmloop_inv_dyn_r2_proxy": 0.0,
        }

    features = _latent_delta_features(latents)
    targets = _aligned_actions(actions, target_frames=features.shape[1]).to(device=features.device, dtype=features.dtype)
    flat_x = features.reshape(-1, features.shape[-1])
    flat_y = targets.reshape(-1, targets.shape[-1]).detach()
    if flat_x.shape[0] < 2:
        return base_loss, {
            "train/wmloop_inv_dyn_reward_loss": 0.0,
            "train/wmloop_inv_dyn_reward_weight": float(weight),
            "train/wmloop_inv_dyn_r2_proxy": 0.0,
        }

    coeff, bias = _ridge_inverse_head(flat_x.detach(), flat_y)
    pred = flat_x @ coeff + bias
    aux_loss = torch.nn.functional.mse_loss(pred, flat_y)
    total = base_loss + base_loss.new_tensor(weight) * aux_loss.to(dtype=base_loss.dtype)
    variance = flat_y.float().var(unbiased=False).clamp_min(1e-8)
    r2_proxy = 1.0 - float(aux_loss.detach().float().cpu() / variance.detach().cpu())
    return total, {
        "train/wmloop_inv_dyn_reward_loss": float(aux_loss.detach().cpu()),
        "train/wmloop_inv_dyn_reward_weight": float(weight),
        "train/wmloop_inv_dyn_r2_proxy": float(r2_proxy),
    }


def _latent_delta_features(latents: torch.Tensor) -> torch.Tensor:
    delta = latents.float()[:, 1:] - latents.float()[:, :-1]
    pooled = delta.mean(dim=(3, 4))
    scale = pooled.detach().float().std(dim=-1, keepdim=True).clamp_min(1e-6)
    return pooled / scale


def _aligned_actions(actions: torch.Tensor, *, target_frames: int) -> torch.Tensor:
    if actions.shape[1] >= target_frames:
        return actions[:, :target_frames]
    pad = actions[:, -1:].repeat(1, target_frames - actions.shape[1], 1)
    return torch.cat([actions, pad], dim=1)


def _ridge_inverse_head(features: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    ones = torch.ones(features.shape[0], 1, device=features.device, dtype=features.dtype)
    design = torch.cat([features, ones], dim=-1)
    regularizer = 1e-3 * torch.eye(design.shape[-1], device=design.device, dtype=design.dtype)
    lhs = design.transpose(0, 1) @ design + regularizer
    rhs = design.transpose(0, 1) @ targets
    try:
        coeff = torch.linalg.solve(lhs.float(), rhs.float()).to(dtype=features.dtype)
    except RuntimeError:
        coeff = torch.linalg.pinv(lhs.float()) @ rhs.float()
        coeff = coeff.to(dtype=features.dtype)
    return coeff[:-1], coeff[-1]


@lru_cache(maxsize=1)
def _configured_weight() -> float:
    payload = _load_sidecar()
    params = payload.get("params")
    if not isinstance(params, Mapping):
        raise RuntimeError("INV_DYN_PARAMS_INVALID")
    value = params.get("reward_weight")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise RuntimeError("INV_DYN_REWARD_WEIGHT_INVALID")
    weight = float(value)
    if not math.isfinite(weight) or weight <= 0.0:
        raise RuntimeError("INV_DYN_REWARD_WEIGHT_INVALID")
    return weight


def _load_sidecar() -> Mapping[str, object]:
    try:
        payload = json.loads(_SIDECAR.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("INV_DYN_SIDECAR_UNREADABLE") from exc
    if not isinstance(payload, Mapping):
        raise RuntimeError("INV_DYN_SIDECAR_INVALID")
    if (
        payload.get("artifact_type") != "wmloop-materialized-primitive-smoke"
        or payload.get("primitive") != "inv_dyn_reward_finetune"
        or payload.get("hook") != "H3"
        or payload.get("materialization_state") != "acwm_runtime_hook_smoke"
    ):
        raise RuntimeError("INV_DYN_SIDECAR_INVALID")
    return payload
'''
