from __future__ import annotations

import difflib
import json
import math
from pathlib import Path
from typing import Mapping


def apply(repo_worktree: str, params: Mapping[str, object]) -> str:
    worktree = Path(repo_worktree)
    weight = _validate_weight(params)
    dynamics_path = worktree / "acwm" / "dynamics" / "diffusion_forcing_wm.py"
    trainer_path = worktree / "acwm" / "trainer" / "train_dynamics.py"
    original_dynamics = dynamics_path.read_text(encoding="utf-8")
    original_trainer = trainer_path.read_text(encoding="utf-8")
    patched_dynamics = _patch_dynamics(original_dynamics)
    patched_trainer = _patch_trainer(original_trainer)
    sidecar = _sidecar_payload(weight)
    return "\n".join(
        part.rstrip("\n")
        for part in (
            _unified_diff("acwm/dynamics/diffusion_forcing_wm.py", original_dynamics, patched_dynamics),
            _unified_diff("acwm/trainer/train_dynamics.py", original_trainer, patched_trainer),
            _optional_new_file_diff(
                worktree,
                "acwm/wmloop_hooks/__init__.py",
                '"""Runtime hooks materialized by reviewed wm-loop primitive templates."""\n',
            ),
            _new_file_diff("acwm/wmloop_hooks/action_contrastive_finetune.py", _HOOK_MODULE),
            _new_file_diff(
                "wmloop_interventions/action_contrastive_finetune.json",
                json.dumps(sidecar, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            ),
        )
        if part.strip()
    ) + "\n"


def _validate_weight(params: Mapping[str, object]) -> float:
    value = params.get("weight")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError("ACTION_CONTRASTIVE_FINETUNE_WEIGHT_INVALID")
    weight = float(value)
    if not math.isfinite(weight) or weight <= 0.0 or weight > 1.0:
        raise ValueError("ACTION_CONTRASTIVE_FINETUNE_WEIGHT_INVALID")
    return weight


def _patch_dynamics(original: str) -> str:
    import_marker = "from acwm.model.interface import DIT_CLASS_MAP, VAE_CLASS_MAP\n"
    import_replacement = (
        "from acwm.model.interface import DIT_CLASS_MAP, VAE_CLASS_MAP\n"
        "from acwm.wmloop_hooks.action_contrastive_finetune import apply_action_contrastive_loss\n"
    )
    loss_marker = "        loss = (weights.view(B, T, 1, 1, 1) * loss_map).mean()\n        return loss\n"
    loss_replacement = (
        "        loss = (weights.view(B, T, 1, 1, 1) * loss_map).mean()\n"
        "        loss, self._wmloop_action_contrastive_metrics = apply_action_contrastive_loss(\n"
        "            base_loss=loss,\n"
        "            dynamics_model=self.model,\n"
        "            z_t=z_t,\n"
        "            t_values=t_values,\n"
        "            actions=a,\n"
        "            v_pred=v_pred,\n"
        "            v_target=v_target,\n"
        "            weights=weights,\n"
        "        )\n"
        "        return loss\n"
    )
    patched = original
    if import_marker not in patched:
        raise ValueError("ACTION_CONTRASTIVE_FINETUNE_DYNAMICS_IMPORT_ANCHOR_MISSING")
    if loss_marker not in patched:
        raise ValueError("ACTION_CONTRASTIVE_FINETUNE_DYNAMICS_LOSS_ANCHOR_MISSING")
    patched = patched.replace(import_marker, import_replacement, 1)
    return patched.replace(loss_marker, loss_replacement, 1)


def _patch_trainer(original: str) -> str:
    loss_marker = (
        "            loss = model.module.training_loss(z, action) if hasattr(model, 'module') else model.training_loss(z, action)\n"
        "            loss.backward()\n"
    )
    loss_replacement = (
        "            loss = model.module.training_loss(z, action) if hasattr(model, 'module') else model.training_loss(z, action)\n"
        "            wmloop_base_model = model.module if hasattr(model, 'module') else model\n"
        "            wmloop_primitive_metrics = dict(getattr(wmloop_base_model, '_wmloop_action_contrastive_metrics', {}))\n"
        "            loss.backward()\n"
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
    if loss_marker not in patched:
        raise ValueError("ACTION_CONTRASTIVE_FINETUNE_TRAINER_LOSS_ANCHOR_MISSING")
    if log_marker not in patched:
        raise ValueError("ACTION_CONTRASTIVE_FINETUNE_TRAINER_LOG_ANCHOR_MISSING")
    patched = patched.replace(loss_marker, loss_replacement, 1)
    return patched.replace(log_marker, log_replacement, 1)


def _sidecar_payload(weight: float) -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_type": "wmloop-materialized-primitive-smoke",
        "primitive": "action_contrastive_finetune",
        "layer": "L3",
        "hook": "H3",
        "materialization_state": "acwm_runtime_hook_smoke",
        "params": {"weight": weight, "margin": 0.05},
        "runtime_hook": {
            "module": "acwm.wmloop_hooks.action_contrastive_finetune",
            "function": "apply_action_contrastive_loss",
            "dynamics_patch": "acwm/dynamics/diffusion_forcing_wm.py",
            "trainer_patch": "acwm/trainer/train_dynamics.py",
        },
        "runtime_hook_paths": [
            "acwm/dynamics/diffusion_forcing_wm.py",
            "acwm/trainer/train_dynamics.py",
            "acwm/wmloop_hooks/action_contrastive_finetune.py",
        ],
        "intent_to_code_contract": {
            "method_intent": "Improve action binding by separating true-action denoising from counterfactual batch-shuffled actions.",
            "runtime_behavior": (
                "Reuses the same noisy latent, timestep, and target velocity, evaluates a batch-shuffled action negative, "
                "and adds a margin ranking loss requiring true-action error to be lower."
            ),
            "declared_proxy": "The counterfactual action margin is the executable action-binding objective.",
            "not_claimed": [
                "does not use held-out or verdict evidence as training labels",
                "does not modify official evaluator files",
                "does not prove improvement until the frozen paired gate passes",
            ],
        },
        "notes": [
            "The auxiliary loss has gradient through both true and counterfactual DiT forward passes.",
            "A fixed margin is recorded in the sidecar so configuration intent and runtime behavior remain aligned.",
        ],
    }


def _unified_diff(path: str, original: str, patched: str) -> str:
    if original == patched:
        raise ValueError(f"ACTION_CONTRASTIVE_FINETUNE_PATCH_EMPTY:{path}")
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


_SIDECAR = Path("wmloop_interventions/action_contrastive_finetune.json")


def apply_action_contrastive_loss(
    *,
    base_loss: torch.Tensor,
    dynamics_model: torch.nn.Module,
    z_t: torch.Tensor,
    t_values: torch.Tensor,
    actions: torch.Tensor,
    v_pred: torch.Tensor,
    v_target: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Apply a gradient-bearing counterfactual action-binding margin loss."""

    config = _configured()
    if actions.ndim != 3 or actions.shape[0] < 2:
        return base_loss, {"train/wmloop_action_contrastive_active": 0.0, "train/wmloop_action_contrastive_loss": 0.0}
    negative_actions = actions.roll(shifts=1, dims=0)
    negative_pred = dynamics_model(z_t, t_values, negative_actions)
    weight_view = weights.view(weights.shape[0], weights.shape[1], 1, 1, 1)
    true_error = (weight_view * (v_pred - v_target).square()).flatten(1).mean(dim=1)
    negative_error = (weight_view * (negative_pred - v_target).square()).flatten(1).mean(dim=1)
    ranking_loss = torch.relu(config["margin"] + true_error - negative_error).mean()
    total = base_loss + config["weight"] * ranking_loss
    return total, {
        "train/wmloop_action_contrastive_active": 1.0,
        "train/wmloop_action_contrastive_loss": float(ranking_loss.detach().cpu()),
        "train/wmloop_action_contrastive_weight": config["weight"],
    }


@lru_cache(maxsize=1)
def _configured() -> dict[str, float]:
    try:
        payload = json.loads(_SIDECAR.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("ACTION_CONTRASTIVE_FINETUNE_SIDECAR_UNREADABLE") from exc
    if not isinstance(payload, Mapping):
        raise RuntimeError("ACTION_CONTRASTIVE_FINETUNE_SIDECAR_INVALID")
    if (
        payload.get("artifact_type") != "wmloop-materialized-primitive-smoke"
        or payload.get("primitive") != "action_contrastive_finetune"
        or payload.get("hook") != "H3"
        or payload.get("materialization_state") != "acwm_runtime_hook_smoke"
    ):
        raise RuntimeError("ACTION_CONTRASTIVE_FINETUNE_SIDECAR_INVALID")
    params = payload.get("params")
    if not isinstance(params, Mapping):
        raise RuntimeError("ACTION_CONTRASTIVE_FINETUNE_PARAMS_INVALID")
    weight = params.get("weight")
    margin = params.get("margin")
    if (
        not isinstance(weight, (int, float))
        or isinstance(weight, bool)
        or not math.isfinite(float(weight))
        or float(weight) <= 0.0
        or float(weight) > 1.0
        or not isinstance(margin, (int, float))
        or isinstance(margin, bool)
        or not math.isfinite(float(margin))
        or float(margin) <= 0.0
    ):
        raise RuntimeError("ACTION_CONTRASTIVE_FINETUNE_PARAMS_INVALID")
    return {"weight": float(weight), "margin": float(margin)}
'''
