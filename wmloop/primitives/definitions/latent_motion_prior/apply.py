from __future__ import annotations

import difflib
import json
import math
from pathlib import Path
from typing import Mapping


def apply(repo_worktree: str, params: Mapping[str, object]) -> str:
    worktree = Path(repo_worktree)
    weight = _validate_weight(params)
    trainer_path = worktree / "acwm" / "trainer" / "train_dynamics.py"
    original = trainer_path.read_text(encoding="utf-8")
    patched = _patch_trainer(original)
    sidecar = _sidecar_payload(weight)
    return "\n".join(
        part.rstrip("\n")
        for part in (
            _unified_diff(
                "acwm/trainer/train_dynamics.py",
                original,
                patched,
            ),
            _optional_new_file_diff(
                worktree,
                "acwm/wmloop_hooks/__init__.py",
                '"""Runtime hooks materialized by reviewed wm-loop primitive templates."""\n',
            ),
            _new_file_diff("acwm/wmloop_hooks/latent_motion_prior.py", _HOOK_MODULE),
            _new_file_diff(
                "wmloop_interventions/latent_motion_prior.json",
                json.dumps(sidecar, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            ),
        )
        if part.strip()
    ) + "\n"


def _validate_weight(params: Mapping[str, object]) -> float:
    value = params.get("weight")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError("LATENT_MOTION_PRIOR_WEIGHT_INVALID")
    weight = float(value)
    if not math.isfinite(weight) or weight <= 0.0:
        raise ValueError("LATENT_MOTION_PRIOR_WEIGHT_INVALID")
    return weight


def _patch_trainer(original: str) -> str:
    import_marker = "from acwm.utils.visualization import visualize_layout\n"
    import_replacement = (
        "from acwm.utils.visualization import visualize_layout\n"
        "from acwm.wmloop_hooks.latent_motion_prior import apply_latent_motion_prior\n"
    )
    loss_marker = (
        "            loss = model.module.training_loss(z, action) if hasattr(model, 'module') else model.training_loss(z, action)\n"
        "            loss.backward()\n"
    )
    loss_replacement = (
        "            loss = model.module.training_loss(z, action) if hasattr(model, 'module') else model.training_loss(z, action)\n"
        "            wmloop_primitive_metrics = {}\n"
        "            loss, wmloop_latent_motion_metrics = apply_latent_motion_prior(loss, z, action)\n"
        "            wmloop_primitive_metrics.update(wmloop_latent_motion_metrics)\n"
        "            loss.backward()\n"
    )
    existing_metrics_backward_marker = "            loss.backward()\n"
    existing_metrics_backward_replacement = (
        "            loss, wmloop_latent_motion_metrics = apply_latent_motion_prior(loss, z, action)\n"
        "            wmloop_primitive_metrics.update(wmloop_latent_motion_metrics)\n"
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
    if import_marker not in patched:
        raise ValueError("LATENT_MOTION_PRIOR_IMPORT_ANCHOR_MISSING")
    patched = patched.replace(import_marker, import_replacement, 1)
    if loss_marker in patched:
        patched = patched.replace(loss_marker, loss_replacement, 1)
    elif "wmloop_primitive_metrics" in patched and existing_metrics_backward_marker in patched:
        patched = patched.replace(existing_metrics_backward_marker, existing_metrics_backward_replacement, 1)
    else:
        raise ValueError("LATENT_MOTION_PRIOR_LOSS_ANCHOR_MISSING")
    if log_marker in patched:
        patched = patched.replace(log_marker, log_replacement, 1)
    elif "wandb.log(train_metrics, step=step)" not in patched:
        raise ValueError("LATENT_MOTION_PRIOR_LOG_ANCHOR_MISSING")
    return patched


def _sidecar_payload(weight: float) -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_type": "wmloop-materialized-primitive-smoke",
        "primitive": "latent_motion_prior",
        "layer": "L3",
        "hook": "H3",
        "materialization_state": "acwm_runtime_hook_smoke",
        "params": {"weight": weight},
        "runtime_hook": {
            "module": "acwm.wmloop_hooks.latent_motion_prior",
            "function": "apply_latent_motion_prior",
            "trainer_patch": "acwm/trainer/train_dynamics.py",
        },
        "runtime_hook_paths": [
            "acwm/trainer/train_dynamics.py",
            "acwm/wmloop_hooks/latent_motion_prior.py",
        ],
        "intent_to_code_contract": {
            "method_intent": "Regularize latent transitions so predicted dynamics remain action-consistent over rollout.",
            "runtime_behavior": (
                "Adds an H3 auxiliary loss on temporal latent deltas inside ACWM train_dynamics immediately after "
                "the base model training_loss is computed."
            ),
            "declared_proxy": (
                "Uses latent temporal smoothness normalized by action energy as the executable proxy for a motion prior."
            ),
            "not_claimed": [
                "does not change ACWM evaluator files",
                "does not inject new trajectories",
                "does not prove improvement until paired eval passes",
            ],
        },
        "notes": [
            "Adds a reviewed H3 auxiliary latent-motion loss to ACWM training_loss assembly.",
            "The patch is trial-worktree only and does not modify frozen evaluator files.",
        ],
    }


def _unified_diff(path: str, original: str, patched: str) -> str:
    if original == patched:
        raise ValueError("LATENT_MOTION_PRIOR_PATCH_EMPTY")
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


_SIDECAR = Path("wmloop_interventions/latent_motion_prior.json")


def apply_latent_motion_prior(
    base_loss: torch.Tensor,
    latents: torch.Tensor,
    actions: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Add a bounded latent-motion auxiliary loss inside the ACWM training path."""

    weight = _configured_weight()
    if latents.ndim != 5 or latents.shape[1] < 2:
        raise RuntimeError("LATENT_MOTION_PRIOR_LATENT_SHAPE_INVALID")
    if actions.ndim != 3:
        raise RuntimeError("LATENT_MOTION_PRIOR_ACTION_SHAPE_INVALID")
    latent_delta = latents.float()[:, 1:] - latents.float()[:, :-1]
    latent_motion = latent_delta.square().mean()
    action_prefix = actions[:, : max(1, min(actions.shape[1], latent_delta.shape[1]))].float()
    action_energy = action_prefix.square().mean().detach()
    aux_loss = weight * latent_motion / (1.0 + action_energy)
    total = base_loss + aux_loss.to(dtype=base_loss.dtype)
    return total, {
        "train/wmloop_latent_motion_prior_loss": float(aux_loss.detach().cpu()),
        "train/wmloop_latent_motion_prior_weight": float(weight),
    }


@lru_cache(maxsize=1)
def _configured_weight() -> float:
    payload = _load_sidecar()
    params = payload.get("params")
    if not isinstance(params, Mapping):
        raise RuntimeError("LATENT_MOTION_PRIOR_PARAMS_INVALID")
    value = params.get("weight")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise RuntimeError("LATENT_MOTION_PRIOR_WEIGHT_INVALID")
    weight = float(value)
    if not math.isfinite(weight) or weight <= 0.0:
        raise RuntimeError("LATENT_MOTION_PRIOR_WEIGHT_INVALID")
    return weight


def _load_sidecar() -> Mapping[str, object]:
    try:
        payload = json.loads(_SIDECAR.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("LATENT_MOTION_PRIOR_SIDECAR_UNREADABLE") from exc
    if not isinstance(payload, Mapping):
        raise RuntimeError("LATENT_MOTION_PRIOR_SIDECAR_INVALID")
    if (
        payload.get("artifact_type") != "wmloop-materialized-primitive-smoke"
        or payload.get("primitive") != "latent_motion_prior"
        or payload.get("hook") != "H3"
        or payload.get("materialization_state") != "acwm_runtime_hook_smoke"
    ):
        raise RuntimeError("LATENT_MOTION_PRIOR_SIDECAR_INVALID")
    return payload
'''
