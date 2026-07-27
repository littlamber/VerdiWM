from __future__ import annotations

import difflib
import json
import math
from pathlib import Path
from typing import Mapping


def apply(repo_worktree: str, params: Mapping[str, object]) -> str:
    worktree = Path(repo_worktree)
    chunks, steps, lr = _validate_params(params)
    trainer_path = worktree / "acwm" / "trainer" / "train_dynamics.py"
    original = trainer_path.read_text(encoding="utf-8")
    patched = _patch_trainer(original)
    sidecar = _sidecar_payload(chunks=chunks, steps=steps, lr=lr)
    return "\n".join(
        part.rstrip("\n")
        for part in (
            _unified_diff("acwm/trainer/train_dynamics.py", original, patched),
            _optional_new_file_diff(
                worktree,
                "acwm/wmloop_hooks/__init__.py",
                '"""Runtime hooks materialized by reviewed wm-loop primitive templates."""\n',
            ),
            _new_file_diff("acwm/wmloop_hooks/next_forcing.py", _HOOK_MODULE),
            _new_file_diff(
                "wmloop_interventions/next_forcing.json",
                json.dumps(sidecar, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            ),
        )
        if part.strip()
    ) + "\n"


def _validate_params(params: Mapping[str, object]) -> tuple[int, int, float]:
    chunks_value = params.get("chunks")
    steps_value = params.get("steps")
    lr_value = params.get("lr")
    if not isinstance(chunks_value, int) or isinstance(chunks_value, bool):
        raise ValueError("NEXT_FORCING_CHUNKS_INVALID")
    if chunks_value < 2 or chunks_value > 8:
        raise ValueError("NEXT_FORCING_CHUNKS_INVALID")
    if not isinstance(steps_value, int) or isinstance(steps_value, bool) or steps_value < 1:
        raise ValueError("NEXT_FORCING_STEPS_INVALID")
    if not isinstance(lr_value, (int, float)) or isinstance(lr_value, bool):
        raise ValueError("NEXT_FORCING_LR_INVALID")
    lr = float(lr_value)
    if not math.isfinite(lr) or lr <= 0.0:
        raise ValueError("NEXT_FORCING_LR_INVALID")
    return int(chunks_value), int(steps_value), lr


def _patch_trainer(original: str) -> str:
    import_marker = "from acwm.utils.visualization import visualize_layout\n"
    import_replacement = (
        "from acwm.utils.visualization import visualize_layout\n"
        "from acwm.wmloop_hooks.next_forcing import apply_next_forcing_loss\n"
    )
    clean_loss_marker = (
        "            loss = model.module.training_loss(z, action) if hasattr(model, 'module') else model.training_loss(z, action)\n"
        "            loss.backward()\n"
    )
    clean_loss_replacement = (
        "            wmloop_primitive_metrics = {}\n"
        "            wmloop_base_model = model.module if hasattr(model, 'module') else model\n"
        "            loss, wmloop_next_forcing_metrics = apply_next_forcing_loss(wmloop_base_model, z, action)\n"
        "            wmloop_primitive_metrics.update(wmloop_next_forcing_metrics)\n"
        "            loss.backward()\n"
    )
    latent_loss_marker = (
        "            loss = model.module.training_loss(z, action) if hasattr(model, 'module') else model.training_loss(z, action)\n"
        "            loss, wmloop_primitive_metrics = apply_latent_motion_prior(loss, z, action)\n"
        "            loss.backward()\n"
    )
    latent_loss_replacement = (
        "            wmloop_primitive_metrics = {}\n"
        "            wmloop_base_model = model.module if hasattr(model, 'module') else model\n"
        "            loss, wmloop_next_forcing_metrics = apply_next_forcing_loss(wmloop_base_model, z, action)\n"
        "            wmloop_primitive_metrics.update(wmloop_next_forcing_metrics)\n"
        "            loss, wmloop_latent_motion_metrics = apply_latent_motion_prior(loss, z, action)\n"
        "            wmloop_primitive_metrics.update(wmloop_latent_motion_metrics)\n"
        "            loss.backward()\n"
    )
    existing_metrics_backward_marker = "            loss.backward()\n"
    existing_metrics_backward_replacement = (
        "            wmloop_base_model = model.module if hasattr(model, 'module') else model\n"
        "            loss, wmloop_next_forcing_metrics = apply_next_forcing_loss(wmloop_base_model, z, action)\n"
        "            wmloop_primitive_metrics.update(wmloop_next_forcing_metrics)\n"
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
        raise ValueError("NEXT_FORCING_IMPORT_ANCHOR_MISSING")
    patched = patched.replace(import_marker, import_replacement, 1)
    if latent_loss_marker in patched:
        patched = patched.replace(latent_loss_marker, latent_loss_replacement, 1)
    elif clean_loss_marker in patched:
        patched = patched.replace(clean_loss_marker, clean_loss_replacement, 1)
    elif "wmloop_primitive_metrics" in patched and existing_metrics_backward_marker in patched:
        patched = patched.replace(existing_metrics_backward_marker, existing_metrics_backward_replacement, 1)
    else:
        raise ValueError("NEXT_FORCING_LOSS_ANCHOR_MISSING")
    if clean_log_marker in patched:
        patched = patched.replace(clean_log_marker, clean_log_replacement, 1)
    elif "wandb.log(train_metrics, step=step)" not in patched:
        raise ValueError("NEXT_FORCING_LOG_ANCHOR_MISSING")
    return patched


def _sidecar_payload(*, chunks: int, steps: int, lr: float) -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_type": "wmloop-materialized-primitive-smoke",
        "primitive": "next_forcing",
        "layer": "L3",
        "hook": "H3",
        "materialization_state": "acwm_runtime_hook_smoke",
        "params": {"chunks": chunks, "steps": steps, "lr": lr},
        "runtime_hook": {
            "module": "acwm.wmloop_hooks.next_forcing",
            "function": "apply_next_forcing_loss",
            "trainer_patch": "acwm/trainer/train_dynamics.py",
        },
        "runtime_hook_paths": [
            "acwm/trainer/train_dynamics.py",
            "acwm/wmloop_hooks/next_forcing.py",
        ],
        "intent_to_code_contract": {
            "method_intent": "Reduce train-inference mismatch by adding supervised losses on shorter rollout prefixes.",
            "runtime_behavior": (
                "Replaces the single base loss call with a base loss plus bounded prefix-window auxiliary losses "
                "computed through the model's existing training_loss."
            ),
            "declared_proxy": "Uses latent prefix windows as the executable proxy for multi-chunk next-token forcing.",
            "not_claimed": [
                "does not run model-generated rollouts inside this smoke hook",
                "does not change held-out evaluation",
                "does not override the frozen primitive conflict rules",
            ],
        },
        "notes": [
            "Adds bounded multi-window next-forcing auxiliary losses around the ACWM training_loss path.",
            "The lr parameter is consumed by the wm-loop training config for this primitive.",
            "The hook is trial-worktree only and does not modify frozen evaluator files.",
        ],
    }


def _unified_diff(path: str, original: str, patched: str) -> str:
    if original == patched:
        raise ValueError("NEXT_FORCING_PATCH_EMPTY")
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


_SIDECAR = Path("wmloop_interventions/next_forcing.json")


@dataclass(frozen=True)
class NextForcingConfig:
    chunks: int
    steps: int
    lr: float


def apply_next_forcing_loss(
    model: object,
    latents: torch.Tensor,
    actions: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Add bounded prefix-window losses to reduce train/inference mismatch."""

    config = _configured_params()
    if latents.ndim != 5:
        raise RuntimeError("NEXT_FORCING_LATENT_SHAPE_INVALID")
    if actions.ndim != 3:
        raise RuntimeError("NEXT_FORCING_ACTION_SHAPE_INVALID")
    training_loss = getattr(model, "training_loss", None)
    if not callable(training_loss):
        raise RuntimeError("NEXT_FORCING_MODEL_TRAINING_LOSS_MISSING")

    base_loss = training_loss(latents, actions)
    if latents.shape[1] < 3:
        return base_loss, {
            "train/wmloop_next_forcing_aux_loss": 0.0,
            "train/wmloop_next_forcing_chunks": 0.0,
            "train/wmloop_next_forcing_lr": float(config.lr),
        }

    window_ends = _window_ends(latents.shape[1], chunks=config.chunks, steps=config.steps)
    losses: list[torch.Tensor] = []
    for end in window_ends:
        z_window = latents[:, :end]
        action_window = _slice_actions_for_latent_window(model, actions, latent_frames=end)
        losses.append(training_loss(z_window, action_window))
    if not losses:
        aux_loss = base_loss.new_tensor(0.0)
    else:
        aux_loss = torch.stack([loss.to(dtype=base_loss.dtype) for loss in losses]).mean()
    aux_weight = 1.0 / float(max(1, config.chunks))
    total = base_loss + base_loss.new_tensor(aux_weight) * aux_loss
    return total, {
        "train/wmloop_next_forcing_base_loss": float(base_loss.detach().cpu()),
        "train/wmloop_next_forcing_aux_loss": float(aux_loss.detach().cpu()),
        "train/wmloop_next_forcing_aux_weight": float(aux_weight),
        "train/wmloop_next_forcing_chunks": float(len(losses)),
        "train/wmloop_next_forcing_lr": float(config.lr),
    }


def _window_ends(total_frames: int, *, chunks: int, steps: int) -> list[int]:
    candidates = []
    for index in range(1, chunks + 1):
        raw = 2 + round((total_frames - 2) * index / chunks)
        end = max(2, min(total_frames, int(raw)))
        candidates.append(end)
    unique = sorted(set(candidates))
    limit = min(len(unique), max(1, int(steps)))
    return unique[:limit]


def _slice_actions_for_latent_window(model: object, actions: torch.Tensor, *, latent_frames: int) -> torch.Tensor:
    compress_rate = _action_compress_rate(model)
    requested = compress_rate * max(0, latent_frames - 1) + 1
    length = max(1, min(actions.shape[1], requested))
    return actions[:, :length]


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


@lru_cache(maxsize=1)
def _configured_params() -> NextForcingConfig:
    payload = _load_sidecar()
    params = payload.get("params")
    if not isinstance(params, Mapping):
        raise RuntimeError("NEXT_FORCING_PARAMS_INVALID")
    chunks = params.get("chunks")
    steps = params.get("steps")
    lr = params.get("lr")
    if not isinstance(chunks, int) or isinstance(chunks, bool) or chunks < 2 or chunks > 8:
        raise RuntimeError("NEXT_FORCING_CHUNKS_INVALID")
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
        raise RuntimeError("NEXT_FORCING_STEPS_INVALID")
    if not isinstance(lr, (int, float)) or isinstance(lr, bool):
        raise RuntimeError("NEXT_FORCING_LR_INVALID")
    lr_value = float(lr)
    if not math.isfinite(lr_value) or lr_value <= 0.0:
        raise RuntimeError("NEXT_FORCING_LR_INVALID")
    return NextForcingConfig(chunks=int(chunks), steps=int(steps), lr=lr_value)


def _load_sidecar() -> Mapping[str, object]:
    try:
        payload = json.loads(_SIDECAR.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("NEXT_FORCING_SIDECAR_UNREADABLE") from exc
    if not isinstance(payload, Mapping):
        raise RuntimeError("NEXT_FORCING_SIDECAR_INVALID")
    if (
        payload.get("artifact_type") != "wmloop-materialized-primitive-smoke"
        or payload.get("primitive") != "next_forcing"
        or payload.get("hook") != "H3"
        or payload.get("materialization_state") != "acwm_runtime_hook_smoke"
    ):
        raise RuntimeError("NEXT_FORCING_SIDECAR_INVALID")
    return payload
'''
