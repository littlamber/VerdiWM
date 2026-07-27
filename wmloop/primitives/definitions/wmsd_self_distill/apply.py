from __future__ import annotations

import difflib
import json
import math
from pathlib import Path
from typing import Mapping


def apply(repo_worktree: str, params: Mapping[str, object]) -> str:
    worktree = Path(repo_worktree)
    teacher_ema, steps, lr = _validate_params(params)
    trainer_path = worktree / "acwm" / "trainer" / "train_dynamics.py"
    original = trainer_path.read_text(encoding="utf-8")
    patched = _patch_trainer(original)
    sidecar = _sidecar_payload(teacher_ema=teacher_ema, steps=steps, lr=lr)
    return "\n".join(
        part.rstrip("\n")
        for part in (
            _unified_diff("acwm/trainer/train_dynamics.py", original, patched),
            _optional_new_file_diff(
                worktree,
                "acwm/wmloop_hooks/__init__.py",
                '"""Runtime hooks materialized by reviewed wm-loop primitive templates."""\n',
            ),
            _new_file_diff("acwm/wmloop_hooks/wmsd_self_distill.py", _HOOK_MODULE),
            _new_file_diff(
                "wmloop_interventions/wmsd_self_distill.json",
                json.dumps(sidecar, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            ),
        )
        if part.strip()
    ) + "\n"


def _validate_params(params: Mapping[str, object]) -> tuple[float, int, float]:
    ema_value = params.get("teacher_ema")
    steps_value = params.get("steps")
    lr_value = params.get("lr")
    if not isinstance(ema_value, (int, float)) or isinstance(ema_value, bool):
        raise ValueError("WMSD_TEACHER_EMA_INVALID")
    teacher_ema = float(ema_value)
    if not math.isfinite(teacher_ema) or teacher_ema <= 0.0 or teacher_ema > 1.0:
        raise ValueError("WMSD_TEACHER_EMA_INVALID")
    if not isinstance(steps_value, int) or isinstance(steps_value, bool) or steps_value < 1:
        raise ValueError("WMSD_STEPS_INVALID")
    if not isinstance(lr_value, (int, float)) or isinstance(lr_value, bool):
        raise ValueError("WMSD_LR_INVALID")
    lr = float(lr_value)
    if not math.isfinite(lr) or lr <= 0.0:
        raise ValueError("WMSD_LR_INVALID")
    return teacher_ema, int(steps_value), lr


def _patch_trainer(original: str) -> str:
    import_marker = "from acwm.utils.visualization import visualize_layout\n"
    import_replacement = (
        "from acwm.utils.visualization import visualize_layout\n"
        "from acwm.wmloop_hooks.wmsd_self_distill import apply_wmsd_self_distill_loss\n"
    )
    clean_loss_marker = (
        "            loss = model.module.training_loss(z, action) if hasattr(model, 'module') else model.training_loss(z, action)\n"
        "            loss.backward()\n"
    )
    clean_loss_replacement = (
        "            wmloop_primitive_metrics = {}\n"
        "            wmloop_base_model = model.module if hasattr(model, 'module') else model\n"
        "            loss = wmloop_base_model.training_loss(z, action)\n"
        "            loss, wmloop_wmsd_metrics = apply_wmsd_self_distill_loss(loss, wmloop_base_model, z, action)\n"
        "            wmloop_primitive_metrics.update(wmloop_wmsd_metrics)\n"
        "            loss.backward()\n"
    )
    existing_metrics_backward_marker = "            loss.backward()\n"
    existing_metrics_backward_replacement = (
        "            wmloop_base_model = model.module if hasattr(model, 'module') else model\n"
        "            loss, wmloop_wmsd_metrics = apply_wmsd_self_distill_loss(loss, wmloop_base_model, z, action)\n"
        "            wmloop_primitive_metrics.update(wmloop_wmsd_metrics)\n"
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
        raise ValueError("WMSD_IMPORT_ANCHOR_MISSING")
    patched = patched.replace(import_marker, import_replacement, 1)
    if clean_loss_marker in patched:
        patched = patched.replace(clean_loss_marker, clean_loss_replacement, 1)
    elif "wmloop_primitive_metrics" in patched and existing_metrics_backward_marker in patched:
        patched = patched.replace(existing_metrics_backward_marker, existing_metrics_backward_replacement, 1)
    else:
        raise ValueError("WMSD_LOSS_ANCHOR_MISSING")
    if clean_log_marker in patched:
        patched = patched.replace(clean_log_marker, clean_log_replacement, 1)
    elif "wandb.log(train_metrics, step=step)" not in patched:
        raise ValueError("WMSD_LOG_ANCHOR_MISSING")
    return patched


def _sidecar_payload(*, teacher_ema: float, steps: int, lr: float) -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_type": "wmloop-materialized-primitive-smoke",
        "primitive": "wmsd_self_distill",
        "layer": "L3",
        "hook": "H3",
        "materialization_state": "acwm_runtime_hook_smoke",
        "params": {"teacher_ema": teacher_ema, "steps": steps, "lr": lr},
        "runtime_hook": {
            "module": "acwm.wmloop_hooks.wmsd_self_distill",
            "function": "apply_wmsd_self_distill_loss",
            "trainer_patch": "acwm/trainer/train_dynamics.py",
        },
        "runtime_hook_paths": [
            "acwm/trainer/train_dynamics.py",
            "acwm/wmloop_hooks/wmsd_self_distill.py",
        ],
        "intent_to_code_contract": {
            "method_intent": (
                "Reduce train-inference mismatch by distilling the current dynamics predictions toward an online "
                "EMA teacher under on-policy noisy latent prefixes."
            ),
            "runtime_behavior": (
                "Adds an H3 auxiliary consistency loss after ACWM training_loss. The hook runs one additional DiT "
                "velocity prediction, pools it over spatial dimensions, and maintains a detached EMA teacher target."
            ),
            "declared_proxy": (
                "Uses pooled latent velocity EMA consistency as the executable WMSD proxy to avoid silently adding "
                "a large checkpointed teacher model or changing evaluator behavior."
            ),
            "not_claimed": [
                "does not add a persistent full-model EMA teacher checkpoint",
                "does not decode pixels or alter held-out evaluation",
                "does not prove OOD or train-inference improvement until GPU smoke and paired eval pass",
            ],
        },
        "notes": [
            "Adds a real H3 train-objective hook instead of a sidecar-only request.",
            "The steps and lr parameters are recorded for scheduler/orchestrator policy; this smoke hook consumes teacher_ema.",
            "The hook is trial-worktree only and does not modify frozen evaluator files.",
        ],
    }


def _unified_diff(path: str, original: str, patched: str) -> str:
    if original == patched:
        raise ValueError("WMSD_PATCH_EMPTY")
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


_SIDECAR = Path("wmloop_interventions/wmsd_self_distill.json")
_TEACHER_SUMMARY: torch.Tensor | None = None


@dataclass(frozen=True)
class WMSDConfig:
    teacher_ema: float
    steps: int
    lr: float


def apply_wmsd_self_distill_loss(
    base_loss: torch.Tensor,
    model: object,
    latents: torch.Tensor,
    actions: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Add an online EMA-teacher self-distillation loss over latent velocity summaries."""

    config = _configured_params()
    if latents.ndim != 5:
        raise RuntimeError("WMSD_LATENT_SHAPE_INVALID")
    if actions.ndim != 3:
        raise RuntimeError("WMSD_ACTION_SHAPE_INVALID")
    inner_model = getattr(model, "model", None)
    scheduler = getattr(model, "scheduler", None)
    if not callable(inner_model):
        raise RuntimeError("WMSD_INNER_MODEL_MISSING")
    _validate_scheduler(scheduler)

    t_values = _distill_timesteps(scheduler, latents)
    noisy_latents, _noise = scheduler.add_independent_noise(latents.float(), t_values)
    v_student = inner_model(noisy_latents.to(dtype=latents.dtype), t_values, actions)
    student_summary = _velocity_summary(v_student)
    teacher_target, teacher_warm = _update_teacher_summary(student_summary, teacher_ema=config.teacher_ema)
    aux_loss = torch.nn.functional.mse_loss(student_summary.float(), teacher_target.to(student_summary.device).float())
    aux_weight = 0.0 if teacher_warm else (1.0 - float(config.teacher_ema))
    total = base_loss + base_loss.new_tensor(aux_weight) * aux_loss.to(dtype=base_loss.dtype)
    return total, {
        "train/wmloop_wmsd_self_distill_loss": float(aux_loss.detach().cpu()),
        "train/wmloop_wmsd_teacher_ema": float(config.teacher_ema),
        "train/wmloop_wmsd_aux_weight": float(aux_weight),
        "train/wmloop_wmsd_teacher_warm": float(teacher_warm),
        "train/wmloop_wmsd_lr": float(config.lr),
    }


def _velocity_summary(velocity: torch.Tensor) -> torch.Tensor:
    if velocity.ndim != 5:
        raise RuntimeError("WMSD_VELOCITY_SHAPE_INVALID")
    return velocity.float().mean(dim=(2, 3))


def _update_teacher_summary(student_summary: torch.Tensor, *, teacher_ema: float) -> tuple[torch.Tensor, bool]:
    global _TEACHER_SUMMARY
    detached = student_summary.detach()
    warm = _TEACHER_SUMMARY is None or tuple(_TEACHER_SUMMARY.shape) != tuple(detached.shape)
    if warm:
        _TEACHER_SUMMARY = detached.clone()
        return _TEACHER_SUMMARY, True
    teacher = _TEACHER_SUMMARY.to(device=detached.device, dtype=detached.dtype)
    target = teacher.clone()
    _TEACHER_SUMMARY = teacher_ema * teacher + (1.0 - teacher_ema) * detached
    return target, False


def _distill_timesteps(scheduler: object, latents: torch.Tensor) -> torch.Tensor:
    timesteps = getattr(scheduler, "timesteps")
    if not isinstance(timesteps, torch.Tensor) or timesteps.ndim != 1 or timesteps.numel() < 1:
        raise RuntimeError("WMSD_TIMESTEPS_INVALID")
    index = min(max(0, int(timesteps.numel()) // 2), int(timesteps.numel()) - 1)
    t_value = timesteps[index].to(device=latents.device)
    return torch.full((latents.shape[0], latents.shape[1]), t_value, device=latents.device, dtype=timesteps.dtype)


def _validate_scheduler(scheduler: object) -> None:
    if scheduler is None:
        raise RuntimeError("WMSD_SCHEDULER_MISSING")
    if not callable(getattr(scheduler, "add_independent_noise", None)):
        raise RuntimeError("WMSD_SCHEDULER_ADD_NOISE_MISSING")
    timesteps = getattr(scheduler, "timesteps", None)
    if not isinstance(timesteps, torch.Tensor) or timesteps.ndim != 1 or timesteps.numel() < 1:
        raise RuntimeError("WMSD_TIMESTEPS_INVALID")


@lru_cache(maxsize=1)
def _configured_params() -> WMSDConfig:
    payload = _load_sidecar()
    params = payload.get("params")
    if not isinstance(params, Mapping):
        raise RuntimeError("WMSD_PARAMS_INVALID")
    ema = params.get("teacher_ema")
    steps = params.get("steps")
    lr = params.get("lr")
    if not isinstance(ema, (int, float)) or isinstance(ema, bool):
        raise RuntimeError("WMSD_TEACHER_EMA_INVALID")
    ema_value = float(ema)
    if not math.isfinite(ema_value) or ema_value <= 0.0 or ema_value > 1.0:
        raise RuntimeError("WMSD_TEACHER_EMA_INVALID")
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
        raise RuntimeError("WMSD_STEPS_INVALID")
    if not isinstance(lr, (int, float)) or isinstance(lr, bool):
        raise RuntimeError("WMSD_LR_INVALID")
    lr_value = float(lr)
    if not math.isfinite(lr_value) or lr_value <= 0.0:
        raise RuntimeError("WMSD_LR_INVALID")
    return WMSDConfig(teacher_ema=ema_value, steps=int(steps), lr=lr_value)


def _load_sidecar() -> Mapping[str, object]:
    try:
        payload = json.loads(_SIDECAR.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("WMSD_SIDECAR_UNREADABLE") from exc
    if not isinstance(payload, Mapping):
        raise RuntimeError("WMSD_SIDECAR_INVALID")
    if (
        payload.get("artifact_type") != "wmloop-materialized-primitive-smoke"
        or payload.get("primitive") != "wmsd_self_distill"
        or payload.get("hook") != "H3"
        or payload.get("materialization_state") != "acwm_runtime_hook_smoke"
    ):
        raise RuntimeError("WMSD_SIDECAR_INVALID")
    return payload
'''
