from __future__ import annotations

import difflib
import json
import math
from pathlib import Path
from typing import Mapping


def apply(repo_worktree: str, params: Mapping[str, object]) -> str:
    worktree = Path(repo_worktree)
    guidance_start, guidance_end = _validate_params(params)
    dynamics_path = worktree / "acwm" / "dynamics" / "diffusion_forcing_wm.py"
    original = dynamics_path.read_text(encoding="utf-8")
    patched = _patch_dynamics(original)
    sidecar = _sidecar_payload(guidance_start=guidance_start, guidance_end=guidance_end)
    return "\n".join(
        part.rstrip("\n")
        for part in (
            _unified_diff("acwm/dynamics/diffusion_forcing_wm.py", original, patched),
            _optional_new_file_diff(
                worktree,
                "acwm/wmloop_hooks/__init__.py",
                '"""Runtime hooks materialized by reviewed wm-loop primitive templates."""\n',
            ),
            _new_file_diff("acwm/wmloop_hooks/cfg_guidance_schedule.py", _HOOK_MODULE),
            _new_file_diff(
                "wmloop_interventions/cfg_guidance_schedule.json",
                json.dumps(sidecar, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            ),
        )
        if part.strip()
    ) + "\n"


def _validate_params(params: Mapping[str, object]) -> tuple[float, float]:
    start = params.get("guidance_start")
    end = params.get("guidance_end")
    if not isinstance(start, (int, float)) or isinstance(start, bool):
        raise ValueError("CFG_GUIDANCE_START_INVALID")
    if not isinstance(end, (int, float)) or isinstance(end, bool):
        raise ValueError("CFG_GUIDANCE_END_INVALID")
    guidance_start = float(start)
    guidance_end = float(end)
    if not math.isfinite(guidance_start) or guidance_start < 0.0:
        raise ValueError("CFG_GUIDANCE_START_INVALID")
    if not math.isfinite(guidance_end) or guidance_end < 0.0:
        raise ValueError("CFG_GUIDANCE_END_INVALID")
    return guidance_start, guidance_end


def _patch_dynamics(original: str) -> str:
    import_marker = "from acwm.model.interface import DIT_CLASS_MAP, VAE_CLASS_MAP\n"
    import_replacement = (
        "from acwm.model.interface import DIT_CLASS_MAP, VAE_CLASS_MAP\n"
        "from acwm.wmloop_hooks.cfg_guidance_schedule import apply_cfg_guidance_schedule\n"
    )
    parallel_marker = (
        "                with torch.no_grad():\n"
        "                    v_pred = self.model(z, t, a)\n"
        "                    z = self.scheduler.step(v_pred, t, z)\n"
    )
    parallel_replacement = (
        "                with torch.no_grad():\n"
        "                    v_pred = apply_cfg_guidance_schedule(\n"
        "                        self.model, z, t, a, step_index=i, total_steps=len(self.scheduler.timesteps)\n"
        "                    )\n"
        "                    z = self.scheduler.step(v_pred, t, z)\n"
    )
    autoregressive_marker = (
        "                    with torch.no_grad():\n"
        "                        v_pred = self.model(z_curr, t_seq, a_curr)\n"
        "                        z_curr = self.scheduler.step(v_pred, t_seq, z_curr)\n"
    )
    autoregressive_replacement = (
        "                    with torch.no_grad():\n"
        "                        v_pred = apply_cfg_guidance_schedule(\n"
        "                            self.model, z_curr, t_seq, a_curr, step_index=i, total_steps=len(self.scheduler.timesteps)\n"
        "                        )\n"
        "                        z_curr = self.scheduler.step(v_pred, t_seq, z_curr)\n"
    )
    patched = original
    for marker, replacement, code in (
        (import_marker, import_replacement, "CFG_GUIDANCE_IMPORT_ANCHOR_MISSING"),
        (parallel_marker, parallel_replacement, "CFG_GUIDANCE_PARALLEL_ANCHOR_MISSING"),
        (autoregressive_marker, autoregressive_replacement, "CFG_GUIDANCE_AUTOREGRESSIVE_ANCHOR_MISSING"),
    ):
        if marker not in patched:
            raise ValueError(code)
        patched = patched.replace(marker, replacement, 1)
    return patched


def _sidecar_payload(*, guidance_start: float, guidance_end: float) -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_type": "wmloop-materialized-primitive-smoke",
        "primitive": "cfg_guidance_schedule",
        "layer": "L4",
        "hook": "H4",
        "materialization_state": "acwm_runtime_hook_smoke",
        "params": {"guidance_start": guidance_start, "guidance_end": guidance_end},
        "runtime_hook": {
            "module": "acwm.wmloop_hooks.cfg_guidance_schedule",
            "function": "apply_cfg_guidance_schedule",
            "dynamics_patch": "acwm/dynamics/diffusion_forcing_wm.py",
        },
        "runtime_hook_paths": [
            "acwm/dynamics/diffusion_forcing_wm.py",
            "acwm/wmloop_hooks/cfg_guidance_schedule.py",
        ],
        "intent_to_code_contract": {
            "method_intent": "Schedule action-conditioned classifier-free guidance during ACWM inference.",
            "runtime_behavior": (
                "Wraps DiffusionForcing_WM.generate denoising model calls with cond/uncond predictions and blends "
                "them using a linear guidance_start-to-guidance_end schedule."
            ),
            "declared_proxy": (
                "Uses the model's null-action convention when available, falling back to an all-zero action tensor "
                "with a null bit in the final action channel."
            ),
            "not_claimed": [
                "does not add text guidance",
                "does not retrain the model",
                "does not modify frozen evaluator files",
            ],
        },
        "notes": [
            "Adds action-conditioned CFG scheduling to ACWM generation without changing tensor shapes.",
            "The hook is trial-worktree only and does not modify frozen evaluator files.",
        ],
    }


def _unified_diff(path: str, original: str, patched: str) -> str:
    if original == patched:
        raise ValueError("CFG_GUIDANCE_PATCH_EMPTY")
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


_SIDECAR = Path("wmloop_interventions/cfg_guidance_schedule.json")


@dataclass(frozen=True)
class CfgGuidanceSchedule:
    guidance_start: float
    guidance_end: float


def apply_cfg_guidance_schedule(
    model: object,
    latents: torch.Tensor,
    timesteps: torch.Tensor,
    actions: torch.Tensor,
    *,
    step_index: int,
    total_steps: int,
) -> torch.Tensor:
    """Run action-conditioned classifier-free guidance with a bounded schedule."""

    config = _configured_params()
    if latents.ndim != 5:
        raise RuntimeError("CFG_GUIDANCE_LATENT_SHAPE_INVALID")
    if timesteps.ndim != 2:
        raise RuntimeError("CFG_GUIDANCE_TIMESTEP_SHAPE_INVALID")
    if actions.ndim != 3:
        raise RuntimeError("CFG_GUIDANCE_ACTION_SHAPE_INVALID")
    if total_steps < 1:
        raise RuntimeError("CFG_GUIDANCE_TOTAL_STEPS_INVALID")
    forward = getattr(model, "__call__", None)
    if not callable(forward):
        raise RuntimeError("CFG_GUIDANCE_MODEL_CALL_MISSING")

    cond = model(latents, timesteps, actions)
    scale = _scale_for_step(config, step_index=step_index, total_steps=total_steps)
    if scale == 1.0:
        return cond
    uncond = model(latents, timesteps, _null_actions(model, actions))
    return uncond + cond.new_tensor(scale) * (cond - uncond)


def _scale_for_step(config: CfgGuidanceSchedule, *, step_index: int, total_steps: int) -> float:
    if total_steps <= 1:
        fraction = 1.0
    else:
        fraction = min(1.0, max(0.0, float(step_index) / float(total_steps - 1)))
    return float(config.guidance_start + fraction * (config.guidance_end - config.guidance_start))


def _null_actions(model: object, actions: torch.Tensor) -> torch.Tensor:
    for candidate in (getattr(model, "model", None), model):
        get_null_cond = getattr(candidate, "get_null_cond", None)
        if callable(get_null_cond):
            return get_null_cond(actions)
    output = torch.zeros_like(actions)
    if output.shape[-1] > 0:
        output[..., -1] = 1
    return output


@lru_cache(maxsize=1)
def _configured_params() -> CfgGuidanceSchedule:
    payload = _load_sidecar()
    params = payload.get("params")
    if not isinstance(params, Mapping):
        raise RuntimeError("CFG_GUIDANCE_PARAMS_INVALID")
    start = params.get("guidance_start")
    end = params.get("guidance_end")
    if not isinstance(start, (int, float)) or isinstance(start, bool):
        raise RuntimeError("CFG_GUIDANCE_START_INVALID")
    if not isinstance(end, (int, float)) or isinstance(end, bool):
        raise RuntimeError("CFG_GUIDANCE_END_INVALID")
    guidance_start = float(start)
    guidance_end = float(end)
    if not math.isfinite(guidance_start) or guidance_start < 0.0:
        raise RuntimeError("CFG_GUIDANCE_START_INVALID")
    if not math.isfinite(guidance_end) or guidance_end < 0.0:
        raise RuntimeError("CFG_GUIDANCE_END_INVALID")
    return CfgGuidanceSchedule(guidance_start=guidance_start, guidance_end=guidance_end)


def _load_sidecar() -> Mapping[str, object]:
    try:
        payload = json.loads(_SIDECAR.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("CFG_GUIDANCE_SIDECAR_UNREADABLE") from exc
    if not isinstance(payload, Mapping):
        raise RuntimeError("CFG_GUIDANCE_SIDECAR_INVALID")
    if (
        payload.get("artifact_type") != "wmloop-materialized-primitive-smoke"
        or payload.get("primitive") != "cfg_guidance_schedule"
        or payload.get("hook") != "H4"
        or payload.get("materialization_state") != "acwm_runtime_hook_smoke"
    ):
        raise RuntimeError("CFG_GUIDANCE_SIDECAR_INVALID")
    return payload
'''
