from __future__ import annotations

import difflib
import json
import math
from pathlib import Path
from typing import Mapping


def apply(repo_worktree: str, params: Mapping[str, object]) -> str:
    worktree = Path(repo_worktree)
    anchor_every, anchor_weight = _validate_params(params)
    dynamics_path = worktree / "acwm" / "dynamics" / "diffusion_forcing_wm.py"
    original = dynamics_path.read_text(encoding="utf-8")
    patched = _patch_dynamics(original)
    sidecar = _sidecar_payload(anchor_every=anchor_every, anchor_weight=anchor_weight)
    return "\n".join(
        part.rstrip("\n")
        for part in (
            _unified_diff("acwm/dynamics/diffusion_forcing_wm.py", original, patched),
            _optional_new_file_diff(
                worktree,
                "acwm/wmloop_hooks/__init__.py",
                '"""Runtime hooks materialized by reviewed wm-loop primitive templates."""\n',
            ),
            _new_file_diff("acwm/wmloop_hooks/first_frame_anchor.py", _HOOK_MODULE),
            _new_file_diff(
                "wmloop_interventions/first_frame_anchor.json",
                json.dumps(sidecar, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            ),
        )
        if part.strip()
    ) + "\n"


def _validate_params(params: Mapping[str, object]) -> tuple[int, float]:
    every = params.get("anchor_every")
    weight = params.get("anchor_weight", 0.25)
    if not isinstance(every, int) or isinstance(every, bool):
        raise ValueError("FIRST_FRAME_ANCHOR_EVERY_INVALID")
    if every < 4 or every > 32:
        raise ValueError("FIRST_FRAME_ANCHOR_EVERY_INVALID")
    if not isinstance(weight, (int, float)) or isinstance(weight, bool):
        raise ValueError("FIRST_FRAME_ANCHOR_WEIGHT_INVALID")
    anchor_weight = float(weight)
    if not math.isfinite(anchor_weight) or anchor_weight < 0.0 or anchor_weight > 1.0:
        raise ValueError("FIRST_FRAME_ANCHOR_WEIGHT_INVALID")
    return int(every), anchor_weight


def _patch_dynamics(original: str) -> str:
    import_marker = "from acwm.model.interface import DIT_CLASS_MAP, VAE_CLASS_MAP\n"
    import_replacement = (
        "from acwm.model.interface import DIT_CLASS_MAP, VAE_CLASS_MAP\n"
        "from acwm.wmloop_hooks.first_frame_anchor import apply_first_frame_anchor\n"
    )
    parallel_variants = (
        (
            (
                "                with torch.no_grad():\n"
                "                    z_context = apply_drift_token_trim(z, anchor=z_0)\n"
                "                    v_pred = apply_cfg_guidance_schedule(\n"
                "                        self.model, z_context, t, a, step_index=i, total_steps=len(self.scheduler.timesteps)\n"
                "                    )\n"
                "                    z = self.scheduler.step(v_pred, t, z)\n"
            ),
            (
                "                with torch.no_grad():\n"
                "                    z_context = apply_first_frame_anchor(\n"
                "                        z, z_0, step_index=i, total_steps=len(self.scheduler.timesteps)\n"
                "                    )\n"
                "                    z_context = apply_drift_token_trim(z_context, anchor=z_0)\n"
                "                    v_pred = apply_cfg_guidance_schedule(\n"
                "                        self.model, z_context, t, a, step_index=i, total_steps=len(self.scheduler.timesteps)\n"
                "                    )\n"
                "                    z = self.scheduler.step(v_pred, t, z)\n"
            ),
        ),
        (
            (
                "                with torch.no_grad():\n"
                "                    z_context = apply_drift_token_trim(z, anchor=z_0)\n"
                "                    v_pred = self.model(z_context, t, a)\n"
                "                    z = self.scheduler.step(v_pred, t, z)\n"
            ),
            (
                "                with torch.no_grad():\n"
                "                    z_context = apply_first_frame_anchor(\n"
                "                        z, z_0, step_index=i, total_steps=len(self.scheduler.timesteps)\n"
                "                    )\n"
                "                    z_context = apply_drift_token_trim(z_context, anchor=z_0)\n"
                "                    v_pred = self.model(z_context, t, a)\n"
                "                    z = self.scheduler.step(v_pred, t, z)\n"
            ),
        ),
        (
            (
                "                with torch.no_grad():\n"
                "                    v_pred = apply_cfg_guidance_schedule(\n"
                "                        self.model, z, t, a, step_index=i, total_steps=len(self.scheduler.timesteps)\n"
                "                    )\n"
                "                    z = self.scheduler.step(v_pred, t, z)\n"
            ),
            (
                "                with torch.no_grad():\n"
                "                    z_context = apply_first_frame_anchor(\n"
                "                        z, z_0, step_index=i, total_steps=len(self.scheduler.timesteps)\n"
                "                    )\n"
                "                    v_pred = apply_cfg_guidance_schedule(\n"
                "                        self.model, z_context, t, a, step_index=i, total_steps=len(self.scheduler.timesteps)\n"
                "                    )\n"
                "                    z = self.scheduler.step(v_pred, t, z)\n"
            ),
        ),
        (
            (
                "                with torch.no_grad():\n"
                "                    v_pred = self.model(z, t, a)\n"
                "                    z = self.scheduler.step(v_pred, t, z)\n"
            ),
            (
                "                with torch.no_grad():\n"
                "                    z_context = apply_first_frame_anchor(\n"
                "                        z, z_0, step_index=i, total_steps=len(self.scheduler.timesteps)\n"
                "                    )\n"
                "                    v_pred = self.model(z_context, t, a)\n"
                "                    z = self.scheduler.step(v_pred, t, z)\n"
            ),
        ),
    )
    autoregressive_variants = (
        (
            (
                "                    with torch.no_grad():\n"
                "                        z_context = apply_drift_token_trim(z_curr, anchor=z_0)\n"
                "                        v_pred = apply_cfg_guidance_schedule(\n"
                "                            self.model, z_context, t_seq, a_curr, step_index=i, total_steps=len(self.scheduler.timesteps)\n"
                "                        )\n"
                "                        z_curr = self.scheduler.step(v_pred, t_seq, z_curr)\n"
            ),
            (
                "                    with torch.no_grad():\n"
                "                        z_context = apply_first_frame_anchor(\n"
                "                            z_curr, z_0, step_index=i, total_steps=len(self.scheduler.timesteps)\n"
                "                        )\n"
                "                        z_context = apply_drift_token_trim(z_context, anchor=z_0)\n"
                "                        v_pred = apply_cfg_guidance_schedule(\n"
                "                            self.model, z_context, t_seq, a_curr, step_index=i, total_steps=len(self.scheduler.timesteps)\n"
                "                        )\n"
                "                        z_curr = self.scheduler.step(v_pred, t_seq, z_curr)\n"
            ),
        ),
        (
            (
                "                    with torch.no_grad():\n"
                "                        z_context = apply_drift_token_trim(z_curr, anchor=z_0)\n"
                "                        v_pred = self.model(z_context, t_seq, a_curr)\n"
                "                        z_curr = self.scheduler.step(v_pred, t_seq, z_curr)\n"
            ),
            (
                "                    with torch.no_grad():\n"
                "                        z_context = apply_first_frame_anchor(\n"
                "                            z_curr, z_0, step_index=i, total_steps=len(self.scheduler.timesteps)\n"
                "                        )\n"
                "                        z_context = apply_drift_token_trim(z_context, anchor=z_0)\n"
                "                        v_pred = self.model(z_context, t_seq, a_curr)\n"
                "                        z_curr = self.scheduler.step(v_pred, t_seq, z_curr)\n"
            ),
        ),
        (
            (
                "                    with torch.no_grad():\n"
                "                        v_pred = apply_cfg_guidance_schedule(\n"
                "                            self.model, z_curr, t_seq, a_curr, step_index=i, total_steps=len(self.scheduler.timesteps)\n"
                "                        )\n"
                "                        z_curr = self.scheduler.step(v_pred, t_seq, z_curr)\n"
            ),
            (
                "                    with torch.no_grad():\n"
                "                        z_context = apply_first_frame_anchor(\n"
                "                            z_curr, z_0, step_index=i, total_steps=len(self.scheduler.timesteps)\n"
                "                        )\n"
                "                        v_pred = apply_cfg_guidance_schedule(\n"
                "                            self.model, z_context, t_seq, a_curr, step_index=i, total_steps=len(self.scheduler.timesteps)\n"
                "                        )\n"
                "                        z_curr = self.scheduler.step(v_pred, t_seq, z_curr)\n"
            ),
        ),
        (
            (
                "                    with torch.no_grad():\n"
                "                        v_pred = self.model(z_curr, t_seq, a_curr)\n"
                "                        z_curr = self.scheduler.step(v_pred, t_seq, z_curr)\n"
            ),
            (
                "                    with torch.no_grad():\n"
                "                        z_context = apply_first_frame_anchor(\n"
                "                            z_curr, z_0, step_index=i, total_steps=len(self.scheduler.timesteps)\n"
                "                        )\n"
                "                        v_pred = self.model(z_context, t_seq, a_curr)\n"
                "                        z_curr = self.scheduler.step(v_pred, t_seq, z_curr)\n"
            ),
        ),
    )
    patched = original
    if import_marker not in patched:
        raise ValueError("FIRST_FRAME_ANCHOR_IMPORT_ANCHOR_MISSING")
    patched = patched.replace(import_marker, import_replacement, 1)
    patched = _replace_one_variant(
        patched,
        variants=parallel_variants,
        missing_code="FIRST_FRAME_ANCHOR_PARALLEL_ANCHOR_MISSING",
    )
    patched = _replace_one_variant(
        patched,
        variants=autoregressive_variants,
        missing_code="FIRST_FRAME_ANCHOR_AUTOREGRESSIVE_ANCHOR_MISSING",
    )
    return patched


def _replace_one_variant(text: str, *, variants: tuple[tuple[str, str], ...], missing_code: str) -> str:
    for marker, replacement in variants:
        if marker in text:
            return text.replace(marker, replacement, 1)
    raise ValueError(missing_code)


def _sidecar_payload(*, anchor_every: int, anchor_weight: float) -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_type": "wmloop-materialized-primitive-smoke",
        "primitive": "first_frame_anchor",
        "layer": "L2",
        "hook": "H2",
        "materialization_state": "acwm_runtime_hook_smoke",
        "params": {"anchor_every": anchor_every, "anchor_weight": anchor_weight},
        "runtime_hook": {
            "module": "acwm.wmloop_hooks.first_frame_anchor",
            "function": "apply_first_frame_anchor",
            "dynamics_patch": "acwm/dynamics/diffusion_forcing_wm.py",
        },
        "runtime_hook_paths": [
            "acwm/dynamics/diffusion_forcing_wm.py",
            "acwm/wmloop_hooks/first_frame_anchor.py",
        ],
        "intent_to_code_contract": {
            "method_intent": "Use the observed first-frame latent as a memory anchor against long-horizon appearance drift.",
            "runtime_behavior": (
                "Injects a periodic, shape-preserving first-frame latent blend before ACWM denoising model calls in "
                "both parallel and autoregressive generation."
            ),
            "declared_proxy": (
                "Uses a configurable step cadence and blend weight as the executable proxy for first-frame memory anchoring."
            ),
            "not_claimed": [
                "does not change training loss",
                "does not overwrite the first latent frame produced by the evaluator path",
                "does not add external memory features",
            ],
        },
        "notes": [
            "Adds first-frame latent anchoring inside DiffusionForcing_WM.generate.",
            "The hook is trial-worktree only and does not modify frozen evaluator files.",
        ],
    }


def _unified_diff(path: str, original: str, patched: str) -> str:
    if original == patched:
        raise ValueError("FIRST_FRAME_ANCHOR_PATCH_EMPTY")
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


_SIDECAR = Path("wmloop_interventions/first_frame_anchor.json")


@dataclass(frozen=True)
class FirstFrameAnchorConfig:
    anchor_every: int
    anchor_weight: float


def apply_first_frame_anchor(
    latents: torch.Tensor,
    first_frame: torch.Tensor,
    *,
    step_index: int,
    total_steps: int,
) -> torch.Tensor:
    """Blend future latent frames toward the encoded first frame on a fixed cadence."""

    config = _configured_params()
    if config.anchor_weight == 0.0:
        return latents
    if latents.ndim != 5:
        raise RuntimeError("FIRST_FRAME_ANCHOR_LATENT_SHAPE_INVALID")
    if total_steps < 1:
        raise RuntimeError("FIRST_FRAME_ANCHOR_TOTAL_STEPS_INVALID")
    anchor = _normalize_anchor(first_frame, like=latents)
    if latents.shape[1] < 2:
        return latents
    if int(step_index) % config.anchor_every != 0 and int(step_index) != total_steps - 1:
        return latents
    output = latents.clone()
    weight = output.new_tensor(config.anchor_weight)
    output[:, 1:] = (1.0 - weight) * output[:, 1:] + weight * anchor
    output[:, 0] = latents[:, 0]
    return output


def _normalize_anchor(first_frame: torch.Tensor, *, like: torch.Tensor) -> torch.Tensor:
    if not isinstance(first_frame, torch.Tensor):
        raise RuntimeError("FIRST_FRAME_ANCHOR_INPUT_INVALID")
    anchor = first_frame
    if anchor.ndim == 4:
        anchor = anchor.unsqueeze(1)
    if anchor.ndim != 5 or anchor.shape[0] != like.shape[0] or anchor.shape[2:] != like.shape[2:]:
        raise RuntimeError("FIRST_FRAME_ANCHOR_INPUT_INVALID")
    return anchor[:, :1].to(device=like.device, dtype=like.dtype).detach()


@lru_cache(maxsize=1)
def _configured_params() -> FirstFrameAnchorConfig:
    payload = _load_sidecar()
    params = payload.get("params")
    if not isinstance(params, Mapping):
        raise RuntimeError("FIRST_FRAME_ANCHOR_PARAMS_INVALID")
    every = params.get("anchor_every")
    weight = params.get("anchor_weight", 0.25)
    if not isinstance(every, int) or isinstance(every, bool) or every < 4 or every > 32:
        raise RuntimeError("FIRST_FRAME_ANCHOR_EVERY_INVALID")
    if not isinstance(weight, (int, float)) or isinstance(weight, bool):
        raise RuntimeError("FIRST_FRAME_ANCHOR_WEIGHT_INVALID")
    anchor_weight = float(weight)
    if not math.isfinite(anchor_weight) or anchor_weight < 0.0 or anchor_weight > 1.0:
        raise RuntimeError("FIRST_FRAME_ANCHOR_WEIGHT_INVALID")
    return FirstFrameAnchorConfig(anchor_every=int(every), anchor_weight=anchor_weight)


def _load_sidecar() -> Mapping[str, object]:
    try:
        payload = json.loads(_SIDECAR.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("FIRST_FRAME_ANCHOR_SIDECAR_UNREADABLE") from exc
    if not isinstance(payload, Mapping):
        raise RuntimeError("FIRST_FRAME_ANCHOR_SIDECAR_INVALID")
    if (
        payload.get("artifact_type") != "wmloop-materialized-primitive-smoke"
        or payload.get("primitive") != "first_frame_anchor"
        or payload.get("hook") != "H2"
        or payload.get("materialization_state") != "acwm_runtime_hook_smoke"
    ):
        raise RuntimeError("FIRST_FRAME_ANCHOR_SIDECAR_INVALID")
    return payload
'''
