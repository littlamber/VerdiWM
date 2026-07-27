from __future__ import annotations

import difflib
import json
import math
from pathlib import Path
from typing import Mapping


def apply(repo_worktree: str, params: Mapping[str, object]) -> str:
    worktree = Path(repo_worktree)
    keep_tokens = _validate_keep_tokens(params)
    dynamics_path = worktree / "acwm" / "dynamics" / "diffusion_forcing_wm.py"
    original = dynamics_path.read_text(encoding="utf-8")
    patched = _patch_dynamics(original)
    sidecar = _sidecar_payload(keep_tokens)
    return "\n".join(
        part.rstrip("\n")
        for part in (
            _unified_diff("acwm/dynamics/diffusion_forcing_wm.py", original, patched),
            _optional_new_file_diff(
                worktree,
                "acwm/wmloop_hooks/__init__.py",
                '"""Runtime hooks materialized by reviewed wm-loop primitive templates."""\n',
            ),
            _new_file_diff("acwm/wmloop_hooks/drift_token_trim.py", _HOOK_MODULE),
            _new_file_diff(
                "wmloop_interventions/drift_token_trim.json",
                json.dumps(sidecar, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            ),
        )
        if part.strip()
    ) + "\n"


def _validate_keep_tokens(params: Mapping[str, object]) -> int:
    value = params.get("keep_tokens")
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError("DRIFT_TOKEN_TRIM_KEEP_TOKENS_INVALID")
    if value < 1 or not math.isfinite(float(value)):
        raise ValueError("DRIFT_TOKEN_TRIM_KEEP_TOKENS_INVALID")
    return int(value)


def _patch_dynamics(original: str) -> str:
    import_marker = "from acwm.model.interface import DIT_CLASS_MAP, VAE_CLASS_MAP\n"
    import_replacement = (
        "from acwm.model.interface import DIT_CLASS_MAP, VAE_CLASS_MAP\n"
        "from acwm.wmloop_hooks.drift_token_trim import apply_drift_token_trim\n"
    )
    parallel_marker = (
        "                with torch.no_grad():\n"
        "                    v_pred = self.model(z, t, a)\n"
        "                    z = self.scheduler.step(v_pred, t, z)\n"
    )
    cfg_parallel_marker = (
        "                with torch.no_grad():\n"
        "                    v_pred = apply_cfg_guidance_schedule(\n"
        "                        self.model, z, t, a, step_index=i, total_steps=len(self.scheduler.timesteps)\n"
        "                    )\n"
        "                    z = self.scheduler.step(v_pred, t, z)\n"
    )
    parallel_replacement = (
        "                with torch.no_grad():\n"
        "                    z_context = apply_drift_token_trim(z, anchor=z_0)\n"
        "                    v_pred = self.model(z_context, t, a)\n"
        "                    z = self.scheduler.step(v_pred, t, z)\n"
    )
    cfg_parallel_replacement = (
        "                with torch.no_grad():\n"
        "                    z_context = apply_drift_token_trim(z, anchor=z_0)\n"
        "                    v_pred = apply_cfg_guidance_schedule(\n"
        "                        self.model, z_context, t, a, step_index=i, total_steps=len(self.scheduler.timesteps)\n"
        "                    )\n"
        "                    z = self.scheduler.step(v_pred, t, z)\n"
    )
    autoregressive_marker = (
        "                    with torch.no_grad():\n"
        "                        v_pred = self.model(z_curr, t_seq, a_curr)\n"
        "                        z_curr = self.scheduler.step(v_pred, t_seq, z_curr)\n"
    )
    cfg_autoregressive_marker = (
        "                    with torch.no_grad():\n"
        "                        v_pred = apply_cfg_guidance_schedule(\n"
        "                            self.model, z_curr, t_seq, a_curr, step_index=i, total_steps=len(self.scheduler.timesteps)\n"
        "                        )\n"
        "                        z_curr = self.scheduler.step(v_pred, t_seq, z_curr)\n"
    )
    autoregressive_replacement = (
        "                    with torch.no_grad():\n"
        "                        z_context = apply_drift_token_trim(z_curr, anchor=z_0)\n"
        "                        v_pred = self.model(z_context, t_seq, a_curr)\n"
        "                        z_curr = self.scheduler.step(v_pred, t_seq, z_curr)\n"
    )
    cfg_autoregressive_replacement = (
        "                    with torch.no_grad():\n"
        "                        z_context = apply_drift_token_trim(z_curr, anchor=z_0)\n"
        "                        v_pred = apply_cfg_guidance_schedule(\n"
        "                            self.model, z_context, t_seq, a_curr, step_index=i, total_steps=len(self.scheduler.timesteps)\n"
        "                        )\n"
        "                        z_curr = self.scheduler.step(v_pred, t_seq, z_curr)\n"
    )
    patched = original
    if import_marker not in patched:
        raise ValueError("DRIFT_TOKEN_TRIM_IMPORT_ANCHOR_MISSING")
    patched = patched.replace(import_marker, import_replacement, 1)
    if cfg_parallel_marker in patched:
        patched = patched.replace(cfg_parallel_marker, cfg_parallel_replacement, 1)
    elif parallel_marker in patched:
        patched = patched.replace(parallel_marker, parallel_replacement, 1)
    else:
        raise ValueError("DRIFT_TOKEN_TRIM_PARALLEL_ANCHOR_MISSING")
    if cfg_autoregressive_marker in patched:
        patched = patched.replace(cfg_autoregressive_marker, cfg_autoregressive_replacement, 1)
    elif autoregressive_marker in patched:
        patched = patched.replace(autoregressive_marker, autoregressive_replacement, 1)
    else:
        raise ValueError("DRIFT_TOKEN_TRIM_AUTOREGRESSIVE_ANCHOR_MISSING")
    return patched


def _sidecar_payload(keep_tokens: int) -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_type": "wmloop-materialized-primitive-smoke",
        "primitive": "drift_token_trim",
        "layer": "L4",
        "hook": "H4",
        "materialization_state": "acwm_runtime_hook_smoke",
        "params": {"keep_tokens": keep_tokens},
        "runtime_hook": {
            "module": "acwm.wmloop_hooks.drift_token_trim",
            "function": "apply_drift_token_trim",
            "dynamics_patch": "acwm/dynamics/diffusion_forcing_wm.py",
        },
        "runtime_hook_paths": [
            "acwm/dynamics/diffusion_forcing_wm.py",
            "acwm/wmloop_hooks/drift_token_trim.py",
        ],
        "intent_to_code_contract": {
            "method_intent": "Reduce long-horizon appearance drift by suppressing stale latent context tokens during inference.",
            "runtime_behavior": (
                "Wraps DiffusionForcing_WM.generate denoising calls with a shape-preserving latent context trim that "
                "keeps the first frame and most recent tokens intact."
            ),
            "declared_proxy": (
                "Uses interpolation of stale non-anchor latent frames toward the first-frame anchor as the executable "
                "proxy for token trimming."
            ),
            "not_claimed": [
                "does not retrain the model",
                "does not change the evaluator",
                "does not remove tokens from tensor shapes",
            ],
        },
        "notes": [
            "Applies shape-preserving stale latent-context trimming inside DiffusionForcing_WM.generate.",
            "The hook is trial-worktree only and does not modify frozen evaluator files.",
        ],
    }


def _unified_diff(path: str, original: str, patched: str) -> str:
    if original == patched:
        raise ValueError("DRIFT_TOKEN_TRIM_PATCH_EMPTY")
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


_SIDECAR = Path("wmloop_interventions/drift_token_trim.json")


def apply_drift_token_trim(latents: torch.Tensor, *, anchor: torch.Tensor | None = None) -> torch.Tensor:
    """Apply shape-preserving stale-context trimming before denoising calls."""

    keep_tokens = _configured_keep_tokens()
    if latents.ndim != 5:
        raise RuntimeError("DRIFT_TOKEN_TRIM_LATENT_SHAPE_INVALID")
    if latents.shape[1] <= keep_tokens + 1:
        return latents
    output = latents.clone()
    trim_end = latents.shape[1] - keep_tokens
    if trim_end <= 1:
        return latents
    reference = _reference_token(output, anchor)
    stale = output[:, 1:trim_end]
    output[:, 1:trim_end] = reference + 0.25 * (stale - reference)
    output[:, 0] = latents[:, 0]
    output[:, -keep_tokens:] = latents[:, -keep_tokens:]
    return output


def _reference_token(latents: torch.Tensor, anchor: torch.Tensor | None) -> torch.Tensor:
    if anchor is None:
        return latents[:, :1].detach()
    if not isinstance(anchor, torch.Tensor):
        raise RuntimeError("DRIFT_TOKEN_TRIM_ANCHOR_INVALID")
    if anchor.ndim == 4:
        anchor = anchor.unsqueeze(1)
    if anchor.ndim != 5 or anchor.shape[0] != latents.shape[0] or anchor.shape[2:] != latents.shape[2:]:
        raise RuntimeError("DRIFT_TOKEN_TRIM_ANCHOR_INVALID")
    return anchor[:, :1].to(device=latents.device, dtype=latents.dtype).detach()


@lru_cache(maxsize=1)
def _configured_keep_tokens() -> int:
    payload = _load_sidecar()
    params = payload.get("params")
    if not isinstance(params, Mapping):
        raise RuntimeError("DRIFT_TOKEN_TRIM_PARAMS_INVALID")
    value = params.get("keep_tokens")
    if not isinstance(value, int) or isinstance(value, bool):
        raise RuntimeError("DRIFT_TOKEN_TRIM_KEEP_TOKENS_INVALID")
    if value < 1 or not math.isfinite(float(value)):
        raise RuntimeError("DRIFT_TOKEN_TRIM_KEEP_TOKENS_INVALID")
    return int(value)


def _load_sidecar() -> Mapping[str, object]:
    try:
        payload = json.loads(_SIDECAR.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("DRIFT_TOKEN_TRIM_SIDECAR_UNREADABLE") from exc
    if not isinstance(payload, Mapping):
        raise RuntimeError("DRIFT_TOKEN_TRIM_SIDECAR_INVALID")
    if (
        payload.get("artifact_type") != "wmloop-materialized-primitive-smoke"
        or payload.get("primitive") != "drift_token_trim"
        or payload.get("hook") != "H4"
        or payload.get("materialization_state") != "acwm_runtime_hook_smoke"
    ):
        raise RuntimeError("DRIFT_TOKEN_TRIM_SIDECAR_INVALID")
    return payload
'''
