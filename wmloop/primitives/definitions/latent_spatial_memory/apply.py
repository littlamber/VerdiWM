from __future__ import annotations

import difflib
import json
import math
from pathlib import Path
from typing import Mapping


def apply(repo_worktree: str, params: Mapping[str, object]) -> str:
    worktree = Path(repo_worktree)
    memory_slots, memory_weight = _validate_params(params)
    dynamics_path = worktree / "acwm" / "dynamics" / "diffusion_forcing_wm.py"
    original = dynamics_path.read_text(encoding="utf-8")
    patched = _patch_dynamics(original)
    sidecar = _sidecar_payload(memory_slots=memory_slots, memory_weight=memory_weight)
    return "\n".join(
        part.rstrip("\n")
        for part in (
            _unified_diff("acwm/dynamics/diffusion_forcing_wm.py", original, patched),
            _optional_new_file_diff(
                worktree,
                "acwm/wmloop_hooks/__init__.py",
                '"""Runtime hooks materialized by reviewed wm-loop primitive templates."""\n',
            ),
            _new_file_diff("acwm/wmloop_hooks/latent_spatial_memory.py", _HOOK_MODULE),
            _new_file_diff(
                "wmloop_interventions/latent_spatial_memory.json",
                json.dumps(sidecar, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            ),
        )
        if part.strip()
    ) + "\n"


def _validate_params(params: Mapping[str, object]) -> tuple[int, float]:
    slots = params.get("memory_slots")
    weight = params.get("memory_weight")
    if not isinstance(slots, int) or isinstance(slots, bool):
        raise ValueError("LATENT_SPATIAL_MEMORY_SLOTS_INVALID")
    if slots < 1 or slots > 128:
        raise ValueError("LATENT_SPATIAL_MEMORY_SLOTS_INVALID")
    if not isinstance(weight, (int, float)) or isinstance(weight, bool):
        raise ValueError("LATENT_SPATIAL_MEMORY_WEIGHT_INVALID")
    memory_weight = float(weight)
    if not math.isfinite(memory_weight) or memory_weight <= 0.0 or memory_weight > 1.0:
        raise ValueError("LATENT_SPATIAL_MEMORY_WEIGHT_INVALID")
    return int(slots), memory_weight


def _patch_dynamics(original: str) -> str:
    import_marker = "from acwm.model.interface import DIT_CLASS_MAP, VAE_CLASS_MAP\n"
    import_replacement = (
        "from acwm.model.interface import DIT_CLASS_MAP, VAE_CLASS_MAP\n"
        "from acwm.wmloop_hooks.latent_spatial_memory import apply_latent_spatial_memory\n"
    )
    parallel_variants = (
        (
            (
                "                    z_context = apply_drift_token_trim(z_context, anchor=z_0)\n"
                "                    v_pred = apply_cfg_guidance_schedule(\n"
                "                        self.model, z_context, t, a, step_index=i, total_steps=len(self.scheduler.timesteps)\n"
                "                    )\n"
            ),
            (
                "                    z_context = apply_drift_token_trim(z_context, anchor=z_0)\n"
                "                    z_context = apply_latent_spatial_memory(z_context)\n"
                "                    v_pred = apply_cfg_guidance_schedule(\n"
                "                        self.model, z_context, t, a, step_index=i, total_steps=len(self.scheduler.timesteps)\n"
                "                    )\n"
            ),
        ),
        (
            (
                "                    z_context = apply_drift_token_trim(z_context, anchor=z_0)\n"
                "                    v_pred = self.model(z_context, t, a)\n"
            ),
            (
                "                    z_context = apply_drift_token_trim(z_context, anchor=z_0)\n"
                "                    z_context = apply_latent_spatial_memory(z_context)\n"
                "                    v_pred = self.model(z_context, t, a)\n"
            ),
        ),
        (
            (
                "                    z_context = apply_first_frame_anchor(\n"
                "                        z, z_0, step_index=i, total_steps=len(self.scheduler.timesteps)\n"
                "                    )\n"
                "                    v_pred = apply_cfg_guidance_schedule(\n"
            ),
            (
                "                    z_context = apply_first_frame_anchor(\n"
                "                        z, z_0, step_index=i, total_steps=len(self.scheduler.timesteps)\n"
                "                    )\n"
                "                    z_context = apply_latent_spatial_memory(z_context)\n"
                "                    v_pred = apply_cfg_guidance_schedule(\n"
            ),
        ),
        (
            (
                "                    z_context = apply_first_frame_anchor(\n"
                "                        z, z_0, step_index=i, total_steps=len(self.scheduler.timesteps)\n"
                "                    )\n"
                "                    v_pred = self.model(z_context, t, a)\n"
            ),
            (
                "                    z_context = apply_first_frame_anchor(\n"
                "                        z, z_0, step_index=i, total_steps=len(self.scheduler.timesteps)\n"
                "                    )\n"
                "                    z_context = apply_latent_spatial_memory(z_context)\n"
                "                    v_pred = self.model(z_context, t, a)\n"
            ),
        ),
        (
            (
                "                    z_context = apply_drift_token_trim(z, anchor=z_0)\n"
                "                    v_pred = apply_cfg_guidance_schedule(\n"
                "                        self.model, z_context, t, a, step_index=i, total_steps=len(self.scheduler.timesteps)\n"
                "                    )\n"
            ),
            (
                "                    z_context = apply_drift_token_trim(z, anchor=z_0)\n"
                "                    z_context = apply_latent_spatial_memory(z_context)\n"
                "                    v_pred = apply_cfg_guidance_schedule(\n"
                "                        self.model, z_context, t, a, step_index=i, total_steps=len(self.scheduler.timesteps)\n"
                "                    )\n"
            ),
        ),
        (
            (
                "                    z_context = apply_drift_token_trim(z, anchor=z_0)\n"
                "                    v_pred = self.model(z_context, t, a)\n"
            ),
            (
                "                    z_context = apply_drift_token_trim(z, anchor=z_0)\n"
                "                    z_context = apply_latent_spatial_memory(z_context)\n"
                "                    v_pred = self.model(z_context, t, a)\n"
            ),
        ),
        (
            (
                "                    v_pred = apply_cfg_guidance_schedule(\n"
                "                        self.model, z, t, a, step_index=i, total_steps=len(self.scheduler.timesteps)\n"
                "                    )\n"
            ),
            (
                "                    z_context = apply_latent_spatial_memory(z)\n"
                "                    v_pred = apply_cfg_guidance_schedule(\n"
                "                        self.model, z_context, t, a, step_index=i, total_steps=len(self.scheduler.timesteps)\n"
                "                    )\n"
            ),
        ),
        (
            "                    v_pred = self.model(z, t, a)\n",
            (
                "                    z_context = apply_latent_spatial_memory(z)\n"
                "                    v_pred = self.model(z_context, t, a)\n"
            ),
        ),
    )
    autoregressive_variants = (
        (
            (
                "                        z_context = apply_drift_token_trim(z_context, anchor=z_0)\n"
                "                        v_pred = apply_cfg_guidance_schedule(\n"
                "                            self.model, z_context, t_seq, a_curr, step_index=i, total_steps=len(self.scheduler.timesteps)\n"
                "                        )\n"
            ),
            (
                "                        z_context = apply_drift_token_trim(z_context, anchor=z_0)\n"
                "                        z_context = apply_latent_spatial_memory(z_context)\n"
                "                        v_pred = apply_cfg_guidance_schedule(\n"
                "                            self.model, z_context, t_seq, a_curr, step_index=i, total_steps=len(self.scheduler.timesteps)\n"
                "                        )\n"
            ),
        ),
        (
            (
                "                        z_context = apply_drift_token_trim(z_context, anchor=z_0)\n"
                "                        v_pred = self.model(z_context, t_seq, a_curr)\n"
            ),
            (
                "                        z_context = apply_drift_token_trim(z_context, anchor=z_0)\n"
                "                        z_context = apply_latent_spatial_memory(z_context)\n"
                "                        v_pred = self.model(z_context, t_seq, a_curr)\n"
            ),
        ),
        (
            (
                "                        z_context = apply_first_frame_anchor(\n"
                "                            z_curr, z_0, step_index=i, total_steps=len(self.scheduler.timesteps)\n"
                "                        )\n"
                "                        v_pred = apply_cfg_guidance_schedule(\n"
            ),
            (
                "                        z_context = apply_first_frame_anchor(\n"
                "                            z_curr, z_0, step_index=i, total_steps=len(self.scheduler.timesteps)\n"
                "                        )\n"
                "                        z_context = apply_latent_spatial_memory(z_context)\n"
                "                        v_pred = apply_cfg_guidance_schedule(\n"
            ),
        ),
        (
            (
                "                        z_context = apply_first_frame_anchor(\n"
                "                            z_curr, z_0, step_index=i, total_steps=len(self.scheduler.timesteps)\n"
                "                        )\n"
                "                        v_pred = self.model(z_context, t_seq, a_curr)\n"
            ),
            (
                "                        z_context = apply_first_frame_anchor(\n"
                "                            z_curr, z_0, step_index=i, total_steps=len(self.scheduler.timesteps)\n"
                "                        )\n"
                "                        z_context = apply_latent_spatial_memory(z_context)\n"
                "                        v_pred = self.model(z_context, t_seq, a_curr)\n"
            ),
        ),
        (
            (
                "                        z_context = apply_drift_token_trim(z_curr, anchor=z_0)\n"
                "                        v_pred = apply_cfg_guidance_schedule(\n"
                "                            self.model, z_context, t_seq, a_curr, step_index=i, total_steps=len(self.scheduler.timesteps)\n"
                "                        )\n"
            ),
            (
                "                        z_context = apply_drift_token_trim(z_curr, anchor=z_0)\n"
                "                        z_context = apply_latent_spatial_memory(z_context)\n"
                "                        v_pred = apply_cfg_guidance_schedule(\n"
                "                            self.model, z_context, t_seq, a_curr, step_index=i, total_steps=len(self.scheduler.timesteps)\n"
                "                        )\n"
            ),
        ),
        (
            (
                "                        z_context = apply_drift_token_trim(z_curr, anchor=z_0)\n"
                "                        v_pred = self.model(z_context, t_seq, a_curr)\n"
            ),
            (
                "                        z_context = apply_drift_token_trim(z_curr, anchor=z_0)\n"
                "                        z_context = apply_latent_spatial_memory(z_context)\n"
                "                        v_pred = self.model(z_context, t_seq, a_curr)\n"
            ),
        ),
        (
            (
                "                        v_pred = apply_cfg_guidance_schedule(\n"
                "                            self.model, z_curr, t_seq, a_curr, step_index=i, total_steps=len(self.scheduler.timesteps)\n"
                "                        )\n"
            ),
            (
                "                        z_context = apply_latent_spatial_memory(z_curr)\n"
                "                        v_pred = apply_cfg_guidance_schedule(\n"
                "                            self.model, z_context, t_seq, a_curr, step_index=i, total_steps=len(self.scheduler.timesteps)\n"
                "                        )\n"
            ),
        ),
        (
            "                        v_pred = self.model(z_curr, t_seq, a_curr)\n",
            (
                "                        z_context = apply_latent_spatial_memory(z_curr)\n"
                "                        v_pred = self.model(z_context, t_seq, a_curr)\n"
            ),
        ),
    )
    patched = original
    if import_marker not in patched:
        raise ValueError("LATENT_SPATIAL_MEMORY_IMPORT_ANCHOR_MISSING")
    patched = patched.replace(import_marker, import_replacement, 1)
    patched = _replace_one_variant(
        patched,
        variants=parallel_variants,
        missing_code="LATENT_SPATIAL_MEMORY_PARALLEL_ANCHOR_MISSING",
    )
    patched = _replace_one_variant(
        patched,
        variants=autoregressive_variants,
        missing_code="LATENT_SPATIAL_MEMORY_AUTOREGRESSIVE_ANCHOR_MISSING",
    )
    return patched


def _replace_one_variant(text: str, *, variants: tuple[tuple[str, str], ...], missing_code: str) -> str:
    for marker, replacement in variants:
        if marker in text:
            return text.replace(marker, replacement, 1)
    raise ValueError(missing_code)


def _sidecar_payload(*, memory_slots: int, memory_weight: float) -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_type": "wmloop-materialized-primitive-smoke",
        "primitive": "latent_spatial_memory",
        "layer": "L2",
        "hook": "H2",
        "materialization_state": "acwm_runtime_hook_smoke",
        "params": {"memory_slots": memory_slots, "memory_weight": memory_weight},
        "runtime_hook": {
            "module": "acwm.wmloop_hooks.latent_spatial_memory",
            "function": "apply_latent_spatial_memory",
            "dynamics_patch": "acwm/dynamics/diffusion_forcing_wm.py",
        },
        "runtime_hook_paths": [
            "acwm/dynamics/diffusion_forcing_wm.py",
            "acwm/wmloop_hooks/latent_spatial_memory.py",
        ],
        "intent_to_code_contract": {
            "method_intent": "Carry a compact latent scene memory through long-horizon ACWM generation.",
            "runtime_behavior": (
                "Builds a mean latent memory from the first memory_slots frames available in the denoising context "
                "and blends future latent frames toward it before model prediction."
            ),
            "declared_proxy": (
                "Uses internal ACWM latent history as the memory source; no external representation extractor is used."
            ),
            "not_claimed": [
                "does not add DINO or other pretrained visual features",
                "does not change training loss",
                "does not alter frozen evaluator files",
            ],
        },
        "notes": [
            "Adds latent spatial memory injection inside DiffusionForcing_WM.generate.",
            "The hook is trial-worktree only and does not modify frozen evaluator files.",
        ],
    }


def _unified_diff(path: str, original: str, patched: str) -> str:
    if original == patched:
        raise ValueError("LATENT_SPATIAL_MEMORY_PATCH_EMPTY")
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


_SIDECAR = Path("wmloop_interventions/latent_spatial_memory.json")


@dataclass(frozen=True)
class LatentSpatialMemoryConfig:
    memory_slots: int
    memory_weight: float


def apply_latent_spatial_memory(latents: torch.Tensor) -> torch.Tensor:
    """Inject a compact latent scene-memory summary into future context frames."""

    config = _configured_params()
    if latents.ndim != 5:
        raise RuntimeError("LATENT_SPATIAL_MEMORY_LATENT_SHAPE_INVALID")
    if latents.shape[1] < 2 or config.memory_weight == 0.0:
        return latents
    slots = max(1, min(int(config.memory_slots), int(latents.shape[1])))
    memory = latents[:, :slots].float().mean(dim=1, keepdim=True).to(dtype=latents.dtype).detach()
    output = latents.clone()
    weight = output.new_tensor(config.memory_weight)
    output[:, 1:] = (1.0 - weight) * output[:, 1:] + weight * memory
    output[:, 0] = latents[:, 0]
    return output


@lru_cache(maxsize=1)
def _configured_params() -> LatentSpatialMemoryConfig:
    payload = _load_sidecar()
    params = payload.get("params")
    if not isinstance(params, Mapping):
        raise RuntimeError("LATENT_SPATIAL_MEMORY_PARAMS_INVALID")
    slots = params.get("memory_slots")
    weight = params.get("memory_weight")
    if not isinstance(slots, int) or isinstance(slots, bool) or slots < 1 or slots > 128:
        raise RuntimeError("LATENT_SPATIAL_MEMORY_SLOTS_INVALID")
    if not isinstance(weight, (int, float)) or isinstance(weight, bool):
        raise RuntimeError("LATENT_SPATIAL_MEMORY_WEIGHT_INVALID")
    memory_weight = float(weight)
    if not math.isfinite(memory_weight) or memory_weight <= 0.0 or memory_weight > 1.0:
        raise RuntimeError("LATENT_SPATIAL_MEMORY_WEIGHT_INVALID")
    return LatentSpatialMemoryConfig(memory_slots=int(slots), memory_weight=memory_weight)


def _load_sidecar() -> Mapping[str, object]:
    try:
        payload = json.loads(_SIDECAR.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("LATENT_SPATIAL_MEMORY_SIDECAR_UNREADABLE") from exc
    if not isinstance(payload, Mapping):
        raise RuntimeError("LATENT_SPATIAL_MEMORY_SIDECAR_INVALID")
    if (
        payload.get("artifact_type") != "wmloop-materialized-primitive-smoke"
        or payload.get("primitive") != "latent_spatial_memory"
        or payload.get("hook") != "H2"
        or payload.get("materialization_state") != "acwm_runtime_hook_smoke"
    ):
        raise RuntimeError("LATENT_SPATIAL_MEMORY_SIDECAR_INVALID")
    return payload
'''
