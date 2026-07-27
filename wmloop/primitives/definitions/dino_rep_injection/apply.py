from __future__ import annotations

import difflib
import json
import math
from pathlib import Path
from typing import Mapping


def apply(repo_worktree: str, params: Mapping[str, object]) -> str:
    worktree = Path(repo_worktree)
    injection_weight = _validate_weight(params)
    dynamics_path = worktree / "acwm" / "dynamics" / "diffusion_forcing_wm.py"
    trainer_path = worktree / "acwm" / "trainer" / "train_dynamics.py"
    dynamics_original = dynamics_path.read_text(encoding="utf-8")
    trainer_original = trainer_path.read_text(encoding="utf-8")
    dynamics_patched = _patch_dynamics(dynamics_original)
    trainer_patched = _patch_trainer(trainer_original)
    sidecar = _sidecar_payload(injection_weight)
    return "\n".join(
        part.rstrip("\n")
        for part in (
            _unified_diff(
                "acwm/dynamics/diffusion_forcing_wm.py",
                dynamics_original,
                dynamics_patched,
            ),
            _unified_diff(
                "acwm/trainer/train_dynamics.py",
                trainer_original,
                trainer_patched,
            ),
            _optional_new_file_diff(
                worktree,
                "acwm/wmloop_hooks/__init__.py",
                '"""Runtime hooks materialized by reviewed wm-loop primitive templates."""\n',
            ),
            _new_file_diff("acwm/wmloop_hooks/dino_rep_injection.py", _HOOK_MODULE),
            _new_file_diff(
                "wmloop_interventions/dino_rep_injection.json",
                json.dumps(sidecar, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            ),
        )
        if part.strip()
    ) + "\n"


def _validate_weight(params: Mapping[str, object]) -> float:
    value = params.get("injection_weight")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError("DINO_REP_INJECTION_WEIGHT_INVALID")
    weight = float(value)
    if not math.isfinite(weight) or weight < 0.0 or weight > 1.0:
        raise ValueError("DINO_REP_INJECTION_WEIGHT_INVALID")
    return weight


def _patch_dynamics(original: str) -> str:
    import_marker = "from acwm.model.interface import DIT_CLASS_MAP, VAE_CLASS_MAP\n"
    import_replacement = (
        "from acwm.model.interface import DIT_CLASS_MAP, VAE_CLASS_MAP\n"
        "from acwm.wmloop_hooks.dino_rep_injection import apply_dino_representation_loss\n"
    )
    loss_marker = "        loss = (weights.view(B, T, 1, 1, 1) * loss_map).mean()\n"
    loss_replacement = (
        "        loss = (weights.view(B, T, 1, 1, 1) * loss_map).mean()\n"
        "        sigma = self.scheduler.sigmas[t_indices].to(device=z_t.device, dtype=z_t.dtype)\n"
        "        predicted_clean = z_t - sigma.view(B, T, 1, 1, 1) * v_pred\n"
        "        wmloop_dino_observations = getattr(self, '_wmloop_dino_observations', None)\n"
        "        loss, _ = apply_dino_representation_loss(loss, predicted_clean, wmloop_dino_observations)\n"
    )
    patched = original
    for marker, replacement, code in (
        (import_marker, import_replacement, "DINO_REP_INJECTION_IMPORT_ANCHOR_MISSING"),
        (loss_marker, loss_replacement, "DINO_REP_INJECTION_LOSS_ANCHOR_MISSING"),
    ):
        if marker not in patched:
            raise ValueError(code)
        patched = patched.replace(marker, replacement, 1)
    return patched


def _patch_trainer(original: str) -> str:
    loss_marker = (
        "            loss = model.module.training_loss(z, action) if hasattr(model, 'module') else model.training_loss(z, action)\n"
    )
    loss_replacement = (
        "            wmloop_dino_target = model.module if hasattr(model, 'module') else model\n"
        "            wmloop_dino_target._wmloop_dino_observations = obs\n"
        "            loss = model.module.training_loss(z, action) if hasattr(model, 'module') else model.training_loss(z, action)\n"
    )
    if loss_marker not in original:
        raise ValueError("DINO_REP_INJECTION_TRAINER_LOSS_ANCHOR_MISSING")
    return original.replace(loss_marker, loss_replacement, 1)


def _sidecar_payload(injection_weight: float) -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_type": "wmloop-materialized-primitive-smoke",
        "primitive": "dino_rep_injection",
        "layer": "L2",
        "hook": "H2",
        "materialization_state": "acwm_runtime_hook_smoke",
        "params": {"injection_weight": injection_weight},
        "runtime_hook": {
            "module": "acwm.wmloop_hooks.dino_rep_injection",
            "function": "apply_dino_representation_loss",
            "dynamics_patch": "acwm/dynamics/diffusion_forcing_wm.py",
            "trainer_patch": "acwm/trainer/train_dynamics.py",
        },
        "runtime_hook_paths": [
            "acwm/dynamics/diffusion_forcing_wm.py",
            "acwm/trainer/train_dynamics.py",
            "acwm/wmloop_hooks/dino_rep_injection.py",
        ],
        "asset_contract": {
            "repo_environment_variable": "WMLOOP_DINO_REPO",
            "weight_environment_variable": "WMLOOP_DINO_WEIGHT",
            "accepted_architectures": ["dino_vits8", "dino_vits16", "dino_vitb8", "dino_vitb16"],
        },
        "intent_to_code_contract": {
            "method_intent": "Inject frozen DINO appearance structure into ACWM fine-tuning without changing the evaluator.",
            "runtime_behavior": (
                "Reconstructs the predicted clean flow-matching latent, compares its first/last-frame cosine relation "
                "against a frozen DINO teacher relation, and adds the weighted robust alignment loss every eight steps."
            ),
            "declared_proxy": (
                "Uses first-to-last temporal relation distillation rather than direct feature projection because ACWM "
                "and DINO feature dimensions differ and the frozen registry exposes only injection_weight."
            ),
            "not_claimed": [
                "does not modify frozen evaluator files",
                "does not train or checkpoint the DINO teacher",
                "does not claim quality improvement until the official paired gate passes",
            ],
        },
        "notes": [
            "The DINO teacher is loaded from explicit runtime assets and remains frozen.",
            "The fixed eight-step cadence bounds training overhead and is recorded in the intent contract.",
        ],
    }


def _unified_diff(path: str, original: str, patched: str) -> str:
    if original == patched:
        raise ValueError("DINO_REP_INJECTION_PATCH_EMPTY")
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

import importlib.util
import json
import math
import os
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Mapping

import torch
import torch.nn.functional as F


_SIDECAR = Path("wmloop_interventions/dino_rep_injection.json")
_CADENCE = 8
_CALL_COUNT = 0
_TEACHER: torch.nn.Module | None = None
_TEACHER_DEVICE: torch.device | None = None


def apply_dino_representation_loss(
    base_loss: torch.Tensor,
    predicted_clean_latents: torch.Tensor,
    observations: torch.Tensor | None,
    *,
    force: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Align predicted clean-latent temporal relations with a frozen DINO teacher."""

    global _CALL_COUNT
    weight = _configured_weight()
    if predicted_clean_latents.ndim != 5 or predicted_clean_latents.shape[1] < 2:
        raise RuntimeError("DINO_REP_INJECTION_LATENT_SHAPE_INVALID")
    if observations is None:
        return base_loss, {"train/wmloop_dino_rep_injection_active": 0.0}
    if observations.ndim != 5 or observations.shape[1] < 2:
        raise RuntimeError("DINO_REP_INJECTION_OBSERVATION_SHAPE_INVALID")
    _CALL_COUNT += 1
    if weight == 0.0:
        return base_loss, {"train/wmloop_dino_rep_injection_active": 0.0}
    if not force and (_CALL_COUNT - 1) % _CADENCE != 0:
        return base_loss, {"train/wmloop_dino_rep_injection_active": 0.0}

    latent_endpoints = predicted_clean_latents.float()[:, (0, -1)].mean(dim=(2, 3))
    predicted_similarity = F.cosine_similarity(latent_endpoints[:, 0], latent_endpoints[:, 1], dim=-1)
    teacher_similarity = _teacher_similarity(observations).to(
        device=predicted_similarity.device,
        dtype=predicted_similarity.dtype,
    )
    if teacher_similarity.shape != predicted_similarity.shape:
        raise RuntimeError("DINO_REP_INJECTION_TEACHER_SHAPE_INVALID")
    alignment = F.smooth_l1_loss(predicted_similarity, teacher_similarity)
    auxiliary = base_loss.new_tensor(weight) * alignment.to(dtype=base_loss.dtype)
    total = base_loss + auxiliary
    return total, {
        "train/wmloop_dino_rep_injection_active": 1.0,
        "train/wmloop_dino_rep_injection_loss": float(auxiliary.detach().cpu()),
        "train/wmloop_dino_rep_injection_weight": float(weight),
    }


def _teacher_similarity(observations: torch.Tensor) -> torch.Tensor:
    endpoints = observations[:, (0, -1)]
    if endpoints.shape[2] == 3:
        images = endpoints
    elif endpoints.shape[-1] == 3:
        images = endpoints.permute(0, 1, 4, 2, 3).contiguous()
    else:
        raise RuntimeError("DINO_REP_INJECTION_CHANNEL_SHAPE_INVALID")
    batch = images.shape[0]
    images = images.reshape(batch * 2, 3, images.shape[-2], images.shape[-1]).float()
    images = F.interpolate(images, size=(224, 224), mode="bicubic", align_corners=False, antialias=True)
    mean = images.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = images.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    images = (images.clamp(0.0, 1.0) - mean) / std
    teacher = _teacher(images.device)
    with torch.no_grad():
        features = teacher(images)
    if not isinstance(features, torch.Tensor) or features.ndim != 2 or features.shape[0] != batch * 2:
        raise RuntimeError("DINO_REP_INJECTION_TEACHER_OUTPUT_INVALID")
    features = features.float().reshape(batch, 2, -1)
    return F.cosine_similarity(features[:, 0], features[:, 1], dim=-1).detach()


def _teacher(device: torch.device) -> torch.nn.Module:
    global _TEACHER, _TEACHER_DEVICE
    if _TEACHER is None:
        repo = Path(os.environ.get("WMLOOP_DINO_REPO", "")).expanduser()
        weight = Path(os.environ.get("WMLOOP_DINO_WEIGHT", "")).expanduser()
        if not repo.is_dir() or not weight.is_file() or repo.is_symlink() or weight.is_symlink():
            raise RuntimeError("DINO_REP_INJECTION_ASSET_MISSING")
        module = _load_vision_transformer(repo / "vision_transformer.py")
        state = torch.load(weight, map_location="cpu", weights_only=True)
        if not isinstance(state, Mapping):
            raise RuntimeError("DINO_REP_INJECTION_WEIGHT_INVALID")
        projection = state.get("patch_embed.proj.weight")
        if not isinstance(projection, torch.Tensor) or projection.ndim != 4:
            raise RuntimeError("DINO_REP_INJECTION_WEIGHT_INVALID")
        dimension, _, patch_height, patch_width = projection.shape
        if patch_height != patch_width or dimension not in {384, 768} or patch_height not in {8, 16}:
            raise RuntimeError("DINO_REP_INJECTION_ARCHITECTURE_UNSUPPORTED")
        constructor_name = "vit_small" if dimension == 384 else "vit_base"
        constructor = getattr(module, constructor_name, None)
        if not callable(constructor):
            raise RuntimeError("DINO_REP_INJECTION_CONSTRUCTOR_MISSING")
        teacher = constructor(patch_size=int(patch_height), num_classes=0)
        teacher.load_state_dict(state, strict=True)
        teacher.requires_grad_(False)
        teacher.eval()
        _TEACHER = teacher
    if _TEACHER_DEVICE != device:
        _TEACHER = _TEACHER.to(device)
        _TEACHER_DEVICE = device
    return _TEACHER


def _load_vision_transformer(path: Path) -> ModuleType:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("DINO_REP_INJECTION_REPO_INVALID")
    spec = importlib.util.spec_from_file_location("wmloop_dino_vision_transformer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("DINO_REP_INJECTION_REPO_INVALID")
    module = importlib.util.module_from_spec(spec)
    repo = str(path.parent)
    sys.path.insert(0, repo)
    try:
        spec.loader.exec_module(module)
    finally:
        if sys.path and sys.path[0] == repo:
            sys.path.pop(0)
    return module


@lru_cache(maxsize=1)
def _configured_weight() -> float:
    payload = _load_sidecar()
    params = payload.get("params")
    if not isinstance(params, Mapping):
        raise RuntimeError("DINO_REP_INJECTION_PARAMS_INVALID")
    value = params.get("injection_weight")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise RuntimeError("DINO_REP_INJECTION_WEIGHT_INVALID")
    weight = float(value)
    if not math.isfinite(weight) or weight < 0.0 or weight > 1.0:
        raise RuntimeError("DINO_REP_INJECTION_WEIGHT_INVALID")
    return weight


def _load_sidecar() -> Mapping[str, object]:
    try:
        payload = json.loads(_SIDECAR.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("DINO_REP_INJECTION_SIDECAR_UNREADABLE") from exc
    if not isinstance(payload, Mapping):
        raise RuntimeError("DINO_REP_INJECTION_SIDECAR_INVALID")
    if (
        payload.get("artifact_type") != "wmloop-materialized-primitive-smoke"
        or payload.get("primitive") != "dino_rep_injection"
        or payload.get("hook") != "H2"
        or payload.get("materialization_state") != "acwm_runtime_hook_smoke"
    ):
        raise RuntimeError("DINO_REP_INJECTION_SIDECAR_INVALID")
    return payload
'''
