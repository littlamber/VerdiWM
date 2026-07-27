from __future__ import annotations

import difflib
import json
import math
from pathlib import Path
from typing import Mapping


def apply(repo_worktree: str, params: Mapping[str, object]) -> str:
    worktree = Path(repo_worktree)
    history_noise = _validate_history_noise(params)
    trainer_path = worktree / "acwm" / "trainer" / "train_dynamics.py"
    original = trainer_path.read_text(encoding="utf-8")
    patched = _patch_trainer(original)
    sidecar = _sidecar_payload(history_noise)
    return "\n".join(
        part.rstrip("\n")
        for part in (
            _unified_diff("acwm/trainer/train_dynamics.py", original, patched),
            _optional_new_file_diff(
                worktree,
                "acwm/wmloop_hooks/__init__.py",
                '"""Runtime hooks materialized by reviewed wm-loop primitive templates."""\n',
            ),
            _new_file_diff("acwm/wmloop_hooks/history_noise_schedule.py", _HOOK_MODULE),
            _new_file_diff(
                "wmloop_interventions/history_noise_schedule.json",
                json.dumps(sidecar, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            ),
        )
        if part.strip()
    ) + "\n"


def _validate_history_noise(params: Mapping[str, object]) -> float:
    value = params.get("history_noise")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError("HISTORY_NOISE_SCHEDULE_VALUE_INVALID")
    history_noise = float(value)
    if not math.isfinite(history_noise) or history_noise < 0.0 or history_noise > 1.0:
        raise ValueError("HISTORY_NOISE_SCHEDULE_VALUE_INVALID")
    return history_noise


def _patch_trainer(original: str) -> str:
    import_marker = "from acwm.utils.visualization import visualize_layout\n"
    import_replacement = (
        "from acwm.utils.visualization import visualize_layout\n"
        "from acwm.wmloop_hooks.history_noise_schedule import apply_history_noise_schedule\n"
    )
    encode_marker = (
        "            # Encode\n"
        "            with torch.no_grad():\n"
        "                z = model.module.encode_obs(obs) if hasattr(model, 'module') else model.encode_obs(obs)\n"
        "            \n"
        "            # Forward & Backward\n"
    )
    encode_replacement = (
        "            # Encode\n"
        "            with torch.no_grad():\n"
        "                z = model.module.encode_obs(obs) if hasattr(model, 'module') else model.encode_obs(obs)\n"
        "            z = apply_history_noise_schedule(z)\n"
        "            \n"
        "            # Forward & Backward\n"
    )
    patched = original
    for marker, replacement, code in (
        (import_marker, import_replacement, "HISTORY_NOISE_SCHEDULE_IMPORT_ANCHOR_MISSING"),
        (encode_marker, encode_replacement, "HISTORY_NOISE_SCHEDULE_ENCODE_ANCHOR_MISSING"),
    ):
        if marker not in patched:
            raise ValueError(code)
        patched = patched.replace(marker, replacement, 1)
    return patched


def _sidecar_payload(history_noise: float) -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_type": "wmloop-materialized-primitive-smoke",
        "primitive": "history_noise_schedule",
        "layer": "L2",
        "hook": "H2",
        "materialization_state": "acwm_runtime_hook_smoke",
        "params": {"history_noise": history_noise},
        "runtime_hook": {
            "module": "acwm.wmloop_hooks.history_noise_schedule",
            "function": "apply_history_noise_schedule",
            "trainer_patch": "acwm/trainer/train_dynamics.py",
        },
        "runtime_hook_paths": [
            "acwm/trainer/train_dynamics.py",
            "acwm/wmloop_hooks/history_noise_schedule.py",
        ],
        "intent_to_code_contract": {
            "method_intent": "Expose the model to imperfect latent histories during training to reduce autoregressive brittleness.",
            "runtime_behavior": (
                "Adds bounded Gaussian perturbations only to encoded history latents before ACWM training_loss; "
                "the current target latent remains unchanged."
            ),
            "declared_proxy": "Uses per-sample latent standard deviation as the scale for train-time history corruption.",
            "not_claimed": [
                "does not change held-out rollouts",
                "does not alter action labels",
                "does not claim robustness without paired eval evidence",
            ],
        },
        "notes": [
            "Adds bounded latent history noise before ACWM training_loss to reduce train/inference history brittleness.",
            "The hook is trial-worktree only and does not modify frozen evaluator files.",
        ],
    }


def _unified_diff(path: str, original: str, patched: str) -> str:
    if original == patched:
        raise ValueError("HISTORY_NOISE_SCHEDULE_PATCH_EMPTY")
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


_SIDECAR = Path("wmloop_interventions/history_noise_schedule.json")


def apply_history_noise_schedule(latents: torch.Tensor) -> torch.Tensor:
    """Inject bounded noise into latent history frames during training."""

    strength = _configured_strength()
    if strength == 0.0:
        return latents
    if latents.ndim != 5:
        raise RuntimeError("HISTORY_NOISE_SCHEDULE_LATENT_SHAPE_INVALID")
    if latents.shape[1] < 2:
        return latents
    output = latents.clone()
    history = output[:, :-1]
    scale = history.float().std(dim=(2, 3, 4), keepdim=True).clamp_min(1e-6).detach()
    history.add_(torch.randn_like(history) * (strength * scale).to(dtype=history.dtype))
    return output


@lru_cache(maxsize=1)
def _configured_strength() -> float:
    payload = _load_sidecar()
    params = payload.get("params")
    if not isinstance(params, Mapping):
        raise RuntimeError("HISTORY_NOISE_SCHEDULE_PARAMS_INVALID")
    value = params.get("history_noise")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise RuntimeError("HISTORY_NOISE_SCHEDULE_VALUE_INVALID")
    strength = float(value)
    if not math.isfinite(strength) or strength < 0.0 or strength > 1.0:
        raise RuntimeError("HISTORY_NOISE_SCHEDULE_VALUE_INVALID")
    return strength


def _load_sidecar() -> Mapping[str, object]:
    try:
        payload = json.loads(_SIDECAR.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("HISTORY_NOISE_SCHEDULE_SIDECAR_UNREADABLE") from exc
    if not isinstance(payload, Mapping):
        raise RuntimeError("HISTORY_NOISE_SCHEDULE_SIDECAR_INVALID")
    if (
        payload.get("artifact_type") != "wmloop-materialized-primitive-smoke"
        or payload.get("primitive") != "history_noise_schedule"
        or payload.get("hook") != "H2"
        or payload.get("materialization_state") != "acwm_runtime_hook_smoke"
    ):
        raise RuntimeError("HISTORY_NOISE_SCHEDULE_SIDECAR_INVALID")
    return payload
'''
