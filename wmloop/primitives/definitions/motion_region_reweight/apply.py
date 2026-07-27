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
            _unified_diff("acwm/trainer/train_dynamics.py", original, patched),
            _optional_new_file_diff(
                worktree,
                "acwm/wmloop_hooks/__init__.py",
                '"""Runtime hooks materialized by reviewed wm-loop primitive templates.\"\"\"\n',
            ),
            _new_file_diff("acwm/wmloop_hooks/motion_region_reweight.py", _HOOK_MODULE),
            _new_file_diff(
                "wmloop_interventions/motion_region_reweight.json",
                json.dumps(sidecar, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            ),
        )
        if part.strip()
    ) + "\n"


def _validate_weight(params: Mapping[str, object]) -> float:
    value = params.get("weight")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError("MOTION_REGION_REWEIGHT_WEIGHT_INVALID")
    weight = float(value)
    if not math.isfinite(weight) or weight <= 0.0 or weight > 4.0:
        raise ValueError("MOTION_REGION_REWEIGHT_WEIGHT_INVALID")
    return weight


def _patch_trainer(original: str) -> str:
    import_marker = "from acwm.utils.visualization import visualize_layout\n"
    import_replacement = (
        "from acwm.utils.visualization import visualize_layout\n"
        "from acwm.wmloop_hooks.motion_region_reweight import apply_motion_region_reweight_config\n"
    )
    call_marker = "    # Automatically set action_dim from dataset registry\n"
    call_replacement = (
        "    config = apply_motion_region_reweight_config(config)\n\n"
        "    # Automatically set action_dim from dataset registry\n"
    )
    if import_marker not in original:
        raise ValueError("MOTION_REGION_REWEIGHT_IMPORT_ANCHOR_MISSING")
    if call_marker not in original:
        raise ValueError("MOTION_REGION_REWEIGHT_CONFIG_ANCHOR_MISSING")
    patched = original.replace(import_marker, import_replacement, 1)
    return patched.replace(call_marker, call_replacement, 1)


def _sidecar_payload(weight: float) -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_type": "wmloop-materialized-primitive-smoke",
        "primitive": "motion_region_reweight",
        "layer": "L5",
        "hook": "H5",
        "materialization_state": "acwm_runtime_hook_smoke",
        "params": {"weight": weight},
        "runtime_hook": {
            "module": "acwm.wmloop_hooks.motion_region_reweight",
            "function": "apply_motion_region_reweight_config",
            "trainer_patch": "acwm/trainer/train_dynamics.py",
            "upstream_parameter": "model_config.motion_weighting_gamma",
        },
        "runtime_hook_paths": [
            "acwm/trainer/train_dynamics.py",
            "acwm/wmloop_hooks/motion_region_reweight.py",
            "acwm/dynamics/diffusion_forcing_wm.py",
        ],
        "intent_to_code_contract": {
            "method_intent": "Increase the training weight of latent regions with real temporal motion so dynamic physics are not dominated by static appearance.",
            "runtime_behavior": "Reads the reviewed weight from the intervention sidecar and writes it to model_config.motion_weighting_gamma before the ACWM dynamics model is constructed; the upstream training_loss then applies 1 + gamma * latent temporal difference.",
            "declared_proxy": "The official ACWM-Phys latent temporal-difference weighting implementation is used as the executable proxy for motion-region reweighting.",
            "not_claimed": [
                "does not alter held-out evaluation or evaluator code",
                "does not use ground-truth rollouts as training labels",
                "does not establish quality improvement until the official metric gate passes",
            ],
        },
        "notes": [
            "Uses the vendor's existing motion_weighting_gamma path, which is disabled by default at gamma=0.",
            "The sidecar and config mutation are trial-worktree artifacts and are recorded in the materialization receipt.",
        ],
    }


def _unified_diff(path: str, original: str, patched: str) -> str:
    if original == patched:
        raise ValueError("MOTION_REGION_REWEIGHT_PATCH_EMPTY")
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


_SIDECAR = Path("wmloop_interventions/motion_region_reweight.json")


def apply_motion_region_reweight_config(config: Mapping[str, object]) -> dict[str, object]:
    """Bind the reviewed intervention weight to ACWM's existing gamma control."""

    output = dict(config)
    model_config = dict(output.get("model_config") or {})
    model_config["motion_weighting_gamma"] = _configured_weight()
    output["model_config"] = model_config
    return output


@lru_cache(maxsize=1)
def _configured_weight() -> float:
    try:
        payload = json.loads(_SIDECAR.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("MOTION_REGION_REWEIGHT_SIDECAR_UNREADABLE") from exc
    if not isinstance(payload, Mapping) or payload.get("primitive") != "motion_region_reweight":
        raise RuntimeError("MOTION_REGION_REWEIGHT_SIDECAR_INVALID")
    params = payload.get("params")
    value = params.get("weight") if isinstance(params, Mapping) else None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise RuntimeError("MOTION_REGION_REWEIGHT_WEIGHT_INVALID")
    weight = float(value)
    if not math.isfinite(weight) or weight <= 0.0 or weight > 4.0:
        raise RuntimeError("MOTION_REGION_REWEIGHT_WEIGHT_INVALID")
    return weight
'''
