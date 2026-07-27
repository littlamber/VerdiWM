from __future__ import annotations

import difflib
import json
import math
from pathlib import Path
from typing import Mapping


def apply(repo_worktree: str, params: Mapping[str, object]) -> str:
    worktree = Path(repo_worktree)
    blend, max_gain = _validate_params(params)
    dit_path = worktree / "acwm" / "model" / "dit" / "dit.py"
    original = dit_path.read_text(encoding="utf-8")
    patched = _patch_dit(original)
    sidecar = _sidecar_payload(blend=blend, max_gain=max_gain)
    return "\n".join(
        part.rstrip("\n")
        for part in (
            _unified_diff("acwm/model/dit/dit.py", original, patched),
            _optional_new_file_diff(
                worktree,
                "acwm/wmloop_hooks/__init__.py",
                '"""Runtime hooks materialized by reviewed wm-loop primitive templates."""\n',
            ),
            _new_file_diff("acwm/wmloop_hooks/action_dimension_balancing.py", _HOOK_MODULE),
            _new_file_diff(
                "wmloop_interventions/action_dimension_balancing.json",
                json.dumps(sidecar, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            ),
        )
        if part.strip()
    ) + "\n"


def _validate_params(params: Mapping[str, object]) -> tuple[float, float]:
    blend_value = params.get("blend")
    gain_value = params.get("max_gain")
    if not isinstance(blend_value, (int, float)) or isinstance(blend_value, bool):
        raise ValueError("ACTION_DIMENSION_BALANCING_BLEND_INVALID")
    if not isinstance(gain_value, (int, float)) or isinstance(gain_value, bool):
        raise ValueError("ACTION_DIMENSION_BALANCING_MAX_GAIN_INVALID")
    blend = float(blend_value)
    max_gain = float(gain_value)
    if not math.isfinite(blend) or blend <= 0.0 or blend > 1.0:
        raise ValueError("ACTION_DIMENSION_BALANCING_BLEND_INVALID")
    if not math.isfinite(max_gain) or max_gain < 1.0 or max_gain > 8.0:
        raise ValueError("ACTION_DIMENSION_BALANCING_MAX_GAIN_INVALID")
    return blend, max_gain


def _patch_dit(original: str) -> str:
    import_marker = "from enum import Enum\n"
    import_replacement = (
        "from enum import Enum\n"
        "from acwm.wmloop_hooks.action_dimension_balancing import balance_action_dimensions\n"
    )
    forward_marker = (
        "        action = action.to(self.mlp_in[0].weight.dtype)\n"
        "        x = self.mlp_in(action)  # [B, L, dim]\n"
    )
    forward_replacement = (
        "        action = action.to(self.mlp_in[0].weight.dtype)\n"
        "        action = balance_action_dimensions(action)\n"
        "        x = self.mlp_in(action)  # [B, L, dim]\n"
    )
    if import_marker not in original:
        raise ValueError("ACTION_DIMENSION_BALANCING_IMPORT_ANCHOR_MISSING")
    if forward_marker not in original:
        raise ValueError("ACTION_DIMENSION_BALANCING_FORWARD_ANCHOR_MISSING")
    patched = original.replace(import_marker, import_replacement, 1)
    return patched.replace(forward_marker, forward_replacement, 1)


def _sidecar_payload(*, blend: float, max_gain: float) -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_type": "wmloop-materialized-primitive-smoke",
        "primitive": "action_dimension_balancing",
        "layer": "L2",
        "hook": "H2",
        "materialization_state": "acwm_runtime_hook_smoke",
        "params": {"blend": blend, "max_gain": max_gain, "epsilon": 1e-6},
        "runtime_hook": {
            "module": "acwm.wmloop_hooks.action_dimension_balancing",
            "function": "balance_action_dimensions",
            "model_patch": "acwm/model/dit/dit.py",
            "attachment": "ActionEmbedder.forward before the existing mlp_in",
        },
        "runtime_hook_paths": [
            "acwm/model/dit/dit.py",
            "acwm/wmloop_hooks/action_dimension_balancing.py",
        ],
        "intent_to_code_contract": {
            "method_intent": (
                "Reduce action-binding imbalance caused by heterogeneous per-dimension action magnitudes."
            ),
            "runtime_behavior": (
                "For each sample, computes temporal RMS per action dimension, moves it toward the sample-global RMS "
                "using a clipped gain, and blends the result with the original action before the unchanged ActionEmbedder."
            ),
            "declared_proxy": (
                "Per-dimension temporal action RMS imbalance before ActionEmbedder projection; success requires the "
                "frozen action-binding and official rollout metrics rather than the proxy alone."
            ),
            "checkpoint_compatibility": (
                "Adds no trainable parameter or persistent buffer; the official checkpoint state dict remains shape-compatible."
            ),
            "null_action_rule": "Classifier-free null-action sequences are returned unchanged.",
            "not_claimed": [
                "does not reproduce the full DexAC architecture",
                "does not add semantic branches, dimension tokens, or local/global modulation modules",
                "does not modify evaluator files or use held-out evidence as training input",
            ],
        },
        "literature_source": {
            "arxiv_id": "2606.27325v1",
            "staged_candidate": "action_dimension_balancing_v1",
        },
    }


def _unified_diff(path: str, original: str, patched: str) -> str:
    if original == patched:
        raise ValueError(f"ACTION_DIMENSION_BALANCING_PATCH_EMPTY:{path}")
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


_SIDECAR = Path("wmloop_interventions/action_dimension_balancing.json")
_CALL_COUNT = 0
_LAST_RECEIPT = {"state": "not_invoked", "call_count": 0}


def balance_action_dimensions(action: torch.Tensor) -> torch.Tensor:
    """Apply bounded per-sample action-dimension RMS balancing."""

    global _CALL_COUNT, _LAST_RECEIPT
    config = _configured()
    if action.ndim != 3 or action.shape[-1] < 1 or not action.is_floating_point():
        raise RuntimeError("ACTION_DIMENSION_BALANCING_INPUT_INVALID")
    source = action
    working = source.float()
    epsilon = config["epsilon"]
    dimension_rms = working.square().mean(dim=1, keepdim=True).add(epsilon).sqrt()
    global_rms = working.square().mean(dim=(1, 2), keepdim=True).add(epsilon).sqrt()
    raw_gain = global_rms / dimension_rms
    gain = raw_gain.clamp(min=1.0 / config["max_gain"], max=config["max_gain"])
    balanced = working * gain
    output = working + config["blend"] * (balanced - working)

    if action.shape[-1] >= 2:
        null_rows = action[..., :-1].eq(0).all(dim=-1) & action[..., -1].eq(1)
        null_sequences = null_rows.all(dim=1, keepdim=True).unsqueeze(-1)
        output = torch.where(null_sequences, working, output)

    result = output.to(dtype=source.dtype)
    _CALL_COUNT += 1
    _LAST_RECEIPT = {
        "state": "ready",
        "call_count": _CALL_COUNT,
        "shape": list(source.shape),
        "dtype": str(source.dtype),
        "observed_gain_min": float(gain.detach().amin().cpu()),
        "observed_gain_max": float(gain.detach().amax().cpu()),
        "mean_abs_delta": float((result.detach().float() - source.detach().float()).abs().mean().cpu()),
        "configured_blend": config["blend"],
        "configured_max_gain": config["max_gain"],
    }
    return result


def runtime_hook_receipt() -> dict[str, object]:
    return dict(_LAST_RECEIPT)


@lru_cache(maxsize=1)
def _configured() -> dict[str, float]:
    try:
        payload = json.loads(_SIDECAR.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("ACTION_DIMENSION_BALANCING_SIDECAR_UNREADABLE") from exc
    if not isinstance(payload, Mapping):
        raise RuntimeError("ACTION_DIMENSION_BALANCING_SIDECAR_INVALID")
    if (
        payload.get("artifact_type") != "wmloop-materialized-primitive-smoke"
        or payload.get("primitive") != "action_dimension_balancing"
        or payload.get("hook") != "H2"
        or payload.get("materialization_state") != "acwm_runtime_hook_smoke"
    ):
        raise RuntimeError("ACTION_DIMENSION_BALANCING_SIDECAR_INVALID")
    params = payload.get("params")
    if not isinstance(params, Mapping):
        raise RuntimeError("ACTION_DIMENSION_BALANCING_PARAMS_INVALID")
    blend = params.get("blend")
    max_gain = params.get("max_gain")
    epsilon = params.get("epsilon")
    values = (blend, max_gain, epsilon)
    if any(not isinstance(value, (int, float)) or isinstance(value, bool) for value in values):
        raise RuntimeError("ACTION_DIMENSION_BALANCING_PARAMS_INVALID")
    blend_value = float(blend)
    gain_value = float(max_gain)
    epsilon_value = float(epsilon)
    if (
        not math.isfinite(blend_value)
        or blend_value <= 0.0
        or blend_value > 1.0
        or not math.isfinite(gain_value)
        or gain_value < 1.0
        or gain_value > 8.0
        or not math.isfinite(epsilon_value)
        or epsilon_value <= 0.0
    ):
        raise RuntimeError("ACTION_DIMENSION_BALANCING_PARAMS_INVALID")
    return {"blend": blend_value, "max_gain": gain_value, "epsilon": epsilon_value}
'''
