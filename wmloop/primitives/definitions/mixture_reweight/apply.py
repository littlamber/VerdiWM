from __future__ import annotations

import difflib
import json
import math
from pathlib import Path
from typing import Mapping


def apply(repo_worktree: str, params: Mapping[str, object]) -> str:
    worktree = Path(repo_worktree)
    frontier_weight = _validate_frontier_weight(params)
    trainer_path = worktree / "acwm" / "trainer" / "train_dynamics.py"
    original = trainer_path.read_text(encoding="utf-8")
    patched = _patch_trainer(original)
    sidecar = _sidecar_payload(frontier_weight)
    return "\n".join(
        part.rstrip("\n")
        for part in (
            _unified_diff("acwm/trainer/train_dynamics.py", original, patched),
            _optional_new_file_diff(
                worktree,
                "acwm/wmloop_hooks/__init__.py",
                '"""Runtime hooks materialized by reviewed wm-loop primitive templates."""\n',
            ),
            _new_file_diff("acwm/wmloop_hooks/mixture_reweight.py", _HOOK_MODULE),
            _new_file_diff(
                "wmloop_interventions/mixture_reweight.json",
                json.dumps(sidecar, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            ),
        )
        if part.strip()
    ) + "\n"


def _validate_frontier_weight(params: Mapping[str, object]) -> float:
    value = params.get("frontier_weight")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError("MIXTURE_REWEIGHT_FRONTIER_WEIGHT_INVALID")
    frontier_weight = float(value)
    if not math.isfinite(frontier_weight) or frontier_weight < 0.0:
        raise ValueError("MIXTURE_REWEIGHT_FRONTIER_WEIGHT_INVALID")
    return frontier_weight


def _patch_trainer(original: str) -> str:
    import_marker = "from acwm.utils.visualization import visualize_layout\n"
    import_replacement = (
        "from acwm.utils.visualization import visualize_layout\n"
        "from acwm.wmloop_hooks.mixture_reweight import build_mixture_reweight_sampler\n"
    )
    sampler_marker = (
        "    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank) if world_size > 1 else None\n"
        "    train_loader = DataLoader(\n"
        "        train_dataset, \n"
        "        batch_size=config['training']['batch_size'], \n"
        "        sampler=train_sampler, \n"
        "        shuffle=(train_sampler is None),\n"
        "        num_workers=config['training']['num_workers'],\n"
        "        pin_memory=True\n"
        "    )\n"
    )
    sampler_replacement = (
        "    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank) if world_size > 1 else None\n"
        "    wmloop_mixture_sampler = None\n"
        "    if world_size == 1:\n"
        "        wmloop_mixture_sampler = build_mixture_reweight_sampler(train_dataset)\n"
        "        if wmloop_mixture_sampler is not None:\n"
        "            train_sampler = wmloop_mixture_sampler\n"
        "    train_loader = DataLoader(\n"
        "        train_dataset, \n"
        "        batch_size=config['training']['batch_size'], \n"
        "        sampler=train_sampler, \n"
        "        shuffle=(train_sampler is None),\n"
        "        num_workers=config['training']['num_workers'],\n"
        "        pin_memory=True\n"
        "    )\n"
    )
    epoch_marker = (
        "        if train_sampler:\n"
        "            train_sampler.set_epoch(epoch)\n"
    )
    epoch_replacement = (
        "        if train_sampler and hasattr(train_sampler, \"set_epoch\"):\n"
        "            train_sampler.set_epoch(epoch)\n"
    )
    patched = original
    for marker, replacement, code in (
        (import_marker, import_replacement, "MIXTURE_REWEIGHT_IMPORT_ANCHOR_MISSING"),
        (sampler_marker, sampler_replacement, "MIXTURE_REWEIGHT_SAMPLER_ANCHOR_MISSING"),
        (epoch_marker, epoch_replacement, "MIXTURE_REWEIGHT_SET_EPOCH_ANCHOR_MISSING"),
    ):
        if marker not in patched:
            raise ValueError(code)
        patched = patched.replace(marker, replacement, 1)
    return patched


def _sidecar_payload(frontier_weight: float) -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_type": "wmloop-materialized-primitive-smoke",
        "primitive": "mixture_reweight",
        "layer": "L1",
        "hook": "H1",
        "materialization_state": "acwm_runtime_hook_smoke",
        "params": {"frontier_weight": frontier_weight},
        "runtime_hook": {
            "module": "acwm.wmloop_hooks.mixture_reweight",
            "function": "build_mixture_reweight_sampler",
            "trainer_patch": "acwm/trainer/train_dynamics.py",
        },
        "runtime_hook_paths": [
            "acwm/trainer/train_dynamics.py",
            "acwm/wmloop_hooks/mixture_reweight.py",
        ],
        "intent_to_code_contract": {
            "method_intent": "Reweight the training data mixture toward frontier/high-motion windows.",
            "runtime_behavior": (
                "For single-process ACWM training, replace the default shuffled DataLoader path with a "
                "WeightedRandomSampler whose weights are derived from action-window frontier scores."
            ),
            "declared_proxy": (
                "Uses action magnitude from ACWM metadata as the frontier score because no external diagnostic "
                "frontier labels are available inside the frozen training split."
            ),
            "not_claimed": [
                "does not collect new trajectories",
                "does not alter held-out/evaluator data",
                "does not use verdict probes as training labels",
            ],
        },
        "notes": [
            "Adds a sampler-level data-mixture reweighting hook before ACWM training DataLoader construction.",
            "The hook is trial-worktree only and does not modify frozen evaluator files.",
        ],
    }


def _unified_diff(path: str, original: str, patched: str) -> str:
    if original == patched:
        raise ValueError("MIXTURE_REWEIGHT_PATCH_EMPTY")
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
from torch.utils.data import WeightedRandomSampler


_SIDECAR = Path("wmloop_interventions/mixture_reweight.json")


def build_mixture_reweight_sampler(dataset: object) -> WeightedRandomSampler | None:
    """Build a frontier-weighted sampler for ACWM training windows."""

    frontier_weight = _configured_frontier_weight()
    if frontier_weight == 0.0:
        return None
    weights = _window_weights(dataset, frontier_weight=frontier_weight)
    if weights.numel() == 0:
        return None
    return WeightedRandomSampler(weights.double(), num_samples=int(weights.numel()), replacement=True)


def _window_weights(dataset: object, *, frontier_weight: float) -> torch.Tensor:
    base_dataset, subset_indices = _unwrap_subset(dataset)
    indices = getattr(base_dataset, "indices", None)
    if not isinstance(indices, list):
        return torch.ones(len(dataset), dtype=torch.float64)
    metadata = getattr(base_dataset, "full_metadata", None)
    if not isinstance(metadata, list):
        return torch.ones(len(dataset), dtype=torch.float64)
    config = getattr(base_dataset, "config", None)
    seq_len = int(getattr(config, "seq_len", 1) or 1)
    sampling_rate = int(getattr(config, "sampling_rate", 1) or 1)
    action_dim = int(getattr(config, "action_dim", 1) or 1)
    scores: list[float] = []
    logical_indices = list(subset_indices) if subset_indices is not None else list(range(len(indices)))
    for logical_index in logical_indices:
        if logical_index < 0 or logical_index >= len(indices):
            scores.append(0.0)
            continue
        traj_idx, start_f = indices[logical_index]
        if not isinstance(traj_idx, int) or traj_idx < 0 or traj_idx >= len(metadata):
            scores.append(0.0)
            continue
        entry = metadata[traj_idx]
        if not isinstance(entry, Mapping):
            scores.append(0.0)
            continue
        required_span = max(1, (seq_len - 1) * sampling_rate + 1)
        action = _action_slice(entry, int(start_f), int(start_f) + required_span, action_dim=action_dim)
        if action.numel() == 0:
            scores.append(0.0)
            continue
        sampled = action[torch.arange(0, action.shape[0], max(1, sampling_rate))]
        if sampled.numel() == 0:
            scores.append(0.0)
            continue
        scores.append(float(sampled.float().norm(dim=-1).mean().item()))
    score_tensor = torch.tensor(scores, dtype=torch.float64)
    if score_tensor.numel() == 0:
        return score_tensor
    max_score = torch.max(score_tensor)
    if not torch.isfinite(max_score) or float(max_score) <= 0.0:
        normalized = torch.zeros_like(score_tensor)
    else:
        normalized = score_tensor / max_score.clamp_min(1e-12)
    weights = 1.0 + float(frontier_weight) * normalized
    return weights.clamp_min(1e-6)


def _unwrap_subset(dataset: object) -> tuple[object, list[int] | None]:
    base = getattr(dataset, "dataset", None)
    indices = getattr(dataset, "indices", None)
    if base is not None and indices is not None:
        return base, [int(index) for index in indices]
    return dataset, None


def _action_slice(entry: Mapping[str, object], start: int, end: int, *, action_dim: int) -> torch.Tensor:
    if "actions" in entry:
        return _as_action_tensor(entry["actions"], action_dim=action_dim)[start:end]
    commands = entry.get("commands")
    if isinstance(commands, Mapping):
        lin = commands.get("linear_velocity")
        ang = commands.get("angular_velocity")
        if lin is not None and ang is not None:
            return torch.stack(
                [
                    _as_action_tensor(lin, action_dim=1)[start:end].reshape(-1),
                    _as_action_tensor(ang, action_dim=1)[start:end].reshape(-1),
                ],
                dim=-1,
            )
    if commands is not None:
        return _as_action_tensor(commands, action_dim=action_dim)[start:end]
    return torch.zeros((max(0, end - start), action_dim), dtype=torch.float32)


def _as_action_tensor(value: object, *, action_dim: int) -> torch.Tensor:
    try:
        tensor = torch.as_tensor(value)
    except (TypeError, ValueError):
        return torch.zeros((0, action_dim), dtype=torch.float32)
    if tensor.ndim == 0:
        return tensor.reshape(1, 1).float()
    if tensor.ndim == 1:
        return tensor.reshape(-1, 1).float()
    return tensor.float()


@lru_cache(maxsize=1)
def _configured_frontier_weight() -> float:
    payload = _load_sidecar()
    params = payload.get("params")
    if not isinstance(params, Mapping):
        raise RuntimeError("MIXTURE_REWEIGHT_PARAMS_INVALID")
    value = params.get("frontier_weight")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise RuntimeError("MIXTURE_REWEIGHT_FRONTIER_WEIGHT_INVALID")
    frontier_weight = float(value)
    if not math.isfinite(frontier_weight) or frontier_weight < 0.0:
        raise RuntimeError("MIXTURE_REWEIGHT_FRONTIER_WEIGHT_INVALID")
    return frontier_weight


def _load_sidecar() -> Mapping[str, object]:
    try:
        payload = json.loads(_SIDECAR.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("MIXTURE_REWEIGHT_SIDECAR_UNREADABLE") from exc
    if not isinstance(payload, Mapping):
        raise RuntimeError("MIXTURE_REWEIGHT_SIDECAR_INVALID")
    if (
        payload.get("artifact_type") != "wmloop-materialized-primitive-smoke"
        or payload.get("primitive") != "mixture_reweight"
        or payload.get("hook") != "H1"
        or payload.get("materialization_state") != "acwm_runtime_hook_smoke"
    ):
        raise RuntimeError("MIXTURE_REWEIGHT_SIDECAR_INVALID")
    return payload
'''
