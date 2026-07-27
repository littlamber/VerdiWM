from __future__ import annotations

import difflib
import json
import math
from pathlib import Path
from typing import Mapping


def apply(repo_worktree: str, params: Mapping[str, object]) -> str:
    worktree = Path(repo_worktree)
    event_weight, event_quantile, visual_motion_blend = _validate_params(params)
    trainer_path = worktree / "acwm" / "trainer" / "train_dynamics.py"
    original = trainer_path.read_text(encoding="utf-8")
    patched = _patch_trainer(original)
    sidecar = _sidecar_payload(
        event_weight=event_weight,
        event_quantile=event_quantile,
        visual_motion_blend=visual_motion_blend,
    )
    return "\n".join(
        part.rstrip("\n")
        for part in (
            _unified_diff("acwm/trainer/train_dynamics.py", original, patched),
            _optional_new_file_diff(
                worktree,
                "acwm/wmloop_hooks/__init__.py",
                '"""Runtime hooks materialized by reviewed wm-loop primitive templates."""\n',
            ),
            _new_file_diff("acwm/wmloop_hooks/event_window_reweight.py", _HOOK_MODULE),
            _new_file_diff(
                "wmloop_interventions/event_window_reweight.json",
                json.dumps(sidecar, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            ),
        )
        if part.strip()
    ) + "\n"


def _validate_params(params: Mapping[str, object]) -> tuple[float, float, float]:
    values = tuple(params.get(name) for name in ("event_weight", "event_quantile", "visual_motion_blend"))
    if any(not isinstance(value, (int, float)) or isinstance(value, bool) for value in values):
        raise ValueError("EVENT_WINDOW_REWEIGHT_PARAMS_INVALID")
    event_weight, event_quantile, visual_motion_blend = (float(value) for value in values)
    if (
        not all(math.isfinite(value) for value in (event_weight, event_quantile, visual_motion_blend))
        or not 0.0 < event_weight <= 16.0
        or not 0.5 <= event_quantile <= 0.95
        or not 0.0 <= visual_motion_blend <= 1.0
    ):
        raise ValueError("EVENT_WINDOW_REWEIGHT_PARAMS_INVALID")
    return event_weight, event_quantile, visual_motion_blend


def _patch_trainer(original: str) -> str:
    import_marker = "from acwm.utils.visualization import visualize_layout\n"
    import_replacement = (
        "from acwm.utils.visualization import visualize_layout\n"
        "from acwm.wmloop_hooks.event_window_reweight import build_event_window_sampler\n"
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
        "    if world_size == 1:\n"
        "        wmloop_event_sampler = build_event_window_sampler(train_dataset)\n"
        "        if wmloop_event_sampler is not None:\n"
        "            train_sampler = wmloop_event_sampler\n"
        "    train_loader = DataLoader(\n"
        "        train_dataset, \n"
        "        batch_size=config['training']['batch_size'], \n"
        "        sampler=train_sampler, \n"
        "        shuffle=(train_sampler is None),\n"
        "        num_workers=config['training']['num_workers'],\n"
        "        pin_memory=True\n"
        "    )\n"
    )
    epoch_marker = "        if train_sampler:\n            train_sampler.set_epoch(epoch)\n"
    epoch_replacement = (
        "        if train_sampler and hasattr(train_sampler, \"set_epoch\"):\n"
        "            train_sampler.set_epoch(epoch)\n"
    )
    patched = original
    for marker, replacement, code in (
        (import_marker, import_replacement, "EVENT_WINDOW_REWEIGHT_IMPORT_ANCHOR_MISSING"),
        (sampler_marker, sampler_replacement, "EVENT_WINDOW_REWEIGHT_SAMPLER_ANCHOR_MISSING"),
        (epoch_marker, epoch_replacement, "EVENT_WINDOW_REWEIGHT_SET_EPOCH_ANCHOR_MISSING"),
    ):
        if marker not in patched:
            raise ValueError(code)
        patched = patched.replace(marker, replacement, 1)
    return patched


def _sidecar_payload(*, event_weight: float, event_quantile: float, visual_motion_blend: float) -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_type": "wmloop-materialized-primitive-smoke",
        "primitive": "event_window_reweight",
        "layer": "L1",
        "hook": "H1",
        "materialization_state": "acwm_runtime_hook_smoke",
        "params": {
            "event_weight": event_weight,
            "event_quantile": event_quantile,
            "visual_motion_blend": visual_motion_blend,
        },
        "runtime_hook": {
            "module": "acwm.wmloop_hooks.event_window_reweight",
            "function": "build_event_window_sampler",
            "trainer_patch": "acwm/trainer/train_dynamics.py",
            "runtime_receipt": "runtime_hook_receipt",
        },
        "runtime_hook_paths": [
            "acwm/trainer/train_dynamics.py",
            "acwm/wmloop_hooks/event_window_reweight.py",
        ],
        "intent_to_code_contract": {
            "method_intent": "Increase exposure to sparse physical-transition windows that are underrepresented by uniform sliding-window sampling.",
            "runtime_behavior": "Score every training window from training-split action transitions and optional unlabeled visual motion, then install a WeightedRandomSampler that boosts only the configured upper event-score quantile.",
            "declared_proxy": "Action finite differences and frame-to-frame RGB change are label-free proxies for sparse physical-event activity; the runtime receipt records whether both signals were observed and how many windows were boosted.",
            "not_claimed": [
                "does not inspect held-out videos, official metrics, or verdict labels",
                "does not hard-code pour-water frame numbers or environment identities",
                "does not establish quality improvement until the frozen official and event gates pass",
            ],
        },
        "notes": [
            "The sampler is constructed only for single-process training; distributed training remains unchanged.",
            "Visual scoring fails over to action-only scoring when videos or OpenCV are unavailable, and this source is exposed in the runtime receipt.",
        ],
    }


def _unified_diff(path: str, original: str, patched: str) -> str:
    if original == patched:
        raise ValueError("EVENT_WINDOW_REWEIGHT_PATCH_EMPTY")
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


_SIDECAR = Path("wmloop_interventions/event_window_reweight.json")
_RECEIPT = {
    "state": "not_called",
    "call_count": 0,
    "window_count": 0,
    "selected_window_count": 0,
    "score_source": "none",
    "max_weight": 0.0,
    "mean_weight": 0.0,
}


def build_event_window_sampler(dataset: object) -> WeightedRandomSampler:
    """Install event-focused sampling using training-split signals only."""

    params = _configured_params()
    weights, source = _window_weights(
        dataset,
        event_weight=params["event_weight"],
        event_quantile=params["event_quantile"],
        visual_motion_blend=params["visual_motion_blend"],
    )
    if weights.numel() == 0 or not torch.isfinite(weights).all():
        raise RuntimeError("EVENT_WINDOW_REWEIGHT_WEIGHTS_INVALID")
    selected = int((weights > 1.0 + 1e-12).sum().item())
    if selected < 1:
        raise RuntimeError("EVENT_WINDOW_REWEIGHT_SIGNAL_UNAVAILABLE")
    _RECEIPT.update(
        {
            "state": "ready",
            "call_count": int(_RECEIPT["call_count"]) + 1,
            "window_count": int(weights.numel()),
            "selected_window_count": selected,
            "score_source": source,
            "max_weight": float(weights.max().item()),
            "mean_weight": float(weights.mean().item()),
            "event_weight": params["event_weight"],
            "event_quantile": params["event_quantile"],
            "visual_motion_blend": params["visual_motion_blend"],
        }
    )
    print("WMLOOP_EVENT_WINDOW_REWEIGHT_RECEIPT=" + json.dumps(_RECEIPT, sort_keys=True))
    return WeightedRandomSampler(weights.double(), num_samples=int(weights.numel()), replacement=True)


def runtime_hook_receipt() -> dict[str, object]:
    return dict(_RECEIPT)


def _window_weights(
    dataset: object,
    *,
    event_weight: float,
    event_quantile: float,
    visual_motion_blend: float,
) -> tuple[torch.Tensor, str]:
    base_dataset, subset_indices = _unwrap_subset(dataset)
    indices = getattr(base_dataset, "indices", None)
    metadata = getattr(base_dataset, "full_metadata", None)
    if not isinstance(indices, list) or not isinstance(metadata, list):
        raise RuntimeError("EVENT_WINDOW_REWEIGHT_DATASET_CONTRACT_INVALID")
    config = getattr(base_dataset, "config", None)
    seq_len = int(getattr(config, "seq_len", 1) or 1)
    sampling_rate = int(getattr(config, "sampling_rate", 1) or 1)
    action_dim = int(getattr(config, "action_dim", 1) or 1)
    required_span = max(2, (seq_len - 1) * sampling_rate + 1)
    logical_indices = list(subset_indices) if subset_indices is not None else list(range(len(indices)))
    action_scores: list[float] = []
    visual_scores: list[float] = []
    motion_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for logical_index in logical_indices:
        if logical_index < 0 or logical_index >= len(indices):
            raise RuntimeError("EVENT_WINDOW_REWEIGHT_INDEX_INVALID")
        traj_idx, start_f = indices[logical_index]
        if not isinstance(traj_idx, int) or traj_idx < 0 or traj_idx >= len(metadata):
            raise RuntimeError("EVENT_WINDOW_REWEIGHT_TRAJECTORY_INVALID")
        entry = metadata[traj_idx]
        if not isinstance(entry, Mapping):
            raise RuntimeError("EVENT_WINDOW_REWEIGHT_METADATA_INVALID")
        action = _action_slice(entry, int(start_f), int(start_f) + required_span, action_dim=action_dim)
        sampled = action[::max(1, sampling_rate)]
        if sampled.shape[0] >= 2:
            transition = sampled[1:] - sampled[:-1]
            action_score = float(transition.float().norm(dim=-1).mean().item())
        else:
            action_score = 0.0
        action_scores.append(action_score)
        if traj_idx not in motion_cache:
            motion_cache[traj_idx] = _video_motion_curve(base_dataset, entry)
        frame_numbers, curve = motion_cache[traj_idx]
        if curve.numel() == 0:
            visual_scores.append(0.0)
        else:
            in_window = (frame_numbers >= int(start_f)) & (frame_numbers < int(start_f) + required_span)
            values = curve[in_window]
            visual_scores.append(float(values.mean().item()) if values.numel() else 0.0)
    action_tensor = _normalize_scores(torch.tensor(action_scores, dtype=torch.float64))
    visual_tensor = _normalize_scores(torch.tensor(visual_scores, dtype=torch.float64))
    has_action = bool(action_tensor.numel() and float(action_tensor.max()) > 0.0)
    has_visual = bool(visual_tensor.numel() and float(visual_tensor.max()) > 0.0)
    if has_action and has_visual:
        combined = (1.0 - visual_motion_blend) * action_tensor + visual_motion_blend * visual_tensor
        source = "action_and_visual_motion"
    elif has_visual:
        combined = visual_tensor
        source = "visual_motion_only"
    elif has_action:
        combined = action_tensor
        source = "action_transition_only"
    else:
        raise RuntimeError("EVENT_WINDOW_REWEIGHT_SIGNAL_UNAVAILABLE")
    threshold = torch.quantile(combined, event_quantile)
    maximum = combined.max()
    denominator = maximum - threshold
    if not torch.isfinite(denominator) or float(denominator) <= 1e-12:
        eventness = (combined >= maximum).to(torch.float64)
    else:
        eventness = ((combined - threshold) / denominator).clamp(0.0, 1.0)
    return (1.0 + event_weight * eventness).clamp_min(1e-6), source


def _video_motion_curve(dataset: object, entry: Mapping[str, object]) -> tuple[torch.Tensor, torch.Tensor]:
    root = getattr(dataset, "effective_root", None)
    relative = entry.get("video_path")
    if not isinstance(root, str) or not isinstance(relative, str) or not relative:
        return torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.float64)
    try:
        import cv2
    except ImportError:
        return torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.float64)
    capture = cv2.VideoCapture(str(Path(root) / relative))
    frame_numbers: list[int] = []
    values: list[float] = []
    previous = None
    frame_index = 0
    stride = 4
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % stride == 0:
                gray = cv2.cvtColor(cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
                current = torch.from_numpy(gray).to(torch.float32) / 255.0
                if previous is not None:
                    frame_numbers.append(frame_index)
                    values.append(float((current - previous).abs().mean().item()))
                previous = current
            frame_index += 1
    finally:
        capture.release()
    return torch.tensor(frame_numbers, dtype=torch.int64), torch.tensor(values, dtype=torch.float64)


def _normalize_scores(values: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        return values
    finite = torch.where(torch.isfinite(values), values, torch.zeros_like(values))
    low = torch.quantile(finite, 0.05)
    high = torch.quantile(finite, 0.95)
    if float(high - low) <= 1e-12:
        maximum = finite.max()
        return finite / maximum if float(maximum) > 0.0 else torch.zeros_like(finite)
    return ((finite - low) / (high - low)).clamp(0.0, 1.0)


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
        linear = commands.get("linear_velocity")
        angular = commands.get("angular_velocity")
        if linear is not None and angular is not None:
            return torch.stack(
                [
                    _as_action_tensor(linear, action_dim=1)[start:end].reshape(-1),
                    _as_action_tensor(angular, action_dim=1)[start:end].reshape(-1),
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
def _configured_params() -> dict[str, float]:
    try:
        payload = json.loads(_SIDECAR.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("EVENT_WINDOW_REWEIGHT_SIDECAR_UNREADABLE") from exc
    if (
        not isinstance(payload, Mapping)
        or payload.get("artifact_type") != "wmloop-materialized-primitive-smoke"
        or payload.get("primitive") != "event_window_reweight"
        or payload.get("hook") != "H1"
        or payload.get("materialization_state") != "acwm_runtime_hook_smoke"
    ):
        raise RuntimeError("EVENT_WINDOW_REWEIGHT_SIDECAR_INVALID")
    params = payload.get("params")
    if not isinstance(params, Mapping):
        raise RuntimeError("EVENT_WINDOW_REWEIGHT_PARAMS_INVALID")
    values = {name: params.get(name) for name in ("event_weight", "event_quantile", "visual_motion_blend")}
    if any(not isinstance(value, (int, float)) or isinstance(value, bool) for value in values.values()):
        raise RuntimeError("EVENT_WINDOW_REWEIGHT_PARAMS_INVALID")
    output = {name: float(value) for name, value in values.items()}
    if (
        not all(math.isfinite(value) for value in output.values())
        or not 0.0 < output["event_weight"] <= 16.0
        or not 0.5 <= output["event_quantile"] <= 0.95
        or not 0.0 <= output["visual_motion_blend"] <= 1.0
    ):
        raise RuntimeError("EVENT_WINDOW_REWEIGHT_PARAMS_INVALID")
    return output
'''
