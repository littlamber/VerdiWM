from __future__ import annotations

import difflib
import json
import math
from pathlib import Path
from typing import Mapping


def apply(repo_worktree: str, params: Mapping[str, object]) -> str:
    worktree = Path(repo_worktree)
    condition, n_episodes = _validate_params(params)
    trainer_path = worktree / "acwm" / "trainer" / "train_dynamics.py"
    original = trainer_path.read_text(encoding="utf-8")
    patched = _patch_trainer(original)
    sidecar = _sidecar_payload(condition=condition, n_episodes=n_episodes)
    return "\n".join(
        part.rstrip("\n")
        for part in (
            _unified_diff("acwm/trainer/train_dynamics.py", original, patched),
            _optional_new_file_diff(
                worktree,
                "acwm/wmloop_hooks/__init__.py",
                '"""Runtime hooks materialized by reviewed wm-loop primitive templates."""\n',
            ),
            _new_file_diff("acwm/wmloop_hooks/frontier_collection.py", _HOOK_MODULE),
            _new_file_diff(
                "wmloop_interventions/frontier_collection.json",
                json.dumps(sidecar, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            ),
        )
        if part.strip()
    ) + "\n"


def _validate_params(params: Mapping[str, object]) -> tuple[str, int]:
    condition = params.get("condition")
    if not isinstance(condition, str) or not condition.strip():
        raise ValueError("FRONTIER_COLLECTION_CONDITION_INVALID")
    n_episodes = params.get("n_episodes")
    if not isinstance(n_episodes, int) or isinstance(n_episodes, bool) or n_episodes < 1:
        raise ValueError("FRONTIER_COLLECTION_N_EPISODES_INVALID")
    return condition.strip(), int(n_episodes)


def _patch_trainer(original: str) -> str:
    import_marker = "from acwm.utils.visualization import visualize_layout\n"
    import_replacement = (
        "from acwm.utils.visualization import visualize_layout\n"
        "from acwm.wmloop_hooks.frontier_collection import record_frontier_observation\n"
    )
    backward_marker = "            loss.backward()\n"
    backward_replacement = (
        "            loss.backward()\n"
        "            record_frontier_observation(\n"
        "                loss,\n"
        "                action,\n"
        "                step=step,\n"
        "                epoch=epoch,\n"
        "                traj_ids=batch.get('traj_idx'),\n"
        "                starts=batch.get('start_f'),\n"
        "            )\n"
    )
    patched = original
    for marker, replacement, code in (
        (import_marker, import_replacement, "FRONTIER_COLLECTION_IMPORT_ANCHOR_MISSING"),
        (backward_marker, backward_replacement, "FRONTIER_COLLECTION_BACKWARD_ANCHOR_MISSING"),
    ):
        if marker not in patched:
            raise ValueError(code)
        patched = patched.replace(marker, replacement, 1)
    return patched


def _sidecar_payload(*, condition: str, n_episodes: int) -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_type": "wmloop-materialized-primitive-smoke",
        "primitive": "frontier_collection",
        "layer": "L1",
        "hook": "H1",
        "materialization_state": "acwm_runtime_hook_smoke",
        "params": {"condition": condition, "n_episodes": n_episodes},
        "runtime_hook": {
            "module": "acwm.wmloop_hooks.frontier_collection",
            "function": "record_frontier_observation",
            "trainer_patch": "acwm/trainer/train_dynamics.py",
        },
        "runtime_hook_paths": [
            "acwm/trainer/train_dynamics.py",
            "acwm/wmloop_hooks/frontier_collection.py",
        ],
        "intent_to_code_contract": {
            "method_intent": (
                "Collect bounded failure-frontier training-window evidence for later data-extension or "
                "primitive-routing decisions."
            ),
            "runtime_behavior": (
                "Records lightweight per-step frontier observations from ACWM train_dynamics after backward, "
                "including loss, action magnitude, trajectory ids, and start indices, capped by n_episodes."
            ),
            "declared_proxy": (
                "Uses high training loss and action-window magnitude as an executable frontier proxy; it is "
                "routing evidence, not a verdict metric."
            ),
            "not_claimed": [
                "does not collect new simulator trajectories by itself",
                "does not alter loss, sampler, checkpoints, evaluator, or held-out splits",
                "does not prove model improvement until a separate closed-loop trial passes",
            ],
        },
        "notes": [
            "Adds a reviewed H1 frontier-observation hook to the ACWM training loop.",
            "The hook writes bounded JSONL routing evidence under wmloop_interventions/frontier_collection_records.jsonl.",
        ],
    }


def _unified_diff(path: str, original: str, patched: str) -> str:
    if original == patched:
        raise ValueError("FRONTIER_COLLECTION_PATCH_EMPTY")
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
import os
from functools import lru_cache
from pathlib import Path
from typing import Mapping

import torch


_SIDECAR = Path("wmloop_interventions/frontier_collection.json")
_RECORDS_PATH = Path("wmloop_interventions/frontier_collection_records.jsonl")
_RECORDED_COUNT = 0


def record_frontier_observation(
    loss: torch.Tensor,
    actions: torch.Tensor,
    *,
    step: int,
    epoch: int,
    traj_ids: object = None,
    starts: object = None,
) -> dict[str, float]:
    """Record bounded frontier evidence without changing ACWM optimization."""

    config = _configured_request()
    global _RECORDED_COUNT
    if _RECORDED_COUNT >= config["n_episodes"]:
        return {"train/wmloop_frontier_collection_recorded": float(_RECORDED_COUNT)}
    if actions.ndim < 2:
        raise RuntimeError("FRONTIER_COLLECTION_ACTION_SHAPE_INVALID")
    loss_value = float(loss.detach().float().mean().cpu())
    action_magnitude = float(actions.detach().float().square().mean().sqrt().cpu())
    if not math.isfinite(loss_value) or not math.isfinite(action_magnitude):
        raise RuntimeError("FRONTIER_COLLECTION_SCORE_INVALID")
    record = {
        "schema_version": 1,
        "artifact_type": "wmloop-frontier-observation",
        "primitive": "frontier_collection",
        "role": "diagnostic_routing_evidence",
        "condition": config["condition"],
        "rank": int(os.environ.get("RANK", "0") or 0),
        "step": int(step),
        "epoch": int(epoch),
        "loss": loss_value,
        "action_magnitude": action_magnitude,
        "frontier_score": loss_value * (1.0 + action_magnitude),
        "traj_ids": _to_jsonable(traj_ids),
        "starts": _to_jsonable(starts),
    }
    _RECORDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _RECORDS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\\n")
    _RECORDED_COUNT += 1
    return {
        "train/wmloop_frontier_collection_recorded": float(_RECORDED_COUNT),
        "train/wmloop_frontier_collection_score": float(record["frontier_score"]),
    }


@lru_cache(maxsize=1)
def _configured_request() -> dict[str, object]:
    payload = _load_sidecar()
    params = payload.get("params")
    if not isinstance(params, Mapping):
        raise RuntimeError("FRONTIER_COLLECTION_PARAMS_INVALID")
    condition = params.get("condition")
    if not isinstance(condition, str) or not condition.strip():
        raise RuntimeError("FRONTIER_COLLECTION_CONDITION_INVALID")
    n_episodes = params.get("n_episodes")
    if not isinstance(n_episodes, int) or isinstance(n_episodes, bool) or n_episodes < 1:
        raise RuntimeError("FRONTIER_COLLECTION_N_EPISODES_INVALID")
    return {"condition": condition.strip(), "n_episodes": int(n_episodes)}


def _load_sidecar() -> Mapping[str, object]:
    try:
        payload = json.loads(_SIDECAR.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("FRONTIER_COLLECTION_SIDECAR_UNREADABLE") from exc
    if not isinstance(payload, Mapping):
        raise RuntimeError("FRONTIER_COLLECTION_SIDECAR_INVALID")
    if (
        payload.get("artifact_type") != "wmloop-materialized-primitive-smoke"
        or payload.get("primitive") != "frontier_collection"
        or payload.get("hook") != "H1"
        or payload.get("materialization_state") != "acwm_runtime_hook_smoke"
    ):
        raise RuntimeError("FRONTIER_COLLECTION_SIDECAR_INVALID")
    return payload


def _to_jsonable(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return repr(value)
'''
