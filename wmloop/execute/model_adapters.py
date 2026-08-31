"""Narrow model-family launch adapters used by the generic model runner."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path


class ModelLaunchAdapterError(RuntimeError):
    """A profile-backed external runtime cannot satisfy its declared protocol."""


def preflight_model_adapter(manifest: Mapping[str, object]) -> None:
    """Verify model-family capabilities before a GPU lease is acquired."""

    adapter = manifest.get("adapter")
    if not isinstance(adapter, Mapping):
        raise ModelLaunchAdapterError("MODEL_ADAPTER_PROFILE_INVALID")
    runner = manifest.get("runner")
    adapter_id = runner.get("adapter_id") if isinstance(runner, Mapping) else None
    if adapter_id is None:
        return
    if adapter_id != "ctrl_world_train_wm_v1":
        # Unknown adapter ids still use the manifest's fully validated command
        # protocol. Family-specific preflight is opt-in, not a second registry.
        return
    source = manifest.get("source")
    if not isinstance(source, Mapping):
        raise ModelLaunchAdapterError("MODEL_ADAPTER_SOURCE_INVALID")
    root = Path(str(source.get("model_root") or "")).resolve()
    if not (root / "scripts" / "train_wm.py").is_file():
        raise ModelLaunchAdapterError("CTRL_WORLD_TRAIN_ENTRYPOINT_MISSING")
    commands = manifest.get("runtime")
    evaluation = manifest.get("evaluation")
    train_command = commands.get("train_command") if isinstance(commands, Mapping) else None
    evaluate_command = evaluation.get("command") if isinstance(evaluation, Mapping) else None
    if not isinstance(train_command, list) or not train_command or not Path(str(train_command[0])).is_file():
        raise ModelLaunchAdapterError("CTRL_WORLD_MODEL_RUN_TRAIN_WRAPPER_MISSING")
    if not isinstance(evaluate_command, list) or not evaluate_command or not Path(str(evaluate_command[0])).is_file():
        raise ModelLaunchAdapterError("CTRL_WORLD_MODEL_RUN_EVAL_WRAPPER_MISSING")


def runner_adapter_id(manifest: Mapping[str, object]) -> str | None:
    runner = manifest.get("runner")
    value = runner.get("adapter_id") if isinstance(runner, Mapping) else None
    return str(value) if isinstance(value, str) and value else None
