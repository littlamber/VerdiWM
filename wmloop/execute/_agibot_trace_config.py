"""Trace configuration and native bridge command bindings."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path

if __package__:
    from ._agibot_trace_contract import (
        AgibotTraceError,
        SCHEMA_VERSION,
        TRACE_CONFIG_TYPE,
        _is_sha256,
        _read_json,
        _write_json,
        canonical_digest,
    )
    from ._agibot_trace_validation import _validate_config
else:
    from _agibot_trace_contract import (
        AgibotTraceError,
        SCHEMA_VERSION,
        TRACE_CONFIG_TYPE,
        _is_sha256,
        _read_json,
        _write_json,
        canonical_digest,
    )
    from _agibot_trace_validation import _validate_config

def write_trace_config(
    output_root: Path,
    *,
    trace_id: str,
    support: Mapping[str, object],
    native_config_sha256: str,
    world_size: int,
    batch_size: int,
    gradient_accumulation: int,
    expected_optimizer_updates: int,
    seed: int,
    support_bindings: Mapping[str, object],
    init_from_path: str | None = None,
) -> Path:
    """Persist the small, immutable runtime binding consumed by the bridge."""

    if not trace_id or not isinstance(trace_id, str):
        raise AgibotTraceError("AGIBOT_TRACE_ID_INVALID")
    if not _is_sha256(native_config_sha256):
        raise AgibotTraceError("AGIBOT_TRACE_NATIVE_CONFIG_DIGEST_INVALID")
    if type(world_size) is not int or world_size < 1:
        raise AgibotTraceError("AGIBOT_TRACE_WORLD_SIZE_INVALID")
    if type(batch_size) is not int or batch_size < 1:
        raise AgibotTraceError("AGIBOT_TRACE_BATCH_SIZE_INVALID")
    if type(gradient_accumulation) is not int or gradient_accumulation < 1:
        raise AgibotTraceError("AGIBOT_TRACE_GRADIENT_ACCUMULATION_INVALID")
    if type(expected_optimizer_updates) is not int or expected_optimizer_updates < 1:
        raise AgibotTraceError("AGIBOT_TRACE_EXPECTED_UPDATES_INVALID")
    if type(seed) is not int or seed < 0:
        raise AgibotTraceError("AGIBOT_TRACE_SEED_INVALID")
    if not isinstance(support_bindings, Mapping):
        raise AgibotTraceError("AGIBOT_TRACE_SUPPORT_BINDINGS_INVALID")
    if init_from_path is not None and (
        not isinstance(init_from_path, str)
        or not init_from_path
        or not Path(init_from_path).is_absolute()
        or not Path(init_from_path).is_dir()
    ):
        raise AgibotTraceError("AGIBOT_TRACE_INIT_FROM_PATH_INVALID")
    validation = support.get("validation_support")
    if not isinstance(validation, Mapping):
        raise AgibotTraceError("AGIBOT_TRACE_SUPPORT_INVALID")
    ids = validation.get("sample_window_ids")
    if (
        not isinstance(ids, list)
        or not ids
        or any(not isinstance(item, str) or not item for item in ids)
        or len(ids) != len(set(ids))
    ):
        raise AgibotTraceError("AGIBOT_TRACE_VALIDATION_IDS_INVALID")
    digest = canonical_digest(support)
    config = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": TRACE_CONFIG_TYPE,
        "trace_id": trace_id,
        "support_digest": digest,
        "source_fingerprint": canonical_digest(_source_fingerprint_payload(
            digest, native_config_sha256, init_from_path
        )),
        "native_config_sha256": native_config_sha256,
        "output_root": str(output_root.resolve()),
        "world_size": world_size,
        "batch_size": batch_size,
        "gradient_accumulation": gradient_accumulation,
        "expected_optimizer_updates": expected_optimizer_updates,
        "seed": seed,
        "support_bindings": dict(support_bindings),
        "validation_sample_window_ids": list(ids),
        "validation_padding_rule": "complete_frozen_panel_per_rank_v1",
    }
    if init_from_path is not None:
        config["init_from_path"] = str(Path(init_from_path).resolve())
    runtime_root = output_root / "runtime"
    runtime_root.mkdir(parents=True, exist_ok=True)
    path = runtime_root / "agibot-trace-config.json"
    _write_json(path, config)
    return path


def build_bridge_command(
    runtime_python: str,
    native_command: Sequence[str],
    *,
    trace_config_path: Path,
    bridge_path: Path,
    world_size: int,
) -> list[str]:
    """Launch one bridge process per native training rank.

    ``native_command`` is the argument tail after the runtime interpreter, as
    stored in a model-run manifest. The fixed, reviewed trainer entrypoint is
    run by the bridge in each torchrun rank. Other entrypoints fail closed.
    """

    if (
        not runtime_python
        or not isinstance(native_command, Sequence)
        or isinstance(native_command, (str, bytes))
        or not native_command
        or type(world_size) is not int
        or world_size < 1
    ):
        raise AgibotTraceError("AGIBOT_TRACE_NATIVE_COMMAND_INVALID")
    command = [str(token) for token in native_command]
    if any(not token for token in command):
        raise AgibotTraceError("AGIBOT_TRACE_NATIVE_COMMAND_INVALID")
    if not Path(command[0]).as_posix().endswith("scripts/training/train_world_model.py"):
        raise AgibotTraceError("AGIBOT_TRACE_NATIVE_ENTRYPOINT_UNSUPPORTED")
    bridge = bridge_path.resolve()
    config = trace_config_path.resolve()
    return [
        runtime_python,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
        str(world_size),
        str(bridge),
        "--trace-config",
        str(config),
        "--",
        *command,
    ]


def validate_native_training_command(
    native_command: Sequence[str], config: Mapping[str, object]
) -> tuple[Path, Path]:
    """Bind the explicit native YAML argument and enforce WMTrainer selection.

    This project run uses the Wan2.2 5B ``WMTrainer`` path, so the bridge
    requires ``wan2.2`` and its explicit checkpoint path before importing or
    executing the trainer.
    """

    if not native_command:
        raise AgibotTraceError("AGIBOT_TRACE_NATIVE_COMMAND_INVALID")
    command = [str(token) for token in native_command]
    entrypoint = Path(command[0]).expanduser().resolve()
    if not entrypoint.as_posix().endswith("scripts/training/train_world_model.py"):
        raise AgibotTraceError("AGIBOT_TRACE_NATIVE_ENTRYPOINT_UNSUPPORTED")
    if not entrypoint.is_file():
        raise AgibotTraceError("AGIBOT_TRACE_NATIVE_ENTRYPOINT_MISSING")
    expected_init_from = config.get("init_from_path")
    if expected_init_from is None:
        if len(command) != 3 or command[1] != "--config" or not command[2]:
            raise AgibotTraceError("AGIBOT_TRACE_NATIVE_CONFIG_ARGUMENT_INVALID")
    elif (
        len(command) != 5
        or command[1] != "--config"
        or not command[2]
        or command[3] != "--init_from"
        or not command[4]
        or str(Path(command[4]).expanduser().resolve()) != expected_init_from
        or not Path(command[4]).is_dir()
        or Path(command[4]).is_symlink()
    ):
        raise AgibotTraceError("AGIBOT_TRACE_NATIVE_CONFIG_ARGUMENT_INVALID")
    config_value = command[2]
    native_config_path = Path(config_value).expanduser().resolve()
    output_root = Path(str(config.get("output_root", ""))).expanduser().resolve()
    if native_config_path != output_root / "config" / "agibot-training.yaml":
        raise AgibotTraceError("AGIBOT_TRACE_NATIVE_CONFIG_PATH_INVALID")
    try:
        payload = native_config_path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        import yaml

        native_config = yaml.safe_load(payload.decode("utf-8"))
    except Exception as exc:
        raise AgibotTraceError("AGIBOT_TRACE_NATIVE_CONFIG_INVALID") from exc
    if digest != config.get("native_config_sha256"):
        raise AgibotTraceError("AGIBOT_TRACE_NATIVE_CONFIG_DIGEST_MISMATCH")
    world_model = native_config.get("world_model") if isinstance(native_config, Mapping) else None
    checkpoint_value = world_model.get("checkpoint") if isinstance(world_model, Mapping) else None
    if (
        not isinstance(world_model, Mapping)
        or world_model.get("backbone") != "wan2.2"
        or not isinstance(checkpoint_value, str)
        or not checkpoint_value.strip()
        or not Path(checkpoint_value).expanduser().exists()
    ):
        raise AgibotTraceError("AGIBOT_TRACE_WAN22_WMTRAINER_CONFIG_REQUIRED")
    return entrypoint, native_config_path


def _source_fingerprint_payload(
    support_digest: str, native_config_sha256: str, init_from_path: str | None
) -> dict[str, str]:
    payload = {
        "support_digest": support_digest,
        "native_config_sha256": native_config_sha256,
    }
    if init_from_path is not None:
        payload["init_from_path"] = str(Path(init_from_path).resolve())
    return payload


def load_trace_config(config_path: Path) -> dict[str, object]:
    """Load and validate the immutable config shared by every rank."""

    config = _read_json(config_path)
    _validate_config(config)
    return config


def trace_shard_path(config: Mapping[str, object], rank: int) -> Path:
    """Derive the protocol-owned shard path from the bound native config output."""

    output_root = Path(str(config["output_root"])).resolve()
    return output_root / "receipts" / "training-trace" / f"rank-{rank:05d}.jsonl"
