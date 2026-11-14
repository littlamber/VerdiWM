"""Agibot native support, training trace, and bridge bindings for protocol v3."""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

from wmloop.contracts import ContractValidationError, validate_document
from wmloop.execute.model_runner import ModelRunnerError
from wmloop.storage import checked_path, sha256_file

def _write_agibot_trace_config(
    manifest: Mapping[str, object], output: Path
) -> Path:
    try:
        from wmloop.execute.agibot_trace_runtime import (
            AgibotTraceError,
            load_trace_config,
            write_trace_config,
        )

        training = manifest["training"]
        plan = training["scale_plan"]
        parallelism = plan["parallelism"]
        native_config = training["agibot_native_config"]
        path = write_trace_config(
            output,
            trace_id=str(manifest["model_run_id"]),
            support=training["agibot_training_support"],
            native_config_sha256=str(native_config["sha256"]),
            world_size=int(parallelism["world_size"]),
            batch_size=int(parallelism["batch_size"]),
            gradient_accumulation=int(parallelism["gradient_accumulation"]),
            expected_optimizer_updates=int(plan["updates"]["planned_steps"]),
            seed=int(plan["replication"]["seeds"][0]),
            support_bindings=_agibot_bound_receipt_fields(manifest),
            init_from_path=str(native_config["init_from_path"]),
        )
        config = load_trace_config(path)
        _validate_trace_config_binding(config, manifest, output)
        return path
    except (AgibotTraceError, KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, ModelRunnerError):
            raise
        raise ModelRunnerError(f"MODEL_RUN_AGIBOT_TRACE_CONFIG_INVALID:{exc}") from exc


def _validate_trace_config_binding(
    config: Mapping[str, object], manifest: Mapping[str, object], output: Path
) -> None:
    try:
        from wmloop.execute.agibot_trace_runtime import canonical_digest

        training = manifest["training"]
        support = training["agibot_training_support"]
        native_config = training["agibot_native_config"]
        plan = training["scale_plan"]
        parallelism = plan["parallelism"]
        expected = {
            "trace_id": manifest["model_run_id"],
            "support_digest": canonical_digest(support),
            "native_config_sha256": native_config["sha256"],
            "init_from_path": native_config["init_from_path"],
            "output_root": str(output.resolve()),
            "world_size": parallelism["world_size"],
            "batch_size": parallelism["batch_size"],
            "gradient_accumulation": parallelism["gradient_accumulation"],
            "expected_optimizer_updates": plan["updates"]["planned_steps"],
            "seed": plan["replication"]["seeds"][0],
            "support_bindings": _agibot_bound_receipt_fields(manifest),
            "validation_sample_window_ids": support["validation_support"]["sample_window_ids"],
        }
    except (KeyError, TypeError) as exc:
        raise ModelRunnerError("MODEL_RUN_AGIBOT_TRACE_BINDING_INVALID") from exc
    if any(config.get(key) != value for key, value in expected.items()):
        raise ModelRunnerError("MODEL_RUN_AGIBOT_TRACE_BINDING_MISMATCH")


def _build_agibot_bridge_command(
    manifest: Mapping[str, object],
    command: Sequence[str],
    *,
    trace_config_path: Path,
) -> list[str]:
    try:
        from wmloop.execute.agibot_trace_runtime import (
            AgibotTraceError,
            build_bridge_command,
        )

        plan = manifest["training"]["scale_plan"]
        if (
            len(command) != 6
            or command[2] != "--config"
            or command[4] != "--init_from"
        ):
            raise ModelRunnerError("MODEL_RUN_AGIBOT_NATIVE_TRAIN_COMMAND_INVALID")
        return build_bridge_command(
            command[0],
            command[1:],
            trace_config_path=trace_config_path,
            bridge_path=Path(__file__).with_name("agibot_trace_bridge.py"),
            world_size=int(plan["parallelism"]["world_size"]),
        )
    except (AgibotTraceError, KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, ModelRunnerError):
            raise
        raise ModelRunnerError(f"MODEL_RUN_AGIBOT_TRACE_BRIDGE_INVALID:{exc}") from exc


def _merge_agibot_training_trace(
    manifest: Mapping[str, object],
    output: Path,
    *,
    trace_config_path: Path,
) -> tuple[Path, str]:
    try:
        from wmloop.execute.agibot_trace_runtime import (
            AgibotTraceError,
            load_trace_config,
            merge_trace_shards,
            trace_shard_path,
        )

        config = load_trace_config(trace_config_path)
        _validate_trace_config_binding(config, manifest, output)
        world_size = int(config["world_size"])
        shard_paths = [trace_shard_path(config, rank) for rank in range(world_size)]
        trace_root = checked_path(
            output / "receipts" / "training-trace",
            code="MODEL_RUN_AGIBOT_TRACE_PATH_INVALID",
            error=ModelRunnerError,
        )
        merged_path = trace_root / "merged.json"
        merged = merge_trace_shards(
            trace_config_path, shard_paths, output_path=merged_path
        )
        if (
            merged.get("optimizer_update_count") != config["expected_optimizer_updates"]
            or merged.get("validation_pass_count") != 1
            or merged.get("rank_count") != world_size
            or merged.get("rank_completion_count") != world_size
        ):
            raise ModelRunnerError("AGIBOT_TRACE_SETTLEMENT_INCOMPLETE")
        return merged_path, sha256_file(merged_path)
    except (AgibotTraceError, OSError, KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, ModelRunnerError):
            raise
        raise ModelRunnerError(f"MODEL_RUN_AGIBOT_TRACE_INVALID:{exc}") from exc


def _validate_agibot_training_trace(
    manifest: Mapping[str, object], output: Path
) -> tuple[Path, str]:
    from wmloop.execute.agibot_trace_runtime import (
        AgibotTraceError,
        load_trace_config,
        merge_trace_shards,
        trace_shard_path,
    )

    try:
        config_path = checked_path(
            output / "runtime" / "agibot-trace-config.json",
            code="MODEL_RUN_AGIBOT_TRACE_PATH_INVALID",
            error=ModelRunnerError,
        )
        config = load_trace_config(config_path)
        _validate_trace_config_binding(config, manifest, output)
        world_size = int(config["world_size"])
        merged_path = checked_path(
            output / "receipts" / "training-trace" / "merged.json",
            code="MODEL_RUN_AGIBOT_TRACE_PATH_INVALID",
            error=ModelRunnerError,
        )
        merged = merge_trace_shards(
            config_path,
            [trace_shard_path(config, rank) for rank in range(world_size)],
        )
        persisted = json.loads(merged_path.read_text(encoding="utf-8"))
        if (
            persisted != merged
            or merged.get("optimizer_update_count") != config["expected_optimizer_updates"]
            or merged.get("validation_pass_count") != 1
            or merged.get("rank_count") != world_size
            or merged.get("rank_completion_count") != world_size
        ):
            raise ModelRunnerError("AGIBOT_TRACE_SETTLEMENT_INCOMPLETE")
        return merged_path, sha256_file(merged_path)
    except (AgibotTraceError, OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, ModelRunnerError):
            raise
        raise ModelRunnerError(f"MODEL_RUN_AGIBOT_TRACE_INVALID:{exc}") from exc


def validate_agibot_support_at_execution(manifest: Mapping[str, object]) -> None:
    training = manifest["training"]  # type: ignore[index]
    source = manifest["source"]  # type: ignore[index]
    support = training["agibot_training_support"]
    try:
        from wmloop.execute.agibot_support import (
            AgibotSupportError,
            agibot_receipt_bindings,
            validate_agibot_training_support,
        )

        if not isinstance(support, Mapping):
            raise ModelRunnerError("MODEL_RUN_AGIBOT_SUPPORT_INVALID")
        validate_agibot_training_support(
            support,
            expected_dataset_root=Path(str(source["dataset_root"])),
            expected_native_source_root=Path(str(source["source_root"])),
        )
        if training.get("agibot_receipt_bindings") != agibot_receipt_bindings(support):
            raise ModelRunnerError("MODEL_RUN_AGIBOT_RECEIPT_BINDINGS_MISMATCH")
    except AgibotSupportError as exc:
        raise ModelRunnerError(f"MODEL_RUN_AGIBOT_SUPPORT_INVALID:{exc}") from exc


def _agibot_bound_receipt_fields(manifest: Mapping[str, object]) -> dict[str, object]:
    training = manifest["training"]  # type: ignore[index]
    bindings = training["agibot_receipt_bindings"]
    if not isinstance(bindings, Mapping):
        raise ModelRunnerError("MODEL_RUN_AGIBOT_RECEIPT_BINDINGS_INVALID")
    return {key: value for key, value in bindings.items() if key != "validation_sample_window_ids"}


def _agibot_receipt_bindings(manifest: Mapping[str, object]) -> Mapping[str, object]:
    training = manifest["training"]  # type: ignore[index]
    bindings = training["agibot_receipt_bindings"]
    if not isinstance(bindings, Mapping):
        raise ModelRunnerError("MODEL_RUN_AGIBOT_RECEIPT_BINDINGS_INVALID")
    return bindings


def _build_agibot_checkpoint_evaluator_command(
    manifest: Mapping[str, object],
    *,
    trace_config_path: Path,
    checkpoint: Path,
    checkpoint_sha256: str,
    raw_output: Path,
) -> list[str]:
    """Launch the package-owned evaluator that reloads the saved weights.

    The profile's evaluation command is descriptive only for protocol v3. A
    profile worker cannot manufacture a checkpoint or evaluation receipt.
    """

    runtime = str(manifest["runtime"]["python"])  # type: ignore[index]
    source = manifest["source"]  # type: ignore[index]
    plan = manifest["training"]["scale_plan"]  # type: ignore[index]
    world_size = int(plan["parallelism"]["world_size"])
    native_root = Path(str(source["source_root"])).resolve()
    args = [
        "--trace-config",
        str(trace_config_path.resolve()),
        "--native-source-root",
        str(native_root),
        "--checkpoint",
        str(checkpoint.resolve()),
        "--checkpoint-sha256",
        checkpoint_sha256,
        "--raw-output",
        str(raw_output.resolve()),
    ]
    module = "wmloop.execute.agibot_checkpoint_evaluator"
    if world_size == 1:
        return [runtime, "-m", module, *args]
    return [
        runtime,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
        str(world_size),
        "--module",
        module,
        *args,
    ]
