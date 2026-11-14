"""Explicit rank-local entrypoint for tracing the native Agibot WMTrainer run."""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path
from typing import Sequence

if __package__:
    from .agibot_trace_runtime import (
        AgibotTraceError,
        install,
        load_trace_config,
        validate_native_training_command,
    )
else:  # Direct script execution puts this file's directory on sys.path.
    from agibot_trace_runtime import (
        AgibotTraceError,
        install,
        load_trace_config,
        validate_native_training_command,
    )


def _parse_args(argv: Sequence[str]) -> tuple[Path, list[str]]:
    try:
        separator = list(argv).index("--")
    except ValueError as exc:
        raise AgibotTraceError("AGIBOT_TRACE_BRIDGE_SEPARATOR_REQUIRED") from exc
    bridge_args = list(argv[:separator])
    native_command = list(argv[separator + 1 :])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-config", required=True, type=Path)
    try:
        parsed = parser.parse_args(bridge_args)
    except SystemExit as exc:
        raise AgibotTraceError("AGIBOT_TRACE_BRIDGE_ARGUMENTS_INVALID") from exc
    if not native_command:
        raise AgibotTraceError("AGIBOT_TRACE_NATIVE_COMMAND_INVALID")
    return parsed.trace_config.expanduser().resolve(), native_command


def main(argv: Sequence[str] | None = None) -> None:
    """Validate the immutable run binding, install hooks, then run native main."""

    config_path, native_command = _parse_args(sys.argv[1:] if argv is None else argv)
    config = load_trace_config(config_path)
    entrypoint, _ = validate_native_training_command(native_command, config)
    try:
        launched_world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))
    except ValueError as exc:
        raise AgibotTraceError("AGIBOT_TRACE_DISTRIBUTED_ENV_INVALID") from exc
    if launched_world_size != int(config["world_size"]):
        raise AgibotTraceError("AGIBOT_TRACE_WORLD_SIZE_MISMATCH")
    if rank < 0 or rank >= launched_world_size:
        raise AgibotTraceError("AGIBOT_TRACE_RANK_INVALID")
    native_source_root = entrypoint.parents[2]
    if not native_source_root.is_dir():
        raise AgibotTraceError("AGIBOT_TRACE_NATIVE_SOURCE_ROOT_MISSING")
    native_source_text = str(native_source_root)
    if native_source_text not in sys.path:
        sys.path.insert(0, native_source_text)

    # Import and patch the reviewed trainer only after the source root is
    # available, but before the native entrypoint imports its trainer classes.
    install(config_path)
    sys.argv = [str(entrypoint), *native_command[1:]]
    runpy.run_path(str(entrypoint), run_name="__main__")


if __name__ == "__main__":
    main()
