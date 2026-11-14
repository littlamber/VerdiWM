"""Best-effort host-process isolation for the Agibot native runner.

This module keeps routine process files and telemetry clients inside a model
run's output directory. It is not an operating-system sandbox: native code
can still open arbitrary files or sockets available to the current user.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
import stat
import tempfile
from typing import Callable

from wmloop.runtime_env import runtime_subprocess_env
from wmloop.storage import checked_path


_PASSTHROUGH_PREFIXES = (
    "CUDA_",
    "CUDNN_",
    "CUBLAS_",
    "NCCL_",
    "GLOO_",
    "TORCH_",
    "PYTORCH_",
    "OMP_",
    "MKL_",
)
_PASSTHROUGH_NAMES = {
    "CMAKE_PREFIX_PATH",
    "LD_PRELOAD",
    "LIBRARY_PATH",
    "LANG",
    "LC_ALL",
}

# Python's multiprocessing AF_UNIX listener adds ``pymp-*/listener-*`` below
# TMPDIR. Keep enough headroom for that suffix on Linux and macOS.
_MAX_TMPDIR_BYTES = 64


def agibot_runtime_context(
    manifest: Mapping[str, object],
    output: Path,
    *,
    lease_environment: Mapping[str, str],
) -> tuple[Path, dict[str, str], Callable[[], None]]:
    """Return a private working directory and a CUDA-capable minimal env.

    PATH, the pinned runtime's library paths, and explicit CUDA/PyTorch/NCCL
    tuning variables survive. Credentials, proxies, arbitrary PYTHONPATH, and
    caller HOME/cache paths do not.
    """

    source = manifest["source"]
    runtime = manifest["runtime"]
    if not isinstance(source, Mapping) or not isinstance(runtime, Mapping):
        raise ValueError("MODEL_RUN_RUNTIME_BINDING_INVALID")
    output_root = checked_path(output, code="MODEL_RUN_OUTPUT_PATH_INVALID")
    runtime_root = _private_directory(output_root, "runtime")
    working_directory = _private_directory(output_root, "runtime/workspace")
    home = _private_directory(output_root, "runtime/home")
    temporary = _private_directory(output_root, "runtime/tmp")
    cache = _private_directory(output_root, "runtime/cache")
    telemetry = _private_directory(output_root, "runtime/telemetry")
    short_temporary, cleanup_temporary_alias = _short_temporary_alias(temporary)

    runtime_python = Path(str(runtime["python"]))
    env = runtime_subprocess_env(runtime_python, base=os.environ)
    for name, value in os.environ.items():
        if name in _PASSTHROUGH_NAMES or name.startswith(_PASSTHROUGH_PREFIXES):
            env[name] = value

    project_root = Path(__file__).resolve().parents[2]
    native_source_root = Path(str(source["source_root"])).resolve()
    env.update(
        {
            "PYTHONPATH": os.pathsep.join((str(native_source_root), str(project_root))),
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "HOME": str(home),
            "TMPDIR": str(short_temporary),
            "TMP": str(short_temporary),
            "TEMP": str(short_temporary),
            "XDG_CACHE_HOME": str(cache),
            "HF_HOME": str(cache / "huggingface"),
            "TORCH_HOME": str(cache / "torch"),
            "TORCH_EXTENSIONS_DIR": str(cache / "torch_extensions"),
            "CUDA_CACHE_PATH": str(cache / "cuda"),
            "WANDB_MODE": "disabled",
            "WANDB_DISABLED": "true",
            "WANDB_CONSOLE": "off",
            "WANDB_DIR": str(telemetry / "wandb"),
            "SWANLAB_MODE": "disabled",
            "SWANLAB_LOG_DIR": str(telemetry / "swanlab"),
            "HF_HUB_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "DO_NOT_TRACK": "1",
            "PIP_NO_INDEX": "1",
        }
    )
    env.update({str(name): str(value) for name, value in lease_environment.items()})
    # GPU lease variables cannot override the private scratch contract.
    env.update(
        {
            "TMPDIR": str(short_temporary),
            "TMP": str(short_temporary),
            "TEMP": str(short_temporary),
        }
    )

    # Validate all derived write locations remain in the campaign output.
    try:
        for key in (
            "HOME",
            "XDG_CACHE_HOME",
            "WANDB_DIR",
            "SWANLAB_LOG_DIR",
            "TORCH_EXTENSIONS_DIR",
            "CUDA_CACHE_PATH",
        ):
            checked = checked_path(Path(env[key]), code="MODEL_RUN_RUNTIME_PATH_INVALID")
            try:
                checked.relative_to(output_root)
            except ValueError as exc:
                raise ValueError("MODEL_RUN_RUNTIME_PATH_INVALID") from exc
        # Verify the intentional symlink points exactly to the private run dir.
        if Path(env["TMPDIR"]).resolve(strict=True) != temporary.resolve(strict=True):
            raise ValueError("MODEL_RUN_RUNTIME_TMP_PATH_INVALID")
        # Keep the runtime root referenced so its existence is guaranteed before
        # the child starts, even though individual paths live in its subdirectories.
        if not runtime_root.is_dir():
            raise ValueError("MODEL_RUN_RUNTIME_PATH_INVALID")
    except BaseException:
        cleanup_temporary_alias()
        raise
    return working_directory, env, cleanup_temporary_alias


def _short_temporary_alias(target: Path) -> tuple[Path, Callable[[], None]]:
    """Make a short TMPDIR alias whose files still land in the private run dir.

    Campaign output roots can be deeply nested. Python multiprocessing creates
    AF_UNIX listener sockets below TMPDIR, where Linux and macOS impose a small
    sockaddr path limit. A private, short symlink in the host temp area keeps
    the socket name short while the symlink target keeps temporary files
    inside this run's isolated output directory.
    """

    if os.name == "nt":
        # Windows multiprocessing uses named pipes rather than AF_UNIX paths.
        return target, lambda: None

    parent = _short_temporary_parent()
    namespace = tempfile.TemporaryDirectory(prefix="v-", dir=str(parent))
    alias = Path(namespace.name) / "t"
    try:
        alias.symlink_to(target, target_is_directory=True)
        if len(os.fsencode(alias)) > _MAX_TMPDIR_BYTES:
            raise ValueError("MODEL_RUN_RUNTIME_TMP_PATH_TOO_LONG")
    except BaseException:
        namespace.cleanup()
        raise
    return alias, namespace.cleanup


def _short_temporary_parent() -> Path:
    """Select a writable, private or sticky temp ancestor with path headroom."""

    temporary_root = Path(tempfile.gettempdir()).expanduser().absolute()
    candidates = (temporary_root, *temporary_root.parents)
    for candidate in candidates:
        if candidate == candidate.parent:
            continue
        try:
            metadata = candidate.stat()
        except OSError:
            continue
        if not candidate.is_dir() or not os.access(candidate, os.W_OK | os.X_OK):
            continue
        is_private_owner_dir = (
            hasattr(os, "getuid")
            and metadata.st_uid == os.getuid()
            and not metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
        )
        is_sticky_directory = bool(metadata.st_mode & stat.S_ISVTX)
        if not (is_private_owner_dir or is_sticky_directory):
            continue
        # TemporaryDirectory adds "v-" plus a random suffix, and the alias
        # adds "/t". Keep the resulting path within the AF_UNIX safety budget.
        if len(os.fsencode(candidate)) + 12 <= _MAX_TMPDIR_BYTES:
            return candidate
    raise ValueError("MODEL_RUN_RUNTIME_TMP_PATH_UNAVAILABLE")


def _private_directory(output_root: Path, relative: str) -> Path:
    path = checked_path(
        output_root / relative, code="MODEL_RUN_RUNTIME_PATH_INVALID"
    )
    try:
        path.relative_to(output_root)
    except ValueError as exc:
        raise ValueError("MODEL_RUN_RUNTIME_PATH_INVALID") from exc
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not path.is_dir():
        raise ValueError("MODEL_RUN_RUNTIME_PATH_INVALID")
    return checked_path(path, code="MODEL_RUN_RUNTIME_PATH_INVALID")
