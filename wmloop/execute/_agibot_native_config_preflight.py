"""Check native Agibot config and warm-start architecture before GPU admission."""

from __future__ import annotations

import json
import os
import re
import subprocess
from collections.abc import Mapping
from pathlib import Path

from wmloop.runtime_env import runtime_subprocess_env
from wmloop.storage import checked_path


_PREFLIGHT_TIMEOUT_SECONDS = 30
_PATH_FRAGMENT = re.compile(
    r"(?<![A-Za-z0-9])(?:/[^\s'\"<>|]+|~/[^\s'\"<>|]+|[A-Za-z]:[\\/][^\s'\"<>|]+)"
)
_PREFLIGHT_SCRIPT = """
import json
from pathlib import Path
import sys

try:
    from sa_wm.config import load_config

    config = load_config(sys.argv[1], [])

    from sa_wm.wan.model import WanModelAction
    from sa_wm.world_model.trainer import WMTrainer
    from safetensors import safe_open
    import torch

    from wmloop.execute._agibot_checkpoint_compatibility import (
        compare_agibot_checkpoint_architecture,
    )
    from wmloop.execute._agibot_history_reference_integrity import (
        inspect_agibot_history_reference_parameters,
        should_scan_restored_history_reference,
    )

    checkpoint_root = Path(sys.argv[2])
    checkpoint_path = checkpoint_root / "dit_model.safetensors"

    # Build the exact WMTrainer DiT architecture on meta tensors. This calls
    # the native _init_dit configuration path while avoiding both base-model
    # weight loading and GPU/large-RAM allocation.
    original_from_pretrained = WanModelAction.from_pretrained

    @classmethod
    def _from_pretrained_meta(cls, pretrained_model_name_or_path, **kwargs):
        model_config = cls.load_config(pretrained_model_name_or_path)
        action_config = kwargs.get("action_config")
        with torch.device("meta"):
            model = cls.from_config(model_config, action_config=action_config)
        model.to = lambda *args, **options: model
        return model

    WanModelAction.from_pretrained = _from_pretrained_meta
    trainer = object.__new__(WMTrainer)
    trainer.mcfg = config.world_model
    trainer.tcfg = config.wm_training
    trainer.device = torch.device("cpu")
    trainer.dtype = torch.float32
    WMTrainer._init_dit(trainer)
    target_shapes = {
        key: tuple(value.shape)
        for key, value in trainer._dit.state_dict().items()
    }
    target_parameter_shapes = {
        key: tuple(value.shape)
        for key, value in trainer._dit.named_parameters()
        if key.startswith("history_reference.")
    }
    WanModelAction.from_pretrained = original_from_pretrained

    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as checkpoint:
        checkpoint_shapes = {
            key: tuple(checkpoint.get_slice(key).get_shape())
            for key in checkpoint.keys()
        }
        report = compare_agibot_checkpoint_architecture(
            checkpoint_shapes,
            target_shapes,
            history_ref_enabled=bool(config.world_model.history_ref_enabled),
        )
        if not report.compatible:
            raise RuntimeError("checkpoint architecture mismatch: " + report.summary())

        # A wholly absent branch is a supported native warm-start case: the
        # native loader initializes it after materialization and the trainer's
        # own state gate checks that initialized branch. Only scan a restored
        # branch, using the same rules as _validate_reference_training_state.
        scan_history_reference = should_scan_restored_history_reference(
            enabled=bool(config.world_model.history_ref_enabled),
            null_enabled=bool(config.world_model.history_ref_null),
            expected_names=report.history_reference_expected,
            missing_names=report.history_reference_missing,
        )
        if scan_history_reference:
            integrity = inspect_agibot_history_reference_parameters(
                target_parameter_shapes,
                read_chunk=lambda name, start, stop: (
                    checkpoint.get_tensor(name)
                    if start is None
                    else checkpoint.get_slice(name)[start:stop]
                ),
                is_finite=lambda values: bool(torch.isfinite(values).all().item()),
                has_nonzero=lambda values: bool(torch.count_nonzero(values).item()),
            )
            if not integrity.compatible:
                affected = tuple(
                    dict.fromkeys((*integrity.non_finite, *integrity.dead_weights))
                )
                details = ",".join(affected[:3])
                if len(affected) > 3:
                    details += ",…"
                categories = []
                if integrity.non_finite:
                    categories.append("non-finite")
                if integrity.dead_weights:
                    categories.append("dead encoder/QKV weights")
                reason = (
                    "history-reference checkpoint integrity failed ("
                    + " and ".join(categories)
                    + "). Verify the corrected H1 checkpoint-134750 before use, "
                    "then retry with a new campaign ID"
                )
                if details:
                    reason += "; affected=" + details
                raise RuntimeError(reason)
except Exception as exc:
    reason = " ".join(str(exc).split())[:500]
    sys.stdout.write(json.dumps({"error_type": type(exc).__name__, "reason": reason}) + "\\n")
    raise SystemExit(1)
"""


class AgibotNativeConfigPreflightError(RuntimeError):
    """The selected native runtime rejected the frozen Agibot config."""


def preflight_agibot_native_config(manifest: Mapping[str, object]) -> None:
    """Validate config and state-dict compatibility in the selected runtime."""

    source = manifest.get("source")
    runtime = manifest.get("runtime")
    training = manifest.get("training")
    if not all(isinstance(value, Mapping) for value in (source, runtime, training)):
        raise AgibotNativeConfigPreflightError(
            "MODEL_RUN_AGIBOT_NATIVE_CONFIG_PREFLIGHT_FAILED:manifest binding invalid"
        )

    native_config = training.get("agibot_native_config")
    if not isinstance(native_config, Mapping):
        raise AgibotNativeConfigPreflightError(
            "MODEL_RUN_AGIBOT_NATIVE_CONFIG_PREFLIGHT_FAILED:config binding missing"
        )

    try:
        source_root = checked_path(
            Path(str(source.get("source_root"))),
            code="MODEL_RUN_AGIBOT_NATIVE_CONFIG_PREFLIGHT_FAILED:source path invalid",
            error=AgibotNativeConfigPreflightError,
        )
        config_path = checked_path(
            Path(str(native_config.get("path"))),
            code="MODEL_RUN_AGIBOT_NATIVE_CONFIG_PREFLIGHT_FAILED:config path invalid",
            error=AgibotNativeConfigPreflightError,
        )
        runtime_python = Path(str(runtime.get("python"))).expanduser()
    except (TypeError, ValueError, OSError) as exc:
        if isinstance(exc, AgibotNativeConfigPreflightError):
            raise
        raise AgibotNativeConfigPreflightError(
            "MODEL_RUN_AGIBOT_NATIVE_CONFIG_PREFLIGHT_FAILED:runtime binding invalid"
        ) from exc

    if not source_root.is_dir() or not config_path.is_file() or not runtime_python.is_file():
        raise AgibotNativeConfigPreflightError(
            "MODEL_RUN_AGIBOT_NATIVE_CONFIG_PREFLIGHT_FAILED:runtime input unavailable"
        )

    try:
        from wmloop.control.agibot_native_config import (
            AgibotNativeConfigError,
            validate_agibot_initialization_assets,
        )

        import yaml

        parsed_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        world_model = (
            parsed_config.get("world_model")
            if isinstance(parsed_config, Mapping)
            else None
        )
        asset_bindings = source.get("asset_bindings")
        if not isinstance(world_model, Mapping) or not isinstance(asset_bindings, Mapping):
            raise AgibotNativeConfigPreflightError(
                "MODEL_RUN_AGIBOT_INITIALIZATION_ASSET_PREFLIGHT_FAILED:config bindings invalid"
            )
        validate_agibot_initialization_assets(
            base_model=str(native_config.get("base_model_path", "")),
            init_from=str(native_config.get("init_from_path", "")),
            vae_checkpoint=str(native_config.get("vae_checkpoint_path", "")),
            world_model=world_model,
        )
        if (
            asset_bindings.get("--base-model") != native_config.get("base_model_path")
            or asset_bindings.get("--init_from") != native_config.get("init_from_path")
            or asset_bindings.get("--vae-checkpoint")
            != native_config.get("vae_checkpoint_path")
        ):
            raise AgibotNativeConfigPreflightError(
                "MODEL_RUN_AGIBOT_INITIALIZATION_ASSET_PREFLIGHT_FAILED:role binding mismatch"
            )
    except AgibotNativeConfigPreflightError:
        raise
    except AgibotNativeConfigError as exc:
        raise AgibotNativeConfigPreflightError(
            f"MODEL_RUN_AGIBOT_INITIALIZATION_ASSET_PREFLIGHT_FAILED:{exc}"
        ) from exc
    except (OSError, ValueError, TypeError) as exc:
        raise AgibotNativeConfigPreflightError(
            "MODEL_RUN_AGIBOT_INITIALIZATION_ASSET_PREFLIGHT_FAILED:config could not be read"
        ) from exc

    project_root = Path(__file__).resolve().parents[2]
    env = runtime_subprocess_env(runtime_python, base=os.environ)
    env.update(
        {
            "PYTHONPATH": os.pathsep.join((str(source_root), str(project_root))),
            "PYTHONNOUSERSITE": "1",
            "CUDA_VISIBLE_DEVICES": "",
            "NVIDIA_VISIBLE_DEVICES": "void",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "HF_HUB_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
        }
    )

    try:
        completed = subprocess.run(
            [
                str(runtime_python),
                "-c",
                _PREFLIGHT_SCRIPT,
                str(config_path),
                str(native_config.get("init_from_path", "")),
            ],
            cwd=source_root,
            env=env,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=_PREFLIGHT_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise AgibotNativeConfigPreflightError(
            "MODEL_RUN_AGIBOT_NATIVE_CONFIG_PREFLIGHT_FAILED:loader timed out"
        ) from exc
    except OSError as exc:
        raise AgibotNativeConfigPreflightError(
            "MODEL_RUN_AGIBOT_NATIVE_CONFIG_PREFLIGHT_FAILED:loader could not start"
        ) from exc

    if completed.returncode == 0:
        return

    reason = _loader_failure_reason(completed.stdout)
    raise AgibotNativeConfigPreflightError(
        f"MODEL_RUN_AGIBOT_NATIVE_CONFIG_PREFLIGHT_FAILED:{reason}"
    )


def _loader_failure_reason(stdout: str) -> str:
    safe_stdout = _redact_paths(stdout)
    for line in reversed(safe_stdout.splitlines()):
        try:
            result = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(result, Mapping):
            continue
        error_type = result.get("error_type")
        reason = result.get("reason")
        if isinstance(error_type, str) and isinstance(reason, str):
            message = " ".join(reason.split())[:240]
            return f"{error_type}: {message}" if message else error_type
    return "native loader rejected config"


def _redact_paths(text: str) -> str:
    """Hide absolute filesystem paths from native-runtime diagnostics."""

    return _PATH_FRAGMENT.sub("<path>", text)
