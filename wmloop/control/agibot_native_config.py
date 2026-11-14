"""Private, hash-bound Agibot training configuration derivation."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path

from wmloop.storage import checked_path, exclusive_file_lock


class AgibotNativeConfigError(ValueError):
    """An Agibot source or derived config cannot be bound safely."""


def bind_agibot_native_config(
    value: object,
    plan: Mapping[str, object],
    *,
    output_root: Path,
    dataset_root: Path,
    asset_bindings: Mapping[str, str],
) -> dict[str, object]:
    """Derive a campaign-private config with explicit model and VAE inputs."""

    source_path = _file(value, "MODEL_RUN_AGIBOT_NATIVE_CONFIG_REQUIRED")
    try:
        source_bytes = source_path.read_bytes()
        source = _parse(source_bytes)
        derived = _copy_mapping(source)
        base_model, init_from, vae_checkpoint = _initialization_assets(asset_bindings)
        output = checked_path(output_root, code="MODEL_RUN_AGIBOT_NATIVE_CONFIG_INVALID", error=AgibotNativeConfigError)
        checkpoint_root = output / "checkpoints"
        derived["output_dir"] = str(checkpoint_root)
        world_model = derived.get("world_model")
        world_size = _world_size(plan)
        if not isinstance(world_model, dict):
            raise AgibotNativeConfigError("MODEL_RUN_AGIBOT_NATIVE_CONFIG_INVALID")
        _validate_fsdp_plan(source, world_size)
        # The Diffusers base model initializes the architecture; the user's
        # action checkpoint is loaded separately by train_world_model.py.
        # Never inherit machine-local paths from the supplied YAML.
        validate_agibot_initialization_assets(
            base_model=base_model,
            init_from=init_from,
            vae_checkpoint=vae_checkpoint,
            world_model=world_model,
        )
        world_model["checkpoint"] = str(base_model)
        world_model["vae_checkpoint"] = str(vae_checkpoint)
        if "fsdp_shard_size" in world_model:
            world_model["fsdp_shard_size"] = world_size
        world_model["batch_size"] = _parallelism(plan)["batch_size"]
        world_model["grad_accum_steps"] = _parallelism(plan)["gradient_accumulation"]
        updates = _planned_updates(plan)
        accumulation = _parallelism(plan)["gradient_accumulation"]
        microsteps = updates * accumulation
        world_model["max_train_steps"] = microsteps
        derived["seed"] = _seed(plan)
        wm_training = derived.get("wm_training")
        if not isinstance(wm_training, dict):
            raise AgibotNativeConfigError("MODEL_RUN_AGIBOT_NATIVE_CONFIG_INVALID")
        wm_training["warmup_steps"] = _bounded_warmup_steps(source, updates)
        wm_training["output_dir"] = str(checkpoint_root)
        wm_training["dataset_roots"] = [str(dataset_root)]
        wm_training["dataset_probs"] = [1.0]
        wm_training["val_dataset_probs"] = [1.0]
        # The protocol needs one checkpoint before its one frozen validation
        # pass. Disable inherited periodic diagnostics and RGB rollout metrics,
        # which can consume validation samples outside that panel.
        wm_training["save_interval"] = microsteps
        wm_training["save_final_checkpoint"] = True
        wm_training["val_interval"] = microsteps
        wm_training["diagnose_interval"] = 0
        wm_training["val_metrics_enabled"] = False
        wm_training["val_metrics_interval"] = 0
        wm_training["val_metrics_num_samples"] = 0
        wm_training["require_val_split"] = True
        derived_bytes = _dump(derived)
        config_dir = checked_path(output / "config", code="MODEL_RUN_AGIBOT_NATIVE_CONFIG_INVALID", error=AgibotNativeConfigError)
        config_path = config_dir / "agibot-training.yaml"
        _persist_immutable(config_path, derived_bytes, output)
    except AgibotNativeConfigError:
        raise
    except (OSError, ValueError, TypeError) as exc:
        raise AgibotNativeConfigError("MODEL_RUN_AGIBOT_NATIVE_CONFIG_INVALID") from exc

    bound = {
        "source_path": str(source_path),
        "source_sha256": _sha256(source_bytes),
        "path": str(config_path),
        "sha256": _sha256(derived_bytes),
        "base_model_path": str(base_model),
        "init_from_path": str(init_from),
        "vae_checkpoint_path": str(vae_checkpoint),
        **_capacities(source),
    }
    validate_agibot_native_config(
        bound,
        plan,
        output_root=output,
        dataset_root=dataset_root,
        asset_bindings=asset_bindings,
    )
    return bound


def validate_agibot_native_config(
    value: Mapping[str, object],
    plan: Mapping[str, object],
    *,
    output_root: Path,
    dataset_root: Path,
    asset_bindings: Mapping[str, str],
) -> None:
    """Verify source identity, initialization assets, and plan capacity."""

    source_path = _file(value.get("source_path"), "MODEL_RUN_AGIBOT_NATIVE_CONFIG_INVALID")
    derived_path = _file(value.get("path"), "MODEL_RUN_AGIBOT_NATIVE_CONFIG_INVALID")
    output = checked_path(output_root, code="MODEL_RUN_AGIBOT_NATIVE_CONFIG_INVALID", error=AgibotNativeConfigError)
    expected_path = output / "config" / "agibot-training.yaml"
    if derived_path != expected_path or not _within(derived_path, output / "config"):
        raise AgibotNativeConfigError("MODEL_RUN_AGIBOT_NATIVE_CONFIG_PATH_INVALID")
    try:
        source_bytes = source_path.read_bytes()
        derived_bytes = derived_path.read_bytes()
        source = _parse(source_bytes)
        derived = _parse(derived_bytes)
        base_model, init_from, vae_checkpoint = _initialization_assets(asset_bindings)
    except (OSError, ValueError, TypeError) as exc:
        raise AgibotNativeConfigError("MODEL_RUN_AGIBOT_NATIVE_CONFIG_INVALID") from exc
    world_size = _world_size(plan)
    _validate_fsdp_plan(source, world_size)
    source_digest = _sha256(source_bytes)
    derived_digest = _sha256(derived_bytes)
    if value.get("source_sha256") != source_digest:
        raise AgibotNativeConfigError("MODEL_RUN_AGIBOT_NATIVE_CONFIG_SOURCE_BINDING_MISMATCH")
    if value.get("sha256") != derived_digest:
        raise AgibotNativeConfigError("MODEL_RUN_AGIBOT_NATIVE_CONFIG_BINDING_MISMATCH")
    expected_derived = _copy_mapping(source)
    checkpoint_root = str(output / "checkpoints")
    expected_derived["output_dir"] = checkpoint_root
    expected_world_model = expected_derived.get("world_model")
    if not isinstance(expected_world_model, dict):
        raise AgibotNativeConfigError("MODEL_RUN_AGIBOT_NATIVE_CONFIG_INVALID")
    validate_agibot_initialization_assets(
        base_model=base_model,
        init_from=init_from,
        vae_checkpoint=vae_checkpoint,
        world_model=expected_world_model,
    )
    expected_world_model["checkpoint"] = str(base_model)
    expected_world_model["vae_checkpoint"] = str(vae_checkpoint)
    if "fsdp_shard_size" in expected_world_model:
        expected_world_model["fsdp_shard_size"] = world_size
    expected_parallelism = _parallelism(plan)
    expected_world_model["batch_size"] = expected_parallelism["batch_size"]
    expected_world_model["grad_accum_steps"] = expected_parallelism[
        "gradient_accumulation"
    ]
    expected_updates = _planned_updates(plan)
    expected_world_model["max_train_steps"] = (
        expected_updates * expected_parallelism["gradient_accumulation"]
    )
    expected_derived["seed"] = _seed(plan)
    expected_training = expected_derived.get("wm_training")
    if not isinstance(expected_training, dict):
        raise AgibotNativeConfigError("MODEL_RUN_AGIBOT_NATIVE_CONFIG_INVALID")
    expected_training["warmup_steps"] = _bounded_warmup_steps(source, expected_updates)
    expected_training["output_dir"] = checkpoint_root
    expected_training["dataset_roots"] = [str(dataset_root)]
    expected_training["dataset_probs"] = [1.0]
    expected_training["val_dataset_probs"] = [1.0]
    expected_microsteps = (
        expected_updates * expected_parallelism["gradient_accumulation"]
    )
    expected_training["save_interval"] = expected_microsteps
    expected_training["save_final_checkpoint"] = True
    expected_training["val_interval"] = expected_microsteps
    expected_training["diagnose_interval"] = 0
    expected_training["val_metrics_enabled"] = False
    expected_training["val_metrics_interval"] = 0
    expected_training["val_metrics_num_samples"] = 0
    expected_training["require_val_split"] = True
    if derived != expected_derived:
        raise AgibotNativeConfigError("MODEL_RUN_AGIBOT_NATIVE_CONFIG_PATH_OVERRIDE_INVALID")
    expected = {
        "source_path": str(source_path),
        "source_sha256": source_digest,
        "path": str(derived_path),
        "sha256": derived_digest,
        "base_model_path": str(base_model),
        "init_from_path": str(init_from),
        "vae_checkpoint_path": str(vae_checkpoint),
        **_capacities(source),
    }
    if dict(value) != expected:
        raise AgibotNativeConfigError("MODEL_RUN_AGIBOT_NATIVE_CONFIG_BINDING_MISMATCH")
    dataset = plan.get("dataset")
    domain_support = dataset.get("domain_support") if isinstance(dataset, Mapping) else None
    required_capacity = domain_support.get("required_action_num_domains") if isinstance(domain_support, Mapping) else None
    capacities = _capacities(derived)
    if (
        isinstance(required_capacity, bool)
        or not isinstance(required_capacity, int)
        or required_capacity < 1
        or capacities["world_model_action_num_domains"] < required_capacity
        or capacities["wm_training_action_num_domains"] < required_capacity
    ):
        raise AgibotNativeConfigError("MODEL_RUN_AGIBOT_NATIVE_CONFIG_DOMAIN_CAPACITY_TOO_SMALL")


def _persist_immutable(path: Path, payload: bytes, output_root: Path) -> None:
    path = checked_path(path, code="MODEL_RUN_AGIBOT_NATIVE_CONFIG_INVALID", error=AgibotNativeConfigError)
    if not _within(path, output_root / "config"):
        raise AgibotNativeConfigError("MODEL_RUN_AGIBOT_NATIVE_CONFIG_PATH_INVALID")
    lock_root = output_root.parent / ".config-locks"
    lock_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock_path = lock_root / f"{output_root.name}.agibot-native-config.lock"
    with exclusive_file_lock(lock_path):
        if path.exists():
            if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
                raise AgibotNativeConfigError("MODEL_RUN_AGIBOT_NATIVE_CONFIG_CONFLICT")
            return
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        checked_path(path.parent, code="MODEL_RUN_AGIBOT_NATIVE_CONFIG_INVALID", error=AgibotNativeConfigError)
        try:
            with path.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            path.chmod(0o400)
        except FileExistsError:
            if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
                raise AgibotNativeConfigError("MODEL_RUN_AGIBOT_NATIVE_CONFIG_CONFLICT")


def _parse(raw: bytes) -> dict[str, object]:
    try:
        import yaml

        parsed = yaml.safe_load(raw.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValueError("invalid YAML") from exc
    if not isinstance(parsed, Mapping):
        raise ValueError("config root must be a mapping")
    result = dict(parsed)
    if not isinstance(result.get("wm_training"), Mapping):
        raise ValueError("wm_training section missing")
    _capacities(result)
    return result


def _dump(value: Mapping[str, object]) -> bytes:
    import yaml

    return yaml.safe_dump(dict(value), sort_keys=False, allow_unicode=True).encode("utf-8")


def _capacities(value: Mapping[str, object]) -> dict[str, int]:
    world_model = value.get("world_model")
    wm_training = value.get("wm_training")
    if not isinstance(world_model, Mapping) or not isinstance(wm_training, Mapping):
        raise ValueError("config sections missing")
    capacities = {
        "world_model_action_num_domains": world_model.get("action_num_domains"),
        "wm_training_action_num_domains": wm_training.get("action_num_domains"),
    }
    if any(
        isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1
        for capacity in capacities.values()
    ):
        raise ValueError("action domain capacities must be positive integers")
    return {key: int(capacity) for key, capacity in capacities.items()}


def _initialization_assets(
    asset_bindings: Mapping[str, str],
) -> tuple[Path, Path, Path]:
    """Bind the Diffusers base, action warm-start, and VAE by their roles."""

    raw_base_model = asset_bindings.get("--base-model")
    if not raw_base_model:
        raise AgibotNativeConfigError("MODEL_RUN_AGIBOT_BASE_MODEL_REQUIRED")
    base_model = _asset_directory(
        raw_base_model, "MODEL_RUN_AGIBOT_BASE_MODEL_INVALID"
    )
    raw_init_from = asset_bindings.get("--init_from")
    if not raw_init_from:
        raise AgibotNativeConfigError("MODEL_RUN_AGIBOT_WARMSTART_REQUIRED")
    init_from = _asset_directory(
        raw_init_from, "MODEL_RUN_AGIBOT_WARMSTART_INVALID"
    )

    raw_vae_checkpoint = asset_bindings.get("--vae-checkpoint")
    if raw_vae_checkpoint:
        vae_checkpoint = _asset_file(
            raw_vae_checkpoint, "MODEL_RUN_AGIBOT_VAE_CHECKPOINT_INVALID"
        )
    else:
        default_vae = base_model / "Wan2.2_VAE.pth"
        try:
            vae_checkpoint = _asset_file(
                default_vae, "MODEL_RUN_AGIBOT_VAE_CHECKPOINT_REQUIRED"
            )
        except AgibotNativeConfigError as exc:
            if str(exc) == "MODEL_RUN_AGIBOT_VAE_CHECKPOINT_REQUIRED":
                raise AgibotNativeConfigError(
                    "MODEL_RUN_AGIBOT_VAE_CHECKPOINT_REQUIRED:provide --vae-checkpoint or place Wan2.2_VAE.pth in the selected base Diffusers directory"
                ) from exc
            raise
    return base_model, init_from, vae_checkpoint


def validate_agibot_initialization_assets(
    *,
    base_model: str | Path,
    init_from: str | Path,
    vae_checkpoint: str | Path,
    world_model: Mapping[str, object],
) -> None:
    """Validate the two model inputs and their metadata before GPU admission.

    These checks establish asset completeness and declared config agreement.
    Tensor-key and tensor-shape compatibility still belongs to the native
    training loader and is not claimed by this CPU preflight.
    """

    base_root = _asset_directory(
        base_model, "MODEL_RUN_AGIBOT_BASE_MODEL_INVALID"
    )
    warmstart_root = _asset_directory(
        init_from, "MODEL_RUN_AGIBOT_WARMSTART_INVALID"
    )
    vae_path = _asset_file(
        vae_checkpoint, "MODEL_RUN_AGIBOT_VAE_CHECKPOINT_INVALID"
    )
    _validate_diffusers_base_model(base_root)
    if vae_path.stat().st_size == 0:
        raise AgibotNativeConfigError("MODEL_RUN_AGIBOT_VAE_CHECKPOINT_INVALID:empty file")
    _validate_warmstart_checkpoint(warmstart_root, world_model)


def _validate_diffusers_base_model(root: Path) -> None:
    config_path = root / "config.json"
    config_path = _asset_file(
        config_path, "MODEL_RUN_AGIBOT_BASE_MODEL_CONFIG_REQUIRED"
    )
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AgibotNativeConfigError(
            "MODEL_RUN_AGIBOT_BASE_MODEL_CONFIG_INVALID"
        ) from exc
    if not isinstance(config, Mapping) or not isinstance(config.get("model_type"), str):
        raise AgibotNativeConfigError("MODEL_RUN_AGIBOT_BASE_MODEL_CONFIG_INVALID")

    index_path = root / "diffusion_pytorch_model.safetensors.index.json"
    if index_path.exists() or index_path.is_symlink():
        index_path = _asset_file(
            index_path, "MODEL_RUN_AGIBOT_BASE_MODEL_INDEX_INVALID"
        )
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AgibotNativeConfigError(
                "MODEL_RUN_AGIBOT_BASE_MODEL_INDEX_INVALID"
            ) from exc
        weight_map = index.get("weight_map") if isinstance(index, Mapping) else None
        if not isinstance(weight_map, Mapping) or not weight_map:
            raise AgibotNativeConfigError(
                "MODEL_RUN_AGIBOT_BASE_MODEL_INDEX_INVALID"
            )
        for shard_name in set(weight_map.values()):
            if not isinstance(shard_name, str) or not shard_name:
                raise AgibotNativeConfigError(
                    "MODEL_RUN_AGIBOT_BASE_MODEL_INDEX_INVALID"
                )
            relative = Path(shard_name)
            if relative.is_absolute() or ".." in relative.parts:
                raise AgibotNativeConfigError(
                    "MODEL_RUN_AGIBOT_BASE_MODEL_INDEX_INVALID"
                )
            shard = root / relative
            shard = _asset_file(
                shard,
                f"MODEL_RUN_AGIBOT_BASE_MODEL_SHARD_MISSING:{relative.name}",
            )
            if shard.stat().st_size == 0:
                raise AgibotNativeConfigError(
                    f"MODEL_RUN_AGIBOT_BASE_MODEL_SHARD_EMPTY:{relative.name}"
                )
    elif not any(
        (root / name).is_file() and not (root / name).is_symlink()
        for name in (
            "diffusion_pytorch_model.safetensors",
            "diffusion_pytorch_model.bin",
        )
    ):
        raise AgibotNativeConfigError(
            "MODEL_RUN_AGIBOT_BASE_MODEL_WEIGHTS_REQUIRED"
        )
    else:
        weights = next(
            root / name
            for name in (
                "diffusion_pytorch_model.safetensors",
                "diffusion_pytorch_model.bin",
            )
            if (root / name).is_file() and not (root / name).is_symlink()
        )
        weights = _asset_file(
            weights, "MODEL_RUN_AGIBOT_BASE_MODEL_WEIGHTS_REQUIRED"
        )
        if weights.stat().st_size == 0:
            raise AgibotNativeConfigError("MODEL_RUN_AGIBOT_BASE_MODEL_WEIGHTS_EMPTY")


def _validate_warmstart_checkpoint(
    root: Path, world_model: Mapping[str, object]
) -> None:
    weights = root / "dit_model.safetensors"
    weights = _asset_file(weights, "MODEL_RUN_AGIBOT_WARMSTART_WEIGHTS_REQUIRED")
    if weights.stat().st_size == 0:
        raise AgibotNativeConfigError("MODEL_RUN_AGIBOT_WARMSTART_WEIGHTS_EMPTY")
    metadata_path = root / "meta.json"
    metadata_path = _asset_file(
        metadata_path, "MODEL_RUN_AGIBOT_WARMSTART_METADATA_REQUIRED"
    )
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AgibotNativeConfigError(
            "MODEL_RUN_AGIBOT_WARMSTART_METADATA_INVALID"
        ) from exc
    model_config = (
        metadata.get("model_config_dict") if isinstance(metadata, Mapping) else None
    )
    if not isinstance(model_config, Mapping):
        raise AgibotNativeConfigError(
            "MODEL_RUN_AGIBOT_WARMSTART_METADATA_INVALID:model_config_dict required"
        )
    for field in (
        "backbone",
        "action_schema",
        "action_dim",
        "pred_frames",
        "history_length",
    ):
        expected = world_model.get(field)
        actual = model_config.get(field)
        if expected is None or actual is None:
            raise AgibotNativeConfigError(
                f"MODEL_RUN_AGIBOT_WARMSTART_METADATA_INVALID:{field} required"
            )
        if actual != expected or (
            isinstance(expected, int)
            and (isinstance(actual, bool) or not isinstance(actual, int))
        ):
            raise AgibotNativeConfigError(
                f"MODEL_RUN_AGIBOT_WARMSTART_CONFIG_MISMATCH:{field}"
            )


def _asset_directory(value: str | Path, code: str) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_symlink() or not candidate.exists():
        raise AgibotNativeConfigError(code)
    resolved = checked_path(candidate, code=code, error=AgibotNativeConfigError)
    if not resolved.is_dir():
        raise AgibotNativeConfigError(code)
    return resolved


def _asset_file(value: str | Path, code: str) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_symlink() or not candidate.exists():
        raise AgibotNativeConfigError(code)
    resolved = checked_path(candidate, code=code, error=AgibotNativeConfigError)
    if not resolved.is_file():
        raise AgibotNativeConfigError(code)
    return resolved


def _world_size(plan: Mapping[str, object]) -> int:
    parallelism = plan.get("parallelism")
    value = parallelism.get("world_size") if isinstance(parallelism, Mapping) else None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise AgibotNativeConfigError("MODEL_RUN_SCALE_PLAN_INVALID")
    return value


def _validate_fsdp_plan(config: Mapping[str, object], world_size: int) -> None:
    world_model = config.get("world_model")
    if (
        isinstance(world_model, Mapping)
        and world_model.get("fsdp_enabled") is True
        and world_size < 2
    ):
        raise AgibotNativeConfigError(
            "MODEL_RUN_AGIBOT_FSDP_REQUIRES_MULTI_GPU_PLAN:"
            "world_model.fsdp_enabled=true requires a feasible multi-GPU training "
            "scale plan with parallelism.world_size >= 2 and at least that many "
            "allocated GPUs"
        )


def _planned_updates(plan: Mapping[str, object]) -> int:
    updates = plan.get("updates")
    value = updates.get("planned_steps") if isinstance(updates, Mapping) else None
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 1
        or value > 8
    ):
        raise AgibotNativeConfigError(
            "MODEL_RUN_PROTOCOL_V3_ENGINEERING_DIAGNOSTIC_POLICY_INVALID"
        )
    return value


def _seed(plan: Mapping[str, object]) -> int:
    replication = plan.get("replication")
    seeds = replication.get("seeds") if isinstance(replication, Mapping) else None
    if (
        not isinstance(seeds, list)
        or not seeds
        or isinstance(seeds[0], bool)
        or not isinstance(seeds[0], int)
        or seeds[0] < 0
    ):
        raise AgibotNativeConfigError("MODEL_RUN_SCALE_PLAN_INVALID")
    return seeds[0]


def _parallelism(plan: Mapping[str, object]) -> dict[str, int]:
    parallelism = plan.get("parallelism")
    if not isinstance(parallelism, Mapping):
        raise AgibotNativeConfigError("MODEL_RUN_SCALE_PLAN_INVALID")
    result: dict[str, int] = {}
    for key in ("batch_size", "gradient_accumulation"):
        value = parallelism.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise AgibotNativeConfigError("MODEL_RUN_SCALE_PLAN_INVALID")
        result[key] = value
    return result


def _bounded_warmup_steps(source: Mapping[str, object], updates: int) -> int:
    training = source.get("wm_training")
    warmup = training.get("warmup_steps", 0) if isinstance(training, Mapping) else None
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise AgibotNativeConfigError("MODEL_RUN_AGIBOT_NATIVE_CONFIG_INVALID")
    return min(warmup, updates - 1)


def _copy_mapping(value: Mapping[str, object]) -> dict[str, object]:
    # YAML mappings/lists contain only safe-loaded primitive containers.
    import copy

    return copy.deepcopy(dict(value))


def _file(value: object, code: str) -> Path:
    if not isinstance(value, (str, Path)) or not value:
        raise AgibotNativeConfigError(code)
    candidate = Path(value).expanduser()
    if candidate.is_symlink() or not candidate.is_file():
        raise AgibotNativeConfigError(code)
    return checked_path(candidate, code=code, error=AgibotNativeConfigError)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False
