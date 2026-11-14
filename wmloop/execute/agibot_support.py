"""Bindings for the native Agibot video-latent engineering diagnostic."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from wmloop.contracts import ContractValidationError, validate_document
from wmloop.storage import canonical_bytes, checked_path, sha256_file


class AgibotSupportError(ValueError):
    """The native Agibot support set cannot be bound or revalidated."""


_NATIVE_FILES = (
    "scripts/training/train_world_model.py",
    "sa_wm/world_model/trainer.py",
    "sa_wm/data/video_latent_dataset.py",
    "sa_wm/data/video_latent_reader.py",
    "sa_wm/data/video_latent.py",
    "sa_wm/world_model/action_contract.py",
)
_NATIVE_PACKAGE_ROOT = "sa_wm"
_NATIVE_RUNTIME_EXCLUDES = frozenset(
    {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
)
_VAE_TEMPORAL_STRIDE = 4
_TRAIN_WINDOW_RULE = (
    "native_video_latent_training_dataset_v1_all_windows_sorted_episode_uid_then_first_future_video_frame"
)


def build_agibot_training_support(
    dataset_root: Path,
    native_source_root: Path,
    *,
    loader: Mapping[str, object],
    validation_sample_count: int = 128,
) -> dict[str, object]:
    """Bind all native train windows and a deterministic held-out diagnostic set.

    The train index, frozen loader, native source, and enumeration rule bind the
    full train support without allocating one string per window. Only the fixed
    held-out panel is materialized for the diagnostic receipt.
    """

    data_root = _directory(dataset_root, "AGIBOT_SUPPORT_DATA_ROOT_INVALID")
    source_root = _directory(native_source_root, "AGIBOT_SUPPORT_SOURCE_ROOT_INVALID")
    normalized_loader = _validate_loader(loader)
    if (
        isinstance(validation_sample_count, bool)
        or not isinstance(validation_sample_count, int)
        or validation_sample_count < 1
    ):
        raise AgibotSupportError("AGIBOT_SUPPORT_VALIDATION_SAMPLE_COUNT_INVALID")

    paths = {
        "manifest": data_root / "manifest.json",
        "train_index": data_root / "train.index.jsonl",
        "validation_index": data_root / "val.index.jsonl",
        "arm_condition_stats": data_root / "stats" / "arm_slot_eef_pose.json",
        "body_condition_stats": data_root / "stats" / "body_control.json",
    }
    payloads = {key: _json_file(path, f"AGIBOT_SUPPORT_{key.upper()}_INVALID") for key, path in paths.items() if key in {"manifest", "arm_condition_stats", "body_condition_stats"}}
    manifest = payloads["manifest"]
    _validate_agibot_manifest(manifest, payloads["body_condition_stats"])
    train_rows = _jsonl(paths["train_index"], "train")
    val_rows = _jsonl(paths["validation_index"], "validation")
    train_entries = _window_entries(
        train_rows, split="train", loader=normalized_loader
    )
    val_entries = _window_entries(
        val_rows, split="val", loader=normalized_loader
    )
    train_episodes = sorted(entry[0] for entry in train_entries)
    val_episodes = sorted(entry[0] for entry in val_entries)
    if len(set(train_episodes)) != len(train_episodes) or len(set(val_episodes)) != len(val_episodes):
        raise AgibotSupportError("AGIBOT_SUPPORT_EPISODE_UID_DUPLICATE")
    overlap = sorted(set(train_episodes) & set(val_episodes))
    if overlap:
        raise AgibotSupportError("AGIBOT_SUPPORT_TRAIN_VALIDATION_LEAKAGE")

    train_window_count = sum(entry[3] for entry in train_entries)
    if train_window_count == 0:
        raise AgibotSupportError("AGIBOT_SUPPORT_TRAIN_WINDOWS_EMPTY")
    validation_window_count, validation_ids = _sample_native_windows(
        val_entries,
        loader=normalized_loader,
        sample_count=validation_sample_count,
    )
    if validation_sample_count > validation_window_count:
        raise AgibotSupportError("AGIBOT_SUPPORT_VALIDATION_SAMPLE_COUNT_EXCEEDS_SUPPORT")
    input_digests = {key: sha256_file(path) for key, path in paths.items()}
    # Bind every train window without materializing millions of IDs. The raw
    # index, frozen loader, native source digests, and versioned enumeration
    # rule fully determine the support set.
    train_window_digest = _digest(
        {
            "split": "train",
            "index_sha256": input_digests["train_index"],
            "loader": normalized_loader,
            "window_count": train_window_count,
            "window_rule": _TRAIN_WINDOW_RULE,
        }
    )
    shard_catalog = _build_shard_catalog(data_root, train_rows, val_rows)

    native_files = _native_source_files(source_root)
    native_source_digest = _digest(native_files)
    support = {
        "schema_version": 1,
        "artifact_type": "verdiwm-agibot-training-support",
        "state": "bound",
        "dataset_root": str(data_root),
        "native_source_root": str(source_root),
        "inputs": {
            key: input_digests[key] for key in paths
        },
        "production_contract": {
            "dataset_key": manifest["production_contract"]["dataset_key"],
            "family": "agibot",
            "condition_view": "arm_slot_eef_pose",
            "body_conditioning": True,
            "body_stats_source": "train_only",
        },
        "loader": normalized_loader,
        "train_support": {
            "split": "train",
            "episode_count": len(train_episodes),
            "episode_uids_sha256": _string_set_digest(train_episodes),
            "window_count": train_window_count,
            "window_set_sha256": train_window_digest,
            "window_rule": _TRAIN_WINDOW_RULE,
        },
        "validation_support": {
            "split": "val",
            "episode_count": len(val_episodes),
            "episode_uids_sha256": _string_set_digest(val_episodes),
            "window_count": validation_window_count,
            "sample_rule": "evenly_spaced_over_native_sorted_windows_including_first_and_last_v1",
            "sample_count": len(validation_ids),
            "sample_window_ids": validation_ids,
            "sample_window_ids_sha256": _string_set_digest(validation_ids),
            "episode_uids_disjoint_from_train": True,
        },
        "native_source": {
            "files": native_files,
            "sha256": native_source_digest,
        },
        "shard_catalog": shard_catalog,
        "claim_boundary": (
            "Engineering diagnostic support only. Native validation loss may be recorded, "
            "but this contract grants no model-improvement, promotion, or generalization claim."
        ),
    }
    try:
        validate_document("agibot_training_support", support)
    except ContractValidationError as exc:
        raise AgibotSupportError("AGIBOT_SUPPORT_SCHEMA_INVALID") from exc
    return support


def _native_source_files(source_root: Path) -> dict[str, str]:
    """Hash the trainer entrypoint and complete imported ``sa_wm`` package.

    The trainer imports modules beyond the dataset and trainer files that
    define the support contract. Hashing the package as a whole keeps model,
    sampler, normalization, camera, and validation changes bound without
    maintaining a fragile duplicate of Python's import graph.
    """

    package_root = source_root / _NATIVE_PACKAGE_ROOT
    if not package_root.is_dir() or package_root.is_symlink():
        raise AgibotSupportError("AGIBOT_SUPPORT_NATIVE_SOURCE_INVALID")
    required_paths = set(_NATIVE_FILES)
    result: dict[str, str] = {}
    stack = [package_root]
    while stack:
        directory = stack.pop()
        try:
            children = sorted(directory.iterdir(), key=lambda path: path.name)
        except OSError as exc:
            raise AgibotSupportError("AGIBOT_SUPPORT_NATIVE_SOURCE_INVALID") from exc
        for child in children:
            if child.name in _NATIVE_RUNTIME_EXCLUDES:
                continue
            if child.is_symlink():
                raise AgibotSupportError("AGIBOT_SUPPORT_NATIVE_SOURCE_INVALID")
            if child.is_dir():
                stack.append(child)
                continue
            if child.suffix != ".py" or not child.is_file():
                continue
            relative = child.relative_to(source_root).as_posix()
            checked = checked_path(
                child,
                code="AGIBOT_SUPPORT_NATIVE_SOURCE_INVALID",
                error=AgibotSupportError,
            )
            result[relative] = sha256_file(checked)

    entrypoint = source_root / "scripts" / "training" / "train_world_model.py"
    checked_entrypoint = checked_path(
        entrypoint,
        code="AGIBOT_SUPPORT_NATIVE_SOURCE_INVALID",
        error=AgibotSupportError,
    )
    if not checked_entrypoint.is_file():
        raise AgibotSupportError("AGIBOT_SUPPORT_NATIVE_SOURCE_INVALID")
    result[entrypoint.relative_to(source_root).as_posix()] = sha256_file(
        checked_entrypoint
    )
    if not required_paths.issubset(result):
        raise AgibotSupportError("AGIBOT_SUPPORT_NATIVE_SOURCE_INVALID")
    return dict(sorted(result.items()))


def validate_agibot_training_support(
    support: Mapping[str, object],
    *,
    expected_dataset_root: Path | None = None,
    expected_native_source_root: Path | None = None,
) -> None:
    """Rebuild and compare the support binding against current local inputs."""

    if support.get("schema_version") != 1 or support.get("artifact_type") != "verdiwm-agibot-training-support":
        raise AgibotSupportError("AGIBOT_SUPPORT_SCHEMA_INVALID")
    data_root = Path(str(support.get("dataset_root", "")))
    source_root = Path(str(support.get("native_source_root", "")))
    if expected_dataset_root is not None and data_root.resolve() != expected_dataset_root.resolve():
        raise AgibotSupportError("AGIBOT_SUPPORT_DATA_ROOT_MISMATCH")
    if expected_native_source_root is not None and source_root.resolve() != expected_native_source_root.resolve():
        raise AgibotSupportError("AGIBOT_SUPPORT_SOURCE_ROOT_MISMATCH")
    loader = support.get("loader")
    validation = support.get("validation_support")
    if not isinstance(loader, Mapping) or not isinstance(validation, Mapping):
        raise AgibotSupportError("AGIBOT_SUPPORT_SCHEMA_INVALID")
    rebuilt = build_agibot_training_support(
        data_root,
        source_root,
        loader=loader,
        validation_sample_count=int(validation.get("sample_count", 0)),
    )
    if dict(support) != rebuilt:
        raise AgibotSupportError("AGIBOT_SUPPORT_BINDING_MISMATCH")


def agibot_receipt_bindings(support: Mapping[str, object]) -> dict[str, object]:
    """Return the immutable train and held-out identities required by receipts."""

    train = support["train_support"]
    validation = support["validation_support"]
    catalog = support["shard_catalog"]
    return {
        "agibot_support_digest": _digest(support),
        "shard_catalog_sha256": catalog["sha256"],
        "train_episode_uids_sha256": train["episode_uids_sha256"],
        "train_window_set_sha256": train["window_set_sha256"],
        "validation_sample_window_ids_sha256": validation["sample_window_ids_sha256"],
        "validation_sample_window_ids": list(validation["sample_window_ids"]),
    }


def _validate_loader(value: Mapping[str, object]) -> dict[str, object]:
    required_types = {
        "num_history": int,
        "num_frames": int,
        "sample_stride_video_frames": int,
        "history_selector": str,
        "history_offsets": list,
        "history_dilations": list,
        "history_eval_dilation": int,
        "generated_history_enabled": bool,
        "condition_view": str,
        "body_conditioning": bool,
    }
    if set(value) != {*required_types, "history_collapse_prob"} or any(
        type(value[key]) is not kind for key, kind in required_types.items()
    ):
        raise AgibotSupportError("AGIBOT_SUPPORT_LOADER_CONFIG_INVALID")
    collapse_prob = value["history_collapse_prob"]
    if isinstance(collapse_prob, bool) or not isinstance(collapse_prob, (int, float)):
        raise AgibotSupportError("AGIBOT_SUPPORT_LOADER_CONFIG_INVALID")
    result = dict(value)
    result["history_collapse_prob"] = float(collapse_prob)
    if (
        result["num_history"] < 1
        or result["num_frames"] < 1
        or result["sample_stride_video_frames"] < 1
        or result["history_selector"] not in {"recent", "sparse", "first_recent", "first_offset_recent"}
        or result["condition_view"] != "arm_slot_eef_pose"
        or result["body_conditioning"] is not True
        or result["generated_history_enabled"] is not False
        or any(type(offset) is not int or offset < 0 for offset in result["history_offsets"])
        or not result["history_dilations"]
        or any(type(dilation) is not int or dilation < 1 for dilation in result["history_dilations"])
        or not 0.0 <= result["history_collapse_prob"] <= 1.0
        or result["history_eval_dilation"] < 1
    ):
        raise AgibotSupportError("AGIBOT_SUPPORT_LOADER_CONFIG_INVALID")
    return result


def _validate_agibot_manifest(manifest: Mapping[str, Any], body_stats: Mapping[str, Any]) -> None:
    contract = manifest.get("production_contract")
    schema = manifest.get("condition_schema")
    body = manifest.get("condition_schema_v2", {}).get("body") if isinstance(manifest.get("condition_schema_v2"), Mapping) else None
    if (
        not isinstance(contract, Mapping)
        or contract.get("family") != "agibot"
        or contract.get("condition_view") != "arm_slot_eef_pose"
        or contract.get("body_conditioning") is not True
        or not isinstance(schema, Mapping)
        or schema.get("condition_view") != "arm_slot_eef_pose"
        or not isinstance(body, Mapping)
        or body.get("normalization") != "stats/body_control.json schema_version=3 per-slot contract"
        or body_stats.get("schema_version") != 3
        or body_stats.get("source") != "train_only"
    ):
        raise AgibotSupportError("AGIBOT_SUPPORT_PRODUCTION_CONTRACT_INVALID")


def _window_entries(
    rows: list[dict[str, Any]], *, split: str, loader: Mapping[str, object]
) -> list[tuple[str, int, int, int]]:
    """Return compact per-episode window metadata in native sorted order."""

    result: list[tuple[str, int, int, int]] = []
    for row in rows:
        if row.get("split") != split or row.get("dataset") != "agibot":
            raise AgibotSupportError("AGIBOT_SUPPORT_INDEX_SPLIT_INVALID")
        time_doc = row.get("time")
        if not isinstance(time_doc, Mapping):
            raise AgibotSupportError("AGIBOT_SUPPORT_INDEX_TIME_INVALID")
        video_frames = _positive(time_doc.get("num_video_frames"), "AGIBOT_SUPPORT_INDEX_TIME_INVALID")
        latent_frames = _positive(time_doc.get("num_latent_frames"), "AGIBOT_SUPPORT_INDEX_TIME_INVALID")
        episode_uid = row.get("episode_uid")
        if not isinstance(episode_uid, str) or not episode_uid:
            raise AgibotSupportError("AGIBOT_SUPPORT_EPISODE_UID_MISSING")
        count = _native_window_count(
            video_frames=video_frames, latent_frames=latent_frames, loader=loader
        )
        result.append((episode_uid, video_frames, latent_frames, count))
    result.sort(key=lambda entry: entry[0])
    return result


def _native_window_count(
    *, video_frames: int, latent_frames: int, loader: Mapping[str, object]
) -> int:
    required_future = int(loader["num_frames"])
    if latent_frames < int(loader["num_history"]) + required_future:
        return 0
    first = _VAE_TEMPORAL_STRIDE if loader["history_selector"] == "sparse" else 0
    last = min(
        max(0, (latent_frames - required_future) * _VAE_TEMPORAL_STRIDE),
        video_frames - 1,
    )
    if last < first:
        return 0
    return (last - first) // int(loader["sample_stride_video_frames"]) + 1


def _sample_native_windows(
    entries: list[tuple[str, int, int, int]],
    *,
    loader: Mapping[str, object],
    sample_count: int,
) -> tuple[int, list[str]]:
    """Select the existing evenly spaced panel without building the full set."""

    total = sum(entry[3] for entry in entries)
    if total == 0:
        raise AgibotSupportError("AGIBOT_SUPPORT_VAL_WINDOWS_EMPTY")
    if sample_count > total:
        raise AgibotSupportError("AGIBOT_SUPPORT_VALIDATION_SAMPLE_COUNT_EXCEEDS_SUPPORT")
    if sample_count == 1:
        positions = [0]
    else:
        positions = [
            round(index * (total - 1) / (sample_count - 1))
            for index in range(sample_count)
        ]

    sampled: list[str] = []
    position_index = 0
    offset = 0
    for episode_uid, _video_frames, latent_frames, window_count in entries:
        while (
            position_index < len(positions)
            and positions[position_index] < offset + window_count
        ):
            first_future_index = positions[position_index] - offset
            first_future = (
                (_VAE_TEMPORAL_STRIDE if loader["history_selector"] == "sparse" else 0)
                + first_future_index * int(loader["sample_stride_video_frames"])
            )
            sampled.append(f"{episode_uid}:{first_future:06d}")
            position_index += 1
        offset += window_count
    return total, sampled


def _build_shard_catalog(
    data_root: Path,
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
) -> dict[str, object]:
    """Bind referenced native shard names, sizes and index metadata without hashing huge data files."""

    references_by_path: dict[str, list[dict[str, object]]] = {}
    for split, rows in (("train", train_rows), ("val", validation_rows)):
        for row in rows:
            episode_uid = row.get("episode_uid")
            if not isinstance(episode_uid, str) or not episode_uid:
                raise AgibotSupportError("AGIBOT_SUPPORT_EPISODE_UID_MISSING")
            descriptors: list[tuple[str, Mapping[str, Any]]] = []
            for key in ("latent", "signals"):
                descriptor = row.get(key)
                if not isinstance(descriptor, Mapping):
                    raise AgibotSupportError("AGIBOT_SUPPORT_SHARD_REFERENCE_INVALID")
                descriptors.append((key, descriptor))
            conditions = row.get("model_condition_views")
            if not isinstance(conditions, Mapping) or not conditions:
                raise AgibotSupportError("AGIBOT_SUPPORT_SHARD_REFERENCE_INVALID")
            for condition_name, descriptor in sorted(conditions.items()):
                if not isinstance(condition_name, str) or not isinstance(descriptor, Mapping):
                    raise AgibotSupportError("AGIBOT_SUPPORT_SHARD_REFERENCE_INVALID")
                descriptors.append((f"condition:{condition_name}", descriptor))

            for component, descriptor in descriptors:
                relative = _relative_shard_path(descriptor.get("shard"))
                offset_bytes = descriptor.get("offset_bytes")
                if (
                    isinstance(offset_bytes, bool)
                    or not isinstance(offset_bytes, int)
                    or offset_bytes < 0
                ):
                    raise AgibotSupportError("AGIBOT_SUPPORT_SHARD_REFERENCE_INVALID")
                # NativeVideoLatentReader resolves descriptors as
                # ``data_root / "shards" / shard_name``. Keep the index value
                # as a single filename here and add the directory only when
                # resolving the file below.
                relative_text = relative.name
                reference = {
                    "split": split,
                    "episode_uid": episode_uid,
                    "component": component,
                    "metadata_sha256": _digest(
                        {key: value for key, value in descriptor.items() if key != "shard"}
                    ),
                }
                references_by_path.setdefault(relative_text, []).append(reference)

    shard_root = checked_path(
        data_root / "shards",
        code="AGIBOT_SUPPORT_SHARD_PATH_INVALID",
        error=AgibotSupportError,
    )
    files: dict[str, object] = {}
    for relative_text, references in sorted(references_by_path.items()):
        shard_path = checked_path(
            shard_root / relative_text,
            code="AGIBOT_SUPPORT_SHARD_PATH_INVALID",
            error=AgibotSupportError,
        )
        try:
            shard_path.relative_to(shard_root)
        except ValueError as exc:
            raise AgibotSupportError("AGIBOT_SUPPORT_SHARD_PATH_INVALID") from exc
        if not shard_path.is_file():
            raise AgibotSupportError("AGIBOT_SUPPORT_SHARD_MISSING")
        references.sort(
            key=lambda item: (
                str(item["split"]),
                str(item["episode_uid"]),
                str(item["component"]),
            )
        )
        files[f"shards/{relative_text}"] = {
            "size_bytes": shard_path.stat().st_size,
            "reference_count": len(references),
            "references_sha256": hashlib.sha256(canonical_bytes(references)).hexdigest(),
        }
    if not files:
        raise AgibotSupportError("AGIBOT_SUPPORT_SHARD_CATALOG_EMPTY")
    reference_count = sum(int(item["reference_count"]) for item in files.values())
    identity = {
        "files": files,
        "reference_count": reference_count,
    }
    return {
        "file_count": len(files),
        "reference_count": reference_count,
        "files": files,
        "sha256": hashlib.sha256(canonical_bytes(identity)).hexdigest(),
        "claim_boundary": (
            "Catalog binds shard paths and sizes, per-shard reference counts and canonical "
            "reference metadata digests; the train/validation index digests bind source rows. "
            "It does not hash the full shard contents."
        ),
    }


def _relative_shard_path(value: object) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise AgibotSupportError("AGIBOT_SUPPORT_SHARD_REFERENCE_INVALID")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or len(path.parts) != 1
        or path.name != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise AgibotSupportError("AGIBOT_SUPPORT_SHARD_REFERENCE_INVALID")
    return path


def _jsonl(path: Path, split: str) -> list[dict[str, Any]]:
    resolved = checked_path(path, code="AGIBOT_SUPPORT_INDEX_INVALID", error=AgibotSupportError)
    if not resolved.is_file():
        raise AgibotSupportError("AGIBOT_SUPPORT_INDEX_INVALID")
    rows: list[dict[str, Any]] = []
    try:
        with resolved.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise AgibotSupportError("AGIBOT_SUPPORT_INDEX_ROW_INVALID")
                rows.append(row)
    except (OSError, json.JSONDecodeError) as exc:
        raise AgibotSupportError(f"AGIBOT_SUPPORT_{split.upper()}_INDEX_INVALID") from exc
    if not rows:
        raise AgibotSupportError(f"AGIBOT_SUPPORT_{split.upper()}_INDEX_EMPTY")
    return rows


def _json_file(path: Path, code: str) -> dict[str, Any]:
    resolved = checked_path(path, code=code, error=AgibotSupportError)
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AgibotSupportError(code) from exc
    if not isinstance(value, dict):
        raise AgibotSupportError(code)
    return value


def _directory(path: Path, code: str) -> Path:
    resolved = checked_path(path, code=code, error=AgibotSupportError)
    if not resolved.is_dir():
        raise AgibotSupportError(code)
    return resolved


def _positive(value: object, code: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise AgibotSupportError(code)
    return value


def _string_set_digest(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode("utf-8")).hexdigest()


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()
