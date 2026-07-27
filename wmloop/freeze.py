"""Frozen evaluator and held-out split creation for M0.

These utilities never select an evaluator or a benchmark themselves.  They
only turn explicitly supplied paths and trajectory identifiers into a durable,
verifiable baseline protocol.  Any symlink, replacement race, duplicate ID or
manifest inconsistency fails closed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import uuid
from collections.abc import Iterable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from wmloop.acwm_data import (
    AcwmEnvironmentSpec,
    CANONICAL_ACWM_ENVIRONMENTS,
    canonical_dataset_relative_files,
    inspect_acwm_dataset,
)


class FreezeVerificationError(ValueError):
    """A frozen source or deterministic split could not be trusted."""


def freeze_evaluator_files(source_root: Path, relative_paths: Iterable[str]) -> dict[str, Any]:
    """Return a canonical SHA-256 freeze manifest for explicit evaluator files."""

    root = _trusted_root(source_root)
    paths = _normalized_paths(relative_paths)
    entries = [_freeze_entry(root, path) for path in paths]
    return {"schema_version": 1, "algorithm": "sha256", "entries": entries}


def verify_evaluator_freeze(source_root: Path, manifest: Mapping[str, Any]) -> None:
    """Re-open every frozen member and fail if its identity or digest changed."""

    root = _trusted_root(source_root)
    if manifest.get("schema_version") != 1 or manifest.get("algorithm") != "sha256":
        raise FreezeVerificationError("EVALUATOR_MANIFEST_UNSUPPORTED")
    raw_entries = manifest.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise FreezeVerificationError("EVALUATOR_MANIFEST_ENTRIES_INVALID")
    seen: set[str] = set()
    for raw in raw_entries:
        if not isinstance(raw, Mapping):
            raise FreezeVerificationError("EVALUATOR_MANIFEST_ENTRY_INVALID")
        path = str(raw.get("path") or "")
        expected_digest = str(raw.get("sha256") or "")
        expected_size = raw.get("size")
        if path in seen or not _is_digest(expected_digest) or not isinstance(expected_size, int) or expected_size < 0:
            raise FreezeVerificationError("EVALUATOR_MANIFEST_ENTRY_INVALID")
        seen.add(path)
        actual = _freeze_entry(root, path)
        if actual["sha256"] != expected_digest or actual["size"] != expected_size:
            raise FreezeVerificationError(f"EVALUATOR_HASH_MISMATCH:{path}")


def write_evaluator_freeze(destination: Path, manifest: Mapping[str, Any]) -> None:
    """Validate then publish a canonical evaluator freeze manifest atomically."""

    _validate_manifest_shape(manifest)
    _atomic_write_json(destination, manifest)


def make_heldout_split(ids: Iterable[str], *, seed: int, dev_ratio: float) -> dict[str, Any]:
    """Partition stable trajectory identifiers without input-order dependence."""

    normalized = [str(value) for value in ids]
    if len(normalized) < 2:
        raise FreezeVerificationError("SPLIT_IDS_INSUFFICIENT")
    if any(not value for value in normalized):
        raise FreezeVerificationError("SPLIT_ID_EMPTY")
    if len(set(normalized)) != len(normalized):
        raise FreezeVerificationError("SPLIT_IDS_DUPLICATE")
    if not 0.0 < dev_ratio < 1.0:
        raise FreezeVerificationError("SPLIT_RATIO_INVALID")
    ranked = sorted(
        normalized,
        key=lambda value: (hashlib.sha256(f"{seed}\0{value}".encode("utf-8")).hexdigest(), value),
    )
    dev_count = min(len(ranked) - 1, max(1, math.floor(len(ranked) * dev_ratio)))
    return {
        "schema_version": 1,
        "partition_method": "sha256-seeded-rank-v1",
        "seed": seed,
        "dev_ratio": dev_ratio,
        "dev": ranked[:dev_count],
        "accept": ranked[dev_count:],
    }


def write_heldout_split(destination: Path, split: Mapping[str, Any]) -> None:
    _validate_split_shape(split)
    _atomic_write_json(destination, split)


def freeze_acwm_dataset(
    data_root: Path,
    *,
    environment_specs: Iterable[AcwmEnvironmentSpec] = CANONICAL_ACWM_ENVIRONMENTS,
    required_splits: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Hash the complete published ACWM dataset layout as one provenance artifact.

    A normal Hugging Face resume tree may have metadata sidecars and partially
    fetched blobs.  The exact split inventory is checked before reading a
    single byte, so a partial sync cannot be promoted into a baseline freeze.
    """

    root = _trusted_root(data_root)
    specs = tuple(environment_specs)
    scope = _freeze_scope(required_splits, specs=specs)
    inventory = inspect_acwm_dataset(root, environment_specs=specs)
    try:
        paths = canonical_dataset_relative_files(inventory, environment_specs=specs, required_splits=scope)
    except ValueError as exc:
        raise FreezeVerificationError(str(exc)) from exc
    return {
        "schema_version": 2,
        "artifact_type": "acwm-dataset-freeze",
        "algorithm": "sha256",
        "required_splits": list(scope),
        "entries": [_freeze_entry(root, path) for path in paths],
    }


def verify_acwm_dataset_freeze(
    data_root: Path,
    manifest: Mapping[str, Any],
    *,
    environment_specs: Iterable[AcwmEnvironmentSpec] = CANONICAL_ACWM_ENVIRONMENTS,
    required_splits: Iterable[str] | None = None,
) -> None:
    """Require current exact inventory and every frozen file digest to agree."""

    _validate_acwm_dataset_manifest(manifest)
    root = _trusted_root(data_root)
    specs = tuple(environment_specs)
    scope = _dataset_freeze_scope(manifest, specs=specs)
    if required_splits is not None and tuple(required_splits) != scope:
        raise FreezeVerificationError("ACWM_DATASET_FREEZE_SCOPE_MISMATCH")
    inventory = inspect_acwm_dataset(root, environment_specs=specs)
    try:
        expected_paths = canonical_dataset_relative_files(inventory, environment_specs=specs, required_splits=scope)
    except ValueError as exc:
        raise FreezeVerificationError(str(exc)) from exc
    entries = manifest["entries"]
    paths = tuple(str(entry["path"]) for entry in entries)
    if paths != expected_paths:
        raise FreezeVerificationError("ACWM_DATASET_FREEZE_FILESET_MISMATCH")
    for entry in entries:
        actual = _freeze_entry(root, str(entry["path"]))
        if actual != dict(entry):
            raise FreezeVerificationError(f"ACWM_DATASET_FREEZE_HASH_MISMATCH:{entry['path']}")


def dataset_freeze_sha256(manifest: Mapping[str, Any]) -> str:
    """Return the canonical identity used by baseline records and split plans."""

    _validate_acwm_dataset_manifest(manifest)
    content = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def make_acwm_heldout_protocol(
    dataset_freeze: Mapping[str, Any],
    *,
    seed: int,
    dev_ratio: float,
    environment_specs: Iterable[AcwmEnvironmentSpec] = CANONICAL_ACWM_ENVIRONMENTS,
) -> dict[str, Any]:
    """Partition InD model-selection trajectories while reserving all OoD data.

    The provider's public evaluator can still report a whole split.  This
    protocol is the controller-side authority that distinguishes tuning
    feedback (`ind_dev`) from the final InD and OoD acceptance evidence.
    """

    _validate_acwm_dataset_manifest(dataset_freeze)
    specs = tuple(environment_specs)
    per_environment = _acwm_environment_partitions(dataset_freeze, specs=specs, seed=seed, dev_ratio=dev_ratio)
    ind_dev = tuple(path for environment in per_environment.values() for path in environment["ind_dev"])
    ind_accept = tuple(path for environment in per_environment.values() for path in environment["ind_accept"])
    ood_accept = tuple(path for environment in per_environment.values() for path in environment["ood_accept"])
    return {
        "schema_version": 1,
        "artifact_type": "acwm-heldout-protocol",
        "dataset_freeze_sha256": dataset_freeze_sha256(dataset_freeze),
        "seed": seed,
        "dev_ratio": dev_ratio,
        "ind_dev": list(ind_dev),
        "ind_accept": list(ind_accept),
        "ood_accept": list(ood_accept),
        "environment_partitions": per_environment,
    }


def verify_acwm_heldout_protocol(
    dataset_freeze: Mapping[str, Any],
    protocol: Mapping[str, Any],
    *,
    environment_specs: Iterable[AcwmEnvironmentSpec] = CANONICAL_ACWM_ENVIRONMENTS,
) -> None:
    """Fail if a protocol is detached from its data freeze or its deterministic split."""

    _validate_acwm_dataset_manifest(dataset_freeze)
    if protocol.get("schema_version") != 1 or protocol.get("artifact_type") != "acwm-heldout-protocol":
        raise FreezeVerificationError("ACWM_HELDOUT_PROTOCOL_UNSUPPORTED")
    if protocol.get("dataset_freeze_sha256") != dataset_freeze_sha256(dataset_freeze):
        raise FreezeVerificationError("ACWM_HELDOUT_FREEZE_MISMATCH")
    seed, dev_ratio = protocol.get("seed"), protocol.get("dev_ratio")
    if not isinstance(seed, int) or isinstance(seed, bool) or not isinstance(dev_ratio, (int, float)):
        raise FreezeVerificationError("ACWM_HELDOUT_PROTOCOL_INVALID")
    expected = make_acwm_heldout_protocol(
        dataset_freeze,
        seed=seed,
        dev_ratio=float(dev_ratio),
        environment_specs=environment_specs,
    )
    if dict(protocol) != expected:
        raise FreezeVerificationError("ACWM_HELDOUT_PROTOCOL_MISMATCH")


def write_acwm_dataset_freeze(destination: Path, manifest: Mapping[str, Any]) -> None:
    _validate_acwm_dataset_manifest(manifest)
    _atomic_write_json(destination, manifest)


def write_acwm_heldout_protocol(destination: Path, protocol: Mapping[str, Any]) -> None:
    if protocol.get("schema_version") != 1 or protocol.get("artifact_type") != "acwm-heldout-protocol":
        raise FreezeVerificationError("ACWM_HELDOUT_PROTOCOL_UNSUPPORTED")
    _atomic_write_json(destination, protocol)


def _trusted_root(source_root: Path) -> Path:
    root = Path(source_root)
    try:
        metadata = os.lstat(root)
    except FileNotFoundError as exc:
        raise FreezeVerificationError("EVALUATOR_ROOT_MISSING") from exc
    if stat.S_ISLNK(metadata.st_mode):
        raise FreezeVerificationError("EVALUATOR_SYMLINK:root")
    if not stat.S_ISDIR(metadata.st_mode):
        raise FreezeVerificationError("EVALUATOR_ROOT_NOT_DIRECTORY")
    return root.resolve(strict=True)


def _normalized_paths(relative_paths: Iterable[str]) -> list[str]:
    paths = sorted({_normalize_relative_path(value) for value in relative_paths})
    if not paths:
        raise FreezeVerificationError("EVALUATOR_FILESET_EMPTY")
    return paths


def _normalize_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise FreezeVerificationError("EVALUATOR_PATH_INVALID")
    return path.as_posix()


def _freeze_entry(root: Path, relative_path: str) -> dict[str, Any]:
    normalized = _normalize_relative_path(relative_path)
    path = root
    for part in PurePosixPath(normalized).parts:
        path = path / part
        try:
            metadata = os.lstat(path)
        except FileNotFoundError as exc:
            raise FreezeVerificationError(f"EVALUATOR_MEMBER_MISSING:{normalized}") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise FreezeVerificationError(f"EVALUATOR_SYMLINK:{normalized}")
    if not stat.S_ISREG(metadata.st_mode):
        raise FreezeVerificationError(f"EVALUATOR_NON_REGULAR:{normalized}")
    digest, size = _read_stable_digest(path, metadata, normalized)
    return {"path": normalized, "sha256": digest, "size": size}


def _read_stable_digest(path: Path, before: os.stat_result, label: str) -> tuple[str, int]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise FreezeVerificationError(f"EVALUATOR_MEMBER_OPEN_FAILED:{label}") from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or _identity(before) != _identity(opened):
            raise FreezeVerificationError(f"EVALUATOR_MEMBER_CHANGED:{label}")
        hasher = hashlib.sha256()
        total = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
            total += len(chunk)
    finally:
        os.close(descriptor)
    after = os.lstat(path)
    if _identity(before) != _identity(after) or after.st_size != total:
        raise FreezeVerificationError(f"EVALUATOR_MEMBER_CHANGED:{label}")
    return hasher.hexdigest(), total


def _identity(metadata: os.stat_result) -> tuple[int, int, int, int]:
    return metadata.st_dev, metadata.st_ino, metadata.st_mode, metadata.st_size


def _validate_manifest_shape(manifest: Mapping[str, Any], *, schema_version: int = 1) -> None:
    if manifest.get("schema_version") != schema_version or manifest.get("algorithm") != "sha256":
        raise FreezeVerificationError("EVALUATOR_MANIFEST_UNSUPPORTED")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise FreezeVerificationError("EVALUATOR_MANIFEST_ENTRIES_INVALID")
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise FreezeVerificationError("EVALUATOR_MANIFEST_ENTRY_INVALID")
        path = _normalize_relative_path(str(entry.get("path") or ""))
        if path in seen or not _is_digest(str(entry.get("sha256") or "")):
            raise FreezeVerificationError("EVALUATOR_MANIFEST_ENTRY_INVALID")
        if not isinstance(entry.get("size"), int) or entry["size"] < 0:
            raise FreezeVerificationError("EVALUATOR_MANIFEST_ENTRY_INVALID")
        seen.add(path)


def _validate_split_shape(split: Mapping[str, Any]) -> None:
    if split.get("schema_version") != 1 or split.get("partition_method") != "sha256-seeded-rank-v1":
        raise FreezeVerificationError("SPLIT_SCHEMA_UNSUPPORTED")
    dev, accept = split.get("dev"), split.get("accept")
    if not isinstance(dev, list) or not isinstance(accept, list) or not dev or not accept:
        raise FreezeVerificationError("SPLIT_PARTITIONS_INVALID")
    combined = [*dev, *accept]
    if any(not isinstance(value, str) or not value for value in combined) or len(set(combined)) != len(combined):
        raise FreezeVerificationError("SPLIT_IDS_INVALID")


def _validate_acwm_dataset_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("artifact_type") != "acwm-dataset-freeze" or manifest.get("schema_version") != 2:
        raise FreezeVerificationError("ACWM_DATASET_FREEZE_UNSUPPORTED")
    _validate_manifest_shape(manifest, schema_version=2)
    _dataset_freeze_scope(manifest, specs=CANONICAL_ACWM_ENVIRONMENTS)


def _freeze_scope(required_splits: Iterable[str] | None, *, specs: tuple[AcwmEnvironmentSpec, ...]) -> tuple[str, ...]:
    scope = tuple(required_splits) if required_splits is not None else tuple(split for split, _ in specs[0].split_sizes)
    if not scope or len(set(scope)) != len(scope) or any(not isinstance(split, str) or not split for split in scope):
        raise FreezeVerificationError("ACWM_DATASET_FREEZE_SCOPE_INVALID")
    if any(set(scope) - {split for split, _ in spec.split_sizes} for spec in specs):
        raise FreezeVerificationError("ACWM_DATASET_FREEZE_SCOPE_INVALID")
    return scope


def _dataset_freeze_scope(
    manifest: Mapping[str, Any], *, specs: tuple[AcwmEnvironmentSpec, ...]
) -> tuple[str, ...]:
    raw = manifest.get("required_splits")
    if not isinstance(raw, list):
        raise FreezeVerificationError("ACWM_DATASET_FREEZE_SCOPE_INVALID")
    return _freeze_scope(tuple(raw), specs=specs)


def _dataset_video_ids(dataset_freeze: Mapping[str, Any], split: str) -> tuple[str, ...]:
    marker = f"/{split}/"
    ids = tuple(
        str(entry["path"])
        for entry in dataset_freeze["entries"]
        if str(entry["path"]).endswith(".mp4") and marker in str(entry["path"])
    )
    if not ids:
        raise FreezeVerificationError(f"ACWM_HELDOUT_SPLIT_EMPTY:{split}")
    return ids


def _acwm_environment_partitions(
    dataset_freeze: Mapping[str, Any],
    *,
    specs: tuple[AcwmEnvironmentSpec, ...],
    seed: int,
    dev_ratio: float,
) -> dict[str, dict[str, list[str]]]:
    if not specs or len({spec.environment for spec in specs}) != len(specs):
        raise FreezeVerificationError("ACWM_HELDOUT_ENVIRONMENT_SPECS_INVALID")
    partitions: dict[str, dict[str, list[str]]] = {}
    for spec in specs:
        base = spec.dataset_relative_path.rstrip("/")
        ind_test = tuple(path for path in _dataset_video_ids(dataset_freeze, "ind_test") if path.startswith(f"{base}/"))
        ood_test = tuple(path for path in _dataset_video_ids(dataset_freeze, "ood_test") if path.startswith(f"{base}/"))
        if not ind_test or not ood_test:
            raise FreezeVerificationError(f"ACWM_HELDOUT_ENVIRONMENT_SPLIT_EMPTY:{spec.environment}")
        partition = make_heldout_split(ind_test, seed=seed, dev_ratio=dev_ratio)
        partitions[spec.environment] = {
            "ind_dev": partition["dev"],
            "ind_accept": partition["accept"],
            "ood_accept": list(ood_test),
        }
    return partitions


def _atomic_write_json(destination: Path, payload: Mapping[str, Any]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = (json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _is_digest(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)
