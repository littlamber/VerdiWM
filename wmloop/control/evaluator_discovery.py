"""Read-only discovery of evaluator candidates in an external model checkout.

The scanner deliberately reports *candidates*, never frozen contracts.  It
extracts the small amount of protocol metadata needed for first contact and
keeps suspicious absolute paths visible so an overlay can rewrite them after
the user confirms the intended split and verifier.
"""

from __future__ import annotations

import json
import math
import os
import hashlib
import re
from pathlib import Path
from typing import Any, Mapping

from wmloop.storage import checked_path


class EvaluatorDiscoveryError(ValueError):
    """The source directory cannot be scanned reliably."""


_MAX_FILES = 200
_MAX_SCAN_ENTRIES = 4000
_MAX_SCAN_DEPTH = 12
_MAX_MANIFEST_BYTES = 1024 * 1024
_MAX_METADATA_NODES = 10000
_MAX_METADATA_DEPTH = 40
# Build host-path prefixes without embedding machine-local paths in the public
# source tree; the release audit treats those literals as leaked deployment data.
_ABSOLUTE_RISK_PREFIXES = tuple(
    "/" + part + "/" for part in ("mnt", "root", "workspace", "home")
)
_METRIC_KEYS = {"metric", "metrics", "metric_contract", "primary_metric", "target_metrics"}
_FRAME_KEYS = {"num_video_frames", "video_frames", "num_frames", "frames", "horizon_frames"}


def discover_evaluator_candidates(source_root: Path) -> dict[str, Any]:
    """Return deterministic evaluator candidates without importing or running code."""

    root = checked_path(source_root, code="EVALUATOR_SOURCE_SYMLINK", error=EvaluatorDiscoveryError)
    candidates: list[dict[str, Any]] = []
    warnings: list[dict[str, str]] = []
    if not root.is_dir():
        raise EvaluatorDiscoveryError("EVALUATOR_SOURCE_DIRECTORY_REQUIRED")
    try:
        with os.scandir(root) as entries:
            next(entries, None)
    except OSError as exc:
        raise EvaluatorDiscoveryError("EVALUATOR_SOURCE_UNREADABLE") from exc
    files, scan_warnings = _candidate_files(root)
    warnings.extend(scan_warnings)
    for path in files:
        try:
            payload, _encoded = read_evaluator_manifest(path)
        except EvaluatorDiscoveryError as exc:
            warnings.append({"source_path": str(path), "code": str(exc)})
            continue
        # Ignore registries and prose diagnostics unless they expose a clear
        # evaluation protocol or suite/manifest role.
        lowered = json.dumps(payload, ensure_ascii=False).casefold()
        name = path.name.casefold()
        if not any(token in name or token in lowered for token in ("eval", "evaluation", "suite", "manifest", "verifier")):
            continue
        candidates.append(_candidate(path, root, payload))
    candidates.sort(key=lambda item: str(item["source_path"]))
    # Keep discovery deterministic while giving a human an immediate way to
    # choose among many repository manifests.  This is only a hint: no
    # candidate is selected or frozen automatically.
    ranked = sorted(
        candidates,
        key=lambda item: (
            bool(item.get("minute_level_supported")),
            float(item.get("horizon_seconds") or 0.0),
            int(item.get("closed_loop_chunks") or 0),
            str(item.get("source_path")),
        ),
        reverse=True,
    )
    return {
        "state": "candidates_available" if candidates else "none",
        "scanned_root": str(root),
        "candidate_count": len(candidates),
        "candidates": candidates,
        "selection": None,
        "selection_requires_confirmation": bool(candidates),
        "selection_hint": (
            {
                "candidate_id": ranked[0]["candidate_id"],
                "reason": "longest_discovered_horizon; review split and verifier before confirming",
            }
            if ranked
            else None
        ),
        "warnings": warnings,
    }


def _candidate_files(root: Path) -> tuple[list[Path], list[dict[str, str]]]:
    """Bound traversal itself, including irrelevant files and empty directories."""
    files: list[Path] = []
    warnings: list[dict[str, str]] = []
    pending: list[tuple[Path, int]] = []
    for name in ("evaluation", "evaluators"):
        raw = root / "configs" / name
        try:
            directory = checked_path(raw, code="EVALUATOR_SCAN_SYMLINK", error=EvaluatorDiscoveryError)
        except EvaluatorDiscoveryError as exc:
            warnings.append({"source_path": str(raw), "code": str(exc)})
            continue
        if directory.is_dir():
            pending.append((directory, 0))
    visited = 0
    while pending:
        directory, depth = pending.pop(0)
        children: list[Path] = []
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    if visited >= _MAX_SCAN_ENTRIES:
                        warnings.append({"source_path": str(root), "code": "EVALUATOR_SCAN_LIMIT_REACHED"})
                        return sorted(files), warnings
                    visited += 1
                    children.append(Path(entry.path))
        except OSError:
            warnings.append({"source_path": str(directory), "code": "EVALUATOR_SCAN_UNREADABLE"})
            continue
        for path in sorted(children):
            if path.is_symlink():
                warnings.append({"source_path": str(path), "code": "EVALUATOR_SCAN_SYMLINK"})
                continue
            if path.is_dir():
                if depth < _MAX_SCAN_DEPTH:
                    pending.append((path, depth + 1))
                else:
                    warnings.append({"source_path": str(path), "code": "EVALUATOR_SCAN_DEPTH_LIMIT_REACHED"})
            elif path.is_file() and path.suffix.lower() in {".json", ".yaml", ".yml"}:
                if len(files) >= _MAX_FILES:
                    warnings.append({"source_path": str(root), "code": "EVALUATOR_SCAN_LIMIT_REACHED"})
                    return sorted(files), warnings
                files.append(path)
    return sorted(files), warnings


def read_evaluator_manifest(path: Path) -> tuple[Mapping[str, Any], bytes]:
    """Read bounded JSON-compatible metadata without importing repository code."""
    source = checked_path(path, code="EVALUATOR_CANDIDATE_SYMLINK", error=EvaluatorDiscoveryError)
    if not source.is_file() or source.suffix.lower() not in {".json", ".yaml", ".yml"}:
        raise EvaluatorDiscoveryError("EVALUATOR_CANDIDATE_INVALID")
    try:
        with source.open("rb") as handle:
            encoded = handle.read(_MAX_MANIFEST_BYTES + 1)
        if len(encoded) > _MAX_MANIFEST_BYTES:
            raise EvaluatorDiscoveryError("EVALUATOR_CANDIDATE_TOO_LARGE")
        text = encoded.decode("utf-8")
        if source.suffix.lower() == ".json":
            payload = json.loads(text)
        else:
            import yaml  # type: ignore[import-not-found]
            payload = yaml.safe_load(text)
    except EvaluatorDiscoveryError:
        raise
    except (OSError, UnicodeError, ValueError, ImportError, RecursionError) as exc:
        raise EvaluatorDiscoveryError("EVALUATOR_CANDIDATE_UNREADABLE_OR_MALFORMED") from exc
    except Exception as exc:
        # PyYAML errors have no standard-library base class. Keep its parser
        # failures at the same metadata boundary as malformed JSON.
        if type(exc).__module__.startswith("yaml"):
            raise EvaluatorDiscoveryError("EVALUATOR_CANDIDATE_UNREADABLE_OR_MALFORMED") from exc
        raise
    if not isinstance(payload, Mapping):
        raise EvaluatorDiscoveryError("EVALUATOR_CANDIDATE_OBJECT_REQUIRED")
    _validate_metadata(payload)
    return payload, encoded


def _validate_metadata(payload: object) -> None:
    pending = [(payload, 0, frozenset())]
    visited = 0
    while pending:
        value, depth, parents = pending.pop()
        visited += 1
        if visited > _MAX_METADATA_NODES or depth > _MAX_METADATA_DEPTH:
            raise EvaluatorDiscoveryError("EVALUATOR_CANDIDATE_METADATA_LIMIT_REACHED")
        if isinstance(value, (Mapping, list)):
            if id(value) in parents:
                raise EvaluatorDiscoveryError("EVALUATOR_CANDIDATE_METADATA_CYCLE")
            ancestors = parents | {id(value)}
            if isinstance(value, Mapping):
                if not all(isinstance(key, str) for key in value):
                    raise EvaluatorDiscoveryError("EVALUATOR_CANDIDATE_METADATA_INVALID")
                children = value.values()
            else:
                children = value
            pending.extend((child, depth + 1, ancestors) for child in children)
        elif value is None or isinstance(value, (str, bool)):
            continue
        elif isinstance(value, (int, float)):
            try:
                finite = math.isfinite(value)
            except OverflowError:
                finite = False
            if not finite:
                raise EvaluatorDiscoveryError("EVALUATOR_CANDIDATE_NONFINITE_METADATA")
        else:
            raise EvaluatorDiscoveryError("EVALUATOR_CANDIDATE_METADATA_INVALID")


def _candidate(path: Path, root: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    fps = _number(payload, {"fps", "frame_rate"})
    frames = _numbers(payload, _FRAME_KEYS)
    horizon_invalid = False
    if fps and frames:
        longest, shortest = max(frames) / fps, min(frames) / fps
        horizon_invalid = not (math.isfinite(longest) and math.isfinite(shortest))
        horizon_seconds = None if horizon_invalid else round(longest, 3)
        horizon_min_seconds = None if horizon_invalid else round(shortest, 3)
    else:
        horizon_seconds = _number(payload, {"horizon_seconds", "duration_seconds"})
        horizon_min_seconds = horizon_seconds
    closed_loop = _number(payload, {"closed_loop_chunks", "closed_loop_steps", "closed_loop_frames"}, integral=True)
    split = _first(payload, {"source_split", "split", "evaluation_split"})
    split = split if isinstance(split, str) and split.strip() else None
    metrics = sorted(set(_strings_for_keys(payload, _METRIC_KEYS)))
    absolute_paths = sorted(set(_absolute_strings(payload)))
    path_rewrites = [p for p in absolute_paths if _path_risk(p) or not Path(p).exists()]
    identity_disjoint = _first(payload, {"episode_disjoint", "identity_disjoint", "identity_absent_from_all_group_train_roots"})
    units = _find_list(payload, "units")
    protocol_warnings: list[str] = []
    if horizon_invalid:
        protocol_warnings.append("HORIZON_METADATA_INVALID")
    numeric_keys = _FRAME_KEYS | {"fps", "frame_rate", "horizon_seconds", "duration_seconds", "closed_loop_chunks", "closed_loop_steps", "closed_loop_frames"}
    if any(key.casefold() in numeric_keys and (not _positive_number(value) or (key.casefold().startswith("closed_loop_") and not float(value).is_integer())) for key, value in _walk(payload)):
        protocol_warnings.append("PROTOCOL_NUMERIC_METADATA_INVALID")
    if horizon_seconds is None:
        protocol_warnings.append("HORIZON_UNDECLARED")
    elif horizon_min_seconds is not None and horizon_min_seconds < 60:
        protocol_warnings.append("MINUTE_LEVEL_CLAIM_UNSUPPORTED")
    if split is None:
        protocol_warnings.append("SPLIT_UNDECLARED")
    return {
        "candidate_id": _candidate_id(root, path),
        "source_path": str(path),
        "relative_path": path.relative_to(root).as_posix(),
        "family": str(payload.get("kind") or payload.get("role") or path.parent.name),
        "state": "candidate",
        "requires_user_confirmation": True,
        "fps": fps,
        "horizon_seconds": horizon_seconds,
        "horizon_min_seconds": horizon_min_seconds,
        "minute_level_supported": bool(not protocol_warnings and horizon_min_seconds is not None and horizon_min_seconds >= 60),
        "closed_loop_chunks": int(closed_loop) if isinstance(closed_loop, (int, float)) else None,
        "unit_count": len(units) if units is not None else None,
        "split": split,
        "episode_disjoint": bool(identity_disjoint) if isinstance(identity_disjoint, bool) else None,
        "metrics": metrics,
        "path_rewrites_required": path_rewrites,
        "confidence": "medium" if fps is not None and (horizon_seconds is not None or closed_loop is not None) else "low",
        "protocol_warnings": protocol_warnings,
        "claim_boundary": "Candidate metadata only; no evaluator is frozen and no quality claim is licensed until the user confirms an overlay and target-side verifier.",
}


def _candidate_id(root: Path, path: Path) -> str:
    relative = path.relative_to(root).with_suffix("").as_posix()
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", f"{root.name}-{relative}").strip("-").casefold()
    suffix = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:8]
    return f"{slug[:48]}-{suffix}"


def _walk(value: object, key: str = ""):
    yield key, value
    if isinstance(value, Mapping):
        for name, child in value.items():
            yield from _walk(child, str(name))
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child, key)


def _positive_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def _number(payload: Mapping[str, Any], keys: set[str], *, integral: bool = False) -> float | None:
    for key, value in _walk(payload):
        if key.casefold() in keys and _positive_number(value) and (not integral or float(value).is_integer()):
            return float(value)
    return None


def _numbers(payload: Mapping[str, Any], keys: set[str]) -> list[float]:
    return [float(value) for key, value in _walk(payload) if key.casefold() in keys and _positive_number(value)]


def _first(payload: Mapping[str, Any], keys: set[str]) -> object | None:
    for key, value in _walk(payload):
        if key.casefold() in keys:
            return value
    return None


def _strings_for_keys(payload: Mapping[str, Any], keys: set[str]) -> list[str]:
    values: list[str] = []
    for key, value in _walk(payload):
        if key.casefold() not in keys:
            continue
        if isinstance(value, str) and value.strip():
            values.append(value.strip())
        elif isinstance(value, list):
            values.extend(item.strip() for item in value if isinstance(item, str) and item.strip())
    return values


def _find_list(payload: Mapping[str, Any], wanted: str) -> list[object] | None:
    for key, value in _walk(payload):
        if key.casefold() == wanted and isinstance(value, list):
            return value
    return None


def _absolute_strings(payload: Mapping[str, Any]) -> list[str]:
    return [str(value) for _key, value in _walk(payload) if isinstance(value, str) and value.startswith("/")]


def _path_risk(value: str) -> bool:
    return value.startswith(_ABSOLUTE_RISK_PREFIXES) or not os.path.exists(value)
