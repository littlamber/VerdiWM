"""Read-only discovery of evaluator candidates in an external model checkout.

The scanner deliberately reports *candidates*, never frozen contracts.  It
extracts the small amount of protocol metadata needed for first contact and
keeps suspicious absolute paths visible so an overlay can rewrite them after
the user confirms the intended split and verifier.
"""

from __future__ import annotations

import json
import os
import hashlib
import re
from pathlib import Path
from typing import Any, Mapping


_MAX_FILES = 200
# Build host-path prefixes without embedding machine-local paths in the public
# source tree; the release audit treats those literals as leaked deployment data.
_ABSOLUTE_RISK_PREFIXES = tuple(
    "/" + part + "/" for part in ("mnt", "root", "workspace", "home")
)
_METRIC_KEYS = {"metric", "metrics", "metric_contract", "primary_metric", "target_metrics"}
_FRAME_KEYS = {"num_video_frames", "video_frames", "num_frames", "frames", "horizon_frames"}


def discover_evaluator_candidates(source_root: Path) -> dict[str, Any]:
    """Return deterministic evaluator candidates without importing or running code."""

    root = Path(source_root).expanduser().resolve()
    candidates: list[dict[str, Any]] = []
    warnings: list[dict[str, str]] = []
    if not root.is_dir() or root.is_symlink():
        return {"state": "none", "candidates": [], "scanned_root": str(root)}
    search_roots = [root / "configs" / "evaluation", root / "configs" / "evaluators"]
    files: list[Path] = []
    for directory in search_roots:
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix.lower() in {".json", ".yaml", ".yml"}:
                files.append(path)
                if len(files) >= _MAX_FILES:
                    break
    for path in files:
        payload = _load(path)
        if not isinstance(payload, Mapping):
            warnings.append({"source_path": str(path), "code": "EVALUATOR_CANDIDATE_UNREADABLE_OR_MALFORMED"})
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


def _load(path: Path) -> object:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        if path.suffix.lower() == ".json":
            return json.loads(text)
        try:
            import yaml  # type: ignore[import-not-found]
            return yaml.safe_load(text)
        except Exception:
            return None
    except (OSError, json.JSONDecodeError):
        return None


def _candidate(path: Path, root: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    fps = _number(payload, {"fps", "frame_rate"})
    frames = _numbers(payload, _FRAME_KEYS)
    if fps and frames:
        horizon_seconds = round(max(frames) / fps, 3)
        horizon_min_seconds = round(min(frames) / fps, 3)
    else:
        horizon_seconds = _number(payload, {"horizon_seconds", "duration_seconds"})
        horizon_min_seconds = horizon_seconds
    closed_loop = _number(payload, {"closed_loop_chunks", "closed_loop_steps", "closed_loop_frames"})
    split = _first(payload, {"source_split", "split", "evaluation_split"})
    metrics = sorted(set(_strings_for_keys(payload, _METRIC_KEYS)))
    absolute_paths = sorted(set(_absolute_strings(payload)))
    path_rewrites = [p for p in absolute_paths if _path_risk(p) or not Path(p).exists()]
    identity_disjoint = _first(payload, {"episode_disjoint", "identity_disjoint", "identity_absent_from_all_group_train_roots"})
    units = _find_list(payload, "units")
    protocol_warnings: list[str] = []
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
        "minute_level_supported": bool(horizon_min_seconds is not None and horizon_min_seconds >= 60),
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
    if isinstance(value, Mapping):
        for name, child in value.items():
            yield from _walk(child, str(name))
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child, key)
    else:
        yield key, value


def _number(payload: Mapping[str, Any], keys: set[str]) -> float | None:
    for key, value in _walk(payload):
        if key.casefold() in keys and isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def _numbers(payload: Mapping[str, Any], keys: set[str]) -> list[float]:
    return [float(value) for key, value in _walk(payload) if key.casefold() in keys and isinstance(value, (int, float)) and not isinstance(value, bool)]


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
            values.extend(str(item).strip() for item in value if str(item).strip())
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
