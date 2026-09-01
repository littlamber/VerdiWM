"""Discover unified IRG assets and build a compact atlas projection for UIs.

The atlas is a read-only presentation projection: source IRG assets remain
authoritative. Discovery scans a state root for ``verdiwm-unified-irg-asset``
documents, either directly or through a ``verdiwm-unified-irg-asset-index``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

_MAX_SCAN_FILES = 5000
_MAX_ASSETS = 64


class AtlasError(ValueError):
    """The atlas input or discovery contract is invalid."""


def discover_irg_assets(root: Path) -> list[Path]:
    """Return asset document paths under root, index-referenced first."""

    base = Path(root).expanduser().resolve()
    if not base.is_dir() or base.is_symlink():
        raise AtlasError("ATLAS_ROOT_INVALID")
    indexed: list[Path] = []
    referenced: set[Path] = set()
    standalone: list[Path] = []
    scanned = 0
    for path in sorted(base.rglob("*.json")):
        if scanned >= _MAX_SCAN_FILES:
            break
        if path.is_symlink() or not path.is_file():
            continue
        scanned += 1
        try:
            head = path.read_text(encoding="utf-8", errors="strict")[:4096]
        except (OSError, UnicodeDecodeError):
            continue
        if '"verdiwm-unified-irg-asset-index"' in head:
            try:
                index = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            asset_paths = index.get("asset_paths")
            if not isinstance(asset_paths, Mapping):
                continue
            for rel in sorted(asset_paths.values()):
                if not isinstance(rel, str):
                    continue
                asset = (path.parent / rel).resolve()
                if asset.is_file() and not asset.is_symlink() and asset.is_relative_to(base):
                    indexed.append(asset)
                    referenced.add(asset)
        elif '"verdiwm-unified-irg-asset"' in head:
            resolved = path.resolve()
            if resolved not in referenced:
                standalone.append(resolved)
    assets = []
    for path in indexed + standalone:
        if path not in assets:
            assets.append(path)
    return assets[:_MAX_ASSETS]


def build_atlas(root: Path) -> dict[str, Any]:
    """Build the atlas projection consumed by the workbench UI."""

    base = Path(root).expanduser().resolve()
    asset_paths = discover_irg_assets(base)
    if not asset_paths:
        raise AtlasError("ATLAS_ASSETS_NOT_FOUND")
    environments: list[dict[str, Any]] = []
    probe_names: list[str] = []
    outcome_names: list[str] = []
    outcome_weights: list[float] = []
    threshold: float | None = None
    for path in asset_paths:
        try:
            asset = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if asset.get("artifact_type") != "verdiwm-unified-irg-asset":
            continue
        env = _project_asset(asset, source=str(path))
        if env is None:
            continue
        environments.append(env)
        for name in asset.get("probe_path_names") or []:
            if name not in probe_names:
                probe_names.append(name)
        if not outcome_names and isinstance(asset.get("outcome_names"), list):
            outcome_names = list(asset["outcome_names"])
            weights = asset.get("outcome_weights")
            outcome_weights = [float(w) for w in weights] if isinstance(weights, list) else [1.0] * len(outcome_names)
        if threshold is None and isinstance(asset.get("locality_threshold"), (int, float)):
            threshold = float(asset["locality_threshold"])
    if not environments:
        raise AtlasError("ATLAS_ASSETS_NOT_FOUND")
    environments.sort(key=lambda e: (e["backbone_family"], e["environment"]))
    return {
        "artifact_type": "verdiwm-atlas",
        "schema_version": 1,
        "state": "ready",
        "input_root": str(base),
        "environment_count": len(environments),
        "probe_path_names": probe_names,
        "outcome_names": outcome_names,
        "outcome_weights": outcome_weights,
        "locality_threshold": threshold,
        "claim_boundary": (
            "This is a presentation projection of measured local routing charts. "
            "Unsupported paths remain masked, and cross-group covariance gaps still "
            "force transfer abstention at the source assets."
        ),
        "environments": environments,
    }


def _project_asset(asset: Mapping[str, Any], *, source: str) -> dict[str, Any] | None:
    probe_names = asset.get("probe_path_names")
    coords = asset.get("response_coordinate")
    if not isinstance(probe_names, list) or not isinstance(coords, list):
        return None
    outcomes = asset.get("outcome_names") or []
    weights = asset.get("outcome_weights") or [1.0] * len(outcomes)
    names = asset.get("coordinate_names") or []
    raw_coords = asset.get("raw_response_coordinate") or coords
    repair = asset.get("repair_metric") or [None] * len(probe_names)
    raw_repair = asset.get("raw_repair_metric") or repair
    support = asset.get("support_mask") or [True] * len(probe_names)
    locality = asset.get("locality_residuals") or {}
    total_weight = sum(float(w) for w in weights) or 1.0
    probes = []
    for p, probe in enumerate(probe_names):
        values: dict[str, float] = {}
        raw_values: dict[str, float] = {}
        for o, outcome in enumerate(outcomes):
            key = f"{outcome}:{probe}"
            index = names.index(key) if key in names else o * len(probe_names) + p
            if index < len(coords) and isinstance(coords[index], (int, float)):
                values[str(outcome)] = float(coords[index])
            if index < len(raw_coords) and isinstance(raw_coords[index], (int, float)):
                raw_values[str(outcome)] = float(raw_coords[index])
        weighted = (
            sum(values.get(str(outcome), 0.0) * float(weights[o]) for o, outcome in enumerate(outcomes))
            / total_weight
        )
        probes.append(
            {
                "probe": probe,
                "supported": bool(support[p]) if p < len(support) else True,
                "repair_metric": repair[p] if p < len(repair) else None,
                "raw_repair_metric": raw_repair[p] if p < len(raw_repair) else None,
                "locality_residual": locality.get(probe) if isinstance(locality, Mapping) else None,
                "weighted_response": weighted,
                "responses": values,
                "raw_responses": raw_values,
            }
        )
    return {
        "environment": str(asset.get("environment") or Path(source).stem),
        "asset_id": asset.get("asset_id"),
        "backbone_family": str(asset.get("backbone_family") or "unknown"),
        "routing_state": asset.get("routing_state"),
        "transfer_state": asset.get("transfer_state"),
        "transfer_blockers": list(asset.get("transfer_blockers") or []),
        "supported_probe_path_count": asset.get("supported_probe_path_count"),
        "seeds": list(asset.get("seeds") or []),
        "checkpoint_steps": list(asset.get("checkpoint_steps") or []),
        "source": source,
        "probes": probes,
    }
