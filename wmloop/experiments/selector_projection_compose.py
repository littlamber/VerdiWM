"""Compose independently calibrated diagnostic probes into one selector frame."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from wmloop.experiments._artifacts import canonical_json, write_bundle


class SelectorProjectionComposeError(ValueError):
    """Projection sources cannot be combined without changing their semantics."""


def compose_selector_projections(
    *,
    primary_projection: Path,
    primary_path_order: Sequence[str],
    extension_projections: Sequence[tuple[Path, Sequence[str]]],
    output_root: Path,
    archive_db: Path | None = None,
    cas_root: Path | None = None,
) -> dict[str, object]:
    primary_rows = _load_jsonl(Path(primary_projection))
    primary = _index(primary_rows)
    environments = sorted({environment for environment, _selector in primary})
    if len(environments) < 2:
        raise SelectorProjectionComposeError("PROJECTION_COMPOSE_ENVIRONMENTS_INVALID")
    expected = {(environment, selector) for environment in environments for selector in _SELECTORS}
    if set(primary) != expected:
        raise SelectorProjectionComposeError("PROJECTION_COMPOSE_PRIMARY_FRAME_INVALID")

    sources: list[tuple[str, tuple[str, ...], dict[tuple[str, str], Mapping[str, Any]]]] = [
        (str(Path(primary_projection).resolve()), _validate_path_order(primary_path_order), primary)
    ]
    for path, raw_order in extension_projections:
        indexed = _index(_load_jsonl(Path(path)))
        if set(indexed) != expected:
            raise SelectorProjectionComposeError("PROJECTION_COMPOSE_EXTENSION_FRAME_INVALID")
        sources.append((str(Path(path).resolve()), _validate_path_order(raw_order), indexed))

    rows: list[dict[str, object]] = []
    for environment in environments:
        for selector in _SELECTORS:
            if selector != "irg":
                rows.append(dict(primary[(environment, selector)]))
                continue
            feature_names: list[str] = []
            features: list[float] = []
            campaign_ids: list[str] = []
            for _source_path, path_order, indexed in sources:
                row = indexed[(environment, "irg")]
                names, values = _explicit_irg_features(row, path_order)
                feature_names.extend(names)
                features.extend(values)
                campaign_ids.append(str(row.get("campaign_id") or "unknown"))
            if len(set(feature_names)) != len(feature_names):
                raise SelectorProjectionComposeError("PROJECTION_COMPOSE_FEATURE_COLLISION")
            rows.append(
                {
                    "schema_version": 1,
                    "artifact_type": "verdiwm-selector-input-projection",
                    "campaign_id": "+".join(campaign_ids),
                    "environment": environment,
                    "selector": "irg",
                    "feature_names": feature_names,
                    "features": features,
                    "uses_intervention_response": True,
                    "uses_uncertainty": True,
                    "aligned_geometry": True,
                    "composed_probe_count": len(sources),
                }
            )

    report = {
        "schema_version": 1,
        "artifact_type": "verdiwm-selector-projection-stack",
        "state": "ready",
        "environment_count": len(environments),
        "selector_row_count": len(rows),
        "probe_source_count": len(sources),
        "probe_sources": [
            {"projection_path": path, "path_order": list(order)}
            for path, order, _indexed in sources
        ],
        "claim_boundary": "This artifact composes diagnostic coordinates only. It does not change source measurements, locality admission, effect labels, or verdict metrics.",
    }
    destination = Path(output_root).resolve()
    return write_bundle(
        output_root=destination,
        files={
            "projection-stack.json": canonical_json(report),
            "selector-input-projections.jsonl": b"".join(canonical_json(row) for row in rows),
        },
        manifest_fields={
            "artifact_type": "verdiwm-selector-projection-stack-manifest",
            "state": "ready",
            "environment_count": len(environments),
            "selector_row_count": len(rows),
            "probe_source_count": len(sources),
            "report_path": str(destination / "projection-stack.json"),
        },
        archive_db=archive_db,
        cas_root=cas_root,
    )


_SELECTORS = ("environment_label", "static_probe", "raw_response", "irg")


def _explicit_irg_features(row: Mapping[str, Any], path_order: Sequence[str]) -> tuple[list[str], list[float]]:
    raw_names = row.get("feature_names")
    raw_features = row.get("features")
    if (
        not isinstance(raw_names, list)
        or not isinstance(raw_features, list)
        or len(raw_names) != len(raw_features)
    ):
        raise SelectorProjectionComposeError("PROJECTION_COMPOSE_IRG_ROW_INVALID")
    names: list[str] = []
    features: list[float] = []
    for raw_name, raw_value in zip(raw_names, raw_features, strict=True):
        name = str(raw_name)
        if name.startswith("response_coordinate:") or name.startswith("covariance_diagonal:"):
            prefix, suffix = name.rsplit(":", 1)
            if not suffix.isdigit():
                raise SelectorProjectionComposeError("PROJECTION_COMPOSE_IRG_LAYOUT_INVALID")
            coordinate = int(suffix)
            path = path_order[coordinate % len(path_order)]
            outcome = coordinate // len(path_order)
            name = f"{prefix}:{outcome}:{path}"
        elif name.startswith("locality:") or name.startswith("path_supported:"):
            if name.split(":", 1)[1] not in path_order:
                raise SelectorProjectionComposeError("PROJECTION_COMPOSE_IRG_LAYOUT_INVALID")
        else:
            raise SelectorProjectionComposeError("PROJECTION_COMPOSE_IRG_FEATURE_UNKNOWN")
        names.append(name)
        features.append(float(raw_value))
    return names, features


def _validate_path_order(values: Sequence[str]) -> tuple[str, ...]:
    result = tuple(str(value) for value in values)
    if not result or len(set(result)) != len(result) or any(not value for value in result):
        raise SelectorProjectionComposeError("PROJECTION_COMPOSE_PATH_ORDER_INVALID")
    return result


def _index(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], Mapping[str, Any]]:
    result: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in rows:
        key = (str(row.get("environment") or ""), str(row.get("selector") or ""))
        if key[1] not in _SELECTORS or not key[0] or key in result:
            raise SelectorProjectionComposeError("PROJECTION_COMPOSE_ROW_INVALID")
        result[key] = row
    return result


def _load_jsonl(path: Path) -> list[Mapping[str, Any]]:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise SelectorProjectionComposeError(f"PROJECTION_COMPOSE_JSONL_INVALID:{path}") from exc
    if any(not isinstance(row, Mapping) for row in rows):
        raise SelectorProjectionComposeError(f"PROJECTION_COMPOSE_JSONL_INVALID:{path}")
    return rows
