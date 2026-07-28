#!/usr/bin/env python3
"""Materialize one canonical IRG asset per ACWM-Phys environment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from wmloop.experiments._artifacts import canonical_json, write_bundle
from wmloop.geometry.assets import IRGChartSource, compose_irg_asset, validate_irg_asset
from wmloop.geometry.types import GeometryValidationError


REPO_ROOT = Path(__file__).resolve().parents[2]


class UnifiedIRGExportError(ValueError):
    """The frozen response-chart stack cannot form an auditable asset."""


def export_unified_irg_assets(
    *,
    projection_stack_root: Path,
    output_root: Path,
    backbone_family: str = "acwm_phys",
    capability_class: str = "latent_dit_action_conditioned",
    locality_threshold: float = 0.5,
    baseline_atol: float = 1e-4,
    ridge: float = 1e-6,
    archive_db: Path | None = None,
    cas_root: Path | None = None,
) -> dict[str, object]:
    stack_root = Path(projection_stack_root).resolve(strict=True)
    stack = _load_json(stack_root / "projection-stack.json")
    source_specs = _load_source_specs(stack)
    environments = _environment_order(source_specs[0])
    assets: list[dict[str, object]] = []
    files: dict[str, bytes] = {}
    for environment in environments:
        sources, backbone_ref = _environment_sources(source_specs, environment)
        asset = compose_irg_asset(
            asset_id=f"acwm-phys:{environment}:active-r20",
            environment=environment,
            backbone_family=backbone_family,
            capability_class=capability_class,
            backbone_instance_ref=backbone_ref,
            sources=sources,
            locality_threshold=locality_threshold,
            baseline_atol=baseline_atol,
            ridge=ridge,
        )
        validate_irg_asset(asset)
        assets.append(asset)
        files[f"assets/{environment}.json"] = canonical_json(asset)

    path_order = assets[0]["probe_path_names"]
    if any(asset["probe_path_names"] != path_order for asset in assets):
        raise UnifiedIRGExportError("UNIFIED_IRG_PATH_ORDER_MISMATCH")
    summary_rows = [
        {
            "environment": asset["environment"],
            "outcome_count": asset["dimensions"]["outcome_count"],
            "probe_path_count": asset["dimensions"]["probe_path_count"],
            "supported_probe_path_count": asset["supported_probe_path_count"],
            "baseline_group_count": asset["covariance_contract"]["joint_baseline_group_count"],
            "routing_state": asset["routing_state"],
            "transfer_state": asset["transfer_state"],
            "transfer_blockers": ";".join(asset["transfer_blockers"]),
        }
        for asset in assets
    ]
    index = {
        "schema_version": 1,
        "artifact_type": "verdiwm-unified-irg-asset-index",
        "state": "ready",
        "selector_revision": "active-r20",
        "environment_count": len(assets),
        "probe_source_count": len(source_specs),
        "probe_path_count": len(path_order),
        "probe_path_names": path_order,
        "outcome_names": assets[0]["outcome_names"],
        "outcome_weights": assets[0]["outcome_weights"],
        "routing_ready_environment_count": sum(asset["routing_state"] == "ready" for asset in assets),
        "transfer_ready_environment_count": sum(asset["transfer_state"] == "ready" for asset in assets),
        "asset_paths": {
            str(asset["environment"]): f"assets/{asset['environment']}.json"
            for asset in assets
        },
        "source_projection_stack": _portable_path(stack_root / "projection-stack.json"),
        "claim_boundary": (
            "These assets unify measured local response charts for routing. Baseline-incompatible source "
            "groups retain explicit covariance gaps, so the current ACWM assets abstain from transfer "
            "licensing until a jointly paired probe campaign is measured."
        ),
    }
    files["index.json"] = canonical_json(index)
    files["tables/asset-summary.csv"] = _csv(summary_rows).encode("utf-8")
    files["README.md"] = _readme(index, summary_rows).encode("utf-8")
    return write_bundle(
        output_root=Path(output_root),
        files=files,
        manifest_fields={
            "artifact_type": "verdiwm-unified-irg-asset-bundle",
            "state": "ready",
            "environment_count": len(assets),
            "probe_source_count": len(source_specs),
            "probe_path_count": len(path_order),
            "routing_ready_environment_count": index["routing_ready_environment_count"],
            "transfer_ready_environment_count": index["transfer_ready_environment_count"],
            "index_path": "index.json",
        },
        archive_db=archive_db,
        cas_root=cas_root,
    )


def _load_source_specs(stack: Mapping[str, Any]) -> tuple[dict[str, object], ...]:
    rows = stack.get("probe_sources")
    if not isinstance(rows, list) or not rows:
        raise UnifiedIRGExportError("UNIFIED_IRG_PROBE_SOURCES_INVALID")
    result = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise UnifiedIRGExportError("UNIFIED_IRG_PROBE_SOURCE_INVALID")
        projection = Path(str(row.get("projection_path") or "")).resolve(strict=True)
        root = projection.parent
        campaign = _load_json(root / "input-campaign.json")
        atlas = _load_json(root / "fingerprint-atlas.json")
        path_order = tuple(str(value) for value in row.get("path_order", ()))
        if not path_order or len(set(path_order)) != len(path_order):
            raise UnifiedIRGExportError("UNIFIED_IRG_PATH_ORDER_INVALID")
        result.append(
            {
                "root": root,
                "campaign": campaign,
                "atlas": atlas,
                "path_order": path_order,
                "campaign_path": root / "input-campaign.json",
            }
        )
    return tuple(result)


def _environment_order(source: Mapping[str, object]) -> tuple[str, ...]:
    campaign = source["campaign"]
    environments = campaign.get("environments") if isinstance(campaign, Mapping) else None
    if not isinstance(environments, Mapping) or not environments:
        raise UnifiedIRGExportError("UNIFIED_IRG_ENVIRONMENTS_INVALID")
    return tuple(str(value) for value in environments)


def _environment_sources(
    specs: Sequence[Mapping[str, object]], environment: str
) -> tuple[tuple[IRGChartSource, ...], str]:
    sources: list[IRGChartSource] = []
    backbone_refs: set[str] = set()
    checkpoint_steps: set[int] = set()
    for spec in specs:
        root = Path(spec["root"])
        campaign = spec["campaign"]
        atlas = spec["atlas"]
        path_order = tuple(spec["path_order"])
        if not isinstance(campaign, Mapping) or not isinstance(atlas, Mapping):
            raise UnifiedIRGExportError("UNIFIED_IRG_SOURCE_PAYLOAD_INVALID")
        environment_rows = atlas.get("environments")
        if not isinstance(environment_rows, Mapping) or environment not in environment_rows:
            raise UnifiedIRGExportError(f"UNIFIED_IRG_ENVIRONMENT_MISSING:{environment}")
        atlas_environment = environment_rows[environment]
        if not isinstance(atlas_environment, Mapping):
            raise UnifiedIRGExportError("UNIFIED_IRG_ENVIRONMENT_INVALID")
        chart_path = root / "charts" / f"{environment}.json"
        chart = _load_json(chart_path)
        if tuple(str(value) for value in chart.get("intervention_names", ())) != path_order:
            raise UnifiedIRGExportError(f"UNIFIED_IRG_CHART_PATH_ORDER_MISMATCH:{environment}")
        measurement_path = _resolve_repo_path(str(atlas_environment.get("measurements_ref") or ""))
        measurements = _load_jsonl(measurement_path)
        repeat_jacobians, baseline_vectors = _repeat_jacobians(
            campaign=campaign,
            atlas=atlas,
            chart=chart,
            path_order=path_order,
            environment=environment,
            measurements=measurements,
        )
        outcomes = tuple(str(row["name"]) for row in campaign["goal_oriented_outcomes"])
        weights = tuple(float(row["weight"]) for row in campaign["goal_oriented_outcomes"])
        checkpoint_step = int(atlas_environment["checkpoint_step"])
        checkpoint_steps.add(checkpoint_step)
        backbone_ref = str(campaign.get("backbone_instance_ref") or "")
        backbone_refs.add(backbone_ref)
        campaign_path = Path(spec["campaign_path"])
        sources.append(
            IRGChartSource(
                source_id=root.name,
                campaign_id=str(campaign["campaign_id"]),
                path_names=path_order,
                outcome_names=outcomes,
                outcome_weights=weights,
                seeds=tuple(int(value) for value in campaign["seeds"]),
                jacobian=_matrix(chart["jacobian"]),
                covariance=_matrix(chart["covariance"]),
                locality_residuals={
                    str(key): float(value)
                    for key, value in chart["locality_residuals"].items()
                },
                repeat_jacobians=repeat_jacobians,
                baseline_vectors=baseline_vectors,
                checkpoint_step=checkpoint_step,
                provenance={
                    "chart_ref": _portable_path(chart_path),
                    "chart_sha256": _sha256(chart_path),
                    "measurement_ref": _portable_path(measurement_path),
                    "measurement_sha256": _sha256(measurement_path),
                    "campaign_ref": _portable_path(campaign_path),
                    "campaign_sha256": _sha256(campaign_path),
                    "protocol": str(atlas.get("protocol") or "unknown"),
                },
            )
        )
    if len(backbone_refs) != 1 or "" in backbone_refs:
        raise UnifiedIRGExportError("UNIFIED_IRG_BACKBONE_FRAME_MISMATCH")
    if len(checkpoint_steps) != 1:
        raise UnifiedIRGExportError(f"UNIFIED_IRG_CHECKPOINT_STEP_MISMATCH:{environment}")
    return tuple(sources), next(iter(backbone_refs))


def _repeat_jacobians(
    *,
    campaign: Mapping[str, Any],
    atlas: Mapping[str, Any],
    chart: Mapping[str, Any],
    path_order: tuple[str, ...],
    environment: str,
    measurements: Sequence[Mapping[str, Any]],
) -> tuple[tuple[tuple[tuple[float, ...], ...], ...], tuple[tuple[float, ...], ...]]:
    seeds = tuple(int(value) for value in campaign["seeds"])
    by_pair = {
        (float(row["dose"]), int(row["seed"])): _goal_vector(campaign, row["metrics"])
        for row in measurements
        if str(row.get("environment")) == environment
    }
    baseline = tuple(by_pair[(0.0, seed)] for seed in seeds)
    per_path: dict[str, tuple[tuple[float, ...], ...]] = {}
    policy_ref = atlas.get("path_calibration_policy")
    if policy_ref:
        policy = _load_json(_resolve_repo_path(str(policy_ref)))
        policy_rows = {
            str(row["path_name"]): str(row["source_dose_sign"])
            for row in policy["paths"]
        }
        if set(policy_rows) != set(path_order):
            raise UnifiedIRGExportError("UNIFIED_IRG_PATH_POLICY_MISMATCH")
        source_doses = tuple(float(value) for value in campaign["probe"]["doses"] if float(value) != 0.0)
        for path in path_order:
            sign = policy_rows[path]
            selected = sorted(
                (dose for dose in source_doses if (dose > 0.0) is (sign == "positive")),
                key=abs,
            )
            if not selected:
                raise UnifiedIRGExportError("UNIFIED_IRG_PATH_DOSE_MISSING")
            dose = selected[0]
            semantic_dose = abs(dose)
            per_path[path] = tuple(
                tuple((by_pair[(dose, seed)][index] - baseline_row[index]) / semantic_dose for index in range(len(baseline_row)))
                for seed, baseline_row in zip(seeds, baseline, strict=True)
            )
    else:
        probe_id = str(campaign["probe"]["probe_id"])
        if path_order != (probe_id,):
            raise UnifiedIRGExportError("UNIFIED_IRG_SOURCE_PATH_UNSUPPORTED")
        doses = tuple(float(value) for value in campaign["probe"]["doses"] if float(value) != 0.0)
        positive = sorted(dose for dose in doses if dose > 0.0)
        negative = sorted((dose for dose in doses if dose < 0.0), key=abs)
        if positive and negative and math.isclose(positive[0], abs(negative[0]), rel_tol=1e-9, abs_tol=1e-12):
            plus, minus = positive[0], negative[0]
            per_path[probe_id] = tuple(
                tuple((by_pair[(plus, seed)][index] - by_pair[(minus, seed)][index]) / (plus - minus) for index in range(len(baseline_row)))
                for seed, baseline_row in zip(seeds, baseline, strict=True)
            )
        else:
            dose = positive[0] if positive else negative[0]
            per_path[probe_id] = tuple(
                tuple((by_pair[(dose, seed)][index] - baseline_row[index]) / dose for index in range(len(baseline_row)))
                for seed, baseline_row in zip(seeds, baseline, strict=True)
            )
    repeats = tuple(
        tuple(
            tuple(per_path[path][repeat][outcome] for path in path_order)
            for outcome in range(len(baseline[0]))
        )
        for repeat in range(len(seeds))
    )
    return repeats, baseline


def _goal_vector(campaign: Mapping[str, Any], metrics: Mapping[str, Any]) -> tuple[float, ...]:
    return tuple(
        float(metrics[str(row["source_metric"])]) * float(row["sign"])
        for row in campaign["goal_oriented_outcomes"]
    )


def _matrix(values: object) -> tuple[tuple[float, ...], ...]:
    if not isinstance(values, list) or not values or any(not isinstance(row, list) for row in values):
        raise UnifiedIRGExportError("UNIFIED_IRG_MATRIX_INVALID")
    return tuple(tuple(float(value) for value in row) for row in values)


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise UnifiedIRGExportError(f"UNIFIED_IRG_JSON_INVALID:{path}") from exc
    if not isinstance(payload, Mapping):
        raise UnifiedIRGExportError(f"UNIFIED_IRG_JSON_INVALID:{path}")
    return payload


def _load_jsonl(path: Path) -> list[Mapping[str, Any]]:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise UnifiedIRGExportError(f"UNIFIED_IRG_JSONL_INVALID:{path}") from exc
    if not rows or any(not isinstance(row, Mapping) for row in rows):
        raise UnifiedIRGExportError(f"UNIFIED_IRG_JSONL_INVALID:{path}")
    return rows


def _resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path.resolve(strict=True) if path.is_absolute() else (REPO_ROOT / path).resolve(strict=True)


def _portable_path(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError as exc:
        raise UnifiedIRGExportError(f"UNIFIED_IRG_SOURCE_OUTSIDE_REPO:{resolved}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _csv(rows: Sequence[Mapping[str, object]]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _readme(index: Mapping[str, Any], rows: Sequence[Mapping[str, object]]) -> str:
    abstained = sum(row["transfer_state"] == "abstain" for row in rows)
    return f"""# ACWM-Phys Unified IRG Assets

This bundle materializes one canonical IRG asset for each ACWM-Phys environment.
Every asset stores raw and locality-masked `J_X`, `G_X`, and `r_X`, paired-seed
response covariance, support masks, and source hashes.

- Environments: {index['environment_count']}
- Probe paths: {index['probe_path_count']}
- Routing-ready environments: {index['routing_ready_environment_count']}
- Transfer-abstaining environments: {abstained}

The current source campaigns share outcomes, weights, seeds, and checkpoints,
but use more than one zero-dose baseline frame. Covariance is measured only
within compatible groups; cross-group entries are explicitly unobserved and
zero-filled. These assets support audited routing, not a cross-backbone transfer
claim.
"""


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projection-stack-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--backbone-family", default="acwm_phys")
    parser.add_argument("--capability-class", default="latent_dit_action_conditioned")
    parser.add_argument("--locality-threshold", type=float, default=0.5)
    parser.add_argument("--baseline-atol", type=float, default=1e-4)
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--archive-db", type=Path)
    parser.add_argument("--cas-root", type=Path)
    args = parser.parse_args(argv)
    report = export_unified_irg_assets(
        projection_stack_root=args.projection_stack_root,
        output_root=args.output_root,
        backbone_family=args.backbone_family,
        capability_class=args.capability_class,
        locality_threshold=args.locality_threshold,
        baseline_atol=args.baseline_atol,
        ridge=args.ridge,
        archive_db=args.archive_db,
        cas_root=args.cas_root,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
