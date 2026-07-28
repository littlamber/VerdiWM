"""Canonical, provenance-bearing assets for composed IRG charts."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

from wmloop.geometry.types import GeometryValidationError


Vector = tuple[float, ...]
Matrix = tuple[Vector, ...]


@dataclass(frozen=True)
class IRGChartSource:
    """One measured source chart and the repeats needed to audit composition."""

    source_id: str
    campaign_id: str
    path_names: tuple[str, ...]
    outcome_names: tuple[str, ...]
    outcome_weights: Vector
    seeds: tuple[int, ...]
    jacobian: Matrix
    covariance: Matrix
    locality_residuals: Mapping[str, float]
    repeat_jacobians: tuple[Matrix, ...]
    baseline_vectors: tuple[Vector, ...]
    checkpoint_step: int
    provenance: Mapping[str, object]


def compose_irg_asset(
    *,
    asset_id: str,
    environment: str,
    backbone_family: str,
    capability_class: str,
    backbone_instance_ref: str,
    sources: Sequence[IRGChartSource],
    locality_threshold: float = 0.5,
    baseline_atol: float = 1e-4,
    ridge: float = 1e-6,
) -> dict[str, object]:
    """Compose source charts without inventing cross-frame covariance.

    Response covariance is estimated jointly only for sources whose paired
    zero-dose baselines agree within ``baseline_atol``. Cross-group entries are
    represented as zero in the stored matrix and are explicitly declared
    unobserved, which forces transfer to abstain while preserving routing use.
    """

    if not asset_id or not environment or not backbone_family or not capability_class:
        raise GeometryValidationError("IRG_ASSET_IDENTITY_INVALID")
    if not backbone_instance_ref or not sources:
        raise GeometryValidationError("IRG_ASSET_SOURCE_MISSING")
    threshold = _finite(locality_threshold, "IRG_ASSET_LOCALITY_THRESHOLD_INVALID")
    tolerance = _finite(baseline_atol, "IRG_ASSET_BASELINE_TOLERANCE_INVALID")
    regularizer = _finite(ridge, "IRG_ASSET_RIDGE_INVALID")
    if threshold < 0.0 or tolerance < 0.0 or regularizer < 0.0:
        raise GeometryValidationError("IRG_ASSET_NUMERIC_POLICY_INVALID")

    source_rows = tuple(sources)
    outcomes = source_rows[0].outcome_names
    weights = source_rows[0].outcome_weights
    seeds = source_rows[0].seeds
    if not outcomes or not weights or not seeds:
        raise GeometryValidationError("IRG_ASSET_FRAME_EMPTY")
    if len(outcomes) != len(weights) or any(value <= 0.0 for value in weights):
        raise GeometryValidationError("IRG_ASSET_OUTCOME_FRAME_INVALID")
    if len(set(seeds)) != len(seeds):
        raise GeometryValidationError("IRG_ASSET_SEED_FRAME_INVALID")

    paths: list[str] = []
    source_path_indices: list[tuple[int, ...]] = []
    for source in source_rows:
        _validate_source(source, outcomes=outcomes, weights=weights, seeds=seeds)
        indices = tuple(range(len(paths), len(paths) + len(source.path_names)))
        source_path_indices.append(indices)
        paths.extend(source.path_names)
    if len(set(paths)) != len(paths):
        raise GeometryValidationError("IRG_ASSET_PATH_DUPLICATED")

    raw_jacobian = tuple(
        tuple(value for source in source_rows for value in source.jacobian[outcome])
        for outcome in range(len(outcomes))
    )
    locality = {
        path: float(source.locality_residuals[path])
        for source in source_rows
        for path in source.path_names
    }
    support_mask = tuple(locality[path] <= threshold for path in paths)
    effective_jacobian = tuple(
        tuple(value if support_mask[column] else 0.0 for column, value in enumerate(row))
        for row in raw_jacobian
    )
    raw_coordinate = _response_coordinate(raw_jacobian, weights)
    coordinate_mask = tuple(mask for _outcome in outcomes for mask in support_mask)
    effective_coordinate = tuple(
        value if coordinate_mask[index] else 0.0
        for index, value in enumerate(raw_coordinate)
    )
    raw_metric = _repair_metric(raw_jacobian, weights, regularizer, tuple(True for _ in paths))
    effective_metric = _repair_metric(effective_jacobian, weights, regularizer, support_mask)

    groups = _baseline_groups(source_rows, tolerance)
    raw_covariance = [[0.0 for _ in raw_coordinate] for _ in raw_coordinate]
    observed_blocks: list[dict[str, object]] = []
    for group_index, source_indices in enumerate(groups):
        path_indices = tuple(
            index
            for source_index in source_indices
            for index in source_path_indices[source_index]
        )
        coordinate_indices = tuple(
            outcome * len(paths) + path
            for outcome in range(len(outcomes))
            for path in path_indices
        )
        repeats = tuple(
            tuple(
                math.sqrt(weights[outcome])
                * _repeat_path_value(
                    source_rows,
                    source_indices,
                    source_path_indices,
                    repeat,
                    outcome,
                    path,
                )
                for outcome in range(len(outcomes))
                for path in path_indices
            )
            for repeat in range(len(seeds))
        )
        covariance = _sample_covariance(repeats)
        for local_i, global_i in enumerate(coordinate_indices):
            for local_j, global_j in enumerate(coordinate_indices):
                raw_covariance[global_i][global_j] = covariance[local_i][local_j]
        observed_blocks.append(
            {
                "group_id": f"baseline-group-{group_index + 1}",
                "source_ids": [source_rows[index].source_id for index in source_indices],
                "path_names": [paths[index] for index in path_indices],
                "coordinate_indices": list(coordinate_indices),
                "repeat_count": len(seeds),
                "maximum_baseline_deviation": _maximum_group_deviation(
                    source_rows, source_indices
                ),
            }
        )
    effective_covariance = tuple(
        tuple(
            value if coordinate_mask[row] and coordinate_mask[column] else 0.0
            for column, value in enumerate(values)
        )
        for row, values in enumerate(raw_covariance)
    )
    raw_covariance_tuple = tuple(tuple(row) for row in raw_covariance)
    transfer_blockers = (
        [] if len(groups) == 1 else ["joint_baseline_frame_mismatch"]
    )
    coordinate_names = [
        f"{outcome}:{path}" for outcome in outcomes for path in paths
    ]
    checkpoint_steps = sorted({source.checkpoint_step for source in source_rows})
    return {
        "schema_version": 1,
        "artifact_type": "verdiwm-unified-irg-asset",
        "asset_id": asset_id,
        "environment": environment,
        "backbone_family": backbone_family,
        "capability_class": capability_class,
        "backbone_instance_ref": backbone_instance_ref,
        "goal_schema": "acwm_phys_goal_oriented_pixel_metrics_v1",
        "outcome_names": list(outcomes),
        "outcome_weights": list(weights),
        "probe_path_names": paths,
        "seeds": list(seeds),
        "checkpoint_steps": checkpoint_steps,
        "dimensions": {
            "outcome_count": len(outcomes),
            "probe_path_count": len(paths),
            "response_coordinate_count": len(raw_coordinate),
        },
        "symbol_table": {
            "J_X": "jacobian",
            "G_X": "repair_metric",
            "r_X": "response_coordinate",
            "Sigma_X": "response_covariance",
        },
        "raw_jacobian": [list(row) for row in raw_jacobian],
        "jacobian": [list(row) for row in effective_jacobian],
        "raw_repair_metric": [list(row) for row in raw_metric],
        "repair_metric": [list(row) for row in effective_metric],
        "raw_response_coordinate": list(raw_coordinate),
        "response_coordinate": list(effective_coordinate),
        "coordinate_names": coordinate_names,
        "raw_response_covariance": [list(row) for row in raw_covariance_tuple],
        "response_covariance": [list(row) for row in effective_covariance],
        "locality_residuals": locality,
        "locality_threshold": threshold,
        "support_mask": list(support_mask),
        "supported_probe_path_count": sum(support_mask),
        "ridge": regularizer,
        "metric_contract": "G=J_supported^T W J_supported with ridge only on supported path diagonals",
        "covariance_contract": {
            "estimator": "paired-seed sample covariance within baseline-compatible source groups",
            "baseline_absolute_tolerance": tolerance,
            "observed_blocks": observed_blocks,
            "cross_group_entries": "unmeasured_zero_filled",
            "joint_baseline_group_count": len(groups),
        },
        "routing_state": "ready" if any(support_mask) else "abstain",
        "transfer_state": "ready" if not transfer_blockers else "abstain",
        "transfer_blockers": transfer_blockers,
        "source_charts": [
            {
                "source_id": source.source_id,
                "campaign_id": source.campaign_id,
                "probe_path_names": list(source.path_names),
                "checkpoint_step": source.checkpoint_step,
                **dict(source.provenance),
            }
            for source in source_rows
        ],
        "claim_boundary": (
            "This asset is a measured local routing chart. Unsupported paths are retained as raw "
            "finite-dose evidence but masked from J, G, and r. Cross-baseline-group covariance is "
            "unobserved and cannot license transfer."
        ),
    }


def validate_irg_asset(asset: Mapping[str, object]) -> None:
    """Validate dynamic matrix dimensions not expressible in JSON Schema."""

    if asset.get("artifact_type") != "verdiwm-unified-irg-asset":
        raise GeometryValidationError("IRG_ASSET_TYPE_INVALID")
    dimensions = asset.get("dimensions")
    if not isinstance(dimensions, Mapping):
        raise GeometryValidationError("IRG_ASSET_DIMENSIONS_INVALID")
    outcomes = int(dimensions.get("outcome_count", 0))
    paths = int(dimensions.get("probe_path_count", 0))
    coordinates = int(dimensions.get("response_coordinate_count", 0))
    if outcomes < 1 or paths < 1 or coordinates != outcomes * paths:
        raise GeometryValidationError("IRG_ASSET_DIMENSIONS_INVALID")
    for key, rows, columns in (
        ("raw_jacobian", outcomes, paths),
        ("jacobian", outcomes, paths),
        ("raw_repair_metric", paths, paths),
        ("repair_metric", paths, paths),
        ("raw_response_covariance", coordinates, coordinates),
        ("response_covariance", coordinates, coordinates),
    ):
        _validate_serialized_matrix(asset.get(key), rows, columns, key)
    for key, width in (
        ("raw_response_coordinate", coordinates),
        ("response_coordinate", coordinates),
        ("coordinate_names", coordinates),
        ("probe_path_names", paths),
        ("support_mask", paths),
    ):
        values = asset.get(key)
        if not isinstance(values, list) or len(values) != width:
            raise GeometryValidationError(f"IRG_ASSET_{key.upper()}_INVALID")
    symbols = asset.get("symbol_table")
    if not isinstance(symbols, Mapping) or symbols != {
        "J_X": "jacobian",
        "G_X": "repair_metric",
        "r_X": "response_coordinate",
        "Sigma_X": "response_covariance",
    }:
        raise GeometryValidationError("IRG_ASSET_SYMBOL_TABLE_INVALID")
    support = asset["support_mask"]
    if any(type(value) is not bool for value in support):
        raise GeometryValidationError("IRG_ASSET_SUPPORT_MASK_INVALID")
    if asset.get("supported_probe_path_count") != sum(support):
        raise GeometryValidationError("IRG_ASSET_SUPPORTED_COUNT_INVALID")
    jacobian = asset["jacobian"]
    metric = asset["repair_metric"]
    coordinate = asset["response_coordinate"]
    for path, is_supported in enumerate(support):
        if is_supported:
            continue
        if any(float(row[path]) != 0.0 for row in jacobian):
            raise GeometryValidationError("IRG_ASSET_UNSUPPORTED_JACOBIAN_NONZERO")
        if any(float(metric[path][column]) != 0.0 for column in range(paths)):
            raise GeometryValidationError("IRG_ASSET_UNSUPPORTED_METRIC_NONZERO")
        if any(float(metric[row][path]) != 0.0 for row in range(paths)):
            raise GeometryValidationError("IRG_ASSET_UNSUPPORTED_METRIC_NONZERO")
        if any(float(coordinate[outcome * paths + path]) != 0.0 for outcome in range(outcomes)):
            raise GeometryValidationError("IRG_ASSET_UNSUPPORTED_COORDINATE_NONZERO")


def _validate_source(
    source: IRGChartSource,
    *,
    outcomes: tuple[str, ...],
    weights: Vector,
    seeds: tuple[int, ...],
) -> None:
    if (
        not source.source_id
        or not source.campaign_id
        or not source.path_names
        or len(set(source.path_names)) != len(source.path_names)
    ):
        raise GeometryValidationError("IRG_ASSET_SOURCE_IDENTITY_INVALID")
    if source.outcome_names != outcomes or source.outcome_weights != weights or source.seeds != seeds:
        raise GeometryValidationError("IRG_ASSET_SOURCE_FRAME_MISMATCH")
    if source.checkpoint_step <= 0 or len(source.repeat_jacobians) != len(seeds):
        raise GeometryValidationError("IRG_ASSET_SOURCE_REPEAT_INVALID")
    _validate_matrix(source.jacobian, len(outcomes), len(source.path_names), "IRG_ASSET_SOURCE_JACOBIAN_INVALID")
    coordinate_width = len(outcomes) * len(source.path_names)
    _validate_matrix(source.covariance, coordinate_width, coordinate_width, "IRG_ASSET_SOURCE_COVARIANCE_INVALID")
    for matrix in source.repeat_jacobians:
        _validate_matrix(matrix, len(outcomes), len(source.path_names), "IRG_ASSET_SOURCE_REPEAT_INVALID")
    if len(source.baseline_vectors) != len(seeds) or any(len(row) != len(outcomes) for row in source.baseline_vectors):
        raise GeometryValidationError("IRG_ASSET_SOURCE_BASELINE_INVALID")
    if set(source.locality_residuals) != set(source.path_names):
        raise GeometryValidationError("IRG_ASSET_SOURCE_LOCALITY_INVALID")
    mean = _mean_matrices(source.repeat_jacobians)
    if _maximum_matrix_deviation(mean, source.jacobian) > 1e-8:
        raise GeometryValidationError("IRG_ASSET_SOURCE_REPEAT_MEAN_MISMATCH")
    repeated_coordinates = tuple(
        _response_coordinate(matrix, weights) for matrix in source.repeat_jacobians
    )
    covariance = _sample_covariance(repeated_coordinates)
    if _maximum_matrix_deviation(covariance, source.covariance) > 1e-8:
        raise GeometryValidationError("IRG_ASSET_SOURCE_COVARIANCE_MISMATCH")


def _baseline_groups(sources: Sequence[IRGChartSource], tolerance: float) -> tuple[tuple[int, ...], ...]:
    groups: list[list[int]] = []
    for index, source in enumerate(sources):
        for group in groups:
            if all(_baseline_deviation(source, sources[member]) <= tolerance for member in group):
                group.append(index)
                break
        else:
            groups.append([index])
    return tuple(tuple(group) for group in groups)


def _baseline_deviation(left: IRGChartSource, right: IRGChartSource) -> float:
    return max(
        abs(a - b)
        for left_row, right_row in zip(left.baseline_vectors, right.baseline_vectors, strict=True)
        for a, b in zip(left_row, right_row, strict=True)
    )


def _maximum_group_deviation(sources: Sequence[IRGChartSource], indices: Sequence[int]) -> float:
    return max(
        (_baseline_deviation(sources[left], sources[right]) for left in indices for right in indices),
        default=0.0,
    )


def _repeat_path_value(
    sources: Sequence[IRGChartSource],
    source_indices: Sequence[int],
    source_path_indices: Sequence[tuple[int, ...]],
    repeat: int,
    outcome: int,
    global_path: int,
) -> float:
    for source_index in source_indices:
        indices = source_path_indices[source_index]
        if global_path in indices:
            local_path = indices.index(global_path)
            return sources[source_index].repeat_jacobians[repeat][outcome][local_path]
    raise GeometryValidationError("IRG_ASSET_GROUP_PATH_INVALID")


def _repair_metric(jacobian: Matrix, weights: Vector, ridge: float, support: Sequence[bool]) -> Matrix:
    width = len(jacobian[0])
    return tuple(
        tuple(
            sum(weights[k] * jacobian[k][i] * jacobian[k][j] for k in range(len(jacobian)))
            + (ridge if i == j and support[i] else 0.0)
            for j in range(width)
        )
        for i in range(width)
    )


def _response_coordinate(jacobian: Matrix, weights: Vector) -> Vector:
    return tuple(
        math.sqrt(weights[outcome]) * value
        for outcome, row in enumerate(jacobian)
        for value in row
    )


def _sample_covariance(vectors: Sequence[Vector]) -> Matrix:
    if not vectors or not vectors[0] or any(len(row) != len(vectors[0]) for row in vectors):
        raise GeometryValidationError("IRG_ASSET_COVARIANCE_INPUT_INVALID")
    width = len(vectors[0])
    mean = tuple(sum(row[index] for row in vectors) / len(vectors) for index in range(width))
    if len(vectors) == 1:
        return tuple(tuple(0.0 for _ in range(width)) for _ in range(width))
    return tuple(
        tuple(
            sum((row[i] - mean[i]) * (row[j] - mean[j]) for row in vectors)
            / (len(vectors) - 1)
            for j in range(width)
        )
        for i in range(width)
    )


def _mean_matrices(matrices: Sequence[Matrix]) -> Matrix:
    rows = len(matrices[0])
    columns = len(matrices[0][0])
    return tuple(
        tuple(sum(matrix[row][column] for matrix in matrices) / len(matrices) for column in range(columns))
        for row in range(rows)
    )


def _maximum_matrix_deviation(left: Matrix, right: Matrix) -> float:
    return max(
        abs(a - b)
        for left_row, right_row in zip(left, right, strict=True)
        for a, b in zip(left_row, right_row, strict=True)
    )


def _validate_matrix(values: Matrix, rows: int, columns: int, code: str) -> None:
    if len(values) != rows or any(len(row) != columns for row in values):
        raise GeometryValidationError(code)
    if any(not math.isfinite(float(value)) for row in values for value in row):
        raise GeometryValidationError(code)


def _validate_serialized_matrix(value: object, rows: int, columns: int, key: str) -> None:
    if not isinstance(value, list) or len(value) != rows:
        raise GeometryValidationError(f"IRG_ASSET_{key.upper()}_INVALID")
    if any(not isinstance(row, list) or len(row) != columns for row in value):
        raise GeometryValidationError(f"IRG_ASSET_{key.upper()}_INVALID")
    if any(not math.isfinite(float(item)) for row in value for item in row):
        raise GeometryValidationError(f"IRG_ASSET_{key.upper()}_INVALID")


def _finite(value: float, code: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise GeometryValidationError(code)
    return result
