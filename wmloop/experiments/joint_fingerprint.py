"""Joint-frame ACWM fingerprint contracts and full-covariance chart fitting."""

from __future__ import annotations

from dataclasses import dataclass
import copy
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from wmloop.experiments.acwm_fingerprint import goal_vector, load_campaign
from wmloop.geometry import IRGChartSource, ResponseChart, compose_irg_asset, estimate_response_chart


class JointFingerprintError(ValueError):
    """A joint probe campaign would violate its paired experimental frame."""


@dataclass(frozen=True)
class JointFingerprintFit:
    chart: ResponseChart
    repeat_jacobians: tuple[tuple[tuple[float, ...], ...], ...]
    baseline_vectors: tuple[tuple[float, ...], ...]


def load_joint_campaign(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("artifact_type") != "verdiwm-acwm-joint-fingerprint-campaign":
        raise JointFingerprintError("JOINT_FINGERPRINT_CAMPAIGN_TYPE_INVALID")
    if payload.get("generation_mode") != "autoregressive":
        raise JointFingerprintError("JOINT_FINGERPRINT_GENERATION_MODE_INVALID")
    source_rows = payload.get("source_campaigns")
    path_rows = payload.get("semantic_paths")
    if not isinstance(source_rows, list) or not source_rows:
        raise JointFingerprintError("JOINT_FINGERPRINT_SOURCES_INVALID")
    if not isinstance(path_rows, list) or not path_rows:
        raise JointFingerprintError("JOINT_FINGERPRINT_PATHS_INVALID")
    source_ids = tuple(str(row.get("source_probe_id") or "") for row in source_rows if isinstance(row, Mapping))
    if len(source_ids) != len(source_rows) or "" in source_ids or len(set(source_ids)) != len(source_ids):
        raise JointFingerprintError("JOINT_FINGERPRINT_SOURCE_IDS_INVALID")
    path_names = tuple(str(row.get("path_name") or "") for row in path_rows if isinstance(row, Mapping))
    if len(path_names) != len(path_rows) or "" in path_names or len(set(path_names)) != len(path_names):
        raise JointFingerprintError("JOINT_FINGERPRINT_PATH_NAMES_INVALID")
    transforms = {"signed", "positive_magnitude", "negative_magnitude"}
    for row in path_rows:
        if (
            not isinstance(row, Mapping)
            or row.get("source_probe_id") not in source_ids
            or row.get("dose_transform") not in transforms
        ):
            raise JointFingerprintError("JOINT_FINGERPRINT_PATH_MAPPING_INVALID")
    return payload


def load_joint_sources(
    joint: Mapping[str, Any], *, repo_root: Path
) -> tuple[dict[str, Any], ...]:
    root = Path(repo_root).resolve(strict=True)
    sources: list[dict[str, Any]] = []
    for row in joint["source_campaigns"]:
        path = Path(str(row["campaign_ref"]))
        resolved = path.resolve(strict=True) if path.is_absolute() else (root / path).resolve(strict=True)
        campaign = load_campaign(resolved)
        if campaign["probe"]["probe_id"] != row["source_probe_id"]:
            raise JointFingerprintError("JOINT_FINGERPRINT_SOURCE_PROBE_MISMATCH")
        sources.append(campaign)
    validate_joint_frame(joint, sources)
    effective_sources: list[dict[str, Any]] = []
    for row, source in zip(joint["source_campaigns"], sources, strict=True):
        effective = copy.deepcopy(source)
        effective["probe"]["generation_mode"] = str(row["generation_mode_override"])
        effective_sources.append(effective)
    return tuple(effective_sources)


def validate_joint_frame(
    joint: Mapping[str, Any], sources: Sequence[Mapping[str, Any]]
) -> None:
    if not sources:
        raise JointFingerprintError("JOINT_FINGERPRINT_SOURCES_INVALID")
    source_rows = joint["source_campaigns"]
    if len(source_rows) != len(sources):
        raise JointFingerprintError("JOINT_FINGERPRINT_SOURCE_COUNT_MISMATCH")
    reference = sources[0]
    shared_keys = (
        "backbone_instance_ref",
        "goal_spec_ref",
        "goal_oriented_outcomes",
        "seeds",
        "environments",
        "protocols",
    )
    for row, source in zip(source_rows, sources, strict=True):
        if row.get("generation_mode_override") != joint.get("generation_mode"):
            raise JointFingerprintError("JOINT_FINGERPRINT_MODE_OVERRIDE_MISSING")
        if any(source.get(key) != reference.get(key) for key in shared_keys):
            raise JointFingerprintError("JOINT_FINGERPRINT_SHARED_FRAME_MISMATCH")
    seeds = tuple(int(value) for value in reference["seeds"])
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise JointFingerprintError("JOINT_FINGERPRINT_SEEDS_INVALID")
    if len(reference["environments"]) != 8:
        raise JointFingerprintError("JOINT_FINGERPRINT_ENVIRONMENT_FRAME_INVALID")
    source_ids = {str(source["probe"]["probe_id"]) for source in sources}
    mapped = {str(row["source_probe_id"]) for row in joint["semantic_paths"]}
    if mapped != source_ids:
        raise JointFingerprintError("JOINT_FINGERPRINT_SOURCE_PATH_COVERAGE_INVALID")


def condition_schedule(
    joint: Mapping[str, Any], sources: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, object], ...]:
    validate_joint_frame(joint, sources)
    seeds = tuple(int(value) for value in sources[0]["seeds"])
    rows: list[dict[str, object]] = []
    for seed in seeds:
        rows.append(
            {
                "condition_kind": "baseline",
                "condition_id": f"baseline__s{seed}",
                "source_probe_id": None,
                "dose": 0.0,
                "seed": seed,
            }
        )
        for source in sources:
            probe_id = str(source["probe"]["probe_id"])
            for dose in source["probe"]["doses"]:
                value = float(dose)
                if value == 0.0:
                    continue
                rows.append(
                    {
                        "condition_kind": "probe",
                        "condition_id": f"{probe_id}__{_dose_tag(value)}__s{seed}",
                        "source_probe_id": probe_id,
                        "dose": value,
                        "seed": seed,
                    }
                )
    return tuple(rows)


def fit_joint_fingerprint(
    joint: Mapping[str, Any],
    sources: Sequence[Mapping[str, Any]],
    *,
    environment: str,
    measurements: Sequence[Mapping[str, Any]],
) -> JointFingerprintFit:
    validate_joint_frame(joint, sources)
    expected = condition_schedule(joint, sources)
    by_identity: dict[tuple[str, str | None, float, int], Mapping[str, Any]] = {}
    for row in measurements:
        if row.get("environment") != environment:
            continue
        identity = (
            str(row.get("condition_kind") or ""),
            str(row["source_probe_id"]) if row.get("source_probe_id") is not None else None,
            float(row["dose"]),
            int(row["seed"]),
        )
        if identity in by_identity:
            raise JointFingerprintError("JOINT_FINGERPRINT_MEASUREMENT_DUPLICATED")
        by_identity[identity] = row
    expected_identities = {
        (
            str(row["condition_kind"]),
            str(row["source_probe_id"]) if row["source_probe_id"] is not None else None,
            float(row["dose"]),
            int(row["seed"]),
        )
        for row in expected
    }
    if set(by_identity) != expected_identities:
        raise JointFingerprintError("JOINT_FINGERPRINT_MEASUREMENT_FRAME_INCOMPLETE")

    reference = sources[0]
    seeds = tuple(int(value) for value in reference["seeds"])
    source_by_id = {str(source["probe"]["probe_id"]): source for source in sources}
    baseline = tuple(
        goal_vector(
            reference,
            by_identity[("baseline", None, 0.0, seed)]["metrics"],
        )
        for seed in seeds
    )
    observations: dict[str, dict[float, tuple[tuple[float, ...], ...]]] = {}
    path_order: list[str] = []
    for path in joint["semantic_paths"]:
        path_name = str(path["path_name"])
        probe_id = str(path["source_probe_id"])
        transform = str(path["dose_transform"])
        source = source_by_id[probe_id]
        dose_rows: dict[float, tuple[tuple[float, ...], ...]] = {}
        for raw_dose in source["probe"]["doses"]:
            raw = float(raw_dose)
            semantic = _semantic_dose(raw, transform)
            if semantic is None:
                continue
            if semantic in dose_rows:
                raise JointFingerprintError("JOINT_FINGERPRINT_SEMANTIC_DOSE_COLLISION")
            dose_rows[semantic] = tuple(
                goal_vector(
                    reference,
                    by_identity[("probe", probe_id, raw, seed)]["metrics"],
                )
                for seed in seeds
            )
        if not dose_rows:
            raise JointFingerprintError("JOINT_FINGERPRINT_PATH_OBSERVATIONS_EMPTY")
        observations[path_name] = dose_rows
        path_order.append(path_name)

    chart = estimate_response_chart(
        chart_id=f"{joint['campaign_id']}:{environment}",
        goal_schema="acwm_phys_goal_oriented_pixel_metrics_v1",
        outcome_names=tuple(str(row["name"]) for row in reference["goal_oriented_outcomes"]),
        outcome_weights=tuple(float(row["weight"]) for row in reference["goal_oriented_outcomes"]),
        baseline_repeats=baseline,
        dose_observations=observations,
    )
    ordered_chart = _reorder_chart(chart, tuple(path_order))
    repeats = _repeat_jacobians(observations, baseline, tuple(path_order))
    return JointFingerprintFit(
        chart=ordered_chart,
        repeat_jacobians=repeats,
        baseline_vectors=baseline,
    )


def compose_joint_irg_asset(
    joint: Mapping[str, Any],
    sources: Sequence[Mapping[str, Any]],
    fit: JointFingerprintFit,
    *,
    environment: str,
    checkpoint_step: int,
    provenance: Mapping[str, object],
    locality_threshold: float = 0.5,
) -> dict[str, object]:
    reference = sources[0]
    source = IRGChartSource(
        source_id=str(joint["campaign_id"]),
        campaign_id=str(joint["campaign_id"]),
        path_names=fit.chart.intervention_names,
        outcome_names=fit.chart.outcome_names,
        outcome_weights=tuple(float(row["weight"]) for row in reference["goal_oriented_outcomes"]),
        seeds=tuple(int(value) for value in reference["seeds"]),
        jacobian=fit.chart.jacobian,
        covariance=fit.chart.covariance,
        locality_residuals=fit.chart.locality_residuals,
        repeat_jacobians=fit.repeat_jacobians,
        baseline_vectors=fit.baseline_vectors,
        checkpoint_step=int(checkpoint_step),
        provenance=dict(provenance),
    )
    return compose_irg_asset(
        asset_id=f"acwm-phys:{environment}:joint-autoregressive-r1",
        environment=environment,
        backbone_family="acwm_phys",
        capability_class="latent_dit_action_conditioned",
        backbone_instance_ref=str(reference["backbone_instance_ref"]),
        sources=(source,),
        locality_threshold=locality_threshold,
    )


def _semantic_dose(raw: float, transform: str) -> float | None:
    if raw == 0.0:
        return None
    if transform == "signed":
        return raw
    if transform == "positive_magnitude":
        return raw if raw > 0.0 else None
    if transform == "negative_magnitude":
        return abs(raw) if raw < 0.0 else None
    raise JointFingerprintError("JOINT_FINGERPRINT_DOSE_TRANSFORM_INVALID")


def _repeat_jacobians(
    observations: Mapping[str, Mapping[float, tuple[tuple[float, ...], ...]]],
    baseline: tuple[tuple[float, ...], ...],
    path_order: tuple[str, ...],
) -> tuple[tuple[tuple[float, ...], ...], ...]:
    per_path = {path: _smallest_slopes(observations[path], baseline) for path in path_order}
    repeat_count = min(len(values) for values in per_path.values())
    outcome_count = len(baseline[0])
    return tuple(
        tuple(
            tuple(per_path[path][repeat][outcome] for path in path_order)
            for outcome in range(outcome_count)
        )
        for repeat in range(repeat_count)
    )


def _smallest_slopes(
    observations: Mapping[float, tuple[tuple[float, ...], ...]],
    baseline: tuple[tuple[float, ...], ...],
) -> tuple[tuple[float, ...], ...]:
    positive = sorted(dose for dose in observations if dose > 0.0)
    negative = sorted((dose for dose in observations if dose < 0.0), key=abs)
    if positive and negative and math.isclose(positive[0], abs(negative[0]), rel_tol=1e-9, abs_tol=1e-12):
        plus, minus = observations[positive[0]], observations[negative[0]]
        return tuple(
            tuple((a - b) / (positive[0] - negative[0]) for a, b in zip(left, right, strict=True))
            for left, right in zip(plus, minus, strict=True)
        )
    dose = positive[0] if positive else negative[0]
    return tuple(
        tuple((value - zero) / dose for value, zero in zip(treated, control, strict=True))
        for treated, control in zip(observations[dose], baseline, strict=True)
    )


def _reorder_chart(chart: ResponseChart, path_order: tuple[str, ...]) -> ResponseChart:
    if set(path_order) != set(chart.intervention_names) or len(path_order) != len(chart.intervention_names):
        raise JointFingerprintError("JOINT_FINGERPRINT_CHART_PATH_FRAME_INVALID")
    old_positions = {name: index for index, name in enumerate(chart.intervention_names)}
    permutation = tuple(old_positions[name] for name in path_order)
    width = len(path_order)
    coordinate_permutation = tuple(
        outcome * width + old
        for outcome in range(len(chart.outcome_names))
        for old in permutation
    )
    return ResponseChart(
        chart_id=chart.chart_id,
        goal_schema=chart.goal_schema,
        outcome_names=chart.outcome_names,
        intervention_names=path_order,
        jacobian=tuple(tuple(row[index] for index in permutation) for row in chart.jacobian),
        repair_metric=tuple(
            tuple(chart.repair_metric[left][right] for right in permutation)
            for left in permutation
        ),
        response_coordinate=tuple(chart.response_coordinate[index] for index in coordinate_permutation),
        covariance=tuple(
            tuple(chart.covariance[left][right] for right in coordinate_permutation)
            for left in coordinate_permutation
        ),
        locality_residuals={name: chart.locality_residuals[name] for name in path_order},
        repeat_count=chart.repeat_count,
    )


def _dose_tag(dose: float) -> str:
    return f"{dose:+.4f}".replace("+", "p").replace("-", "m").replace(".", "d")
