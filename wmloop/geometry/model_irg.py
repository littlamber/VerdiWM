"""Model-conditioned IRG bindings for routing, evidence, and collisions.

The response chart is the numeric IRG.  This module binds that vector to one
immutable model portrait and records the semantic axes, method effects, and
counterexample lineage that make the vector useful for autonomous research.
It deliberately grants ranking/diagnostic authority only; target verification
still belongs to the frozen experiment protocol.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from wmloop.contracts import ContractValidationError, validate_document
from wmloop.control.model_portrait import ModelPortraitError, validate_model_portrait
from wmloop.geometry.assets import validate_irg_asset
from wmloop.geometry.evolution import (
    AtlasPoint,
    EffectEstimate,
    RepairCollision,
    detect_repair_collisions,
)
from wmloop.geometry.evidence_ir import is_content_addressed, reject_runtime_bindings
from wmloop.geometry.types import GeometryValidationError


class ModelIRGError(ValueError):
    """A model-conditioned IRG binding or query is invalid."""


def build_model_irg(
    *,
    portrait: Mapping[str, object],
    asset: Mapping[str, object],
    diagnostic_axes: Sequence[Mapping[str, object]] = (),
    method_effects: Sequence[Mapping[str, object]] = (),
    collision_refs: Sequence[str] = (),
    evolution_refs: Sequence[str] = (),
    evidence_refs: Sequence[str] = (),
    root: Any = None,
) -> dict[str, object]:
    """Bind one measured response vector to a portrait and method evidence."""

    try:
        validate_model_portrait(portrait, root=root)
    except ModelPortraitError as exc:
        raise ModelIRGError(f"MODEL_IRG_PORTRAIT_INVALID:{exc}") from exc
    try:
        validate_irg_asset(asset)
    except GeometryValidationError as exc:
        raise ModelIRGError(f"MODEL_IRG_ASSET_INVALID:{exc}") from exc
    portrait_digest = "sha256:" + _digest(portrait)
    asset_digest = "sha256:" + _digest(asset)
    portrait_binding = {
        "portrait_id": str(portrait["portrait_id"]),
        "portrait_digest": portrait_digest,
        "model_capability_id": str(portrait["model_capability_id"]),
        "model_family": str(portrait["model_family"]),
    }
    asset_binding = {
        "asset_id": str(asset["asset_id"]),
        "asset_digest": asset_digest,
        "environment": str(asset["environment"]),
        "backbone_family": str(asset["backbone_family"]),
        "goal_schema": str(asset["goal_schema"]),
    }
    paths = [str(value) for value in asset["probe_path_names"]]
    outcomes = [str(value) for value in asset["outcome_names"]]
    vector = [float(value) for value in asset["response_coordinate"]]
    support = [bool(value) for value in asset["support_mask"]]
    axes = _normalize_axes(
        diagnostic_axes,
        asset=asset,
        portrait_digest=portrait_digest,
        asset_digest=asset_digest,
    )
    effects = _normalize_effects(method_effects)
    refs = _refs(
        (
            *evidence_refs,
            portrait_digest,
            asset_digest,
            *portrait.get("evidence_refs", []),
        )
    )
    for row in axes:
        refs.update(str(value) for value in row["evidence_refs"])
    for row in effects:
        refs.update(str(value) for value in row["evidence_refs"])
    refs.update(_refs(collision_refs))
    refs.update(_refs(evolution_refs))
    body: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "verdiwm-model-irg",
        "irg_id": "",
        "portrait_binding": portrait_binding,
        "asset_binding": asset_binding,
        "dimensions": {
            "coordinate_count": len(vector),
            "probe_count": len(paths),
            "outcome_count": len(outcomes),
        },
        "coordinate_names": [str(value) for value in asset["coordinate_names"]],
        "response_vector": vector,
        "response_covariance": [
            list(map(float, row)) for row in asset["response_covariance"]
        ],
        "support_mask": support,
        "diagnostic_axes": axes,
        "method_effects": effects,
        "collision_refs": sorted(_refs(collision_refs)),
        "evolution_refs": sorted(_refs(evolution_refs)),
        "evidence_refs": sorted(refs),
        "routing_state": str(asset["routing_state"]),
        "claim_boundary": (
            "This IRG is a measured, portrait-bound diagnostic vector for ranking and "
            "collision discovery. It does not replace target-side frozen verification."
        ),
    }
    body["irg_id"] = "model-irg-" + _digest(
        {key: value for key, value in body.items() if key != "irg_id"}
    )[:24]
    validate_model_irg(body, portrait=portrait, asset=asset, root=root)
    return body


def validate_model_irg(
    document: Mapping[str, object],
    *,
    portrait: Mapping[str, object] | None = None,
    asset: Mapping[str, object] | None = None,
    root: Any = None,
) -> None:
    """Validate schema, source digests, dimensions, and vector identity."""

    if not isinstance(document, Mapping):
        raise ModelIRGError("MODEL_IRG_DOCUMENT_INVALID")
    reject_runtime_bindings(document)
    try:
        validate_document("model_irg", document, root=root)
    except ContractValidationError as exc:
        raise ModelIRGError(f"MODEL_IRG_SCHEMA_INVALID:{exc}") from exc
    if portrait is not None:
        try:
            validate_model_portrait(portrait, root=root)
        except ModelPortraitError as exc:
            raise ModelIRGError(f"MODEL_IRG_PORTRAIT_INVALID:{exc}") from exc
        binding = document["portrait_binding"]
        if binding["portrait_id"] != portrait["portrait_id"]:
            raise ModelIRGError("MODEL_IRG_PORTRAIT_ID_MISMATCH")
        if binding["portrait_digest"] != "sha256:" + _digest(portrait):
            raise ModelIRGError("MODEL_IRG_PORTRAIT_DIGEST_MISMATCH")
    if asset is not None:
        try:
            validate_irg_asset(asset)
        except GeometryValidationError as exc:
            raise ModelIRGError(f"MODEL_IRG_ASSET_INVALID:{exc}") from exc
        binding = document["asset_binding"]
        if binding["asset_id"] != asset["asset_id"]:
            raise ModelIRGError("MODEL_IRG_ASSET_ID_MISMATCH")
        if binding["asset_digest"] != "sha256:" + _digest(asset):
            raise ModelIRGError("MODEL_IRG_ASSET_DIGEST_MISMATCH")
        if list(document["coordinate_names"]) != list(asset["coordinate_names"]):
            raise ModelIRGError("MODEL_IRG_COORDINATE_NAMES_MISMATCH")
        if list(document["response_vector"]) != list(asset["response_coordinate"]):
            raise ModelIRGError("MODEL_IRG_RESPONSE_VECTOR_MISMATCH")
        if list(document["support_mask"]) != list(asset["support_mask"]):
            raise ModelIRGError("MODEL_IRG_SUPPORT_MASK_MISMATCH")
        if list(document["response_covariance"]) != list(asset["response_covariance"]):
            raise ModelIRGError("MODEL_IRG_RESPONSE_COVARIANCE_MISMATCH")
    dimensions = document["dimensions"]
    if dimensions["coordinate_count"] != len(document["response_vector"]):
        raise ModelIRGError("MODEL_IRG_COORDINATE_COUNT_MISMATCH")
    if dimensions["coordinate_count"] != len(document["coordinate_names"]):
        raise ModelIRGError("MODEL_IRG_COORDINATE_NAME_COUNT_MISMATCH")
    if len(document["diagnostic_axes"]) != dimensions["coordinate_count"]:
        raise ModelIRGError("MODEL_IRG_DIAGNOSTIC_AXIS_COUNT_MISMATCH")
    if dimensions["probe_count"] != len(document["support_mask"]):
        raise ModelIRGError("MODEL_IRG_PROBE_PATH_COUNT_MISMATCH")
    covariance = document["response_covariance"]
    if len(covariance) != dimensions["coordinate_count"] or any(
        len(row) != dimensions["coordinate_count"] for row in covariance
    ):
        raise ModelIRGError("MODEL_IRG_COVARIANCE_SHAPE_INVALID")
    axis_names = [str(row["axis"]) for row in document["diagnostic_axes"]]
    if len(axis_names) != len(set(axis_names)):
        raise ModelIRGError("MODEL_IRG_DIAGNOSTIC_AXIS_DUPLICATE")
    if set(axis_names) != set(str(value) for value in document["coordinate_names"]):
        raise ModelIRGError("MODEL_IRG_DIAGNOSTIC_AXIS_COORDINATE_MISMATCH")
    effect_ids = [str(row["effect_id"]) for row in document["method_effects"]]
    if len(effect_ids) != len(set(effect_ids)):
        raise ModelIRGError("MODEL_IRG_EFFECT_DUPLICATE")
    primitives = [str(row["primitive"]) for row in document["method_effects"]]
    if len(primitives) != len(set(primitives)):
        raise ModelIRGError("MODEL_IRG_EFFECT_PRIMITIVE_DUPLICATE")
    for row in document["method_effects"]:
        if row["lower_bound"] > row["mean_effect"] or row["mean_effect"] > row["upper_bound"]:
            raise ModelIRGError("MODEL_IRG_EFFECT_INTERVAL_INVALID")
    body = dict(document)
    body.pop("irg_id", None)
    if document["irg_id"] != "model-irg-" + _digest(body)[:24]:
        raise ModelIRGError("MODEL_IRG_ID_MISMATCH")


def model_irg_distance(left: Mapping[str, object], right: Mapping[str, object]) -> float:
    """Compare two compatible portrait-bound response vectors with uncertainty."""

    validate_model_irg(left)
    validate_model_irg(right)
    if left["coordinate_names"] != right["coordinate_names"]:
        raise ModelIRGError("MODEL_IRG_COORDINATES_INCOMPATIBLE")
    if left["asset_binding"]["goal_schema"] != right["asset_binding"]["goal_schema"]:
        raise ModelIRGError("MODEL_IRG_GOAL_INCOMPATIBLE")
    shared = [
        index
        for index, (a, b) in enumerate(
            zip(left["support_mask"], right["support_mask"], strict=True)
        )
        if a and b
    ]
    if not shared:
        raise ModelIRGError("MODEL_IRG_NO_SHARED_SUPPORT")
    total = 0.0
    for index in shared:
        variance = float(left["response_covariance"][index][index]) + float(
            right["response_covariance"][index][index]
        )
        delta = float(left["response_vector"][index]) - float(
            right["response_vector"][index]
        )
        total += (delta * delta) / (1.0 + max(variance, 0.0))
    asymmetric_support = sum(
        bool(a) != bool(b)
        for a, b in zip(left["support_mask"], right["support_mask"], strict=True)
    )
    missing_fraction = asymmetric_support / len(left["support_mask"])
    return math.sqrt(total + missing_fraction)


def rank_method_effects_by_irg(
    target: Mapping[str, object],
    candidates: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    """Rank compatible method effects by IRG proximity as ranking-only priors."""

    validate_model_irg(target)
    rows: list[dict[str, object]] = []
    for candidate in candidates:
        validate_model_irg(candidate)
        try:
            distance = model_irg_distance(target, candidate)
        except ModelIRGError:
            continue
        prior_only = (
            candidate["portrait_binding"]["portrait_id"]
            != target["portrait_binding"]["portrait_id"]
        )
        for effect in candidate["method_effects"]:
            rows.append(
                {
                    "method_id": effect["method_id"],
                    "primitive": effect["primitive"],
                    "effect_status": effect["effect_status"],
                    "mean_effect": effect["mean_effect"],
                    "effect_id": effect["effect_id"],
                    "source_irg_id": candidate["irg_id"],
                    "irg_distance": distance,
                    "claim_scope": "ranking_only",
                    "prior_only": prior_only,
                    "evidence_refs": list(effect["evidence_refs"]),
                }
            )
    return tuple(
        sorted(
            rows,
            key=lambda row: (
                float(row["irg_distance"]),
                str(row["method_id"]),
                str(row["effect_id"]),
            ),
        )
    )


def detect_model_irg_collisions(
    bindings: Sequence[Mapping[str, object]],
    *,
    distance_threshold: float,
    minimum_effect: float,
    fdr_alpha: float,
) -> tuple[RepairCollision, ...]:
    """Detect nearby IRGs whose bound method effects confidently disagree."""

    points: list[AtlasPoint] = []
    for binding in bindings:
        validate_model_irg(binding)
        effects = {
            str(row["primitive"]): EffectEstimate(
                mean=float(row["mean_effect"]),
                lower=float(row["lower_bound"]),
                upper=float(row["upper_bound"]),
                sign_q_value=float(row["sign_q_value"]),
            )
            for row in binding["method_effects"]
        }
        points.append(
            AtlasPoint(
                campaign_id=str(binding["irg_id"]),
                chart_id=str(binding["asset_binding"]["goal_schema"]),
                coordinates=tuple(float(value) for value in binding["response_vector"]),
                effects=effects,
            )
        )
    return detect_repair_collisions(
        points,
        distance_threshold=distance_threshold,
        minimum_effect=minimum_effect,
        fdr_alpha=fdr_alpha,
    )


def _normalize_axes(
    rows: Sequence[Mapping[str, object]],
    *,
    asset: Mapping[str, object],
    portrait_digest: str,
    asset_digest: str,
) -> list[dict[str, object]]:
    if rows:
        normalized = [dict(row) for row in rows]
        for row in normalized:
            row["evidence_refs"] = sorted(_refs(row.get("evidence_refs", ())))
        return normalized
    paths = [str(value) for value in asset["probe_path_names"]]
    outcomes = [str(value) for value in asset["outcome_names"]]
    vector = [float(value) for value in asset["response_coordinate"]]
    support = [bool(value) for value in asset["support_mask"]]
    refs = [portrait_digest, asset_digest]
    result = []
    for index, value in enumerate(vector):
        path_index = index % len(paths)
        outcome_index = index // len(paths)
        supported = support[path_index]
        result.append(
            {
                "axis": str(asset["coordinate_names"][index]),
                "probe_id": paths[path_index],
                "outcome": outcomes[outcome_index],
                "response": value,
                "magnitude": abs(value),
                "support_state": "supported" if supported else "unsupported",
                "diagnosis": "response_axis" if supported else "locality_unsupported",
                "evidence_refs": refs,
            }
        )
    return result


def _normalize_effects(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    normalized = [dict(row) for row in rows]
    for row in normalized:
        row["evidence_refs"] = sorted(_refs(row.get("evidence_refs", ())))
    return normalized


def _refs(values: Sequence[object]) -> set[str]:
    result: set[str] = set()
    for value in values:
        if not isinstance(value, str) or not is_content_addressed(value):
            raise ModelIRGError("MODEL_IRG_EVIDENCE_REF_INVALID")
        result.add(value)
    return result


def _digest(value: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
