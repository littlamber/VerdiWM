"""ACWM-Phys paired-dose fingerprint contracts and response-chart fitting."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from wmloop.geometry import (
    CapabilityProfile,
    InterventionDescriptor,
    compile_intervention,
    estimate_response_chart,
)


class FingerprintCampaignError(ValueError):
    """A fingerprint campaign or measurement set is malformed."""


def load_campaign(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("artifact_type") != "verdiwm-acwm-fingerprint-campaign":
        raise FingerprintCampaignError("FINGERPRINT_CAMPAIGN_TYPE_INVALID")
    probe = payload.get("probe")
    if not isinstance(probe, Mapping):
        raise FingerprintCampaignError("FINGERPRINT_PROBE_MISSING")
    doses = tuple(float(value) for value in probe.get("doses", ()))
    if 0.0 not in doses or len(doses) < 3 or len(set(doses)) != len(doses):
        raise FingerprintCampaignError("FINGERPRINT_DOSE_FRAME_INVALID")
    nonzero = {value for value in doses if value != 0.0}
    if not any(-value in nonzero for value in nonzero):
        raise FingerprintCampaignError("FINGERPRINT_SYMMETRIC_DOSE_MISSING")
    seeds = tuple(int(value) for value in payload.get("seeds", ()))
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise FingerprintCampaignError("FINGERPRINT_SEEDS_INVALID")
    environments = payload.get("environments")
    if not isinstance(environments, Mapping) or len(environments) != 8:
        raise FingerprintCampaignError("FINGERPRINT_ENVIRONMENT_FRAME_INVALID")
    outcomes = payload.get("goal_oriented_outcomes")
    if not isinstance(outcomes, list) or not outcomes:
        raise FingerprintCampaignError("FINGERPRINT_OUTCOMES_INVALID")
    return payload


def descriptor_from_campaign(campaign: Mapping[str, Any]) -> InterventionDescriptor:
    probe = campaign["probe"]
    return InterventionDescriptor(
        name=str(probe["probe_id"]),
        kind=str(probe["kind"]),
        hook_type=str(probe["hook_type"]),
        transformation=str(probe["transformation"]),
        scope=str(probe["scope"]),
        dose_unit=str(probe["dose_unit"]),
        schedule=str(probe["schedule"]),
        preconditions=tuple(str(value) for value in probe["preconditions"]),
        invariants=tuple(str(value) for value in probe["invariants"]),
        prediction=str(probe["prediction"]),
        required_capabilities=frozenset({"action_embedding_hook", "paired_seed_control"}),
        inference_only=bool(probe["inference_only"]),
        reversible=bool(probe["reversible"]),
    )


def compile_probe_receipt(campaign: Mapping[str, Any], *, dose: float) -> dict[str, object]:
    if dose == 0.0:
        return {
            "schema_version": 1,
            "artifact_type": "verdiwm-typed-compile-receipt",
            "descriptor_name": campaign["probe"]["probe_id"],
            "backbone_family": "acwm_phys",
            "capability_class": "latent_dit_action_conditioned",
            "compiled": True,
            "hook_type": campaign["probe"]["hook_type"],
            "dose_unit": campaign["probe"]["dose_unit"],
            "dose_direction": 0.0,
            "semantic_obligations": ["control_condition_no_hook_effect"],
            "invariant_checks": {name: True for name in campaign["probe"]["invariants"]},
            "blockers": [],
            "control_condition": True,
        }
    receipt = compile_intervention(
        descriptor_from_campaign(campaign),
        CapabilityProfile(
            backbone_family="acwm_phys",
            capability_class="latent_dit_action_conditioned",
            capabilities=frozenset({"action_embedding_hook", "paired_seed_control"}),
            hook_types=frozenset({"H2"}),
        ),
        invariant_checks={name: True for name in campaign["probe"]["invariants"]},
        dose_direction=dose,
    )
    return receipt.to_dict()


def goal_vector(campaign: Mapping[str, Any], metrics: Mapping[str, float]) -> tuple[float, ...]:
    values: list[float] = []
    for outcome in campaign["goal_oriented_outcomes"]:
        value = float(metrics[str(outcome["source_metric"])]) * float(outcome["sign"])
        if not math.isfinite(value):
            raise FingerprintCampaignError("FINGERPRINT_METRIC_NONFINITE")
        values.append(value)
    return tuple(values)


def fit_chart(
    campaign: Mapping[str, Any],
    *,
    environment: str,
    measurements: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    probe_id = str(campaign["probe"]["probe_id"])
    by_dose: dict[float, dict[int, tuple[float, ...]]] = {}
    for row in measurements:
        if row.get("environment") != environment:
            continue
        dose = float(row["dose"])
        seed = int(row["seed"])
        by_dose.setdefault(dose, {})[seed] = goal_vector(campaign, row["metrics"])
    expected_seeds = tuple(int(value) for value in campaign["seeds"])
    doses = tuple(float(value) for value in campaign["probe"]["doses"])
    for dose in doses:
        if set(by_dose.get(dose, {})) != set(expected_seeds):
            raise FingerprintCampaignError(f"FINGERPRINT_MEASUREMENT_INCOMPLETE:{environment}:{dose}")
    baseline = tuple(by_dose[0.0][seed] for seed in expected_seeds)
    observations = {
        probe_id: {
            dose: tuple(by_dose[dose][seed] for seed in expected_seeds)
            for dose in doses
            if dose != 0.0
        }
    }
    chart = estimate_response_chart(
        chart_id=f"{campaign['campaign_id']}:{environment}",
        goal_schema="acwm_phys_goal_oriented_pixel_metrics_v1",
        outcome_names=tuple(str(value["name"]) for value in campaign["goal_oriented_outcomes"]),
        outcome_weights=tuple(float(value["weight"]) for value in campaign["goal_oriented_outcomes"]),
        baseline_repeats=baseline,
        dose_observations=observations,
    )
    return chart.to_dict()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
