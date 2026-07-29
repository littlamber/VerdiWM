"""Evolve a symmetric Cosmos3 probe into a held-out-testable directional path."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

from wmloop.experiments._artifacts import canonical_json, write_bundle
from wmloop.geometry.irg import estimate_response_chart


class Cosmos3DirectionalProbeError(ValueError):
    """The directional probe evolution contract is invalid."""


def build_cosmos3_directional_probe_evolution(
    *,
    source_fingerprint_root: Path,
    successor_campaign_path: Path,
    output_root: Path,
) -> dict[str, object]:
    source_root = Path(source_fingerprint_root).resolve(strict=True)
    source_campaign = _load_json(source_root / "input-campaign.json")
    source_fingerprint = _load_json(source_root / "target-local-fingerprint.json")
    successor = _load_json(Path(successor_campaign_path).resolve(strict=True))
    measurements = [
        json.loads(line)
        for line in (source_root / "measurements.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if (
        source_campaign.get("artifact_type") != "verdiwm-cosmos3-fingerprint-campaign"
        or source_fingerprint.get("artifact_type") != "verdiwm-cosmos3-target-local-fingerprint"
        or source_fingerprint.get("protocol") != "pilot"
        or source_fingerprint.get("split") != "dev"
        or successor.get("artifact_type") != "verdiwm-cosmos3-fingerprint-campaign"
    ):
        raise Cosmos3DirectionalProbeError("COSMOS3_DIRECTIONAL_SOURCE_INVALID")
    predecessor = successor.get("predecessor_campaign")
    if (
        not isinstance(predecessor, Mapping)
        or predecessor.get("campaign_id") != source_campaign.get("campaign_id")
        or predecessor.get("fingerprint_sha256") != _sha256(source_root / "target-local-fingerprint.json")
    ):
        raise Cosmos3DirectionalProbeError("COSMOS3_DIRECTIONAL_PREDECESSOR_MISMATCH")
    source_probe = source_campaign.get("probe")
    successor_probe = successor.get("probe")
    if not isinstance(source_probe, Mapping) or not isinstance(successor_probe, Mapping):
        raise Cosmos3DirectionalProbeError("COSMOS3_DIRECTIONAL_PROBE_INVALID")
    if source_probe.get("probe_id") != successor_probe.get("probe_id"):
        raise Cosmos3DirectionalProbeError("COSMOS3_DIRECTIONAL_PROBE_ID_MISMATCH")
    source_doses = {float(value) for value in source_probe.get("doses", [])}
    successor_doses = tuple(float(value) for value in successor_probe.get("doses", []))
    nonzero = tuple(value for value in successor_doses if value != 0.0)
    if (
        0.0 not in successor_doses
        or len(nonzero) < 2
        or not set(successor_doses).issubset(source_doses)
        or not (all(value > 0.0 for value in nonzero) or all(value < 0.0 for value in nonzero))
    ):
        raise Cosmos3DirectionalProbeError("COSMOS3_DIRECTIONAL_DOSE_DOMAIN_INVALID")
    if successor.get("outcomes") != source_campaign.get("outcomes"):
        raise Cosmos3DirectionalProbeError("COSMOS3_DIRECTIONAL_OUTCOME_DRIFT")
    source_threshold = float(source_campaign["locality_admission"]["maximum_residual"])
    successor_threshold = float(successor["locality_admission"]["maximum_residual"])
    if successor_threshold != source_threshold:
        raise Cosmos3DirectionalProbeError("COSMOS3_DIRECTIONAL_THRESHOLD_DRIFT")
    selection = successor.get("selection_policy")
    if (
        not isinstance(selection, Mapping)
        or selection.get("selection_split") != "dev"
        or selection.get("validation_split") != "accept"
        or selection.get("accept_data_used_for_selection") is not False
    ):
        raise Cosmos3DirectionalProbeError("COSMOS3_DIRECTIONAL_SELECTION_POLICY_INVALID")

    by_key: dict[tuple[float, int, int], tuple[float, ...]] = {}
    for row in measurements:
        if not isinstance(row, Mapping) or not isinstance(row.get("outcomes"), list):
            raise Cosmos3DirectionalProbeError("COSMOS3_DIRECTIONAL_MEASUREMENT_INVALID")
        key = (float(row["dose"]), int(row["sample_index"]), int(row["seed"]))
        if key in by_key:
            raise Cosmos3DirectionalProbeError("COSMOS3_DIRECTIONAL_MEASUREMENT_DUPLICATE")
        by_key[key] = tuple(float(value) for value in row["outcomes"])
    identities = tuple(sorted((sample, seed) for dose, sample, seed in by_key if dose == 0.0))
    required = {(dose, *identity) for dose in successor_doses for identity in identities}
    if len(identities) < 2 or not required.issubset(by_key):
        raise Cosmos3DirectionalProbeError("COSMOS3_DIRECTIONAL_MEASUREMENT_INCOMPLETE")
    outcomes = successor["outcomes"]
    chart = estimate_response_chart(
        chart_id=f"{successor['campaign_id']}:dev_selection",
        goal_schema="cosmos3_acwm_forward_dynamics_v1",
        outcome_names=tuple(str(item["name"]) for item in outcomes),
        outcome_weights=tuple(float(item["weight"]) for item in outcomes),
        baseline_repeats=tuple(by_key[(0.0, *identity)] for identity in identities),
        dose_observations={
            str(successor_probe["probe_id"]): {
                dose: tuple(by_key[(dose, *identity)] for identity in identities)
                for dose in nonzero
            }
        },
    ).to_dict()
    probe_id = str(successor_probe["probe_id"])
    residual = float(chart["locality_residuals"][probe_id])
    admitted = residual <= successor_threshold
    source_residual = float(source_fingerprint["locality_admission"]["path_residuals"][probe_id])
    report = {
        "schema_version": 1,
        "artifact_type": "verdiwm-cosmos3-directional-probe-evolution",
        "state": "dev_selected" if admitted else "dev_abstained",
        "decision": "freeze_direction_then_validate_on_accept" if admitted else "retain_counterexample_and_abstain",
        "source_campaign_id": source_campaign["campaign_id"],
        "successor_campaign_id": successor["campaign_id"],
        "probe_id": probe_id,
        "dose_domain": successor_probe.get("dose_domain"),
        "selected_doses": list(successor_doses),
        "source_symmetric_residual": source_residual,
        "selected_directional_residual": residual,
        "maximum_residual": successor_threshold,
        "dev_locality_admitted": admitted,
        "dev_chart": chart,
        "source_fingerprint_sha256": _sha256(source_root / "target-local-fingerprint.json"),
        "source_measurements_sha256": _sha256(source_root / "measurements.jsonl"),
        "selection_policy": dict(selection),
        "claim_boundary": (
            "A directional probe was selected using dev receipts only. Transfer remains prohibited "
            "until the frozen direction passes independent accept locality and dev/accept alignment."
        ),
    }
    selected_fingerprint = {
        "schema_version": 1,
        "artifact_type": "verdiwm-cosmos3-target-local-fingerprint",
        "state": "ready",
        "campaign_id": successor["campaign_id"],
        "protocol": "pilot",
        "split": "dev",
        "measurement_count": len(required),
        "repeat_count": len(identities),
        "chart": chart,
        "locality_admission": {
            "state": "passed" if admitted else "failed",
            "maximum_residual": successor_threshold,
            "path_residuals": {probe_id: residual},
            "supported_local_paths": [probe_id] if admitted else [],
            "cross_backbone_transfer_eligible": False,
            "failure_policy": "require_independent_accept_settlement",
        },
        "derivation": {
            "type": "dev_only_directional_subset",
            "source_campaign_id": source_campaign["campaign_id"],
            "source_fingerprint_sha256": report["source_fingerprint_sha256"],
            "source_measurements_sha256": report["source_measurements_sha256"],
        },
        "claim_boundary": report["claim_boundary"],
    }
    return write_bundle(
        output_root=output_root,
        files={
            "directional-probe-evolution.json": canonical_json(report),
            "selected-dev-fingerprint.json": canonical_json(selected_fingerprint),
            "input-successor-campaign.json": canonical_json(successor),
        },
        manifest_fields={
            "artifact_type": "verdiwm-cosmos3-directional-probe-evolution-manifest",
            "state": report["state"],
            "successor_campaign_id": successor["campaign_id"],
            "dev_locality_admitted": admitted,
            "selected_directional_residual": residual,
            "report_path": str(Path(output_root).resolve() / "directional-probe-evolution.json"),
        },
    )


def _load_json(path: Path) -> Mapping[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise Cosmos3DirectionalProbeError(f"COSMOS3_DIRECTIONAL_JSON_INVALID:{path}")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
