"""Settle a dev-selected Cosmos3 directional probe on an independent split."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from wmloop.experiments._artifacts import canonical_json, write_bundle


class Cosmos3DirectionalSettlementError(ValueError):
    """Directional dev and accept fingerprints violate one frozen contract."""


def settle_cosmos3_directional_probe(
    *,
    dev_selection_root: Path,
    accept_fingerprint_root: Path,
    output_root: Path,
) -> dict[str, object]:
    dev_root = Path(dev_selection_root).resolve(strict=True)
    accept_root = Path(accept_fingerprint_root).resolve(strict=True)
    dev_manifest_path = dev_root / "manifest.json"
    dev_report_path = dev_root / "directional-probe-evolution.json"
    dev_fingerprint_path = dev_root / "selected-dev-fingerprint.json"
    dev_campaign_path = dev_root / "input-successor-campaign.json"
    accept_manifest_path = accept_root / "manifest.json"
    accept_fingerprint_path = accept_root / "target-local-fingerprint.json"
    accept_campaign_path = accept_root / "input-campaign.json"

    dev_manifest = _load_mapping(dev_manifest_path)
    dev_report = _load_mapping(dev_report_path)
    dev_fingerprint = _load_mapping(dev_fingerprint_path)
    dev_campaign = _load_mapping(dev_campaign_path)
    accept_manifest = _load_mapping(accept_manifest_path)
    accept_fingerprint = _load_mapping(accept_fingerprint_path)
    accept_campaign = _load_mapping(accept_campaign_path)

    if _sha256(dev_campaign_path) != _sha256(accept_campaign_path):
        raise Cosmos3DirectionalSettlementError("COSMOS3_DIRECTIONAL_CAMPAIGN_SHA_MISMATCH")
    campaign_id = str(dev_campaign.get("campaign_id", ""))
    if (
        dev_campaign.get("artifact_type") != "verdiwm-cosmos3-fingerprint-campaign"
        or campaign_id != accept_campaign.get("campaign_id")
        or dev_manifest.get("artifact_type")
        != "verdiwm-cosmos3-directional-probe-evolution-manifest"
        or dev_manifest.get("state") != "dev_selected"
        or dev_report.get("state") != "dev_selected"
        or dev_fingerprint.get("artifact_type")
        != "verdiwm-cosmos3-target-local-fingerprint"
        or dev_fingerprint.get("campaign_id") != campaign_id
        or dev_fingerprint.get("protocol") != "pilot"
        or dev_fingerprint.get("split") != "dev"
        or accept_manifest.get("artifact_type")
        != "verdiwm-cosmos3-target-local-fingerprint-manifest"
        or accept_manifest.get("state") != "ready"
        or accept_fingerprint.get("artifact_type")
        != "verdiwm-cosmos3-target-local-fingerprint"
        or accept_fingerprint.get("campaign_id") != campaign_id
        or accept_fingerprint.get("protocol") != "paper"
        or accept_fingerprint.get("split") != "accept"
    ):
        raise Cosmos3DirectionalSettlementError("COSMOS3_DIRECTIONAL_SETTLEMENT_INPUT_INVALID")

    dev_chart = _mapping(dev_fingerprint.get("chart"), "COSMOS3_DIRECTIONAL_DEV_CHART_INVALID")
    accept_chart = _mapping(
        accept_fingerprint.get("chart"), "COSMOS3_DIRECTIONAL_ACCEPT_CHART_INVALID"
    )
    for field in ("goal_schema", "outcome_names", "intervention_names"):
        if dev_chart.get(field) != accept_chart.get(field):
            raise Cosmos3DirectionalSettlementError(
                f"COSMOS3_DIRECTIONAL_CHART_COORDINATE_MISMATCH:{field}"
            )
    probe_id = str(dev_campaign.get("probe", {}).get("probe_id", ""))
    if dev_chart.get("intervention_names") != [probe_id]:
        raise Cosmos3DirectionalSettlementError("COSMOS3_DIRECTIONAL_PROBE_MISMATCH")

    dev_jacobian = _flatten_matrix(dev_chart.get("jacobian"), "DEV")
    accept_jacobian = _flatten_matrix(accept_chart.get("jacobian"), "ACCEPT")
    if len(dev_jacobian) != len(accept_jacobian):
        raise Cosmos3DirectionalSettlementError("COSMOS3_DIRECTIONAL_JACOBIAN_SHAPE_MISMATCH")
    alignment_error = _normalized_alignment_error(dev_jacobian, accept_jacobian)

    acceptance = _mapping(
        dev_campaign.get("acceptance"), "COSMOS3_DIRECTIONAL_ACCEPTANCE_POLICY_INVALID"
    )
    maximum_alignment_error = _finite_nonnegative(
        acceptance.get("maximum_dev_accept_alignment_error"),
        "COSMOS3_DIRECTIONAL_ALIGNMENT_THRESHOLD_INVALID",
    )
    require_accept_locality = acceptance.get("require_accept_locality_admission") is True
    dev_locality = _mapping(
        dev_fingerprint.get("locality_admission"), "COSMOS3_DIRECTIONAL_DEV_LOCALITY_INVALID"
    )
    accept_locality = _mapping(
        accept_fingerprint.get("locality_admission"),
        "COSMOS3_DIRECTIONAL_ACCEPT_LOCALITY_INVALID",
    )
    terms = {
        "dev_locality_admitted": dev_locality.get("state") == "passed",
        "accept_locality_admitted": (
            accept_locality.get("state") == "passed" if require_accept_locality else True
        ),
        "dev_accept_jacobian_aligned": alignment_error <= maximum_alignment_error,
    }
    licensed = all(terms.values())
    state = "settled_licensed" if licensed else "settled_abstained"
    report = {
        "schema_version": 1,
        "artifact_type": "verdiwm-cosmos3-directional-probe-settlement",
        "state": state,
        "campaign_id": campaign_id,
        "probe_id": probe_id,
        "decision": (
            "license_directional_probe_for_later_transfer_experiments"
            if licensed
            else "retain_counterexample_and_abstain_from_transfer"
        ),
        "cross_backbone_transfer_eligible": licensed,
        "terms": terms,
        "abstention_reasons": [name for name, passed in terms.items() if not passed],
        "alignment": {
            "metric": "l2_distance_between_unit_frobenius_jacobians",
            "error": alignment_error,
            "maximum_error": maximum_alignment_error,
            "dev_jacobian": dev_jacobian,
            "accept_jacobian": accept_jacobian,
        },
        "locality": {
            "maximum_residual": float(dev_campaign["locality_admission"]["maximum_residual"]),
            "dev_state": dev_locality.get("state"),
            "dev_residual": float(dev_locality["path_residuals"][probe_id]),
            "accept_state": accept_locality.get("state"),
            "accept_residual": float(accept_locality["path_residuals"][probe_id]),
        },
        "inputs": {
            "dev_selection_manifest_sha256": _sha256(dev_manifest_path),
            "dev_fingerprint_sha256": _sha256(dev_fingerprint_path),
            "accept_manifest_sha256": _sha256(accept_manifest_path),
            "accept_fingerprint_sha256": _sha256(accept_fingerprint_path),
            "campaign_sha256": _sha256(dev_campaign_path),
        },
        "claim_boundary": (
            "This settlement tests whether one dev-selected diagnostic direction is stable on the "
            "independent accept split. It is not evidence of model improvement. A licensed result "
            "would permit, but not prove, a later cross-backbone transfer experiment."
        ),
    }
    destination = Path(output_root).resolve()
    return write_bundle(
        output_root=destination,
        files={"directional-probe-settlement.json": canonical_json(report)},
        manifest_fields={
            "artifact_type": "verdiwm-cosmos3-directional-probe-settlement-manifest",
            "state": state,
            "campaign_id": campaign_id,
            "probe_id": probe_id,
            "alignment_error": alignment_error,
            "maximum_alignment_error": maximum_alignment_error,
            "cross_backbone_transfer_eligible": licensed,
            "report_path": str(destination / "directional-probe-settlement.json"),
        },
    )


def _normalized_alignment_error(left: Sequence[float], right: Sequence[float]) -> float:
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm <= 0.0 or right_norm <= 0.0:
        raise Cosmos3DirectionalSettlementError("COSMOS3_DIRECTIONAL_JACOBIAN_NORM_ZERO")
    return math.sqrt(
        sum(
            (left_value / left_norm - right_value / right_norm) ** 2
            for left_value, right_value in zip(left, right, strict=True)
        )
    )


def _flatten_matrix(value: object, label: str) -> tuple[float, ...]:
    if not isinstance(value, list) or not value or any(not isinstance(row, list) for row in value):
        raise Cosmos3DirectionalSettlementError(f"COSMOS3_DIRECTIONAL_{label}_JACOBIAN_INVALID")
    rows = [row for row in value if isinstance(row, list)]
    widths = {len(row) for row in rows}
    if len(widths) != 1 or 0 in widths:
        raise Cosmos3DirectionalSettlementError(f"COSMOS3_DIRECTIONAL_{label}_JACOBIAN_INVALID")
    return tuple(
        _finite(number, f"COSMOS3_DIRECTIONAL_{label}_JACOBIAN_INVALID")
        for row in rows
        for number in row
    )


def _mapping(value: object, code: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Cosmos3DirectionalSettlementError(code)
    return value


def _load_mapping(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Cosmos3DirectionalSettlementError(f"COSMOS3_DIRECTIONAL_JSON_INVALID:{path}") from exc
    return _mapping(payload, f"COSMOS3_DIRECTIONAL_JSON_INVALID:{path}")


def _finite(value: object, code: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise Cosmos3DirectionalSettlementError(code) from exc
    if not math.isfinite(number):
        raise Cosmos3DirectionalSettlementError(code)
    return number


def _finite_nonnegative(value: object, code: str) -> float:
    number = _finite(value, code)
    if number < 0.0:
        raise Cosmos3DirectionalSettlementError(code)
    return number


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
