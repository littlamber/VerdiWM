"""Settle Ctrl-World locality-radius calibration without overstating transfer."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from wmloop.experiments._artifacts import canonical_json, write_bundle


class CtrlWorldFingerprintSettlementError(ValueError):
    """Fingerprint candidates cannot be settled under one protocol."""


def settle_ctrl_world_fingerprints(
    *,
    fingerprint_roots: Sequence[Path],
    protocol: str,
    output_root: Path,
) -> dict[str, object]:
    """Select the widest admitted local chart, or settle an abstention."""
    if not fingerprint_roots:
        raise CtrlWorldFingerprintSettlementError("CTRL_WORLD_FINGERPRINT_CANDIDATES_EMPTY")
    candidates = [_load_candidate(Path(root), protocol=protocol) for root in fingerprint_roots]
    campaign_ids = [str(candidate["campaign_id"]) for candidate in candidates]
    if len(set(campaign_ids)) != len(campaign_ids):
        raise CtrlWorldFingerprintSettlementError("CTRL_WORLD_FINGERPRINT_CAMPAIGN_DUPLICATE")
    passed = [candidate for candidate in candidates if candidate["locality_state"] == "passed"]
    selected = max(
        passed,
        key=lambda candidate: (float(candidate["dose_radius"]), -float(candidate["maximum_residual"])),
        default=None,
    )
    settled_state = "settled_admitted" if selected is not None else "settled_abstained"
    report = {
        "schema_version": 1,
        "artifact_type": "verdiwm-ctrl-world-fingerprint-settlement",
        "state": settled_state,
        "protocol": protocol,
        "selection_policy": "widest_locality_admitted_radius_then_lowest_residual",
        "selected_campaign_id": selected["campaign_id"] if selected is not None else None,
        "selected_dose_radius": selected["dose_radius"] if selected is not None else None,
        "cross_backbone_transfer_eligible": selected is not None,
        "candidates": candidates,
        "claim_boundary": (
            "This settles a target-local Ctrl-World ACWM response chart only. Admission permits later "
            "selector experiments; it is not model-improvement or cross-backbone transfer evidence."
        ),
    }
    destination = Path(output_root).resolve()
    return write_bundle(
        output_root=destination,
        files={
            "settlement.json": canonical_json(report),
            "candidates.jsonl": b"".join(canonical_json(candidate) for candidate in candidates),
        },
        manifest_fields={
            "artifact_type": "verdiwm-ctrl-world-fingerprint-settlement-manifest",
            "state": settled_state,
            "protocol": protocol,
            "candidate_count": len(candidates),
            "selected_campaign_id": report["selected_campaign_id"],
            "cross_backbone_transfer_eligible": report["cross_backbone_transfer_eligible"],
            "report_path": str(destination / "settlement.json"),
        },
    )


def _load_candidate(root: Path, *, protocol: str) -> dict[str, object]:
    resolved = root.resolve(strict=True)
    manifest_path = resolved / "manifest.json"
    report_path = resolved / "target-local-fingerprint.json"
    campaign_path = resolved / "input-campaign.json"
    manifest = _load_mapping(manifest_path, "CTRL_WORLD_FINGERPRINT_MANIFEST_INVALID")
    report = _load_mapping(report_path, "CTRL_WORLD_FINGERPRINT_REPORT_INVALID")
    campaign = _load_mapping(campaign_path, "CTRL_WORLD_FINGERPRINT_CAMPAIGN_INVALID")
    if manifest.get("artifact_type") != "verdiwm-ctrl-world-target-local-fingerprint-manifest":
        raise CtrlWorldFingerprintSettlementError("CTRL_WORLD_FINGERPRINT_MANIFEST_TYPE_INVALID")
    if report.get("artifact_type") != "verdiwm-ctrl-world-target-local-fingerprint":
        raise CtrlWorldFingerprintSettlementError("CTRL_WORLD_FINGERPRINT_REPORT_TYPE_INVALID")
    if campaign.get("artifact_type") != "verdiwm-ctrl-world-fingerprint-campaign":
        raise CtrlWorldFingerprintSettlementError("CTRL_WORLD_FINGERPRINT_CAMPAIGN_TYPE_INVALID")
    campaign_id = str(campaign.get("campaign_id"))
    if (
        manifest.get("campaign_id") != campaign_id
        or report.get("campaign_id") != campaign_id
        or manifest.get("protocol") != protocol
        or report.get("protocol") != protocol
    ):
        raise CtrlWorldFingerprintSettlementError("CTRL_WORLD_FINGERPRINT_SETTLEMENT_CONTRACT_MISMATCH")
    locality = report.get("locality_admission")
    if not isinstance(locality, Mapping) or locality.get("state") not in {"passed", "failed"}:
        raise CtrlWorldFingerprintSettlementError("CTRL_WORLD_FINGERPRINT_LOCALITY_INVALID")
    residuals = locality.get("path_residuals")
    if not isinstance(residuals, Mapping) or not residuals:
        raise CtrlWorldFingerprintSettlementError("CTRL_WORLD_FINGERPRINT_RESIDUALS_INVALID")
    residual_values = [float(value) for value in residuals.values()]
    if any(not math.isfinite(value) or value < 0.0 for value in residual_values):
        raise CtrlWorldFingerprintSettlementError("CTRL_WORLD_FINGERPRINT_RESIDUAL_INVALID")
    doses = tuple(float(value) for value in campaign.get("probe", {}).get("doses", ()))
    if not doses or 0.0 not in doses:
        raise CtrlWorldFingerprintSettlementError("CTRL_WORLD_FINGERPRINT_DOSES_INVALID")
    expected_measurements = len(doses) * int(campaign["protocols"][protocol]["required_receipts_per_dose"])
    if int(report.get("measurement_count", -1)) != expected_measurements:
        raise CtrlWorldFingerprintSettlementError("CTRL_WORLD_FINGERPRINT_MEASUREMENT_COUNT_INVALID")
    locality_state = str(locality["state"])
    if manifest.get("locality_admission_state") != locality_state:
        raise CtrlWorldFingerprintSettlementError("CTRL_WORLD_FINGERPRINT_LOCALITY_STATE_MISMATCH")
    return {
        "campaign_id": campaign_id,
        "probe_id": campaign["probe"]["probe_id"],
        "doses": list(doses),
        "dose_radius": max(abs(value) for value in doses),
        "measurement_count": expected_measurements,
        "locality_state": locality_state,
        "maximum_residual": max(residual_values),
        "locality_threshold": float(locality["maximum_residual"]),
        "supported_local_paths": list(locality.get("supported_local_paths", ())),
        "fingerprint_root": str(resolved),
        "manifest_sha256": _sha256(manifest_path),
        "report_sha256": _sha256(report_path),
    }


def _load_mapping(path: Path, code: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CtrlWorldFingerprintSettlementError(f"{code}:{path}") from exc
    if not isinstance(payload, Mapping):
        raise CtrlWorldFingerprintSettlementError(f"{code}:{path}")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
