"""Fit a target-local Cosmos3 fingerprint from paired-dose receipts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from wmloop.evaluate.adapters.cosmos3_predictive import evaluate_cosmos3_prediction_receipt
from wmloop.experiments._artifacts import canonical_json, write_bundle
from wmloop.geometry.irg import estimate_response_chart


class Cosmos3FingerprintError(ValueError):
    """Cosmos3 paired-dose evidence is incomplete or inconsistent."""


def fit_cosmos3_fingerprint(
    *,
    campaign_path: Path,
    shard_manifests: Sequence[Path],
    split_path: Path,
    protocol: str,
    output_root: Path,
) -> dict[str, object]:
    campaign = _load_json(campaign_path)
    if campaign.get("artifact_type") != "verdiwm-cosmos3-fingerprint-campaign":
        raise Cosmos3FingerprintError("COSMOS3_FINGERPRINT_CAMPAIGN_TYPE_INVALID")
    protocols = campaign.get("protocols", {})
    if protocol not in protocols:
        raise Cosmos3FingerprintError("COSMOS3_FINGERPRINT_PROTOCOL_INVALID")
    split_name = str(protocols[protocol]["split"])
    split = _load_json(split_path)
    expected_identities = {
        (int(row["sample_index"]), int(row["seed"])) for row in split[split_name]
    }
    doses = tuple(float(value) for value in campaign["probe"]["doses"])
    probe_id = str(campaign["probe"]["probe_id"])
    by_key: dict[tuple[float, int, int], tuple[float, ...]] = {}
    evidence_rows: list[dict[str, object]] = []
    for manifest_path in shard_manifests:
        shard = _load_json(manifest_path)
        if (
            shard.get("artifact_type") != "verdiwm-cosmos3-fingerprint-campaign-shard"
            or shard.get("state") != "ready"
            or shard.get("campaign_id") != campaign["campaign_id"]
            or shard.get("protocol") != protocol
        ):
            raise Cosmos3FingerprintError("COSMOS3_FINGERPRINT_SHARD_INVALID")
        for row in shard.get("records", []):
            gpu_audit = row.get("gpu_exclusivity_audit")
            if (
                not isinstance(gpu_audit, Mapping)
                or gpu_audit.get("artifact_type")
                != "verdiwm-cosmos3-gpu-exclusivity-audit"
                or gpu_audit.get("state") != "ready"
                or gpu_audit.get("foreign_pid_events") != []
                or int(gpu_audit.get("sample_count", 0)) < 1
                or not str(gpu_audit.get("gpu_uuid", "")).startswith("GPU-")
            ):
                raise Cosmos3FingerprintError("COSMOS3_FINGERPRINT_GPU_AUDIT_INVALID")
            dose = float(row["dose"])
            identity = (int(row["sample_index"]), int(row["seed"]))
            key = (dose, *identity)
            if dose not in doses or identity not in expected_identities or key in by_key:
                raise Cosmos3FingerprintError("COSMOS3_FINGERPRINT_RECORD_IDENTITY_INVALID")
            receipt_path = Path(str(row["receipt_ref"])).resolve(strict=True)
            if _sha256(receipt_path) != row["receipt_sha256"]:
                raise Cosmos3FingerprintError("COSMOS3_FINGERPRINT_RECEIPT_SHA_MISMATCH")
            evidence = evaluate_cosmos3_prediction_receipt(
                receipt_path=receipt_path,
                heldout_split_path=split_path,
                split_name=split_name,
            )
            intervention = evidence.get("intervention")
            if (
                not isinstance(intervention, Mapping)
                or float(intervention.get("dose", float("nan"))) != dose
                or intervention.get("probe_id") != probe_id
            ):
                raise Cosmos3FingerprintError("COSMOS3_FINGERPRINT_RECEIPT_INTERVENTION_MISMATCH")
            receipt = _load_json(receipt_path)
            _verify_intervention_ref(receipt_path.parent, receipt)
            values = tuple(
                float(evidence["metrics"][str(outcome["source_metric"])])
                * float(outcome["sign"])
                for outcome in campaign["outcomes"]
            )
            by_key[key] = values
            evidence_rows.append(
                {
                    "schema_version": 1,
                    "artifact_type": "verdiwm-cosmos3-fingerprint-measurement",
                    "campaign_id": campaign["campaign_id"],
                    "protocol": protocol,
                    "dose": dose,
                    "sample_index": identity[0],
                    "seed": identity[1],
                    "outcomes": list(values),
                    "receipt_sha256": row["receipt_sha256"],
                    "gpu_uuid": gpu_audit["gpu_uuid"],
                    "gpu_audit_sample_count": gpu_audit["sample_count"],
                }
            )
    expected_keys = {(dose, *identity) for dose in doses for identity in expected_identities}
    if set(by_key) != expected_keys:
        missing = sorted(expected_keys - set(by_key))
        raise Cosmos3FingerprintError(f"COSMOS3_FINGERPRINT_INCOMPLETE:{missing}")
    identities = tuple(sorted(expected_identities))
    chart = estimate_response_chart(
        chart_id=f"{campaign['campaign_id']}:{protocol}",
        goal_schema="cosmos3_acwm_forward_dynamics_v1",
        outcome_names=tuple(str(item["name"]) for item in campaign["outcomes"]),
        outcome_weights=tuple(float(item["weight"]) for item in campaign["outcomes"]),
        baseline_repeats=tuple(by_key[(0.0, *identity)] for identity in identities),
        dose_observations={
            str(campaign["probe"]["probe_id"]): {
                dose: tuple(by_key[(dose, *identity)] for identity in identities)
                for dose in doses
                if dose != 0.0
            }
        },
    ).to_dict()
    threshold = float(campaign["locality_admission"]["maximum_residual"])
    residuals = {str(key): float(value) for key, value in chart["locality_residuals"].items()}
    supported = sorted(name for name, value in residuals.items() if value <= threshold)
    report: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "verdiwm-cosmos3-target-local-fingerprint",
        "state": "ready",
        "campaign_id": campaign["campaign_id"],
        "protocol": protocol,
        "split": split_name,
        "measurement_count": len(evidence_rows),
        "repeat_count": len(identities),
        "chart": chart,
        "locality_admission": {
            "state": "passed" if supported else "failed",
            "maximum_residual": threshold,
            "path_residuals": residuals,
            "supported_local_paths": supported,
            "cross_backbone_transfer_eligible": bool(supported),
            "failure_policy": campaign["locality_admission"]["failure_policy"],
        },
        "claim_boundary": campaign["claim_scope"],
    }
    destination = Path(output_root).resolve()
    return write_bundle(
        output_root=destination,
        files={
            "target-local-fingerprint.json": canonical_json(report),
            "measurements.jsonl": b"".join(canonical_json(row) for row in evidence_rows),
            "input-campaign.json": canonical_json(campaign),
        },
        manifest_fields={
            "artifact_type": "verdiwm-cosmos3-target-local-fingerprint-manifest",
            "state": "ready",
            "campaign_id": campaign["campaign_id"],
            "protocol": protocol,
            "split": split_name,
            "measurement_count": len(evidence_rows),
            "repeat_count": len(identities),
            "locality_admission_state": report["locality_admission"]["state"],
            "cross_backbone_transfer_eligible": bool(supported),
            "report_path": str(destination / "target-local-fingerprint.json"),
        },
    )


def _verify_intervention_ref(root: Path, receipt: Mapping[str, Any]) -> None:
    ref = receipt.get("intervention_ref")
    intervention = receipt.get("intervention")
    if not isinstance(ref, str) or not isinstance(intervention, Mapping):
        raise Cosmos3FingerprintError("COSMOS3_FINGERPRINT_INTERVENTION_REF_MISSING")
    relative = Path(ref)
    if relative.is_absolute() or ".." in relative.parts:
        raise Cosmos3FingerprintError("COSMOS3_FINGERPRINT_INTERVENTION_REF_NONPORTABLE")
    path = (root / relative).resolve(strict=True)
    if root.resolve() not in path.parents or _sha256(path) != intervention["hook_receipt_sha256"]:
        raise Cosmos3FingerprintError("COSMOS3_FINGERPRINT_INTERVENTION_REF_SHA_MISMATCH")


def _load_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise Cosmos3FingerprintError(f"COSMOS3_FINGERPRINT_JSON_INVALID:{path}")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
