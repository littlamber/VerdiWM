"""Recover completed Cosmos3 records from an interrupted campaign shard."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

from wmloop.evaluate.adapters.cosmos3_predictive import evaluate_cosmos3_prediction_receipt


class Cosmos3ShardRecoveryError(ValueError):
    """A completed record cannot be admitted as a recovered ready shard."""


def recover_cosmos3_shard(
    *,
    campaign_path: Path,
    source_manifest_path: Path,
    split_path: Path,
    protocol: str,
    doses: Sequence[float],
    output_path: Path,
) -> dict[str, object]:
    campaign = _load_json(campaign_path)
    source = _load_json(source_manifest_path)
    split = _load_json(split_path)
    selected_doses = tuple(float(value) for value in doses)
    frozen_doses = {float(value) for value in campaign.get("probe", {}).get("doses", [])}
    if (
        campaign.get("artifact_type") != "verdiwm-cosmos3-fingerprint-campaign"
        or source.get("artifact_type") != "verdiwm-cosmos3-fingerprint-campaign-shard"
        or source.get("campaign_id") != campaign.get("campaign_id")
        or source.get("protocol") != protocol
        or protocol not in campaign.get("protocols", {})
    ):
        raise Cosmos3ShardRecoveryError("COSMOS3_SHARD_RECOVERY_SOURCE_INVALID")
    if not selected_doses or len(set(selected_doses)) != len(selected_doses):
        raise Cosmos3ShardRecoveryError("COSMOS3_SHARD_RECOVERY_DOSES_INVALID")
    if any(dose not in frozen_doses for dose in selected_doses):
        raise Cosmos3ShardRecoveryError("COSMOS3_SHARD_RECOVERY_DOSE_OUTSIDE_FREEZE")

    split_name = str(campaign["protocols"][protocol]["split"])
    frozen_identities = {
        (int(row["sample_index"]), int(row["seed"])) for row in split[split_name]
    }
    identities = tuple(
        sorted(
            (int(row["sample_index"]), int(row["seed"]))
            for row in source.get("identities", [])
        )
    )
    if not identities or any(identity not in frozen_identities for identity in identities):
        raise Cosmos3ShardRecoveryError("COSMOS3_SHARD_RECOVERY_IDENTITY_INVALID")

    expected_keys = {(dose, *identity) for dose in selected_doses for identity in identities}
    recovered: dict[tuple[float, int, int], dict[str, object]] = {}
    for raw_row in source.get("records", []):
        row = dict(raw_row)
        key = (float(row["dose"]), int(row["sample_index"]), int(row["seed"]))
        if key not in expected_keys:
            continue
        if key in recovered:
            raise Cosmos3ShardRecoveryError("COSMOS3_SHARD_RECOVERY_DUPLICATE_RECORD")
        _validate_record(
            row,
            split_path=split_path,
            split_name=split_name,
            probe_id=str(campaign["probe"]["probe_id"]),
        )
        recovered[key] = row
    if set(recovered) != expected_keys:
        raise Cosmos3ShardRecoveryError(
            f"COSMOS3_SHARD_RECOVERY_INCOMPLETE:{sorted(expected_keys - set(recovered))}"
        )

    destination = Path(output_path).resolve()
    if destination.exists() or destination.is_symlink():
        raise Cosmos3ShardRecoveryError("COSMOS3_SHARD_RECOVERY_OUTPUT_EXISTS")
    manifest: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "verdiwm-cosmos3-fingerprint-campaign-shard",
        "state": "ready",
        "campaign_id": campaign["campaign_id"],
        "protocol": protocol,
        "split": split_name,
        "probe_id": campaign["probe"]["probe_id"],
        "doses": list(selected_doses),
        "identities": [
            {"sample_index": sample, "seed": seed} for sample, seed in identities
        ],
        "expected_receipt_count": len(expected_keys),
        "receipt_count": len(recovered),
        "records": [recovered[key] for key in sorted(recovered)],
        "recovery": {
            "source_manifest_ref": str(Path(source_manifest_path).resolve()),
            "source_manifest_sha256": _sha256(Path(source_manifest_path)),
            "source_state": source.get("state"),
            "policy": "admit_only_completed_records_after_receipt_hook_gpu_and_split_revalidation",
        },
        "claim_boundary": source.get(
            "claim_boundary",
            "Recovered campaign execution evidence only; no transfer or model-improvement claim.",
        ),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_name, destination)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return manifest


def _validate_record(
    row: Mapping[str, object],
    *,
    split_path: Path,
    split_name: str,
    probe_id: str,
) -> None:
    audit = row.get("gpu_exclusivity_audit")
    if (
        not isinstance(audit, Mapping)
        or audit.get("artifact_type") != "verdiwm-cosmos3-gpu-exclusivity-audit"
        or audit.get("state") != "ready"
        or audit.get("foreign_pid_events") != []
        or int(audit.get("sample_count", 0)) < 1
        or not str(audit.get("gpu_uuid", "")).startswith("GPU-")
    ):
        raise Cosmos3ShardRecoveryError("COSMOS3_SHARD_RECOVERY_GPU_AUDIT_INVALID")
    receipt_path = Path(str(row.get("receipt_ref", ""))).resolve(strict=True)
    if _sha256(receipt_path) != row.get("receipt_sha256"):
        raise Cosmos3ShardRecoveryError("COSMOS3_SHARD_RECOVERY_RECEIPT_SHA_MISMATCH")
    evidence = evaluate_cosmos3_prediction_receipt(
        receipt_path=receipt_path,
        heldout_split_path=split_path,
        split_name=split_name,
    )
    intervention = evidence.get("intervention")
    if (
        not isinstance(intervention, Mapping)
        or intervention.get("probe_id") != probe_id
        or float(intervention.get("dose", float("nan"))) != float(row["dose"])
    ):
        raise Cosmos3ShardRecoveryError("COSMOS3_SHARD_RECOVERY_INTERVENTION_MISMATCH")
    if evidence.get("metrics") != row.get("metrics"):
        raise Cosmos3ShardRecoveryError("COSMOS3_SHARD_RECOVERY_METRICS_MISMATCH")
    receipt = _load_json(receipt_path)
    hook_ref = receipt.get("intervention_ref")
    if not isinstance(hook_ref, str):
        raise Cosmos3ShardRecoveryError("COSMOS3_SHARD_RECOVERY_HOOK_MISSING")
    relative = Path(hook_ref)
    if relative.is_absolute() or ".." in relative.parts:
        raise Cosmos3ShardRecoveryError("COSMOS3_SHARD_RECOVERY_HOOK_NONPORTABLE")
    hook_path = (receipt_path.parent / relative).resolve(strict=True)
    if (
        receipt_path.parent.resolve() not in hook_path.parents
        or _sha256(hook_path) != intervention.get("hook_receipt_sha256")
    ):
        raise Cosmos3ShardRecoveryError("COSMOS3_SHARD_RECOVERY_HOOK_SHA_MISMATCH")


def _load_json(path: Path) -> Mapping[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise Cosmos3ShardRecoveryError(f"COSMOS3_SHARD_RECOVERY_JSON_INVALID:{path}")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
