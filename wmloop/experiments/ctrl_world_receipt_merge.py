"""Merge sharded Ctrl-World fingerprint receipts under the frozen contract."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from wmloop.evaluate.adapters.ctrl_world_predictive import evaluate_ctrl_world_prediction_receipt
from wmloop.experiments._artifacts import canonical_json
from wmloop.experiments.ctrl_world_fingerprint import load_ctrl_world_campaign


class CtrlWorldReceiptMergeError(ValueError):
    """Receipt shards cannot form one complete paired-dose campaign."""


def merge_ctrl_world_receipt_indexes(
    *,
    campaign_path: Path,
    heldout_split_path: Path,
    protocol: str,
    receipt_index_paths: Sequence[Path],
    output_root: Path,
) -> dict[str, object]:
    """Validate, copy, and atomically merge complete receipt shards."""
    if not receipt_index_paths:
        raise CtrlWorldReceiptMergeError("CTRL_WORLD_RECEIPT_SHARDS_EMPTY")
    campaign = load_ctrl_world_campaign(Path(campaign_path))
    protocols = campaign["protocols"]
    if protocol not in protocols:
        raise CtrlWorldReceiptMergeError("CTRL_WORLD_RECEIPT_PROTOCOL_INVALID")
    split_name = str(protocols[protocol]["split"])
    required = int(protocols[protocol]["required_receipts_per_dose"])
    doses = tuple(float(value) for value in campaign["probe"]["doses"])
    dose_order = {dose: index for index, dose in enumerate(doses)}

    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise CtrlWorldReceiptMergeError("CTRL_WORLD_RECEIPT_MERGE_OUTPUT_EXISTS")
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    source_indexes: list[dict[str, object]] = []
    merged: dict[tuple[float, str, str, int], dict[str, object]] = {}
    try:
        temporary.mkdir(mode=0o700, parents=True)
        receipt_dir = temporary / "receipts"
        receipt_dir.mkdir(mode=0o700)
        for index_path_value in receipt_index_paths:
            index_path = Path(index_path_value).resolve(strict=True)
            index_bytes = index_path.read_bytes()
            index = _load_mapping(index_path, "CTRL_WORLD_RECEIPT_INDEX_INVALID")
            if index.get("artifact_type") != "verdiwm-ctrl-world-fingerprint-receipt-index":
                raise CtrlWorldReceiptMergeError("CTRL_WORLD_RECEIPT_INDEX_TYPE_INVALID")
            if index.get("campaign_id") != campaign["campaign_id"] or index.get("protocol") != protocol:
                raise CtrlWorldReceiptMergeError("CTRL_WORLD_RECEIPT_INDEX_CONTRACT_MISMATCH")
            rows = index.get("rows")
            if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
                raise CtrlWorldReceiptMergeError("CTRL_WORLD_RECEIPT_INDEX_ROWS_INVALID")
            source_indexes.append(
                {
                    "path": str(index_path),
                    "sha256": hashlib.sha256(index_bytes).hexdigest(),
                    "row_count": len(rows),
                }
            )
            for row in rows:
                dose = float(row["dose"])
                if dose not in dose_order:
                    raise CtrlWorldReceiptMergeError("CTRL_WORLD_RECEIPT_DOSE_UNKNOWN")
                receipt_path = Path(str(row["receipt_ref"])).resolve(strict=True)
                receipt = _load_mapping(receipt_path, "CTRL_WORLD_RECEIPT_INVALID")
                evidence = evaluate_ctrl_world_prediction_receipt(
                    receipt_path=receipt_path,
                    heldout_split_path=Path(heldout_split_path),
                    split_name=split_name,
                )
                identity = (
                    str(evidence["task_id"]),
                    str(evidence["episode_id"]),
                    int(evidence["seed"]),
                )
                key = (dose, *identity)
                if key in merged:
                    raise CtrlWorldReceiptMergeError("CTRL_WORLD_RECEIPT_DUPLICATE")
                filename = (
                    f"dose_{_dose_tag(dose)}__{identity[0]}__{identity[1]}__s{identity[2]}.json"
                )
                payload = canonical_json(receipt)
                (receipt_dir / filename).write_bytes(payload)
                merged[key] = {
                    "dose": dose,
                    "filename": filename,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "identity": {
                        "task_id": identity[0],
                        "episode_id": identity[1],
                        "seed": identity[2],
                    },
                }

        identities_by_dose = {
            dose: {(key[1], key[2], key[3]) for key in merged if key[0] == dose} for dose in doses
        }
        baseline_identities = identities_by_dose[0.0]
        if len(baseline_identities) != required:
            raise CtrlWorldReceiptMergeError("CTRL_WORLD_RECEIPT_BASELINE_FRAME_INVALID")
        for dose in doses:
            if identities_by_dose[dose] != baseline_identities:
                raise CtrlWorldReceiptMergeError(f"CTRL_WORLD_RECEIPT_PAIRED_FRAME_INVALID:{dose}")

        ordered = sorted(
            merged.items(),
            key=lambda item: (dose_order[item[0][0]], item[0][1], item[0][2], item[0][3]),
        )
        final_receipt_dir = destination / "receipts"
        rows = [
            {
                "dose": key[0],
                "receipt_ref": str(final_receipt_dir / str(value["filename"])),
            }
            for key, value in ordered
        ]
        index = {
            "artifact_type": "verdiwm-ctrl-world-fingerprint-receipt-index",
            "campaign_id": campaign["campaign_id"],
            "protocol": protocol,
            "rows": rows,
        }
        index_payload = canonical_json(index)
        (temporary / "receipt-index.json").write_bytes(index_payload)
        manifest = {
            "schema_version": 1,
            "artifact_type": "verdiwm-ctrl-world-receipt-merge-manifest",
            "state": "complete",
            "campaign_id": campaign["campaign_id"],
            "protocol": protocol,
            "split": split_name,
            "configured_doses": list(doses),
            "receipt_count": len(rows),
            "repeat_count": len(baseline_identities),
            "source_indexes": source_indexes,
            "receipt_index_sha256": hashlib.sha256(index_payload).hexdigest(),
            "receipts": [value for _, value in ordered],
            "claim_boundary": campaign["claim_scope"],
        }
        (temporary / "manifest.json").write_bytes(canonical_json(manifest))
        os.replace(temporary, destination)
        return {
            **manifest,
            "manifest_path": str(destination / "manifest.json"),
            "receipt_index": str(destination / "receipt-index.json"),
        }
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _load_mapping(path: Path, code: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CtrlWorldReceiptMergeError(f"{code}:{path}") from exc
    if not isinstance(payload, Mapping):
        raise CtrlWorldReceiptMergeError(f"{code}:{path}")
    return payload


def _dose_tag(dose: float) -> str:
    sign = "p" if dose >= 0.0 else "m"
    return f"{sign}{abs(dose):0.4f}".replace(".", "d")
