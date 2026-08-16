#!/usr/bin/env python3
"""Settle whether frozen Ctrl-World direction labels license candidate training."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


class DirectionalDataAdmissionError(ValueError):
    """Frozen selector batches do not form one valid development admission frame."""


def _load(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DirectionalDataAdmissionError(f"DIRECTIONAL_DATA_ADMISSION_JSON_INVALID:{path}") from exc
    if not isinstance(payload, Mapping):
        raise DirectionalDataAdmissionError(f"DIRECTIONAL_DATA_ADMISSION_JSON_INVALID:{path}")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite(value: object, code: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise DirectionalDataAdmissionError(code)
    return float(value)


def _campaign(path: Path) -> dict[str, object]:
    payload = _load(path)
    protocol = payload.get("protocol")
    gate = payload.get("data_admission_gate")
    campaign_id = payload.get("campaign_id")
    if (
        payload.get("artifact_type") != "verdiwm-ctrl-world-local-fingerprint-campaign"
        or not isinstance(campaign_id, str)
        or not campaign_id
        or not isinstance(protocol, Mapping)
        or not isinstance(gate, Mapping)
    ):
        raise DirectionalDataAdmissionError("DIRECTIONAL_DATA_ADMISSION_CAMPAIGN_INVALID")
    classes = tuple(int(value) for value in gate.get("classes", ()))
    minimum = int(gate.get("minimum_contexts_per_class", 0))
    if classes != (-1, 0, 1) or minimum < 1:
        raise DirectionalDataAdmissionError("DIRECTIONAL_DATA_ADMISSION_GATE_INVALID")
    return {
        "campaign_id": campaign_id,
        "candidate_radius": _finite(
            protocol.get("candidate_radius"), "DIRECTIONAL_DATA_ADMISSION_RADIUS_INVALID"
        ),
        "confidence_z": _finite(
            protocol.get("selector_confidence_critical_value"),
            "DIRECTIONAL_DATA_ADMISSION_CONFIDENCE_INVALID",
        ),
        "minimum_gain_lcb": _finite(
            protocol.get("minimum_predicted_gain_lcb"),
            "DIRECTIONAL_DATA_ADMISSION_THRESHOLD_INVALID",
        ),
        "classes": classes,
        "minimum_contexts_per_class": minimum,
        "path": str(path),
        "sha256": _sha256(path),
    }


def _classify(row: Mapping[str, Any], *, radius: float, minimum_gain_lcb: float) -> int:
    action = row.get("selector_action")
    selected = row.get("selected_candidate")
    if action == "abstain" and selected is None:
        return 0
    if action != "execute" or not isinstance(selected, Mapping):
        raise DirectionalDataAdmissionError("DIRECTIONAL_DATA_ADMISSION_SELECTOR_ACTION_INVALID")
    dose = _finite(selected.get("dose"), "DIRECTIONAL_DATA_ADMISSION_DOSE_INVALID")
    lcb = _finite(
        selected.get("predicted_weighted_gain_lcb"),
        "DIRECTIONAL_DATA_ADMISSION_LCB_INVALID",
    )
    if not math.isclose(abs(dose), radius, rel_tol=0.0, abs_tol=1e-12) or lcb <= minimum_gain_lcb:
        raise DirectionalDataAdmissionError("DIRECTIONAL_DATA_ADMISSION_SELECTION_INVALID")
    return 1 if dose > 0.0 else -1


def settle(
    *,
    campaign_paths: Sequence[Path],
    selector_paths: Sequence[Path],
    output_root: Path,
) -> dict[str, object]:
    if not campaign_paths or not selector_paths or output_root.exists() or output_root.is_symlink():
        raise DirectionalDataAdmissionError("DIRECTIONAL_DATA_ADMISSION_ARGUMENT_INVALID")
    campaigns: dict[str, dict[str, object]] = {}
    for raw_path in campaign_paths:
        path = Path(raw_path).resolve(strict=True)
        row = _campaign(path)
        campaign_id = str(row["campaign_id"])
        if campaign_id in campaigns:
            raise DirectionalDataAdmissionError("DIRECTIONAL_DATA_ADMISSION_CAMPAIGN_DUPLICATE")
        campaigns[campaign_id] = row
    frames = {
        (
            float(row["candidate_radius"]),
            float(row["confidence_z"]),
            float(row["minimum_gain_lcb"]),
            tuple(row["classes"]),
            int(row["minimum_contexts_per_class"]),
        )
        for row in campaigns.values()
    }
    if len(frames) != 1:
        raise DirectionalDataAdmissionError("DIRECTIONAL_DATA_ADMISSION_CAMPAIGN_FRAME_MISMATCH")
    radius, confidence_z, minimum_gain_lcb, classes, minimum = frames.pop()

    receipts: list[dict[str, object]] = []
    classifications: list[dict[str, object]] = []
    seen_contexts: set[str] = set()
    seen_campaigns: set[str] = set()
    for raw_path in selector_paths:
        path = Path(raw_path).resolve(strict=True)
        payload = _load(path)
        campaign_id = payload.get("fingerprint_campaign_id")
        contexts = payload.get("contexts")
        if (
            payload.get("artifact_type") != "verdiwm-ctrl-world-frozen-directional-selector"
            or payload.get("state") != "frozen"
            or not isinstance(campaign_id, str)
            or campaign_id not in campaigns
            or campaign_id in seen_campaigns
            or not isinstance(contexts, list)
            or not contexts
            or payload.get("effect_labels_observed") is not False
            or not math.isclose(_finite(payload.get("candidate_radius"), "DIRECTIONAL_DATA_ADMISSION_RADIUS_INVALID"), radius)
            or not math.isclose(_finite(payload.get("confidence_z"), "DIRECTIONAL_DATA_ADMISSION_CONFIDENCE_INVALID"), confidence_z)
            or not math.isclose(
                _finite(payload.get("minimum_predicted_gain_lcb"), "DIRECTIONAL_DATA_ADMISSION_THRESHOLD_INVALID"),
                minimum_gain_lcb,
            )
        ):
            raise DirectionalDataAdmissionError("DIRECTIONAL_DATA_ADMISSION_SELECTOR_INVALID")
        seen_campaigns.add(campaign_id)
        receipts.append(
            {
                "campaign": campaigns[campaign_id],
                "selector_path": str(path),
                "selector_sha256": _sha256(path),
            }
        )
        for row in contexts:
            if not isinstance(row, Mapping) or not isinstance(row.get("context"), Mapping):
                raise DirectionalDataAdmissionError("DIRECTIONAL_DATA_ADMISSION_CONTEXT_INVALID")
            context = row["context"]
            context_id = context.get("context_id")
            if not isinstance(context_id, str) or not context_id or context_id in seen_contexts:
                raise DirectionalDataAdmissionError("DIRECTIONAL_DATA_ADMISSION_CONTEXT_DUPLICATE")
            seen_contexts.add(context_id)
            target = _classify(row, radius=radius, minimum_gain_lcb=minimum_gain_lcb)
            classifications.append(
                {
                    "campaign_id": campaign_id,
                    "context": dict(context),
                    "target_class": target,
                    "selector_action": row.get("selector_action"),
                    "selected_candidate": row.get("selected_candidate"),
                }
            )
    if seen_campaigns != set(campaigns):
        raise DirectionalDataAdmissionError("DIRECTIONAL_DATA_ADMISSION_SELECTOR_COVERAGE_MISMATCH")

    counts = {target: sum(row["target_class"] == target for row in classifications) for target in classes}
    deficits = {target: max(minimum - counts[target], 0) for target in classes}
    admitted = not any(deficits.values())
    output_root.mkdir(mode=0o700, parents=True)
    report = {
        "schema_version": 1,
        "artifact_type": "verdiwm-ctrl-world-directional-data-admission-settlement",
        "state": "admitted" if admitted else "blocked",
        "decision": "license_candidate_training" if admitted else "do_not_train_candidate",
        "candidate_training_licensed": admitted,
        "campaign_count": len(campaigns),
        "context_count": len(classifications),
        "frozen_frame": {
            "candidate_radius": radius,
            "confidence_z": confidence_z,
            "minimum_predicted_gain_lcb": minimum_gain_lcb,
            "classes": list(classes),
            "minimum_contexts_per_class": minimum,
        },
        "class_counts": {str(target): counts[target] for target in classes},
        "class_deficits": {str(target): deficits[target] for target in classes},
        "input_receipts": sorted(receipts, key=lambda row: str(row["campaign"]["campaign_id"])),
        "classifications": sorted(classifications, key=lambda row: str(row["context"]["context_id"])),
        "failure_action": (
            None
            if admitted
            else "Keep the multiscale adapter unimplemented and untrained; expand development contexts or redesign the mechanism."
        ),
        "claim_boundary": "Development-label coverage only. Admission does not establish repair effect or held-out improvement.",
    }
    report_path = output_root / "data-admission-settlement.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "artifact_type": "verdiwm-ctrl-world-directional-data-admission-manifest",
        "state": report["state"],
        "candidate_training_licensed": admitted,
        "class_counts": report["class_counts"],
        "report_path": str(report_path.resolve()),
        "report_sha256": _sha256(report_path),
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, action="append", required=True)
    parser.add_argument("--selector", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = settle(
        campaign_paths=args.campaign,
        selector_paths=args.selector,
        output_root=args.output_root.resolve(),
    )
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
