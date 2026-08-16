#!/usr/bin/env python3
"""Materialize an anonymous FSHC bank from admitted multi-identity shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


FORBIDDEN_TRAINING_KEYS = frozenset(("context_id", "episode_id", "probe_id", "seed", "start_idx"))
FORBIDDEN_PROMOTION_EPISODES = frozenset(("1799",))
EXPECTED_DOSES = (-0.025, 0.0, 0.025)


class FSHCFeatureBankError(ValueError):
    """Admitted probe evidence cannot produce a leakage-free training bank."""


def _load(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FSHCFeatureBankError(f"FSHC_FEATURE_BANK_JSON_INVALID:{path}") from exc
    if not isinstance(value, Mapping):
        raise FSHCFeatureBankError(f"FSHC_FEATURE_BANK_JSON_INVALID:{path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity(raw: object) -> tuple[str, str, int, int]:
    if not isinstance(raw, Mapping):
        raise FSHCFeatureBankError("FSHC_FEATURE_BANK_IDENTITY_INVALID")
    values = (raw.get("context_id"), raw.get("episode_id"), raw.get("start_idx"), raw.get("seed"))
    if (
        not isinstance(values[0], str)
        or not values[0]
        or not isinstance(values[1], str)
        or not values[1]
        or not isinstance(values[2], int)
        or values[2] < 0
        or not isinstance(values[3], int)
    ):
        raise FSHCFeatureBankError("FSHC_FEATURE_BANK_IDENTITY_INVALID")
    if values[1] in FORBIDDEN_PROMOTION_EPISODES:
        raise FSHCFeatureBankError("FSHC_FEATURE_BANK_PROMOTION_EPISODE_FORBIDDEN")
    return values  # type: ignore[return-value]


def _features(raw: object) -> list[list[list[float]]]:
    if not isinstance(raw, list) or not raw:
        raise FSHCFeatureBankError("FSHC_FEATURE_BANK_RECORDS_EMPTY")
    records: list[list[list[float]]] = []
    shape: tuple[int, int] | None = None
    for record in raw:
        if not isinstance(record, list) or not record or any(not isinstance(row, list) for row in record):
            raise FSHCFeatureBankError("FSHC_FEATURE_BANK_FEATURE_SHAPE_INVALID")
        matrix = [[float(value) for value in row] for row in record]
        current_shape = (len(matrix), len(matrix[0]))
        if current_shape[1] < 1 or any(len(row) != current_shape[1] for row in matrix):
            raise FSHCFeatureBankError("FSHC_FEATURE_BANK_FEATURE_SHAPE_INVALID")
        if shape is None:
            shape = current_shape
        if current_shape != shape or any(not math.isfinite(value) for row in matrix for value in row):
            raise FSHCFeatureBankError("FSHC_FEATURE_BANK_FEATURE_VALUE_INVALID")
        records.append(matrix)
    return records


def _assert_no_identity_keys(value: object) -> None:
    if isinstance(value, Mapping):
        if FORBIDDEN_TRAINING_KEYS.intersection(value):
            raise FSHCFeatureBankError("FSHC_FEATURE_BANK_IDENTITY_LEAK")
        for child in value.values():
            _assert_no_identity_keys(child)
    elif isinstance(value, list):
        for child in value:
            _assert_no_identity_keys(child)


def _target(row: Mapping[str, Any]) -> int:
    action = row.get("selector_action")
    selected = row.get("selected_candidate")
    if action == "abstain" and selected is None:
        return 0
    if action != "execute" or not isinstance(selected, Mapping):
        raise FSHCFeatureBankError("FSHC_FEATURE_BANK_LABEL_INVALID")
    dose = float(selected.get("dose"))
    if not math.isfinite(dose) or not math.isclose(abs(dose), 0.025, rel_tol=0.0, abs_tol=1e-12):
        raise FSHCFeatureBankError("FSHC_FEATURE_BANK_LABEL_INVALID")
    return 1 if dose > 0.0 else -1


def _dose_index(value: object) -> int:
    try:
        dose = float(value)
    except (TypeError, ValueError) as exc:
        raise FSHCFeatureBankError("FSHC_FEATURE_BANK_QUALITY_DOSE_FRAME_INVALID") from exc
    for index, expected in enumerate(EXPECTED_DOSES):
        if math.isclose(dose, expected, rel_tol=0.0, abs_tol=1e-12):
            return index
    raise FSHCFeatureBankError("FSHC_FEATURE_BANK_QUALITY_DOSE_FRAME_INVALID")


def materialize(
    *,
    admission_path: Path,
    campaign_paths: Sequence[Path],
    selector_paths: Sequence[Path],
    shard_paths: Sequence[Path],
    output_root: Path,
) -> dict[str, object]:
    if (
        not campaign_paths
        or not selector_paths
        or not shard_paths
        or output_root.exists()
        or output_root.is_symlink()
    ):
        raise FSHCFeatureBankError("FSHC_FEATURE_BANK_INPUT_INVALID")
    admission_path = admission_path.resolve(strict=True)
    admission = _load(admission_path)
    if (
        admission.get("artifact_type") != "verdiwm-ctrl-world-directional-data-admission-settlement"
        or admission.get("state") != "admitted"
        or admission.get("candidate_training_licensed") is not True
    ):
        raise FSHCFeatureBankError("FSHC_FEATURE_BANK_ADMISSION_INVALID")

    campaigns: dict[str, dict[str, object]] = {}
    outcome_frame: tuple[tuple[str, ...], tuple[float, ...]] | None = None
    for raw_path in campaign_paths:
        path = Path(raw_path).resolve(strict=True)
        payload = _load(path)
        campaign_id = payload.get("campaign_id")
        outcomes = payload.get("outcomes")
        if (
            payload.get("artifact_type") != "verdiwm-ctrl-world-local-fingerprint-campaign"
            or not isinstance(campaign_id, str)
            or not campaign_id
            or campaign_id in campaigns
            or not isinstance(outcomes, list)
            or not outcomes
        ):
            raise FSHCFeatureBankError("FSHC_FEATURE_BANK_CAMPAIGN_INVALID")
        names = tuple(str(row.get("name")) for row in outcomes if isinstance(row, Mapping))
        weights = tuple(float(row.get("weight")) for row in outcomes if isinstance(row, Mapping))
        if (
            len(names) != len(outcomes)
            or len(set(names)) != len(names)
            or any(not math.isfinite(weight) or weight <= 0.0 for weight in weights)
        ):
            raise FSHCFeatureBankError("FSHC_FEATURE_BANK_OUTCOME_FRAME_INVALID")
        frame = (names, weights)
        if outcome_frame is None:
            outcome_frame = frame
        elif frame != outcome_frame:
            raise FSHCFeatureBankError("FSHC_FEATURE_BANK_OUTCOME_FRAME_MISMATCH")
        campaigns[campaign_id] = {"path": path, "payload": payload, "sha256": _sha256(path)}
    assert outcome_frame is not None
    outcome_names, outcome_weights = outcome_frame

    selectors: dict[str, dict[str, object]] = {}
    selector_targets: dict[tuple[str, str], tuple[int, Mapping[str, Any]]] = {}
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
            or campaign_id in selectors
            or not isinstance(contexts, list)
        ):
            raise FSHCFeatureBankError("FSHC_FEATURE_BANK_SELECTOR_INVALID")
        selectors[campaign_id] = {"path": path, "payload": payload, "sha256": _sha256(path)}
        for row in contexts:
            if not isinstance(row, Mapping) or not isinstance(row.get("context"), Mapping):
                raise FSHCFeatureBankError("FSHC_FEATURE_BANK_LABEL_INVALID")
            context = row["context"]
            context_id = context.get("context_id")
            episode_id = context.get("episode_id")
            if (
                not isinstance(context_id, str)
                or not context_id
                or not isinstance(episode_id, str)
                or episode_id in FORBIDDEN_PROMOTION_EPISODES
                or (campaign_id, context_id) in selector_targets
            ):
                raise FSHCFeatureBankError("FSHC_FEATURE_BANK_LABEL_INVALID")
            selector_targets[(campaign_id, context_id)] = (_target(row), context)
    if set(selectors) != set(campaigns):
        raise FSHCFeatureBankError("FSHC_FEATURE_BANK_SELECTOR_COVERAGE_MISMATCH")

    receipt_campaigns: set[str] = set()
    receipts = admission.get("input_receipts")
    if not isinstance(receipts, list):
        raise FSHCFeatureBankError("FSHC_FEATURE_BANK_ADMISSION_RECEIPT_INVALID")
    for receipt in receipts:
        if not isinstance(receipt, Mapping) or not isinstance(receipt.get("campaign"), Mapping):
            raise FSHCFeatureBankError("FSHC_FEATURE_BANK_ADMISSION_RECEIPT_INVALID")
        campaign_receipt = receipt["campaign"]
        campaign_id = campaign_receipt.get("campaign_id")
        if (
            not isinstance(campaign_id, str)
            or campaign_id not in campaigns
            or campaign_id in receipt_campaigns
            or campaign_receipt.get("sha256") != campaigns[campaign_id]["sha256"]
            or receipt.get("selector_sha256") != selectors[campaign_id]["sha256"]
        ):
            raise FSHCFeatureBankError("FSHC_FEATURE_BANK_ADMISSION_HASH_MISMATCH")
        receipt_campaigns.add(campaign_id)
    if receipt_campaigns != set(campaigns):
        raise FSHCFeatureBankError("FSHC_FEATURE_BANK_ADMISSION_RECEIPT_INVALID")

    expected: dict[tuple[str, str, str, int, int], int] = {}
    classifications = admission.get("classifications")
    if not isinstance(classifications, list):
        raise FSHCFeatureBankError("FSHC_FEATURE_BANK_ADMISSION_LABEL_INVALID")
    for row in classifications:
        if (
            not isinstance(row, Mapping)
            or not isinstance(row.get("campaign_id"), str)
            or not isinstance(row.get("context"), Mapping)
        ):
            raise FSHCFeatureBankError("FSHC_FEATURE_BANK_ADMISSION_LABEL_INVALID")
        campaign_id = row["campaign_id"]
        context = row["context"]
        context_id = context.get("context_id")
        episode_id = context.get("episode_id")
        start_idx = context.get("start_idx")
        seeds = context.get("seeds")
        target = row.get("target_class")
        selector = selector_targets.get((campaign_id, context_id))
        if (
            campaign_id not in campaigns
            or selector is None
            or not isinstance(context_id, str)
            or not isinstance(episode_id, str)
            or episode_id in FORBIDDEN_PROMOTION_EPISODES
            or not isinstance(start_idx, int)
            or not isinstance(seeds, list)
            or not seeds
            or any(not isinstance(seed, int) for seed in seeds)
            or target not in (-1, 0, 1)
            or selector[0] != target
            or dict(selector[1]) != dict(context)
        ):
            raise FSHCFeatureBankError("FSHC_FEATURE_BANK_ADMISSION_LABEL_INVALID")
        for seed in seeds:
            key = (campaign_id, context_id, episode_id, start_idx, seed)
            if key in expected:
                raise FSHCFeatureBankError("FSHC_FEATURE_BANK_IDENTITY_DUPLICATE_OR_MISMATCH")
            expected[key] = target

    units: list[dict[str, object]] = []
    seen: set[tuple[str, str, str, int, int]] = set()
    feature_shape: tuple[int, int] | None = None
    for raw_path in shard_paths:
        path = Path(raw_path).resolve(strict=True)
        shard = _load(path)
        campaign_id = shard.get("campaign_id")
        measurements = shard.get("measurements")
        checks = shard.get("zero_identity_checks")
        if (
            shard.get("artifact_type") != "verdiwm-ctrl-world-local-fingerprint-probe-result"
            or shard.get("state") != "ready"
            or not isinstance(campaign_id, str)
            or campaign_id not in campaigns
            or shard.get("probe_id") != "fshc_signed_history_gain"
            or shard.get("hook_activation") != {"state": "passed"}
            or not isinstance(measurements, list)
            or not isinstance(checks, list)
        ):
            raise FSHCFeatureBankError(f"FSHC_FEATURE_BANK_SHARD_INVALID:{path}")
        measurements_by_identity: dict[tuple[str, str, int, int], dict[int, Mapping[str, Any]]] = {}
        for row in measurements:
            if not isinstance(row, Mapping):
                raise FSHCFeatureBankError(f"FSHC_FEATURE_BANK_SHARD_INVALID:{path}")
            identity = _identity(row.get("identity"))
            dose_index = _dose_index(row.get("dose"))
            by_dose = measurements_by_identity.setdefault(identity, {})
            if dose_index in by_dose:
                raise FSHCFeatureBankError("FSHC_FEATURE_BANK_QUALITY_DOSE_FRAME_INVALID")
            by_dose[dose_index] = row
        checks_by_identity: dict[tuple[str, str, int, int], Mapping[str, Any]] = {}
        for check in checks:
            if not isinstance(check, Mapping) or check.get("state") != "passed":
                raise FSHCFeatureBankError(f"FSHC_FEATURE_BANK_ZERO_FRAME_INVALID:{path}")
            identity = _identity(check.get("identity"))
            if identity in checks_by_identity:
                raise FSHCFeatureBankError(f"FSHC_FEATURE_BANK_ZERO_FRAME_INVALID:{path}")
            checks_by_identity[identity] = check
        if set(checks_by_identity) != set(measurements_by_identity):
            raise FSHCFeatureBankError(f"FSHC_FEATURE_BANK_ZERO_FRAME_INVALID:{path}")
        for identity, by_dose in measurements_by_identity.items():
            key = (campaign_id, *identity)
            if set(by_dose) != {0, 1, 2}:
                raise FSHCFeatureBankError("FSHC_FEATURE_BANK_QUALITY_DOSE_FRAME_INVALID")
            if key not in expected or key in seen:
                raise FSHCFeatureBankError("FSHC_FEATURE_BANK_IDENTITY_DUPLICATE_OR_MISMATCH")
            seen.add(key)
            records = _features(by_dose[1].get("runtime_feature_records"))
            current_shape = (len(records[0]), len(records[0][0]))
            if feature_shape is None:
                feature_shape = current_shape
            if current_shape != feature_shape:
                raise FSHCFeatureBankError("FSHC_FEATURE_BANK_CROSS_SHARD_SHAPE_MISMATCH")
            losses: list[float] = []
            for dose_index in (0, 1, 2):
                outcomes = by_dose[dose_index].get("outcomes")
                if not isinstance(outcomes, Mapping) or set(outcomes) != set(outcome_names):
                    raise FSHCFeatureBankError("FSHC_FEATURE_BANK_QUALITY_INVALID")
                values = [float(outcomes[name]) for name in outcome_names]
                if any(not math.isfinite(value) for value in values):
                    raise FSHCFeatureBankError("FSHC_FEATURE_BANK_QUALITY_INVALID")
                losses.append(-sum(weight * value for weight, value in zip(outcome_weights, values, strict=True)))
            units.append(
                {
                    "sort_key": key,
                    "identity": identity,
                    "campaign_id": campaign_id,
                    "target": expected[key],
                    "records": records,
                    "counterfactual_losses": losses,
                    "shard_path": path,
                }
            )
    if seen != set(expected) or feature_shape is None:
        raise FSHCFeatureBankError("FSHC_FEATURE_BANK_COVERAGE_MISMATCH")

    training_records: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    for unit in sorted(units, key=lambda row: row["sort_key"]):
        records = unit["records"]
        offset = len(training_records)
        training_records.extend(
            {
                "features": record,
                "target": unit["target"],
                "counterfactual_losses": unit["counterfactual_losses"],
            }
            for record in records
        )
        identity = unit["identity"]
        path = unit["shard_path"]
        audit_rows.append(
            {
                "campaign_id": unit["campaign_id"],
                "identity": {
                    "context_id": identity[0],
                    "episode_id": identity[1],
                    "start_idx": identity[2],
                    "seed": identity[3],
                },
                "target": unit["target"],
                "training_record_offset": offset,
                "training_record_count": len(records),
                "shard_path": str(path),
                "shard_sha256": _sha256(path),
            }
        )

    training = {
        "schema_version": 2,
        "artifact_type": "ctrl-world-fshc-anonymous-runtime-feature-bank",
        "state": "ready",
        "feature_shape": list(feature_shape),
        "record_count": len(training_records),
        "target_counts": {
            str(target): sum(row["target"] == target for row in training_records)
            for target in (-1, 0, 1)
        },
        "source_admission_sha256": _sha256(admission_path),
        "records": training_records,
        "claim_boundary": "Development-only anonymous gate supervision; no identity fields and no promotion episode records.",
    }
    _assert_no_identity_keys(training)
    audit = {
        "schema_version": 2,
        "artifact_type": "verdiwm-ctrl-world-fshc-feature-bank-audit",
        "state": "ready",
        "admission": {"path": str(admission_path), "sha256": _sha256(admission_path)},
        "forbidden_promotion_episodes": sorted(FORBIDDEN_PROMOTION_EPISODES),
        "campaigns": [
            {"campaign_id": campaign_id, "path": str(row["path"]), "sha256": row["sha256"]}
            for campaign_id, row in sorted(campaigns.items())
        ],
        "selectors": [
            {"campaign_id": campaign_id, "path": str(row["path"]), "sha256": row["sha256"]}
            for campaign_id, row in sorted(selectors.items())
        ],
        "identity_count": len(seen),
        "rows": audit_rows,
    }
    output_root.mkdir(mode=0o700, parents=True)
    training_path = output_root / "training-feature-bank.json"
    audit_path = output_root / "audit-receipt.json"
    training_path.write_text(json.dumps(training, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    audit_path.write_text(json.dumps(audit, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 2,
        "artifact_type": "verdiwm-ctrl-world-fshc-feature-bank-manifest",
        "state": "ready",
        "identity_count": len(seen),
        "record_count": len(training_records),
        "target_counts": training["target_counts"],
        "training_path": str(training_path.resolve()),
        "training_sha256": _sha256(training_path),
        "audit_path": str(audit_path.resolve()),
        "audit_sha256": _sha256(audit_path),
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, action="append", required=True)
    parser.add_argument("--selector", type=Path, action="append", required=True)
    parser.add_argument("--shard-result", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = materialize(
        admission_path=args.admission,
        campaign_paths=args.campaign,
        selector_paths=args.selector,
        shard_paths=args.shard_result,
        output_root=args.output_root.resolve(),
    )
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
