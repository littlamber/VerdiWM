#!/usr/bin/env python3
"""Materialize and admit an anonymous CCLVR interaction-local value bank."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_TRAINING_KEYS = frozenset(("context_id", "episode_id", "seed", "start_idx"))
FORBIDDEN_PROMOTION_EPISODES = frozenset(("1799",))


class CCLVRValueBankError(ValueError):
    """Raw local counterfactual evidence cannot form an admitted value bank."""


def _load(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CCLVRValueBankError(f"CCLVR_VALUE_BANK_JSON_INVALID:{path}") from exc
    if not isinstance(payload, Mapping):
        raise CCLVRValueBankError(f"CCLVR_VALUE_BANK_JSON_INVALID:{path}")
    return payload


def _resolve(value: object) -> Path:
    path = Path(str(value))
    return (ROOT / path).resolve(strict=True) if not path.is_absolute() else path.resolve(strict=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(raw: object) -> tuple[str, str, int, int]:
    if not isinstance(raw, Mapping):
        raise CCLVRValueBankError("CCLVR_VALUE_BANK_IDENTITY_INVALID")
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
        raise CCLVRValueBankError("CCLVR_VALUE_BANK_IDENTITY_INVALID")
    if values[1] in FORBIDDEN_PROMOTION_EPISODES:
        raise CCLVRValueBankError("CCLVR_VALUE_BANK_PROMOTION_EPISODE_FORBIDDEN")
    return values  # type: ignore[return-value]


def _finite(value: object, error: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise CCLVRValueBankError(error) from exc
    if not math.isfinite(result):
        raise CCLVRValueBankError(error)
    return result


def _feature_records(raw: object) -> list[list[list[float]]]:
    if not isinstance(raw, list) or not raw:
        raise CCLVRValueBankError("CCLVR_VALUE_BANK_FEATURES_INVALID")
    records: list[list[list[float]]] = []
    shape: tuple[int, int] | None = None
    for raw_record in raw:
        if not isinstance(raw_record, list) or not raw_record or any(not isinstance(row, list) for row in raw_record):
            raise CCLVRValueBankError("CCLVR_VALUE_BANK_FEATURES_INVALID")
        record = [[_finite(value, "CCLVR_VALUE_BANK_FEATURES_INVALID") for value in row] for row in raw_record]
        current = (len(record), len(record[0]))
        if current[1] < 1 or any(len(row) != current[1] for row in record):
            raise CCLVRValueBankError("CCLVR_VALUE_BANK_FEATURES_INVALID")
        if shape is None:
            shape = current
        elif current != shape:
            raise CCLVRValueBankError("CCLVR_VALUE_BANK_FEATURES_INVALID")
        records.append(record)
    return records


def _interactions(raw: object, count: int) -> list[dict[str, float]]:
    if not isinstance(raw, list) or len(raw) != count:
        raise CCLVRValueBankError("CCLVR_VALUE_BANK_INTERACTIONS_INVALID")
    rows: list[dict[str, float]] = []
    for index, row in enumerate(raw):
        if not isinstance(row, Mapping) or row.get("interaction") != index:
            raise CCLVRValueBankError("CCLVR_VALUE_BANK_INTERACTIONS_INVALID")
        rows.append(
            {
                "mean_l1": _finite(row.get("mean_l1"), "CCLVR_VALUE_BANK_INTERACTIONS_INVALID"),
                "final_l1": _finite(row.get("final_l1"), "CCLVR_VALUE_BANK_INTERACTIONS_INVALID"),
            }
        )
    return rows


def _slope(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean_x = (len(values) - 1) / 2.0
    mean_y = sum(values) / len(values)
    numerator = sum((index - mean_x) * (value - mean_y) for index, value in enumerate(values))
    denominator = sum((index - mean_x) ** 2 for index in range(len(values)))
    return numerator / denominator


def _local_error(
    rows: Sequence[Mapping[str, float]],
    interaction: int,
    *,
    suffix_mean_weight: float,
    terminal_weight: float,
    slope_weight: float,
) -> dict[str, float]:
    suffix = [float(row["mean_l1"]) for row in rows[interaction:]]
    suffix_mean = sum(suffix) / len(suffix)
    terminal = float(rows[-1]["final_l1"])
    slope = _slope(suffix)
    loss = suffix_mean_weight * suffix_mean + terminal_weight * terminal + slope_weight * slope
    return {
        "loss": loss,
        "suffix_mean_l1": suffix_mean,
        "terminal_interaction_l1": terminal,
        "suffix_horizon_l1_slope": slope,
    }


def _wilson(successes: int, total: int, z: float) -> dict[str, float | int]:
    if total < 1 or successes < 0 or successes > total or not math.isfinite(z) or z <= 0.0:
        raise CCLVRValueBankError("CCLVR_VALUE_BANK_WILSON_INPUT_INVALID")
    observed = successes / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = (observed + z2 / (2.0 * total)) / denominator
    margin = z * math.sqrt(observed * (1.0 - observed) / total + z2 / (4.0 * total * total)) / denominator
    return {
        "successes": successes,
        "total": total,
        "observed": observed,
        "lower": max(0.0, center - margin),
        "upper": min(1.0, center + margin),
        "z": z,
    }


def _assert_anonymous(value: object) -> None:
    if isinstance(value, Mapping):
        if FORBIDDEN_TRAINING_KEYS.intersection(value):
            raise CCLVRValueBankError("CCLVR_VALUE_BANK_IDENTITY_LEAK")
        for child in value.values():
            _assert_anonymous(child)
    elif isinstance(value, list):
        for child in value:
            _assert_anonymous(child)


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def materialize(*, campaign_path: Path, shard_paths: Sequence[Path], output_root: Path) -> dict[str, object]:
    if not shard_paths or output_root.exists() or output_root.is_symlink():
        raise CCLVRValueBankError("CCLVR_VALUE_BANK_INPUT_INVALID")
    campaign_path = campaign_path.resolve(strict=True)
    campaign = _load(campaign_path)
    protocol = campaign.get("protocol")
    return_spec = campaign.get("local_return")
    admission_spec = campaign.get("data_admission")
    checkpoint = campaign.get("checkpoint")
    contexts_spec = campaign.get("contexts")
    dependencies = campaign.get("dependencies")
    execution = campaign.get("execution")
    if (
        campaign.get("artifact_type") != "verdiwm-ctrl-world-cclvr-local-value-campaign"
        or campaign.get("state") != "frozen_before_execution"
        or not isinstance(protocol, Mapping)
        or not isinstance(return_spec, Mapping)
        or not isinstance(admission_spec, Mapping)
        or not isinstance(checkpoint, Mapping)
        or not isinstance(contexts_spec, Mapping)
        or not isinstance(dependencies, Mapping)
        or not isinstance(execution, Mapping)
    ):
        raise CCLVRValueBankError("CCLVR_VALUE_BANK_CAMPAIGN_INVALID")
    contexts_path = _resolve(contexts_spec.get("path"))
    checkpoint_path = _resolve(checkpoint.get("path"))
    evaluator_path = _resolve(dependencies.get("evaluator"))
    materializer_path = _resolve(dependencies.get("materializer"))
    if _sha256(contexts_path) != contexts_spec.get("sha256"):
        raise CCLVRValueBankError("CCLVR_VALUE_BANK_CONTEXTS_HASH_MISMATCH")
    if _sha256(checkpoint_path) != checkpoint.get("sha256"):
        raise CCLVRValueBankError("CCLVR_VALUE_BANK_CHECKPOINT_HASH_MISMATCH")
    if _sha256(evaluator_path) != dependencies.get("evaluator_sha256"):
        raise CCLVRValueBankError("CCLVR_VALUE_BANK_EVALUATOR_HASH_MISMATCH")
    if _sha256(materializer_path) != dependencies.get("materializer_sha256"):
        raise CCLVRValueBankError("CCLVR_VALUE_BANK_MATERIALIZER_HASH_MISMATCH")

    context_payload = _load(contexts_path)
    context_rows = context_payload.get("contexts")
    if context_payload.get("artifact_type") != "verdiwm-ctrl-world-local-context-set" or not isinstance(context_rows, list):
        raise CCLVRValueBankError("CCLVR_VALUE_BANK_CONTEXTS_INVALID")
    expected: set[tuple[str, str, int, int]] = set()
    for row in context_rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("seeds"), list):
            raise CCLVRValueBankError("CCLVR_VALUE_BANK_CONTEXTS_INVALID")
        for seed in row["seeds"]:
            identity = _identity({**row, "seed": seed})
            if identity in expected:
                raise CCLVRValueBankError("CCLVR_VALUE_BANK_IDENTITY_DUPLICATE")
            expected.add(identity)

    interact_num = int(protocol.get("interact_num", 0))
    inference_steps = int(protocol.get("num_inference_steps", 0))
    target_interactions = tuple(int(value) for value in protocol.get("target_interactions", ()))
    doses = tuple(float(value) for value in protocol.get("doses", ()))
    if (
        interact_num < 2
        or inference_steps < 1
        or target_interactions != tuple(range(interact_num))
        or len(doses) != 3
        or not math.isclose(doses[1], 0.0, abs_tol=1e-12)
        or not math.isclose(doses[0], -doses[2], abs_tol=1e-12)
    ):
        raise CCLVRValueBankError("CCLVR_VALUE_BANK_PROTOCOL_INVALID")
    if len(shard_paths) != int(execution.get("gpus", -1)):
        raise CCLVRValueBankError("CCLVR_VALUE_BANK_SHARD_COUNT_INVALID")

    measurements: dict[tuple[tuple[str, str, int, int], int | None, float], Mapping[str, Any]] = {}
    zero_checks: set[tuple[str, str, int, int]] = set()
    seen_identities: set[tuple[str, str, int, int]] = set()
    shard_receipts: list[dict[str, str]] = []
    for raw_path in shard_paths:
        path = Path(raw_path).resolve(strict=True)
        shard = _load(path)
        rows = shard.get("measurements")
        checks = shard.get("zero_identity_checks")
        shard_input = shard.get("input")
        if (
            shard.get("artifact_type") != "verdiwm-ctrl-world-local-fingerprint-probe-result"
            or shard.get("state") != "ready"
            or shard.get("campaign_id") != campaign.get("campaign_id")
            or shard.get("probe_id") != "fshc_interaction_local_gain"
            or shard.get("hook_activation") != {"state": "passed"}
            or not isinstance(rows, list)
            or not isinstance(checks, list)
            or not isinstance(shard_input, Mapping)
            or Path(str(shard_input.get("checkpoint"))).resolve() != checkpoint_path
            or shard_input.get("contexts_sha256") != contexts_spec.get("sha256")
        ):
            raise CCLVRValueBankError(f"CCLVR_VALUE_BANK_SHARD_INVALID:{path}")
        shard_identities: set[tuple[str, str, int, int]] = set()
        for row in rows:
            if not isinstance(row, Mapping):
                raise CCLVRValueBankError(f"CCLVR_VALUE_BANK_SHARD_INVALID:{path}")
            identity = _identity(row.get("identity"))
            dose = _finite(row.get("dose"), "CCLVR_VALUE_BANK_DOSE_INVALID")
            target = row.get("target_interaction")
            target_value = None if target is None else int(target)
            if not any(math.isclose(dose, expected_dose, abs_tol=1e-12) for expected_dose in doses):
                raise CCLVRValueBankError("CCLVR_VALUE_BANK_DOSE_INVALID")
            if (math.isclose(dose, 0.0, abs_tol=1e-12)) != (target_value is None):
                raise CCLVRValueBankError("CCLVR_VALUE_BANK_LOCAL_FRAME_INVALID")
            if target_value is not None and target_value not in target_interactions:
                raise CCLVRValueBankError("CCLVR_VALUE_BANK_LOCAL_FRAME_INVALID")
            key = (identity, target_value, dose)
            if key in measurements:
                raise CCLVRValueBankError("CCLVR_VALUE_BANK_MEASUREMENT_DUPLICATE")
            measurements[key] = row
            shard_identities.add(identity)
        for check in checks:
            if not isinstance(check, Mapping) or check.get("state") != "passed" or check.get("target_interaction") is not None:
                raise CCLVRValueBankError("CCLVR_VALUE_BANK_ZERO_CHECK_INVALID")
            identity = _identity(check.get("identity"))
            if identity in zero_checks:
                raise CCLVRValueBankError("CCLVR_VALUE_BANK_ZERO_CHECK_DUPLICATE")
            zero_checks.add(identity)
        if not shard_identities or seen_identities.intersection(shard_identities):
            raise CCLVRValueBankError("CCLVR_VALUE_BANK_SHARD_IDENTITY_OVERLAP")
        seen_identities.update(shard_identities)
        shard_receipts.append({"path": str(path), "sha256": _sha256(path)})
    if seen_identities != expected or zero_checks != expected:
        raise CCLVRValueBankError("CCLVR_VALUE_BANK_IDENTITY_COVERAGE_MISMATCH")

    suffix_weight = _finite(return_spec.get("suffix_mean_l1_weight"), "CCLVR_VALUE_BANK_RETURN_INVALID")
    terminal_weight = _finite(return_spec.get("terminal_interaction_l1_weight"), "CCLVR_VALUE_BANK_RETURN_INVALID")
    slope_weight = _finite(return_spec.get("suffix_horizon_l1_slope_weight"), "CCLVR_VALUE_BANK_RETURN_INVALID")
    minimum_benefit = _finite(return_spec.get("minimum_benefit"), "CCLVR_VALUE_BANK_RETURN_INVALID")
    if min(suffix_weight, terminal_weight, slope_weight) <= 0.0 or minimum_benefit < 0.0:
        raise CCLVRValueBankError("CCLVR_VALUE_BANK_RETURN_INVALID")
    prefix_tolerance = _finite(admission_spec.get("prefix_identity_tolerance"), "CCLVR_VALUE_BANK_ADMISSION_INVALID")
    minimum_support = admission_spec.get("minimum_best_arm_support")
    if not isinstance(minimum_support, Mapping):
        raise CCLVRValueBankError("CCLVR_VALUE_BANK_ADMISSION_INVALID")

    training_records: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    feature_shape: tuple[int, int, int] | None = None
    arm_counts = {"negative": 0, "zero": 0, "positive": 0}
    for identity in sorted(expected):
        zero_row = measurements.get((identity, None, 0.0))
        if zero_row is None:
            raise CCLVRValueBankError("CCLVR_VALUE_BANK_ZERO_MEASUREMENT_MISSING")
        zero_interactions = _interactions(zero_row.get("interactions"), interact_num)
        all_features = _feature_records(zero_row.get("runtime_feature_records"))
        if len(all_features) != interact_num * inference_steps:
            raise CCLVRValueBankError("CCLVR_VALUE_BANK_FEATURE_INVOCATION_COUNT_INVALID")
        audit_interactions: list[dict[str, object]] = []
        record_offset = len(training_records)
        for interaction in target_interactions:
            arms: list[Mapping[str, Any]] = []
            for dose in doses:
                row = zero_row if math.isclose(dose, 0.0, abs_tol=1e-12) else measurements.get((identity, interaction, dose))
                if row is None:
                    raise CCLVRValueBankError("CCLVR_VALUE_BANK_LOCAL_ARM_MISSING")
                arms.append(row)
            arm_interactions = [_interactions(row.get("interactions"), interact_num) for row in arms]
            for candidate in arm_interactions:
                for prefix in range(interaction):
                    for metric in ("mean_l1", "final_l1"):
                        if abs(candidate[prefix][metric] - zero_interactions[prefix][metric]) > prefix_tolerance:
                            raise CCLVRValueBankError("CCLVR_VALUE_BANK_LOCAL_PREFIX_CHANGED")
            components = [
                _local_error(
                    rows,
                    interaction,
                    suffix_mean_weight=suffix_weight,
                    terminal_weight=terminal_weight,
                    slope_weight=slope_weight,
                )
                for rows in arm_interactions
            ]
            losses = [row["loss"] for row in components]
            values = [losses[1] - loss for loss in losses]
            best_nonzero = 0 if values[0] >= values[2] else 2
            target_arm = best_nonzero if values[best_nonzero] >= minimum_benefit else 1
            arm_name = ("negative", "zero", "positive")[target_arm]
            arm_counts[arm_name] += 1
            features = all_features[interaction * inference_steps : (interaction + 1) * inference_steps]
            current_shape = (len(features), len(features[0]), len(features[0][0]))
            if feature_shape is None:
                feature_shape = current_shape
            elif current_shape != feature_shape:
                raise CCLVRValueBankError("CCLVR_VALUE_BANK_FEATURE_SHAPE_MISMATCH")
            training_records.append(
                {
                    "features": features,
                    "interaction_index": interaction,
                    "arm_doses": list(doses),
                    "arm_losses": losses,
                    "arm_values": values,
                    "target_arm": target_arm,
                    "target_dose": doses[target_arm],
                }
            )
            audit_interactions.append(
                {
                    "interaction_index": interaction,
                    "arm_components": components,
                    "arm_losses": losses,
                    "arm_values": values,
                    "target_arm": target_arm,
                    "target_dose": doses[target_arm],
                }
            )
        audit_rows.append(
            {
                "identity": {
                    "context_id": identity[0],
                    "episode_id": identity[1],
                    "start_idx": identity[2],
                    "seed": identity[3],
                },
                "training_record_offset": record_offset,
                "training_record_count": interact_num,
                "interactions": audit_interactions,
            }
        )
    if feature_shape is None:
        raise CCLVRValueBankError("CCLVR_VALUE_BANK_EMPTY")

    required_counts = {
        name: int(minimum_support.get(name, -1)) for name in ("negative", "zero", "positive")
    }
    support_checks = {
        name: {
            "observed": arm_counts[name],
            "required": required_counts[name],
            "state": "passed" if arm_counts[name] >= required_counts[name] >= 0 else "failed",
        }
        for name in arm_counts
    }
    admitted = all(row["state"] == "passed" for row in support_checks.values())
    coverage = _wilson(
        arm_counts["negative"] + arm_counts["positive"],
        len(training_records),
        _finite(admission_spec.get("coverage_wilson_z"), "CCLVR_VALUE_BANK_ADMISSION_INVALID"),
    )
    source_hashes = {
        "campaign_sha256": _sha256(campaign_path),
        "contexts_sha256": _sha256(contexts_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "evaluator_sha256": _sha256(evaluator_path),
        "materializer_sha256": _sha256(materializer_path),
        "shards": sorted(shard_receipts, key=lambda row: row["path"]),
    }
    training = {
        "schema_version": 1,
        "artifact_type": "verdiwm-ctrl-world-cclvr-anonymous-local-value-bank",
        "state": "ready" if admitted else "blocked",
        "feature_shape": list(feature_shape),
        "record_count": len(training_records),
        "arm_doses": list(doses),
        "target_arm_counts": arm_counts,
        "local_return": dict(return_spec),
        "coverage_band": coverage,
        "source_hashes": source_hashes,
        "records": training_records,
        "claim_boundary": "Development-only anonymous interaction-local value supervision. It contains no context, episode, seed, or replay-start identity and no promotion episode data.",
    }
    _assert_anonymous(training)
    audit = {
        "schema_version": 1,
        "artifact_type": "verdiwm-ctrl-world-cclvr-value-bank-audit",
        "state": "ready",
        "forbidden_promotion_episodes": sorted(FORBIDDEN_PROMOTION_EPISODES),
        "identity_count": len(expected),
        "record_count": len(training_records),
        "source_hashes": source_hashes,
        "rows": audit_rows,
    }
    settlement = {
        "schema_version": 1,
        "artifact_type": "verdiwm-ctrl-world-cclvr-data-admission-settlement",
        "state": "admitted" if admitted else "blocked",
        "training_authorized": admitted,
        "identity_count": len(expected),
        "local_decision_count": len(training_records),
        "target_arm_counts": arm_counts,
        "support_checks": support_checks,
        "coverage_band": coverage,
        "minimum_benefit": minimum_benefit,
        "promotion_episode_present": False,
        "failure_action": None if admitted else admission_spec.get("failure_action"),
    }
    output_root.mkdir(mode=0o700, parents=True)
    training_path = output_root / "anonymous-local-value-bank.json"
    audit_path = output_root / "audit-receipt.json"
    settlement_path = output_root / "data-admission-settlement.json"
    _atomic_json(training_path, training)
    _atomic_json(audit_path, audit)
    _atomic_json(settlement_path, settlement)
    manifest = {
        "schema_version": 1,
        "artifact_type": "verdiwm-ctrl-world-cclvr-value-bank-manifest",
        "state": settlement["state"],
        "training_authorized": admitted,
        "training_path": str(training_path.resolve()),
        "training_sha256": _sha256(training_path),
        "audit_path": str(audit_path.resolve()),
        "audit_sha256": _sha256(audit_path),
        "settlement_path": str(settlement_path.resolve()),
        "settlement_sha256": _sha256(settlement_path),
        "record_count": len(training_records),
        "target_arm_counts": arm_counts,
        "coverage_band": coverage,
    }
    _atomic_json(output_root / "manifest.json", manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--shard-result", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = materialize(
        campaign_path=args.campaign,
        shard_paths=args.shard_result,
        output_root=args.output_root.resolve(),
    )
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0 if result["state"] == "admitted" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (CCLVRValueBankError, OSError, ValueError) as exc:
        print(str(exc), file=__import__("sys").stderr)
        raise SystemExit(2) from exc
