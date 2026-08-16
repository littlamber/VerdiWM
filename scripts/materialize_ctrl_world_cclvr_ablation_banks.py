#!/usr/bin/env python3
"""Derive anonymous D1/D3/D4 CCLVR supervision banks from admitted evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


FORBIDDEN_KEYS = frozenset(("context_id", "episode_id", "probe_id", "seed", "start_idx"))
ARM_NAMES = ("negative", "zero", "positive")


class CCLVRAblationBankError(ValueError):
    """Admitted local evidence cannot be converted into an ablation bank."""


def _load(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CCLVRAblationBankError(f"CCLVR_ABLATION_JSON_INVALID:{path}") from exc
    if not isinstance(payload, Mapping):
        raise CCLVRAblationBankError(f"CCLVR_ABLATION_JSON_INVALID:{path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: object, error: str = "CCLVR_ABLATION_VALUE_INVALID") -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise CCLVRAblationBankError(error) from exc
    if not math.isfinite(result):
        raise CCLVRAblationBankError(error)
    return result


def _assert_anonymous(value: object) -> None:
    if isinstance(value, Mapping):
        if FORBIDDEN_KEYS.intersection(value):
            raise CCLVRAblationBankError("CCLVR_ABLATION_IDENTITY_LEAK")
        for child in value.values():
            _assert_anonymous(child)
    elif isinstance(value, list):
        for child in value:
            _assert_anonymous(child)


def _feature_tensor(value: object, shape: Sequence[int]) -> list[list[list[float]]]:
    if (
        not isinstance(value, list)
        or len(shape) != 3
        or len(value) != shape[0]
        or any(not isinstance(invocation, list) or len(invocation) != shape[1] for invocation in value)
    ):
        raise CCLVRAblationBankError("CCLVR_ABLATION_FEATURE_SHAPE_INVALID")
    result: list[list[list[float]]] = []
    for invocation in value:
        rows: list[list[float]] = []
        for row in invocation:
            if not isinstance(row, list) or len(row) != shape[2]:
                raise CCLVRAblationBankError("CCLVR_ABLATION_FEATURE_SHAPE_INVALID")
            rows.append([_finite(item) for item in row])
        result.append(rows)
    return result


def _arm_components(value: object) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or len(value) != 3 or any(not isinstance(row, Mapping) for row in value):
        raise CCLVRAblationBankError("CCLVR_ABLATION_COMPONENT_INVALID")
    return list(value)


def _arm_payload(losses: Sequence[float], doses: Sequence[float], minimum_benefit: float) -> dict[str, object]:
    if len(losses) != 3 or len(doses) != 3:
        raise CCLVRAblationBankError("CCLVR_ABLATION_ARM_INVALID")
    values = [losses[1] - loss for loss in losses]
    best_nonzero = 0 if values[0] >= values[2] else 2
    target_arm = best_nonzero if values[best_nonzero] >= minimum_benefit else 1
    return {
        "arm_losses": list(losses),
        "arm_values": values,
        "target_arm": target_arm,
        "target_dose": doses[target_arm],
    }


def _wilson(successes: int, total: int, z: float = 1.96) -> dict[str, float | int]:
    if total < 1 or not 0 <= successes <= total:
        raise CCLVRAblationBankError("CCLVR_ABLATION_COVERAGE_INVALID")
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


def _mean_episode_features(records: Sequence[Mapping[str, Any]], feature_shape: Sequence[int]) -> list[list[float]]:
    invocation_count, history_length, feature_dim = feature_shape
    denominator = len(records) * invocation_count
    totals = [[0.0] * feature_dim for _ in range(history_length)]
    for record in records:
        features = _feature_tensor(record.get("features"), feature_shape)
        for invocation in features:
            for history_index, row in enumerate(invocation):
                for feature_index, value in enumerate(row):
                    totals[history_index][feature_index] += value
    return [[value / denominator for value in row] for row in totals]


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _bank(
    *,
    variant: str,
    granularity: str,
    return_weights: Mapping[str, float],
    feature_shape: Sequence[int],
    records: Sequence[Mapping[str, object]],
    doses: Sequence[float],
    source_hashes: Mapping[str, str],
) -> dict[str, object]:
    counts = {name: 0 for name in ARM_NAMES}
    for record in records:
        target = int(record["target_arm"])
        if target not in (0, 1, 2):
            raise CCLVRAblationBankError("CCLVR_ABLATION_TARGET_INVALID")
        counts[ARM_NAMES[target]] += 1
    payload: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "verdiwm-ctrl-world-cclvr-anonymous-ablation-bank",
        "state": "ready",
        "variant": variant,
        "route_granularity": granularity,
        "feature_shape": list(feature_shape),
        "record_count": len(records),
        "arm_doses": list(doses),
        "target_arm_counts": counts,
        "coverage_band": _wilson(counts["negative"] + counts["positive"], len(records)),
        "return_weights": dict(return_weights),
        "source_hashes": dict(source_hashes),
        "records": list(records),
        "claim_boundary": "Anonymous development supervision for the authorized D1-D4 64-step screen only; no promotion episode evidence is present.",
    }
    _assert_anonymous(payload)
    return payload


def materialize(*, source_bank_path: Path, audit_path: Path, output_root: Path) -> Mapping[str, object]:
    if output_root.exists() or output_root.is_symlink():
        raise CCLVRAblationBankError("CCLVR_ABLATION_OUTPUT_EXISTS")
    source_bank_path = source_bank_path.resolve(strict=True)
    audit_path = audit_path.resolve(strict=True)
    source = _load(source_bank_path)
    audit = _load(audit_path)
    records = source.get("records")
    audit_rows = audit.get("rows")
    feature_shape = source.get("feature_shape")
    doses = source.get("arm_doses")
    local_return = source.get("local_return")
    if (
        source.get("artifact_type") != "verdiwm-ctrl-world-cclvr-anonymous-local-value-bank"
        or source.get("state") != "ready"
        or not isinstance(records, list)
        or int(source.get("record_count", -1)) != len(records)
        or not isinstance(feature_shape, list)
        or len(feature_shape) != 3
        or any(not isinstance(value, int) or value < 1 for value in feature_shape)
        or not isinstance(doses, list)
        or len(doses) != 3
        or not isinstance(local_return, Mapping)
    ):
        raise CCLVRAblationBankError("CCLVR_ABLATION_SOURCE_INVALID")
    _assert_anonymous(source)
    if (
        audit.get("artifact_type") != "verdiwm-ctrl-world-cclvr-value-bank-audit"
        or audit.get("state") != "ready"
        or not isinstance(audit_rows, list)
        or int(audit.get("record_count", -1)) != len(records)
        or "1799" not in {str(value) for value in audit.get("forbidden_promotion_episodes", ())}
    ):
        raise CCLVRAblationBankError("CCLVR_ABLATION_AUDIT_INVALID")
    if any(
        isinstance(row, Mapping)
        and isinstance(row.get("identity"), Mapping)
        and str(row["identity"].get("episode_id")) == "1799"
        for row in audit_rows
    ):
        raise CCLVRAblationBankError("CCLVR_ABLATION_PROMOTION_EPISODE_FORBIDDEN")

    dose_values = [_finite(value) for value in doses]
    minimum_benefit = _finite(local_return.get("minimum_benefit"))
    source_hashes = {
        "anonymous_local_value_bank_sha256": _sha256(source_bank_path),
        "value_bank_audit_sha256": _sha256(audit_path),
    }
    episode_records: list[dict[str, object]] = []
    local_suffix_records: list[dict[str, object]] = []
    full_return_records: list[dict[str, object]] = []
    covered_offsets: set[int] = set()
    for audit_row in audit_rows:
        if not isinstance(audit_row, Mapping):
            raise CCLVRAblationBankError("CCLVR_ABLATION_AUDIT_ROW_INVALID")
        offset = int(audit_row.get("training_record_offset", -1))
        count = int(audit_row.get("training_record_count", -1))
        interactions = audit_row.get("interactions")
        if (
            offset < 0
            or count < 1
            or offset + count > len(records)
            or not isinstance(interactions, list)
            or len(interactions) != count
            or covered_offsets.intersection(range(offset, offset + count))
        ):
            raise CCLVRAblationBankError("CCLVR_ABLATION_AUDIT_ROW_INVALID")
        covered_offsets.update(range(offset, offset + count))
        source_group = records[offset : offset + count]
        if any(not isinstance(record, Mapping) for record in source_group):
            raise CCLVRAblationBankError("CCLVR_ABLATION_SOURCE_RECORD_INVALID")
        episode_losses = [0.0, 0.0, 0.0]
        for local_index, (source_record, interaction) in enumerate(zip(source_group, interactions, strict=True)):
            if not isinstance(interaction, Mapping) or int(interaction.get("interaction_index", -1)) != local_index:
                raise CCLVRAblationBankError("CCLVR_ABLATION_INTERACTION_INVALID")
            components = _arm_components(interaction.get("arm_components"))
            suffix_losses = [
                _finite(component.get("suffix_mean_l1"), "CCLVR_ABLATION_COMPONENT_INVALID")
                for component in components
            ]
            for arm, loss in enumerate(suffix_losses):
                episode_losses[arm] += loss / count
            features = _feature_tensor(source_record.get("features"), feature_shape)
            suffix_record = {
                "features": features,
                **_arm_payload(suffix_losses, dose_values, minimum_benefit),
            }
            local_suffix_records.append(suffix_record)
            full_losses = [_finite(value) for value in source_record.get("arm_losses", ())]
            if len(full_losses) != 3:
                raise CCLVRAblationBankError("CCLVR_ABLATION_SOURCE_RECORD_INVALID")
            full_return_records.append(
                {
                    "features": features,
                    **_arm_payload(full_losses, dose_values, minimum_benefit),
                }
            )
        episode_records.append(
            {
                "features": _mean_episode_features(source_group, feature_shape),
                **_arm_payload(episode_losses, dose_values, minimum_benefit),
            }
        )
    if covered_offsets != set(range(len(records))):
        raise CCLVRAblationBankError("CCLVR_ABLATION_AUDIT_COVERAGE_INVALID")

    banks = {
        "episode_suffix_mean": _bank(
            variant="episode_suffix_mean",
            granularity="episode",
            return_weights={"suffix_mean_l1": 1.0, "terminal_interaction_l1": 0.0, "suffix_horizon_l1_slope": 0.0},
            feature_shape=feature_shape[1:],
            records=episode_records,
            doses=dose_values,
            source_hashes=source_hashes,
        ),
        "interaction_suffix_mean": _bank(
            variant="interaction_suffix_mean",
            granularity="interaction",
            return_weights={"suffix_mean_l1": 1.0, "terminal_interaction_l1": 0.0, "suffix_horizon_l1_slope": 0.0},
            feature_shape=feature_shape,
            records=local_suffix_records,
            doses=dose_values,
            source_hashes=source_hashes,
        ),
        "interaction_terminal_horizon": _bank(
            variant="interaction_terminal_horizon",
            granularity="interaction",
            return_weights={
                "suffix_mean_l1": _finite(local_return.get("suffix_mean_l1_weight")),
                "terminal_interaction_l1": _finite(local_return.get("terminal_interaction_l1_weight")),
                "suffix_horizon_l1_slope": _finite(local_return.get("suffix_horizon_l1_slope_weight")),
            },
            feature_shape=feature_shape,
            records=full_return_records,
            doses=dose_values,
            source_hashes=source_hashes,
        ),
    }
    output_root.mkdir(mode=0o700, parents=True)
    manifest_banks: dict[str, object] = {}
    for variant, payload in banks.items():
        path = output_root / f"{variant}.json"
        _write_json(path, payload)
        manifest_banks[variant] = {
            "path": str(path.resolve()),
            "sha256": _sha256(path),
            "record_count": payload["record_count"],
            "target_arm_counts": payload["target_arm_counts"],
            "coverage_band": payload["coverage_band"],
        }
    manifest = {
        "schema_version": 1,
        "artifact_type": "verdiwm-ctrl-world-cclvr-ablation-bank-manifest",
        "state": "ready",
        "source_hashes": source_hashes,
        "banks": manifest_banks,
        "promotion_episode_present": False,
    }
    _write_json(output_root / "manifest.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bank", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    result = materialize(
        source_bank_path=args.source_bank,
        audit_path=args.audit,
        output_root=args.output_root.resolve(),
    )
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (CCLVRAblationBankError, OSError, ValueError) as exc:
        print(str(exc), file=__import__("sys").stderr)
        raise SystemExit(2) from exc
