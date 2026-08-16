#!/usr/bin/env python3
"""Rank a development-only Ctrl-World anchor-repair screen and freeze top-1."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


class AnchorRepairSettlementError(ValueError):
    """Repair results do not form a valid paired development screen."""


def _load(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AnchorRepairSettlementError(f"CTRL_WORLD_REPAIR_JSON_INVALID:{path}") from exc
    if not isinstance(payload, Mapping):
        raise AnchorRepairSettlementError(f"CTRL_WORLD_REPAIR_JSON_INVALID:{path}")
    return payload


def _finite(value: object, code: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise AnchorRepairSettlementError(code)
    return float(value)


def _identity(raw: object) -> tuple[str, str, int, int]:
    if not isinstance(raw, Mapping):
        raise AnchorRepairSettlementError("CTRL_WORLD_REPAIR_IDENTITY_INVALID")
    values = (raw.get("context_id"), raw.get("episode_id"), raw.get("start_idx"), raw.get("seed"))
    if (
        not isinstance(values[0], str)
        or not values[0]
        or not isinstance(values[1], str)
        or not values[1]
        or not isinstance(values[2], int)
        or not isinstance(values[3], int)
    ):
        raise AnchorRepairSettlementError("CTRL_WORLD_REPAIR_IDENTITY_INVALID")
    return values  # type: ignore[return-value]


def _mean_standard_error(values: Sequence[float]) -> tuple[float, float]:
    if len(values) < 2:
        raise AnchorRepairSettlementError("CTRL_WORLD_REPAIR_REPEATS_INSUFFICIENT")
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(max(variance, 0.0) / len(values))


def _parse_result(
    path: Path,
) -> tuple[str, tuple[str, ...], dict[float, dict[tuple[str, str, int, int], tuple[float, ...]]], Mapping[str, Any]]:
    payload = _load(path)
    if (
        payload.get("artifact_type") != "verdiwm-ctrl-world-anchor-repair-result"
        or payload.get("state") != "ready"
        or not isinstance(payload.get("hook_activation"), Mapping)
        or payload["hook_activation"].get("state") != "passed"
        or any(
            not isinstance(row, Mapping) or row.get("state") != "passed"
            for row in payload.get("zero_identity_checks", ())
        )
    ):
        raise AnchorRepairSettlementError("CTRL_WORLD_REPAIR_RESULT_INVALID")
    primitive_id = payload.get("primitive_id")
    outcome_names = tuple(str(value) for value in payload.get("outcome_names", ()))
    measurements = payload.get("measurements")
    if not isinstance(primitive_id, str) or not primitive_id or not outcome_names or not isinstance(measurements, list):
        raise AnchorRepairSettlementError("CTRL_WORLD_REPAIR_RESULT_INVALID")
    rows: dict[float, dict[tuple[str, str, int, int], tuple[float, ...]]] = {}
    for raw in measurements:
        if not isinstance(raw, Mapping) or not isinstance(raw.get("outcomes"), Mapping):
            raise AnchorRepairSettlementError("CTRL_WORLD_REPAIR_MEASUREMENT_INVALID")
        strength = _finite(raw.get("strength"), "CTRL_WORLD_REPAIR_STRENGTH_INVALID")
        identity = _identity(raw.get("identity"))
        outcomes = raw["outcomes"]
        if set(outcomes) != set(outcome_names):
            raise AnchorRepairSettlementError("CTRL_WORLD_REPAIR_OUTCOMES_INVALID")
        vector = tuple(_finite(outcomes[name], "CTRL_WORLD_REPAIR_OUTCOMES_INVALID") for name in outcome_names)
        if identity in rows.setdefault(strength, {}):
            raise AnchorRepairSettlementError("CTRL_WORLD_REPAIR_MEASUREMENT_DUPLICATE")
        rows[strength][identity] = vector
    if 0.0 not in rows or len(rows) < 2:
        raise AnchorRepairSettlementError("CTRL_WORLD_REPAIR_STRENGTH_FRAME_INVALID")
    identities = set(rows[0.0])
    if len(identities) < 3 or any(set(values) != identities for values in rows.values()):
        raise AnchorRepairSettlementError("CTRL_WORLD_REPAIR_PAIRED_FRAME_INVALID")
    return primitive_id, outcome_names, rows, payload


def settle(
    *,
    result_paths: Sequence[Path],
    output_root: Path,
    outcome_weights: Sequence[float] = (1.0, 2.0, 2.0, 1.0, 1.0),
    confidence_z: float = 1.96,
    minimum_gain_lcb: float = 0.0,
) -> dict[str, object]:
    target = output_root.resolve()
    if target.exists() or target.is_symlink():
        raise AnchorRepairSettlementError("CTRL_WORLD_REPAIR_OUTPUT_EXISTS")
    confidence_z = _finite(confidence_z, "CTRL_WORLD_REPAIR_CONFIDENCE_INVALID")
    minimum_gain_lcb = _finite(minimum_gain_lcb, "CTRL_WORLD_REPAIR_THRESHOLD_INVALID")
    if confidence_z < 0.0:
        raise AnchorRepairSettlementError("CTRL_WORLD_REPAIR_CONFIDENCE_INVALID")

    parsed = []
    receipts = []
    primitive_ids: set[str] = set()
    canonical_names: tuple[str, ...] | None = None
    canonical_identities: set[tuple[str, str, int, int]] | None = None
    canonical_baseline: dict[tuple[str, str, int, int], tuple[float, ...]] | None = None
    checkpoint: str | None = None
    contexts_sha256: str | None = None
    for raw_path in result_paths:
        path = Path(raw_path).resolve(strict=True)
        primitive_id, names, rows, payload = _parse_result(path)
        if primitive_id in primitive_ids:
            raise AnchorRepairSettlementError("CTRL_WORLD_REPAIR_PRIMITIVE_DUPLICATE")
        primitive_ids.add(primitive_id)
        if canonical_names is None:
            canonical_names = names
        elif names != canonical_names:
            raise AnchorRepairSettlementError("CTRL_WORLD_REPAIR_OUTCOME_FRAME_MISMATCH")
        identities = set(rows[0.0])
        if canonical_identities is None:
            canonical_identities = identities
            canonical_baseline = rows[0.0]
        elif identities != canonical_identities or any(
            max(abs(left - right) for left, right in zip(rows[0.0][key], canonical_baseline[key], strict=True)) > 1e-6
            for key in identities
        ):
            raise AnchorRepairSettlementError("CTRL_WORLD_REPAIR_BASELINE_MISMATCH")
        raw_input = payload.get("input")
        if not isinstance(raw_input, Mapping):
            raise AnchorRepairSettlementError("CTRL_WORLD_REPAIR_INPUT_INVALID")
        current_checkpoint = str(raw_input.get("checkpoint", ""))
        current_contexts_sha = str(raw_input.get("contexts_sha256", ""))
        if not current_checkpoint or not current_contexts_sha:
            raise AnchorRepairSettlementError("CTRL_WORLD_REPAIR_INPUT_INVALID")
        if checkpoint is None:
            checkpoint, contexts_sha256 = current_checkpoint, current_contexts_sha
        elif current_checkpoint != checkpoint or current_contexts_sha != contexts_sha256:
            raise AnchorRepairSettlementError("CTRL_WORLD_REPAIR_INPUT_FRAME_MISMATCH")
        parsed.append((primitive_id, rows, payload))
        receipts.append({"primitive_id": primitive_id, "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})

    if canonical_names is None or canonical_identities is None or canonical_baseline is None:
        raise AnchorRepairSettlementError("CTRL_WORLD_REPAIR_RESULTS_EMPTY")
    weights = tuple(_finite(value, "CTRL_WORLD_REPAIR_WEIGHT_INVALID") for value in outcome_weights)
    if len(weights) != len(canonical_names) or any(value <= 0.0 for value in weights):
        raise AnchorRepairSettlementError("CTRL_WORLD_REPAIR_WEIGHT_INVALID")

    candidates: list[dict[str, object]] = []
    for primitive_id, rows, payload in parsed:
        for strength in sorted(value for value in rows if value > 0.0):
            per_identity: list[dict[str, object]] = []
            gains = []
            for identity in sorted(canonical_identities):
                gain = sum(
                    weight * (after - before)
                    for weight, after, before in zip(
                        weights, rows[strength][identity], canonical_baseline[identity], strict=True
                    )
                )
                gains.append(gain)
                per_identity.append(
                    {
                        "identity": {
                            "context_id": identity[0],
                            "episode_id": identity[1],
                            "start_idx": identity[2],
                            "seed": identity[3],
                        },
                        "weighted_gain": gain,
                    }
                )
            mean, standard_error = _mean_standard_error(gains)
            context_ids = sorted({identity[0] for identity in canonical_identities})
            per_context = []
            for context_id in context_ids:
                values = [
                    row["weighted_gain"]
                    for row in per_identity
                    if row["identity"]["context_id"] == context_id  # type: ignore[index]
                ]
                context_mean, context_se = _mean_standard_error(values)  # type: ignore[arg-type]
                per_context.append(
                    {
                        "context_id": context_id,
                        "mean_weighted_gain": context_mean,
                        "standard_error": context_se,
                        "positive": context_mean > 0.0,
                    }
                )
            candidates.append(
                {
                    "candidate_id": f"{primitive_id}:{strength:g}",
                    "primitive_id": primitive_id,
                    "primitive_type": payload.get("primitive_type"),
                    "target_hook": payload.get("target_hook"),
                    "strength": strength,
                    "mean_weighted_gain": mean,
                    "standard_error": standard_error,
                    "gain_lcb": mean - confidence_z * standard_error,
                    "positive_context_fraction": sum(row["positive"] for row in per_context) / len(per_context),
                    "per_context": per_context,
                    "per_identity": per_identity,
                }
            )
    ranked = sorted(
        candidates,
        key=lambda row: (-float(row["gain_lcb"]), -float(row["mean_weighted_gain"]), str(row["candidate_id"])),
    )
    if not ranked:
        raise AnchorRepairSettlementError("CTRL_WORLD_REPAIR_CANDIDATES_EMPTY")
    top = ranked[0]
    selected = top if float(top["gain_lcb"]) > minimum_gain_lcb else None
    target.mkdir(mode=0o700, parents=True)
    report = {
        "schema_version": 1,
        "artifact_type": "verdiwm-ctrl-world-anchor-repair-development-settlement",
        "state": "ready",
        "checkpoint": checkpoint,
        "contexts_sha256": contexts_sha256,
        "outcome_names": list(canonical_names),
        "outcome_weights": list(weights),
        "confidence_z": confidence_z,
        "minimum_gain_lcb": minimum_gain_lcb,
        "result_receipts": receipts,
        "candidate_count": len(ranked),
        "selector_action": "execute" if selected is not None else "abstain",
        "selected_candidate": selected,
        "ranked_candidates": ranked,
        "claim_boundary": (
            "Development-context frozen-checkpoint repair screen. Ranking is not an independent "
            "repair confirmation and cannot license a training or RSI claim."
        ),
    }
    report_path = target / "development-settlement.json"
    report_path.write_text(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    selector = {
        "schema_version": 1,
        "artifact_type": "verdiwm-ctrl-world-frozen-anchor-repair-selector",
        "state": "frozen",
        "development_settlement": {
            "path": str(report_path),
            "sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        },
        "selector_action": report["selector_action"],
        "selected_candidate": selected,
        "outcome_names": list(canonical_names),
        "outcome_weights": list(weights),
        "confidence_z": confidence_z,
        "minimum_gain_lcb": minimum_gain_lcb,
        "forbidden_confirm_adaptation": [
            "primitive_id",
            "strength",
            "outcome_weights",
            "confidence_z",
            "minimum_gain_lcb",
        ],
    }
    selector_path = target / "selector.json"
    selector_path.write_text(json.dumps(selector, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "artifact_type": "verdiwm-ctrl-world-anchor-repair-development-manifest",
        "state": "ready",
        "selector_action": report["selector_action"],
        "selected_candidate_id": selected["candidate_id"] if selected is not None else None,
        "development_settlement": {"path": str(report_path), "sha256": hashlib.sha256(report_path.read_bytes()).hexdigest()},
        "frozen_selector": {"path": str(selector_path), "sha256": hashlib.sha256(selector_path.read_bytes()).hexdigest()},
    }
    (target / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--outcome-weights", type=float, nargs="+", default=(1.0, 2.0, 2.0, 1.0, 1.0))
    parser.add_argument("--confidence-z", type=float, default=1.96)
    parser.add_argument("--minimum-gain-lcb", type=float, default=0.0)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = settle(
        result_paths=args.result,
        output_root=args.output_root,
        outcome_weights=args.outcome_weights,
        confidence_z=args.confidence_z,
        minimum_gain_lcb=args.minimum_gain_lcb,
    )
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
