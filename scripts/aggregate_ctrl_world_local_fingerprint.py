#!/usr/bin/env python3
"""Assemble source-isolated Ctrl-World paired-dose results into local charts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from wmloop.geometry.irg import estimate_response_chart


OUTCOME_NAMES = (
    "negative_mean_l1",
    "negative_final_interaction_l1",
    "negative_horizon_l1_slope",
    "mean_psnr",
    "final_psnr",
)


class LocalFingerprintAggregationError(ValueError):
    """Raw probe evidence cannot form a trustworthy target-local chart."""


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LocalFingerprintAggregationError(f"LOCAL_FINGERPRINT_RESULT_INVALID:{path}") from exc
    if not isinstance(payload, Mapping):
        raise LocalFingerprintAggregationError(f"LOCAL_FINGERPRINT_RESULT_INVALID:{path}")
    return payload


def _finite(value: object, code: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise LocalFingerprintAggregationError(code)
    return float(value)


def _identity(raw: object) -> tuple[str, str, int, int]:
    if not isinstance(raw, Mapping):
        raise LocalFingerprintAggregationError("LOCAL_FINGERPRINT_MEASUREMENT_IDENTITY_INVALID")
    context_id = raw.get("context_id")
    episode_id = raw.get("episode_id")
    start_idx = raw.get("start_idx")
    seed = raw.get("seed")
    if (
        not isinstance(context_id, str)
        or not context_id
        or not isinstance(episode_id, str)
        or not episode_id
        or not isinstance(start_idx, int)
        or start_idx < 0
        or not isinstance(seed, int)
    ):
        raise LocalFingerprintAggregationError("LOCAL_FINGERPRINT_MEASUREMENT_IDENTITY_INVALID")
    return context_id, episode_id, start_idx, seed


def _vector(raw: object) -> tuple[float, ...]:
    if not isinstance(raw, Mapping) or set(raw) != set(OUTCOME_NAMES):
        raise LocalFingerprintAggregationError("LOCAL_FINGERPRINT_OUTCOME_SCHEMA_INVALID")
    return tuple(_finite(raw[name], "LOCAL_FINGERPRINT_OUTCOME_NONFINITE") for name in OUTCOME_NAMES)


def _load_result(path: Path, *, campaign_id: str) -> tuple[str, dict[float, dict[tuple[str, str, int, int], tuple[float, ...]]], dict[str, object]]:
    payload = _load_json(path)
    if (
        payload.get("artifact_type") != "verdiwm-ctrl-world-local-fingerprint-probe-result"
        or payload.get("state") != "ready"
        or payload.get("campaign_id") != campaign_id
        or tuple(payload.get("outcome_names", ())) != OUTCOME_NAMES
    ):
        raise LocalFingerprintAggregationError("LOCAL_FINGERPRINT_RESULT_NOT_ADMITTED")
    probe_id = payload.get("probe_id")
    if not isinstance(probe_id, str) or not probe_id:
        raise LocalFingerprintAggregationError("LOCAL_FINGERPRINT_PROBE_ID_INVALID")
    checks = payload.get("zero_identity_checks")
    activation = payload.get("hook_activation")
    if (
        not isinstance(checks, list)
        or not checks
        or any(not isinstance(row, Mapping) or row.get("state") != "passed" for row in checks)
        or not isinstance(activation, Mapping)
        or activation.get("state") != "passed"
    ):
        raise LocalFingerprintAggregationError("LOCAL_FINGERPRINT_INVARIANT_GATE_FAILED")
    measurements = payload.get("measurements")
    if not isinstance(measurements, list) or not measurements:
        raise LocalFingerprintAggregationError("LOCAL_FINGERPRINT_MEASUREMENTS_EMPTY")
    by_dose: dict[float, dict[tuple[str, str, int, int], tuple[float, ...]]] = {}
    for row in measurements:
        if not isinstance(row, Mapping):
            raise LocalFingerprintAggregationError("LOCAL_FINGERPRINT_MEASUREMENT_INVALID")
        dose = _finite(row.get("dose"), "LOCAL_FINGERPRINT_MEASUREMENT_DOSE_INVALID")
        identity = _identity(row.get("identity"))
        values = _vector(row.get("outcomes"))
        if identity in by_dose.setdefault(dose, {}):
            raise LocalFingerprintAggregationError("LOCAL_FINGERPRINT_MEASUREMENT_DUPLICATE")
        by_dose[dose][identity] = values
    if 0.0 not in by_dose or len(by_dose) < 3:
        raise LocalFingerprintAggregationError("LOCAL_FINGERPRINT_DOSE_FRAME_INVALID")
    baseline = set(by_dose[0.0])
    if not baseline or any(set(rows) != baseline for rows in by_dose.values()):
        raise LocalFingerprintAggregationError("LOCAL_FINGERPRINT_PAIRED_FRAME_INVALID")
    positive = {dose for dose in by_dose if dose > 0.0}
    if not positive or not any(-dose in by_dose for dose in positive):
        raise LocalFingerprintAggregationError("LOCAL_FINGERPRINT_SYMMETRIC_DOSE_MISSING")
    receipt = {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "probe_id": probe_id,
        "base_probe_family": payload.get("base_probe_family"),
        "measurement_count": len(measurements),
        "identity_count": len(baseline),
        "doses": sorted(by_dose),
    }
    return probe_id, by_dose, receipt


def aggregate(
    *,
    campaign_path: Path,
    result_paths: Sequence[Path],
    output_root: Path,
) -> dict[str, object]:
    campaign = _load_json(campaign_path)
    if campaign.get("artifact_type") != "verdiwm-ctrl-world-local-fingerprint-campaign":
        raise LocalFingerprintAggregationError("LOCAL_FINGERPRINT_CAMPAIGN_INVALID")
    protocol = campaign.get("protocol")
    if not isinstance(protocol, Mapping):
        raise LocalFingerprintAggregationError("LOCAL_FINGERPRINT_CAMPAIGN_PROTOCOL_INVALID")
    campaign_id = campaign.get("campaign_id")
    probe_specs = campaign.get("probe_paths")
    outcome_specs = campaign.get("outcomes")
    locality = campaign.get("locality_admission")
    if (
        not isinstance(campaign_id, str)
        or not campaign_id
        or not isinstance(probe_specs, list)
        or not isinstance(outcome_specs, list)
        or not isinstance(locality, Mapping)
    ):
        raise LocalFingerprintAggregationError("LOCAL_FINGERPRINT_CAMPAIGN_INVALID")
    expected_probes = {str(row.get("probe_id")) for row in probe_specs if isinstance(row, Mapping)}
    if not expected_probes or len(expected_probes) != len(probe_specs):
        raise LocalFingerprintAggregationError("LOCAL_FINGERPRINT_CAMPAIGN_PROBES_INVALID")
    if tuple(str(row.get("name")) for row in outcome_specs if isinstance(row, Mapping)) != OUTCOME_NAMES:
        raise LocalFingerprintAggregationError("LOCAL_FINGERPRINT_CAMPAIGN_OUTCOMES_INVALID")
    weights = tuple(_finite(row.get("weight"), "LOCAL_FINGERPRINT_OUTCOME_WEIGHT_INVALID") for row in outcome_specs)
    if any(value <= 0.0 for value in weights):
        raise LocalFingerprintAggregationError("LOCAL_FINGERPRINT_OUTCOME_WEIGHT_INVALID")
    threshold = _finite(locality.get("maximum_residual"), "LOCAL_FINGERPRINT_LOCALITY_THRESHOLD_INVALID")
    if threshold < 0.0:
        raise LocalFingerprintAggregationError("LOCAL_FINGERPRINT_LOCALITY_THRESHOLD_INVALID")
    if output_root.exists() or output_root.is_symlink():
        raise LocalFingerprintAggregationError("LOCAL_FINGERPRINT_OUTPUT_EXISTS")

    results: dict[str, dict[float, dict[tuple[str, str, int, int], tuple[float, ...]]]] = {}
    input_receipts: list[dict[str, object]] = []
    for raw_path in result_paths:
        path = Path(raw_path).resolve(strict=True)
        probe_id, rows, receipt = _load_result(path, campaign_id=campaign_id)
        if probe_id in results:
            raise LocalFingerprintAggregationError("LOCAL_FINGERPRINT_PROBE_RESULT_DUPLICATE")
        results[probe_id] = rows
        input_receipts.append(receipt)
    if set(results) != expected_probes:
        raise LocalFingerprintAggregationError(
            f"LOCAL_FINGERPRINT_PROBE_RESULT_INCOMPLETE:{sorted(expected_probes - set(results))}"
        )

    identities_by_probe = {probe_id: set(rows[0.0]) for probe_id, rows in results.items()}
    identities = next(iter(identities_by_probe.values()))
    if not identities or any(rows != identities for rows in identities_by_probe.values()):
        raise LocalFingerprintAggregationError("LOCAL_FINGERPRINT_CROSS_PROBE_FRAME_INVALID")
    contexts: dict[tuple[str, str, int], list[tuple[str, str, int, int]]] = {}
    for identity in sorted(identities):
        contexts.setdefault(identity[:3], []).append(identity)
    required_repeats = int(protocol.get("minimum_seed_repeats", 1))
    if required_repeats < 1 or any(len(rows) < required_repeats for rows in contexts.values()):
        raise LocalFingerprintAggregationError("LOCAL_FINGERPRINT_SEED_REPEATS_INSUFFICIENT")

    # Every base path must replay the same zero-dose experiment.  A mismatch
    # here means the purported paired frame is contaminated by RNG, stateful
    # scheduler behavior, or an unintended hook side effect.
    reference_probe = next(iter(sorted(results)))
    for identity in sorted(identities):
        reference = results[reference_probe][0.0][identity]
        for probe_id in sorted(results):
            observed = results[probe_id][0.0][identity]
            if max(abs(a - b) for a, b in zip(reference, observed, strict=True)) > 1e-6:
                raise LocalFingerprintAggregationError("LOCAL_FINGERPRINT_ZERO_DOSE_CROSS_PROBE_MISMATCH")

    context_reports: list[dict[str, object]] = []
    for (context_id, episode_id, start_idx), repeat_identities in sorted(contexts.items()):
        doses_by_path = {
            probe_id: {
                dose: tuple(by_identity[identity] for identity in repeat_identities)
                for dose, by_identity in sorted(rows.items())
                if dose != 0.0
            }
            for probe_id, rows in results.items()
        }
        chart = estimate_response_chart(
            chart_id=f"{campaign_id}:{context_id}",
            goal_schema="ctrl_world_replay_local_predictive_v1",
            outcome_names=OUTCOME_NAMES,
            outcome_weights=weights,
            baseline_repeats=tuple(results[reference_probe][0.0][identity] for identity in repeat_identities),
            dose_observations=doses_by_path,
        ).to_dict()
        residuals = {name: float(value) for name, value in chart["locality_residuals"].items()}
        supported = sorted(name for name, value in residuals.items() if value <= threshold)
        baseline = {
            name: float(sum(results[reference_probe][0.0][identity][index] for identity in repeat_identities) / len(repeat_identities))
            for index, name in enumerate(OUTCOME_NAMES)
        }
        context_reports.append(
            {
                "context": {
                    "context_id": context_id,
                    "episode_id": episode_id,
                    "start_idx": start_idx,
                    "seeds": [identity[3] for identity in repeat_identities],
                },
                "baseline_outcomes": baseline,
                "chart": chart,
                "locality_admission": {
                    "state": "passed" if len(supported) == len(expected_probes) else "partial_or_failed",
                    "maximum_residual": threshold,
                    "path_residuals": residuals,
                    "supported_local_paths": supported,
                    "unsupported_paths": sorted(expected_probes - set(supported)),
                },
            }
        )
    report = {
        "schema_version": 1,
        "artifact_type": "verdiwm-ctrl-world-context-local-fingerprint-atlas",
        "state": "ready",
        "campaign_id": campaign_id,
        "context_count": len(context_reports),
        "probe_count": len(expected_probes),
        "outcome_names": list(OUTCOME_NAMES),
        "outcome_weights": list(weights),
        "input_receipts": input_receipts,
        "contexts": context_reports,
        "routing_readiness": {
            "state": "not_licensed",
            "reason": (
                "Probe locality can admit response coordinates but cannot license repair ranking "
                "until an independently held-out selector-gain comparison is supplied."
            ),
            "required_next_evidence": "paired held-out comparison of fingerprint-guided repair ranking against no-fingerprint ranking",
        },
        "claim_boundary": campaign.get("claim_scope"),
    }
    output_root.mkdir(mode=0o700, parents=True)
    (output_root / "context-local-fingerprint-atlas.json").write_text(
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "artifact_type": "verdiwm-ctrl-world-context-local-fingerprint-atlas-manifest",
        "state": "ready",
        "campaign_id": campaign_id,
        "context_count": len(context_reports),
        "report_path": str((output_root / "context-local-fingerprint-atlas.json").resolve()),
        "report_sha256": hashlib.sha256(
            (output_root / "context-local-fingerprint-atlas.json").read_bytes()
        ).hexdigest(),
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--probe-result", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = aggregate(
        campaign_path=args.campaign.resolve(strict=True),
        result_paths=args.probe_result,
        output_root=args.output_root.resolve(),
    )
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
