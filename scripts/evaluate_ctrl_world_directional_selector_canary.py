#!/usr/bin/env python3
"""Test whether a narrow Ctrl-World fingerprint predicts wider dose effects."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


class DirectionalSelectorCanaryError(ValueError):
    """Fingerprint and candidate effects do not form a non-leaking canary."""


def _load(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DirectionalSelectorCanaryError(f"DIRECTIONAL_SELECTOR_JSON_INVALID:{path}") from exc
    if not isinstance(payload, Mapping):
        raise DirectionalSelectorCanaryError(f"DIRECTIONAL_SELECTOR_JSON_INVALID:{path}")
    return payload


def _finite(value: object, code: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise DirectionalSelectorCanaryError(code)
    return float(value)


def _mean_standard_error(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        raise DirectionalSelectorCanaryError("DIRECTIONAL_SELECTOR_EFFECT_VALUES_EMPTY")
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(max(variance, 0.0) / len(values))


def _identity(raw: object) -> tuple[str, str, int, int]:
    if not isinstance(raw, Mapping):
        raise DirectionalSelectorCanaryError("DIRECTIONAL_SELECTOR_IDENTITY_INVALID")
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
        or not isinstance(seed, int)
    ):
        raise DirectionalSelectorCanaryError("DIRECTIONAL_SELECTOR_IDENTITY_INVALID")
    return context_id, episode_id, start_idx, seed


def _result(path: Path) -> tuple[str, dict[float, dict[tuple[str, str, int, int], tuple[float, ...]]], tuple[str, ...]]:
    payload = _load(path)
    if (
        payload.get("artifact_type") != "verdiwm-ctrl-world-local-fingerprint-probe-result"
        or payload.get("state") != "ready"
        or payload.get("hook_activation", {}).get("state") != "passed"
    ):
        raise DirectionalSelectorCanaryError("DIRECTIONAL_SELECTOR_CANDIDATE_RESULT_INVALID")
    probe_id = payload.get("probe_id")
    outcome_names = tuple(str(value) for value in payload.get("outcome_names", ()))
    measurements = payload.get("measurements")
    if not isinstance(probe_id, str) or not probe_id or not outcome_names or not isinstance(measurements, list):
        raise DirectionalSelectorCanaryError("DIRECTIONAL_SELECTOR_CANDIDATE_RESULT_INVALID")
    by_dose: dict[float, dict[tuple[str, str, int, int], tuple[float, ...]]] = {}
    for row in measurements:
        if not isinstance(row, Mapping) or not isinstance(row.get("outcomes"), Mapping):
            raise DirectionalSelectorCanaryError("DIRECTIONAL_SELECTOR_MEASUREMENT_INVALID")
        dose = _finite(row.get("dose"), "DIRECTIONAL_SELECTOR_DOSE_INVALID")
        identity = _identity(row.get("identity"))
        outcomes = row["outcomes"]
        if set(outcomes) != set(outcome_names):
            raise DirectionalSelectorCanaryError("DIRECTIONAL_SELECTOR_OUTCOME_FRAME_INVALID")
        vector = tuple(_finite(outcomes[name], "DIRECTIONAL_SELECTOR_OUTCOME_INVALID") for name in outcome_names)
        if identity in by_dose.setdefault(dose, {}):
            raise DirectionalSelectorCanaryError("DIRECTIONAL_SELECTOR_MEASUREMENT_DUPLICATE")
        by_dose[dose][identity] = vector
    nonzero = sorted((dose for dose in by_dose if dose != 0.0), key=abs)
    if 0.0 not in by_dose or len(nonzero) != 2 or not math.isclose(nonzero[0], -nonzero[1], abs_tol=1e-12):
        raise DirectionalSelectorCanaryError("DIRECTIONAL_SELECTOR_SYMMETRIC_CANDIDATES_REQUIRED")
    identities = set(by_dose[0.0])
    if not identities or any(set(rows) != identities for rows in by_dose.values()):
        raise DirectionalSelectorCanaryError("DIRECTIONAL_SELECTOR_PAIRED_FRAME_INVALID")
    return probe_id, by_dose, outcome_names


def evaluate(
    *,
    fingerprint_atlas_path: Path,
    candidate_result_paths: Sequence[Path],
    output_root: Path,
    confidence_z: float = 0.0,
    minimum_gain_lcb: float = -1e300,
    independent_contexts: bool = False,
    frozen_selector_path: Path | None = None,
) -> dict[str, object]:
    atlas_path = fingerprint_atlas_path.resolve(strict=True)
    atlas = _load(atlas_path)
    if (
        atlas.get("artifact_type") != "verdiwm-ctrl-world-context-local-fingerprint-atlas"
        or atlas.get("state") != "ready"
    ):
        raise DirectionalSelectorCanaryError("DIRECTIONAL_SELECTOR_FINGERPRINT_INVALID")
    outcome_names = tuple(str(value) for value in atlas.get("outcome_names", ()))
    weights = tuple(_finite(value, "DIRECTIONAL_SELECTOR_WEIGHT_INVALID") for value in atlas.get("outcome_weights", ()))
    if not outcome_names or len(weights) != len(outcome_names) or any(value <= 0.0 for value in weights):
        raise DirectionalSelectorCanaryError("DIRECTIONAL_SELECTOR_OUTCOME_FRAME_INVALID")
    if output_root.exists() or output_root.is_symlink():
        raise DirectionalSelectorCanaryError("DIRECTIONAL_SELECTOR_OUTPUT_EXISTS")
    confidence_z = _finite(confidence_z, "DIRECTIONAL_SELECTOR_CONFIDENCE_INVALID")
    minimum_gain_lcb = _finite(minimum_gain_lcb, "DIRECTIONAL_SELECTOR_GAIN_THRESHOLD_INVALID")
    if confidence_z < 0.0:
        raise DirectionalSelectorCanaryError("DIRECTIONAL_SELECTOR_CONFIDENCE_INVALID")
    candidate_results: dict[str, dict[float, dict[tuple[str, str, int, int], tuple[float, ...]]]] = {}
    candidate_receipts: list[dict[str, object]] = []
    candidate_radius: float | None = None
    for raw_path in candidate_result_paths:
        path = Path(raw_path).resolve(strict=True)
        probe_id, rows, names = _result(path)
        if names != outcome_names or probe_id in candidate_results:
            raise DirectionalSelectorCanaryError("DIRECTIONAL_SELECTOR_CANDIDATE_FRAME_INVALID")
        radius = max(abs(dose) for dose in rows)
        if candidate_radius is None:
            candidate_radius = radius
        elif not math.isclose(candidate_radius, radius, abs_tol=1e-12):
            raise DirectionalSelectorCanaryError("DIRECTIONAL_SELECTOR_CANDIDATE_RADIUS_MISMATCH")
        candidate_results[probe_id] = rows
        candidate_receipts.append(
            {"probe_id": probe_id, "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        )
    if candidate_radius is None:
        raise DirectionalSelectorCanaryError("DIRECTIONAL_SELECTOR_CANDIDATES_EMPTY")
    frozen_selector: Mapping[str, Any] | None = None
    frozen_selector_receipt: dict[str, object] | None = None
    if frozen_selector_path is not None:
        selector_path = Path(frozen_selector_path).resolve(strict=True)
        frozen_selector = _load(selector_path)
        if (
            frozen_selector.get("artifact_type") != "verdiwm-ctrl-world-frozen-directional-selector"
            or frozen_selector.get("state") != "frozen"
            or frozen_selector.get("fingerprint_atlas", {}).get("sha256")
            != hashlib.sha256(atlas_path.read_bytes()).hexdigest()
            or not math.isclose(
                _finite(frozen_selector.get("candidate_radius"), "DIRECTIONAL_SELECTOR_FROZEN_RADIUS_INVALID"),
                candidate_radius,
                abs_tol=1e-12,
            )
        ):
            raise DirectionalSelectorCanaryError("DIRECTIONAL_SELECTOR_FROZEN_SELECTOR_INVALID")
        frozen_selector_receipt = {
            "path": str(selector_path),
            "sha256": hashlib.sha256(selector_path.read_bytes()).hexdigest(),
        }

    context_reports: list[dict[str, object]] = []
    for context_row in atlas.get("contexts", ()):
        if not isinstance(context_row, Mapping):
            raise DirectionalSelectorCanaryError("DIRECTIONAL_SELECTOR_CONTEXT_INVALID")
        context = context_row.get("context")
        chart = context_row.get("chart")
        locality = context_row.get("locality_admission")
        if not isinstance(context, Mapping) or not isinstance(chart, Mapping) or not isinstance(locality, Mapping):
            raise DirectionalSelectorCanaryError("DIRECTIONAL_SELECTOR_CONTEXT_INVALID")
        context_id = str(context.get("context_id", ""))
        episode_id = str(context.get("episode_id", ""))
        start_idx = int(context.get("start_idx", -1))
        seeds = tuple(int(value) for value in context.get("seeds", ()))
        intervention_names = tuple(str(value) for value in chart.get("intervention_names", ()))
        chart_outcomes = tuple(str(value) for value in chart.get("outcome_names", ()))
        jacobian = chart.get("jacobian")
        covariance = chart.get("covariance")
        repeat_count = int(chart.get("repeat_count", 0))
        supported = set(str(value) for value in locality.get("supported_local_paths", ()))
        if (
            not context_id
            or not episode_id
            or start_idx < 0
            or not seeds
            or chart_outcomes != outcome_names
            or not isinstance(jacobian, list)
            or len(jacobian) != len(outcome_names)
            or not isinstance(covariance, list)
            or repeat_count < 2
        ):
            raise DirectionalSelectorCanaryError("DIRECTIONAL_SELECTOR_CONTEXT_INVALID")
        eligible = sorted(supported & set(candidate_results) & set(intervention_names))
        if not eligible:
            continue
        candidates: list[dict[str, object]] = []
        for probe_id in eligible:
            path_index = intervention_names.index(probe_id)
            slope = tuple(_finite(row[path_index], "DIRECTIONAL_SELECTOR_JACOBIAN_INVALID") for row in jacobian)
            coordinate_width = len(outcome_names) * len(intervention_names)
            if len(covariance) != coordinate_width or any(
                not isinstance(row, list) or len(row) != coordinate_width for row in covariance
            ):
                raise DirectionalSelectorCanaryError("DIRECTIONAL_SELECTOR_COVARIANCE_INVALID")
            coefficients = [0.0] * coordinate_width
            for outcome_index, weight in enumerate(weights):
                coefficients[outcome_index * len(intervention_names) + path_index] = math.sqrt(weight)
            slope_variance = sum(
                coefficients[i]
                * _finite(covariance[i][j], "DIRECTIONAL_SELECTOR_COVARIANCE_INVALID")
                * coefficients[j]
                for i in range(coordinate_width)
                for j in range(coordinate_width)
            )
            slope_standard_error = math.sqrt(max(slope_variance, 0.0) / repeat_count)
            rows = candidate_results[probe_id]
            identities = [(context_id, episode_id, start_idx, seed) for seed in seeds]
            if any(identity not in rows[0.0] for identity in identities):
                raise DirectionalSelectorCanaryError("DIRECTIONAL_SELECTOR_CONTEXT_EFFECT_MISSING")
            for dose in sorted(value for value in rows if value != 0.0):
                predicted_gain = dose * sum(weight * value for weight, value in zip(weights, slope, strict=True))
                predicted_standard_error = abs(dose) * slope_standard_error
                predicted_gain_lcb = predicted_gain - confidence_z * predicted_standard_error
                per_seed = []
                for identity in identities:
                    treated = rows[dose][identity]
                    baseline = rows[0.0][identity]
                    per_seed.append(
                        sum(
                            weight * (after - before)
                            for weight, after, before in zip(weights, treated, baseline, strict=True)
                        )
                    )
                actual_gain = sum(per_seed) / len(per_seed)
                candidates.append(
                    {
                        "candidate_id": f"{probe_id}:{dose:+g}",
                        "probe_id": probe_id,
                        "dose": dose,
                        "predicted_weighted_gain": predicted_gain,
                        "predicted_weighted_gain_standard_error": predicted_standard_error,
                        "predicted_weighted_gain_lcb": predicted_gain_lcb,
                        "actual_weighted_gain": actual_gain,
                        "per_seed_actual_weighted_gain": per_seed,
                    }
                )
        ranked = sorted(
            candidates,
            key=lambda row: (
                -float(row["predicted_weighted_gain_lcb"]),
                -float(row["predicted_weighted_gain"]),
                str(row["candidate_id"]),
            ),
        )
        selected = ranked[0] if float(ranked[0]["predicted_weighted_gain_lcb"]) > minimum_gain_lcb else None
        if frozen_selector is not None:
            selector_rows = [
                item
                for item in frozen_selector.get("contexts", ())
                if isinstance(item, Mapping)
                and isinstance(item.get("context"), Mapping)
                and item["context"].get("context_id") == context_id
            ]
            if len(selector_rows) != 1:
                raise DirectionalSelectorCanaryError("DIRECTIONAL_SELECTOR_FROZEN_CONTEXT_MISSING")
            frozen_row = selector_rows[0]
            frozen_selected = frozen_row.get("selected_candidate")
            if frozen_row.get("selector_action") == "abstain":
                selected = None
            elif isinstance(frozen_selected, Mapping):
                candidate_id = frozen_selected.get("candidate_id")
                matches = [item for item in ranked if item["candidate_id"] == candidate_id]
                if len(matches) != 1:
                    raise DirectionalSelectorCanaryError("DIRECTIONAL_SELECTOR_FROZEN_CANDIDATE_MISSING")
                selected = matches[0]
            else:
                raise DirectionalSelectorCanaryError("DIRECTIONAL_SELECTOR_FROZEN_ACTION_INVALID")
        oracle_candidate = max(candidates, key=lambda row: (float(row["actual_weighted_gain"]), str(row["candidate_id"])))
        oracle = oracle_candidate if float(oracle_candidate["actual_weighted_gain"]) > 0.0 else {
            "candidate_id": "no_intervention",
            "probe_id": None,
            "dose": 0.0,
            "actual_weighted_gain": 0.0,
        }
        uniform_gain = sum(float(row["actual_weighted_gain"]) for row in candidates) / len(candidates)
        selected_gain = float(selected["actual_weighted_gain"]) if selected is not None else 0.0
        selected_effect_values = (
            [float(value) for value in selected["per_seed_actual_weighted_gain"]]
            if selected is not None
            else [0.0 for _seed in seeds]
        )
        selected_effect_mean, selected_effect_standard_error = _mean_standard_error(
            selected_effect_values
        )
        selected_effect_lcb = selected_effect_mean - confidence_z * selected_effect_standard_error
        selection_regret = float(oracle["actual_weighted_gain"]) - selected_gain
        uniform_regret = float(oracle["actual_weighted_gain"]) - uniform_gain
        context_reports.append(
            {
                "context": dict(context),
                "candidate_count": len(candidates),
                "selected_candidate": selected,
                "selector_action": "execute" if selected is not None else "abstain",
                "oracle_candidate": oracle,
                "uniform_random_expected_gain": uniform_gain,
                "selection_regret": selection_regret,
                "uniform_random_expected_regret": uniform_regret,
                "regret_reduction_vs_uniform": uniform_regret - selection_regret,
                "top1_positive_hit": selected is not None and selected_gain > 0.0,
                "harmful_execution": selected is not None and selected_gain < 0.0,
                "selected_effect_standard_error": selected_effect_standard_error,
                "selected_effect_lcb": selected_effect_lcb,
                "selected_effect_lcb_positive": selected is not None and selected_effect_lcb > 0.0,
                "ranked_candidates": ranked,
            }
        )
    if not context_reports:
        raise DirectionalSelectorCanaryError("DIRECTIONAL_SELECTOR_NO_SUPPORTED_CONTEXTS")

    count = len(context_reports)
    mean_regret = sum(float(row["selection_regret"]) for row in context_reports) / count
    mean_uniform_regret = sum(float(row["uniform_random_expected_regret"]) for row in context_reports) / count
    regret_reduction = mean_uniform_regret - mean_regret
    positive_rate = sum(bool(row["top1_positive_hit"]) for row in context_reports) / count
    harmful_rate = sum(bool(row["harmful_execution"]) for row in context_reports) / count
    positive_effect_lcb_rate = sum(
        bool(row["selected_effect_lcb_positive"]) for row in context_reports
    ) / count
    independent_pass = (
        independent_contexts
        and regret_reduction > 0.0
        and harmful_rate == 0.0
        and positive_effect_lcb_rate == 1.0
    )
    report = {
        "schema_version": 1,
        "artifact_type": "verdiwm-ctrl-world-directional-selector-canary",
        "state": "ready",
        "fingerprint_campaign_id": atlas.get("campaign_id"),
        "fingerprint_atlas": {"path": str(atlas_path), "sha256": hashlib.sha256(atlas_path.read_bytes()).hexdigest()},
        "candidate_radius": candidate_radius,
        "candidate_receipts": candidate_receipts,
        "frozen_selector_receipt": frozen_selector_receipt,
        "context_count": count,
        "metrics": {
            "top1_positive_hit_rate": positive_rate,
            "harmful_execution_rate": harmful_rate,
            "positive_selected_effect_lcb_rate": positive_effect_lcb_rate,
            "mean_selection_regret": mean_regret,
            "mean_uniform_random_expected_regret": mean_uniform_regret,
            "regret_reduction_vs_uniform": regret_reduction,
        },
        "decision": (
            "passed_independent_accept"
            if independent_pass
            else "promising_dev_only"
            if not independent_contexts and regret_reduction > 0.0 and positive_rate > 0.5
            else "failed_or_inconclusive"
        ),
        "contexts": context_reports,
        "routing_readiness": {
            "state": "directional_selector_validated" if independent_pass else "not_licensed",
            "reason": (
                "Independent context and preregistered LCB abstention passed for base-direction dose selection; retrieved repair methods remain untested."
                if independent_pass
                else "The same development contexts were used for fingerprint calibration; an independently frozen context is still required."
                if not independent_contexts
                else "Independent accept did not demonstrate positive regret reduction, zero harmful execution, and a positive selected-effect lower confidence bound."
            ),
        },
        "selector_policy": {
            "confidence_z": confidence_z,
            "minimum_predicted_gain_lcb": minimum_gain_lcb,
            "abstain_when_no_candidate_passes": True,
            "independent_contexts": independent_contexts,
        },
        "claim_boundary": (
            "This is a dose-extrapolation selector canary for base intervention directions. It is not a "
            "retrieved-method comparison, model improvement, held-out selector certificate, or RSI evidence."
        ),
    }
    output_root.mkdir(mode=0o700, parents=True)
    report_path = output_root / "directional-selector-canary.json"
    report_path.write_text(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "artifact_type": "verdiwm-ctrl-world-directional-selector-canary-manifest",
        "state": "ready",
        "decision": report["decision"],
        "context_count": count,
        "regret_reduction_vs_uniform": regret_reduction,
        "report_path": str(report_path.resolve()),
        "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fingerprint-atlas", type=Path, required=True)
    parser.add_argument("--candidate-result", type=Path, action="append", required=True)
    parser.add_argument("--confidence-z", type=float, default=0.0)
    parser.add_argument("--minimum-gain-lcb", type=float, default=-1e300)
    parser.add_argument("--independent-contexts", action="store_true")
    parser.add_argument("--frozen-selector", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = evaluate(
        fingerprint_atlas_path=args.fingerprint_atlas,
        candidate_result_paths=args.candidate_result,
        output_root=args.output_root.resolve(),
        confidence_z=args.confidence_z,
        minimum_gain_lcb=args.minimum_gain_lcb,
        independent_contexts=bool(args.independent_contexts),
        frozen_selector_path=args.frozen_selector,
    )
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
