#!/usr/bin/env python3
"""Consolidate horizon-effect profiles into scoped transfer experience."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


class AcwmHorizonExperienceMapError(RuntimeError):
    """Horizon experience evidence was invalid or unsafe to merge."""


_POSITIVE_SCOPES = {
    "aggregate_long_horizon_positive",
    "short_horizon_only_positive",
    "hard_case_only_positive",
}


def build_horizon_experience_map(
    *, profile_paths: Sequence[Path], output_root: Path
) -> dict[str, object]:
    """Build routing, anti-condition, and causal-credit views."""

    if not profile_paths:
        raise AcwmHorizonExperienceMapError("HORIZON_EXPERIENCE_PROFILES_EMPTY")
    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise AcwmHorizonExperienceMapError("HORIZON_EXPERIENCE_OUTPUT_EXISTS")
    profiles = [_load_profile(Path(path).resolve(strict=True)) for path in profile_paths]
    edges = [_edge(profile) for profile in profiles]
    routing_priors = [edge for edge in edges if edge["effect_scope"] in _POSITIVE_SCOPES]
    anti_conditions = [edge for edge in edges if edge["effect_scope"] != "aggregate_long_horizon_positive"]
    causal_edges = [edge for edge in edges if edge["causal_credit_eligible"] is True]
    report: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "wmloop-acwm-horizon-experience-map",
        "state": "ready",
        "summary": {
            "profile_count": len(edges),
            "routing_prior_count": len(routing_priors),
            "anti_condition_count": len(anti_conditions),
            "causal_edge_count": len(causal_edges),
            "environment_count": len({str(edge["environment"]) for edge in edges}),
            "primitive_count": len({str(edge["primitive"]) for edge in edges}),
        },
        "observational_edges": edges,
        "routing_priors": routing_priors,
        "anti_conditions": anti_conditions,
        "causal_edges": causal_edges,
        "admission_policy": {
            "observational_edges_can_rank_proposals": True,
            "hard_case_only_can_support_general_uplift": False,
            "causal_edge_requires_profile_causal_credit": True,
            "required_causal_evidence": "factorized heldout replication plus settled verifier verdict",
        },
        "claim_boundary": (
            "Routing priors and anti-conditions are scoped observational experience. Only profiles carrying "
            "verified causal credit may enter causal_edges; ACWM-Phys metric values never transfer as a new "
            "backbone verdict."
        ),
    }
    return _write_bundle(destination, report)


def discover_profiles(report_root: Path) -> list[Path]:
    root = Path(report_root).resolve(strict=True)
    return sorted(root.glob("acwm-horizon-effect-profile-*/horizon-effect-profile.json"))


def _load_profile(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AcwmHorizonExperienceMapError("HORIZON_EXPERIENCE_PROFILE_INVALID") from exc
    if (
        not isinstance(payload, Mapping)
        or payload.get("artifact_type") != "wmloop-acwm-horizon-effect-profile"
        or payload.get("state") != "ready"
    ):
        raise AcwmHorizonExperienceMapError("HORIZON_EXPERIENCE_PROFILE_INVALID")
    return payload


def _edge(profile: Mapping[str, Any]) -> dict[str, object]:
    classification = profile.get("effect_classification")
    transfer = profile.get("transfer_prior")
    source = profile.get("source")
    per_frame = profile.get("per_frame_effect")
    horizon_effects = profile.get("horizon_effects")
    if (
        not isinstance(classification, Mapping)
        or not isinstance(transfer, Mapping)
        or not isinstance(source, Mapping)
        or not isinstance(per_frame, Mapping)
        or not isinstance(horizon_effects, list)
    ):
        raise AcwmHorizonExperienceMapError("HORIZON_EXPERIENCE_PROFILE_FIELDS_INVALID")
    scope = str(classification.get("effect_scope") or "")
    if not scope:
        raise AcwmHorizonExperienceMapError("HORIZON_EXPERIENCE_SCOPE_INVALID")
    horizon_psnr_delta = _horizon_psnr_delta(horizon_effects)
    early_delta = _finite_float(per_frame.get("early_half_mean_delta_psnr"), "early_half_mean_delta_psnr")
    late_delta = _finite_float(per_frame.get("late_half_mean_delta_psnr"), "late_half_mean_delta_psnr")
    tail_delta = _finite_float(per_frame.get("tail_frame_delta_psnr"), "tail_frame_delta_psnr")
    return {
        "environment": str(profile.get("environment") or ""),
        "primitive": str(profile.get("primitive") or ""),
        "training_seed": profile.get("training_seed"),
        "failure_signatures": list(transfer.get("failure_signatures") or []),
        "mechanism_family": transfer.get("mechanism_family"),
        "layer": transfer.get("layer"),
        "effect_scope": scope,
        "response_shape": _response_shape(
            scope=scope,
            early_delta_psnr=early_delta,
            late_delta_psnr=late_delta,
            tail_delta_psnr=tail_delta,
        ),
        "effective_horizons": list(classification.get("aggregate_passing_horizons") or []),
        "first_aggregate_failure_horizon": classification.get("first_aggregate_failure_horizon"),
        "horizon_psnr_delta": horizon_psnr_delta,
        "max_horizon": max(int(value) for value in profile.get("horizons", [])),
        "max_horizon_seconds": max(float(value) for value in profile.get("horizon_seconds", {}).values()),
        "positive_trajectory_rate_at_max_horizon": float(
            classification.get("positive_trajectory_rate_at_max_horizon", 0.0)
        ),
        "early_half_mean_delta_psnr": early_delta,
        "late_half_mean_delta_psnr": late_delta,
        "tail_frame_delta_psnr": tail_delta,
        "transfer_preconditions": transfer.get("transfer_preconditions"),
        "anti_conditions": transfer.get("anti_conditions"),
        "causal_credit_eligible": transfer.get("causal_credit_eligible") is True,
        "causal_credit_blocker": transfer.get("causal_credit_blocker"),
        "candidate_checkpoint": source.get("candidate_checkpoint"),
        "candidate_checkpoint_step": source.get("candidate_checkpoint_step"),
        "checkpoint_ladder": transfer.get("checkpoint_ladder"),
        "effective_training_window": transfer.get("effective_training_window"),
    }


def _horizon_psnr_delta(rows: Sequence[object]) -> dict[str, float]:
    result: dict[str, float] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise AcwmHorizonExperienceMapError("HORIZON_EXPERIENCE_HORIZON_EFFECT_INVALID")
        delta = row.get("delta_candidate_minus_baseline")
        if not isinstance(delta, Mapping):
            raise AcwmHorizonExperienceMapError("HORIZON_EXPERIENCE_HORIZON_EFFECT_INVALID")
        try:
            horizon = str(int(row["horizon"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise AcwmHorizonExperienceMapError("HORIZON_EXPERIENCE_HORIZON_EFFECT_INVALID") from exc
        result[horizon] = _finite_float(delta.get("psnr"), f"horizon_{horizon}_psnr")
    if not result:
        raise AcwmHorizonExperienceMapError("HORIZON_EXPERIENCE_HORIZON_EFFECT_INVALID")
    return result


def _finite_float(value: object, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise AcwmHorizonExperienceMapError(f"HORIZON_EXPERIENCE_VALUE_INVALID:{field}") from exc
    if not math.isfinite(result):
        raise AcwmHorizonExperienceMapError(f"HORIZON_EXPERIENCE_VALUE_INVALID:{field}")
    return result


def _response_shape(
    *, scope: str, early_delta_psnr: float, late_delta_psnr: float, tail_delta_psnr: float
) -> str:
    if scope == "aggregate_long_horizon_positive":
        return "sustained_long_horizon_gain"
    if scope == "short_horizon_only_positive":
        return "short_horizon_gain_then_failure"
    if scope == "hard_case_only_positive":
        return "heterogeneous_hard_case_repair"
    if early_delta_psnr > 0.0 and (late_delta_psnr <= 0.0 or tail_delta_psnr <= 0.0):
        return "early_gain_then_temporal_regression"
    if late_delta_psnr < early_delta_psnr and tail_delta_psnr < 0.0:
        return "degrading_with_horizon"
    return "nonpositive_or_mixed"


def _write_bundle(destination: Path, report: Mapping[str, object]) -> dict[str, object]:
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    try:
        temporary.mkdir(mode=0o700, parents=True)
        _write_json(temporary / "horizon-experience-map.json", report)
        _write_edges_csv(temporary / "horizon-experience-edges.csv", report["observational_edges"])
        (temporary / "horizon-experience-map.md").write_text(_markdown(report), encoding="utf-8")
        manifest = {
            "schema_version": 1,
            "artifact_type": "wmloop-acwm-horizon-experience-map-manifest",
            "state": "ready",
            **dict(report["summary"]),
            "report_path": str(destination / "horizon-experience-map.json"),
            "markdown_path": str(destination / "horizon-experience-map.md"),
            "csv_path": str(destination / "horizon-experience-edges.csv"),
        }
        _write_json(temporary / "manifest.json", manifest)
        os.replace(temporary, destination)
        return {**dict(report), "manifest": manifest}
    except Exception:
        if temporary.exists() and not temporary.is_symlink():
            shutil.rmtree(temporary)
        raise


def _write_edges_csv(path: Path, rows: object) -> None:
    if not isinstance(rows, list):
        raise AcwmHorizonExperienceMapError("HORIZON_EXPERIENCE_EDGES_INVALID")
    fields = (
        "environment",
        "primitive",
        "failure_signatures",
        "mechanism_family",
        "layer",
        "effect_scope",
        "response_shape",
        "effective_horizons",
        "first_aggregate_failure_horizon",
        "horizon_psnr_delta",
        "max_horizon",
        "max_horizon_seconds",
        "positive_trajectory_rate_at_max_horizon",
        "early_half_mean_delta_psnr",
        "late_half_mean_delta_psnr",
        "tail_frame_delta_psnr",
        "causal_credit_eligible",
        "causal_credit_blocker",
        "best_checkpoint_step",
        "later_regression_observed",
        "first_later_regression_step",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            values = {field: row.get(field) for field in fields}
            values["failure_signatures"] = ";".join(row["failure_signatures"])
            values["effective_horizons"] = ";".join(str(value) for value in row["effective_horizons"])
            values["horizon_psnr_delta"] = json.dumps(
                row["horizon_psnr_delta"], sort_keys=True, separators=(",", ":")
            )
            training_window = row.get("effective_training_window")
            if isinstance(training_window, Mapping):
                values["best_checkpoint_step"] = training_window.get("selected_step")
                values["later_regression_observed"] = training_window.get(
                    "later_regression_observed"
                )
                values["first_later_regression_step"] = training_window.get(
                    "first_later_regression_step"
                )
            writer.writerow(values)


def _markdown(report: Mapping[str, object]) -> str:
    summary = report["summary"]
    lines = [
        "# ACWM Horizon Experience Map",
        "",
        f"- Profiles: `{summary['profile_count']}`",
        f"- Routing priors: `{summary['routing_prior_count']}`",
        f"- Anti-conditions: `{summary['anti_condition_count']}`",
        f"- Causal edges: `{summary['causal_edge_count']}`",
        "",
        "| Environment | Primitive | Failure signatures | Response shape | Effective horizons | Causal credit |",
        "|---|---|---|---|---|:---:|",
    ]
    for edge in report["observational_edges"]:
        lines.append(
            f"| {edge['environment']} | `{edge['primitive']}` | "
            f"{', '.join(edge['failure_signatures']) or 'unknown'} | `{edge['response_shape']}` | "
            f"{edge['effective_horizons']} | {edge['causal_credit_eligible']} |"
        )
    lines.extend(["", "## Claim Boundary", "", str(report["claim_boundary"]), ""])
    return "\n".join(lines)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", action="append", type=Path, default=[])
    parser.add_argument("--report-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    profiles = list(args.profile)
    if args.report_root is not None:
        profiles.extend(discover_profiles(args.report_root))
    profiles = list(dict.fromkeys(Path(path).resolve() for path in profiles))
    report = build_horizon_experience_map(profile_paths=profiles, output_root=args.output_root)
    print(json.dumps(report["manifest"], sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
