"""Reject a successor probe whose collision-smoke geometry duplicates its parent."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from wmloop.experiments._artifacts import canonical_json, write_bundle


class ProbeSmokeRedundancyError(ValueError):
    """Probe-smoke response charts are malformed or incomparable."""


def evaluate_probe_smoke_redundancy(
    *,
    reference_campaign_root: Path,
    candidate_campaign_root: Path,
    environments: Sequence[str],
    output_root: Path,
    minimum_cosine_similarity: float = 0.999,
    maximum_relative_l2: float = 0.1,
    maximum_locality_residual: float = 0.5,
    archive_db: Path | None = None,
    cas_root: Path | None = None,
) -> dict[str, object]:
    if not environments or len(set(environments)) != len(environments):
        raise ProbeSmokeRedundancyError("PROBE_REDUNDANCY_ENVIRONMENTS_INVALID")
    if not -1.0 <= minimum_cosine_similarity <= 1.0:
        raise ProbeSmokeRedundancyError("PROBE_REDUNDANCY_COSINE_THRESHOLD_INVALID")
    if maximum_relative_l2 < 0.0 or maximum_locality_residual < 0.0:
        raise ProbeSmokeRedundancyError("PROBE_REDUNDANCY_THRESHOLD_INVALID")
    reference_root = Path(reference_campaign_root).resolve()
    candidate_root = Path(candidate_campaign_root).resolve()
    comparisons: list[dict[str, object]] = []
    reference_probe_ids: set[str] = set()
    candidate_probe_ids: set[str] = set()
    for environment in environments:
        reference = _load_chart(reference_root, environment)
        candidate = _load_chart(candidate_root, environment)
        if reference.get("outcome_names") != candidate.get("outcome_names"):
            raise ProbeSmokeRedundancyError(f"PROBE_REDUNDANCY_OUTCOMES_DIFFER:{environment}")
        reference_probe = _single_probe(reference, environment)
        candidate_probe = _single_probe(candidate, environment)
        reference_probe_ids.add(reference_probe)
        candidate_probe_ids.add(candidate_probe)
        left = _vector(reference.get("response_coordinate"), environment)
        right = _vector(candidate.get("response_coordinate"), environment)
        if len(left) != len(right):
            raise ProbeSmokeRedundancyError(f"PROBE_REDUNDANCY_DIMENSION_MISMATCH:{environment}")
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm <= 1e-12 or right_norm <= 1e-12:
            raise ProbeSmokeRedundancyError(f"PROBE_REDUNDANCY_ZERO_RESPONSE:{environment}")
        cosine = sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)
        relative_l2 = math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right, strict=True))) / left_norm
        locality = float(candidate["locality_residuals"][candidate_probe])
        comparisons.append(
            {
                "environment": environment,
                "cosine_similarity": cosine,
                "relative_l2": relative_l2,
                "candidate_locality_residual": locality,
                "candidate_locality_pass": locality <= maximum_locality_residual,
                "redundant": (
                    cosine >= minimum_cosine_similarity
                    and relative_l2 <= maximum_relative_l2
                    and locality <= maximum_locality_residual
                ),
            }
        )
    if len(reference_probe_ids) != 1 or len(candidate_probe_ids) != 1:
        raise ProbeSmokeRedundancyError("PROBE_REDUNDANCY_PROBE_ID_INCONSISTENT")
    reference_probe_id = next(iter(reference_probe_ids))
    candidate_probe_id = next(iter(candidate_probe_ids))
    if reference_probe_id == candidate_probe_id:
        raise ProbeSmokeRedundancyError("PROBE_REDUNDANCY_SUCCESSOR_NOT_NOVEL")
    redundant = all(bool(row["redundant"]) for row in comparisons)
    report = {
        "schema_version": 1,
        "artifact_type": "verdiwm-probe-smoke-redundancy-gate",
        "state": "ready",
        "decision": "reject_as_redundant" if redundant else "expand_collision_evidence",
        "expand_to_eight_environment_pilot": not redundant,
        "reference_probe_id": reference_probe_id,
        "candidate_probe_id": candidate_probe_id,
        "thresholds": {
            "minimum_cosine_similarity": minimum_cosine_similarity,
            "maximum_relative_l2": maximum_relative_l2,
            "maximum_locality_residual": maximum_locality_residual,
        },
        "comparisons": comparisons,
        "claim_boundary": "This gate only controls diagnostic probe expansion cost. It does not establish primitive quality, transfer, or causal equivalence outside the measured collision environments.",
    }
    destination = Path(output_root).resolve()
    return write_bundle(
        output_root=destination,
        files={
            "probe-smoke-redundancy.json": canonical_json(report),
            "probe-smoke-redundancy.md": _markdown(report).encode("utf-8"),
        },
        manifest_fields={
            "artifact_type": "verdiwm-probe-smoke-redundancy-gate-manifest",
            "state": "ready",
            "decision": report["decision"],
            "expand_to_eight_environment_pilot": not redundant,
            "environment_count": len(comparisons),
            "report_path": str(destination / "probe-smoke-redundancy.json"),
        },
        archive_db=archive_db,
        cas_root=cas_root,
    )


def _load_chart(root: Path, environment: str) -> Mapping[str, Any]:
    path = root / "environments" / environment / "response-chart.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProbeSmokeRedundancyError(f"PROBE_REDUNDANCY_CHART_INVALID:{path}") from exc
    if not isinstance(payload, Mapping):
        raise ProbeSmokeRedundancyError(f"PROBE_REDUNDANCY_CHART_INVALID:{path}")
    return payload


def _single_probe(chart: Mapping[str, Any], environment: str) -> str:
    names = chart.get("intervention_names")
    if not isinstance(names, list) or len(names) != 1 or not isinstance(names[0], str):
        raise ProbeSmokeRedundancyError(f"PROBE_REDUNDANCY_PROBE_INVALID:{environment}")
    residuals = chart.get("locality_residuals")
    if not isinstance(residuals, Mapping) or names[0] not in residuals:
        raise ProbeSmokeRedundancyError(f"PROBE_REDUNDANCY_LOCALITY_INVALID:{environment}")
    return names[0]


def _vector(value: object, environment: str) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise ProbeSmokeRedundancyError(f"PROBE_REDUNDANCY_RESPONSE_INVALID:{environment}")
    result = tuple(float(item) for item in value)
    if any(not math.isfinite(item) for item in result):
        raise ProbeSmokeRedundancyError(f"PROBE_REDUNDANCY_RESPONSE_INVALID:{environment}")
    return result


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Probe Smoke Redundancy Gate",
        "",
        f"Decision: `{report['decision']}`",
        "",
        "| Environment | Cosine | Relative L2 | Locality | Redundant |",
        "|---|---:|---:|---:|:---|",
    ]
    for row in report["comparisons"]:
        lines.append(
            f"| {row['environment']} | {row['cosine_similarity']:.9f} | "
            f"{row['relative_l2']:.6f} | {row['candidate_locality_residual']:.6f} | "
            f"{row['redundant']} |"
        )
    lines.extend(["", str(report["claim_boundary"]), ""])
    return "\n".join(lines)
