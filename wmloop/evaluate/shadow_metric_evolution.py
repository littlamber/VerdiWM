"""Shadow-only metric evolution records.

Metric candidates can be proposed by the autonomous loop, but this module
deliberately refuses to produce an active evaluator or promotion binding. A
separately frozen constitution transition remains the only path to adoption.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from wmloop.contracts import ContractValidationError, validate_document


class ShadowMetricEvolutionError(ValueError):
    """A metric candidate crossed the shadow-only authority boundary."""


def compile_shadow_metric_proposals(
    *,
    candidates: Sequence[Mapping[str, object]],
    protected_metric_ids: Sequence[str],
    output_root: Path,
    root: Path,
) -> dict[str, object]:
    """Persist deterministic shadow proposals and never mutate active metrics."""

    destination = Path(output_root).expanduser().resolve()
    if destination == root or root in destination.parents:
        raise ShadowMetricEvolutionError("SHADOW_METRIC_OUTPUT_INSIDE_SOURCE")
    protected = {str(value) for value in protected_metric_ids}
    rows: list[dict[str, object]] = []
    for candidate in sorted(candidates, key=lambda value: str(value.get("metric_id", ""))):
        _validate_metric(candidate, root=root)
        metric_id = str(candidate["metric_id"])
        if metric_id in protected or candidate.get("role") in {"protected", "primary", "guard"}:
            raise ShadowMetricEvolutionError("SHADOW_METRIC_PROTECTED_OR_ACTIVE_FORBIDDEN:" + metric_id)
        body = {
            "schema_version": 1,
            "artifact_type": "wmloop-shadow-metric-proposal",
            "proposal_id": "shadow-metric-" + hashlib.sha256(_canonical(candidate).encode()).hexdigest()[:24],
            "metric_id": metric_id,
            "metric_contract": dict(candidate),
            "metric_contract_digest": hashlib.sha256(_canonical(candidate).encode()).hexdigest(),
            "state": "shadow",
            "active_metric_mutation": False,
            "active_evaluator_mutation": False,
            "verdict_authority": False,
            "promotion_authority": False,
            "required_next_evidence": ["static_check", "shadow_evaluation", "historical_calibration", "fresh_heldout", "canary"],
            "claim_boundary": "This candidate is diagnostic shadow evidence only. A separately frozen constitution transition is required before future work may use it.",
        }
        body["proposal_digest"] = hashlib.sha256(_canonical(body).encode()).hexdigest()
        rows.append(body)
    destination.mkdir(mode=0o700, parents=True, exist_ok=True)
    paths: list[str] = []
    for row in rows:
        path = destination / f"{row['proposal_id']}.json"
        path.write_text(json.dumps(row, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        paths.append(str(path))
    summary = {
        "schema_version": 1,
        "artifact_type": "wmloop-shadow-metric-evolution-summary",
        "state": "shadow_only",
        "proposal_count": len(rows),
        "proposal_paths": paths,
        "protected_metric_ids": sorted(protected),
        "active_metric_mutation": False,
        "active_evaluator_mutation": False,
        "verdict_authority": False,
        "promotion_authority": False,
        "claim_boundary": "Shadow metric proposals cannot alter the frozen evaluator or promotion gate.",
    }
    summary_path = destination / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


def _validate_metric(candidate: Mapping[str, object], *, root: Path) -> None:
    try:
        validate_document("metric_adequacy", candidate, root=root)
    except ContractValidationError as exc:
        raise ShadowMetricEvolutionError(f"SHADOW_METRIC_CONTRACT_INVALID:{exc}") from exc


def _canonical(payload: Mapping[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
