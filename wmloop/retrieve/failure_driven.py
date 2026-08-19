"""Failure-driven retrieval plans for open trainable methods."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from wmloop.contracts import ContractValidationError, validate_document
from wmloop.geometry.evidence_ir import reject_runtime_bindings
from wmloop.geometry.types import GeometryValidationError


class FailureDrivenRetrievalError(ValueError):
    """A failure signal or retrieval plan was invalid."""


_INTENTS = {
    "missing_capability": (
        "methods that create or avoid the missing semantic capability",
        "target-model integration and hook requirements",
    ),
    "missing_data": (
        "methods trainable under the available data regime",
        "data conversion, supervision, and minimum dataset requirements",
    ),
    "training_instability": (
        "training objectives and schedules that stabilize the observed failure",
        "optimizer, loss, regularization, and ablation details",
    ),
    "rollout_regression": (
        "methods that address the observed long-horizon rollout regression",
        "rollout-aware training objectives and protected-metric ablations",
    ),
    "no_admissible_candidate": (
        "recent methods outside the current primitive and ABI vocabulary",
        "complete training recipe and target integration evidence",
    ),
}


def build_failure_retrieval_plan(
    *,
    portrait_binding: Mapping[str, object],
    probe_binding: Mapping[str, object],
    failures: Sequence[Mapping[str, object]],
    root: Path | None = None,
) -> dict[str, object]:
    normalized = [_failure(row) for row in failures]
    if not normalized:
        raise FailureDrivenRetrievalError("FAILURE_RETRIEVAL_FAILURES_REQUIRED")
    lanes = []
    for failure_class in sorted({str(row["failure_class"]) for row in normalized}):
        intents = _INTENTS.get(failure_class)
        if intents is None:
            raise FailureDrivenRetrievalError("FAILURE_RETRIEVAL_CLASS_INVALID")
        lane: dict[str, object] = {
            "failure_class": failure_class,
            "query_intents": list(intents),
            "required_method_evidence": [
                "mechanism description",
                "implementation or algorithm detail",
                "training objective and optimization recipe",
                "target-relevant evaluation and ablation",
            ],
            "excluded_source_roles": [
                "survey_only",
                "benchmark_only",
                "diagnostic_without_intervention",
                "inference_only_when_training_required",
            ],
        }
        lane["lane_id"] = "retrieval-lane-" + _digest(lane)[:24]
        lanes.append(lane)
    body: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "verdiwm-failure-retrieval-plan",
        "plan_id": "",
        "portrait_binding": dict(portrait_binding),
        "probe_binding": dict(probe_binding),
        "failures": normalized,
        "lanes": lanes,
        "source_requirements": {
            "training_recipe_required": any(
                row["failure_class"] in {"training_instability", "rollout_regression", "no_admissible_candidate"}
                for row in normalized
            ),
            "implementation_detail_required": True,
            "falsification_required": True,
            "source_digest_required": True,
        },
        "authority": {"ranking_only": True, "gpu_scheduling": False, "promotion": False},
        "claim_boundary": (
            "This plan changes retrieval and ranking only. Retrieved methods remain proposals "
            "until target-bound calibration and frozen verification."
        ),
    }
    body["plan_id"] = "failure-retrieval-" + _digest(body, excluded="plan_id")[:24]
    validate_failure_retrieval_plan(body, root=root)
    return body


def validate_failure_retrieval_plan(
    document: Mapping[str, object], *, root: Path | None = None
) -> None:
    try:
        reject_runtime_bindings(document)
        validate_document("failure_retrieval_plan", document, root=root)
    except (ContractValidationError, GeometryValidationError) as exc:
        raise FailureDrivenRetrievalError(f"FAILURE_RETRIEVAL_SCHEMA_INVALID:{exc}") from exc
    expected = "failure-retrieval-" + _digest(document, excluded="plan_id")[:24]
    if document.get("plan_id") != expected:
        raise FailureDrivenRetrievalError("FAILURE_RETRIEVAL_DIGEST_MISMATCH")


def _failure(row: Mapping[str, object]) -> dict[str, object]:
    failure_class = str(row.get("failure_class") or "")
    code = str(row.get("code") or "").strip()
    refs = row.get("evidence_refs")
    if failure_class not in _INTENTS or not code or not isinstance(refs, list):
        raise FailureDrivenRetrievalError("FAILURE_RETRIEVAL_FAILURE_INVALID")
    return {"failure_class": failure_class, "code": code, "evidence_refs": sorted(str(value) for value in refs)}


def _digest(value: object, *, excluded: str | None = None) -> str:
    payload = (
        {key: item for key, item in value.items() if key != excluded}
        if excluded is not None and isinstance(value, Mapping)
        else value
    )
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode()
    ).hexdigest()
