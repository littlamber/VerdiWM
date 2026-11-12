"""Training and frozen evaluation smoke components: routing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from wmloop.execute.training_smoke.errors import OrchestratorTrainingEvalSmokeError
from wmloop.execute.training_smoke.io import _load_json_object
from wmloop.primitives.registry import PrimitiveRegistry


def _augment_failure_with_signature_bank_routes(
    *,
    root: Path,
    failure: Mapping[str, Any],
    requested_primitive: str,
    proposal_routing_plan: Path | None = None,
) -> dict[str, Any]:
    augmented = dict(failure)
    signature_routed = _signature_bank_routed_failure_families(
        root=root,
        environment=str(failure.get("env") or ""),
        primitive=requested_primitive,
    )
    transfer_routed = _positive_transfer_routed_failure_families(
        root=root,
        environment=str(failure.get("env") or ""),
        primitive=requested_primitive,
    )
    targeted_routed, targeted_provenance = _targeted_plan_routed_failure_families(
        root=root,
        plan_path=proposal_routing_plan,
        environment=str(failure.get("env") or ""),
        primitive=requested_primitive,
    )
    routed = tuple(
        dict.fromkeys((*signature_routed, *transfer_routed, *targeted_routed))
    )
    if routed:
        existing = augmented.get("routed_failure_families")
        merged: list[str] = []
        if isinstance(existing, list):
            merged.extend(str(item) for item in existing if isinstance(item, str))
        merged.extend(routed)
        augmented["routed_failure_families"] = list(
            dict.fromkeys(item for item in merged if item and item != "mixed")
        )
        provenance = augmented.get("proposal_routing_provenance")
        provenance_rows = (
            [dict(item) for item in provenance if isinstance(item, Mapping)]
            if isinstance(provenance, list)
            else []
        )
        if signature_routed:
            augmented["signature_bank_routed_primitive"] = requested_primitive
            provenance_rows.append(
                {
                    "source": "failure_signature_bank",
                    "primitive": requested_primitive,
                    "routed_failure_families": list(signature_routed),
                }
            )
        if transfer_routed:
            augmented["positive_transfer_routed_primitive"] = requested_primitive
            provenance_rows.append(
                {
                    "source": "positive_transfer_exemplar",
                    "primitive": requested_primitive,
                    "routed_failure_families": list(transfer_routed),
                }
            )
        if targeted_routed:
            augmented["targeted_gap_plan_routed_primitive"] = requested_primitive
            provenance_rows.append(targeted_provenance)
        augmented["proposal_routing_provenance"] = provenance_rows
    return augmented


def _targeted_plan_routed_failure_families(
    *,
    root: Path,
    plan_path: Path | None,
    environment: str,
    primitive: str,
) -> tuple[tuple[str, ...], dict[str, Any]]:
    """Compile an explicit targeted exploration route without rewriting diagnosis."""

    if plan_path is None:
        return (), {}
    path = Path(plan_path).resolve(strict=True)
    payload = _load_json_object(path)
    if (
        payload.get("artifact_type") != "wmloop-acwm-targeted-gap-plan"
        or payload.get("state") != "ready"
    ):
        raise OrchestratorTrainingEvalSmokeError(
            "M3_TRAINING_EVAL_SMOKE_ROUTING_PLAN_INVALID"
        )
    claim_boundary = payload.get("claim_boundary")
    records = payload.get("environment_records")
    if (
        not isinstance(claim_boundary, str)
        or not claim_boundary
        or not isinstance(records, list)
    ):
        raise OrchestratorTrainingEvalSmokeError(
            "M3_TRAINING_EVAL_SMOKE_ROUTING_PLAN_INVALID"
        )
    matches = [
        record
        for record in records
        if isinstance(record, Mapping)
        and record.get("environment") == environment
        and primitive in (record.get("recommended_existing_primitives") or [])
    ]
    if len(matches) != 1:
        raise OrchestratorTrainingEvalSmokeError(
            f"M3_TRAINING_EVAL_SMOKE_ROUTING_PLAN_SCOPE_MISMATCH:{environment}:{primitive}"
        )
    record = matches[0]
    hypothesis = record.get("mechanism_hypothesis")
    routed = record.get("routed_failure_families")
    probes = record.get("diagnostic_probe_candidates")
    if (
        not isinstance(hypothesis, str)
        or not hypothesis
        or not isinstance(routed, list)
        or not routed
        or not all(
            isinstance(item, str) and item and item != "mixed" for item in routed
        )
        or not isinstance(probes, list)
        or not probes
        or not all(isinstance(item, str) and item for item in probes)
    ):
        raise OrchestratorTrainingEvalSmokeError(
            "M3_TRAINING_EVAL_SMOKE_ROUTING_PLAN_INVALID"
        )
    try:
        targets = set(
            PrimitiveRegistry.from_root(root).manifest(primitive).targets_failures
        )
    except (OSError, ValueError, KeyError) as exc:
        raise OrchestratorTrainingEvalSmokeError(
            f"M3_TRAINING_EVAL_SMOKE_ROUTING_PLAN_PRIMITIVE_INVALID:{primitive}"
        ) from exc
    normalized = tuple(dict.fromkeys(str(item) for item in routed))
    if not set(normalized).issubset(targets):
        raise OrchestratorTrainingEvalSmokeError(
            f"M3_TRAINING_EVAL_SMOKE_ROUTING_PLAN_TARGET_MISMATCH:{primitive}"
        )
    return normalized, {
        "source": "targeted_gap_plan",
        "path": str(path),
        "primitive": primitive,
        "routed_failure_families": list(normalized),
        "diagnostic_probe_candidates": list(probes),
        "mechanism_hypothesis": hypothesis,
        "claim_boundary": claim_boundary,
    }


def _positive_transfer_routed_failure_families(
    *, root: Path, environment: str, primitive: str
) -> tuple[str, ...]:
    """Admit a cross-environment route only after a formal positive exemplar exists."""

    if not environment or not primitive:
        return ()
    reports = Path(root) / "results" / "reports"
    has_positive_source = False
    for path in reports.glob("acwm-autoloop-official-gate-*/manifest.json"):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        gate = (
            manifest.get("official_quality_gate")
            if isinstance(manifest, Mapping)
            else None
        )
        if (
            isinstance(manifest, Mapping)
            and manifest.get("state") == "ready"
            and manifest.get("primitive") == primitive
            and manifest.get("environment") != environment
            and isinstance(gate, Mapping)
            and gate.get("pass") is True
        ):
            has_positive_source = True
            break
    if not has_positive_source:
        return ()
    try:
        manifest = PrimitiveRegistry.from_root(root).manifest(primitive)
    except (OSError, ValueError, KeyError):
        return ()
    return tuple(failure for failure in manifest.targets_failures if failure != "mixed")


def _signature_bank_routed_failure_families(
    *, root: Path, environment: str, primitive: str
) -> tuple[str, ...]:
    if not environment or not primitive:
        return ()
    bank_path = (
        Path(root)
        / "results"
        / "reports"
        / "failure-signature-bank-r1"
        / "failure-signature-bank.json"
    )
    if not bank_path.is_file() or bank_path.is_symlink():
        return ()
    try:
        payload = json.loads(bank_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ()
    if (
        not isinstance(payload, Mapping)
        or payload.get("artifact_type") != "wmloop-failure-signature-bank"
    ):
        return ()
    rows = payload.get("primitive_routing")
    if not isinstance(rows, list):
        return ()
    routed: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        if row.get("environment") != environment or row.get("primitive") != primitive:
            continue
        decision = row.get("routing_decision")
        if decision not in {
            "stage_canary_after_diagnostic_probe",
            "retain_as_source_exemplar",
        }:
            continue
        failures = row.get("target_failures")
        if isinstance(failures, list):
            routed.extend(
                str(item)
                for item in failures
                if isinstance(item, str) and item != "mixed"
            )
    return tuple(dict.fromkeys(routed))
