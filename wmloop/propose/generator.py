"""LLM proposal adapter constrained to the registered intervention language."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from wmloop.contracts import ContractValidationError, validate_document
from wmloop.primitives.registry import PrimitiveRegistry, PrimitiveValidationError


class LLMClient(Protocol):
    def complete(self, prompt: str) -> str:
        """Return exactly one model completion."""


class ProposalGenerationError(RuntimeError):
    """The constrained generator could not produce an admissible proposal."""


CURRENT_LIBRARY_VERSION = "v1.2"


@dataclass(frozen=True)
class ProposalContext:
    failure_report: Mapping[str, Any]
    goal_spec: Mapping[str, Any]
    archive_statistics: Mapping[str, Any]
    registry: PrimitiveRegistry
    library_version: str = CURRENT_LIBRARY_VERSION


@dataclass(frozen=True)
class GeneratedProposal:
    proposal: Mapping[str, Any]
    prompt: str
    raw_responses: tuple[str, ...]
    attempts: int


class ProposalGenerator:
    def __init__(self, client: LLMClient, *, max_attempts: int = 3) -> None:
        if max_attempts < 1:
            raise ValueError("PROPOSAL_MAX_ATTEMPTS_INVALID")
        self._client = client
        self._max_attempts = max_attempts

    def generate(self, context: ProposalContext) -> GeneratedProposal:
        prompt = _build_prompt(context)
        responses: list[str] = []
        errors: list[str] = []
        for _ in range(self._max_attempts):
            response = self._client.complete(prompt)
            responses.append(response)
            try:
                proposal = json.loads(response)
                if not isinstance(proposal, Mapping):
                    raise ProposalGenerationError("PROPOSAL_JSON_OBJECT_REQUIRED")
                _validate_proposal(proposal, context)
            except (json.JSONDecodeError, ContractValidationError, PrimitiveValidationError, ProposalGenerationError, TypeError, ValueError) as exc:
                errors.append(type(exc).__name__)
                continue
            return GeneratedProposal(
                proposal=dict(proposal),
                prompt=prompt,
                raw_responses=tuple(responses),
                attempts=len(responses),
            )
        raise ProposalGenerationError(f"PROPOSAL_RETRY_EXHAUSTED:{','.join(errors) or 'empty'}")


def validate_proposal_for_context(proposal: Mapping[str, Any], context: ProposalContext) -> None:
    """Validate a supplied proposal against the same bounded language as LLM output."""

    _validate_proposal(proposal, context)


def _build_prompt(context: ProposalContext) -> str:
    routing_failures = _routing_failures(context.failure_report, context.registry)
    allowed_manifests = _allowed_manifests(context.failure_report, context.registry)
    allowed = [
        {
            "name": manifest.name,
            "layer": manifest.layer,
            "params_schema": manifest.params_schema,
            "cost_class": manifest.cost_class,
            "targets_failures": manifest.targets_failures,
            "estimated_gpu_hours": manifest.estimated_gpu_hours,
            "hooks": manifest.hooks,
        }
        for manifest in allowed_manifests
    ]
    packet = {
        "instruction": "Return one JSON proposal only. Choose only listed interventions. Do not write source code or commands.",
        "library_version": context.library_version,
        "failure_report": context.failure_report,
        "failure_routing": {
            "dominant_failure": context.failure_report.get("dominant_failure"),
            "routing_failures": routing_failures,
            "provenance": context.failure_report.get("proposal_routing_provenance", []),
        },
        "goal_spec": context.goal_spec,
        "archive_statistics": context.archive_statistics,
        "allowed_primitives": allowed,
    }
    return json.dumps(packet, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _validate_proposal(proposal: Mapping[str, Any], context: ProposalContext) -> None:
    validate_document("proposal", proposal)
    failure = context.failure_report
    if proposal["env"] != failure.get("env") or proposal["round"] != failure.get("round"):
        raise ProposalGenerationError("PROPOSAL_CONTEXT_MISMATCH")
    goal_id = context.goal_spec.get("goal_id")
    if (
        not isinstance(goal_id, str)
        or proposal["goal_id"] != goal_id
        or failure.get("goal_id") != goal_id
    ):
        raise ProposalGenerationError("PROPOSAL_GOAL_MISMATCH")
    if proposal["library_version"] != context.library_version:
        raise ProposalGenerationError("PROPOSAL_LIBRARY_VERSION_MISMATCH")
    if proposal["based_on_failure"] != failure.get("dominant_failure"):
        raise ProposalGenerationError("PROPOSAL_DIAGNOSIS_MISMATCH")
    intervention_rows = proposal["interventions"]
    selections = context.registry.validate_combination(
        [(str(row["primitive"]), row["params"]) for row in intervention_rows]
    )
    allowed_names = {manifest.name for manifest in _allowed_manifests(failure, context.registry)}
    for selection in selections:
        if selection.name not in allowed_names:
            raise ProposalGenerationError(f"PROPOSAL_PRIMITIVE_NOT_ROUTED:{selection.name}")
    if any(row["layer"] != selection.layer for row, selection in zip(intervention_rows, selections)):
        raise ProposalGenerationError("PROPOSAL_LAYER_MISMATCH")
    estimate = float(proposal["budget_estimate_gpu_hours"])
    if not math.isfinite(estimate) or estimate < sum(selection.estimated_gpu_hours for selection in selections):
        raise ProposalGenerationError("PROPOSAL_BUDGET_UNDERESTIMATED")


def _allowed_manifests(failure_report: Mapping[str, Any], registry: PrimitiveRegistry) -> tuple[Any, ...]:
    manifests = []
    seen: set[str] = set()
    for failure in _routing_failures(failure_report, registry):
        for manifest in registry.available_for(failure=failure):
            if manifest.name not in seen:
                manifests.append(manifest)
                seen.add(manifest.name)
    return tuple(manifests)


def _routing_failures(failure_report: Mapping[str, Any], registry: PrimitiveRegistry) -> tuple[str, ...]:
    dominant = str(failure_report.get("dominant_failure") or "")
    ordered: list[str] = []
    if dominant != "mixed":
        ordered.append(dominant)
    candidates = failure_report.get("dominant_failure_candidates")
    if isinstance(candidates, list):
        ordered.extend(str(candidate) for candidate in candidates if isinstance(candidate, str) and candidate != "mixed")
    routed = failure_report.get("routed_failure_families")
    if isinstance(routed, list):
        ordered.extend(str(failure) for failure in routed if isinstance(failure, str) and failure != "mixed")
    if not ordered and dominant == "mixed":
        ordered.extend(_registered_failure_types(registry))
    return tuple(dict.fromkeys(ordered))


def _registered_failure_types(registry: PrimitiveRegistry) -> tuple[str, ...]:
    failures = {
        failure
        for name in registry.names()
        for failure in registry.manifest(name).targets_failures
        if failure != "mixed"
    }
    return tuple(sorted(failures))
