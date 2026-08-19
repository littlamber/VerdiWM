"""Fail-closed contracts for shadow constitutional evolution.

This module deliberately stops at proposal governance.  A proposal can make
new goals, verifiers, metrics, or actions explicit and can pass shadow gates,
but it cannot create verdict authority.  A separately frozen constitution is
still required before a candidate can affect scientific claims.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from wmloop.contracts import ContractValidationError, validate_document


class ConstitutionProposalError(ValueError):
    """A metric contract or constitutional proposal failed closed."""


_CONTENT_REF = re.compile(r"^(?:cas://|urn:|sha256:)[^\s]+$")
_STATES = {"candidate", "shadow", "probation", "approved", "rejected", "revoked"}
_TRANSITIONS = {
    "candidate": {"shadow", "rejected"},
    "shadow": {"probation", "rejected"},
    "probation": {"approved", "rejected"},
    "approved": {"revoked"},
    "rejected": set(),
    "revoked": set(),
}
_PROBATION_GATES = (
    "incremental_predictive_value",
    "anti_goodhart_passed",
    "regression_passed",
)
_APPROVAL_GATES = _PROBATION_GATES + (
    "independent_validation_passed",
    "canary_passed",
)
_AUTONOMOUS_EVIDENCE_FIELDS = (
    "static_check_refs",
    "shadow_evaluation_refs",
    "historical_calibration_refs",
    "fresh_heldout_refs",
    "canary_refs",
    "metric_adequacy_report_refs",
)


def validate_metric_adequacy(
    document: Mapping[str, object], *, root: Path | None = None
) -> None:
    """Validate one metric contract without granting it scientific authority."""

    try:
        validate_document("metric_adequacy", document, root=root)
    except ContractValidationError as exc:
        raise ConstitutionProposalError(f"METRIC_ADEQUACY_SCHEMA_INVALID:{exc}") from exc
    metric_id = str(document["metric_id"])
    guards = [str(value) for value in document.get("required_guard_metrics", [])]
    if metric_id in guards:
        raise ConstitutionProposalError("METRIC_ADEQUACY_SELF_GUARD_FORBIDDEN")
    if len(guards) != len(set(guards)):
        raise ConstitutionProposalError("METRIC_ADEQUACY_GUARD_DUPLICATE")
    horizons = [int(value) for value in document["horizons"]]
    if len(horizons) != len(set(horizons)):
        raise ConstitutionProposalError("METRIC_ADEQUACY_HORIZON_DUPLICATE")
    splits = [str(value) for value in document["splits"]]
    if len(splits) != len(set(splits)):
        raise ConstitutionProposalError("METRIC_ADEQUACY_SPLIT_DUPLICATE")
    if not _CONTENT_REF.fullmatch(str(document["evaluator_binding"])):
        raise ConstitutionProposalError("METRIC_ADEQUACY_EVALUATOR_REF_INVALID")


def validate_constitution_proposal(
    document: Mapping[str, object],
    *,
    parent_protected_metric_ids: Sequence[str] | None = None,
    root: Path | None = None,
) -> None:
    """Validate a candidate and its immutable authority boundary."""

    try:
        validate_document("constitution_proposal", document, root=root)
    except ContractValidationError as exc:
        raise ConstitutionProposalError(f"CONSTITUTION_PROPOSAL_SCHEMA_INVALID:{exc}") from exc

    expected_digest = _digest_without(document, "proposal_digest")
    if document["proposal_digest"] != expected_digest:
        raise ConstitutionProposalError("CONSTITUTION_PROPOSAL_DIGEST_MISMATCH")
    metric_contracts = document["metric_contracts"]
    if not isinstance(metric_contracts, list):  # schema validation should catch this
        raise ConstitutionProposalError("CONSTITUTION_PROPOSAL_METRICS_INVALID")
    by_id: dict[str, Mapping[str, object]] = {}
    for metric in metric_contracts:
        if not isinstance(metric, Mapping):
            raise ConstitutionProposalError("CONSTITUTION_PROPOSAL_METRIC_INVALID")
        validate_metric_adequacy(metric, root=root)
        metric_id = str(metric["metric_id"])
        if metric_id in by_id:
            raise ConstitutionProposalError(f"CONSTITUTION_PROPOSAL_METRIC_DUPLICATE:{metric_id}")
        by_id[metric_id] = metric

    protected_ids = [str(value) for value in document["protected_metric_ids"]]
    if len(protected_ids) != len(set(protected_ids)):
        raise ConstitutionProposalError("CONSTITUTION_PROPOSAL_PROTECTED_DUPLICATE")
    if any(metric_id not in by_id for metric_id in protected_ids):
        raise ConstitutionProposalError("CONSTITUTION_PROPOSAL_PROTECTED_METRIC_UNKNOWN")
    if any(by_id[metric_id].get("role") != "protected" for metric_id in protected_ids):
        raise ConstitutionProposalError("CONSTITUTION_PROPOSAL_PROTECTED_ROLE_REQUIRED")
    if parent_protected_metric_ids is None:
        raise ConstitutionProposalError("CONSTITUTION_PARENT_PROTECTED_SET_REQUIRED")
    parent_ids = {str(value) for value in parent_protected_metric_ids}
    if not parent_ids:
        raise ConstitutionProposalError("CONSTITUTION_PARENT_PROTECTED_SET_EMPTY")
    if not parent_ids.issubset(set(protected_ids)):
        raise ConstitutionProposalError("CONSTITUTION_PROPOSAL_PROTECTED_METRIC_REMOVED")

    promoted_ids = [str(value) for value in document["promoted_metric_ids"]]
    if len(promoted_ids) != len(set(promoted_ids)):
        raise ConstitutionProposalError("CONSTITUTION_PROPOSAL_PROMOTED_DUPLICATE")
    if any(metric_id not in by_id for metric_id in promoted_ids):
        raise ConstitutionProposalError("CONSTITUTION_PROPOSAL_PROMOTED_METRIC_UNKNOWN")
    state = str(document["state"])
    if state in {"candidate", "shadow", "probation"}:
        if promoted_ids:
            raise ConstitutionProposalError("CONSTITUTION_PROPOSAL_PROMOTION_BEFORE_APPROVAL")
        if any(
            metric.get("role") not in {"protected", "diagnostic"}
            for metric in metric_contracts
        ):
            raise ConstitutionProposalError("CONSTITUTION_PROPOSAL_VERDICT_ROLE_BEFORE_APPROVAL")
    if any(by_id[metric_id].get("role") == "protected" for metric_id in promoted_ids):
        raise ConstitutionProposalError("CONSTITUTION_PROPOSAL_PROTECTED_PROMOTION_FORBIDDEN")

    for metric in metric_contracts:
        for guard_id in metric.get("required_guard_metrics", []):
            if guard_id not in by_id:
                raise ConstitutionProposalError(
                    f"CONSTITUTION_PROPOSAL_GUARD_METRIC_UNKNOWN:{guard_id}"
                )
            if by_id[guard_id].get("role") not in {"protected", "guard"}:
                raise ConstitutionProposalError(
                    f"CONSTITUTION_PROPOSAL_GUARD_ROLE_INVALID:{guard_id}"
                )

    parent_constitution = str(document["parent_constitution"])
    if not _CONTENT_REF.fullmatch(parent_constitution):
        raise ConstitutionProposalError("CONSTITUTION_PROPOSAL_PARENT_REF_INVALID")
    for field in ("counterexample_refs", "evidence_refs"):
        refs = document[field]
        if any(not _CONTENT_REF.fullmatch(str(value)) for value in refs):
            raise ConstitutionProposalError(f"CONSTITUTION_PROPOSAL_{field.upper()}_INVALID")
    governance = document["governance"]
    if not isinstance(governance, Mapping):
        raise ConstitutionProposalError("CONSTITUTION_PROPOSAL_GOVERNANCE_INVALID")
    if any(
        not _CONTENT_REF.fullmatch(str(value))
        for value in governance.get("approval_refs", [])
    ):
        raise ConstitutionProposalError("CONSTITUTION_PROPOSAL_APPROVAL_REF_INVALID")
    _validate_governance_mode(document, governance)


def build_constitution_proposal(
    *,
    proposal_id: str,
    parent_constitution: str,
    candidate_version: str,
    objective: str,
    metric_contracts: Sequence[Mapping[str, object]],
    protected_metric_ids: Sequence[str],
    counterexample_refs: Sequence[str],
    evidence_refs: Sequence[str],
    changes: Mapping[str, str],
    parent_protected_metric_ids: Sequence[str],
    promoted_metric_ids: Sequence[str] = (),
    state: str = "candidate",
    gates: Mapping[str, bool] | None = None,
    approval_refs: Sequence[str] = (),
    approval_quorum: int = 2,
    schema_version: int = 1,
    approval_required: bool = True,
    transition_policy_digest: str | None = None,
    autonomous_promotion_metric_ids: Sequence[str] = (),
    autonomous_promotion_role: str | None = None,
    autonomous_evidence: Mapping[str, Sequence[str]] | None = None,
    root: Path | None = None,
) -> dict[str, object]:
    """Build a deterministic L2 candidate; the result remains non-authoritative."""

    body: dict[str, object] = {
        "schema_version": schema_version,
        "artifact_type": "wmloop-constitution-proposal",
        "proposal_id": proposal_id,
        "parent_constitution": parent_constitution,
        "candidate_version": candidate_version,
        "state": state,
        "authority_level": "L2",
        "objective": objective,
        "metric_contracts": [dict(metric) for metric in metric_contracts],
        "protected_metric_ids": list(protected_metric_ids),
        "promoted_metric_ids": list(promoted_metric_ids),
        "counterexample_refs": list(counterexample_refs),
        "evidence_refs": list(evidence_refs),
        "changes": dict(changes),
        "gates": dict(gates or {name: False for name in _APPROVAL_GATES}),
        "governance": {
            "approval_required": approval_required,
            "approval_refs": list(approval_refs),
            "approval_quorum": approval_quorum,
        },
    }
    if schema_version == 2:
        if transition_policy_digest is not None:
            body["transition_policy_digest"] = transition_policy_digest
        body["autonomous_promotion_metric_ids"] = list(autonomous_promotion_metric_ids)
        if autonomous_promotion_role is not None:
            body["autonomous_promotion_role"] = autonomous_promotion_role
        if autonomous_evidence is not None:
            body["autonomous_evidence"] = {
                name: list(values) for name, values in autonomous_evidence.items()
            }
    body["proposal_digest"] = _digest_without(body, "proposal_digest")
    validate_constitution_proposal(
        body,
        parent_protected_metric_ids=parent_protected_metric_ids,
        root=root,
    )
    return body


def _validate_governance_mode(
    document: Mapping[str, object], governance: Mapping[str, object]
) -> None:
    schema_version = int(document["schema_version"])
    approval_required = governance.get("approval_required")
    approval_refs = governance.get("approval_refs", [])
    if schema_version == 1:
        if approval_required is not True:
            raise ConstitutionProposalError("CONSTITUTION_PROPOSAL_APPROVAL_REQUIRED")
        if any(
            name in document
            for name in (
                "transition_policy_digest",
                "autonomous_promotion_metric_ids",
                "autonomous_promotion_role",
                "autonomous_evidence",
            )
        ):
            raise ConstitutionProposalError("CONSTITUTION_PROPOSAL_AUTONOMOUS_FIELDS_REQUIRE_V2")
        return
    if approval_required is not False:
        raise ConstitutionProposalError("CONSTITUTION_PROPOSAL_AUTONOMOUS_APPROVAL_MODE_REQUIRED")
    if approval_refs:
        raise ConstitutionProposalError("CONSTITUTION_PROPOSAL_AUTONOMOUS_APPROVAL_REFS_FORBIDDEN")
    if str(document["state"]) == "approved":
        raise ConstitutionProposalError("CONSTITUTION_PROPOSAL_AUTONOMOUS_DIRECT_APPROVAL_FORBIDDEN")
    policy_digest = document.get("transition_policy_digest")
    if not isinstance(policy_digest, str) or not re.fullmatch(r"[0-9a-f]{64}", policy_digest):
        raise ConstitutionProposalError("CONSTITUTION_PROPOSAL_TRANSITION_POLICY_INVALID")
    promotion_ids = document.get("autonomous_promotion_metric_ids")
    if not isinstance(promotion_ids, list) or not promotion_ids:
        raise ConstitutionProposalError("CONSTITUTION_PROPOSAL_AUTONOMOUS_PROMOTION_REQUIRED")
    if len({str(value) for value in promotion_ids}) != len(promotion_ids):
        raise ConstitutionProposalError("CONSTITUTION_PROPOSAL_AUTONOMOUS_PROMOTION_DUPLICATE")
    role = document.get("autonomous_promotion_role")
    if role not in {"primary", "guard"}:
        raise ConstitutionProposalError("CONSTITUTION_PROPOSAL_AUTONOMOUS_PROMOTION_ROLE_INVALID")
    evidence = document.get("autonomous_evidence")
    if not isinstance(evidence, Mapping):
        raise ConstitutionProposalError("CONSTITUTION_PROPOSAL_AUTONOMOUS_EVIDENCE_REQUIRED")
    for name in _AUTONOMOUS_EVIDENCE_FIELDS:
        refs = evidence.get(name)
        if not isinstance(refs, list) or not refs:
            raise ConstitutionProposalError(
                "CONSTITUTION_PROPOSAL_AUTONOMOUS_EVIDENCE_MISSING:" + name
            )
        if any(not _CONTENT_REF.fullmatch(str(value)) for value in refs):
            raise ConstitutionProposalError(
                "CONSTITUTION_PROPOSAL_AUTONOMOUS_EVIDENCE_INVALID:" + name
            )


def evaluate_constitution_proposal(
    document: Mapping[str, object],
    *,
    parent_protected_metric_ids: Sequence[str] | None = None,
    root: Path | None = None,
) -> dict[str, object]:
    """Report shadow/probation readiness without granting claim authority."""

    validate_constitution_proposal(
        document,
        parent_protected_metric_ids=parent_protected_metric_ids,
        root=root,
    )
    state = str(document["state"])
    gates = document["gates"]
    governance = document["governance"]
    blockers: list[str] = []
    if state == "probation":
        blockers.extend(
            f"GATE_NOT_PASSED:{name}" for name in _PROBATION_GATES if gates[name] is not True
        )
    elif state == "approved":
        blockers.extend(
            f"GATE_NOT_PASSED:{name}" for name in _APPROVAL_GATES if gates[name] is not True
        )
        approval_count = len(governance["approval_refs"])
        if approval_count < int(governance["approval_quorum"]):
            blockers.append("EXTERNAL_APPROVAL_QUORUM_NOT_MET")
    return {
        "schema_version": 1,
        "artifact_type": "wmloop-constitution-proposal-verdict",
        "proposal_id": document["proposal_id"],
        "proposal_digest": document["proposal_digest"],
        "state": state,
        "shadow_ready": state in {"shadow", "probation", "approved"} and not blockers,
        "approval_ready": state == "approved" and not blockers,
        "verdict_authority": False,
        "claim_boundary": "A proposal is a shadow candidate; only a separately frozen constitution can grant verdict authority.",
        "blockers": blockers,
    }


def advance_constitution_proposal(
    document: Mapping[str, object],
    target_state: str,
    *,
    parent_protected_metric_ids: Sequence[str] | None = None,
    root: Path | None = None,
) -> dict[str, object]:
    """Advance one proposal through the narrow, auditable state machine."""

    validate_constitution_proposal(
        document,
        parent_protected_metric_ids=parent_protected_metric_ids,
        root=root,
    )
    current = str(document["state"])
    if target_state not in _STATES:
        raise ConstitutionProposalError("CONSTITUTION_PROPOSAL_STATE_INVALID")
    if target_state not in _TRANSITIONS[current]:
        raise ConstitutionProposalError(
            f"CONSTITUTION_PROPOSAL_TRANSITION_FORBIDDEN:{current}->{target_state}"
        )
    candidate = copy.deepcopy(dict(document))
    candidate["state"] = target_state
    candidate["proposal_digest"] = _digest_without(candidate, "proposal_digest")
    assessment = evaluate_constitution_proposal(
        candidate,
        parent_protected_metric_ids=parent_protected_metric_ids,
        root=root,
    )
    if assessment["blockers"]:
        raise ConstitutionProposalError(
            "CONSTITUTION_PROPOSAL_GATES_BLOCKED:" + ",".join(assessment["blockers"])
        )
    return candidate


def _digest_without(document: Mapping[str, object], field: str) -> str:
    payload = {key: value for key, value in document.items() if key != field}
    try:
        encoded = json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ConstitutionProposalError("CONSTITUTION_PROPOSAL_PAYLOAD_INVALID") from exc
    return hashlib.sha256(encoded).hexdigest()


def _load(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConstitutionProposalError("CONSTITUTION_PROPOSAL_FILE_INVALID") from exc
    if not isinstance(payload, dict):
        raise ConstitutionProposalError("CONSTITUTION_PROPOSAL_FILE_INVALID")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "evaluate"))
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--parent-protected-metric", action="append", default=[])
    args = parser.parse_args(argv)
    proposal = _load(args.proposal)
    kwargs = {"parent_protected_metric_ids": args.parent_protected_metric}
    if args.command == "validate":
        validate_constitution_proposal(proposal, **kwargs)
        result: Mapping[str, Any] = {"valid": True, "proposal_id": proposal["proposal_id"]}
    else:
        result = evaluate_constitution_proposal(proposal, **kwargs)
    print(json.dumps(result, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
