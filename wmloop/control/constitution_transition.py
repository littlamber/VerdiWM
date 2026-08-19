"""Policy-bounded authorization for future constitutional metric transitions.

The transition receipt never mutates an active constitution and never grants
verdict authority.  It only says that an explicitly frozen successor may use
the named diagnostic metrics for future work after every pre-authorized gate
has been evidenced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from wmloop.contracts import ContractValidationError, validate_document
from wmloop.control.constitution_proposal import (
    ConstitutionProposalError,
    validate_constitution_proposal,
)
from wmloop.evaluate.metric_adequacy import (
    MetricAdequacyError,
    validate_metric_adequacy_report,
)


class ConstitutionTransitionError(ValueError):
    """A policy-bounded constitutional transition was invalid or blocked."""


_CONTENT_REF = re.compile(r"^(?:cas://|urn:|sha256:)[^\s]+$")
_REQUIRED_GATES = (
    "incremental_predictive_value",
    "anti_goodhart_passed",
    "regression_passed",
    "independent_validation_passed",
    "canary_passed",
)
_EVIDENCE_STAGES = (
    "static_check",
    "shadow_evaluation",
    "historical_calibration",
    "fresh_heldout",
    "canary",
)


def validate_transition_policy(
    document: Mapping[str, object], *, root: Path | None = None
) -> None:
    """Validate the small immutable policy that bounds automatic transitions."""

    try:
        validate_document("constitution_transition_policy", document, root=root)
    except ContractValidationError as exc:
        raise ConstitutionTransitionError(
            f"CONSTITUTION_TRANSITION_POLICY_SCHEMA_INVALID:{exc}"
        ) from exc
    if document["policy_digest"] != _digest_without(document, "policy_digest"):
        raise ConstitutionTransitionError("CONSTITUTION_TRANSITION_POLICY_DIGEST_MISMATCH")
    if not _CONTENT_REF.fullmatch(str(document["parent_constitution"])):
        raise ConstitutionTransitionError("CONSTITUTION_TRANSITION_PARENT_REF_INVALID")
    protected = [str(value) for value in document["parent_protected_metric_ids"]]
    if len(protected) != len(set(protected)):
        raise ConstitutionTransitionError("CONSTITUTION_TRANSITION_PARENT_PROTECTED_DUPLICATE")
    roles = [str(value) for value in document["allowed_successor_roles"]]
    if len(roles) != len(set(roles)):
        raise ConstitutionTransitionError("CONSTITUTION_TRANSITION_SUCCESSOR_ROLE_DUPLICATE")
    stages = [str(value) for value in document["required_evidence_stages"]]
    if tuple(stages) != _EVIDENCE_STAGES:
        raise ConstitutionTransitionError("CONSTITUTION_TRANSITION_EVIDENCE_STAGES_INVALID")


def build_transition_policy(
    *,
    policy_id: str,
    parent_constitution: str,
    parent_protected_metric_ids: Sequence[str],
    allowed_successor_roles: Sequence[str] = ("primary", "guard"),
    maximum_promoted_metric_count: int = 1,
    root: Path | None = None,
) -> dict[str, object]:
    """Build a deterministic policy; storing it in a freeze is a deployment step."""

    body: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "wmloop-constitution-transition-policy",
        "policy_id": policy_id,
        "parent_constitution": parent_constitution,
        "parent_protected_metric_ids": list(parent_protected_metric_ids),
        "allowed_successor_roles": list(allowed_successor_roles),
        "maximum_promoted_metric_count": maximum_promoted_metric_count,
        "required_evidence_stages": list(_EVIDENCE_STAGES),
        "allowed_changes": {
            "goal": "unchanged",
            "verifier": "unchanged",
            "protected_metrics": "append_only",
            "allowed_actions": "unchanged",
        },
        "effective_scope": "future_work_only",
        "history_mode": "append_only",
        "successor_freeze_required": True,
    }
    body["policy_digest"] = _digest_without(body, "policy_digest")
    validate_transition_policy(body, root=root)
    return body


def authorize_constitution_transition(
    proposal: Mapping[str, object],
    policy: Mapping[str, object],
    *,
    parent_protected_metric_ids: Sequence[str] | None = None,
    metric_reports: Sequence[Mapping[str, object]] = (),
    root: Path | None = None,
) -> dict[str, object]:
    """Return a deterministic authorization receipt or the precise blockers.

    Invalid contracts raise.  Valid but insufficient evidence returns a
    `blocked` receipt, making the next required evidence leg explicit without
    granting any runtime side effect.
    """

    validate_transition_policy(policy, root=root)
    expected_parent = tuple(
        str(value) for value in policy["parent_protected_metric_ids"]
    )
    supplied_parent = tuple(parent_protected_metric_ids or expected_parent)
    if supplied_parent != expected_parent:
        raise ConstitutionTransitionError("CONSTITUTION_TRANSITION_PARENT_PROTECTED_SET_MISMATCH")
    try:
        validate_constitution_proposal(
            proposal,
            parent_protected_metric_ids=supplied_parent,
            root=root,
        )
    except ConstitutionProposalError as exc:
        raise ConstitutionTransitionError(str(exc)) from exc
    reports_by_metric: dict[str, Mapping[str, object]] = {}
    for report in metric_reports:
        try:
            validate_metric_adequacy_report(report, root=root)
        except MetricAdequacyError as exc:
            raise ConstitutionTransitionError(str(exc)) from exc
        metric_id = str(report["metric_id"])
        if metric_id in reports_by_metric:
            raise ConstitutionTransitionError(
                "AUTONOMOUS_TRANSITION_METRIC_ADEQUACY_REPORT_DUPLICATE:" + metric_id
            )
        reports_by_metric[metric_id] = report

    blockers: list[str] = []
    if proposal["schema_version"] != 2:
        blockers.append("AUTONOMOUS_TRANSITION_PROPOSAL_V2_REQUIRED")
    if proposal["state"] != "probation":
        blockers.append("AUTONOMOUS_TRANSITION_PROBATION_REQUIRED")
    if proposal["parent_constitution"] != policy["parent_constitution"]:
        blockers.append("AUTONOMOUS_TRANSITION_PARENT_CONSTITUTION_MISMATCH")
    if proposal.get("transition_policy_digest") != policy["policy_digest"]:
        blockers.append("AUTONOMOUS_TRANSITION_POLICY_DIGEST_MISMATCH")
    if proposal["changes"] != policy["allowed_changes"]:
        blockers.append("AUTONOMOUS_TRANSITION_IMMUTABLE_CORE_CHANGE_FORBIDDEN")
    gates = proposal["gates"]
    for gate in _REQUIRED_GATES:
        if gates[gate] is not True:
            blockers.append("AUTONOMOUS_TRANSITION_GATE_NOT_PASSED:" + gate)

    metric_by_id = {str(metric["metric_id"]): metric for metric in proposal["metric_contracts"]}
    protected_ids = {str(value) for value in proposal["protected_metric_ids"]}
    candidate_ids = [str(value) for value in proposal.get("autonomous_promotion_metric_ids", [])]
    if len(candidate_ids) > int(policy["maximum_promoted_metric_count"]):
        blockers.append("AUTONOMOUS_TRANSITION_PROMOTION_LIMIT_EXCEEDED")
    for metric_id in candidate_ids:
        metric = metric_by_id.get(metric_id)
        if metric is None:
            blockers.append("AUTONOMOUS_TRANSITION_PROMOTION_METRIC_UNKNOWN:" + metric_id)
        elif metric_id in protected_ids or metric["role"] != "diagnostic":
            blockers.append("AUTONOMOUS_TRANSITION_PROMOTION_NOT_DIAGNOSTIC:" + metric_id)
        else:
            report = reports_by_metric.get(metric_id)
            if report is None:
                blockers.append("AUTONOMOUS_TRANSITION_METRIC_ADEQUACY_REPORT_MISSING:" + metric_id)
            elif report["state"] != "adequate":
                blockers.append("AUTONOMOUS_TRANSITION_METRIC_ADEQUACY_INADEQUATE:" + metric_id)
            elif report["metric_contract_digest"] != _document_digest(metric):
                blockers.append("AUTONOMOUS_TRANSITION_METRIC_ADEQUACY_CONTRACT_MISMATCH:" + metric_id)
            elif report["evaluator_binding"] != metric["evaluator_binding"]:
                blockers.append("AUTONOMOUS_TRANSITION_METRIC_ADEQUACY_EVALUATOR_MISMATCH:" + metric_id)
            elif (
                "sha256:" + str(report["report_digest"])
                not in proposal["autonomous_evidence"]["metric_adequacy_report_refs"]
            ):
                blockers.append("AUTONOMOUS_TRANSITION_METRIC_ADEQUACY_REF_MISSING:" + metric_id)
    role = str(proposal.get("autonomous_promotion_role") or "none")
    if role not in set(str(value) for value in policy["allowed_successor_roles"]):
        blockers.append("AUTONOMOUS_TRANSITION_SUCCESSOR_ROLE_FORBIDDEN:" + role)

    receipt: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "wmloop-constitution-transition-receipt",
        "receipt_id": "constitution-transition-" + _receipt_id(policy, proposal),
        "state": "authorized" if not blockers else "blocked",
        "policy_id": policy["policy_id"],
        "policy_digest": policy["policy_digest"],
        "parent_constitution": policy["parent_constitution"],
        "proposal_id": proposal["proposal_id"],
        "proposal_digest": proposal["proposal_digest"],
        "authorized_metric_ids": candidate_ids if not blockers else [],
        "successor_metric_role": role if not blockers else "none",
        "effective_scope": "future_work_only",
        "successor_freeze_required": True,
        "historical_verdicts_preserved": True,
        "verdict_authority": False,
        "blockers": blockers,
    }
    receipt["receipt_digest"] = _digest_without(receipt, "receipt_digest")
    validate_transition_receipt(receipt, root=root)
    return receipt


def validate_transition_receipt(
    document: Mapping[str, object], *, root: Path | None = None
) -> None:
    """Validate an authorization receipt before any future freeze consumes it."""

    try:
        validate_document("constitution_transition_receipt", document, root=root)
    except ContractValidationError as exc:
        raise ConstitutionTransitionError(
            f"CONSTITUTION_TRANSITION_RECEIPT_SCHEMA_INVALID:{exc}"
        ) from exc
    if document["receipt_digest"] != _digest_without(document, "receipt_digest"):
        raise ConstitutionTransitionError("CONSTITUTION_TRANSITION_RECEIPT_DIGEST_MISMATCH")
    expected_id = "constitution-transition-" + hashlib.sha256(
        (str(document["policy_digest"]) + ":" + str(document["proposal_digest"])).encode("ascii")
    ).hexdigest()[:24]
    if document["receipt_id"] != expected_id:
        raise ConstitutionTransitionError("CONSTITUTION_TRANSITION_RECEIPT_ID_MISMATCH")
    if document["state"] == "authorized" and not document["authorized_metric_ids"]:
        raise ConstitutionTransitionError("CONSTITUTION_TRANSITION_AUTHORIZATION_EMPTY")
    if document["state"] == "blocked" and document["authorized_metric_ids"]:
        raise ConstitutionTransitionError("CONSTITUTION_TRANSITION_BLOCKED_AUTHORIZATION_FORBIDDEN")


def write_transition_receipt(path: Path, receipt: Mapping[str, object], *, root: Path | None = None) -> None:
    """Persist a receipt idempotently without exposing or modifying runtime state."""

    validate_transition_receipt(receipt, root=root)
    payload = _canonical_json(receipt)
    destination = Path(path).resolve()
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or not destination.is_file():
            raise ConstitutionTransitionError("CONSTITUTION_TRANSITION_RECEIPT_DESTINATION_UNSAFE")
        if destination.read_bytes() != payload:
            raise ConstitutionTransitionError("CONSTITUTION_TRANSITION_RECEIPT_ALREADY_EXISTS")
        return
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def _receipt_id(policy: Mapping[str, object], proposal: Mapping[str, object]) -> str:
    return hashlib.sha256(
        (str(policy["policy_digest"]) + ":" + str(proposal["proposal_digest"])).encode("ascii")
    ).hexdigest()[:24]


def _document_digest(document: Mapping[str, object]) -> str:
    try:
        payload = json.dumps(
            document,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ConstitutionTransitionError("CONSTITUTION_TRANSITION_PAYLOAD_INVALID") from exc
    return hashlib.sha256(payload).hexdigest()


def _digest_without(document: Mapping[str, object], field: str) -> str:
    payload = {key: value for key, value in document.items() if key != field}
    try:
        return hashlib.sha256(_canonical_json(payload)).hexdigest()
    except (TypeError, ValueError) as exc:
        raise ConstitutionTransitionError("CONSTITUTION_TRANSITION_PAYLOAD_INVALID") from exc


def _canonical_json(payload: Mapping[str, object]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _load(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConstitutionTransitionError("CONSTITUTION_TRANSITION_FILE_INVALID") from exc
    if not isinstance(payload, dict):
        raise ConstitutionTransitionError("CONSTITUTION_TRANSITION_FILE_INVALID")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--parent-protected-metric", action="append", default=[])
    parser.add_argument("--metric-adequacy-report", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    receipt = authorize_constitution_transition(
        _load(args.proposal),
        _load(args.policy),
        parent_protected_metric_ids=args.parent_protected_metric or None,
        metric_reports=[_load(path) for path in args.metric_adequacy_report],
    )
    if args.output is not None:
        write_transition_receipt(args.output, receipt)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0 if receipt["state"] == "authorized" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
