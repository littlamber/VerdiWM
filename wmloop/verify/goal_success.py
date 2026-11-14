"""Read settled, current-iteration evidence for a predeclared goal stop.

This reader never constructs a verdict. A successful pipeline or an exploratory
confirm stage cannot grant the authority of a frozen, replicated judge verdict.
"""

from __future__ import annotations

import json
import math
import sqlite3
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from wmloop.archive.receipt_evidence import ReceiptEvidenceError, verify_receipt_evidence
from wmloop.archive.settlement_evidence import read_settled_trial_rows
from wmloop.archive.store import ContentAddressedStore
from wmloop.contracts import load_yaml_document, validate_document
from wmloop.storage import checked_path, sha256_bytes
from wmloop.verify.acceptance_policy import has_frozen_policy_binding, verification_policy_from_goal


def resolve_frozen_goal(evaluator_path: Path) -> Path | None:
    """Resolve the evaluator's optional goal, validating a formal v2 policy."""
    evaluator = checked_path(evaluator_path, code="EVOLUTION_EVALUATOR_INVALID", error=ValueError)
    document = _object(evaluator.read_bytes())
    value = document.get("frozen_goal")
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError("EVOLUTION_FROZEN_GOAL_INVALID")
    path = Path(value)
    if not path.is_absolute():
        path = evaluator.parent / path
    path = checked_path(path, code="EVOLUTION_FROZEN_GOAL_INVALID", error=ValueError)
    goal = load_yaml_document(path)
    if goal.get("schema_version") != 2 or verification_policy_from_goal(goal) is None:
        raise ValueError("EVOLUTION_FROZEN_GOAL_POLICY_REQUIRED")
    return path


def evaluate_iteration_goal_success(
    *, iteration_root: Path, archive_db: Path, cas_root: Path, frozen_goal: Path | None,
    prior_receipt_refs: Sequence[str] = (),
) -> dict[str, object]:
    """Only inspect confirm receipts named in this iteration's queue execution.

    A formal trial must be referenced by a CAS artifact of the settled outer
    confirm receipt. Scanning unrelated historic positive trials is forbidden.
    Invalid or missing evidence stays unconfirmed and supplies a visible reason.
    """
    def pending(reason: str) -> dict[str, object]:
        return {"state": "not_confirmed", "reason": reason, "evidence_refs": []}

    if frozen_goal is None:
        return pending("FROZEN_GOAL_NOT_CONFIGURED")
    try:
        goal_path = checked_path(frozen_goal, code="GOAL_SUCCESS_GOAL_INVALID", error=ValueError)
        goal = load_yaml_document(goal_path)
        policy = verification_policy_from_goal(goal)
        if goal.get("schema_version") != 2 or policy is None:
            return pending("FROZEN_GOAL_POLICY_REQUIRED")
        root = checked_path(iteration_root, code="GOAL_SUCCESS_ITERATION_INVALID", error=ValueError)
        queue = root / "pipeline" / "compiled" / "queue"
        execution_path = checked_path(queue / "execution.json", code="GOAL_SUCCESS_QUEUE_INVALID", error=ValueError)
        execution = _object(execution_path.read_bytes())
        batch = _object((root / "inputs" / "candidate-batch.json").read_bytes())
        queue_document = _object((queue / "queue.json").read_bytes())
        campaign_id = batch["campaign_id"]
        if queue_document.get("campaign_id") != campaign_id or execution.get("campaign_id") != campaign_id:
            return pending("CONFIRM_ITERATION_BINDING_INVALID")
        plans = {
            str(selected["candidate_id"]) + ":confirm": stage
            for selected in queue_document["selected"]
            for stage in selected["stages"] if stage["stage"] == "confirm"
        }
        results = execution.get("results")
        if not isinstance(results, Mapping):
            return pending("CONFIRM_EVIDENCE_MISSING")
        database = checked_path(archive_db, code="GOAL_SUCCESS_ARCHIVE_INVALID", error=ValueError)
        if not database.is_file() or not (Path(cas_root) / "cas").is_dir():
            return pending("SETTLED_EVIDENCE_MISSING")
        cas = ContentAddressedStore(cas_root)
        with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as connection:
            connection.row_factory = sqlite3.Row
            connection.execute("BEGIN")
            for result_key, result in results.items():
                if not isinstance(result, Mapping) or result.get("stage") != "confirm" or result.get("state") != "completed":
                    continue
                path = checked_path(Path(str(result.get("manifest_path", ""))), code="GOAL_SUCCESS_MANIFEST_INVALID", error=ValueError)
                if not path.is_relative_to(queue / "runs"):
                    return pending("CONFIRM_ITERATION_BINDING_INVALID")
                manifest = _object(path.read_bytes())
                ref = manifest.get("receipt_ref")
                if not isinstance(ref, str):
                    continue
                if result_key not in plans:
                    return pending("CONFIRM_PLAN_BINDING_INVALID")
                stage = plans[result_key]
                plan_path = checked_path(queue / stage["plan_path"], code="GOAL_SUCCESS_PLAN_INVALID", error=ValueError)
                if not plan_path.is_relative_to(queue / "plans"):
                    return pending("CONFIRM_PLAN_BINDING_INVALID")
                plan_bytes = plan_path.read_bytes()
                plan = _object(plan_bytes)
                if sha256_bytes(plan_bytes) != stage["plan_sha256"]:
                    return pending("CONFIRM_PLAN_BINDING_INVALID")
                receipt = _object(cas.read_bytes(ref))
                row = _settled_row(connection, ref)
                if row is None or receipt.get("archive_trial_id") != row["trial_id"] or receipt.get("stage") != "confirm" or receipt.get("settlement_state") != "settled":
                    continue
                if (
                    receipt.get("campaign_id") != campaign_id or plan.get("campaign_id") != campaign_id
                    or plan.get("stage") != "confirm" or receipt.get("plan_sha256") != stage["plan_sha256"]
                    or receipt.get("trial_id") != plan.get("trial_id")
                    or any(manifest.get(key) != receipt.get(key) for key in ("stage", "campaign_id", "trial_id", "archive_trial_id", "plan_sha256"))
                    or receipt.get("verdict_ref") != row["verdict_ref"]
                    or receipt.get("failure_context_ref") != row["failure_context_ref"]
                ):
                    return pending("CONFIRM_ITERATION_BINDING_INVALID")
                verify_receipt_evidence(receipt=receipt, cas=cas)
                for name, artifact_ref in receipt["artifact_refs"].items():
                    if not str(name).endswith(".json"):
                        continue
                    artifact = _object(cas.read_bytes(artifact_ref))
                    references = [artifact_ref, artifact.get("receipt_ref")]
                    nested = artifact.get("receipt")
                    if isinstance(nested, Mapping):
                        references.append(nested.get("receipt_ref"))
                    for candidate_ref in references:
                        if not isinstance(candidate_ref, str):
                            continue
                        if candidate_ref in prior_receipt_refs:
                            continue
                        evidence = _formal_success(connection, cas, candidate_ref, policy.sha256, policy.goal_sha256, goal["goal_id"])
                        if evidence is not None:
                            return {"state": "confirmed", "reason": "FROZEN_GOAL_CONFIRMED", "evidence_refs": [ref, artifact_ref, *evidence]}
        return pending("FORMAL_GOAL_EVIDENCE_NOT_CONFIRMED")
    except (OSError, ValueError, TypeError, KeyError, sqlite3.Error, ReceiptEvidenceError) as exc:
        return pending("GOAL_EVIDENCE_INVALID:" + type(exc).__name__)


def capture_settled_receipt_refs(archive_db: Path) -> list[str]:
    """Snapshot prior evidence without creating a database or changing it."""
    try:
        database = checked_path(archive_db, code="GOAL_SUCCESS_BASELINE_INVALID", error=ValueError)
        if not database.exists():
            return []
        with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as connection:
            return [row[0] for row in connection.execute("SELECT receipt_ref FROM receipts ORDER BY receipt_ref")]
    except (OSError, sqlite3.Error) as exc:
        raise ValueError("GOAL_SUCCESS_BASELINE_INVALID") from exc


def _settled_row(connection: sqlite3.Connection, receipt_ref: str) -> sqlite3.Row | None:
    rows = read_settled_trial_rows(connection, receipt_ref=receipt_ref, limit=2)
    return rows[0] if len(rows) == 1 else None


def _formal_success(
    connection: sqlite3.Connection, cas: ContentAddressedStore, receipt_ref: str,
    policy_sha256: str, goal_sha256: str, goal_id: str,
) -> list[str] | None:
    row = _settled_row(connection, receipt_ref)
    if row is None or row["goal_id"] != goal_id or _object(row["settlement_json"]).get("evidence_scope") != "verified":
        return None
    receipt = _object(cas.read_bytes(receipt_ref))
    if receipt.get("state") != "ready" or receipt.get("artifact_type") != "wmloop-m3-training-eval-smoke-execution-receipt":
        return None
    for field, expected in (("archive_trial_id", row["trial_id"]), ("verdict_ref", row["verdict_ref"]), ("failure_context_ref", row["failure_context_ref"])):
        if field in receipt and receipt[field] != expected:
            return None
    _verify_cas_members(receipt, cas)
    verdict = _object(cas.read_bytes(row["verdict_ref"]))
    validate_document("verdict", verdict)
    statistics = verdict.get("replication_statistics", {})
    if (
        verdict.get("schema_version") != 2 or verdict.get("verdict") != "ACCEPT"
        or verdict.get("proposal_id") != row["proposal_id"] or receipt.get("proposal_id") != row["proposal_id"]
        or verdict.get("violation") is not None
        or not has_frozen_policy_binding(statistics, receipt=receipt)
        or statistics.get("policy_sha256") != policy_sha256
        or receipt.get("frozen_goal_ref") != "cas://sha256/" + goal_sha256
        or set(verdict["gates"].values()) != {"pass"}
        or verdict.get("negative_metrics")
    ):
        return None
    goal = _object(cas.read_bytes(receipt["frozen_goal_ref"]))
    policy_document = _object(cas.read_bytes(receipt["verification_policy_ref"]))
    policy = verification_policy_from_goal(goal)
    if policy is None or policy.sha256 != policy_sha256 or policy.to_dict() != policy_document:
        return None
    lower = statistics.get("lower_bound")
    primary = verdict["delta_m_ver"].get(policy.primary_metric)
    action = verdict["action_following_gate"]
    if (
        not _finite(lower) or lower <= policy.minimum_effect
        or not _finite(primary) or not _finite(statistics.get("mean"))
        or not math.isclose(primary, statistics["mean"], rel_tol=1e-9, abs_tol=1e-12)
        or any(not _finite(value) or value < 0 for value in verdict["delta_m_ver"].values())
        or set(verdict["delta_m_ver"]) != set(dict(policy.metric_directions))
        or action.get("enabled") is not True or action.get("pass") is not True
        or action.get("threshold") != policy.action_following_threshold
        or not _finite(action.get("observed")) or action["observed"] < policy.action_following_threshold
    ):
        return None
    cas.read_bytes(row["failure_context_ref"])
    return [receipt_ref, row["verdict_ref"], receipt["frozen_goal_ref"], receipt["verification_policy_ref"]]


def _finite(value: Any) -> bool:
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _verify_cas_members(value: object, cas: ContentAddressedStore) -> None:
    if isinstance(value, Mapping):
        for member in value.values():
            _verify_cas_members(member, cas)
    elif isinstance(value, (list, tuple)):
        for member in value:
            _verify_cas_members(member, cas)
    elif isinstance(value, str) and value.startswith("cas://"):
        cas.read_bytes(value)


def _object(payload: bytes | str) -> dict[str, Any]:
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError("GOAL_SUCCESS_OBJECT_INVALID")
    return value
