"""Evidence-bound next-task decisions and local terminal archiving."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from wmloop.contracts import ContractValidationError, validate_document
from wmloop.control.model_portrait import validate_model_portrait
from wmloop.geometry.evidence_ir import reject_runtime_bindings
from wmloop.geometry.types import GeometryValidationError


class ClosedLoopReplanningError(ValueError):
    """A closed-loop decision, audit, or archive crossed its contract."""


_HARD_AUDIT_STOPS = {
    "duplicate_candidate": "duplicate_candidate",
    "protocol_drift": "protocol_drift",
    "cleanup": "cleanup_failure",
    "authority": "authority_violation",
    "non_portability": "non_portable_knowledge",
}
_POLICY_GAP_TRIGGERS = {
    "pending_interface_extension",
    "requires_evaluator_binding",
    "resource_binding_required",
}


def build_next_task_decision(
    *,
    work_id: str,
    trigger: str,
    signals: Mapping[str, object],
    quality_audit: Mapping[str, object],
    minimum_information_gain: float,
    maximum_replans: int,
    stop_on_confirmed_positive: bool,
    available_tasks: Sequence[str] | None = None,
    root: Path | None = None,
) -> dict[str, object]:
    """Rank one bounded observation, intervention-discovery, or stop action."""

    normalized = _signals(signals)
    allowed = _available_tasks(available_tasks)
    _validate_quality_audit(quality_audit, root=root)
    stop_reason = _audit_stop_reason(quality_audit)
    if stop_reason is None:
        if trigger == "confirmed_positive" and stop_on_confirmed_positive:
            stop_reason = "success"
        elif normalized["replan_count"] >= maximum_replans:
            stop_reason = "exhausted_budget"
        elif trigger == "portfolio_budget_blocked":
            stop_reason = "exhausted_budget"
        elif trigger == "architecture_bound":
            stop_reason = "architecture_mismatch"
        elif trigger in _POLICY_GAP_TRIGGERS:
            stop_reason = "unresolved_policy_gap"
        elif trigger in {"missing_data_regime", "operational_failure"}:
            stop_reason = "insufficient_evidence"
    if stop_reason is not None:
        selected = "stop"
    elif normalized["stale_portrait"] and "observe_portrait" in allowed:
        selected = "observe_portrait"
    elif normalized["stale_portrait"]:
        selected = "stop"
        stop_reason = "unresolved_policy_gap"
    elif normalized["information_gain"] < minimum_information_gain:
        selected = "stop"
        stop_reason = "low_information_gain"
    elif "discover_intervention" in allowed:
        selected = "discover_intervention"
    else:
        selected = "stop"
        stop_reason = "unresolved_policy_gap"

    discover_score = round(
        (
            normalized["residual"]
            + normalized["counterexample"]
            + normalized["uncertainty"]
            + normalized["information_gain"]
        )
        / 4.0,
        8,
    )
    observe_score = round(
        max(
            normalized["uncertainty"],
            1.0 if normalized["stale_portrait"] else 0.0,
        ),
        8,
    )
    stop_score = 1.0 if stop_reason is not None else round(
        max(0.0, minimum_information_gain - normalized["information_gain"]),
        8,
    )
    scores = {
        task: score
        for task, score in {
            "observe_portrait": observe_score,
            "discover_intervention": discover_score,
            "stop": stop_score,
        }.items()
        if task in allowed
    }
    scores[selected] = max(1.0, scores[selected])
    reasons = {
        "observe_portrait": (
            "The active portrait contains stale or unresolved observation coverage."
        ),
        "discover_intervention": (
            "Residual, counterexample, uncertainty, and expected information gain "
            "justify a new bounded source-discovery pass."
        ),
        "stop": "A declared stop condition or quality boundary prevents useful work.",
    }
    ordered = sorted(scores, key=lambda task: (-scores[task], task))
    ranked = [
        {
            "task": task,
            "rank": index + 1,
            "score": scores[task],
            "reason": reasons[task],
        }
        for index, task in enumerate(ordered)
    ]
    body: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "verdiwm-closed-loop-next-task",
        "work_id": _text(work_id, "CLOSED_LOOP_WORK_ID_INVALID"),
        "trigger": _text(trigger, "CLOSED_LOOP_TRIGGER_INVALID"),
        "state": "stop" if selected == "stop" else "continue",
        "selected_task": selected,
        "stop_reason": stop_reason,
        "ranked_tasks": ranked,
        "signals": normalized,
        "authority": {
            "gpu_authority": False,
            "evaluator_authority": False,
            "promotion_authority": False,
            "model_mutation_authority": False,
        },
        "claim_boundary": (
            "This record ranks the next bounded task only. It grants no GPU, "
            "evaluator, model-mutation, or scientific-promotion authority."
        ),
    }
    body["decision_id"] = "closed-loop-decision-" + _digest(body)[:24]
    _validate("closed_loop_next_task", body, root=root)
    return body


def build_quality_audit(
    *,
    snapshot: Mapping[str, object],
    portrait: Mapping[str, object],
    protocol_findings: Sequence[str],
    cleanup_findings: Sequence[str],
    non_portability_findings: Sequence[str],
    archive_receipts: Sequence[Mapping[str, object]],
    root: Path | None = None,
) -> dict[str, object]:
    """Audit the durable loop without converting findings into claim authority."""

    validate_model_portrait(portrait, root=root)
    duplicate_findings = _duplicate_candidate_findings(snapshot)
    stale_findings = _stale_portrait_findings(portrait)
    authority_findings = _authority_findings(snapshot)
    terminal_ids = {
        str(row["work_id"])
        for row in _rows(snapshot.get("work_items"))
        if row.get("state") in {"terminal", "imported_terminal"}
    }
    archived_ids = {
        str(row.get("work_id"))
        for row in archive_receipts
        if row.get("state") == "ready"
    }
    missing_archives = terminal_ids - archived_ids
    cleanup_codes = sorted(
        {
            *(_codes(cleanup_findings)),
            *("TERMINAL_WORK_NOT_ARCHIVED" for _ in missing_archives),
        }
    )
    checks = {
        "duplicate_candidate": _check(duplicate_findings),
        "stale_portrait": _check(stale_findings),
        "protocol_drift": _check(protocol_findings),
        "cleanup": _check(cleanup_codes),
        "authority": _check(authority_findings),
        "non_portability": _check(non_portability_findings),
    }
    body: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "verdiwm-closed-loop-quality-audit",
        "state": (
            "action_required"
            if any(row["state"] == "fail" for row in checks.values())
            else "ready"
        ),
        "checks": checks,
        "terminal_work_count": len(terminal_ids),
        "archived_work_count": len(terminal_ids & archived_ids),
        "claim_boundary": (
            "This local audit detects control-plane quality defects. It does not "
            "promote experiment results or export local execution records."
        ),
    }
    body["audit_id"] = "closed-loop-audit-" + _digest(body)[:24]
    _validate("closed_loop_quality_audit", body, root=root)
    return body


def archive_work(
    *,
    work: Mapping[str, object],
    state_root: Path,
    archive_root: Path,
    terminal_outcome: str,
    root: Path | None = None,
) -> tuple[dict[str, object], Path]:
    """Copy one work item's immutable inputs and receipts into local CAS."""

    state = Path(state_root).expanduser().resolve()
    destination = _prepare_directory(archive_root)
    files: dict[str, Path] = {}
    work_root = state / "work" / str(work["work_id"])
    if not work_root.is_dir() or work_root.is_symlink():
        raise ClosedLoopReplanningError("CLOSED_LOOP_ARCHIVE_WORK_ROOT_INVALID")
    for path in sorted(work_root.rglob("*")):
        if path.is_symlink():
            raise ClosedLoopReplanningError("CLOSED_LOOP_ARCHIVE_SYMLINK_FORBIDDEN")
        if path.is_file():
            files["work/" + path.relative_to(work_root).as_posix()] = path
    for field in ("assessment_path", "idea_path", "work_order_path"):
        value = work.get(field)
        path = Path(str(value or "")).expanduser().resolve()
        if not path.is_file() or path.is_symlink():
            raise ClosedLoopReplanningError(
                "CLOSED_LOOP_ARCHIVE_INPUT_INVALID:" + field
            )
        files["input/" + field] = path
    artifacts = []
    for logical_name, path in sorted(files.items()):
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        cas_path = destination / "cas" / "sha256" / digest[:2] / digest
        _write_bytes_idempotent(cas_path, payload)
        artifacts.append(
            {
                "logical_name": logical_name,
                "sha256": digest,
                "cas_ref": "cas://sha256/" + digest,
                "size_bytes": len(payload),
            }
        )
    body: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "verdiwm-terminal-archive-receipt",
        "work_id": _text(str(work["work_id"]), "CLOSED_LOOP_WORK_ID_INVALID"),
        "state": "ready",
        "terminal_outcome": _text(
            terminal_outcome, "CLOSED_LOOP_TERMINAL_OUTCOME_INVALID"
        ),
        "artifacts": artifacts,
        "claim_boundary": (
            "This local receipt proves byte-level archival only. It carries no "
            "scientific verdict or community-promotion authority."
        ),
    }
    body["archive_id"] = "terminal-archive-" + _digest(body)[:24]
    _validate("terminal_archive_receipt", body, root=root)
    receipt_path = (
        destination
        / "receipts"
        / str(work["work_id"])
        / f"{body['archive_id']}.json"
    )
    _write_bytes_idempotent(receipt_path, _canonical(body) + b"\n")
    return body, receipt_path


def load_archive_receipts(
    archive_root: Path, *, root: Path | None = None
) -> list[dict[str, object]]:
    destination = Path(archive_root).expanduser().resolve()
    receipts = []
    receipt_root = destination / "receipts"
    if not receipt_root.is_dir():
        return receipts
    for path in sorted(receipt_root.rglob("*.json")):
        if path.is_symlink() or not path.is_file():
            raise ClosedLoopReplanningError("CLOSED_LOOP_ARCHIVE_RECEIPT_INVALID")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ClosedLoopReplanningError(
                "CLOSED_LOOP_ARCHIVE_RECEIPT_INVALID"
            ) from exc
        if not isinstance(payload, dict):
            raise ClosedLoopReplanningError("CLOSED_LOOP_ARCHIVE_RECEIPT_INVALID")
        _validate("terminal_archive_receipt", payload, root=root)
        body = dict(payload)
        received = body.pop("archive_id", None)
        if received != "terminal-archive-" + _digest(body)[:24]:
            raise ClosedLoopReplanningError(
                "CLOSED_LOOP_ARCHIVE_RECEIPT_ID_MISMATCH"
            )
        receipts.append(payload)
    return receipts


def _audit_stop_reason(audit: Mapping[str, object]) -> str | None:
    checks = audit.get("checks")
    if not isinstance(checks, Mapping):
        raise ClosedLoopReplanningError("CLOSED_LOOP_AUDIT_INVALID")
    for name, reason in _HARD_AUDIT_STOPS.items():
        row = checks.get(name)
        if isinstance(row, Mapping) and row.get("state") == "fail":
            return reason
    return None


def _validate_quality_audit(
    audit: Mapping[str, object], *, root: Path | None
) -> None:
    _validate("closed_loop_quality_audit", audit, root=root)
    body = dict(audit)
    received = body.pop("audit_id", None)
    if received != "closed-loop-audit-" + _digest(body)[:24]:
        raise ClosedLoopReplanningError("CLOSED_LOOP_AUDIT_ID_MISMATCH")


def _signals(values: Mapping[str, object]) -> dict[str, object]:
    result: dict[str, object] = {}
    for field in ("residual", "counterexample", "uncertainty", "information_gain"):
        value = values.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ClosedLoopReplanningError("CLOSED_LOOP_SIGNAL_INVALID:" + field)
        result[field] = round(max(0.0, min(1.0, float(value))), 8)
    if not isinstance(values.get("stale_portrait"), bool):
        raise ClosedLoopReplanningError("CLOSED_LOOP_SIGNAL_INVALID:stale_portrait")
    count = values.get("replan_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ClosedLoopReplanningError("CLOSED_LOOP_SIGNAL_INVALID:replan_count")
    result["stale_portrait"] = values["stale_portrait"]
    result["replan_count"] = count
    return result


def _available_tasks(values: Sequence[str] | None) -> set[str]:
    allowed = {"observe_portrait", "discover_intervention", "stop"}
    if values is None:
        return allowed
    candidate = {str(value) for value in values}
    if not candidate or not candidate.issubset(allowed) or "stop" not in candidate:
        raise ClosedLoopReplanningError("CLOSED_LOOP_AVAILABLE_TASKS_INVALID")
    return candidate


def _duplicate_candidate_findings(snapshot: Mapping[str, object]) -> list[str]:
    owners: dict[str, set[str]] = defaultdict(set)
    for row in _rows(snapshot.get("stage_receipts")):
        work_id = str(row.get("work_id") or "")
        for candidate_id in _values_for_key(row.get("payload"), "candidate_id"):
            owners[candidate_id].add(work_id)
    return [
        "DUPLICATE_CANDIDATE_ID"
        for work_ids in owners.values()
        if len(work_ids) > 1
    ]


def _stale_portrait_findings(portrait: Mapping[str, object]) -> list[str]:
    coverage = portrait.get("coverage")
    if not isinstance(coverage, Mapping):
        return ["PORTRAIT_COVERAGE_INVALID"]
    findings = []
    if coverage.get("stale_fingerprint_ids"):
        findings.append("STALE_PORTRAIT_FINGERPRINTS")
    if coverage.get("conflicts"):
        findings.append("PORTRAIT_EVIDENCE_CONFLICT")
    if coverage.get("unknown_operational_metrics"):
        findings.append("PORTRAIT_OPERATIONAL_COVERAGE_UNKNOWN")
    return findings


def _authority_findings(snapshot: Mapping[str, object]) -> list[str]:
    findings = []
    forbidden = {
        "gpu_authority",
        "evaluator_authority",
        "promotion_authority",
        "model_mutation_authority",
    }
    for row in _rows(snapshot.get("stage_receipts")):
        for key, value in _mapping_items(row.get("payload")):
            if key in forbidden and value is True:
                findings.append("UNAUTHORIZED_STAGE_AUTHORITY:" + key)
    return sorted(set(findings))


def _values_for_key(value: object, key: str) -> list[str]:
    values = []
    if isinstance(value, Mapping):
        for name, child in value.items():
            if name == key and isinstance(child, str) and child:
                values.append(child)
            values.extend(_values_for_key(child, key))
    elif isinstance(value, list):
        for child in value:
            values.extend(_values_for_key(child, key))
    return values


def _mapping_items(value: object) -> list[tuple[str, object]]:
    items = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            items.append((str(key), child))
            items.extend(_mapping_items(child))
    elif isinstance(value, list):
        for child in value:
            items.extend(_mapping_items(child))
    return items


def _check(findings: Sequence[str]) -> dict[str, object]:
    codes = _codes(findings)
    return {"state": "fail" if codes else "pass", "finding_codes": codes}


def _codes(values: Sequence[str]) -> list[str]:
    return sorted({str(value) for value in values if str(value)})


def _rows(value: object) -> list[Mapping[str, object]]:
    if not isinstance(value, list) or any(not isinstance(row, Mapping) for row in value):
        raise ClosedLoopReplanningError("CLOSED_LOOP_SNAPSHOT_INVALID")
    return list(value)  # type: ignore[return-value]


def _prepare_directory(path: Path) -> Path:
    raw = Path(path).expanduser()
    if raw.is_symlink():
        raise ClosedLoopReplanningError("CLOSED_LOOP_ARCHIVE_ROOT_INVALID")
    destination = raw.resolve()
    destination.mkdir(mode=0o700, parents=True, exist_ok=True)
    return destination


def _write_bytes_idempotent(path: Path, payload: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise ClosedLoopReplanningError("CLOSED_LOOP_ARCHIVE_WRITE_CONFLICT")
        return
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _validate(schema: str, value: Mapping[str, object], *, root: Path | None) -> None:
    try:
        validate_document(schema, value, root=root)
    except ContractValidationError as exc:
        raise ClosedLoopReplanningError(
            f"CLOSED_LOOP_SCHEMA_INVALID:{schema}:{exc}"
        ) from exc


def _text(value: object, code: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ClosedLoopReplanningError(code)
    text = value.strip()
    try:
        reject_runtime_bindings(text)
    except GeometryValidationError:
        raise ClosedLoopReplanningError(code) from None
    return text


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ClosedLoopReplanningError("CLOSED_LOOP_CANONICAL_INVALID") from exc


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()
