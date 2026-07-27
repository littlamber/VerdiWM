"""Audit orchestrator reports for the three mandatory log surfaces."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from wmloop.archive.store import ArchiveStore, ContentAddressedStore


class LogCompletenessAuditError(RuntimeError):
    """A log-completeness audit input or output invariant failed."""


def run_log_completeness_audit(
    *,
    orchestrator_report: Path,
    output_root: Path,
    cas_root: Path | None = None,
    archive_db: Path | None = None,
) -> dict[str, object]:
    """Write a read-only audit for §12.5 orchestrator/receipt/LLM log coverage."""

    report_path, report = _load_json_mapping(orchestrator_report, "LOG_AUDIT_REPORT_INVALID")
    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise LogCompletenessAuditError("LOG_AUDIT_OUTPUT_EXISTS")
    cas_storage_root = Path(cas_root).resolve() if cas_root is not None else _cas_root_from_report(report, report_path=report_path)
    cas = ContentAddressedStore(cas_storage_root)
    archive = ArchiveStore(archive_db) if archive_db is not None else None
    entries = _entries(report)
    observations = [_audit_entry(entry, cas=cas) for entry in entries]
    blockers = [
        blocker
        for observation in observations
        for blocker in observation["blockers"]
        if isinstance(blocker, Mapping)
    ]
    state = "ready" if not blockers and observations else "blocked"
    audit = {
        "schema_version": 1,
        "artifact_type": "wmloop-log-completeness-audit",
        "state": state,
        "source_report_path": str(report_path),
        "source_artifact_type": report.get("artifact_type"),
        "cas_root": str(cas_storage_root),
        "entry_count": len(observations),
        "ready_entry_count": sum(1 for observation in observations if observation["state"] == "ready"),
        "blocker_count": len(blockers),
        "blockers": blockers,
        "observations": observations,
        "required_surfaces": [
            "orchestrator report fields",
            "execution receipt reference",
            "LLM call log with full prompt and every raw response in CAS",
        ],
    }
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    try:
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary.mkdir(mode=0o700)
        audit_bytes = _canonical_json_bytes(audit)
        markdown_bytes = _render_markdown(audit).encode("utf-8")
        _write_bytes_atomic(temporary / "log-completeness-audit.json", audit_bytes)
        _write_bytes_atomic(temporary / "log-completeness-audit.md", markdown_bytes)
        audit_ref = cas.put_bytes(audit_bytes, media_type="application/json").uri
        markdown_ref = cas.put_bytes(markdown_bytes, media_type="text/markdown").uri
        if archive is not None:
            archive.record_artifact_reference(audit_ref)
            archive.record_artifact_reference(markdown_ref)
        manifest = {
            "schema_version": 1,
            "artifact_type": "wmloop-log-completeness-audit-manifest",
            "state": state,
            "source_report_path": str(report_path),
            "source_artifact_type": report.get("artifact_type"),
            "entry_count": len(observations),
            "ready_entry_count": audit["ready_entry_count"],
            "blocker_count": len(blockers),
            "blockers": blockers,
            "report_path": str(destination / "log-completeness-audit.json"),
            "markdown_path": str(destination / "log-completeness-audit.md"),
            "cas_refs": {
                "log_completeness_audit_json": audit_ref,
                "log_completeness_audit_markdown": markdown_ref,
            },
        }
        _write_bytes_atomic(temporary / "manifest.json", _canonical_json_bytes(manifest))
        os.replace(temporary, destination)
        return manifest
    except Exception:
        if temporary.is_symlink():
            temporary.unlink()
        elif temporary.exists():
            shutil.rmtree(temporary)
        raise


def _entries(report: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rounds = report.get("rounds")
    if isinstance(rounds, list):
        entries = []
        for index, item in enumerate(rounds, start=1):
            if not isinstance(item, Mapping):
                raise LogCompletenessAuditError("LOG_AUDIT_ROUND_INVALID")
            entries.append(item)
        return entries
    return [report]


def _audit_entry(entry: Mapping[str, Any], *, cas: ContentAddressedStore) -> dict[str, object]:
    proposal_id = entry.get("proposal_id")
    ordinal = entry.get("ordinal")
    blockers: list[dict[str, object]] = []
    for field in ("proposal_ref", "receipt_ref", "verdict_ref", "settlement_state"):
        if not entry.get(field):
            blockers.append(_blocker(proposal_id=proposal_id, ordinal=ordinal, surface="orchestrator", detail=f"missing {field}"))
    receipt_ref = entry.get("receipt_ref")
    if isinstance(receipt_ref, str):
        _require_cas_payload(
            receipt_ref,
            cas=cas,
            blockers=blockers,
            proposal_id=proposal_id,
            ordinal=ordinal,
            surface="receipt",
            detail="receipt_ref unreadable",
        )
    llm_log = entry.get("llm_call_log")
    if not isinstance(llm_log, Mapping):
        blockers.append(_blocker(proposal_id=proposal_id, ordinal=ordinal, surface="llm_call_log", detail="missing llm_call_log"))
        prompt_ref = entry.get("prompt_ref")
        if not prompt_ref:
            blockers.append(_blocker(proposal_id=proposal_id, ordinal=ordinal, surface="llm_call_log", detail="missing prompt_ref"))
        return _observation(entry, blockers=blockers, prompt_size_bytes=None, raw_response_count=None)
    if llm_log.get("artifact_type") != "wmloop-llm-call-log" or llm_log.get("state") != "ready":
        blockers.append(_blocker(proposal_id=proposal_id, ordinal=ordinal, surface="llm_call_log", detail="invalid llm_call_log header"))
    if llm_log.get("untruncated") is not True:
        blockers.append(_blocker(proposal_id=proposal_id, ordinal=ordinal, surface="llm_call_log", detail="llm_call_log not marked untruncated"))
    prompt_ref = llm_log.get("prompt_ref")
    if prompt_ref != entry.get("prompt_ref"):
        blockers.append(_blocker(proposal_id=proposal_id, ordinal=ordinal, surface="llm_call_log", detail="prompt_ref mismatch"))
    prompt_size = None
    if isinstance(prompt_ref, str):
        prompt_size = _require_cas_payload(
            prompt_ref,
            cas=cas,
            blockers=blockers,
            proposal_id=proposal_id,
            ordinal=ordinal,
            surface="llm_call_log",
            detail="prompt_ref unreadable",
        )
        expected_prompt_size = llm_log.get("prompt_size_bytes")
        if isinstance(expected_prompt_size, int) and prompt_size is not None and prompt_size != expected_prompt_size:
            blockers.append(_blocker(proposal_id=proposal_id, ordinal=ordinal, surface="llm_call_log", detail="prompt size mismatch"))
    else:
        blockers.append(_blocker(proposal_id=proposal_id, ordinal=ordinal, surface="llm_call_log", detail="prompt_ref invalid"))
    response_refs = llm_log.get("raw_response_refs")
    if not isinstance(response_refs, list) or not response_refs:
        blockers.append(_blocker(proposal_id=proposal_id, ordinal=ordinal, surface="llm_call_log", detail="raw_response_refs missing"))
        response_count = 0
    else:
        response_count = len(response_refs)
        for index, item in enumerate(response_refs, start=1):
            if not isinstance(item, Mapping):
                blockers.append(_blocker(proposal_id=proposal_id, ordinal=ordinal, surface="llm_call_log", detail=f"raw response {index} invalid"))
                continue
            ref = item.get("cas_ref")
            if not isinstance(ref, str):
                blockers.append(_blocker(proposal_id=proposal_id, ordinal=ordinal, surface="llm_call_log", detail=f"raw response {index} ref missing"))
                continue
            size = _require_cas_payload(
                ref,
                cas=cas,
                blockers=blockers,
                proposal_id=proposal_id,
                ordinal=ordinal,
                surface="llm_call_log",
                detail=f"raw response {index} unreadable",
            )
            expected_size = item.get("size_bytes")
            if isinstance(expected_size, int) and size is not None and size != expected_size:
                blockers.append(_blocker(proposal_id=proposal_id, ordinal=ordinal, surface="llm_call_log", detail=f"raw response {index} size mismatch"))
    expected_count = llm_log.get("raw_response_count")
    if expected_count != response_count:
        blockers.append(_blocker(proposal_id=proposal_id, ordinal=ordinal, surface="llm_call_log", detail="raw_response_count mismatch"))
    return _observation(entry, blockers=blockers, prompt_size_bytes=prompt_size, raw_response_count=response_count)


def _require_cas_payload(
    ref: str,
    *,
    cas: ContentAddressedStore,
    blockers: list[dict[str, object]],
    proposal_id: object,
    ordinal: object,
    surface: str,
    detail: str,
) -> int | None:
    try:
        payload = cas.read_bytes(ref)
    except Exception as exc:
        blockers.append(_blocker(proposal_id=proposal_id, ordinal=ordinal, surface=surface, detail=f"{detail}: {exc}"))
        return None
    return len(payload)


def _observation(
    entry: Mapping[str, Any],
    *,
    blockers: list[dict[str, object]],
    prompt_size_bytes: int | None,
    raw_response_count: int | None,
) -> dict[str, object]:
    return {
        "proposal_id": entry.get("proposal_id"),
        "ordinal": entry.get("ordinal"),
        "state": "ready" if not blockers else "blocked",
        "settlement_state": entry.get("settlement_state"),
        "receipt_ref": entry.get("receipt_ref"),
        "prompt_ref": entry.get("prompt_ref"),
        "prompt_size_bytes": prompt_size_bytes,
        "raw_response_count": raw_response_count,
        "blockers": blockers,
    }


def _blocker(*, proposal_id: object, ordinal: object, surface: str, detail: str) -> dict[str, object]:
    return {
        "proposal_id": proposal_id,
        "ordinal": ordinal,
        "surface": surface,
        "detail": detail,
    }


def _cas_root_from_report(report: Mapping[str, Any], *, report_path: Path) -> Path:
    raw = report.get("cas_root")
    if isinstance(raw, str) and raw:
        return Path(raw).resolve()
    return report_path.parent.resolve()


def _load_json_mapping(path: Path, error_code: str) -> tuple[Path, Mapping[str, Any]]:
    try:
        resolved = Path(path).resolve(strict=True)
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LogCompletenessAuditError(error_code) from exc
    if not isinstance(payload, Mapping):
        raise LogCompletenessAuditError(error_code)
    return resolved, payload


def _render_markdown(audit: Mapping[str, Any]) -> str:
    lines = [
        "# Log Completeness Audit",
        "",
        f"State: `{audit['state']}`",
        f"Source report: `{audit['source_report_path']}`",
        f"Entries: `{audit['ready_entry_count']}/{audit['entry_count']}` ready",
        f"Blockers: `{audit['blocker_count']}`",
        "",
        "| Entry | Proposal | State | Raw Responses | Blockers |",
        "|--:|:--|:--|--:|--:|",
    ]
    for index, item in enumerate(audit["observations"], start=1):
        lines.append(
            f"| {index} | {item.get('proposal_id')} | {item['state']} | "
            f"{item.get('raw_response_count')} | {len(item['blockers'])} |"
        )
    if audit["blockers"]:
        lines.extend(["", "## Blockers", ""])
        for blocker in audit["blockers"]:
            lines.append(f"- {blocker.get('proposal_id')}: {blocker.get('surface')} - {blocker.get('detail')}")
    return "\n".join(lines) + "\n"


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise LogCompletenessAuditError("LOG_AUDIT_OUTPUT_EXISTS")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="audit a generated orchestrator report for required logs")
    run.add_argument("--orchestrator-report", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--cas-root", type=Path)
    run.add_argument("--archive-db", type=Path)
    args = parser.parse_args(argv)
    if args.command == "run":
        manifest = run_log_completeness_audit(
            orchestrator_report=args.orchestrator_report,
            output_root=args.output_root,
            cas_root=args.cas_root,
            archive_db=args.archive_db,
        )
        print(json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
        return 0
    raise LogCompletenessAuditError("LOG_AUDIT_COMMAND_INVALID")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
