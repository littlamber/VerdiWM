"""Audit execution receipts for required CAS-backed evidence surfaces."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from wmloop.archive.store import ArchiveStore, ContentAddressedStore


class ReceiptCompletenessAuditError(RuntimeError):
    """A receipt-completeness audit input or output invariant failed."""


def run_receipt_completeness_audit(
    *,
    orchestrator_report: Path,
    output_root: Path,
    cas_root: Path | None = None,
    archive_db: Path | None = None,
) -> dict[str, object]:
    report_path, report = _load_json_mapping(orchestrator_report, "RECEIPT_AUDIT_REPORT_INVALID")
    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise ReceiptCompletenessAuditError("RECEIPT_AUDIT_OUTPUT_EXISTS")
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
    state = "ready" if observations and not blockers else "blocked"
    audit = {
        "schema_version": 1,
        "artifact_type": "wmloop-receipt-completeness-audit",
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
            "receipt_ref CAS payload",
            "candidate manifest and diff CAS payloads",
            "per-attempt json/stdout/stderr CAS payloads",
            "GPU sampling curve CAS payload for GPU-backed receipts",
        ],
    }
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    try:
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary.mkdir(mode=0o700)
        audit_bytes = _canonical_json_bytes(audit)
        markdown_bytes = _render_markdown(audit).encode("utf-8")
        _write_bytes_atomic(temporary / "receipt-completeness-audit.json", audit_bytes)
        _write_bytes_atomic(temporary / "receipt-completeness-audit.md", markdown_bytes)
        audit_ref = cas.put_bytes(audit_bytes, media_type="application/json").uri
        markdown_ref = cas.put_bytes(markdown_bytes, media_type="text/markdown").uri
        if archive is not None:
            archive.record_artifact_reference(audit_ref)
            archive.record_artifact_reference(markdown_ref)
        manifest = {
            "schema_version": 1,
            "artifact_type": "wmloop-receipt-completeness-audit-manifest",
            "state": state,
            "source_report_path": str(report_path),
            "source_artifact_type": report.get("artifact_type"),
            "entry_count": len(observations),
            "ready_entry_count": audit["ready_entry_count"],
            "blocker_count": len(blockers),
            "blockers": blockers,
            "report_path": str(destination / "receipt-completeness-audit.json"),
            "markdown_path": str(destination / "receipt-completeness-audit.md"),
            "cas_refs": {
                "receipt_completeness_audit_json": audit_ref,
                "receipt_completeness_audit_markdown": markdown_ref,
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
        return [_require_mapping(item, "RECEIPT_AUDIT_ROUND_INVALID") for item in rounds]
    return [report]


def _audit_entry(entry: Mapping[str, Any], *, cas: ContentAddressedStore) -> dict[str, object]:
    proposal_id = entry.get("proposal_id")
    ordinal = entry.get("ordinal")
    blockers: list[dict[str, object]] = []
    receipt = entry.get("receipt")
    if not isinstance(receipt, Mapping):
        blockers.append(_blocker(proposal_id=proposal_id, ordinal=ordinal, surface="receipt", detail="missing embedded receipt"))
        return _observation(entry, blockers=blockers, attempt_count=0, gpu_sample_count=0)
    for field in ("receipt_ref",):
        ref = entry.get(field) or receipt.get(field)
        if isinstance(ref, str):
            _require_cas_payload(ref, cas=cas, blockers=blockers, proposal_id=proposal_id, ordinal=ordinal, surface="receipt", detail=f"{field} unreadable")
        else:
            blockers.append(_blocker(proposal_id=proposal_id, ordinal=ordinal, surface="receipt", detail=f"missing {field}"))
    for field in ("candidate_manifest_ref", "candidate_diff_ref"):
        ref = receipt.get(field)
        if isinstance(ref, str):
            _require_cas_payload(ref, cas=cas, blockers=blockers, proposal_id=proposal_id, ordinal=ordinal, surface="candidate", detail=f"{field} unreadable")
        else:
            blockers.append(_blocker(proposal_id=proposal_id, ordinal=ordinal, surface="candidate", detail=f"missing {field}"))
    candidate = receipt.get("candidate")
    attempts = candidate.get("receipts") if isinstance(candidate, Mapping) else None
    attempt_refs = receipt.get("attempt_refs")
    if not isinstance(attempts, list) or not attempts:
        blockers.append(_blocker(proposal_id=proposal_id, ordinal=ordinal, surface="attempts", detail="missing candidate receipts"))
        attempt_count = 0
    elif not isinstance(attempt_refs, Mapping):
        blockers.append(_blocker(proposal_id=proposal_id, ordinal=ordinal, surface="attempts", detail="missing attempt_refs"))
        attempt_count = len(attempts)
    else:
        attempt_count = len(attempts)
        for attempt in attempts:
            if not isinstance(attempt, Mapping):
                blockers.append(_blocker(proposal_id=proposal_id, ordinal=ordinal, surface="attempts", detail="invalid attempt receipt"))
                continue
            attempt_ordinal = attempt.get("ordinal")
            if not isinstance(attempt_ordinal, int):
                blockers.append(_blocker(proposal_id=proposal_id, ordinal=ordinal, surface="attempts", detail="attempt ordinal invalid"))
                continue
            for suffix in ("json", "stdout", "stderr"):
                key = f"{attempt_ordinal:03d}_{suffix}"
                ref = attempt_refs.get(key)
                if not isinstance(ref, str):
                    blockers.append(_blocker(proposal_id=proposal_id, ordinal=ordinal, surface="attempts", detail=f"missing {key}"))
                    continue
                _require_cas_payload(ref, cas=cas, blockers=blockers, proposal_id=proposal_id, ordinal=ordinal, surface="attempts", detail=f"{key} unreadable")
            for field in ("label", "exit_code", "timed_out", "duration_seconds", "stdout_sha256", "stderr_sha256", "passed"):
                if field not in attempt:
                    blockers.append(_blocker(proposal_id=proposal_id, ordinal=ordinal, surface="attempts", detail=f"attempt missing {field}"))
    gpu_sample_count = _audit_gpu_sampling(entry=entry, receipt=receipt, cas=cas, blockers=blockers, proposal_id=proposal_id, ordinal=ordinal)
    return _observation(entry, blockers=blockers, attempt_count=attempt_count, gpu_sample_count=gpu_sample_count)


def _audit_gpu_sampling(
    *,
    entry: Mapping[str, Any],
    receipt: Mapping[str, Any],
    cas: ContentAddressedStore,
    blockers: list[dict[str, object]],
    proposal_id: object,
    ordinal: object,
) -> int:
    requires_gpu_sampling = (
        entry.get("run_training") is True
        or "train_steps" in entry
        or (isinstance(receipt.get("actual_gpu_hours"), (float, int)) and float(receipt.get("actual_gpu_hours", 0.0)) > 0.0)
    )
    sampling = receipt.get("gpu_sampling")
    if not isinstance(sampling, Mapping):
        if requires_gpu_sampling:
            blockers.append(_blocker(proposal_id=proposal_id, ordinal=ordinal, surface="gpu_sampling", detail="missing gpu_sampling"))
        return 0
    sample_count = sampling.get("sample_count")
    samples = sampling.get("samples")
    if not isinstance(sample_count, int) or not isinstance(samples, list) or sample_count != len(samples) or sample_count < 1:
        blockers.append(_blocker(proposal_id=proposal_id, ordinal=ordinal, surface="gpu_sampling", detail="invalid sample count"))
    ref = receipt.get("gpu_sampling_ref")
    if isinstance(ref, str):
        _require_cas_payload(ref, cas=cas, blockers=blockers, proposal_id=proposal_id, ordinal=ordinal, surface="gpu_sampling", detail="gpu_sampling_ref unreadable")
    else:
        blockers.append(_blocker(proposal_id=proposal_id, ordinal=ordinal, surface="gpu_sampling", detail="missing gpu_sampling_ref"))
    return int(sample_count) if isinstance(sample_count, int) else 0


def _observation(
    entry: Mapping[str, Any],
    *,
    blockers: list[dict[str, object]],
    attempt_count: int,
    gpu_sample_count: int,
) -> dict[str, object]:
    return {
        "proposal_id": entry.get("proposal_id"),
        "ordinal": entry.get("ordinal"),
        "state": "ready" if not blockers else "blocked",
        "receipt_ref": entry.get("receipt_ref"),
        "attempt_count": attempt_count,
        "gpu_sample_count": gpu_sample_count,
        "blockers": blockers,
    }


def _require_mapping(value: object, error_code: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReceiptCompletenessAuditError(error_code)
    return value


def _require_cas_payload(
    ref: str,
    *,
    cas: ContentAddressedStore,
    blockers: list[dict[str, object]],
    proposal_id: object,
    ordinal: object,
    surface: str,
    detail: str,
) -> None:
    try:
        cas.read_bytes(ref)
    except Exception as exc:
        blockers.append(_blocker(proposal_id=proposal_id, ordinal=ordinal, surface=surface, detail=f"{detail}: {exc}"))


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
        raise ReceiptCompletenessAuditError(error_code) from exc
    if not isinstance(payload, Mapping):
        raise ReceiptCompletenessAuditError(error_code)
    return resolved, payload


def _render_markdown(audit: Mapping[str, Any]) -> str:
    lines = [
        "# Receipt Completeness Audit",
        "",
        f"State: `{audit['state']}`",
        f"Source report: `{audit['source_report_path']}`",
        f"Entries: `{audit['ready_entry_count']}/{audit['entry_count']}` ready",
        f"Blockers: `{audit['blocker_count']}`",
        "",
        "| Entry | Proposal | State | Attempts | GPU Samples | Blockers |",
        "|--:|:--|:--|--:|--:|--:|",
    ]
    for index, item in enumerate(audit["observations"], start=1):
        lines.append(
            f"| {index} | {item.get('proposal_id')} | {item['state']} | {item['attempt_count']} | "
            f"{item['gpu_sample_count']} | {len(item['blockers'])} |"
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
        raise ReceiptCompletenessAuditError("RECEIPT_AUDIT_OUTPUT_EXISTS")
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
    run = commands.add_parser("run", help="audit a generated orchestrator report for receipt completeness")
    run.add_argument("--orchestrator-report", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--cas-root", type=Path)
    run.add_argument("--archive-db", type=Path)
    args = parser.parse_args(argv)
    if args.command == "run":
        manifest = run_receipt_completeness_audit(
            orchestrator_report=args.orchestrator_report,
            output_root=args.output_root,
            cas_root=args.cas_root,
            archive_db=args.archive_db,
        )
        print(json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
        return 0
    raise ReceiptCompletenessAuditError("RECEIPT_AUDIT_COMMAND_INVALID")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
