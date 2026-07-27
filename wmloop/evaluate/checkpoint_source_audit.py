"""Resolve M0 checkpoint source blockers without downloading model weights."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from wmloop.archive.store import ArchiveStore, ContentAddressedStore


class CheckpointSourceAuditError(RuntimeError):
    """Checkpoint source audit evidence could not be produced safely."""


UrlOpen = Callable[..., Any]


def generate_checkpoint_source_audit(
    *,
    checkpoint_step_audit_path: Path,
    output_root: Path,
    repository_id: str = "t1an/ACWM-Phys-checkpoints",
    expected_step: int = 100000,
    model_card_url: str = "https://huggingface.co/t1an/ACWM-Phys-checkpoints",
    remote_timeout_seconds: float = 15.0,
    skip_remote_probes: bool = False,
    external_metadata_manifest: Path | None = None,
    archive_db: Path | None = None,
    cas_root: Path | None = None,
    opener: UrlOpen = urllib.request.urlopen,
) -> dict[str, object]:
    """Write a durable source-resolution report for mismatched checkpoints."""

    if not isinstance(expected_step, int) or isinstance(expected_step, bool) or expected_step < 1:
        raise CheckpointSourceAuditError("CHECKPOINT_SOURCE_AUDIT_EXPECTED_STEP_INVALID")
    if remote_timeout_seconds < 0:
        raise CheckpointSourceAuditError("CHECKPOINT_SOURCE_AUDIT_TIMEOUT_INVALID")
    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise CheckpointSourceAuditError("CHECKPOINT_SOURCE_AUDIT_OUTPUT_EXISTS")
    step_audit = _load_json_mapping(checkpoint_step_audit_path, "CHECKPOINT_SOURCE_AUDIT_STEP_AUDIT_INVALID")
    external_metadata = _load_external_metadata_manifest(
        external_metadata_manifest,
        repository_id=repository_id,
    )
    records = _records_from_step_audit(step_audit, expected_step=expected_step)
    mismatch_records = [record for record in records if record["status"] != "pass"]
    source_records = [
        _source_record(
            record,
            repository_id=repository_id,
            remote_timeout_seconds=remote_timeout_seconds,
            skip_remote_probes=skip_remote_probes,
            external_metadata=external_metadata["records_by_path"],
            opener=opener,
        )
        for record in mismatch_records
    ]
    state = _overall_state(source_records, mismatch_count=len(mismatch_records))
    report = {
        "schema_version": 1,
        "artifact_type": "acwm-m0-checkpoint-source-audit",
        "state": state,
        "expected_step": expected_step,
        "repository_id": repository_id,
        "model_card_url": model_card_url,
        "model_card_claim": "Released ACWM-Phys checkpoints are expected to be trained for 100k steps.",
        "checkpoint_step_audit_path": str(Path(checkpoint_step_audit_path).resolve(strict=True)),
        "external_metadata_manifest": external_metadata["summary"],
        "mismatch_count": len(mismatch_records),
        "source_record_count": len(source_records),
        "replacement_candidate_count": sum(1 for record in source_records if record["source_status"] == "remote_hash_differs_from_local"),
        "remote_unreachable_count": sum(1 for record in source_records if record["source_status"] == "remote_unreachable"),
        "records": source_records,
        "next_actions": _next_actions(state),
        "limitations": [
            "This audit reads local metadata and remote HTTP metadata only; it does not download checkpoint bodies.",
            "External metadata manifests are comparison evidence only; matching external hashes do not prove a checkpoint step is correct.",
            "A remote hash difference is only a source candidate. It must be downloaded to a quarantine path and pass step audit before replacing active checkpoints.",
        ],
    }
    return _write_report_bundle(report=report, output_root=destination, archive_db=archive_db, cas_root=cas_root)


def _records_from_step_audit(step_audit: Mapping[str, Any], *, expected_step: int) -> list[dict[str, object]]:
    if step_audit.get("artifact_type") != "acwm-m0-checkpoint-step-audit" or step_audit.get("schema_version") != 1:
        raise CheckpointSourceAuditError("CHECKPOINT_SOURCE_AUDIT_STEP_AUDIT_INVALID")
    raw_records = step_audit.get("records")
    if not isinstance(raw_records, list):
        raise CheckpointSourceAuditError("CHECKPOINT_SOURCE_AUDIT_STEP_AUDIT_INVALID")
    records: list[dict[str, object]] = []
    for item in raw_records:
        if not isinstance(item, Mapping):
            raise CheckpointSourceAuditError("CHECKPOINT_SOURCE_AUDIT_STEP_AUDIT_INVALID")
        environment = _string_field(item, "environment")
        relative_path = _string_field(item, "checkpoint_relative_path")
        checkpoint_path = _string_field(item, "checkpoint_path")
        status = _string_field(item, "status")
        observed_step = _int_field(item, "observed_step")
        record_expected = _int_field(item, "expected_step")
        if record_expected != expected_step:
            raise CheckpointSourceAuditError("CHECKPOINT_SOURCE_AUDIT_EXPECTED_STEP_MISMATCH")
        hf_metadata = item.get("huggingface_download_metadata")
        if not isinstance(hf_metadata, Mapping):
            raise CheckpointSourceAuditError("CHECKPOINT_SOURCE_AUDIT_STEP_AUDIT_INVALID")
        records.append(
            {
                "environment": environment,
                "checkpoint_relative_path": relative_path,
                "checkpoint_path": checkpoint_path,
                "size_bytes": _int_field(item, "size_bytes"),
                "expected_step": record_expected,
                "observed_step": observed_step,
                "status": status,
                "huggingface_download_metadata": dict(hf_metadata),
            }
        )
    return records


def _source_record(
    record: Mapping[str, object],
    *,
    repository_id: str,
    remote_timeout_seconds: float,
    skip_remote_probes: bool,
    external_metadata: Mapping[str, Mapping[str, object]],
    opener: UrlOpen,
) -> dict[str, object]:
    relative_path = str(record["checkpoint_relative_path"])
    local_hash = _local_hf_hash(record)
    external_manifest_probe = _external_metadata_probe(
        relative_path=relative_path,
        external_metadata=external_metadata,
    )
    if skip_remote_probes:
        model_api_probe = {"status": "skipped", "reason": "skip_remote_probes"}
        file_head_probe = {"status": "skipped", "reason": "skip_remote_probes"}
    else:
        model_api_probe = _probe_model_api(
            repository_id=repository_id,
            relative_path=relative_path,
            timeout_seconds=remote_timeout_seconds,
            opener=opener,
        )
        file_head_probe = _probe_file_head(
            repository_id=repository_id,
            relative_path=relative_path,
            timeout_seconds=remote_timeout_seconds,
            opener=opener,
        )
    remote_hashes = _remote_hashes(model_api_probe, file_head_probe, external_manifest_probe)
    remote_success = (
        model_api_probe["status"] == "ok"
        or file_head_probe["status"] == "ok"
        or external_manifest_probe["status"] == "ok"
    )
    if remote_hashes and local_hash is not None and any(value != local_hash for value in remote_hashes):
        source_status = "remote_hash_differs_from_local"
    elif local_hash is not None and local_hash in remote_hashes:
        source_status = "remote_current_matches_local_mismatch"
    elif not remote_success:
        source_status = "remote_unreachable"
    else:
        source_status = "remote_inconclusive"
    return {
        "environment": record["environment"],
        "checkpoint_relative_path": relative_path,
        "checkpoint_path": record["checkpoint_path"],
        "size_bytes": record["size_bytes"],
        "expected_step": record["expected_step"],
        "observed_step": record["observed_step"],
        "local_hf_download_metadata": record["huggingface_download_metadata"],
        "local_hf_hash": local_hash,
        "source_status": source_status,
        "remote_probes": {
            "model_api": model_api_probe,
            "file_head": file_head_probe,
            "external_manifest": external_manifest_probe,
        },
    }


def _load_external_metadata_manifest(
    path: Path | None,
    *,
    repository_id: str,
) -> dict[str, object]:
    if path is None:
        return {"summary": {"state": "not_provided"}, "records_by_path": {}}
    resolved = Path(path).resolve(strict=True)
    payload = _load_json_mapping(resolved, "CHECKPOINT_SOURCE_AUDIT_EXTERNAL_METADATA_INVALID")
    if payload.get("artifact_type") != "acwm-hf-external-checkpoint-metadata" or payload.get("schema_version") != 1:
        raise CheckpointSourceAuditError("CHECKPOINT_SOURCE_AUDIT_EXTERNAL_METADATA_INVALID")
    records = payload.get("records")
    if not isinstance(records, list):
        raise CheckpointSourceAuditError("CHECKPOINT_SOURCE_AUDIT_EXTERNAL_METADATA_INVALID")
    records_by_path: dict[str, Mapping[str, object]] = {}
    for item in records:
        if not isinstance(item, Mapping):
            raise CheckpointSourceAuditError("CHECKPOINT_SOURCE_AUDIT_EXTERNAL_METADATA_INVALID")
        item_repository = item.get("repository_id")
        if item_repository != repository_id:
            raise CheckpointSourceAuditError("CHECKPOINT_SOURCE_AUDIT_EXTERNAL_METADATA_REPOSITORY_MISMATCH")
        relative_path = item.get("checkpoint_relative_path")
        lfs_sha256 = _normalize_hash(item.get("lfs_sha256"))
        if not isinstance(relative_path, str) or not relative_path or lfs_sha256 is None:
            raise CheckpointSourceAuditError("CHECKPOINT_SOURCE_AUDIT_EXTERNAL_METADATA_INVALID")
        normalized = dict(item)
        normalized["lfs_sha256"] = lfs_sha256
        records_by_path[relative_path] = normalized
    return {
        "summary": {
            "state": "provided",
            "path": str(resolved),
            "sha256": _sha256_file(resolved),
            "artifact_type": payload.get("artifact_type"),
            "observed_at_utc": payload.get("observed_at_utc"),
            "record_count": len(records_by_path),
            "source_urls": payload.get("source_urls", []),
        },
        "records_by_path": records_by_path,
    }


def _external_metadata_probe(
    *,
    relative_path: str,
    external_metadata: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    if not external_metadata:
        return {"status": "not_provided"}
    record = external_metadata.get(relative_path)
    if record is None:
        return {"status": "missing", "reason": "relative_path_not_in_external_metadata"}
    lfs_sha256 = _normalize_hash(record.get("lfs_sha256"))
    if lfs_sha256 is None:
        return {"status": "invalid", "reason": "invalid_lfs_sha256"}
    sibling: dict[str, object] = {
        "rfilename": relative_path,
        "lfs": {"sha256": lfs_sha256},
    }
    size_bytes = record.get("size_bytes")
    if isinstance(size_bytes, int) and not isinstance(size_bytes, bool) and size_bytes >= 0:
        sibling["size"] = size_bytes
        sibling["lfs"]["size"] = size_bytes  # type: ignore[index]
    return {
        "status": "ok",
        "source": "external_metadata_manifest",
        "url": record.get("source_url"),
        "observed_at_utc": record.get("observed_at_utc"),
        "repository_sha": record.get("repository_sha"),
        "commit": record.get("commit"),
        "xet_hash": record.get("xet_hash"),
        "sibling": sibling,
    }


def _probe_model_api(
    *,
    repository_id: str,
    relative_path: str,
    timeout_seconds: float,
    opener: UrlOpen,
) -> dict[str, object]:
    url = f"https://huggingface.co/api/models/{repository_id}"
    try:
        with opener(url, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        return _probe_error(url, exc)
    if not isinstance(payload, Mapping):
        return {"status": "invalid", "url": url, "reason": "non_mapping_payload"}
    siblings = payload.get("siblings")
    sibling = None
    if isinstance(siblings, list):
        for item in siblings:
            if isinstance(item, Mapping) and item.get("rfilename") == relative_path:
                sibling = item
                break
    return {
        "status": "ok",
        "url": url,
        "repository_sha": payload.get("sha"),
        "last_modified": payload.get("lastModified"),
        "sibling": _summarize_sibling(sibling),
    }


def _probe_file_head(
    *,
    repository_id: str,
    relative_path: str,
    timeout_seconds: float,
    opener: UrlOpen,
) -> dict[str, object]:
    url = f"https://huggingface.co/{repository_id}/resolve/main/{relative_path}"
    request = urllib.request.Request(url, method="HEAD")
    try:
        with opener(request, timeout=timeout_seconds) as response:
            headers = {key.lower(): value for key, value in response.headers.items()}
            status_code = getattr(response, "status", None)
    except Exception as exc:
        return _probe_error(url, exc)
    selected_headers = {
        key: headers[key]
        for key in (
            "etag",
            "x-linked-etag",
            "x-repo-commit",
            "x-linked-size",
            "content-length",
            "location",
        )
        if key in headers
    }
    return {"status": "ok", "url": url, "status_code": status_code, "headers": selected_headers}


def _probe_error(url: str, exc: Exception) -> dict[str, object]:
    reason = getattr(exc, "reason", None)
    return {
        "status": "error",
        "url": url,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "reason": str(reason) if reason is not None else None,
    }


def _summarize_sibling(sibling: Mapping[str, Any] | None) -> dict[str, object] | None:
    if sibling is None:
        return None
    summary: dict[str, object] = {"rfilename": sibling.get("rfilename")}
    if "size" in sibling:
        summary["size"] = sibling["size"]
    lfs = sibling.get("lfs")
    if isinstance(lfs, Mapping):
        summary["lfs"] = {key: lfs.get(key) for key in ("sha256", "size", "pointerSize") if key in lfs}
    return summary


def _local_hf_hash(record: Mapping[str, object]) -> str | None:
    metadata = record.get("huggingface_download_metadata")
    if not isinstance(metadata, Mapping):
        return None
    value = metadata.get("etag_or_lfs_sha256")
    return _normalize_hash(value) if isinstance(value, str) else None


def _remote_hashes(*probes: Mapping[str, object]) -> set[str]:
    values: set[str] = set()
    for probe in probes:
        if probe.get("status") != "ok":
            continue
        sibling = probe.get("sibling")
        if isinstance(sibling, Mapping):
            lfs = sibling.get("lfs")
            if isinstance(lfs, Mapping):
                value = lfs.get("sha256")
                normalized = _normalize_hash(value)
                if normalized is not None:
                    values.add(normalized)
        headers = probe.get("headers")
        if isinstance(headers, Mapping):
            for key in ("x-linked-etag", "etag"):
                normalized = _normalize_hash(headers.get(key))
                if normalized is not None:
                    values.add(normalized)
    return values


def _normalize_hash(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip().strip('"')
    if len(stripped) == 64 and all(character in "0123456789abcdef" for character in stripped.lower()):
        return stripped.lower()
    return None


def _overall_state(records: Sequence[Mapping[str, object]], *, mismatch_count: int) -> str:
    if mismatch_count == 0:
        return "ready"
    statuses = {str(record["source_status"]) for record in records}
    if "remote_hash_differs_from_local" in statuses:
        return "replacement_candidate_found"
    if "remote_current_matches_local_mismatch" in statuses:
        return "remote_current_mismatch"
    if statuses == {"remote_unreachable"}:
        return "source_unresolved"
    return "source_inconclusive"


def _next_actions(state: str) -> list[str]:
    if state == "ready":
        return ["No checkpoint source action is required."]
    if state == "replacement_candidate_found":
        return [
            "Download differing remote checkpoint files into a quarantine path, not over active checkpoints.",
            "Run checkpoint step audit on the quarantine path and replace active files only after observed_step equals expected_step.",
            "Rerun M0 baseline launch, checkpoint audit, launch guard smoke, reproduction report, and phase gate.",
        ]
    if state == "remote_current_mismatch":
        return [
            "Do not redownload the same current remote file as a fix; remote metadata matches the locally mismatched checkpoint.",
            "Find a verified alternate revision/source or contact the checkpoint publisher, then rerun checkpoint step audit before replacement.",
        ]
    return [
        "Resolve network/source access or provide a verified alternate 100k-step checkpoint candidate.",
        "Do not overwrite active checkpoints until the candidate passes checkpoint step audit.",
        "Keep M0 T0.3 and M4 launch blocked while the source remains unresolved.",
    ]


def _render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# M0 Checkpoint Source Audit",
        "",
        f"State: `{report['state']}`",
        f"Expected step: `{report['expected_step']}`",
        f"Repository: `{report['repository_id']}`",
        f"Mismatch count: `{report['mismatch_count']}`",
        f"Replacement candidates: `{report['replacement_candidate_count']}`",
        f"Remote unreachable: `{report['remote_unreachable_count']}`",
        f"External metadata: `{report['external_metadata_manifest']['state']}`",
        "",
        "| Environment | Observed | Expected | Local HF Hash | Source Status |",
        "|:--|--:|--:|:--|:--|",
    ]
    for record in report["records"]:
        lines.append(
            f"| {record['environment']} | {record['observed_step']} | {record['expected_step']} | "
            f"`{record['local_hf_hash']}` | `{record['source_status']}` |"
        )
    lines.extend(["", "## Next Actions", ""])
    for item in report["next_actions"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Limitations", ""])
    for item in report["limitations"]:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def _write_report_bundle(
    *,
    report: Mapping[str, object],
    output_root: Path,
    archive_db: Path | None,
    cas_root: Path | None,
) -> dict[str, object]:
    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise CheckpointSourceAuditError("CHECKPOINT_SOURCE_AUDIT_OUTPUT_EXISTS")
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    report_bytes = _canonical_json_bytes(report)
    markdown_bytes = _render_markdown(report).encode("utf-8")
    try:
        temporary.mkdir(mode=0o700)
        _write_bytes_atomic(temporary / "checkpoint-source-audit.json", report_bytes)
        _write_bytes_atomic(temporary / "checkpoint-source-audit.md", markdown_bytes)
        archive = ArchiveStore(archive_db) if archive_db is not None else None
        cas_refs: dict[str, str] = {}
        if archive is not None or cas_root is not None:
            root = cas_root if cas_root is not None else Path(archive_db).resolve().parent  # type: ignore[union-attr]
            cas = ContentAddressedStore(Path(root))
            for name, payload, media_type in (
                ("checkpoint_source_audit_json", report_bytes, "application/json"),
                ("checkpoint_source_audit_markdown", markdown_bytes, "text/markdown"),
            ):
                ref = cas.put_bytes(payload, media_type=media_type).uri
                cas_refs[name] = ref
                if archive is not None:
                    archive.record_artifact_reference(ref)
        manifest = {
            "schema_version": 1,
            "artifact_type": "acwm-m0-checkpoint-source-audit-manifest",
            "state": report["state"],
            "expected_step": report["expected_step"],
            "mismatch_count": report["mismatch_count"],
            "replacement_candidate_count": report["replacement_candidate_count"],
            "remote_unreachable_count": report["remote_unreachable_count"],
            "report_path": str(destination / "checkpoint-source-audit.json"),
            "markdown_path": str(destination / "checkpoint-source-audit.md"),
            "cas_refs": cas_refs,
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


def _load_json_mapping(path: Path, code: str) -> Mapping[str, object]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointSourceAuditError(code) from exc
    if not isinstance(payload, Mapping):
        raise CheckpointSourceAuditError(code)
    return payload


def _sha256_file(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _string_field(payload: Mapping[str, Any], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value:
        raise CheckpointSourceAuditError("CHECKPOINT_SOURCE_AUDIT_STEP_AUDIT_INVALID")
    return value


def _int_field(payload: Mapping[str, Any], name: str) -> int:
    value = payload.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise CheckpointSourceAuditError("CHECKPOINT_SOURCE_AUDIT_STEP_AUDIT_INVALID")
    return value


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
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
    run = commands.add_parser("run", help="audit sources for mismatched M0 checkpoints")
    run.add_argument("--checkpoint-step-audit", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--repository-id", default="t1an/ACWM-Phys-checkpoints")
    run.add_argument("--expected-step", type=int, default=100000)
    run.add_argument("--model-card-url", default="https://huggingface.co/t1an/ACWM-Phys-checkpoints")
    run.add_argument("--remote-timeout-seconds", type=float, default=15.0)
    run.add_argument("--skip-remote-probes", action="store_true")
    run.add_argument("--external-metadata-manifest", type=Path)
    run.add_argument("--archive-db", type=Path)
    run.add_argument("--cas-root", type=Path)
    args = parser.parse_args(argv)
    if args.command == "run":
        manifest = generate_checkpoint_source_audit(
            checkpoint_step_audit_path=args.checkpoint_step_audit,
            output_root=args.output_root,
            repository_id=args.repository_id,
            expected_step=args.expected_step,
            model_card_url=args.model_card_url,
            remote_timeout_seconds=args.remote_timeout_seconds,
            skip_remote_probes=args.skip_remote_probes,
            external_metadata_manifest=args.external_metadata_manifest,
            archive_db=args.archive_db,
            cas_root=args.cas_root,
        )
        print(json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
