"""Scan remote checkpoint revisions for a quarantine-only replacement candidate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import shutil
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from wmloop.archive.store import ArchiveStore, ContentAddressedStore


class CheckpointRevisionWatchError(RuntimeError):
    """Checkpoint revision watch evidence could not be produced safely."""


UrlOpen = Callable[..., Any]


def generate_checkpoint_revision_watch(
    *,
    checkpoint_step_audit_path: Path,
    environment: str,
    output_root: Path,
    repository_id: str = "t1an/ACWM-Phys-checkpoints",
    expected_step: int = 100000,
    revisions: Sequence[str] = (),
    max_discovered_revisions: int = 32,
    remote_timeout_seconds: float = 15.0,
    skip_ref_probe: bool = False,
    archive_db: Path | None = None,
    cas_root: Path | None = None,
    opener: UrlOpen = urllib.request.urlopen,
) -> dict[str, object]:
    """Write a read-only HF revision scan for one checkpoint blocker."""

    if not isinstance(expected_step, int) or isinstance(expected_step, bool) or expected_step < 1:
        raise CheckpointRevisionWatchError("CHECKPOINT_REVISION_WATCH_EXPECTED_STEP_INVALID")
    if not isinstance(max_discovered_revisions, int) or isinstance(max_discovered_revisions, bool) or max_discovered_revisions < 0:
        raise CheckpointRevisionWatchError("CHECKPOINT_REVISION_WATCH_MAX_DISCOVERED_REVISIONS_INVALID")
    if remote_timeout_seconds < 0:
        raise CheckpointRevisionWatchError("CHECKPOINT_REVISION_WATCH_TIMEOUT_INVALID")
    if not isinstance(environment, str) or not environment:
        raise CheckpointRevisionWatchError("CHECKPOINT_REVISION_WATCH_ENVIRONMENT_INVALID")
    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise CheckpointRevisionWatchError("CHECKPOINT_REVISION_WATCH_OUTPUT_EXISTS")

    step_audit = _load_step_audit(checkpoint_step_audit_path, expected_step=expected_step)
    record = _select_environment_record(step_audit, environment)
    relative_path = str(record["checkpoint_relative_path"])
    local_hash = _local_hf_hash(record)
    observed_step = int(record["observed_step"])
    local_status = str(record["status"])
    explicit_revisions = _normalized_revision_inputs(revisions)
    if skip_ref_probe:
        refs_probe = {"status": "skipped", "reason": "skip_ref_probe", "url": None}
        discovered = []
    else:
        refs_probe = _probe_refs_api(
            repository_id=repository_id,
            timeout_seconds=remote_timeout_seconds,
            opener=opener,
        )
        discovered = _revision_targets_from_refs(refs_probe, limit=max_discovered_revisions)
    revision_targets = _dedupe_revision_targets(
        [
            *({"revision": revision, "source": "explicit", "target_commit": None} for revision in explicit_revisions),
            *discovered,
        ]
    )
    records = [
        _revision_record(
            repository_id=repository_id,
            relative_path=relative_path,
            local_hash=local_hash,
            target=target,
            timeout_seconds=remote_timeout_seconds,
            opener=opener,
        )
        for target in revision_targets
    ]
    candidate_records = [item for item in records if item["candidate_available"] is True]
    state = _watch_state(
        local_status=local_status,
        observed_step=observed_step,
        expected_step=expected_step,
        local_hash=local_hash,
        refs_probe=refs_probe,
        revision_targets=revision_targets,
        records=records,
    )
    report = {
        "schema_version": 1,
        "artifact_type": "acwm-m0-checkpoint-revision-watch",
        "state": state,
        "observed_at_utc": _utc_now(),
        "repository_id": repository_id,
        "environment": environment,
        "checkpoint_relative_path": relative_path,
        "checkpoint_path": record["checkpoint_path"],
        "expected_step": expected_step,
        "observed_step": observed_step,
        "local_status": local_status,
        "local_hf_hash": local_hash,
        "explicit_revisions": explicit_revisions,
        "max_discovered_revisions": max_discovered_revisions,
        "refs_probe": refs_probe,
        "scanned_revision_count": len(records),
        "candidate_revision_count": len(candidate_records),
        "candidate_available": bool(candidate_records),
        "downloaded_checkpoint_bytes": False,
        "active_checkpoint_mutated": False,
        "m4_launch_allowed": False,
        "formal_training_allowed": False,
        "revision_records": records,
        "candidate_revisions": candidate_records,
        "next_actions": _next_actions(
            state=state,
            environment=environment,
            relative_path=relative_path,
            candidate_records=candidate_records,
        ),
        "limitations": [
            "This watch reads Hugging Face refs/model metadata and file HEAD metadata only; it never downloads checkpoint bodies.",
            "A differing revision hash is only a replacement candidate. It must be downloaded into quarantine and pass checkpoint_quarantine validate before install.",
            "Revision metadata cannot prove checkpoint training step. Only local quarantine validation with the ACWM runtime can prove observed_step.",
            "This report never authorizes M4 training; a regenerated strict phase gate must report m4_launch_allowed=true.",
        ],
    }
    return _write_report_bundle(report=report, output_root=destination, archive_db=archive_db, cas_root=cas_root)


def _load_step_audit(path: Path, *, expected_step: int) -> Mapping[str, Any]:
    payload = _load_json_mapping(path, "CHECKPOINT_REVISION_WATCH_STEP_AUDIT_INVALID")
    if payload.get("schema_version") != 1 or payload.get("artifact_type") != "acwm-m0-checkpoint-step-audit":
        raise CheckpointRevisionWatchError("CHECKPOINT_REVISION_WATCH_STEP_AUDIT_INVALID")
    if payload.get("expected_step") != expected_step:
        raise CheckpointRevisionWatchError("CHECKPOINT_REVISION_WATCH_EXPECTED_STEP_MISMATCH")
    if not isinstance(payload.get("records"), list):
        raise CheckpointRevisionWatchError("CHECKPOINT_REVISION_WATCH_STEP_AUDIT_INVALID")
    return payload


def _select_environment_record(step_audit: Mapping[str, Any], environment: str) -> Mapping[str, Any]:
    records = step_audit.get("records")
    assert isinstance(records, list)
    for item in records:
        if not isinstance(item, Mapping):
            raise CheckpointRevisionWatchError("CHECKPOINT_REVISION_WATCH_STEP_AUDIT_INVALID")
        if item.get("environment") == environment:
            _validate_record(item)
            return item
    raise CheckpointRevisionWatchError("CHECKPOINT_REVISION_WATCH_ENVIRONMENT_NOT_FOUND")


def _validate_record(record: Mapping[str, Any]) -> None:
    for field in ("environment", "checkpoint_relative_path", "checkpoint_path", "status"):
        if not isinstance(record.get(field), str) or not record.get(field):
            raise CheckpointRevisionWatchError("CHECKPOINT_REVISION_WATCH_STEP_AUDIT_INVALID")
    for field in ("expected_step", "observed_step"):
        if not isinstance(record.get(field), int) or isinstance(record.get(field), bool) or record.get(field) < 0:
            raise CheckpointRevisionWatchError("CHECKPOINT_REVISION_WATCH_STEP_AUDIT_INVALID")
    if not isinstance(record.get("huggingface_download_metadata"), Mapping):
        raise CheckpointRevisionWatchError("CHECKPOINT_REVISION_WATCH_STEP_AUDIT_INVALID")


def _normalized_revision_inputs(revisions: Sequence[str]) -> list[str]:
    normalized: list[str] = []
    for revision in revisions:
        if not isinstance(revision, str) or not revision.strip():
            raise CheckpointRevisionWatchError("CHECKPOINT_REVISION_WATCH_REVISION_INVALID")
        value = revision.strip()
        if value not in normalized:
            normalized.append(value)
    return normalized


def _probe_refs_api(
    *,
    repository_id: str,
    timeout_seconds: float,
    opener: UrlOpen,
) -> dict[str, object]:
    url = f"https://huggingface.co/api/models/{repository_id}/refs"
    try:
        with opener(url, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        return _probe_error(url, exc)
    if not isinstance(payload, Mapping):
        return {"status": "invalid", "url": url, "reason": "non_mapping_payload"}
    return {
        "status": "ok",
        "url": url,
        "branches": _summarize_refs(payload.get("branches")),
        "tags": _summarize_refs(payload.get("tags")),
        "converts": _summarize_refs(payload.get("converts")),
    }


def _summarize_refs(raw: object) -> list[dict[str, object]]:
    if not isinstance(raw, list):
        return []
    result: list[dict[str, object]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        name = item.get("name")
        if not isinstance(name, str) or not name:
            continue
        result.append(
            {
                "name": name,
                "ref": item.get("ref"),
                "target_commit": item.get("targetCommit") or item.get("target_commit"),
            }
        )
    return result


def _revision_targets_from_refs(refs_probe: Mapping[str, object], *, limit: int) -> list[dict[str, object]]:
    if refs_probe.get("status") != "ok" or limit == 0:
        return []
    targets: list[dict[str, object]] = []
    for source_key in ("branches", "tags", "converts"):
        refs = refs_probe.get(source_key)
        if not isinstance(refs, list):
            continue
        for item in refs:
            if not isinstance(item, Mapping):
                continue
            name = item.get("name")
            if isinstance(name, str) and name:
                targets.append(
                    {
                        "revision": name,
                        "source": source_key[:-1],
                        "target_commit": item.get("target_commit"),
                    }
                )
                if len(targets) >= limit:
                    return targets
    return targets


def _dedupe_revision_targets(targets: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    seen: set[str] = set()
    for target in targets:
        revision = target.get("revision")
        if not isinstance(revision, str) or not revision:
            continue
        if revision in seen:
            continue
        seen.add(revision)
        result.append(
            {
                "revision": revision,
                "source": target.get("source") if isinstance(target.get("source"), str) else "unknown",
                "target_commit": target.get("target_commit"),
            }
        )
    return result


def _revision_record(
    *,
    repository_id: str,
    relative_path: str,
    local_hash: str | None,
    target: Mapping[str, object],
    timeout_seconds: float,
    opener: UrlOpen,
) -> dict[str, object]:
    revision = str(target["revision"])
    model_api_probe = _probe_model_api_revision(
        repository_id=repository_id,
        revision=revision,
        relative_path=relative_path,
        timeout_seconds=timeout_seconds,
        opener=opener,
    )
    file_head_probe = _probe_file_head_revision(
        repository_id=repository_id,
        revision=revision,
        relative_path=relative_path,
        timeout_seconds=timeout_seconds,
        opener=opener,
    )
    remote_hashes = sorted(_remote_hashes(model_api_probe, file_head_probe))
    if local_hash is not None and any(item != local_hash for item in remote_hashes):
        source_status = "remote_hash_differs_from_local"
    elif local_hash is not None and local_hash in set(remote_hashes):
        source_status = "remote_matches_local_mismatch"
    elif model_api_probe["status"] == "ok" or file_head_probe["status"] == "ok":
        source_status = "remote_inconclusive"
    else:
        source_status = "remote_unreachable"
    return {
        "revision": revision,
        "source": target.get("source"),
        "target_commit": target.get("target_commit"),
        "remote_hashes": remote_hashes,
        "candidate_available": source_status == "remote_hash_differs_from_local",
        "source_status": source_status,
        "remote_probes": {
            "model_api": model_api_probe,
            "file_head": file_head_probe,
        },
    }


def _probe_model_api_revision(
    *,
    repository_id: str,
    revision: str,
    relative_path: str,
    timeout_seconds: float,
    opener: UrlOpen,
) -> dict[str, object]:
    url = f"https://huggingface.co/api/models/{repository_id}/revision/{_quote_revision(revision)}"
    try:
        with opener(url, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        return _probe_error(url, exc)
    if not isinstance(payload, Mapping):
        return {"status": "invalid", "url": url, "reason": "non_mapping_payload"}
    sibling = None
    siblings = payload.get("siblings")
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


def _probe_file_head_revision(
    *,
    repository_id: str,
    revision: str,
    relative_path: str,
    timeout_seconds: float,
    opener: UrlOpen,
) -> dict[str, object]:
    url = f"https://huggingface.co/{repository_id}/resolve/{_quote_revision(revision)}/{urllib.parse.quote(relative_path, safe='/')}"
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


def _local_hf_hash(record: Mapping[str, Any]) -> str | None:
    metadata = record.get("huggingface_download_metadata")
    if not isinstance(metadata, Mapping):
        return None
    return _normalize_hash(metadata.get("etag_or_lfs_sha256"))


def _remote_hashes(*probes: Mapping[str, object]) -> set[str]:
    values: set[str] = set()
    for probe in probes:
        if probe.get("status") != "ok":
            continue
        sibling = probe.get("sibling")
        if isinstance(sibling, Mapping):
            lfs = sibling.get("lfs")
            if isinstance(lfs, Mapping):
                normalized = _normalize_hash(lfs.get("sha256"))
                if normalized is not None:
                    values.add(normalized)
        headers = probe.get("headers")
        if isinstance(headers, Mapping):
            for key in ("x-linked-etag", "etag"):
                normalized = _normalize_hash(headers.get(key))
                if normalized is not None:
                    values.add(normalized)
    return values


def _watch_state(
    *,
    local_status: str,
    observed_step: int,
    expected_step: int,
    local_hash: str | None,
    refs_probe: Mapping[str, object],
    revision_targets: Sequence[Mapping[str, object]],
    records: Sequence[Mapping[str, object]],
) -> str:
    if local_status == "pass" and observed_step == expected_step:
        return "no_checkpoint_blocker"
    if local_hash is None:
        return "local_hash_unavailable"
    if any(item.get("candidate_available") is True for item in records):
        return "revision_candidate_available"
    if not revision_targets:
        if refs_probe.get("status") == "skipped":
            return "revision_probe_skipped"
        if refs_probe.get("status") == "error":
            return "revision_probe_unreachable"
        if refs_probe.get("status") == "ok":
            return "no_revision_refs_found"
        return "revision_probe_inconclusive"
    statuses = {str(item.get("source_status")) for item in records}
    if statuses == {"remote_unreachable"}:
        return "revision_probe_unreachable"
    if statuses and statuses <= {"remote_matches_local_mismatch"}:
        return "no_revision_candidate_found"
    if "remote_matches_local_mismatch" in statuses:
        return "no_revision_candidate_found"
    return "revision_probe_inconclusive"


def _next_actions(
    *,
    state: str,
    environment: str,
    relative_path: str,
    candidate_records: Sequence[Mapping[str, object]],
) -> list[str]:
    if state == "no_checkpoint_blocker":
        return ["No checkpoint revision watch action is required for this environment."]
    if state == "revision_candidate_available":
        revisions = ", ".join(str(item.get("revision")) for item in candidate_records)
        return [
            f"Download only the differing revision(s) {revisions} into results/quarantine/checkpoints/{environment}/latest.pt, not over the active checkpoint.",
            "Run wmloop.evaluate.checkpoint_quarantine validate on the quarantine file before any install.",
            "Install only after a ready_for_manual_install quarantine report and explicit human confirmation.",
        ]
    if state in {"no_revision_candidate_found", "no_revision_refs_found"}:
        return [
            f"No scanned revision produced a differing hash for {relative_path}.",
            "Continue looking for a publisher-fixed or externally verified 100k-step candidate outside the current refs.",
            "Do not redownload main as the checkpoint fix.",
        ]
    if state == "revision_probe_unreachable":
        return [
            "Resolve Hugging Face refs/revision metadata access and rerun this watch.",
            "Do not overwrite active checkpoints while revision status is unknown.",
        ]
    return [
        "Treat historical revision source resolution as inconclusive.",
        "Provide explicit revision IDs to --revision or a verified external candidate under quarantine.",
    ]


def _render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# M0 Checkpoint Revision Watch",
        "",
        f"State: `{report['state']}`",
        f"Observed at: `{report['observed_at_utc']}`",
        f"Repository: `{report['repository_id']}`",
        f"Environment: `{report['environment']}`",
        f"Expected step: `{report['expected_step']}`",
        f"Observed step: `{report['observed_step']}`",
        f"Scanned revisions: `{report['scanned_revision_count']}`",
        f"Candidate revisions: `{report['candidate_revision_count']}`",
        f"Downloaded checkpoint bytes: `{report['downloaded_checkpoint_bytes']}`",
        f"Active checkpoint mutated: `{report['active_checkpoint_mutated']}`",
        f"M4 launch allowed: `{report['m4_launch_allowed']}`",
        "",
        "## Revision Records",
        "",
        "| Revision | Source | Target Commit | Status | Candidate | Remote Hashes |",
        "|:--|:--|:--|:--|:--|:--|",
    ]
    for record in report["revision_records"]:
        lines.append(
            f"| `{record['revision']}` | `{record['source']}` | `{record['target_commit']}` | "
            f"`{record['source_status']}` | `{record['candidate_available']}` | "
            f"`{', '.join(record['remote_hashes']) or 'none'}` |"
        )
    lines.extend(["", "## Next Actions", ""])
    lines.extend(f"- {item}" for item in report["next_actions"])
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in report["limitations"])
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
        raise CheckpointRevisionWatchError("CHECKPOINT_REVISION_WATCH_OUTPUT_EXISTS")
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    report_bytes = _canonical_json_bytes(report)
    markdown_bytes = _render_markdown(report).encode("utf-8")
    try:
        temporary.mkdir(mode=0o700)
        _write_bytes_atomic(temporary / "checkpoint-revision-watch.json", report_bytes)
        _write_bytes_atomic(temporary / "checkpoint-revision-watch.md", markdown_bytes)
        archive = ArchiveStore(archive_db) if archive_db is not None else None
        cas_refs: dict[str, str] = {}
        if archive is not None or cas_root is not None:
            root = cas_root if cas_root is not None else Path(archive_db).resolve().parent  # type: ignore[union-attr]
            cas = ContentAddressedStore(Path(root))
            for name, payload, media_type in (
                ("checkpoint_revision_watch_json", report_bytes, "application/json"),
                ("checkpoint_revision_watch_markdown", markdown_bytes, "text/markdown"),
            ):
                ref = cas.put_bytes(payload, media_type=media_type).uri
                cas_refs[name] = ref
                if archive is not None:
                    archive.record_artifact_reference(ref)
        manifest = {
            "schema_version": 1,
            "artifact_type": "acwm-m0-checkpoint-revision-watch-manifest",
            "state": report["state"],
            "observed_at_utc": report["observed_at_utc"],
            "repository_id": report["repository_id"],
            "environment": report["environment"],
            "checkpoint_relative_path": report["checkpoint_relative_path"],
            "expected_step": report["expected_step"],
            "observed_step": report["observed_step"],
            "scanned_revision_count": report["scanned_revision_count"],
            "candidate_revision_count": report["candidate_revision_count"],
            "candidate_available": report["candidate_available"],
            "downloaded_checkpoint_bytes": report["downloaded_checkpoint_bytes"],
            "active_checkpoint_mutated": report["active_checkpoint_mutated"],
            "m4_launch_allowed": report["m4_launch_allowed"],
            "formal_training_allowed": report["formal_training_allowed"],
            "report_path": str(destination / "checkpoint-revision-watch.json"),
            "markdown_path": str(destination / "checkpoint-revision-watch.md"),
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
        raise CheckpointRevisionWatchError(code) from exc
    if not isinstance(payload, Mapping):
        raise CheckpointRevisionWatchError(code)
    return payload


def _quote_revision(revision: str) -> str:
    return urllib.parse.quote(revision, safe="")


def _normalize_hash(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip().strip('"')
    if len(stripped) == 64 and all(character in "0123456789abcdef" for character in stripped.lower()):
        return stripped.lower()
    return None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


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
    run = commands.add_parser("run", help="scan remote checkpoint refs/revisions for one environment")
    run.add_argument("--checkpoint-step-audit", type=Path, required=True)
    run.add_argument("--environment", required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--repository-id", default="t1an/ACWM-Phys-checkpoints")
    run.add_argument("--expected-step", type=int, default=100000)
    run.add_argument("--revision", action="append", default=[])
    run.add_argument("--max-discovered-revisions", type=int, default=32)
    run.add_argument("--remote-timeout-seconds", type=float, default=15.0)
    run.add_argument("--skip-ref-probe", action="store_true")
    run.add_argument("--archive-db", type=Path)
    run.add_argument("--cas-root", type=Path)
    args = parser.parse_args(argv)
    if args.command == "run":
        manifest = generate_checkpoint_revision_watch(
            checkpoint_step_audit_path=args.checkpoint_step_audit,
            environment=args.environment,
            output_root=args.output_root,
            repository_id=args.repository_id,
            expected_step=args.expected_step,
            revisions=args.revision,
            max_discovered_revisions=args.max_discovered_revisions,
            remote_timeout_seconds=args.remote_timeout_seconds,
            skip_ref_probe=args.skip_ref_probe,
            archive_db=args.archive_db,
            cas_root=args.cas_root,
        )
        print(json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
