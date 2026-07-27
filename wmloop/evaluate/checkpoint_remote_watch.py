"""Watch remote checkpoint metadata for a replacement candidate without downloading weights."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import shutil
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from wmloop.archive.store import ArchiveStore, ContentAddressedStore


class CheckpointRemoteWatchError(RuntimeError):
    """Checkpoint remote watch evidence could not be produced safely."""


UrlOpen = Callable[..., Any]


def generate_checkpoint_remote_watch(
    *,
    checkpoint_step_audit_path: Path,
    environment: str,
    output_root: Path,
    repository_id: str = "t1an/ACWM-Phys-checkpoints",
    expected_step: int = 100000,
    remote_timeout_seconds: float = 15.0,
    skip_remote_probes: bool = False,
    archive_db: Path | None = None,
    cas_root: Path | None = None,
    opener: UrlOpen = urllib.request.urlopen,
) -> dict[str, object]:
    """Write a durable remote-metadata watch report for one checkpoint blocker."""

    if not isinstance(expected_step, int) or isinstance(expected_step, bool) or expected_step < 1:
        raise CheckpointRemoteWatchError("CHECKPOINT_REMOTE_WATCH_EXPECTED_STEP_INVALID")
    if remote_timeout_seconds < 0:
        raise CheckpointRemoteWatchError("CHECKPOINT_REMOTE_WATCH_TIMEOUT_INVALID")
    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise CheckpointRemoteWatchError("CHECKPOINT_REMOTE_WATCH_OUTPUT_EXISTS")
    if not isinstance(environment, str) or not environment:
        raise CheckpointRemoteWatchError("CHECKPOINT_REMOTE_WATCH_ENVIRONMENT_INVALID")

    step_audit = _load_step_audit(checkpoint_step_audit_path, expected_step=expected_step)
    record = _select_environment_record(step_audit, environment)
    relative_path = str(record["checkpoint_relative_path"])
    local_hash = _local_hf_hash(record)
    observed_step = int(record["observed_step"])
    local_status = str(record["status"])
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
    remote_hashes = sorted(_remote_hashes(model_api_probe, file_head_probe))
    state = _watch_state(
        local_status=local_status,
        observed_step=observed_step,
        expected_step=expected_step,
        local_hash=local_hash,
        remote_hashes=remote_hashes,
        remote_probe_succeeded=model_api_probe["status"] == "ok" or file_head_probe["status"] == "ok",
        remote_probes_skipped=skip_remote_probes,
    )
    candidate_available = state == "remote_candidate_available"
    report = {
        "schema_version": 1,
        "artifact_type": "acwm-m0-checkpoint-remote-watch",
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
        "remote_hashes": remote_hashes,
        "candidate_available": candidate_available,
        "downloaded_checkpoint_bytes": False,
        "active_checkpoint_mutated": False,
        "m4_launch_allowed": False,
        "formal_training_allowed": False,
        "remote_probes": {
            "model_api": model_api_probe,
            "file_head": file_head_probe,
        },
        "next_actions": _next_actions(
            state=state,
            environment=environment,
            relative_path=relative_path,
        ),
        "limitations": [
            "This watch reads remote metadata only and never downloads checkpoint bodies.",
            "A differing remote hash is only a replacement candidate; it must be downloaded to quarantine and pass checkpoint_quarantine validate before install.",
            "This report never authorizes M4 training; a regenerated strict phase gate must report m4_launch_allowed=true.",
        ],
    }
    return _write_report_bundle(report=report, output_root=destination, archive_db=archive_db, cas_root=cas_root)


def _load_step_audit(path: Path, *, expected_step: int) -> Mapping[str, Any]:
    payload = _load_json_mapping(path, "CHECKPOINT_REMOTE_WATCH_STEP_AUDIT_INVALID")
    if payload.get("schema_version") != 1 or payload.get("artifact_type") != "acwm-m0-checkpoint-step-audit":
        raise CheckpointRemoteWatchError("CHECKPOINT_REMOTE_WATCH_STEP_AUDIT_INVALID")
    if payload.get("expected_step") != expected_step:
        raise CheckpointRemoteWatchError("CHECKPOINT_REMOTE_WATCH_EXPECTED_STEP_MISMATCH")
    records = payload.get("records")
    if not isinstance(records, list):
        raise CheckpointRemoteWatchError("CHECKPOINT_REMOTE_WATCH_STEP_AUDIT_INVALID")
    return payload


def _select_environment_record(step_audit: Mapping[str, Any], environment: str) -> Mapping[str, Any]:
    records = step_audit.get("records")
    assert isinstance(records, list)
    for item in records:
        if not isinstance(item, Mapping):
            raise CheckpointRemoteWatchError("CHECKPOINT_REMOTE_WATCH_STEP_AUDIT_INVALID")
        if item.get("environment") == environment:
            _validate_record(item)
            return item
    raise CheckpointRemoteWatchError("CHECKPOINT_REMOTE_WATCH_ENVIRONMENT_NOT_FOUND")


def _validate_record(record: Mapping[str, Any]) -> None:
    required_strings = ("environment", "checkpoint_relative_path", "checkpoint_path", "status")
    for field in required_strings:
        if not isinstance(record.get(field), str) or not record.get(field):
            raise CheckpointRemoteWatchError("CHECKPOINT_REMOTE_WATCH_STEP_AUDIT_INVALID")
    for field in ("expected_step", "observed_step"):
        if not isinstance(record.get(field), int) or isinstance(record.get(field), bool) or record.get(field) < 0:
            raise CheckpointRemoteWatchError("CHECKPOINT_REMOTE_WATCH_STEP_AUDIT_INVALID")
    if not isinstance(record.get("huggingface_download_metadata"), Mapping):
        raise CheckpointRemoteWatchError("CHECKPOINT_REMOTE_WATCH_STEP_AUDIT_INVALID")


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


def _local_hf_hash(record: Mapping[str, Any]) -> str | None:
    metadata = record.get("huggingface_download_metadata")
    if not isinstance(metadata, Mapping):
        return None
    value = metadata.get("etag_or_lfs_sha256")
    return _normalize_hash(value)


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


def _normalize_hash(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip().strip('"')
    if len(stripped) == 64 and all(character in "0123456789abcdef" for character in stripped.lower()):
        return stripped.lower()
    return None


def _watch_state(
    *,
    local_status: str,
    observed_step: int,
    expected_step: int,
    local_hash: str | None,
    remote_hashes: Sequence[str],
    remote_probe_succeeded: bool,
    remote_probes_skipped: bool,
) -> str:
    if local_status == "pass" and observed_step == expected_step:
        return "no_checkpoint_blocker"
    if local_hash is None:
        return "local_hash_unavailable"
    if any(remote_hash != local_hash for remote_hash in remote_hashes):
        return "remote_candidate_available"
    if local_hash in set(remote_hashes):
        return "remote_still_matches_blocked_local"
    if remote_probes_skipped:
        return "remote_probe_skipped"
    if not remote_probe_succeeded:
        return "remote_unreachable"
    return "remote_inconclusive"


def _next_actions(*, state: str, environment: str, relative_path: str) -> list[str]:
    if state == "no_checkpoint_blocker":
        return ["No remote checkpoint watch action is required for this environment."]
    if state == "remote_candidate_available":
        return [
            f"Download the differing remote {environment} checkpoint to results/quarantine/checkpoints/{environment}/latest.pt, not over the active checkpoint.",
            "Run wmloop.evaluate.checkpoint_quarantine validate on the quarantine file.",
            "Install only after a ready_for_manual_install quarantine report and explicit human confirmation.",
        ]
    if state == "remote_still_matches_blocked_local":
        return [
            f"Do not redownload main/{relative_path}; remote metadata still matches the locally blocked checkpoint.",
            "Wait for a publisher-fixed hash or provide a verified alternate 100k-step candidate under quarantine.",
            "Keep M0 checkpoint source resolution and M4 launch blocked.",
        ]
    if state == "remote_unreachable":
        return [
            "Resolve remote access and rerun this watch before deciding whether a publisher fix exists.",
            "Do not overwrite active checkpoints while remote status is unknown.",
        ]
    return [
        "Treat the remote checkpoint source as unresolved.",
        "Provide a verified alternate 100k-step candidate under quarantine or rerun after metadata becomes conclusive.",
    ]


def _render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# M0 Checkpoint Remote Watch",
        "",
        f"State: `{report['state']}`",
        f"Observed at: `{report['observed_at_utc']}`",
        f"Repository: `{report['repository_id']}`",
        f"Environment: `{report['environment']}`",
        f"Expected step: `{report['expected_step']}`",
        f"Observed step: `{report['observed_step']}`",
        f"Candidate available: `{report['candidate_available']}`",
        f"Downloaded checkpoint bytes: `{report['downloaded_checkpoint_bytes']}`",
        f"Active checkpoint mutated: `{report['active_checkpoint_mutated']}`",
        f"M4 launch allowed: `{report['m4_launch_allowed']}`",
        "",
        "## Hashes",
        "",
        f"- Local HF hash: `{report['local_hf_hash']}`",
        f"- Remote hashes: `{', '.join(report['remote_hashes']) or 'none'}`",
        "",
        "## Next Actions",
        "",
    ]
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
        raise CheckpointRemoteWatchError("CHECKPOINT_REMOTE_WATCH_OUTPUT_EXISTS")
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    report_bytes = _canonical_json_bytes(report)
    markdown_bytes = _render_markdown(report).encode("utf-8")
    try:
        temporary.mkdir(mode=0o700)
        _write_bytes_atomic(temporary / "checkpoint-remote-watch.json", report_bytes)
        _write_bytes_atomic(temporary / "checkpoint-remote-watch.md", markdown_bytes)
        archive = ArchiveStore(archive_db) if archive_db is not None else None
        cas_refs: dict[str, str] = {}
        if archive is not None or cas_root is not None:
            root = cas_root if cas_root is not None else Path(archive_db).resolve().parent  # type: ignore[union-attr]
            cas = ContentAddressedStore(Path(root))
            for name, payload, media_type in (
                ("checkpoint_remote_watch_json", report_bytes, "application/json"),
                ("checkpoint_remote_watch_markdown", markdown_bytes, "text/markdown"),
            ):
                ref = cas.put_bytes(payload, media_type=media_type).uri
                cas_refs[name] = ref
                if archive is not None:
                    archive.record_artifact_reference(ref)
        manifest = {
            "schema_version": 1,
            "artifact_type": "acwm-m0-checkpoint-remote-watch-manifest",
            "state": report["state"],
            "observed_at_utc": report["observed_at_utc"],
            "repository_id": report["repository_id"],
            "environment": report["environment"],
            "checkpoint_relative_path": report["checkpoint_relative_path"],
            "expected_step": report["expected_step"],
            "observed_step": report["observed_step"],
            "candidate_available": report["candidate_available"],
            "downloaded_checkpoint_bytes": report["downloaded_checkpoint_bytes"],
            "active_checkpoint_mutated": report["active_checkpoint_mutated"],
            "m4_launch_allowed": report["m4_launch_allowed"],
            "formal_training_allowed": report["formal_training_allowed"],
            "report_path": str(destination / "checkpoint-remote-watch.json"),
            "markdown_path": str(destination / "checkpoint-remote-watch.md"),
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
        raise CheckpointRemoteWatchError(code) from exc
    if not isinstance(payload, Mapping):
        raise CheckpointRemoteWatchError(code)
    return payload


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
    run = commands.add_parser("run", help="watch remote checkpoint metadata for one environment")
    run.add_argument("--checkpoint-step-audit", type=Path, required=True)
    run.add_argument("--environment", required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--repository-id", default="t1an/ACWM-Phys-checkpoints")
    run.add_argument("--expected-step", type=int, default=100000)
    run.add_argument("--remote-timeout-seconds", type=float, default=15.0)
    run.add_argument("--skip-remote-probes", action="store_true")
    run.add_argument("--archive-db", type=Path)
    run.add_argument("--cas-root", type=Path)
    args = parser.parse_args(argv)
    if args.command == "run":
        manifest = generate_checkpoint_remote_watch(
            checkpoint_step_audit_path=args.checkpoint_step_audit,
            environment=args.environment,
            output_root=args.output_root,
            repository_id=args.repository_id,
            expected_step=args.expected_step,
            remote_timeout_seconds=args.remote_timeout_seconds,
            skip_remote_probes=args.skip_remote_probes,
            archive_db=args.archive_db,
            cas_root=args.cas_root,
        )
        print(json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
