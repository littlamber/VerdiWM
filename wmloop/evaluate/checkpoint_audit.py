"""Audit M0 baseline checkpoint training steps against the official target."""

from __future__ import annotations

import argparse
import json
import math
import os
import uuid
from pathlib import Path
from typing import Mapping, Sequence

from wmloop.acwm_data import CANONICAL_ACWM_ENVIRONMENTS
from wmloop.archive.store import ArchiveStore, ContentAddressedStore


class CheckpointAuditError(RuntimeError):
    """Checkpoint audit evidence could not be produced safely."""


def generate_checkpoint_step_audit(
    *,
    checkpoint_root: Path,
    output_root: Path,
    expected_step: int = 100000,
    archive_db: Path | None = None,
    cas_root: Path | None = None,
) -> dict[str, object]:
    """Write a durable fail-closed audit for released ACWM checkpoint steps."""

    if not isinstance(expected_step, int) or isinstance(expected_step, bool) or expected_step < 1:
        raise CheckpointAuditError("CHECKPOINT_AUDIT_EXPECTED_STEP_INVALID")
    root = Path(checkpoint_root).resolve(strict=True)
    records = []
    for spec in CANONICAL_ACWM_ENVIRONMENTS:
        checkpoint = _regular_file(root / spec.checkpoint_relative_path)
        observed_step = _checkpoint_step(checkpoint)
        records.append(
            {
                "environment": spec.environment,
                "checkpoint_relative_path": spec.checkpoint_relative_path,
                "checkpoint_path": str(checkpoint),
                "size_bytes": checkpoint.stat().st_size,
                "huggingface_download_metadata": _huggingface_download_metadata(root, spec.checkpoint_relative_path),
                "expected_step": expected_step,
                "observed_step": observed_step,
                "status": "pass" if observed_step == expected_step else "step_mismatch",
            }
        )
    mismatch_count = sum(1 for record in records if record["status"] != "pass")
    report = {
        "schema_version": 1,
        "artifact_type": "acwm-m0-checkpoint-step-audit",
        "state": "ready" if mismatch_count == 0 else "step_mismatch",
        "expected_step": expected_step,
        "checkpoint_root": str(root),
        "environment_count": len(records),
        "mismatch_count": mismatch_count,
        "records": records,
    }
    markdown = _render_markdown(report)
    return _write_report_bundle(
        report=report,
        markdown=markdown,
        output_root=output_root,
        archive_db=archive_db,
        cas_root=cas_root,
    )


def _regular_file(path: Path) -> Path:
    if path.is_symlink() or not path.is_file():
        raise CheckpointAuditError(f"CHECKPOINT_AUDIT_CHECKPOINT_MISSING:{path}")
    return path.resolve(strict=True)


def _checkpoint_step(path: Path) -> int:
    try:
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise CheckpointAuditError(f"CHECKPOINT_AUDIT_LOAD_FAILED:{path}") from exc
    if not isinstance(payload, Mapping):
        raise CheckpointAuditError(f"CHECKPOINT_AUDIT_PAYLOAD_INVALID:{path}")
    value = payload.get("step")
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise CheckpointAuditError(f"CHECKPOINT_AUDIT_STEP_INVALID:{path}")
    step = int(value)
    if step != float(value) or step < 0:
        raise CheckpointAuditError(f"CHECKPOINT_AUDIT_STEP_INVALID:{path}")
    return step


def _huggingface_download_metadata(checkpoint_root: Path, relative_path: str) -> dict[str, object]:
    metadata_path = checkpoint_root / ".cache" / "huggingface" / "download" / f"{relative_path}.metadata"
    if metadata_path.is_symlink() or not metadata_path.is_file():
        return {"status": "missing", "path": str(metadata_path)}
    try:
        lines = metadata_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise CheckpointAuditError(f"CHECKPOINT_AUDIT_HF_METADATA_INVALID:{metadata_path}") from exc
    if len(lines) < 2 or not lines[0] or not lines[1]:
        raise CheckpointAuditError(f"CHECKPOINT_AUDIT_HF_METADATA_INVALID:{metadata_path}")
    return {
        "status": "available",
        "path": str(metadata_path),
        "download_commit": lines[0],
        "etag_or_lfs_sha256": lines[1],
        "timestamp": lines[2] if len(lines) > 2 else None,
    }


def _render_markdown(report: Mapping[str, object]) -> str:
    lines = [
        "# M0 Checkpoint Step Audit",
        "",
        f"State: `{report['state']}`",
        f"Expected step: `{report['expected_step']}`",
        f"Mismatch count: `{report['mismatch_count']}`",
        "",
        "| Environment | Observed Step | Expected Step | Status | HF Metadata | Checkpoint |",
        "|:--|--:|--:|:--|:--|:--|",
    ]
    for record in report["records"]:
        hf_metadata = record["huggingface_download_metadata"]
        hf_status = hf_metadata["status"]
        if hf_status == "available":
            hf_status = f"{hf_status}:{hf_metadata['etag_or_lfs_sha256']}"
        lines.append(
            f"| {record['environment']} | {record['observed_step']} | {record['expected_step']} | "
            f"{record['status']} | `{hf_status}` | `{record['checkpoint_relative_path']}` |"
        )
    return "\n".join(lines) + "\n"


def _write_report_bundle(
    *,
    report: Mapping[str, object],
    markdown: str,
    output_root: Path,
    archive_db: Path | None,
    cas_root: Path | None,
) -> dict[str, object]:
    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise CheckpointAuditError("CHECKPOINT_AUDIT_OUTPUT_EXISTS")
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    report_bytes = _canonical_json_bytes(report)
    markdown_bytes = markdown.encode("utf-8")
    cas_refs: dict[str, str] = {}
    try:
        temporary.mkdir(mode=0o700)
        _write_bytes_atomic(temporary / "checkpoint-step-audit.json", report_bytes)
        _write_bytes_atomic(temporary / "checkpoint-step-audit.md", markdown_bytes)
        archive = ArchiveStore(archive_db) if archive_db is not None else None
        if archive is not None or cas_root is not None:
            root = cas_root if cas_root is not None else Path(archive_db).resolve().parent  # type: ignore[union-attr]
            cas = ContentAddressedStore(Path(root))
            for name, payload, media_type in (
                ("checkpoint_step_audit_json", report_bytes, "application/json"),
                ("checkpoint_step_audit_markdown", markdown_bytes, "text/markdown"),
            ):
                ref = cas.put_bytes(payload, media_type=media_type).uri
                cas_refs[name] = ref
                if archive is not None:
                    archive.record_artifact_reference(ref)
        manifest = {
            "schema_version": 1,
            "artifact_type": "acwm-m0-checkpoint-step-audit-manifest",
            "state": report["state"],
            "expected_step": report["expected_step"],
            "mismatch_count": report["mismatch_count"],
            "report_path": str(destination / "checkpoint-step-audit.json"),
            "markdown_path": str(destination / "checkpoint-step-audit.md"),
            "cas_refs": cas_refs,
        }
        _write_bytes_atomic(temporary / "manifest.json", _canonical_json_bytes(manifest))
        os.replace(temporary, destination)
        return manifest
    except Exception:
        if temporary.is_symlink():
            temporary.unlink()
        elif temporary.exists():
            import shutil

            shutil.rmtree(temporary)
        raise


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise CheckpointAuditError("CHECKPOINT_AUDIT_OUTPUT_EXISTS")
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
    run = commands.add_parser("run", help="audit ACWM checkpoint training steps")
    run.add_argument("--checkpoint-root", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--expected-step", type=int, default=100000)
    run.add_argument("--archive-db", type=Path)
    run.add_argument("--cas-root", type=Path)
    args = parser.parse_args(argv)
    if args.command == "run":
        result = generate_checkpoint_step_audit(
            checkpoint_root=args.checkpoint_root,
            output_root=args.output_root,
            expected_step=args.expected_step,
            archive_db=args.archive_db,
            cas_root=args.cas_root,
        )
        print(json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
