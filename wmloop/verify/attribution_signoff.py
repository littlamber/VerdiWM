"""Create a human sign-off receipt for a ready M1 attribution review draft."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from wmloop.archive.store import ArchiveStore, ContentAddressedStore


class AttributionSignoffError(RuntimeError):
    """Attribution sign-off failed closed."""


def run_attribution_signoff(
    *,
    attribution_review_manifest: Path,
    reviewer: str,
    output_root: Path,
    confirm_human_signed_off: bool = False,
    archive_db: Path | None = None,
    cas_root: Path | None = None,
) -> dict[str, object]:
    """Sign a ready attribution review draft without mutating the draft."""

    if not confirm_human_signed_off:
        raise AttributionSignoffError("ATTRIBUTION_SIGNOFF_HUMAN_CONFIRMATION_REQUIRED")
    if not reviewer.strip():
        raise AttributionSignoffError("ATTRIBUTION_SIGNOFF_REVIEWER_REQUIRED")
    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise AttributionSignoffError("ATTRIBUTION_SIGNOFF_OUTPUT_EXISTS")

    manifest, manifest_bytes, manifest_path = _read_json(
        attribution_review_manifest,
        "ATTRIBUTION_SIGNOFF_REVIEW_MANIFEST_INVALID",
    )
    report, report_bytes, report_path = _load_review_report(manifest)
    _validate_ready_review(manifest=manifest, report=report)
    receipt = {
        "schema_version": 1,
        "artifact_type": "wmloop-m1-attribution-signoff-receipt",
        "state": "signed",
        "manual_pass": True,
        "human_signed_off": True,
        "reviewer": reviewer.strip(),
        "source_attribution_review": {
            "manifest_path": str(manifest_path),
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "report_path": str(report_path),
            "report_sha256": hashlib.sha256(report_bytes).hexdigest(),
            "state": manifest.get("state"),
            "reviewed_count": manifest.get("reviewed_count"),
            "required_review_count": manifest.get("required_review_count"),
            "expectation_mismatch_count": manifest.get("expectation_mismatch_count"),
        },
        "reviewed_count": manifest.get("reviewed_count"),
        "required_review_count": manifest.get("required_review_count"),
        "expectation_mismatch_count": manifest.get("expectation_mismatch_count"),
        "signed_record_count": len(report.get("records", [])) if isinstance(report.get("records"), list) else None,
        "contract": [
            "This receipt signs an already-ready attribution review draft; it does not alter raw evidence or diagnosis outputs.",
            "The receipt is valid only for the exact attribution review manifest hash recorded here.",
            "M4 phase gate must still verify every other strict prerequisite independently.",
        ],
    }
    return _write_report_bundle(
        receipt=receipt,
        output_root=destination,
        archive_db=archive_db,
        cas_root=cas_root,
    )


def _validate_ready_review(*, manifest: Mapping[str, Any], report: Mapping[str, Any]) -> None:
    if manifest.get("artifact_type") != "wmloop-m1-attribution-review-manifest":
        raise AttributionSignoffError("ATTRIBUTION_SIGNOFF_REVIEW_MANIFEST_INVALID")
    if report.get("artifact_type") != "wmloop-m1-attribution-review-draft":
        raise AttributionSignoffError("ATTRIBUTION_SIGNOFF_REVIEW_REPORT_INVALID")
    reviewed_count = _int(manifest.get("reviewed_count"))
    required_review_count = _int(manifest.get("required_review_count"))
    mismatch_count = _int(manifest.get("expectation_mismatch_count"))
    if (
        manifest.get("state") != "ready_for_human_signoff"
        or report.get("state") != "ready_for_human_signoff"
        or manifest.get("manual_pass") is not False
        or manifest.get("human_signed_off") is not False
        or reviewed_count < required_review_count
        or mismatch_count != 0
        or required_review_count < 1
    ):
        raise AttributionSignoffError("ATTRIBUTION_SIGNOFF_REVIEW_NOT_READY")
    if (
        reviewed_count != _int(report.get("reviewed_count"))
        or required_review_count != _int(report.get("required_review_count"))
        or mismatch_count != _int(report.get("expectation_mismatch_count"))
    ):
        raise AttributionSignoffError("ATTRIBUTION_SIGNOFF_REVIEW_MANIFEST_REPORT_MISMATCH")


def _load_review_report(manifest: Mapping[str, Any]) -> tuple[Mapping[str, Any], bytes, Path]:
    report_path = manifest.get("report_path")
    if not isinstance(report_path, str) or not report_path:
        raise AttributionSignoffError("ATTRIBUTION_SIGNOFF_REVIEW_REPORT_MISSING")
    return _read_json(Path(report_path), "ATTRIBUTION_SIGNOFF_REVIEW_REPORT_INVALID")


def _read_json(path: Path, error: str) -> tuple[Mapping[str, Any], bytes, Path]:
    resolved = Path(path).resolve(strict=True)
    try:
        payload_bytes = resolved.read_bytes()
        payload = json.loads(payload_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise AttributionSignoffError(f"{error}:{resolved}") from exc
    if not isinstance(payload, Mapping):
        raise AttributionSignoffError(f"{error}:{resolved}")
    return payload, payload_bytes, resolved


def _write_report_bundle(
    *,
    receipt: Mapping[str, object],
    output_root: Path,
    archive_db: Path | None,
    cas_root: Path | None,
) -> dict[str, object]:
    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise AttributionSignoffError("ATTRIBUTION_SIGNOFF_OUTPUT_EXISTS")
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    receipt_bytes = _canonical_json_bytes(receipt)
    markdown_bytes = _render_markdown(receipt).encode("utf-8")
    try:
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary.mkdir(mode=0o700)
        cas_refs: dict[str, str] = {}
        archive = ArchiveStore(archive_db) if archive_db is not None else None
        if archive is not None or cas_root is not None:
            root = cas_root if cas_root is not None else Path(archive_db).resolve().parent  # type: ignore[union-attr]
            cas = ContentAddressedStore(Path(root))
            for name, payload, media_type in (
                ("attribution_signoff_receipt_json", receipt_bytes, "application/json"),
                ("attribution_signoff_receipt_markdown", markdown_bytes, "text/markdown"),
            ):
                ref = cas.put_bytes(payload, media_type=media_type).uri
                cas_refs[name] = ref
                if archive is not None:
                    archive.record_artifact_reference(ref)
        _write_bytes_atomic(temporary / "attribution-signoff-receipt.json", receipt_bytes)
        _write_bytes_atomic(temporary / "attribution-signoff-receipt.md", markdown_bytes)
        manifest = {
            "schema_version": 1,
            "artifact_type": "wmloop-m1-attribution-signoff-receipt-manifest",
            "state": receipt["state"],
            "manual_pass": receipt["manual_pass"],
            "human_signed_off": receipt["human_signed_off"],
            "reviewer": receipt["reviewer"],
            "reviewed_count": receipt["reviewed_count"],
            "required_review_count": receipt["required_review_count"],
            "expectation_mismatch_count": receipt["expectation_mismatch_count"],
            "source_attribution_review": receipt["source_attribution_review"],
            "report_path": str(destination / "attribution-signoff-receipt.json"),
            "markdown_path": str(destination / "attribution-signoff-receipt.md"),
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


def _render_markdown(receipt: Mapping[str, Any]) -> str:
    source = receipt["source_attribution_review"]
    lines = [
        "# M1 Attribution Signoff Receipt",
        "",
        f"State: `{receipt['state']}`",
        f"Manual pass: `{receipt['manual_pass']}`",
        f"Human signed off: `{receipt['human_signed_off']}`",
        f"Reviewer: `{receipt['reviewer']}`",
        f"Reviewed: `{receipt['reviewed_count']}/{receipt['required_review_count']}`",
        f"Expectation mismatches: `{receipt['expectation_mismatch_count']}`",
        "",
        "## Source Review",
        "",
        f"- Manifest: `{source['manifest_path']}`",
        f"- Manifest SHA256: `{source['manifest_sha256']}`",
        f"- Report SHA256: `{source['report_sha256']}`",
        "",
        "## Contract",
        "",
    ]
    lines.extend(f"- {item}" for item in receipt["contract"])
    return "\n".join(lines) + "\n"


def _int(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise AttributionSignoffError("ATTRIBUTION_SIGNOFF_OUTPUT_EXISTS")
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
    run = commands.add_parser("run", help="sign a ready attribution review draft")
    run.add_argument("--attribution-review-manifest", type=Path, required=True)
    run.add_argument("--reviewer", required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--confirm-human-signed-off", action="store_true")
    run.add_argument("--archive-db", type=Path)
    run.add_argument("--cas-root", type=Path)
    args = parser.parse_args(argv)
    if args.command == "run":
        manifest = run_attribution_signoff(
            attribution_review_manifest=args.attribution_review_manifest,
            reviewer=args.reviewer,
            output_root=args.output_root,
            confirm_human_signed_off=args.confirm_human_signed_off,
            archive_db=args.archive_db,
            cas_root=args.cas_root,
        )
        print(json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
