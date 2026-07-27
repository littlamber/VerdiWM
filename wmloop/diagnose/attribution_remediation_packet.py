"""Generate a read-only remediation packet for unresolved M1 attribution review."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from wmloop.archive.store import ArchiveStore, ContentAddressedStore


class AttributionRemediationPacketError(RuntimeError):
    """Attribution remediation packet generation failed closed."""


def run_attribution_remediation_packet(
    *,
    attribution_review_manifest: Path,
    output_root: Path,
    archive_db: Path | None = None,
    cas_root: Path | None = None,
) -> dict[str, object]:
    """Write a human-facing remediation packet without altering attribution evidence."""

    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise AttributionRemediationPacketError("ATTRIBUTION_REMEDIATION_OUTPUT_EXISTS")
    manifest, manifest_bytes, manifest_path = _read_json(
        attribution_review_manifest,
        "ATTRIBUTION_REMEDIATION_REVIEW_MANIFEST_INVALID",
    )
    report, report_bytes, report_path = _load_review_report(manifest)
    _validate_review_bundle(manifest=manifest, report=report)
    blockers = _remediation_blockers(manifest=manifest, report=report)
    state = "ready_for_attribution_signoff" if not blockers else "requires_human_attribution_resolution"
    packet = {
        "schema_version": 1,
        "artifact_type": "wmloop-m1-attribution-remediation-packet",
        "state": state,
        "signoff_allowed": not blockers,
        "human_resolution_required": bool(blockers),
        "m4_launch_allowed": False,
        "formal_training_allowed": False,
        "review_summary": {
            "state": manifest.get("state"),
            "manual_pass": manifest.get("manual_pass"),
            "human_signed_off": manifest.get("human_signed_off"),
            "reviewed_count": manifest.get("reviewed_count"),
            "required_review_count": manifest.get("required_review_count"),
            "expectation_mismatch_count": manifest.get("expectation_mismatch_count"),
        },
        "blockers": blockers,
        "mismatches": _mismatches(report),
        "blocked_environments": _blocked_environments(report),
        "action_items": _action_items(blockers),
        "post_resolution_contract": _post_resolution_contract(blockers),
        "source_attribution_review": {
            "manifest_path": str(manifest_path),
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "report_path": str(report_path),
            "report_sha256": hashlib.sha256(report_bytes).hexdigest(),
        },
        "limitations": [
            "This packet is read-only and does not sign attribution, revise expectations, mutate protocol, or launch GPU work.",
            "M4 remains forbidden until a regenerated strict phase gate reports m4_launch_allowed=true.",
        ],
    }
    return _write_report_bundle(packet=packet, output_root=destination, archive_db=archive_db, cas_root=cas_root)


def _validate_review_bundle(*, manifest: Mapping[str, Any], report: Mapping[str, Any]) -> None:
    if manifest.get("artifact_type") != "wmloop-m1-attribution-review-manifest":
        raise AttributionRemediationPacketError("ATTRIBUTION_REMEDIATION_REVIEW_MANIFEST_INVALID")
    if report.get("artifact_type") != "wmloop-m1-attribution-review-draft":
        raise AttributionRemediationPacketError("ATTRIBUTION_REMEDIATION_REVIEW_REPORT_INVALID")
    for key in ("state", "reviewed_count", "required_review_count", "expectation_mismatch_count"):
        if manifest.get(key) != report.get(key):
            raise AttributionRemediationPacketError("ATTRIBUTION_REMEDIATION_REVIEW_MANIFEST_REPORT_MISMATCH")
    if manifest.get("manual_pass") is not False or report.get("manual_pass") is not False:
        raise AttributionRemediationPacketError("ATTRIBUTION_REMEDIATION_REVIEW_ALREADY_MANUAL_PASS")
    if manifest.get("human_signed_off") is not False or report.get("human_signed_off") is not False:
        raise AttributionRemediationPacketError("ATTRIBUTION_REMEDIATION_REVIEW_ALREADY_SIGNED")
    if _int(manifest.get("required_review_count")) < 1:
        raise AttributionRemediationPacketError("ATTRIBUTION_REMEDIATION_REVIEW_COUNT_INVALID")


def _remediation_blockers(*, manifest: Mapping[str, Any], report: Mapping[str, Any]) -> list[dict[str, object]]:
    reviewed = _int(manifest.get("reviewed_count"))
    required = _int(manifest.get("required_review_count"))
    mismatch_count = _int(manifest.get("expectation_mismatch_count"))
    blockers: list[dict[str, object]] = []
    if reviewed < required:
        blockers.append(
            {
                "kind": "insufficient_reviewable_environments",
                "severity": "blocking",
                "observed": reviewed,
                "required": required,
                "missing": required - reviewed,
                "resolution_owner": "protocol_or_data_decision",
                "reason": "The strict M1 manual attribution check requires enough real failure reports before sign-off.",
            }
        )
    if mismatch_count > 0:
        blockers.append(
            {
                "kind": "expectation_mismatch",
                "severity": "blocking",
                "count": mismatch_count,
                "resolution_owner": "human_attribution_reviewer",
                "reason": "Measured dominant_failure records disagree with the documented attribution expectation.",
            }
        )
    blocked = _blocked_environments(report)
    if blocked:
        blockers.append(
            {
                "kind": "protocol_limited_environments",
                "severity": "blocking" if reviewed < required else "informational",
                "count": len(blocked),
                "environments": [item["environment"] for item in blocked],
                "resolution_owner": "human_protocol_decision",
                "reason": "These environments cannot produce formal current-protocol failure reports until the horizon/data path is resolved.",
            }
        )
    if manifest.get("state") == "ready_for_human_signoff" and reviewed >= required and mismatch_count == 0:
        return []
    if not blockers:
        blockers.append(
            {
                "kind": "review_state_not_ready",
                "severity": "blocking",
                "state": manifest.get("state"),
                "resolution_owner": "human_attribution_reviewer",
                "reason": "The review manifest is not marked ready_for_human_signoff.",
            }
        )
    return blockers


def _action_items(blockers: Sequence[Mapping[str, Any]]) -> list[str]:
    if not blockers:
        return [
            "Run wmloop.verify.attribution_signoff with explicit human confirmation if the reviewer accepts the attribution draft.",
            "Regenerate the M4 phase gate with the signoff receipt; do not bypass other M0/M1/M3 blockers.",
        ]
    kinds = {str(blocker.get("kind")) for blocker in blockers}
    actions = [
        "Do not sign the current attribution review and do not launch M4 formal training from this packet.",
    ]
    if "expectation_mismatch" in kinds:
        actions.append(
            "Resolve the mismatch by either accepting the measured dominant_failure with a written rationale, or regenerating evidence under an approved protocol."
        )
    if "insufficient_reviewable_environments" in kinds or "protocol_limited_environments" in kinds:
        actions.append(
            "Select and approve a protocol/data path, then regenerate enough real failure reports to reach the required review count."
        )
    actions.append("After evidence changes, rerun attribution_review, M3 acceptance audit, phase_gate, and blocker_decision_matrix.")
    return actions


def _post_resolution_contract(blockers: Sequence[Mapping[str, Any]]) -> list[str]:
    if not blockers:
        return [
            "A signoff receipt may reference this exact review manifest hash.",
            "The signoff receipt does not resolve checkpoint, raw-evidence, or M3 blockers by itself.",
        ]
    return [
        "Any expectation revision must be explicit human protocol or attribution approval; Codex must not silently edit expectations.",
        "Any protocol-changing path must create a versioned goal config and rerun M1/M3 evidence before M4.",
        "Any new review/signoff must bind to exact manifest and report hashes.",
    ]


def _mismatches(report: Mapping[str, Any]) -> list[dict[str, object]]:
    raw = report.get("missing_or_mismatched_expectations")
    if not isinstance(raw, list):
        return []
    result: list[dict[str, object]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        result.append(
            {
                "environment": item.get("environment"),
                "expected_dominant_failure": item.get("expected_dominant_failure"),
                "observed_dominant_failure": item.get("observed_dominant_failure"),
                "state": item.get("state"),
                "source": item.get("source"),
                "classification": _mismatch_classification(item),
            }
        )
    return result


def _mismatch_classification(item: Mapping[str, Any]) -> str:
    if item.get("state") == "missing_report":
        return "missing_measurement"
    if item.get("expected_dominant_failure") and item.get("observed_dominant_failure"):
        return "expectation_vs_measurement_disagreement"
    return "manual_review_required"


def _blocked_environments(report: Mapping[str, Any]) -> list[dict[str, object]]:
    raw = report.get("blocked_records_from_batch")
    if not isinstance(raw, list):
        return []
    result: list[dict[str, object]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        environment = item.get("environment")
        blockers = item.get("blockers")
        warnings = item.get("confidence_warnings")
        result.append(
            {
                "environment": environment,
                "blockers": blockers if isinstance(blockers, list) else [],
                "confidence_warnings": warnings if isinstance(warnings, list) else [],
            }
        )
    return result


def _load_review_report(manifest: Mapping[str, Any]) -> tuple[Mapping[str, Any], bytes, Path]:
    report_path = manifest.get("report_path")
    if not isinstance(report_path, str) or not report_path:
        raise AttributionRemediationPacketError("ATTRIBUTION_REMEDIATION_REVIEW_REPORT_MISSING")
    return _read_json(Path(report_path), "ATTRIBUTION_REMEDIATION_REVIEW_REPORT_INVALID")


def _read_json(path: Path, error: str) -> tuple[Mapping[str, Any], bytes, Path]:
    resolved = Path(path).resolve(strict=True)
    try:
        payload_bytes = resolved.read_bytes()
        payload = json.loads(payload_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise AttributionRemediationPacketError(f"{error}:{resolved}") from exc
    if not isinstance(payload, Mapping):
        raise AttributionRemediationPacketError(f"{error}:{resolved}")
    return payload, payload_bytes, resolved


def _write_report_bundle(
    *,
    packet: Mapping[str, object],
    output_root: Path,
    archive_db: Path | None,
    cas_root: Path | None,
) -> dict[str, object]:
    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise AttributionRemediationPacketError("ATTRIBUTION_REMEDIATION_OUTPUT_EXISTS")
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    packet_bytes = _canonical_json_bytes(packet)
    markdown_bytes = _render_markdown(packet).encode("utf-8")
    try:
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary.mkdir(mode=0o700)
        _write_bytes_atomic(temporary / "attribution-remediation-packet.json", packet_bytes)
        _write_bytes_atomic(temporary / "attribution-remediation-packet.md", markdown_bytes)
        cas_refs: dict[str, str] = {}
        archive = ArchiveStore(archive_db) if archive_db is not None else None
        if archive is not None or cas_root is not None:
            root = cas_root if cas_root is not None else Path(archive_db).resolve().parent  # type: ignore[union-attr]
            cas = ContentAddressedStore(Path(root))
            for name, payload, media_type in (
                ("attribution_remediation_packet_json", packet_bytes, "application/json"),
                ("attribution_remediation_packet_markdown", markdown_bytes, "text/markdown"),
            ):
                ref = cas.put_bytes(payload, media_type=media_type).uri
                cas_refs[name] = ref
                if archive is not None:
                    archive.record_artifact_reference(ref)
        manifest = {
            "schema_version": 1,
            "artifact_type": "wmloop-m1-attribution-remediation-packet-manifest",
            "state": packet["state"],
            "signoff_allowed": packet["signoff_allowed"],
            "human_resolution_required": packet["human_resolution_required"],
            "m4_launch_allowed": packet["m4_launch_allowed"],
            "formal_training_allowed": packet["formal_training_allowed"],
            "blocker_count": len(packet["blockers"]),  # type: ignore[arg-type]
            "review_summary": packet["review_summary"],
            "source_attribution_review": packet["source_attribution_review"],
            "report_path": str(destination / "attribution-remediation-packet.json"),
            "markdown_path": str(destination / "attribution-remediation-packet.md"),
            "cas_refs": cas_refs,
            "action_items": packet["action_items"],
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


def _render_markdown(packet: Mapping[str, Any]) -> str:
    lines = [
        "# M1 Attribution Remediation Packet",
        "",
        f"State: `{packet['state']}`",
        f"Signoff allowed: `{packet['signoff_allowed']}`",
        f"M4 launch allowed: `{packet['m4_launch_allowed']}`",
        f"Formal training allowed: `{packet['formal_training_allowed']}`",
        "",
        "## Review Summary",
        "",
    ]
    summary = packet["review_summary"]
    lines.extend(
        [
            f"- Review state: `{summary['state']}`",
            f"- Reviewed: `{summary['reviewed_count']}/{summary['required_review_count']}`",
            f"- Expectation mismatches: `{summary['expectation_mismatch_count']}`",
            f"- Human signed off: `{summary['human_signed_off']}`",
        ]
    )
    if packet["blockers"]:
        lines.extend(["", "## Blockers", "", "| Kind | Owner | Reason |", "|:--|:--|:--|"])
        for blocker in packet["blockers"]:
            lines.append(f"| `{blocker['kind']}` | `{blocker['resolution_owner']}` | {blocker['reason']} |")
    if packet["mismatches"]:
        lines.extend(["", "## Mismatches", "", "| Environment | Expected | Observed | Classification |", "|:--|:--|:--|:--|"])
        for mismatch in packet["mismatches"]:
            lines.append(
                f"| `{mismatch['environment']}` | `{mismatch['expected_dominant_failure']}` | "
                f"`{mismatch['observed_dominant_failure']}` | `{mismatch['classification']}` |"
            )
    if packet["blocked_environments"]:
        lines.extend(["", "## Protocol-Limited Environments", ""])
        for item in packet["blocked_environments"]:
            lines.append(
                f"- `{item['environment']}` blockers=`{','.join(str(value) for value in item['blockers'])}` "
                f"warnings=`{','.join(str(value) for value in item['confidence_warnings'])}`"
            )
    lines.extend(["", "## Action Items", ""])
    lines.extend(f"- {item}" for item in packet["action_items"])
    return "\n".join(lines) + "\n"


def _int(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise AttributionRemediationPacketError("ATTRIBUTION_REMEDIATION_OUTPUT_EXISTS")
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
    run = commands.add_parser("run", help="generate an attribution remediation packet")
    run.add_argument("--attribution-review-manifest", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--archive-db", type=Path)
    run.add_argument("--cas-root", type=Path)
    args = parser.parse_args(argv)
    if args.command == "run":
        manifest = run_attribution_remediation_packet(
            attribution_review_manifest=args.attribution_review_manifest,
            output_root=args.output_root,
            archive_db=args.archive_db,
            cas_root=args.cas_root,
        )
        print(json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
