"""Create a human-approved M1 attribution expectation revision receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from wmloop.archive.store import ArchiveStore, ContentAddressedStore


class AttributionExpectationRevisionError(RuntimeError):
    """Attribution expectation revision failed closed."""


def run_attribution_expectation_revision(
    *,
    attribution_remediation_manifest: Path,
    source_expectations: Path,
    environment: str,
    accept_observed_dominant_failure: str,
    reviewer: str,
    rationale: str,
    output_root: Path,
    confirm_human_attribution_resolution: bool = False,
    archive_db: Path | None = None,
    cas_root: Path | None = None,
) -> dict[str, object]:
    """Write revised expectations after explicit human attribution approval."""

    if not confirm_human_attribution_resolution:
        raise AttributionExpectationRevisionError("ATTRIBUTION_EXPECTATION_REVISION_HUMAN_CONFIRMATION_REQUIRED")
    if not environment.strip():
        raise AttributionExpectationRevisionError("ATTRIBUTION_EXPECTATION_REVISION_ENVIRONMENT_REQUIRED")
    if not accept_observed_dominant_failure.strip():
        raise AttributionExpectationRevisionError("ATTRIBUTION_EXPECTATION_REVISION_OBSERVED_FAILURE_REQUIRED")
    if not reviewer.strip():
        raise AttributionExpectationRevisionError("ATTRIBUTION_EXPECTATION_REVISION_REVIEWER_REQUIRED")
    if not rationale.strip():
        raise AttributionExpectationRevisionError("ATTRIBUTION_EXPECTATION_REVISION_RATIONALE_REQUIRED")

    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise AttributionExpectationRevisionError("ATTRIBUTION_EXPECTATION_REVISION_OUTPUT_EXISTS")

    remediation_manifest, remediation_manifest_bytes, remediation_manifest_path = _read_json(
        attribution_remediation_manifest,
        "ATTRIBUTION_EXPECTATION_REVISION_REMEDIATION_MANIFEST_INVALID",
    )
    remediation_packet, remediation_packet_bytes, remediation_packet_path = _load_report(
        remediation_manifest,
        "ATTRIBUTION_EXPECTATION_REVISION_REMEDIATION_PACKET_MISSING",
        "ATTRIBUTION_EXPECTATION_REVISION_REMEDIATION_PACKET_INVALID",
    )
    review_manifest, review_manifest_bytes, review_manifest_path = _load_source_review_manifest(
        remediation_manifest=remediation_manifest,
        remediation_packet=remediation_packet,
    )
    review_report, review_report_bytes, review_report_path = _load_report(
        review_manifest,
        "ATTRIBUTION_EXPECTATION_REVISION_REVIEW_REPORT_MISSING",
        "ATTRIBUTION_EXPECTATION_REVISION_REVIEW_REPORT_INVALID",
    )
    _validate_sources(
        remediation_manifest=remediation_manifest,
        remediation_packet=remediation_packet,
        remediation_manifest_bytes=remediation_manifest_bytes,
        remediation_packet_bytes=remediation_packet_bytes,
        review_manifest=review_manifest,
        review_manifest_bytes=review_manifest_bytes,
        review_report=review_report,
        review_report_bytes=review_report_bytes,
    )
    expectations, expectations_bytes, expectations_path = _read_json(
        source_expectations,
        "ATTRIBUTION_EXPECTATION_REVISION_EXPECTATIONS_INVALID",
    )
    _validate_expectations(expectations)
    mismatch = _matching_mismatch(
        review_report=review_report,
        environment=environment.strip(),
        observed=accept_observed_dominant_failure.strip(),
    )
    revised_expectations = _revised_expectations(
        expectations=expectations,
        mismatch=mismatch,
        reviewer=reviewer.strip(),
        rationale=rationale.strip(),
        review_manifest_sha256=hashlib.sha256(review_manifest_bytes).hexdigest(),
    )
    report = _revision_report(
        reviewer=reviewer.strip(),
        rationale=rationale.strip(),
        mismatch=mismatch,
        source_expectations_path=expectations_path,
        source_expectations_bytes=expectations_bytes,
        remediation_manifest_path=remediation_manifest_path,
        remediation_manifest_bytes=remediation_manifest_bytes,
        remediation_packet_path=remediation_packet_path,
        remediation_packet_bytes=remediation_packet_bytes,
        review_manifest_path=review_manifest_path,
        review_manifest_bytes=review_manifest_bytes,
        review_report_path=review_report_path,
        review_report_bytes=review_report_bytes,
        revised_expectations=revised_expectations,
        output_root=destination,
    )
    return _write_report_bundle(
        report=report,
        revised_expectations=revised_expectations,
        output_root=destination,
        archive_db=archive_db,
        cas_root=cas_root,
    )


def _validate_sources(
    *,
    remediation_manifest: Mapping[str, Any],
    remediation_packet: Mapping[str, Any],
    remediation_manifest_bytes: bytes,
    remediation_packet_bytes: bytes,
    review_manifest: Mapping[str, Any],
    review_manifest_bytes: bytes,
    review_report: Mapping[str, Any],
    review_report_bytes: bytes,
) -> None:
    if remediation_manifest.get("artifact_type") != "wmloop-m1-attribution-remediation-packet-manifest":
        raise AttributionExpectationRevisionError("ATTRIBUTION_EXPECTATION_REVISION_REMEDIATION_MANIFEST_INVALID")
    if remediation_packet.get("artifact_type") != "wmloop-m1-attribution-remediation-packet":
        raise AttributionExpectationRevisionError("ATTRIBUTION_EXPECTATION_REVISION_REMEDIATION_PACKET_INVALID")
    if remediation_manifest.get("state") != "requires_human_attribution_resolution":
        raise AttributionExpectationRevisionError("ATTRIBUTION_EXPECTATION_REVISION_REMEDIATION_NOT_BLOCKED")
    if remediation_manifest.get("human_resolution_required") is not True:
        raise AttributionExpectationRevisionError("ATTRIBUTION_EXPECTATION_REVISION_REMEDIATION_NOT_HUMAN_REQUIRED")
    if remediation_manifest.get("signoff_allowed") is not False or remediation_packet.get("signoff_allowed") is not False:
        raise AttributionExpectationRevisionError("ATTRIBUTION_EXPECTATION_REVISION_REMEDIATION_SIGNOFF_ALLOWED")
    if remediation_manifest.get("m4_launch_allowed") is not False or remediation_packet.get("m4_launch_allowed") is not False:
        raise AttributionExpectationRevisionError("ATTRIBUTION_EXPECTATION_REVISION_REMEDIATION_M4_ALLOWED")
    if remediation_manifest.get("formal_training_allowed") is not False or remediation_packet.get("formal_training_allowed") is not False:
        raise AttributionExpectationRevisionError("ATTRIBUTION_EXPECTATION_REVISION_REMEDIATION_FORMAL_TRAINING_ALLOWED")
    source = remediation_manifest.get("source_attribution_review")
    if not isinstance(source, Mapping):
        raise AttributionExpectationRevisionError("ATTRIBUTION_EXPECTATION_REVISION_REMEDIATION_SOURCE_MISSING")
    source_hash = source.get("manifest_sha256")
    if source_hash != hashlib.sha256(review_manifest_bytes).hexdigest():
        raise AttributionExpectationRevisionError("ATTRIBUTION_EXPECTATION_REVISION_REVIEW_MANIFEST_HASH_MISMATCH")
    packet_source = remediation_packet.get("source_attribution_review")
    if not isinstance(packet_source, Mapping) or packet_source.get("manifest_sha256") != source_hash:
        raise AttributionExpectationRevisionError("ATTRIBUTION_EXPECTATION_REVISION_PACKET_SOURCE_MISMATCH")
    if remediation_manifest.get("report_path") and hashlib.sha256(remediation_packet_bytes).hexdigest() != hashlib.sha256(
        Path(str(remediation_manifest["report_path"])).resolve(strict=True).read_bytes()
    ).hexdigest():
        raise AttributionExpectationRevisionError("ATTRIBUTION_EXPECTATION_REVISION_REMEDIATION_PACKET_HASH_DRIFT")
    if review_manifest.get("artifact_type") != "wmloop-m1-attribution-review-manifest":
        raise AttributionExpectationRevisionError("ATTRIBUTION_EXPECTATION_REVISION_REVIEW_MANIFEST_INVALID")
    if review_report.get("artifact_type") != "wmloop-m1-attribution-review-draft":
        raise AttributionExpectationRevisionError("ATTRIBUTION_EXPECTATION_REVISION_REVIEW_REPORT_INVALID")
    if review_manifest.get("state") != "requires_attention" or review_report.get("state") != "requires_attention":
        raise AttributionExpectationRevisionError("ATTRIBUTION_EXPECTATION_REVISION_REVIEW_NOT_ATTENTION")
    if _int(review_manifest.get("expectation_mismatch_count")) < 1 or _int(review_report.get("expectation_mismatch_count")) < 1:
        raise AttributionExpectationRevisionError("ATTRIBUTION_EXPECTATION_REVISION_REVIEW_NO_MISMATCH")
    if review_manifest.get("report_path") and hashlib.sha256(review_report_bytes).hexdigest() != hashlib.sha256(
        Path(str(review_manifest["report_path"])).resolve(strict=True).read_bytes()
    ).hexdigest():
        raise AttributionExpectationRevisionError("ATTRIBUTION_EXPECTATION_REVISION_REVIEW_REPORT_HASH_DRIFT")
    _ = remediation_manifest_bytes


def _load_source_review_manifest(
    *,
    remediation_manifest: Mapping[str, Any],
    remediation_packet: Mapping[str, Any],
) -> tuple[Mapping[str, Any], bytes, Path]:
    source = remediation_manifest.get("source_attribution_review")
    if not isinstance(source, Mapping):
        raise AttributionExpectationRevisionError("ATTRIBUTION_EXPECTATION_REVISION_REMEDIATION_SOURCE_MISSING")
    path = source.get("manifest_path")
    if not isinstance(path, str) or not path:
        packet_source = remediation_packet.get("source_attribution_review")
        path = packet_source.get("manifest_path") if isinstance(packet_source, Mapping) else None
    if not isinstance(path, str) or not path:
        raise AttributionExpectationRevisionError("ATTRIBUTION_EXPECTATION_REVISION_REVIEW_MANIFEST_MISSING")
    return _read_json(Path(path), "ATTRIBUTION_EXPECTATION_REVISION_REVIEW_MANIFEST_INVALID")


def _load_report(
    manifest: Mapping[str, Any],
    missing_error: str,
    invalid_error: str,
) -> tuple[Mapping[str, Any], bytes, Path]:
    report_path = manifest.get("report_path")
    if not isinstance(report_path, str) or not report_path:
        raise AttributionExpectationRevisionError(missing_error)
    return _read_json(Path(report_path), invalid_error)


def _validate_expectations(expectations: Mapping[str, Any]) -> None:
    if (
        expectations.get("schema_version") != 1
        or expectations.get("artifact_type") != "wmloop-m1-attribution-review-expectations"
        or not isinstance(expectations.get("required_review_count"), int)
        or expectations["required_review_count"] < 1
        or not isinstance(expectations.get("expectations"), list)
    ):
        raise AttributionExpectationRevisionError("ATTRIBUTION_EXPECTATION_REVISION_EXPECTATIONS_INVALID")
    seen: set[str] = set()
    for item in expectations["expectations"]:
        if (
            not isinstance(item, Mapping)
            or not isinstance(item.get("environment"), str)
            or not item["environment"]
            or not isinstance(item.get("dominant_failure"), str)
            or not item["dominant_failure"]
        ):
            raise AttributionExpectationRevisionError("ATTRIBUTION_EXPECTATION_REVISION_EXPECTATIONS_INVALID")
        environment = str(item["environment"])
        if environment in seen:
            raise AttributionExpectationRevisionError("ATTRIBUTION_EXPECTATION_REVISION_EXPECTATION_DUPLICATE")
        seen.add(environment)


def _matching_mismatch(
    *,
    review_report: Mapping[str, Any],
    environment: str,
    observed: str,
) -> dict[str, object]:
    mismatches = review_report.get("missing_or_mismatched_expectations")
    if not isinstance(mismatches, list):
        raise AttributionExpectationRevisionError("ATTRIBUTION_EXPECTATION_REVISION_REVIEW_MISMATCHES_INVALID")
    matches = [
        item
        for item in mismatches
        if isinstance(item, Mapping)
        and item.get("state") == "mismatch"
        and item.get("environment") == environment
        and item.get("observed_dominant_failure") == observed
        and isinstance(item.get("expected_dominant_failure"), str)
    ]
    if len(matches) != 1:
        raise AttributionExpectationRevisionError("ATTRIBUTION_EXPECTATION_REVISION_MATCHING_MISMATCH_NOT_FOUND")
    item = matches[0]
    expected = str(item["expected_dominant_failure"])
    if expected == observed:
        raise AttributionExpectationRevisionError("ATTRIBUTION_EXPECTATION_REVISION_NO_ACTUAL_CHANGE")
    return {
        "environment": environment,
        "previous_dominant_failure": expected,
        "revised_dominant_failure": observed,
        "source": item.get("source"),
    }


def _revised_expectations(
    *,
    expectations: Mapping[str, Any],
    mismatch: Mapping[str, object],
    reviewer: str,
    rationale: str,
    review_manifest_sha256: str,
) -> dict[str, object]:
    environment = str(mismatch["environment"])
    previous = str(mismatch["previous_dominant_failure"])
    revised = str(mismatch["revised_dominant_failure"])
    updated: list[dict[str, object]] = []
    changed = False
    for item in expectations["expectations"]:
        current = dict(item)
        if current.get("environment") == environment:
            if current.get("dominant_failure") != previous:
                raise AttributionExpectationRevisionError("ATTRIBUTION_EXPECTATION_REVISION_EXPECTATION_SOURCE_DRIFT")
            current["dominant_failure"] = revised
            current["source"] = (
                "human-approved attribution expectation revision; "
                f"review_manifest_sha256={review_manifest_sha256}; reviewer={reviewer}"
            )
            current["revision"] = {
                "previous_dominant_failure": previous,
                "accepted_observed_dominant_failure": revised,
                "rationale": rationale,
            }
            changed = True
        updated.append(current)
    if not changed:
        raise AttributionExpectationRevisionError("ATTRIBUTION_EXPECTATION_REVISION_EXPECTATION_ENV_MISSING")
    result = {
        "schema_version": expectations["schema_version"],
        "artifact_type": expectations["artifact_type"],
        "required_review_count": expectations["required_review_count"],
        "expectations": updated,
        "revision_history": list(expectations.get("revision_history", []))
        if isinstance(expectations.get("revision_history"), list)
        else [],
    }
    result["revision_history"].append(
        {
            "environment": environment,
            "previous_dominant_failure": previous,
            "revised_dominant_failure": revised,
            "reviewer": reviewer,
            "rationale": rationale,
            "source_review_manifest_sha256": review_manifest_sha256,
        }
    )
    return result


def _revision_report(
    *,
    reviewer: str,
    rationale: str,
    mismatch: Mapping[str, object],
    source_expectations_path: Path,
    source_expectations_bytes: bytes,
    remediation_manifest_path: Path,
    remediation_manifest_bytes: bytes,
    remediation_packet_path: Path,
    remediation_packet_bytes: bytes,
    review_manifest_path: Path,
    review_manifest_bytes: bytes,
    review_report_path: Path,
    review_report_bytes: bytes,
    revised_expectations: Mapping[str, object],
    output_root: Path,
) -> dict[str, object]:
    revised_bytes = _canonical_json_bytes(revised_expectations)
    return {
        "schema_version": 1,
        "artifact_type": "wmloop-m1-attribution-expectation-revision",
        "state": "applied",
        "human_resolution_confirmed": True,
        "manual_pass": False,
        "human_signed_off": False,
        "reviewer": reviewer,
        "rationale": rationale,
        "active_expectations_mutated": False,
        "m4_launch_allowed": False,
        "formal_training_allowed": False,
        "accepted_resolution": {
            "environment": mismatch["environment"],
            "previous_dominant_failure": mismatch["previous_dominant_failure"],
            "revised_dominant_failure": mismatch["revised_dominant_failure"],
            "source": mismatch.get("source"),
        },
        "source_expectations": {
            "path": str(source_expectations_path),
            "sha256": hashlib.sha256(source_expectations_bytes).hexdigest(),
        },
        "source_attribution_remediation": {
            "manifest_path": str(remediation_manifest_path),
            "manifest_sha256": hashlib.sha256(remediation_manifest_bytes).hexdigest(),
            "report_path": str(remediation_packet_path),
            "report_sha256": hashlib.sha256(remediation_packet_bytes).hexdigest(),
        },
        "source_attribution_review": {
            "manifest_path": str(review_manifest_path),
            "manifest_sha256": hashlib.sha256(review_manifest_bytes).hexdigest(),
            "report_path": str(review_report_path),
            "report_sha256": hashlib.sha256(review_report_bytes).hexdigest(),
        },
        "revised_expectations": {
            "path": str(output_root / "revised-attribution-expectations.json"),
            "sha256": hashlib.sha256(revised_bytes).hexdigest(),
            "created": True,
            "expectation_count": len(revised_expectations["expectations"]),  # type: ignore[arg-type]
        },
        "post_revision_required_reruns": [
            "Regenerate attribution_review with revised-attribution-expectations.json.",
            "Regenerate attribution_remediation_packet from the new attribution review.",
            "Run attribution_signoff only after the regenerated review is ready_for_human_signoff.",
            "Regenerate M4 phase_gate with the new review and signoff; checkpoint blockers remain independent.",
        ],
        "limitations": [
            "This receipt accepts a measured attribution label; it does not sign the attribution review.",
            "This receipt does not mutate the original expectations file.",
            "This receipt does not authorize M4 launch or formal GPU training.",
        ],
    }


def _write_report_bundle(
    *,
    report: Mapping[str, object],
    revised_expectations: Mapping[str, object],
    output_root: Path,
    archive_db: Path | None,
    cas_root: Path | None,
) -> dict[str, object]:
    destination = Path(output_root).resolve()
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    report_bytes = _canonical_json_bytes(report)
    expectations_bytes = _canonical_json_bytes(revised_expectations)
    markdown_bytes = _render_markdown(report).encode("utf-8")
    try:
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary.mkdir(mode=0o700)
        cas_refs: dict[str, str] = {}
        archive = ArchiveStore(archive_db) if archive_db is not None else None
        if archive is not None or cas_root is not None:
            root = cas_root if cas_root is not None else Path(archive_db).resolve().parent  # type: ignore[union-attr]
            cas = ContentAddressedStore(Path(root))
            for name, payload, media_type in (
                ("attribution_expectation_revision_json", report_bytes, "application/json"),
                ("revised_attribution_expectations_json", expectations_bytes, "application/json"),
                ("attribution_expectation_revision_markdown", markdown_bytes, "text/markdown"),
            ):
                ref = cas.put_bytes(payload, media_type=media_type).uri
                cas_refs[name] = ref
                if archive is not None:
                    archive.record_artifact_reference(ref)
        _write_bytes_atomic(temporary / "attribution-expectation-revision.json", report_bytes)
        _write_bytes_atomic(temporary / "revised-attribution-expectations.json", expectations_bytes)
        _write_bytes_atomic(temporary / "attribution-expectation-revision.md", markdown_bytes)
        manifest = {
            "schema_version": 1,
            "artifact_type": "wmloop-m1-attribution-expectation-revision-manifest",
            "state": report["state"],
            "human_resolution_confirmed": report["human_resolution_confirmed"],
            "manual_pass": report["manual_pass"],
            "human_signed_off": report["human_signed_off"],
            "reviewer": report["reviewer"],
            "accepted_resolution": report["accepted_resolution"],
            "active_expectations_mutated": report["active_expectations_mutated"],
            "m4_launch_allowed": report["m4_launch_allowed"],
            "formal_training_allowed": report["formal_training_allowed"],
            "source_expectations": report["source_expectations"],
            "source_attribution_review": report["source_attribution_review"],
            "source_attribution_remediation": report["source_attribution_remediation"],
            "revised_expectations": report["revised_expectations"],
            "report_path": str(destination / "attribution-expectation-revision.json"),
            "markdown_path": str(destination / "attribution-expectation-revision.md"),
            "cas_refs": cas_refs,
            "post_revision_required_reruns": report["post_revision_required_reruns"],
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


def _render_markdown(report: Mapping[str, Any]) -> str:
    resolution = report["accepted_resolution"]
    revised = report["revised_expectations"]
    lines = [
        "# M1 Attribution Expectation Revision",
        "",
        f"State: `{report['state']}`",
        f"Human resolution confirmed: `{report['human_resolution_confirmed']}`",
        f"Reviewer: `{report['reviewer']}`",
        f"M4 launch allowed: `{report['m4_launch_allowed']}`",
        "",
        "## Accepted Resolution",
        "",
        f"- Environment: `{resolution['environment']}`",
        f"- Previous dominant failure: `{resolution['previous_dominant_failure']}`",
        f"- Revised dominant failure: `{resolution['revised_dominant_failure']}`",
        f"- Rationale: {report['rationale']}",
        "",
        "## Revised Expectations",
        "",
        f"- Path: `{revised['path']}`",
        f"- SHA256: `{revised['sha256']}`",
        "",
        "## Required Reruns",
        "",
    ]
    lines.extend(f"- {item}" for item in report["post_revision_required_reruns"])
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in report["limitations"])
    return "\n".join(lines) + "\n"


def _read_json(path: Path, error: str) -> tuple[Mapping[str, Any], bytes, Path]:
    resolved = Path(path).resolve(strict=True)
    try:
        payload_bytes = resolved.read_bytes()
        payload = json.loads(payload_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise AttributionExpectationRevisionError(f"{error}:{resolved}") from exc
    if not isinstance(payload, Mapping):
        raise AttributionExpectationRevisionError(f"{error}:{resolved}")
    return payload, payload_bytes, resolved


def _int(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise AttributionExpectationRevisionError("ATTRIBUTION_EXPECTATION_REVISION_OUTPUT_EXISTS")
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
    run = commands.add_parser("run", help="create a human-approved attribution expectation revision")
    run.add_argument("--attribution-remediation-manifest", type=Path, required=True)
    run.add_argument("--source-expectations", type=Path, required=True)
    run.add_argument("--environment", required=True)
    run.add_argument("--accept-observed-dominant-failure", required=True)
    run.add_argument("--reviewer", required=True)
    run.add_argument("--rationale", required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--confirm-human-attribution-resolution", action="store_true")
    run.add_argument("--archive-db", type=Path)
    run.add_argument("--cas-root", type=Path)
    args = parser.parse_args(argv)
    if args.command == "run":
        manifest = run_attribution_expectation_revision(
            attribution_remediation_manifest=args.attribution_remediation_manifest,
            source_expectations=args.source_expectations,
            environment=args.environment,
            accept_observed_dominant_failure=args.accept_observed_dominant_failure,
            reviewer=args.reviewer,
            rationale=args.rationale,
            output_root=args.output_root,
            confirm_human_attribution_resolution=args.confirm_human_attribution_resolution,
            archive_db=args.archive_db,
            cas_root=args.cas_root,
        )
        print(json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
