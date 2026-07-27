"""Draft human attribution review packets from measured failure reports."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from wmloop.archive.store import ArchiveStore, ContentAddressedStore
from wmloop.contracts import ContractValidationError, validate_document


class AttributionReviewError(RuntimeError):
    """Attribution review draft generation failed closed."""


def generate_attribution_review(
    *,
    failure_batch_manifest: Path,
    expectations_path: Path,
    output_root: Path,
    archive_db: Path | None = None,
    cas_root: Path | None = None,
) -> dict[str, object]:
    """Write a review draft without claiming human sign-off."""

    batch = _load_batch(failure_batch_manifest)
    expectations = _load_expectations(expectations_path)
    reports = [_load_report(Path(item["failure_report_path"])) for item in batch["reports"]]
    records = [_review_record(report) for report in reports]
    expected_by_env = {str(item["environment"]): item for item in expectations["expectations"]}
    expectation_results = [_expectation_result(record, expected_by_env.get(str(record["environment"]))) for record in records]
    missing_expected = [
        {
            "environment": environment,
            "expected_dominant_failure": expectation["dominant_failure"],
            "state": "missing_report",
        }
        for environment, expectation in sorted(expected_by_env.items())
        if environment not in {str(record["environment"]) for record in records}
    ]
    mismatches = [item for item in expectation_results if item["state"] == "mismatch"] + missing_expected
    required = int(expectations["required_review_count"])
    reviewed_count = len(records)
    ready_for_human_signoff = reviewed_count >= required and not mismatches
    report = {
        "schema_version": 1,
        "artifact_type": "wmloop-m1-attribution-review-draft",
        "state": "ready_for_human_signoff" if ready_for_human_signoff else "requires_attention",
        "manual_pass": False,
        "human_signed_off": False,
        "failure_batch_manifest": str(Path(failure_batch_manifest).resolve()),
        "expectations_path": str(Path(expectations_path).resolve()),
        "required_review_count": required,
        "reviewed_count": reviewed_count,
        "expectation_mismatch_count": len(mismatches),
        "records": records,
        "expectation_results": expectation_results,
        "missing_or_mismatched_expectations": mismatches,
        "blocked_records_from_batch": batch.get("blocked_records", []),
        "notes": [
            "This artifact is a Codex-generated review packet; it is not a human sign-off.",
            "manual_pass remains false until a human reviewer explicitly confirms the attribution decisions.",
        ],
    }
    markdown = _render_markdown(report)
    return _write_bundle(report=report, markdown=markdown, output_root=output_root, archive_db=archive_db, cas_root=cas_root)


def _load_batch(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AttributionReviewError("ATTRIBUTION_REVIEW_BATCH_INVALID") from exc
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema_version") != 1
        or payload.get("artifact_type") != "wmloop-m1-raw-failure-report-batch"
        or not isinstance(payload.get("reports"), list)
    ):
        raise AttributionReviewError("ATTRIBUTION_REVIEW_BATCH_INVALID")
    return payload


def _load_expectations(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AttributionReviewError("ATTRIBUTION_REVIEW_EXPECTATIONS_INVALID") from exc
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema_version") != 1
        or payload.get("artifact_type") != "wmloop-m1-attribution-review-expectations"
        or not isinstance(payload.get("required_review_count"), int)
        or payload["required_review_count"] < 1
        or not isinstance(payload.get("expectations"), list)
    ):
        raise AttributionReviewError("ATTRIBUTION_REVIEW_EXPECTATIONS_INVALID")
    seen: set[str] = set()
    for item in payload["expectations"]:
        if (
            not isinstance(item, Mapping)
            or not isinstance(item.get("environment"), str)
            or not item["environment"]
            or not isinstance(item.get("dominant_failure"), str)
            or not item["dominant_failure"]
        ):
            raise AttributionReviewError("ATTRIBUTION_REVIEW_EXPECTATIONS_INVALID")
        environment = str(item["environment"])
        if environment in seen:
            raise AttributionReviewError("ATTRIBUTION_REVIEW_EXPECTATION_DUPLICATE")
        seen.add(environment)
    return payload


def _load_report(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        validate_document("failure_report", payload)
    except (OSError, json.JSONDecodeError, ContractValidationError) as exc:
        raise AttributionReviewError("ATTRIBUTION_REVIEW_FAILURE_REPORT_INVALID") from exc
    return payload


def _review_record(report: Mapping[str, Any]) -> dict[str, object]:
    action = report["action_following"]
    ood = report["ood_profile"]
    appearance = report["appearance_drift"]
    horizon = report["horizon_curve"]
    return {
        "environment": report["env"],
        "dominant_failure": report["dominant_failure"],
        "dominant_failure_candidates": report["dominant_failure_candidates"],
        "low_motion_ssim_64": appearance["low_motion_ssim_64"],
        "inv_dyn_acc_perframe": action["inv_dyn_acc_perframe"],
        "no_action_delta_psnr": action["no_action_delta_psnr"],
        "inverse_dynamics_r2": action["inverse_dynamics_r2"],
        "inverse_dynamics_low_confidence": action["low_confidence"],
        "ood_gap": ood["gap"],
        "worst_ood_condition": ood["worst_ood_condition"],
        "drift_slope": horizon["drift_slope"],
        "segment_worst_drop": horizon["segment_drift"]["worst_drop"],
    }


def _expectation_result(record: Mapping[str, object], expectation: Mapping[str, Any] | None) -> dict[str, object]:
    if expectation is None:
        return {
            "environment": record["environment"],
            "state": "not_specified",
            "observed_dominant_failure": record["dominant_failure"],
            "expected_dominant_failure": None,
        }
    expected = str(expectation["dominant_failure"])
    observed = str(record["dominant_failure"])
    return {
        "environment": record["environment"],
        "state": "match" if observed == expected else "mismatch",
        "observed_dominant_failure": observed,
        "expected_dominant_failure": expected,
        "source": expectation.get("source"),
    }


def _render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# M1 Attribution Review Draft",
        "",
        f"State: `{report['state']}`",
        f"Manual pass: `{report['manual_pass']}`",
        f"Reviewed: `{report['reviewed_count']}/{report['required_review_count']}`",
        f"Expectation mismatches: `{report['expectation_mismatch_count']}`",
        "",
        "| Environment | Observed | Candidates | OOD Gap | No-Action Delta | InvDyn R2 | Low Conf |",
        "|:--|:--|:--|--:|--:|--:|:--|",
    ]
    for record in report["records"]:
        lines.append(
            "| {environment} | {dominant_failure} | {candidates} | {ood_gap:.4f} | {delta:.4f} | {r2:.4f} | {low_conf} |".format(
                environment=record["environment"],
                dominant_failure=record["dominant_failure"],
                candidates=",".join(record["dominant_failure_candidates"]),
                ood_gap=float(record["ood_gap"]),
                delta=float(record["no_action_delta_psnr"]),
                r2=float(record["inverse_dynamics_r2"]),
                low_conf=str(record["inverse_dynamics_low_confidence"]).lower(),
            )
        )
    if report["missing_or_mismatched_expectations"]:
        lines.extend(["", "## Missing Or Mismatched Expectations", ""])
        for item in report["missing_or_mismatched_expectations"]:
            lines.append(
                "- `{environment}` expected `{expected}` but observed `{observed}` ({state}).".format(
                    environment=item["environment"],
                    expected=item.get("expected_dominant_failure"),
                    observed=item.get("observed_dominant_failure"),
                    state=item["state"],
                )
            )
    lines.extend(["", "## Notes", ""])
    for note in report["notes"]:
        lines.append(f"- {note}")
    return "\n".join(lines) + "\n"


def _write_bundle(
    *,
    report: Mapping[str, object],
    markdown: str,
    output_root: Path,
    archive_db: Path | None,
    cas_root: Path | None,
) -> dict[str, object]:
    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise AttributionReviewError("ATTRIBUTION_REVIEW_OUTPUT_EXISTS")
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    report_bytes = _canonical_json_bytes(report)
    markdown_bytes = markdown.encode("utf-8")
    cas_refs: dict[str, str] = {}
    try:
        temporary.mkdir(mode=0o700)
        _write_bytes_atomic(temporary / "attribution-review.json", report_bytes)
        _write_bytes_atomic(temporary / "attribution-review.md", markdown_bytes)
        archive = ArchiveStore(archive_db) if archive_db is not None else None
        if archive is not None or cas_root is not None:
            root = cas_root if cas_root is not None else Path(archive_db).resolve().parent  # type: ignore[union-attr]
            cas = ContentAddressedStore(Path(root))
            for name, payload, media_type in (
                ("attribution_review_json", report_bytes, "application/json"),
                ("attribution_review_markdown", markdown_bytes, "text/markdown"),
            ):
                ref = cas.put_bytes(payload, media_type=media_type).uri
                cas_refs[name] = ref
                if archive is not None:
                    archive.record_artifact_reference(ref)
        manifest = {
            "schema_version": 1,
            "artifact_type": "wmloop-m1-attribution-review-manifest",
            "state": report["state"],
            "manual_pass": report["manual_pass"],
            "human_signed_off": report["human_signed_off"],
            "reviewed_count": report["reviewed_count"],
            "required_review_count": report["required_review_count"],
            "expectation_mismatch_count": report["expectation_mismatch_count"],
            "report_path": str(destination / "attribution-review.json"),
            "markdown_path": str(destination / "attribution-review.md"),
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
        raise AttributionReviewError("ATTRIBUTION_REVIEW_OUTPUT_EXISTS")
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate", help="generate an attribution review draft")
    generate.add_argument("--failure-batch-manifest", type=Path, required=True)
    generate.add_argument("--expectations", type=Path, required=True)
    generate.add_argument("--output-root", type=Path, required=True)
    generate.add_argument("--archive-db", type=Path)
    generate.add_argument("--cas-root", type=Path)
    args = parser.parse_args(argv)
    if args.command == "generate":
        manifest = generate_attribution_review(
            failure_batch_manifest=args.failure_batch_manifest,
            expectations_path=args.expectations,
            output_root=args.output_root,
            archive_db=args.archive_db,
            cas_root=args.cas_root,
        )
        print(json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
