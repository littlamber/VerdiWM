"""Generate the T4.5 verifier meta-validation report."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from wmloop.archive.store import ArchiveStore, ContentAddressedStore


class T45VerifierMetaValidationError(RuntimeError):
    """T4.5 verifier meta-validation failed closed."""


def run_t45_verifier_meta_validation(
    *,
    archive_db: Path,
    output_root: Path,
    trial_manifests: Sequence[Path] = (),
    campaign_manifests: Sequence[Path] = (),
    verifier_case_manifests: Sequence[Path] = (),
    cas_root: Path | None = None,
    truth_delta_threshold: float = 0.0,
    require_archive_coverage: bool = True,
) -> dict[str, object]:
    """Write a read-only verifier-vs-truth settlement report."""

    if not math.isfinite(truth_delta_threshold):
        raise T45VerifierMetaValidationError("T45_META_VALIDATION_THRESHOLD_INVALID")
    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise T45VerifierMetaValidationError("T45_META_VALIDATION_OUTPUT_EXISTS")
    archive = ArchiveStore(archive_db)
    cas_storage_root = Path(cas_root) if cas_root is not None else Path(archive_db).resolve().parent
    cas = ContentAddressedStore(cas_storage_root)
    archive_statistics = archive.archive_statistics()
    visible_settled_trials = archive.visible_settled_trials()
    excluded_archive_trials = _excluded_archive_trials(visible_settled_trials)
    archive_statistics = {
        **archive_statistics,
        "verifier_eligible_settled_trials": int(archive_statistics.get("settled_trials", 0)) - len(excluded_archive_trials),
    }
    sources: list[dict[str, object]] = []
    trials = []
    for manifest_path in trial_manifests:
        source, trial = _load_trial_manifest(manifest_path, cas=cas, archive=archive)
        sources.append(source)
        trials.append(trial)
    for campaign_path in campaign_manifests:
        campaign_source, campaign_trials = _load_campaign_manifest(campaign_path, cas=cas, archive=archive)
        sources.append(campaign_source)
        trials.extend(campaign_trials)
    verifier_cases = []
    for manifest_path in verifier_case_manifests:
        verifier_source, loaded = _load_verifier_case_manifest(manifest_path, cas=cas, archive=archive)
        sources.append(verifier_source)
        verifier_cases.extend(loaded)
    trial_records = [_validation_record(trial, truth_delta_threshold=truth_delta_threshold) for trial in trials]
    verifier_records = [_verifier_case_record(case) for case in verifier_cases]
    report = _report(
        trial_records=trial_records,
        verifier_records=verifier_records,
        sources=sources,
        archive_statistics=archive_statistics,
        excluded_archive_trials=excluded_archive_trials,
        truth_delta_threshold=truth_delta_threshold,
        require_archive_coverage=require_archive_coverage,
    )
    return _write_report_bundle(report=report, output_root=destination, cas=cas, archive=archive)


def _load_campaign_manifest(
    path: Path,
    *,
    cas: ContentAddressedStore,
    archive: ArchiveStore,
) -> tuple[dict[str, object], list[Mapping[str, Any]]]:
    source = _load_source(path, cas=cas, archive=archive)
    payload = source["payload"]
    if payload.get("artifact_type") != "wmloop-training-eval-limited-campaign-manifest":
        raise T45VerifierMetaValidationError(f"T45_META_VALIDATION_CAMPAIGN_INVALID:{path}")
    records = payload.get("records")
    if not isinstance(records, list):
        raise T45VerifierMetaValidationError(f"T45_META_VALIDATION_CAMPAIGN_RECORDS_INVALID:{path}")
    trials: list[Mapping[str, Any]] = []
    trial_sources = []
    for item in records:
        if not isinstance(item, Mapping) or item.get("state") != "ready":
            continue
        manifest_path = item.get("manifest_path")
        if not isinstance(manifest_path, str) or not manifest_path:
            raise T45VerifierMetaValidationError("T45_META_VALIDATION_CAMPAIGN_TRIAL_MANIFEST_MISSING")
        trial_source, trial = _load_trial_manifest(Path(manifest_path), cas=cas, archive=archive)
        trial_sources.append(trial_source["summary"])
        trials.append(trial)
    summary = source["summary"]
    if isinstance(summary, dict):
        summary["trial_sources"] = trial_sources
    return source, trials


def _load_trial_manifest(
    path: Path,
    *,
    cas: ContentAddressedStore,
    archive: ArchiveStore,
) -> tuple[dict[str, object], Mapping[str, Any]]:
    source = _load_source(path, cas=cas, archive=archive)
    payload = source["payload"]
    if payload.get("artifact_type") != "wmloop-m3-training-eval-smoke-manifest":
        raise T45VerifierMetaValidationError(f"T45_META_VALIDATION_TRIAL_INVALID:{path}")
    return source, payload


def _load_verifier_case_manifest(
    path: Path,
    *,
    cas: ContentAddressedStore,
    archive: ArchiveStore,
) -> tuple[dict[str, object], list[Mapping[str, Any]]]:
    source = _load_source(path, cas=cas, archive=archive)
    payload = source["payload"]
    artifact_type = payload.get("artifact_type")
    if artifact_type != "wmloop-m3-judge-gate-smoke-manifest":
        raise T45VerifierMetaValidationError(f"T45_META_VALIDATION_VERIFIER_CASE_INVALID:{path}")
    report_path = payload.get("report_path")
    if not isinstance(report_path, str) or not report_path:
        raise T45VerifierMetaValidationError("T45_META_VALIDATION_VERIFIER_CASE_REPORT_MISSING")
    report_source = _load_source(Path(report_path), cas=cas, archive=archive)
    report = report_source["payload"]
    if report.get("artifact_type") != "wmloop-m3-judge-gate-smoke-report":
        raise T45VerifierMetaValidationError("T45_META_VALIDATION_VERIFIER_CASE_REPORT_INVALID")
    records = report.get("records")
    if not isinstance(records, list):
        raise T45VerifierMetaValidationError("T45_META_VALIDATION_VERIFIER_CASE_RECORDS_INVALID")
    summary = source["summary"]
    if isinstance(summary, dict):
        summary["verifier_case_source"] = report_source["summary"]
    loaded = []
    for record in records:
        if not isinstance(record, Mapping):
            raise T45VerifierMetaValidationError("T45_META_VALIDATION_VERIFIER_CASE_RECORD_INVALID")
        loaded.append({**dict(record), "source_report_path": report_path})
    return source, loaded


def _validation_record(trial: Mapping[str, Any], *, truth_delta_threshold: float) -> dict[str, object]:
    verdict = _string_field(trial, "verdict")
    delta = _primary_delta(trial)
    truth = "true_improvement" if delta > truth_delta_threshold else "no_improvement"
    agreement = None
    if verdict == "ACCEPT":
        agreement = truth == "true_improvement"
    elif verdict == "REJECT":
        agreement = truth == "no_improvement"
    return {
        "case_source": "training_trial",
        "proposal_id": trial.get("proposal_id"),
        "case_name": None,
        "trial_manifest_artifact_type": trial.get("artifact_type"),
        "state": trial.get("state"),
        "environment": trial.get("environment"),
        "goal_id": trial.get("goal_id"),
        "primary_metric": trial.get("primary_metric"),
        "verdict": verdict,
        "truth_label": truth,
        "delta_primary_metric": delta,
        "agreement": agreement,
        "report_path": trial.get("report_path"),
        "receipt_ref": trial.get("receipt_ref"),
        "verdict_ref": trial.get("verdict_ref"),
        "action_following_gate": trial.get("action_following_gate", {}),
    }


def _verifier_case_record(case: Mapping[str, Any]) -> dict[str, object]:
    verdict_payload = case.get("verdict")
    expected = case.get("expected")
    if not isinstance(verdict_payload, Mapping) or not isinstance(expected, Mapping):
        raise T45VerifierMetaValidationError("T45_META_VALIDATION_VERIFIER_CASE_VERDICT_INVALID")
    verdict = verdict_payload.get("verdict")
    expected_verdict = expected.get("verdict")
    if not isinstance(verdict, str) or not verdict or not isinstance(expected_verdict, str) or not expected_verdict:
        raise T45VerifierMetaValidationError("T45_META_VALIDATION_VERIFIER_CASE_VERDICT_INVALID")
    expected_violation = expected.get("violation")
    observed_violation = verdict_payload.get("violation")
    delta = _primary_delta_or_zero(verdict_payload)
    return {
        "case_source": "verifier_gate_case",
        "proposal_id": verdict_payload.get("proposal_id"),
        "case_name": case.get("case"),
        "trial_manifest_artifact_type": "wmloop-m3-judge-gate-smoke-report",
        "state": "ready" if case.get("passed") is True else "failed",
        "environment": None,
        "goal_id": None,
        "primary_metric": next(iter(verdict_payload.get("delta_m_ver", {}) or {}), "verifier_gate_case"),
        "verdict": verdict,
        "truth_label": f"expected_{expected_verdict}",
        "delta_primary_metric": delta,
        "agreement": verdict == expected_verdict and observed_violation == expected_violation,
        "report_path": case.get("source_report_path"),
        "receipt_ref": None,
        "verdict_ref": case.get("verdict_ref"),
        "action_following_gate": verdict_payload.get("action_following_gate", {}),
        "observed_violation": observed_violation,
        "expected_violation": expected_violation,
    }


def _primary_delta(trial: Mapping[str, Any]) -> float:
    deltas = trial.get("delta_m_ver")
    if not isinstance(deltas, Mapping) or not deltas:
        raise T45VerifierMetaValidationError("T45_META_VALIDATION_TRIAL_DELTA_MISSING")
    primary = trial.get("primary_metric")
    candidates: list[object] = []
    if isinstance(primary, str) and primary:
        candidates.append(deltas.get(primary))
    candidates.extend(deltas.values())
    for value in candidates:
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
            return float(value)
    raise T45VerifierMetaValidationError("T45_META_VALIDATION_TRIAL_DELTA_INVALID")


def _primary_delta_or_zero(payload: Mapping[str, Any]) -> float:
    deltas = payload.get("delta_m_ver")
    if isinstance(deltas, Mapping):
        for value in deltas.values():
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
                return float(value)
    return 0.0


def _report(
    *,
    trial_records: Sequence[Mapping[str, object]],
    verifier_records: Sequence[Mapping[str, object]],
    sources: Sequence[Mapping[str, object]],
    archive_statistics: Mapping[str, int],
    excluded_archive_trials: Sequence[str],
    truth_delta_threshold: float,
    require_archive_coverage: bool,
) -> dict[str, object]:
    records = [*trial_records, *verifier_records]
    settled_count = int(archive_statistics.get("settled_trials", 0))
    eligible_settled_count = int(archive_statistics.get("verifier_eligible_settled_trials", settled_count))
    coverage_complete = len(trial_records) == eligible_settled_count if require_archive_coverage else True
    matrix = _confusion_matrix(records)
    verdict_counts = _counts(record["verdict"] for record in records)
    truth_counts = _counts(record["truth_label"] for record in records)
    examples = _case_examples(records)
    blockers = []
    if not trial_records:
        blockers.append("no_trial_evidence")
    if not coverage_complete:
        blockers.append("settled_trial_coverage_incomplete")
    if "INCONCLUSIVE" not in {item["verdict"] for item in examples}:
        blockers.append("inconclusive_real_case_missing")
    if "VOID" not in {item["verdict"] for item in examples}:
        blockers.append("void_real_case_missing")
    ready = not blockers
    return {
        "schema_version": 1,
        "artifact_type": "wmloop-t4-5-verifier-meta-validation",
        "state": "ready" if ready else "blocked",
        "verifier_meta_validation_ready": ready,
        "confusion_matrix_ready": bool(records),
        "settled_trial_count": len(trial_records),
        "verifier_case_count": len(verifier_records),
        "archive_settled_trial_count": settled_count,
        "archive_verifier_eligible_settled_trial_count": eligible_settled_count,
        "excluded_archive_trial_count": len(excluded_archive_trials),
        "excluded_archive_trials": list(excluded_archive_trials),
        "coverage_complete": coverage_complete,
        "truth_delta_threshold": truth_delta_threshold,
        "require_archive_coverage": require_archive_coverage,
        "trial_count": len(records),
        "verdict_counts": verdict_counts,
        "truth_counts": truth_counts,
        "confusion_matrix": matrix,
        "rates": _rates(records),
        "case_examples": examples,
        "records": [dict(record) for record in records],
        "blockers": blockers,
        "sources": [source["summary"] for source in sources],
        "limitations": [
            "Truth labels are derived from frozen ACWM-Phys paired metric deltas in the supplied trial manifests.",
            "Verifier gate cases are constructed fail-closed judge evidence and only validate verifier behavior; they do not count as model-quality training trials.",
            "A ready T4.5 report requires coverage of all visible verifier-eligible settled trials plus at least one real INCONCLUSIVE and one real VOID case.",
            "Archive settlement smoke records are excluded from verifier-eligible coverage because they are archive self-checks, not closed-loop training-eval trials.",
        ],
    }


def _excluded_archive_trials(trial_ids: Sequence[str]) -> list[str]:
    return sorted(trial_id for trial_id in trial_ids if trial_id.startswith("archive-settlement-smoke"))


def _confusion_matrix(records: Sequence[Mapping[str, object]]) -> dict[str, dict[str, int]]:
    matrix: dict[str, dict[str, int]] = {}
    for record in records:
        verdict = str(record["verdict"])
        truth = str(record["truth_label"])
        matrix.setdefault(verdict, {})
        matrix[verdict][truth] = matrix[verdict].get(truth, 0) + 1
    return matrix


def _rates(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    training_records = [record for record in records if record.get("case_source") == "training_trial"]
    accepted = [record for record in training_records if record["verdict"] == "ACCEPT"]
    rejected = [record for record in training_records if record["verdict"] == "REJECT"]
    decided = accepted + rejected
    true_improvements = [record for record in training_records if record["truth_label"] == "true_improvement"]
    accepted_true = [record for record in accepted if record["truth_label"] == "true_improvement"]
    rejected_true = [record for record in rejected if record["truth_label"] == "true_improvement"]
    agreed = [record for record in decided if record.get("agreement") is True]
    return {
        "accept_truth_rate": _ratio(len(accepted_true), len(accepted)),
        "reject_false_negative_rate": _ratio(len(rejected_true), len(true_improvements)),
        "decided_agreement_rate": _ratio(len(agreed), len(decided)),
        "inconclusive_rate": _ratio(len([record for record in training_records if record["verdict"] == "INCONCLUSIVE"]), len(training_records)),
        "void_rate": _ratio(len([record for record in training_records if record["verdict"] == "VOID"]), len(training_records)),
    }


def _case_examples(records: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    examples = []
    for verdict in ("INCONCLUSIVE", "VOID"):
        match = next((record for record in records if record["verdict"] == verdict), None)
        if match is not None:
            examples.append(dict(match))
    return examples


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _counts(values: Sequence[object]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _load_source(path: Path, *, cas: ContentAddressedStore, archive: ArchiveStore) -> dict[str, object]:
    resolved = Path(path).resolve(strict=True)
    try:
        payload_bytes = resolved.read_bytes()
        payload = json.loads(payload_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise T45VerifierMetaValidationError(f"T45_META_VALIDATION_SOURCE_INVALID:{resolved}") from exc
    if not isinstance(payload, dict):
        raise T45VerifierMetaValidationError(f"T45_META_VALIDATION_SOURCE_NOT_OBJECT:{resolved}")
    ref = cas.put_bytes(payload_bytes, media_type="application/json").uri
    archive.record_artifact_reference(ref)
    return {
        "payload": payload,
        "summary": {
            "path": str(resolved),
            "cas_ref": ref,
            "artifact_type": payload.get("artifact_type"),
            "state": payload.get("state"),
        },
    }


def _write_report_bundle(
    *,
    report: Mapping[str, object],
    output_root: Path,
    cas: ContentAddressedStore,
    archive: ArchiveStore,
) -> dict[str, object]:
    destination = Path(output_root).resolve()
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    try:
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary.mkdir(mode=0o700)
        report_bytes = _canonical_json_bytes(report)
        markdown_bytes = _render_markdown(report).encode("utf-8")
        csv_bytes = _render_csv(report).encode("utf-8")
        _write_bytes_atomic(temporary / "verifier-meta-validation.json", report_bytes)
        _write_bytes_atomic(temporary / "verifier-meta-validation.md", markdown_bytes)
        _write_bytes_atomic(temporary / "verifier-meta-validation.csv", csv_bytes)
        report_ref = cas.put_bytes(report_bytes, media_type="application/json").uri
        markdown_ref = cas.put_bytes(markdown_bytes, media_type="text/markdown").uri
        csv_ref = cas.put_bytes(csv_bytes, media_type="text/csv").uri
        for ref in (report_ref, markdown_ref, csv_ref):
            archive.record_artifact_reference(ref)
        manifest = {
            "schema_version": 1,
            "artifact_type": "wmloop-t4-5-verifier-meta-validation-manifest",
            "state": report["state"],
            "verifier_meta_validation_ready": report["verifier_meta_validation_ready"],
            "confusion_matrix_ready": report["confusion_matrix_ready"],
            "settled_trial_count": report["settled_trial_count"],
            "verifier_case_count": report["verifier_case_count"],
            "archive_settled_trial_count": report["archive_settled_trial_count"],
            "archive_verifier_eligible_settled_trial_count": report["archive_verifier_eligible_settled_trial_count"],
            "excluded_archive_trial_count": report["excluded_archive_trial_count"],
            "excluded_archive_trials": report["excluded_archive_trials"],
            "case_examples": report["case_examples"],
            "blockers": report["blockers"],
            "report_path": str(destination / "verifier-meta-validation.json"),
            "markdown_path": str(destination / "verifier-meta-validation.md"),
            "csv_path": str(destination / "verifier-meta-validation.csv"),
            "cas_refs": {
                "verifier_meta_validation_json": report_ref,
                "verifier_meta_validation_markdown": markdown_ref,
                "verifier_meta_validation_csv": csv_ref,
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


def _render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# T4.5 Verifier Meta-Validation",
        "",
        f"State: `{report['state']}`",
        f"Ready: `{report['verifier_meta_validation_ready']}`",
        f"Trials covered: `{report['settled_trial_count']}`",
        f"Verifier cases: `{report['verifier_case_count']}`",
        f"Archive settled trials: `{report['archive_settled_trial_count']}`",
        f"Verifier-eligible archive trials: `{report['archive_verifier_eligible_settled_trial_count']}`",
        f"Excluded archive trials: `{report['excluded_archive_trial_count']}`",
        f"Coverage complete: `{report['coverage_complete']}`",
        "",
        "## Rates",
        "",
    ]
    for key, value in report["rates"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Confusion Matrix", ""])
    matrix = report["confusion_matrix"]
    truth_labels = sorted({truth for row in matrix.values() for truth in row})
    lines.append("| Verdict | " + " | ".join(f"`{truth}`" for truth in truth_labels) + " |")
    lines.append("|:--" + "|--:" * len(truth_labels) + "|")
    for verdict in sorted(matrix):
        row = matrix[verdict]
        lines.append("| `" + verdict + "` | " + " | ".join(f"`{row.get(truth, 0)}`" for truth in truth_labels) + " |")
    if report["blockers"]:
        lines.extend(["", "## Blockers", ""])
        for blocker in report["blockers"]:
            lines.append(f"- `{blocker}`")
    return "\n".join(lines) + "\n"


def _render_csv(report: Mapping[str, Any]) -> str:
    import io

    handle = io.StringIO()
    writer = csv.DictWriter(
        handle,
        fieldnames=[
            "proposal_id",
            "case_source",
            "case_name",
            "environment",
            "goal_id",
            "primary_metric",
            "verdict",
            "truth_label",
            "delta_primary_metric",
            "agreement",
            "report_path",
            "receipt_ref",
            "verdict_ref",
        ],
    )
    writer.writeheader()
    for record in report.get("records", []):
        if not isinstance(record, Mapping):
            continue
        writer.writerow(
            {
                "proposal_id": record.get("proposal_id", ""),
                "case_source": record.get("case_source", ""),
                "case_name": record.get("case_name", ""),
                "environment": record.get("environment", ""),
                "goal_id": record.get("goal_id", ""),
                "primary_metric": record.get("primary_metric", ""),
                "verdict": record.get("verdict", ""),
                "truth_label": record.get("truth_label", ""),
                "delta_primary_metric": record.get("delta_primary_metric", ""),
                "agreement": record.get("agreement", ""),
                "report_path": record.get("report_path", ""),
                "receipt_ref": record.get("receipt_ref", ""),
                "verdict_ref": record.get("verdict_ref", ""),
            }
        )
    return handle.getvalue()


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise T45VerifierMetaValidationError("T45_META_VALIDATION_OUTPUT_EXISTS")
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


def _string_field(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise T45VerifierMetaValidationError(f"T45_META_VALIDATION_FIELD_INVALID:{key}")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="generate T4.5 verifier meta-validation")
    run.add_argument("--archive-db", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--trial-manifest", action="append", type=Path, default=[])
    run.add_argument("--campaign-manifest", action="append", type=Path, default=[])
    run.add_argument("--verifier-case-manifest", action="append", type=Path, default=[])
    run.add_argument("--cas-root", type=Path)
    run.add_argument("--truth-delta-threshold", type=float, default=0.0)
    run.add_argument("--no-require-archive-coverage", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "run":
        manifest = run_t45_verifier_meta_validation(
            archive_db=args.archive_db,
            output_root=args.output_root,
            trial_manifests=args.trial_manifest,
            campaign_manifests=args.campaign_manifest,
            verifier_case_manifests=args.verifier_case_manifest,
            cas_root=args.cas_root,
            truth_delta_threshold=args.truth_delta_threshold,
            require_archive_coverage=not args.no_require_archive_coverage,
        )
        print(json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
        return 0
    raise T45VerifierMetaValidationError("T45_META_VALIDATION_COMMAND_INVALID")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
