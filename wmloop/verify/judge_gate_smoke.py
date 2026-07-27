"""M3 verifier gate smoke for fail-closed verdict behavior."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Mapping

from wmloop.archive.store import ArchiveStore, ContentAddressedStore
from wmloop.verify.judge import VerificationEvidence, judge


class JudgeGateSmokeError(RuntimeError):
    """Verifier gate smoke could not be generated."""


REQUIRED_GATE_CASES = (
    "accept_flow",
    "eval_path_tamper_g1",
    "heldout_split_contamination_g2",
    "hardcoded_metric_diff_g3",
    "static_degradation_af_gate",
    "replication_missing",
)


def run_judge_gate_smoke(
    *,
    output_root: Path,
    archive_db: Path | None = None,
    cas_root: Path | None = None,
) -> dict[str, object]:
    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise JudgeGateSmokeError("JUDGE_GATE_SMOKE_OUTPUT_EXISTS")
    archive = ArchiveStore(archive_db) if archive_db is not None else None
    cas_storage_root = cas_root if cas_root is not None else (Path(archive_db).resolve().parent if archive_db is not None else destination.parent)
    cas = ContentAddressedStore(cas_storage_root)
    cases = _cases()
    records = []
    for name, evidence, expected in cases:
        result = judge(evidence).to_dict()
        passed = result["verdict"] == expected["verdict"] and result["violation"] == expected["violation"]
        if name == "static_degradation_af_gate":
            passed = passed and result["delta_m_ver"] == {"auc_psnr_16_64": 0.0}
        verdict_ref = _put_json(cas, result, archive=archive)
        records.append(
            {
                "case": name,
                "expected": expected,
                "passed": passed,
                "verdict": result,
                "verdict_ref": verdict_ref,
            }
        )
    if not all(record["passed"] for record in records):
        state = "failed"
    else:
        state = "ready"
    report = {
        "schema_version": 1,
        "artifact_type": "wmloop-m3-judge-gate-smoke-report",
        "state": state,
        "case_count": len(records),
        "passed_count": sum(1 for record in records if record["passed"]),
        "case_names": [record["case"] for record in records],
        "required_gate_case_names": list(REQUIRED_GATE_CASES),
        "records": records,
        "cas_root": str(Path(cas_storage_root).resolve()),
        "limitations": [
            "This is constructed verifier evidence; it does not execute provider training or evaluation.",
            "The smoke proves fail-closed verdict behavior for representative G1, G2, G3, AF, replication, and ACCEPT paths.",
        ],
    }
    return _write_report_bundle(report=report, output_root=destination, cas=cas, archive=archive)


def _cases() -> tuple[tuple[str, VerificationEvidence, dict[str, object]], ...]:
    base = {
        "proposal_id": "m3-judge-smoke",
        "readonly_evaluator_verified": True,
        "accept_split_verified": True,
        "extended_horizon_verified": True,
        "diff_audit_passed": True,
        "evidence_complete": True,
        "accept_metric_deltas": {"auc_psnr_16_64": 1.0},
        "replication_deltas": [0.9, 1.0, 1.1],
        "action_following_observed": 0.61,
        "action_following_threshold": 0.55,
    }
    return (
        ("accept_flow", VerificationEvidence(**base), {"verdict": "ACCEPT", "violation": None}),
        (
            "eval_path_tamper_g1",
            VerificationEvidence(**{**base, "proposal_id": "m3-judge-smoke-g1", "readonly_evaluator_verified": False}),
            {"verdict": "VOID", "violation": "G1_READONLY_VIOLATION"},
        ),
        (
            "hardcoded_metric_diff_g3",
            VerificationEvidence(**{**base, "proposal_id": "m3-judge-smoke-g3", "diff_audit_passed": False}),
            {"verdict": "VOID", "violation": "G3_DIFF_AUDIT_VIOLATION"},
        ),
        (
            "heldout_split_contamination_g2",
            VerificationEvidence(**{**base, "proposal_id": "m3-judge-smoke-g2", "accept_split_verified": False}),
            {"verdict": "INCONCLUSIVE", "violation": None},
        ),
        (
            "static_degradation_af_gate",
            VerificationEvidence(**{**base, "proposal_id": "m3-judge-smoke-af", "action_following_observed": 0.1}),
            {"verdict": "REJECT", "violation": "AF_GATE_FAILED"},
        ),
        (
            "replication_missing",
            VerificationEvidence(**{**base, "proposal_id": "m3-judge-smoke-repl", "replication_deltas": [1.0]}),
            {"verdict": "INCONCLUSIVE", "violation": None},
        ),
    )


def _write_report_bundle(
    *,
    report: Mapping[str, object],
    output_root: Path,
    cas: ContentAddressedStore,
    archive: ArchiveStore | None,
) -> dict[str, object]:
    destination = Path(output_root).resolve()
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    report_bytes = _canonical_json_bytes(report)
    markdown_bytes = _render_markdown(report).encode("utf-8")
    try:
        temporary.mkdir(mode=0o700)
        _write_bytes_atomic(temporary / "judge-gate-smoke.json", report_bytes)
        _write_bytes_atomic(temporary / "judge-gate-smoke.md", markdown_bytes)
        report_ref = cas.put_bytes(report_bytes, media_type="application/json").uri
        markdown_ref = cas.put_bytes(markdown_bytes, media_type="text/markdown").uri
        if archive is not None:
            archive.record_artifact_reference(report_ref)
            archive.record_artifact_reference(markdown_ref)
        manifest = {
            "schema_version": 1,
            "artifact_type": "wmloop-m3-judge-gate-smoke-manifest",
            "state": report["state"],
            "case_count": report["case_count"],
            "passed_count": report["passed_count"],
            "case_names": report["case_names"],
            "required_gate_case_names": report["required_gate_case_names"],
            "report_path": str(destination / "judge-gate-smoke.json"),
            "markdown_path": str(destination / "judge-gate-smoke.md"),
            "cas_refs": {
                "judge_gate_smoke_json": report_ref,
                "judge_gate_smoke_markdown": markdown_ref,
            },
            "limitations": report["limitations"],
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
        "# M3 Judge Gate Smoke",
        "",
        f"State: `{report['state']}`",
        f"Passed cases: `{report['passed_count']}/{report['case_count']}`",
        "",
        "| Case | Passed | Verdict | Violation |",
        "|:--|:--|:--|:--|",
    ]
    for record in report["records"]:
        verdict = record["verdict"]
        lines.append(f"| {record['case']} | {record['passed']} | {verdict['verdict']} | {verdict['violation']} |")
    lines.extend(["", "## Limitations", ""])
    for item in report["limitations"]:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def _put_json(cas: ContentAddressedStore, payload: Mapping[str, object], *, archive: ArchiveStore | None) -> str:
    ref = cas.put_bytes(_canonical_json_bytes(payload), media_type="application/json").uri
    if archive is not None:
        archive.record_artifact_reference(ref)
    return ref


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run constructed judge gate smoke")
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--archive-db", type=Path)
    run.add_argument("--cas-root", type=Path)
    args = parser.parse_args(argv)
    if args.command == "run":
        manifest = run_judge_gate_smoke(output_root=args.output_root, archive_db=args.archive_db, cas_root=args.cas_root)
        print(json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
