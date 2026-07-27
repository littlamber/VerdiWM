"""M3 strict acceptance audit over existing evidence manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Mapping

from wmloop.archive.store import ArchiveStore, ContentAddressedStore


class M3AcceptanceAuditError(RuntimeError):
    """M3 acceptance evidence could not be audited."""


REQUIRED_VERIFIER_GATE_CASES = frozenset(
    {
        "accept_flow",
        "eval_path_tamper_g1",
        "heldout_split_contamination_g2",
        "hardcoded_metric_diff_g3",
        "static_degradation_af_gate",
        "replication_missing",
    }
)


def run_m3_acceptance_audit(
    *,
    proposal_readiness_manifest: Path,
    judge_gate_manifest: Path,
    orchestrator_smoke_manifest: Path,
    raw_failure_batch_manifest: Path | None = None,
    horizon_protocol_manifest: Path | None = None,
    training_eval_smoke_manifest: Path | None = None,
    output_root: Path,
    archive_db: Path | None = None,
    cas_root: Path | None = None,
) -> dict[str, object]:
    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise M3AcceptanceAuditError("M3_ACCEPTANCE_AUDIT_OUTPUT_EXISTS")

    archive = ArchiveStore(archive_db) if archive_db is not None else None
    cas_storage_root = cas_root if cas_root is not None else (Path(archive_db).resolve().parent if archive_db is not None else destination.parent)
    cas = ContentAddressedStore(cas_storage_root)

    sources = {
        "proposal_readiness": _load_source(proposal_readiness_manifest, cas=cas, archive=archive),
        "judge_gate": _load_source(judge_gate_manifest, cas=cas, archive=archive),
        "orchestrator_smoke": _load_source(orchestrator_smoke_manifest, cas=cas, archive=archive),
    }
    if raw_failure_batch_manifest is not None:
        sources["raw_failure_batch"] = _load_source(raw_failure_batch_manifest, cas=cas, archive=archive)
    if horizon_protocol_manifest is not None:
        sources["horizon_protocol"] = _load_source(horizon_protocol_manifest, cas=cas, archive=archive)
    if training_eval_smoke_manifest is not None:
        sources["training_eval_smoke"] = _load_source(training_eval_smoke_manifest, cas=cas, archive=archive)

    requirements = _requirements(sources)
    strict_m3_pass = all(
        requirements[name]["passed"] is True
        for name in ("T3.1_proposal_generation", "T3.2_verifier_gates", "T3.3_orchestrator_push_cube")
    )
    state = "ready" if strict_m3_pass else "partial"
    blockers = _blockers(requirements=requirements, sources=sources)
    report = {
        "schema_version": 1,
        "artifact_type": "wmloop-m3-strict-acceptance-audit",
        "state": state,
        "strict_m3_pass": strict_m3_pass,
        "requirements": requirements,
        "blockers": blockers,
        "sources": {name: source["summary"] for name, source in sources.items()},
        "cas_root": str(Path(cas_storage_root).resolve()),
        "archive_db": str(Path(archive_db).resolve()) if archive_db is not None else None,
        "limitations": [
            "This audit reads existing manifests and does not launch training or evaluation.",
            "Training/evaluation smoke evidence is tracked separately from formal model-quality ACCEPT claims.",
            "A partial state records exact unmet strict-DoD requirements instead of promoting smoke evidence.",
        ],
        "next_actions": _next_actions(requirements=requirements, sources=sources),
    }
    return _write_report_bundle(report=report, output_root=destination, cas=cas, archive=archive)


def _requirements(sources: Mapping[str, Mapping[str, object]]) -> dict[str, dict[str, object]]:
    proposal = _payload(sources, "proposal_readiness")
    judge_gate = _payload(sources, "judge_gate")
    orchestrator = _payload(sources, "orchestrator_smoke")
    training_eval = _payload_or_none(sources, "training_eval_smoke")

    t31_passed = proposal.get("strict_t3_1_pass") is True and proposal.get("state") == "ready"
    t32_case_count = _int(judge_gate.get("case_count"))
    t32_passed_count = _int(judge_gate.get("passed_count"))
    t32_case_names = _string_set(judge_gate.get("case_names"))
    t32_missing_cases = sorted(REQUIRED_VERIFIER_GATE_CASES - t32_case_names)
    t32_passed = (
        judge_gate.get("state") == "ready"
        and t32_case_count >= len(REQUIRED_VERIFIER_GATE_CASES)
        and t32_passed_count == t32_case_count
        and not t32_missing_cases
    )
    t33_rounds = _int(orchestrator.get("completed_round_count"))
    t33_passed = orchestrator.get("state") == "ready" and t33_rounds >= 5
    model_quality = _model_quality_summary(training_eval)

    return {
        "T3.1_proposal_generation": {
            "passed": t31_passed,
            "expected": "At least 3 real failure_report inputs produce legal proposals and invalid JSON retry demo passes.",
            "observed": {
                "state": proposal.get("state"),
                "strict_t3_1_pass": proposal.get("strict_t3_1_pass"),
                "failure_report_count": proposal.get("failure_report_count"),
                "legal_proposal_count": proposal.get("legal_proposal_count"),
                "strict_required_reports": proposal.get("strict_required_reports"),
                "blockers": proposal.get("blockers", []),
            },
        },
        "T3.2_verifier_gates": {
            "passed": t32_passed,
            "expected": "Verifier gate smoke covers accept flow plus G1/G2/G3/AF/G4 fail-closed cases.",
            "observed": {
                "state": judge_gate.get("state"),
                "case_count": t32_case_count,
                "passed_count": t32_passed_count,
                "case_names": sorted(t32_case_names),
                "required_case_names": sorted(REQUIRED_VERIFIER_GATE_CASES),
                "missing_case_names": t32_missing_cases,
            },
        },
        "T3.3_orchestrator_push_cube": {
            "passed": t33_passed,
            "expected": "Push Cube autonomous orchestrator smoke completes at least 5 rounds with archived receipts/verdicts.",
            "observed": {
                "state": orchestrator.get("state"),
                "completed_round_count": t33_rounds,
                "verdict_counts": orchestrator.get("verdict_counts", {}),
            },
        },
        "model_quality_acceptance": model_quality,
    }


def _model_quality_summary(training_eval: Mapping[str, Any] | None) -> dict[str, object]:
    if training_eval is None:
        return {
            "passed": False,
            "expected": "Formal model-quality trials are separate from M3 smoke and require a verifier ACCEPT.",
            "observed": {"state": "not_provided"},
            "claim_scope": "none",
        }
    verdict = training_eval.get("verdict")
    return {
        "passed": verdict == "ACCEPT" and training_eval.get("state") == "ready",
        "expected": "Formal model-quality trials require ready evidence and ACCEPT verdict; current smoke does not establish this claim.",
        "observed": {
            "state": training_eval.get("state"),
            "verdict": verdict,
            "delta_m_ver": training_eval.get("delta_m_ver"),
            "gates": training_eval.get("gates"),
            "action_following_gate": training_eval.get("action_following_gate"),
        },
        "claim_scope": "smoke_only" if training_eval.get("state") == "ready" else "none",
    }


def _blockers(*, requirements: Mapping[str, Mapping[str, object]], sources: Mapping[str, Mapping[str, object]]) -> list[dict[str, object]]:
    blockers = []
    if not requirements["T3.1_proposal_generation"]["passed"]:
        proposal_observed = requirements["T3.1_proposal_generation"]["observed"]
        blockers.append(
            {
                "requirement": "T3.1_proposal_generation",
                "reason": "strict_T3_1_requires_three_real_failure_reports",
                "observed": proposal_observed,
            }
        )
    raw_failure = _payload_or_none(sources, "raw_failure_batch")
    if raw_failure is not None and raw_failure.get("state") != "ready":
        blockers.append(
            {
                "requirement": "M1_raw_failure_report_inputs",
                "reason": "raw_failure_report_batch_not_ready",
                "observed": {
                    "state": raw_failure.get("state"),
                    "report_count": raw_failure.get("report_count"),
                    "blocked_count": raw_failure.get("blocked_count"),
                    "blocked_records": raw_failure.get("blocked_records", []),
                },
            }
        )
    horizon = _payload_or_none(sources, "horizon_protocol")
    if horizon is not None and _int(horizon.get("unavailable_horizon_count")) > 0:
        blockers.append(
            {
                "requirement": "M1_horizon_protocol",
                "reason": "dataset_length_limited_horizons_are_not_gpu_rerun_tasks",
                "observed": {
                    "state": horizon.get("state"),
                    "decision_id": horizon.get("decision_id"),
                    "unavailable_record_count": horizon.get("unavailable_record_count"),
                    "unavailable_horizon_count": horizon.get("unavailable_horizon_count"),
                    "next_actions": horizon.get("next_actions", []),
                },
            }
        )
    return blockers


def _next_actions(*, requirements: Mapping[str, Mapping[str, object]], sources: Mapping[str, Mapping[str, object]]) -> list[str]:
    actions = []
    if not requirements["T3.1_proposal_generation"]["passed"]:
        actions.append("Provide a third real raw failure_report under the frozen protocol, or make a human protocol revision before rerunning T3.1.")
    horizon = _payload_or_none(sources, "horizon_protocol")
    if horizon is not None and _int(horizon.get("unavailable_horizon_count")) > 0:
        actions.append("Do not spend GPU time rerunning dataset-length-limited 64-frame horizons until the goal/data protocol is revised.")
    if requirements["T3.2_verifier_gates"]["passed"] and requirements["T3.3_orchestrator_push_cube"]["passed"]:
        actions.append("Keep T3.2/T3.3 evidence as valid smoke linkage; do not relabel it as a model-quality ACCEPT.")
    if not actions:
        actions.append("Proceed to formal multi-seed trials and publish only settled verifier outputs.")
    return actions


def _load_source(path: Path, *, cas: ContentAddressedStore, archive: ArchiveStore | None) -> dict[str, object]:
    resolved = Path(path).resolve()
    try:
        payload_bytes = resolved.read_bytes()
        payload = json.loads(payload_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise M3AcceptanceAuditError(f"M3_ACCEPTANCE_AUDIT_SOURCE_INVALID:{resolved}") from exc
    if not isinstance(payload, dict):
        raise M3AcceptanceAuditError(f"M3_ACCEPTANCE_AUDIT_SOURCE_NOT_OBJECT:{resolved}")
    ref = cas.put_bytes(payload_bytes, media_type="application/json").uri
    if archive is not None:
        archive.record_artifact_reference(ref)
    return {
        "payload": payload,
        "summary": {
            "path": str(resolved),
            "sha256": hashlib.sha256(payload_bytes).hexdigest(),
            "cas_ref": ref,
            "artifact_type": payload.get("artifact_type"),
            "state": payload.get("state"),
        },
    }


def _payload(sources: Mapping[str, Mapping[str, object]], name: str) -> Mapping[str, Any]:
    try:
        payload = sources[name]["payload"]
    except KeyError as exc:
        raise M3AcceptanceAuditError(f"M3_ACCEPTANCE_AUDIT_REQUIRED_SOURCE_MISSING:{name}") from exc
    if not isinstance(payload, Mapping):
        raise M3AcceptanceAuditError(f"M3_ACCEPTANCE_AUDIT_SOURCE_INVALID:{name}")
    return payload


def _payload_or_none(sources: Mapping[str, Mapping[str, object]], name: str) -> Mapping[str, Any] | None:
    if name not in sources:
        return None
    return _payload(sources, name)


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
        temporary.mkdir(mode=0o700, parents=True)
        _write_bytes_atomic(temporary / "m3-acceptance-audit.json", report_bytes)
        _write_bytes_atomic(temporary / "m3-acceptance-audit.md", markdown_bytes)
        report_ref = cas.put_bytes(report_bytes, media_type="application/json").uri
        markdown_ref = cas.put_bytes(markdown_bytes, media_type="text/markdown").uri
        if archive is not None:
            archive.record_artifact_reference(report_ref)
            archive.record_artifact_reference(markdown_ref)
        manifest = {
            "schema_version": 1,
            "artifact_type": "wmloop-m3-strict-acceptance-audit-manifest",
            "state": report["state"],
            "strict_m3_pass": report["strict_m3_pass"],
            "requirements": report["requirements"],
            "report_path": str(destination / "m3-acceptance-audit.json"),
            "markdown_path": str(destination / "m3-acceptance-audit.md"),
            "cas_refs": {
                "m3_acceptance_audit_json": report_ref,
                "m3_acceptance_audit_markdown": markdown_ref,
            },
            "blockers": report["blockers"],
            "next_actions": report["next_actions"],
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
        "# M3 Strict Acceptance Audit",
        "",
        f"State: `{report['state']}`",
        f"Strict M3 pass: `{report['strict_m3_pass']}`",
        "",
        "| Requirement | Passed | Observed |",
        "|:--|:--|:--|",
    ]
    for name, requirement in report["requirements"].items():
        observed = json.dumps(requirement["observed"], sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        lines.append(f"| {name} | {requirement['passed']} | `{observed}` |")
    if report["blockers"]:
        lines.extend(["", "## Blockers", ""])
        for blocker in report["blockers"]:
            lines.append(f"- `{json.dumps(blocker, sort_keys=True, separators=(',', ':'), ensure_ascii=False)}`")
    lines.extend(["", "## Next Actions", ""])
    for action in report["next_actions"]:
        lines.append(f"- {action}")
    lines.extend(["", "## Limitations", ""])
    for limitation in report["limitations"]:
        lines.append(f"- {limitation}")
    return "\n".join(lines) + "\n"


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise M3AcceptanceAuditError("M3_ACCEPTANCE_AUDIT_OUTPUT_EXISTS")
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


def _int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    return 0


def _string_set(value: object) -> set[str]:
    if not isinstance(value, list):
        return set()
    return {item for item in value if isinstance(item, str)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="audit M3 strict acceptance manifests")
    run.add_argument("--proposal-readiness-manifest", type=Path, required=True)
    run.add_argument("--judge-gate-manifest", type=Path, required=True)
    run.add_argument("--orchestrator-smoke-manifest", type=Path, required=True)
    run.add_argument("--raw-failure-batch-manifest", type=Path)
    run.add_argument("--horizon-protocol-manifest", type=Path)
    run.add_argument("--training-eval-smoke-manifest", type=Path)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--archive-db", type=Path)
    run.add_argument("--cas-root", type=Path)
    args = parser.parse_args(argv)
    if args.command == "run":
        manifest = run_m3_acceptance_audit(
            proposal_readiness_manifest=args.proposal_readiness_manifest,
            judge_gate_manifest=args.judge_gate_manifest,
            orchestrator_smoke_manifest=args.orchestrator_smoke_manifest,
            raw_failure_batch_manifest=args.raw_failure_batch_manifest,
            horizon_protocol_manifest=args.horizon_protocol_manifest,
            training_eval_smoke_manifest=args.training_eval_smoke_manifest,
            output_root=args.output_root,
            archive_db=args.archive_db,
            cas_root=args.cas_root,
        )
        print(json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
