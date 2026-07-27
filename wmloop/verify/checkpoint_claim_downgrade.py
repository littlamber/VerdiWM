"""Human-approved checkpoint claim downgrade receipt for M4 launch.

This receipt does not certify a warned checkpoint as an official 100k
reproduction checkpoint.  It records that the current official asset may be
used for paired baseline-vs-candidate M4 deltas while the stronger reproduction
claim remains forbidden for warned environments.
"""

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


class CheckpointClaimDowngradeError(RuntimeError):
    """Checkpoint claim downgrade evidence is invalid or unsafe."""


CLAIM_BOUNDARY_TEXT = (
    "Official-current warned checkpoints may be used for paired baseline-vs-candidate M4 deltas; "
    "official 100k reproduction claims remain disallowed for warned environments."
)


def run_checkpoint_claim_downgrade(
    *,
    limited_gate_manifest: Path,
    baseline_reproduction_manifest: Path,
    checkpoint_source_manifest: Path,
    output_root: Path,
    archive_db: Path,
    checkpoint_launch_guard_manifest: Path | None = None,
    reviewer: str = "user",
    rationale: str = "Human decision accepts official-current checkpoint warning for paired-delta M4 trials.",
    human_approval_confirmed: bool = False,
    cas_root: Path | None = None,
) -> dict[str, object]:
    """Write a receipt that downgrades strict checkpoint claims, not evidence."""

    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise CheckpointClaimDowngradeError("CHECKPOINT_CLAIM_DOWNGRADE_OUTPUT_EXISTS")
    if not human_approval_confirmed:
        raise CheckpointClaimDowngradeError("CHECKPOINT_CLAIM_DOWNGRADE_HUMAN_APPROVAL_REQUIRED")
    if not reviewer:
        raise CheckpointClaimDowngradeError("CHECKPOINT_CLAIM_DOWNGRADE_REVIEWER_REQUIRED")
    archive = ArchiveStore(archive_db)
    cas_storage_root = Path(cas_root).resolve() if cas_root is not None else Path(archive_db).resolve().parent
    cas = ContentAddressedStore(cas_storage_root)

    limited_gate = _load_manifest_with_report(
        limited_gate_manifest,
        manifest_artifact_type="wmloop-limited-campaign-gate-manifest",
        report_artifact_type="wmloop-limited-campaign-gate",
        cas=cas,
        archive=archive,
    )
    baseline = _load_manifest_with_report(
        baseline_reproduction_manifest,
        manifest_artifact_type="acwm-m0-baseline-reproduction-manifest",
        report_artifact_type="acwm-m0-baseline-reproduction-report",
        cas=cas,
        archive=archive,
    )
    checkpoint_source = _load_manifest_with_report(
        checkpoint_source_manifest,
        manifest_artifact_type="acwm-m0-checkpoint-source-audit-manifest",
        report_artifact_type="acwm-m0-checkpoint-source-audit",
        cas=cas,
        archive=archive,
    )
    checkpoint_launch_guard = (
        _load_source(checkpoint_launch_guard_manifest, cas=cas, archive=archive)
        if checkpoint_launch_guard_manifest is not None
        else None
    )

    requirements = _requirements(
        limited_gate=limited_gate,
        baseline=baseline,
        checkpoint_source=checkpoint_source,
        checkpoint_launch_guard=checkpoint_launch_guard,
    )
    ready = all(item["passed"] is True for item in requirements.values())
    observed = requirements["official_current_warning"]["observed"]
    warned_envs = _string_list(observed.get("warned_envs"))
    allowed_envs = _string_list(observed.get("allowed_envs"))
    warning_records = observed.get("official_current_warning_records")
    if not isinstance(warning_records, Mapping):
        warning_records = {}
    report = {
        "schema_version": 1,
        "artifact_type": "wmloop-m4-checkpoint-claim-downgrade",
        "state": "ready" if ready else "blocked",
        "phase": "M4_launch",
        "m4_launch_claim_downgrade_allowed": ready,
        "human_approval_confirmed": True,
        "reviewer": reviewer,
        "rationale": rationale,
        "allowed_envs": allowed_envs,
        "warned_envs": warned_envs,
        "official_current_warning_records": {
            str(env): dict(record)
            for env, record in warning_records.items()
            if isinstance(env, str) and isinstance(record, Mapping)
        },
        "checkpoint_policy": {
            "allow_official_current_checkpoint_warning": True,
            "claim_boundary": CLAIM_BOUNDARY_TEXT,
        },
        "claim_boundary": {
            "paired_delta_m4_trials_allowed": True,
            "official_100k_reproduction_claim_disallowed_for_warned_envs": True,
            "source_checkpoint_mutation_allowed": False,
            "evaluator_or_protocol_mutation_allowed": False,
        },
        "requirements": requirements,
        "blockers": _blockers(requirements),
        "sources": {
            "limited_campaign_gate": limited_gate["summary"],
            "baseline_reproduction": baseline["summary"],
            "checkpoint_source": checkpoint_source["summary"],
            **({"checkpoint_launch_guard": checkpoint_launch_guard["summary"]} if checkpoint_launch_guard else {}),
        },
        "limitations": [
            "This receipt changes the M4 claim boundary only; it does not mutate checkpoints, evaluators, data, or the active protocol.",
            CLAIM_BOUNDARY_TEXT,
        ],
    }
    return _write_report_bundle(report=report, output_root=destination, cas=cas, archive=archive)


def _requirements(
    *,
    limited_gate: Mapping[str, object],
    baseline: Mapping[str, object],
    checkpoint_source: Mapping[str, object],
    checkpoint_launch_guard: Mapping[str, object] | None,
) -> dict[str, dict[str, object]]:
    limited_manifest = _payload(limited_gate, "manifest")
    limited_report = _payload(limited_gate, "report")
    baseline_manifest = _payload(baseline, "manifest")
    baseline_report = _payload(baseline, "report")
    source_manifest = _payload(checkpoint_source, "manifest")
    source_report = _payload(checkpoint_source, "report")
    observed_warning = _official_current_warning_observed(limited_manifest=limited_manifest, limited_report=limited_report)
    source_alignment = _source_alignment_observed(
        limited_report=limited_report,
        baseline=baseline,
        checkpoint_source=checkpoint_source,
        checkpoint_launch_guard=checkpoint_launch_guard,
    )
    warned_envs = _string_list(observed_warning.get("warned_envs"))
    return {
        "limited_gate_ready_with_warning_policy": {
            "passed": _limited_gate_ready(limited_manifest=limited_manifest, limited_report=limited_report),
            "expected": "A ready limited campaign gate explicitly permits official-current checkpoint warnings.",
            "observed": _limited_gate_observed(limited_manifest=limited_manifest, limited_report=limited_report),
        },
        "official_current_warning": {
            "passed": _official_current_warning_passes(observed_warning),
            "expected": "Every included checkpoint mismatch is a warned official-current file with remote-current provenance.",
            "observed": observed_warning,
        },
        "source_manifest_alignment": {
            "passed": source_alignment["passed"] is True,
            "expected": "The receipt inputs match the source manifest hashes recorded by the limited gate.",
            "observed": source_alignment["observed"],
        },
        "baseline_reproduction_scope": {
            "passed": _baseline_scope_passes(
                baseline_manifest=baseline_manifest,
                baseline_report=baseline_report,
                allowed_envs=_string_list(observed_warning.get("allowed_envs")),
            ),
            "expected": "Baseline reproduction covers every allowed environment and is blocked only by the accepted checkpoint-step warning.",
            "observed": _baseline_scope_observed(
                baseline_manifest=baseline_manifest,
                baseline_report=baseline_report,
                allowed_envs=_string_list(observed_warning.get("allowed_envs")),
            ),
        },
        "checkpoint_source_scope": {
            "passed": _checkpoint_source_scope_passes(
                source_manifest=source_manifest,
                source_report=source_report,
                warned_envs=warned_envs,
                warning_records=observed_warning.get("official_current_warning_records"),
            ),
            "expected": "Checkpoint-source mismatch scope exactly matches the warned official-current environments.",
            "observed": _checkpoint_source_scope_observed(
                source_manifest=source_manifest,
                source_report=source_report,
                warned_envs=warned_envs,
                warning_records=observed_warning.get("official_current_warning_records"),
            ),
        },
        "claim_boundary": {
            "passed": True,
            "expected": "The downgrade allows paired-delta trials but forbids official 100k reproduction claims for warned environments.",
            "observed": {
                "paired_delta_m4_trials_allowed": True,
                "official_100k_reproduction_claim_disallowed_for_warned_envs": True,
                "source_checkpoint_mutation_allowed": False,
                "evaluator_or_protocol_mutation_allowed": False,
            },
        },
    }


def _limited_gate_ready(*, limited_manifest: Mapping[str, Any], limited_report: Mapping[str, Any]) -> bool:
    policy = limited_manifest.get("checkpoint_policy")
    report_policy = limited_report.get("checkpoint_policy")
    return (
        limited_manifest.get("state") == "ready"
        and limited_manifest.get("limited_campaign_allowed") is True
        and limited_manifest.get("m4_launch_allowed") is False
        and limited_manifest.get("formal_training_allowed") is False
        and isinstance(policy, Mapping)
        and policy.get("allow_official_current_checkpoint_warning") is True
        and limited_report.get("state") == "ready"
        and isinstance(report_policy, Mapping)
        and report_policy.get("allow_official_current_checkpoint_warning") is True
    )


def _limited_gate_observed(*, limited_manifest: Mapping[str, Any], limited_report: Mapping[str, Any]) -> dict[str, object]:
    policy = limited_manifest.get("checkpoint_policy")
    report_policy = limited_report.get("checkpoint_policy")
    return {
        "manifest_state": limited_manifest.get("state"),
        "report_state": limited_report.get("state"),
        "limited_campaign_allowed": limited_manifest.get("limited_campaign_allowed"),
        "m4_launch_allowed": limited_manifest.get("m4_launch_allowed"),
        "formal_training_allowed": limited_manifest.get("formal_training_allowed"),
        "manifest_warning_policy": policy.get("allow_official_current_checkpoint_warning") if isinstance(policy, Mapping) else None,
        "report_warning_policy": report_policy.get("allow_official_current_checkpoint_warning") if isinstance(report_policy, Mapping) else None,
    }


def _official_current_warning_observed(
    *,
    limited_manifest: Mapping[str, Any],
    limited_report: Mapping[str, Any],
) -> dict[str, object]:
    requirements = limited_report.get("requirements")
    checkpoint_requirement = requirements.get("checkpoint_steps_for_scope") if isinstance(requirements, Mapping) else None
    observed = checkpoint_requirement.get("observed") if isinstance(checkpoint_requirement, Mapping) else None
    if not isinstance(observed, Mapping):
        return {"state": "missing"}
    included = _string_list(limited_manifest.get("included_envs"))
    warned = _string_list(observed.get("included_provenance_warning"))
    included_non_pass = _string_list(observed.get("included_non_pass"))
    warning_records = observed.get("official_current_warning_records")
    return {
        "state": observed.get("state"),
        "checkpoint_requirement_passed": checkpoint_requirement.get("passed"),
        "allow_official_current_checkpoint_warning": observed.get("allow_official_current_checkpoint_warning"),
        "allowed_envs": included,
        "warned_envs": warned,
        "included_non_pass": included_non_pass,
        "mismatch_envs": _string_list(observed.get("mismatch_envs")),
        "mismatch_outside_excluded": _string_list(observed.get("mismatch_outside_excluded")),
        "expected_step": observed.get("expected_step"),
        "claim_boundary": observed.get("claim_boundary"),
        "official_current_warning_records": warning_records if isinstance(warning_records, Mapping) else {},
    }


def _official_current_warning_passes(observed: Mapping[str, object]) -> bool:
    warned = _string_list(observed.get("warned_envs"))
    included_non_pass = _string_list(observed.get("included_non_pass"))
    warning_records = observed.get("official_current_warning_records")
    if (
        observed.get("checkpoint_requirement_passed") is not True
        or observed.get("allow_official_current_checkpoint_warning") is not True
        or observed.get("mismatch_outside_excluded") != []
        or not warned
        or set(warned) != set(included_non_pass)
        or not isinstance(warning_records, Mapping)
    ):
        return False
    for env in warned:
        record = warning_records.get(env)
        if not isinstance(record, Mapping):
            return False
        if record.get("source_status") != "remote_current_matches_local_mismatch":
            return False
        if _int(record.get("observed_step")) == 0 or _int(record.get("expected_step")) == 0:
            return False
        if _int(record.get("observed_step")) >= _int(record.get("expected_step")):
            return False
        local_hash = record.get("local_hf_hash")
        if not isinstance(local_hash, str) or len(local_hash) != 64:
            return False
    return True


def _source_alignment_observed(
    *,
    limited_report: Mapping[str, Any],
    baseline: Mapping[str, object],
    checkpoint_source: Mapping[str, object],
    checkpoint_launch_guard: Mapping[str, object] | None,
) -> dict[str, object]:
    sources = limited_report.get("sources")
    if not isinstance(sources, Mapping):
        return {"passed": False, "observed": {"state": "missing_limited_gate_sources"}}
    expected = {
        "baseline_reproduction": _source_sha(sources, "baseline_reproduction"),
        "checkpoint_source": _source_sha(sources, "checkpoint_source"),
    }
    observed = {
        "baseline_reproduction": _summary_sha(baseline),
        "checkpoint_source": _summary_sha(checkpoint_source),
    }
    if checkpoint_launch_guard is not None:
        observed["checkpoint_launch_guard"] = _summary_sha(checkpoint_launch_guard)
    mismatches = sorted(name for name, expected_sha in expected.items() if not expected_sha or observed.get(name) != expected_sha)
    return {"passed": not mismatches, "observed": {"expected": expected, "observed": observed, "mismatches": mismatches}}


def _source_sha(sources: Mapping[str, object], name: str) -> str | None:
    source = sources.get(name)
    value = source.get("sha256") if isinstance(source, Mapping) else None
    return value if isinstance(value, str) and value else None


def _summary_sha(source: Mapping[str, object]) -> str | None:
    summary = source.get("summary")
    value = summary.get("sha256") if isinstance(summary, Mapping) else None
    return value if isinstance(value, str) and value else None


def _baseline_scope_passes(
    *,
    baseline_manifest: Mapping[str, Any],
    baseline_report: Mapping[str, Any],
    allowed_envs: Sequence[str],
) -> bool:
    by_environment = baseline_report.get("by_environment_unweighted")
    return (
        baseline_manifest.get("state") == "checkpoint_step_mismatch"
        and baseline_manifest.get("strict_m0_t03_pass") is False
        and baseline_report.get("state") == "checkpoint_step_mismatch"
        and baseline_report.get("strict_m0_t03_pass") is False
        and isinstance(by_environment, Mapping)
        and all(env in by_environment for env in allowed_envs)
    )


def _baseline_scope_observed(
    *,
    baseline_manifest: Mapping[str, Any],
    baseline_report: Mapping[str, Any],
    allowed_envs: Sequence[str],
) -> dict[str, object]:
    by_environment = baseline_report.get("by_environment_unweighted")
    covered = sorted(env for env in allowed_envs if isinstance(by_environment, Mapping) and env in by_environment)
    return {
        "manifest_state": baseline_manifest.get("state"),
        "report_state": baseline_report.get("state"),
        "manifest_strict_m0_t03_pass": baseline_manifest.get("strict_m0_t03_pass"),
        "report_strict_m0_t03_pass": baseline_report.get("strict_m0_t03_pass"),
        "allowed_env_count": len(allowed_envs),
        "covered_envs": covered,
        "missing_envs": sorted(set(allowed_envs) - set(covered)),
        "warnings": baseline_manifest.get("warnings", []),
    }


def _checkpoint_source_scope_passes(
    *,
    source_manifest: Mapping[str, Any],
    source_report: Mapping[str, Any],
    warned_envs: Sequence[str],
    warning_records: object,
) -> bool:
    if source_manifest.get("state") != "remote_current_mismatch" or source_report.get("state") != "remote_current_mismatch":
        return False
    if _int(source_manifest.get("mismatch_count")) != len(warned_envs):
        return False
    if _int(source_report.get("mismatch_count")) != len(warned_envs):
        return False
    if not isinstance(warning_records, Mapping):
        return False
    records = source_report.get("records")
    if not isinstance(records, list):
        return False
    by_env = {record.get("environment"): record for record in records if isinstance(record, Mapping)}
    for env in warned_envs:
        source_record = by_env.get(env)
        warning_record = warning_records.get(env)
        if not isinstance(source_record, Mapping) or not isinstance(warning_record, Mapping):
            return False
        if source_record.get("source_status") != "remote_current_matches_local_mismatch":
            return False
        for field in ("observed_step", "expected_step", "local_hf_hash", "checkpoint_relative_path"):
            if source_record.get(field) != warning_record.get(field):
                return False
    return True


def _checkpoint_source_scope_observed(
    *,
    source_manifest: Mapping[str, Any],
    source_report: Mapping[str, Any],
    warned_envs: Sequence[str],
    warning_records: object,
) -> dict[str, object]:
    records = source_report.get("records")
    by_env = {
        record.get("environment"): record
        for record in records
        if isinstance(records, list) and isinstance(record, Mapping)
    } if isinstance(records, list) else {}
    matched = []
    mismatched = []
    if isinstance(warning_records, Mapping):
        for env in warned_envs:
            source_record = by_env.get(env)
            warning_record = warning_records.get(env)
            ok = (
                isinstance(source_record, Mapping)
                and isinstance(warning_record, Mapping)
                and source_record.get("source_status") == "remote_current_matches_local_mismatch"
                and all(
                    source_record.get(field) == warning_record.get(field)
                    for field in ("observed_step", "expected_step", "local_hf_hash", "checkpoint_relative_path")
                )
            )
            (matched if ok else mismatched).append(env)
    return {
        "manifest_state": source_manifest.get("state"),
        "report_state": source_report.get("state"),
        "manifest_mismatch_count": source_manifest.get("mismatch_count"),
        "report_mismatch_count": source_report.get("mismatch_count"),
        "warned_envs": list(warned_envs),
        "matched_warning_records": matched,
        "mismatched_warning_records": mismatched,
    }


def _blockers(requirements: Mapping[str, Mapping[str, object]]) -> list[dict[str, object]]:
    return [
        {"requirement": name, "expected": item["expected"], "observed": item["observed"]}
        for name, item in requirements.items()
        if item.get("passed") is not True
    ]


def _load_manifest_with_report(
    path: Path,
    *,
    manifest_artifact_type: str,
    report_artifact_type: str,
    cas: ContentAddressedStore,
    archive: ArchiveStore,
) -> dict[str, object]:
    manifest_source = _load_source(path, cas=cas, archive=archive)
    manifest = _payload(manifest_source, "payload")
    if manifest.get("artifact_type") != manifest_artifact_type:
        raise CheckpointClaimDowngradeError(f"CHECKPOINT_CLAIM_DOWNGRADE_MANIFEST_TYPE_INVALID:{path}")
    report_path = manifest.get("report_path")
    if not isinstance(report_path, str) or not report_path:
        raise CheckpointClaimDowngradeError(f"CHECKPOINT_CLAIM_DOWNGRADE_REPORT_PATH_MISSING:{path}")
    report_source = _load_source(Path(report_path), cas=cas, archive=archive)
    report = _payload(report_source, "payload")
    if report.get("artifact_type") != report_artifact_type:
        raise CheckpointClaimDowngradeError(f"CHECKPOINT_CLAIM_DOWNGRADE_REPORT_TYPE_INVALID:{report_path}")
    return {
        "manifest": manifest,
        "report": report,
        "summary": {**manifest_source["summary"], "report_ref": report_source["summary"]["cas_ref"]},
    }


def _load_source(path: Path | None, *, cas: ContentAddressedStore, archive: ArchiveStore) -> dict[str, object]:
    if path is None:
        raise CheckpointClaimDowngradeError("CHECKPOINT_CLAIM_DOWNGRADE_SOURCE_MISSING")
    resolved = Path(path).resolve(strict=True)
    try:
        payload_bytes = resolved.read_bytes()
        payload = json.loads(payload_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointClaimDowngradeError(f"CHECKPOINT_CLAIM_DOWNGRADE_SOURCE_INVALID:{resolved}") from exc
    if not isinstance(payload, dict):
        raise CheckpointClaimDowngradeError(f"CHECKPOINT_CLAIM_DOWNGRADE_SOURCE_NOT_OBJECT:{resolved}")
    ref = cas.put_bytes(payload_bytes, media_type="application/json").uri
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


def _payload(source: Mapping[str, object], key: str) -> Mapping[str, Any]:
    value = source.get(key)
    if not isinstance(value, Mapping):
        raise CheckpointClaimDowngradeError("CHECKPOINT_CLAIM_DOWNGRADE_SOURCE_INVALID")
    return value


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
        _write_bytes_atomic(temporary / "checkpoint-claim-downgrade.json", report_bytes)
        _write_bytes_atomic(temporary / "checkpoint-claim-downgrade.md", markdown_bytes)
        report_ref = cas.put_bytes(report_bytes, media_type="application/json").uri
        markdown_ref = cas.put_bytes(markdown_bytes, media_type="text/markdown").uri
        archive.record_artifact_reference(report_ref)
        archive.record_artifact_reference(markdown_ref)
        manifest = {
            "schema_version": 1,
            "artifact_type": "wmloop-m4-checkpoint-claim-downgrade-manifest",
            "state": report["state"],
            "phase": report["phase"],
            "m4_launch_claim_downgrade_allowed": report["m4_launch_claim_downgrade_allowed"],
            "human_approval_confirmed": report["human_approval_confirmed"],
            "reviewer": report["reviewer"],
            "allowed_envs": report["allowed_envs"],
            "warned_envs": report["warned_envs"],
            "checkpoint_policy": report["checkpoint_policy"],
            "claim_boundary": report["claim_boundary"],
            "sources": report["sources"],
            "blockers": report["blockers"],
            "report_path": str(destination / "checkpoint-claim-downgrade.json"),
            "markdown_path": str(destination / "checkpoint-claim-downgrade.md"),
            "cas_refs": {
                "checkpoint_claim_downgrade_json": report_ref,
                "checkpoint_claim_downgrade_markdown": markdown_ref,
            },
        }
        _write_bytes_atomic(temporary / "manifest.json", _canonical_json_bytes(manifest))
        os.replace(temporary, destination)
        return manifest
    except Exception:
        if temporary.is_dir():
            shutil.rmtree(temporary, ignore_errors=True)
        elif temporary.exists() or temporary.is_symlink():
            temporary.unlink()
        raise


def _render_markdown(report: Mapping[str, object]) -> str:
    lines = [
        "# Checkpoint Claim Downgrade",
        "",
        f"State: `{report['state']}`",
        f"M4 claim downgrade allowed: `{report['m4_launch_claim_downgrade_allowed']}`",
        f"Reviewer: `{report['reviewer']}`",
        f"Allowed envs: `{report['allowed_envs']}`",
        f"Warned envs: `{report['warned_envs']}`",
        "",
        "## Claim Boundary",
        "",
        CLAIM_BOUNDARY_TEXT,
        "",
        "## Blockers",
        "",
    ]
    blockers = report.get("blockers")
    if isinstance(blockers, list) and blockers:
        lines.extend(f"- `{item}`" for item in blockers)
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item]


def _int(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise CheckpointClaimDowngradeError("CHECKPOINT_CLAIM_DOWNGRADE_OUTPUT_EXISTS")
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
    run = commands.add_parser("run", help="write a checkpoint claim downgrade receipt")
    run.add_argument("--limited-gate-manifest", type=Path, required=True)
    run.add_argument("--baseline-reproduction-manifest", type=Path, required=True)
    run.add_argument("--checkpoint-source-manifest", type=Path, required=True)
    run.add_argument("--checkpoint-launch-guard-manifest", type=Path)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--archive-db", type=Path, required=True)
    run.add_argument("--reviewer", default="user")
    run.add_argument("--rationale", default="Human decision accepts official-current checkpoint warning for paired-delta M4 trials.")
    run.add_argument("--confirm-human-approval", action="store_true")
    run.add_argument("--cas-root", type=Path)
    args = parser.parse_args(argv)
    if args.command == "run":
        manifest = run_checkpoint_claim_downgrade(
            limited_gate_manifest=args.limited_gate_manifest,
            baseline_reproduction_manifest=args.baseline_reproduction_manifest,
            checkpoint_source_manifest=args.checkpoint_source_manifest,
            checkpoint_launch_guard_manifest=args.checkpoint_launch_guard_manifest,
            output_root=args.output_root,
            archive_db=args.archive_db,
            reviewer=args.reviewer,
            rationale=args.rationale,
            human_approval_confirmed=args.confirm_human_approval,
            cas_root=args.cas_root,
        )
        print(json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
        return 0
    raise CheckpointClaimDowngradeError("CHECKPOINT_CLAIM_DOWNGRADE_COMMAND_INVALID")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
