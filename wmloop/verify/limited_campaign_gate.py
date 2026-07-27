"""Scoped gate for non-formal limited campaign pilots.

This gate intentionally does not authorize M4.  It exists so a subset of
environments can exercise the closed-loop control path while a known excluded
environment remains under recovery.
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


class LimitedCampaignGateError(RuntimeError):
    """A limited campaign scope is invalid or not ready."""


def run_limited_campaign_gate(
    *,
    goal_config: Path,
    checkpoint_step_audit_manifest: Path,
    raw_failure_batch_manifest: Path,
    constitutional_audit_manifest: Path,
    m3_acceptance_manifest: Path,
    output_root: Path,
    archive_db: Path,
    baseline_reproduction_manifest: Path | None = None,
    checkpoint_source_manifest: Path | None = None,
    included_envs: Sequence[str] | None = None,
    excluded_envs: Sequence[str] = (),
    allow_official_current_checkpoint_warning: bool = False,
    cas_root: Path | None = None,
) -> dict[str, object]:
    """Write a read-only readiness receipt for a scoped pilot campaign."""

    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise LimitedCampaignGateError("LIMITED_CAMPAIGN_GATE_OUTPUT_EXISTS")
    archive = ArchiveStore(archive_db)
    cas_storage_root = Path(cas_root) if cas_root is not None else Path(archive_db).resolve().parent
    cas = ContentAddressedStore(cas_storage_root)
    goal_path, goal, goal_bytes = _load_yaml_mapping(goal_config)
    goal_envs = _string_list(goal.get("envs"), "LIMITED_CAMPAIGN_GOAL_ENVS_INVALID")
    excluded = tuple(_dedupe_preserve(excluded_envs))
    if included_envs is None or len(included_envs) == 0:
        included = tuple(environment for environment in goal_envs if environment not in set(excluded))
    else:
        included = tuple(_dedupe_preserve(included_envs))
    scope = _scope_definition(goal_envs=goal_envs, included=included, excluded=excluded)

    sources = {
        "checkpoint_step_audit": _load_manifest_with_report(
            checkpoint_step_audit_manifest,
            manifest_artifact_type="acwm-m0-checkpoint-step-audit-manifest",
            report_artifact_type="acwm-m0-checkpoint-step-audit",
            cas=cas,
            archive=archive,
        ),
        "raw_failure_batch": _load_source(raw_failure_batch_manifest, cas=cas, archive=archive),
        "constitutional_audit": _load_source(constitutional_audit_manifest, cas=cas, archive=archive),
        "m3_acceptance": _load_source(m3_acceptance_manifest, cas=cas, archive=archive),
    }
    if baseline_reproduction_manifest is not None:
        sources["baseline_reproduction"] = _load_manifest_with_report(
            baseline_reproduction_manifest,
            manifest_artifact_type="acwm-m0-baseline-reproduction-manifest",
            report_artifact_type="acwm-m0-baseline-reproduction-report",
            cas=cas,
            archive=archive,
        )
    if checkpoint_source_manifest is not None:
        sources["checkpoint_source"] = _load_manifest_with_report(
            checkpoint_source_manifest,
            manifest_artifact_type="acwm-m0-checkpoint-source-audit-manifest",
            report_artifact_type="acwm-m0-checkpoint-source-audit",
            cas=cas,
            archive=archive,
        )

    archive_statistics = archive.archive_statistics()
    requirements = _requirements(
        goal_envs=goal_envs,
        included=included,
        excluded=excluded,
        scope=scope,
        sources=sources,
        archive_statistics=archive_statistics,
        allow_official_current_checkpoint_warning=allow_official_current_checkpoint_warning,
    )
    ready = all(item["passed"] is True for item in requirements.values())
    report = {
        "schema_version": 1,
        "artifact_type": "wmloop-limited-campaign-gate",
        "state": "ready" if ready else "blocked",
        "scope_type": "limited_pilot",
        "limited_campaign_allowed": ready,
        "m4_launch_allowed": False,
        "formal_training_allowed": False,
        "checkpoint_policy": {
            "allow_official_current_checkpoint_warning": allow_official_current_checkpoint_warning,
            "claim_boundary": (
                "Paired-delta closed-loop trials may use official-current checkpoints with provenance warnings; "
                "official 100k reproduction claims remain disallowed for warned environments."
                if allow_official_current_checkpoint_warning
                else "Every included environment must pass the strict checkpoint-step audit."
            ),
        },
        "goal_config": {
            "path": str(goal_path),
            "sha256": hashlib.sha256(goal_bytes).hexdigest(),
            "goal_id": goal.get("goal_id"),
        },
        "included_envs": list(included),
        "excluded_envs": list(excluded),
        "requirements": requirements,
        "blockers": _blockers(requirements),
        "archive_statistics": archive_statistics,
        "sources": {name: source["summary"] for name, source in sources.items()},
        "limitations": [
            "This gate is for scoped pilot execution only and must not be used as an M4 launch authorization.",
            "Excluded environments cannot contribute formal claims, cross-environment coverage, or M4 settled-trial counts.",
            "Full M4 remains blocked until the strict phase gate and M4 launch guard are ready.",
        ],
        "next_actions": _next_actions(ready=ready, requirements=requirements, excluded=excluded),
    }
    return _write_report_bundle(report=report, output_root=destination, cas=cas, archive=archive)


def _requirements(
    *,
    goal_envs: Sequence[str],
    included: Sequence[str],
    excluded: Sequence[str],
    scope: Mapping[str, object],
    sources: Mapping[str, Mapping[str, object]],
    archive_statistics: Mapping[str, int],
    allow_official_current_checkpoint_warning: bool,
) -> dict[str, dict[str, object]]:
    checkpoint_report = _payload(sources, "checkpoint_step_audit", report=True)
    checkpoint_source = _payload_or_none(sources, "checkpoint_source", report=True)
    raw_failure = _payload(sources, "raw_failure_batch")
    constitutional = _payload(sources, "constitutional_audit")
    m3 = _payload(sources, "m3_acceptance")
    baseline = _payload_or_none(sources, "baseline_reproduction", report=True)
    return {
        "scope_definition": {
            "passed": scope["state"] == "ready",
            "expected": "Included and excluded environments form a non-empty subset of the goal environments.",
            "observed": scope,
        },
        "generation_zero_baselines_for_scope": {
            "passed": _baseline_coverage_passes(
                baseline=baseline,
                archive_statistics=archive_statistics,
                included=included,
            ),
            "expected": "Generation-zero baseline records and reproduction metrics cover every included environment.",
            "observed": _baseline_observed(
                baseline=baseline,
                archive_statistics=archive_statistics,
                included=included,
            ),
        },
        "checkpoint_steps_for_scope": {
            "passed": _checkpoint_steps_pass(
                checkpoint_report=checkpoint_report,
                checkpoint_source=checkpoint_source,
                included=included,
                excluded=excluded,
                allow_official_current_checkpoint_warning=allow_official_current_checkpoint_warning,
            ),
            "expected": (
                "Every included environment has a 100k checkpoint, unless an included mismatch is proven to be the "
                "official-current file and is used only for paired-delta claims with a provenance warning."
            ),
            "observed": _checkpoint_steps_observed(
                checkpoint_report=checkpoint_report,
                checkpoint_source=checkpoint_source,
                included=included,
                excluded=excluded,
                allow_official_current_checkpoint_warning=allow_official_current_checkpoint_warning,
            ),
        },
        "raw_failure_reports_for_scope": {
            "passed": _raw_failure_passes(raw_failure=raw_failure, included=included),
            "expected": "Ready raw failure reports cover every included environment under the selected goal protocol.",
            "observed": _raw_failure_observed(raw_failure=raw_failure, included=included),
        },
        "constitutional_layer_ready": {
            "passed": _constitutional_passes(constitutional),
            "expected": "The scoped pilot uses a ready constitutional audit and does not mutate the active constitution.",
            "observed": _constitutional_observed(constitutional),
        },
        "m3_strict_acceptance_ready": {
            "passed": m3.get("state") == "ready" and m3.get("strict_m3_pass") is True,
            "expected": "M3 strict acceptance passes before any scoped pilot side effects.",
            "observed": {
                "state": m3.get("state"),
                "strict_m3_pass": m3.get("strict_m3_pass"),
                "blockers": m3.get("blockers", []),
            },
        },
        "full_m4_not_authorized_by_this_gate": {
            "passed": True,
            "expected": "The limited pilot gate never authorizes formal M4 launch.",
            "observed": {"m4_launch_allowed": False, "formal_training_allowed": False},
        },
    }


def _scope_definition(*, goal_envs: Sequence[str], included: Sequence[str], excluded: Sequence[str]) -> dict[str, object]:
    goal_set = set(goal_envs)
    included_set = set(included)
    excluded_set = set(excluded)
    blockers = []
    if not included:
        blockers.append("included_envs_empty")
    unknown = sorted((included_set | excluded_set) - goal_set)
    if unknown:
        blockers.append("env_not_in_goal")
    overlap = sorted(included_set & excluded_set)
    if overlap:
        blockers.append("included_excluded_overlap")
    return {
        "state": "ready" if not blockers else "blocked",
        "goal_env_count": len(goal_envs),
        "included_count": len(included),
        "excluded_count": len(excluded),
        "unknown_envs": unknown,
        "overlap_envs": overlap,
        "blockers": blockers,
    }


def _baseline_coverage_passes(
    *,
    baseline: Mapping[str, Any] | None,
    archive_statistics: Mapping[str, int],
    included: Sequence[str],
) -> bool:
    if archive_statistics.get("baselines", 0) < len(included):
        return False
    if baseline is None:
        return True
    by_environment = baseline.get("by_environment_unweighted")
    if not isinstance(by_environment, Mapping):
        return False
    return all(environment in by_environment for environment in included)


def _baseline_observed(
    *,
    baseline: Mapping[str, Any] | None,
    archive_statistics: Mapping[str, int],
    included: Sequence[str],
) -> dict[str, object]:
    covered: list[str] = []
    if baseline is not None and isinstance(baseline.get("by_environment_unweighted"), Mapping):
        by_environment = baseline["by_environment_unweighted"]
        covered = sorted(environment for environment in included if environment in by_environment)
    return {
        "archive_baselines": archive_statistics.get("baselines", 0),
        "required_for_scope": len(included),
        "baseline_reproduction_state": baseline.get("state") if baseline is not None else "not_provided",
        "covered_envs": covered,
        "missing_envs": sorted(set(included) - set(covered)) if baseline is not None else [],
    }


def _checkpoint_steps_pass(
    *,
    checkpoint_report: Mapping[str, Any],
    checkpoint_source: Mapping[str, Any] | None,
    included: Sequence[str],
    excluded: Sequence[str],
    allow_official_current_checkpoint_warning: bool,
) -> bool:
    observed = _checkpoint_steps_observed(
        checkpoint_report=checkpoint_report,
        checkpoint_source=checkpoint_source,
        included=included,
        excluded=excluded,
        allow_official_current_checkpoint_warning=allow_official_current_checkpoint_warning,
    )
    warning_envs = set(observed["included_provenance_warning"])
    non_pass = set(observed["included_non_pass"])
    return (
        observed["included_missing"] == []
        and observed["mismatch_outside_excluded"] == []
        and (non_pass == set() or (allow_official_current_checkpoint_warning and non_pass == warning_envs))
    )


def _checkpoint_steps_observed(
    *,
    checkpoint_report: Mapping[str, Any],
    checkpoint_source: Mapping[str, Any] | None,
    included: Sequence[str],
    excluded: Sequence[str],
    allow_official_current_checkpoint_warning: bool,
) -> dict[str, object]:
    records = _records_by_environment(checkpoint_report.get("records"))
    official_current = _official_current_warning_records(checkpoint_source)
    included_non_pass = []
    included_provenance_warning = []
    for environment in included:
        record = records.get(environment)
        if record is None:
            continue
        if record.get("status") != "pass" or record.get("observed_step") != record.get("expected_step"):
            included_non_pass.append(environment)
            if environment in official_current:
                included_provenance_warning.append(environment)
    mismatches = sorted(
        environment for environment, record in records.items() if record.get("status") != "pass"
    )
    ignored_for_scope = set(included_provenance_warning) if allow_official_current_checkpoint_warning else set()
    return {
        "state": checkpoint_report.get("state"),
        "expected_step": checkpoint_report.get("expected_step"),
        "checkpoint_source_state": checkpoint_source.get("state") if checkpoint_source is not None else "not_provided",
        "allow_official_current_checkpoint_warning": allow_official_current_checkpoint_warning,
        "included_missing": sorted(set(included) - set(records)),
        "included_non_pass": sorted(included_non_pass),
        "included_provenance_warning": sorted(included_provenance_warning),
        "official_current_warning_records": {
            environment: official_current[environment] for environment in sorted(set(included) & set(official_current))
        },
        "mismatch_envs": mismatches,
        "excluded_envs": list(excluded),
        "mismatch_outside_excluded": sorted(set(mismatches) - set(excluded) - ignored_for_scope),
        "claim_boundary": (
            "Warned environments may contribute paired baseline-vs-candidate deltas only; do not claim official 100k reproduction for them."
            if included_provenance_warning
            else "No included official-current checkpoint warning was needed."
        ),
    }


def _official_current_warning_records(checkpoint_source: Mapping[str, Any] | None) -> dict[str, dict[str, object]]:
    if checkpoint_source is None:
        return {}
    records = checkpoint_source.get("records")
    if not isinstance(records, list):
        return {}
    result: dict[str, dict[str, object]] = {}
    for item in records:
        if not isinstance(item, Mapping):
            continue
        environment = item.get("environment")
        if not isinstance(environment, str) or not environment:
            continue
        if item.get("source_status") != "remote_current_matches_local_mismatch":
            continue
        result[environment] = {
            "source_status": item.get("source_status"),
            "observed_step": item.get("observed_step"),
            "expected_step": item.get("expected_step"),
            "local_hf_hash": item.get("local_hf_hash"),
            "checkpoint_relative_path": item.get("checkpoint_relative_path"),
        }
    return result


def _raw_failure_passes(*, raw_failure: Mapping[str, Any], included: Sequence[str]) -> bool:
    observed = _raw_failure_observed(raw_failure=raw_failure, included=included)
    return (
        raw_failure.get("state") == "ready"
        and observed["included_missing"] == []
        and observed["included_blocked"] == []
    )


def _raw_failure_observed(*, raw_failure: Mapping[str, Any], included: Sequence[str]) -> dict[str, object]:
    reports = raw_failure.get("reports")
    reported_envs = set()
    if isinstance(reports, list):
        for item in reports:
            if isinstance(item, Mapping) and isinstance(item.get("environment"), str):
                reported_envs.add(item["environment"])
    blocked = raw_failure.get("blocked_records")
    blocked_envs = set()
    if isinstance(blocked, list):
        for item in blocked:
            if isinstance(item, Mapping) and isinstance(item.get("environment"), str):
                blocked_envs.add(item["environment"])
    return {
        "state": raw_failure.get("state"),
        "report_count": raw_failure.get("report_count"),
        "blocked_count": raw_failure.get("blocked_count"),
        "included_missing": sorted(set(included) - reported_envs),
        "included_blocked": sorted(set(included) & blocked_envs),
    }


def _constitutional_passes(constitutional: Mapping[str, Any]) -> bool:
    return (
        constitutional.get("artifact_type") == "wmloop-constitutional-audit-manifest"
        and constitutional.get("state") == "ready"
        and _int(constitutional.get("ready_surface_count")) == 5
        and _int(constitutional.get("surface_count")) == 5
        and _int(constitutional.get("blocker_count")) == 0
        and constitutional.get("active_constitution_mutated") is False
        and constitutional.get("m4_launch_allowed") is False
        and constitutional.get("formal_training_allowed") is False
    )


def _constitutional_observed(constitutional: Mapping[str, Any]) -> dict[str, object]:
    return {
        "state": constitutional.get("state"),
        "ready_surface_count": constitutional.get("ready_surface_count"),
        "surface_count": constitutional.get("surface_count"),
        "blocker_count": constitutional.get("blocker_count"),
        "active_constitution_mutated": constitutional.get("active_constitution_mutated"),
        "m4_launch_allowed": constitutional.get("m4_launch_allowed"),
        "formal_training_allowed": constitutional.get("formal_training_allowed"),
    }


def _records_by_environment(value: object) -> dict[str, Mapping[str, Any]]:
    records: dict[str, Mapping[str, Any]] = {}
    if not isinstance(value, list):
        return records
    for item in value:
        if isinstance(item, Mapping) and isinstance(item.get("environment"), str):
            records[item["environment"]] = item
    return records


def _load_manifest_with_report(
    path: Path,
    *,
    manifest_artifact_type: str,
    report_artifact_type: str,
    cas: ContentAddressedStore,
    archive: ArchiveStore,
) -> dict[str, object]:
    source = _load_source(path, cas=cas, archive=archive)
    payload = source["payload"]
    if not isinstance(payload, Mapping) or payload.get("artifact_type") != manifest_artifact_type:
        raise LimitedCampaignGateError(f"LIMITED_CAMPAIGN_MANIFEST_INVALID:{path}")
    report_path = payload.get("report_path")
    if not isinstance(report_path, str) or not report_path:
        raise LimitedCampaignGateError(f"LIMITED_CAMPAIGN_REPORT_PATH_INVALID:{path}")
    report_source = _load_source(Path(report_path), cas=cas, archive=archive)
    report = report_source["payload"]
    if not isinstance(report, Mapping) or report.get("artifact_type") != report_artifact_type:
        raise LimitedCampaignGateError(f"LIMITED_CAMPAIGN_REPORT_INVALID:{report_path}")
    source["report_payload"] = report
    source["report_summary"] = report_source["summary"]
    return source


def _load_source(path: Path, *, cas: ContentAddressedStore, archive: ArchiveStore) -> dict[str, object]:
    resolved = Path(path).resolve(strict=True)
    try:
        payload_bytes = resolved.read_bytes()
        payload = json.loads(payload_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise LimitedCampaignGateError(f"LIMITED_CAMPAIGN_SOURCE_INVALID:{resolved}") from exc
    if not isinstance(payload, dict):
        raise LimitedCampaignGateError(f"LIMITED_CAMPAIGN_SOURCE_NOT_OBJECT:{resolved}")
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


def _load_yaml_mapping(path: Path) -> tuple[Path, Mapping[str, Any], bytes]:
    try:
        import yaml  # type: ignore[import-not-found]

        resolved = Path(path).resolve(strict=True)
        payload_bytes = resolved.read_bytes()
        payload = yaml.safe_load(payload_bytes.decode("utf-8"))
    except Exception as exc:
        raise LimitedCampaignGateError(f"LIMITED_CAMPAIGN_GOAL_CONFIG_INVALID:{path}") from exc
    if not isinstance(payload, Mapping):
        raise LimitedCampaignGateError(f"LIMITED_CAMPAIGN_GOAL_CONFIG_INVALID:{resolved}")
    return resolved, payload, payload_bytes


def _payload(sources: Mapping[str, Mapping[str, object]], name: str, *, report: bool = False) -> Mapping[str, Any]:
    try:
        source = sources[name]
    except KeyError as exc:
        raise LimitedCampaignGateError(f"LIMITED_CAMPAIGN_SOURCE_MISSING:{name}") from exc
    key = "report_payload" if report else "payload"
    payload = source.get(key)
    if not isinstance(payload, Mapping):
        raise LimitedCampaignGateError(f"LIMITED_CAMPAIGN_SOURCE_INVALID:{name}")
    return payload


def _payload_or_none(
    sources: Mapping[str, Mapping[str, object]],
    name: str,
    *,
    report: bool = False,
) -> Mapping[str, Any] | None:
    if name not in sources:
        return None
    return _payload(sources, name, report=report)


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
        _write_bytes_atomic(temporary / "limited-campaign-gate.json", report_bytes)
        _write_bytes_atomic(temporary / "limited-campaign-gate.md", markdown_bytes)
        report_ref = cas.put_bytes(report_bytes, media_type="application/json").uri
        markdown_ref = cas.put_bytes(markdown_bytes, media_type="text/markdown").uri
        archive.record_artifact_reference(report_ref)
        archive.record_artifact_reference(markdown_ref)
        manifest = {
            "schema_version": 1,
            "artifact_type": "wmloop-limited-campaign-gate-manifest",
            "state": report["state"],
            "scope_type": report["scope_type"],
            "limited_campaign_allowed": report["limited_campaign_allowed"],
            "m4_launch_allowed": False,
            "formal_training_allowed": False,
            "checkpoint_policy": report["checkpoint_policy"],
            "included_envs": report["included_envs"],
            "excluded_envs": report["excluded_envs"],
            "blockers": report["blockers"],
            "report_path": str(destination / "limited-campaign-gate.json"),
            "markdown_path": str(destination / "limited-campaign-gate.md"),
            "cas_refs": {
                "limited_campaign_gate_json": report_ref,
                "limited_campaign_gate_markdown": markdown_ref,
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
        "# Limited Campaign Gate",
        "",
        f"State: `{report['state']}`",
        f"Limited campaign allowed: `{report['limited_campaign_allowed']}`",
        f"M4 launch allowed: `{report['m4_launch_allowed']}`",
        f"Included envs: `{','.join(report['included_envs'])}`",
        f"Excluded envs: `{','.join(report['excluded_envs'])}`",
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


def _next_actions(
    *,
    ready: bool,
    requirements: Mapping[str, Mapping[str, object]],
    excluded: Sequence[str],
) -> list[str]:
    if ready:
        return [
            "Run scoped closed-loop pilot only on included_envs.",
            "Keep excluded_envs out of formal claims until their checkpoint and baseline audits pass.",
        ]
    actions = []
    for name, requirement in requirements.items():
        if requirement.get("passed") is not True:
            actions.append(f"Resolve limited campaign blocker: {name}.")
    if excluded:
        actions.append(f"Do not include excluded environments yet: {','.join(excluded)}.")
    return actions


def _blockers(requirements: Mapping[str, Mapping[str, object]]) -> list[dict[str, object]]:
    return [
        {"requirement": name, "expected": item["expected"], "observed": item["observed"]}
        for name, item in requirements.items()
        if item.get("passed") is not True
    ]


def _dedupe_preserve(items: Sequence[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _string_list(value: object, error_code: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise LimitedCampaignGateError(error_code)
    return list(value)


def _int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    return 0


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise LimitedCampaignGateError("LIMITED_CAMPAIGN_GATE_OUTPUT_EXISTS")
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
    run = commands.add_parser("run", help="audit a scoped non-formal campaign pilot")
    run.add_argument("--goal-config", type=Path, required=True)
    run.add_argument("--checkpoint-step-audit-manifest", type=Path, required=True)
    run.add_argument("--raw-failure-batch-manifest", type=Path, required=True)
    run.add_argument("--constitutional-audit-manifest", type=Path, required=True)
    run.add_argument("--m3-acceptance-manifest", type=Path, required=True)
    run.add_argument("--baseline-reproduction-manifest", type=Path)
    run.add_argument("--checkpoint-source-manifest", type=Path)
    run.add_argument("--include-env", action="append", default=[])
    run.add_argument("--exclude-env", action="append", default=[])
    run.add_argument("--allow-official-current-checkpoint-warning", action="store_true")
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--archive-db", type=Path, required=True)
    run.add_argument("--cas-root", type=Path)
    args = parser.parse_args(argv)
    if args.command == "run":
        manifest = run_limited_campaign_gate(
            goal_config=args.goal_config,
            checkpoint_step_audit_manifest=args.checkpoint_step_audit_manifest,
            raw_failure_batch_manifest=args.raw_failure_batch_manifest,
            constitutional_audit_manifest=args.constitutional_audit_manifest,
            m3_acceptance_manifest=args.m3_acceptance_manifest,
            baseline_reproduction_manifest=args.baseline_reproduction_manifest,
            checkpoint_source_manifest=args.checkpoint_source_manifest,
            included_envs=args.include_env,
            excluded_envs=args.exclude_env,
            allow_official_current_checkpoint_warning=args.allow_official_current_checkpoint_warning,
            output_root=args.output_root,
            archive_db=args.archive_db,
            cas_root=args.cas_root,
        )
        print(json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
