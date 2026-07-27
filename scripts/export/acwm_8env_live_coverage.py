#!/usr/bin/env python3
"""Export live eight-environment ACWM closed-loop coverage."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.export.acwm_autoloop_replenisher import (
    EVENT_SEMANTIC_REQUIRED_ENVIRONMENTS,
    _active_daemon_queue_paths,
    _attempted_parameter_signatures,
    _officially_positive_environments,
    _pending_internal_positive_environments,
    _pending_positive_confirmation_environments,
    _quality_discovery_terminal_positive_environments,
    _running_screen_environments,
    _checkpoint_delta_confirmation_key,
    _formally_confirmed_checkpoint_delta_keys,
    _formally_confirmed_runtime_only_keys,
    _event_semantic_positive_methods,
    _training_finalization_records,
)
from wmloop.acwm_data import CANONICAL_ACWM_ENVIRONMENTS
from wmloop.execute.acwm_primitive_routes import (
    INVALIDATED_QUALITY_PRIMITIVES,
    RUNTIME_ONLY_PRIMITIVES,
    TRAINING_QUALITY_SCREEN_PRIMITIVES,
)


DEFAULT_REPORT_ROOT = ROOT / "results/reports"
DEFAULT_OUTPUT_ROOT = DEFAULT_REPORT_ROOT / "acwm-8env-live-coverage-r1"


class Acwm8EnvLiveCoverageError(RuntimeError):
    """Live coverage export failed closed."""


def export_live_coverage(*, report_root: Path, output_root: Path) -> dict[str, object]:
    reports = Path(report_root).resolve(strict=True)
    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise Acwm8EnvLiveCoverageError("ACWM_8ENV_LIVE_COVERAGE_OUTPUT_EXISTS")

    report = build_live_coverage(report_root=reports)
    return _write_coverage_bundle(destination=destination, report=report)


def build_live_coverage(
    *, report_root: Path, active_queue_paths: set[Path] | None = None
) -> dict[str, object]:
    reports = Path(report_root).resolve(strict=True)

    confirmed = _officially_positive_environments(reports) | _quality_discovery_terminal_positive_environments(
        reports
    )
    confirmation_pending = _pending_positive_confirmation_environments(reports)
    internal_positive = _pending_internal_positive_environments(reports)
    running = _running_screen_environments(reports)
    attempted = _attempted_parameter_signatures(reports)
    active_queues = (
        {Path(path).resolve() for path in active_queue_paths}
        if active_queue_paths is not None
        else _active_daemon_queue_paths(reports)
    )
    queue_counts = _active_queue_counts(active_queues)
    confirmed_methods = _confirmed_methods(reports)
    official_screen_passes = _official_screen_passes(reports)
    event_positive_methods = _event_semantic_positive_methods(reports)

    rows: list[dict[str, object]] = []
    for spec in CANONICAL_ACWM_ENVIRONMENTS:
        environment = spec.environment
        event_required = environment in EVENT_SEMANTIC_REQUIRED_ENVIRONMENTS
        event_validated = bool(event_positive_methods.get(environment))
        claim_ready = environment in confirmed and (not event_required or event_validated)
        state = coverage_state(
            confirmed=environment in confirmed,
            confirmation_pending=environment in confirmation_pending,
            official_screen_pass=bool(official_screen_passes.get(environment)),
            internal_positive=environment in internal_positive,
            running=environment in running,
            event_required=event_required,
            event_validated=event_validated,
        )
        rows.append(
            {
                "environment": environment,
                "coverage_state": state,
                "formally_confirmed": environment in confirmed,
                "claim_ready": claim_ready,
                "confirmed_methods": ";".join(confirmed_methods.get(environment, ())),
                "confirmed_method_count": len(confirmed_methods.get(environment, ())),
                "official_screen_pass_methods": ";".join(official_screen_passes.get(environment, ())),
                "event_semantic_required": event_required,
                "event_semantic_validated": event_validated,
                "event_semantic_positive_methods": ";".join(event_positive_methods.get(environment, ())),
                "confirmation_pending": environment in confirmation_pending,
                "internal_positive_pending_official": environment in internal_positive,
                "screen_running": environment in running,
                "effective_attempted_signature_count": sum(1 for item in attempted if item[0] == environment),
                "active_queue_count": queue_counts.get(environment, 0),
                "next_action": _next_action(state),
            }
        )

    summary = {
        "environment_count": len(rows),
        "formally_confirmed_environment_count": sum(bool(row["formally_confirmed"]) for row in rows),
        "claim_ready_environment_count": sum(bool(row["claim_ready"]) for row in rows),
        "event_validation_pending_environment_count": sum(
            bool(row["event_semantic_required"]) and not bool(row["event_semantic_validated"])
            for row in rows
        ),
        "confirmation_pending_environment_count": sum(bool(row["confirmation_pending"]) for row in rows),
        "running_screen_environment_count": sum(bool(row["screen_running"]) for row in rows),
        "active_daemon_queue_count": len(active_queues),
        "effective_attempted_signature_count": len(attempted),
        "coverage_complete": all(bool(row["claim_ready"]) for row in rows),
    }
    return {
        "schema_version": 1,
        "artifact_type": "wmloop-acwm-8env-live-coverage",
        "state": "ready",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "protocol": "512 screen -> official 50-step gate -> independent 800/1000 gates -> best passing checkpoint -> environment-specific semantic event gate",
        "summary": summary,
        "rows": rows,
        "sources": {"report_root": str(reports)},
    }


def _write_coverage_bundle(*, destination: Path, report: Mapping[str, object]) -> dict[str, object]:
    summary = report["summary"]
    if not isinstance(summary, Mapping):
        raise Acwm8EnvLiveCoverageError("ACWM_8ENV_LIVE_COVERAGE_SUMMARY_INVALID")
    rows = report["rows"]
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise Acwm8EnvLiveCoverageError("ACWM_8ENV_LIVE_COVERAGE_ROWS_INVALID")

    destination.mkdir(mode=0o700, parents=True)
    report_path = destination / "acwm-8env-live-coverage.json"
    csv_path = destination / "acwm-8env-live-coverage.csv"
    markdown_path = destination / "acwm-8env-live-coverage.md"
    _write_json(report_path, report)
    _write_csv(csv_path, rows)
    markdown_path.write_text(_render_markdown(report), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "artifact_type": "wmloop-acwm-8env-live-coverage-manifest",
        "state": "ready",
        "summary": summary,
        "report_path": str(report_path),
        "csv_path": str(csv_path),
        "markdown_path": str(markdown_path),
    }
    _write_json(destination / "manifest.json", manifest)
    return manifest


def coverage_state(
    *,
    confirmed: bool,
    confirmation_pending: bool,
    official_screen_pass: bool,
    internal_positive: bool,
    running: bool,
    event_required: bool = False,
    event_validated: bool = False,
) -> str:
    if confirmed:
        if event_required and not event_validated:
            return "metric_positive_pending_event_validation"
        return "formally_confirmed_positive"
    if confirmation_pending:
        return "confirmation_pending"
    if official_screen_pass:
        return "official_screen_pass_pending_confirmation"
    if internal_positive:
        return "internal_positive_pending_official_gate"
    if running:
        return "screen_running"
    return "exploring_unconfirmed"


def _confirmed_methods(report_root: Path) -> dict[str, tuple[str, ...]]:
    values: dict[str, set[str]] = {}
    for record in _training_finalization_records(report_root):
        if record["confirmation_passed"] is True:
            values.setdefault(str(record["environment"]), set()).add(str(record["primitive"]))
    for environment, primitive, _ in _formally_confirmed_runtime_only_keys(report_root):
        values.setdefault(environment, set()).add(primitive)
    confirmed_checkpoint_delta = _formally_confirmed_checkpoint_delta_keys(report_root)
    for path in {
        *report_root.glob("acwm-autoloop-official-gate-*/manifest.json"),
        *report_root.glob("acwm-official-gate-*/manifest.json"),
    }:
        manifest = _load_optional_json(path)
        gate = manifest.get("official_quality_gate")
        environment = str(manifest.get("environment") or _environment_from_path(path))
        primitive = str(manifest.get("primitive") or _primitive_from_path(path, environment))
        if (
            not environment
            or not primitive
            or primitive in INVALIDATED_QUALITY_PRIMITIVES
            or primitive in TRAINING_QUALITY_SCREEN_PRIMITIVES
            or primitive in RUNTIME_ONLY_PRIMITIVES
            or manifest.get("state") != "ready"
            or not isinstance(gate, Mapping)
            or gate.get("pass") is not True
        ):
            continue
        if primitive == "checkpoint_delta_scaling":
            key = _checkpoint_delta_confirmation_key(manifest)
            if key is None or key not in confirmed_checkpoint_delta:
                continue
        values.setdefault(environment, set()).add(primitive)
    return {environment: tuple(sorted(methods)) for environment, methods in values.items()}


def _official_screen_passes(report_root: Path) -> dict[str, tuple[str, ...]]:
    values: dict[str, set[str]] = {}
    for path in report_root.glob("acwm-autoloop-official-gate-*/manifest.json"):
        manifest = _load_optional_json(path)
        gate = manifest.get("official_quality_gate")
        environment = str(manifest.get("environment") or _environment_from_path(path))
        primitive = str(manifest.get("primitive") or _primitive_from_path(path, environment))
        if (
            not environment
            or not primitive
            or primitive in INVALIDATED_QUALITY_PRIMITIVES
            or primitive in RUNTIME_ONLY_PRIMITIVES
            or manifest.get("state") != "ready"
            or not isinstance(gate, Mapping)
            or gate.get("pass") is not True
        ):
            continue
        values.setdefault(environment, set()).add(primitive)
    return {environment: tuple(sorted(methods)) for environment, methods in values.items()}


def _active_queue_counts(queue_paths: set[Path]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in queue_paths:
        queue = _load_optional_json(path)
        environments = {
            str(row.get("environment") or "")
            for row in queue.get("rows", [])
            if isinstance(row, Mapping) and row.get("environment")
        }
        for environment in environments:
            counts[environment] = counts.get(environment, 0) + 1
    return counts


def _environment_from_path(path: Path) -> str:
    name = path.parent.name
    for spec in CANONICAL_ACWM_ENVIRONMENTS:
        marker = f"-{spec.environment}-"
        if marker in name:
            return spec.environment
    return ""


def _primitive_from_path(path: Path, environment: str) -> str:
    if not environment:
        return ""
    suffix = path.parent.name.split(f"-{environment}-", 1)[-1]
    for marker in ("-s", "-a"):
        if marker in suffix:
            return suffix.split(marker, 1)[0]
    return ""


def _next_action(state: str) -> str:
    return {
        "formally_confirmed_positive": "Retain evidence and route transferable experience to uncovered environments.",
        "metric_positive_pending_event_validation": "Run an event-covering rollout and continue diagnosis-matched search until physical event improvement passes.",
        "confirmation_pending": "Finish the 800/1000 ladder and select the best passing checkpoint.",
        "official_screen_pass_pending_confirmation": "Launch the staged 1k confirmation.",
        "internal_positive_pending_official_gate": "Run the frozen official 50-step metric gate.",
        "screen_running": "Finish the 512 screen and apply the official gate if internally positive.",
        "exploring_unconfirmed": "Continue diagnosis-matched primitive search with effective queue leases.",
    }[state]


def _render_markdown(report: Mapping[str, object]) -> str:
    summary = report["summary"]
    rows = report["rows"]
    assert isinstance(summary, Mapping) and isinstance(rows, list)
    lines = [
        "# ACWM 8-Environment Live Coverage",
        "",
        f"Protocol: `{report['protocol']}`",
        "",
        f"Confirmed environments: `{summary['formally_confirmed_environment_count']}/{summary['environment_count']}`",
        f"Claim-ready environments: `{summary['claim_ready_environment_count']}/{summary['environment_count']}`",
        "",
        "| Environment | State | Confirmed methods | Event methods | Attempts | Active queues | Next action |",
        "|:--|:--|:--|:--|--:|--:|:--|",
    ]
    for row in rows:
        assert isinstance(row, Mapping)
        lines.append(
            "| {environment} | {coverage_state} | {methods} | {event_methods} | {attempts} | {queues} | {next_action} |".format(
                environment=row["environment"],
                coverage_state=row["coverage_state"],
                methods=row["confirmed_methods"] or "none",
                event_methods=row["event_semantic_positive_methods"] or "none",
                attempts=row["effective_attempted_signature_count"],
                queues=row["active_queue_count"],
                next_action=row["next_action"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_optional_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", nargs="?")
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)
    manifest = export_live_coverage(report_root=args.report_root, output_root=args.output_root)
    print(json.dumps(manifest, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
