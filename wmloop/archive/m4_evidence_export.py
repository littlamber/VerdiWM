"""Export numeric M4/M5 evidence tables from the settled archive."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import re
import shutil
import sqlite3
import statistics
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from wmloop.archive.store import ArchiveStore, ContentAddressedStore


class M4EvidenceExportError(RuntimeError):
    """M4 evidence export failed closed."""


def run_m4_evidence_export(
    *,
    output_root: Path,
    archive_db: Path,
    cas_root: Path | None = None,
    m4_completion_manifest: Path | None = None,
    verifier_meta_validation_manifest: Path | None = None,
    consolidation_manifest: Path | None = None,
    settled_trial_target: int = 150,
) -> dict[str, object]:
    """Write read-only numeric evidence exports for paper consolidation."""

    if settled_trial_target < 1:
        raise M4EvidenceExportError("M4_EVIDENCE_SETTLED_TARGET_INVALID")
    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise M4EvidenceExportError("M4_EVIDENCE_OUTPUT_EXISTS")
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"

    archive = ArchiveStore(archive_db)
    cas = ContentAddressedStore(cas_root if cas_root is not None else Path(archive_db).resolve().parent)
    source_manifests = _load_source_manifests(
        {
            "m4_completion": m4_completion_manifest,
            "verifier_meta_validation": verifier_meta_validation_manifest,
            "m5_consolidation": consolidation_manifest,
        }
    )
    archive_summary = _archive_summary(archive)
    trial_rows = _trial_rows(archive_db=archive_db, cas=cas)
    evidence_trial_rows = _evidence_trial_rows(trial_rows)
    cell_rows = _cell_rows(archive)
    env_primitive_rows = _aggregate_trial_rows(evidence_trial_rows, group_keys=("environment", "primitive_family"))
    env_rows = _aggregate_trial_rows(evidence_trial_rows, group_keys=("environment",))
    best_recipe_rows = _best_recipe_rows(cell_rows)
    verdict_rows = _verdict_count_rows(trial_rows, scope="archive") + _verdict_count_rows(
        evidence_trial_rows,
        scope="evidence",
    )
    cost_rows = _cost_rows(evidence_trial_rows)
    blockers = _blockers(
        archive_summary=archive_summary,
        trial_rows=trial_rows,
        evidence_trial_rows=evidence_trial_rows,
        cell_rows=cell_rows,
        source_manifests=source_manifests,
        settled_trial_target=settled_trial_target,
    )
    state = "ready" if not blockers else "blocked"
    report = {
        "schema_version": 1,
        "artifact_type": "wmloop-m4-evidence-export-report",
        "state": state,
        "m4_evidence_export_complete": state == "ready",
        "archive_db": str(Path(archive_db).resolve()),
        "settled_trial_target": settled_trial_target,
        "archive_summary": archive_summary,
        "source_manifests": source_manifests,
        "trial_count": len(trial_rows),
        "evidence_trial_count": len(evidence_trial_rows),
        "excluded_trial_count": len(trial_rows) - len(evidence_trial_rows),
        "cell_count": len(cell_rows),
        "env_primitive_count": len(env_primitive_rows),
        "best_recipe_count": len(best_recipe_rows),
        "blockers": blockers,
        "limitations": [
            "This export is read-only and does not launch training or evaluation.",
            "Best-recipe rows are empirical cell summaries, not causal guarantees.",
            "Surrogate usage remains proposal sorting only; exported prediction values are not verifier inputs.",
        ],
    }
    files = {
        "m4_evidence_export_json": ("m4-evidence-export.json", _canonical_json_bytes(report), "application/json"),
        "m4_evidence_export_markdown": (
            "m4-evidence-export.md",
            _render_markdown(
                report=report,
                env_primitive_rows=env_primitive_rows,
                best_recipe_rows=best_recipe_rows,
                cost_rows=cost_rows,
            ).encode("utf-8"),
            "text/markdown",
        ),
        "trial_records_csv": ("tables/trial-records.csv", _csv_bytes(trial_rows), "text/csv"),
        "env_summary_csv": ("tables/env-summary.csv", _csv_bytes(env_rows), "text/csv"),
        "env_primitive_summary_csv": (
            "tables/env-primitive-summary.csv",
            _csv_bytes(env_primitive_rows),
            "text/csv",
        ),
        "cell_summary_csv": ("tables/cell-summary.csv", _csv_bytes(cell_rows), "text/csv"),
        "best_recipe_csv": ("tables/best-recipe.csv", _csv_bytes(best_recipe_rows), "text/csv"),
        "verdict_counts_csv": ("tables/verdict-counts.csv", _csv_bytes(verdict_rows), "text/csv"),
        "cost_benefit_csv": ("tables/cost-benefit.csv", _csv_bytes(cost_rows), "text/csv"),
        "best_recipe_latex": (
            "latex/best-recipe.tex",
            _latex_table(best_recipe_rows, "Best empirical cell per environment").encode("utf-8"),
            "text/x-tex",
        ),
        "env_primitive_summary_latex": (
            "latex/env-primitive-summary.tex",
            _latex_table(env_primitive_rows, "Environment by primitive summary").encode("utf-8"),
            "text/x-tex",
        ),
    }
    try:
        temporary.mkdir(mode=0o700)
        cas_refs: dict[str, str] = {}
        for key, (relative, payload, media_type) in files.items():
            _write_bytes_atomic(temporary / relative, payload)
            ref = cas.put_bytes(payload, media_type=media_type).uri
            archive.record_artifact_reference(ref)
            cas_refs[key] = ref
        manifest = {
            "schema_version": 1,
            "artifact_type": "wmloop-m4-evidence-export-manifest",
            "state": state,
            "m4_evidence_export_complete": state == "ready",
            "settled_trial_target": settled_trial_target,
            "archive_summary": archive_summary,
            "trial_count": len(trial_rows),
            "evidence_trial_count": len(evidence_trial_rows),
            "excluded_trial_count": len(trial_rows) - len(evidence_trial_rows),
            "cell_count": len(cell_rows),
            "env_primitive_count": len(env_primitive_rows),
            "ready_best_recipe_count": len(best_recipe_rows),
            "blocker_count": len(blockers),
            "report_path": str(destination / "m4-evidence-export.json"),
            "markdown_path": str(destination / "m4-evidence-export.md"),
            "tables_dir": str(destination / "tables"),
            "latex_dir": str(destination / "latex"),
            "cas_refs": cas_refs,
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


def _archive_summary(archive: ArchiveStore) -> dict[str, int]:
    stats = archive.archive_statistics()
    cells = archive.list_cells()
    return {
        **stats,
        "cells": len(cells),
        "cell_observations": sum(record.stats.visits for record in cells),
    }


def _load_source_manifests(paths: Mapping[str, Path | None]) -> dict[str, dict[str, object]]:
    loaded: dict[str, dict[str, object]] = {}
    for name, path in paths.items():
        if path is None:
            continue
        resolved = Path(path).resolve(strict=True)
        payload = _load_json_mapping(resolved)
        loaded[name] = {
            "path": str(resolved),
            "artifact_type": payload.get("artifact_type"),
            "state": payload.get("state"),
            "blocker_count": len(payload.get("blockers") or []),
        }
    return loaded


def _trial_rows(*, archive_db: Path, cas: ContentAddressedStore) -> list[dict[str, object]]:
    connection = sqlite3.connect(archive_db)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            """
            SELECT trial_id, proposal_id, goal_id, library_version, failure_context_ref,
                   verdict_ref, receipt_ref, cost_json, settlement_json
            FROM trials ORDER BY trial_id
            """
        ).fetchall()
    finally:
        connection.close()
    exported: list[dict[str, object]] = []
    for row in rows:
        failure = _read_json_ref(cas, str(row["failure_context_ref"]))
        verdict = _read_json_ref(cas, str(row["verdict_ref"]))
        receipt = _read_json_ref(cas, str(row["receipt_ref"]))
        cost = _loads_mapping(row["cost_json"], "M4_EVIDENCE_COST_JSON_INVALID")
        settlement = _loads_mapping(row["settlement_json"], "M4_EVIDENCE_SETTLEMENT_JSON_INVALID")
        proposal_id = str(row["proposal_id"])
        parsed = _parse_proposal(proposal_id)
        environment = str(failure.get("env") or parsed.get("environment") or "unknown")
        primitive = str(parsed.get("primitive_family") or _receipt_primitive(receipt) or "unknown")
        gpu_hours = _finite_or_none(cost.get("gpu_hours"))
        delta = _nested_finite(verdict, ("delta_m_ver", "ladder_auc_psnr_envmax"))
        evidence_exclusion_reason = _evidence_exclusion_reason(
            environment=environment,
            primitive_family=primitive,
            delta=delta,
        )
        action_following = verdict.get("action_following_gate")
        action_observed = _finite_or_none(action_following.get("observed")) if isinstance(action_following, Mapping) else None
        action_pass = action_following.get("pass") if isinstance(action_following, Mapping) else None
        exported.append(
            {
                "trial_id": str(row["trial_id"]),
                "proposal_id": proposal_id,
                "environment": environment,
                "primitive_family": primitive,
                "goal_id": str(row["goal_id"]),
                "library_version": str(row["library_version"]),
                "verdict": str(verdict.get("verdict", "")),
                "settlement_state": str(settlement.get("state", "")),
                "delta_ladder_auc_psnr_envmax": delta,
                "evidence_eligible": evidence_exclusion_reason is None,
                "evidence_exclusion_reason": evidence_exclusion_reason or "",
                "action_following_observed": action_observed,
                "action_following_pass": action_pass,
                "gpu_hours": gpu_hours,
                "failure_context_ref": str(row["failure_context_ref"]),
                "verdict_ref": str(row["verdict_ref"]),
                "receipt_ref": str(row["receipt_ref"]),
            }
        )
    return exported


def _evidence_exclusion_reason(
    *,
    environment: str,
    primitive_family: str,
    delta: float | None,
) -> str | None:
    if environment == "unknown":
        return "ENVIRONMENT_UNKNOWN"
    if primitive_family == "unknown":
        return "PRIMITIVE_UNKNOWN"
    if delta is None:
        return "DELTA_UNAVAILABLE"
    return None


def _evidence_trial_rows(rows: Sequence[Mapping[str, object]]) -> list[Mapping[str, object]]:
    return [row for row in rows if row.get("evidence_eligible") is True]


def _cell_rows(archive: ArchiveStore) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in archive.list_cells():
        rows.append(
            {
                "environment": record.cell.environment,
                "layer": record.cell.layer,
                "primitive_family": record.cell.primitive_family,
                "parameter_bucket": record.cell.parameter_bucket,
                "visits": record.stats.visits,
                "mean_verified_gain": record.stats.mean_verified_improvement,
                "effect_sign": _effect_sign(record.stats.mean_verified_improvement),
            }
        )
    rows.sort(key=lambda row: (str(row["environment"]), -float(row["mean_verified_gain"]), str(row["primitive_family"])))
    return rows


def _aggregate_trial_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    group_keys: Sequence[str],
) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[Mapping[str, object]]] = {}
    for row in rows:
        key = tuple(row.get(group_key, "") for group_key in group_keys)
        groups.setdefault(key, []).append(row)
    exported: list[dict[str, object]] = []
    for key, members in groups.items():
        deltas = [_finite_or_none(member.get("delta_ladder_auc_psnr_envmax")) for member in members]
        gpu_hours = [_finite_or_none(member.get("gpu_hours")) for member in members]
        finite_deltas = [value for value in deltas if value is not None]
        finite_gpu = [value for value in gpu_hours if value is not None]
        verdict_counts: dict[str, int] = {}
        for member in members:
            verdict = str(member.get("verdict", ""))
            verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
        row: dict[str, object] = {group_key: key[index] for index, group_key in enumerate(group_keys)}
        row.update(
            {
                "trial_count": len(members),
                "mean_delta_ladder_auc_psnr_envmax": _mean_or_none(finite_deltas),
                "median_delta_ladder_auc_psnr_envmax": _median_or_none(finite_deltas),
                "min_delta_ladder_auc_psnr_envmax": min(finite_deltas) if finite_deltas else None,
                "max_delta_ladder_auc_psnr_envmax": max(finite_deltas) if finite_deltas else None,
                "positive_delta_count": sum(1 for value in finite_deltas if value > 0),
                "negative_delta_count": sum(1 for value in finite_deltas if value < 0),
                "zero_delta_count": sum(1 for value in finite_deltas if value == 0),
                "total_gpu_hours": sum(finite_gpu),
                "mean_gpu_hours": _mean_or_none(finite_gpu),
                "verdict_counts_json": json.dumps(verdict_counts, sort_keys=True, separators=(",", ":")),
            }
        )
        exported.append(row)
    exported.sort(
        key=lambda row: (
            str(row.get("environment", "")),
            str(row.get("primitive_family", "")),
        )
    )
    return exported


def _best_recipe_rows(cell_rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    by_environment: dict[str, list[Mapping[str, object]]] = {}
    for row in cell_rows:
        by_environment.setdefault(str(row["environment"]), []).append(row)
    exported: list[dict[str, object]] = []
    for environment, rows in sorted(by_environment.items()):
        best = max(rows, key=lambda row: float(row["mean_verified_gain"]))
        exported.append(
            {
                "scope": "environment",
                "environment": environment,
                "layer": best["layer"],
                "primitive_family": best["primitive_family"],
                "parameter_bucket": best["parameter_bucket"],
                "visits": best["visits"],
                "mean_verified_gain": best["mean_verified_gain"],
                "effect_sign": best["effect_sign"],
            }
        )
    if cell_rows:
        global_best = max(cell_rows, key=lambda row: float(row["mean_verified_gain"]))
        exported.insert(
            0,
            {
                "scope": "global",
                "environment": global_best["environment"],
                "layer": global_best["layer"],
                "primitive_family": global_best["primitive_family"],
                "parameter_bucket": global_best["parameter_bucket"],
                "visits": global_best["visits"],
                "mean_verified_gain": global_best["mean_verified_gain"],
                "effect_sign": global_best["effect_sign"],
            },
        )
    return exported


def _verdict_count_rows(rows: Sequence[Mapping[str, object]], *, scope: str) -> list[dict[str, object]]:
    counts: dict[str, int] = {}
    for row in rows:
        verdict = str(row.get("verdict", ""))
        counts[verdict] = counts.get(verdict, 0) + 1
    return [{"scope": scope, "verdict": verdict, "count": count} for verdict, count in sorted(counts.items())]


def _cost_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    gpu_hours = [_finite_or_none(row.get("gpu_hours")) for row in rows]
    finite_gpu = [value for value in gpu_hours if value is not None]
    deltas = [_finite_or_none(row.get("delta_ladder_auc_psnr_envmax")) for row in rows]
    finite_deltas = [value for value in deltas if value is not None]
    positive = [value for value in finite_deltas if value > 0]
    return [
        {"metric": "evidence_trial_count", "value": len(rows)},
        {"metric": "total_gpu_hours", "value": sum(finite_gpu)},
        {"metric": "mean_gpu_hours_per_trial", "value": _mean_or_none(finite_gpu)},
        {"metric": "mean_delta_ladder_auc_psnr_envmax", "value": _mean_or_none(finite_deltas)},
        {"metric": "positive_delta_trial_count", "value": len(positive)},
        {"metric": "negative_delta_trial_count", "value": sum(1 for value in finite_deltas if value < 0)},
        {"metric": "positive_delta_rate", "value": len(positive) / len(finite_deltas) if finite_deltas else None},
    ]


def _blockers(
    *,
    archive_summary: Mapping[str, int],
    trial_rows: Sequence[Mapping[str, object]],
    evidence_trial_rows: Sequence[Mapping[str, object]],
    cell_rows: Sequence[Mapping[str, object]],
    source_manifests: Mapping[str, Mapping[str, object]],
    settled_trial_target: int,
) -> list[dict[str, object]]:
    blockers: list[dict[str, object]] = []
    if int(archive_summary.get("settled_trials", 0)) < settled_trial_target:
        blockers.append(
            {
                "requirement": "settled_trial_volume",
                "expected": settled_trial_target,
                "observed": archive_summary.get("settled_trials", 0),
            }
        )
    if len(evidence_trial_rows) < settled_trial_target:
        blockers.append(
            {
                "requirement": "evidence_trial_volume",
                "expected": settled_trial_target,
                "observed": len(evidence_trial_rows),
                "reason": "Evidence tables require known environment, known primitive, and finite delta.",
            }
        )
    if not trial_rows:
        blockers.append({"requirement": "decodable_trial_rows", "reason": "NO_TRIAL_ROWS"})
    if not evidence_trial_rows:
        blockers.append({"requirement": "evidence_trial_rows", "reason": "NO_EVIDENCE_ELIGIBLE_TRIAL_ROWS"})
    if not cell_rows:
        blockers.append({"requirement": "cell_projection_rows", "reason": "NO_CELL_ROWS"})
    for source_name, expected_type in (
        ("m4_completion", "wmloop-m4-completion-gate-manifest"),
        ("verifier_meta_validation", "wmloop-t4-5-verifier-meta-validation-manifest"),
        ("m5_consolidation", "wmloop-m5-consolidation-manifest"),
    ):
        source = source_manifests.get(source_name)
        if source is None:
            continue
        if source.get("artifact_type") != expected_type or source.get("state") != "ready":
            blockers.append(
                {
                    "requirement": source_name,
                    "expected": {"artifact_type": expected_type, "state": "ready"},
                    "observed": {
                        "artifact_type": source.get("artifact_type"),
                        "state": source.get("state"),
                    },
                }
            )
    return blockers


def _read_json_ref(cas: ContentAddressedStore, uri: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(cas.read_bytes(uri).decode("utf-8"))
    except Exception as exc:
        raise M4EvidenceExportError(f"M4_EVIDENCE_CAS_JSON_INVALID:{uri}") from exc
    if not isinstance(payload, Mapping):
        raise M4EvidenceExportError(f"M4_EVIDENCE_CAS_JSON_INVALID:{uri}")
    return payload


def _loads_mapping(payload: object, code: str) -> Mapping[str, Any]:
    try:
        value = json.loads(str(payload))
    except json.JSONDecodeError as exc:
        raise M4EvidenceExportError(code) from exc
    if not isinstance(value, Mapping):
        raise M4EvidenceExportError(code)
    return value


def _load_json_mapping(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise M4EvidenceExportError(f"M4_EVIDENCE_MANIFEST_INVALID:{path}") from exc
    if not isinstance(payload, Mapping):
        raise M4EvidenceExportError(f"M4_EVIDENCE_MANIFEST_INVALID:{path}")
    return payload


def _parse_proposal(proposal_id: str) -> dict[str, str]:
    match = re.match(r"^m3-(?P<environment>.+)-training-eval-smoke-(?P<primitive>.+?)-unlabeled", proposal_id)
    if not match:
        return {}
    return {"environment": match.group("environment"), "primitive_family": match.group("primitive")}


def _receipt_primitive(receipt: Mapping[str, Any]) -> str | None:
    candidate = receipt.get("candidate")
    if isinstance(candidate, Mapping):
        changed_paths = candidate.get("changed_paths")
        if isinstance(changed_paths, list):
            for path in changed_paths:
                if isinstance(path, str) and "wmloop_interventions/" in path and path.endswith(".json"):
                    return Path(path).stem
    return None


def _nested_finite(mapping: Mapping[str, Any], keys: Sequence[str]) -> float | None:
    current: object = mapping
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return _finite_or_none(current)


def _finite_or_none(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _mean_or_none(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _median_or_none(values: Sequence[float]) -> float | None:
    return statistics.median(values) if values else None


def _effect_sign(value: float) -> str:
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "zero"


def _csv_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    if not rows:
        return b"\n"
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(str(key))
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: "" if row.get(key) is None else row.get(key, "") for key in columns})
    return output.getvalue().encode("utf-8")


def _latex_table(rows: Sequence[Mapping[str, object]], caption: str) -> str:
    if not rows:
        return "% empty table\n"
    columns = [str(key) for key in rows[0].keys()]
    lines = [
        "\\begin{tabular}{" + "l" * len(columns) + "}",
        "\\hline",
        " & ".join(_latex_escape(column) for column in columns) + " \\\\",
        "\\hline",
    ]
    for row in rows:
        lines.append(" & ".join(_latex_escape(str(row.get(column, ""))) for column in columns) + " \\\\")
    lines.extend(["\\hline", "\\end{tabular}", f"% {caption}", ""])
    return "\n".join(lines)


def _latex_escape(value: str) -> str:
    return (
        value.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("_", "\\_")
        .replace("{", "\\{")
        .replace("}", "\\}")
    )


def _render_markdown(
    *,
    report: Mapping[str, object],
    env_primitive_rows: Sequence[Mapping[str, object]],
    best_recipe_rows: Sequence[Mapping[str, object]],
    cost_rows: Sequence[Mapping[str, object]],
) -> str:
    archive_summary = report["archive_summary"]
    lines = [
        "# M4 Evidence Export",
        "",
        f"State: `{report['state']}`",
        f"Evidence export complete: `{report['m4_evidence_export_complete']}`",
        (
            "Evidence rows: "
            f"`trial_count={report['trial_count']}`, "
            f"`evidence_trial_count={report['evidence_trial_count']}`, "
            f"`excluded_trial_count={report['excluded_trial_count']}`"
        ),
        (
            "Archive: "
            f"`baselines={archive_summary['baselines']}`, "
            f"`settled_trials={archive_summary['settled_trials']}`, "
            f"`cells={archive_summary['cells']}`, "
            f"`cell_observations={archive_summary['cell_observations']}`"
        ),
        "",
        "## Best Recipes",
        "",
        "| Scope | Environment | Primitive | Visits | Mean Gain | Sign |",
        "|:--|:--|:--|--:|--:|:--|",
    ]
    for row in best_recipe_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["scope"]),
                    str(row["environment"]),
                    str(row["primitive_family"]),
                    str(row["visits"]),
                    str(row["mean_verified_gain"]),
                    str(row["effect_sign"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Environment x Primitive",
            "",
            "| Environment | Primitive | Trials | Mean Delta | Positive | Negative | GPU Hours |",
            "|:--|:--|--:|--:|--:|--:|--:|",
        ]
    )
    for row in env_primitive_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("environment", "")),
                    str(row.get("primitive_family", "")),
                    str(row["trial_count"]),
                    str(row["mean_delta_ladder_auc_psnr_envmax"]),
                    str(row["positive_delta_count"]),
                    str(row["negative_delta_count"]),
                    str(row["total_gpu_hours"]),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Cost Summary", "", "| Metric | Value |", "|:--|--:|"])
    for row in cost_rows:
        lines.append(f"| {row['metric']} | {row['value']} |")
    if report["blockers"]:
        lines.extend(["", "## Blockers", ""])
        for blocker in report["blockers"]:
            lines.append(f"- `{json.dumps(blocker, sort_keys=True, ensure_ascii=False)}`")
    lines.extend(["", "## Limitations", ""])
    for item in report["limitations"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise M4EvidenceExportError("M4_EVIDENCE_OUTPUT_EXISTS")
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
    run = commands.add_parser("run", help="export numeric M4/M5 evidence tables")
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--archive-db", type=Path, required=True)
    run.add_argument("--cas-root", type=Path)
    run.add_argument("--m4-completion-manifest", type=Path)
    run.add_argument("--verifier-meta-validation-manifest", type=Path)
    run.add_argument("--consolidation-manifest", type=Path)
    run.add_argument("--settled-trial-target", type=int, default=150)
    args = parser.parse_args(argv)
    if args.command == "run":
        manifest = run_m4_evidence_export(
            output_root=args.output_root,
            archive_db=args.archive_db,
            cas_root=args.cas_root,
            m4_completion_manifest=args.m4_completion_manifest,
            verifier_meta_validation_manifest=args.verifier_meta_validation_manifest,
            consolidation_manifest=args.consolidation_manifest,
            settled_trial_target=args.settled_trial_target,
        )
        print(json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
        return 0
    raise M4EvidenceExportError("M4_EVIDENCE_COMMAND_INVALID")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
