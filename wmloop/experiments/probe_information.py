"""Export the probe-information and collision study from settled evidence."""

from __future__ import annotations

import csv
import io
import json
import math
from collections.abc import Mapping
from pathlib import Path
from statistics import fmean
from typing import Any

from wmloop.experiments._artifacts import canonical_json, write_bundle


class ProbeInformationError(ValueError):
    """S4 inputs are missing, malformed, or semantically incomplete."""


METRICS = (
    "top1_positive_hit",
    "benefit_sign_accuracy",
    "ranking_kendall_tau",
    "selection_regret",
    "negative_selection",
    "probe_gpu_hours",
    "gram_condition_number",
)


def export_probe_information_study(
    *,
    config_path: Path,
    output_root: Path,
    archive_db: Path | None = None,
    cas_root: Path | None = None,
) -> dict[str, object]:
    config_file = Path(config_path).resolve(strict=True)
    config = _load_mapping(config_file)
    if config.get("study_id") != "S4_probe_information_and_collision":
        raise ProbeInformationError("S4_CONFIG_STUDY_ID_INVALID")
    conditions = config.get("conditions")
    if not isinstance(conditions, Mapping) or not conditions:
        raise ProbeInformationError("S4_CONFIG_CONDITIONS_INVALID")

    condition_rows: list[dict[str, object]] = []
    source_refs: list[dict[str, object]] = [{"role": "config", **_source_ref(config_file)}]
    for condition_name, condition in conditions.items():
        if not isinstance(condition_name, str) or not isinstance(condition, Mapping):
            raise ProbeInformationError("S4_CONFIG_CONDITION_INVALID")
        status = str(condition.get("status", "observed"))
        report_value = condition.get("selector_replay_report")
        if status == "pending":
            condition_rows.append(
                {
                    "condition": condition_name,
                    "status": "pending",
                    "observed": False,
                    "claim_eligible": False,
                    "reason": str(condition.get("reason", "pending evidence")),
                    "source_report": None,
                }
            )
            continue
        if not isinstance(report_value, str) or not report_value:
            raise ProbeInformationError(f"S4_CONDITION_REPORT_MISSING:{condition_name}")
        report_path = _resolve_path(report_value, config_file.parents[2])
        metrics_path = report_path / "tables" / "selector-metrics.csv"
        manifest_path = report_path / "manifest.json"
        metrics = _load_metrics(metrics_path)
        manifest = _load_mapping(manifest_path)
        source_refs.extend(
            [
                {"role": f"{condition_name}:manifest", **_source_ref(manifest_path)},
                {"role": f"{condition_name}:selector_metrics", **_source_ref(metrics_path)},
            ]
        )
        condition_rows.append(
            {
                "condition": condition_name,
                "status": "observed",
                "observed": True,
                "claim_eligible": False,
                "interpretation": str(condition.get("interpretation", "")),
                "source_report": str(report_path),
                "source_state": manifest.get("state"),
                "evaluated_cell_count": manifest.get("evaluated_cell_count"),
                "abstained_cell_count": manifest.get("abstained_cell_count"),
                "metrics": _summarize_metrics(metrics),
            }
        )

    collision_rows: list[dict[str, object]] = []
    collision_sources = config.get("collision_redundancy_reports", [])
    if not isinstance(collision_sources, list):
        raise ProbeInformationError("S4_COLLISION_SOURCES_INVALID")
    for value in collision_sources:
        if not isinstance(value, str):
            raise ProbeInformationError("S4_COLLISION_SOURCE_INVALID")
        report_path = _resolve_path(value, config_file.parents[2])
        report_file = report_path / "probe-smoke-redundancy.json"
        report = _load_mapping(report_file)
        source_refs.append({"role": "collision_redundancy", **_source_ref(report_file)})
        comparisons = report.get("comparisons")
        if not isinstance(comparisons, list):
            raise ProbeInformationError("S4_COLLISION_COMPARISONS_INVALID")
        for comparison in comparisons:
            if not isinstance(comparison, Mapping):
                raise ProbeInformationError("S4_COLLISION_COMPARISON_INVALID")
            collision_rows.append(
                {
                    "reference_probe_id": report.get("reference_probe_id"),
                    "candidate_probe_id": report.get("candidate_probe_id"),
                    "environment": comparison.get("environment"),
                    "cosine_similarity": comparison.get("cosine_similarity"),
                    "relative_l2": comparison.get("relative_l2"),
                    "candidate_locality_residual": comparison.get("candidate_locality_residual"),
                    "candidate_locality_pass": comparison.get("candidate_locality_pass"),
                    "redundant": comparison.get("redundant"),
                    "collision_label": "non_redundant_successor" if not comparison.get("redundant") else "redundant",
                    "source_report": str(report_file),
                }
            )

    collision_label_report: Mapping[str, Any] | None = None
    collision_label_rows: list[dict[str, object]] = []
    collision_label_value = config.get("collision_label_report")
    if collision_label_value is not None:
        if not isinstance(collision_label_value, str) or not collision_label_value:
            raise ProbeInformationError("S4_COLLISION_LABEL_REPORT_INVALID")
        collision_label_root = _resolve_path(collision_label_value, config_file.parents[2])
        collision_label_file = collision_label_root / "collision-label-evaluation.json"
        collision_label_report = _load_mapping(collision_label_file)
        if (
            collision_label_report.get("artifact_type") != "verdiwm-collision-label-evaluation"
            or collision_label_report.get("state") != "ready"
        ):
            raise ProbeInformationError("S4_COLLISION_LABEL_REPORT_INVALID")
        raw_cases = collision_label_report.get("cases")
        if not isinstance(raw_cases, list) or any(not isinstance(row, Mapping) for row in raw_cases):
            raise ProbeInformationError("S4_COLLISION_LABEL_CASES_INVALID")
        collision_label_rows = [dict(row) for row in raw_cases]
        source_refs.append({"role": "collision_labels", **_source_ref(collision_label_file)})

    observed_conditions = sum(bool(row["observed"]) for row in condition_rows)
    pending_conditions = len(condition_rows) - observed_conditions
    ground_truth_available = bool(
        collision_label_report
        and int(collision_label_report.get("positive_collision_count", 0)) > 0
        and int(collision_label_report.get("negative_collision_count", 0)) > 0
    )
    collision_f1 = collision_label_report.get("collision_detection_f1") if collision_label_report else None
    post_evolution_collision_rate = (
        collision_label_report.get("post_evolution_collision_rate") if collision_label_report else None
    )
    collision_metrics_complete = collision_f1 is not None and post_evolution_collision_rate is not None
    report = {
        "schema_version": 1,
        "artifact_type": "verdiwm-probe-information-and-collision-study",
        "study_id": config["study_id"],
        "state": (
            "ready"
            if pending_conditions == 0 and ground_truth_available and collision_metrics_complete
            else "partial"
        ),
        "claim_boundary": (
            "This report describes diagnostic probe information and redundancy smoke evidence. "
            "It does not establish primitive quality, selector superiority, or transfer. "
            "Any pending random-expansion evidence, missing collision labels, or zero accepted "
            "post-evolution coverage remains an explicit blocker."
        ),
        "condition_count": len(condition_rows),
        "observed_condition_count": observed_conditions,
        "pending_condition_count": pending_conditions,
        "conditions": condition_rows,
        "collision": {
            "comparison_count": len(collision_rows),
            "labeled_case_count": len(collision_label_rows),
            "ground_truth_available": ground_truth_available,
            "collision_detection_f1": collision_f1,
            "post_evolution_collision_rate": post_evolution_collision_rate,
            "accepted_case_count": (
                collision_label_report.get("accepted_case_count") if collision_label_report else None
            ),
            "accepted_coverage": (
                collision_label_report.get("accepted_coverage") if collision_label_report else None
            ),
            "pre_certificate_collision_rate": (
                collision_label_report.get("pre_certificate_collision_rate")
                if collision_label_report
                else None
            ),
            "rows": collision_rows,
            "labeled_cases": collision_label_rows,
        },
        "work_orders": [
            {
                "id": "S4-RANDOM-EXPANSION",
                "status": "open" if pending_conditions else "closed",
                "action": "Run preregistered deterministic random probe-subset expansion on the frozen ACWM-Phys split.",
            },
            {
                "id": "S4-COLLISION-GROUND-TRUTH",
                "status": "open" if not ground_truth_available else "closed",
                "action": "Add held-out effect-sign labels for paired collision/non-collision cases before reporting F1.",
            },
            {
                "id": "S4-COLLISION-ACCEPTED-COVERAGE",
                "status": "open" if post_evolution_collision_rate is None else "closed",
                "action": "Obtain non-zero certificate-accepted evolved coverage before reporting post-evolution collision rate.",
            },
        ],
        "source_refs": source_refs,
    }
    files = {
        "probe-information-and-collision.json": canonical_json(report),
        "probe-information-and-collision.md": _markdown(report).encode("utf-8"),
        "tables/conditions.csv": _conditions_csv(condition_rows).encode("utf-8"),
        "tables/collision-smoke.csv": _collision_csv(collision_rows).encode("utf-8"),
        "tables/collision-labeled-cases.csv": _generic_csv(collision_label_rows).encode("utf-8"),
        "tables/paper-summary.tex": _latex(report).encode("utf-8"),
        "source-refs.json": canonical_json({"sources": source_refs}),
    }
    destination = Path(output_root).resolve()
    return write_bundle(
        output_root=destination,
        files=files,
        manifest_fields={
            "artifact_type": "verdiwm-probe-information-and-collision-study-manifest",
            "study_id": report["study_id"],
            "state": report["state"],
            "condition_count": len(condition_rows),
            "observed_condition_count": observed_conditions,
            "collision_comparison_count": len(collision_rows),
            "collision_labeled_case_count": len(collision_label_rows),
            "collision_detection_f1": collision_f1,
            "post_evolution_collision_rate": post_evolution_collision_rate,
            "report_path": str(destination / "probe-information-and-collision.json"),
        },
        archive_db=archive_db,
        cas_root=cas_root,
    )


def _load_mapping(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProbeInformationError(f"S4_JSON_INVALID:{path}") from exc
    if not isinstance(payload, Mapping):
        raise ProbeInformationError(f"S4_MAPPING_INVALID:{path}")
    return payload


def _resolve_path(value: str, base: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    return path.resolve(strict=True)


def _source_ref(path: Path) -> dict[str, object]:
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha256(path)}


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_metrics(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise ProbeInformationError(f"S4_METRICS_INVALID:{path}") from exc
    if not rows or not all("selector" in row for row in rows):
        raise ProbeInformationError(f"S4_METRICS_EMPTY:{path}")
    return rows


def _summarize_metrics(rows: list[dict[str, str]]) -> dict[str, object]:
    summary = _metric_values(rows)
    summary["by_selector"] = {
        selector: _metric_values([row for row in rows if row["selector"] == selector])
        for selector in sorted({row["selector"] for row in rows})
    }
    return summary


def _metric_values(rows: list[dict[str, str]]) -> dict[str, object]:
    summary: dict[str, object] = {"row_count": len(rows)}
    for metric in METRICS:
        values = []
        for row in rows:
            value = row.get(metric)
            if value in (None, ""):
                continue
            number = float(value)
            if math.isfinite(number):
                values.append(number)
        summary[metric] = fmean(values) if values else None
        summary[f"{metric}_observed_rows"] = len(values)
    return summary


def _conditions_csv(rows: list[dict[str, object]]) -> str:
    output = io.StringIO()
    fields = ["condition", "status", "observed", "claim_eligible", "evaluated_cell_count", "abstained_cell_count"]
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _collision_csv(rows: list[dict[str, object]]) -> str:
    output = io.StringIO()
    fields = ["reference_probe_id", "candidate_probe_id", "environment", "cosine_similarity", "relative_l2", "candidate_locality_residual", "candidate_locality_pass", "redundant", "collision_label"]
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _generic_csv(rows: list[dict[str, object]]) -> str:
    if not rows:
        return ""
    output = io.StringIO()
    fields = sorted({key for row in rows for key in row})
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# S4 Probe Information and Collision Study",
        "",
        f"State: `{report['state']}`",
        "",
        "This is a diagnostic evidence report. It is not a repair-quality or transfer claim.",
        "",
        "| Condition | Status | Observed | Claim eligible |",
        "|---|---|:---:|:---:|",
    ]
    for row in report["conditions"]:
        lines.append(f"| {row['condition']} | {row['status']} | {row['observed']} | {row['claim_eligible']} |")
    lines.extend(
        [
            "",
            f"Collision smoke comparisons: `{report['collision']['comparison_count']}`",
            f"Independently labeled cases: `{report['collision']['labeled_case_count']}`",
            "",
            f"Collision F1: `{report['collision']['collision_detection_f1']}`",
            f"Accepted coverage: `{report['collision']['accepted_coverage']}`",
            f"Post-evolution collision rate: `{report['collision']['post_evolution_collision_rate']}`",
            "",
            "## Open Work",
        ]
    )
    for work_order in report["work_orders"]:
        lines.append(f"- `{work_order['id']}` ({work_order['status']}): {work_order['action']}")
    return "\n".join(lines) + "\n"


def _latex(report: Mapping[str, Any]) -> str:
    lines = [
        "% S4 remains partial until all conditions and certificate-accepted collision metrics are observed.",
        "\\begin{tabular}{lrr}",
        "Condition & Observed & Claim eligible \\\\",
        "\\hline",
    ]
    for row in report["conditions"]:
        condition = str(row["condition"]).replace("_", "\\_")
        lines.append(
            "{} & {} & {}".format(
                condition,
                int(bool(row["observed"])),
                int(bool(row["claim_eligible"])),
            )
            + r" \\"
        )
    lines.extend(["\\end{tabular}", ""])
    return "\n".join(lines)
