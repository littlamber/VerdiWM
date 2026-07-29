"""Audit ACWM progressive-fidelity efficiency from settled historical evidence."""

from __future__ import annotations

import csv
import io
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from wmloop.experiments._artifacts import canonical_json, write_bundle


class ProgressiveFidelityError(ValueError):
    """Historical screen, gate, or confirmation evidence is inconsistent."""


CandidateKey = tuple[str, str, int]


def export_progressive_fidelity_efficiency(
    *,
    reports_root: Path,
    effect_label_index_path: Path,
    output_root: Path,
    archive_db: Path | None = None,
    cas_root: Path | None = None,
) -> dict[str, object]:
    """Export S6 using settled 512 gates and independent 800/1000 finalizers."""

    root = Path(reports_root).resolve(strict=True)
    label_index = _load_json(Path(effect_label_index_path).resolve(strict=True))
    if label_index.get("artifact_type") != "verdiwm-settled-effect-label-index":
        raise ProgressiveFidelityError("PROGRESSIVE_FIDELITY_LABEL_INDEX_INVALID")
    candidates, excluded = _discover_candidates(root=root, label_index=label_index)
    confirmations, confirm_excluded = _discover_confirmations(root=root)
    excluded.extend(confirm_excluded)
    report = build_progressive_fidelity_report(
        candidates=candidates,
        confirmations=confirmations,
        excluded=excluded,
    )
    destination = Path(output_root).resolve()
    files = {
        "progressive-fidelity-efficiency.json": canonical_json(report),
        "progressive-fidelity-efficiency.md": _markdown(report).encode("utf-8"),
        "tables/candidate-ledger.csv": _csv(report["candidate_rows"]),
        "tables/cost-summary.csv": _csv(report["cost_rows"]),
        "tables/transition-summary.csv": _csv(report["transition_rows"]),
        "tables/paper-summary.tex": _latex(report).encode("utf-8"),
        "excluded-evidence.json": canonical_json(excluded),
        "input-effect-label-index.json": canonical_json(label_index),
    }
    return write_bundle(
        output_root=destination,
        files=files,
        manifest_fields={
            "artifact_type": "verdiwm-progressive-fidelity-efficiency-manifest",
            "state": report["state"],
            "study_id": "S6_progressive_fidelity_efficiency",
            "candidate_count": report["candidate_count"],
            "gate_pair_count": report["gate_pair_count"],
            "confirm_pair_count": report["confirm_pair_count"],
            "gpu_hour_reduction": report["metrics"]["gpu_hour_reduction"],
            "positive_recall": report["metrics"]["positive_recall"],
            "report_path": str(destination / "progressive-fidelity-efficiency.json"),
            "paper_table_path": str(destination / "tables" / "paper-summary.tex"),
        },
        archive_db=archive_db,
        cas_root=cas_root,
    )


def build_progressive_fidelity_report(
    *,
    candidates: Sequence[Mapping[str, Any]],
    confirmations: Sequence[Mapping[str, Any]],
    excluded: Sequence[Mapping[str, Any]] = (),
) -> dict[str, object]:
    """Compute measured transitions and a measured-cost confirm-all projection."""

    by_key = {_key(row): dict(row) for row in candidates}
    if len(by_key) != len(candidates):
        raise ProgressiveFidelityError("PROGRESSIVE_FIDELITY_CANDIDATE_DUPLICATE")
    confirm_by_key = {_key(row): dict(row) for row in confirmations}
    if len(confirm_by_key) != len(confirmations):
        raise ProgressiveFidelityError("PROGRESSIVE_FIDELITY_CONFIRM_DUPLICATE")
    orphan_confirms = sorted(set(confirm_by_key) - set(by_key))

    confirm_costs = [
        float(row["confirm_gpu_hours"])
        for row in confirmations
        if _positive_finite(row.get("confirm_gpu_hours"))
    ]
    confirm_gate_costs = [
        float(row["confirm_gate_gpu_hours"])
        for row in confirmations
        if _nonnegative_finite(row.get("confirm_gate_gpu_hours"))
    ]
    if not confirm_costs:
        raise ProgressiveFidelityError("PROGRESSIVE_FIDELITY_CONFIRM_COST_MISSING")
    global_confirm = statistics.median(confirm_costs)
    global_confirm_gate = statistics.median(confirm_gate_costs) if confirm_gate_costs else 0.0
    confirm_by_environment = _cost_medians(confirmations, "confirm_gpu_hours")
    confirm_gate_by_environment = _cost_medians(confirmations, "confirm_gate_gpu_hours")

    rows: list[dict[str, object]] = []
    progressive_cost = 0.0
    counterfactual_cost = 0.0
    projected_count = 0
    for key, candidate in sorted(by_key.items()):
        confirmation = confirm_by_key.get(key)
        screen_cost = _required_nonnegative(candidate, "screen_gpu_hours")
        gate_cost = _required_nonnegative(candidate, "gate_gpu_hours")
        progressive_cost += screen_cost + gate_cost
        if confirmation is not None:
            confirm_cost = _required_nonnegative(confirmation, "confirm_gpu_hours")
            confirm_gate_cost = _required_nonnegative(confirmation, "confirm_gate_gpu_hours")
            progressive_cost += confirm_cost + confirm_gate_cost
            projection_source = "measured_exact_candidate"
        else:
            environment = key[0]
            confirm_cost = confirm_by_environment.get(environment, global_confirm)
            confirm_gate_cost = confirm_gate_by_environment.get(environment, global_confirm_gate)
            projection_source = (
                "measured_environment_median"
                if environment in confirm_by_environment
                else "measured_global_median"
            )
            projected_count += 1
        projected_full_cost = confirm_cost + confirm_gate_cost
        counterfactual_cost += projected_full_cost
        rows.append(
            {
                "environment": key[0],
                "primitive": key[1],
                "seed": key[2],
                "screen_score": candidate.get("screen_score"),
                "screen_positive": bool(candidate["screen_positive"]),
                "gate_score": candidate.get("gate_score"),
                "gate_positive": bool(candidate["gate_positive"]),
                "confirmation_observed": confirmation is not None,
                "confirm_score": confirmation.get("confirm_score") if confirmation else None,
                "confirm_positive": confirmation.get("confirm_positive") if confirmation else None,
                "screen_gpu_hours": screen_cost,
                "gate_gpu_hours": gate_cost,
                "confirm_gpu_hours": confirmation.get("confirm_gpu_hours") if confirmation else None,
                "confirm_gate_gpu_hours": confirmation.get("confirm_gate_gpu_hours") if confirmation else None,
                "projected_full_confirm_gpu_hours": projected_full_cost,
                "projection_source": projection_source,
                "screen_manifest": candidate.get("screen_manifest"),
                "gate_manifest": candidate.get("gate_manifest"),
                "confirmation_manifest": confirmation.get("confirmation_manifest") if confirmation else None,
            }
        )

    gate_positives = [row for row in rows if row["gate_positive"] is True]
    screen_rejects = [row for row in rows if row["screen_positive"] is False]
    true_screen_positives = sum(row["screen_positive"] is True for row in gate_positives)
    false_rejections = sum(row["gate_positive"] is True for row in screen_rejects)
    positive_recall = _ratio(true_screen_positives, len(gate_positives))
    false_rejection_rate = _ratio(false_rejections, len(screen_rejects))
    paired_scores = [
        (float(row["screen_score"]), float(row["confirm_score"]))
        for row in rows
        if _finite(row.get("screen_score")) and _finite(row.get("confirm_score"))
    ]
    rank_correlation = _spearman(paired_scores)
    gpu_hour_reduction = (
        None
        if counterfactual_cost <= 0.0
        else (counterfactual_cost - progressive_cost) / counterfactual_cost
    )

    metrics = {
        "gpu_hour_reduction": _round(gpu_hour_reduction),
        "positive_recall": _round(positive_recall),
        "false_rejection_rate": _round(false_rejection_rate),
        "screen_confirm_rank_correlation": _round(rank_correlation),
        "progressive_fidelity_gpu_hours": _round(progressive_cost),
        "confirm_every_candidate_projected_gpu_hours": _round(counterfactual_cost),
    }
    transition_rows = [
        {
            "transition": "screen_to_frozen_512_gate",
            "paired_count": len(rows),
            "higher_fidelity_positive_count": len(gate_positives),
            "positive_recall": metrics["positive_recall"],
            "false_rejection_rate": metrics["false_rejection_rate"],
            "rank_correlation": _round(
                _spearman(
                    [
                        (float(row["screen_score"]), float(row["gate_score"]))
                        for row in rows
                        if _finite(row.get("screen_score")) and _finite(row.get("gate_score"))
                    ]
                )
            ),
        },
        {
            "transition": "screen_to_independent_800_1000_confirmation",
            "paired_count": len(paired_scores),
            "higher_fidelity_positive_count": sum(
                row.get("confirm_positive") is True for row in rows
            ),
            "positive_recall": None,
            "false_rejection_rate": None,
            "rank_correlation": metrics["screen_confirm_rank_correlation"],
        },
    ]
    cost_rows = [
        {
            "condition": "progressive_fidelity_observed",
            "gpu_hours": metrics["progressive_fidelity_gpu_hours"],
            "measured_candidate_count": len(rows),
            "projected_candidate_count": 0,
        },
        {
            "condition": "confirm_every_candidate_measured_cost_projection",
            "gpu_hours": metrics["confirm_every_candidate_projected_gpu_hours"],
            "measured_candidate_count": len(rows) - projected_count,
            "projected_candidate_count": projected_count,
        },
    ]
    limitations = [
        "Positive recall and false rejection are identified against settled frozen 512-step official gates, not screen labels.",
        "Screen-to-confirm rank correlation uses only candidates with independent 800/1000 confirmation evidence.",
        "Confirm-every-candidate GPU hours are a counterfactual projection from measured exact, environment-median, or global-median confirmation costs; no synthetic training run is represented as observed.",
        "Historical retries and selector-inadmissible labels are excluded; excluded evidence remains enumerated in the bundle.",
    ]
    state = "ready" if rows and paired_scores and counterfactual_cost > 0.0 else "partial"
    return {
        "schema_version": 1,
        "artifact_type": "verdiwm-progressive-fidelity-efficiency",
        "study_id": "S6_progressive_fidelity_efficiency",
        "state": state,
        "candidate_count": len(rows),
        "gate_pair_count": len(rows),
        "confirm_pair_count": len(paired_scores),
        "gate_positive_count": len(gate_positives),
        "confirmation_positive_count": sum(row.get("confirm_positive") is True for row in rows),
        "false_rejection_count": false_rejections,
        "counterfactual_projected_candidate_count": projected_count,
        "orphan_confirmation_keys": [list(key) for key in orphan_confirms],
        "excluded_evidence_count": len(excluded),
        "metrics": metrics,
        "transition_rows": transition_rows,
        "cost_rows": cost_rows,
        "candidate_rows": rows,
        "limitations": limitations,
        "claim_boundary": "Efficiency evidence only. Model-quality positives remain governed by settled frozen official-gate and checkpoint-ladder receipts.",
    }


def _discover_candidates(
    *, root: Path, label_index: Mapping[str, Any]
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    candidates: dict[CandidateKey, dict[str, object]] = {}
    excluded: list[dict[str, object]] = []
    labels = label_index.get("labels")
    if not isinstance(labels, list):
        raise ProgressiveFidelityError("PROGRESSIVE_FIDELITY_LABELS_INVALID")
    for label in labels:
        if not isinstance(label, Mapping):
            continue
        if label.get("label_source") != "retained_checkpoint_completion_gate":
            continue
        if label.get("settled") is not True or not isinstance(label.get("positive"), bool):
            excluded.append(
                {
                    "stage": "gate",
                    "reason": "selector_inadmissible_or_unsettled",
                    "label_id": label.get("label_id"),
                }
            )
            continue
        key = _key(label)
        gate_manifest = Path(str(label["evidence_ref"])).resolve(strict=True)
        gate = _load_json(gate_manifest)
        if _key(gate) != key:
            raise ProgressiveFidelityError("PROGRESSIVE_FIDELITY_GATE_IDENTITY_MISMATCH")
        row = _candidate_from_gate(gate_manifest=gate_manifest, gate=gate)
        if row["gate_positive"] is not label["positive"]:
            raise ProgressiveFidelityError("PROGRESSIVE_FIDELITY_GATE_LABEL_MISMATCH")
        existing = candidates.get(key)
        if existing is not None:
            excluded.append(
                {
                    "stage": "gate",
                    "reason": "superseded_retry",
                    "key": list(key),
                    "excluded_manifest": existing["gate_manifest"],
                    "selected_manifest": str(gate_manifest),
                }
            )
        candidates[key] = row

    # The label index intentionally names independent 800/1000 gates, while the
    # finalizer is the authoritative link back to each candidate's 512 gate.
    for finalizer_path in sorted(
        root.glob("acwm-autoloop-checkpoint-finalize-*/checkpoint-ladder-finalization.json")
    ):
        finalizer = _load_json(finalizer_path)
        records = finalizer.get("selection", {}).get("records", [])
        initial = [
            row
            for row in records
            if isinstance(row, Mapping) and int(row.get("checkpoint_step", 0)) == 512
        ]
        if len(initial) != 1:
            excluded.append(
                {
                    "stage": "gate",
                    "reason": "finalizer_512_gate_missing_or_ambiguous",
                    "path": str(finalizer_path),
                }
            )
            continue
        gate_manifest = Path(str(initial[0]["official_manifest_path"])).resolve(strict=True)
        gate = _load_json(gate_manifest)
        row = _candidate_from_gate(gate_manifest=gate_manifest, gate=gate)
        key = _key(row)
        if key in candidates:
            continue
        candidates[key] = row
    return list(candidates.values()), excluded


def _candidate_from_gate(
    *, gate_manifest: Path, gate: Mapping[str, Any]
) -> dict[str, object]:
    quality_gate = gate.get("official_quality_gate")
    if (
        gate.get("state") != "ready"
        or not isinstance(quality_gate, Mapping)
        or not isinstance(quality_gate.get("pass"), bool)
    ):
        raise ProgressiveFidelityError("PROGRESSIVE_FIDELITY_GATE_NOT_SETTLED")
    key = _key(gate)
    checkpoint = Path(str(gate.get("candidate_checkpoint") or "")).resolve(strict=True)
    screen_manifest = checkpoint.parents[1] / "manifest.json"
    screen = _load_json(screen_manifest)
    screen_report = _load_json(Path(str(screen["report_path"])).resolve(strict=True))
    if (
        screen.get("environment") != key[0]
        or screen.get("seed") != key[2]
        or screen_report.get("proposal_primitive") != key[1]
    ):
        raise ProgressiveFidelityError("PROGRESSIVE_FIDELITY_SCREEN_IDENTITY_MISMATCH")
    screen_cost = _required_nonnegative(screen_report, "actual_gpu_hours")
    metric = str(screen.get("primary_metric") or screen_report.get("primary_metric") or "")
    score = _metric(screen.get("delta_m_ver"), metric)
    action_gate = screen.get("action_following_gate")
    action_pass = (
        not isinstance(action_gate, Mapping)
        or action_gate.get("enabled") is not True
        or action_gate.get("pass") is True
    )
    screen_positive = (
        screen.get("state") == "ready"
        and action_pass
        and score is not None
        and score >= 0.0
    )
    return {
        "environment": key[0],
        "primitive": key[1],
        "seed": key[2],
        "screen_score": score,
        "screen_positive": screen_positive,
        "gate_score": _metric(quality_gate.get("delta_candidate_minus_baseline"), "psnr"),
        "gate_positive": bool(quality_gate["pass"]),
        "screen_gpu_hours": screen_cost,
        "gate_gpu_hours": _manifest_gpu_hours(gate),
        "screen_manifest": str(screen_manifest),
        "gate_manifest": str(gate_manifest),
    }


def _discover_confirmations(
    *, root: Path
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    confirmations: dict[CandidateKey, dict[str, object]] = {}
    excluded: list[dict[str, object]] = []
    for report_path in sorted(root.glob("acwm-autoloop-checkpoint-finalize-*/checkpoint-ladder-finalization.json")):
        report = _load_json(report_path)
        if report.get("state") not in {"ready", "checks_failed"}:
            excluded.append({"stage": "confirm", "reason": "finalizer_not_settled", "path": str(report_path)})
            continue
        key = _key(report)
        checkpoint_manifest = Path(str(report["checkpoint_manifest"])).resolve(strict=True)
        confirm_env_manifest = checkpoint_manifest.parents[1] / "manifest.json"
        confirm_env = _load_json(confirm_env_manifest)
        confirm_report = _load_json(Path(str(confirm_env["report_path"])).resolve(strict=True))
        confirmation_records = [
            row
            for row in report.get("selection", {}).get("records", [])
            if isinstance(row, Mapping) and int(row.get("checkpoint_step", 0)) >= 800
        ]
        gate_cost = 0.0
        scores: list[float] = []
        for row in confirmation_records:
            gate_path = Path(str(row["official_manifest_path"])).resolve(strict=True)
            gate = _load_json(gate_path)
            gate_cost += _manifest_gpu_hours(gate)
            score = _metric(gate.get("official_quality_gate", {}).get("delta_candidate_minus_baseline"), "psnr")
            if score is not None:
                scores.append(score)
        row = {
            "environment": key[0],
            "primitive": key[1],
            "seed": key[2],
            "confirm_score": max(scores) if scores else None,
            "confirm_positive": bool(report.get("confirmation_passed")),
            "confirm_gpu_hours": _required_nonnegative(confirm_report, "actual_gpu_hours"),
            "confirm_gate_gpu_hours": gate_cost,
            "confirmation_manifest": str(report_path),
        }
        if key in confirmations:
            excluded.append(
                {
                    "stage": "confirm",
                    "reason": "superseded_finalizer",
                    "key": list(key),
                    "excluded_manifest": confirmations[key]["confirmation_manifest"],
                    "selected_manifest": str(report_path),
                }
            )
        confirmations[key] = row
    return list(confirmations.values()), excluded


def _cost_medians(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        value = row.get(field)
        if not _nonnegative_finite(value):
            continue
        grouped.setdefault(str(row["environment"]), []).append(float(value))
    return {key: statistics.median(values) for key, values in grouped.items()}


def _manifest_gpu_hours(payload: Mapping[str, Any]) -> float:
    if _nonnegative_finite(payload.get("actual_gpu_hours")):
        return float(payload["actual_gpu_hours"])
    created = payload.get("created_at")
    completed = payload.get("completed_at")
    if isinstance(created, str) and isinstance(completed, str):
        seconds = (_timestamp(completed) - _timestamp(created)).total_seconds()
        if seconds >= 0.0:
            return seconds / 3600.0
    if _nonnegative_finite(payload.get("duration_seconds")):
        return float(payload["duration_seconds"]) / 3600.0
    raise ProgressiveFidelityError("PROGRESSIVE_FIDELITY_GATE_DURATION_MISSING")


def _timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _key(row: Mapping[str, Any]) -> CandidateKey:
    environment = row.get("environment")
    primitive = row.get("primitive")
    seed = row.get("seed")
    if not isinstance(environment, str) or not environment or not isinstance(primitive, str) or not primitive or not isinstance(seed, int):
        raise ProgressiveFidelityError("PROGRESSIVE_FIDELITY_IDENTITY_INVALID")
    return environment, primitive, seed


def _metric(raw: object, name: str) -> float | None:
    if not isinstance(raw, Mapping):
        return None
    candidates = [name, "ladder_auc_psnr_envmax", "auc_psnr_envmax"]
    candidates.extend(str(key) for key in raw)
    for candidate in candidates:
        value = raw.get(candidate)
        if _finite(value):
            return float(value)
    return None


def _required_nonnegative(row: Mapping[str, Any], field: str) -> float:
    value = row.get(field)
    if not _nonnegative_finite(value):
        raise ProgressiveFidelityError(f"PROGRESSIVE_FIDELITY_COST_INVALID:{field}")
    return float(value)


def _positive_finite(value: object) -> bool:
    return _finite(value) and float(value) > 0.0


def _nonnegative_finite(value: object) -> bool:
    return _finite(value) and float(value) >= 0.0


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 9)


def _spearman(pairs: Sequence[tuple[float, float]]) -> float | None:
    if len(pairs) < 2:
        return None
    x = _ranks([pair[0] for pair in pairs])
    y = _ranks([pair[1] for pair in pairs])
    x_mean = statistics.mean(x)
    y_mean = statistics.mean(y)
    numerator = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y, strict=True))
    denominator = math.sqrt(
        sum((a - x_mean) ** 2 for a in x) * sum((b - y_mean) ** 2 for b in y)
    )
    return None if denominator == 0.0 else numerator / denominator


def _ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        rank = (index + 1 + end) / 2.0
        for original, _ in ordered[index:end]:
            ranks[original] = rank
        index = end
    return ranks


def _csv(rows: object) -> bytes:
    records = list(rows)  # type: ignore[arg-type]
    if not records:
        return b"\n"
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(records[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(records)
    return output.getvalue().encode("utf-8")


def _latex(report: Mapping[str, Any]) -> str:
    metrics = report["metrics"]
    values = [
        ("GPU-hour reduction", metrics["gpu_hour_reduction"]),
        ("Positive recall", metrics["positive_recall"]),
        ("False rejection rate", metrics["false_rejection_rate"]),
        ("Screen--confirm rank correlation", metrics["screen_confirm_rank_correlation"]),
    ]
    lines = ["\\begin{tabular}{lr}", "\\toprule", "Metric & Value \\\\", "\\midrule"]
    for label, value in values:
        rendered = "--" if value is None else f"{float(value):.3f}"
        lines.append(f"{label} & {rendered} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines) + "\n"


def _markdown(report: Mapping[str, Any]) -> str:
    metrics = report["metrics"]
    return "\n".join(
        [
            "# Progressive-Fidelity Efficiency",
            "",
            f"State: `{report['state']}`",
            f"Settled 512 gate pairs: `{report['gate_pair_count']}`",
            f"Independent 800/1000 confirmation pairs: `{report['confirm_pair_count']}`",
            f"GPU-hour reduction: `{metrics['gpu_hour_reduction']}`",
            f"Positive recall: `{metrics['positive_recall']}`",
            f"False rejection rate: `{metrics['false_rejection_rate']}`",
            f"Screen-confirm rank correlation: `{metrics['screen_confirm_rank_correlation']}`",
            "",
            "## Claim Boundary",
            "",
            str(report["claim_boundary"]),
            "",
            "## Limitations",
            "",
            *[f"- {item}" for item in report["limitations"]],
            "",
        ]
    )


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProgressiveFidelityError(f"PROGRESSIVE_FIDELITY_JSON_INVALID:{path}") from exc
    if not isinstance(payload, Mapping):
        raise ProgressiveFidelityError(f"PROGRESSIVE_FIDELITY_JSON_INVALID:{path}")
    return payload
