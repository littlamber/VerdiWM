"""Measure retrospective surrogate proposal-sorting benefit from archive cells."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from wmloop.archive.store import ArchiveStore, CellProjectionRecord, ContentAddressedStore
from wmloop.archive.surrogate import RankingCandidate, SURROGATE_MIN_SETTLED_TRIALS, _predict_candidate


class SurrogateBenefitError(RuntimeError):
    """The surrogate benefit export failed closed."""


def run_surrogate_benefit(
    *,
    output_root: Path,
    archive_db: Path,
    cas_root: Path | None = None,
    surrogate_readiness_manifest: Path | None = None,
    min_settled_trials: int = SURROGATE_MIN_SETTLED_TRIALS,
    positive_gain_threshold: float = 0.0,
) -> dict[str, object]:
    """Write a read-only leave-one-cell-out surrogate benefit report."""

    if min_settled_trials < 1:
        raise SurrogateBenefitError("SURROGATE_BENEFIT_MIN_SETTLED_INVALID")
    if not math.isfinite(positive_gain_threshold):
        raise SurrogateBenefitError("SURROGATE_BENEFIT_POSITIVE_THRESHOLD_INVALID")
    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise SurrogateBenefitError("SURROGATE_BENEFIT_OUTPUT_EXISTS")
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"

    archive = ArchiveStore(archive_db)
    cas = ContentAddressedStore(cas_root if cas_root is not None else Path(archive_db).resolve().parent)
    stats = archive.archive_statistics()
    cells = archive.list_cells()
    cell_observation_count = sum(record.stats.visits for record in cells)
    source_manifests = _load_source_manifests({"surrogate_readiness": surrogate_readiness_manifest})
    blockers = _blockers(
        settled_trial_count=stats["settled_trials"],
        cell_observation_count=cell_observation_count,
        cell_count=len(cells),
        positive_cell_count=sum(1 for record in cells if record.stats.mean_verified_improvement > positive_gain_threshold),
        min_settled_trials=min_settled_trials,
        source_manifests=source_manifests,
    )
    state = "ready" if not blockers else "blocked"
    ranking_rows = _leave_one_out_rows(cells, positive_gain_threshold=positive_gain_threshold) if state == "ready" else []
    benefit_summary = _benefit_summary(ranking_rows)
    report = {
        "schema_version": 1,
        "artifact_type": "wmloop-surrogate-benefit-report",
        "state": state,
        "surrogate_benefit_measured": state == "ready",
        "archive_db": str(Path(archive_db).resolve()),
        "source_manifests": source_manifests,
        "min_settled_trials": min_settled_trials,
        "settled_trial_count": stats["settled_trials"],
        "cell_count": len(cells),
        "cell_observation_count": cell_observation_count,
        "positive_gain_threshold": positive_gain_threshold,
        "method": "leave_one_cell_out_weighted_cell_knn",
        "prediction_usage": "proposal_sorting_only",
        "prediction_values_enter_reward_or_verdict": False,
        "blockers": blockers,
        "benefit_summary": benefit_summary,
        "ranking_rows": ranking_rows,
        "limitations": [
            "This is a retrospective cell-level leave-one-out measurement, not an online causal trial.",
            "Surrogate predictions are proposal sorting evidence only and never verifier inputs.",
            "Savings factors below 1.0 indicate worse-than-random ordering for that target.",
        ],
    }
    files = {
        "surrogate_benefit_json": ("surrogate-benefit.json", _canonical_json_bytes(report), "application/json"),
        "surrogate_benefit_markdown": (
            "surrogate-benefit.md",
            _render_markdown(report).encode("utf-8"),
            "text/markdown",
        ),
        "benefit_summary_csv": (
            "tables/benefit-summary.csv",
            _csv_bytes(_summary_rows(benefit_summary)),
            "text/csv",
        ),
        "loo_ranking_csv": ("tables/loo-ranking.csv", _csv_bytes(ranking_rows), "text/csv"),
        "benefit_summary_latex": (
            "latex/benefit-summary.tex",
            _latex_table(_summary_rows(benefit_summary), "Surrogate sorting benefit").encode("utf-8"),
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
            "artifact_type": "wmloop-surrogate-benefit-manifest",
            "state": state,
            "surrogate_benefit_measured": state == "ready",
            "method": report["method"],
            "prediction_usage": report["prediction_usage"],
            "prediction_values_enter_reward_or_verdict": False,
            "settled_trial_count": stats["settled_trials"],
            "cell_count": len(cells),
            "cell_observation_count": cell_observation_count,
            "positive_gain_threshold": positive_gain_threshold,
            "benefit_summary": benefit_summary,
            "blocker_count": len(blockers),
            "report_path": str(destination / "surrogate-benefit.json"),
            "markdown_path": str(destination / "surrogate-benefit.md"),
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
            "surrogate_training_allowed": payload.get("surrogate_training_allowed"),
        }
    return loaded


def _blockers(
    *,
    settled_trial_count: int,
    cell_observation_count: int,
    cell_count: int,
    positive_cell_count: int,
    min_settled_trials: int,
    source_manifests: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    blockers: list[dict[str, object]] = []
    if settled_trial_count < min_settled_trials:
        blockers.append(
            {
                "code": "SURROGATE_BENEFIT_MIN_SETTLED_TRIALS_NOT_MET",
                "expected": min_settled_trials,
                "observed": settled_trial_count,
            }
        )
    if cell_observation_count < min_settled_trials:
        blockers.append(
            {
                "code": "SURROGATE_BENEFIT_MIN_CELL_OBSERVATIONS_NOT_MET",
                "expected": min_settled_trials,
                "observed": cell_observation_count,
            }
        )
    if cell_count < 2:
        blockers.append({"code": "SURROGATE_BENEFIT_LOO_CELL_COUNT_TOO_SMALL", "observed": cell_count})
    if positive_cell_count < 1:
        blockers.append({"code": "SURROGATE_BENEFIT_NO_POSITIVE_CELL", "observed": positive_cell_count})
    readiness = source_manifests.get("surrogate_readiness")
    if readiness is not None and (
        readiness.get("artifact_type") != "wmloop-surrogate-readiness-manifest"
        or readiness.get("state") != "ready"
        or readiness.get("surrogate_training_allowed") is not True
    ):
        blockers.append(
            {
                "code": "SURROGATE_BENEFIT_READINESS_SOURCE_NOT_READY",
                "observed": {
                    "artifact_type": readiness.get("artifact_type"),
                    "state": readiness.get("state"),
                    "surrogate_training_allowed": readiness.get("surrogate_training_allowed"),
                },
            }
        )
    return blockers


def _leave_one_out_rows(
    cells: Sequence[CellProjectionRecord],
    *,
    positive_gain_threshold: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, record in enumerate(cells):
        training_cells = [candidate for candidate_index, candidate in enumerate(cells) if candidate_index != index]
        prediction = _predict_candidate(
            RankingCandidate(_loo_proposal_id(record), record.cell),
            training_cells,
        )
        rows.append(
            {
                "environment": record.cell.environment,
                "layer": record.cell.layer,
                "primitive_family": record.cell.primitive_family,
                "parameter_bucket": record.cell.parameter_bucket,
                "visits": record.stats.visits,
                "true_mean_verified_gain": record.stats.mean_verified_improvement,
                "predicted_verified_gain_loo": prediction["predicted_verified_gain"],
                "nearest_cosine_distance_loo": prediction["nearest_cosine_distance"],
                "nearest_environment": prediction["nearest_cell"]["environment"],
                "nearest_primitive_family": prediction["nearest_cell"]["primitive_family"],
                "true_positive": record.stats.mean_verified_improvement > positive_gain_threshold,
            }
        )
    by_predicted = sorted(rows, key=lambda row: (-float(row["predicted_verified_gain_loo"]), _cell_sort_key(row)))
    for rank, row in enumerate(by_predicted, start=1):
        row["surrogate_rank"] = rank
    by_measured = sorted(rows, key=lambda row: (-float(row["true_mean_verified_gain"]), _cell_sort_key(row)))
    for rank, row in enumerate(by_measured, start=1):
        row["measured_rank"] = rank
    return by_predicted


def _benefit_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not rows:
        return {}
    true_gains = [float(row["true_mean_verified_gain"]) for row in rows]
    predicted = [float(row["predicted_verified_gain_loo"]) for row in rows]
    errors = [prediction - true_gain for prediction, true_gain in zip(predicted, true_gains)]
    positive_rows = [row for row in rows if row.get("true_positive") is True]
    best_row = min(rows, key=lambda row: int(row["measured_rank"]))
    n = len(rows)
    positive_count = len(positive_rows)
    surrogate_rank_first_positive = min((int(row["surrogate_rank"]) for row in positive_rows), default=None)
    random_expected_rank_first_positive = (n + 1) / (positive_count + 1) if positive_count else None
    random_expected_rank_best_cell = (n + 1) / 2.0
    surrogate_rank_best_cell = int(best_row["surrogate_rank"])
    concordance = _pairwise_concordance(rows)
    summary = {
        "cell_count": n,
        "positive_cell_count": positive_count,
        "random_expected_rank_first_positive": random_expected_rank_first_positive,
        "surrogate_rank_first_positive": surrogate_rank_first_positive,
        "sorting_savings_factor_first_positive": _ratio(
            random_expected_rank_first_positive,
            surrogate_rank_first_positive,
        ),
        "random_expected_rank_best_cell": random_expected_rank_best_cell,
        "surrogate_rank_best_cell": surrogate_rank_best_cell,
        "sorting_savings_factor_best_cell": _ratio(random_expected_rank_best_cell, surrogate_rank_best_cell),
        "top1_true_mean_verified_gain": float(rows[0]["true_mean_verified_gain"]),
        "top3_positive_count": sum(1 for row in rows[:3] if row.get("true_positive") is True),
        "top5_positive_count": sum(1 for row in rows[:5] if row.get("true_positive") is True),
        "mean_absolute_error": sum(abs(error) for error in errors) / len(errors),
        "root_mean_squared_error": math.sqrt(sum(error * error for error in errors) / len(errors)),
        **concordance,
    }
    return summary


def _pairwise_concordance(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    concordant = 0
    discordant = 0
    ties = 0
    for left_index, left in enumerate(rows):
        for right in rows[left_index + 1 :]:
            true_diff = float(left["true_mean_verified_gain"]) - float(right["true_mean_verified_gain"])
            predicted_diff = float(left["predicted_verified_gain_loo"]) - float(right["predicted_verified_gain_loo"])
            if true_diff == 0.0 or predicted_diff == 0.0:
                ties += 1
            elif true_diff * predicted_diff > 0:
                concordant += 1
            else:
                discordant += 1
    comparable = concordant + discordant
    return {
        "pairwise_concordant_count": concordant,
        "pairwise_discordant_count": discordant,
        "pairwise_tie_count": ties,
        "pairwise_concordance_rate": concordant / comparable if comparable else None,
    }


def _summary_rows(summary: Mapping[str, object]) -> list[dict[str, object]]:
    return [{"metric": key, "value": value} for key, value in summary.items()]


def _ratio(numerator: float | None, denominator: int | float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return float(numerator) / float(denominator)


def _loo_proposal_id(record: CellProjectionRecord) -> str:
    cell = record.cell
    return "loo:" + ":".join([cell.environment, cell.layer, cell.primitive_family])


def _cell_sort_key(row: Mapping[str, object]) -> tuple[str, str, str, str]:
    return (
        str(row["environment"]),
        str(row["layer"]),
        str(row["primitive_family"]),
        str(row["parameter_bucket"]),
    )


def _load_json_mapping(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SurrogateBenefitError(f"SURROGATE_BENEFIT_MANIFEST_INVALID:{path}") from exc
    if not isinstance(payload, Mapping):
        raise SurrogateBenefitError(f"SURROGATE_BENEFIT_MANIFEST_INVALID:{path}")
    return payload


def _render_markdown(report: Mapping[str, object]) -> str:
    lines = [
        "# Surrogate Benefit",
        "",
        f"State: `{report['state']}`",
        f"Benefit measured: `{report['surrogate_benefit_measured']}`",
        f"Method: `{report['method']}`",
        f"Prediction usage: `{report['prediction_usage']}`",
        f"Prediction enters reward/verdict: `{report['prediction_values_enter_reward_or_verdict']}`",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|:--|--:|",
    ]
    for row in _summary_rows(report["benefit_summary"]):
        lines.append(f"| {row['metric']} | {row['value']} |")
    rows = report.get("ranking_rows")
    if isinstance(rows, list) and rows:
        lines.extend(
            [
                "",
                "## LOO Ranking",
                "",
                "| Rank | Environment | Primitive | True Gain | Predicted Gain | Positive |",
                "|--:|:--|:--|--:|--:|:--|",
            ]
        )
        for row in rows:
            if isinstance(row, Mapping):
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            str(row["surrogate_rank"]),
                            str(row["environment"]),
                            str(row["primitive_family"]),
                            str(row["true_mean_verified_gain"]),
                            str(row["predicted_verified_gain_loo"]),
                            str(row["true_positive"]),
                        ]
                    )
                    + " |"
                )
    if report["blockers"]:
        lines.extend(["", "## Blockers", ""])
        for blocker in report["blockers"]:
            lines.append(f"- `{json.dumps(blocker, sort_keys=True, ensure_ascii=False)}`")
    lines.extend(["", "## Limitations", ""])
    for item in report["limitations"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


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


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise SurrogateBenefitError("SURROGATE_BENEFIT_OUTPUT_EXISTS")
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
    run = commands.add_parser("run", help="measure retrospective surrogate sorting benefit")
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--archive-db", type=Path, required=True)
    run.add_argument("--cas-root", type=Path)
    run.add_argument("--surrogate-readiness-manifest", type=Path)
    run.add_argument("--min-settled-trials", type=int, default=SURROGATE_MIN_SETTLED_TRIALS)
    run.add_argument("--positive-gain-threshold", type=float, default=0.0)
    args = parser.parse_args(argv)
    if args.command == "run":
        manifest = run_surrogate_benefit(
            output_root=args.output_root,
            archive_db=args.archive_db,
            cas_root=args.cas_root,
            surrogate_readiness_manifest=args.surrogate_readiness_manifest,
            min_settled_trials=args.min_settled_trials,
            positive_gain_threshold=args.positive_gain_threshold,
        )
        print(json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
        return 0
    raise SurrogateBenefitError("SURROGATE_BENEFIT_COMMAND_INVALID")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
