"""Fail-closed aggregation for completed M0 baseline launch tasks."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from wmloop.evaluate.launch import BaselineLaunchPlan, BaselineLaunchTask, baseline_task_status, load_baseline_launch_plan


class BaselineResultsError(RuntimeError):
    """A baseline result artifact could not be summarized safely."""


_METRICS = ("mse", "masked_mse", "psnr", "ssim")
_ROW = re.compile(r"^\|(.+)\|\s*$")


@dataclass(frozen=True)
class BaselineTaskMetrics:
    task_id: str
    environment: str
    cohort: str
    split: str
    steps: int
    trajectory_count: int
    window_count: int | None
    metrics: Mapping[str, float]
    results_path: Path
    receipt_path: Path

    def to_document(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "environment": self.environment,
            "cohort": self.cohort,
            "split": self.split,
            "steps": self.steps,
            "trajectory_count": self.trajectory_count,
            "window_count": self.window_count,
            "metrics": dict(self.metrics),
            "results_path": str(self.results_path),
            "receipt_path": str(self.receipt_path),
        }


def summarize_baseline_results(plan: BaselineLaunchPlan) -> dict[str, object]:
    """Summarize completed M0 metrics without inventing missing evidence."""

    statuses = [baseline_task_status(task) for task in plan.tasks]
    state_counts: dict[str, int] = {}
    for status in statuses:
        state_counts[status.state] = state_counts.get(status.state, 0) + 1
    incomplete = [status.to_document() for status in statuses if status.state != "completed"]
    if incomplete:
        return _summary_document(plan, state="incomplete", state_counts=state_counts, task_metrics=(), blockers=incomplete)
    task_metrics: list[BaselineTaskMetrics] = []
    blockers: list[dict[str, object]] = []
    for task in plan.tasks:
        try:
            task_metrics.append(read_task_metrics(task))
        except BaselineResultsError as exc:
            blockers.append({"task_id": task.task_id, "state": "metrics_unavailable", "reason": str(exc)})
    if blockers:
        return _summary_document(plan, state="metrics_unavailable", state_counts=state_counts, task_metrics=tuple(task_metrics), blockers=blockers)
    return _summary_document(plan, state="ready", state_counts=state_counts, task_metrics=tuple(task_metrics), blockers=())


def read_task_metrics(task: BaselineLaunchTask) -> BaselineTaskMetrics:
    results_path = task.output_root / "results.md"
    payload = _read_regular_text(results_path, "BASELINE_RESULTS_MISSING")
    rows = _parse_results_table(payload)
    steps = _int_option(task.evaluation_command, "--steps")
    expected_env = task.selection.vendor_environment
    for row in rows:
        if row.get("env") == expected_env and row.get("split") == task.split and _parse_int(row.get("steps", "")) == steps:
            metrics = {
                "mse": _finite_float(row.get("mse", ""), "BASELINE_RESULTS_METRIC_INVALID"),
                "masked_mse": _finite_float(row.get("masked-mse", ""), "BASELINE_RESULTS_METRIC_INVALID"),
                "psnr": _finite_float(row.get("psnr", ""), "BASELINE_RESULTS_METRIC_INVALID"),
                "ssim": _finite_float(row.get("ssim", ""), "BASELINE_RESULTS_METRIC_INVALID"),
            }
            return BaselineTaskMetrics(
                task_id=task.task_id,
                environment=task.environment,
                cohort=task.cohort,
                split=task.split,
                steps=steps,
                trajectory_count=len(task.selection.trajectory_ids),
                window_count=_parse_window_count(task.task_root / "evaluation.stdout"),
                metrics=metrics,
                results_path=results_path,
                receipt_path=task.task_root / "receipt.json",
            )
    raise BaselineResultsError("BASELINE_RESULTS_ROW_MISSING")


def write_baseline_summary(summary: Mapping[str, object], output_path: Path) -> Path:
    target = Path(output_path)
    if target.exists() or target.is_symlink():
        raise BaselineResultsError("BASELINE_RESULTS_OUTPUT_EXISTS")
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    _write_json_atomic(target, summary)
    return target


def _summary_document(
    plan: BaselineLaunchPlan,
    *,
    state: str,
    state_counts: Mapping[str, int],
    task_metrics: Sequence[BaselineTaskMetrics],
    blockers: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    ready = state == "ready"
    return {
        "schema_version": 1,
        "artifact_type": "acwm-m0-baseline-summary",
        "state": state,
        "ready_for_archive": ready,
        "run_root": str(plan.run_root),
        "dataset_freeze_sha256": plan.dataset_freeze_sha256,
        "heldout_protocol_sha256": plan.heldout_protocol_sha256,
        "source_revision": plan.source_revision,
        "evaluator_freeze_sha256": plan.evaluator_freeze_sha256,
        "total_tasks": len(plan.tasks),
        "state_counts": dict(sorted(state_counts.items())),
        "metrics_ready_tasks": len(task_metrics),
        "blockers": [dict(item) for item in blockers],
        "task_metrics": [item.to_document() for item in task_metrics],
        "aggregate_metrics": _aggregate_metrics(task_metrics) if ready else {},
    }


def _aggregate_metrics(task_metrics: Sequence[BaselineTaskMetrics]) -> dict[str, object]:
    by_cohort: dict[str, list[BaselineTaskMetrics]] = {}
    by_environment: dict[str, list[BaselineTaskMetrics]] = {}
    for item in task_metrics:
        by_cohort.setdefault(item.cohort, []).append(item)
        by_environment.setdefault(item.environment, []).append(item)
    return {
        "overall_unweighted": _mean_metrics(task_metrics),
        "by_cohort_unweighted": {key: _mean_metrics(values) for key, values in sorted(by_cohort.items())},
        "by_environment_unweighted": {key: _mean_metrics(values) for key, values in sorted(by_environment.items())},
    }


def _mean_metrics(values: Sequence[BaselineTaskMetrics]) -> dict[str, float]:
    if not values:
        raise BaselineResultsError("BASELINE_RESULTS_EMPTY_AGGREGATE")
    return {metric: sum(item.metrics[metric] for item in values) / len(values) for metric in _METRICS}


def _parse_results_table(payload: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    header: list[str] | None = None
    for line in payload.splitlines():
        match = _ROW.match(line.strip())
        if not match:
            continue
        cells = [cell.strip().lower() for cell in match.group(1).split("|")]
        if not cells or all(set(cell) <= {":", "-"} for cell in cells):
            continue
        if header is None:
            header = cells
            continue
        if len(cells) != len(header):
            raise BaselineResultsError("BASELINE_RESULTS_TABLE_INVALID")
        rows.append(dict(zip(header, cells)))
    expected = {"env", "split", "steps", "mse", "masked-mse", "psnr", "ssim"}
    if header is None or not expected.issubset(set(header)):
        raise BaselineResultsError("BASELINE_RESULTS_TABLE_INVALID")
    return rows


def _parse_window_count(path: Path) -> int | None:
    if not path.exists():
        return None
    payload = _read_regular_text(path, "BASELINE_RESULTS_STDOUT_INVALID")
    matches = re.findall(r"Dataset size:\s*([0-9]+)\s+windows", payload)
    if len(matches) != 1:
        return None
    return int(matches[0])


def _read_regular_text(path: Path, code: str) -> str:
    if path.is_symlink() or not path.is_file():
        raise BaselineResultsError(code)
    if path.stat().st_size > 1024 * 1024:
        raise BaselineResultsError(code)
    return path.read_text(encoding="utf-8")


def _int_option(command: Sequence[str], option: str) -> int:
    try:
        index = command.index(option)
        value = command[index + 1]
    except (ValueError, IndexError) as exc:
        raise BaselineResultsError("BASELINE_RESULTS_COMMAND_INVALID") from exc
    return _parse_int(value)


def _parse_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise BaselineResultsError("BASELINE_RESULTS_INT_INVALID") from exc
    if parsed < 1:
        raise BaselineResultsError("BASELINE_RESULTS_INT_INVALID")
    return parsed


def _finite_float(value: str, code: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise BaselineResultsError(code) from exc
    if not math.isfinite(parsed):
        raise BaselineResultsError(code)
    return parsed


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    summarize = commands.add_parser("summarize", help="summarize completed M0 baseline metrics")
    summarize.add_argument("--launch-plan", type=Path, required=True)
    summarize.add_argument("--repo-root", type=Path)
    summarize.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.command == "summarize":
        plan = load_baseline_launch_plan(args.launch_plan, repo_root=args.repo_root)
        summary = summarize_baseline_results(plan)
        if args.output is not None:
            write_baseline_summary(summary, args.output)
        print(json.dumps(summary, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
