"""Import a ready M0 baseline summary into the immutable archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any, Mapping, Sequence

from wmloop.archive.store import ArchiveStore, BaselineRecord, ContentAddressedStore
from wmloop.evaluate.launch import BaselineLaunchPlan, BaselineLaunchTask, load_baseline_launch_plan


class BaselineArchiveImportError(RuntimeError):
    """A baseline summary could not be promoted into archive records."""


def import_baseline_summary(
    *,
    launch_plan_path: Path,
    summary_path: Path,
    archive_db: Path,
    repo_root: Path | None = None,
    cas_root: Path | None = None,
) -> dict[str, object]:
    """Record generation-zero baseline rows from a ready M0 summary."""

    plan = load_baseline_launch_plan(launch_plan_path, repo_root=repo_root)
    summary = _load_summary(summary_path)
    _verify_summary_matches_plan(summary, plan)
    archive = ArchiveStore(archive_db)
    cas = ContentAddressedStore(cas_root if cas_root is not None else Path(archive_db).resolve().parent)
    by_environment = _tasks_by_environment(plan.tasks)
    metrics_by_environment = _environment_metrics(summary)
    records: list[BaselineRecord] = []
    for environment in sorted(by_environment):
        metrics = metrics_by_environment.get(environment)
        if metrics is None:
            raise BaselineArchiveImportError(f"BASELINE_ARCHIVE_ENVIRONMENT_METRICS_MISSING:{environment}")
        tasks = by_environment[environment]
        checkpoint_path = _single_checkpoint_path(environment, tasks)
        model_ref = cas.put_bytes(
            _canonical_json_bytes(
                {
                    "schema_version": 1,
                    "artifact_type": "acwm-m0-baseline-model-ref",
                    "environment": environment,
                    "generation": 0,
                    "source_revision": plan.source_revision,
                    "dataset_freeze_sha256": plan.dataset_freeze_sha256,
                    "heldout_protocol_sha256": plan.heldout_protocol_sha256,
                    "evaluator_freeze_sha256": plan.evaluator_freeze_sha256,
                    "checkpoint_path": str(checkpoint_path),
                    "checkpoint_sha256": _sha256_regular_file(checkpoint_path),
                    "checkpoint_size": _regular_file_size(checkpoint_path),
                }
            ),
            media_type="application/json",
        ).uri
        receipt_ref = cas.put_bytes(
            _canonical_json_bytes(_receipt_bundle(environment, plan, tasks, summary)),
            media_type="application/json",
        ).uri
        record = BaselineRecord(
            environment=environment,
            model_ref=model_ref,
            evaluator_freeze_sha256=plan.evaluator_freeze_sha256,
            heldout_split_sha256=plan.heldout_protocol_sha256,
            receipt_ref=receipt_ref,
            metrics=metrics,
        )
        archive.record_baseline(record)
        records.append(record)
    return {
        "schema_version": 1,
        "artifact_type": "acwm-m0-baseline-archive-import",
        "state": "recorded",
        "archive_db": str(Path(archive_db).resolve()),
        "cas_root": str((cas_root if cas_root is not None else Path(archive_db).resolve().parent).resolve()),
        "recorded_baselines": len(records),
        "environments": [record.environment for record in records],
        "archive_statistics": archive.archive_statistics(),
    }


def _load_summary(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BaselineArchiveImportError("BASELINE_ARCHIVE_SUMMARY_INVALID") from exc
    if not isinstance(payload, Mapping):
        raise BaselineArchiveImportError("BASELINE_ARCHIVE_SUMMARY_INVALID")
    return payload


def _verify_summary_matches_plan(summary: Mapping[str, Any], plan: BaselineLaunchPlan) -> None:
    if summary.get("schema_version") != 1 or summary.get("artifact_type") != "acwm-m0-baseline-summary":
        raise BaselineArchiveImportError("BASELINE_ARCHIVE_SUMMARY_INVALID")
    if summary.get("state") != "ready" or summary.get("ready_for_archive") is not True:
        raise BaselineArchiveImportError("BASELINE_ARCHIVE_SUMMARY_NOT_READY")
    expected = {
        "dataset_freeze_sha256": plan.dataset_freeze_sha256,
        "heldout_protocol_sha256": plan.heldout_protocol_sha256,
        "source_revision": plan.source_revision,
        "evaluator_freeze_sha256": plan.evaluator_freeze_sha256,
        "total_tasks": len(plan.tasks),
        "metrics_ready_tasks": len(plan.tasks),
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise BaselineArchiveImportError(f"BASELINE_ARCHIVE_SUMMARY_PLAN_MISMATCH:{key}")
    if summary.get("state_counts") != {"completed": len(plan.tasks)}:
        raise BaselineArchiveImportError("BASELINE_ARCHIVE_SUMMARY_STATE_COUNTS_INVALID")


def _tasks_by_environment(tasks: Sequence[BaselineLaunchTask]) -> dict[str, tuple[BaselineLaunchTask, ...]]:
    grouped: dict[str, list[BaselineLaunchTask]] = {}
    for task in tasks:
        grouped.setdefault(task.environment, []).append(task)
    return {environment: tuple(items) for environment, items in grouped.items()}


def _environment_metrics(summary: Mapping[str, Any]) -> dict[str, Mapping[str, float]]:
    aggregate = summary.get("aggregate_metrics")
    if not isinstance(aggregate, Mapping):
        raise BaselineArchiveImportError("BASELINE_ARCHIVE_METRICS_INVALID")
    raw = aggregate.get("by_environment_unweighted")
    if not isinstance(raw, Mapping):
        raise BaselineArchiveImportError("BASELINE_ARCHIVE_METRICS_INVALID")
    return {str(environment): _metrics_mapping(metrics) for environment, metrics in raw.items()}


def _metrics_mapping(payload: object) -> Mapping[str, float]:
    if not isinstance(payload, Mapping):
        raise BaselineArchiveImportError("BASELINE_ARCHIVE_METRICS_INVALID")
    metrics: dict[str, float] = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not isinstance(value, (int, float)) or isinstance(value, bool):
            raise BaselineArchiveImportError("BASELINE_ARCHIVE_METRICS_INVALID")
        metrics[key] = float(value)
    return metrics


def _single_checkpoint_path(environment: str, tasks: Sequence[BaselineLaunchTask]) -> Path:
    paths = {task.checkpoint_path for task in tasks}
    if len(paths) != 1:
        raise BaselineArchiveImportError(f"BASELINE_ARCHIVE_CHECKPOINT_INCONSISTENT:{environment}")
    return next(iter(paths))


def _receipt_bundle(
    environment: str,
    plan: BaselineLaunchPlan,
    tasks: Sequence[BaselineLaunchTask],
    summary: Mapping[str, Any],
) -> Mapping[str, object]:
    metrics_by_task = _task_metrics(summary)
    return {
        "schema_version": 1,
        "artifact_type": "acwm-m0-baseline-receipt-bundle",
        "environment": environment,
        "generation": 0,
        "run_root": str(plan.run_root),
        "dataset_freeze_sha256": plan.dataset_freeze_sha256,
        "heldout_protocol_sha256": plan.heldout_protocol_sha256,
        "evaluator_freeze_sha256": plan.evaluator_freeze_sha256,
        "tasks": [_task_receipt_document(task, metrics_by_task) for task in sorted(tasks, key=lambda item: item.task_id)],
    }


def _task_metrics(summary: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    raw = summary.get("task_metrics")
    if not isinstance(raw, list):
        raise BaselineArchiveImportError("BASELINE_ARCHIVE_TASK_METRICS_INVALID")
    metrics: dict[str, Mapping[str, Any]] = {}
    for item in raw:
        if not isinstance(item, Mapping) or not isinstance(item.get("task_id"), str):
            raise BaselineArchiveImportError("BASELINE_ARCHIVE_TASK_METRICS_INVALID")
        task_id = str(item["task_id"])
        if task_id in metrics:
            raise BaselineArchiveImportError("BASELINE_ARCHIVE_TASK_METRICS_INVALID")
        metrics[task_id] = item
    return metrics


def _task_receipt_document(task: BaselineLaunchTask, metrics_by_task: Mapping[str, Mapping[str, Any]]) -> Mapping[str, object]:
    metrics = metrics_by_task.get(task.task_id)
    if metrics is None:
        raise BaselineArchiveImportError(f"BASELINE_ARCHIVE_TASK_METRICS_MISSING:{task.task_id}")
    receipt_payload = _read_json_regular(task.task_root / "receipt.json", "BASELINE_ARCHIVE_RECEIPT_INVALID")
    results_path = task.output_root / "results.md"
    return {
        "task_id": task.task_id,
        "cohort": task.cohort,
        "split": task.split,
        "planned_gpu": task.gpu,
        "actual_gpu": receipt_payload.get("actual_gpu", task.gpu),
        "receipt_path": str(task.task_root / "receipt.json"),
        "receipt_sha256": _sha256_regular_file(task.task_root / "receipt.json"),
        "results_path": str(results_path),
        "results_sha256": _sha256_regular_file(results_path),
        "metrics": metrics.get("metrics"),
        "trajectory_count": metrics.get("trajectory_count"),
        "window_count": metrics.get("window_count"),
    }


def _read_json_regular(path: Path, code: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(_read_regular_bytes(path).decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BaselineArchiveImportError(code) from exc
    if not isinstance(payload, Mapping):
        raise BaselineArchiveImportError(code)
    return payload


def _sha256_regular_file(path: Path) -> str:
    handle = _open_regular_file(path)
    try:
        digest = hashlib.sha256()
        while True:
            chunk = os.read(handle, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        return digest.hexdigest()
    finally:
        os.close(handle)


def _read_regular_bytes(path: Path) -> bytes:
    handle = _open_regular_file(path)
    try:
        chunks: list[bytes] = []
        while True:
            chunk = os.read(handle, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(handle)


def _regular_file_size(path: Path) -> int:
    metadata = _regular_file_metadata(path)
    return int(metadata.st_size)


def _open_regular_file(path: Path) -> int:
    before = _regular_file_metadata(path)
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise BaselineArchiveImportError("BASELINE_ARCHIVE_REGULAR_FILE_INVALID") from exc
    opened = os.fstat(descriptor)
    if (before.st_dev, before.st_ino, before.st_size) != (opened.st_dev, opened.st_ino, opened.st_size):
        os.close(descriptor)
        raise BaselineArchiveImportError("BASELINE_ARCHIVE_REGULAR_FILE_CHANGED")
    return descriptor


def _regular_file_metadata(path: Path) -> os.stat_result:
    try:
        metadata = os.lstat(path)
    except OSError as exc:
        raise BaselineArchiveImportError("BASELINE_ARCHIVE_REGULAR_FILE_INVALID") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise BaselineArchiveImportError("BASELINE_ARCHIVE_REGULAR_FILE_INVALID")
    return metadata


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    import_command = commands.add_parser("import", help="record ready M0 baseline rows into archive.db")
    import_command.add_argument("--launch-plan", type=Path, required=True)
    import_command.add_argument("--summary", type=Path, required=True)
    import_command.add_argument("--archive-db", type=Path, required=True)
    import_command.add_argument("--repo-root", type=Path)
    import_command.add_argument("--cas-root", type=Path)
    args = parser.parse_args(argv)
    if args.command == "import":
        result = import_baseline_summary(
            launch_plan_path=args.launch_plan,
            summary_path=args.summary,
            archive_db=args.archive_db,
            repo_root=args.repo_root,
            cas_root=args.cas_root,
        )
        print(json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
