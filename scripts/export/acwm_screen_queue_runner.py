#!/usr/bin/env python3
"""Launch prepared ACWM screen queue rows after fresh GPU audits."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence


class AcwmScreenQueueRunnerError(RuntimeError):
    """ACWM screen queue launch failed closed."""


def run_queue_runner(
    *,
    queue_path: Path,
    output_root: Path,
    max_launches: int = 3,
    dry_run: bool = False,
    post_launch_wait_seconds: float = 8.0,
) -> dict[str, object]:
    """Run row-specific GPU audits and launch ready queue rows."""

    if max_launches < 1:
        raise AcwmScreenQueueRunnerError("ACWM_SCREEN_QUEUE_MAX_LAUNCHES_INVALID")
    if post_launch_wait_seconds < 0:
        raise AcwmScreenQueueRunnerError("ACWM_SCREEN_QUEUE_WAIT_INVALID")
    queue = _load_json_mapping(queue_path)
    rows = queue.get("rows")
    if not isinstance(rows, list):
        raise AcwmScreenQueueRunnerError("ACWM_SCREEN_QUEUE_ROWS_INVALID")
    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise AcwmScreenQueueRunnerError("ACWM_SCREEN_QUEUE_RUNNER_OUTPUT_EXISTS")
    launched_count = 0
    records: list[dict[str, object]] = []
    for raw in sorted((row for row in rows if isinstance(row, Mapping)), key=lambda row: int(row.get("rank", 999999))):
        row = dict(raw)
        if launched_count >= max_launches:
            records.append(_row_record(row, state="not_attempted", reason="MAX_LAUNCHES_REACHED"))
            continue
        preflight = _preflight_row(row)
        if preflight is not None:
            records.append(_row_record(row, state="skipped", reason=preflight))
            continue
        if dry_run:
            records.append(_row_record(row, state="dry_run_ready", reason="DRY_RUN"))
            launched_count += 1
            continue
        audit = _run_audit(row)
        if audit["state"] != "ready":
            records.append(_row_record(row, state="audit_blocked", reason=str(audit["reason"]), audit=audit))
            continue
        launch = _launch_row(row)
        launched_count += 1
        if post_launch_wait_seconds:
            time.sleep(post_launch_wait_seconds)
        records.append(
            _row_record(
                row,
                state="launched",
                reason="LAUNCHED",
                audit=audit,
                launch=launch,
                status=_load_optional_json(Path(str(row["output_root"])) / "status.json"),
            )
        )
    state = "ready" if any(record["state"] in {"launched", "dry_run_ready"} for record in records) else "blocked"
    report = {
        "schema_version": 1,
        "artifact_type": "wmloop-acwm-screen-queue-runner",
        "state": state,
        "dry_run": dry_run,
        "queue_path": str(Path(queue_path).resolve()),
        "max_launches": max_launches,
        "attempted_launch_slot_count": launched_count,
        "launched_count": sum(1 for record in records if record["state"] == "launched"),
        "dry_run_ready_count": sum(1 for record in records if record["state"] == "dry_run_ready"),
        "skipped_count": sum(1 for record in records if record["state"] == "skipped"),
        "audit_blocked_count": sum(1 for record in records if record["state"] == "audit_blocked"),
        "records": records,
        "limitations": [
            "The runner consumes an already prepared queue; it does not choose new primitives.",
            "Every real launch must pass a fresh row-specific GPU exclusivity audit.",
            "Existing audit roots are skipped rather than reused, because stale GPU audits are not launch authorization.",
        ],
    }
    _write_bundle(destination, report)
    return {
        key: report[key]
        for key in (
            "schema_version",
            "artifact_type",
            "state",
            "dry_run",
            "queue_path",
            "max_launches",
            "attempted_launch_slot_count",
            "launched_count",
            "dry_run_ready_count",
            "skipped_count",
            "audit_blocked_count",
        )
    } | {
        "report_path": str(destination / "queue-runner.json"),
        "markdown_path": str(destination / "queue-runner.md"),
        "csv_path": str(destination / "queue-runner.csv"),
    }


def _preflight_row(row: Mapping[str, object]) -> str | None:
    required = ("rank", "environment", "primitive", "seed", "train_steps", "audit_command", "audit_root", "audit_manifest", "launch_command", "output_root")
    missing = [key for key in required if not row.get(key)]
    if missing:
        return f"ROW_FIELDS_MISSING:{','.join(missing)}"
    output_root = Path(str(row["output_root"]))
    audit_root = Path(str(row["audit_root"]))
    if output_root.exists() or output_root.is_symlink():
        return "OUTPUT_ROOT_EXISTS"
    if audit_root.exists() or audit_root.is_symlink():
        return "AUDIT_ROOT_EXISTS_STALE_NOT_REUSED"
    return None


def _run_audit(row: Mapping[str, object]) -> dict[str, object]:
    command = shlex.split(str(row["audit_command"]))
    completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=120)
    manifest_path = Path(str(row["audit_manifest"]))
    manifest = _load_optional_json(manifest_path)
    state = manifest.get("state") if isinstance(manifest, Mapping) else None
    ready = completed.returncode == 0 and state == "ready" and manifest.get("gpu_exclusivity_ready") is True
    return {
        "state": "ready" if ready else "blocked",
        "reason": "AUDIT_READY" if ready else "AUDIT_NOT_READY",
        "returncode": completed.returncode,
        "stdout_tail": _tail(completed.stdout),
        "stderr_tail": _tail(completed.stderr),
        "manifest_path": str(manifest_path),
        "manifest_state": state,
        "blocker_count": manifest.get("blocker_count") if isinstance(manifest, Mapping) else None,
    }


def _launch_row(row: Mapping[str, object]) -> dict[str, object]:
    command = ["setsid", *shlex.split(str(row["launch_command"]))]
    output_root = Path(str(row["output_root"]))
    launch_log = output_root.with_name(f"{output_root.name}.launch.log")
    launch_log.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if launch_log.exists() or launch_log.is_symlink():
        raise AcwmScreenQueueRunnerError(f"ACWM_SCREEN_QUEUE_LAUNCH_LOG_EXISTS:{launch_log}")
    with launch_log.open("wb") as stdout:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=subprocess.STDOUT, close_fds=True)
    return {
        "pid": process.pid,
        "launch_log": str(launch_log),
        "command": command,
    }


def _row_record(
    row: Mapping[str, object],
    *,
    state: str,
    reason: str,
    audit: Mapping[str, object] | None = None,
    launch: Mapping[str, object] | None = None,
    status: Mapping[str, object] | None = None,
) -> dict[str, object]:
    return {
        "rank": row.get("rank"),
        "campaign_id": row.get("campaign_id"),
        "environment": row.get("environment"),
        "primitive": row.get("primitive"),
        "seed": row.get("seed"),
        "train_steps": row.get("train_steps"),
        "state": state,
        "reason": reason,
        "output_root": row.get("output_root"),
        "audit_root": row.get("audit_root"),
        "audit_manifest": row.get("audit_manifest"),
        "audit": dict(audit) if audit is not None else {},
        "launch": dict(launch) if launch is not None else {},
        "status_state": status.get("state") if isinstance(status, Mapping) else "",
        "status_path": str(Path(str(row.get("output_root", ""))) / "status.json") if row.get("output_root") else "",
    }


def _write_bundle(destination: Path, report: Mapping[str, object]) -> None:
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    try:
        temporary.mkdir(mode=0o700)
        _write_json(temporary / "queue-runner.json", report)
        _write_markdown(temporary / "queue-runner.md", report)
        _write_csv(temporary / "queue-runner.csv", report.get("records", []))
        os.replace(temporary, destination)
    except Exception:
        if temporary.is_symlink():
            temporary.unlink()
        elif temporary.exists():
            shutil.rmtree(temporary)
        raise


def _write_markdown(path: Path, report: Mapping[str, object]) -> None:
    lines = [
        "# ACWM Screen Queue Runner",
        "",
        f"State: `{report['state']}`",
        f"Dry run: `{report['dry_run']}`",
        f"Launched count: `{report['launched_count']}`",
        f"Dry-run ready count: `{report['dry_run_ready_count']}`",
        f"Skipped count: `{report['skipped_count']}`",
        f"Audit blocked count: `{report['audit_blocked_count']}`",
        "",
        "| Rank | Env | Primitive | Steps | State | Reason |",
        "|---:|---|---|---:|---|---|",
    ]
    records = report.get("records")
    if isinstance(records, list):
        for record in records:
            if isinstance(record, Mapping):
                lines.append(
                    "| {rank} | {environment} | {primitive} | {train_steps} | `{state}` | `{reason}` |".format(
                        rank=record.get("rank", ""),
                        environment=record.get("environment", ""),
                        primitive=record.get("primitive", ""),
                        train_steps=record.get("train_steps", ""),
                        state=record.get("state", ""),
                        reason=record.get("reason", ""),
                    )
                )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_csv(path: Path, records: object) -> None:
    fieldnames = [
        "rank",
        "campaign_id",
        "environment",
        "primitive",
        "seed",
        "train_steps",
        "state",
        "reason",
        "output_root",
        "audit_root",
        "audit_manifest",
        "status_state",
        "status_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        if isinstance(records, list):
            for record in records:
                if isinstance(record, Mapping):
                    writer.writerow({name: record.get(name, "") for name in fieldnames})


def _load_json_mapping(path: Path) -> dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise AcwmScreenQueueRunnerError(f"ACWM_SCREEN_QUEUE_JSON_NOT_OBJECT:{path}")
    return payload


def _load_optional_json(path: Path) -> dict[str, object]:
    try:
        if not path.is_file():
            return {}
        return _load_json_mapping(path)
    except (OSError, json.JSONDecodeError):
        return {}


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _tail(value: str, limit: int = 2000) -> str:
    return value[-limit:]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", nargs="?")
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-launches", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--post-launch-wait-seconds", type=float, default=8.0)
    args = parser.parse_args(argv)
    manifest = run_queue_runner(
        queue_path=args.queue,
        output_root=args.output_root,
        max_launches=args.max_launches,
        dry_run=args.dry_run,
        post_launch_wait_seconds=args.post_launch_wait_seconds,
    )
    print(json.dumps(manifest, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
