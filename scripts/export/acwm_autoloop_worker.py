#!/usr/bin/env python3
"""Run a dependency-aware ACWM autoloop queue.

The worker is deliberately conservative:
- it never reuses old GPU audits;
- it never writes into an existing output root;
- official eval gates wait for a positive 512 manifest;
- staged confirmation rows wait for a passing official eval.py quality gate;
- every launch receives a fresh GPU exclusivity audit.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
CPU_ONLY_PHASES = {
    "event_semantic_gate",
    "horizon_effect_profile",
    "horizon_triptych",
    "horizon_experience_map",
}


class AcwmAutoloopWorkerError(RuntimeError):
    """ACWM autoloop worker failed closed."""


def run_autoloop_worker(
    *,
    queue_path: Path,
    output_root: Path,
    max_launches: int = 1,
    iterations: int = 1,
    sleep_seconds: float = 60.0,
    dry_run: bool = False,
    now_token: str | None = None,
    phase_allowlist: set[str] | None = None,
    allowed_gpu_indices: set[int] | None = None,
) -> dict[str, object]:
    if max_launches < 1:
        raise AcwmAutoloopWorkerError("ACWM_AUTOLOOP_WORKER_MAX_LAUNCHES_INVALID")
    if iterations < 1:
        raise AcwmAutoloopWorkerError("ACWM_AUTOLOOP_WORKER_ITERATIONS_INVALID")
    if sleep_seconds < 0:
        raise AcwmAutoloopWorkerError("ACWM_AUTOLOOP_WORKER_SLEEP_INVALID")
    if allowed_gpu_indices is not None and any(
        isinstance(gpu, bool) or not isinstance(gpu, int) or gpu < 0
        for gpu in allowed_gpu_indices
    ):
        raise AcwmAutoloopWorkerError("ACWM_AUTOLOOP_WORKER_ALLOWED_GPUS_INVALID")
    queue = _load_json_object(queue_path)
    rows = queue.get("rows")
    if not isinstance(rows, list):
        raise AcwmAutoloopWorkerError("ACWM_AUTOLOOP_QUEUE_ROWS_INVALID")
    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise AcwmAutoloopWorkerError("ACWM_AUTOLOOP_WORKER_OUTPUT_EXISTS")

    token = now_token or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    launched_count = 0
    reserved_gpus: set[int] = set()
    active_gpu_pids: dict[int, int] = {}
    released_gpu_launch_count = 0
    reserved_output_roots: set[str] = set()
    records: list[dict[str, object]] = []
    for iteration in range(1, iterations + 1):
        for gpu, pid in list(active_gpu_pids.items()):
            if _pid_is_running(pid):
                continue
            del active_gpu_pids[gpu]
            released_gpu_launch_count += 1
        idle_gpus = _idle_gpu_indices() - reserved_gpus - set(active_gpu_pids)
        if allowed_gpu_indices is not None:
            idle_gpus &= allowed_gpu_indices
        launched_this_iteration = False
        for raw in sorted((row for row in rows if isinstance(row, Mapping)), key=lambda item: int(item.get("rank", 999999))):
            row = dict(raw)
            if launched_count >= max_launches:
                records.append(_record(row, iteration=iteration, state="not_attempted", reason="MAX_LAUNCHES_REACHED"))
                break
            output_root = str(Path(str(row.get("output_root") or "")).resolve())
            if output_root in reserved_output_roots:
                records.append(_record(row, iteration=iteration, state="skipped", reason="OUTPUT_ROOT_RESERVED"))
                continue
            if phase_allowlist is not None and str(row.get("phase") or "") not in phase_allowlist:
                records.append(_record(row, iteration=iteration, state="skipped", reason="PHASE_EXCLUDED"))
                continue
            preflight = _row_preflight(row)
            if preflight is not None:
                records.append(_record(row, iteration=iteration, state="skipped", reason=preflight))
                continue
            if _row_resource_class(row) == "cpu":
                if dry_run:
                    records.append(
                        _record(
                            row,
                            iteration=iteration,
                            state="dry_run_ready",
                            reason="DRY_RUN",
                        )
                    )
                else:
                    launch = _launch_row(
                        row,
                        gpu=-1,
                        audit_manifest="",
                        attempt_id=f"{token}-i{iteration:03d}",
                    )
                    records.append(
                        _record(
                            row,
                            iteration=iteration,
                            state="launched",
                            reason="LAUNCHED_CPU_ONLY",
                            launch=launch,
                        )
                    )
                launched_count += 1
                reserved_output_roots.add(output_root)
                launched_this_iteration = True
                break
            candidate_gpus = [gpu for gpu in _row_gpus(row) if gpu in idle_gpus]
            if not candidate_gpus and row.get("allow_any_idle_gpu") is True:
                candidate_gpus = sorted(idle_gpus)
            if not candidate_gpus:
                records.append(_record(row, iteration=iteration, state="waiting_for_gpu", reason="NO_IDLE_CANDIDATE_GPU"))
                continue
            if dry_run:
                records.append(_record(row, iteration=iteration, state="dry_run_ready", reason="DRY_RUN", selected_gpu=candidate_gpus[0]))
                launched_count += 1
                reserved_gpus.add(candidate_gpus[0])
                reserved_output_roots.add(output_root)
                launched_this_iteration = True
                break
            launch = _launch_first_ready_gpu(row, candidate_gpus=candidate_gpus, attempt_id=f"{token}-i{iteration:03d}")
            records.append(
                _record(
                    row,
                    iteration=iteration,
                    state=str(launch["state"]),
                    reason=str(launch["reason"]),
                    selected_gpu=launch.get("selected_gpu"),
                    audit=launch.get("audit"),
                    launch=launch.get("launch"),
                )
            )
            if launch["state"] == "launched":
                launched_count += 1
                reserved_output_roots.add(output_root)
                selected_gpu = launch.get("selected_gpu")
                if isinstance(selected_gpu, int):
                    launch_payload = launch.get("launch")
                    pid = launch_payload.get("pid") if isinstance(launch_payload, Mapping) else None
                    if isinstance(pid, int) and pid > 0:
                        active_gpu_pids[selected_gpu] = pid
                launched_this_iteration = True
                break
        if launched_count >= max_launches:
            break
        if iteration < iterations and not launched_this_iteration:
            time.sleep(sleep_seconds)

    state = "ready" if launched_count > 0 or dry_run else "blocked"
    launched_gpu_indices = sorted(
        {
            int(record["selected_gpu"])
            for record in records
            if record.get("state") in {"launched", "dry_run_ready"}
            and isinstance(record.get("selected_gpu"), int)
            and not isinstance(record.get("selected_gpu"), bool)
        }
    )
    launched_cpu_count = sum(
        1
        for record in records
        if record.get("state") in {"launched", "dry_run_ready"}
        and record.get("resource_class") == "cpu"
    )
    report = {
        "schema_version": 1,
        "artifact_type": "wmloop-acwm-autoloop-worker",
        "state": state,
        "queue_path": str(Path(queue_path).resolve()),
        "dry_run": dry_run,
        "max_launches": max_launches,
        "iterations": iterations,
        "sleep_seconds": sleep_seconds,
        "phase_allowlist": sorted(phase_allowlist) if phase_allowlist is not None else None,
        "allowed_gpu_indices": (
            sorted(allowed_gpu_indices) if allowed_gpu_indices is not None else None
        ),
        "launched_count": launched_count,
        "launched_gpu_indices": launched_gpu_indices,
        "launched_cpu_count": launched_cpu_count,
        "released_gpu_launch_count": released_gpu_launch_count,
        "records": records,
        "limitations": [
            "This worker starts queued training/eval jobs only; it does not mutate the primitive registry.",
            "Positive 512-screen evidence is candidate discovery only.",
            "Staged confirmation rows require a passing official eval.py PSNR/SSIM/MSE quality gate.",
            "Every launch uses a fresh GPU exclusivity audit rooted by the worker attempt id.",
        ],
    }
    return _write_bundle(destination, report)


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    stat_path = Path("/proc") / str(pid) / "stat"
    try:
        fields = stat_path.read_text(encoding="ascii").split()
    except (OSError, PermissionError):
        return False
    return len(fields) >= 3 and fields[2] != "Z"


def _row_preflight(row: Mapping[str, object]) -> str | None:
    for key in ("rank", "phase", "campaign_id", "environment", "primitive", "seed", "train_steps", "output_root", "launch_argv_template"):
        value = row.get(key)
        if value is None or value == "":
            return f"ROW_FIELD_MISSING:{key}"
    output_root = Path(str(row["output_root"]))
    if _row_resource_class(row) not in {"cpu", "gpu"}:
        return "ROW_RESOURCE_CLASS_INVALID"
    if output_root.exists() or output_root.is_symlink():
        return "OUTPUT_ROOT_EXISTS"
    marker = _launch_marker_path(output_root)
    if marker.exists() or marker.is_symlink():
        try:
            pid = int(marker.read_text(encoding="ascii").strip())
        except (OSError, ValueError):
            pid = -1
        if pid > 0:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                marker.unlink(missing_ok=True)
            except PermissionError:
                return "OUTPUT_ROOT_LAUNCHING"
            else:
                return "OUTPUT_ROOT_LAUNCHING"
        else:
            marker.unlink(missing_ok=True)
    dependency = str(row.get("requires_positive_manifest") or "")
    if dependency:
        blocker = _positive_dependency_blocker(Path(dependency))
        if blocker is not None:
            return blocker
    ready_dependency = str(row.get("requires_ready_manifest") or "")
    if ready_dependency:
        blocker = _ready_dependency_blocker(Path(ready_dependency))
        if blocker is not None:
            return blocker
    ready_dependencies = row.get("requires_ready_manifests", [])
    if ready_dependencies:
        if not isinstance(ready_dependencies, list) or not all(
            isinstance(value, str) and value for value in ready_dependencies
        ):
            return "READY_DEPENDENCIES_INVALID"
        for index, value in enumerate(ready_dependencies):
            blocker = _ready_dependency_blocker(Path(value))
            if blocker is not None:
                return f"READY_DEPENDENCY_{index}:{blocker}"
    official_dependency = str(row.get("requires_official_quality_manifest") or "")
    if official_dependency:
        return _official_quality_dependency_blocker(Path(official_dependency))
    if row.get("phase") in {"confirm_4k", "confirm_staged"}:
        return "OFFICIAL_QUALITY_GATE_REQUIRED"
    return None


def _row_resource_class(row: Mapping[str, object]) -> str:
    value = row.get("resource_class")
    if value is None or value == "":
        return "cpu" if str(row.get("phase") or "") in CPU_ONLY_PHASES else "gpu"
    return str(value)


def _ready_dependency_blocker(manifest_path: Path) -> str | None:
    if not manifest_path.is_file():
        return "READY_DEPENDENCY_PENDING"
    manifest = _load_json_object(manifest_path)
    if manifest.get("state") != "ready":
        return f"READY_DEPENDENCY_NOT_READY:{manifest.get('state')}"
    return None


def _positive_dependency_blocker(manifest_path: Path) -> str | None:
    if not manifest_path.is_file():
        return "DEPENDENCY_PENDING"
    manifest = _load_json_object(manifest_path)
    if manifest.get("state") != "ready":
        return f"DEPENDENCY_NOT_READY:{manifest.get('state')}"
    primary_metric = str(manifest.get("primary_metric") or "ladder_auc_psnr_envmax")
    delta = _metric_delta(manifest, primary_metric)
    if delta is None:
        return "DEPENDENCY_METRIC_UNAVAILABLE"
    if delta <= 0.0:
        return f"DEPENDENCY_NONPOSITIVE:{delta}"
    action_gate = manifest.get("action_following_gate")
    if isinstance(action_gate, Mapping) and action_gate.get("enabled") is True and action_gate.get("pass") is not True:
        return "DEPENDENCY_ACTION_GATE_FAILED"
    return None


def _metric_delta(manifest: Mapping[str, object], primary_metric: str) -> float | None:
    raw = manifest.get("delta_m_ver")
    if not isinstance(raw, Mapping):
        return None
    for key in (primary_metric, "ladder_auc_psnr_envmax"):
        value = raw.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        value = float(value)
        if math.isfinite(value):
            return value
    return None


def _official_quality_dependency_blocker(manifest_path: Path) -> str | None:
    if not manifest_path.is_file():
        return "OFFICIAL_QUALITY_GATE_PENDING"
    manifest = _load_json_object(manifest_path)
    if manifest.get("state") != "ready":
        return f"OFFICIAL_QUALITY_GATE_NOT_READY:{manifest.get('state')}"
    gate = manifest.get("official_quality_gate")
    if not isinstance(gate, Mapping):
        return "OFFICIAL_QUALITY_GATE_MISSING"
    if gate.get("pass") is not True:
        delta = gate.get("delta_candidate_minus_baseline")
        psnr_delta = delta.get("psnr") if isinstance(delta, Mapping) else "unknown"
        return f"OFFICIAL_QUALITY_GATE_FAILED:{psnr_delta}"
    return None


def _launch_first_ready_gpu(
    row: Mapping[str, object],
    *,
    candidate_gpus: Sequence[int],
    attempt_id: str,
) -> dict[str, object]:
    audit_attempts = []
    for gpu in candidate_gpus:
        audit = _run_gpu_audit(row, gpu=gpu, attempt_id=attempt_id)
        audit_attempts.append(audit)
        if audit["state"] != "ready":
            continue
        launch = _launch_row(row, gpu=gpu, audit_manifest=str(audit["manifest_path"]), attempt_id=attempt_id)
        return {
            "state": "launched",
            "reason": "LAUNCHED",
            "selected_gpu": gpu,
            "audit": audit,
            "launch": launch,
        }
    return {
        "state": "audit_blocked",
        "reason": "NO_GPU_AUDIT_READY",
        "audit": {"attempts": audit_attempts},
        "launch": {},
    }


def _run_gpu_audit(row: Mapping[str, object], *, gpu: int, attempt_id: str) -> dict[str, object]:
    audit_root = Path(str(row["gpu_audit_root_template"]).format(gpu=gpu, attempt_id=attempt_id)).resolve()
    if audit_root.exists() or audit_root.is_symlink():
        audit_root = audit_root.with_name(f"{audit_root.name}-{uuid.uuid4().hex[:8]}")
    archive_db = str(row.get("archive_db") or ROOT / "results/archive.db")
    cas_root = str(row.get("cas_root") or ROOT / "results")
    command = [
        str(ROOT / ".venv/bin/python3"),
        "-m",
        "wmloop.execute.gpu_exclusivity_audit",
        "run",
        "--output-root",
        str(audit_root),
        "--gpus",
        str(gpu),
        "--archive-db",
        archive_db,
        "--cas-root",
        cas_root,
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=180)
    manifest_path = audit_root / "manifest.json"
    manifest = _load_optional_json(manifest_path)
    ready = (
        completed.returncode == 0
        and manifest.get("state") == "ready"
        and manifest.get("gpu_exclusivity_ready") is True
    )
    return {
        "state": "ready" if ready else "blocked",
        "returncode": completed.returncode,
        "gpu": gpu,
        "audit_root": str(audit_root),
        "manifest_path": str(manifest_path),
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-2000:],
        "manifest_state": manifest.get("state"),
        "blocker_count": manifest.get("blocker_count"),
    }


def _launch_row(row: Mapping[str, object], *, gpu: int, audit_manifest: str, attempt_id: str) -> dict[str, object]:
    argv_template = row.get("launch_argv_template")
    if not isinstance(argv_template, list):
        raise AcwmAutoloopWorkerError("ACWM_AUTOLOOP_LAUNCH_TEMPLATE_INVALID")
    argv = [
        str(part).format(gpu=gpu, gpu_audit_manifest=audit_manifest, attempt_id=attempt_id)
        for part in argv_template
    ]
    output_root = Path(str(row["output_root"]))
    launch_log = output_root.with_name(f"{output_root.name}.{attempt_id}.launch.log")
    launch_log.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with launch_log.open("wb") as handle:
        process = subprocess.Popen(["setsid", *argv], stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT, close_fds=True)
    marker = _launch_marker_path(output_root)
    marker.write_text(str(process.pid), encoding="ascii")
    return {
        "pid": process.pid,
        "launch_log": str(launch_log),
        "argv": argv,
    }


def _launch_marker_path(output_root: Path) -> Path:
    return output_root.with_name(f"{output_root.name}.launching")


def _idle_gpu_indices() -> set[int]:
    try:
        gpu_raw = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        )
        apps_raw = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return set()
    app_present = bool(apps_raw.strip())
    idle: set[int] = set()
    for row in csv.reader(gpu_raw.splitlines()):
        if len(row) < 3:
            continue
        try:
            index = int(row[0].strip())
            memory = int(row[1].strip())
            utilization = int(row[2].strip())
        except ValueError:
            continue
        if app_present:
            # If any compute app exists, rely on fresh per-GPU audit instead of
            # treating high-level utilization as sufficient.  The memory check
            # still prevents obviously busy GPUs from audit churn.
            if memory <= 1024:
                idle.add(index)
        elif memory <= 1024 and utilization <= 10:
            idle.add(index)
    return idle - _reserved_gpu_indices_from_processes()


def _reserved_gpu_indices_from_processes(proc_root: Path = Path("/proc")) -> set[int]:
    """Treat launched autoloop jobs as reservations before CUDA initializes."""

    reserved: set[int] = set()
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            parts = [
                part.decode("utf-8", errors="replace")
                for part in (entry / "cmdline").read_bytes().split(b"\0")
                if part
            ]
        except (OSError, PermissionError):
            continue
        markers = (
            "wmloop.execute.training_eval_limited_campaign",
            "wmloop.execute.primitive_runtime_smoke",
            "wmloop.orchestrator_training_eval_smoke",
            "wmloop.diagnose.horizon_runtime",
            "scripts/export/acwm_failed_screen_salvage.py",
            "scripts/export/acwm_runtime_only_screen.py",
            "scripts/export/acwm_formal_visualization.py",
        )
        if not parts or not any(
            part == marker or part.endswith(f"/{marker}")
            for part in parts
            for marker in markers
        ):
            continue
        for flag in ("--gpus", "--gpu-index"):
            try:
                value = int(parts[parts.index(flag) + 1])
            except (ValueError, IndexError):
                continue
            if value >= 0:
                reserved.add(value)
    return reserved


def _row_gpus(row: Mapping[str, object]) -> list[int]:
    values = row.get("candidate_gpus")
    if not isinstance(values, list):
        return []
    output = []
    for value in values:
        if isinstance(value, bool):
            continue
        try:
            output.append(int(value))
        except (TypeError, ValueError):
            continue
    return output


def _record(
    row: Mapping[str, object],
    *,
    iteration: int,
    state: str,
    reason: str,
    selected_gpu: object = "",
    audit: object = None,
    launch: object = None,
) -> dict[str, object]:
    return {
        "iteration": iteration,
        "rank": row.get("rank"),
        "phase": row.get("phase"),
        "campaign_id": row.get("campaign_id"),
        "environment": row.get("environment"),
        "primitive": row.get("primitive"),
        "seed": row.get("seed"),
        "train_steps": row.get("train_steps"),
        "state": state,
        "reason": reason,
        "selected_gpu": selected_gpu,
        "output_root": row.get("output_root"),
        "requires_positive_manifest": row.get("requires_positive_manifest"),
        "requires_ready_manifest": row.get("requires_ready_manifest"),
        "requires_official_quality_manifest": row.get("requires_official_quality_manifest"),
        "resource_class": _row_resource_class(row),
        "audit": audit if isinstance(audit, Mapping) else {},
        "launch": launch if isinstance(launch, Mapping) else {},
    }


def _write_bundle(destination: Path, report: Mapping[str, object]) -> dict[str, object]:
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    try:
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary.mkdir(mode=0o700)
        _write_json(temporary / "autoloop-worker.json", report)
        _write_csv(temporary / "autoloop-worker.csv", report.get("records", []))
        _write_markdown(temporary / "autoloop-worker.md", report)
        manifest = {
            "schema_version": 1,
            "artifact_type": "wmloop-acwm-autoloop-worker-manifest",
            "state": report["state"],
            "queue_path": report["queue_path"],
            "launched_count": report["launched_count"],
            "launched_gpu_indices": report["launched_gpu_indices"],
            "launched_cpu_count": report["launched_cpu_count"],
            "released_gpu_launch_count": report["released_gpu_launch_count"],
            "report_path": str(destination / "autoloop-worker.json"),
            "markdown_path": str(destination / "autoloop-worker.md"),
            "csv_path": str(destination / "autoloop-worker.csv"),
        }
        _write_json(temporary / "manifest.json", manifest)
        os.replace(temporary, destination)
        return manifest
    except Exception:
        if temporary.is_symlink():
            temporary.unlink()
        elif temporary.exists():
            shutil.rmtree(temporary)
        raise


def _write_markdown(path: Path, report: Mapping[str, object]) -> None:
    lines = [
        "# ACWM Autoloop Worker",
        "",
        f"State: `{report['state']}`",
        f"Launched: `{report['launched_count']}`",
        "",
        "| Iter | Rank | Phase | Env | Primitive | Steps | State | Reason | GPU |",
        "|---:|---:|---|---|---|---:|---|---|---:|",
    ]
    records = report.get("records")
    if isinstance(records, list):
        for record in records:
            if isinstance(record, Mapping):
                lines.append(
                    "| {iteration} | {rank} | `{phase}` | {environment} | {primitive} | {train_steps} | `{state}` | `{reason}` | {selected_gpu} |".format(
                        iteration=record.get("iteration", ""),
                        rank=record.get("rank", ""),
                        phase=record.get("phase", ""),
                        environment=record.get("environment", ""),
                        primitive=record.get("primitive", ""),
                        train_steps=record.get("train_steps", ""),
                        state=record.get("state", ""),
                        reason=record.get("reason", ""),
                        selected_gpu=record.get("selected_gpu", ""),
                    )
                )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_csv(path: Path, records: object) -> None:
    fieldnames = [
        "iteration",
        "rank",
        "phase",
        "campaign_id",
        "environment",
        "primitive",
        "seed",
        "train_steps",
        "state",
        "reason",
        "selected_gpu",
        "output_root",
        "requires_positive_manifest",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        if isinstance(records, list):
            for record in records:
                if isinstance(record, Mapping):
                    writer.writerow({name: record.get(name, "") for name in fieldnames})


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json_object(path: Path) -> dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise AcwmAutoloopWorkerError(f"ACWM_AUTOLOOP_JSON_NOT_OBJECT:{path}")
    return payload


def _load_optional_json(path: Path) -> dict[str, object]:
    try:
        if not path.is_file():
            return {}
        return _load_json_object(path)
    except (OSError, json.JSONDecodeError):
        return {}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", nargs="?")
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-launches", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--sleep-seconds", type=float, default=60.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    manifest = run_autoloop_worker(
        queue_path=args.queue,
        output_root=args.output_root,
        max_launches=args.max_launches,
        iterations=args.iterations,
        sleep_seconds=args.sleep_seconds,
        dry_run=args.dry_run,
    )
    print(json.dumps(manifest, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
