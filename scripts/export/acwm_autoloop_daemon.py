#!/usr/bin/env python3
"""Continuously advance one or more dependency-aware ACWM autoloop queues."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Sequence

from scripts.export.acwm_8env_live_coverage import build_live_coverage
from scripts.export.acwm_candidate_frontier import build_candidate_frontier
from scripts.export.acwm_autoloop_queue import (
    DEFAULT_CHECKPOINT_ROOT,
    DEFAULT_DATA_ROOT,
    DEFAULT_RUNTIME_PYTHON,
)
from scripts.export.acwm_autoloop_replenisher import replenish_autoloop_queue
from wmloop.execute.acwm_primitive_routes import INVALIDATED_QUALITY_PRIMITIVES
from wmloop.execute.training_monitor_policy import DEFAULT_CONFIRMATION_STEPS
from scripts.export.acwm_autoloop_worker import (
    CPU_ONLY_PHASES,
    _idle_gpu_indices,
    run_autoloop_worker,
)


Replenisher = Callable[[int, int], Mapping[str, object]]
FrontierRefresher = Callable[[int], Mapping[str, object]]

QUALITY_DISCOVERY_PHASES = {
    "materialize_runtime_admission",
    "failed_screen_salvage",
    "screen_512",
    "official_eval_gate",
}
POST_CONFIRMATION_EVIDENCE_PHASES = {
    "long_horizon_baseline",
    "long_horizon_candidate",
    "event_semantic_gate",
    "horizon_effect_profile",
    "horizon_triptych",
    "horizon_experience_map",
}
_CONFIRM_OFFICIAL_STEP = re.compile(r"-step(\d+)-r\d+")
_SCREEN_CHECKPOINT_STEP = re.compile(r"acwm-autoloop-screen-.+-t(\d+)-r\d+(?:-retry\d+)?")
_CAMPAIGN_SEED = re.compile(r"(?:^|-)s(\d+)(?:-|$)")
_COVERAGE_REFRESH_INTERVAL_CYCLES = 5
_DEFAULT_STATIC_QUEUE_SCAN_BUDGET_SECONDS = 10.0
_DEFAULT_STATIC_RECENT_QUEUE_COUNT = 8


def _static_queue_scan_indices(
    queue_count: int,
    *,
    cursor: int,
    recent_count: int,
) -> tuple[list[int], int, int]:
    """Prioritize recent queues, then round-robin the older backlog."""

    recent_start = max(0, queue_count - recent_count)
    recent_indices = list(range(queue_count - 1, recent_start - 1, -1))
    older_count = recent_start
    if older_count == 0:
        return recent_indices, len(recent_indices), 0
    normalized_cursor = cursor % older_count
    older_indices = [
        (normalized_cursor + offset) % older_count
        for offset in range(older_count)
    ]
    return recent_indices + older_indices, len(recent_indices), older_count


def _worker_launched_gpu_indices(
    manifest: Mapping[str, object],
    *,
    allowed_gpu_indices: set[int],
) -> set[int]:
    """Fail closed when an older worker omits its per-launch GPU receipt."""

    raw = manifest.get("launched_gpu_indices")
    launched = {
        int(gpu)
        for gpu in raw
        if isinstance(raw, list)
        and isinstance(gpu, int)
        and not isinstance(gpu, bool)
        and gpu in allowed_gpu_indices
    } if isinstance(raw, list) else set()
    count = manifest.get("launched_count")
    cpu_count = manifest.get("launched_cpu_count", 0)
    gpu_count = (
        count - cpu_count
        if isinstance(count, int)
        and not isinstance(count, bool)
        and isinstance(cpu_count, int)
        and not isinstance(cpu_count, bool)
        else 0
    )
    if gpu_count > 0 and not launched:
        return set(allowed_gpu_indices)
    return launched


def _queue_has_pending_cpu_rows(queue_path: Path) -> bool:
    try:
        payload = json.loads(Path(queue_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    rows = payload.get("rows") if isinstance(payload, Mapping) else None
    if not isinstance(rows, list):
        return False
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        phase = str(row.get("phase") or "")
        resource_class = str(row.get("resource_class") or "")
        if resource_class != "cpu" and phase not in CPU_ONLY_PHASES:
            continue
        output_value = row.get("output_root")
        if isinstance(output_value, str) and output_value and not Path(output_value).exists():
            return True
    return False


def _resolve_seed_start(report_root: Path, requested_seed_start: int) -> int:
    if isinstance(requested_seed_start, bool) or requested_seed_start < 0:
        raise ValueError("ACWM_AUTOLOOP_DAEMON_SEED_START_INVALID")
    highest = requested_seed_start - 1
    try:
        entries = Path(report_root).iterdir()
        for entry in entries:
            match = _CAMPAIGN_SEED.search(entry.name)
            if match is not None:
                highest = max(highest, int(match.group(1)))
    except OSError as error:
        raise ValueError(f"ACWM_AUTOLOOP_DAEMON_REPORT_ROOT_UNREADABLE:{error}") from error
    return highest + 1


def run_daemon(
    *,
    queues: Sequence[Path],
    output_root: Path,
    poll_seconds: float = 120.0,
    max_cycles: int = 720,
    replenisher: Replenisher | None = None,
    quality_discovery_only: bool = False,
    promote_current_contract_queues: bool = False,
    report_root: Path | None = None,
    frontier_refresher: FrontierRefresher | None = None,
    static_queue_scan_budget_seconds: float = _DEFAULT_STATIC_QUEUE_SCAN_BUDGET_SECONDS,
    static_recent_queue_count: int = _DEFAULT_STATIC_RECENT_QUEUE_COUNT,
) -> None:
    if (
        not queues
        or poll_seconds < 1.0
        or max_cycles < 1
        or static_queue_scan_budget_seconds <= 0.0
        or static_recent_queue_count < 1
    ):
        raise ValueError("ACWM_AUTOLOOP_DAEMON_ARGUMENT_INVALID")
    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise ValueError("ACWM_AUTOLOOP_DAEMON_OUTPUT_EXISTS")
    destination.mkdir(mode=0o700, parents=True)
    queue_paths = [Path(path).resolve(strict=True) for path in queues]
    state = {
        "schema_version": 1,
        "artifact_type": "wmloop-acwm-autoloop-daemon",
        "state": "running",
        "pid": os.getpid(),
        "queues": [str(path) for path in queue_paths],
        "poll_seconds": poll_seconds,
        "max_cycles": max_cycles,
        "started_at": _utc_now(),
        "cycle": 0,
        "launch_count": 0,
        "replenishment_count": 0,
        "static_queue_cursor": 0,
        "static_queue_scan_budget_seconds": static_queue_scan_budget_seconds,
        "static_recent_queue_count": static_recent_queue_count,
        "quality_discovery_only": quality_discovery_only,
        "promote_current_contract_queues": promote_current_contract_queues,
        "quality_gate_inventory_path": str(destination / "quality-gates.json"),
        "metric_pass_retention_path": str(destination / "metric-pass-retention.json"),
        "environment_coverage_path": str(destination / "environment-coverage.json"),
        "candidate_frontier_manifest_path": None,
        "official_quality_gate_count": 0,
        "official_quality_gate_pass_count": 0,
        "metric_pass_retained_count": 0,
    }
    _refresh_quality_gate_inventory(
        destination=destination,
        queues=queue_paths,
        state=state,
        report_root=report_root,
    )
    _refresh_environment_coverage(
        destination=destination,
        queues=queue_paths,
        state=state,
        report_root=report_root,
    )
    _write_json(destination / "status.json", state)
    stop = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    try:
        for cycle in range(1, max_cycles + 1):
            if stop:
                break
            idle_before = sorted(_idle_gpu_indices())
            cycle_available_gpus = set(idle_before)
            cycle_record: dict[str, object] = {
                "cycle": cycle,
                "timestamp": _utc_now(),
                "idle_gpus_before": idle_before,
                "workers": [],
            }
            static_launch_count = 0
            launched_static_queues: list[tuple[int, Path]] = []
            pending_cpu_queues = {
                path for path in queue_paths if _queue_has_pending_cpu_rows(path)
            }
            if idle_before or pending_cpu_queues:
                scan_started = time.monotonic()
                scan_indices, recent_priority_count, older_queue_count = _static_queue_scan_indices(
                    len(queue_paths),
                    cursor=int(state["static_queue_cursor"]),
                    recent_count=static_recent_queue_count,
                )
                static_scan_count = 0
                static_scan_budget_exhausted = False
                for scan_position, queue_offset in enumerate(scan_indices):
                    if (
                        scan_position >= recent_priority_count
                        and time.monotonic() - scan_started >= static_queue_scan_budget_seconds
                    ):
                        static_scan_budget_exhausted = True
                        break
                    queue = queue_paths[queue_offset]
                    if not cycle_available_gpus and queue not in pending_cpu_queues:
                        static_scan_count += 1
                        if queue_offset < older_queue_count:
                            state["static_queue_cursor"] = (queue_offset + 1) % older_queue_count
                        continue
                    queue_index = queue_offset + 1
                    worker_root = destination / "workers" / f"cycle-{cycle:04d}-queue-{queue_index:02d}"
                    worker_allowed_gpus = set(cycle_available_gpus)
                    try:
                        manifest = run_autoloop_worker(
                            queue_path=queue,
                            output_root=worker_root,
                            max_launches=1,
                            iterations=1,
                            sleep_seconds=0,
                            phase_allowlist=_phase_allowlist_for_queue(
                                queue,
                                quality_discovery_only=quality_discovery_only,
                                promote_current_contract_queues=promote_current_contract_queues,
                            ),
                            allowed_gpu_indices=worker_allowed_gpus,
                        )
                        cycle_record["workers"].append(manifest)  # type: ignore[union-attr]
                        cycle_available_gpus -= _worker_launched_gpu_indices(
                            manifest,
                            allowed_gpu_indices=worker_allowed_gpus,
                        )
                        launched = int(manifest.get("launched_count", 0))
                        static_launch_count += launched
                        state["launch_count"] = int(state["launch_count"]) + launched
                        if launched > 0:
                            launched_static_queues.append((queue_index, queue))
                    except Exception as error:
                        cycle_record["workers"].append(  # type: ignore[union-attr]
                            {"state": "failed", "queue": str(queue), "error": f"{type(error).__name__}:{error}"}
                        )
                    finally:
                        static_scan_count += 1
                        if queue_offset < older_queue_count:
                            state["static_queue_cursor"] = (queue_offset + 1) % older_queue_count
                cycle_record["static_queue_scan_count"] = static_scan_count
                cycle_record["static_queue_scan_budget_exhausted"] = static_scan_budget_exhausted
                cycle_record["static_queue_scan_elapsed_seconds"] = round(
                    time.monotonic() - scan_started,
                    6,
                )
            idle_after_static = sorted(_idle_gpu_indices() & cycle_available_gpus)
            if static_launch_count > 0 and idle_after_static:
                for queue_index, queue in launched_static_queues:
                    if not cycle_available_gpus:
                        break
                    worker_root = (
                        destination
                        / "workers"
                        / f"cycle-{cycle:04d}-static-retry-queue-{queue_index:02d}"
                    )
                    worker_allowed_gpus = set(cycle_available_gpus)
                    try:
                        manifest = run_autoloop_worker(
                            queue_path=queue,
                            output_root=worker_root,
                            max_launches=1,
                            iterations=1,
                            sleep_seconds=0,
                            phase_allowlist=_phase_allowlist_for_queue(
                                queue,
                                quality_discovery_only=quality_discovery_only,
                                promote_current_contract_queues=promote_current_contract_queues,
                            ),
                            allowed_gpu_indices=worker_allowed_gpus,
                        )
                        cycle_record["workers"].append(manifest)  # type: ignore[union-attr]
                        cycle_available_gpus -= _worker_launched_gpu_indices(
                            manifest,
                            allowed_gpu_indices=worker_allowed_gpus,
                        )
                        state["launch_count"] = int(state["launch_count"]) + int(
                            manifest.get("launched_count", 0)
                        )
                    except Exception as error:
                        cycle_record["workers"].append(  # type: ignore[union-attr]
                            {"state": "failed", "queue": str(queue), "error": f"{type(error).__name__}:{error}"}
                        )
                idle_after_static = sorted(_idle_gpu_indices() & cycle_available_gpus)
            if replenisher is not None and idle_after_static:
                cycle_record["replenishments"] = []
                for gpu in idle_after_static:
                    sequence = int(state["replenishment_count"]) + 1
                    try:
                        result = dict(replenisher(gpu, sequence))
                        cycle_record["replenishments"].append(result)  # type: ignore[union-attr]
                        state["replenishment_count"] = sequence
                        queue_value = result.get("queue_path")
                        if result.get("state") != "ready" or not isinstance(queue_value, str):
                            continue
                        queue = Path(queue_value).resolve(strict=True)
                        if queue not in queue_paths:
                            queue_paths.append(queue)
                            state["queues"] = [str(path) for path in queue_paths]
                        worker_root = destination / "workers" / f"cycle-{cycle:04d}-replenishment-{sequence:04d}"
                        manifest = run_autoloop_worker(
                            queue_path=queue,
                            output_root=worker_root,
                            max_launches=1,
                            iterations=1,
                            sleep_seconds=0,
                            phase_allowlist=_phase_allowlist_for_queue(
                                queue,
                                quality_discovery_only=quality_discovery_only,
                                promote_current_contract_queues=promote_current_contract_queues,
                            ),
                            allowed_gpu_indices={gpu},
                        )
                        cycle_record["workers"].append(manifest)  # type: ignore[union-attr]
                        cycle_available_gpus -= _worker_launched_gpu_indices(
                            manifest,
                            allowed_gpu_indices={gpu},
                        )
                        state["launch_count"] = int(state["launch_count"]) + int(manifest.get("launched_count", 0))
                    except Exception as error:
                        cycle_record["replenishments"].append(  # type: ignore[union-attr]
                            {"state": "failed", "gpu": gpu, "error": f"{type(error).__name__}:{error}"}
                        )
            cycle_record["idle_gpus_after"] = sorted(
                _idle_gpu_indices() & cycle_available_gpus
            )
            inventory = _refresh_quality_gate_inventory(
                destination=destination,
                queues=queue_paths,
                state=state,
                report_root=report_root,
            )
            cycle_record["official_quality_gate_count"] = inventory["gate_count"]
            cycle_record["official_quality_gate_pass_count"] = inventory["pass_count"]
            if (
                cycle == 1
                or cycle % _COVERAGE_REFRESH_INTERVAL_CYCLES == 0
                or static_launch_count > 0
                or "replenishments" in cycle_record
            ):
                coverage = _refresh_environment_coverage(
                    destination=destination,
                    queues=queue_paths,
                    state=state,
                    report_root=report_root,
                )
                if coverage is not None:
                    cycle_record["formally_confirmed_environment_count"] = coverage["summary"][
                        "formally_confirmed_environment_count"
                    ]
                if frontier_refresher is not None:
                    try:
                        frontier = dict(frontier_refresher(cycle))
                        cycle_record["candidate_frontier"] = frontier
                        state["candidate_frontier_manifest_path"] = frontier.get("manifest_path")
                        state["candidate_frontier_summary"] = frontier.get("summary")
                        state.pop("candidate_frontier_error", None)
                    except Exception as error:
                        failure = f"{type(error).__name__}:{error}"
                        cycle_record["candidate_frontier"] = {"state": "failed", "error": failure}
                        state["candidate_frontier_error"] = failure
            _append_jsonl(destination / "cycles.jsonl", cycle_record)
            state.update({"cycle": cycle, "updated_at": _utc_now(), "idle_gpus": cycle_record["idle_gpus_after"]})
            _write_json(destination / "status.json", state)
            if cycle < max_cycles and not stop:
                time.sleep(poll_seconds)
    finally:
        _refresh_quality_gate_inventory(
            destination=destination,
            queues=queue_paths,
            state=state,
            report_root=report_root,
        )
        _refresh_environment_coverage(
            destination=destination,
            queues=queue_paths,
            state=state,
            report_root=report_root,
        )
        state.update({"state": "stopped" if stop else "completed", "completed_at": _utc_now()})
        _write_json(destination / "status.json", state)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: object) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")


def _refresh_quality_gate_inventory(
    *,
    destination: Path,
    queues: Sequence[Path],
    state: dict[str, object],
    report_root: Path | None = None,
) -> dict[str, object]:
    inventory = _collect_quality_gate_inventory(queues, report_root=report_root)
    _write_json(destination / "quality-gates.json", inventory)
    retention = _metric_pass_retention_inventory(inventory)
    _write_json(destination / "metric-pass-retention.json", retention)
    state["official_quality_gate_count"] = inventory["gate_count"]
    state["official_quality_gate_pass_count"] = inventory["pass_count"]
    state["official_quality_gate_fail_count"] = inventory["fail_count"]
    state["metric_pass_retained_count"] = retention["record_count"]
    state["metric_pass_effective_method_count"] = retention["effective_method_count"]
    state["metric_pass_invalidated_audit_count"] = retention["invalidated_method_audit_count"]
    state["metric_pass_confirmed_count"] = retention["confirmed_record_count"]
    state["metric_pass_pending_confirmation_count"] = retention["pending_confirmation_record_count"]
    return inventory


def _refresh_environment_coverage(
    *,
    destination: Path,
    queues: Sequence[Path],
    state: dict[str, object],
    report_root: Path | None,
) -> dict[str, object] | None:
    if report_root is None:
        return None
    coverage = build_live_coverage(
        report_root=Path(report_root),
        active_queue_paths={Path(path).resolve() for path in queues},
    )
    _write_json(destination / "environment-coverage.json", coverage)
    summary = coverage["summary"]
    if isinstance(summary, Mapping):
        state["formally_confirmed_environment_count"] = summary.get(
            "formally_confirmed_environment_count", 0
        )
        state["environment_count"] = summary.get("environment_count", 0)
        state["environment_coverage_complete"] = summary.get("coverage_complete", False)
        state["environment_coverage_updated_at"] = coverage.get("created_at")
    return coverage


def _metric_pass_retention_inventory(inventory: Mapping[str, object]) -> dict[str, object]:
    raw_records = inventory.get("records")
    records = [
        dict(record)
        for record in raw_records
        if isinstance(record, Mapping) and record.get("pass") is True
    ] if isinstance(raw_records, list) else []
    effective = 0
    invalidated = 0
    confirmed = 0
    pending_confirmation = 0
    for record in records:
        retention = record.get("retention")
        method_invalidated = isinstance(retention, Mapping) and retention.get("method_invalidated") is True
        if method_invalidated:
            invalidated += 1
        else:
            effective += 1
        claim_tier = str(retention.get("claim_tier") or "") if isinstance(retention, Mapping) else ""
        if claim_tier.startswith("A_"):
            confirmed += 1
        elif claim_tier.startswith("B_"):
            pending_confirmation += 1
    return {
        "schema_version": 1,
        "artifact_type": "wmloop-acwm-metric-pass-retention-inventory",
        "state": "ready",
        "generated_at": str(inventory.get("generated_at") or _utc_now()),
        "source_artifact_type": str(inventory.get("artifact_type") or ""),
        "record_count": len(records),
        "effective_method_count": effective,
        "invalidated_method_audit_count": invalidated,
        "confirmed_record_count": confirmed,
        "pending_confirmation_record_count": pending_confirmation,
        "retention_policy": {
            "retain_every_official_gate_pass": True,
            "visual_prominence_required_for_retention": False,
            "tier_a_use": "method_specific_confirmation_satisfied_and_claim_eligible",
            "tier_b_use": "quantitative_fallback_and_visual_candidate_pending_human_review",
            "tier_c_use": "invalidated_method_implementation_audit_only",
            "gt_injection_allowed": False,
            "claim_boundary": "Only the official 50-step PSNR/SSIM/MSE/masked-MSE gate establishes a retained metric pass.",
        },
        "records": records,
    }


def _collect_quality_gate_inventory(
    queues: Sequence[Path], *, report_root: Path | None = None
) -> dict[str, object]:
    records_by_manifest: dict[str, dict[str, object]] = {}
    for queue_path in queues:
        try:
            queue = json.loads(Path(queue_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        rows = queue.get("rows") if isinstance(queue, Mapping) else None
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, Mapping) or row.get("phase") not in {
                "official_eval_gate",
                "confirm_official_eval_gate",
            }:
                continue
            output_root = row.get("output_root")
            if not isinstance(output_root, str) or not output_root:
                continue
            manifest_path = Path(output_root).resolve() / "manifest.json"
            if manifest_path.is_symlink() or not manifest_path.is_file():
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            gate = manifest.get("official_quality_gate") if isinstance(manifest, Mapping) else None
            if not isinstance(gate, Mapping) or not isinstance(gate.get("pass"), bool):
                continue
            manifest_key = str(manifest_path)
            records_by_manifest[manifest_key] = {
                "environment": str(manifest.get("environment") or row.get("environment") or ""),
                "primitive": str(manifest.get("primitive") or row.get("primitive") or ""),
                "seed": int(manifest.get("seed") or row.get("seed") or 0),
                "phase": str(row.get("phase") or ""),
                "checkpoint_step": row.get("checkpoint_step"),
                "pass": bool(gate["pass"]),
                "delta_candidate_minus_baseline": dict(gate.get("delta_candidate_minus_baseline") or {}),
                "candidate_checkpoint_sha256": manifest.get("candidate_checkpoint_sha256"),
                "candidate_runtime_sha256": manifest.get("candidate_runtime_sha256"),
                "execution_mode": manifest.get("execution_mode"),
                "runtime_parameters": manifest.get("runtime_parameters"),
                "runtime_effect_gate": manifest.get("runtime_effect_gate"),
                "candidate_runtime_hook_receipt": manifest.get("candidate_runtime_hook_receipt"),
                "checkpoint_transform_provenance": manifest.get("checkpoint_transform_provenance"),
                "manifest_path": manifest_key,
                "manifest_sha256": _sha256(manifest_path),
                "retention": _retention_evidence(manifest, gate),
            }
    if report_root is not None:
        root = Path(report_root).resolve()
        gate_paths = {
            path.resolve()
            for pattern in (
                "acwm-autoloop-official-gate-*/manifest.json",
                "acwm-official-gate-*/manifest.json",
                "acwm-autoloop-confirm-official-gate-*/manifest.json",
            )
            for path in root.glob(pattern)
        }
        for manifest_path in sorted(gate_paths):
            if manifest_path.is_symlink() or not manifest_path.is_file():
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(manifest, Mapping) or manifest.get("state") != "ready":
                continue
            gate = manifest.get("official_quality_gate")
            if not isinstance(gate, Mapping) or not isinstance(gate.get("pass"), bool):
                continue
            discovered_record = {
                "environment": str(manifest.get("environment") or ""),
                "primitive": str(manifest.get("primitive") or ""),
                "seed": int(manifest.get("seed") or manifest.get("eval_seed") or 0),
                "phase": "confirm_official_eval_gate"
                if "confirm-official-gate" in manifest_path.parent.name
                else "official_eval_gate",
                "checkpoint_step": _gate_checkpoint_step(manifest_path.parent.name, manifest),
                "pass": bool(gate["pass"]),
                "delta_candidate_minus_baseline": dict(gate.get("delta_candidate_minus_baseline") or {}),
                "candidate_checkpoint_sha256": manifest.get("candidate_checkpoint_sha256"),
                "candidate_runtime_sha256": manifest.get("candidate_runtime_sha256"),
                "execution_mode": manifest.get("execution_mode"),
                "runtime_parameters": manifest.get("runtime_parameters"),
                "runtime_effect_gate": manifest.get("runtime_effect_gate"),
                "candidate_runtime_hook_receipt": manifest.get("candidate_runtime_hook_receipt"),
                "checkpoint_transform_provenance": manifest.get("checkpoint_transform_provenance"),
                "manifest_path": str(manifest_path),
                "manifest_sha256": _sha256(manifest_path),
                "retention": _retention_evidence(manifest, gate),
            }
            records_by_manifest.setdefault(str(manifest_path), discovered_record)
    records = [records_by_manifest[key] for key in sorted(records_by_manifest)]
    _annotate_confirmation_tiers(records)
    pass_count = sum(record["pass"] is True for record in records)
    return {
        "schema_version": 1,
        "artifact_type": "wmloop-acwm-official-quality-gate-inventory",
        "state": "ready",
        "generated_at": _utc_now(),
        "gate_count": len(records),
        "pass_count": pass_count,
        "fail_count": len(records) - pass_count,
        "records": records,
    }


def _gate_checkpoint_step(name: str, manifest: Mapping[str, object]) -> object:
    match = _CONFIRM_OFFICIAL_STEP.search(name)
    if match is not None:
        return int(match.group(1))
    checkpoint_step = manifest.get("checkpoint_step")
    if isinstance(checkpoint_step, int) and not isinstance(checkpoint_step, bool):
        return checkpoint_step
    candidate_checkpoint = manifest.get("candidate_checkpoint")
    if isinstance(candidate_checkpoint, str):
        screen_match = _SCREEN_CHECKPOINT_STEP.search(candidate_checkpoint)
        if screen_match is not None:
            return int(screen_match.group(1))
    if manifest.get("execution_mode") == "runtime_only":
        return 0
    return None


def _annotate_confirmation_tiers(records: list[dict[str, object]]) -> None:
    runtime_passes: dict[tuple[str, str, str], set[int]] = {}
    checkpoint_transform_passes: dict[tuple[str, str, str, str], set[int]] = {}
    for record in records:
        parameters = record.get("runtime_parameters")
        effect_gate = record.get("runtime_effect_gate")
        receipt = record.get("candidate_runtime_hook_receipt")
        if (
            record.get("pass") is not True
            or record.get("execution_mode") != "runtime_only"
            or not isinstance(parameters, Mapping)
            or not isinstance(effect_gate, Mapping)
            or effect_gate.get("pass") is not True
            or not isinstance(receipt, Mapping)
            or receipt.get("state") != "ready"
            or not isinstance(receipt.get("call_count"), int)
            or int(receipt["call_count"]) < 1
        ):
            continue
        key = (
            str(record.get("environment") or ""),
            str(record.get("primitive") or ""),
            json.dumps(dict(parameters), ensure_ascii=True, sort_keys=True, separators=(",", ":")),
        )
        seed = record.get("seed")
        if isinstance(seed, int) and not isinstance(seed, bool):
            runtime_passes.setdefault(key, set()).add(seed)
    for record in records:
        provenance = record.get("checkpoint_transform_provenance")
        seed = record.get("seed")
        if (
            record.get("pass") is not True
            or record.get("primitive") != "checkpoint_delta_scaling"
            or not isinstance(provenance, Mapping)
            or provenance.get("state") != "verified"
            or isinstance(seed, bool)
            or not isinstance(seed, int)
        ):
            continue
        key = (
            str(record.get("environment") or ""),
            str(provenance.get("source_primitive") or ""),
            str(provenance.get("scaled_checkpoint_sha256") or ""),
            str(provenance.get("alpha")),
        )
        checkpoint_transform_passes.setdefault(key, set()).add(seed)

    for record in records:
        retention = record.get("retention")
        if not isinstance(retention, dict) or record.get("pass") is not True:
            continue
        if retention.get("method_invalidated") is True:
            retention["confirmation_state"] = "invalidated_audit_only"
            continue
        primitive = str(record.get("primitive") or "")
        if primitive == "checkpoint_delta_scaling":
            provenance = record.get("checkpoint_transform_provenance")
            if not isinstance(provenance, Mapping) or provenance.get("state") != "verified":
                retention["claim_tier"] = "C_checkpoint_transform_provenance_incomplete"
                retention["confirmation_state"] = "checkpoint_transform_provenance_incomplete"
                continue
            key = (
                str(record.get("environment") or ""),
                str(provenance.get("source_primitive") or ""),
                str(provenance.get("scaled_checkpoint_sha256") or ""),
                str(provenance.get("alpha")),
            )
            pass_count = len(checkpoint_transform_passes.get(key, set()))
            confirmed = pass_count >= 2
            retention.update(
                {
                    "claim_tier": (
                        "A_checkpoint_transform_multiseed_confirmed"
                        if confirmed
                        else "B_checkpoint_transform_single_seed_confirmation_pending"
                    ),
                    "confirmation_state": "confirmed" if confirmed else "pending_independent_seed",
                    "confirmation_pass_count": pass_count,
                    "confirmation_required_passes": 2,
                }
            )
            continue
        if record.get("execution_mode") == "runtime_only":
            parameters = record.get("runtime_parameters")
            effect_gate = record.get("runtime_effect_gate")
            receipt = record.get("candidate_runtime_hook_receipt")
            if (
                not isinstance(parameters, Mapping)
                or not isinstance(effect_gate, Mapping)
                or effect_gate.get("pass") is not True
                or not isinstance(receipt, Mapping)
                or receipt.get("state") != "ready"
                or not isinstance(receipt.get("call_count"), int)
                or int(receipt["call_count"]) < 1
            ):
                retention["claim_tier"] = "C_runtime_materialization_evidence_incomplete"
                retention["confirmation_state"] = "runtime_materialization_evidence_incomplete"
                continue
            key = (
                str(record.get("environment") or ""),
                primitive,
                json.dumps(dict(parameters), ensure_ascii=True, sort_keys=True, separators=(",", ":")),
            )
            pass_count = len(runtime_passes.get(key, set()))
            confirmed = pass_count >= 2
            retention.update(
                {
                    "claim_tier": (
                        "A_runtime_multiseed_confirmed"
                        if confirmed
                        else "B_runtime_single_seed_confirmation_pending"
                    ),
                    "confirmation_state": "confirmed" if confirmed else "pending_independent_seed",
                    "confirmation_pass_count": pass_count,
                    "confirmation_required_passes": 2,
                }
            )
            continue
        checkpoint_step = record.get("checkpoint_step")
        if (
            record.get("phase") == "confirm_official_eval_gate"
            and isinstance(checkpoint_step, int)
            and not isinstance(checkpoint_step, bool)
            and checkpoint_step in {800, 1000}
        ):
            retention.update(
                {
                    "claim_tier": "A_training_1k_checkpoint_confirmed",
                    "confirmation_state": "confirmed",
                    "confirmation_checkpoint_step": checkpoint_step,
                }
            )
            continue
        retention["claim_tier"] = "B_gate_pass_confirmation_pending"
        retention["confirmation_state"] = "pending_method_specific_confirmation"


def _retention_evidence(
    manifest: Mapping[str, object], gate: Mapping[str, object]
) -> dict[str, object]:
    checkpoint = manifest.get("candidate_checkpoint")
    runtime = manifest.get("candidate_runtime_root")
    paired_videos = manifest.get("paired_videos")
    video_paths: list[str] = []
    if isinstance(paired_videos, list):
        video_paths = [str(path) for path in paired_videos if isinstance(path, str)]
    hard_case = manifest.get("hard_case_visualization")
    if isinstance(hard_case, Mapping):
        selected = hard_case.get("selected")
        if isinstance(selected, list):
            for item in selected:
                if not isinstance(item, Mapping):
                    continue
                path = item.get("paired_video_path")
                if isinstance(path, str):
                    video_paths.append(path)
    video_paths = list(dict.fromkeys(video_paths))
    checkpoint_path = str(checkpoint) if isinstance(checkpoint, str) else None
    runtime_path = str(runtime) if isinstance(runtime, str) else None
    primitive = str(manifest.get("primitive") or "")
    invalidated = primitive in INVALIDATED_QUALITY_PRIMITIVES
    gate_pass = gate.get("pass") is True
    return {
        "class": (
            "official_gate_pass_method_invalidated_retained"
            if gate_pass and invalidated
            else "official_gate_pass_metric_retained"
            if gate_pass
            else "official_gate_failure_audit_retained"
        ),
        "claim_tier": (
            "C_gate_pass_method_invalidated_audit_only"
            if gate_pass and invalidated
            else "B_gate_pass_visual_pending"
            if gate_pass
            else "D_gate_failed_audit_only"
        ),
        "method_invalidated": invalidated,
        "method_invalidation_reason": (
            "auxiliary_loss_uses_detached_latents_and_does_not_update_trainable_model_parameters"
            if invalidated
            else None
        ),
        "candidate_checkpoint_path": checkpoint_path,
        "candidate_checkpoint_exists": bool(checkpoint_path and Path(checkpoint_path).is_file()),
        "candidate_runtime_root": runtime_path,
        "candidate_runtime_exists": bool(runtime_path and Path(runtime_path).exists()),
        "paired_video_paths": video_paths,
        "paired_video_exists": [Path(path).is_file() for path in video_paths],
        "metric_gate_pass": gate_pass,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _phase_allowlist_for_queue(
    queue_path: Path,
    *,
    quality_discovery_only: bool,
    promote_current_contract_queues: bool,
) -> set[str] | None:
    if not quality_discovery_only:
        return None
    if _has_post_confirmation_evidence_contract(queue_path):
        return POST_CONFIRMATION_EVIDENCE_PHASES
    if promote_current_contract_queues and _has_current_confirmation_contract(queue_path):
        return None
    return QUALITY_DISCOVERY_PHASES


def _has_post_confirmation_evidence_contract(queue_path: Path) -> bool:
    """Admit evidence-only phases after two independent official gate passes."""

    try:
        payload = json.loads(Path(queue_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(payload, Mapping):
        return False
    contract = payload.get("post_confirmation_evidence_contract")
    rows = payload.get("rows")
    if (
        not isinstance(contract, Mapping)
        or contract.get("state") != "verified"
        or contract.get("kind") != "multi_seed_checkpoint_delta_long_horizon_evidence"
        or not isinstance(rows, list)
        or not rows
    ):
        return False
    phases = {
        str(row.get("phase") or "")
        for row in rows
        if isinstance(row, Mapping)
    }
    if not phases or not phases.issubset(POST_CONFIRMATION_EVIDENCE_PHASES):
        return False
    pass_manifests = contract.get("pass_manifests")
    expected_sha = str(contract.get("candidate_checkpoint_sha256") or "")
    if not isinstance(pass_manifests, list) or len(pass_manifests) < 2 or not expected_sha:
        return False
    seeds: set[int] = set()
    for raw_path in pass_manifests:
        if not isinstance(raw_path, str) or not raw_path:
            return False
        try:
            manifest = json.loads(Path(raw_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        if not isinstance(manifest, Mapping):
            return False
        gate = manifest.get("official_quality_gate")
        seed = manifest.get("eval_seed", manifest.get("seed"))
        if (
            manifest.get("state") != "ready"
            or manifest.get("primitive") != "checkpoint_delta_scaling"
            or manifest.get("candidate_checkpoint_sha256") != expected_sha
            or not isinstance(gate, Mapping)
            or gate.get("pass") is not True
            or isinstance(seed, bool)
            or not isinstance(seed, int)
        ):
            return False
        seeds.add(seed)
    return len(seeds) >= 2


def _has_current_confirmation_contract(queue_path: Path) -> bool:
    """Admit full execution only for the frozen 1k checkpoint-ladder contract."""

    try:
        payload = json.loads(Path(queue_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(payload, Mapping):
        return False
    confirmation_steps = payload.get("confirmation_steps")
    gate_count = payload.get("confirmation_official_gate_row_count")
    if (
        isinstance(confirmation_steps, bool)
        or not isinstance(confirmation_steps, int)
        or confirmation_steps < DEFAULT_CONFIRMATION_STEPS
        or isinstance(gate_count, bool)
        or not isinstance(gate_count, int)
        or gate_count < 2
    ):
        return False
    rows = payload.get("rows")
    if not isinstance(rows, list):
        return False
    confirmation_rows = [
        row for row in rows if isinstance(row, Mapping) and row.get("phase") == "confirm_staged"
    ]
    if len(confirmation_rows) != 1:
        return False
    confirmation = confirmation_rows[0]
    argv = confirmation.get("launch_argv_template")
    train_steps = confirmation.get("train_steps")
    if (
        isinstance(train_steps, bool)
        or not isinstance(train_steps, int)
        or train_steps < DEFAULT_CONFIRMATION_STEPS
        or not isinstance(argv, list)
    ):
        return False
    ladder_steps = {
        int(row["checkpoint_step"])
        for row in rows
        if isinstance(row, Mapping)
        and row.get("phase") == "confirm_official_eval_gate"
        and isinstance(row.get("checkpoint_step"), int)
        and not isinstance(row.get("checkpoint_step"), bool)
    }
    return {800, 1000}.issubset(ladder_steps)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", nargs="?")
    parser.add_argument("--queue", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=120.0)
    parser.add_argument("--max-cycles", type=int, default=720)
    parser.add_argument("--auto-replenish", action="store_true")
    parser.add_argument(
        "--staging-plan",
        type=Path,
        default=Path("results/reports/acwm-gap-driven-staging-plan-r1/acwm-gap-driven-staging-plan.json"),
    )
    parser.add_argument(
        "--materialization-gate",
        type=Path,
        default=Path("results/reports/primitive-materialization-gate-expanded-gpu-r1/manifest.json"),
    )
    parser.add_argument("--report-root", type=Path, default=Path("results/reports"))
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--runtime-python", type=Path, default=DEFAULT_RUNTIME_PYTHON)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--seed-start", type=int, default=900)
    parser.add_argument(
        "--static-queue-scan-budget-seconds",
        type=float,
        default=_DEFAULT_STATIC_QUEUE_SCAN_BUDGET_SECONDS,
        help="Maximum wall time spent scanning older static queues before replenishing idle GPUs.",
    )
    parser.add_argument(
        "--static-recent-queue-count",
        type=int,
        default=_DEFAULT_STATIC_RECENT_QUEUE_COUNT,
        help="Newest queues that are always checked before the bounded older-queue scan.",
    )
    parser.add_argument("--quality-discovery-only", action="store_true")
    parser.add_argument(
        "--promote-current-contract-queues",
        action="store_true",
        help="Allow full execution only for queues that prove the current 1k checkpoint-ladder contract.",
    )
    args = parser.parse_args(argv)
    resolved_seed_start = _resolve_seed_start(
        args.report_root.resolve(strict=True),
        args.seed_start,
    )
    dynamic_replenisher = None
    dynamic_frontier_refresher = None
    if args.auto_replenish:
        def dynamic_replenisher(gpu: int, sequence: int) -> Mapping[str, object]:
            return replenish_autoloop_queue(
                staging_plan=args.staging_plan.resolve(strict=True),
                materialization_gate=args.materialization_gate.resolve(strict=True),
                report_root=args.report_root.resolve(),
                output_root=args.output_root.resolve() / "replenishments" / f"candidate-{sequence:04d}-gpu{gpu}",
                gpu=gpu,
                seed=resolved_seed_start + sequence - 1,
                repo_root=args.repo_root.resolve(strict=True),
                runtime_python=args.runtime_python.resolve(strict=True),
                data_root=args.data_root.resolve(strict=True),
                checkpoint_root=args.checkpoint_root.resolve(strict=True),
                quality_discovery_only=args.quality_discovery_only,
                promote_current_contract_queues=args.promote_current_contract_queues,
            )
        def dynamic_frontier_refresher(cycle: int) -> Mapping[str, object]:
            output = args.output_root.resolve() / "candidate-frontiers" / f"cycle-{cycle:04d}"
            manifest = build_candidate_frontier(
                staging_plan=args.staging_plan.resolve(strict=True),
                materialization_gate=args.materialization_gate.resolve(strict=True),
                report_root=args.report_root.resolve(),
                output_root=output,
                repo_root=args.repo_root.resolve(strict=True),
                quality_discovery_only=args.quality_discovery_only,
            )
            return {**manifest, "manifest_path": str(output / "manifest.json")}
    run_daemon(
        queues=args.queue,
        output_root=args.output_root,
        poll_seconds=args.poll_seconds,
        max_cycles=args.max_cycles,
        replenisher=dynamic_replenisher,
        quality_discovery_only=args.quality_discovery_only,
        promote_current_contract_queues=args.promote_current_contract_queues,
        report_root=args.report_root.resolve(),
        frontier_refresher=dynamic_frontier_refresher,
        static_queue_scan_budget_seconds=args.static_queue_scan_budget_seconds,
        static_recent_queue_count=args.static_recent_queue_count,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
