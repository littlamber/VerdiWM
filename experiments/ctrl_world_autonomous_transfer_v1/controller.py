#!/usr/bin/env python3
"""Durable autonomous discovery-to-knowledge loop for Ctrl-World transfers."""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import datetime as dt
import fcntl
import hashlib
import json
import os
import signal
import socket
import sys
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.ctrl_world_autonomous_transfer_v1 import workflow  # noqa: E402
from experiments.ctrl_world_autonomous_transfer_v1.state import (  # noqa: E402
    DurableLoopStore,
)
from wmloop.execute.gpu_lease import GpuLeaseError  # noqa: E402


class AutonomousTransferControllerError(RuntimeError):
    """The durable controller could not preserve its ownership boundary."""


@dataclasses.dataclass(frozen=True)
class ControllerDrivers:
    discovery: Callable[..., dict[str, object]] = workflow.run_discovery
    portrait_gate: Callable[..., workflow.StageResult] = workflow.evaluate_portrait_gate
    observation_plan: Callable[..., workflow.StageResult] = workflow.plan_observation
    observation_execute: Callable[..., workflow.StageResult] = workflow.execute_observation
    shadow_probe_admission: Callable[..., workflow.StageResult] = workflow.admit_shadow_probe
    gap_plan: Callable[..., workflow.StageResult] = workflow.plan_capability_gaps
    portfolio_plan: Callable[..., workflow.StageResult] = workflow.plan_experiment_portfolio
    materialization: Callable[..., workflow.StageResult] = workflow.materialize
    open_method_calibration: Callable[..., workflow.StageResult] = (
        workflow.calibrate_open_method
    )
    resource_admission: Callable[..., workflow.StageResult] = workflow.admit_screen_resources
    resource_reallocation: Callable[..., workflow.StageResult] = (
        workflow.reallocate_confirm_resources
    )
    gpu_stage: Callable[..., workflow.StageResult] = workflow.run_gpu_stage
    verifier: Callable[..., workflow.StageResult] = workflow.verify
    knowledge: Callable[..., workflow.StageResult] = workflow.rebuild_knowledge_graph
    replan: Callable[..., workflow.StageResult] = workflow.run_closed_loop_replan
    evidence_import: Callable[..., list[dict[str, object]]] = workflow.import_evidence


class ControllerLease:
    """A process lifetime lock; durable state remains in SQLite and receipts."""

    def __init__(self, state_root: Path, *, loop_id: str, config_digest: str) -> None:
        self.state_root = Path(state_root).expanduser().resolve()
        self.loop_id = loop_id
        self.config_digest = config_digest
        self._handle: Any | None = None

    def __enter__(self) -> "ControllerLease":
        self.state_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        lock_path = self.state_root / "controller.lock"
        handle = lock_path.open("a+", encoding="ascii")
        os.chmod(lock_path, 0o600)
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.close()
            raise AutonomousTransferControllerError("AUTONOMOUS_CONTROLLER_ALREADY_RUNNING") from exc
        self._handle = handle
        _write_json_atomic(
            self.state_root / "controller-owner.json",
            {
                "schema_version": 1,
                "artifact_type": "verdiwm-autonomous-controller-owner",
                "state": "held",
                "loop_id": self.loop_id,
                "config_digest": self.config_digest,
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
                "started_at": _utc_now(),
            },
        )
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        handle = self._handle
        if handle is None:
            return
        _write_json_atomic(
            self.state_root / "controller-owner.json",
            {
                "schema_version": 1,
                "artifact_type": "verdiwm-autonomous-controller-owner",
                "state": "released",
                "loop_id": self.loop_id,
                "config_digest": self.config_digest,
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
                "released_at": _utc_now(),
            },
        )
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
            self._handle = None


class ControllerRuntime:
    """Own one durable controller lease and execute resumable scheduling cycles."""

    def __init__(
        self,
        *,
        config: Mapping[str, object],
        state_root: Path,
        drivers: ControllerDrivers | None = None,
    ) -> None:
        self.config = dict(config)
        self.state_root = Path(state_root).expanduser().resolve()
        self.drivers = drivers or ControllerDrivers()
        source = Path(str(self.config["paths"]["ctrl_world_root"])).resolve()  # type: ignore[index]
        if (
            self.state_root == PROJECT_ROOT
            or PROJECT_ROOT in self.state_root.parents
            or self.state_root == source
            or source in self.state_root.parents
        ):
            raise AutonomousTransferControllerError("AUTONOMOUS_STATE_INSIDE_SOURCE")
        self.lease = ControllerLease(
            self.state_root,
            loop_id=str(self.config["loop_id"]),
            config_digest=str(self.config["config_digest"]),
        )
        self.store: DurableLoopStore | None = None

    def __enter__(self) -> "ControllerRuntime":
        self.lease.__enter__()
        try:
            self.store = DurableLoopStore(
                self.state_root / "controller.db",
                loop_id=str(self.config["loop_id"]),
                config_digest=str(self.config["config_digest"]),
            )
            recovered = self.store.recover_interrupted()
            self._write_status("running", {"recovered_work_items": recovered})
            self._import_existing_evidence()
            return self
        except Exception:
            # ``with`` does not call __exit__ when __enter__ itself fails.
            # Release the owner receipt and flock before propagating the fault.
            self.lease.__exit__(*sys.exc_info())
            raise

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        try:
            self._write_status(
                "failed" if exc is not None else "stopped",
                {"last_error": str(exc)[:1000] if exc is not None else None},
            )
        finally:
            self.lease.__exit__(exc_type, exc, traceback)

    def run_cycle(self, *, force_discovery: bool = False) -> dict[str, object]:
        store = self._store()
        cycle_started = _utc_now()
        discovery = None
        if force_discovery or store.discovery_due(
            # Keep unattended research responsive even when an older revision
            # carried a multi-hour discovery interval. The persisted config
            # digest remains unchanged; this is only a scheduler upper bound.
            interval_seconds=min(
                300.0, max(1.0, float(self.config["discovery_interval_seconds"]))
            )
        ):
            discovery = self._run_discovery()

        progress = 0
        for _ in range(16):
            pass_progress = 0
            pass_progress += self._run_portrait_gates()
            pass_progress += self._run_observation_plans()
            pass_progress += self._run_probe_executions()
            pass_progress += self._run_shadow_probe_admissions()
            pass_progress += self._run_gap_plans()
            pass_progress += self._run_portfolios()
            pass_progress += self._run_materializations()
            pass_progress += self._run_open_method_calibrations()
            pass_progress += self._run_resource_admissions()
            pass_progress += self._run_resource_reallocations()
            pass_progress += self._run_gpu_work()
            pass_progress += self._run_verifiers()
            pass_progress += self._run_knowledge_updates()
            pass_progress += self._run_replans()
            progress += pass_progress
            if pass_progress == 0:
                break
        status = store.status()
        result = {
            "schema_version": 1,
            "artifact_type": "verdiwm-autonomous-controller-cycle",
            "state": "idle" if progress == 0 else "progressed",
            "loop_id": self.config["loop_id"],
            "config_digest": self.config["config_digest"],
            "cycle_started_at": cycle_started,
            "cycle_finished_at": _utc_now(),
            "discovery": discovery,
            "stage_transitions": progress,
            "status": status,
        }
        _write_json_atomic(self.state_root / "last-cycle.json", result)
        self._write_status("running", {"last_cycle": result})
        return result

    def _run_discovery(self) -> dict[str, object]:
        store = self._store()
        token = f"{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
        output_root = self.state_root / "discovery" / f"cycle-{token}"
        cycle_id = store.begin_discovery(output_root)
        try:
            manifest = self.drivers.discovery(
                self.config,
                output_root=output_root,
                failure_context=_failure_context(self.state_root, store.status()),
            )
            ingested = store.ingest_intake(
                cycle_id,
                manifest,
                initial_state=(
                    "pending_portrait"
                    if isinstance(self.config.get("portrait_gate"), Mapping)
                    else "pending_materialization"
                ),
            )
            store.finish_discovery(cycle_id, state="completed", manifest=manifest)
            return {"cycle_id": cycle_id, "manifest": manifest, **ingested}
        except Exception as exc:
            store.finish_discovery(
                cycle_id,
                state="failed",
                error=f"{type(exc).__name__}:{str(exc)[:800]}",
            )
            return {
                "cycle_id": cycle_id,
                "state": "failed",
                "error": f"{type(exc).__name__}:{str(exc)[:800]}",
            }

    def _run_portrait_gates(self) -> int:
        progressed = 0
        for raw in self._store().list_work(["pending_portrait"], limit=32):
            work = self._store().claim(raw["work_id"], pending_state="pending_portrait")
            if work is None:
                continue
            self._execute_portrait_gate(work)
            progressed += 1
        return progressed

    def _run_materializations(self) -> int:
        progressed = 0
        for raw in self._store().list_work(["pending_materialization"], limit=32):
            work = self._store().claim(raw["work_id"], pending_state="pending_materialization")
            if work is None:
                continue
            self._execute_materialization(work)
            progressed += 1
        return progressed

    def _run_open_method_calibrations(self) -> int:
        if not isinstance(self.config.get("open_method_generation"), Mapping):
            return 0
        progressed = 0
        for raw in self._store().list_work(
            ["pending_open_method_calibration"], limit=32
        ):
            work = self._store().claim(
                raw["work_id"], pending_state="pending_open_method_calibration"
            )
            if work is None:
                continue
            self._execute_open_method_calibration(work)
            progressed += 1
        return progressed

    def _run_gap_plans(self) -> int:
        progressed = 0
        for raw in self._store().list_work(["pending_gap_planning"], limit=32):
            work = self._store().claim(
                raw["work_id"], pending_state="pending_gap_planning"
            )
            if work is None:
                continue
            self._execute_gap_plan(work)
            progressed += 1
        return progressed

    def _run_portfolios(self) -> int:
        progressed = 0
        for raw in self._store().list_work(["pending_portfolio"], limit=32):
            work = self._store().claim(
                raw["work_id"], pending_state="pending_portfolio"
            )
            if work is None:
                continue
            self._execute_portfolio(work)
            progressed += 1
        return progressed

    def _run_observation_plans(self) -> int:
        if not isinstance(self.config.get("observation_planning"), Mapping):
            return 0
        progressed = 0
        for raw in self._store().list_work(["pending_observation"], limit=32):
            work = self._store().claim(raw["work_id"], pending_state="pending_observation")
            if work is None:
                continue
            self._execute_observation_plan(work)
            progressed += 1
        return progressed

    def _run_probe_executions(self) -> int:
        if not isinstance(self.config.get("observation_execution"), Mapping):
            return 0
        progressed = 0
        for raw in self._store().list_work(["pending_probe_execution"], limit=32):
            work = self._store().claim(
                raw["work_id"], pending_state="pending_probe_execution"
            )
            if work is None:
                continue
            self._execute_probe_execution(work)
            progressed += 1
        return progressed

    def _run_shadow_probe_admissions(self) -> int:
        if not isinstance(self.config.get("observation_execution"), Mapping):
            return 0
        progressed = 0
        for raw in self._store().list_work(
            ["pending_shadow_probe_admission"], limit=32
        ):
            work = self._store().claim(
                raw["work_id"], pending_state="pending_shadow_probe_admission"
            )
            if work is None:
                continue
            self._execute_shadow_probe_admission(work)
            progressed += 1
        return progressed

    def _execute_portrait_gate(self, work: Mapping[str, object]) -> None:
        attempt, root = self._store().begin_stage(
            str(work["work_id"]), "portrait", self._work_root(work) / "portrait"
        )
        try:
            result = self.drivers.portrait_gate(
                self.config, work=work, attempt_root=root
            )
            staging = workflow.stage_portrait_knowledge(
                self.config, state_root=self.state_root
            )
            payload = {**result.payload, **staging}
            self._store().finish_stage(
                str(work["work_id"]),
                "portrait",
                attempt,
                state=result.state,
                payload=payload,
                receipt_path=result.receipt_path,
            )
            next_state = (
                (
                    "pending_materialization"
                    if isinstance(self.config.get("open_method_generation"), Mapping)
                    else "pending_gap_planning"
                )
                if result.outcome == "ready_for_gap_planning"
                else "pending_observation"
            )
            self._store().transition(
                str(work["work_id"]),
                expected_state="running_portrait",
                next_state=next_state,
                context_update=payload,
            )
        except Exception as exc:
            self._handle_stage_exception(work, "portrait", attempt, root, exc)

    def _execute_observation_plan(self, work: Mapping[str, object]) -> None:
        attempt, root = self._store().begin_stage(
            str(work["work_id"]), "observation", self._work_root(work) / "observation"
        )
        try:
            result = self.drivers.observation_plan(
                self.config, work=work, attempt_root=root
            )
            next_state = str(result.payload.get("observation_next_state") or "")
            if next_state not in {
                "pending_probe_execution",
                "pending_shadow_probe_admission",
                "pending_interface_extension",
                "requires_evaluator_binding",
                "missing_data_regime",
                "architecture_bound",
            }:
                raise AutonomousTransferControllerError(
                    "AUTONOMOUS_OBSERVATION_NEXT_STATE_INVALID"
                )
            self._store().finish_stage(
                str(work["work_id"]),
                "observation",
                attempt,
                state=result.state,
                payload=result.payload,
                receipt_path=result.receipt_path,
            )
            self._store().transition(
                str(work["work_id"]),
                expected_state="running_observation",
                next_state=next_state,
                context_update=result.payload,
            )
        except Exception as exc:
            self._handle_stage_exception(work, "observation", attempt, root, exc)

    def _execute_probe_execution(self, work: Mapping[str, object]) -> None:
        attempt, root = self._store().begin_stage(
            str(work["work_id"]),
            "probe_execution",
            self._work_root(work) / "probe-execution",
        )
        try:
            result = self.drivers.observation_execute(
                self.config, work=work, attempt_root=root
            )
            next_state = str(
                result.payload.get("observation_execution_next_state") or ""
            )
            if next_state not in {
                "pending_portrait",
                "observation_execution_blocked",
            }:
                raise AutonomousTransferControllerError(
                    "AUTONOMOUS_OBSERVATION_EXECUTION_NEXT_STATE_INVALID"
                )
            self._store().finish_stage(
                str(work["work_id"]),
                "probe_execution",
                attempt,
                state=result.state,
                payload=result.payload,
                receipt_path=result.receipt_path,
            )
            self._store().transition(
                str(work["work_id"]),
                expected_state="running_probe_execution",
                next_state=next_state,
                context_update=result.payload,
            )
        except Exception as exc:
            self._handle_stage_exception(work, "probe_execution", attempt, root, exc)

    def _execute_shadow_probe_admission(self, work: Mapping[str, object]) -> None:
        attempt, root = self._store().begin_stage(
            str(work["work_id"]),
            "shadow_probe_admission",
            self._work_root(work) / "shadow-probe-admission",
        )
        try:
            result = self.drivers.shadow_probe_admission(
                self.config, work=work, attempt_root=root
            )
            next_state = str(
                result.payload.get("shadow_probe_admission_next_state") or ""
            )
            if next_state not in {
                "shadow_probe_evaluator_binding_required",
                "shadow_probe_admission_blocked",
            }:
                raise AutonomousTransferControllerError(
                    "AUTONOMOUS_SHADOW_PROBE_ADMISSION_NEXT_STATE_INVALID"
                )
            self._store().finish_stage(
                str(work["work_id"]),
                "shadow_probe_admission",
                attempt,
                state=result.state,
                payload=result.payload,
                receipt_path=result.receipt_path,
            )
            self._store().transition(
                str(work["work_id"]),
                expected_state="running_shadow_probe_admission",
                next_state=next_state,
                context_update=result.payload,
            )
        except Exception as exc:
            self._handle_stage_exception(
                work, "shadow_probe_admission", attempt, root, exc
            )

    def _execute_gap_plan(self, work: Mapping[str, object]) -> None:
        attempt, root = self._store().begin_stage(
            str(work["work_id"]),
            "gap_planning",
            self._work_root(work) / "gap-planning",
        )
        try:
            result = self.drivers.gap_plan(
                self.config, work=work, attempt_root=root
            )
            staging = workflow.stage_gap_knowledge(
                self.config, state_root=self.state_root, planning=result
            )
            payload = {**result.payload, **staging}
            next_state = str(result.payload.get("capability_gap_next_state") or "")
            if next_state not in {
                "pending_materialization",
                "pending_portfolio",
                "pending_interface_extension",
                "missing_data_regime",
                "architecture_bound",
            }:
                raise AutonomousTransferControllerError(
                    "AUTONOMOUS_GAP_NEXT_STATE_INVALID"
                )
            self._store().finish_stage(
                str(work["work_id"]),
                "gap_planning",
                attempt,
                state=result.state,
                payload=payload,
                receipt_path=result.receipt_path,
            )
            self._store().transition(
                str(work["work_id"]),
                expected_state="running_gap_planning",
                next_state=next_state,
                context_update=payload,
            )
        except Exception as exc:
            self._handle_stage_exception(work, "gap_planning", attempt, root, exc)

    def _execute_portfolio(self, work: Mapping[str, object]) -> None:
        attempt, root = self._store().begin_stage(
            str(work["work_id"]),
            "portfolio",
            self._work_root(work) / "portfolio",
        )
        try:
            result = self.drivers.portfolio_plan(
                self.config, work=work, attempt_root=root
            )
            next_state = str(result.payload.get("experiment_portfolio_next_state") or "")
            if next_state not in {
                "pending_materialization",
                "pending_resource_admission",
                "portfolio_budget_blocked",
            }:
                raise AutonomousTransferControllerError(
                    "AUTONOMOUS_PORTFOLIO_NEXT_STATE_INVALID"
                )
            self._store().finish_stage(
                str(work["work_id"]),
                "portfolio",
                attempt,
                state=result.state,
                payload=result.payload,
                receipt_path=result.receipt_path,
            )
            self._store().transition(
                str(work["work_id"]),
                expected_state="running_portfolio",
                next_state=next_state,
                context_update=result.payload,
            )
        except Exception as exc:
            self._handle_stage_exception(work, "portfolio", attempt, root, exc)

    def _run_gpu_work(self) -> int:
        rows = self._store().list_work(["pending_screen"], limit=64)
        if not rows:
            rows = self._store().list_work(["pending_confirm"], limit=64)
        maximum = min(
            int(self.config["max_parallel_gpu_jobs"]),
            len(self.config["gpu_indices"]),  # type: ignore[arg-type]
        )
        claimed = []
        for raw in rows:
            work = self._store().claim(raw["work_id"], pending_state=str(raw["state"]))
            if work is not None:
                claimed.append(work)
            if len(claimed) >= maximum:
                break
        if not claimed:
            return 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=maximum) as executor:
            futures = [executor.submit(self._execute_gpu_stage, work) for work in claimed]
            for future in concurrent.futures.as_completed(futures):
                future.result()
        return len(claimed)

    def _run_verifiers(self) -> int:
        progressed = 0
        for raw in self._store().list_work(["pending_verify"], limit=32):
            work = self._store().claim(raw["work_id"], pending_state="pending_verify")
            if work is None:
                continue
            self._execute_verifier(work)
            progressed += 1
        return progressed

    def _run_resource_admissions(self) -> int:
        progressed = 0
        for raw in self._store().list_work(["pending_resource_admission"], limit=32):
            work = self._store().claim(
                raw["work_id"], pending_state="pending_resource_admission"
            )
            if work is None:
                continue
            self._execute_resource_admission(work)
            progressed += 1
        return progressed

    def _run_resource_reallocations(self) -> int:
        progressed = 0
        for raw in self._store().list_work(["pending_resource_reallocation"], limit=32):
            work = self._store().claim(
                raw["work_id"], pending_state="pending_resource_reallocation"
            )
            if work is None:
                continue
            self._execute_resource_reallocation(work)
            progressed += 1
        return progressed

    def _run_knowledge_updates(self) -> int:
        progressed = 0
        for raw in self._store().list_work(["pending_knowledge"], limit=64):
            work = self._store().claim(raw["work_id"], pending_state="pending_knowledge")
            if work is None:
                continue
            self._execute_knowledge(work)
            progressed += 1
        return progressed

    def _run_replans(self) -> int:
        if not isinstance(self.config.get("closed_loop"), Mapping):
            return 0
        blocker_states = (
            "pending_interface_extension",
            "requires_evaluator_binding",
            "observation_execution_blocked",
            "shadow_probe_evaluator_binding_required",
            "shadow_probe_admission_blocked",
            "resource_binding_required",
            "missing_data_regime",
            "architecture_bound",
            "portfolio_budget_blocked",
        )
        for raw in self._store().list_work(blocker_states, limit=32):
            state = str(raw["state"])
            self._store().transition(
                str(raw["work_id"]),
                expected_state=state,
                next_state="pending_replan",
                context_update={"replan_trigger": state},
            )
        progressed = 0
        for raw in self._store().list_work(["pending_replan"], limit=32):
            work = self._store().claim(raw["work_id"], pending_state="pending_replan")
            if work is None:
                continue
            self._execute_replan(work)
            progressed += 1
        return progressed

    def _execute_materialization(self, work: Mapping[str, object]) -> None:
        running = "running_materialization"
        attempt, root = self._store().begin_stage(
            str(work["work_id"]), "materialization", self._work_root(work) / "materialization"
        )
        try:
            result = self.drivers.materialization(self.config, work=work, attempt_root=root)
            self._store().finish_stage(
                str(work["work_id"]),
                "materialization",
                attempt,
                state=result.state,
                payload=result.payload,
                receipt_path=result.receipt_path,
            )
            requested_next = result.payload.get("materialization_next_state")
            if isinstance(requested_next, str) and requested_next:
                allowed = {
                    "pending_open_method_calibration",
                    "pending_interface_extension",
                    "missing_data_regime",
                    "architecture_bound",
                    "pending_replan",
                }
                if requested_next not in allowed:
                    raise AutonomousTransferControllerError(
                        "AUTONOMOUS_MATERIALIZATION_NEXT_STATE_INVALID"
                    )
                next_state = requested_next
            else:
                next_state = (
                    (
                        "pending_resource_admission"
                        if isinstance(self.config.get("resource_portfolio"), Mapping)
                        else "pending_screen"
                    )
                    if result.state == "completed"
                    else "pending_knowledge"
                )
            update = {**result.payload, "pre_verifier_outcome": result.outcome}
            self._store().transition(
                str(work["work_id"]),
                expected_state=running,
                next_state=next_state,
                context_update=update,
            )
        except Exception as exc:
            self._handle_stage_exception(work, "materialization", attempt, root, exc)

    def _execute_open_method_calibration(self, work: Mapping[str, object]) -> None:
        attempt, root = self._store().begin_stage(
            str(work["work_id"]),
            "open_method_calibration",
            self._work_root(work) / "open-method-calibration",
        )
        try:
            result = self.drivers.open_method_calibration(
                self.config, work=work, attempt_root=root
            )
            next_state = str(
                result.payload.get("open_method_calibration_next_state") or ""
            )
            if next_state not in {
                "pending_resource_admission",
                "pending_screen",
                "pending_replan",
                "pending_knowledge",
                "pending_interface_extension",
            }:
                raise AutonomousTransferControllerError(
                    "AUTONOMOUS_OPEN_METHOD_CALIBRATION_NEXT_STATE_INVALID"
                )
            self._store().finish_stage(
                str(work["work_id"]),
                "open_method_calibration",
                attempt,
                state=result.state,
                payload=result.payload,
                receipt_path=result.receipt_path,
            )
            self._store().transition(
                str(work["work_id"]),
                expected_state="running_open_method_calibration",
                next_state=next_state,
                context_update=result.payload,
            )
        except Exception as exc:
            self._handle_stage_exception(
                work, "open_method_calibration", attempt, root, exc
            )

    def _execute_resource_admission(self, work: Mapping[str, object]) -> None:
        attempt, root = self._store().begin_stage(
            str(work["work_id"]),
            "resource_admission",
            self._work_root(work) / "resource-admission",
        )
        try:
            result = self.drivers.resource_admission(
                self.config, work=work, attempt_root=root
            )
            next_state = str(result.payload.get("resource_portfolio_next_state") or "")
            if next_state not in {"pending_screen", "resource_binding_required"}:
                raise AutonomousTransferControllerError(
                    "AUTONOMOUS_RESOURCE_ADMISSION_NEXT_STATE_INVALID"
                )
            self._store().finish_stage(
                str(work["work_id"]),
                "resource_admission",
                attempt,
                state=result.state,
                payload=result.payload,
                receipt_path=result.receipt_path,
            )
            self._store().transition(
                str(work["work_id"]),
                expected_state="running_resource_admission",
                next_state=next_state,
                context_update=result.payload,
            )
        except Exception as exc:
            self._handle_stage_exception(
                work, "resource_admission", attempt, root, exc
            )

    def _execute_resource_reallocation(self, work: Mapping[str, object]) -> None:
        attempt, root = self._store().begin_stage(
            str(work["work_id"]),
            "resource_reallocation",
            self._work_root(work) / "resource-reallocation",
        )
        try:
            result = self.drivers.resource_reallocation(
                self.config, work=work, attempt_root=root
            )
            next_state = str(result.payload.get("resource_portfolio_next_state") or "")
            if next_state != "pending_confirm":
                raise AutonomousTransferControllerError(
                    "AUTONOMOUS_RESOURCE_REALLOCATION_NEXT_STATE_INVALID"
                )
            self._store().finish_stage(
                str(work["work_id"]),
                "resource_reallocation",
                attempt,
                state=result.state,
                payload=result.payload,
                receipt_path=result.receipt_path,
            )
            self._store().transition(
                str(work["work_id"]),
                expected_state="running_resource_reallocation",
                next_state=next_state,
                context_update=result.payload,
            )
        except Exception as exc:
            self._handle_stage_exception(
                work, "resource_reallocation", attempt, root, exc
            )

    def _execute_gpu_stage(self, work: Mapping[str, object]) -> None:
        stage = str(work["state"])[len("running_") :]
        running = f"running_{stage}"
        attempt, root = self._store().begin_stage(
            str(work["work_id"]), stage, self._work_root(work) / "campaign" / stage
        )
        try:
            result = self.drivers.gpu_stage(
                self.config, work=work, stage=stage, attempt_root=root
            )
            self._store().finish_stage(
                str(work["work_id"]),
                stage,
                attempt,
                state=result.state,
                payload=result.payload,
                receipt_path=result.receipt_path,
            )
            payload = dict(result.payload)
            if result.receipt_path is not None:
                receipt_path = Path(result.receipt_path).expanduser().resolve()
                payload[f"{stage}_stage_receipt_path"] = str(receipt_path)
                payload[f"{stage}_stage_receipt_sha256"] = hashlib.sha256(
                    receipt_path.read_bytes()
                ).hexdigest()
            if stage == "screen" and bool(payload.get("screen_accepted")):
                next_state = (
                    "pending_resource_reallocation"
                    if isinstance(self.config.get("resource_portfolio"), Mapping)
                    else "pending_confirm"
                )
            else:
                next_state = "pending_verify"
            self._store().transition(
                str(work["work_id"]),
                expected_state=running,
                next_state=next_state,
                context_update=payload,
            )
        except GpuLeaseError as exc:
            self._store().finish_stage(
                str(work["work_id"]),
                stage,
                attempt,
                state="deferred",
                payload={"error": str(exc)},
                receipt_path=None,
            )
            self._store().transition(
                str(work["work_id"]),
                expected_state=running,
                next_state=f"pending_{stage}",
                error=str(exc),
            )
        except Exception as exc:
            self._handle_stage_exception(work, stage, attempt, root, exc)

    def _execute_verifier(self, work: Mapping[str, object]) -> None:
        attempt, root = self._store().begin_stage(
            str(work["work_id"]), "verify", self._work_root(work) / "verifier"
        )
        try:
            result = self.drivers.verifier(self.config, work=work, attempt_root=root)
            staging = workflow.stage_verified_knowledge(
                self.config,
                state_root=self.state_root,
                work=work,
                verification=result,
            )
            payload = {**result.payload, **staging}
            self._store().finish_stage(
                str(work["work_id"]),
                "verify",
                attempt,
                state=result.state,
                payload=payload,
                receipt_path=result.receipt_path,
            )
            self._store().transition(
                str(work["work_id"]),
                expected_state="running_verify",
                next_state="pending_knowledge",
                context_update=payload,
            )
        except Exception as exc:
            self._handle_stage_exception(work, "verify", attempt, root, exc)

    def _execute_knowledge(self, work: Mapping[str, object]) -> None:
        attempt, root = self._store().begin_stage(
            str(work["work_id"]), "knowledge", self._work_root(work) / "knowledge"
        )
        try:
            result = self.drivers.knowledge(self.config, state_root=self.state_root)
            self._store().finish_stage(
                str(work["work_id"]),
                "knowledge",
                attempt,
                state=result.state,
                payload=result.payload,
                receipt_path=result.receipt_path,
            )
            context = work.get("context")
            assert isinstance(context, Mapping)
            outcome = str(
                context.get("decision")
                or context.get("pre_verifier_outcome")
                or "knowledge_only"
            )
            next_state = (
                "pending_replan"
                if isinstance(self.config.get("closed_loop"), Mapping)
                else "terminal"
            )
            self._store().transition(
                str(work["work_id"]),
                expected_state="running_knowledge",
                next_state=next_state,
                context_update=result.payload,
                terminal_outcome=outcome if next_state == "terminal" else None,
            )
        except Exception as exc:
            self._handle_stage_exception(work, "knowledge", attempt, root, exc)

    def _execute_replan(self, work: Mapping[str, object]) -> None:
        attempt, root = self._store().begin_stage(
            str(work["work_id"]), "replan", self._work_root(work) / "replan"
        )
        try:
            root.mkdir(mode=0o700, parents=True, exist_ok=True)
            result = self.drivers.replan(
                self.config,
                state_root=self.state_root,
                work=work,
                attempt_root=root,
                snapshot=self._store().snapshot(),
            )
            next_state = str(result.payload.get("replan_next_state") or "")
            if next_state not in {
                "pending_observation",
                "pending_gap_planning",
                "terminal",
            }:
                raise AutonomousTransferControllerError(
                    "AUTONOMOUS_REPLAN_NEXT_STATE_INVALID"
                )
            self._store().finish_stage(
                str(work["work_id"]),
                "replan",
                attempt,
                state=result.state,
                payload=result.payload,
                receipt_path=result.receipt_path,
            )
            outcome = str(result.payload.get("next_task_stop_reason") or result.outcome)
            self._store().transition(
                str(work["work_id"]),
                expected_state="running_replan",
                next_state=next_state,
                context_update=result.payload,
                terminal_outcome=outcome if next_state == "terminal" else None,
            )
        except Exception as exc:
            self._handle_stage_exception(work, "replan", attempt, root, exc)

    def _handle_stage_exception(
        self,
        work: Mapping[str, object],
        stage: str,
        attempt: int,
        root: Path,
        exc: Exception,
    ) -> None:
        error = f"{type(exc).__name__}:{str(exc)[:1000]}"
        failure_path = root.parent / f"attempt-{attempt:03d}-controller-failure.json"
        _write_json_atomic(
            failure_path,
            {
                "schema_version": 1,
                "artifact_type": "verdiwm-autonomous-stage-failure",
                "state": "failed",
                "work_id": work["work_id"],
                "stage": stage,
                "attempt": attempt,
                "error": error,
                "recorded_at": _utc_now(),
            },
        )
        self._store().finish_stage(
            str(work["work_id"]),
            stage,
            attempt,
            state="failed",
            payload={"error": error},
            receipt_path=failure_path,
        )
        failures = int(work["failure_count"]) + 1
        retry_limit = int(self.config["max_operational_retries"])
        if failures <= retry_limit:
            next_state = f"pending_{stage if stage != 'verify' else 'verify'}"
            terminal = None
        elif stage == "knowledge":
            next_state = "terminal"
            terminal = "knowledge_projection_failure"
        elif stage == "replan":
            next_state = "terminal"
            terminal = "replan_failure"
        else:
            next_state = "pending_knowledge"
            terminal = "operational_failure_unverified"
        self._store().transition(
            str(work["work_id"]),
            expected_state=f"running_{stage}",
            next_state=next_state,
            context_update={"pre_verifier_outcome": terminal} if terminal else None,
            terminal_outcome=terminal if next_state == "terminal" else None,
            error=error,
            increment_failure=True,
        )

    def _import_existing_evidence(self) -> None:
        imported_any = False
        for row in self.drivers.evidence_import(self.config, state_root=self.state_root):
            inserted = self._store().import_verified_evidence(
                source_path=Path(row["source_path"]),
                local_path=Path(row["local_path"]),
                record=row["record"],
            )
            imported_any = inserted or imported_any
        if isinstance(self.config.get("closed_loop"), Mapping):
            self._store().queue_imported_replans()
        graph_root = Path(str(self.config["knowledge_graph_root"])).expanduser().resolve()
        portable_graph_root = Path(
            str(
                self.config.get("portable_knowledge_root")
                or graph_root.with_name(graph_root.name + "-portable")
            )
        ).expanduser().resolve()
        if (
            imported_any
            or not (graph_root / "manifest.json").is_file()
            or not (portable_graph_root / "manifest.json").is_file()
        ):
            self.drivers.knowledge(self.config, state_root=self.state_root)

    def _work_root(self, work: Mapping[str, object]) -> Path:
        return self.state_root / "work" / str(work["work_id"])

    def _store(self) -> DurableLoopStore:
        if self.store is None:
            raise AutonomousTransferControllerError("AUTONOMOUS_STORE_NOT_OPEN")
        return self.store

    def _write_status(self, state: str, extra: Mapping[str, object]) -> None:
        status = self.store.status() if self.store is not None else {}
        _write_json_atomic(
            self.state_root / "status.json",
            {
                "schema_version": 1,
                "artifact_type": "verdiwm-autonomous-controller-status",
                "state": state,
                "loop_id": self.config["loop_id"],
                "config_digest": self.config["config_digest"],
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
                "heartbeat_at": _utc_now(),
                "persistent_state": str(self.state_root / "controller.db"),
                "gpu_indices": self.config["gpu_indices"],
                "max_parallel_gpu_jobs": self.config["max_parallel_gpu_jobs"],
                "controller_status": status,
                **extra,
            },
        )


def run_once(
    *,
    config_path: Path,
    state_root: Path,
    force_discovery: bool = False,
    drivers: ControllerDrivers | None = None,
) -> dict[str, object]:
    config = workflow.load_and_validate_config(config_path, project_root=PROJECT_ROOT)
    with ControllerRuntime(config=config, state_root=state_root, drivers=drivers) as runtime:
        return runtime.run_cycle(force_discovery=force_discovery)


def run_forever(
    *,
    config_path: Path,
    state_root: Path,
    force_discovery: bool = False,
    max_cycles: int = 0,
    drivers: ControllerDrivers | None = None,
) -> dict[str, object]:
    config = workflow.load_and_validate_config(config_path, project_root=PROJECT_ROOT)
    stop = threading.Event()

    def request_stop(_signal: int, _frame: object) -> None:
        stop.set()

    previous = {
        signum: signal.signal(signum, request_stop)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    cycles = 0
    last: dict[str, object] = {}
    try:
        with ControllerRuntime(config=config, state_root=state_root, drivers=drivers) as runtime:
            while not stop.is_set():
                last = runtime.run_cycle(force_discovery=force_discovery and cycles == 0)
                cycles += 1
                if max_cycles and cycles >= max_cycles:
                    break
                stop.wait(float(config["poll_seconds"]))
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    return {"state": "stopped", "cycles": cycles, "last_cycle": last}


def read_status(*, config_path: Path, state_root: Path) -> dict[str, object]:
    config = workflow.load_and_validate_config(config_path, project_root=PROJECT_ROOT)
    root = Path(state_root).expanduser().resolve()
    status_path = root / "status.json"
    if not status_path.is_file() or status_path.is_symlink():
        return {
            "state": "not_started",
            "loop_id": config["loop_id"],
            "state_root": str(root),
        }
    payload = json.loads(status_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AutonomousTransferControllerError("AUTONOMOUS_STATUS_INVALID")
    return payload


def _failure_context(state_root: Path, status: Mapping[str, object]) -> tuple[str, ...]:
    values: set[str] = set()
    outcomes = status.get("terminal_outcome_counts")
    if isinstance(outcomes, Mapping):
        values.update(str(key) for key in outcomes)
    for path in sorted(state_root.rglob("verified-evidence.jsonl")):
        if path.is_symlink() or not path.is_file():
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            blockers = row.get("blockers") if isinstance(row, Mapping) else None
            if isinstance(blockers, list):
                values.update(str(value) for value in blockers if str(value))
    return tuple(sorted(values))[:12]


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{threading.get_ident()}")
    with temporary.open("wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    once = subparsers.add_parser("run-once")
    once.add_argument("--force-discovery", action="store_true")
    run = subparsers.add_parser("run")
    run.add_argument("--force-discovery", action="store_true")
    run.add_argument("--max-cycles", type=int, default=0)
    subparsers.add_parser("status")
    subparsers.add_parser("validate")
    args = parser.parse_args(argv)
    if args.command == "validate":
        result = workflow.load_and_validate_config(args.config, project_root=PROJECT_ROOT)
        output: Mapping[str, object] = {
            "state": "validated",
            "loop_id": result["loop_id"],
            "config_digest": result["config_digest"],
        }
    elif args.command == "status":
        output = read_status(config_path=args.config, state_root=args.state_root)
    elif args.command == "run-once":
        output = run_once(
            config_path=args.config,
            state_root=args.state_root,
            force_discovery=bool(args.force_discovery),
        )
    else:
        output = run_forever(
            config_path=args.config,
            state_root=args.state_root,
            force_discovery=bool(args.force_discovery),
            max_cycles=int(args.max_cycles),
        )
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
