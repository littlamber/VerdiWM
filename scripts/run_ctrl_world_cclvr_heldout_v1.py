#!/usr/bin/env python3
"""Run the frozen CCLVR v1 held-out routing and action-sensitivity campaign."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
PLAN_V1 = "verdiwm-ctrl-world-cclvr-heldout-plan-v1"
PLAN_V2 = "verdiwm-ctrl-world-cclvr-heldout-plan-v2"


class CCLVRHeldoutLaunchError(RuntimeError):
    """The frozen CCLVR held-out plan or one of its jobs is invalid."""


def _load(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CCLVRHeldoutLaunchError(f"CCLVR_HELDOUT_JSON_INVALID:{path}") from exc
    if not isinstance(payload, Mapping):
        raise CCLVRHeldoutLaunchError(f"CCLVR_HELDOUT_JSON_INVALID:{path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _resolve(value: object) -> Path:
    path = Path(str(value))
    return (ROOT / path).resolve(strict=True) if not path.is_absolute() else path.resolve(strict=True)


def _validate_hash(path: Path, expected: object, error: str) -> None:
    if not isinstance(expected, str) or len(expected) != 64 or _sha256(path) != expected:
        raise CCLVRHeldoutLaunchError(error)


def _validate_adapter_source(plan_type: object, dependencies: Mapping[str, Any]) -> None:
    if plan_type == PLAN_V1:
        return
    if plan_type != PLAN_V2:
        raise CCLVRHeldoutLaunchError("CCLVR_HELDOUT_PLAN_INVALID")
    ctrl_world_root = _resolve(dependencies.get("ctrl_world_root"))
    expected_source = (ctrl_world_root / "models" / "multiscale_history_adapter.py").resolve(strict=True)
    adapter_source = _resolve(dependencies.get("adapter_source"))
    if adapter_source != expected_source:
        raise CCLVRHeldoutLaunchError("CCLVR_HELDOUT_ADAPTER_SOURCE_INVALID")
    _validate_hash(
        adapter_source,
        dependencies.get("adapter_source_sha256"),
        "CCLVR_HELDOUT_ADAPTER_SOURCE_HASH_MISMATCH",
    )


def _validate_plan(path: Path) -> tuple[Mapping[str, Any], dict[str, Mapping[str, Any]], Path, Path, Path]:
    plan = _load(path)
    if (
        plan.get("artifact_type") not in {PLAN_V1, PLAN_V2}
        or plan.get("state") != "frozen_before_execution"
        or str(plan.get("promotion_episode")) != "1799"
        or plan.get("confirmation_authorized") is not False
    ):
        raise CCLVRHeldoutLaunchError("CCLVR_HELDOUT_PLAN_INVALID")
    dependencies = plan.get("dependencies")
    references = plan.get("references")
    cells = plan.get("cells")
    routing_jobs = plan.get("routing_jobs")
    action_jobs = plan.get("action_jobs")
    if (
        not isinstance(dependencies, Mapping)
        or not isinstance(references, Mapping)
        or not isinstance(cells, list)
        or not isinstance(routing_jobs, list)
        or not isinstance(action_jobs, list)
    ):
        raise CCLVRHeldoutLaunchError("CCLVR_HELDOUT_PLAN_INVALID")
    _validate_adapter_source(plan.get("artifact_type"), dependencies)
    evaluator = _resolve(dependencies.get("evaluator"))
    aggregator = _resolve(dependencies.get("aggregator"))
    launcher = _resolve(dependencies.get("launcher"))
    base_evaluator = _resolve(dependencies.get("base_evaluator"))
    _validate_hash(evaluator, dependencies.get("evaluator_sha256"), "CCLVR_HELDOUT_EVALUATOR_HASH_MISMATCH")
    _validate_hash(aggregator, dependencies.get("aggregator_sha256"), "CCLVR_HELDOUT_AGGREGATOR_HASH_MISMATCH")
    _validate_hash(launcher, dependencies.get("launcher_sha256"), "CCLVR_HELDOUT_LAUNCHER_HASH_MISMATCH")
    _validate_hash(
        base_evaluator,
        dependencies.get("base_evaluator_sha256"),
        "CCLVR_HELDOUT_BASE_EVALUATOR_HASH_MISMATCH",
    )
    contexts = _resolve(plan.get("contexts_json"))
    _validate_hash(contexts, plan.get("contexts_sha256"), "CCLVR_HELDOUT_CONTEXTS_HASH_MISMATCH")
    context_payload = _load(contexts)
    context_rows = context_payload.get("contexts")
    if (
        context_payload.get("artifact_type") != "verdiwm-ctrl-world-local-context-set"
        or not isinstance(context_rows, list)
        or len(context_rows) != 1
        or str(context_rows[0].get("episode_id")) != "1799"
        or str(context_rows[0].get("context_id")) != str(plan.get("context_id"))
        or [int(value) for value in context_rows[0].get("seeds", ())]
        != [int(value) for value in plan.get("seeds", ())]
    ):
        raise CCLVRHeldoutLaunchError("CCLVR_HELDOUT_CONTEXTS_INVALID")
    source_settlement = _resolve(plan.get("source_training_settlement"))
    _validate_hash(
        source_settlement,
        plan.get("source_training_settlement_sha256"),
        "CCLVR_HELDOUT_TRAINING_SETTLEMENT_HASH_MISMATCH",
    )
    if _load(source_settlement).get("promotion_episode_used") is not False:
        raise CCLVRHeldoutLaunchError("CCLVR_HELDOUT_TRAINING_LEAKAGE_INVALID")

    cell_map: dict[str, Mapping[str, Any]] = {}
    for cell in cells:
        if not isinstance(cell, Mapping) or str(cell.get("id")) not in {"d1", "d2", "d3", "d4"}:
            raise CCLVRHeldoutLaunchError("CCLVR_HELDOUT_CELL_INVALID")
        cell_id = str(cell["id"])
        if cell_id in cell_map or cell.get("route_scope") not in {"episode", "interaction"}:
            raise CCLVRHeldoutLaunchError("CCLVR_HELDOUT_CELL_INVALID")
        checkpoint = _resolve(cell.get("checkpoint"))
        bank = _resolve(cell.get("supervision_bank"))
        _validate_hash(checkpoint, cell.get("checkpoint_sha256"), "CCLVR_HELDOUT_CHECKPOINT_HASH_MISMATCH")
        _validate_hash(bank, cell.get("supervision_bank_sha256"), "CCLVR_HELDOUT_BANK_HASH_MISMATCH")
        cell_map[cell_id] = {**cell, "checkpoint": str(checkpoint), "supervision_bank": str(bank)}
    if set(cell_map) != {"d1", "d2", "d3", "d4"}:
        raise CCLVRHeldoutLaunchError("CCLVR_HELDOUT_CELL_INVALID")
    if cell_map["d1"]["route_scope"] != "episode" or cell_map["d2"]["route_scope"] != "episode":
        raise CCLVRHeldoutLaunchError("CCLVR_HELDOUT_CELL_SCOPE_INVALID")
    if cell_map["d3"]["route_scope"] != "interaction" or cell_map["d4"]["route_scope"] != "interaction":
        raise CCLVRHeldoutLaunchError("CCLVR_HELDOUT_CELL_SCOPE_INVALID")

    if len(routing_jobs) != 8 or len({int(job["gpu"]) for job in routing_jobs}) != 8:
        raise CCLVRHeldoutLaunchError("CCLVR_HELDOUT_ROUTING_JOBS_INVALID")
    routing_coverage: dict[str, list[int]] = {cell_id: [] for cell_id in cell_map}
    output_rels: set[str] = set()
    for job in routing_jobs:
        if not isinstance(job, Mapping) or str(job.get("cell_id")) not in cell_map:
            raise CCLVRHeldoutLaunchError("CCLVR_HELDOUT_ROUTING_JOBS_INVALID")
        cell_id = str(job["cell_id"])
        seeds = [int(value) for value in job.get("seeds", ())]
        if not seeds:
            raise CCLVRHeldoutLaunchError("CCLVR_HELDOUT_ROUTING_JOBS_INVALID")
        routing_coverage[cell_id].extend(seeds)
        output_rel = str(job.get("output_rel", ""))
        if not output_rel or output_rel in output_rels or Path(output_rel).is_absolute():
            raise CCLVRHeldoutLaunchError("CCLVR_HELDOUT_OUTPUT_FRAME_INVALID")
        output_rels.add(output_rel)
    expected_seeds = sorted(int(value) for value in plan["seeds"])
    if any(sorted(values) != expected_seeds for values in routing_coverage.values()):
        raise CCLVRHeldoutLaunchError("CCLVR_HELDOUT_ROUTING_COVERAGE_INVALID")
    if len(action_jobs) != len(expected_seeds):
        raise CCLVRHeldoutLaunchError("CCLVR_HELDOUT_ACTION_JOBS_INVALID")
    if sorted(int(job["seed"]) for job in action_jobs) != expected_seeds:
        raise CCLVRHeldoutLaunchError("CCLVR_HELDOUT_ACTION_JOBS_INVALID")
    if len({int(job["gpu"]) for job in action_jobs}) != len(action_jobs):
        raise CCLVRHeldoutLaunchError("CCLVR_HELDOUT_ACTION_JOBS_INVALID")
    for job in action_jobs:
        output_rel = str(job.get("output_rel", ""))
        if not output_rel or output_rel in output_rels or Path(output_rel).is_absolute():
            raise CCLVRHeldoutLaunchError("CCLVR_HELDOUT_OUTPUT_FRAME_INVALID")
        output_rels.add(output_rel)
    for path_key, hash_key in (
        ("a0_action_result", "a0_action_result_sha256"),
        ("cbma_report", "cbma_report_sha256"),
    ):
        reference = _resolve(references.get(path_key))
        _validate_hash(reference, references.get(hash_key), "CCLVR_HELDOUT_REFERENCE_HASH_MISMATCH")
    return plan, cell_map, contexts, evaluator, aggregator


def _runtime_python(value: object) -> Path:
    path = Path(str(value))
    if not path.is_absolute() or not path.is_file() or not os.access(path, os.X_OK):
        raise CCLVRHeldoutLaunchError("CCLVR_HELDOUT_RUNTIME_INVALID")
    return path


def run(*, plan_path: Path) -> dict[str, object]:
    plan_path = plan_path.resolve(strict=True)
    plan, cells, contexts, evaluator, aggregator = _validate_plan(plan_path)
    output_root = Path(str(plan["output_root"])).resolve()
    if output_root.exists() or output_root.is_symlink():
        raise CCLVRHeldoutLaunchError("CCLVR_HELDOUT_OUTPUT_EXISTS")
    dependencies = plan["dependencies"]
    runtime = _runtime_python(dependencies["runtime_python"])
    preflight = subprocess.run(
        [str(runtime), "-c", "import diffusers, mediapy, torch; assert torch.cuda.device_count() >= 1"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if preflight.returncode != 0:
        raise CCLVRHeldoutLaunchError(f"CCLVR_HELDOUT_PREFLIGHT_FAILED:{preflight.stdout.strip()}")
    output_root.mkdir(mode=0o700, parents=True)
    status_path = output_root / "campaign-status.json"
    status: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "verdiwm-ctrl-world-cclvr-heldout-status-v1",
        "state": "running",
        "experiment_id": plan["experiment_id"],
        "plan": str(plan_path),
        "plan_sha256": _sha256(plan_path),
        "started_at_unix": time.time(),
        "phases": {"routing": {"state": "pending", "jobs": {}}, "action_sensitivity": {"state": "pending", "jobs": {}}},
    }
    _atomic_json(status_path, status)
    lock = threading.Lock()

    def command_for(job: Mapping[str, Any], *, phase: str) -> tuple[list[str], Path, Path]:
        cell_id = str(job.get("cell_id", "d4"))
        cell = cells[cell_id]
        job_root = output_root / str(job["output_rel"])
        job_id = str(job["id"])
        log_path = output_root / "logs" / f"{phase}-{job_id}.log"
        seeds = [int(job["seed"])] if phase == "action_sensitivity" else [int(value) for value in job["seeds"]]
        command = [
            str(runtime),
            "-u",
            str(evaluator),
            "--campaign-id",
            str(plan["experiment_id"]),
            "--cell-id",
            cell_id,
            "--mode",
            phase,
            "--route-scope",
            str(cell["route_scope"]),
            "--ctrl-world-root",
            str(dependencies["ctrl_world_root"]),
            "--dataset-root",
            str(dependencies["dataset_root"]),
            "--data-stat",
            str(dependencies["data_stat"]),
            "--svd-model-path",
            str(dependencies["svd_model_path"]),
            "--clip-model-path",
            str(dependencies["clip_model_path"]),
            "--ckpt-path",
            str(cell["checkpoint"]),
            "--contexts-json",
            str(contexts),
            "--context-id",
            str(plan["context_id"]),
            "--seeds",
            *[str(value) for value in seeds],
            "--interact-num",
            "4",
            "--num-inference-steps",
            "4",
            "--cclvr-supervision-path",
            str(cell["supervision_bank"]),
            "--cclvr-supervision-sha256",
            str(cell["supervision_bank_sha256"]),
            "--cclvr-supervision-variant",
            str(cell["supervision_variant"]),
            "--cclvr-value-hidden-dim",
            str(cell.get("cclvr_value_hidden_dim", 128)),
            "--cclvr-policy-temperature",
            str(cell.get("cclvr_policy_temperature", 0.1)),
            "--output-root",
            str(job_root),
        ]
        if phase == "action_sensitivity":
            command.extend(["--action-doses", "-0.1", "0", "0.1"])
        return command, job_root, log_path

    def execute(job: Mapping[str, Any], *, phase: str) -> tuple[str, int]:
        job_id = str(job["id"])
        gpu = int(job["gpu"])
        command, job_root, log_path = command_for(job, phase=phase)
        log_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        launch = {
            "state": "running",
            "physical_gpu": gpu,
            "command": command,
            "output_root": str(job_root),
            "log_path": str(log_path),
            "started_at_unix": time.time(),
        }
        with lock:
            status["phases"][phase]["jobs"][job_id] = launch
            _atomic_json(status_path, status)
        child_env = dict(os.environ)
        child_env.update(
            {
                "CUDA_VISIBLE_DEVICES": str(gpu),
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "PYTHONDONTWRITEBYTECODE": "1",
            }
        )
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.run(
                command,
                cwd=ROOT,
                env=child_env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        result_path = job_root / "result.json"
        ready = process.returncode == 0 and result_path.is_file() and _load(result_path).get("state") == "ready"
        receipt = {
            **launch,
            "schema_version": 1,
            "artifact_type": "verdiwm-ctrl-world-cclvr-heldout-job-receipt-v1",
            "state": "completed" if ready else "failed",
            "return_code": process.returncode,
            "result_path": str(result_path),
            "result_sha256": _sha256(result_path) if result_path.is_file() else None,
            "completed_at_unix": time.time(),
        }
        _atomic_json(output_root / "receipts" / f"{phase}-{job_id}.json", receipt)
        with lock:
            status["phases"][phase]["jobs"][job_id] = receipt
            _atomic_json(status_path, status)
        return job_id, 0 if ready else max(process.returncode, 1)

    def run_phase(phase: str, jobs: Sequence[Mapping[str, Any]]) -> None:
        status["phases"][phase]["state"] = "running"
        _atomic_json(status_path, status)
        return_codes: dict[str, int] = {}
        with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
            futures = [executor.submit(execute, job, phase=phase) for job in jobs]
            for future in as_completed(futures):
                job_id, return_code = future.result()
                return_codes[job_id] = return_code
        failed = {job_id: code for job_id, code in return_codes.items() if code != 0}
        status["phases"][phase]["state"] = "failed" if failed else "completed"
        status["phases"][phase]["return_codes"] = return_codes
        _atomic_json(status_path, status)
        if failed:
            status["state"] = "failed"
            status["failed_phase"] = phase
            _atomic_json(status_path, status)
            raise CCLVRHeldoutLaunchError(f"CCLVR_HELDOUT_PHASE_FAILED:{phase}:{failed}")

    run_phase("routing", plan["routing_jobs"])
    run_phase("action_sensitivity", plan["action_jobs"])
    report_path = output_root / "heldout-report.json"
    aggregate_log = output_root / "logs" / "aggregate.log"
    with aggregate_log.open("w", encoding="utf-8") as log:
        process = subprocess.run(
            [str(runtime), "-u", str(aggregator), "--plan", str(plan_path), "--output", str(report_path)],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if process.returncode != 0 or not report_path.is_file() or _load(report_path).get("state") != "ready":
        status["state"] = "failed"
        status["failed_phase"] = "aggregation"
        _atomic_json(status_path, status)
        raise CCLVRHeldoutLaunchError("CCLVR_HELDOUT_AGGREGATION_FAILED")
    report = _load(report_path)
    settlement = {
        "schema_version": 1,
        "artifact_type": "verdiwm-ctrl-world-cclvr-heldout-settlement-v1",
        "state": "completed",
        "experiment_id": plan["experiment_id"],
        "plan": str(plan_path),
        "plan_sha256": _sha256(plan_path),
        "report": str(report_path),
        "report_sha256": _sha256(report_path),
        "promotion": report["promotion"],
        "confirmation_authorized": False,
        "promotion_episode": "1799",
        "promotion_episode_reuse_forbidden": True,
        "completed_at_unix": time.time(),
    }
    settlement_path = output_root / "SETTLEMENT.json"
    _atomic_json(settlement_path, settlement)
    status.update(
        {
            "state": "completed",
            "report": str(report_path),
            "report_sha256": _sha256(report_path),
            "settlement": str(settlement_path),
            "settlement_sha256": _sha256(settlement_path),
            "completed_at_unix": time.time(),
        }
    )
    _atomic_json(status_path, status)
    return settlement


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run(plan_path=args.plan)
    print(json.dumps(result["promotion"], ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
