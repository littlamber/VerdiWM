#!/usr/bin/env python3
"""Compile or run a resumable source-grounded Hybrid Memory ACWM campaign."""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from wmloop.control.acwm_campaign import canonical_json_bytes, load_mapping  # noqa: E402
from wmloop.control.acwm_materialized_campaign import (  # noqa: E402
    ACWMMaterializedCampaignError,
    build_evaluator_command,
    compile_materialized_candidate_batch,
    failed_materialized_candidate,
    load_baseline,
    load_materialized_measurement,
    settle_materialized_candidate,
    sha256_file,
    terminal_state,
    validate_materialized_candidate_batch,
)


class HybridMemoryCampaignRunnerError(RuntimeError):
    """The campaign cannot be launched or resumed without violating its lock."""


def compile_batch(args: argparse.Namespace) -> dict[str, object]:
    contract_path = _require_file(args.contract, "HYBRID_CAMPAIGN_CONTRACT_INVALID")
    contract = load_mapping(contract_path, error_code="HYBRID_CAMPAIGN_CONTRACT_INVALID")
    batch, compilation = compile_materialized_candidate_batch(
        catalog_path=args.catalog,
        assessment_path=args.assessment,
        contract=contract,
        stage=args.stage,
        batch_id=args.batch_id,
        objective=args.objective,
        hypothesis=args.hypothesis,
        falsification_criterion=args.falsification_criterion,
        selection_reason=args.selection_reason,
        expected_gpu_hours_per_candidate=args.expected_gpu_hours,
        root=PROJECT_ROOT,
    )
    output = Path(args.output).expanduser().resolve()
    _write_json_idempotent(output, batch)
    report_path = Path(args.compilation_report).expanduser().resolve()
    _write_json_idempotent(report_path, compilation)
    return {
        "state": "compiled",
        "batch_path": str(output),
        "batch_digest": batch["batch_digest"],
        "compilation_report_path": str(report_path),
    }


def run_campaign(args: argparse.Namespace) -> dict[str, object]:
    contract_path = _require_file(args.contract, "HYBRID_CAMPAIGN_CONTRACT_INVALID")
    batch_path = _require_file(args.batch, "HYBRID_CAMPAIGN_BATCH_INVALID")
    baseline_path = _require_file(args.baseline, "HYBRID_CAMPAIGN_BASELINE_INVALID")
    evaluator = _require_file(args.evaluator, "HYBRID_CAMPAIGN_EVALUATOR_INVALID")
    base_evaluator = _require_file(
        args.base_evaluator, "HYBRID_CAMPAIGN_BASE_EVALUATOR_INVALID"
    )
    runtime_python = _require_file(
        args.runtime_python, "HYBRID_CAMPAIGN_RUNTIME_PYTHON_INVALID"
    )
    contract = load_mapping(contract_path, error_code="HYBRID_CAMPAIGN_CONTRACT_INVALID")
    batch = load_mapping(batch_path, error_code="HYBRID_CAMPAIGN_BATCH_INVALID")
    validate_materialized_candidate_batch(batch, contract, root=PROJECT_ROOT)
    baseline, baseline_sha256 = load_baseline(
        baseline_path, contract=contract, stage=str(batch["stage"]), root=PROJECT_ROOT
    )
    assets = _asset_paths(args)
    destination = Path(args.output_root).expanduser().resolve()
    _validate_output_root(destination, source_root=assets["ctrl_world_root"])
    gpu_indices = _gpu_indices(args.gpu_indices, len(batch["candidates"]))
    input_lock = _input_lock(
        batch=batch,
        batch_path=batch_path,
        contract=contract,
        contract_path=contract_path,
        baseline_path=baseline_path,
        baseline_sha256=baseline_sha256,
        evaluator=evaluator,
        base_evaluator=base_evaluator,
        runtime_python=runtime_python,
        assets=assets,
        gpu_indices=gpu_indices,
    )
    if args.dry_run:
        return {
            "state": "validated",
            "batch_id": batch["batch_id"],
            "candidate_count": len(batch["candidates"]),
            "gpu_indices": gpu_indices,
            "input_sha256": input_lock["input_sha256"],
            "gpu_execution_started": False,
        }
    _prepare_output_root(destination, input_lock=input_lock, resume=bool(args.resume))
    _write_json_atomic(
        destination / "status.json",
        {
            "schema_version": 1,
            "artifact_type": "verdiwm-acwm-materialized-campaign-status",
            "state": "running",
            "batch_id": batch["batch_id"],
            "input_sha256": input_lock["input_sha256"],
            "updated_at": _utc_now(),
        },
    )
    candidates = batch["candidates"]
    assert isinstance(candidates, list)
    worker_results: dict[str, dict[str, object]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(candidates)) as executor:
        futures = {
            executor.submit(
                _run_one_candidate,
                args=args,
                batch=batch,
                contract=contract,
                baseline=baseline,
                baseline_sha256=baseline_sha256,
                baseline_path=baseline_path,
                candidate=candidate,
                gpu_index=gpu_indices[index],
                destination=destination,
                evaluator=evaluator,
                base_evaluator=base_evaluator,
                runtime_python=runtime_python,
                assets=assets,
                input_lock=input_lock,
            ): str(candidate["candidate_id"])
            for index, candidate in enumerate(candidates)
            if isinstance(candidate, Mapping)
        }
        for future in concurrent.futures.as_completed(futures):
            candidate_id = futures[future]
            try:
                worker_results[candidate_id] = future.result()
            except Exception as exc:
                worker_results[candidate_id] = {
                    "candidate_id": candidate_id,
                    "state": "failed",
                    "error": str(exc),
                }
    settlements = [worker_results[str(row["candidate_id"])] for row in candidates]
    _write_knowledge(destination, settlements)
    counts = dict(sorted(Counter(str(row.get("state")) for row in settlements).items()))
    summary = {
        "schema_version": 1,
        "artifact_type": "verdiwm-acwm-materialized-campaign-summary",
        "state": "settled",
        "batch_id": batch["batch_id"],
        "batch_digest": batch["batch_digest"],
        "contract_id": contract["contract_id"],
        "contract_digest": contract["contract_digest"],
        "stage": batch["stage"],
        "input_sha256": input_lock["input_sha256"],
        "counts": counts,
        "candidates": [
            {
                "candidate_id": row.get("candidate", {}).get("candidate_id", row.get("candidate_id")),
                "gpu_index": row.get("gpu_index"),
                "state": row.get("state"),
                "accepted": row.get("accepted", False),
                "settlement_sha256": _settlement_sha(destination, row),
            }
            for row in settlements
        ],
        "claim_boundary": (
            "This campaign settles screen or confirmation evidence only. "
            "No result enters shared verified knowledge before frozen verification."
        ),
    }
    _write_json_idempotent(destination / "campaign-summary.json", summary)
    _write_json_atomic(destination / "status.json", {**summary, "updated_at": _utc_now()})
    return summary


def _run_one_candidate(
    *,
    args: argparse.Namespace,
    batch: Mapping[str, object],
    contract: Mapping[str, object],
    baseline: Mapping[str, object],
    baseline_sha256: str,
    baseline_path: Path,
    candidate: Mapping[str, object],
    gpu_index: int,
    destination: Path,
    evaluator: Path,
    base_evaluator: Path,
    runtime_python: Path,
    assets: Mapping[str, Path],
    input_lock: Mapping[str, object],
) -> dict[str, object]:
    candidate_id = str(candidate["candidate_id"])
    candidate_root = destination / "candidates" / candidate_id
    candidate_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    candidate_path = candidate_root / "candidate.json"
    _write_json_idempotent(candidate_path, dict(candidate))
    settlement_path = candidate_root / "settlement.json"
    if settlement_path.is_file():
        settlement = load_mapping(
            settlement_path, error_code="HYBRID_CAMPAIGN_SETTLEMENT_INVALID"
        )
        if settlement.get("input_sha256") != input_lock["input_sha256"] or not terminal_state(settlement):
            raise HybridMemoryCampaignRunnerError("HYBRID_CAMPAIGN_SETTLEMENT_RESUME_MISMATCH")
        return settlement
    measurement_path = candidate_root / "measurement" / "measurement.json"
    worker_receipt_path = candidate_root / "worker-receipt.json"
    try:
        if measurement_path.is_file():
            measurement, measurement_sha256 = load_materialized_measurement(
                measurement_path,
                contract=contract,
                stage=str(batch["stage"]),
                candidate=candidate,
                root=PROJECT_ROOT,
            )
            worker_receipt = {
                "schema_version": 1,
                "artifact_type": "verdiwm-acwm-materialized-worker-receipt",
                "candidate_id": candidate_id,
                "gpu_index": gpu_index,
                "state": "recovered_measurement",
                "measurement_path": str(measurement_path),
            }
            _write_json_idempotent(worker_receipt_path, worker_receipt)
        elif measurement_path.exists():
            raise HybridMemoryCampaignRunnerError("HYBRID_CAMPAIGN_MEASUREMENT_PATH_INVALID")
        else:
            worker_receipt = _launch_worker(
                args=args,
                batch=batch,
                candidate=candidate,
                candidate_path=candidate_path,
                gpu_index=gpu_index,
                candidate_root=candidate_root,
                measurement_path=measurement_path,
                evaluator=evaluator,
                base_evaluator=base_evaluator,
                runtime_python=runtime_python,
                assets=assets,
            )
            _write_json_idempotent(worker_receipt_path, worker_receipt)
            if worker_receipt["state"] != "completed":
                return _persist_settlement(
                    settlement_path,
                    failed_materialized_candidate(
                        batch=batch,
                        contract=contract,
                        candidate=candidate,
                        gpu_index=gpu_index,
                        failure_code=str(worker_receipt["failure_code"]),
                        worker_receipt_path=worker_receipt_path,
                    ),
                    input_lock=input_lock,
                )
            measurement, measurement_sha256 = load_materialized_measurement(
                measurement_path,
                contract=contract,
                stage=str(batch["stage"]),
                candidate=candidate,
                root=PROJECT_ROOT,
            )
        settlement = settle_materialized_candidate(
            batch=batch,
            contract=contract,
            baseline=baseline,
            baseline_path=baseline_path,
            baseline_sha256=baseline_sha256,
            candidate=candidate,
            measurement=measurement,
            measurement_path=measurement_path,
            measurement_sha256=measurement_sha256,
            gpu_index=gpu_index,
            root=PROJECT_ROOT,
        )
        return _persist_settlement(settlement_path, settlement, input_lock=input_lock)
    except Exception as exc:
        failure_code = (
            str(exc)
            if isinstance(exc, (ACWMMaterializedCampaignError, HybridMemoryCampaignRunnerError))
            else "HYBRID_CAMPAIGN_INTERNAL_ERROR"
        )
        if not worker_receipt_path.exists():
            _write_json_idempotent(
                worker_receipt_path,
                {
                    "schema_version": 1,
                    "artifact_type": "verdiwm-acwm-materialized-worker-receipt",
                    "candidate_id": candidate_id,
                    "gpu_index": gpu_index,
                    "state": "failed",
                    "failure_code": failure_code,
                },
            )
        return _persist_settlement(
            settlement_path,
            failed_materialized_candidate(
                batch=batch,
                contract=contract,
                candidate=candidate,
                gpu_index=gpu_index,
                failure_code=failure_code,
                worker_receipt_path=worker_receipt_path,
            ),
            input_lock=input_lock,
        )


def _launch_worker(
    *,
    args: argparse.Namespace,
    batch: Mapping[str, object],
    candidate: Mapping[str, object],
    candidate_path: Path,
    gpu_index: int,
    candidate_root: Path,
    measurement_path: Path,
    evaluator: Path,
    base_evaluator: Path,
    runtime_python: Path,
    assets: Mapping[str, Path],
) -> dict[str, object]:
    command = build_evaluator_command(
        runtime_python=runtime_python,
        evaluator=evaluator,
        base_evaluator=base_evaluator,
        contract=Path(args.contract),
        stage=str(batch["stage"]),
        candidate_path=candidate_path,
        output_root=measurement_path.parent,
        **assets,
    )
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    started_at = _utc_now()
    log_path = candidate_root / "worker.log"
    observation = None
    return_code = None
    timed_out = False
    try:
        with log_path.open("x", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=str(PROJECT_ROOT),
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            if int(args.activity_probe_seconds) > 0:
                time.sleep(int(args.activity_probe_seconds))
                observation = _gpu_observation(
                    gpu_index,
                    worker_pid=process.pid,
                    process_alive=process.poll() is None,
                )
            try:
                return_code = process.wait(timeout=float(args.worker_timeout_seconds))
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
    except (OSError, ValueError) as exc:
        return {
            "schema_version": 1,
            "artifact_type": "verdiwm-acwm-materialized-worker-receipt",
            "candidate_id": candidate["candidate_id"],
            "gpu_index": gpu_index,
            "state": "failed",
            "failure_code": "HYBRID_CAMPAIGN_WORKER_LAUNCH_FAILED",
            "detail": type(exc).__name__,
            "command": command,
            "started_at": started_at,
            "ended_at": _utc_now(),
            "gpu_observation": observation,
        }
    completed = not timed_out and return_code == 0
    return {
        "schema_version": 1,
        "artifact_type": "verdiwm-acwm-materialized-worker-receipt",
        "candidate_id": candidate["candidate_id"],
        "gpu_index": gpu_index,
        "state": "completed" if completed else "failed",
        "failure_code": None if completed else (
            "HYBRID_CAMPAIGN_WORKER_TIMEOUT" if timed_out else "HYBRID_CAMPAIGN_WORKER_EXIT_NONZERO"
        ),
        "exit_code": return_code,
        "timed_out": timed_out,
        "command": command,
        "started_at": started_at,
        "ended_at": _utc_now(),
        "gpu_observation": observation,
        "output_artifact": str(measurement_path),
    }


def _input_lock(
    *,
    batch: Mapping[str, object],
    batch_path: Path,
    contract: Mapping[str, object],
    contract_path: Path,
    baseline_path: Path,
    baseline_sha256: str,
    evaluator: Path,
    base_evaluator: Path,
    runtime_python: Path,
    assets: Mapping[str, Path],
    gpu_indices: list[int],
) -> dict[str, object]:
    payload = {
        "schema_version": 1,
        "artifact_type": "verdiwm-acwm-materialized-campaign-input-lock",
        "batch_id": batch["batch_id"],
        "batch_digest": batch["batch_digest"],
        "batch_path": str(batch_path),
        "batch_file_sha256": sha256_file(batch_path),
        "contract_id": contract["contract_id"],
        "contract_digest": contract["contract_digest"],
        "contract_path": str(contract_path),
        "contract_file_sha256": sha256_file(contract_path),
        "baseline_path": str(baseline_path),
        "baseline_sha256": baseline_sha256,
        "evaluator_path": str(evaluator),
        "evaluator_sha256": sha256_file(evaluator),
        "base_evaluator_path": str(base_evaluator),
        "base_evaluator_sha256": sha256_file(base_evaluator),
        "runtime_python": str(runtime_python),
        "assets": {name: str(path) for name, path in sorted(assets.items())},
        "gpu_indices": gpu_indices,
    }
    payload["input_sha256"] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    return payload


def _persist_settlement(
    path: Path, settlement: dict[str, object], *, input_lock: Mapping[str, object]
) -> dict[str, object]:
    row = {**settlement, "input_sha256": input_lock["input_sha256"]}
    _write_json_idempotent(path, row)
    return row


def _prepare_output_root(
    destination: Path, *, input_lock: Mapping[str, object], resume: bool
) -> None:
    lock_path = destination / "input-lock.json"
    if destination.exists():
        if destination.is_symlink() or not destination.is_dir() or not resume:
            raise HybridMemoryCampaignRunnerError("HYBRID_CAMPAIGN_OUTPUT_EXISTS")
        existing = load_mapping(lock_path, error_code="HYBRID_CAMPAIGN_INPUT_LOCK_INVALID")
        if existing.get("input_sha256") != input_lock.get("input_sha256"):
            raise HybridMemoryCampaignRunnerError("HYBRID_CAMPAIGN_INPUT_LOCK_MISMATCH")
        return
    destination.mkdir(mode=0o700, parents=True, exist_ok=False)
    _write_json_idempotent(lock_path, dict(input_lock))


def _asset_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "ctrl_world_root": _require_directory(args.ctrl_world_root, "HYBRID_CAMPAIGN_CTRL_WORLD_ROOT_INVALID"),
        "dataset_root": _require_directory(args.dataset_root, "HYBRID_CAMPAIGN_DATASET_ROOT_INVALID"),
        "data_stat": _require_file(args.data_stat, "HYBRID_CAMPAIGN_DATA_STAT_INVALID"),
        "checkpoint": _require_file(args.checkpoint, "HYBRID_CAMPAIGN_CHECKPOINT_INVALID"),
        "svd_model": _require_directory(args.svd_model, "HYBRID_CAMPAIGN_SVD_MODEL_INVALID"),
        "clip_model": _require_directory(args.clip_model, "HYBRID_CAMPAIGN_CLIP_MODEL_INVALID"),
    }


def _gpu_indices(raw: str, count: int) -> list[int]:
    try:
        values = [int(value.strip()) for value in raw.split(",") if value.strip()]
    except ValueError as exc:
        raise HybridMemoryCampaignRunnerError("HYBRID_CAMPAIGN_GPU_INDICES_INVALID") from exc
    if len(values) < count or len(values) != len(set(values)) or any(value < 0 for value in values):
        raise HybridMemoryCampaignRunnerError("HYBRID_CAMPAIGN_GPU_INDICES_INVALID")
    return values[:count]


def _gpu_observation(gpu_index: int, *, worker_pid: int, process_alive: bool) -> dict[str, object]:
    command = [
        "nvidia-smi", "--id", str(gpu_index),
        "--query-gpu=index,uuid,name,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=10)
        identity = {
            "state": "observed" if completed.returncode == 0 else "failed",
            "return_code": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        identity = {"state": "unavailable", "detail": type(exc).__name__}
    return {
        "physical_gpu_index": gpu_index,
        "worker_pid": worker_pid,
        "worker_alive_at_probe": process_alive,
        "identity": identity,
    }


def _write_knowledge(destination: Path, settlements: list[dict[str, object]]) -> None:
    payload = b"".join(canonical_json_bytes(row) for row in settlements)
    _write_bytes_idempotent(destination / "knowledge.jsonl", payload)
    _write_json_idempotent(
        destination / "knowledge-index.json",
        {
            "schema_version": 1,
            "artifact_type": "verdiwm-acwm-materialized-campaign-knowledge-index",
            "record_count": len(settlements),
            "knowledge_sha256": hashlib.sha256(payload).hexdigest(),
            "verification_authority": False,
        },
    )


def _settlement_sha(destination: Path, row: Mapping[str, object]) -> str | None:
    candidate = row.get("candidate")
    candidate_id = candidate.get("candidate_id") if isinstance(candidate, Mapping) else row.get("candidate_id")
    if not isinstance(candidate_id, str):
        return None
    path = destination / "candidates" / candidate_id / "settlement.json"
    return sha256_file(path) if path.is_file() else None


def _validate_output_root(destination: Path, *, source_root: Path) -> None:
    for protected in (PROJECT_ROOT.resolve(), source_root.resolve()):
        if destination == protected or protected in destination.parents:
            raise HybridMemoryCampaignRunnerError("HYBRID_CAMPAIGN_OUTPUT_INSIDE_SOURCE")


def _require_file(path: Path, code: str) -> Path:
    raw = Path(path).expanduser()
    if raw.is_symlink():
        raise HybridMemoryCampaignRunnerError(code)
    resolved = raw.resolve()
    if not resolved.is_file():
        raise HybridMemoryCampaignRunnerError(code)
    return resolved


def _require_directory(path: Path, code: str) -> Path:
    raw = Path(path).expanduser()
    if raw.is_symlink():
        raise HybridMemoryCampaignRunnerError(code)
    resolved = raw.resolve()
    if not resolved.is_dir():
        raise HybridMemoryCampaignRunnerError(code)
    return resolved


def _write_json_idempotent(path: Path, payload: Mapping[str, object]) -> None:
    _write_bytes_idempotent(path, canonical_json_bytes(payload))


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    data = canonical_json_bytes(payload)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def _write_bytes_idempotent(path: Path, payload: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise HybridMemoryCampaignRunnerError("HYBRID_CAMPAIGN_IMMUTABLE_WRITE_CONFLICT")
        return
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(payload)
    os.replace(temporary, path)


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--base-evaluator", type=Path, required=True)
    parser.add_argument("--runtime-python", type=Path, required=True)
    parser.add_argument("--ctrl-world-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--data-stat", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--svd-model", type=Path, required=True)
    parser.add_argument("--clip-model", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu-indices", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--activity-probe-seconds", type=int, default=10)
    parser.add_argument("--worker-timeout-seconds", type=float, default=7200)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    compile_parser = subparsers.add_parser("compile")
    compile_parser.add_argument("--catalog", type=Path, required=True)
    compile_parser.add_argument("--assessment", type=Path, required=True)
    compile_parser.add_argument("--contract", type=Path, required=True)
    compile_parser.add_argument("--stage", choices=("screen", "confirm"), required=True)
    compile_parser.add_argument("--batch-id", required=True)
    compile_parser.add_argument("--objective", required=True)
    compile_parser.add_argument("--hypothesis", required=True)
    compile_parser.add_argument("--falsification-criterion", required=True)
    compile_parser.add_argument("--selection-reason", required=True)
    compile_parser.add_argument("--expected-gpu-hours", type=float, required=True)
    compile_parser.add_argument("--output", type=Path, required=True)
    compile_parser.add_argument("--compilation-report", type=Path, required=True)
    run_parser = subparsers.add_parser("run")
    _add_runtime_arguments(run_parser)
    args = parser.parse_args(argv)
    result = compile_batch(args) if args.command == "compile" else run_campaign(args)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
