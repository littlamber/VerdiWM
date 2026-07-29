#!/usr/bin/env python3
"""Run a shard of the frozen Cosmos3 paired-dose fingerprint campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path


def _dose_tag(dose: float) -> str:
    return f"{dose:+.4f}".replace("+", "p").replace("-", "m").replace(".", "d")


def run_campaign(args: argparse.Namespace) -> dict[str, object]:
    campaign = json.loads(args.campaign.read_text(encoding="utf-8"))
    split = json.loads(args.split_path.read_text(encoding="utf-8"))
    protocol = campaign["protocols"][args.protocol]
    split_name = str(protocol["split"])
    configured_doses = tuple(float(value) for value in campaign["probe"]["doses"])
    selected_doses = tuple(float(value) for value in (args.doses or configured_doses))
    if any(value not in configured_doses for value in selected_doses):
        raise ValueError("COSMOS3_CAMPAIGN_DOSE_OUTSIDE_FROZEN_CONFIG")
    identities = [dict(item) for item in split[split_name]]
    if args.sample_indices:
        selected = set(args.sample_indices)
        identities = [item for item in identities if int(item["sample_index"]) in selected]
        if {int(item["sample_index"]) for item in identities} != selected:
            raise ValueError("COSMOS3_CAMPAIGN_SAMPLE_OUTSIDE_FROZEN_SPLIT")

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(args.verdiwm_root), str(args.cosmos_root), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    _preflight_runtime(
        args.runtime_python,
        modules=("imageio_ffmpeg", "PIL", "cosmos_framework", "wmloop"),
        code="COSMOS3_RUNNER_PREFLIGHT_FAILED",
        cwd=args.cosmos_root,
        env=env,
    )
    _preflight_runtime(
        args.evaluator_python,
        modules=("imageio.v3", "numpy", "PIL", "jsonschema", "cosmos_framework", "wmloop"),
        code="COSMOS3_EVALUATOR_PREFLIGHT_FAILED",
        cwd=args.verdiwm_root,
        env=env,
    )

    output_root = args.output_root.resolve()
    if output_root.exists() or output_root.is_symlink():
        raise ValueError("COSMOS3_CAMPAIGN_OUTPUT_EXISTS")
    output_root.mkdir(parents=True)
    records: list[dict[str, object]] = []
    started = time.time()
    try:
        for dose in selected_doses:
            for identity in identities:
                sample_index = int(identity["sample_index"])
                seed = int(identity["seed"])
                name = f"dose-{_dose_tag(dose)}-sample-{sample_index}-seed-{seed}"
                run_root = output_root / "runner" / name
                eval_root = output_root / "receipts" / name
                log_root = output_root / "logs" / name
                log_root.mkdir(parents=True)
                runner_command = [
                    str(args.runtime_python),
                    str(args.runner),
                    "--repo-root", str(args.cosmos_root),
                    "--dataset-root", str(args.dataset_root),
                    "--checkpoint-path", str(args.checkpoint_path),
                    "--config-file", str(args.config_file),
                    "--output-dir", str(run_root),
                    "--num-chunks", "1",
                    "--chunk-length", "16",
                    "--start-index", str(sample_index),
                    "--seed", str(seed),
                    "--action-dose", str(dose),
                    "--cuda-visible-devices", str(args.gpu_index),
                    "--master-port", str(args.master_port + len(records)),
                ]
                runner_gpu_audit = _run(
                    runner_command,
                    cwd=args.cosmos_root,
                    env=env,
                    stdout_path=log_root / "runner.stdout.log",
                    stderr_path=log_root / "runner.stderr.log",
                    exclusive_gpu_index=args.gpu_index,
                    gpu_audit_path=log_root / "gpu-exclusivity-audit.json",
                )
                action = run_root / "inputs/robotics_droid_action_chunk_00.json"
                hook = run_root / "inputs/robotics_droid_action_chunk_00.hook.json"
                sample_args = run_root / "inputs/action_forward_dynamics_robotics_chunk_00.jsonl"
                rollout = run_root / "robotics_action_cond_chunk_00/vision.mp4"
                evaluator_command = [
                    str(args.evaluator_python),
                    str(args.evaluator),
                    "--cosmos-root", str(args.cosmos_root),
                    "--dataset-root", str(args.dataset_root),
                    "--split-path", str(args.split_path),
                    "--split-name", split_name,
                    "--sample-index", str(sample_index),
                    "--seed", str(seed),
                    "--rollout", str(rollout),
                    "--sample-args", str(sample_args),
                    "--action-input", str(action),
                    "--action-hook-receipt", str(hook),
                    "--action-dose", str(dose),
                    "--output-root", str(eval_root),
                ]
                _run(
                    evaluator_command,
                    cwd=args.verdiwm_root,
                    env=env,
                    stdout_path=log_root / "evaluator.stdout.log",
                    stderr_path=log_root / "evaluator.stderr.log",
                )
                receipt_path = eval_root / "prediction-receipt.json"
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                records.append(
                    {
                        "dose": dose,
                        "sample_index": sample_index,
                        "seed": seed,
                        "receipt_ref": str(receipt_path),
                        "receipt_sha256": _sha256(receipt_path),
                        "metrics": receipt["metrics"],
                        "gpu_exclusivity_audit": runner_gpu_audit,
                    }
                )
                _write_state(output_root, args, split_name, selected_doses, identities, records, started)
        manifest = _write_state(output_root, args, split_name, selected_doses, identities, records, started)
        return manifest
    except BaseException:
        _write_state(output_root, args, split_name, selected_doses, identities, records, started, state="failed")
        raise


def _preflight_runtime(
    python: Path,
    *,
    modules: Sequence[str],
    code: str,
    cwd: Path,
    env: dict[str, str],
) -> None:
    imports = "; ".join(f"import {module}" for module in modules)
    completed = subprocess.run(
        [str(python), "-c", imports],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip().replace("\n", " | ")[-2000:]
        raise RuntimeError(f"{code}:{completed.returncode}:{detail}")


def _run(
    command: Sequence[str],
    *,
    cwd: Path,
    env: dict[str, str],
    stdout_path: Path,
    stderr_path: Path,
    exclusive_gpu_index: int | None = None,
    gpu_audit_path: Path | None = None,
) -> dict[str, object] | None:
    audit = None
    target_uuid = None
    if exclusive_gpu_index is not None:
        if gpu_audit_path is None:
            raise ValueError("COSMOS3_GPU_AUDIT_PATH_REQUIRED")
        target_uuid = _gpu_uuid(exclusive_gpu_index)
        foreign = _foreign_compute_pids(target_uuid, _compute_process_rows(), os.getpid())
        if foreign:
            raise RuntimeError(f"COSMOS3_GPU_NOT_EXCLUSIVE_BEFORE_RUN:{target_uuid}:{foreign}")
        audit = {
            "schema_version": 1,
            "artifact_type": "verdiwm-cosmos3-gpu-exclusivity-audit",
            "state": "running",
            "gpu_index": exclusive_gpu_index,
            "gpu_uuid": target_uuid,
            "monitor_interval_seconds": 0.5,
            "sample_count": 0,
            "observed_campaign_pids": [],
            "foreign_pid_events": [],
            "started_at_unix": time.time(),
        }
        _write_gpu_audit(gpu_audit_path, audit)
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        contamination: list[int] = []
        try:
            while process.poll() is None:
                if audit is not None and target_uuid is not None:
                    rows = _compute_process_rows()
                    own = sorted(
                        pid
                        for uuid, pid in rows
                        if uuid == target_uuid and _pid_is_descendant(pid, os.getpid())
                    )
                    foreign = _foreign_compute_pids(target_uuid, rows, os.getpid())
                    audit["sample_count"] = int(audit["sample_count"]) + 1
                    audit["observed_campaign_pids"] = sorted(
                        set(audit["observed_campaign_pids"]) | set(own)
                    )
                    if foreign:
                        contamination = foreign
                        audit["foreign_pid_events"].append(
                            {"observed_at_unix": time.time(), "pids": foreign}
                        )
                        _terminate_process_group(process)
                        break
                    _write_gpu_audit(gpu_audit_path, audit)
                time.sleep(0.5)
            returncode = process.wait()
        except BaseException:
            _terminate_process_group(process)
            if audit is not None:
                audit["completed_at_unix"] = time.time()
                audit["state"] = "failed"
                audit["termination_reason"] = "campaign_interrupted"
                _write_gpu_audit(gpu_audit_path, audit)
            raise
    if audit is not None:
        audit["completed_at_unix"] = time.time()
        audit["state"] = "failed" if contamination or returncode != 0 else "ready"
        _write_gpu_audit(gpu_audit_path, audit)
    if contamination:
        raise RuntimeError(f"COSMOS3_GPU_EXCLUSIVITY_VIOLATION:{target_uuid}:{contamination}")
    if returncode != 0:
        raise RuntimeError(f"COSMOS3_CAMPAIGN_COMMAND_FAILED:{returncode}:{command[0]}")
    return audit


def _gpu_uuid(gpu_index: int) -> str:
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError("COSMOS3_GPU_INVENTORY_QUERY_FAILED")
    mapping = {}
    for line in completed.stdout.splitlines():
        index, uuid = (part.strip() for part in line.split(",", maxsplit=1))
        mapping[int(index)] = uuid
    if gpu_index not in mapping:
        raise RuntimeError(f"COSMOS3_GPU_INDEX_INVALID:{gpu_index}")
    return mapping[gpu_index]


def _compute_process_rows() -> list[tuple[str, int]]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError("COSMOS3_GPU_PROCESS_QUERY_FAILED")
    rows = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        uuid, raw_pid = (part.strip() for part in line.split(",", maxsplit=1))
        rows.append((uuid, int(raw_pid)))
    return rows


def _foreign_compute_pids(
    target_uuid: str,
    rows: Sequence[tuple[str, int]],
    root_pid: int,
    parent_reader: Callable[[int], int | None] | None = None,
) -> list[int]:
    return sorted(
        pid
        for uuid, pid in rows
        if uuid == target_uuid and not _pid_is_descendant(pid, root_pid, parent_reader)
    )


def _pid_is_descendant(
    pid: int, root_pid: int, parent_reader: Callable[[int], int | None] | None = None
) -> bool:
    read_parent = parent_reader or _read_parent_pid
    current = pid
    visited = set()
    while current > 1 and current not in visited:
        if current == root_pid:
            return True
        visited.add(current)
        parent = read_parent(current)
        if parent is None:
            return False
        current = parent
    return current == root_pid


def _read_parent_pid(pid: int) -> int | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("PPid:"):
                return int(line.split(":", maxsplit=1)[1].strip())
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None
    return None


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
    except ProcessLookupError:
        return


def _write_gpu_audit(path: Path, audit: dict[str, object]) -> None:
    path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_state(
    root: Path,
    args: argparse.Namespace,
    split_name: str,
    doses: Sequence[float],
    identities: Sequence[dict[str, object]],
    records: Sequence[dict[str, object]],
    started: float,
    *,
    state: str = "ready",
) -> dict[str, object]:
    expected = len(doses) * len(identities)
    settled_state = state if state == "failed" else ("ready" if len(records) == expected else "running")
    manifest: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "verdiwm-cosmos3-fingerprint-campaign-shard",
        "state": settled_state,
        "campaign_id": json.loads(args.campaign.read_text(encoding="utf-8"))["campaign_id"],
        "protocol": args.protocol,
        "split": split_name,
        "gpu_index": args.gpu_index,
        "doses": list(doses),
        "identities": identities,
        "expected_receipt_count": expected,
        "receipt_count": len(records),
        "elapsed_seconds": time.time() - started,
        "records": list(records),
        "claim_boundary": "Campaign execution evidence only; locality and transfer claims require the complete fitted chart.",
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--protocol", choices=("pilot", "paper"), required=True)
    parser.add_argument("--split-path", type=Path, required=True)
    parser.add_argument("--doses", type=float, nargs="+")
    parser.add_argument("--sample-indices", type=int, nargs="+")
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--master-port", type=int, default=29710)
    parser.add_argument("--verdiwm-root", type=Path, required=True)
    parser.add_argument("--cosmos-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--config-file", type=Path, required=True)
    parser.add_argument("--runtime-python", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--evaluator-python", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    manifest = run_campaign(parse_args(argv))
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
