#!/usr/bin/env python3
"""Run a resumable joint-frame ACWM fingerprint campaign on explicit GPUs."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import threading
import time
from typing import Sequence

from wmloop.experiments.joint_fingerprint import load_joint_campaign, load_joint_sources


REPO_ROOT = Path(__file__).resolve().parents[1]


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> dict[str, object]:
    joint_path = args.joint_campaign.resolve(strict=True)
    joint = load_joint_campaign(joint_path)
    sources = load_joint_sources(joint, repo_root=REPO_ROOT)
    environments = list(sources[0]["environments"])
    if args.environment:
        unknown = sorted(set(args.environment) - set(environments))
        if unknown:
            raise ValueError(f"JOINT_FINGERPRINT_ENVIRONMENT_UNKNOWN:{','.join(unknown)}")
        environments = list(args.environment)
    runtime_python = args.runtime_python.resolve(strict=True)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "status.json"
    lock = threading.Lock()
    status: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "verdiwm-acwm-joint-fingerprint-campaign-status",
        "campaign_id": joint["campaign_id"],
        "protocol": args.protocol,
        "state": "running",
        "started_at_unix": time.time(),
        "runtime_python": str(runtime_python),
        "gpus": list(args.gpu),
        "environments": {environment: {"state": "pending"} for environment in environments},
    }
    _atomic_json(status_path, status)

    def execute(environment: str, gpu: int) -> tuple[str, int]:
        env_output = output_root / "environments" / environment
        env_output.mkdir(parents=True, exist_ok=True)
        log_path = output_root / "logs" / f"{environment}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            str(runtime_python),
            str(Path(__file__).with_name("run_acwm_joint_fingerprint_probe.py")),
            "--joint-campaign", str(joint_path),
            "--environment", environment,
            "--protocol", args.protocol,
            "--vendor-root", str(args.vendor_root.resolve()),
            "--data-root", str(args.data_root.resolve()),
            "--checkpoint-root", str(args.checkpoint_root.resolve()),
            "--vae-path", str(args.vae_path.resolve()),
            "--output-root", str(env_output),
            "--physical-gpu", str(gpu),
        ]
        child_env = dict(os.environ)
        child_env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        child_env["PYTHONDONTWRITEBYTECODE"] = "1"
        with lock:
            status["environments"][environment] = {  # type: ignore[index]
                "state": "running",
                "physical_gpu": gpu,
                "command": command,
                "log_path": str(log_path),
            }
            _atomic_json(status_path, status)
        with log_path.open("a", encoding="utf-8") as log:
            process = subprocess.run(
                command,
                env=child_env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        with lock:
            status["environments"][environment] = {  # type: ignore[index]
                "state": "ready" if process.returncode == 0 else "failed",
                "physical_gpu": gpu,
                "return_code": process.returncode,
                "log_path": str(log_path),
                "manifest_path": str(env_output / "manifest.json"),
            }
            _atomic_json(status_path, status)
        return environment, process.returncode

    results: dict[str, int] = {}
    if not args.gpu or len(set(args.gpu)) != len(args.gpu):
        raise ValueError("JOINT_FINGERPRINT_GPU_SET_INVALID")
    pending: Queue[str] = Queue()
    for environment in environments:
        pending.put(environment)

    def execute_queue(gpu: int) -> dict[str, int]:
        completed: dict[str, int] = {}
        while True:
            try:
                environment = pending.get_nowait()
            except Empty:
                return completed
            try:
                completed[environment] = execute(environment, gpu)[1]
            finally:
                pending.task_done()

    with ThreadPoolExecutor(max_workers=len(args.gpu)) as executor:
        futures = [
            executor.submit(execute_queue, gpu)
            for gpu in args.gpu
        ]
        for future in as_completed(futures):
            results.update(future.result())
    status["state"] = "ready" if len(results) == len(environments) and all(code == 0 for code in results.values()) else "failed"
    status["completed_at_unix"] = time.time()
    status["return_codes"] = results
    _atomic_json(status_path, status)
    return status


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint-campaign", type=Path, required=True)
    parser.add_argument("--protocol", choices=("smoke", "pilot", "paper"), default="pilot")
    parser.add_argument("--runtime-python", type=Path, required=True)
    parser.add_argument("--gpu", type=int, action="append", required=True)
    parser.add_argument("--environment", action="append", default=[])
    parser.add_argument("--vendor-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--vae-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run(args), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
