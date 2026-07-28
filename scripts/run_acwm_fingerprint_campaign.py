#!/usr/bin/env python3
"""Run the ACWM-Phys fingerprint campaign over an explicit GPU set."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wmloop.experiments.acwm_fingerprint import load_campaign


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> dict[str, object]:
    campaign = load_campaign(args.campaign)
    environments = list(campaign["environments"])
    if args.environment:
        unknown = sorted(set(args.environment) - set(environments))
        if unknown:
            raise ValueError(f"FINGERPRINT_ENVIRONMENT_UNKNOWN:{','.join(unknown)}")
        environments = list(args.environment)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "status.json"
    lock = threading.Lock()
    status: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "verdiwm-acwm-fingerprint-campaign-status",
        "campaign_id": campaign["campaign_id"],
        "protocol": args.protocol,
        "state": "running",
        "started_at_unix": time.time(),
        "gpus": list(args.gpu),
        "environments": {environment: {"state": "pending"} for environment in environments},
    }
    _atomic_json(status_path, status)

    def execute(index_environment: tuple[int, str]) -> tuple[str, int]:
        index, environment = index_environment
        gpu = args.gpu[index % len(args.gpu)]
        env_output = output_root / "environments" / environment
        env_output.mkdir(parents=True, exist_ok=True)
        log_path = output_root / "logs" / f"{environment}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(Path(__file__).with_name("run_acwm_fingerprint_probe.py")),
            "--campaign", str(args.campaign.resolve()),
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
                "state": "running", "physical_gpu": gpu, "command": command, "log_path": str(log_path)
            }
            _atomic_json(status_path, status)
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.run(command, env=child_env, stdout=log, stderr=subprocess.STDOUT, check=False)
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
    with ThreadPoolExecutor(max_workers=len(args.gpu)) as executor:
        futures = [executor.submit(execute, item) for item in enumerate(environments)]
        for future in as_completed(futures):
            environment, return_code = future.result()
            results[environment] = return_code
    status["state"] = "ready" if all(value == 0 for value in results.values()) else "failed"
    status["completed_at_unix"] = time.time()
    status["return_codes"] = results
    _atomic_json(status_path, status)
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--protocol", choices=("smoke", "pilot", "paper"), default="pilot")
    parser.add_argument("--gpu", type=int, action="append", required=True)
    parser.add_argument("--environment", action="append", default=[])
    parser.add_argument("--vendor-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--vae-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
