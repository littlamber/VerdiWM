#!/usr/bin/env python3
"""Run prepared ACWM CPBE canary campaigns in parallel, one candidate per GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def build_candidate_gpu_assignments(
    campaigns: list[dict[str, object]], gpus: list[int]
) -> list[tuple[dict[str, object], int]]:
    if len(campaigns) != len(gpus) or not campaigns or len(set(gpus)) != len(gpus):
        raise ValueError("ACWM_CPBE_CANDIDATE_GPU_FRAME_INVALID")
    probe_ids = [str(row.get("probe_id", "")) for row in campaigns]
    if any(not value for value in probe_ids) or len(set(probe_ids)) != len(probe_ids):
        raise ValueError("ACWM_CPBE_CANDIDATE_FRAME_INVALID")
    return list(zip(sorted(campaigns, key=lambda row: str(row["probe_id"])), gpus, strict=True))


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(args: argparse.Namespace) -> dict[str, object]:
    preparation_root = args.preparation_root.resolve(strict=True)
    manifest = json.loads((preparation_root / "manifest.json").read_text(encoding="utf-8"))
    spec = json.loads((preparation_root / manifest["collision_spec_path"]).read_text(encoding="utf-8"))
    campaigns = manifest.get("campaigns")
    if manifest.get("state") != "ready" or not isinstance(campaigns, list):
        raise RuntimeError("ACWM_CPBE_PREPARATION_NOT_READY")
    assignments = build_candidate_gpu_assignments(campaigns, list(args.gpu))
    environments = [str(row["environment"]) for row in spec["labels"]]
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    logs = output_root / "logs"
    logs.mkdir()
    status_path = output_root / "status.json"
    lock = threading.Lock()
    status: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "verdiwm-acwm-cpbe-canary-campaign-status",
        "state": "running",
        "started_at_unix": time.time(),
        "protocol": args.protocol,
        "environments": environments,
        "candidates": {},
    }
    _atomic_json(status_path, status)

    def execute(entry: dict[str, object], gpu: int) -> tuple[str, int]:
        probe_id = str(entry["probe_id"])
        campaign_path = preparation_root / str(entry["path"])
        if _sha256(campaign_path) != entry["sha256"]:
            raise RuntimeError(f"ACWM_CPBE_CAMPAIGN_HASH_MISMATCH:{probe_id}")
        candidate_output = output_root / "candidates" / probe_id
        command = [
            sys.executable,
            str(Path(__file__).with_name("run_acwm_fingerprint_campaign.py")),
            "--campaign",
            str(campaign_path),
            "--protocol",
            args.protocol,
            "--gpu",
            str(gpu),
        ]
        for environment in environments:
            command.extend(["--environment", environment])
        command.extend(
            [
                "--vendor-root",
                str(args.vendor_root.resolve(strict=True)),
                "--data-root",
                str(args.data_root.resolve(strict=True)),
                "--checkpoint-root",
                str(args.checkpoint_root.resolve(strict=True)),
                "--vae-path",
                str(args.vae_path.resolve(strict=True)),
                "--output-root",
                str(candidate_output),
            ]
        )
        log_path = logs / f"{probe_id}.log"
        with lock:
            status["candidates"][probe_id] = {  # type: ignore[index]
                "state": "running",
                "physical_gpu": gpu,
                "campaign_sha256": entry["sha256"],
                "command": command,
                "log_path": str(log_path),
            }
            _atomic_json(status_path, status)
        child_env = dict(os.environ)
        child_env["PYTHONDONTWRITEBYTECODE"] = "1"
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.run(
                command,
                env=child_env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        with lock:
            status["candidates"][probe_id] = {  # type: ignore[index]
                "state": "ready" if process.returncode == 0 else "failed",
                "physical_gpu": gpu,
                "campaign_sha256": entry["sha256"],
                "return_code": process.returncode,
                "log_path": str(log_path),
                "output_root": str(candidate_output),
            }
            _atomic_json(status_path, status)
        return probe_id, process.returncode

    results: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=len(assignments)) as executor:
        futures = [executor.submit(execute, campaign, gpu) for campaign, gpu in assignments]
        for future in as_completed(futures):
            probe_id, return_code = future.result()
            results[probe_id] = return_code
    status["state"] = "ready" if all(value == 0 for value in results.values()) else "failed"
    status["completed_at_unix"] = time.time()
    status["return_codes"] = results
    _atomic_json(status_path, status)
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation-root", type=Path, required=True)
    parser.add_argument("--protocol", choices=("smoke", "pilot", "paper"), default="pilot")
    parser.add_argument("--gpu", type=int, action="append", required=True)
    parser.add_argument("--vendor-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--vae-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    print(json.dumps(run(parser.parse_args()), sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
