#!/usr/bin/env python3
"""Run one ACWM-Phys environment through a joint-frame probe campaign."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_acwm_fingerprint_probe import _dose_context, _load_eval_module
from wmloop.experiments.acwm_fingerprint import compile_probe_receipt, sha256_file
from wmloop.experiments.joint_fingerprint import (
    compose_joint_irg_asset,
    condition_schedule,
    fit_joint_fingerprint,
    load_joint_campaign,
    load_joint_sources,
)
from wmloop.geometry.assets import validate_irg_asset


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _tensor_digest(*tensors: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        value = tensor.detach().cpu().contiguous()
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _load_measurements(root: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not root.exists():
        return rows
    for path in sorted(root.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        condition_id = str(payload.get("condition_id") or "")
        if not condition_id or condition_id in rows:
            raise RuntimeError("JOINT_FINGERPRINT_CONDITION_FILE_INVALID")
        rows[condition_id] = payload
    return rows


def _validate_existing(
    rows: Mapping[str, Mapping[str, Any]],
    *,
    expected_ids: set[str],
    campaign_id: str,
    environment: str,
    protocol: str,
    frame_identity: Mapping[str, object],
) -> None:
    if not set(rows).issubset(expected_ids):
        raise RuntimeError("JOINT_FINGERPRINT_RESUME_CONDITION_UNKNOWN")
    for condition_id, row in rows.items():
        if (
            row.get("condition_id") != condition_id
            or row.get("campaign_id") != campaign_id
            or row.get("environment") != environment
            or row.get("protocol") != protocol
            or row.get("frame_identity") != frame_identity
        ):
            raise RuntimeError("JOINT_FINGERPRINT_RESUME_FRAME_MISMATCH")


def run(args: argparse.Namespace) -> dict[str, object]:
    joint_path = args.joint_campaign.resolve(strict=True)
    joint = load_joint_campaign(joint_path)
    sources = load_joint_sources(joint, repo_root=REPO_ROOT)
    source_by_id = {str(source["probe"]["probe_id"]): source for source in sources}
    reference = sources[0]
    if args.environment not in reference["environments"]:
        raise RuntimeError(f"JOINT_FINGERPRINT_ENVIRONMENT_UNKNOWN:{args.environment}")
    protocol = reference["protocols"][args.protocol]
    env_spec = reference["environments"][args.environment]
    vendor_root = args.vendor_root.resolve(strict=True)
    checkpoint = (args.checkpoint_root / env_spec["checkpoint_dir"] / "latest.pt").resolve(strict=True)
    config_path = (vendor_root / env_spec["config"]).resolve(strict=True)
    evaluator_path = (vendor_root / "eval.py").resolve(strict=True)
    vae_path = args.vae_path.resolve(strict=True)
    data_root = args.data_root.resolve(strict=True)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    condition_root = output_root / "conditions"
    condition_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "status.json"
    schedule = condition_schedule(joint, sources)
    expected_ids = {str(row["condition_id"]) for row in schedule}

    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["ACWM_DATA_ROOT"] = str(data_root)
    os.chdir(vendor_root)
    if str(vendor_root) not in sys.path:
        sys.path.insert(0, str(vendor_root))
    official_eval = _load_eval_module(vendor_root)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["model_config"]["vae_config"] = [str(vae_path)]
    dataset_name = str(env_spec["dataset_name"])
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    started = time.time()
    model = official_eval.load_model(config, device)
    checkpoint_step = int(official_eval.load_checkpoint(model, str(checkpoint), device))
    dataset_kwargs = dict(config.get("dataset", {}))
    for key in ("name", "test_cuts", "train_size", "ind_test_size", "ood_test_size"):
        dataset_kwargs.pop(key, None)
    dataset = official_eval.RoboticsDatasetWrapper.get_dataset(
        dataset_name,
        split=str(protocol["split"]),
        max_trajs=int(protocol["max_trajs"]),
        test_cuts=int(protocol["test_cuts"]),
        **dataset_kwargs,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(protocol["batch_size"]),
        shuffle=False,
        num_workers=0,
    )
    batch = next(iter(loader))
    obs = batch["obs"].to(device)
    action = batch["action"].to(device)
    o_0 = obs[:, 0].permute(0, 2, 3, 1).contiguous()
    gt_video = obs.permute(0, 1, 3, 4, 2).contiguous()
    frame_identity = {
        "checkpoint_sha256": sha256_file(checkpoint),
        "config_sha256": sha256_file(config_path),
        "dataset_name": dataset_name,
        "split": str(protocol["split"]),
        "trajectory_batch_digest": _tensor_digest(obs, action),
        "evaluator_sha256": sha256_file(evaluator_path),
        "generation_mode": str(joint["generation_mode"]),
        "inference_steps": int(protocol["inference_steps"]),
    }
    existing = _load_measurements(condition_root)
    _validate_existing(
        existing,
        expected_ids=expected_ids,
        campaign_id=str(joint["campaign_id"]),
        environment=args.environment,
        protocol=args.protocol,
        frame_identity=frame_identity,
    )
    status: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "verdiwm-acwm-joint-fingerprint-environment-status",
        "campaign_id": joint["campaign_id"],
        "environment": args.environment,
        "protocol": args.protocol,
        "state": "running",
        "physical_gpu": args.physical_gpu,
        "condition_count": len(schedule),
        "completed_condition_count": len(existing),
        "frame_identity": frame_identity,
        "started_at_unix": started,
    }
    _atomic_json(status_path, status)

    for ordinal, condition in enumerate(schedule, start=1):
        condition_id = str(condition["condition_id"])
        if condition_id in existing:
            continue
        seed = int(condition["seed"])
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        probe_id = condition["source_probe_id"]
        dose = float(condition["dose"])
        source = source_by_id[str(probe_id)] if probe_id is not None else None
        context = (
            _dose_context(model, source, dose)
            if source is not None
            else nullcontext()
        )
        with context:
            with torch.no_grad():
                prediction = model.generate(
                    o_0,
                    action,
                    num_inference_steps=int(protocol["inference_steps"]),
                    noise_level=0.0,
                    mode=str(joint["generation_mode"]),
                )
        metrics = {
            str(key): float(value)
            for key, value in official_eval.compute_metrics(prediction, gt_video).items()
        }
        compile_receipt = (
            compile_probe_receipt(source, dose=dose)
            if source is not None
            else {
                "schema_version": 1,
                "artifact_type": "verdiwm-joint-no-hook-baseline-receipt",
                "compiled": True,
                "control_condition": True,
                "hook_policy": "no_hook_context",
                "blockers": [],
            }
        )
        measurement = {
            "schema_version": 1,
            "artifact_type": "verdiwm-acwm-joint-probe-measurement",
            "campaign_id": joint["campaign_id"],
            "environment": args.environment,
            "dataset_name": dataset_name,
            "protocol": args.protocol,
            **condition,
            "ordinal": ordinal,
            "metrics": metrics,
            "compile_receipt": compile_receipt,
            "frame_identity": frame_identity,
        }
        _atomic_json(condition_root / f"{condition_id}.json", measurement)
        existing[condition_id] = measurement
        status["completed_condition_count"] = len(existing)
        status["last_condition_id"] = condition_id
        _atomic_json(status_path, status)

    ordered_measurements = [existing[str(condition["condition_id"])] for condition in schedule]
    fit = fit_joint_fingerprint(
        joint,
        sources,
        environment=args.environment,
        measurements=ordered_measurements,
    )
    measurement_path = output_root / "measurements.jsonl"
    measurement_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in ordered_measurements),
        encoding="utf-8",
    )
    chart_path = output_root / "response-chart.json"
    chart_path.write_text(
        json.dumps(fit.chart.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    asset = compose_joint_irg_asset(
        joint,
        sources,
        fit,
        environment=args.environment,
        checkpoint_step=checkpoint_step,
        locality_threshold=float(joint["locality_threshold"]),
        provenance={
            "joint_campaign_sha256": sha256_file(joint_path),
            "measurement_sha256": sha256_file(measurement_path),
            **frame_identity,
        },
    )
    validate_irg_asset(asset)
    asset_path = output_root / "irg-asset.json"
    asset_path.write_text(json.dumps(asset, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "artifact_type": "verdiwm-acwm-joint-fingerprint-environment-manifest",
        "state": "ready",
        "campaign_id": joint["campaign_id"],
        "environment": args.environment,
        "protocol": args.protocol,
        "physical_gpu": args.physical_gpu,
        "checkpoint": str(checkpoint),
        "checkpoint_step": checkpoint_step,
        "frame_identity": frame_identity,
        "source_probe_ids": list(source_by_id),
        "semantic_path_names": list(fit.chart.intervention_names),
        "condition_count": len(schedule),
        "baseline_condition_count": sum(row["condition_kind"] == "baseline" for row in schedule),
        "measurement_sha256": sha256_file(measurement_path),
        "response_chart_sha256": sha256_file(chart_path),
        "irg_asset_sha256": sha256_file(asset_path),
        "joint_baseline_group_count": asset["covariance_contract"]["joint_baseline_group_count"],
        "routing_state": asset["routing_state"],
        "transfer_state": asset["transfer_state"],
        "elapsed_seconds": round(time.time() - started, 6),
        "claim_boundary": joint["claim_scope"],
    }
    _atomic_json(output_root / "manifest.json", manifest)
    status["state"] = "ready"
    status["completed_condition_count"] = len(schedule)
    status["completed_at_unix"] = time.time()
    _atomic_json(status_path, status)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint-campaign", type=Path, required=True)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--protocol", choices=("smoke", "pilot", "paper"), default="smoke")
    parser.add_argument("--vendor-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--vae-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run(args), sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
