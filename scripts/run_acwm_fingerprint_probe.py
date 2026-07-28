#!/usr/bin/env python3
"""Run one ACWM-Phys environment through a paired action-conditioning probe."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
import time
from pathlib import Path
from types import MethodType

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wmloop.experiments.acwm_fingerprint import (
    compile_probe_receipt,
    fit_chart,
    load_campaign,
    sha256_file,
)


def _load_eval_module(vendor_root: Path):
    path = vendor_root / "eval.py"
    spec = importlib.util.spec_from_file_location("acwm_official_eval", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("ACWM_EVAL_IMPORT_FAILED")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ActionEmbeddingDose:
    def __init__(self, dynamics_model, dose: float) -> None:
        self.dose = float(dose)
        self.embedder = dynamics_model.model.action_embedder
        self.original = self.embedder.forward

    def __enter__(self):
        scale = 1.0 + self.dose
        original = self.original

        def scaled_forward(_module, action):
            return original(action) * scale

        self.embedder.forward = MethodType(scaled_forward, self.embedder)
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.embedder.forward = self.original
        return False


def run(args: argparse.Namespace) -> dict[str, object]:
    campaign_path = args.campaign.resolve(strict=True)
    campaign = load_campaign(campaign_path)
    env_spec = campaign["environments"][args.environment]
    protocol = campaign["protocols"][args.protocol]
    vendor_root = args.vendor_root.resolve(strict=True)
    checkpoint = (args.checkpoint_root / env_spec["checkpoint_dir"] / "latest.pt").resolve(strict=True)
    config_path = (vendor_root / env_spec["config"]).resolve(strict=True)
    vae_path = args.vae_path.resolve(strict=True)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["ACWM_DATA_ROOT"] = str(args.data_root.resolve(strict=True))
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
    checkpoint_step = official_eval.load_checkpoint(model, str(checkpoint), device)
    if not hasattr(model.model, "action_embedder"):
        raise RuntimeError("ACWM_ACTION_EMBEDDER_HOOK_MISSING")
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
    loader = DataLoader(dataset, batch_size=int(protocol["batch_size"]), shuffle=False, num_workers=0)
    batch = next(iter(loader))
    obs = batch["obs"].to(device)
    action = batch["action"].to(device)
    o_0 = obs[:, 0].permute(0, 2, 3, 1).contiguous()
    gt_video = obs.permute(0, 1, 3, 4, 2).contiguous()
    measurements: list[dict[str, object]] = []
    for seed in campaign["seeds"]:
        for dose in campaign["probe"]["doses"]:
            random.seed(int(seed))
            np.random.seed(int(seed))
            torch.manual_seed(int(seed))
            torch.cuda.manual_seed_all(int(seed))
            with ActionEmbeddingDose(model, float(dose)):
                with torch.no_grad():
                    prediction = model.generate(
                        o_0,
                        action,
                        num_inference_steps=int(protocol["inference_steps"]),
                        noise_level=0.0,
                        mode="parallel",
                    )
            metrics = official_eval.compute_metrics(prediction, gt_video)
            measurements.append(
                {
                    "schema_version": 1,
                    "artifact_type": "verdiwm-acwm-probe-measurement",
                    "campaign_id": campaign["campaign_id"],
                    "environment": args.environment,
                    "dataset_name": dataset_name,
                    "protocol": args.protocol,
                    "probe_id": campaign["probe"]["probe_id"],
                    "hook_type": campaign["probe"]["hook_type"],
                    "dose": float(dose),
                    "seed": int(seed),
                    "metrics": metrics,
                    "compile_receipt": compile_probe_receipt(campaign, dose=float(dose)),
                    "invariants": {name: True for name in campaign["probe"]["invariants"]},
                }
            )
    chart = fit_chart(campaign, environment=args.environment, measurements=measurements)
    measurement_path = output_root / "measurements.jsonl"
    measurement_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in measurements), encoding="utf-8"
    )
    chart_path = output_root / "response-chart.json"
    chart_path.write_text(json.dumps(chart, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "artifact_type": "verdiwm-acwm-fingerprint-environment-manifest",
        "state": "ready",
        "campaign_id": campaign["campaign_id"],
        "environment": args.environment,
        "protocol": args.protocol,
        "physical_gpu": args.physical_gpu,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_step": int(checkpoint_step),
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "measurement_count": len(measurements),
        "repeat_count": chart["repeat_count"],
        "probe_id": campaign["probe"]["probe_id"],
        "doses": campaign["probe"]["doses"],
        "seeds": campaign["seeds"],
        "elapsed_seconds": round(time.time() - started, 6),
        "measurement_sha256": sha256_file(measurement_path),
        "response_chart_sha256": sha256_file(chart_path),
        "claim_boundary": campaign["claim_scope"],
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--protocol", choices=("smoke", "pilot", "paper"), default="smoke")
    parser.add_argument("--vendor-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--vae-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args), sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
