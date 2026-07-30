#!/usr/bin/env python3
"""Evaluate one official Cosmos3 DROID forward-dynamics rollout."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image

from wmloop.contracts import validate_document
from wmloop.evaluate.adapters.cosmos3_predictive import evaluate_cosmos3_prediction_receipt
from wmloop.evaluate.cosmos3_paired_gt import compute_cosmos3_paired_metrics, sha256_file
from wmloop.primitives.adapters.cosmos3_hooks import (
    apply_action_probe,
    cosmos3_probe_dose_unit,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cosmos-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-path", type=Path, required=True)
    parser.add_argument("--split-name", choices=("dev", "accept"), required=True)
    parser.add_argument("--sample-index", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--sample-args", type=Path, required=True)
    parser.add_argument("--action-input", type=Path, required=True)
    parser.add_argument("--action-hook-receipt", type=Path)
    parser.add_argument(
        "--action-probe",
        choices=(
            "action_conditioning_scale",
            "action_dimension_anisotropy",
            "action_dimension_interaction",
            "action_embedding_temporal_mix",
            "action_translation_scale",
        ),
        default="action_conditioning_scale",
    )
    parser.add_argument("--action-dose", type=float, default=0.0)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    split = json.loads(args.split_path.read_text(encoding="utf-8"))
    validate_document("cosmos3_forward_dynamics_split", split)
    identity = {"sample_index": args.sample_index, "seed": args.seed}
    if identity not in split[args.split_name]:
        raise ValueError("COSMOS3_EVAL_IDENTITY_OUTSIDE_FROZEN_SPLIT")

    cosmos_root = args.cosmos_root.resolve()
    sys.path.insert(0, str(cosmos_root))
    from cosmos_framework.data.vfm.action.datasets import DROIDLeRobotDataset

    sample = DROIDLeRobotDataset(
        root=str(args.dataset_root.resolve()), chunk_length=16, mode="forward_dynamics"
    )[args.sample_index]
    if sample["mode"] != "forward_dynamics" or sample["viewpoint"] != "concat_view":
        raise ValueError("COSMOS3_EVAL_DATASET_MODE_OR_VIEWPOINT_INVALID")
    gt = sample["video"].permute(1, 2, 3, 0).cpu().numpy()
    expected_action = np.asarray(
        apply_action_probe(
            sample["action"].cpu().numpy().tolist(),
            probe_id=args.action_probe,
            dose=args.action_dose,
        ),
        dtype=np.float32,
    )

    sample_args = json.loads(args.sample_args.read_text(encoding="utf-8"))
    if sample_args.get("model_mode") != "forward_dynamics":
        raise ValueError("COSMOS3_EVAL_POLICY_MODE_FORBIDDEN")
    if int(sample_args.get("seed", -1)) != args.seed:
        raise ValueError("COSMOS3_EVAL_SEED_MISMATCH")
    if sample_args.get("view_point") != "concat_view":
        raise ValueError("COSMOS3_EVAL_VIEWPOINT_MISMATCH")
    action = np.asarray(json.loads(args.action_input.read_text(encoding="utf-8")), dtype=np.float32)
    if action.shape != (16, 10) or not np.allclose(action, expected_action, rtol=0.0, atol=1e-6):
        raise ValueError("COSMOS3_EVAL_ACTION_MISMATCH")
    hook_receipt = None
    if args.action_hook_receipt is not None:
        hook_receipt = json.loads(args.action_hook_receipt.read_text(encoding="utf-8"))
        if (
            hook_receipt.get("mode") != args.action_probe
            or float(hook_receipt.get("parameters", {}).get("dose", float("nan"))) != args.action_dose
            or hook_receipt.get("dose_unit") != cosmos3_probe_dose_unit(args.action_probe)
            or hook_receipt.get("output_sha256") != sha256_file(args.action_input)
            or hook_receipt.get("shape") != [16, 10]
        ):
            raise ValueError("COSMOS3_EVAL_ACTION_HOOK_RECEIPT_MISMATCH")
    if args.action_dose != 0.0 and hook_receipt is None:
        raise ValueError("COSMOS3_EVAL_ACTION_HOOK_RECEIPT_REQUIRED")

    rollout = iio.imread(args.rollout)
    metrics, alignment = compute_cosmos3_paired_metrics(ground_truth=gt, rollout=rollout)
    output_root = args.output_root.resolve()
    if output_root.exists() or output_root.is_symlink():
        raise ValueError("COSMOS3_EVAL_OUTPUT_EXISTS")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent))
    try:
        gt_path = temporary / "ground-truth.npy"
        condition_path = temporary / "conditioning.png"
        action_path = temporary / "action-input.json"
        hook_path = temporary / "action-hook-receipt.json"
        rollout_path = temporary / "rollout.mp4"
        receipt_path = temporary / "prediction-receipt.json"
        np.save(gt_path, gt, allow_pickle=False)
        Image.fromarray(gt[0]).save(condition_path)
        shutil.copyfile(args.action_input, action_path)
        if args.action_hook_receipt is not None:
            shutil.copyfile(args.action_hook_receipt, hook_path)
        shutil.copyfile(args.rollout, rollout_path)
        receipt = {
            "schema_version": 1,
            "artifact_type": "verdiwm-cosmos3-prediction-receipt",
            "evidence_source": "paired_ground_truth_rollout",
            "model_mode": "forward_dynamics",
            "split_id": split["split_id"],
            "split_name": args.split_name,
            "dataset_freeze_id": "cosmos3_droid_lerobot_cookbook_sample_v1",
            "sample_index": args.sample_index,
            "seed": args.seed,
            "viewpoint": "concat_view",
            "action_shape": [16, 10],
            "horizon_frames": 16,
            "metrics": metrics,
            "frame_alignment": alignment,
            "action_conditioned": True,
            "conditioning_ref": condition_path.name,
            "action_ref": action_path.name,
            "ground_truth_ref": gt_path.name,
            "rollout_ref": rollout_path.name,
            "sha256": {
                "action_input": sha256_file(action_path),
                "conditioning": sha256_file(condition_path),
                "ground_truth": sha256_file(gt_path),
                "rollout": sha256_file(rollout_path),
            },
            "evaluator_version": "cosmos3-paired-gt-v1",
        }
        if hook_receipt is not None:
            receipt["intervention_ref"] = hook_path.name
            receipt["intervention"] = {
                "probe_id": args.action_probe,
                "dose": args.action_dose,
                "dose_unit": cosmos3_probe_dose_unit(args.action_probe),
                "hook_receipt_sha256": sha256_file(hook_path),
            }
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        evaluate_cosmos3_prediction_receipt(
            receipt_path=receipt_path,
            heldout_split_path=args.split_path,
            split_name=args.split_name,
        )
        manifest = {
            "schema_version": 1,
            "artifact_type": "verdiwm-cosmos3-paired-gt-evaluation",
            "state": "ready",
            "identity": identity,
            "split": args.split_name,
            "receipt": receipt_path.name,
            "metrics": metrics,
            "claim_boundary": "Paired predictive metrics on one frozen official DROID window; no transfer or model-improvement claim.",
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, output_root)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
