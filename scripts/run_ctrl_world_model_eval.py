#!/usr/bin/env python3
"""Run the real VerdiWM Ctrl-World paired ACWM evaluator from a model run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_ctrl_world_acwm_dual import run_evaluation


def run(manifest_path: Path) -> dict[str, object]:
    manifest = _load(manifest_path)
    assets = manifest["source"]["asset_bindings"]
    checkpoint_receipt = _load(Path(str(manifest["training"]["checkpoint_receipt"])))
    checkpoint = checkpoint_receipt.get("checkpoint_path")
    if not isinstance(checkpoint, str) or not Path(checkpoint).is_file():
        raise RuntimeError("CTRL_WORLD_CHECKPOINT_RECEIPT_INVALID")
    output = Path(str(manifest["output"]["root"])).resolve() / "acwm-evaluation"
    contract = ROOT / "configs/experiments/ctrl_world_acwm_dual_evaluation_v1.json"
    args = argparse.Namespace(
        contract=contract,
        stage="confirm",
        ctrl_world_root=Path(str(manifest["source"]["model_root"])),
        dataset_root=Path(str(assets["--dataset_root_path"])),
        data_stat=Path(str(assets["--data_stat_path"])),
        checkpoint=Path(checkpoint),
        svd_model=Path(str(assets["--svd_model_path"])),
        clip_model=Path(str(assets["--clip_model_path"])),
        output_root=output,
        candidate_id=str(manifest["model_run_id"]),
        guidance_scale=1.0,
    )
    measurement = run_evaluation(args)
    receipt_path = Path(str(manifest["evaluation"]["evidence_receipt"])).resolve()
    receipt = {
        "schema_version": 1,
        "artifact_type": "verdiwm-heldout-evidence-receipt",
        "state": "complete",
        "model_run_id": manifest["model_run_id"],
        "evaluator_contract": str(contract),
        "measurement_path": str(output / "measurement.json"),
        "metrics": measurement["metrics"],
        "evidence_source": "paired_ground_truth_rollout",
        "claim_boundary": "Held-out ACWM measurement only; promotion remains governed by the frozen verifier.",
    }
    receipt_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return receipt


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("CTRL_WORLD_MODEL_RUN_MANIFEST_INVALID")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verdiwm-model-run", type=Path, required=True)
    args = parser.parse_args(argv)
    run(args.verdiwm_model_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
