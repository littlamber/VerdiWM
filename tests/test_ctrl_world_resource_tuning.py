from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.settle_ctrl_world_resource_tuning import settle


def _write(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _receipt(path: Path, batch_size: int, fraction: float) -> Path:
    ranks = [
        {
            "peak_memory_fraction": fraction,
            "absolute_identity_loss_difference": 0.0,
            "backbone_mismatch_count": 0,
            "history_corrector_gradient_norm": 0.1,
            "adapter_parameter_l1_delta": 0.1,
        }
        for _ in range(8)
    ]
    return _write(
        path,
        {
            "artifact_type": "ctrl-world-multiscale-history-adapter-distributed-smoke",
            "state": "passed",
            "checkpoint": "checkpoint.pt",
            "world_size": 8,
            "mixed_precision": "bf16",
            "optimizer": "torch.optim.AdamW",
            "trainable_parameter_count": 10,
            "per_device_batch_size": batch_size,
            "effective_global_batch_size": batch_size * 8,
            "completed_optimizer_steps": 1,
            "all_official_checkpoint_tensors_unchanged": True,
            "rank_receipts": ranks,
        },
    )


def test_selects_largest_batch_below_memory_gate(tmp_path: Path) -> None:
    receipts = [
        _receipt(tmp_path / "bs1.json", 1, 0.4),
        _receipt(tmp_path / "bs2.json", 2, 0.55),
        _receipt(tmp_path / "bs4.json", 4, 0.86),
    ]
    plan = _write(
        tmp_path / "plan.json",
        {
            "artifact_type": "verdiwm-ctrl-world-resource-tuning-plan",
            "state": "frozen_before_bs4_execution",
            "candidate": "candidate",
            "evidence": {
                "bs1_receipt_sha256": _sha256(receipts[0]),
                "bs2_receipt_sha256": _sha256(receipts[1]),
            },
            "next_rung": {"per_device_batch_size": 4},
        },
    )

    manifest = settle(plan_path=plan, receipt_paths=receipts, output_root=tmp_path / "out")

    assert manifest["selected_per_device_batch_size"] == 4
    assert manifest["selected_effective_global_batch_size"] == 32
