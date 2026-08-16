#!/usr/bin/env python3
"""Select the largest admitted Ctrl-World per-device batch size."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


class ResourceTuningError(ValueError):
    """Resource smoke receipts do not form one valid tuning ladder."""


def _load(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResourceTuningError(f"RESOURCE_TUNING_JSON_INVALID:{path}") from exc
    if not isinstance(payload, Mapping):
        raise ResourceTuningError(f"RESOURCE_TUNING_JSON_INVALID:{path}")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def settle(*, plan_path: Path, receipt_paths: Sequence[Path], output_root: Path) -> dict[str, object]:
    if not receipt_paths or output_root.exists() or output_root.is_symlink():
        raise ResourceTuningError("RESOURCE_TUNING_ARGUMENT_INVALID")
    plan_path = plan_path.resolve(strict=True)
    plan = _load(plan_path)
    if (
        plan.get("artifact_type") != "verdiwm-ctrl-world-resource-tuning-plan"
        or plan.get("state") != "frozen_before_bs4_execution"
        or not isinstance(plan.get("next_rung"), Mapping)
    ):
        raise ResourceTuningError("RESOURCE_TUNING_PLAN_INVALID")

    receipts: dict[int, dict[str, object]] = {}
    invariant_frame: tuple[object, ...] | None = None
    for raw_path in receipt_paths:
        path = Path(raw_path).resolve(strict=True)
        payload = _load(path)
        batch_size = payload.get("per_device_batch_size")
        ranks = payload.get("rank_receipts")
        if (
            payload.get("artifact_type") != "ctrl-world-multiscale-history-adapter-distributed-smoke"
            or payload.get("state") != "passed"
            or not isinstance(batch_size, int)
            or batch_size < 1
            or batch_size in receipts
            or payload.get("completed_optimizer_steps") != 1
            or payload.get("all_official_checkpoint_tensors_unchanged") is not True
            or not isinstance(ranks, list)
            or len(ranks) != int(payload.get("world_size", 0))
        ):
            raise ResourceTuningError("RESOURCE_TUNING_RECEIPT_INVALID")
        frame = (
            payload.get("checkpoint"),
            payload.get("world_size"),
            payload.get("mixed_precision"),
            payload.get("optimizer"),
            payload.get("trainable_parameter_count"),
        )
        if invariant_frame is None:
            invariant_frame = frame
        elif frame != invariant_frame:
            raise ResourceTuningError("RESOURCE_TUNING_RECEIPT_FRAME_MISMATCH")
        fractions: list[float] = []
        for rank in ranks:
            if not isinstance(rank, Mapping):
                raise ResourceTuningError("RESOURCE_TUNING_RANK_RECEIPT_INVALID")
            fraction = float(rank.get("peak_memory_fraction"))
            if (
                not math.isfinite(fraction)
                or fraction <= 0.0
                or rank.get("absolute_identity_loss_difference") != 0.0
                or int(rank.get("backbone_mismatch_count", -1)) != 0
                or float(rank.get("history_corrector_gradient_norm", 0.0)) <= 0.0
                or float(rank.get("adapter_parameter_l1_delta", 0.0)) <= 0.0
            ):
                raise ResourceTuningError("RESOURCE_TUNING_RANK_RECEIPT_INVALID")
            fractions.append(fraction)
        effective = int(payload.get("effective_global_batch_size", 0))
        world_size = int(payload.get("world_size", 0))
        if effective != batch_size * world_size:
            raise ResourceTuningError("RESOURCE_TUNING_GLOBAL_BATCH_INVALID")
        receipts[batch_size] = {
            "path": str(path),
            "sha256": _sha256(path),
            "per_device_batch_size": batch_size,
            "effective_global_batch_size": effective,
            "maximum_peak_memory_fraction": max(fractions),
            "all_ranks_below_90_percent_memory": max(fractions) < 0.9,
        }
    if set(receipts) != {1, 2, 4}:
        raise ResourceTuningError("RESOURCE_TUNING_LADDER_INCOMPLETE")
    evidence = plan.get("evidence")
    if (
        not isinstance(evidence, Mapping)
        or evidence.get("bs1_receipt_sha256") != receipts[1]["sha256"]
        or evidence.get("bs2_receipt_sha256") != receipts[2]["sha256"]
        or int(plan["next_rung"].get("per_device_batch_size", 0)) != 4
    ):
        raise ResourceTuningError("RESOURCE_TUNING_PLAN_HASH_MISMATCH")
    accepted = [batch for batch, row in receipts.items() if row["all_ranks_below_90_percent_memory"]]
    if not accepted:
        raise ResourceTuningError("RESOURCE_TUNING_NO_ACCEPTED_BATCH")
    selected = max(accepted)

    output_root.mkdir(mode=0o700, parents=True)
    report = {
        "schema_version": 1,
        "artifact_type": "verdiwm-ctrl-world-resource-tuning-settlement",
        "state": "settled",
        "decision": "use_selected_batch_for_64_step_screen",
        "candidate": plan.get("candidate"),
        "selected_per_device_batch_size": selected,
        "selected_effective_global_batch_size": receipts[selected]["effective_global_batch_size"],
        "memory_safety_threshold": 0.9,
        "plan": {"path": str(plan_path), "sha256": _sha256(plan_path)},
        "receipts": [receipts[batch] for batch in sorted(receipts)],
        "claim_boundary": "Infrastructure utilization tuning only; no predictive-quality or promotion claim.",
    }
    report_path = output_root / "resource-tuning-settlement.json"
    report_path.write_text(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "artifact_type": "verdiwm-ctrl-world-resource-tuning-manifest",
        "state": "settled",
        "selected_per_device_batch_size": selected,
        "selected_effective_global_batch_size": receipts[selected]["effective_global_batch_size"],
        "report_path": str(report_path.resolve()),
        "report_sha256": _sha256(report_path),
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    result = settle(
        plan_path=args.plan,
        receipt_paths=args.receipt,
        output_root=args.output_root.resolve(),
    )
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
