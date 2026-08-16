#!/usr/bin/env python3
"""Freeze the repaired CCLVR v2 held-out plan from an audited training settlement."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
CTRL_WORLD_ROOT = Path(os.environ.get("VERDIWM_CTRL_WORLD_ROOT", "/path/to/Ctrl-World"))
TEMPLATE = ROOT / "configs" / "experiments" / "ctrl_world_cclvr_heldout_v1.json"
CONTEXTS = ROOT / "configs" / "probes" / "ctrl_world_fshc_heldout_contexts_v1.json"
EVALUATOR = ROOT / "scripts" / "evaluate_ctrl_world_cclvr_heldout_v1.py"
AGGREGATOR = ROOT / "scripts" / "aggregate_ctrl_world_cclvr_heldout_v1.py"
LAUNCHER = ROOT / "scripts" / "run_ctrl_world_cclvr_heldout_v1.py"
BASE_EVALUATOR = ROOT / "scripts" / "run_ctrl_world_local_fingerprint_probe.py"
CELL_ORDER = ("d1", "d2", "d3", "d4")
ROUTE_SCOPES = {"d1": "episode", "d2": "episode", "d3": "interaction", "d4": "interaction"}


class CCLVRHeldoutPlanError(ValueError):
    """Training evidence cannot safely freeze a CCLVR v2 held-out plan."""


def _load(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CCLVRHeldoutPlanError(f"CCLVR_HELDOUT_FREEZE_JSON_INVALID:{path}") from exc
    if not isinstance(payload, Mapping):
        raise CCLVRHeldoutPlanError(f"CCLVR_HELDOUT_FREEZE_JSON_INVALID:{path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def freeze(
    *,
    training_plan_path: Path,
    training_settlement_path: Path,
    output_path: Path,
    evaluation_output_root: Path,
    experiment_id: str = "ctrl-world-cclvr-heldout-screen-64-v2",
) -> dict[str, object]:
    if not experiment_id or any(character.isspace() for character in experiment_id):
        raise CCLVRHeldoutPlanError("CCLVR_HELDOUT_FREEZE_EXPERIMENT_ID_INVALID")
    if (
        output_path.exists()
        or output_path.is_symlink()
        or evaluation_output_root.exists()
        or evaluation_output_root.is_symlink()
    ):
        raise CCLVRHeldoutPlanError("CCLVR_HELDOUT_FREEZE_OUTPUT_EXISTS")
    training_plan_path = training_plan_path.resolve(strict=True)
    training_settlement_path = training_settlement_path.resolve(strict=True)
    training = _load(training_plan_path)
    settlement = _load(training_settlement_path)
    if (
        training.get("artifact_type") != "ctrl-world-cclvr-ablation-plan"
        or training.get("state") != "frozen_before_execution"
        or training.get("confirmation_authorized") is not False
        or settlement.get("artifact_type") != "ctrl-world-cclvr-64-step-training-settlement"
        or settlement.get("state") != "training_screen_completed"
        or settlement.get("promotion_episode_used") is not False
        or settlement.get("confirmation_authorized") is not False
        or str(settlement.get("promotion_episode")) != "1799"
        or Path(str(settlement.get("plan"))).resolve(strict=True) != training_plan_path
        or settlement.get("plan_sha256") != _sha256(training_plan_path)
    ):
        raise CCLVRHeldoutPlanError("CCLVR_HELDOUT_FREEZE_TRAINING_INVALID")
    adapter_source = (CTRL_WORLD_ROOT / "models" / "multiscale_history_adapter.py").resolve(strict=True)
    source_hashes = training.get("source_hashes")
    if (
        not isinstance(source_hashes, Mapping)
        or source_hashes.get("models/multiscale_history_adapter.py") != _sha256(adapter_source)
    ):
        raise CCLVRHeldoutPlanError("CCLVR_HELDOUT_FREEZE_ADAPTER_SOURCE_DRIFT")

    experiments = {
        str(row["id"]): row
        for row in training.get("experiments", ())
        if isinstance(row, Mapping) and isinstance(row.get("id"), str)
    }
    settled_cells = {
        str(row["cell_id"]): row
        for row in settlement.get("cells", ())
        if isinstance(row, Mapping) and isinstance(row.get("cell_id"), str)
    }
    if tuple(training.get("execution_order", ())) != CELL_ORDER or set(experiments) != set(CELL_ORDER):
        raise CCLVRHeldoutPlanError("CCLVR_HELDOUT_FREEZE_CELL_FRAME_INVALID")
    if set(settled_cells) != set(CELL_ORDER):
        raise CCLVRHeldoutPlanError("CCLVR_HELDOUT_FREEZE_CELL_FRAME_INVALID")

    cells: list[dict[str, object]] = []
    common = training["common_training"]
    for cell_id in CELL_ORDER:
        experiment = experiments[cell_id]
        settled = settled_cells[cell_id]
        scope = settled.get("checkpoint_scope")
        if (
            settled.get("optimization_steps") != 64
            or not isinstance(scope, Mapping)
            or scope.get("state") != "passed"
        ):
            raise CCLVRHeldoutPlanError(f"CCLVR_HELDOUT_FREEZE_CELL_INVALID:{cell_id}")
        checkpoint = Path(str(settled["checkpoint"])).resolve(strict=True)
        if settled.get("checkpoint_sha256") != _sha256(checkpoint):
            raise CCLVRHeldoutPlanError(f"CCLVR_HELDOUT_FREEZE_CHECKPOINT_HASH_MISMATCH:{cell_id}")
        bank_spec = experiment.get("supervision_bank")
        if not isinstance(bank_spec, Mapping):
            raise CCLVRHeldoutPlanError(f"CCLVR_HELDOUT_FREEZE_BANK_INVALID:{cell_id}")
        bank = Path(str(bank_spec.get("path"))).resolve(strict=True)
        if bank_spec.get("sha256") != _sha256(bank):
            raise CCLVRHeldoutPlanError(f"CCLVR_HELDOUT_FREEZE_BANK_HASH_MISMATCH:{cell_id}")
        cells.append(
            {
                "id": cell_id,
                "name": experiment["name"],
                "route_scope": ROUTE_SCOPES[cell_id],
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": settled["checkpoint_sha256"],
                "supervision_bank": str(bank),
                "supervision_bank_sha256": bank_spec["sha256"],
                "supervision_variant": bank_spec["variant"],
                "cclvr_value_hidden_dim": common["cclvr_value_hidden_dim"],
                "cclvr_policy_temperature": common["cclvr_policy_temperature"],
            }
        )

    plan = copy.deepcopy(dict(_load(TEMPLATE)))
    plan.update(
        {
            "schema_version": 2,
            "artifact_type": "verdiwm-ctrl-world-cclvr-heldout-plan-v2",
            "state": "frozen_before_execution",
            "experiment_id": experiment_id,
            "objective": "Test whether the repaired complete-residual CCLVR mechanism improves held-out autoregressive replay through cached hard negative-zero-positive multiscale routing.",
            "source_training_plan": str(training_plan_path),
            "source_training_plan_sha256": _sha256(training_plan_path),
            "source_training_settlement": str(training_settlement_path),
            "source_training_settlement_sha256": _sha256(training_settlement_path),
            "output_root": str(evaluation_output_root.resolve()),
            "contexts_json": str(CONTEXTS.resolve(strict=True)),
            "contexts_sha256": _sha256(CONTEXTS),
            "cells": cells,
            "confirmation_authorized": False,
            "operator_repair": "Route times signed dose scales the complete projected residual, including projection bias; zero dose is exact identity and signed endpoints are exact opposites.",
        }
    )
    dependencies = dict(plan["dependencies"])
    dependencies.update(
        {
            "ctrl_world_root": str(CTRL_WORLD_ROOT.resolve(strict=True)),
            "evaluator": str(EVALUATOR.resolve(strict=True)),
            "evaluator_sha256": _sha256(EVALUATOR),
            "aggregator": str(AGGREGATOR.resolve(strict=True)),
            "aggregator_sha256": _sha256(AGGREGATOR),
            "launcher": str(LAUNCHER.resolve(strict=True)),
            "launcher_sha256": _sha256(LAUNCHER),
            "base_evaluator": str(BASE_EVALUATOR.resolve(strict=True)),
            "base_evaluator_sha256": _sha256(BASE_EVALUATOR),
            "adapter_source": str(adapter_source),
            "adapter_source_sha256": _sha256(adapter_source),
        }
    )
    plan["dependencies"] = dependencies
    output_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    output_path.write_text(json.dumps(plan, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "artifact_type": "verdiwm-ctrl-world-cclvr-heldout-plan-manifest-v2",
        "state": "frozen",
        "plan_path": str(output_path.resolve()),
        "plan_sha256": _sha256(output_path),
        "cell_count": len(cells),
        "confirmation_authorized": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-plan", type=Path, required=True)
    parser.add_argument("--training-settlement", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluation-output-root", type=Path, required=True)
    parser.add_argument(
        "--experiment-id",
        default="ctrl-world-cclvr-heldout-screen-64-v2",
    )
    args = parser.parse_args(argv)
    result = freeze(
        training_plan_path=args.training_plan,
        training_settlement_path=args.training_settlement,
        output_path=args.output.resolve(),
        evaluation_output_root=args.evaluation_output_root.resolve(),
        experiment_id=args.experiment_id,
    )
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
