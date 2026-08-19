#!/usr/bin/env python3
"""Freeze a held-out CBMA screen from a completed 64-step sequence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
CTRL_WORLD_ROOT = Path(os.environ.get("VERDIWM_CTRL_WORLD_ROOT", "/path/to/Ctrl-World"))
CONTEXTS = ROOT / "configs" / "probes" / "ctrl_world_fshc_heldout_contexts_v1.json"
EVALUATOR = ROOT / "scripts" / "run_ctrl_world_local_fingerprint_probe.py"
LAUNCHER = ROOT / "scripts" / "run_ctrl_world_fshc_heldout_evaluation.py"
AGGREGATOR = ROOT / "scripts" / "aggregate_ctrl_world_fshc_heldout_evaluation.py"


class CBMAHeldoutPlanError(ValueError):
    """Completed training artifacts cannot freeze one held-out screen."""


def _load(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CBMAHeldoutPlanError(f"CBMA_HELDOUT_JSON_INVALID:{path}") from exc
    if not isinstance(payload, Mapping):
        raise CBMAHeldoutPlanError(f"CBMA_HELDOUT_JSON_INVALID:{path}")
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
    sequence_path: Path,
    output_path: Path,
    evaluation_output_root: Path,
) -> dict[str, object]:
    if output_path.exists() or output_path.is_symlink() or evaluation_output_root.exists():
        raise CBMAHeldoutPlanError("CBMA_HELDOUT_OUTPUT_EXISTS")
    training_plan_path = training_plan_path.resolve(strict=True)
    sequence_path = sequence_path.resolve(strict=True)
    training = _load(training_plan_path)
    sequence = _load(sequence_path)
    if (
        training.get("artifact_type") != "ctrl-world-fshc-ablation-plan"
        or training.get("state") != "frozen_before_execution"
        or sequence.get("artifact_type") != "ctrl-world-fshc-ablation-sequence"
        or sequence.get("state") != "completed"
        or sequence.get("experiment_id") != training.get("experiment_id")
    ):
        raise CBMAHeldoutPlanError("CBMA_HELDOUT_TRAINING_NOT_COMPLETE")
    experiments = {
        str(row["id"]): row
        for row in training.get("experiments", ())
        if isinstance(row, Mapping) and isinstance(row.get("id"), str)
    }
    cells = sequence.get("cells")
    if not isinstance(cells, list) or set(experiments) != {str(row.get("cell_id")) for row in cells if isinstance(row, Mapping)}:
        raise CBMAHeldoutPlanError("CBMA_HELDOUT_CELL_COVERAGE_MISMATCH")

    official = Path(str(training["baseline"]["checkpoint"])).resolve(strict=True)
    models: list[dict[str, object]] = [
        {
            "id": "a0",
            "name": "official_checkpoint_no_training",
            "checkpoint": str(official),
            "checkpoint_size_bytes": official.stat().st_size,
            "checkpoint_sha256": _sha256(official),
            "enable_signed_history_correction": False,
            "unsigned_history_gate": False,
            "enable_multiscale_history_adapter": False,
            "multiscale_history_always_on": False,
        }
    ]
    for cell in cells:
        if (
            not isinstance(cell, Mapping)
            or cell.get("state") != "completed"
            or cell.get("return_code") != 0
        ):
            raise CBMAHeldoutPlanError("CBMA_HELDOUT_CELL_INCOMPLETE")
        cell_id = str(cell["cell_id"])
        experiment = experiments[cell_id]
        receipt = _load(Path(str(cell["receipt"])).resolve(strict=True))
        if receipt.get("state") != "completed" or receipt.get("return_code") != 0:
            raise CBMAHeldoutPlanError("CBMA_HELDOUT_CELL_RECEIPT_INVALID")
        checkpoint = (Path(str(cell["output_dir"])) / "checkpoint-64.pt").resolve(strict=True)
        flags = set(str(value) for value in experiment.get("flags", ()))
        models.append(
            {
                "id": cell_id,
                "name": experiment["name"],
                "checkpoint": str(checkpoint),
                "checkpoint_size_bytes": checkpoint.stat().st_size,
                "checkpoint_sha256": _sha256(checkpoint),
                "enable_signed_history_correction": "enable_signed_history_correction" in flags,
                "unsigned_history_gate": "unsigned_history_gate" in flags,
                "enable_multiscale_history_adapter": "enable_multiscale_history_adapter" in flags,
                "multiscale_history_always_on": "multiscale_history_always_on" in flags,
            }
        )
    model_ids = [str(row["id"]) for row in models]
    expected_ids = ["a0", "b2", "b3", "c1", "c2", "c3"]
    if model_ids != expected_ids:
        raise CBMAHeldoutPlanError("CBMA_HELDOUT_MODEL_ORDER_INVALID")

    plan = {
        "schema_version": 1,
        "artifact_type": "ctrl-world-fshc-heldout-evaluation-plan",
        "experiment_id": "ctrl-world-cbma-heldout-screen-64-v1",
        "state": "frozen_before_execution",
        "objective": "Test whether the counterfactual-benefit-routed multiscale adapter improves held-out autoregressive replay beyond the official model, scalar routes, reconstruction-only multiscale adaptation, and direction-only multiscale routing.",
        "claim_boundary": "One held-out episode with three paired seeds supports a 64-step mechanism screen only. A full pass licenses a separately frozen 512-step confirmation plan.",
        "source_training_plan": str(training_plan_path),
        "source_training_plan_sha256": _sha256(training_plan_path),
        "source_training_sequence": str(sequence_path),
        "source_training_sequence_sha256": _sha256(sequence_path),
        "output_root": str(evaluation_output_root.resolve()),
        "contexts_json": str(CONTEXTS.resolve(strict=True)),
        "contexts_sha256": _sha256(CONTEXTS),
        "seeds": [101, 202, 303],
        "dependencies": {
            "runtime_python": os.environ.get("VERDIWM_CTRL_WORLD_PYTHON", sys.executable),
            "evaluator": str(EVALUATOR.resolve(strict=True)),
            "evaluator_sha256": _sha256(EVALUATOR),
            "launcher": str(LAUNCHER.resolve(strict=True)),
            "launcher_sha256": _sha256(LAUNCHER),
            "aggregator": str(AGGREGATOR.resolve(strict=True)),
            "aggregator_sha256": _sha256(AGGREGATOR),
            # The freeze records external runtime bindings; the launcher performs
            # strict existence checks when the actual held-out job is admitted.
            "ctrl_world_root": str(CTRL_WORLD_ROOT.resolve()),
            "dataset_root": str((CTRL_WORLD_ROOT / "dataset_example" / "droid_subset").resolve()),
            "data_stat": str((CTRL_WORLD_ROOT / "dataset_meta_info" / "droid_subset" / "stat.json").resolve()),
            "svd_model_path": os.environ.get(
                "VERDIWM_SVD_MODEL_PATH", "/path/to/models/stable-video-diffusion-img2vid"
            ),
            "clip_model_path": os.environ.get(
                "VERDIWM_CLIP_MODEL_PATH", "/path/to/models/clip-vit-base-patch32"
            ),
        },
        "models": models,
        "phases": {
            "action_sensitivity": {
                "probe_id": "action_conditioning_scale",
                "doses": [-0.1, 0.0, 0.1],
                "interact_num": 4,
                "num_inference_steps": 4,
                "zero_reference_mode": "unwrapped",
                "models": expected_ids,
                "gpu_assignments": {model_id: index for index, model_id in enumerate(expected_ids)},
                "metric_definition": "Held-out quality plus normalized RGB response to paired action-embedding perturbations."
            },
            "routing": {
                "probe_id": "fshc_signed_history_gain",
                "fshc_dose_mode": "normalized_mechanism",
                "doses": [-0.99, 0.0, 0.99],
                "interact_num": 4,
                "num_inference_steps": 4,
                "zero_reference_mode": "probe-zero",
                "models": ["b2", "b3", "c2", "c3"],
                "gpu_assignments": {"b2": 0, "b3": 1, "c2": 2, "c3": 3},
                "outcome_weights": [1.0, 2.0, 2.0, 1.0, 1.0],
                "minimum_composite_benefit": 0.01,
                "active_probability_threshold": 0.5,
                "metric_definition": "Normalized mechanism endpoints compare scalar max-gain and multiscale max-scale interventions; learned use probability is scored against whether either endpoint beats no correction."
            }
        },
        "promotion_gate": {
            "candidate": "c3",
            "quality_comparators": ["a0", "b2", "b3", "c1", "c2"],
            "minimum_paired_seed_wins": 2,
            "action_sensitivity_references": ["a0", "b3"],
            "minimum_action_sensitivity_retention": 0.9,
            "routing_comparators": ["b3", "c2"],
            "maximum_harmful_routing_rate": 0.1,
            "pass_action": "Freeze a separate 512-step confirmation plan.",
            "fail_action": "Do not extend training; settle the failed axis and return to mechanism retrieval."
        }
    }
    output_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    output_path.write_text(json.dumps(plan, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "artifact_type": "ctrl-world-cbma-heldout-plan-manifest",
        "state": "frozen",
        "plan_path": str(output_path.resolve()),
        "plan_sha256": _sha256(output_path),
        "model_count": len(models),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-plan", type=Path, required=True)
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluation-output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    result = freeze(
        training_plan_path=args.training_plan,
        sequence_path=args.sequence,
        output_path=args.output.resolve(),
        evaluation_output_root=args.evaluation_output_root.resolve(),
    )
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
