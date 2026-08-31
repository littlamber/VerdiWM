#!/usr/bin/env python3
"""Evaluate one receipt-bound Hybrid Memory candidate on frozen ACWM contexts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from wmloop.control.acwm_materialized_campaign import (  # noqa: E402
    candidate_binding_digest,
    sha256_file,
    validate_materialized_candidate,
)


class CtrlWorldHybridMemoryEvaluationError(RuntimeError):
    """The frozen candidate or evaluation surface could not be executed faithfully."""


def run_evaluation(args: argparse.Namespace) -> dict[str, object]:
    base_path = _require_file(args.base_evaluator, "HYBRID_MEMORY_BASE_EVALUATOR_INVALID")
    base = _load_module(base_path, "verdiwm_acwm_dual_v1")
    contract = base._load_mapping(args.contract)
    base._validate_contract(contract)
    candidate_path = _require_file(args.candidate, "HYBRID_MEMORY_CANDIDATE_INVALID")
    candidate = _load_mapping(candidate_path, "HYBRID_MEMORY_CANDIDATE_INVALID")
    try:
        validate_materialized_candidate(candidate)
    except ValueError as exc:
        raise CtrlWorldHybridMemoryEvaluationError(str(exc)) from exc
    adapter_path = _adapter_path(candidate)
    adapter = _load_module(adapter_path, "verdiwm_materialized_hybrid_memory")
    protocol = base._protocol_for_stage(contract, args.stage)
    paths = base._validate_paths(args)
    numpy, torch = base._runtime_dependencies()
    base._seed_everything(torch, int(contract["protocol"]["seed"]))
    rollout_module = base._load_rollout_module(paths["ctrl_world_root"])
    runtime_args = base._build_runtime_args(
        ctrl_world_root=paths["ctrl_world_root"],
        dataset_root=paths["dataset_root"],
        dataset_name=str(getattr(args, "dataset_name", "droid_subset")),
        data_stat=paths["data_stat"],
        checkpoint=paths["checkpoint"],
        svd_model=paths["svd_model"],
        clip_model=paths["clip_model"],
        interactions=max(
            int(protocol["paired_prediction_interactions"]),
            int(protocol["rollout_interactions"]),
        ),
        inference_steps=int(protocol["num_inference_steps"]),
        guidance_scale=1.0,
    )
    parameters = candidate["parameters"]
    assert isinstance(parameters, Mapping)
    if int(parameters["max_items"]) != int(getattr(runtime_args, "num_history")):
        raise CtrlWorldHybridMemoryEvaluationError(
            "HYBRID_MEMORY_MAX_ITEMS_CONDITION_MISMATCH"
        )
    agent = rollout_module.agent(runtime_args)

    paired_rows = []
    for offset, context in enumerate(protocol["paired_prediction_contexts"]):
        base._seed_everything(torch, int(contract["protocol"]["seed"]) + offset)
        paired_rows.append(
            _run_context(
                base=base,
                adapter=adapter,
                agent=agent,
                runtime_args=runtime_args,
                context=context,
                interactions=int(protocol["paired_prediction_interactions"]),
                generated_history=False,
                parameters=parameters,
                numpy=numpy,
                torch=torch,
            )
        )
    rollout_rows = []
    for offset, context in enumerate(protocol["rollout_contexts"]):
        base._seed_everything(
            torch, int(contract["protocol"]["seed"]) + 10_000 + offset
        )
        rollout_rows.append(
            _run_context(
                base=base,
                adapter=adapter,
                agent=agent,
                runtime_args=runtime_args,
                context=context,
                interactions=int(protocol["rollout_interactions"]),
                generated_history=True,
                parameters=parameters,
                numpy=numpy,
                torch=torch,
            )
        )
    metrics = base._aggregate_metrics(paired_rows, rollout_rows, numpy=numpy)
    provenance = candidate["provenance"]
    assert isinstance(provenance, Mapping)
    result = {
        "schema_version": 1,
        "artifact_type": "verdiwm-ctrl-world-acwm-measurement",
        "contract_id": contract["contract_id"],
        "contract_digest": contract["contract_digest"],
        "stage": args.stage,
        "scope": "acwm_only",
        "evidence_source": contract["evidence_source"],
        "candidate": candidate,
        "candidate_binding_sha256": candidate_binding_digest(candidate),
        "metrics": metrics,
        "protocol": protocol,
        "materialization_binding": {
            "implementation_revision": provenance["implementation_revision"],
            "materialization_receipt_sha256": provenance["materialization_receipt_sha256"],
            "descriptor_sha256": provenance["descriptor_sha256"],
            "source_assessment_sha256": provenance["source_assessment_sha256"],
            "source_digest": provenance["source_digest"],
            "assessment_digest": provenance["assessment_digest"],
            "required_file_sha256": {
                str(row["name"]): str(row["sha256"])
                for row in provenance["required_files"]
                if isinstance(row, Mapping)
            },
        },
        "evaluator_binding": {
            "implementation_sha256": sha256_file(Path(__file__).resolve()),
            "base_evaluator_sha256": sha256_file(base_path),
        },
        "asset_fingerprints": {
            name: base._fingerprint(path)
            for name, path in sorted(paths.items())
            if name != "ctrl_world_root"
        },
        "source_revision": base._source_revision(paths["ctrl_world_root"]),
        "paired_prediction": paired_rows,
        "giga_style_rollout": rollout_rows,
        "runtime": base._runtime_receipt(torch),
        "claim_boundary": (
            "Source-grounded Hybrid Memory mechanism-transfer measurement only. "
            "It is not a HyDRA reproduction and executes no policy or task-success signal."
        ),
    }
    base._validate_metrics(result["metrics"])
    _write_result(Path(args.output_root), result)
    return result


def _run_context(
    *,
    base: Any,
    adapter: Any,
    agent: object,
    runtime_args: object,
    context: Mapping[str, object],
    interactions: int,
    generated_history: bool,
    parameters: Mapping[str, object],
    numpy: Any,
    torch: Any,
) -> dict[str, object]:
    pred_step = int(getattr(runtime_args, "pred_step"))
    num_history = int(getattr(runtime_args, "num_history"))
    num_frames = int(getattr(runtime_args, "num_frames"))
    episode_id = str(context["episode_id"])
    start_idx = int(context["start_idx"])
    eef_gt, _joint_pos, _videos, video_latents, instruction = agent.get_traj_info(
        episode_id, start_idx=start_idx, steps=int(pred_step * interactions + 8)
    )
    first_latent = torch.cat([value[0] for value in video_latents], dim=1).unsqueeze(0)
    if tuple(first_latent.shape) != (1, 4, 72, 40):
        raise CtrlWorldHybridMemoryEvaluationError(
            "HYBRID_MEMORY_INITIAL_LATENT_SHAPE_INVALID"
        )
    latent_history = [first_latent for _ in range(num_history * 4)]
    eef_history = [eef_gt[0:1] for _ in range(num_history * 4)]
    true_batches: list[Any] = []
    prediction_batches: list[Any] = []
    interaction_errors: list[float] = []
    retrieval_trace: list[dict[str, object]] = []

    for interaction in range(interactions):
        start = interaction * (pred_step - 1)
        end = start + pred_step
        target_latents = [value[start:end] for value in video_latents]
        action = eef_gt[start:end]
        if tuple(action.shape) != (pred_step, 7):
            raise CtrlWorldHybridMemoryEvaluationError(
                "HYBRID_MEMORY_ACTION_WINDOW_INVALID"
            )
        current = latent_history[-1]
        history_tensor = torch.stack(latent_history, dim=1)
        action_tensor = torch.as_tensor(
            numpy.concatenate(eef_history, axis=0),
            device=current.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        query_action = torch.as_tensor(
            action[0], device=current.device, dtype=torch.float32
        ).unsqueeze(0)
        retrieved = adapter.retrieve_relevant_history(
            history_tensor,
            action_tensor,
            current,
            query_action,
            max_items=int(parameters["max_items"]),
            token_grid_size=int(parameters["token_grid_size"]),
            spatial_weight=float(parameters["spatial_weight"]),
            action_weight=float(parameters["action_weight"]),
            temporal_weight=float(parameters["temporal_weight"]),
        )
        history_cond = retrieved.history
        history_pose = retrieved.actions[0].detach().cpu().numpy()
        action_cond = numpy.concatenate((history_pose, action), axis=0)
        if tuple(action_cond.shape) != (num_history + num_frames, 7):
            raise CtrlWorldHybridMemoryEvaluationError(
                "HYBRID_MEMORY_CONDITION_WINDOW_INVALID"
            )
        if tuple(history_cond.shape[:2]) != (1, num_history):
            raise CtrlWorldHybridMemoryEvaluationError(
                "HYBRID_MEMORY_RETRIEVED_HISTORY_SHAPE_INVALID"
            )
        retrieval_trace.append(
            {
                "interaction": interaction,
                "available_history_items": len(latent_history),
                "selected_indices": retrieved.indices[0].detach().cpu().tolist(),
                "normalized_weights": [
                    float(value) for value in retrieved.weights[0].detach().cpu().tolist()
                ],
                "memory_token_shape": list(retrieved.memory_tokens.shape),
            }
        )
        _visual, true_video, prediction, predicted_latents = agent.forward_wm(
            action_cond,
            target_latents,
            current,
            his_cond=history_cond,
            text=instruction if bool(getattr(runtime_args, "text_cond")) else None,
        )
        error = base._rgb_l1(true_video, prediction, numpy=numpy)
        interaction_errors.append(error)
        true_batches.append(true_video)
        prediction_batches.append(prediction)
        eef_history.append(action[pred_step - 1 : pred_step])
        if generated_history:
            next_latent = torch.cat(
                [value[pred_step - 1] for value in predicted_latents], dim=1
            ).unsqueeze(0)
        else:
            next_latent = torch.cat(
                [value[end - 1] for value in video_latents], dim=1
            ).unsqueeze(0)
        latent_history.append(next_latent)

    true_all = numpy.concatenate(true_batches, axis=1)
    prediction_all = numpy.concatenate(prediction_batches, axis=1)
    return {
        "context": {"episode_id": episode_id, "start_idx": start_idx},
        "interactions": interactions,
        "mean_rgb_l1": base._rgb_l1(true_all, prediction_all, numpy=numpy),
        "motion_rgb_l1": base._motion_l1(true_all, prediction_all, numpy=numpy),
        "interaction_rgb_l1": interaction_errors,
        "horizon_drift_slope": base._slope(interaction_errors, numpy=numpy),
        "retrieval_trace": retrieval_trace,
    }


def _adapter_path(candidate: Mapping[str, object]) -> Path:
    provenance = candidate["provenance"]
    assert isinstance(provenance, Mapping)
    matches = [
        Path(str(row["path"]))
        for row in provenance["required_files"]
        if isinstance(row, Mapping) and row.get("name") == "hybrid_relevance_memory"
    ]
    if len(matches) != 1:
        raise CtrlWorldHybridMemoryEvaluationError(
            "HYBRID_MEMORY_ADAPTER_BINDING_INVALID"
        )
    return _require_file(matches[0], "HYBRID_MEMORY_ADAPTER_BINDING_INVALID")


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise CtrlWorldHybridMemoryEvaluationError("HYBRID_MEMORY_MODULE_LOAD_FAILED")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_mapping(path: Path, code: str) -> dict[str, object]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CtrlWorldHybridMemoryEvaluationError(code) from exc
    if not isinstance(payload, dict):
        raise CtrlWorldHybridMemoryEvaluationError(code)
    return payload


def _require_file(path: Path, code: str) -> Path:
    raw = Path(path).expanduser()
    if raw.is_symlink():
        raise CtrlWorldHybridMemoryEvaluationError(code)
    resolved = raw.resolve()
    if not resolved.is_file():
        raise CtrlWorldHybridMemoryEvaluationError(code)
    return resolved


def _write_result(destination: Path, result: Mapping[str, object]) -> None:
    destination = destination.expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise CtrlWorldHybridMemoryEvaluationError("HYBRID_MEMORY_OUTPUT_EXISTS")
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        payload = json.dumps(result, sort_keys=True, ensure_ascii=True, indent=2) + "\n"
        (temporary / "measurement.json").write_text(payload, encoding="utf-8")
        manifest = {
            "schema_version": 1,
            "artifact_type": "verdiwm-ctrl-world-acwm-materialized-measurement-manifest",
            "measurement_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
            "stage": result["stage"],
            "contract_digest": result["contract_digest"],
            "candidate_id": result["candidate"]["candidate_id"],
            "candidate_binding_sha256": result["candidate_binding_sha256"],
            "evaluator_binding": result["evaluator_binding"],
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-evaluator", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--stage", choices=("screen", "confirm"), required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--ctrl-world-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-name", default="droid_subset")
    parser.add_argument("--data-stat", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--svd-model", type=Path, required=True)
    parser.add_argument("--clip-model", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_evaluation(args)
    print(
        json.dumps(
            {"metrics": result["metrics"], "output_root": str(args.output_root)},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
