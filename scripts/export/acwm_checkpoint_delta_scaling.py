#!/usr/bin/env python3
"""Create auditable checkpoint interpolation candidates for ACWM evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


class CheckpointDeltaScalingError(RuntimeError):
    """Checkpoint interpolation could not be completed without ambiguity."""


def run_checkpoint_delta_scaling(
    *,
    output_root: Path,
    baseline_checkpoint: Path,
    candidate_checkpoint: Path,
    environment: str,
    source_primitive: str,
    seed: int,
    alphas: Sequence[float],
) -> dict[str, object]:
    destination = Path(output_root).resolve()
    baseline_path = Path(baseline_checkpoint).resolve()
    candidate_path = Path(candidate_checkpoint).resolve()
    if destination.exists() or destination.is_symlink():
        raise CheckpointDeltaScalingError("CHECKPOINT_DELTA_SCALING_OUTPUT_EXISTS")
    if not environment or not source_primitive or seed < 1:
        raise CheckpointDeltaScalingError("CHECKPOINT_DELTA_SCALING_ARGUMENT_INVALID")
    normalized_alphas = _validate_alphas(alphas)
    for path in (baseline_path, candidate_path):
        if path.is_symlink() or not path.is_file():
            raise CheckpointDeltaScalingError(f"CHECKPOINT_DELTA_SCALING_INPUT_INVALID:{path}")

    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    try:
        temporary.mkdir(mode=0o700, parents=True)
        baseline = _load_checkpoint(baseline_path)
        candidate = _load_checkpoint(candidate_path)
        baseline_state = _model_state(baseline, label="baseline")
        candidate_state = _model_state(candidate, label="candidate")
        _validate_state_compatibility(baseline_state, candidate_state)

        outputs: list[dict[str, object]] = []
        for alpha in normalized_alphas:
            output_path = temporary / f"alpha_{alpha:.2f}.pt"
            scaled_state = _scaled_state(baseline_state, candidate_state, alpha=alpha)
            payload = {
                "model_state_dict": scaled_state,
                "optimizer_state_dict": {},
                "step": candidate.get("step", baseline.get("step", 0)),
                "epoch": candidate.get("epoch", baseline.get("epoch", 0)),
                "wandb_run_id": candidate.get("wandb_run_id", ""),
                "wmloop_delta_scaling": {
                    "alpha": alpha,
                    "baseline_checkpoint_sha256": _sha256(baseline_path),
                    "candidate_checkpoint_sha256": _sha256(candidate_path),
                    "rule": "theta_scaled = theta_baseline + alpha * (theta_candidate - theta_baseline)",
                },
            }
            torch.save(payload, output_path)
            outputs.append(
                {
                    "alpha": alpha,
                    "path": str(destination / output_path.name),
                    "sha256": _sha256(output_path),
                    "size_bytes": output_path.stat().st_size,
                }
            )

        report = {
            "schema_version": 1,
            "artifact_type": "wmloop-checkpoint-delta-scaling",
            "state": "ready",
            "environment": environment,
            "source_primitive": source_primitive,
            "seed": seed,
            "baseline_checkpoint": str(baseline_path),
            "baseline_sha256": _sha256(baseline_path),
            "candidate_checkpoint": str(candidate_path),
            "candidate_sha256": _sha256(candidate_path),
            "rule": "theta_scaled = theta_baseline + alpha * (theta_candidate - theta_baseline)",
            "outputs": outputs,
            "intent_to_code_contract": {
                "method_intent": "Apply a signed scale to the exact learned checkpoint update; negative alpha reflects a verifier-identified harmful direction.",
                "runtime_behavior": "Interpolate floating model tensors only; require all keys, shapes, and dtypes to match.",
                "not_claimed": [
                    "does not alter the frozen evaluator",
                    "does not interpolate optimizer state",
                    "does not establish a quality gain without the official gate",
                ],
            },
            "claim_boundary": "CPU-side checkpoint transform only; no quality claim until the frozen official gate passes.",
        }
        _write_json(temporary / "manifest.json", report)
        os.replace(temporary, destination)
        return report
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _validate_alphas(alphas: Sequence[float]) -> tuple[float, ...]:
    normalized: list[float] = []
    for raw in alphas:
        alpha = float(raw)
        if not math.isfinite(alpha) or alpha == 0.0 or alpha < -1.0 or alpha > 1.0:
            raise CheckpointDeltaScalingError("CHECKPOINT_DELTA_SCALING_ALPHA_INVALID")
        if alpha not in normalized:
            normalized.append(alpha)
    if not normalized:
        raise CheckpointDeltaScalingError("CHECKPOINT_DELTA_SCALING_ALPHA_INVALID")
    return tuple(sorted(normalized))


def _load_checkpoint(path: Path) -> Mapping[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(payload, Mapping):
        raise CheckpointDeltaScalingError("CHECKPOINT_DELTA_SCALING_PAYLOAD_INVALID")
    return payload


def _model_state(payload: Mapping[str, Any], *, label: str) -> Mapping[str, torch.Tensor]:
    state = payload.get("model_state_dict")
    if not isinstance(state, Mapping) or not state:
        raise CheckpointDeltaScalingError(f"CHECKPOINT_DELTA_SCALING_MODEL_STATE_INVALID:{label}")
    if any(not isinstance(key, str) or not isinstance(value, torch.Tensor) for key, value in state.items()):
        raise CheckpointDeltaScalingError(f"CHECKPOINT_DELTA_SCALING_MODEL_STATE_INVALID:{label}")
    return state


def _validate_state_compatibility(
    baseline: Mapping[str, torch.Tensor], candidate: Mapping[str, torch.Tensor]
) -> None:
    if tuple(baseline) != tuple(candidate):
        raise CheckpointDeltaScalingError("CHECKPOINT_DELTA_SCALING_KEYS_MISMATCH")
    for key in baseline:
        left = baseline[key]
        right = candidate[key]
        if left.shape != right.shape or left.dtype != right.dtype:
            raise CheckpointDeltaScalingError(f"CHECKPOINT_DELTA_SCALING_TENSOR_MISMATCH:{key}")


def _scaled_state(
    baseline: Mapping[str, torch.Tensor],
    candidate: Mapping[str, torch.Tensor],
    *,
    alpha: float,
) -> dict[str, torch.Tensor]:
    output: dict[str, torch.Tensor] = {}
    for key, baseline_tensor in baseline.items():
        candidate_tensor = candidate[key]
        if baseline_tensor.is_floating_point() or baseline_tensor.is_complex():
            output[key] = torch.lerp(baseline_tensor, candidate_tensor, alpha).contiguous()
        else:
            if not torch.equal(baseline_tensor, candidate_tensor):
                raise CheckpointDeltaScalingError(f"CHECKPOINT_DELTA_SCALING_NONFLOAT_MISMATCH:{key}")
            output[key] = baseline_tensor.clone()
    return output


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--source-primitive", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--alpha", type=float, action="append", required=True)
    args = parser.parse_args(argv)
    report = run_checkpoint_delta_scaling(
        output_root=args.output_root,
        baseline_checkpoint=args.baseline_checkpoint,
        candidate_checkpoint=args.candidate_checkpoint,
        environment=args.environment,
        source_primitive=args.source_primitive,
        seed=args.seed,
        alphas=args.alpha,
    )
    print(json.dumps(report, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
