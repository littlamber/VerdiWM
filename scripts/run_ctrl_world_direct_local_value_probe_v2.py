#!/usr/bin/env python3
"""Measure direct interaction-local multiscale endpoints for CCLVR supervision v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

try:
    from scripts import evaluate_ctrl_world_cclvr_heldout_v1 as direct
    from scripts import run_ctrl_world_local_fingerprint_probe as fingerprint
except ImportError:
    import evaluate_ctrl_world_cclvr_heldout_v1 as direct
    import run_ctrl_world_local_fingerprint_probe as fingerprint


class DirectLocalValueProbeError(RuntimeError):
    """A frozen direct local-value shard could not be measured faithfully."""


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _maximum_quality_difference(left: Mapping[str, object], right: Mapping[str, object]) -> float:
    left_interactions = left.get("interactions")
    right_interactions = right.get("interactions")
    if not isinstance(left_interactions, list) or not isinstance(right_interactions, list):
        raise DirectLocalValueProbeError("DIRECT_LOCAL_INTERACTIONS_INVALID")
    values = []
    for left_row, right_row in zip(left_interactions, right_interactions, strict=True):
        for key in ("mean_l1", "final_l1"):
            values.append(abs(float(left_row[key]) - float(right_row[key])))
    return max(values, default=0.0)


def run(args: argparse.Namespace) -> dict[str, object]:
    output_root = Path(args.output_root).resolve()
    if output_root.exists() or output_root.is_symlink():
        raise DirectLocalValueProbeError("DIRECT_LOCAL_OUTPUT_EXISTS")
    if (
        args.probe_id != "fshc_interaction_local_gain"
        or args.fshc_dose_mode != "normalized_mechanism"
        or args.zero_reference_mode != "probe-zero"
        or not bool(args.enable_multiscale_history_adapter)
        or int(args.interact_num) != 4
        or int(args.num_inference_steps) != 4
    ):
        raise DirectLocalValueProbeError("DIRECT_LOCAL_PROTOCOL_INVALID")
    doses = tuple(float(value) for value in args.doses)
    targets = tuple(int(value) for value in args.target_interactions)
    if doses != (-0.99, 0.0, 0.99) or targets != (0, 1, 2, 3):
        raise DirectLocalValueProbeError("DIRECT_LOCAL_GRID_INVALID")
    contexts_path = Path(args.contexts_json).resolve(strict=True)
    contexts = fingerprint.load_contexts(contexts_path, args.seeds, args.context_ids)
    if any(str(context["episode_id"]) == "1799" for context in contexts):
        raise DirectLocalValueProbeError("DIRECT_LOCAL_PROMOTION_EPISODE_FORBIDDEN")
    ctrl_world_root = Path(args.ctrl_world_root).resolve(strict=True)
    module = fingerprint._load_rollout_module(ctrl_world_root)
    runtime_args = fingerprint._runtime_args(
        dataset_root=Path(args.dataset_root).resolve(strict=True),
        data_stat=Path(args.data_stat).resolve(strict=True),
        svd_model_path=Path(args.svd_model_path).resolve(strict=True),
        clip_model_path=Path(args.clip_model_path).resolve(strict=True),
        ckpt_path=Path(args.ckpt_path).resolve(strict=True),
        interact_num=4,
        num_inference_steps=4,
        enable_signed_history_correction=True,
        unsigned_history_gate=False,
        enable_multiscale_history_adapter=True,
        multiscale_history_always_on=False,
    )
    output_root.mkdir(mode=0o700, parents=True)
    rollout_agent = module.agent(runtime_args)
    measurements: list[dict[str, object]] = []
    zero_checks: list[dict[str, object]] = []
    sample_visual: np.ndarray | None = None
    for context in contexts:
        prepared = direct._prepare_context(rollout_agent, runtime_args, context)
        zero, visual = direct._run_prepared_context(
            rollout_agent=rollout_agent,
            runtime_args=runtime_args,
            prepared=prepared,
            route_mode="fixed",
            route_scope="interaction",
            fixed_dose=0.0,
        )
        direct._validate_route_row(
            zero,
            interact_num=4,
            inference_steps=4,
            route_scope="interaction",
            fixed_dose=0.0,
            target_interaction=None,
        )
        zero.update(
            {
                "dose": 0.0,
                "target_interaction": None,
                "hook_audit": zero["route_audit"],
            }
        )
        measurements.append(zero)
        zero_checks.append(
            {
                "identity": zero["identity"],
                "target_interaction": None,
                "maximum_abs_multiscale_residual": zero["route_audit"]["maximum_abs_residual"],
                "state": "passed",
            }
        )
        sample_visual = visual if sample_visual is None else sample_visual
        for interaction in targets:
            for dose in (doses[0], doses[2]):
                endpoint, _visual = direct._run_prepared_context(
                    rollout_agent=rollout_agent,
                    runtime_args=runtime_args,
                    prepared=prepared,
                    route_mode="fixed",
                    route_scope="interaction",
                    fixed_dose=dose,
                    target_interaction=interaction,
                )
                direct._validate_route_row(
                    endpoint,
                    interact_num=4,
                    inference_steps=4,
                    route_scope="interaction",
                    fixed_dose=dose,
                    target_interaction=interaction,
                )
                for prefix in range(interaction):
                    for key in ("mean_l1", "final_l1"):
                        if abs(
                            float(endpoint["interactions"][prefix][key])
                            - float(zero["interactions"][prefix][key])
                        ) > float(args.zero_identity_tolerance):
                            raise DirectLocalValueProbeError("DIRECT_LOCAL_PREFIX_CHANGED")
                endpoint.update(
                    {
                        "dose": dose,
                        "target_interaction": interaction,
                        "hook_audit": endpoint["route_audit"],
                    }
                )
                measurements.append(endpoint)
    if sample_visual is None:
        raise DirectLocalValueProbeError("DIRECT_LOCAL_VISUAL_MISSING")
    result = {
        "schema_version": 1,
        "artifact_type": "verdiwm-ctrl-world-local-fingerprint-probe-result",
        "state": "ready",
        "campaign_id": str(args.campaign_id),
        "probe_id": "fshc_interaction_local_gain",
        "base_probe_family": "direct_interaction_local_multiscale_endpoint_v2",
        "outcome_names": list(fingerprint.OUTCOME_NAMES),
        "input": {
            "checkpoint": str(Path(args.ckpt_path).resolve()),
            "checkpoint_sha256": _sha256(Path(args.ckpt_path)),
            "contexts_json": str(contexts_path),
            "contexts_sha256": _sha256(contexts_path),
            "doses": list(doses),
            "interact_num": 4,
            "num_inference_steps": 4,
            "selected_identities": [
                {"context_id": str(context["context_id"]), "seed": int(context["seed"])}
                for context in contexts
            ],
            "enable_multiscale_history_adapter": True,
            "fshc_dose_mode": "normalized_mechanism",
            "zero_reference_mode": "probe-zero",
            "target_interactions": list(targets),
            "endpoint_semantics": "direct_adapter_scale_override_after_projection_v2",
        },
        "runtime": fingerprint._runtime_receipt(),
        "unwrapped_references": [],
        "measurements": measurements,
        "zero_identity_checks": zero_checks,
        "hook_activation": {"state": "passed"},
        "artifacts": {"zero_dose_rollout": "zero-dose-rollout.mp4"},
        "claim_boundary": (
            "Development-only direct endpoint measurement. Episode 1799 is excluded and no held-out "
            "quality or promotion claim is made."
        ),
    }
    import mediapy

    try:
        import imageio_ffmpeg

        mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
    except ImportError:
        pass
    mediapy.write_video(output_root / "zero-dose-rollout.mp4", sample_visual, fps=4)
    _atomic_json(output_root / "result.json", result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--probe-id", required=True)
    parser.add_argument("--ctrl-world-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--data-stat", type=Path, required=True)
    parser.add_argument("--svd-model-path", type=Path, required=True)
    parser.add_argument("--clip-model-path", type=Path, required=True)
    parser.add_argument("--ckpt-path", type=Path, required=True)
    parser.add_argument("--contexts-json", type=Path, required=True)
    parser.add_argument("--doses", type=float, nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="*")
    parser.add_argument("--context-ids", type=str, nargs="*")
    parser.add_argument("--target-interactions", type=int, nargs="+", required=True)
    parser.add_argument("--interact-num", type=int, required=True)
    parser.add_argument("--num-inference-steps", type=int, required=True)
    parser.add_argument("--fshc-dose-mode", required=True)
    parser.add_argument("--zero-reference-mode", required=True)
    parser.add_argument("--enable-multiscale-history-adapter", action="store_true")
    parser.add_argument("--zero-identity-tolerance", type=float, default=1e-6)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    result = run(_parser().parse_args(argv))
    print(json.dumps({"state": result["state"], "campaign_id": result["campaign_id"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
