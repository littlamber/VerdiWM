#!/usr/bin/env python3
"""Run one ACWM-Phys environment through a paired inference-only probe."""

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


class ActionEmbeddingTemporalMixDose:
    """Scale temporal action contrast while preserving its trajectory mean."""

    def __init__(self, dynamics_model, dose: float) -> None:
        self.dose = float(dose)
        self.embedder = dynamics_model.model.action_embedder
        self.original = self.embedder.forward

    def __enter__(self):
        dose = self.dose
        original = self.original

        def mixed_forward(_module, action):
            embedding = original(action)
            if embedding.ndim < 3:
                raise RuntimeError("ACWM_ACTION_EMBEDDING_TEMPORAL_SHAPE_INVALID")
            temporal_mean = embedding.mean(dim=1, keepdim=True)
            return embedding + dose * (temporal_mean - embedding)

        self.embedder.forward = MethodType(mixed_forward, self.embedder)
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.embedder.forward = self.original
        return False


class ActionEmbeddingEventAlignmentDose:
    """Dose mean-preserving action-embedding contrast at action transitions."""

    def __init__(self, dynamics_model, dose: float) -> None:
        self.dose = float(dose)
        self.embedder = dynamics_model.model.action_embedder
        self.original = self.embedder.forward

    def __enter__(self):
        dose = self.dose
        original = self.original

        def aligned_forward(_module, action):
            embedding = original(action)
            if action.ndim != 3 or embedding.ndim != 3 or action.shape[0] != embedding.shape[0]:
                raise RuntimeError("ACWM_ACTION_EVENT_ALIGNMENT_SHAPE_INVALID")
            action_delta = torch.cat(
                [torch.zeros_like(action[:, :1]), action[:, 1:] - action[:, :-1]], dim=1
            ).abs().mean(dim=-1)
            event_scale = action_delta.amax(dim=1, keepdim=True)
            event_weight = torch.where(
                event_scale > 1e-12,
                action_delta / event_scale.clamp_min(1e-12),
                torch.zeros_like(action_delta),
            )
            if event_weight.shape[1] != embedding.shape[1]:
                event_weight = torch.nn.functional.interpolate(
                    event_weight.unsqueeze(1),
                    size=embedding.shape[1],
                    mode="linear",
                    align_corners=True,
                ).squeeze(1)
            temporal_mean = embedding.mean(dim=1, keepdim=True)
            perturbation = event_weight.unsqueeze(-1).to(embedding.dtype) * (
                embedding - temporal_mean
            )
            perturbation = perturbation - perturbation.mean(dim=1, keepdim=True)
            return embedding + dose * perturbation

        self.embedder.forward = MethodType(aligned_forward, self.embedder)
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.embedder.forward = self.original
        return False


class AutoregressiveHistoryDose:
    """Scale generated latent history around the first conditioned latent."""

    def __init__(self, dynamics_model, dose: float) -> None:
        self.dose = float(dose)
        self.dit = dynamics_model.model
        self.original = self.dit.forward

    def __enter__(self):
        scale = 1.0 + self.dose
        original = self.original

        def scaled_forward(_module, z, t, action):
            if z.shape[1] <= 1:
                return original(z, t, action)
            anchor = z[:, :1]
            history = anchor + scale * (z[:, :-1] - anchor)
            return original(torch.cat([history, z[:, -1:]], dim=1), t, action)

        self.dit.forward = MethodType(scaled_forward, self.dit)
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.dit.forward = self.original
        return False


class AutoregressiveHistoryTemporalMixDose:
    """Mix each generated history state toward its immediate predecessor."""

    def __init__(self, dynamics_model, dose: float) -> None:
        self.dose = float(dose)
        self.dit = dynamics_model.model
        self.original = self.dit.forward

    def __enter__(self):
        dose = self.dose
        original = self.original

        def mixed_forward(_module, z, t, action):
            history = z[:, :-1]
            if history.shape[1] <= 1:
                return original(z, t, action)
            mixed_tail = history[:, 1:] + dose * (history[:, :-1] - history[:, 1:])
            mixed_history = torch.cat([history[:, :1], mixed_tail], dim=1)
            return original(torch.cat([mixed_history, z[:, -1:]], dim=1), t, action)

        self.dit.forward = MethodType(mixed_forward, self.dit)
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.dit.forward = self.original
        return False


class AutoregressiveLatestFeedbackDose:
    """Mix only the latest generated history state toward its predecessor."""

    def __init__(self, dynamics_model, dose: float) -> None:
        self.dose = float(dose)
        self.dit = dynamics_model.model
        self.original = self.dit.forward

    def __enter__(self):
        dose = self.dose
        original = self.original

        def mixed_forward(_module, z, t, action):
            history = z[:, :-1]
            if history.shape[1] <= 1:
                return original(z, t, action)
            latest = history[:, -1:] + dose * (history[:, -2:-1] - history[:, -1:])
            mixed_history = torch.cat([history[:, :-1], latest], dim=1)
            return original(torch.cat([mixed_history, z[:, -1:]], dim=1), t, action)

        self.dit.forward = MethodType(mixed_forward, self.dit)
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.dit.forward = self.original
        return False


class AutoregressiveTeacherRecoveryDose:
    """Mix generated history toward paired teacher latents at a small dose."""

    def __init__(self, dynamics_model, dose: float, teacher_history: torch.Tensor) -> None:
        self.dose = float(dose)
        self.teacher_history = teacher_history.detach()
        self.dit = dynamics_model.model
        self.original = self.dit.forward

    def __enter__(self):
        dose = self.dose
        teacher_history = self.teacher_history
        original = self.original

        def recovered_forward(_module, z, t, action):
            history = z[:, :-1]
            if history.shape[1] <= 1:
                return original(z, t, action)
            if (
                teacher_history.shape[0] != history.shape[0]
                or teacher_history.shape[1] < history.shape[1]
                or teacher_history.shape[2:] != history.shape[2:]
            ):
                raise RuntimeError("ACWM_TEACHER_RECOVERY_HISTORY_SHAPE_INVALID")
            teacher = teacher_history[:, : history.shape[1]].to(
                device=history.device, dtype=history.dtype
            )
            recovered_tail = history[:, 1:] + dose * (teacher[:, 1:] - history[:, 1:])
            recovered_history = torch.cat([history[:, :1], recovered_tail], dim=1)
            return original(torch.cat([recovered_history, z[:, -1:]], dim=1), t, action)

        self.dit.forward = MethodType(recovered_forward, self.dit)
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.dit.forward = self.original
        return False


class AutoregressiveTeacherHorizonRecoveryDose(AutoregressiveTeacherRecoveryDose):
    """Recover later generated-history states more strongly than early states."""

    def __enter__(self):
        dose = self.dose
        teacher_history = self.teacher_history
        original = self.original

        def recovered_forward(_module, z, t, action):
            history = z[:, :-1]
            if history.shape[1] <= 1:
                return original(z, t, action)
            if (
                teacher_history.shape[0] != history.shape[0]
                or teacher_history.shape[1] < history.shape[1]
                or teacher_history.shape[2:] != history.shape[2:]
            ):
                raise RuntimeError("ACWM_TEACHER_HORIZON_RECOVERY_HISTORY_SHAPE_INVALID")
            teacher = teacher_history[:, : history.shape[1]].to(
                device=history.device, dtype=history.dtype
            )
            shape = (1, history.shape[1]) + (1,) * (history.ndim - 2)
            horizon_weight = torch.linspace(
                0.0,
                1.0,
                steps=history.shape[1],
                device=history.device,
                dtype=history.dtype,
            ).reshape(shape)
            recovered_history = history + dose * horizon_weight * (teacher - history)
            return original(torch.cat([recovered_history, z[:, -1:]], dim=1), t, action)

        self.dit.forward = MethodType(recovered_forward, self.dit)
        return self


class AutoregressiveMotionDose:
    """Scale temporal increments in generated latent history."""

    def __init__(self, dynamics_model, dose: float) -> None:
        self.dose = float(dose)
        self.dit = dynamics_model.model
        self.original = self.dit.forward

    def __enter__(self):
        scale = 1.0 + self.dose
        original = self.original

        def scaled_forward(_module, z, t, action):
            history = z[:, :-1]
            if history.shape[1] <= 1:
                return original(z, t, action)
            first = history[:, :1]
            increments = scale * (history[:, 1:] - history[:, :-1])
            motion_scaled = torch.cat([first, first + torch.cumsum(increments, dim=1)], dim=1)
            return original(torch.cat([motion_scaled, z[:, -1:]], dim=1), t, action)

        self.dit.forward = MethodType(scaled_forward, self.dit)
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.dit.forward = self.original
        return False


class AutoregressiveMotionRegionDose:
    """Dose generated-history increments in proportion to spatial motion."""

    def __init__(self, dynamics_model, dose: float) -> None:
        self.dose = float(dose)
        self.dit = dynamics_model.model
        self.original = self.dit.forward

    def __enter__(self):
        dose = self.dose
        original = self.original

        def scaled_forward(_module, z, t, action):
            history = z[:, :-1]
            if history.shape[1] <= 1:
                return original(z, t, action)
            first = history[:, :1]
            increments = history[:, 1:] - history[:, :-1]
            motion = increments.abs().mean(dim=-1, keepdim=True)
            spatial_dims = tuple(range(2, motion.ndim - 1))
            if spatial_dims:
                scale = motion.amax(dim=spatial_dims, keepdim=True).clamp_min(1e-12)
                motion_weight = (motion / scale).clamp(0.0, 1.0)
            else:
                motion_weight = (motion > 0).to(motion.dtype)
            focused_increments = increments * (1.0 + dose * motion_weight)
            focused_history = torch.cat(
                [first, first + torch.cumsum(focused_increments, dim=1)], dim=1
            )
            return original(torch.cat([focused_history, z[:, -1:]], dim=1), t, action)

        self.dit.forward = MethodType(scaled_forward, self.dit)
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.dit.forward = self.original
        return False


class AutoregressiveMotionEventAlignmentDose:
    """Dose motion regions only when they coincide with action transitions."""

    def __init__(self, dynamics_model, dose: float) -> None:
        self.dose = float(dose)
        self.dit = dynamics_model.model
        self.original = self.dit.forward

    def __enter__(self):
        dose = self.dose
        original = self.original

        def aligned_forward(_module, z, t, action):
            history = z[:, :-1]
            if history.shape[1] <= 1:
                return original(z, t, action)
            if action.ndim < 3 or action.shape[0] != history.shape[0] or action.shape[1] < history.shape[1]:
                raise RuntimeError("ACWM_MOTION_EVENT_ACTION_SHAPE_INVALID")

            first = history[:, :1]
            increments = history[:, 1:] - history[:, :-1]
            motion = increments.abs().mean(dim=-1, keepdim=True)
            spatial_dims = tuple(range(2, motion.ndim - 1))
            if spatial_dims:
                motion_scale = motion.amax(dim=spatial_dims, keepdim=True).clamp_min(1e-12)
                motion_weight = (motion / motion_scale).clamp(0.0, 1.0)
            else:
                motion_weight = (motion > 0).to(motion.dtype)

            action_history = action[:, : history.shape[1]].reshape(history.shape[0], history.shape[1], -1)
            action_delta = (action_history[:, 1:] - action_history[:, :-1]).abs().mean(dim=-1)
            event_scale = action_delta.amax(dim=1, keepdim=True)
            event_weight = torch.where(
                event_scale > 1e-12,
                action_delta / event_scale.clamp_min(1e-12),
                torch.zeros_like(action_delta),
            )
            while event_weight.ndim < increments.ndim:
                event_weight = event_weight.unsqueeze(-1)

            aligned_weight = motion_weight * event_weight.to(dtype=motion_weight.dtype)
            focused_increments = increments * (1.0 + dose * aligned_weight)
            focused_history = torch.cat(
                [first, first + torch.cumsum(focused_increments, dim=1)], dim=1
            )
            return original(torch.cat([focused_history, z[:, -1:]], dim=1), t, action)

        self.dit.forward = MethodType(aligned_forward, self.dit)
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.dit.forward = self.original
        return False


def _dose_context(
    dynamics_model,
    campaign: dict[str, object],
    dose: float,
    *,
    teacher_history: torch.Tensor | None = None,
):
    probe = campaign["probe"]
    if not isinstance(probe, dict):
        raise RuntimeError("ACWM_PROBE_INVALID")
    probe_id = probe.get("probe_id")
    if probe_id == "action_conditioning_scale":
        if not hasattr(dynamics_model.model, "action_embedder"):
            raise RuntimeError("ACWM_ACTION_EMBEDDER_HOOK_MISSING")
        return ActionEmbeddingDose(dynamics_model, dose)
    if probe_id == "action_embedding_temporal_mix":
        if not hasattr(dynamics_model.model, "action_embedder"):
            raise RuntimeError("ACWM_ACTION_EMBEDDER_HOOK_MISSING")
        return ActionEmbeddingTemporalMixDose(dynamics_model, dose)
    if probe_id == "action_temporal_alignment":
        if not hasattr(dynamics_model.model, "action_embedder"):
            raise RuntimeError("ACWM_ACTION_EVENT_ALIGNMENT_HOOK_MISSING")
        return ActionEmbeddingEventAlignmentDose(dynamics_model, dose)
    if probe_id == "self_rollout_history_scale":
        if str(probe.get("generation_mode", "")) != "autoregressive":
            raise RuntimeError("ACWM_HISTORY_PROBE_REQUIRES_AUTOREGRESSIVE_MODE")
        if not hasattr(dynamics_model, "model") or not hasattr(dynamics_model.model, "forward"):
            raise RuntimeError("ACWM_HISTORY_LATENT_HOOK_MISSING")
        return AutoregressiveHistoryDose(dynamics_model, dose)
    if probe_id == "self_rollout_temporal_mix":
        if str(probe.get("generation_mode", "")) != "autoregressive":
            raise RuntimeError("ACWM_HISTORY_TEMPORAL_MIX_REQUIRES_AUTOREGRESSIVE_MODE")
        if not hasattr(dynamics_model, "model") or not hasattr(dynamics_model.model, "forward"):
            raise RuntimeError("ACWM_HISTORY_TEMPORAL_MIX_HOOK_MISSING")
        return AutoregressiveHistoryTemporalMixDose(dynamics_model, dose)
    if probe_id == "self_rollout_latest_feedback":
        if str(probe.get("generation_mode", "")) != "autoregressive":
            raise RuntimeError("ACWM_LATEST_FEEDBACK_REQUIRES_AUTOREGRESSIVE_MODE")
        if not hasattr(dynamics_model, "model") or not hasattr(dynamics_model.model, "forward"):
            raise RuntimeError("ACWM_LATEST_FEEDBACK_HISTORY_HOOK_MISSING")
        return AutoregressiveLatestFeedbackDose(dynamics_model, dose)
    if probe_id == "self_rollout_teacher_recovery":
        if str(probe.get("generation_mode", "")) != "autoregressive":
            raise RuntimeError("ACWM_TEACHER_RECOVERY_REQUIRES_AUTOREGRESSIVE_MODE")
        if not hasattr(dynamics_model, "model") or not hasattr(dynamics_model.model, "forward"):
            raise RuntimeError("ACWM_TEACHER_RECOVERY_HISTORY_HOOK_MISSING")
        if teacher_history is None:
            raise RuntimeError("ACWM_TEACHER_RECOVERY_REFERENCE_MISSING")
        return AutoregressiveTeacherRecoveryDose(dynamics_model, dose, teacher_history)
    if probe_id == "self_rollout_horizon_recovery":
        if str(probe.get("generation_mode", "")) != "autoregressive":
            raise RuntimeError("ACWM_TEACHER_HORIZON_RECOVERY_REQUIRES_AUTOREGRESSIVE_MODE")
        if not hasattr(dynamics_model, "model") or not hasattr(dynamics_model.model, "forward"):
            raise RuntimeError("ACWM_TEACHER_HORIZON_RECOVERY_HISTORY_HOOK_MISSING")
        if teacher_history is None:
            raise RuntimeError("ACWM_TEACHER_HORIZON_RECOVERY_REFERENCE_MISSING")
        return AutoregressiveTeacherHorizonRecoveryDose(dynamics_model, dose, teacher_history)
    if probe_id == "motion_history_scale":
        if str(probe.get("generation_mode", "")) != "autoregressive":
            raise RuntimeError("ACWM_MOTION_PROBE_REQUIRES_AUTOREGRESSIVE_MODE")
        if not hasattr(dynamics_model, "model") or not hasattr(dynamics_model.model, "forward"):
            raise RuntimeError("ACWM_MOTION_LATENT_HOOK_MISSING")
        return AutoregressiveMotionDose(dynamics_model, dose)
    if probe_id == "motion_region_scale":
        if str(probe.get("generation_mode", "")) != "autoregressive":
            raise RuntimeError("ACWM_MOTION_REGION_PROBE_REQUIRES_AUTOREGRESSIVE_MODE")
        if not hasattr(dynamics_model, "model") or not hasattr(dynamics_model.model, "forward"):
            raise RuntimeError("ACWM_MOTION_REGION_LATENT_HOOK_MISSING")
        return AutoregressiveMotionRegionDose(dynamics_model, dose)
    if probe_id == "motion_region_event_alignment":
        if str(probe.get("generation_mode", "")) != "autoregressive":
            raise RuntimeError("ACWM_MOTION_EVENT_PROBE_REQUIRES_AUTOREGRESSIVE_MODE")
        if not hasattr(dynamics_model, "model") or not hasattr(dynamics_model.model, "forward"):
            raise RuntimeError("ACWM_MOTION_EVENT_LATENT_HOOK_MISSING")
        return AutoregressiveMotionEventAlignmentDose(dynamics_model, dose)
    raise RuntimeError(f"ACWM_PROBE_RUNTIME_UNSUPPORTED:{probe_id}")


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
    teacher_history = None
    if campaign["probe"]["probe_id"] in {
        "self_rollout_teacher_recovery",
        "self_rollout_horizon_recovery",
    }:
        with torch.no_grad():
            teacher_history = model.encode_obs(gt_video)
    measurements: list[dict[str, object]] = []
    generation_mode = str(campaign["probe"].get("generation_mode", "parallel"))
    for seed in campaign["seeds"]:
        for dose in campaign["probe"]["doses"]:
            random.seed(int(seed))
            np.random.seed(int(seed))
            torch.manual_seed(int(seed))
            torch.cuda.manual_seed_all(int(seed))
            with _dose_context(
                model,
                campaign,
                float(dose),
                teacher_history=teacher_history,
            ):
                with torch.no_grad():
                    prediction = model.generate(
                        o_0,
                        action,
                        num_inference_steps=int(protocol["inference_steps"]),
                        noise_level=0.0,
                        mode=generation_mode,
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
