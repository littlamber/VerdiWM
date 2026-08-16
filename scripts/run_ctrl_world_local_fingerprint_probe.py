#!/usr/bin/env python3
"""Run one reversible Ctrl-World base-probe path on frozen replay contexts.

This program does not change the upstream Ctrl-World checkout, checkpoint, or
dataset.  It produces raw paired-dose measurements that are assembled into a
target-local fingerprint by ``aggregate_ctrl_world_local_fingerprint.py``.
"""

from __future__ import annotations

import argparse
from dataclasses import is_dataclass, replace
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
from pathlib import Path
from types import MethodType
from typing import Any, Mapping, Sequence

import numpy as np
import torch


OUTCOME_NAMES = (
    "negative_mean_l1",
    "negative_final_interaction_l1",
    "negative_horizon_l1_slope",
    "mean_psnr",
    "final_psnr",
)


class LocalFingerprintProbeError(RuntimeError):
    """A frozen paired-dose probe could not be executed faithfully."""


class ReplayProbe:
    """One temporary semantic intervention plus its runtime audit."""

    def __init__(self, model: object, dose: float) -> None:
        if not math.isfinite(dose) or abs(dose) >= 1.0:
            raise LocalFingerprintProbeError("LOCAL_FINGERPRINT_DOSE_INVALID")
        self.model = model
        self.dose = float(dose)
        self.audit: dict[str, float] = {
            "invocation_count": 0.0,
            "mean_abs_tensor_delta": 0.0,
            "maximum_abs_tensor_delta": 0.0,
        }

    def __enter__(self) -> "ReplayProbe":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        return False

    def transform_conditions(
        self, *, history: torch.Tensor, current: torch.Tensor, first: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del first
        return history, current

    def _record_delta(self, before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
        if before.shape != after.shape:
            raise LocalFingerprintProbeError("LOCAL_FINGERPRINT_PROBE_SHAPE_CHANGED")
        delta = (after - before).abs()
        self.audit["invocation_count"] += 1.0
        self.audit["mean_abs_tensor_delta"] += float(delta.mean().item())
        self.audit["maximum_abs_tensor_delta"] = max(
            self.audit["maximum_abs_tensor_delta"], float(delta.max().item())
        )
        return after

    def result(self) -> dict[str, float]:
        count = max(self.audit["invocation_count"], 1.0)
        return {
            "invocation_count": self.audit["invocation_count"],
            "mean_abs_tensor_delta": self.audit["mean_abs_tensor_delta"] / count,
            "maximum_abs_tensor_delta": self.audit["maximum_abs_tensor_delta"],
        }

    def runtime_feature_records(self) -> list[list[list[float]]]:
        return []

    def learned_gate_records(self) -> list[dict[str, object]]:
        return []

    def set_interaction(self, interaction: int) -> None:
        if interaction < 0:
            raise LocalFingerprintProbeError("LOCAL_FINGERPRINT_INTERACTION_INVALID")


class ActionConditioningScale(ReplayProbe):
    """Scale only the action encoder output, after the action trajectory is fixed."""

    def __enter__(self) -> "ActionConditioningScale":
        encoder = getattr(self.model, "action_encoder", None)
        original = getattr(encoder, "forward", None)
        if encoder is None or not callable(original):
            raise LocalFingerprintProbeError("LOCAL_FINGERPRINT_ACTION_HOOK_MISSING")
        self._encoder = encoder
        self._original = original

        def wrapped(_module: object, *args: object, **kwargs: object) -> object:
            embedding = original(*args, **kwargs)
            if not isinstance(embedding, torch.Tensor) or embedding.ndim != 3:
                raise LocalFingerprintProbeError("LOCAL_FINGERPRINT_ACTION_EMBEDDING_INVALID")
            if self.dose == 0.0:
                return self._record_delta(embedding, embedding)
            return self._record_delta(embedding, embedding * (1.0 + self.dose))

        encoder.forward = MethodType(wrapped, encoder)
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        self._encoder.forward = self._original
        return False


class HistoryRetentionGain(ReplayProbe):
    """Increase or attenuate historical variation relative to the newest slot.

    Ctrl-World has a fixed-width history input.  This is therefore a signed
    target-local realization of controlled context retention: negative doses
    contract older history toward the newest retained state, positive doses
    preserve the same slots while increasing their historical contrast.
    """

    def transform_conditions(
        self, *, history: torch.Tensor, current: torch.Tensor, first: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del first
        if history.ndim != 5 or history.shape[1] < 2:
            raise LocalFingerprintProbeError("LOCAL_FINGERPRINT_HISTORY_INVALID")
        if self.dose == 0.0:
            return self._record_delta(history, history), current
        newest = history[:, -1:].expand_as(history)
        transformed = newest + (1.0 + self.dose) * (history - newest)
        return self._record_delta(history, transformed), current


class FirstFrameAnchorBlend(ReplayProbe):
    """Blend the frozen observed first latent into the current image condition."""

    def transform_conditions(
        self, *, history: torch.Tensor, current: torch.Tensor, first: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if current.shape != first.shape:
            raise LocalFingerprintProbeError("LOCAL_FINGERPRINT_FIRST_ANCHOR_SHAPE_INVALID")
        if self.dose == 0.0:
            return history, self._record_delta(current, current)
        transformed = current + self.dose * (first - current)
        return history, self._record_delta(current, transformed)


class SamplerInitialNoiseGain(ReplayProbe):
    """Scale only the sampled initial diffusion latents before denoising."""

    def __enter__(self) -> "SamplerInitialNoiseGain":
        pipeline = getattr(self.model, "pipeline", None)
        original = getattr(pipeline, "prepare_latents", None)
        if pipeline is None or not callable(original):
            raise LocalFingerprintProbeError("LOCAL_FINGERPRINT_SAMPLER_HOOK_MISSING")
        self._pipeline = pipeline
        self._original = original

        def wrapped(_pipeline: object, *args: object, **kwargs: object) -> object:
            latents = original(*args, **kwargs)
            if not isinstance(latents, torch.Tensor):
                raise LocalFingerprintProbeError("LOCAL_FINGERPRINT_SAMPLER_LATENTS_INVALID")
            if self.dose == 0.0:
                return self._record_delta(latents, latents)
            return self._record_delta(latents, latents * (1.0 + self.dose))

        pipeline.prepare_latents = MethodType(wrapped, pipeline)
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        self._pipeline.prepare_latents = self._original
        return False


class FSHCSignedHistoryGain(ReplayProbe):
    """Apply a fixed signed gain through the exact trainable FSHC operator."""

    def __init__(self, model: object, dose: float, *, normalized_mechanism: bool = False) -> None:
        super().__init__(model, dose)
        self.normalized_mechanism = bool(normalized_mechanism)

    def __enter__(self) -> "FSHCSignedHistoryGain":
        corrector = getattr(self.model, "history_corrector", None)
        original = getattr(corrector, "forward", None)
        if corrector is None or not callable(original):
            raise LocalFingerprintProbeError("LOCAL_FINGERPRINT_FSHC_HOOK_MISSING")
        self._corrector = corrector
        self._original = original
        self._runtime_features: list[list[list[float]]] = []
        self._learned_gates: list[dict[str, object]] = []

        def wrapped(_module: object, *args: object, **kwargs: object) -> object:
            if not args or not isinstance(args[0], torch.Tensor):
                raise LocalFingerprintProbeError("LOCAL_FINGERPRINT_FSHC_HISTORY_INVALID")
            if kwargs.get("correction_gain_override") is not None:
                raise LocalFingerprintProbeError("LOCAL_FINGERPRINT_FSHC_OVERRIDE_CONFLICT")
            history = args[0]
            runtime_features_fn = getattr(self._corrector, "runtime_features", None)
            predict_gate_fn = getattr(self._corrector, "predict_gate", None)
            if callable(runtime_features_fn) and callable(predict_gate_fn) and len(args) >= 4:
                learned_features = runtime_features_fn(
                    *args[:4],
                    rollout_consistency=kwargs.get("rollout_consistency"),
                )
                learned_gate = predict_gate_fn(learned_features)
                self._learned_gates.append(
                    {
                        "signed_gain": learned_gate.signed_gain[0].detach().float().cpu().tolist(),
                        "confidence": learned_gate.confidence[0].detach().float().cpu().tolist(),
                        "use_probability": learned_gate.use_probability[0].detach().float().cpu().tolist(),
                    }
                )
            multiscale = getattr(self._corrector, "multiscale_side_adapter", None) is not None
            overrides = dict(kwargs)
            active_dose = self._active_dose()
            if multiscale:
                if overrides.get("adapter_scale_override") is not None:
                    raise LocalFingerprintProbeError("LOCAL_FINGERPRINT_FSHC_OVERRIDE_CONFLICT")
            else:
                gain = active_dose
                if self.normalized_mechanism:
                    gain *= float(getattr(self._corrector, "max_gain"))
                overrides["correction_gain_override"] = gain
            output = original(*args, **overrides)
            if multiscale:
                down_residuals = getattr(output, "down_block_additional_residuals", None)
                mid_residual = getattr(output, "mid_block_additional_residual", None)
                if not isinstance(down_residuals, tuple) or not isinstance(mid_residual, torch.Tensor):
                    raise LocalFingerprintProbeError("LOCAL_FINGERPRINT_FSHC_OUTPUT_INVALID")
                scaled_down_residuals = tuple(value * active_dose for value in down_residuals)
                scaled_mid_residual = mid_residual * active_dose
                updates = {
                    "down_block_additional_residuals": scaled_down_residuals,
                    "mid_block_additional_residual": scaled_mid_residual,
                    "maximum_abs_multiscale_residual": torch.stack(
                        [
                            value.detach().float().abs().max()
                            for value in (*scaled_down_residuals, scaled_mid_residual)
                        ]
                    ).max(),
                }
                if is_dataclass(output):
                    output = replace(output, **updates)
                else:
                    for name, value in updates.items():
                        setattr(output, name, value)
            corrected = getattr(output, "history", None)
            runtime_features = getattr(output, "runtime_features", None)
            if (
                not isinstance(corrected, torch.Tensor)
                or not isinstance(runtime_features, torch.Tensor)
                or runtime_features.ndim != 3
                or runtime_features.shape[0] != 1
                or not torch.isfinite(runtime_features).all()
            ):
                raise LocalFingerprintProbeError("LOCAL_FINGERPRINT_FSHC_OUTPUT_INVALID")
            self._runtime_features.append(
                runtime_features[0].detach().float().cpu().tolist()
            )
            down_residuals = getattr(output, "down_block_additional_residuals", None)
            mid_residual = getattr(output, "mid_block_additional_residual", None)
            if multiscale and isinstance(down_residuals, tuple) and isinstance(mid_residual, torch.Tensor):
                signal = torch.stack(
                    [value.detach().float().abs().max() for value in (*down_residuals, mid_residual)]
                )
                self._record_delta(torch.zeros_like(signal), signal)
            else:
                self._record_delta(history, corrected)
            return output

        corrector.forward = MethodType(wrapped, corrector)
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        self._corrector.forward = self._original
        return False

    def runtime_feature_records(self) -> list[list[list[float]]]:
        return list(self._runtime_features)

    def learned_gate_records(self) -> list[dict[str, object]]:
        return list(self._learned_gates)

    def _active_dose(self) -> float:
        return self.dose


class FSHCInteractionLocalGain(FSHCSignedHistoryGain):
    """Apply the FSHC dose only inside one explicitly selected interaction."""

    def __init__(
        self,
        model: object,
        dose: float,
        *,
        target_interaction: int | None,
        normalized_mechanism: bool = False,
    ) -> None:
        super().__init__(model, dose, normalized_mechanism=normalized_mechanism)
        if target_interaction is None and dose != 0.0:
            raise LocalFingerprintProbeError("LOCAL_FINGERPRINT_LOCAL_TARGET_REQUIRED")
        if target_interaction is not None and target_interaction < 0:
            raise LocalFingerprintProbeError("LOCAL_FINGERPRINT_INTERACTION_INVALID")
        self.target_interaction = target_interaction
        self.current_interaction: int | None = None
        self.active_invocation_count = 0

    def set_interaction(self, interaction: int) -> None:
        super().set_interaction(interaction)
        self.current_interaction = int(interaction)

    def _active_dose(self) -> float:
        if self.current_interaction is None:
            raise LocalFingerprintProbeError("LOCAL_FINGERPRINT_INTERACTION_UNSET")
        if self.target_interaction != self.current_interaction:
            return 0.0
        self.active_invocation_count += 1
        return self.dose

    def result(self) -> dict[str, float | int | None]:
        return {
            **super().result(),
            "target_interaction": self.target_interaction,
            "active_invocation_count": self.active_invocation_count,
        }


PROBES = {
    "action_conditioning_scale": ActionConditioningScale,
    "fshc_interaction_local_gain": FSHCInteractionLocalGain,
    "fshc_signed_history_gain": FSHCSignedHistoryGain,
    "history_retention_gain": HistoryRetentionGain,
    "first_frame_anchor_blend": FirstFrameAnchorBlend,
    "sampler_initial_noise_gain": SamplerInitialNoiseGain,
}


def _canonical_json(payload: object) -> bytes:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _load_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LocalFingerprintProbeError(f"LOCAL_FINGERPRINT_JSON_INVALID:{path}") from exc


def load_contexts(
    path: Path,
    selected_seeds: Sequence[int] | None = None,
    selected_context_ids: Sequence[str] | None = None,
) -> list[dict[str, object]]:
    payload = _load_json(path)
    if not isinstance(payload, Mapping) or payload.get("artifact_type") != "verdiwm-ctrl-world-local-context-set":
        raise LocalFingerprintProbeError("LOCAL_FINGERPRINT_CONTEXT_SET_INVALID")
    contexts = payload.get("contexts")
    if not isinstance(contexts, list) or not contexts:
        raise LocalFingerprintProbeError("LOCAL_FINGERPRINT_CONTEXT_SET_INVALID")
    wanted = set(selected_seeds or ())
    wanted_contexts = set(selected_context_ids or ())
    expanded: list[dict[str, object]] = []
    identities: set[tuple[str, int]] = set()
    for raw in contexts:
        if not isinstance(raw, Mapping):
            raise LocalFingerprintProbeError("LOCAL_FINGERPRINT_CONTEXT_SET_INVALID")
        context_id = str(raw.get("context_id", ""))
        episode_id = str(raw.get("episode_id", ""))
        start_idx = raw.get("start_idx")
        seeds = raw.get("seeds")
        if not context_id or not episode_id or not isinstance(start_idx, int) or start_idx < 0:
            raise LocalFingerprintProbeError("LOCAL_FINGERPRINT_CONTEXT_SET_INVALID")
        if not isinstance(seeds, list) or not seeds or any(not isinstance(value, int) for value in seeds):
            raise LocalFingerprintProbeError("LOCAL_FINGERPRINT_CONTEXT_SET_INVALID")
        if wanted_contexts and context_id not in wanted_contexts:
            continue
        for seed in seeds:
            if wanted and seed not in wanted:
                continue
            identity = (context_id, seed)
            if identity in identities:
                raise LocalFingerprintProbeError("LOCAL_FINGERPRINT_CONTEXT_IDENTITY_DUPLICATE")
            identities.add(identity)
            expanded.append(
                {
                    "context_id": context_id,
                    "episode_id": episode_id,
                    "start_idx": start_idx,
                    "seed": seed,
                }
            )
    if not expanded:
        raise LocalFingerprintProbeError("LOCAL_FINGERPRINT_CONTEXT_SELECTION_EMPTY")
    return expanded


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_rollout_module(ctrl_world_root: Path) -> Any:
    root = ctrl_world_root.resolve(strict=True)
    for value in (str(root), str(root / "scripts")):
        if value in sys.path:
            sys.path.remove(value)
        sys.path.insert(0, value)
    source = root / "scripts" / "rollout_replay_traj.py"
    if not source.is_file():
        raise LocalFingerprintProbeError("LOCAL_FINGERPRINT_REPLAY_ENTRYPOINT_MISSING")
    spec = importlib.util.spec_from_file_location("verdiwm_ctrl_world_replay", source)
    if spec is None or spec.loader is None:
        raise LocalFingerprintProbeError("LOCAL_FINGERPRINT_REPLAY_IMPORT_SPEC_INVALID")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _runtime_args(
    *,
    dataset_root: Path,
    data_stat: Path,
    svd_model_path: Path,
    clip_model_path: Path,
    ckpt_path: Path,
    interact_num: int,
    num_inference_steps: int,
    enable_signed_history_correction: bool,
    unsigned_history_gate: bool,
    enable_multiscale_history_adapter: bool,
    multiscale_history_always_on: bool,
) -> object:
    from config import wm_args

    args = wm_args(task_type="replay")
    args.svd_model_path = str(svd_model_path)
    args.clip_model_path = str(clip_model_path)
    args.ckpt_path = str(ckpt_path)
    args.val_model_path = str(ckpt_path)
    args.val_dataset_dir = str(dataset_root)
    args.data_stat_path = str(data_stat)
    args.interact_num = int(interact_num)
    args.num_inference_steps = int(num_inference_steps)
    args.enable_signed_history_correction = bool(enable_signed_history_correction)
    args.unsigned_history_gate = bool(unsigned_history_gate)
    args.enable_multiscale_history_adapter = bool(enable_multiscale_history_adapter)
    args.multiscale_history_always_on = bool(multiscale_history_always_on)
    args.save_dir = "unused-by-local-fingerprint"
    return args


def _metrics(true_video: np.ndarray, predicted_video: np.ndarray) -> dict[str, object]:
    if true_video.shape != predicted_video.shape or true_video.ndim != 5:
        raise LocalFingerprintProbeError("LOCAL_FINGERPRINT_VIDEO_SHAPE_INVALID")
    true = true_video.astype(np.float32)
    predicted = predicted_video.astype(np.float32)
    absolute = np.abs(true - predicted).mean(axis=(0, 2, 3, 4))
    mse = np.square(true - predicted).mean(axis=(0, 2, 3, 4))
    psnr = 10.0 * np.log10((255.0**2) / np.maximum(mse, 1e-9))
    return {
        "l1_per_frame": [float(value) for value in absolute],
        "psnr_per_frame": [float(value) for value in psnr],
        "mean_l1": float(absolute.mean()),
        "final_l1": float(absolute[-1]),
        "mean_psnr": float(psnr.mean()),
        "final_psnr": float(psnr[-1]),
    }


def _outcome_vector(interactions: Sequence[Mapping[str, object]]) -> dict[str, float]:
    if not interactions:
        raise LocalFingerprintProbeError("LOCAL_FINGERPRINT_INTERACTION_METRICS_EMPTY")
    mean_l1 = np.asarray([float(row["mean_l1"]) for row in interactions], dtype=np.float64)
    final_l1 = np.asarray([float(row["final_l1"]) for row in interactions], dtype=np.float64)
    mean_psnr = np.asarray([float(row["mean_psnr"]) for row in interactions], dtype=np.float64)
    final_psnr = np.asarray([float(row["final_psnr"]) for row in interactions], dtype=np.float64)
    slope = float(np.polyfit(np.arange(len(mean_l1), dtype=np.float64), mean_l1, 1)[0]) if len(mean_l1) > 1 else 0.0
    return {
        "negative_mean_l1": -float(mean_l1.mean()),
        "negative_final_interaction_l1": -float(final_l1[-1]),
        "negative_horizon_l1_slope": -slope,
        "mean_psnr": float(mean_psnr.mean()),
        "final_psnr": float(final_psnr[-1]),
    }


def _paired_grid(true_video: np.ndarray, predicted_video: np.ndarray) -> np.ndarray:
    ground_truth = np.concatenate([true_video[index] for index in range(true_video.shape[0])], axis=1)
    prediction = np.concatenate([predicted_video[index] for index in range(predicted_video.shape[0])], axis=1)
    return np.concatenate((ground_truth, prediction), axis=1)


def _prediction_response(predicted_video: np.ndarray) -> dict[str, object]:
    """Keep a compact spatial response for paired counterfactual comparisons."""
    if predicted_video.ndim != 5 or predicted_video.shape[-1] not in (1, 3, 4):
        raise LocalFingerprintProbeError("LOCAL_FINGERPRINT_PREDICTION_RESPONSE_SHAPE_INVALID")
    tensor = torch.from_numpy(np.ascontiguousarray(predicted_video)).float()
    camera_count, frame_count, height, width, channel_count = tensor.shape
    pooled = torch.nn.functional.adaptive_avg_pool2d(
        tensor.permute(0, 1, 4, 2, 3).reshape(-1, channel_count, height, width),
        output_size=(4, 4),
    )
    if float(tensor.max()) > 1.5:
        pooled = pooled / 255.0
    if not torch.isfinite(pooled).all():
        raise LocalFingerprintProbeError("LOCAL_FINGERPRINT_PREDICTION_RESPONSE_NONFINITE")
    response = pooled.reshape(camera_count, frame_count, channel_count, 4, 4)
    return {
        "shape": list(response.shape),
        "values": response.reshape(-1).tolist(),
        "source_sha256": hashlib.sha256(np.ascontiguousarray(predicted_video).tobytes()).hexdigest(),
    }


def _run_context(
    *,
    rollout_agent: object,
    runtime_args: object,
    context: Mapping[str, object],
    probe_id: str | None,
    dose: float,
    fshc_dose_mode: str,
    target_interaction: int | None = None,
) -> tuple[dict[str, object], np.ndarray]:
    _set_seed(int(context["seed"]))
    pred_step = int(getattr(runtime_args, "pred_step"))
    interact_num = int(getattr(runtime_args, "interact_num"))
    num_history = int(getattr(runtime_args, "num_history"))
    num_frames = int(getattr(runtime_args, "num_frames"))
    eef_gt, _joint_pos_gt, _video_dict, video_latents, instruction = rollout_agent.get_traj_info(
        str(context["episode_id"]),
        start_idx=int(context["start_idx"]),
        steps=int(pred_step * interact_num + 8),
    )
    first_latent = torch.cat([value[0] for value in video_latents], dim=1).unsqueeze(0)
    if tuple(first_latent.shape) != (1, 4, 72, 40):
        raise LocalFingerprintProbeError(
            f"LOCAL_FINGERPRINT_INITIAL_LATENT_SHAPE:{tuple(first_latent.shape)}"
        )
    history_latents = [first_latent for _ in range(num_history * 4)]
    history_eef = [eef_gt[0:1] for _ in range(num_history * 4)]
    probe_cls = PROBES.get(probe_id) if probe_id else None
    if probe_cls is FSHCInteractionLocalGain:
        probe = probe_cls(
            getattr(rollout_agent, "model", None),
            dose,
            target_interaction=target_interaction,
            normalized_mechanism=fshc_dose_mode == "normalized_mechanism",
        )
    elif probe_cls is FSHCSignedHistoryGain:
        probe = probe_cls(
            getattr(rollout_agent, "model", None),
            dose,
            normalized_mechanism=fshc_dose_mode == "normalized_mechanism",
        )
    else:
        probe = probe_cls(getattr(rollout_agent, "model", None), dose) if probe_cls else ReplayProbe(object(), 0.0)
    all_true: list[np.ndarray] = []
    all_prediction: list[np.ndarray] = []
    interactions: list[dict[str, object]] = []
    with probe:
        for interaction in range(interact_num):
            probe.set_interaction(interaction)
            start_id = int(interaction * (pred_step - 1))
            end_id = start_id + pred_step
            target_latents = [value[start_id:end_id] for value in video_latents]
            cartesian_pose = eef_gt[start_id:end_id]
            if cartesian_pose.shape != (pred_step, 7):
                raise LocalFingerprintProbeError("LOCAL_FINGERPRINT_ACTION_WINDOW_INVALID")
            history_idx = [0, 0, -8, -6, -4, -2]
            history_pose = np.concatenate([history_eef[index] for index in history_idx], axis=0)
            action_cond = np.concatenate((history_pose, cartesian_pose), axis=0)
            history_input = torch.cat([history_latents[index] for index in history_idx], dim=0).unsqueeze(0)
            current_latent = history_latents[-1]
            if action_cond.shape != (num_history + num_frames, 7):
                raise LocalFingerprintProbeError("LOCAL_FINGERPRINT_CONDITION_WINDOW_INVALID")
            if history_input.shape != (1, num_history, 4, 72, 40):
                raise LocalFingerprintProbeError("LOCAL_FINGERPRINT_HISTORY_WINDOW_INVALID")
            history_input, current_latent = probe.transform_conditions(
                history=history_input, current=current_latent, first=first_latent
            )
            videos_cat, true_video, prediction, predicted_latents = rollout_agent.forward_wm(
                action_cond,
                target_latents,
                current_latent,
                his_cond=history_input,
                text=instruction if bool(getattr(runtime_args, "text_cond")) else None,
            )
            del videos_cat
            metric = _metrics(true_video, prediction)
            metric["interaction"] = interaction
            interactions.append(metric)
            all_true.append(true_video)
            all_prediction.append(prediction)
            history_eef.append(cartesian_pose[pred_step - 1 : pred_step])
            history_latents.append(
                torch.cat([value[pred_step - 1] for value in predicted_latents], dim=1).unsqueeze(0)
            )
    outcome = _outcome_vector(interactions)
    if not all(math.isfinite(value) for value in outcome.values()):
        raise LocalFingerprintProbeError("LOCAL_FINGERPRINT_OUTCOME_NONFINITE")
    predicted_video = np.concatenate(all_prediction, axis=1)
    row = {
        "identity": {
            "context_id": str(context["context_id"]),
            "episode_id": str(context["episode_id"]),
            "start_idx": int(context["start_idx"]),
            "seed": int(context["seed"]),
        },
        "interactions": interactions,
        "outcomes": outcome,
        "hook_audit": probe.result(),
        "runtime_feature_records": probe.runtime_feature_records(),
        "learned_gate_records": probe.learned_gate_records(),
        "prediction_response": _prediction_response(predicted_video),
    }
    return row, _paired_grid(np.concatenate(all_true, axis=1), predicted_video)


def _outcome_difference(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    if set(left) != set(OUTCOME_NAMES) or set(right) != set(OUTCOME_NAMES):
        raise LocalFingerprintProbeError("LOCAL_FINGERPRINT_OUTCOME_SCHEMA_INVALID")
    return max(abs(float(left[name]) - float(right[name])) for name in OUTCOME_NAMES)


def _runtime_receipt() -> dict[str, object]:
    devices: list[dict[str, object]] = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "logical_index": index,
                    "name": properties.name,
                    "total_memory_bytes": int(properties.total_memory),
                    "uuid": str(getattr(properties, "uuid", "")),
                }
            )
    return {
        "python_executable": sys.executable,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_available": torch.cuda.is_available(),
        "devices": devices,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    output_root = Path(args.output_root).resolve()
    if output_root.exists() or output_root.is_symlink():
        raise LocalFingerprintProbeError("LOCAL_FINGERPRINT_OUTPUT_EXISTS")
    ctrl_world_root = Path(args.ctrl_world_root).resolve(strict=True)
    contexts_path = Path(args.contexts_json).resolve(strict=True)
    contexts = load_contexts(contexts_path, args.seeds, args.context_ids)
    doses = tuple(float(value) for value in args.doses)
    if 0.0 not in doses or len(doses) < 3 or len(doses) != len(set(doses)):
        raise LocalFingerprintProbeError("LOCAL_FINGERPRINT_DOSE_GRID_INVALID")
    if not any(math.isclose(-dose, other, abs_tol=1e-12) for dose in doses if dose > 0.0 for other in doses):
        raise LocalFingerprintProbeError("LOCAL_FINGERPRINT_SYMMETRIC_DOSE_MISSING")
    if args.probe_id not in PROBES or int(args.interact_num) < 2 or int(args.num_inference_steps) < 1:
        raise LocalFingerprintProbeError("LOCAL_FINGERPRINT_ARGUMENT_INVALID")
    local_interaction_probe = args.probe_id == "fshc_interaction_local_gain"
    target_interactions = tuple(int(value) for value in (args.target_interactions or ()))
    if local_interaction_probe:
        if (
            not target_interactions
            or len(target_interactions) != len(set(target_interactions))
            or any(value < 0 or value >= int(args.interact_num) for value in target_interactions)
        ):
            raise LocalFingerprintProbeError("LOCAL_FINGERPRINT_TARGET_INTERACTIONS_INVALID")
    elif target_interactions:
        raise LocalFingerprintProbeError("LOCAL_FINGERPRINT_TARGET_INTERACTIONS_UNSUPPORTED")
    enable_corrector = bool(args.enable_signed_history_correction) or bool(
        args.enable_multiscale_history_adapter
    ) or (
        args.probe_id in {"fshc_interaction_local_gain", "fshc_signed_history_gain"}
    )
    if bool(args.unsigned_history_gate) and not enable_corrector:
        raise LocalFingerprintProbeError("LOCAL_FINGERPRINT_UNSIGNED_GATE_WITHOUT_CORRECTOR")

    module = _load_rollout_module(ctrl_world_root)
    runtime_args = _runtime_args(
        dataset_root=Path(args.dataset_root).resolve(strict=True),
        data_stat=Path(args.data_stat).resolve(strict=True),
        svd_model_path=Path(args.svd_model_path).resolve(strict=True),
        clip_model_path=Path(args.clip_model_path).resolve(strict=True),
        ckpt_path=Path(args.ckpt_path).resolve(strict=True),
        interact_num=int(args.interact_num),
        num_inference_steps=int(args.num_inference_steps),
        enable_signed_history_correction=enable_corrector,
        unsigned_history_gate=bool(args.unsigned_history_gate),
        enable_multiscale_history_adapter=bool(args.enable_multiscale_history_adapter),
        multiscale_history_always_on=bool(args.multiscale_history_always_on),
    )
    output_root.mkdir(mode=0o700, parents=True)
    rollout_agent = module.agent(runtime_args)
    references: dict[tuple[str, int], Mapping[str, object]] = {}
    reference_visual: np.ndarray | None = None
    reference_probe_id = (
        str(args.probe_id) if args.zero_reference_mode == "probe-zero" else None
    )
    for context in contexts:
        row, visual = _run_context(
            rollout_agent=rollout_agent,
            runtime_args=runtime_args,
            context=context,
            probe_id=reference_probe_id,
            dose=0.0,
            fshc_dose_mode=str(args.fshc_dose_mode),
            target_interaction=None,
        )
        key = (str(context["context_id"]), int(context["seed"]))
        references[key] = row
        if reference_visual is None:
            reference_visual = visual

    measurements: list[dict[str, object]] = []
    zero_identity_checks: list[dict[str, object]] = []
    visual_sample: np.ndarray | None = None
    ordered_doses = (0.0,) + tuple(value for value in doses if value != 0.0)
    measurement_frame = (
        ((0.0, None),)
        + tuple(
            (dose, target_interaction)
            for target_interaction in target_interactions
            for dose in ordered_doses
            if dose != 0.0
        )
        if local_interaction_probe
        else tuple((dose, None) for dose in ordered_doses)
    )
    for dose, target_interaction in measurement_frame:
        for context in contexts:
            row, visual = _run_context(
                rollout_agent=rollout_agent,
                runtime_args=runtime_args,
                context=context,
                probe_id=str(args.probe_id),
                dose=dose,
                fshc_dose_mode=str(args.fshc_dose_mode),
                target_interaction=target_interaction,
            )
            row["dose"] = dose
            if local_interaction_probe:
                row["target_interaction"] = target_interaction
            measurements.append(row)
            if dose == 0.0:
                key = (str(context["context_id"]), int(context["seed"]))
                reference = references[key]
                difference = _outcome_difference(
                    reference["outcomes"], row["outcomes"]  # type: ignore[arg-type]
                )
                audit = row["hook_audit"]
                passed = (
                    difference <= float(args.zero_identity_tolerance)
                    and float(audit["invocation_count"]) >= float(args.interact_num)
                    and float(audit["maximum_abs_tensor_delta"]) == 0.0
                )
                zero_identity_checks.append(
                    {
                        "identity": row["identity"],
                        "target_interaction": target_interaction,
                        "maximum_outcome_abs_difference": difference,
                        "tolerance": float(args.zero_identity_tolerance),
                        "state": "passed" if passed else "failed",
                    }
                )
                if visual_sample is None:
                    visual_sample = visual

    if reference_visual is None or visual_sample is None:
        raise LocalFingerprintProbeError("LOCAL_FINGERPRINT_VIDEO_MISSING")
    nonzero_measurements = [row for row in measurements if float(row["dose"]) != 0.0]
    nonzero_hook_active = all(
        float(row["hook_audit"]["invocation_count"]) >= float(args.interact_num)
        and float(row["hook_audit"]["maximum_abs_tensor_delta"]) > 0.0
        and (
            not local_interaction_probe
            or int(row["hook_audit"]["active_invocation_count"])
            == int(args.num_inference_steps)
        )
        for row in nonzero_measurements
    )
    zero_identity_passed = all(row["state"] == "passed" for row in zero_identity_checks)
    result = {
        "schema_version": 1,
        "artifact_type": "verdiwm-ctrl-world-local-fingerprint-probe-result",
        "state": "ready" if zero_identity_passed and nonzero_hook_active else "failed",
        "campaign_id": str(args.campaign_id),
        "probe_id": str(args.probe_id),
        "base_probe_family": {
            "action_conditioning_scale": "action_scaling",
            "fshc_interaction_local_gain": "interaction_local_trainable_history_correction",
            "fshc_signed_history_gain": "trainable_signed_history_correction",
            "history_retention_gain": "controlled_context_retention",
            "first_frame_anchor_blend": "first_frame_anchoring_strength",
            "sampler_initial_noise_gain": "sampler_noise_stress",
        }[str(args.probe_id)],
        "outcome_names": list(OUTCOME_NAMES),
        "input": {
            "checkpoint": str(Path(args.ckpt_path).resolve()),
            "contexts_json": str(contexts_path),
            "contexts_sha256": hashlib.sha256(contexts_path.read_bytes()).hexdigest(),
            "doses": list(ordered_doses),
            "interact_num": int(args.interact_num),
            "num_inference_steps": int(args.num_inference_steps),
            "selected_identities": [
                {
                    "context_id": str(context["context_id"]),
                    "seed": int(context["seed"]),
                }
                for context in contexts
            ],
            "enable_signed_history_correction": enable_corrector,
            "unsigned_history_gate": bool(args.unsigned_history_gate),
            "enable_multiscale_history_adapter": bool(args.enable_multiscale_history_adapter),
            "multiscale_history_always_on": bool(args.multiscale_history_always_on),
            "fshc_dose_mode": str(args.fshc_dose_mode),
            "zero_reference_mode": str(args.zero_reference_mode),
            "target_interactions": list(target_interactions),
        },
        "runtime": _runtime_receipt(),
        "unwrapped_references": list(references.values()),
        "measurements": measurements,
        "zero_identity_checks": zero_identity_checks,
        "hook_activation": {"state": "passed" if nonzero_hook_active else "failed"},
        "artifacts": {"zero_dose_rollout": "zero-dose-rollout.mp4"},
        "claim_boundary": (
            "Inference-only, source-isolated paired-dose measurement. It establishes neither "
            "a repair effect nor fingerprint-guided repair-selection value."
        ),
    }
    import mediapy

    try:
        import imageio_ffmpeg

        mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
    except ImportError:
        pass
    mediapy.write_video(output_root / "zero-dose-rollout.mp4", visual_sample, fps=4)
    (output_root / "result.json").write_text(
        json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--probe-id", choices=tuple(sorted(PROBES)), required=True)
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
    parser.add_argument("--target-interactions", type=int, nargs="*")
    parser.add_argument("--interact-num", type=int, default=4)
    parser.add_argument("--num-inference-steps", type=int, default=2)
    parser.add_argument("--enable-signed-history-correction", action="store_true")
    parser.add_argument("--unsigned-history-gate", action="store_true")
    parser.add_argument("--enable-multiscale-history-adapter", action="store_true")
    parser.add_argument("--multiscale-history-always-on", action="store_true")
    parser.add_argument(
        "--fshc-dose-mode",
        choices=("absolute_scalar", "normalized_mechanism"),
        default="absolute_scalar",
    )
    parser.add_argument(
        "--zero-reference-mode",
        choices=("unwrapped", "probe-zero"),
        default="unwrapped",
    )
    parser.add_argument("--zero-identity-tolerance", type=float, default=1e-6)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    result = run(_parser().parse_args(argv))
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
