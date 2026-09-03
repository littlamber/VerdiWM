#!/usr/bin/env python3
"""Small, real WAN2.2-DROID adapter runner.

The runner is an external runtime plugin for the VerdiWM control plane.  It
loads the official WAN2.2 source checkout and the user supplied weights,
freezes the 5B backbone, trains only :class:`Wan22DroidTokenAdapter`, then
emits a short latent rollout and a WorldArena input package.  The 45-frame
chunk is one autoregressive unit; four such units are required for a 150-frame
confirmation rollout.  No source checkout is modified.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from wmloop.execute.window_scheduler import build_window_schedule
from wmloop.execute.paired_visualization import create_paired_visualization
from wmloop.execute.training_contract import (
    TrainingContractError,
    minimum_training_episode_count,
    validate_training_binding,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _source_revision(source: Path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _load_adapter(adapter_path: Path):
    """Load the explicitly bound adapter without relying on module search paths."""

    path = adapter_path.expanduser().resolve(strict=True)
    spec = importlib.util.spec_from_file_location(
        f"wan22_droid_adapter_{hashlib.sha256(str(path).encode()).hexdigest()[:12]}",
        path,
    )
    if spec is None or spec.loader is None:
        raise ValueError(f"WAN22_DROID_ADAPTER_IMPORT_INVALID:{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    adapter_class = getattr(module, "Wan22DroidTokenAdapter", None)
    validate_window = getattr(module, "validate_window", None)
    if adapter_class is None or not callable(validate_window):
        raise ValueError(f"WAN22_DROID_ADAPTER_CONTRACT_INVALID:{path}")
    return adapter_class, validate_window, path


def _load_record(manifest_path: Path, index: int, *, video_frames: int, start_offset: int = 0):
    import imageio.v2 as imageio

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("WAN22_DROID_MANIFEST_EMPTY")
    record = records[index % len(records)]
    root = Path(str(manifest["data_root"])).expanduser().resolve()
    annotation = json.loads((root / str(record["annotation_path"])).read_text(encoding="utf-8"))
    start = int(record["start_frame"]) + int(start_offset)
    stop = start + video_frames
    actions = np.asarray(annotation["action"][start:stop], dtype=np.float32)
    proprio = np.asarray(annotation["proprio"][start:stop], dtype=np.float32)
    if actions.shape != (video_frames, 7) or proprio.shape != (video_frames, 14):
        raise ValueError("WAN22_DROID_CONDITIONING_WINDOW_INVALID")
    video_path = root / str(record["video_path"])
    reader = imageio.get_reader(str(video_path))
    frames = []
    try:
        for frame_index in range(start, stop):
            frames.append(reader.get_data(frame_index))
    finally:
        reader.close()
    if len(frames) != video_frames:
        raise ValueError("WAN22_DROID_VIDEO_WINDOW_INVALID")
    video = np.stack(frames).astype(np.float32) / 127.5 - 1.0
    video = __import__("torch").from_numpy(video).permute(3, 0, 1, 2).contiguous()
    return manifest, record, actions, proprio, video, video_path


def _assert_episode_disjoint(train_manifest: Mapping[str, Any], validation_manifest: Mapping[str, Any]) -> None:
    """Fail closed if any episode identity is shared by train and validation."""

    train_records = train_manifest.get("records")
    validation_records = validation_manifest.get("records")
    if (
        not isinstance(train_records, list)
        or not train_records
        or not isinstance(validation_records, list)
        or not validation_records
    ):
        raise ValueError("WAN22_DROID_MANIFEST_RECORDS_INVALID")
    if not all(isinstance(row, Mapping) and str(row.get("episode_id", "")).strip() for row in train_records):
        raise ValueError("WAN22_DROID_TRAIN_EPISODE_ID_MISSING")
    if not all(isinstance(row, Mapping) and str(row.get("episode_id", "")).strip() for row in validation_records):
        raise ValueError("WAN22_DROID_VALIDATION_EPISODE_ID_MISSING")
    train_ids = {str(row["episode_id"]).strip() for row in train_records}
    validation_ids = {str(row["episode_id"]).strip() for row in validation_records}
    overlap = sorted(train_ids & validation_ids)
    if overlap:
        raise ValueError("WAN22_DROID_EPISODE_SPLIT_OVERLAP:" + ",".join(overlap[:8]))


def _validate_control_plane_training_contract(
    args: argparse.Namespace,
) -> dict[str, object] | None:
    """Verify every control-plane field when launched by VerdiWM.

    Direct plugin use remains possible when no contract environment is bound,
    but a scheduler-launched process cannot silently ignore stage, budget, or
    scale-plan identity fields.
    """

    contract = os.environ.get("VERDIWM_TRAINING_CONTRACT")
    if contract is None:
        return None
    required = (
        "VERDIWM_TRAINING_STAGE",
        "VERDIWM_TRAINING_MODE",
        "VERDIWM_TRAINING_STEPS",
        "VERDIWM_TRAINING_RECORD_LIMIT",
        "VERDIWM_TRAINING_SAMPLER",
        "VERDIWM_TRAINING_SEED_COUNT",
        "VERDIWM_TRAINING_SCALE_PLAN_SHA256",
    )
    missing = [key for key in required if key not in os.environ]
    if missing:
        raise ValueError(
            "WAN22_DROID_TRAINING_CONTRACT_FIELDS_MISSING:" + ",".join(missing)
        )
    try:
        stage = os.environ["VERDIWM_TRAINING_STAGE"]
        binding = {
            "train_manifest": str(args.train_manifest.expanduser().resolve()),
            "validation_manifest": str(args.validation_manifest.expanduser().resolve()),
            "mode": os.environ["VERDIWM_TRAINING_MODE"],
            "steps": int(os.environ["VERDIWM_TRAINING_STEPS"]),
            "record_limit": int(os.environ["VERDIWM_TRAINING_RECORD_LIMIT"]),
            "sampler": os.environ["VERDIWM_TRAINING_SAMPLER"],
            "seed_count": int(os.environ["VERDIWM_TRAINING_SEED_COUNT"]),
            "validation_panel_size": int(os.environ.get("VERDIWM_VALIDATION_PANEL_SIZE", "1")),
            "scale_plan_sha256": os.environ["VERDIWM_TRAINING_SCALE_PLAN_SHA256"],
            "runner_contract": contract,
        }
        normalized = validate_training_binding(binding, expected_stage=stage)
    except (KeyError, ValueError, TrainingContractError) as exc:
        raise ValueError(f"WAN22_DROID_TRAINING_CONTRACT_INVALID:{exc}") from exc
    if contract != "VERDIWM_TRAINING_CONTRACT_V1":
        raise ValueError("WAN22_DROID_TRAINING_CONTRACT_VERSION_INVALID")
    if stage != args.training_stage:
        raise ValueError(
            "WAN22_DROID_TRAINING_STAGE_MISMATCH:"
            f"{stage}:expected={args.training_stage}"
        )
    return normalized


def _conditioning_for_mode(
    actions: np.ndarray,
    proprio: np.ndarray,
    mode: str,
    history_decay: float = 0.8,
) -> tuple[np.ndarray, np.ndarray]:
    """Project the same DROID stream into a declared, comparable conditioning arm."""

    if mode not in {"visual_anchor_only", "action", "action_proprio", "action_proprio_history", "action_proprio_ema"}:
        raise ValueError(f"WAN22_DROID_CONDITIONING_MODE_INVALID:{mode}")
    if not 0.0 < float(history_decay) <= 1.0:
        raise ValueError("WAN22_DROID_HISTORY_DECAY_INVALID")
    actions = np.asarray(actions, dtype=np.float32).copy()
    proprio = np.asarray(proprio, dtype=np.float32).copy()
    if mode == "visual_anchor_only":
        actions.fill(0.0)
        proprio.fill(0.0)
    elif mode == "action":
        proprio.fill(0.0)
    elif mode == "action_proprio_history":
        # Causal prefix means: no future condition leaks into an earlier token.
        actions = np.cumsum(actions, axis=0) / np.arange(1, len(actions) + 1, dtype=np.float32)[:, None]
        proprio = np.cumsum(proprio, axis=0) / np.arange(1, len(proprio) + 1, dtype=np.float32)[:, None]
    elif mode == "action_proprio_ema":
        # Causal exponential state: old observations remain available but
        # their influence decays, limiting long-horizon stale-state leakage.
        decay = float(history_decay)
        state_a = np.zeros((actions.shape[1],), dtype=np.float32)
        state_p = np.zeros((proprio.shape[1],), dtype=np.float32)
        for index in range(len(actions)):
            state_a = decay * state_a + (1.0 - decay) * actions[index]
            state_p = decay * state_p + (1.0 - decay) * proprio[index]
            actions[index] = state_a
            proprio[index] = state_p
    return actions, proprio


def _chunk_anchor(first_reference: Any, previous_last: Any, chunk_index: int, policy: str, refresh_strength: float) -> tuple[Any, str]:
    """Select a causal chunk boundary anchor from already observed/generated state."""

    if policy not in {"previous_generated", "initial_reference_blend"}:
        raise ValueError(f"WAN22_DROID_ANCHOR_POLICY_INVALID:{policy}")
    if not 0.0 <= float(refresh_strength) <= 1.0:
        raise ValueError("WAN22_DROID_ANCHOR_REFRESH_STRENGTH_INVALID")
    if chunk_index == 0:
        return first_reference, "first_observed_frame"
    if previous_last is None:
        raise ValueError("WAN22_DROID_PREVIOUS_ANCHOR_MISSING")
    if policy == "previous_generated":
        return previous_last, "previous_generated_last_latent"
    strength = float(refresh_strength)
    return (
        previous_last * (1.0 - strength) + first_reference * strength,
        "previous_generated_initial_reference_blend",
    )


def _select_branch_index(
    terminal_states: list[np.ndarray],
    first_reference: np.ndarray,
    previous_last: np.ndarray | None,
    reference_weight: float,
) -> tuple[int, list[float]]:
    """Select a rollout branch using only causal latent-state consistency."""

    if not terminal_states:
        raise ValueError("WAN22_DROID_BRANCHES_EMPTY")
    if not 0.0 <= float(reference_weight) <= 1.0:
        raise ValueError("WAN22_DROID_BRANCH_REFERENCE_WEIGHT_INVALID")

    def cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
        left_flat = np.asarray(left, dtype=np.float64).reshape(-1)
        right_flat = np.asarray(right, dtype=np.float64).reshape(-1)
        denominator = float(np.linalg.norm(left_flat) * np.linalg.norm(right_flat))
        return 1.0 if denominator <= 1e-12 else 1.0 - float(np.dot(left_flat, right_flat) / denominator)

    reference = np.asarray(first_reference)
    continuity = reference if previous_last is None else np.asarray(previous_last)
    weight = float(reference_weight)
    scores = [
        weight * cosine_distance(state, reference)
        + (1.0 - weight) * cosine_distance(state, continuity)
        for state in terminal_states
    ]
    return min(range(len(scores)), key=scores.__getitem__), scores


def _training_record_schedule(
    records: list[Mapping[str, Any]], sample_index: int, mode: str,
    record_limit: int, steps: int, *, sampler: str = "episode_balanced",
    seed: int = 0, chunk_frames: int = 45,
) -> list[tuple[int, int]]:
    """Compatibility wrapper for the generic wmloop window scheduler."""

    try:
        return build_window_schedule(
            records, steps=steps, mode=mode, record_limit=record_limit,
            sampler=sampler, seed=seed, chunk_frames=chunk_frames,
            sample_index=sample_index,
        )
    except ValueError as exc:
        raise ValueError(str(exc).replace("WINDOW_SCHEDULER_", "WAN22_DROID_TRAINING_", 1)) from exc


def _validate_scheduled_training_coverage(
    records: list[Mapping[str, Any]],
    schedule: list[tuple[int, int]],
    stage: str,
) -> None:
    """Require the declared formal stage's episode coverage in used windows."""

    episode_ids = {
        str(records[record_index].get("episode_id", "")).strip()
        for record_index, _offset in schedule
    }
    episode_ids.discard("")
    required = minimum_training_episode_count(stage)
    if len(episode_ids) < required:
        raise ValueError(
            "WAN22_DROID_TRAINING_EPISODE_COVERAGE_TOO_LOW:"
            f"have={len(episode_ids)}:need={required}"
        )


def _load_runtime(source: Path, model_path: Path, device: Any):
    source = source.expanduser().resolve()
    sys.path.insert(0, str(source))
    import torch
    import torch.nn.functional as F
    import wan.modules.attention as attention_module
    import wan.modules.model as model_module
    from wan.modules.model import WanModel
    from wan.modules.vae2_2 import Wan2_2_VAE

    # The official checkout makes FlashAttention-2 mandatory in its low-level
    # helper.  SDPA is a semantically equivalent fallback available in the
    # supported PyTorch runtime, so the runner remains portable across GPU
    # environments without altering the imported source tree.
    def sdpa_fallback(q, k, v, q_lens=None, k_lens=None, dropout_p=0.0,
                      softmax_scale=None, q_scale=None, causal=False,
                      window_size=(-1, -1), deterministic=False,
                      dtype=torch.bfloat16, version=None):
        del window_size, deterministic, version
        out_dtype = q.dtype
        q = q if q.dtype in (torch.float16, torch.bfloat16) else q.to(dtype)
        k = k if k.dtype in (torch.float16, torch.bfloat16) else k.to(dtype)
        v = v if v.dtype in (torch.float16, torch.bfloat16) else v.to(dtype)
        if q_scale is not None:
            q = q * q_scale
        # Wan attention uses [B, L, heads, head_dim]; SDPA uses [B, heads, L, head_dim].
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        mask = None
        if k_lens is not None:
            max_k = k.shape[-2]
            lengths = k_lens.to(device=k.device).view(-1, 1, 1, 1)
            mask = torch.arange(max_k, device=k.device).view(1, 1, 1, max_k) < lengths
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=dropout_p,
            is_causal=causal, scale=softmax_scale,
        )
        return out.transpose(1, 2).contiguous().to(out_dtype)

    attention_module.flash_attention = sdpa_fallback
    model_module.flash_attention = sdpa_fallback

    model = WanModel.from_pretrained(str(model_path), torch_dtype=torch.bfloat16)
    model.eval().requires_grad_(False).to(device)
    vae = Wan2_2_VAE(
        vae_pth=str(model_path / "Wan2.2_VAE.pth"),
        device=device,
    )
    vae.model.eval().requires_grad_(False)
    return torch, model, vae


def _context(torch, model, device):
    # A zero context keeps the G0 path independent of a separately managed T5
    # service.  Production candidates may provide a bound text-encoder plugin.
    return [torch.zeros((1, int(model.text_dim)), device=device, dtype=torch.bfloat16)]


def _seq_len(x: Any, model: Any) -> int:
    _, frames, height, width = x.shape
    pt, ph, pw = tuple(model.patch_size)
    return int((frames // pt) * (height // ph) * (width // pw))


def _model_call(torch, model, adapter, x, actions, proprio, context, *, t):
    adapter.set_conditions(actions, proprio)
    seq_len = _seq_len(x, model)
    timestep = torch.full((1, seq_len), float(t), device=x.device, dtype=torch.float32)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return model([x], t=timestep, context=context, seq_len=seq_len)[0]


def _decode_frames(vae, latent):
    decoded = vae.decode([latent])[0].detach().float().cpu().clamp(-1, 1)
    return ((decoded.permute(1, 2, 3, 0).numpy() + 1.0) * 127.5).round().astype(np.uint8)


def _write_mp4(frames: np.ndarray, output: Path) -> int:
    import imageio.v2 as imageio

    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    imageio.mimsave(str(output), list(frames), fps=5, codec="libx264", macro_block_size=1)
    return int(frames.shape[0])


def _validation_panel_indices(
    validation_manifest: Mapping[str, Any],
    *,
    requested: list[int] | None,
    fallback_index: int,
    formal: bool,
) -> list[int]:
    """Resolve a frozen, episode-diverse validation panel before GPU work."""

    records = validation_manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("WAN22_DROID_VALIDATION_PANEL_EMPTY")
    if requested is None:
        indices = [fallback_index] if not formal else [fallback_index + offset for offset in range(3)]
    else:
        indices = [int(value) for value in requested]
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("WAN22_DROID_VALIDATION_PANEL_INDICES_INVALID")
    if any(index < 0 or index >= len(records) for index in indices):
        raise ValueError("WAN22_DROID_VALIDATION_PANEL_INDEX_OUT_OF_RANGE")
    episodes = [str(records[index].get("episode_id", "")).strip() for index in indices]
    if any(not episode for episode in episodes):
        raise ValueError("WAN22_DROID_VALIDATION_PANEL_EPISODE_ID_MISSING")
    if len(set(episodes)) != len(episodes):
        raise ValueError("WAN22_DROID_VALIDATION_PANEL_EPISODES_NOT_DISTINCT")
    if formal and len(indices) < 3:
        raise ValueError("WAN22_DROID_FORMAL_VALIDATION_PANEL_TOO_SMALL")
    return indices


def _rollout_validation_sample(
    *, torch: Any, backbone: Any, adapter: Any, vae: Any, validate_window: Any, context: Any,
    device: Any, args: argparse.Namespace, sample_index: int,
    output: Path, chunk_sizes: list[int], horizon: int,
) -> dict[str, Any]:
    """Generate one panel member and all paired inspection handoff artifacts."""

    import imageio.v2 as imageio

    manifest, first_record, _, _, _, _ = _load_record(
        args.validation_manifest, sample_index, video_frames=chunk_sizes[0]
    )
    generated_frames_list: list[np.ndarray] = []
    target_frames_list: list[np.ndarray] = []
    action_sequence: list[np.ndarray] = []
    proprio_sequence: list[np.ndarray] = []
    chunk_receipts: list[dict[str, Any]] = []
    previous_last = None
    first_reference = None
    for chunk_index, chunk_size in enumerate(chunk_sizes):
        offset = sum(chunk_sizes[:chunk_index])
        _, record, actions_np, proprio_np, video, video_path = _load_record(
            args.validation_manifest, sample_index, video_frames=chunk_size, start_offset=offset
        )
        actions_np, proprio_np = _conditioning_for_mode(
            actions_np, proprio_np, args.conditioning_mode, args.history_decay
        )
        validate_window(actions_np.tolist(), proprio_np.tolist(), horizon_frames=chunk_size)
        actions = torch.from_numpy(actions_np).to(device=device)
        proprio = torch.from_numpy(proprio_np).to(device=device)
        action_sequence.append(actions_np)
        proprio_sequence.append(proprio_np)
        with torch.no_grad():
            target = vae.encode([video.to(device)])[0].to(device=device, dtype=torch.bfloat16)
        if target.ndim != 4 or target.shape[0] != 48:
            raise ValueError(f"WAN22_VAE_LATENT_SHAPE_INVALID:{tuple(target.shape)}")
        if first_reference is None:
            first_reference = target[:, :1].detach()
        anchor, anchor_source = _chunk_anchor(
            first_reference, previous_last, chunk_index, args.anchor_policy,
            args.anchor_refresh_strength,
        )
        adapter.eval()
        from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
        if args.branch_count < 1:
            raise ValueError("WAN22_DROID_BRANCH_COUNT_INVALID")
        if args.branch_selection == "first" and args.branch_count != 1:
            raise ValueError("WAN22_DROID_FIRST_BRANCH_SELECTION_REQUIRES_ONE_BRANCH")
        branch_states = []
        for _branch_index in range(args.branch_count):
            scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=1000, shift=1, use_dynamic_shifting=False
            )
            scheduler.set_timesteps(args.rollout_steps, device=device)
            branch = torch.randn_like(target)
            branch[:, :1] = anchor
            with torch.no_grad():
                for timestep in scheduler.timesteps:
                    prediction = _model_call(
                        torch, backbone, adapter, branch, actions, proprio, context,
                        t=float(timestep),
                    )
                    branch = scheduler.step(
                        prediction.unsqueeze(0), timestep, branch.unsqueeze(0),
                        return_dict=False,
                    )[0].squeeze(0)
                    branch[:, :1] = anchor
            branch_states.append(branch)
        branch_scores = [0.0]
        selected_branch = 0
        if args.branch_selection == "terminal_reference_consistency":
            selected_branch, branch_scores = _select_branch_index(
                [state[:, -1:].detach().float().cpu().numpy() for state in branch_states],
                first_reference.detach().float().cpu().numpy(),
                None if previous_last is None else previous_last.detach().float().cpu().numpy(),
                args.branch_reference_weight,
            )
        generated = branch_states[selected_branch]
        generated_chunk_frames = _decode_frames(vae, generated)
        target_chunk_frames = _decode_frames(vae, target)
        generated_frames_list.append(generated_chunk_frames)
        target_frames_list.append(target_chunk_frames)
        previous_last = generated[:, -1:].detach()
        chunk_receipts.append({
            "chunk_index": chunk_index,
            "start_frame": int(record["start_frame"]) + offset,
            "frames": int(chunk_size),
            "published_frames": int(generated_chunk_frames.shape[0]),
            "latent_shape": list(target.shape),
            "anchor": anchor_source,
            "branch_count": args.branch_count,
            "branch_selection": args.branch_selection,
            "branch_reference_weight": args.branch_reference_weight if args.branch_selection == "terminal_reference_consistency" else None,
            "branch_scores": branch_scores,
            "selected_branch": selected_branch,
            "video_path": str(video_path),
        })
    generated_frames_np = np.concatenate(generated_frames_list, axis=0)[:horizon]
    target_frames_np = np.concatenate(target_frames_list, axis=0)[:horizon]
    if generated_frames_np.shape[0] != horizon or target_frames_np.shape[0] != horizon:
        raise RuntimeError(
            f"WAN22_DROID_PUBLISHED_HORIZON_INVALID:{generated_frames_np.shape[0]}:{target_frames_np.shape[0]}"
        )
    output.mkdir(mode=0o700, parents=True, exist_ok=True)
    generated_path = output / "generated_150f.mp4"
    generated_frames = _write_mp4(generated_frames_np, generated_path)
    gt_path = output / "ground_truth_150f.mp4"
    _write_mp4(target_frames_np, gt_path)
    visualization = create_paired_visualization(
        generated_video=generated_path,
        ground_truth_video=gt_path,
        output_root=output / "acwm-gt-visualization",
        metadata={
            "seed": args.seed,
            "sample_index": sample_index,
            "model_id": str(args.model.expanduser().resolve()),
            "training_steps": args.steps,
            "training_mode": args.training_mode,
            "conditioning_mode": args.conditioning_mode,
        },
    )
    first_frame_path = output / "first_frame.png"
    imageio.imwrite(str(first_frame_path), generated_frames_np[0])
    np.savez_compressed(
        output / "droid_conditioning.npz",
        action=np.concatenate(action_sequence, axis=0)[:horizon],
        proprio=np.concatenate(proprio_sequence, axis=0)[:horizon],
    )
    worldarena_input = {
        "schema_version": 1,
        "artifact_type": "verdiwm-wan22-droid-worldarena-input",
        "generated_video": str(generated_path),
        "ground_truth_video": str(gt_path),
        "paired_visualization": visualization["manifest_path"],
        "sample_index": sample_index,
        "sample_id": str(first_record["sample_id"]),
        "episode_id": str(first_record["episode_id"]),
        "start_frame": int(first_record["start_frame"]),
        "generated_frames": generated_frames,
        "fps": 5,
        "action_dim": 7,
        "evaluator_contract": str(args.evaluator_contract.expanduser().resolve()),
        "droid_conditioning": str(output / "droid_conditioning.npz"),
        "droid_action_sequence": str(output / "droid_conditioning.npz"),
        "droid_proprio_sequence": str(output / "droid_conditioning.npz"),
        "worldarena_summary": str(output / "worldarena_summary.json"),
        "chunk_receipts": chunk_receipts,
        "metrics": ["subject_consistency", "background_consistency", "motion_smoothness", "photometric_smoothness", "trajectory_accuracy", "action_following"],
    }
    (output / "worldarena_input.json").write_text(
        json.dumps(worldarena_input, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = [{
        "gt_path": str(gt_path),
        "image": str(first_frame_path),
        "prompt": [str(first_record.get("instruction", "DROID robot action-conditioned future prediction"))],
    }]
    (output / "worldarena_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "evaluator_receipt.json").write_text(json.dumps({
        "schema_version": 1,
        "artifact_type": "verdiwm-wan22-droid-evaluator-receipt",
        "evaluator_id": "wan22-droid-worldarena-30s-v1",
        "state": "not_launched",
        "command_bound": False,
        "input": str(output / "worldarena_input.json"),
        "sample_id": str(first_record["sample_id"]),
        "episode_id": str(first_record["episode_id"]),
        "generated_frames": generated_frames,
        "fps": 5,
        "claim_boundary": "No metric claim is valid until the frozen WorldArena command returns zero and emits metrics for this input.",
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "sample_index": sample_index,
        "sample_id": str(first_record["sample_id"]),
        "episode_id": str(first_record["episode_id"]),
        "generated_video": str(generated_path),
        "ground_truth_video": str(gt_path),
        "generated_frames": generated_frames,
        "paired_visualization": visualization["manifest_path"],
        "chunk_receipts": chunk_receipts,
        "action_sequence": np.concatenate(action_sequence, axis=0)[:horizon],
        "proprio_sequence": np.concatenate(proprio_sequence, axis=0)[:horizon],
    }


def _validate_stage_training_mode(training_stage: str, training_mode: str) -> None:
    if training_stage in {"screen", "pilot", "confirm"} and training_mode != "long":
        raise ValueError("WAN22_DROID_FORMAL_STAGE_REQUIRES_LONG_TRAINING")


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    started = time.monotonic()
    if not torch.cuda.is_available():
        raise RuntimeError("RUNTIME_TORCH_CUDA_UNAVAILABLE")
    if args.max_gpu_hours <= 0 or args.max_gpu_hours > 40:
        raise ValueError("GPU_BUDGET_EXCEEDS_40_HOURS")
    _validate_stage_training_mode(args.training_stage, args.training_mode)
    if args.training_mode == "long":
        if args.steps < 64:
            raise ValueError("WAN22_DROID_LONG_TRAINING_STEPS_TOO_LOW")
        if args.train_record_limit and args.train_record_limit < 4:
            raise ValueError("WAN22_DROID_LONG_TRAINING_RECORD_DIVERSITY_TOO_LOW")
        if args.training_sampler != "episode_balanced":
            raise ValueError("WAN22_DROID_FORMAL_TRAINING_EPISODE_BALANCED_REQUIRED")
    elif args.steps > 8:
        raise ValueError("WAN22_DROID_PROBE_STEP_LIMIT_EXCEEDED")
    control_plane_binding = _validate_control_plane_training_contract(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(f"cuda:{args.gpu_index}")
    output = args.output_root.expanduser().resolve()
    output.mkdir(mode=0o700, parents=True, exist_ok=True)
    train_manifest_payload = json.loads(args.train_manifest.read_text(encoding="utf-8"))
    validation_manifest_payload = json.loads(args.validation_manifest.read_text(encoding="utf-8"))
    _assert_episode_disjoint(train_manifest_payload, validation_manifest_payload)
    panel_indices = _validation_panel_indices(
        validation_manifest_payload,
        requested=getattr(args, "validation_sample_indices", None),
        fallback_index=int(args.sample_index),
        formal=args.training_mode == "long",
    )
    torch, backbone, vae = _load_runtime(args.source, args.model, device)
    Wan22DroidTokenAdapter, validate_window, adapter_path = _load_adapter(args.adapter)
    adapter = Wan22DroidTokenAdapter(model_dim=int(backbone.dim)).to(device)
    adapter.attach(backbone)
    context = _context(torch, backbone, device)
    losses: list[float] = []
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.learning_rate)

    # The adapter is optimized only on train data.  Every published rollout
    # and paired ground truth is selected from validation data, and the split
    # identities are checked before any GPU work begins.
    # A 150-frame confirmation is represented as three full 45-frame latent
    # chunks plus a 17-frame tail.  Wan2.2's causal VAE emits two fewer frames
    # for the shortest tail; generating two extra source frames and truncating
    # the final decoded stream keeps the published artifact exactly 150 frames.
    # The first latent frame of every
    # chunk is anchored to the previous generated chunk, so this is genuinely
    # autoregressive rather than four independent samples.
    horizon = int(args.horizon_frames)
    chunk = int(args.chunk_frames)
    if horizon != 150 or chunk != 45:
        raise ValueError("WAN22_DROID_CONFIRMATION_REQUIRES_150X45")
    chunk_sizes = [chunk] * (horizon // chunk) + ([horizon % chunk + 2] if horizon % chunk else [])
    train_records = train_manifest_payload.get("records", [])
    if not isinstance(train_records, list) or not train_records:
        raise ValueError("WAN22_DROID_TRAIN_RECORDS_INVALID")
    schedule = _training_record_schedule(
        train_records, int(args.sample_index), args.training_mode,
        int(args.train_record_limit), int(args.steps),
        sampler=args.training_sampler, seed=int(args.seed), chunk_frames=45,
    )
    if args.training_mode == "long":
        _validate_scheduled_training_coverage(
            train_records, schedule, args.training_stage
        )
    training_window_indices: set[int] = set()
    training_episode_ids: set[str] = set()
    training_chunk_offsets: set[int] = set()
    progress_path = output / "training_progress.json"
    for update_index, (record_index, start_offset) in enumerate(schedule, start=1):
        _, train_record, actions_np, proprio_np, video, _ = _load_record(
            args.train_manifest, record_index, video_frames=45, start_offset=start_offset,
        )
        actions_np, proprio_np = _conditioning_for_mode(actions_np, proprio_np, args.conditioning_mode, args.history_decay)
        train_actions = torch.from_numpy(actions_np).to(device=device)
        train_proprio = torch.from_numpy(proprio_np).to(device=device)
        with torch.no_grad():
            train_target = vae.encode([video.to(device)])[0].to(device=device, dtype=torch.bfloat16)
        if train_target.ndim != 4 or train_target.shape[0] != 48:
            raise ValueError(f"WAN22_VAE_LATENT_SHAPE_INVALID:{tuple(train_target.shape)}")
        noise = torch.randn_like(train_target)
        backbone.zero_grad(set_to_none=True)
        adapter.set_conditions(train_actions, train_proprio)
        optimizer.zero_grad(set_to_none=True)
        x_t = train_target * 0.5 + noise * 0.5
        x_t[:, :1] = train_target[:, :1]
        predicted = _model_call(torch, backbone, adapter, x_t, train_actions, train_proprio, context, t=500.0)
        target_velocity = noise - train_target
        loss = (predicted[:, 1:] - target_velocity[:, 1:]).float().square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError("WAN22_DROID_NONFINITE_LOSS")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        training_window_indices.add(record_index)
        training_episode_ids.add(str(train_record["episode_id"]))
        training_chunk_offsets.add(start_offset)
        if update_index == 1 or update_index % 16 == 0 or update_index == len(schedule):
            _write_json_atomic(progress_path, {
                "schema_version": 1,
                "artifact_type": "verdiwm-wan22-droid-training-progress",
                "state": "completed" if update_index == len(schedule) else "running",
                "training_mode": args.training_mode,
                "training_stage": args.training_stage,
                "training_sampler": args.training_sampler,
                "optimization_updates_completed": update_index,
                "optimization_updates_total": len(schedule),
                "training_windows": len(training_window_indices),
                "training_episodes": len(training_episode_ids),
                "training_chunk_offsets": sorted(training_chunk_offsets),
                "latest_loss": losses[-1],
                "recent_mean_loss": float(np.mean(losses[-16:])),
            })

    checkpoint = output / "wan22_droid_adapter.pt"
    torch.save({"adapter": adapter.state_dict(), "model_dim": int(backbone.dim), "action_dim": 7, "proprio_dim": 14}, checkpoint)
    panel_rows: list[dict[str, Any]] = []
    panel_results: list[dict[str, Any]] = []
    for position, sample_index in enumerate(panel_indices):
        sample_output = output if position == 0 else output / "validation_panel" / f"sample-{sample_index}"
        result = _rollout_validation_sample(
            torch=torch, backbone=backbone, adapter=adapter, vae=vae,
            validate_window=validate_window, context=context, device=device,
            args=args, sample_index=sample_index,
            output=sample_output, chunk_sizes=chunk_sizes, horizon=horizon,
        )
        panel_results.append(result)
        panel_rows.append({
            "sample_index": result["sample_index"],
            "sample_id": result["sample_id"],
            "episode_id": result["episode_id"],
            "run_root": str(sample_output),
            "generated_video": result["generated_video"],
            "ground_truth_video": result["ground_truth_video"],
            "paired_visualization": result["paired_visualization"],
        })
    _write_json_atomic(output / "validation_panel.json", {
        "schema_version": 1,
        "artifact_type": "verdiwm-wan22-droid-validation-panel",
        "state": "frozen",
        "panel_size": len(panel_rows),
        "selection_policy": "explicit_indices_or_first_three_episode_diverse_records_v1",
        "rows": panel_rows,
        "claim_boundary": "The panel is a fixed evaluation sample set; it does not by itself establish model quality or promotion.",
    })
    first_result = panel_results[0]
    first_record = {"episode_id": first_result["episode_id"], "sample_id": first_result["sample_id"]}
    generated_path = Path(str(first_result["generated_video"]))
    gt_path = Path(str(first_result["ground_truth_video"]))
    generated_frames = int(first_result["generated_frames"])
    chunk_receipts = first_result["chunk_receipts"]
    evaluator_receipt = {
        "schema_version": 1,
        "artifact_type": "verdiwm-wan22-droid-evaluator-receipt",
        "evaluator_id": "wan22-droid-worldarena-30s-v1",
        "state": "not_launched" if not args.worldarena_command else "launched",
        "command_bound": bool(args.worldarena_command),
        "input": str(output / "worldarena_input.json"),
        "sample_id": first_result["sample_id"],
        "episode_id": first_result["episode_id"],
        "generated_frames": generated_frames,
        "fps": 5,
        "claim_boundary": "No metric claim is valid until the frozen WorldArena command returns zero and emits metrics for this input.",
    }
    (output / "evaluator_receipt.json").write_text(json.dumps(evaluator_receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    evaluator_result = None
    if args.worldarena_command:
        command = shlex.split(args.worldarena_command)
        completed = subprocess.run(command, cwd=str(args.source.expanduser().resolve()), capture_output=True, text=True)
        (output / "worldarena.stdout.log").write_text(completed.stdout, encoding="utf-8")
        (output / "worldarena.stderr.log").write_text(completed.stderr, encoding="utf-8")
        evaluator_result = {"returncode": completed.returncode, "command": command}
    elapsed_hours = (time.monotonic() - started) / 3600.0
    if elapsed_hours > args.max_gpu_hours:
        raise RuntimeError("GPU_BUDGET_EXCEEDED")
    receipt = {
        "schema_version": 1,
        "artifact_type": "verdiwm-wan22-droid-training-receipt",
        "state": "evaluated" if evaluator_result and evaluator_result["returncode"] == 0 else "awaiting_worldarena_evaluation",
        "gpu_index": args.gpu_index,
        "seed": args.seed,
        "conditioning_mode": args.conditioning_mode,
        "history_decay": args.history_decay if args.conditioning_mode == "action_proprio_ema" else None,
        "anchor_policy": args.anchor_policy,
        "anchor_refresh_strength": args.anchor_refresh_strength if args.anchor_policy == "initial_reference_blend" else None,
        "branch_count": args.branch_count,
        "branch_selection": args.branch_selection,
        "branch_reference_weight": args.branch_reference_weight if args.branch_selection == "terminal_reference_consistency" else None,
        "gpu_hours": elapsed_hours,
        "budget_gpu_hours": args.max_gpu_hours,
        "steps": args.steps,
        "training_stage": args.training_stage,
        "training_mode": args.training_mode,
        "training_sampler": args.training_sampler,
        "training_scheduler": "wmloop.execute.window_scheduler:build_window_schedule@v1",
        "training_windows": len(training_window_indices),
        "training_episodes": len(training_episode_ids),
        "training_chunk_offsets": sorted(training_chunk_offsets),
        "train_record_count": len(train_records),
        "train_record_limit": args.train_record_limit,
        "optimization_updates": len(losses),
        "training_progress": str(progress_path),
        "losses": losses,
        "model": str(args.model.expanduser().resolve()),
        "model_config_sha256": _sha256(args.model.expanduser().resolve() / "config.json"),
        "source": str(args.source.expanduser().resolve()),
        "source_revision": _source_revision(args.source),
        "adapter": str(adapter_path),
        "adapter_sha256": _sha256(adapter_path),
        "training_manifest": str(args.train_manifest.expanduser().resolve()),
        "training_manifest_sha256": _sha256(args.train_manifest.expanduser().resolve()),
        "validation_manifest": str(args.validation_manifest.expanduser().resolve()),
        "validation_manifest_sha256": _sha256(args.validation_manifest.expanduser().resolve()),
        "control_plane_training_contract": control_plane_binding,
        "episode_disjoint_validation": True,
        "sample_id": str(first_record["sample_id"]),
        "validation_panel": str(output / "validation_panel.json"),
        "validation_panel_size": len(panel_rows),
        "validation_sample_indices": panel_indices,
        "validation_sample_ids": [row["sample_id"] for row in panel_rows],
        "validation_episode_ids": [row["episode_id"] for row in panel_rows],
        "horizon_frames": horizon,
        "chunk_frames": chunk,
        "autoregressive_chunks": len(chunk_sizes),
        "generated_video": str(generated_path),
        "ground_truth_video": str(gt_path),
        "paired_visualization": first_result["paired_visualization"],
        "adapter_checkpoint": str(checkpoint),
        "worldarena_input": str(output / "worldarena_input.json"),
        "evaluator_receipt": str(output / "evaluator_receipt.json"),
        "worldarena_result": evaluator_result,
        "claim_boundary": "This receipt proves a real adapter optimization and rollout artifact. Only a successful frozen WorldArena confirmation can support a 30-second quality claim.",
    }
    (output / "training_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--evaluator-contract", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=4101)
    parser.add_argument("--conditioning-mode", choices=("visual_anchor_only", "action", "action_proprio", "action_proprio_history", "action_proprio_ema"), default="action_proprio")
    parser.add_argument("--history-decay", type=float, default=0.8)
    parser.add_argument("--anchor-policy", choices=("previous_generated", "initial_reference_blend"), default="previous_generated")
    parser.add_argument("--anchor-refresh-strength", type=float, default=0.25)
    parser.add_argument("--branch-count", type=int, default=1)
    parser.add_argument("--branch-selection", choices=("first", "terminal_reference_consistency"), default="first")
    parser.add_argument("--branch-reference-weight", type=float, default=0.7)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--validation-sample-indices", type=int, nargs="+", default=None,
                        help="frozen validation panel indices; formal long runs require three distinct episodes")
    parser.add_argument("--horizon-frames", type=int, default=150)
    parser.add_argument("--chunk-frames", type=int, default=45)
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--training-stage", choices=("screen", "pilot", "confirm"), default="pilot")
    parser.add_argument("--training-mode", choices=("probe", "long"), default="long")
    parser.add_argument("--training-sampler", choices=("sequential", "episode_balanced"), default="episode_balanced")
    parser.add_argument("--train-record-limit", type=int, default=256)
    parser.add_argument("--rollout-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-gpu-hours", type=float, default=12.0)
    parser.add_argument("--worldarena-command")
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        print(json.dumps(run(args), ensure_ascii=False, sort_keys=True, indent=2))
        return 0
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        print(json.dumps({"state": "blocked", "error": str(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
