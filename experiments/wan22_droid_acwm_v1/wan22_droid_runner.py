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
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import imageio.v2 as imageio
import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_revision(source: Path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _load_record(manifest_path: Path, index: int, *, video_frames: int, start_offset: int = 0):
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
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    imageio.mimsave(str(output), list(frames), fps=5, codec="libx264", macro_block_size=1)
    return int(frames.shape[0])


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    started = time.monotonic()
    if not torch.cuda.is_available():
        raise RuntimeError("RUNTIME_TORCH_CUDA_UNAVAILABLE")
    if args.max_gpu_hours <= 0 or args.max_gpu_hours > 40:
        raise ValueError("GPU_BUDGET_EXCEEDS_40_HOURS")
    device = torch.device(f"cuda:{args.gpu_index}")
    output = args.output_root.expanduser().resolve()
    output.mkdir(mode=0o700, parents=True, exist_ok=True)
    torch, backbone, vae = _load_runtime(args.source, args.model, device)
    from wan22_droid_adapter import Wan22DroidTokenAdapter, validate_window
    adapter = Wan22DroidTokenAdapter(model_dim=int(backbone.dim)).to(device)
    adapter.attach(backbone)
    context = _context(torch, backbone, device)
    losses: list[float] = []
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.learning_rate)

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
    manifest, first_record, _, _, _, _ = _load_record(args.train_manifest, args.sample_index, video_frames=chunk_sizes[0])
    generated_frames_list: list[np.ndarray] = []
    target_frames_list: list[np.ndarray] = []
    action_sequence: list[np.ndarray] = []
    proprio_sequence: list[np.ndarray] = []
    chunk_receipts: list[dict[str, Any]] = []
    previous_last = None
    for chunk_index, chunk_size in enumerate(chunk_sizes):
        offset = sum(chunk_sizes[:chunk_index])
        manifest, record, actions_np, proprio_np, video, video_path = _load_record(
            args.train_manifest, args.sample_index, video_frames=chunk_size, start_offset=offset
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
        if chunk_index == 0:
            anchor = target[:, :1]
        else:
            anchor = previous_last
        # Only the first chunk is used for adapter optimization in G0/screen;
        # later chunks reuse the same explicitly bound adapter checkpoint.
        if chunk_index == 0:
            noise = torch.randn_like(target)
            backbone.zero_grad(set_to_none=True)
            for step in range(args.steps):
                optimizer.zero_grad(set_to_none=True)
                x_t = target * 0.5 + noise * 0.5
                x_t[:, :1] = anchor
                predicted = _model_call(torch, backbone, adapter, x_t, actions, proprio, context, t=500.0)
                target_velocity = noise - target
                loss = (predicted[:, 1:] - target_velocity[:, 1:]).float().square().mean()
                if not torch.isfinite(loss):
                    raise RuntimeError("WAN22_DROID_NONFINITE_LOSS")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
        adapter.eval()
        from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
        scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
        scheduler.set_timesteps(args.rollout_steps, device=device)
        generated = torch.randn_like(target)
        generated[:, :1] = anchor
        with torch.no_grad():
            for timestep in scheduler.timesteps:
                prediction = _model_call(torch, backbone, adapter, generated, actions, proprio, context, t=float(timestep))
                generated = scheduler.step(prediction.unsqueeze(0), timestep, generated.unsqueeze(0), return_dict=False)[0].squeeze(0)
                generated[:, :1] = anchor
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
            "anchor": "first_observed_frame" if chunk_index == 0 else "previous_generated_last_latent",
            "video_path": str(video_path),
        })
    checkpoint = output / "wan22_droid_adapter.pt"
    torch.save({"adapter": adapter.state_dict(), "model_dim": int(backbone.dim), "action_dim": 7, "proprio_dim": 14}, checkpoint)

    generated_frames_np = np.concatenate(generated_frames_list, axis=0)[:horizon]
    target_frames_np = np.concatenate(target_frames_list, axis=0)[:horizon]
    if generated_frames_np.shape[0] != horizon or target_frames_np.shape[0] != horizon:
        raise RuntimeError(f"WAN22_DROID_PUBLISHED_HORIZON_INVALID:{generated_frames_np.shape[0]}:{target_frames_np.shape[0]}")
    generated_path = output / "generated_150f.mp4"
    generated_frames = _write_mp4(generated_frames_np, generated_path)
    gt_path = output / "ground_truth_150f.mp4"
    _write_mp4(target_frames_np, gt_path)
    worldarena_input = {
        "schema_version": 1,
        "artifact_type": "verdiwm-wan22-droid-worldarena-input",
        "generated_video": str(generated_path),
        "ground_truth_video": str(gt_path),
        "episode_id": str(record["episode_id"]),
        "start_frame": int(first_record["start_frame"]),
        "generated_frames": generated_frames,
        "fps": 5,
        "action_dim": 7,
        "evaluator_contract": str(args.evaluator_contract.expanduser().resolve()),
        "droid_action_sequence": str(output / "droid_conditioning.npz"),
        "droid_proprio_sequence": str(output / "droid_conditioning.npz"),
        "worldarena_summary": str(output / "worldarena_summary.json"),
        "chunk_receipts": chunk_receipts,
        "metrics": ["subject_consistency", "background_consistency", "motion_smoothness", "photometric_smoothness", "trajectory_accuracy", "action_following"],
    }
    (output / "worldarena_input.json").write_text(json.dumps(worldarena_input, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    first_frame_path = output / "first_frame.png"
    imageio.imwrite(str(first_frame_path), generated_frames_np[0])
    summary = [{
        "gt_path": str(gt_path),
        "image": str(first_frame_path),
        "prompt": [str(record.get("instruction", "DROID robot action-conditioned future prediction"))],
    }]
    (output / "worldarena_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    np.savez_compressed(output / "droid_conditioning.npz", action=np.concatenate(action_sequence, axis=0)[:horizon], proprio=np.concatenate(proprio_sequence, axis=0)[:horizon])
    evaluator_receipt = {
        "schema_version": 1,
        "artifact_type": "verdiwm-wan22-droid-evaluator-receipt",
        "evaluator_id": "wan22-droid-worldarena-30s-v1",
        "state": "not_launched" if not args.worldarena_command else "launched",
        "command_bound": bool(args.worldarena_command),
        "input": str(output / "worldarena_input.json"),
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
        "gpu_hours": elapsed_hours,
        "budget_gpu_hours": args.max_gpu_hours,
        "steps": args.steps,
        "losses": losses,
        "model": str(args.model.expanduser().resolve()),
        "model_config_sha256": _sha256(args.model.expanduser().resolve() / "config.json"),
        "source": str(args.source.expanduser().resolve()),
        "source_revision": _source_revision(args.source),
        "training_manifest": str(args.train_manifest.expanduser().resolve()),
        "training_manifest_sha256": _sha256(args.train_manifest.expanduser().resolve()),
        "sample_id": str(first_record["sample_id"]),
        "horizon_frames": horizon,
        "chunk_frames": chunk,
        "autoregressive_chunks": len(chunk_sizes),
        "generated_video": str(generated_path),
        "ground_truth_video": str(gt_path),
        "adapter_checkpoint": str(checkpoint),
        "worldarena_input": str(output / "worldarena_input.json"),
        "evaluator_receipt": str(output / "evaluator_receipt.json"),
        "worldarena_result": evaluator_result,
        "claim_boundary": "This receipt proves a real adapter optimization and rollout artifact. Only a successful frozen WorldArena confirmation can support a 30-second quality claim.",
    }
    (output / "training_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--evaluator-contract", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--horizon-frames", type=int, default=150)
    parser.add_argument("--chunk-frames", type=int, default=45)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--rollout-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-gpu-hours", type=float, default=1.0)
    parser.add_argument("--worldarena-command")
    args = parser.parse_args()
    try:
        print(json.dumps(run(args), ensure_ascii=False, sort_keys=True, indent=2))
        return 0
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        print(json.dumps({"state": "blocked", "error": str(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
