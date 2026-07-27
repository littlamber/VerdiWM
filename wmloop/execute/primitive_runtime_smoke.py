"""M2 runtime smoke for a medium primitive on the real ACWM training entrypoint."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from wmloop.archive.store import ArchiveStore, ContentAddressedStore
from wmloop.execute.agent_staging import AgentRepairSession
from wmloop.execute.gpu_exclusivity_audit import verify_gpu_exclusivity_ready
from wmloop.execute.primitive_smoke import _apply_diff, _default_hook_ios
from wmloop.execute.sandbox import SandboxLease, WorktreeSandbox
from wmloop.primitives.registry import PrimitiveRegistry
from wmloop.primitives.render import PrimitiveRenderer
from wmloop.runtime_env import runtime_subprocess_env
from wmloop.vendor import verify_vendor_checkout


class PrimitiveRuntimeSmokeError(RuntimeError):
    """A medium primitive runtime smoke could not produce trustworthy evidence."""


def run_primitive_runtime_smoke(
    *,
    repo_root: Path,
    output_root: Path,
    runtime_python: Path,
    data_root: Path,
    checkpoint_root: Path,
    primitive: str = "latent_motion_prior",
    gpu_index: int = 1,
    condition: str = "smoke_frontier",
    n_episodes: int = 2,
    weight: float = 0.2,
    action_balance_blend: float = 0.5,
    action_balance_max_gain: float = 4.0,
    history_noise: float = 0.1,
    keep_tokens: int = 2,
    frontier_weight: float = 1.0,
    event_weight: float = 4.0,
    event_quantile: float = 0.75,
    event_visual_blend: float = 0.7,
    guidance_start: float = 1.0,
    guidance_end: float = 1.5,
    anchor_every: int = 4,
    anchor_weight: float = 0.25,
    memory_slots: int = 2,
    memory_weight: float = 0.25,
    reward_weight: float = 0.5,
    inv_dyn_steps: int = 1,
    inv_dyn_lr: float = 0.00001,
    self_forcing_rollout_horizon: int = 16,
    self_forcing_steps: int = 1,
    self_forcing_lr: float = 0.00001,
    wmsd_teacher_ema: float = 0.9,
    wmsd_steps: int = 1,
    wmsd_lr: float = 0.00001,
    next_forcing_chunks: int = 2,
    next_forcing_steps: int = 1,
    next_forcing_lr: float = 0.00001,
    run_training: bool = True,
    hook_timeout_seconds: float = 60.0,
    training_timeout_seconds: float = 1200.0,
    keep_worktree: bool = False,
    archive_db: Path | None = None,
    cas_root: Path | None = None,
    gpu_exclusivity_audit_manifest: Path | None = None,
    gpu_exclusivity_max_age_seconds: float | None = 300.0,
) -> dict[str, object]:
    root = Path(repo_root).resolve()
    runtime = Path(runtime_python).expanduser().absolute()
    data = Path(data_root).resolve()
    checkpoints = Path(checkpoint_root).resolve()
    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise PrimitiveRuntimeSmokeError("PRIMITIVE_RUNTIME_SMOKE_OUTPUT_EXISTS")
    if gpu_index < 0:
        raise PrimitiveRuntimeSmokeError("PRIMITIVE_RUNTIME_SMOKE_GPU_INVALID")
    gpu_exclusivity = None
    if run_training:
        gpu_exclusivity = verify_gpu_exclusivity_ready(
            gpu_exclusivity_audit_manifest,
            gpu_index=gpu_index,
            max_age_seconds=gpu_exclusivity_max_age_seconds,
        )
        _validate_training_assets(runtime=runtime, data_root=data, checkpoint_root=checkpoints)

    source_revision = verify_vendor_checkout(root)
    registry = PrimitiveRegistry.from_root(root)
    primitive_manifest = registry.manifest(primitive)
    primitive_params = _primitive_params(
        primitive=primitive,
        condition=condition,
        n_episodes=n_episodes,
        weight=weight,
        action_balance_blend=action_balance_blend,
        action_balance_max_gain=action_balance_max_gain,
        history_noise=history_noise,
        keep_tokens=keep_tokens,
        frontier_weight=frontier_weight,
        event_weight=event_weight,
        event_quantile=event_quantile,
        event_visual_blend=event_visual_blend,
        guidance_start=guidance_start,
        guidance_end=guidance_end,
        anchor_every=anchor_every,
        anchor_weight=anchor_weight,
        memory_slots=memory_slots,
        memory_weight=memory_weight,
        reward_weight=reward_weight,
        inv_dyn_steps=inv_dyn_steps,
        inv_dyn_lr=inv_dyn_lr,
        self_forcing_rollout_horizon=self_forcing_rollout_horizon,
        self_forcing_steps=self_forcing_steps,
        self_forcing_lr=self_forcing_lr,
        wmsd_teacher_ema=wmsd_teacher_ema,
        wmsd_steps=wmsd_steps,
        wmsd_lr=wmsd_lr,
        next_forcing_chunks=next_forcing_chunks,
        next_forcing_steps=next_forcing_steps,
        next_forcing_lr=next_forcing_lr,
    )
    renderer = PrimitiveRenderer(registry)
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    sandbox_root = temporary / "sandbox"
    sandbox = WorktreeSandbox(vendor_root=root / "vendor" / "ACWM-Phys", runs_root=sandbox_root)
    lease: SandboxLease | None = None
    worktree_removed = False
    try:
        temporary.mkdir(mode=0o700, parents=True)
        lease = sandbox.create(trial_id=f"{primitive.replace('_', '-')}-runtime-smoke", expected_revision=source_revision)
        rendered = renderer.render_checked(
            worktree=lease.worktree,
            interventions=({"primitive": primitive, "params": primitive_params},),
            hook_ios=_default_hook_ios(),
        )
        for item in rendered:
            _apply_diff(lease.worktree, item.diff)
        training_config_path = temporary / f"{primitive.replace('_', '-')}-train-smoke.yaml"
        training_config = _training_config(
            checkpoints,
            primitive=primitive,
            weight=weight,
            learning_rate=next_forcing_lr if primitive == "next_forcing" else None,
        )
        _write_bytes_atomic(training_config_path, _canonical_json_bytes(training_config))
        required_labels = ["runtime_hook_unit"]
        if run_training:
            required_labels.append("acwm_train_smoke")
        session = AgentRepairSession(
            worktree=lease.worktree,
            staging_root=temporary / "staging",
            candidate_id=f"{primitive.replace('_', '-')}-runtime",
            source_revision=source_revision,
            registry_digest=registry.digest(),
            required_check_labels=required_labels,
            environment=_runtime_environment(
                runtime_python=runtime,
                worktree=lease.worktree,
                repo_root=root,
                output_root=temporary,
                data_root=data,
                checkpoint_root=checkpoints,
                gpu_index=gpu_index,
            ),
        )
        session.run(
            label="runtime_hook_unit",
            argv=(str(runtime), "-c", hook_unit_script(primitive)),
            timeout_seconds=hook_timeout_seconds,
        )
        if run_training:
            session.run(
                label="acwm_train_smoke",
                argv=(str(runtime), "train.py", "--config", str(training_config_path)),
                timeout_seconds=training_timeout_seconds,
            )
        candidate = session.seal()
        if not keep_worktree and lease is not None:
            sandbox.remove(lease)
            worktree_removed = True
        report = {
            "schema_version": 1,
            "artifact_type": "wmloop-m2-medium-primitive-runtime-smoke-report",
            "state": "ready" if run_training and candidate.ready_for_promotion else "hook_only" if candidate.ready_for_promotion else "checks_failed",
            "primitive": primitive,
            "primitive_cost_class": primitive_manifest.cost_class,
            "primitive_params": primitive_params,
            "source_revision": source_revision,
            "registry_digest": registry.digest(),
            "runtime_python": str(runtime),
            "data_root": str(data),
            "checkpoint_root": str(checkpoints),
            "gpu_index": gpu_index,
            "run_training": run_training,
            "gpu_exclusivity_audit": gpu_exclusivity,
            "worktree": str(lease.worktree),
            "worktree_removed": worktree_removed,
            "rendered_primitives": [{"name": item.name, "diff_sha256": item.sha256} for item in rendered],
            "training_config_path": str(training_config_path),
            "training_config_sha256": _sha256_bytes(_canonical_json_bytes(training_config)),
            "candidate": candidate.to_document(),
            "candidate_manifest_path": str(candidate.manifest_path),
            "candidate_diff_path": str(candidate.diff_path),
            "limitations": [
                "This is a one-step runtime smoke for hook connectivity, not a model-quality fine-tune.",
                "The trial worktree patch does not modify ACWM evaluator files or frozen dataset splits.",
            ],
        }
        manifest = _write_report_bundle(
            report=report,
            output_root=destination,
            archive_db=archive_db,
            cas_root=cas_root,
            extra_files={
                "candidate_manifest": candidate.manifest_path,
                "candidate_diff": candidate.diff_path,
                "training_config": training_config_path,
            },
            extra_directories={"attempts": candidate.manifest_path.parent / "attempts"},
        )
        return manifest
    except Exception:
        if lease is not None and not keep_worktree:
            try:
                sandbox.remove(lease)
            except Exception:
                pass
        raise
    finally:
        if (temporary.exists() or temporary.is_symlink()) and not keep_worktree:
            shutil.rmtree(temporary, ignore_errors=True)


def _validate_training_assets(*, runtime: Path, data_root: Path, checkpoint_root: Path) -> None:
    if not runtime.is_file():
        raise PrimitiveRuntimeSmokeError("PRIMITIVE_RUNTIME_SMOKE_RUNTIME_MISSING")
    if not data_root.is_dir():
        raise PrimitiveRuntimeSmokeError("PRIMITIVE_RUNTIME_SMOKE_DATA_ROOT_MISSING")
    if not (checkpoint_root / "Wan2.1_VAE.pth").is_file():
        raise PrimitiveRuntimeSmokeError("PRIMITIVE_RUNTIME_SMOKE_VAE_MISSING")
    if not (data_root / "rigid_dynamics" / "push_block" / "ind_train" / "metadata.pt").is_file() and not (
        data_root / "rigid_dynamics" / "push_block" / "ind_train" / "metadata_lite.pt"
    ).is_file():
        raise PrimitiveRuntimeSmokeError("PRIMITIVE_RUNTIME_SMOKE_PUSH_CUBE_TRAIN_SPLIT_MISSING")


def _runtime_environment(
    *,
    runtime_python: Path,
    worktree: Path,
    repo_root: Path,
    output_root: Path,
    data_root: Path,
    checkpoint_root: Path,
    gpu_index: int,
) -> dict[str, str]:
    dino_root = repo_root.parent / "models" / "dino_model"
    dino_repo = dino_root / "facebookresearch_dino_main"
    dino_weight = repo_root.parent / "models" / "RAEv2-models" / "encoders" / "dino" / "dino_vit_small_patch8_224.pth"
    dino_environment: dict[str, str] = {}
    if dino_repo.is_dir() and dino_weight.is_file():
        dino_environment = {
            "WMLOOP_DINO_REPO": str(dino_repo),
            "WMLOOP_DINO_WEIGHT": str(dino_weight),
        }
    return runtime_subprocess_env(
        runtime_python,
        extra={
            "ACWM_DATA_ROOT": str(data_root),
            "WAN_VAE_PATH": str(checkpoint_root / "Wan2.1_VAE.pth"),
            "CUDA_VISIBLE_DEVICES": str(gpu_index),
            "PYTHONPATH": os.pathsep.join((str(worktree), str(repo_root))),
            "WANDB_MODE": "disabled",
            "WANDB_SILENT": "true",
            "HF_HOME": str(output_root / "hf-home"),
            "TORCH_HOME": str(output_root / "torch-home"),
            **dino_environment,
        },
    )


def _training_config(
    checkpoint_root: Path,
    *,
    primitive: str,
    weight: float,
    learning_rate: float | None = None,
) -> dict[str, object]:
    model_config: dict[str, object] = {
        "in_channels": 16,
        "patch_size": 2,
        "action_compress_rate": 4,
        "max_frames": 2,
        "action_dropout_prob": 0.0,
        "temporal_causal": False,
        "vae_name": "WanVAE",
        "vae_config": [str(Path(checkpoint_root).resolve() / "Wan2.1_VAE.pth")],
        "scheduler": "FlowMatch",
        "training_timesteps": 8,
        "motion_weighting_gamma": 0.0,
        "focal_alpha": 0.0,
        "dim": 64,
        "num_layers": 1,
        "num_heads": 4,
        "use_flash_attn": False,
    }
    if primitive == "latent_motion_prior":
        model_config["wmloop_latent_motion_prior_weight"] = weight
    if primitive == "motion_region_reweight":
        model_config["motion_weighting_gamma"] = weight
    return {
        "model_name": "VideoDiT",
        "dynamics_class": "DiffusionForcing_WM",
        "model_config": model_config,
        "dataset": {
            "name": "push_cube",
            "seq_len": 5,
            "obs_shape": [3, 64, 64],
            "train_size": 1,
            "ind_test_size": 1,
            "ood_test_size": 1,
            "test_cuts": 1,
            "cache_size": 2,
        },
        "training": {
            "batch_size": 1,
            "val_batch_size": 1,
            "learning_rate": float(learning_rate) if learning_rate is not None else 0.0001,
            "num_epochs": 1,
            "total_steps": 1,
            "num_workers": 0,
            "grad_clip": 1.0,
            "log_freq": 1,
            "val_freq": 1000,
            "checkpoint_freq": 9999,
            "gen_mode": "parallel",
            "inference_steps": 1,
        },
        "wandb": {
            "project": "wmloop-runtime-smoke",
            "run_name": f"{primitive}_runtime_smoke",
        },
        "distributed": {"use_fsdp": False},
    }


def _primitive_params(
    *,
    primitive: str,
    condition: str,
    n_episodes: int,
    weight: float,
    action_balance_blend: float,
    action_balance_max_gain: float,
    history_noise: float,
    keep_tokens: int,
    frontier_weight: float,
    event_weight: float,
    event_quantile: float,
    event_visual_blend: float,
    guidance_start: float,
    guidance_end: float,
    anchor_every: int,
    anchor_weight: float,
    memory_slots: int,
    memory_weight: float,
    reward_weight: float,
    inv_dyn_steps: int,
    inv_dyn_lr: float,
    self_forcing_rollout_horizon: int,
    self_forcing_steps: int,
    self_forcing_lr: float,
    wmsd_teacher_ema: float,
    wmsd_steps: int,
    wmsd_lr: float,
    next_forcing_chunks: int,
    next_forcing_steps: int,
    next_forcing_lr: float,
) -> dict[str, object]:
    if primitive == "frontier_collection":
        if not isinstance(condition, str) or not condition.strip():
            raise PrimitiveRuntimeSmokeError("PRIMITIVE_RUNTIME_SMOKE_FRONTIER_CONDITION_INVALID")
        if n_episodes < 1:
            raise PrimitiveRuntimeSmokeError("PRIMITIVE_RUNTIME_SMOKE_FRONTIER_N_EPISODES_INVALID")
        return {"condition": condition.strip(), "n_episodes": int(n_episodes)}
    if primitive in {"latent_motion_prior", "motion_region_reweight"}:
        if not math.isfinite(float(weight)) or weight <= 0.0 or (primitive == "motion_region_reweight" and weight > 4.0):
            raise PrimitiveRuntimeSmokeError("PRIMITIVE_RUNTIME_SMOKE_WEIGHT_INVALID")
        return {"weight": weight}
    if primitive == "action_contrastive_finetune":
        if not math.isfinite(float(weight)) or weight <= 0.0 or weight > 1.0:
            raise PrimitiveRuntimeSmokeError("PRIMITIVE_RUNTIME_SMOKE_ACTION_CONTRASTIVE_WEIGHT_INVALID")
        return {"weight": float(weight)}
    if primitive == "action_dimension_balancing":
        if not math.isfinite(float(action_balance_blend)) or action_balance_blend <= 0.0 or action_balance_blend > 1.0:
            raise PrimitiveRuntimeSmokeError("PRIMITIVE_RUNTIME_SMOKE_ACTION_BALANCE_BLEND_INVALID")
        if not math.isfinite(float(action_balance_max_gain)) or action_balance_max_gain < 1.0 or action_balance_max_gain > 8.0:
            raise PrimitiveRuntimeSmokeError("PRIMITIVE_RUNTIME_SMOKE_ACTION_BALANCE_MAX_GAIN_INVALID")
        return {"blend": float(action_balance_blend), "max_gain": float(action_balance_max_gain)}
    if primitive == "dino_rep_injection":
        if not math.isfinite(float(weight)) or weight <= 0.0 or weight > 1.0:
            raise PrimitiveRuntimeSmokeError("PRIMITIVE_RUNTIME_SMOKE_DINO_WEIGHT_INVALID")
        return {"injection_weight": float(weight)}
    if primitive == "history_noise_schedule":
        return {"history_noise": history_noise}
    if primitive == "drift_token_trim":
        if keep_tokens < 1:
            raise PrimitiveRuntimeSmokeError("PRIMITIVE_RUNTIME_SMOKE_KEEP_TOKENS_INVALID")
        return {"keep_tokens": int(keep_tokens)}
    if primitive == "mixture_reweight":
        if not math.isfinite(float(frontier_weight)) or frontier_weight < 0.0:
            raise PrimitiveRuntimeSmokeError("PRIMITIVE_RUNTIME_SMOKE_FRONTIER_WEIGHT_INVALID")
        return {"frontier_weight": float(frontier_weight)}
    if primitive == "event_window_reweight":
        if not math.isfinite(float(event_weight)) or event_weight <= 0.0 or event_weight > 16.0:
            raise PrimitiveRuntimeSmokeError("PRIMITIVE_RUNTIME_SMOKE_EVENT_WEIGHT_INVALID")
        if not math.isfinite(float(event_quantile)) or event_quantile < 0.5 or event_quantile > 0.95:
            raise PrimitiveRuntimeSmokeError("PRIMITIVE_RUNTIME_SMOKE_EVENT_QUANTILE_INVALID")
        if not math.isfinite(float(event_visual_blend)) or event_visual_blend < 0.0 or event_visual_blend > 1.0:
            raise PrimitiveRuntimeSmokeError("PRIMITIVE_RUNTIME_SMOKE_EVENT_VISUAL_BLEND_INVALID")
        return {
            "event_weight": float(event_weight),
            "event_quantile": float(event_quantile),
            "visual_motion_blend": float(event_visual_blend),
        }
    if primitive == "cfg_guidance_schedule":
        if not math.isfinite(float(guidance_start)) or guidance_start < 0.0:
            raise PrimitiveRuntimeSmokeError("PRIMITIVE_RUNTIME_SMOKE_GUIDANCE_START_INVALID")
        if not math.isfinite(float(guidance_end)) or guidance_end < 0.0:
            raise PrimitiveRuntimeSmokeError("PRIMITIVE_RUNTIME_SMOKE_GUIDANCE_END_INVALID")
        return {"guidance_start": float(guidance_start), "guidance_end": float(guidance_end)}
    if primitive == "first_frame_anchor":
        if anchor_every < 4 or anchor_every > 32:
            raise PrimitiveRuntimeSmokeError("PRIMITIVE_RUNTIME_SMOKE_ANCHOR_EVERY_INVALID")
        if not math.isfinite(float(anchor_weight)) or anchor_weight < 0.0 or anchor_weight > 1.0:
            raise PrimitiveRuntimeSmokeError("PRIMITIVE_RUNTIME_SMOKE_ANCHOR_WEIGHT_INVALID")
        return {"anchor_every": int(anchor_every), "anchor_weight": float(anchor_weight)}
    if primitive == "latent_spatial_memory":
        if memory_slots < 1 or memory_slots > 128:
            raise PrimitiveRuntimeSmokeError("PRIMITIVE_RUNTIME_SMOKE_MEMORY_SLOTS_INVALID")
        if not math.isfinite(float(memory_weight)) or memory_weight <= 0.0 or memory_weight > 1.0:
            raise PrimitiveRuntimeSmokeError("PRIMITIVE_RUNTIME_SMOKE_MEMORY_WEIGHT_INVALID")
        return {"memory_slots": int(memory_slots), "memory_weight": float(memory_weight)}
    if primitive == "inv_dyn_reward_finetune":
        if not math.isfinite(float(reward_weight)) or reward_weight <= 0.0:
            raise PrimitiveRuntimeSmokeError("PRIMITIVE_RUNTIME_SMOKE_INV_DYN_REWARD_WEIGHT_INVALID")
        if inv_dyn_steps < 1:
            raise PrimitiveRuntimeSmokeError("PRIMITIVE_RUNTIME_SMOKE_INV_DYN_STEPS_INVALID")
        if not math.isfinite(float(inv_dyn_lr)) or inv_dyn_lr <= 0.0:
            raise PrimitiveRuntimeSmokeError("PRIMITIVE_RUNTIME_SMOKE_INV_DYN_LR_INVALID")
        return {"reward_weight": float(reward_weight), "steps": int(inv_dyn_steps), "lr": float(inv_dyn_lr)}
    if primitive == "self_forcing_finetune":
        if self_forcing_rollout_horizon < 2:
            raise PrimitiveRuntimeSmokeError("PRIMITIVE_RUNTIME_SMOKE_SELF_FORCING_ROLLOUT_HORIZON_INVALID")
        if self_forcing_steps < 1:
            raise PrimitiveRuntimeSmokeError("PRIMITIVE_RUNTIME_SMOKE_SELF_FORCING_STEPS_INVALID")
        if not math.isfinite(float(self_forcing_lr)) or self_forcing_lr <= 0.0:
            raise PrimitiveRuntimeSmokeError("PRIMITIVE_RUNTIME_SMOKE_SELF_FORCING_LR_INVALID")
        return {
            "rollout_horizon": int(self_forcing_rollout_horizon),
            "steps": int(self_forcing_steps),
            "lr": float(self_forcing_lr),
        }
    if primitive == "wmsd_self_distill":
        if not math.isfinite(float(wmsd_teacher_ema)) or wmsd_teacher_ema <= 0.0 or wmsd_teacher_ema > 1.0:
            raise PrimitiveRuntimeSmokeError("PRIMITIVE_RUNTIME_SMOKE_WMSD_TEACHER_EMA_INVALID")
        if wmsd_steps < 1:
            raise PrimitiveRuntimeSmokeError("PRIMITIVE_RUNTIME_SMOKE_WMSD_STEPS_INVALID")
        if not math.isfinite(float(wmsd_lr)) or wmsd_lr <= 0.0:
            raise PrimitiveRuntimeSmokeError("PRIMITIVE_RUNTIME_SMOKE_WMSD_LR_INVALID")
        return {"teacher_ema": float(wmsd_teacher_ema), "steps": int(wmsd_steps), "lr": float(wmsd_lr)}
    if primitive == "next_forcing":
        if next_forcing_chunks < 2 or next_forcing_chunks > 8:
            raise PrimitiveRuntimeSmokeError("PRIMITIVE_RUNTIME_SMOKE_NEXT_FORCING_CHUNKS_INVALID")
        if next_forcing_steps < 1:
            raise PrimitiveRuntimeSmokeError("PRIMITIVE_RUNTIME_SMOKE_NEXT_FORCING_STEPS_INVALID")
        if next_forcing_lr <= 0.0:
            raise PrimitiveRuntimeSmokeError("PRIMITIVE_RUNTIME_SMOKE_NEXT_FORCING_LR_INVALID")
        return {"chunks": int(next_forcing_chunks), "steps": int(next_forcing_steps), "lr": float(next_forcing_lr)}
    raise PrimitiveRuntimeSmokeError(f"PRIMITIVE_RUNTIME_SMOKE_PRIMITIVE_UNSUPPORTED:{primitive}")


def _write_report_bundle(
    *,
    report: Mapping[str, object],
    output_root: Path,
    archive_db: Path | None,
    cas_root: Path | None,
    extra_files: Mapping[str, Path],
    extra_directories: Mapping[str, Path],
) -> dict[str, object]:
    destination = Path(output_root).resolve()
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    report_bytes = _canonical_json_bytes(report)
    markdown_bytes = _render_markdown(report).encode("utf-8")
    cas_refs: dict[str, str] = {}
    try:
        temporary.mkdir(mode=0o700)
        _write_bytes_atomic(temporary / "primitive-runtime-smoke.json", report_bytes)
        _write_bytes_atomic(temporary / "primitive-runtime-smoke.md", markdown_bytes)
        for name, source in extra_files.items():
            target = temporary / f"{name}{source.suffix or '.txt'}"
            shutil.copy2(source, target)
        for name, source in extra_directories.items():
            if source.is_dir():
                shutil.copytree(source, temporary / name)
        archive = ArchiveStore(archive_db) if archive_db is not None else None
        if archive is not None or cas_root is not None:
            root = cas_root if cas_root is not None else Path(archive_db).resolve().parent  # type: ignore[union-attr]
            cas = ContentAddressedStore(Path(root))
            for name, payload, media_type in (
                ("primitive_runtime_smoke_json", report_bytes, "application/json"),
                ("primitive_runtime_smoke_markdown", markdown_bytes, "text/markdown"),
            ):
                ref = cas.put_bytes(payload, media_type=media_type).uri
                cas_refs[name] = ref
                if archive is not None:
                    archive.record_artifact_reference(ref)
            for name, source in extra_files.items():
                payload = source.read_bytes()
                media_type = "application/json" if source.suffix == ".json" else "text/plain"
                ref = cas.put_bytes(payload, media_type=media_type).uri
                cas_refs[name] = ref
                if archive is not None:
                    archive.record_artifact_reference(ref)
            for directory_name, source in extra_directories.items():
                if not source.is_dir():
                    continue
                for member in sorted(path for path in source.rglob("*") if path.is_file()):
                    relative = member.relative_to(source).as_posix()
                    key = f"{directory_name}_{relative}".replace("/", "_").replace(".", "_")
                    media_type = "application/json" if member.suffix == ".json" else "text/plain"
                    ref = cas.put_bytes(member.read_bytes(), media_type=media_type).uri
                    cas_refs[key] = ref
                    if archive is not None:
                        archive.record_artifact_reference(ref)
        manifest = {
            "schema_version": 1,
            "artifact_type": "wmloop-m2-medium-primitive-runtime-smoke-manifest",
            "state": report["state"],
            "primitive": report["primitive"],
            "run_training": report["run_training"],
            "worktree_removed": report["worktree_removed"],
            "gpu_exclusivity_audit": report.get("gpu_exclusivity_audit"),
            "report_path": str(destination / "primitive-runtime-smoke.json"),
            "markdown_path": str(destination / "primitive-runtime-smoke.md"),
            "cas_refs": cas_refs,
            "limitations": report["limitations"],
        }
        _write_bytes_atomic(temporary / "manifest.json", _canonical_json_bytes(manifest))
        os.replace(temporary, destination)
        return manifest
    except Exception:
        if temporary.is_symlink():
            temporary.unlink()
        elif temporary.exists():
            shutil.rmtree(temporary)
        raise


def _render_markdown(report: Mapping[str, Any]) -> str:
    candidate = report["candidate"]
    receipts = candidate["receipts"] if isinstance(candidate, Mapping) else ()
    lines = [
        "# M2 Medium Primitive Runtime Smoke",
        "",
        f"State: `{report['state']}`",
        f"Primitive: `{report['primitive']}`",
        f"Runtime training launched: `{report['run_training']}`",
        f"GPU index: `{report['gpu_index']}`",
        f"Worktree removed: `{report['worktree_removed']}`",
        "",
        "## Receipts",
        "",
        "| Label | Passed | Timed out | Exit code | Seconds |",
        "|:--|:--|:--|:--|--:|",
    ]
    for receipt in receipts:
        lines.append(
            f"| {receipt['label']} | {receipt['passed']} | {receipt['timed_out']} | "
            f"{receipt['exit_code']} | {receipt['duration_seconds']:.3f} |"
        )
    lines.extend(["", "## Limitations", ""])
    for item in report["limitations"]:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def _sha256_bytes(payload: bytes) -> str:
    import hashlib

    return hashlib.sha256(payload).hexdigest()


def hook_unit_script(primitive: str) -> str:
    if primitive == "dino_rep_injection":
        return _DINO_REP_INJECTION_HOOK_UNIT_SCRIPT
    if primitive == "frontier_collection":
        return _FRONTIER_COLLECTION_HOOK_UNIT_SCRIPT
    if primitive == "latent_motion_prior":
        return _HOOK_UNIT_SCRIPT
    if primitive == "motion_region_reweight":
        return _MOTION_REGION_REWEIGHT_HOOK_UNIT_SCRIPT
    if primitive == "action_contrastive_finetune":
        return _ACTION_CONTRASTIVE_HOOK_UNIT_SCRIPT
    if primitive == "action_dimension_balancing":
        return _ACTION_DIMENSION_BALANCING_HOOK_UNIT_SCRIPT
    if primitive == "history_noise_schedule":
        return _HISTORY_NOISE_HOOK_UNIT_SCRIPT
    if primitive == "drift_token_trim":
        return _DRIFT_TOKEN_TRIM_HOOK_UNIT_SCRIPT
    if primitive == "cfg_guidance_schedule":
        return _CFG_GUIDANCE_HOOK_UNIT_SCRIPT
    if primitive == "first_frame_anchor":
        return _FIRST_FRAME_ANCHOR_HOOK_UNIT_SCRIPT
    if primitive == "latent_spatial_memory":
        return _LATENT_SPATIAL_MEMORY_HOOK_UNIT_SCRIPT
    if primitive == "mixture_reweight":
        return _MIXTURE_REWEIGHT_HOOK_UNIT_SCRIPT
    if primitive == "event_window_reweight":
        return _EVENT_WINDOW_REWEIGHT_HOOK_UNIT_SCRIPT
    if primitive == "inv_dyn_reward_finetune":
        return _INV_DYN_REWARD_HOOK_UNIT_SCRIPT
    if primitive == "self_forcing_finetune":
        return _SELF_FORCING_HOOK_UNIT_SCRIPT
    if primitive == "wmsd_self_distill":
        return _WMSD_SELF_DISTILL_HOOK_UNIT_SCRIPT
    if primitive == "next_forcing":
        return _NEXT_FORCING_HOOK_UNIT_SCRIPT
    raise PrimitiveRuntimeSmokeError(f"PRIMITIVE_RUNTIME_SMOKE_HOOK_UNIT_UNSUPPORTED:{primitive}")


_ACTION_CONTRASTIVE_HOOK_UNIT_SCRIPT = """
import json
import torch
from acwm.wmloop_hooks.action_contrastive_finetune import apply_action_contrastive_loss

class DummyInner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.5))
    def forward(self, z, t, a):
        del t
        action_bias = a.float().mean(dim=(1, 2)).view(-1, 1, 1, 1, 1)
        return self.scale * z + 0.01 * action_bias

model = DummyInner()
z_t = torch.arange(2 * 3 * 2 * 2 * 4, dtype=torch.float32).reshape(2, 3, 2, 2, 4) / 100.0
t_values = torch.ones(2, 3)
actions = torch.stack([torch.zeros(5, 2), torch.ones(5, 2)], dim=0)
v_pred = model(z_t, t_values, actions)
v_target = torch.zeros_like(v_pred)
weights = torch.ones(2, 3)
base = (v_pred - v_target).square().mean()
loss, metrics = apply_action_contrastive_loss(
    base_loss=base,
    dynamics_model=model,
    z_t=z_t,
    t_values=t_values,
    actions=actions,
    v_pred=v_pred,
    v_target=v_target,
    weights=weights,
)
loss.backward()
assert loss.item() >= base.detach().item()
assert model.scale.grad is not None
assert model.scale.grad.detach().abs().sum().item() > 0.0
assert metrics["train/wmloop_action_contrastive_active"] == 1.0
assert metrics["train/wmloop_action_contrastive_loss"] > 0.0
print(json.dumps({"loss": float(loss.detach()), "metrics": metrics}, sort_keys=True))
""".strip()


_ACTION_DIMENSION_BALANCING_HOOK_UNIT_SCRIPT = """
import json
import torch
from acwm.wmloop_hooks.action_dimension_balancing import balance_action_dimensions, runtime_hook_receipt

actions = torch.tensor(
    [[[0.1, 4.0, -0.2], [0.2, 4.0, -0.1], [0.1, 4.0, -0.3], [0.2, 4.0, -0.2]]],
    dtype=torch.float16,
)
balanced = balance_action_dimensions(actions)
assert balanced.shape == actions.shape
assert balanced.dtype == actions.dtype
assert not torch.equal(balanced, actions)
null_action = torch.tensor([[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]], dtype=torch.float32)
null_balanced = balance_action_dimensions(null_action)
assert torch.equal(null_balanced, null_action)
receipt = runtime_hook_receipt()
assert receipt["state"] == "ready"
assert receipt["call_count"] == 2
assert receipt["observed_gain_min"] >= 0.0
assert receipt["observed_gain_max"] <= 8.0
print(json.dumps({"receipt": receipt, "balanced": balanced.float().tolist()}, sort_keys=True))
""".strip()


_DINO_REP_INJECTION_HOOK_UNIT_SCRIPT = """
import json
import torch
import acwm.wmloop_hooks.dino_rep_injection as hook

real_teacher_similarity = hook._teacher_similarity
hook._teacher_similarity = lambda observations: observations.new_tensor([0.25, 0.5])
base = torch.tensor(1.0, requires_grad=True)
predicted = torch.arange(2 * 3 * 2 * 2 * 4, dtype=torch.float32).reshape(2, 3, 2, 2, 4)
predicted.requires_grad_(True)
observations = torch.rand(2, 5, 3, 32, 32)
loss, metrics = hook.apply_dino_representation_loss(base, predicted, observations, force=True)
loss.backward()
assert loss.item() > base.detach().item()
assert predicted.grad is not None
assert metrics["train/wmloop_dino_rep_injection_active"] == 1.0
assert metrics["train/wmloop_dino_rep_injection_loss"] > 0.0
hook._teacher_similarity = real_teacher_similarity
with torch.no_grad():
    real_similarity = real_teacher_similarity(torch.rand(1, 2, 3, 32, 32))
assert real_similarity.shape == (1,)
assert torch.isfinite(real_similarity).all()
print(json.dumps({"loss": float(loss.detach()), "metrics": metrics, "real_teacher_similarity": real_similarity.tolist()}, sort_keys=True))
""".strip()


_FRONTIER_COLLECTION_HOOK_UNIT_SCRIPT = """
import json
from pathlib import Path

import torch

from acwm.wmloop_hooks.frontier_collection import record_frontier_observation

records = Path("wmloop_interventions/frontier_collection_records.jsonl")
if records.exists():
    records.unlink()
loss = torch.tensor(2.0, requires_grad=True)
actions = torch.ones(2, 5, 3)
metrics = record_frontier_observation(loss, actions, step=7, epoch=1, traj_ids=torch.tensor([3, 4]), starts=torch.tensor([0, 8]))
assert records.is_file()
lines = records.read_text(encoding="utf-8").strip().splitlines()
assert len(lines) == 1
record = json.loads(lines[0])
assert record["primitive"] == "frontier_collection"
assert record["role"] == "diagnostic_routing_evidence"
assert record["condition"]
assert record["step"] == 7
assert record["traj_ids"] == [3, 4]
assert "train/wmloop_frontier_collection_score" in metrics
print(json.dumps({"record": record, "metrics": metrics}, sort_keys=True))
""".strip()


_HOOK_UNIT_SCRIPT = """
import json
import torch
from acwm.wmloop_hooks.latent_motion_prior import apply_latent_motion_prior

base = torch.tensor(1.0, requires_grad=True)
latents = torch.arange(2 * 3 * 4 * 4 * 2, dtype=torch.float32).reshape(2, 3, 4, 4, 2)
latents.requires_grad_(True)
actions = torch.ones(2, 5, 2)
loss, metrics = apply_latent_motion_prior(base, latents, actions)
loss.backward()
assert loss.item() > 1.0
assert latents.grad is not None
assert "train/wmloop_latent_motion_prior_loss" in metrics
print(json.dumps({"loss": float(loss.detach()), "metrics": metrics}, sort_keys=True))
""".strip()


_MOTION_REGION_REWEIGHT_HOOK_UNIT_SCRIPT = """
import json
from acwm.wmloop_hooks.motion_region_reweight import apply_motion_region_reweight_config

source = {"model_config": {"motion_weighting_gamma": 0.0, "dim": 64}, "training": {"total_steps": 1}}
configured = apply_motion_region_reweight_config(source)
assert configured is not source
assert configured["model_config"]["motion_weighting_gamma"] > 0.0
assert configured["model_config"]["dim"] == 64
assert source["model_config"]["motion_weighting_gamma"] == 0.0
print(json.dumps({"gamma": configured["model_config"]["motion_weighting_gamma"]}, sort_keys=True))
""".strip()


_HISTORY_NOISE_HOOK_UNIT_SCRIPT = """
import json
import torch
from acwm.wmloop_hooks.history_noise_schedule import apply_history_noise_schedule

torch.manual_seed(7)
latents = torch.arange(2 * 4 * 3 * 3 * 2, dtype=torch.float32).reshape(2, 4, 3, 3, 2)
original = latents.clone()
out = apply_history_noise_schedule(latents)
assert out.shape == latents.shape
assert torch.allclose(out[:, -1], latents[:, -1])
assert not torch.allclose(out[:, :-1], latents[:, :-1])
assert torch.allclose(latents, original)
print(json.dumps({"changed": bool(not torch.allclose(out, latents)), "shape": list(out.shape)}, sort_keys=True))
""".strip()


_DRIFT_TOKEN_TRIM_HOOK_UNIT_SCRIPT = """
import json
import torch
from acwm.wmloop_hooks.drift_token_trim import apply_drift_token_trim

latents = torch.arange(2 * 5 * 3 * 3 * 2, dtype=torch.float32).reshape(2, 5, 3, 3, 2)
anchor = latents[:, :1].clone()
original = latents.clone()
out = apply_drift_token_trim(latents, anchor=anchor)
assert out.shape == latents.shape
assert torch.allclose(out[:, 0], latents[:, 0])
assert torch.allclose(out[:, -2:], latents[:, -2:])
assert not torch.allclose(out[:, 1:3], latents[:, 1:3])
assert torch.allclose(latents, original)
print(json.dumps({"changed": bool(not torch.allclose(out, latents)), "shape": list(out.shape)}, sort_keys=True))
""".strip()


_CFG_GUIDANCE_HOOK_UNIT_SCRIPT = """
import json
import torch
from acwm.wmloop_hooks.cfg_guidance_schedule import apply_cfg_guidance_schedule

class DummyInner:
    def get_null_cond(self, actions):
        out = torch.zeros_like(actions)
        out[..., -1] = 1
        return out

class DummyModel:
    def __init__(self):
        self.model = DummyInner()
        self.calls = []
    def __call__(self, z, t, a):
        self.calls.append(float(a.mean().detach()))
        return z + a.mean().to(dtype=z.dtype)

model = DummyModel()
latents = torch.ones(2, 4, 3, 3, 2)
timesteps = torch.zeros(2, 4)
actions = torch.ones(2, 13, 3)
out = apply_cfg_guidance_schedule(model, latents, timesteps, actions, step_index=1, total_steps=3)
with open("wmloop_interventions/cfg_guidance_schedule.json", "r", encoding="utf-8") as handle:
    params = json.load(handle)["params"]
scale = float(params["guidance_start"]) + 0.5 * (
    float(params["guidance_end"]) - float(params["guidance_start"])
)
cond = latents + 1.0
uncond = latents + (1.0 / 3.0)
expected = cond if scale == 1.0 else uncond + scale * (cond - uncond)
assert out.shape == latents.shape
assert len(model.calls) == (1 if scale == 1.0 else 2)
assert model.calls[0] == 1.0
if scale != 1.0:
    assert abs(model.calls[1] - (1.0 / 3.0)) < 1e-6
assert torch.allclose(out, expected)
assert torch.isfinite(out).all()
print(json.dumps({"shape": list(out.shape), "calls": model.calls, "scale": scale}, sort_keys=True))
""".strip()


_FIRST_FRAME_ANCHOR_HOOK_UNIT_SCRIPT = """
import json
import torch
from acwm.wmloop_hooks.first_frame_anchor import apply_first_frame_anchor

latents = torch.arange(2 * 5 * 3 * 3 * 2, dtype=torch.float32).reshape(2, 5, 3, 3, 2)
anchor = latents[:, :1].clone()
original = latents.clone()
out = apply_first_frame_anchor(latents, anchor, step_index=0, total_steps=8)
assert out.shape == latents.shape
assert torch.allclose(out[:, 0], latents[:, 0])
assert not torch.allclose(out[:, 1:], latents[:, 1:])
assert torch.allclose(latents, original)
between_cadence = apply_first_frame_anchor(latents, anchor, step_index=1, total_steps=8)
assert torch.allclose(between_cadence, latents)
print(json.dumps({"changed_on_cadence": bool(not torch.allclose(out, latents)), "unchanged_between_cadence": bool(torch.allclose(between_cadence, latents)), "shape": list(out.shape)}, sort_keys=True))
""".strip()


_LATENT_SPATIAL_MEMORY_HOOK_UNIT_SCRIPT = """
import json
import torch
from acwm.wmloop_hooks.latent_spatial_memory import apply_latent_spatial_memory

latents = torch.arange(2 * 5 * 3 * 3 * 2, dtype=torch.float32).reshape(2, 5, 3, 3, 2)
original = latents.clone()
out = apply_latent_spatial_memory(latents)
assert out.shape == latents.shape
assert torch.allclose(out[:, 0], latents[:, 0])
assert not torch.allclose(out[:, 1:], latents[:, 1:])
assert torch.allclose(latents, original)
print(json.dumps({"changed": bool(not torch.allclose(out, latents)), "shape": list(out.shape)}, sort_keys=True))
""".strip()


_MIXTURE_REWEIGHT_HOOK_UNIT_SCRIPT = """
import json
import torch
from acwm.wmloop_hooks.mixture_reweight import build_mixture_reweight_sampler, _window_weights

class DummyConfig:
    seq_len = 3
    sampling_rate = 1
    action_dim = 2

class DummyDataset:
    config = DummyConfig()
    indices = [(0, 0), (1, 0), (2, 0)]
    def __init__(self):
        self._metadata = [
            {"actions": torch.zeros(3, 2)},
            {"actions": torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])},
            {"actions": torch.tensor([[0.0, 0.0], [4.0, 0.0], [8.0, 0.0]])},
        ]
    @property
    def full_metadata(self):
        return self._metadata
    def __len__(self):
        return len(self.indices)

dataset = DummyDataset()
weights = _window_weights(dataset, frontier_weight=2.0)
assert weights.shape[0] == 3
assert float(weights[-1]) > float(weights[1]) > float(weights[0])
sampler = build_mixture_reweight_sampler(dataset)
assert sampler is not None
draws = list(iter(sampler))
assert len(draws) == len(dataset)
print(json.dumps({"weights": [float(item) for item in weights], "draw_count": len(draws)}, sort_keys=True))
""".strip()


_EVENT_WINDOW_REWEIGHT_HOOK_UNIT_SCRIPT = """
import json
import torch
from acwm.wmloop_hooks.event_window_reweight import (
    _window_weights,
    build_event_window_sampler,
    runtime_hook_receipt,
)

class DummyConfig:
    seq_len = 4
    sampling_rate = 1
    action_dim = 2

class DummyDataset:
    config = DummyConfig()
    indices = [(0, start) for start in range(7)]
    effective_root = None
    def __init__(self):
        action = torch.zeros(10, 2)
        action[4:7, 0] = torch.tensor([0.0, 5.0, 0.0])
        self._metadata = [{"actions": action}]
    @property
    def full_metadata(self):
        return self._metadata
    def __len__(self):
        return len(self.indices)

dataset = DummyDataset()
weights, source = _window_weights(
    dataset,
    event_weight=4.0,
    event_quantile=0.7,
    visual_motion_blend=0.7,
)
assert weights.shape[0] == len(dataset)
assert source == "action_transition_only"
assert float(weights.max()) > 1.0
assert int((weights > 1.0).sum()) < len(dataset)
sampler = build_event_window_sampler(dataset)
assert len(list(iter(sampler))) == len(dataset)
receipt = runtime_hook_receipt()
assert receipt["state"] == "ready"
assert receipt["call_count"] == 1
assert receipt["selected_window_count"] >= 1
assert receipt["score_source"] == "action_transition_only"
print(json.dumps({"weights": weights.tolist(), "receipt": receipt}, sort_keys=True))
""".strip()


_INV_DYN_REWARD_HOOK_UNIT_SCRIPT = """
import json
import torch
from acwm.wmloop_hooks.inv_dyn_reward_finetune import apply_inv_dyn_reward_loss

base = torch.tensor(1.0, requires_grad=True)
latents = torch.arange(2 * 5 * 3 * 2 * 2, dtype=torch.float32).reshape(2, 5, 3, 2, 2)
latents.requires_grad_(True)
actions = torch.stack([
    torch.linspace(0.0, 1.0, 4),
    torch.linspace(1.0, 2.0, 4),
], dim=0).unsqueeze(-1).repeat(1, 1, 2)
loss, metrics = apply_inv_dyn_reward_loss(base, latents, actions)
loss.backward()
assert loss.item() >= 1.0
assert latents.grad is not None
assert "train/wmloop_inv_dyn_reward_loss" in metrics
assert "train/wmloop_inv_dyn_r2_proxy" in metrics
print(json.dumps({"loss": float(loss.detach()), "metrics": metrics}, sort_keys=True))
""".strip()


_SELF_FORCING_HOOK_UNIT_SCRIPT = """
import json
import torch
from acwm.wmloop_hooks.self_forcing_finetune import apply_self_forcing_loss

class DummyScheduler:
    def __init__(self):
        self.timesteps = torch.linspace(1000.0, 0.0, 5)
        self.sigmas = torch.linspace(1.0, 0.0, 5)
    def _indices(self, timestep):
        flat = timestep.reshape(-1, 1)
        return torch.argmin((flat - self.timesteps.reshape(1, -1).to(timestep.device)).abs(), dim=-1).reshape(timestep.shape)
    def add_independent_noise(self, samples, timestep):
        noise = torch.ones_like(samples) * 0.25
        idx = self._indices(timestep)
        sigma = self.sigmas.to(samples.device)[idx].reshape(*idx.shape, *([1] * (samples.ndim - idx.ndim)))
        return (1.0 - sigma) * samples + sigma * noise, noise
    def step(self, model_output, timestep, sample, to_final=False):
        idx = self._indices(timestep)
        sigma = self.sigmas.to(sample.device)[idx].reshape(*idx.shape, *([1] * (sample.ndim - idx.ndim)))
        sigma_next = torch.zeros_like(sigma) if to_final else sigma * 0.5
        return sample + model_output * (sigma_next - sigma)

class DummyInner(torch.nn.Module):
    action_compress_rate = 2
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.5))
    def forward(self, z, t, a):
        del t, a
        return self.scale * z

class DummyModel:
    model_config = {"action_compress_rate": 2}
    def __init__(self):
        self.model = DummyInner()
        self.scheduler = DummyScheduler()

model = DummyModel()
base = torch.tensor(1.0, requires_grad=True)
latents = torch.arange(2 * 5 * 2 * 2 * 3, dtype=torch.float32).reshape(2, 5, 2, 2, 3) / 100.0
actions = torch.ones(2, 9, 4)
loss, metrics = apply_self_forcing_loss(base, model, latents, actions)
loss.backward()
assert loss.item() > 1.0
assert model.model.scale.grad is not None
assert "train/wmloop_self_forcing_aux_loss" in metrics
assert metrics["train/wmloop_self_forcing_latent_frames"] >= 2
print(json.dumps({"loss": float(loss.detach()), "metrics": metrics}, sort_keys=True))
""".strip()


_WMSD_SELF_DISTILL_HOOK_UNIT_SCRIPT = """
import json
import torch
from acwm.wmloop_hooks.wmsd_self_distill import apply_wmsd_self_distill_loss

class DummyScheduler:
    def __init__(self):
        self.timesteps = torch.linspace(1000.0, 0.0, 5)
    def add_independent_noise(self, samples, timestep):
        del timestep
        noise = torch.ones_like(samples) * 0.25
        return samples + 0.1 * noise, noise

class DummyInner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.5))
    def forward(self, z, t, a):
        del t, a
        return self.scale * z

class DummyModel:
    def __init__(self):
        self.model = DummyInner()
        self.scheduler = DummyScheduler()

model = DummyModel()
base1 = torch.tensor(1.0, requires_grad=True)
latents = torch.arange(2 * 4 * 2 * 2 * 3, dtype=torch.float32).reshape(2, 4, 2, 2, 3) / 100.0
actions = torch.ones(2, 7, 4)
loss1, metrics1 = apply_wmsd_self_distill_loss(base1, model, latents, actions)
base2 = torch.tensor(1.0, requires_grad=True)
loss2, metrics2 = apply_wmsd_self_distill_loss(base2, model, latents + 0.1, actions)
loss2.backward()
assert loss1.item() >= 1.0
assert loss2.item() >= 1.0
assert model.model.scale.grad is not None
assert metrics1["train/wmloop_wmsd_teacher_warm"] == 1.0
assert metrics2["train/wmloop_wmsd_teacher_warm"] == 0.0
assert "train/wmloop_wmsd_self_distill_loss" in metrics2
print(json.dumps({"loss": float(loss2.detach()), "metrics": metrics2}, sort_keys=True))
""".strip()


_NEXT_FORCING_HOOK_UNIT_SCRIPT = """
import json
import torch
from acwm.wmloop_hooks.next_forcing import apply_next_forcing_loss

class DummyModel:
    model_config = {"action_compress_rate": 2}
    def __init__(self):
        self.calls = []
    def training_loss(self, z, a):
        self.calls.append((tuple(z.shape), tuple(a.shape)))
        return z.float().square().mean() + 0.01 * a.float().square().mean()

model = DummyModel()
latents = torch.arange(2 * 5 * 2 * 2 * 3, dtype=torch.float32).reshape(2, 5, 2, 2, 3)
latents.requires_grad_(True)
actions = torch.ones(2, 9, 4)
loss, metrics = apply_next_forcing_loss(model, latents, actions)
loss.backward()
assert loss.item() > 0
assert latents.grad is not None
assert len(model.calls) >= 2
assert model.calls[0][0][1] == 5
assert model.calls[1][0][1] >= 2
assert "train/wmloop_next_forcing_aux_loss" in metrics
print(json.dumps({"loss": float(loss.detach()), "calls": model.calls, "metrics": metrics}, sort_keys=True))
""".strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run the medium primitive runtime smoke")
    run.add_argument("--repo-root", type=Path, default=Path("."))
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--runtime-python", type=Path, required=True)
    run.add_argument("--data-root", type=Path, required=True)
    run.add_argument("--checkpoint-root", type=Path, required=True)
    run.add_argument("--primitive", default="latent_motion_prior")
    run.add_argument("--gpu-index", type=int, default=1)
    run.add_argument("--condition", default="smoke_frontier")
    run.add_argument("--n-episodes", type=int, default=2)
    run.add_argument("--weight", type=float, default=0.2)
    run.add_argument("--action-balance-blend", type=float, default=0.5)
    run.add_argument("--action-balance-max-gain", type=float, default=4.0)
    run.add_argument("--history-noise", type=float, default=0.1)
    run.add_argument("--keep-tokens", type=int, default=2)
    run.add_argument("--frontier-weight", type=float, default=1.0)
    run.add_argument("--event-weight", type=float, default=4.0)
    run.add_argument("--event-quantile", type=float, default=0.75)
    run.add_argument("--event-visual-blend", type=float, default=0.7)
    run.add_argument("--guidance-start", type=float, default=1.0)
    run.add_argument("--guidance-end", type=float, default=1.5)
    run.add_argument("--anchor-every", type=int, default=4)
    run.add_argument("--anchor-weight", type=float, default=0.25)
    run.add_argument("--memory-slots", type=int, default=2)
    run.add_argument("--memory-weight", type=float, default=0.25)
    run.add_argument("--reward-weight", type=float, default=0.5)
    run.add_argument("--inv-dyn-steps", type=int, default=1)
    run.add_argument("--inv-dyn-lr", type=float, default=0.00001)
    run.add_argument("--self-forcing-rollout-horizon", type=int, default=16)
    run.add_argument("--self-forcing-steps", type=int, default=1)
    run.add_argument("--self-forcing-lr", type=float, default=0.00001)
    run.add_argument("--wmsd-teacher-ema", type=float, default=0.9)
    run.add_argument("--wmsd-steps", type=int, default=1)
    run.add_argument("--wmsd-lr", type=float, default=0.00001)
    run.add_argument("--next-forcing-chunks", type=int, default=2)
    run.add_argument("--next-forcing-steps", type=int, default=1)
    run.add_argument("--next-forcing-lr", type=float, default=0.00001)
    run.add_argument("--run-training", action="store_true")
    run.add_argument("--hook-timeout-seconds", type=float, default=60.0)
    run.add_argument("--training-timeout-seconds", type=float, default=1200.0)
    run.add_argument("--keep-worktree", action="store_true")
    run.add_argument("--archive-db", type=Path)
    run.add_argument("--cas-root", type=Path)
    run.add_argument("--gpu-exclusivity-audit-manifest", type=Path)
    run.add_argument("--gpu-exclusivity-max-age-seconds", type=float, default=300.0)
    args = parser.parse_args(argv)
    if args.command == "run":
        manifest = run_primitive_runtime_smoke(
            repo_root=args.repo_root,
            output_root=args.output_root,
            runtime_python=args.runtime_python,
            data_root=args.data_root,
            checkpoint_root=args.checkpoint_root,
            primitive=args.primitive,
            gpu_index=args.gpu_index,
            condition=args.condition,
            n_episodes=args.n_episodes,
            weight=args.weight,
            action_balance_blend=args.action_balance_blend,
            action_balance_max_gain=args.action_balance_max_gain,
            history_noise=args.history_noise,
            keep_tokens=args.keep_tokens,
            frontier_weight=args.frontier_weight,
            event_weight=args.event_weight,
            event_quantile=args.event_quantile,
            event_visual_blend=args.event_visual_blend,
            guidance_start=args.guidance_start,
            guidance_end=args.guidance_end,
            anchor_every=args.anchor_every,
            anchor_weight=args.anchor_weight,
            memory_slots=args.memory_slots,
            memory_weight=args.memory_weight,
            reward_weight=args.reward_weight,
            inv_dyn_steps=args.inv_dyn_steps,
            inv_dyn_lr=args.inv_dyn_lr,
            self_forcing_rollout_horizon=args.self_forcing_rollout_horizon,
            self_forcing_steps=args.self_forcing_steps,
            self_forcing_lr=args.self_forcing_lr,
            wmsd_teacher_ema=args.wmsd_teacher_ema,
            wmsd_steps=args.wmsd_steps,
            wmsd_lr=args.wmsd_lr,
            next_forcing_chunks=args.next_forcing_chunks,
            next_forcing_steps=args.next_forcing_steps,
            next_forcing_lr=args.next_forcing_lr,
            run_training=args.run_training,
            hook_timeout_seconds=args.hook_timeout_seconds,
            training_timeout_seconds=args.training_timeout_seconds,
            keep_worktree=args.keep_worktree,
            archive_db=args.archive_db,
            cas_root=args.cas_root,
            gpu_exclusivity_audit_manifest=args.gpu_exclusivity_audit_manifest,
            gpu_exclusivity_max_age_seconds=args.gpu_exclusivity_max_age_seconds,
        )
        print(json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
