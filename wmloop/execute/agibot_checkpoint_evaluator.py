"""Independently reload a native Agibot checkpoint and evaluate a frozen panel.

This entrypoint runs inside the model runtime selected for a model run. It
loads the saved Wan2.2 WMTrainer checkpoint in a fresh process, verifies that
the loaded model tensors match the safetensors file, then evaluates exactly
the validation sample IDs bound to the run's training trace configuration.
The output is an intermediate trace; protocol-v3 derives the formal receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import sys
from collections.abc import Mapping, Sequence
from datetime import timedelta
from pathlib import Path
from typing import Any

from wmloop.execute.agibot_trace_runtime import (
    AgibotTraceError,
    _FrozenValidationView,
    _TRACE_KEY,
    load_trace_config,
)
from wmloop.storage import atomic_write, canonical_bytes, checked_path, sha256_file

logger = logging.getLogger(__name__)
_SAMPLE_TRACE_KEY = "_verdi_trace_identity"
_RAW_ARTIFACT_TYPE = "verdiwm-agibot-checkpoint-evaluation-trace"


class CheckpointEvaluationError(RuntimeError):
    """The saved checkpoint could not be independently evaluated."""


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-config", required=True, type=Path)
    parser.add_argument("--native-source-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--raw-output", required=True, type=Path)
    try:
        return parser.parse_args(argv)
    except SystemExit as exc:
        if exc.code == 0:
            raise
        raise CheckpointEvaluationError("AGIBOT_EVALUATOR_ARGUMENTS_INVALID") from exc


def _finite_float(value: object) -> float:
    try:
        if callable(getattr(value, "detach", None)):
            value = value.detach()
        if callable(getattr(value, "item", None)):
            value = value.item()
        result = float(value)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise CheckpointEvaluationError("AGIBOT_EVALUATOR_LOSS_INVALID") from exc
    if not math.isfinite(result):
        raise CheckpointEvaluationError("AGIBOT_EVALUATOR_LOSS_INVALID")
    return result


def _validate_checkpoint_inputs(
    *,
    checkpoint: Path,
    checkpoint_sha256: str,
    trace_config: Mapping[str, object],
    native_source_root: Path,
) -> tuple[Path, Path, dict[str, object]]:
    output_root = Path(str(trace_config["output_root"])).resolve()
    checkpoint_path = checked_path(
        checkpoint,
        code="AGIBOT_EVALUATOR_CHECKPOINT_INVALID",
        error=CheckpointEvaluationError,
    )
    try:
        checkpoint_path.relative_to(output_root / "checkpoints")
    except ValueError as exc:
        raise CheckpointEvaluationError("AGIBOT_EVALUATOR_CHECKPOINT_OUTSIDE_RUN") from exc
    if (
        checkpoint_path.name != "dit_model.safetensors"
        or not checkpoint_path.is_file()
        or not _is_sha256(checkpoint_sha256)
        or sha256_file(checkpoint_path) != checkpoint_sha256
    ):
        raise CheckpointEvaluationError("AGIBOT_EVALUATOR_CHECKPOINT_DIGEST_MISMATCH")
    metadata_path = checked_path(
        checkpoint_path.parent / "meta.json",
        code="AGIBOT_EVALUATOR_CHECKPOINT_METADATA_INVALID",
        error=CheckpointEvaluationError,
    )
    native_config_path = checked_path(
        output_root / "config" / "agibot-training.yaml",
        code="AGIBOT_EVALUATOR_NATIVE_CONFIG_INVALID",
        error=CheckpointEvaluationError,
    )
    if sha256_file(native_config_path) != trace_config.get("native_config_sha256"):
        raise CheckpointEvaluationError("AGIBOT_EVALUATOR_NATIVE_CONFIG_DIGEST_MISMATCH")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointEvaluationError("AGIBOT_EVALUATOR_CHECKPOINT_METADATA_INVALID") from exc
    if not isinstance(metadata, dict):
        raise CheckpointEvaluationError("AGIBOT_EVALUATOR_CHECKPOINT_METADATA_INVALID")
    if (
        metadata.get("checkpoint_contract_version") != 2
        or metadata.get("step") != trace_config.get("expected_optimizer_updates", 0)
        * trace_config.get("gradient_accumulation", 0)
        or metadata.get("world_size") != trace_config.get("world_size")
        or metadata.get("format") != "safetensors+optim_shards"
        or metadata.get("safetensors_exported") is not True
    ):
        raise CheckpointEvaluationError("AGIBOT_EVALUATOR_CHECKPOINT_METADATA_MISMATCH")
    source_root = checked_path(
        native_source_root,
        code="AGIBOT_EVALUATOR_NATIVE_SOURCE_INVALID",
        error=CheckpointEvaluationError,
    )
    if not source_root.is_dir():
        raise CheckpointEvaluationError("AGIBOT_EVALUATOR_NATIVE_SOURCE_INVALID")
    return checkpoint_path, metadata_path, metadata


def _verify_loaded_checkpoint_tensors(
    checkpoint_path: Path,
    state_dict: Mapping[str, object],
    *,
    torch: Any,
) -> int:
    """Require an exact key, shape, and value match after native loading."""

    try:
        from safetensors import safe_open

        with safe_open(str(checkpoint_path), framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            if not keys or keys != set(state_dict):
                raise CheckpointEvaluationError("AGIBOT_EVALUATOR_CHECKPOINT_KEYS_MISMATCH")
            for key in sorted(keys):
                saved = handle.get_tensor(key)
                loaded = state_dict[key]
                if not torch.is_tensor(loaded):
                    raise CheckpointEvaluationError(
                        "AGIBOT_EVALUATOR_CHECKPOINT_TENSOR_INVALID"
                    )
                loaded_cpu = loaded.detach().to(device="cpu")
                if tuple(saved.shape) != tuple(loaded_cpu.shape) or not torch.equal(
                    saved, loaded_cpu
                ):
                    raise CheckpointEvaluationError(
                        "AGIBOT_EVALUATOR_CHECKPOINT_TENSOR_MISMATCH"
                    )
            return len(keys)
    except CheckpointEvaluationError:
        raise
    except Exception as exc:
        raise CheckpointEvaluationError("AGIBOT_EVALUATOR_CHECKPOINT_RELOAD_FAILED") from exc


class _TracedFrozenPanel:
    """Add stable sample identities to the shared frozen-panel view."""

    def __init__(self, dataset: object, sample_ids: Sequence[str]):
        self.view = _FrozenValidationView(dataset, sample_ids)
        self._identity_by_ref: dict[tuple[int, int], dict[str, object]] = {}
        dataset_keys = getattr(dataset, "dataset_keys", None)
        entries_all = getattr(dataset, "entries_all", None)
        if not isinstance(dataset_keys, list) or not isinstance(entries_all, list):
            raise CheckpointEvaluationError("AGIBOT_EVALUATOR_VALIDATION_DATASET_INVALID")
        for sample_id, ref in zip(sample_ids, self.view.refs, strict=True):
            dataset_index, sample_index = ref
            episode_uid, frame = sample_id.rsplit(":", 1)
            episode_entry_index = int(
                dataset.samples_all[dataset_index][sample_index][0]
            )
            self._identity_by_ref[ref] = {
                "split_id": "val",
                "dataset_index": int(dataset_index),
                "dataset_key": str(dataset_keys[dataset_index]),
                "episode_entry_index": episode_entry_index,
                "source_sample_index": int(sample_index),
                "episode_uid": episode_uid,
                "first_future_frame": int(frame),
                "sample_window_id": sample_id,
            }

    def root_balanced_validation_refs(self, *args: object, **kwargs: object):
        return self.view.root_balanced_validation_refs(*args, **kwargs)

    def get_source_sample(self, dataset_index: int, sample_index: int) -> dict[str, Any]:
        sample = self.view.get_source_sample(dataset_index, sample_index)
        identity = self._identity_by_ref.get((dataset_index, sample_index))
        if identity is None:
            raise CheckpointEvaluationError("AGIBOT_EVALUATOR_VALIDATION_SAMPLE_UNBOUND")
        sample[_SAMPLE_TRACE_KEY] = dict(identity)
        return sample

    def describe_source_sample(self, dataset_index: int, sample_index: int) -> dict[str, Any]:
        return self.view.describe_source_sample(dataset_index, sample_index)

    def __getattr__(self, name: str) -> object:
        return getattr(self.view, name)


def _build_validation_dataset(config: Any, *, dataset_class: Any) -> object:
    training = config.wm_training
    world_model = config.world_model
    dataset_probs = list(getattr(training, "val_dataset_probs", [])) or list(
        getattr(training, "dataset_probs", [])
    )
    dataset = dataset_class(
        dataset_roots=training.dataset_roots,
        mode="val",
        num_history=world_model.history_length,
        num_frames=world_model.pred_frames,
        num_cameras=training.num_cameras,
        dataset_probs=dataset_probs,
        condition_view=str(getattr(training, "video_latent_condition_view")),
        sample_stride_video_frames=int(getattr(training, "video_latent_sample_stride")),
        action_dim=int(world_model.action_dim),
        latent_height_per_view=int(getattr(training, "latent_height_per_view")),
        latent_width=int(getattr(training, "latent_width")),
        expected_video_keys=list(getattr(training, "video_latent_video_keys", [])),
        require_production_contract=bool(
            getattr(training, "video_latent_require_production_contract", False)
        ),
        required_target_fps=float(
            getattr(training, "video_latent_required_target_fps", 0.0)
        ),
        required_target_image_hw=list(
            getattr(training, "video_latent_required_target_image_hw", [])
        ),
        history_selector=str(getattr(training, "history_selector", "recent")),
        history_offsets=list(getattr(training, "history_offsets", [])),
        history_dilations=list(getattr(training, "history_dilations", [1, 2])),
        history_collapse_prob=0.0,
        history_eval_dilation=int(getattr(training, "history_eval_dilation", 2)),
        generated_history_enabled=False,
        camera_conditioning=bool(
            getattr(training, "video_latent_camera_conditioning", False)
        ),
        camera_sidecar_roots=list(
            getattr(training, "video_latent_camera_sidecar_roots", [])
        ),
        camera_intrinsics_fallback=str(
            getattr(training, "video_latent_camera_intrinsics_fallback", "identity")
        ),
        camera_extrinsics_convention=str(
            getattr(training, "video_latent_camera_extrinsics_convention", "world_from_camera")
        ),
        eef_projection_roots=list(getattr(training, "video_latent_eef_projection_roots", [])),
        eef_projection_mode=str(getattr(training, "video_latent_eef_projection_mode", "sidecar")),
        eef_projection_load_heatmap=bool(
            getattr(training, "video_latent_eef_projection_load_heatmap", False)
        ),
        eef_projection_heatmap_size=str(
            getattr(training, "video_latent_eef_projection_heatmap_size", "")
        ),
    )
    if getattr(dataset, "mode", None) != "val" or len(dataset) < 1:
        raise CheckpointEvaluationError("AGIBOT_EVALUATOR_VALIDATION_DATASET_EMPTY")
    return dataset


def _install_evaluation_hooks(trainer: object, panel: _TracedFrozenPanel) -> list[dict[str, object]]:
    try:
        dataset_module = __import__(
            "sa_wm.data.video_latent_dataset", fromlist=["VideoLatentTrainingDataset"]
        )
        collate_module = __import__(
            "sa_wm.data.video_latent_collate", fromlist=["collate_video_latent_batch"]
        )
        trainer_module = __import__("sa_wm.world_model.trainer", fromlist=["WMTrainer"])
    except Exception as exc:
        raise CheckpointEvaluationError("AGIBOT_EVALUATOR_NATIVE_IMPORT_FAILED") from exc
    del dataset_module
    original_collate = collate_module.collate_video_latent_batch
    rows: list[dict[str, object]] = []

    def traced_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        result = original_collate(batch)
        identities = [item.get(_SAMPLE_TRACE_KEY) for item in batch]
        if any(not isinstance(identity, Mapping) for identity in identities):
            raise CheckpointEvaluationError("AGIBOT_EVALUATOR_SAMPLE_TRACE_MISSING")
        result[_TRACE_KEY] = [dict(identity) for identity in identities]
        return result

    collate_module.collate_video_latent_batch = traced_collate
    trainer_module.collate_video_latent_batch = traced_collate
    original_compute_loss = trainer.compute_loss

    def traced_compute_loss(batch: Mapping[str, object]):
        result = original_compute_loss(batch)
        identities = batch.get(_TRACE_KEY)
        if not isinstance(identities, list) or len(identities) != 1:
            raise CheckpointEvaluationError("AGIBOT_EVALUATOR_VALIDATION_BATCH_INVALID")
        identity = identities[0]
        if not isinstance(identity, Mapping):
            raise CheckpointEvaluationError("AGIBOT_EVALUATOR_SAMPLE_TRACE_INVALID")
        rows.append(
            {
                "window_id": str(identity.get("sample_window_id", "")),
                "loss": _finite_float(result[0]),
            }
        )
        return result

    trainer.compute_loss = traced_compute_loss
    return rows


def _run_evaluation(args: argparse.Namespace) -> dict[str, object] | None:
    trace_config_path = args.trace_config.expanduser().resolve()
    trace_config = load_trace_config(trace_config_path)
    checkpoint_path, metadata_path, metadata = _validate_checkpoint_inputs(
        checkpoint=args.checkpoint,
        checkpoint_sha256=args.checkpoint_sha256,
        trace_config=trace_config,
        native_source_root=args.native_source_root,
    )
    native_source_root = args.native_source_root.expanduser().resolve()
    native_source_text = str(native_source_root)
    if native_source_text not in sys.path:
        sys.path.insert(0, native_source_text)

    try:
        import torch
        import torch.distributed as dist
        from sa_wm.config import load_config
        from sa_wm.data.video_latent_dataset import VideoLatentTrainingDataset
        from sa_wm.data.video_latent_collate import collate_video_latent_batch
        from sa_wm.world_model.trainer import WMTrainer
    except Exception as exc:
        raise CheckpointEvaluationError("AGIBOT_EVALUATOR_NATIVE_IMPORT_FAILED") from exc

    output_root = Path(str(trace_config["output_root"])).resolve()
    config_path = output_root / "config" / "agibot-training.yaml"
    config = load_config(str(config_path), [])
    if getattr(config.world_model, "backbone", None) != "wan2.2":
        raise CheckpointEvaluationError("AGIBOT_EVALUATOR_WAN22_WMTRAINER_REQUIRED")
    sample_ids = trace_config.get("validation_sample_window_ids")
    if not isinstance(sample_ids, list) or not sample_ids:
        raise CheckpointEvaluationError("AGIBOT_EVALUATOR_VALIDATION_PANEL_INVALID")
    planned_world_size = int(trace_config["world_size"])
    try:
        launched_world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    except ValueError as exc:
        raise CheckpointEvaluationError("AGIBOT_EVALUATOR_DISTRIBUTED_ENV_INVALID") from exc
    if launched_world_size != planned_world_size or rank < 0 or rank >= planned_world_size:
        raise CheckpointEvaluationError("AGIBOT_EVALUATOR_WORLD_SIZE_MISMATCH")
    if planned_world_size > 1:
        timeout_min = int(os.environ.get("SA_WM_DIST_TIMEOUT_MIN", "180"))
        dist.init_process_group("nccl", timeout=timedelta(minutes=timeout_min))
        torch.cuda.set_device(local_rank)

    rows: list[dict[str, object]] = []
    try:
        torch.manual_seed(int(trace_config["seed"]) + rank)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(trace_config["seed"]) + rank)
        dataset = _build_validation_dataset(
            config, dataset_class=VideoLatentTrainingDataset
        )
        panel = _TracedFrozenPanel(dataset, sample_ids)
        config.wm_training.val_video_num = len(sample_ids)
        config.wm_training.val_metrics_seed = int(trace_config["seed"])
        trainer = WMTrainer(
            training_config=config.wm_training,
            model_config=config.world_model,
        )
        # The evaluator must not initialize an optional remote logger.
        trainer._init_wandb = lambda: None
        native_load = trainer.load_pretrained_weights
        reload_counts: list[int] = []

        def verified_load(path: str | Path) -> None:
            if Path(path).resolve() != checkpoint_path.parent:
                raise CheckpointEvaluationError("AGIBOT_EVALUATOR_CHECKPOINT_PATH_MISMATCH")
            native_load(path)
            count = _verify_loaded_checkpoint_tensors(
                checkpoint_path,
                trainer._dit.state_dict(),
                torch=torch,
            )
            reload_counts.append(count)

        trainer.load_pretrained_weights = verified_load
        trainer.setup(init_from=str(checkpoint_path.parent))
        if len(reload_counts) != 1:
            raise CheckpointEvaluationError("AGIBOT_EVALUATOR_CHECKPOINT_RELOAD_UNPROVEN")
        rows = _install_evaluation_hooks(trainer, panel)
        validation_dir = output_root / "evaluation" / checkpoint_path.parent.name
        result = trainer.validate(
            panel,
            int(metadata["step"]),
            output_dir=validation_dir,
        )
        expected_ids = list(sample_ids)
        observed_ids = [row["window_id"] for row in rows]
        if observed_ids != expected_ids:
            raise CheckpointEvaluationError("AGIBOT_EVALUATOR_VALIDATION_PANEL_MISMATCH")
        losses = [float(row["loss"]) for row in rows]
        mean_loss = sum(losses) / len(losses)
        reported_loss = _finite_float(result.get("val_loss"))
        if not math.isclose(reported_loss, mean_loss, rel_tol=1e-7, abs_tol=1e-9):
            raise CheckpointEvaluationError("AGIBOT_EVALUATOR_VALIDATION_MEAN_MISMATCH")
        if rank != 0:
            return None
        raw = {
            "schema_version": 1,
            "artifact_type": _RAW_ARTIFACT_TYPE,
            "state": "complete",
            "model_run_id": str(trace_config["trace_id"]),
            "checkpoint_identity": {
                "path": str(checkpoint_path),
                "sha256": args.checkpoint_sha256,
            },
            "checkpoint_meta_sha256": sha256_file(metadata_path),
            "checkpoint_reload_verified": True,
            "checkpoint_state_key_count": reload_counts[0],
            "native_config_sha256": trace_config["native_config_sha256"],
            "support_bindings": dict(trace_config["support_bindings"]),
            "consumption_trace": rows,
            "native_validation_loss": mean_loss,
        }
        return raw
    finally:
        if planned_world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    raw = _run_evaluation(args)
    if raw is None:
        return
    output = args.raw_output.expanduser().resolve()
    output_root = Path(str(load_trace_config(args.trace_config.expanduser().resolve())["output_root"])).resolve()
    try:
        output.relative_to(output_root / "receipts")
    except ValueError as exc:
        raise CheckpointEvaluationError("AGIBOT_EVALUATOR_OUTPUT_OUTSIDE_RUN") from exc
    checked_path(
        output,
        code="AGIBOT_EVALUATOR_OUTPUT_INVALID",
        error=CheckpointEvaluationError,
    )
    try:
        atomic_write(output, canonical_bytes(raw) + b"\n")
    except OSError as exc:
        raise CheckpointEvaluationError("AGIBOT_EVALUATOR_OUTPUT_WRITE_FAILED") from exc


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


if __name__ == "__main__":
    main()
