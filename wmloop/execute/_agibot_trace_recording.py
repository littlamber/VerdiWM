"""Native WMTrainer hooks and per-rank consumption trace recording."""

from __future__ import annotations

import json
import math
import threading
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

if __package__:
    from ._agibot_trace_config import trace_shard_path
    from ._agibot_trace_contract import (
        AgibotTraceError,
        SCHEMA_VERSION,
        _TRACE_KEY,
        _finite_float,
        _rank_from_environment,
        _validate_sample,
        _read_json,
    )
    from ._agibot_trace_validation import _validate_config
else:
    from _agibot_trace_config import trace_shard_path
    from _agibot_trace_contract import (
        AgibotTraceError,
        SCHEMA_VERSION,
        _TRACE_KEY,
        _finite_float,
        _rank_from_environment,
        _validate_sample,
        _read_json,
    )
    from _agibot_trace_validation import _validate_config

class _Recorder:
    def __init__(self, config: Mapping[str, object], shard_path: Path, rank: int):
        self.config = dict(config)
        self.rank = rank
        self.shard_path = shard_path
        self.lock = threading.Lock()
        self.sequence = 0
        self.draw_counts: Counter[str] = Counter()
        self.validation_pass = 0
        self.optimizer_update = 0
        self.finalized = False
        self.pending: list[dict[str, object]] = []
        self.header = {
            "record_type": "header",
            "schema_version": SCHEMA_VERSION,
            "trace_id": config["trace_id"],
            "support_digest": config["support_digest"],
            "source_fingerprint": config["source_fingerprint"],
            "rank": rank,
        }
        self.shard_path.parent.mkdir(parents=True, exist_ok=True)
        if self.shard_path.exists():
            raise AgibotTraceError("AGIBOT_TRACE_SHARD_ALREADY_EXISTS")
        self._append(self.header)

    def _append(self, value: Mapping[str, object]) -> None:
        encoded = json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        with self.lock:
            with self.shard_path.open("a", encoding="utf-8") as stream:
                stream.write(encoded + "\n")
                stream.flush()

    def add_train_batch(
        self,
        *,
        samples: Sequence[Mapping[str, object]],
        microstep: int,
        loss: float,
    ) -> None:
        if type(microstep) is not int or microstep < 1 or not math.isfinite(loss):
            raise AgibotTraceError("AGIBOT_TRACE_TRAIN_LOSS_INVALID")
        rows: list[dict[str, object]] = []
        for slot, sample in enumerate(samples):
            identity = _validate_sample(sample, split="train")
            self.draw_counts[identity["sample_window_id"]] += 1
            rows.append(
                {
                    **identity,
                    "phase": "train",
                    "optimizer_update": None,
                    "microstep": microstep,
                    "batch_slot": slot,
                    "draw_index": self.draw_counts[identity["sample_window_id"]],
                    "batch_loss": loss,
                }
            )
        self.pending.append({"microstep": microstep, "samples": rows})

    def commit_optimizer_update(
        self,
        *,
        expected_microbatches: int,
        learning_rate_min: float,
        learning_rate_max: float,
        parameter_delta_max_abs: float,
        parameter_sample_count: int,
        parameter_changed_sample_count: int,
    ) -> None:
        if (
            type(expected_microbatches) is not int
            or expected_microbatches < 1
            or len(self.pending) != expected_microbatches
        ):
            raise AgibotTraceError("AGIBOT_TRACE_OPTIMIZER_ACCUMULATION_MISMATCH")
        if (
            not math.isfinite(learning_rate_min)
            or not math.isfinite(learning_rate_max)
            or learning_rate_min <= 0
            or learning_rate_max < learning_rate_min
            or not math.isfinite(parameter_delta_max_abs)
            or parameter_delta_max_abs <= 0
            or type(parameter_sample_count) is not int
            or parameter_sample_count < 1
            or type(parameter_changed_sample_count) is not int
            or parameter_changed_sample_count < 1
            or parameter_changed_sample_count > parameter_sample_count
        ):
            raise AgibotTraceError("AGIBOT_TRACE_PARAMETER_UPDATE_NOT_PROVEN")
        self.optimizer_update += 1
        for batch in self.pending:
            for row in batch["samples"]:
                self._write_event(
                    {
                        **row,
                        "optimizer_update": self.optimizer_update,
                        "learning_rate_min": learning_rate_min,
                        "learning_rate_max": learning_rate_max,
                        "parameter_delta_max_abs": parameter_delta_max_abs,
                        "parameter_sample_count": parameter_sample_count,
                        "parameter_changed_sample_count": parameter_changed_sample_count,
                    }
                )
        self.pending.clear()

    def begin_validation_pass(self) -> int:
        self.validation_pass += 1
        return self.validation_pass

    def add_validation_batch(
        self,
        *,
        samples: Sequence[Mapping[str, object]],
        batch_index: int,
        loss: float,
    ) -> None:
        if type(batch_index) is not int or batch_index < 0 or not math.isfinite(loss):
            raise AgibotTraceError("AGIBOT_TRACE_VALIDATION_LOSS_INVALID")
        for slot, sample in enumerate(samples):
            identity = _validate_sample(sample, split="val")
            self.draw_counts[identity["sample_window_id"]] += 1
            self._write_event(
                {
                    **identity,
                    "phase": "validation",
                    "validation_pass": self.validation_pass,
                    "optimizer_update": self.optimizer_update,
                    "batch_index": batch_index,
                    "batch_slot": slot,
                    "draw_index": self.draw_counts[identity["sample_window_id"]],
                    "batch_loss": loss,
                }
            )

    def _write_event(self, event: Mapping[str, object]) -> None:
        self.sequence += 1
        self._append(
            {
                "record_type": "event",
                "schema_version": SCHEMA_VERSION,
                "trace_id": self.config["trace_id"],
                "support_digest": self.config["support_digest"],
                "source_fingerprint": self.config["source_fingerprint"],
                "rank": self.rank,
                "sequence": self.sequence,
                **event,
            }
        )

    def finalize(self, *, expected_optimizer_updates: int) -> None:
        if self.finalized:
            raise AgibotTraceError("AGIBOT_TRACE_RANK_ALREADY_FINALIZED")
        if self.pending:
            raise AgibotTraceError("AGIBOT_TRACE_UNCOMMITTED_MICROBATCHES")
        if self.optimizer_update != expected_optimizer_updates:
            raise AgibotTraceError("AGIBOT_TRACE_EXPECTED_UPDATE_COUNT_MISMATCH")
        if self.validation_pass < 1:
            raise AgibotTraceError("AGIBOT_TRACE_VALIDATION_PASS_MISSING")
        footer = {
            "record_type": "footer",
            "schema_version": SCHEMA_VERSION,
            "trace_id": self.config["trace_id"],
            "support_digest": self.config["support_digest"],
            "source_fingerprint": self.config["source_fingerprint"],
            "rank": self.rank,
            "optimizer_update_count": self.optimizer_update,
            "validation_pass_count": self.validation_pass,
            "state": "complete",
        }
        self._append(footer)
        self.finalized = True


class _FrozenValidationView:
    """Proxy selecting exactly the frozen panel in the native WMTrainer."""

    def __init__(self, dataset: object, sample_ids: Sequence[str]):
        if getattr(dataset, "mode", None) != "val":
            raise AgibotTraceError("AGIBOT_TRACE_VALIDATION_SPLIT_INVALID")
        roots = getattr(dataset, "samples_all", None)
        entries_all = getattr(dataset, "entries_all", None)
        if not isinstance(roots, list) or not isinstance(entries_all, list):
            raise AgibotTraceError("AGIBOT_TRACE_VALIDATION_DATASET_UNSUPPORTED")
        wanted = set(sample_ids)
        refs: dict[str, tuple[int, int]] = {}
        for dataset_index, samples in enumerate(roots):
            for sample_index, ref in enumerate(samples):
                entry_index, first_future_frame = ref
                episode_uid = str(
                    entries_all[dataset_index][int(entry_index)].get("episode_uid", "")
                )
                sample_id = f"{episode_uid}:{int(first_future_frame):06d}"
                if sample_id in wanted:
                    if sample_id in refs:
                        raise AgibotTraceError("AGIBOT_TRACE_VALIDATION_ID_AMBIGUOUS")
                    refs[sample_id] = (dataset_index, sample_index)
        if set(refs) != wanted:
            raise AgibotTraceError("AGIBOT_TRACE_VALIDATION_ID_UNAVAILABLE")
        self.dataset = dataset
        self.sample_ids = list(sample_ids)
        self.refs = [refs[sample_id] for sample_id in self.sample_ids]

    def root_balanced_validation_refs(
        self,
        num_samples: int,
        *,
        seed: int,
        future_latents_required: int | None = None,
        window_fraction: float | None = None,
    ) -> list[tuple[int, int]]:
        del seed, future_latents_required, window_fraction
        if int(num_samples) != len(self.refs):
            raise AgibotTraceError("AGIBOT_TRACE_VALIDATION_REQUEST_MISMATCH")
        return list(self.refs)

    def get_source_sample(self, dataset_index: int, sample_index: int) -> dict[str, Any]:
        return self.dataset.get_source_sample(dataset_index, sample_index)

    def describe_source_sample(self, dataset_index: int, sample_index: int) -> dict[str, Any]:
        return self.dataset.describe_source_sample(dataset_index, sample_index)

    def __getattr__(self, name: str) -> object:
        return getattr(self.dataset, name)


def _sample_optimizer_parameters(optimizer: object) -> tuple[Any, int]:
    """Take a small deterministic sample from every local trainable tensor."""

    try:
        torch = __import__("torch")
        values: list[Any] = []
        sample_count = 0
        for group in optimizer.param_groups:
            for parameter in group.get("params", ()):
                if not bool(getattr(parameter, "requires_grad", False)):
                    continue
                local_parameter = (
                    parameter.to_local()
                    if callable(getattr(parameter, "to_local", None))
                    else parameter
                )
                flat = local_parameter.detach().reshape(-1)
                count = int(flat.numel())
                if count < 1:
                    continue
                indices = sorted({0, count // 4, count // 2, (3 * count) // 4, count - 1})
                index_tensor = torch.tensor(indices, dtype=torch.long, device=flat.device)
                values.append(flat.index_select(0, index_tensor).float())
                sample_count += len(indices)
        if not values:
            raise AgibotTraceError("AGIBOT_TRACE_PARAMETER_SAMPLE_EMPTY")
        return torch.cat(values).detach().clone(), sample_count
    except AgibotTraceError:
        raise
    except Exception as exc:
        raise AgibotTraceError("AGIBOT_TRACE_PARAMETER_SAMPLE_FAILED") from exc


def _parameter_sample_delta(before: Any, after: Any) -> tuple[float, int]:
    try:
        torch = __import__("torch")
        difference = (after - before).abs()
        maximum = float(difference.max().item())
        changed = int(torch.count_nonzero(difference).item())
    except Exception as exc:
        raise AgibotTraceError("AGIBOT_TRACE_PARAMETER_SAMPLE_FAILED") from exc
    if not math.isfinite(maximum) or maximum <= 0 or changed < 1:
        raise AgibotTraceError("AGIBOT_TRACE_PARAMETER_UPDATE_NOT_PROVEN")
    return maximum, changed


def install(config_path: Path) -> None:
    """Patch the reviewed Wan2.2 WMTrainer path before native entrypoint import."""

    config = _read_json(config_path)
    _validate_config(config)
    try:
        dataset_module = __import__("sa_wm.data.video_latent_dataset", fromlist=["VideoLatentTrainingDataset"])
        trainer_module = __import__("sa_wm.world_model.trainer", fromlist=["WMTrainer"])
        collate_module = __import__("sa_wm.data.video_latent_collate", fromlist=["collate_video_latent_batch"])
    except Exception as exc:  # pragma: no cover - exercised by actual bridge startup
        raise AgibotTraceError("AGIBOT_TRACE_NATIVE_IMPORT_FAILED") from exc
    dataset_cls = dataset_module.VideoLatentTrainingDataset
    trainer_cls = trainer_module.WMTrainer
    original_get_source_sample = dataset_cls.get_source_sample
    original_collate = trainer_module.collate_video_latent_batch
    original_train = trainer_cls.train
    original_validate = trainer_cls.validate
    original_compute_loss = trainer_cls.compute_loss
    installed_key = "_verdi_trace_installed"
    if getattr(trainer_cls, installed_key, False):
        raise AgibotTraceError("AGIBOT_TRACE_HOOK_ALREADY_INSTALLED")

    def traced_get_source_sample(self: object, dataset_index: int, sample_index: int, **kwargs: object) -> dict[str, Any]:
        sample = original_get_source_sample(self, dataset_index, sample_index, **kwargs)
        try:
            split = str(self.mode)
            entry_index, first_future_frame = self.samples_all[int(dataset_index)][int(sample_index)]
            entry = self.entries_all[int(dataset_index)][int(entry_index)]
            episode_uid = str(entry.get("episode_uid", ""))
            if not episode_uid:
                raise AgibotTraceError("AGIBOT_TRACE_EPISODE_UID_MISSING")
            sample["_verdi_trace_identity"] = {
                "split_id": split,
                "dataset_index": int(dataset_index),
                "dataset_key": str(self.dataset_keys[int(dataset_index)]),
                "episode_entry_index": int(entry_index),
                "source_sample_index": int(sample_index),
                "episode_uid": episode_uid,
                "first_future_frame": int(first_future_frame),
                "sample_window_id": f"{episode_uid}:{int(first_future_frame):06d}",
            }
            return sample
        except Exception as exc:
            if isinstance(exc, AgibotTraceError):
                raise
            raise AgibotTraceError("AGIBOT_TRACE_SAMPLE_IDENTITY_MISSING") from exc

    def traced_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        result = original_collate(batch)
        identities = [item.get("_verdi_trace_identity") for item in batch]
        if any(not isinstance(item, Mapping) for item in identities):
            raise AgibotTraceError("AGIBOT_TRACE_COLLATE_IDENTITY_MISSING")
        result[_TRACE_KEY] = [dict(item) for item in identities]
        return result

    dataset_cls.get_source_sample = traced_get_source_sample
    trainer_module.collate_video_latent_batch = traced_collate
    collate_module.collate_video_latent_batch = traced_collate

    def traced_train(self: object, train_dataset: object, val_dataset: object = None) -> None:
        rank = _rank_from_environment()
        if rank >= int(config["world_size"]):
            raise AgibotTraceError("AGIBOT_TRACE_RANK_INVALID")
        if int(getattr(self, "world_size", -1)) != int(config["world_size"]):
            raise AgibotTraceError("AGIBOT_TRACE_WORLD_SIZE_MISMATCH")
        if int(self.mcfg.batch_size) != int(config["batch_size"]):
            raise AgibotTraceError("AGIBOT_TRACE_BATCH_SIZE_MISMATCH")
        grad_accum = max(1, int(self.mcfg.grad_accum_steps))
        expected_microsteps = int(config["expected_optimizer_updates"]) * grad_accum
        if int(self.mcfg.max_train_steps) != expected_microsteps:
            raise AgibotTraceError("AGIBOT_TRACE_EXPECTED_MICROSTEPS_MISMATCH")
        if getattr(train_dataset, "mode", None) != "train":
            raise AgibotTraceError("AGIBOT_TRACE_TRAIN_SPLIT_INVALID")
        if val_dataset is None:
            raise AgibotTraceError("AGIBOT_TRACE_VALIDATION_DATASET_REQUIRED")
        if int(getattr(self, "_global_step", 0)) != 0 or int(getattr(self, "_accum_count", 0)) != 0:
            raise AgibotTraceError("AGIBOT_TRACE_TRAINER_MUST_START_FRESH")
        recorder = _Recorder(config, trace_shard_path(config, rank), rank)
        state: dict[str, object] = {
            "phase": "train",
            "validation_batch": 0,
            "validation_ids": [],
            "recorder": recorder,
        }
        optimizer = getattr(self, "_optimizer", None)
        if optimizer is None or not callable(getattr(optimizer, "step", None)):
            raise AgibotTraceError("AGIBOT_TRACE_OPTIMIZER_UNAVAILABLE")
        native_optimizer_step = optimizer.step
        if not callable(native_optimizer_step):
            raise AgibotTraceError("AGIBOT_TRACE_OPTIMIZER_UNAVAILABLE")
        def traced_optimizer_step(*step_args: object, **step_kwargs: object) -> Any:
            before, sampled_count = _sample_optimizer_parameters(optimizer)
            learning_rates = [
                _finite_float(group.get("lr"))
                for group in optimizer.param_groups
            ]
            if not learning_rates or any(value <= 0 for value in learning_rates):
                raise AgibotTraceError("AGIBOT_TRACE_LEARNING_RATE_INVALID")
            result = native_optimizer_step(*step_args, **step_kwargs)
            after, after_count = _sample_optimizer_parameters(optimizer)
            if after_count != sampled_count:
                raise AgibotTraceError("AGIBOT_TRACE_PARAMETER_SAMPLE_CHANGED")
            delta_max, changed_count = _parameter_sample_delta(before, after)
            recorder.commit_optimizer_update(
                expected_microbatches=grad_accum,
                learning_rate_min=min(learning_rates),
                learning_rate_max=max(learning_rates),
                parameter_delta_max_abs=delta_max,
                parameter_sample_count=sampled_count,
                parameter_changed_sample_count=changed_count,
            )
            return result

        optimizer.step = traced_optimizer_step
        self._verdi_trace_state = state
        original_compute = self.compute_loss

        def traced_compute(batch: dict[str, Any]) -> tuple[Any, Any]:
            result = original_compute(batch)
            samples = batch.get(_TRACE_KEY)
            if not isinstance(samples, list) or not samples:
                raise AgibotTraceError("AGIBOT_TRACE_BATCH_IDENTITY_MISSING")
            if state["phase"] == "train":
                recorder.add_train_batch(
                    samples=samples,
                    microstep=int(getattr(self, "_global_step", 0)) + 1,
                    loss=_finite_float(result[0]),
                )
            elif state["phase"] == "validation":
                recorder.add_validation_batch(
                    samples=samples,
                    batch_index=int(state["validation_batch"]),
                    loss=_finite_float(result[0]),
                )
                validation_ids = state["validation_ids"]
                if not isinstance(validation_ids, list):
                    raise AgibotTraceError("AGIBOT_TRACE_VALIDATION_ORDER_INVALID")
                validation_ids.extend(
                    str(sample["sample_window_id"]) for sample in samples
                )
                state["validation_batch"] += 1
            else:
                raise AgibotTraceError("AGIBOT_TRACE_PHASE_INVALID")
            return result

        self.compute_loss = traced_compute

        def traced_validate(dataset: object, step: int, output_dir: Path | str = ".") -> dict[str, Any]:
            if recorder.optimizer_update != int(config["expected_optimizer_updates"]):
                raise AgibotTraceError("AGIBOT_TRACE_VALIDATION_BEFORE_FINAL_UPDATE")
            view = _FrozenValidationView(dataset, config["validation_sample_window_ids"])
            old_limit = self.tcfg.val_video_num
            state["phase"] = "validation"
            state["validation_batch"] = 0
            state["validation_ids"] = []
            validation_pass = recorder.begin_validation_pass()
            self.tcfg.val_video_num = len(config["validation_sample_window_ids"])
            try:
                result = original_validate(self, view, step, output_dir)
            finally:
                self.tcfg.val_video_num = old_limit
                state["phase"] = "train"
            if state["validation_batch"] != len(config["validation_sample_window_ids"]):
                raise AgibotTraceError("AGIBOT_TRACE_VALIDATION_BATCH_COVERAGE_INVALID")
            if state["validation_ids"] != list(config["validation_sample_window_ids"]):
                raise AgibotTraceError("AGIBOT_TRACE_VALIDATION_ORDER_MISMATCH")
            if validation_pass != 1:
                raise AgibotTraceError("AGIBOT_TRACE_VALIDATION_PASS_COUNT_INVALID")
            return result

        self.validate = traced_validate

        original_val_interval = self.tcfg.val_interval
        original_val_video_num = self.tcfg.val_video_num
        # WMTrainer checks validation after every microstep. Set the interval
        # to the final microstep so the fixed panel is consumed once, after the
        # planned optimizer update has completed.
        self.tcfg.val_interval = expected_microsteps
        try:
            result = original_train(self, train_dataset, val_dataset)
            if recorder.validation_pass != 1:
                raise AgibotTraceError("AGIBOT_TRACE_VALIDATION_PASS_COUNT_INVALID")
            if recorder.pending:
                raise AgibotTraceError("AGIBOT_TRACE_UNCOMMITTED_MICROBATCHES")
            if recorder.optimizer_update != int(config["expected_optimizer_updates"]):
                raise AgibotTraceError("AGIBOT_TRACE_OPTIMIZER_UPDATE_COUNT_MISMATCH")
            recorder.finalize(expected_optimizer_updates=int(config["expected_optimizer_updates"]))
            return result
        finally:
            self.compute_loss = original_compute
            self.validate = original_validate.__get__(self, trainer_cls)
            self.tcfg.val_interval = original_val_interval
            self.tcfg.val_video_num = original_val_video_num
            optimizer.step = native_optimizer_step
            if hasattr(self, "_verdi_trace_state"):
                del self._verdi_trace_state

    trainer_cls.train = traced_train
    setattr(trainer_cls, installed_key, True)
