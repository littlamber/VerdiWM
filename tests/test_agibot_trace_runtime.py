from __future__ import annotations

import builtins
import hashlib
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from wmloop.execute import agibot_trace_bridge, agibot_trace_runtime as trace


def _support(ids: list[str]) -> dict[str, object]:
    return {"validation_support": {"sample_window_ids": ids}}


def _support_bindings() -> dict[str, str]:
    return {
        "agibot_support_digest": "a" * 64,
        "shard_catalog_sha256": "b" * 64,
        "train_episode_uids_sha256": "c" * 64,
        "train_window_set_sha256": "d" * 64,
        "validation_sample_window_ids_sha256": "e" * 64,
    }


def _identity(split: str, episode_uid: str, frame: int) -> dict[str, object]:
    return {
        "split_id": split,
        "dataset_index": 0,
        "dataset_key": "agibot",
        "episode_entry_index": 0,
        "source_sample_index": frame,
        "episode_uid": episode_uid,
        "first_future_frame": frame,
        "sample_window_id": f"{episode_uid}:{frame:06d}",
    }


def test_traced_compute_calls_native_loss_once(monkeypatch, tmp_path: Path):
    class StopAfterCompute(Exception):
        pass

    class DummyOptimizer:
        param_groups = [{"params": []}]

        def step(self):
            raise AssertionError("the stopped fake train loop must not step")

    class FakeDataset:
        mode = "train"

        def get_source_sample(self, dataset_index, sample_index, **kwargs):
            return {"dataset_index": dataset_index, "sample_index": sample_index}

    class FakeTrainer:
        def __init__(self):
            self.world_size = 1
            self.mcfg = SimpleNamespace(
                batch_size=1, grad_accum_steps=1, max_train_steps=1
            )
            self.tcfg = SimpleNamespace(val_interval=10, val_video_num=4)
            self._global_step = 0
            self._accum_count = 0
            self._optimizer = DummyOptimizer()
            self.compute_calls = 0

        def compute_loss(self, batch):
            self.compute_calls += 1
            return 2.5, {"loss": 2.5}

        def validate(self, dataset, step, output_dir="."):
            return {"val_loss": 1.0}

        def train(self, train_dataset, val_dataset=None):
            self.compute_loss(
                {trace._TRACE_KEY: [_identity("train", "train-episode", 4)]}
            )
            raise StopAfterCompute

    dataset_module = types.ModuleType("sa_wm.data.video_latent_dataset")
    dataset_module.VideoLatentTrainingDataset = FakeDataset
    trainer_module = types.ModuleType("sa_wm.world_model.trainer")
    trainer_module.WMTrainer = FakeTrainer
    trainer_module.collate_video_latent_batch = lambda batch: batch
    collate_module = types.ModuleType("sa_wm.data.video_latent_collate")
    collate_module.collate_video_latent_batch = lambda batch: batch
    modules = {
        "sa_wm": types.ModuleType("sa_wm"),
        "sa_wm.data": types.ModuleType("sa_wm.data"),
        "sa_wm.world_model": types.ModuleType("sa_wm.world_model"),
        "sa_wm.data.video_latent_dataset": dataset_module,
        "sa_wm.data.video_latent_collate": collate_module,
        "sa_wm.world_model.trainer": trainer_module,
    }
    for package_name in ("sa_wm", "sa_wm.data", "sa_wm.world_model"):
        modules[package_name].__path__ = []
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    output_root = tmp_path / "out"
    config_path = trace.write_trace_config(
        output_root,
        trace_id="trace-once",
        support=_support(["val-episode:000000"]),
        native_config_sha256="a" * 64,
        world_size=1,
        batch_size=1,
        gradient_accumulation=1,
        expected_optimizer_updates=1,
        seed=17,
        support_bindings=_support_bindings(),
    )
    trace.install(config_path)
    trainer = FakeTrainer()

    with pytest.raises(StopAfterCompute):
        trainer.train(FakeDataset(), FakeDataset())

    assert trainer.compute_calls == 1
    shard = trace.trace_shard_path(trace.load_trace_config(config_path), 0)
    assert shard.exists()
    assert len(shard.read_text(encoding="utf-8").splitlines()) == 1


def test_bridge_installs_hooks_before_native_main_and_binds_checkpoint(
    monkeypatch, tmp_path: Path
):
    source_root = tmp_path / "native"
    entrypoint = source_root / "scripts" / "training" / "train_world_model.py"
    entrypoint.parent.mkdir(parents=True)
    entrypoint.write_text(
        "import builtins\nbuiltins._agibot_bridge_order.append('main')\n",
        encoding="utf-8",
    )
    checkpoint = tmp_path / "wan2.2-checkpoint"
    checkpoint.write_bytes(b"checkpoint")
    init_from = tmp_path / "target-action-checkpoint"
    init_from.mkdir()
    (init_from / "model.safetensors").write_bytes(b"target checkpoint")
    substituted_init_from = tmp_path / "other-action-checkpoint"
    substituted_init_from.mkdir()
    (substituted_init_from / "model.safetensors").write_bytes(b"other checkpoint")
    output_root = tmp_path / "run"
    native_config_path = output_root / "config" / "agibot-training.yaml"
    native_config_path.parent.mkdir(parents=True)
    native_config_path.write_text(
        "world_model:\n  backbone: wan2.2\n  checkpoint: "
        + json.dumps(str(checkpoint))
        + "\n",
        encoding="utf-8",
    )
    config_path = trace.write_trace_config(
        output_root,
        trace_id="bridge-order",
        support=_support(["val-episode:000000"]),
        native_config_sha256=hashlib.sha256(
            native_config_path.read_bytes()
        ).hexdigest(),
        world_size=1,
        batch_size=1,
        gradient_accumulation=1,
        expected_optimizer_updates=1,
        seed=17,
        support_bindings=_support_bindings(),
        init_from_path=str(init_from),
    )

    order: list[str] = []
    monkeypatch.setattr(builtins, "_agibot_bridge_order", order, raising=False)
    original_argv = list(sys.argv)
    original_path = list(sys.path)

    def fake_install(path: Path) -> None:
        assert path == config_path.resolve()
        assert str(source_root) in sys.path
        order.append("install")

    monkeypatch.setattr(agibot_trace_bridge, "install", fake_install)
    try:
        agibot_trace_bridge.main(
            [
                "--trace-config",
                str(config_path),
                "--",
                str(entrypoint),
                "--config",
                str(native_config_path),
                "--init_from",
                str(init_from),
            ]
        )
    finally:
        sys.argv = original_argv
        sys.path[:] = original_path

    assert order == ["install", "main"]
    with pytest.raises(
        trace.AgibotTraceError, match="NATIVE_CONFIG_ARGUMENT_INVALID"
    ):
        trace.validate_native_training_command(
            [
                str(entrypoint),
                "--config",
                str(native_config_path),
                "--init_from",
                str(init_from),
                "world_model.backbone=wan2.2_causal",
            ],
            trace.load_trace_config(config_path),
        )
    with pytest.raises(
        trace.AgibotTraceError, match="AGIBOT_TRACE_NATIVE_CONFIG_ARGUMENT_INVALID"
    ):
        trace.validate_native_training_command(
            [
                str(entrypoint),
                "--config",
                str(native_config_path),
                "--init_from",
                str(substituted_init_from),
            ],
            trace.load_trace_config(config_path),
        )


def test_bridge_command_uses_torchrun_and_only_native_entrypoint(tmp_path: Path):
    entrypoint = tmp_path / "scripts" / "training" / "train_world_model.py"
    command = trace.build_bridge_command(
        "/opt/wan/bin/python",
        [str(entrypoint), "--config", "/run/config/agibot-training.yaml"],
        trace_config_path=tmp_path / "agibot-trace-config.json",
        bridge_path=tmp_path / "agibot_trace_bridge.py",
        world_size=4,
    )

    assert command[:5] == [
        "/opt/wan/bin/python",
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
        "4",
    ]
    assert command[5].endswith("agibot_trace_bridge.py")
    assert command[-4:] == [
        "--",
        str(entrypoint),
        "--config",
        "/run/config/agibot-training.yaml",
    ]


def test_validation_order_is_checked_for_each_rank():
    validation_ids = ["val-a:000000", "val-b:000000"]
    support = _support(validation_ids)
    support_digest = trace.canonical_digest(support)
    native_config_digest = "f" * 64
    config = {
        "trace_id": "validation-order",
        "support_digest": support_digest,
        "source_fingerprint": trace.canonical_digest(
            {
                "support_digest": support_digest,
                "native_config_sha256": native_config_digest,
            }
        ),
        "native_config_sha256": native_config_digest,
        "output_root": "/tmp/validation-order",
        "world_size": 2,
        "batch_size": 1,
        "gradient_accumulation": 1,
        "expected_optimizer_updates": 1,
        "seed": 17,
        "support_bindings": _support_bindings(),
        "validation_sample_window_ids": validation_ids,
    }
    events: list[dict[str, object]] = []
    for rank in range(2):
        for sequence, phase, sample_id in (
            (1, "train", "train:000000"),
            (2, "validation", "val-a:000000"),
            (3, "validation", "val-b:000000"),
        ):
            split, episode_uid, frame_text = (
                ("train", "train", "000000")
                if phase == "train"
                else ("val", *sample_id.rsplit(":", 1))
            )
            event = {
                **_identity(split, episode_uid, int(frame_text)),
                "trace_id": config["trace_id"],
                "support_digest": config["support_digest"],
                "source_fingerprint": config["source_fingerprint"],
                "rank": rank,
                "phase": phase,
                "optimizer_update": 1,
                "validation_pass": 1,
                "sequence": sequence,
                "batch_slot": 0,
                "batch_loss": 1.0,
            }
            if phase == "train":
                event.update(
                    microstep=1,
                    learning_rate_min=1e-6,
                    learning_rate_max=1e-6,
                    parameter_delta_max_abs=1e-4,
                    parameter_sample_count=1,
                    parameter_changed_sample_count=1,
                )
            else:
                event["batch_index"] = sequence - 2
            events.append(event)

    trace._validate_global_events(events, config)
    events[-2], events[-1] = events[-1], events[-2]
    with pytest.raises(trace.AgibotTraceError, match="VALIDATION_ORDER_MISMATCH"):
        trace._validate_global_events(events, config)


def _write_complete_shard(
    config: dict[str, object], shard_path: Path, rank: int, *, reverse_validation: bool = False
) -> None:
    recorder = trace._Recorder(config, shard_path, rank)
    recorder.add_train_batch(
        samples=[_identity("train", f"train-rank-{rank}", 8)],
        microstep=1,
        loss=1.25,
    )
    recorder.commit_optimizer_update(
        expected_microbatches=1,
        learning_rate_min=1e-6,
        learning_rate_max=1e-6,
        parameter_delta_max_abs=1e-4,
        parameter_sample_count=5,
        parameter_changed_sample_count=1,
    )
    recorder.begin_validation_pass()
    validation_ids = ["val-a:000000", "val-b:000000"]
    if reverse_validation:
        validation_ids.reverse()
    for batch_index, sample_id in enumerate(validation_ids):
        episode_uid, frame = sample_id.rsplit(":", 1)
        recorder.add_validation_batch(
            samples=[_identity("val", episode_uid, int(frame))],
            batch_index=batch_index,
            loss=0.75,
        )
    recorder.finalize(expected_optimizer_updates=1)


def test_trace_merge_requires_every_rank_and_identical_frozen_panel_order(
    tmp_path: Path,
):
    output_root = tmp_path / "merge-run"
    config_path = trace.write_trace_config(
        output_root,
        trace_id="merge-trace",
        support=_support(["val-a:000000", "val-b:000000"]),
        native_config_sha256="b" * 64,
        world_size=2,
        batch_size=1,
        gradient_accumulation=1,
        expected_optimizer_updates=1,
        seed=17,
        support_bindings=_support_bindings(),
    )
    config = trace.load_trace_config(config_path)
    shard_paths = [trace.trace_shard_path(config, rank) for rank in range(2)]
    for rank, path in enumerate(shard_paths):
        _write_complete_shard(config, path, rank)

    merged = trace.merge_trace_shards(config_path, shard_paths)
    assert merged["state"] == "complete"
    assert merged["rank_completion_count"] == 2
    assert merged["optimizer_update_count"] == 1
    assert merged["validation_pass_count"] == 1

    with pytest.raises(trace.AgibotTraceError, match="RANK_SHARD_MISSING"):
        trace.merge_trace_shards(config_path, shard_paths[:1])

    shard_paths[1].unlink()
    _write_complete_shard(config, shard_paths[1], 1, reverse_validation=True)
    with pytest.raises(trace.AgibotTraceError, match="VALIDATION_ORDER_MISMATCH"):
        trace.merge_trace_shards(config_path, shard_paths)
