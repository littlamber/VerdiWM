from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from wmloop.contracts import ContractValidationError, validate_document
from wmloop.execute.agibot_support import (
    AgibotSupportError,
    agibot_receipt_bindings,
    build_agibot_training_support,
    validate_agibot_training_support,
)
from wmloop.storage import canonical_bytes


NATIVE_FILES = (
    "scripts/training/train_world_model.py",
    "sa_wm/world_model/trainer.py",
    "sa_wm/data/video_latent_dataset.py",
    "sa_wm/data/video_latent_reader.py",
    "sa_wm/data/video_latent.py",
    "sa_wm/world_model/action_contract.py",
)
LOADER = {
    "num_history": 2,
    "num_frames": 2,
    "sample_stride_video_frames": 4,
    "history_selector": "recent",
    "history_offsets": [],
    "history_dilations": [1, 2],
    "history_collapse_prob": 0.15,
    "history_eval_dilation": 2,
    "generated_history_enabled": False,
    "condition_view": "arm_slot_eef_pose",
    "body_conditioning": True,
}


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _row(split: str, episode_uid: str, shard_prefix: str) -> dict[str, object]:
    def descriptor(component: str) -> dict[str, object]:
        return {
            "shard": f"{shard_prefix}-{component}.bin",
            "offset_bytes": 0,
            "shape": [6, 4],
            "dtype": "float32",
        }

    return {
        "dataset": "agibot",
        "split": split,
        "episode_uid": episode_uid,
        "time": {"num_video_frames": 24, "num_latent_frames": 6},
        "latent": descriptor("latent"),
        "signals": descriptor("signals"),
        "model_condition_views": {"arm_slot_eef_pose": descriptor("condition")},
    }


def _fixture(root: Path) -> tuple[Path, Path]:
    dataset = root / "dataset"
    source = root / "source"
    _write_json(
        dataset / "manifest.json",
        {
            "production_contract": {
                "dataset_key": "agibot-arm-slot-fixture",
                "family": "agibot",
                "condition_view": "arm_slot_eef_pose",
                "body_conditioning": True,
            },
            "condition_schema": {"condition_view": "arm_slot_eef_pose"},
            "condition_schema_v2": {
                "body": {
                    "normalization": "stats/body_control.json schema_version=3 per-slot contract"
                }
            },
        },
    )
    _write_json(dataset / "stats" / "arm_slot_eef_pose.json", {"schema_version": 1})
    _write_json(
        dataset / "stats" / "body_control.json",
        {"schema_version": 3, "source": "train_only"},
    )
    train_rows = [_row("train", "episode-train-001", "train")]
    val_rows = [_row("val", "episode-val-001", "val")]
    for name, rows in (("train.index.jsonl", train_rows), ("val.index.jsonl", val_rows)):
        path = dataset / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    for row in train_rows + val_rows:
        for descriptor in [row["latent"], row["signals"], *row["model_condition_views"].values()]:
            shard = dataset / "shards" / str(descriptor["shard"])
            shard.parent.mkdir(parents=True, exist_ok=True)
            shard.write_bytes(b"fixture shard bytes")
    for relative in NATIVE_FILES:
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# fixture source: {relative}\n", encoding="utf-8")
    return dataset, source


def _build(dataset: Path, source: Path) -> dict[str, object]:
    return build_agibot_training_support(
        dataset,
        source,
        loader=LOADER,
        validation_sample_count=3,
    )


def test_agibot_support_binds_full_native_windows_and_fixed_heldout_panel(tmp_path: Path):
    dataset, source = _fixture(tmp_path)

    first = _build(dataset, source)
    second = _build(dataset, source)
    validate_document("agibot_training_support", first)
    validate_agibot_training_support(
        first,
        expected_dataset_root=dataset,
        expected_native_source_root=source,
    )

    assert first == second
    assert first["train_support"]["window_count"] == 5
    assert first["train_support"]["window_set_sha256"]
    assert first["validation_support"]["sample_window_ids"] == [
        "episode-val-001:000000",
        "episode-val-001:000008",
        "episode-val-001:000016",
    ]
    assert first["validation_support"]["sample_window_ids_sha256"]
    assert first["train_support"]["episode_count"] == 1
    assert "episode_uids" not in first["train_support"]
    assert first["train_support"]["episode_uids_sha256"] == hashlib.sha256(
        b"episode-train-001\n"
    ).hexdigest()
    assert first["validation_support"]["episode_uids_disjoint_from_train"] is True
    assert first["shard_catalog"]["file_count"] == 6
    assert "shards/train-latent.bin" in first["shard_catalog"]["files"]
    assert first["shard_catalog"]["files"]["shards/train-latent.bin"]["reference_count"] == 1
    assert "references_sha256" in first["shard_catalog"]["files"]["shards/train-latent.bin"]
    assert "references" not in first["shard_catalog"]["files"]["shards/train-latent.bin"]
    assert "does not hash the full shard contents" in first["shard_catalog"]["claim_boundary"]
    assert "no model-improvement" in first["claim_boundary"]

    bindings = agibot_receipt_bindings(first)
    assert bindings["agibot_support_digest"]
    assert bindings["shard_catalog_sha256"] == first["shard_catalog"]["sha256"]
    assert bindings["train_window_set_sha256"] == first["train_support"]["window_set_sha256"]
    assert bindings["validation_sample_window_ids"] == first["validation_support"]["sample_window_ids"]


def test_agibot_support_binds_large_train_window_set_without_materializing_ids(
    tmp_path: Path,
):
    dataset, source = _fixture(tmp_path)
    index_path = dataset / "train.index.jsonl"
    row = json.loads(index_path.read_text(encoding="utf-8"))
    row["time"] = {"num_video_frames": 2_000_000, "num_latent_frames": 1_000_002}
    index_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    support = _build(dataset, source)
    train = support["train_support"]
    assert train["window_count"] == 500_000
    assert "sample_window_ids" not in train
    expected_binding = {
        "split": "train",
        "index_sha256": support["inputs"]["train_index"],
        "loader": support["loader"],
        "window_count": train["window_count"],
        "window_rule": train["window_rule"],
    }
    assert train["window_set_sha256"] == hashlib.sha256(
        canonical_bytes(expected_binding)
    ).hexdigest()


def test_agibot_support_revalidation_detects_index_source_and_shard_size_changes(
    tmp_path: Path,
):
    dataset, source = _fixture(tmp_path)
    support = _build(dataset, source)

    with (dataset / "shards" / "train-latent.bin").open("ab") as handle:
        handle.write(b"changed-size")
    with pytest.raises(AgibotSupportError, match="BINDING_MISMATCH"):
        validate_agibot_training_support(support)

    dataset, source = _fixture(tmp_path / "fresh")
    support = _build(dataset, source)
    native_path = source / NATIVE_FILES[0]
    native_path.write_text("# changed native implementation\n", encoding="utf-8")
    with pytest.raises(AgibotSupportError, match="BINDING_MISMATCH"):
        validate_agibot_training_support(support)

    dataset, source = _fixture(tmp_path / "fresh-index")
    support = _build(dataset, source)
    index_path = dataset / "train.index.jsonl"
    row = json.loads(index_path.read_text(encoding="utf-8"))
    row["index_binding_probe"] = "changed but still valid JSONL"
    index_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(AgibotSupportError, match="BINDING_MISMATCH"):
        validate_agibot_training_support(support)


def test_agibot_support_binds_native_wmtrainer_implementation(tmp_path: Path):
    dataset, source = _fixture(tmp_path)
    support = _build(dataset, source)

    trainer_path = source / "sa_wm" / "world_model" / "trainer.py"
    assert support["native_source"]["files"]["sa_wm/world_model/trainer.py"] == hashlib.sha256(
        trainer_path.read_bytes()
    ).hexdigest()

    trainer_path.write_text("# changed WMTrainer implementation\n", encoding="utf-8")
    with pytest.raises(AgibotSupportError, match="BINDING_MISMATCH"):
        validate_agibot_training_support(support)


def test_agibot_support_binds_unlisted_native_package_dependencies(tmp_path: Path):
    dataset, source = _fixture(tmp_path)
    dependency = source / "sa_wm" / "world_model" / "model_dependency.py"
    dependency.write_text("MODEL_REVISION = 1\n", encoding="utf-8")

    support = _build(dataset, source)
    assert "sa_wm/world_model/model_dependency.py" in support["native_source"]["files"]

    dependency.write_text("MODEL_REVISION = 2\n", encoding="utf-8")
    with pytest.raises(AgibotSupportError, match="BINDING_MISMATCH"):
        validate_agibot_training_support(support)


@pytest.mark.parametrize(
    "bad_shard",
    ["../escape.bin", "/absolute.bin", "bad\\path.bin", "nested/shard.bin", "./train-latent.bin"],
)
def test_agibot_support_rejects_unsafe_shard_paths(tmp_path: Path, bad_shard: str):
    dataset, source = _fixture(tmp_path)
    index_path = dataset / "train.index.jsonl"
    row = json.loads(index_path.read_text())
    row["latent"]["shard"] = bad_shard
    index_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(AgibotSupportError, match="SHARD_REFERENCE_INVALID"):
        _build(dataset, source)


@pytest.mark.parametrize("symlink_target", ["file", "directory"])
def test_agibot_support_rejects_shard_symlinks(tmp_path: Path, symlink_target: str):
    dataset, source = _fixture(tmp_path)
    outside = tmp_path / "outside"

    if symlink_target == "file":
        outside.write_bytes(b"outside shard")
        shard = dataset / "shards" / "train-latent.bin"
        shard.unlink()
        shard.symlink_to(outside)
    else:
        (dataset / "shards").rename(dataset / "real-shards")
        (dataset / "shards").symlink_to(dataset / "real-shards", target_is_directory=True)

    with pytest.raises(AgibotSupportError, match="SHARD_PATH_INVALID"):
        _build(dataset, source)


def test_agibot_support_rejects_missing_shard_and_train_validation_episode_leakage(
    tmp_path: Path,
):
    dataset, source = _fixture(tmp_path)
    (dataset / "shards" / "train-latent.bin").unlink()
    with pytest.raises(AgibotSupportError, match="SHARD_MISSING"):
        _build(dataset, source)

    dataset, source = _fixture(tmp_path / "leak")
    val_path = dataset / "val.index.jsonl"
    val_row = json.loads(val_path.read_text())
    val_row["episode_uid"] = "episode-train-001"
    val_path.write_text(json.dumps(val_row) + "\n", encoding="utf-8")
    with pytest.raises(AgibotSupportError, match="TRAIN_VALIDATION_LEAKAGE"):
        _build(dataset, source)


def test_agibot_support_schema_rejects_false_or_expanded_claims(tmp_path: Path):
    dataset, source = _fixture(tmp_path)
    support = _build(dataset, source)
    invalid = json.loads(json.dumps(support))
    invalid["claim_boundary"] = "This proves model improvement."

    with pytest.raises(ContractValidationError):
        validate_document("agibot_training_support", invalid)

    invalid = json.loads(json.dumps(support))
    invalid["shard_catalog"]["files"]["../escape.bin"] = {
        "size_bytes": 1,
        "reference_count": 1,
        "references_sha256": "0" * 64,
    }
    with pytest.raises(ContractValidationError):
        validate_document("agibot_training_support", invalid)


def test_sparse_history_loader_parameters_are_bound_into_support_and_receipts(
    tmp_path: Path,
):
    dataset, source = _fixture(tmp_path)
    sparse_loader = {
        **LOADER,
        "history_selector": "sparse",
        "history_dilations": [1, 2],
        "history_collapse_prob": 0.15,
        "history_eval_dilation": 2,
    }
    support = build_agibot_training_support(
        dataset, source, loader=sparse_loader, validation_sample_count=3
    )

    assert support["loader"] == sparse_loader
    assert support["train_support"]["window_count"] == 4
    assert support["validation_support"]["sample_window_ids"] == [
        "episode-val-001:000004",
        "episode-val-001:000012",
        "episode-val-001:000016",
    ]
    assert agibot_receipt_bindings(support)["agibot_support_digest"]

    changed_eval_loader = {**sparse_loader, "history_eval_dilation": 1}
    changed_eval_support = build_agibot_training_support(
        dataset, source, loader=changed_eval_loader, validation_sample_count=3
    )
    assert support["validation_support"]["sample_window_ids"] == changed_eval_support[
        "validation_support"
    ]["sample_window_ids"]
    assert (
        agibot_receipt_bindings(support)["agibot_support_digest"]
        != agibot_receipt_bindings(changed_eval_support)["agibot_support_digest"]
    )


@pytest.mark.parametrize(
    "updates",
    [
        {"history_dilations": []},
        {"history_dilations": [1, 0]},
        {"history_dilations": [1, True]},
        {"history_collapse_prob": -0.01},
        {"history_collapse_prob": 1.01},
        {"history_collapse_prob": float("nan")},
        {"history_eval_dilation": 0},
    ],
)
def test_agibot_support_rejects_invalid_sparse_history_loader_parameters(
    tmp_path: Path, updates: dict[str, object]
):
    dataset, source = _fixture(tmp_path)
    loader = {**LOADER, "history_selector": "sparse", **updates}

    with pytest.raises(AgibotSupportError, match="LOADER_CONFIG_INVALID"):
        build_agibot_training_support(
            dataset, source, loader=loader, validation_sample_count=3
        )
