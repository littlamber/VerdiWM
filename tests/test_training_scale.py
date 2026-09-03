import json
from pathlib import Path

import pytest

from wmloop.execute.training_contract import TrainingContractError, validate_training_binding
from wmloop.experiments.training_scale import (
    TrainingScaleError,
    build_training_ladder,
    build_training_scale_plan,
)


def _manifest(path: Path, prefix: str, count: int) -> Path:
    path.write_text(
        json.dumps({
            "records": [
                {
                    "episode_id": f"{prefix}-{index}",
                    "sample_id": f"{prefix}-{index}:0:150",
                    "horizon_frames": 150,
                }
                for index in range(count)
            ]
        }),
        encoding="utf-8",
    )
    return path


def test_probe_is_runtime_only_and_formal_plan_is_long(tmp_path: Path):
    train = _manifest(tmp_path / "train.json", "train", 100)
    validation = _manifest(tmp_path / "val.json", "val", 20)

    probe = build_training_scale_plan(
        train_manifest=train, val_manifest=validation, stage="probe"
    )
    confirm = build_training_scale_plan(
        train_manifest=train, val_manifest=validation, stage="confirm"
    )

    assert probe["training_mode"] == "probe"
    assert probe["quality_eligible"] is False
    assert confirm["training_mode"] == "long"
    assert confirm["evidence_class"] == "formal"
    assert confirm["quality_eligible"] is True
    assert confirm["dataset"]["validation_episode_disjoint"] is True


def test_selection_covers_episodes_before_repeating_windows(tmp_path: Path):
    train = _manifest(tmp_path / "train.json", "train", 100)
    validation = _manifest(tmp_path / "val.json", "val", 20)
    plan = build_training_scale_plan(
        train_manifest=train, val_manifest=validation, stage="screen"
    )
    selected = plan["dataset"]["selected_train_episode_ids"]
    assert len(selected) == plan["dataset"]["selected_train_episode_count"]
    assert len(selected) == 10


def test_train_validation_overlap_blocks_plan(tmp_path: Path):
    train = _manifest(tmp_path / "train.json", "shared", 20)
    validation = _manifest(tmp_path / "val.json", "shared", 20)
    plan = build_training_scale_plan(
        train_manifest=train, val_manifest=validation, stage="pilot"
    )
    assert plan["state"] == "blocked"
    assert any("TRAINING_SCALE_TRAIN_VALIDATION_EPISODE_OVERLAP" in item for item in plan["blockers"])


def test_missing_episode_identity_fails_closed(tmp_path: Path):
    train = tmp_path / "train.json"
    train.write_text(json.dumps({"records": [{"sample_id": "x"}]}), encoding="utf-8")
    validation = _manifest(tmp_path / "val.json", "val", 4)
    with pytest.raises(TrainingScaleError, match="EPISODE_ID_MISSING"):
        build_training_scale_plan(
            train_manifest=train, val_manifest=validation, stage="screen"
        )


def test_ladder_automatically_constructs_formal_upgrades(tmp_path: Path):
    train = _manifest(tmp_path / "train.json", "train", 100)
    validation = _manifest(tmp_path / "val.json", "val", 20)
    ladder = build_training_ladder(
        train_manifest=train,
        val_manifest=validation,
        current_stage="probe",
        target_stage="confirm",
    )
    assert ladder["state"] == "ready"
    assert [row["stage"] for row in ladder["plans"]] == ["screen", "pilot", "confirm"]
    assert all(row["automatic"] is True for row in ladder["transitions"])
    assert all(row["screen_failure_veto"] is False for row in ladder["transitions"])


def test_ladder_accepts_declared_smoke_as_current_stage(tmp_path: Path):
    train = _manifest(tmp_path / "train.json", "train", 100)
    validation = _manifest(tmp_path / "val.json", "val", 20)
    ladder = build_training_ladder(
        train_manifest=train,
        val_manifest=validation,
        current_stage="smoke",
        target_stage="confirm",
    )
    assert [row["stage"] for row in ladder["plans"]] == ["screen", "pilot", "confirm"]
    assert ladder["transitions"][0]["from_stage"] == "smoke"


def test_formal_scheduler_binding_cannot_be_a_probe():
    binding = {
        "train_manifest": "/data/train.json",
        "validation_manifest": "/data/val.json",
        "mode": "probe",
        "steps": 1,
        "record_limit": 1,
        "sampler": "sequential",
        "seed_count": 1,
        "scale_plan_sha256": "0" * 64,
        "runner_contract": "VERDIWM_TRAINING_CONTRACT_V1",
    }
    with pytest.raises(TrainingContractError, match="PROBE_NOT_ALLOWED"):
        validate_training_binding(binding, expected_stage="confirm")


def test_formal_scheduler_binding_requires_validation_panel():
    binding = {
        "train_manifest": "/data/train.json",
        "validation_manifest": "/data/val.json",
        "mode": "long",
        "steps": 1024,
        "record_limit": 16,
        "sampler": "episode_balanced",
        "seed_count": 3,
        "scale_plan_sha256": "0" * 64,
        "runner_contract": "VERDIWM_TRAINING_CONTRACT_V1",
    }
    with pytest.raises(TrainingContractError, match="VALIDATION_PANEL_REQUIRED"):
        validate_training_binding(binding, expected_stage="confirm")
    binding["validation_panel_size"] = 3
    normalized = validate_training_binding(binding, expected_stage="confirm")
    assert normalized["validation_panel_size"] == 3


def test_pilot_is_a_supported_alias_for_generic_formal_binding():
    binding = {
        "train_manifest": "/data/train.json",
        "validation_manifest": "/data/val.json",
        "mode": "long",
        "steps": 256,
        "record_limit": 8,
        "sampler": "episode_balanced",
        "seed_count": 3,
        "validation_panel_size": 3,
        "scale_plan_sha256": "0" * 64,
        "runner_contract": "VERDIWM_TRAINING_CONTRACT_V1",
    }
    assert validate_training_binding(binding, expected_stage="pilot")["expected_stage"] == "pilot"
