import json
import hashlib
from pathlib import Path

import pytest

from wmloop.contracts import ContractValidationError, validate_document
from wmloop.execute.training_contract import TrainingContractError, validate_training_binding
from wmloop.experiments.training_scale import (
    TrainingScaleError,
    build_training_ladder,
    build_training_scale_plan,
    validate_training_scale_checkpoint_schedule,
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


def _agibot_index(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    train_rows = [
        {"dataset": "agibot", "split": "train", "episode_id": 0, "episode_uid": "train-α", "domain": {"dataset": "agibot", "domain_id": 0}},
        {"dataset": "agibot", "split": "train", "episode_id": 1, "episode_uid": "train-β", "domain": {"dataset": "agibot", "domain_id": 9}},
        {"dataset": "agibot", "split": "train", "episode_id": 2, "episode_uid": "train-γ", "domain": {"dataset": "agibot", "domain_id": 9}},
    ]
    validation_rows = [
        {"dataset": "agibot", "split": "val", "episode_id": 3, "episode_uid": "val-α", "domain": {"dataset": "agibot", "domain_id": 9}},
        {"dataset": "agibot", "split": "val", "episode_id": 4, "episode_uid": "val-β", "domain": {"dataset": "agibot", "domain_id": 0}},
    ]
    for name, rows in (
        ("train.index.jsonl", train_rows),
        ("val.index.jsonl", validation_rows),
    ):
        (root / name).write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
    return root


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


def test_agibot_engineering_diagnostic_is_full_support_but_never_quality_eligible(
    tmp_path: Path,
):
    train = _manifest(tmp_path / "train.json", "train", 100)
    validation = _manifest(tmp_path / "val.json", "val", 20)

    plan = build_training_scale_plan(
        train_manifest=train,
        val_manifest=validation,
        stage="engineering_diagnostic",
    )

    assert plan["state"] == "ready"
    assert plan["training_mode"] == "probe"
    assert plan["evidence_class"] == "runtime_only"
    assert plan["quality_eligible"] is False
    assert plan["dataset"]["selection_fraction"] == 1.0
    assert plan["dataset"]["selected_train_examples"] == plan["dataset"]["train_examples"]
    assert plan["updates"]["planned_steps"] <= 8
    assert plan["replication"]["requested_seed_count"] == 1
    assert len(plan["replication"]["seeds"]) == 1


def test_native_agibot_dataset_root_binds_episode_uid_domain_registry_and_index_hashes(
    tmp_path: Path,
):
    dataset_root = _agibot_index(tmp_path / "agibot")

    plan = build_training_scale_plan(
        dataset_root=dataset_root,
        stage="engineering_diagnostic",
    )

    dataset = plan["dataset"]
    assert plan["state"] == "ready"
    assert dataset["input_format"] == "agibot_native_index_jsonl"
    assert dataset["count_unit"] == "agibot_episode_index_record"
    assert dataset["train_episode_count"] == 3
    assert dataset["validation_episode_count"] == 2
    assert dataset["validation_episode_disjoint"] is True
    assert dataset["selected_train_episode_ids_binding"]["count"] == 3
    assert dataset["selected_train_episode_ids_binding"]["sha256"] == hashlib.sha256(
        "train-α\ntrain-β\ntrain-γ\n".encode()
    ).hexdigest()
    assert "selected_train_episode_ids" not in dataset
    assert dataset["manifest_paths"]["train"].endswith("train.index.jsonl")
    assert dataset["manifest_paths"]["validation"].endswith("val.index.jsonl")
    assert dataset["train_manifest_sha256"] == hashlib.sha256(
        (dataset_root / "train.index.jsonl").read_bytes()
    ).hexdigest()
    assert dataset["validation_manifest_sha256"] == hashlib.sha256(
        (dataset_root / "val.index.jsonl").read_bytes()
    ).hexdigest()
    assert dataset["domain_support"] == {
        "source": "agibot_native_index_v1",
        "domain_ids": [0, 9],
        "required_action_num_domains": 10,
    }
    assert dataset["selected_train_record_ids_binding"]["count"] == 3
    assert "selected_train_record_ids" not in dataset
    assert plan["updates"]["planned_steps"] <= 8
    assert plan["updates"]["steps_per_epoch"] is None
    assert plan["updates"]["target_epochs"] is None
    assert plan["updates"]["realized_epochs"] is None
    assert "1 optimizer step(s)" in plan["rationale"]
    assert "8-step value is the policy cap" in plan["rationale"]
    assert plan["updates"]["checkpoint_eval_steps"][-1] == plan["updates"]["planned_steps"]


def test_custom_checkpoint_ladder_always_includes_final_update(tmp_path: Path):
    train = _manifest(tmp_path / "train.json", "train", 100)
    validation = _manifest(tmp_path / "val.json", "val", 20)
    profile = {
        "status": "local_validated",
        "planner_policy": "admitted",
        "planning": {"checkpoint_ladder": [0.25, 0.5]},
    }

    plan = build_training_scale_plan(
        train_manifest=train,
        val_manifest=validation,
        stage="screen",
        training_profile=profile,
    )

    checkpoint_steps = plan["updates"]["checkpoint_eval_steps"]
    assert checkpoint_steps == sorted(set(checkpoint_steps))
    assert checkpoint_steps[-1] == plan["updates"]["planned_steps"]

    malformed = json.loads(json.dumps(plan))
    malformed["updates"]["checkpoint_eval_steps"] = [
        plan["updates"]["planned_steps"] - 1
    ]
    with pytest.raises(
        TrainingScaleError, match="TRAINING_SCALE_CHECKPOINT_SCHEDULE_INVALID"
    ):
        validate_training_scale_checkpoint_schedule(malformed)


@pytest.mark.parametrize(
    ("filename", "mutate", "error"),
    [
        ("train.index.jsonl", lambda row: row.update(episode_uid="  "), "ROW_CONTRACT_INVALID"),
        ("train.index.jsonl", lambda row: row["domain"].update(domain_id=True), "ROW_CONTRACT_INVALID"),
        ("train.index.jsonl", lambda row: row.update(split="val"), "ROW_CONTRACT_INVALID"),
        ("train.index.jsonl", lambda row: row.update(episode_id="0"), "ROW_CONTRACT_INVALID"),
        ("val.index.jsonl", lambda row: row.update(dataset="droid"), "ROW_CONTRACT_INVALID"),
    ],
)
def test_native_agibot_index_contract_fails_closed(
    tmp_path: Path, filename: str, mutate, error: str
):
    dataset_root = _agibot_index(tmp_path / "agibot")
    index_path = dataset_root / filename
    rows = [json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines()]
    mutate(rows[0])
    index_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    with pytest.raises(TrainingScaleError, match=error):
        build_training_scale_plan(
            dataset_root=dataset_root,
            stage="engineering_diagnostic",
        )


def test_native_agibot_indexes_are_not_used_for_quality_stage_planning(tmp_path: Path):
    dataset_root = _agibot_index(tmp_path / "agibot")

    with pytest.raises(TrainingScaleError, match="ENGINEERING_DIAGNOSTIC_ONLY"):
        build_training_scale_plan(dataset_root=dataset_root, stage="screen")


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("quality_eligible",), True),
        (("updates", "planned_steps"), 9),
        (("dataset", "selection_fraction"), 0.5),
        (("replication", "requested_seed_count"), 2),
        (("replication", "seeds"), [1, 2]),
        (("evidence_class",), "formal"),
    ],
)
def test_agibot_engineering_diagnostic_schema_rejects_quality_or_budget_escalation(
    tmp_path: Path, path: tuple[str, ...], value: object
):
    train = _manifest(tmp_path / "train.json", "train", 100)
    validation = _manifest(tmp_path / "val.json", "val", 20)
    plan = build_training_scale_plan(
        train_manifest=train,
        val_manifest=validation,
        stage="engineering_diagnostic",
    )
    mutated = json.loads(json.dumps(plan))
    target = mutated
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value

    with pytest.raises(ContractValidationError):
        validate_document("training_scale_plan", mutated)


@pytest.mark.parametrize(
    "profile",
    [
        {"status": "local_validated", "planner_policy": "admitted", "planning": {"dataset_fraction": 0.5}},
        {"status": "local_validated", "planner_policy": "admitted", "planning": {"max_steps": 9}},
        {"status": "local_validated", "planner_policy": "admitted", "planning": {"min_steps": 9}},
    ],
)
def test_agibot_engineering_diagnostic_rejects_overridden_policy_bounds(
    tmp_path: Path, profile: dict[str, object]
):
    train = _manifest(tmp_path / "train.json", "train", 100)
    validation = _manifest(tmp_path / "val.json", "val", 20)

    with pytest.raises(TrainingScaleError, match="POLICY_EXCEEDS_BOUND"):
        build_training_scale_plan(
            train_manifest=train,
            val_manifest=validation,
            stage="engineering_diagnostic",
            training_profile=profile,
        )


def test_agibot_engineering_diagnostic_rejects_multiple_requested_seeds(
    tmp_path: Path,
):
    train = _manifest(tmp_path / "train.json", "train", 100)
    validation = _manifest(tmp_path / "val.json", "val", 20)

    with pytest.raises(TrainingScaleError, match="SINGLE_SEED_REQUIRED"):
        build_training_scale_plan(
            train_manifest=train,
            val_manifest=validation,
            stage="engineering_diagnostic",
            requested_seed_count=2,
        )


def test_selection_covers_episodes_before_repeating_windows(tmp_path: Path):
    train = _manifest(tmp_path / "train.json", "train", 100)
    validation = _manifest(tmp_path / "val.json", "val", 20)
    plan = build_training_scale_plan(
        train_manifest=train, val_manifest=validation, stage="screen"
    )
    selected = plan["dataset"]["selected_train_episode_ids"]
    assert len(selected) == plan["dataset"]["selected_train_episode_count"]
    assert len(selected) == 10


def test_selected_record_ids_are_stable_and_distinguish_duplicate_rows(tmp_path: Path):
    duplicate = {
        "episode_id": "train-shared",
        "sample_id": "same-window",
        "horizon_frames": 150,
    }
    train = tmp_path / "train-duplicates.json"
    train.write_text(json.dumps({"records": [duplicate, duplicate]}), encoding="utf-8")
    validation = _manifest(tmp_path / "val.json", "val", 4)
    profile = {
        "status": "local_validated",
        "planner_policy": "admitted",
        "planning": {"dataset_fraction": 1.0},
    }

    first = build_training_scale_plan(
        train_manifest=train,
        val_manifest=validation,
        stage="screen",
        training_profile=profile,
    )
    second = build_training_scale_plan(
        train_manifest=train,
        val_manifest=validation,
        stage="screen",
        training_profile=profile,
    )

    dataset = first["dataset"]
    ids = dataset["selected_train_record_ids"]
    assert ids == second["dataset"]["selected_train_record_ids"]
    assert len(ids) == dataset["selected_train_examples"] == 2
    assert len(set(ids)) == 2
    assert {record_id.split("-")[2] for record_id in ids} == {"00000001", "00000002"}
    assert len({record_id.rsplit("-", 1)[-1] for record_id in ids}) == 1


def test_training_scale_plan_v1_remains_readable_and_v2_requires_record_ids(
    tmp_path: Path,
):
    train = _manifest(tmp_path / "train.json", "train", 20)
    validation = _manifest(tmp_path / "val.json", "val", 4)
    plan = build_training_scale_plan(
        train_manifest=train, val_manifest=validation, stage="screen"
    )

    legacy_plan = json.loads(json.dumps(plan))
    legacy_plan["schema_version"] = 1
    del legacy_plan["dataset"]["selected_train_record_ids"]
    validate_document("training_scale_plan", legacy_plan)

    incomplete_v2 = json.loads(json.dumps(plan))
    incomplete_v2["schema_version"] = 2
    del incomplete_v2["dataset"]["selected_train_record_ids"]
    with pytest.raises(ContractValidationError):
        validate_document("training_scale_plan", incomplete_v2)


def test_native_agibot_plan_compacts_large_episode_identity_sets(tmp_path: Path):
    dataset_root = _agibot_index(tmp_path / "agibot")
    train_index = dataset_root / "train.index.jsonl"
    base = json.loads(train_index.read_text(encoding="utf-8").splitlines()[0])
    rows = []
    for index in range(10_000):
        row = json.loads(json.dumps(base))
        row["episode_id"] = index
        row["episode_uid"] = f"train-{index:05d}"
        row["domain"]["domain_id"] = index % 10
        rows.append(json.dumps(row))
    train_index.write_text("\n".join(rows) + "\n", encoding="utf-8")

    plan = build_training_scale_plan(
        dataset_root=dataset_root,
        stage="engineering_diagnostic",
    )

    dataset = plan["dataset"]
    assert dataset["selected_train_examples"] == 10_000
    assert dataset["selected_train_record_ids_binding"]["count"] == 10_000
    assert dataset["selected_train_episode_ids_binding"]["count"] == 10_000
    assert "selected_train_record_ids" not in dataset
    assert "selected_train_episode_ids" not in dataset
    assert len(json.dumps(plan)) < 10_000


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
