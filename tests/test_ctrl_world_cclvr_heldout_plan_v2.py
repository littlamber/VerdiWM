from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import freeze_ctrl_world_cclvr_heldout_plan_v2 as freezer


def _write(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_freezes_v2_plan_from_audited_cells_and_binds_adapter(monkeypatch, tmp_path: Path) -> None:
    ctrl_world = tmp_path / "Ctrl-World"
    adapter = ctrl_world / "models" / "multiscale_history_adapter.py"
    adapter.parent.mkdir(parents=True)
    adapter.write_text("REPAIRED = True\n", encoding="utf-8")
    dependencies = {}
    template = _write(
        tmp_path / "template.json",
        {
            "dependencies": dependencies,
            "routing_jobs": [],
            "action_jobs": [],
            "references": {},
            "promotion_gate": {},
        },
    )
    contexts = _write(tmp_path / "contexts.json", {"contexts": []})
    dependency_files = []
    for name in ("evaluator.py", "aggregator.py", "launcher.py", "base.py"):
        path = tmp_path / name
        path.write_text(name, encoding="utf-8")
        dependency_files.append(path)
    monkeypatch.setattr(freezer, "CTRL_WORLD_ROOT", ctrl_world)
    monkeypatch.setattr(freezer, "TEMPLATE", template)
    monkeypatch.setattr(freezer, "CONTEXTS", contexts)
    monkeypatch.setattr(freezer, "EVALUATOR", dependency_files[0])
    monkeypatch.setattr(freezer, "AGGREGATOR", dependency_files[1])
    monkeypatch.setattr(freezer, "LAUNCHER", dependency_files[2])
    monkeypatch.setattr(freezer, "BASE_EVALUATOR", dependency_files[3])

    experiments = []
    settled_cells = []
    for cell_id in freezer.CELL_ORDER:
        checkpoint = tmp_path / f"{cell_id}.pt"
        checkpoint.write_text(cell_id, encoding="utf-8")
        bank = _write(tmp_path / f"{cell_id}.json", {"cell_id": cell_id})
        experiments.append(
            {
                "id": cell_id,
                "name": f"name-{cell_id}",
                "supervision_bank": {
                    "path": str(bank),
                    "sha256": _digest(bank),
                    "variant": f"variant-{cell_id}",
                },
            }
        )
        settled_cells.append(
            {
                "cell_id": cell_id,
                "optimization_steps": 64,
                "checkpoint_scope": {"state": "passed"},
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": _digest(checkpoint),
            }
        )
    training = _write(
        tmp_path / "training.json",
        {
            "artifact_type": "ctrl-world-cclvr-ablation-plan",
            "state": "frozen_before_execution",
            "confirmation_authorized": False,
            "execution_order": list(freezer.CELL_ORDER),
            "source_hashes": {"models/multiscale_history_adapter.py": _digest(adapter)},
            "common_training": {"cclvr_value_hidden_dim": 128, "cclvr_policy_temperature": 0.1},
            "experiments": experiments,
        },
    )
    settlement = _write(
        tmp_path / "settlement.json",
        {
            "artifact_type": "ctrl-world-cclvr-64-step-training-settlement",
            "state": "training_screen_completed",
            "promotion_episode_used": False,
            "confirmation_authorized": False,
            "promotion_episode": "1799",
            "plan": str(training),
            "plan_sha256": _digest(training),
            "cells": settled_cells,
        },
    )
    output = tmp_path / "heldout-v2.json"

    freezer.freeze(
        training_plan_path=training,
        training_settlement_path=settlement,
        output_path=output,
        evaluation_output_root=tmp_path / "evaluation-output",
        experiment_id="ctrl-world-cclvr-heldout-screen-64-v3",
    )

    plan = json.loads(output.read_text(encoding="utf-8"))
    assert plan["artifact_type"] == "verdiwm-ctrl-world-cclvr-heldout-plan-v2"
    assert plan["experiment_id"] == "ctrl-world-cclvr-heldout-screen-64-v3"
    assert [cell["id"] for cell in plan["cells"]] == list(freezer.CELL_ORDER)
    assert plan["dependencies"]["adapter_source"] == str(adapter)
    assert plan["dependencies"]["adapter_source_sha256"] == _digest(adapter)
    assert plan["confirmation_authorized"] is False


def test_rejects_adapter_source_drift(monkeypatch, tmp_path: Path) -> None:
    ctrl_world = tmp_path / "Ctrl-World"
    adapter = ctrl_world / "models" / "multiscale_history_adapter.py"
    adapter.parent.mkdir(parents=True)
    adapter.write_text("REPAIRED = True\n", encoding="utf-8")
    monkeypatch.setattr(freezer, "CTRL_WORLD_ROOT", ctrl_world)
    training = _write(
        tmp_path / "training.json",
        {
            "artifact_type": "ctrl-world-cclvr-ablation-plan",
            "state": "frozen_before_execution",
            "confirmation_authorized": False,
            "source_hashes": {"models/multiscale_history_adapter.py": "0" * 64},
        },
    )
    settlement = _write(
        tmp_path / "settlement.json",
        {
            "artifact_type": "ctrl-world-cclvr-64-step-training-settlement",
            "state": "training_screen_completed",
            "promotion_episode_used": False,
            "confirmation_authorized": False,
            "promotion_episode": "1799",
            "plan": str(training),
            "plan_sha256": _digest(training),
        },
    )

    with pytest.raises(freezer.CCLVRHeldoutPlanError, match="ADAPTER_SOURCE_DRIFT"):
        freezer.freeze(
            training_plan_path=training,
            training_settlement_path=settlement,
            output_path=tmp_path / "heldout-v2.json",
            evaluation_output_root=tmp_path / "evaluation-output",
        )


def test_rejects_training_settlement_that_used_promotion_episode(tmp_path: Path) -> None:
    training = _write(
        tmp_path / "training.json",
        {
            "artifact_type": "ctrl-world-cclvr-ablation-plan",
            "state": "frozen_before_execution",
            "confirmation_authorized": False,
        },
    )
    settlement = _write(
        tmp_path / "settlement.json",
        {
            "artifact_type": "ctrl-world-cclvr-64-step-training-settlement",
            "state": "training_screen_completed",
            "promotion_episode_used": True,
            "confirmation_authorized": False,
            "promotion_episode": "1799",
            "plan": str(training),
            "plan_sha256": _digest(training),
        },
    )

    with pytest.raises(freezer.CCLVRHeldoutPlanError, match="TRAINING_INVALID"):
        freezer.freeze(
            training_plan_path=training,
            training_settlement_path=settlement,
            output_path=tmp_path / "heldout-v2.json",
            evaluation_output_root=tmp_path / "evaluation-output",
        )
