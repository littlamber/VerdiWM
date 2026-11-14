from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from wmloop.cli import _parser, main
from wmloop.control.commands.inputs import _resolve_run_inputs


def test_short_cli_name_is_used_when_invoked_as_verdi(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["/tmp/bin/verdi"])
    assert _parser().prog == "verdi"


def test_legacy_cli_name_remains_stable(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["/tmp/bin/verdiwm"])
    assert _parser().prog == "verdiwm"


def test_bare_verdi_prints_guide_and_exits_successfully(capsys) -> None:
    assert main([]) == 0
    output = capsys.readouterr().out
    assert "research" in output
    assert "Verdi" in output


def test_research_evaluate_irg_is_a_first_class_command() -> None:
    args = _parser().parse_args(
        [
            "research",
            "evaluate-irg",
            "--plan",
            "plan.json",
            "--projections",
            "projections.jsonl",
            "--labels",
            "labels.json",
            "--output",
            "results/irg",
        ]
    )
    assert args.research_command == "evaluate-irg"
    assert args.output.name == "irg"


def test_research_evaluate_predictor_is_a_first_class_command() -> None:
    args = _parser().parse_args([
        "research", "evaluate-predictor", "--examples", "examples.json",
        "--plan", "plan.json", "--labels", "labels.json", "--output", "results/predictor",
    ])
    assert args.research_command == "evaluate-predictor"
    assert args.examples.name == "examples.json"
    assert args.output.name == "predictor"


def test_research_plan_protected_metrics_are_distinct_from_irg_metrics() -> None:
    args = _parser().parse_args([
        "research", "plan", "--protected-metrics", "action_fidelity", "latency",
        "--irg-protected-metric", "safety",
    ])
    assert args.protected_metrics == ["action_fidelity", "latency"]
    assert args.irg_protected_metric == ["safety"]


def test_model_run_accepts_dataset_root_alias() -> None:
    args = _parser().parse_args(["run", "--model", "model", "--dataset-root", "data/agibot"])
    assert args.command == "run"
    assert args.data == "data/agibot"


def test_model_run_accepts_native_training_config() -> None:
    args = _parser().parse_args(
        ["run", "--model", "model", "--dataset-root", "data/agibot", "--training-config", "train.yaml"]
    )
    assert args.training_config == Path("train.yaml")


def test_model_run_accepts_positive_world_size_for_auto_agibot_plan() -> None:
    args = _parser().parse_args(
        [
            "run",
            "--model",
            "model",
            "--dataset-root",
            "data/agibot",
            "--training-config",
            "train.yaml",
            "--world-size",
            "4",
        ]
    )
    assert args.world_size == 4


def test_model_run_rejects_non_positive_world_size() -> None:
    with pytest.raises(SystemExit):
        _parser().parse_args(["run", "--world-size", "0"])


def test_research_plan_accepts_dataset_root_alias() -> None:
    args = _parser().parse_args([
        "research", "plan", "--model", "model", "--dataset-root", "data/agibot"
    ])
    assert args.research_command == "plan"
    assert args.data == "data/agibot"


def test_plan_training_parser_accepts_native_agibot_diagnostic_inputs(tmp_path: Path) -> None:
    args = _parser().parse_args(
        [
            "plan-training",
            "--agibot-root",
            str(tmp_path / "agibot"),
            "--stage",
            "engineering_diagnostic",
        ]
    )
    assert args.dataset_root == tmp_path / "agibot"
    assert args.stage == "engineering_diagnostic"


def _write_native_agibot_indexes(root: Path) -> None:
    root.mkdir(parents=True)
    for name, split, episodes in (
        ("train.index.jsonl", "train", ["train-a", "train-b", "train-c"]),
        ("val.index.jsonl", "val", ["val-a", "val-b"]),
    ):
        rows = [
            {
                "dataset": "agibot",
                "split": split,
                "episode_id": index,
                "episode_uid": episode_uid,
                "domain": {"dataset": "agibot", "domain_id": 9},
            }
            for index, episode_uid in enumerate(episodes)
        ]
        (root / name).write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )


def test_plan_training_cli_accepts_dataset_root_and_paired_native_indexes(
    tmp_path: Path, capsys
) -> None:
    from wmloop.cli import main

    dataset_root = tmp_path / "agibot"
    _write_native_agibot_indexes(dataset_root)
    invocations = (
        [
            "plan-training",
            "--dataset-root",
            str(dataset_root),
            "--stage",
            "engineering_diagnostic",
        ],
        [
            "plan-training",
            "--train-manifest",
            str(dataset_root / "train.index.jsonl"),
            "--val-manifest",
            str(dataset_root / "val.index.jsonl"),
            "--stage",
            "engineering_diagnostic",
        ],
    )
    for argv in invocations:
        assert main(argv) == 0
        result = json.loads(capsys.readouterr().out)
        assert result["state"] == "ready"
        assert result["stage"] == "engineering_diagnostic"
        assert result["dataset"]["input_format"] == "agibot_native_index_jsonl"
        assert result["dataset"]["domain_support"]["required_action_num_domains"] == 10


def test_run_uses_dataset_root_project_setting(monkeypatch, tmp_path: Path) -> None:
    model = tmp_path / "model"
    data = tmp_path / "agibot"
    source = tmp_path / "source"
    for directory in (model, data, source):
        directory.mkdir()
    monkeypatch.setattr(
        "wmloop.control.commands.inputs._load_project_intent",
        lambda: {
            "model": str(model),
            "data": None,
            "dataset_root": str(data),
            "source": str(source),
            "goal": "reduce validation loss",
        },
    )
    args = Namespace(
        model=None,
        data=None,
        goal=None,
        intent=None,
        source=None,
        target_metrics=None,
        budget=None,
        adapter="auto",
        mode=None,
        adapter_profile=None,
        runtime_python=None,
        state_root=tmp_path / "state",
        campaign_id=None,
    )

    resolved = _resolve_run_inputs(args)

    assert resolved["data"] == str(data)
