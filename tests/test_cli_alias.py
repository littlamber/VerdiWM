from __future__ import annotations

import sys

from wmloop.cli import _parser, main


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
