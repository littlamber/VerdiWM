"""Public CLI contract for the campaign dispatcher."""

from __future__ import annotations

import json
from pathlib import Path

from wmloop.control import campaign_dispatcher


def test_dispatcher_cli_state_root_matches_verdi_run_state_root(
    tmp_path: Path, monkeypatch, capsys
):
    state_root = tmp_path / "user-state"
    observed = []
    monkeypatch.setattr(
        campaign_dispatcher,
        "run_dispatcher",
        lambda options: observed.append(options) or {"pending": 0},
    )
    monkeypatch.setattr(
        "sys.argv",
        ["verdiwm-campaign-dispatcher", "--state-root", str(state_root)],
    )

    assert campaign_dispatcher.main() == 0

    assert observed[0].state_root == state_root / "campaigns"
    assert json.loads(capsys.readouterr().out) == {"pending": 0}


def test_dispatcher_cli_help_explains_state_root_semantics(monkeypatch, capsys):
    monkeypatch.setattr(
        "sys.argv", ["verdiwm-campaign-dispatcher", "--help"]
    )

    try:
        campaign_dispatcher.main()
    except SystemExit as exc:
        assert exc.code == 0
    else:  # pragma: no cover - argparse always exits for --help
        raise AssertionError("argparse help did not exit")

    assert "containing campaigns/" in capsys.readouterr().out
