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
