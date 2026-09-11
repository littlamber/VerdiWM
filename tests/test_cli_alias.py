from __future__ import annotations

import sys

from wmloop.cli import _parser


def test_short_cli_name_is_used_when_invoked_as_verdi(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["/tmp/bin/verdi"])
    assert _parser().prog == "verdi"


def test_legacy_cli_name_remains_stable(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["/tmp/bin/verdiwm"])
    assert _parser().prog == "verdiwm"
