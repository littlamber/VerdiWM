"""Daemon locks use kernel ownership and survive stale metadata safely."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from wmloop.execute import campaign_daemon, evolution_daemon, pipeline_daemon


@pytest.mark.parametrize(
    ("module", "acquire", "release", "error"),
    [
        (campaign_daemon, "_acquire_daemon_lock", "_release_daemon_lock", campaign_daemon.CampaignDaemonError),
        (pipeline_daemon, "_acquire_daemon_lock", "_release_daemon_lock", pipeline_daemon.PipelineDaemonError),
        (evolution_daemon, "_acquire_lock", "_release_lock", evolution_daemon.EvolutionDaemonError),
    ],
)
def test_stale_metadata_does_not_allow_two_daemon_owners(
    tmp_path: Path, module, acquire: str, release: str, error: type[Exception]
) -> None:
    root = tmp_path / "daemon"
    root.mkdir()
    # A crash can leave the human-readable projection behind.  It must never
    # be used as the ownership primitive or be unlinked while another owner
    # holds the kernel lock.
    (root / "daemon.lock").write_text(json.dumps({"pid": 999999}), encoding="utf-8")
    held = getattr(module, acquire)(root)
    try:
        with pytest.raises(error, match="ALREADY_RUNNING"):
            getattr(module, acquire)(root)
        assert (root / ".daemon.lock").is_file()
        assert json.loads((root / "daemon.lock").read_text(encoding="utf-8"))["pid"] > 0
    finally:
        getattr(module, release)(root, held)
    assert not (root / "daemon.lock").exists()


@pytest.mark.parametrize(
    ("module", "acquire", "release"),
    [
        (campaign_daemon, "_acquire_daemon_lock", "_release_daemon_lock"),
        (pipeline_daemon, "_acquire_daemon_lock", "_release_daemon_lock"),
        (evolution_daemon, "_acquire_lock", "_release_lock"),
    ],
)
def test_metadata_write_failure_releases_kernel_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, module, acquire: str, release: str
) -> None:
    root = tmp_path / "daemon"
    root.mkdir()

    def fail(*_args, **_kwargs):
        raise OSError("injected metadata failure")

    monkeypatch.setattr(module, "atomic_write", fail)
    with pytest.raises(OSError, match="injected metadata failure"):
        getattr(module, acquire)(root)
    monkeypatch.undo()
    held = getattr(module, acquire)(root)
    getattr(module, release)(root, held)

