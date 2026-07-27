"""Git-worktree isolation for provider-side trial execution.

The frozen vendor checkout remains read-only. A trial obtains a distinct,
writable worktree at an explicitly recorded revision; no caller may point the
sandbox at an arbitrary path or reuse an existing trial directory.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


class SandboxError(RuntimeError):
    """A trial workspace could not be created or retired safely."""


_TRIAL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_REVISION = re.compile(r"[0-9a-f]{40}\Z")


@dataclass(frozen=True)
class SandboxLease:
    trial_id: str
    worktree: Path
    source_revision: str


class WorktreeSandbox:
    """Create and remove trial worktrees through Git's administration API."""

    def __init__(self, *, vendor_root: Path, runs_root: Path) -> None:
        self._vendor_root = Path(vendor_root).resolve()
        self._runs_root = Path(runs_root).resolve()

    def create(self, *, trial_id: str, expected_revision: str) -> SandboxLease:
        if not _TRIAL_ID.fullmatch(trial_id):
            raise SandboxError("SANDBOX_TRIAL_ID_INVALID")
        if not _REVISION.fullmatch(expected_revision):
            raise SandboxError("SANDBOX_REVISION_INVALID")
        if not self._vendor_root.is_dir():
            raise SandboxError("SANDBOX_VENDOR_MISSING")
        destination = self._runs_root / trial_id / "worktree"
        if destination.exists() or destination.is_symlink():
            raise SandboxError("SANDBOX_TRIAL_ALREADY_EXISTS")
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        completed = subprocess.run(
            ["git", "-C", str(self._vendor_root), "worktree", "add", "--detach", str(destination), expected_revision],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise SandboxError("SANDBOX_WORKTREE_CREATE_FAILED")
        actual = _git_revision(destination)
        if actual != expected_revision:
            self.remove(SandboxLease(trial_id=trial_id, worktree=destination, source_revision=actual))
            raise SandboxError("SANDBOX_REVISION_MISMATCH")
        return SandboxLease(trial_id=trial_id, worktree=destination, source_revision=actual)

    def remove(self, lease: SandboxLease) -> None:
        expected = (self._runs_root / lease.trial_id / "worktree").resolve()
        if lease.worktree.resolve() != expected or not _TRIAL_ID.fullmatch(lease.trial_id):
            raise SandboxError("SANDBOX_LEASE_PATH_INVALID")
        if not lease.worktree.exists() and not lease.worktree.is_symlink():
            raise SandboxError("SANDBOX_WORKTREE_MISSING")
        completed = subprocess.run(
            ["git", "-C", str(self._vendor_root), "worktree", "remove", "--force", str(lease.worktree)],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0 or lease.worktree.exists() or lease.worktree.is_symlink():
            raise SandboxError("SANDBOX_WORKTREE_REMOVE_FAILED")


def _git_revision(worktree: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(worktree), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    revision = completed.stdout.strip()
    if completed.returncode != 0 or not _REVISION.fullmatch(revision):
        raise SandboxError("SANDBOX_WORKTREE_NOT_GIT")
    return revision
