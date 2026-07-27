"""ACWM-Phys source freeze verification.

The source record is intentionally an authority object: a URL without a local
resolved commit is not a frozen dependency and cannot start evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
import stat
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .contracts import ContractValidationError


class VendorSourceState(str, Enum):
    UNMATERIALIZED = "unmaterialized"
    MATERIALIZED = "materialized"


@dataclass(frozen=True)
class VendorSource:
    repository: str
    vendor_path: str
    expected_revision: str | None
    state: VendorSourceState
    readonly_required: bool


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_vendor_source(root: Path | None = None) -> VendorSource:
    base = (root or repository_root()).resolve()
    raw = json.loads((base / "vendor" / "acwm_phys.source.json").read_text(encoding="utf-8"))
    try:
        state = VendorSourceState(raw["state"])
    except (KeyError, ValueError) as exc:
        raise ContractValidationError("VENDOR_SOURCE_STATE_INVALID") from exc
    return VendorSource(
        repository=str(raw.get("repository") or ""),
        vendor_path=str(raw.get("vendor_path") or ""),
        expected_revision=raw.get("expected_revision"),
        state=state,
        readonly_required=raw.get("readonly_required") is True,
    )


def verify_vendor_checkout(root: Path | None = None) -> str:
    base = (root or repository_root()).resolve()
    source = load_vendor_source(base)
    if source.state is VendorSourceState.UNMATERIALIZED:
        raise ContractValidationError("VENDOR_UNMATERIALIZED: ACWM-Phys has no recorded immutable commit")
    if not source.repository or not source.vendor_path or not source.expected_revision:
        raise ContractValidationError("VENDOR_FREEZE_RECORD_INCOMPLETE")
    vendor = (base / source.vendor_path).resolve()
    if base not in vendor.parents or not vendor.is_dir():
        raise ContractValidationError("VENDOR_CHECKOUT_MISSING")
    completed = subprocess.run(
        ["git", "-C", str(vendor), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise ContractValidationError("VENDOR_NOT_GIT_CHECKOUT")
    actual = completed.stdout.strip()
    if actual != source.expected_revision:
        raise ContractValidationError("VENDOR_REVISION_MISMATCH")
    remote = _git_output(vendor, ["remote", "get-url", "origin"], "VENDOR_REMOTE_UNAVAILABLE")
    if remote != source.repository:
        raise ContractValidationError("VENDOR_REMOTE_MISMATCH")
    if _git_output(vendor, ["status", "--porcelain=v1", "--untracked-files=all"], "VENDOR_STATUS_UNAVAILABLE"):
        raise ContractValidationError("VENDOR_DIRTY")
    if source.readonly_required:
        _verify_readonly_worktree(vendor)
    return actual


def _git_output(vendor: Path, args: list[str], code: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(vendor), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise ContractValidationError(code)
    return completed.stdout.strip()


def _verify_readonly_worktree(vendor: Path) -> None:
    """Reject a writable baseline checkout before an evaluator may use it."""

    root_mode = os.lstat(vendor).st_mode
    if root_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise ContractValidationError("VENDOR_WORKTREE_WRITABLE")
    tracked = _git_output(vendor, ["ls-files", "-z"], "VENDOR_TRACKED_FILES_UNAVAILABLE")
    for relative in filter(None, tracked.split("\0")):
        member = vendor / relative
        try:
            mode = os.lstat(member).st_mode
        except FileNotFoundError as exc:
            raise ContractValidationError("VENDOR_TRACKED_FILE_MISSING") from exc
        if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
            raise ContractValidationError("VENDOR_TRACKED_FILE_UNSAFE")
        if mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
            raise ContractValidationError("VENDOR_WORKTREE_WRITABLE")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("verify",))
    args = parser.parse_args()
    if args.command == "verify":
        print(verify_vendor_checkout())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
