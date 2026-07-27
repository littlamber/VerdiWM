"""Auditable agent repair sessions for newly proposed ACWM primitives.

An agent may edit and debug arbitrary *trial* code in a Git worktree.  This
module records the commands it actually ran and seals the resulting patch into
an immutable staging record.  It deliberately is not an operating-system
sandbox: callers that execute untrusted code still need a container or another
host-level isolation backend.  Its responsibility is source-revision binding,
timeout cleanup, frozen-evaluator protection, and reproducible evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

from wmloop.execute.backends import CommandBackend, LocalSubprocessBackend
from wmloop.primitives.render import PrimitiveRenderError, assert_diff_does_not_modify_frozen_evaluator


class AgentStagingError(RuntimeError):
    """A candidate repair session could not produce trustworthy evidence."""


_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_GIT_REVISION = re.compile(r"[0-9a-f]{40}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_FROZEN_EVALUATOR_PATHS = frozenset({"eval.py", "scripts/eval_all.sh"})


@dataclass(frozen=True)
class CommandReceipt:
    ordinal: int
    label: str
    argv: tuple[str, ...]
    timeout_seconds: float
    exit_code: int | None
    timed_out: bool
    duration_seconds: float
    stdout_sha256: str
    stderr_sha256: str

    @property
    def passed(self) -> bool:
        return not self.timed_out and self.exit_code == 0

    def to_document(self) -> dict[str, object]:
        return {**asdict(self), "argv": list(self.argv), "passed": self.passed}


@dataclass(frozen=True)
class StagedPrimitiveCandidate:
    candidate_id: str
    source_revision: str
    registry_digest: str
    worktree_diff_sha256: str
    changed_paths: tuple[str, ...]
    receipts: tuple[CommandReceipt, ...]
    required_check_labels: tuple[str, ...]
    ready_for_promotion: bool
    manifest_path: Path
    diff_path: Path

    def to_document(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "artifact_type": "staged-agent-primitive-candidate",
            "candidate_id": self.candidate_id,
            "source_revision": self.source_revision,
            "registry_digest": self.registry_digest,
            "worktree_diff_sha256": self.worktree_diff_sha256,
            "changed_paths": list(self.changed_paths),
            "required_check_labels": list(self.required_check_labels),
            "ready_for_promotion": self.ready_for_promotion,
            "receipts": [receipt.to_document() for receipt in self.receipts],
        }


class AgentRepairSession:
    """Run agent-selected repair checks, then seal the final worktree patch.

    ``required_check_labels`` identify the last checks whose successful result
    is required for promotion.  Earlier failed diagnostics are preserved and do
    not make a repaired candidate ineligible.  This supports the normal agent
    loop: inspect failure -> edit worktree -> rerun the gate.
    """

    def __init__(
        self,
        *,
        worktree: Path,
        staging_root: Path,
        candidate_id: str,
        source_revision: str,
        registry_digest: str,
        required_check_labels: Sequence[str],
        environment: Mapping[str, str] | None = None,
        command_backend: CommandBackend | None = None,
        max_command_timeout_seconds: float = 86_400.0,
    ) -> None:
        _validate_identifier(candidate_id, "AGENT_STAGING_CANDIDATE_ID_INVALID")
        _validate_git_revision(source_revision, "AGENT_STAGING_SOURCE_REVISION_INVALID")
        _validate_sha256(registry_digest, "AGENT_STAGING_REGISTRY_DIGEST_INVALID")
        labels = tuple(str(label) for label in required_check_labels)
        if not labels or len(labels) != len(set(labels)) or any(not _IDENTIFIER.fullmatch(label) for label in labels):
            raise AgentStagingError("AGENT_STAGING_REQUIRED_CHECKS_INVALID")
        self._worktree = Path(worktree).resolve(strict=True)
        self._source_revision = source_revision
        if _git_revision(self._worktree) != source_revision:
            raise AgentStagingError("AGENT_STAGING_SOURCE_REVISION_MISMATCH")
        root = Path(staging_root).resolve()
        destination = root / candidate_id
        if destination.exists() or destination.is_symlink():
            raise AgentStagingError("AGENT_STAGING_CANDIDATE_EXISTS")
        destination.mkdir(mode=0o700, parents=True)
        (destination / "attempts").mkdir(mode=0o700)
        self._destination = destination
        self._candidate_id = candidate_id
        self._registry_digest = registry_digest
        self._required_check_labels = labels
        self._environment = _execution_environment(destination, environment)
        self._command_backend = command_backend or LocalSubprocessBackend()
        if not math.isfinite(max_command_timeout_seconds) or max_command_timeout_seconds <= 0:
            raise AgentStagingError("AGENT_STAGING_MAX_TIMEOUT_INVALID")
        self._max_command_timeout_seconds = float(max_command_timeout_seconds)
        self._receipts: list[CommandReceipt] = []
        self._sealed = False
        self._write_document(
            destination / "session.json",
            {
                "schema_version": 1,
                "candidate_id": candidate_id,
                "source_revision": source_revision,
                "registry_digest": registry_digest,
                "required_check_labels": list(labels),
            },
        )

    def run(self, *, label: str, argv: Sequence[str], timeout_seconds: float) -> CommandReceipt:
        """Execute one agent-selected command in the writable trial worktree."""

        if self._sealed:
            raise AgentStagingError("AGENT_STAGING_SESSION_SEALED")
        _validate_identifier(label, "AGENT_STAGING_COMMAND_LABEL_INVALID")
        command = _validate_argv(argv)
        if (
            not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
            or timeout_seconds > self._max_command_timeout_seconds
        ):
            raise AgentStagingError("AGENT_STAGING_COMMAND_TIMEOUT_INVALID")
        try:
            result = self._command_backend.run(
                worktree=self._worktree,
                command=command,
                environment=self._environment,
                timeout_seconds=timeout_seconds,
            )
        except OSError as exc:
            raise AgentStagingError("AGENT_STAGING_COMMAND_START_FAILED") from exc
        ordinal = len(self._receipts) + 1
        stdout_path = self._destination / "attempts" / f"{ordinal:03d}.stdout"
        stderr_path = self._destination / "attempts" / f"{ordinal:03d}.stderr"
        _write_bytes(stdout_path, result.stdout)
        _write_bytes(stderr_path, result.stderr)
        receipt = CommandReceipt(
            ordinal=ordinal,
            label=label,
            argv=command,
            timeout_seconds=timeout_seconds,
            exit_code=result.exit_code,
            timed_out=result.timed_out,
            duration_seconds=result.duration_seconds,
            stdout_sha256=hashlib.sha256(result.stdout).hexdigest(),
            stderr_sha256=hashlib.sha256(result.stderr).hexdigest(),
        )
        self._write_document(self._destination / "attempts" / f"{ordinal:03d}.json", receipt.to_document())
        self._receipts.append(receipt)
        return receipt

    def seal(self) -> StagedPrimitiveCandidate:
        """Persist the candidate diff and promotion eligibility without mutating it."""

        if self._sealed:
            raise AgentStagingError("AGENT_STAGING_SESSION_SEALED")
        if _git_revision(self._worktree) != self._source_revision:
            raise AgentStagingError("AGENT_STAGING_SOURCE_REVISION_CHANGED")
        patch, changed_paths = _capture_worktree_patch(self._worktree)
        if not patch:
            raise AgentStagingError("AGENT_STAGING_WORKTREE_UNCHANGED")
        try:
            patch_text = patch.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise AgentStagingError("AGENT_STAGING_DIFF_NOT_TEXT") from exc
        try:
            assert_diff_does_not_modify_frozen_evaluator(patch_text)
        except PrimitiveRenderError as exc:
            raise AgentStagingError("AGENT_STAGING_FROZEN_EVALUATOR_MODIFIED") from exc
        if set(changed_paths) & _FROZEN_EVALUATOR_PATHS:
            raise AgentStagingError("AGENT_STAGING_FROZEN_EVALUATOR_MODIFIED")
        _verify_changed_members(self._worktree, changed_paths)
        diff_path = self._destination / "candidate.diff"
        _write_bytes(diff_path, patch)
        latest = {receipt.label: receipt for receipt in self._receipts}
        ready = all(label in latest and latest[label].passed for label in self._required_check_labels)
        manifest_path = self._destination / "candidate.json"
        candidate = StagedPrimitiveCandidate(
            candidate_id=self._candidate_id,
            source_revision=self._source_revision,
            registry_digest=self._registry_digest,
            worktree_diff_sha256=hashlib.sha256(patch).hexdigest(),
            changed_paths=changed_paths,
            receipts=tuple(self._receipts),
            required_check_labels=self._required_check_labels,
            ready_for_promotion=ready,
            manifest_path=manifest_path,
            diff_path=diff_path,
        )
        self._write_document(manifest_path, candidate.to_document())
        self._sealed = True
        return candidate

    @staticmethod
    def _write_document(path: Path, document: Mapping[str, object]) -> None:
        _write_bytes(path, json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8") + b"\n")


def _execution_environment(destination: Path, requested: Mapping[str, str] | None) -> dict[str, str]:
    home = destination / "agent-home"
    home.mkdir(mode=0o700)
    environment = {
        "HOME": str(home),
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
    }
    if requested is not None:
        for key, value in requested.items():
            if not isinstance(key, str) or not key or "=" in key or "\x00" in key or not isinstance(value, str) or "\x00" in value:
                raise AgentStagingError("AGENT_STAGING_ENVIRONMENT_INVALID")
            environment[key] = value
    return environment


def _capture_worktree_patch(worktree: Path) -> tuple[bytes, tuple[str, ...]]:
    tracked = _git_output_bytes(worktree, ("diff", "--binary", "--full-index", "--no-ext-diff", "HEAD"), expected=(0,))
    tracked_names = _git_output_bytes(worktree, ("diff", "--name-only", "-z", "HEAD"), expected=(0,))
    untracked_names = _git_output_bytes(worktree, ("ls-files", "--others", "--exclude-standard", "-z"), expected=(0,))
    untracked = _parse_relative_names(untracked_names)
    parts = [tracked]
    for relative in untracked:
        member = worktree / relative
        _require_regular_member(member, "AGENT_STAGING_UNTRACKED_MEMBER_INVALID")
        completed = subprocess.run(
            ["git", "diff", "--binary", "--full-index", "--no-index", "--", "/dev/null", relative],
            cwd=worktree,
            capture_output=True,
            check=False,
        )
        if completed.returncode not in (0, 1):
            raise AgentStagingError("AGENT_STAGING_UNTRACKED_DIFF_FAILED")
        parts.append(completed.stdout)
    changed = tuple(sorted(set(_parse_relative_names(tracked_names)) | set(untracked)))
    return b"".join(parts), changed


def _verify_changed_members(worktree: Path, changed_paths: Sequence[str]) -> None:
    if not changed_paths:
        raise AgentStagingError("AGENT_STAGING_WORKTREE_UNCHANGED")
    for relative in changed_paths:
        if relative in _FROZEN_EVALUATOR_PATHS:
            raise AgentStagingError("AGENT_STAGING_FROZEN_EVALUATOR_MODIFIED")
        member = worktree / relative
        if member.exists() or member.is_symlink():
            _require_regular_member(member, "AGENT_STAGING_CHANGED_MEMBER_INVALID")


def _parse_relative_names(payload: bytes) -> tuple[str, ...]:
    names: list[str] = []
    for raw in filter(None, payload.split(b"\0")):
        try:
            relative = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise AgentStagingError("AGENT_STAGING_PATH_ENCODING_INVALID") from exc
        candidate = Path(relative)
        if (
            candidate.is_absolute()
            or not relative
            or any(part in {"", ".", ".."} for part in candidate.parts)
            or ".git" in candidate.parts
            or any(character in relative for character in ("\n", "\r"))
        ):
            raise AgentStagingError("AGENT_STAGING_CHANGED_PATH_INVALID")
        names.append(relative)
    return tuple(names)


def _require_regular_member(path: Path, code: str) -> None:
    try:
        metadata = os.lstat(path)
    except FileNotFoundError as exc:
        raise AgentStagingError(code) from exc
    if not os.path.isfile(path) or os.path.islink(path) or metadata.st_size > 16 * 1024 * 1024:
        raise AgentStagingError(code)


def _git_revision(worktree: Path) -> str:
    output = _git_output_bytes(worktree, ("rev-parse", "HEAD"), expected=(0,))
    try:
        revision = output.decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise AgentStagingError("AGENT_STAGING_WORKTREE_NOT_GIT") from exc
    _validate_git_revision(revision, "AGENT_STAGING_WORKTREE_NOT_GIT")
    return revision


def _git_output_bytes(worktree: Path, arguments: Sequence[str], *, expected: tuple[int, ...]) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(worktree), *arguments],
        capture_output=True,
        check=False,
    )
    if completed.returncode not in expected:
        raise AgentStagingError("AGENT_STAGING_GIT_COMMAND_FAILED")
    return completed.stdout


def _validate_identifier(value: str, code: str) -> None:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise AgentStagingError(code)


def _validate_git_revision(value: str, code: str) -> None:
    if not isinstance(value, str) or not _GIT_REVISION.fullmatch(value):
        raise AgentStagingError(code)


def _validate_sha256(value: str, code: str) -> None:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise AgentStagingError(code)


def _validate_argv(argv: Sequence[str]) -> tuple[str, ...]:
    command = tuple(argv)
    if not command or any(not isinstance(item, str) or not item or "\x00" in item for item in command):
        raise AgentStagingError("AGENT_STAGING_COMMAND_INVALID")
    return command


def _write_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()
