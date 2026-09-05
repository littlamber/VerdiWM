"""Portable description of one long-running experiment job.

The control plane stores a :class:`JobSpec` separately from the worker state.
That separation makes a job resumable without requiring the submitting CLI
process (or a notebook/frontend) to stay alive.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


class JobSpecError(ValueError):
    """A job specification is incomplete or unsafe to execute."""


def _positive_int(value: object, code: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise JobSpecError(code)
    return value


@dataclass(frozen=True)
class JobSpec:
    """Immutable command and resource binding for one background job."""

    command: tuple[str, ...]
    cwd: Path
    job_root: Path
    environment: Mapping[str, str] = field(default_factory=dict)
    output_root: Path | None = None
    timeout_seconds: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> "JobSpec":
        if not self.command or any(not isinstance(item, str) or not item for item in self.command):
            raise JobSpecError("JOB_COMMAND_INVALID")
        cwd = Path(self.cwd).expanduser().resolve()
        if not cwd.is_dir() or cwd.is_symlink():
            raise JobSpecError("JOB_CWD_INVALID")
        root = Path(self.job_root).expanduser().resolve()
        if root == cwd:
            raise JobSpecError("JOB_ROOT_EQUALS_CWD_INVALID")
        if self.output_root is not None:
            output = Path(self.output_root).expanduser().resolve()
            if output == root or root.is_relative_to(output) or output.is_relative_to(root):
                raise JobSpecError("JOB_OUTPUT_ROOT_OVERLAP_INVALID")
        if self.timeout_seconds is not None and (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(float(self.timeout_seconds))
            or float(self.timeout_seconds) <= 0
        ):
            raise JobSpecError("JOB_TIMEOUT_INVALID")
        for key, value in self.environment.items():
            if not isinstance(key, str) or not key or "=" in key:
                raise JobSpecError("JOB_ENVIRONMENT_KEY_INVALID")
            if not isinstance(value, str):
                raise JobSpecError("JOB_ENVIRONMENT_VALUE_INVALID")
        return JobSpec(
            command=tuple(self.command),
            cwd=cwd,
            job_root=root,
            environment=dict(self.environment),
            output_root=(Path(self.output_root).expanduser().resolve() if self.output_root else None),
            timeout_seconds=(float(self.timeout_seconds) if self.timeout_seconds is not None else None),
            metadata=dict(self.metadata),
        )

    def to_document(self) -> dict[str, object]:
        spec = self.validate()
        return {
            "schema_version": 1,
            "artifact_type": "verdiwm-background-job-spec",
            "command": list(spec.command),
            "cwd": str(spec.cwd),
            "job_root": str(spec.job_root),
            "output_root": str(spec.output_root) if spec.output_root else None,
            "timeout_seconds": spec.timeout_seconds,
            "metadata": dict(spec.metadata),
            "environment_keys": sorted(spec.environment),
            "environment_sha256": environment_digest(spec.environment),
        }

    @classmethod
    def from_document(
        cls,
        document: Mapping[str, object],
        *,
        environment: Mapping[str, str],
    ) -> "JobSpec":
        if document.get("artifact_type") != "verdiwm-background-job-spec":
            raise JobSpecError("JOB_SPEC_ARTIFACT_TYPE_INVALID")
        command = document.get("command")
        if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
            raise JobSpecError("JOB_COMMAND_INVALID")
        metadata = document.get("metadata", {})
        if not isinstance(metadata, dict):
            raise JobSpecError("JOB_METADATA_INVALID")
        output = document.get("output_root")
        timeout = document.get("timeout_seconds")
        return cls(
            command=tuple(command),
            cwd=Path(str(document.get("cwd", ""))),
            job_root=Path(str(document.get("job_root", ""))),
            environment=dict(environment),
            output_root=Path(str(output)) if output else None,
            timeout_seconds=float(timeout) if timeout is not None else None,
            metadata=metadata,
        ).validate()


def environment_digest(environment: Mapping[str, str]) -> str:
    """Hash environment content without placing values in public receipts."""

    payload = json.dumps(
        sorted((str(key), str(value)) for key, value in environment.items()),
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def effective_batch_size(
    per_device_batch_size: int,
    gradient_accumulation_steps: int = 1,
    world_size: int = 1,
) -> int:
    """Return the optimizer's global batch size for explicit training receipts."""

    batch = _positive_int(per_device_batch_size, "PER_DEVICE_BATCH_SIZE_INVALID")
    accumulation = _positive_int(
        gradient_accumulation_steps, "GRADIENT_ACCUMULATION_STEPS_INVALID"
    )
    workers = _positive_int(world_size, "WORLD_SIZE_INVALID")
    return batch * accumulation * workers

