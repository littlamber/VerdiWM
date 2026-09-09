"""Common stage implementation for the Ctrl-World workflow."""

from __future__ import annotations
import datetime as dt
import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from wmloop.control.model_portrait import ModelPortraitError, validate_model_portrait


class AutonomousTransferWorkflowError(RuntimeError):
    """A strict workflow stage could not preserve its declared boundary."""


@dataclass(frozen=True)
class StageResult:
    """Terminal result for one durable controller stage attempt."""

    state: str
    outcome: str
    payload: dict[str, object]
    receipt_path: Path | None = None


def _load_bound_portrait(
    gate: Mapping[str, object], *, project_root: Path
) -> dict[str, object]:
    path = _require_file(
        Path(str(gate.get("model_portrait") or "")),
        "AUTONOMOUS_MODEL_PORTRAIT_INVALID",
    )
    expected = gate.get("model_portrait_sha256")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if not isinstance(expected, str) or actual != expected:
        raise AutonomousTransferWorkflowError("AUTONOMOUS_MODEL_PORTRAIT_HASH_MISMATCH")
    portrait = _load(path, "AUTONOMOUS_MODEL_PORTRAIT_INVALID")
    try:
        validate_model_portrait(portrait, root=project_root)
    except ModelPortraitError as exc:
        raise AutonomousTransferWorkflowError(
            f"AUTONOMOUS_MODEL_PORTRAIT_INVALID:{exc}"
        ) from exc
    return portrait


def _load_active_portrait(
    config: Mapping[str, object], *, project_root: Path, state_root: Path | None
) -> dict[str, object]:
    """Load the immutable onboarding portrait or the durable derived portrait."""

    gate = config.get("portrait_gate")
    if not isinstance(gate, Mapping):
        raise AutonomousTransferWorkflowError("AUTONOMOUS_PORTRAIT_GATE_REQUIRED")
    candidate: Path | None = None
    expected: str | None = None
    if state_root is not None:
        active = Path(state_root).expanduser().resolve() / "active-portrait.json"
        if active.is_file() and not active.is_symlink():
            candidate = active
    closed_loop = config.get("closed_loop")
    if candidate is None and isinstance(closed_loop, Mapping):
        configured = closed_loop.get("active_portrait_path")
        if isinstance(configured, str) and configured:
            path = Path(configured).expanduser().resolve()
            if path.is_file() and not path.is_symlink():
                candidate = path
                value = closed_loop.get("active_portrait_sha256")
                expected = value if isinstance(value, str) else None
    if candidate is None:
        return _load_bound_portrait(gate, project_root=project_root)
    actual = hashlib.sha256(candidate.read_bytes()).hexdigest()
    if expected is not None and actual != expected:
        raise AutonomousTransferWorkflowError("AUTONOMOUS_ACTIVE_PORTRAIT_HASH_MISMATCH")
    portrait = _load(candidate, "AUTONOMOUS_MODEL_PORTRAIT_INVALID")
    try:
        validate_model_portrait(portrait, root=project_root)
    except ModelPortraitError as exc:
        raise AutonomousTransferWorkflowError(
            f"AUTONOMOUS_MODEL_PORTRAIT_INVALID:{exc}"
        ) from exc
    return portrait


def _canonical_any_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                ensure_ascii=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise AutonomousTransferWorkflowError(
            "AUTONOMOUS_OBSERVATION_PAYLOAD_INVALID"
        ) from exc


def _state_root_from_attempt(attempt_root: Path) -> Path | None:
    for parent in Path(attempt_root).resolve().parents:
        if (parent / "controller.db").is_file():
            return parent
    return None


def _paths(config: Mapping[str, object]) -> dict[str, Path]:
    raw = config.get("paths")
    if not isinstance(raw, Mapping):
        raise AutonomousTransferWorkflowError("AUTONOMOUS_PATHS_INVALID")
    return {
        str(name): Path(str(value)).expanduser().resolve() for name, value in raw.items()
    }


def _attempt_number(path: Path) -> int:
    try:
        return int(path.name.rsplit("-", 1)[1])
    except (IndexError, ValueError) as exc:
        raise AutonomousTransferWorkflowError("AUTONOMOUS_ATTEMPT_ROOT_INVALID") from exc


def _load(path: Path, code: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AutonomousTransferWorkflowError(code) from exc
    if not isinstance(payload, dict):
        raise AutonomousTransferWorkflowError(code)
    return payload


def _require_file(path: Path, code: str) -> Path:
    raw = Path(path).expanduser()
    if raw.is_symlink():
        raise AutonomousTransferWorkflowError(code)
    resolved = raw.resolve()
    if not resolved.is_file():
        raise AutonomousTransferWorkflowError(code)
    return resolved


def _require_directory(path: Path, code: str) -> Path:
    raw = Path(path).expanduser()
    if raw.is_symlink():
        raise AutonomousTransferWorkflowError(code)
    resolved = raw.resolve()
    if not resolved.is_dir():
        raise AutonomousTransferWorkflowError(code)
    return resolved


def _write_json_idempotent(path: Path, payload: Mapping[str, object]) -> None:
    _write_bytes_idempotent(path, _canonical_bytes(payload))


def _replace_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    encoded = _canonical_bytes(payload)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.exists() and (path.is_symlink() or not path.is_file()):
        raise AutonomousTransferWorkflowError("AUTONOMOUS_ACTIVE_PORTRAIT_INVALID")
    if path.is_file() and path.read_bytes() == encoded:
        return
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _write_bytes_idempotent(path: Path, payload: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise AutonomousTransferWorkflowError("AUTONOMOUS_IMMUTABLE_WRITE_CONFLICT")
        return
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(payload)
    os.replace(temporary, path)


def _canonical_bytes(payload: Mapping[str, object]) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")
