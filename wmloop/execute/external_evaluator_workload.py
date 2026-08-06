"""Run a declared external evaluator and normalize its runtime evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence


class ExternalEvaluatorError(RuntimeError):
    """An external evaluator request violates the workload boundary."""


IdentityProvider = Callable[[int], Mapping[str, str]]


@dataclass(frozen=True)
class ExternalEvaluatorOptions:
    """Inputs for one already-admitted external evaluator process."""

    command: tuple[str, ...]
    scratch_root: Path
    working_directory: Path
    artifacts: tuple[str, ...]
    result_path: str = "result.json"
    artifact_index_path: str = "artifact-index.json"
    stdout_path: str = "external-stdout.log"
    stderr_path: str = "external-stderr.log"
    metrics_path: str | None = None
    metric_prefix: str = ""


def run_external_evaluator(
    options: ExternalEvaluatorOptions,
    *,
    environment: Mapping[str, str] | None = None,
    identity_provider: IdentityProvider | None = None,
) -> int:
    """Execute the command and write a standard auto-experiment result."""

    if not options.command:
        raise ExternalEvaluatorError("EXTERNAL_EVALUATOR_COMMAND_REQUIRED")
    if any("\x00" in token for token in options.command):
        raise ExternalEvaluatorError("EXTERNAL_EVALUATOR_COMMAND_INVALID")
    scratch = Path(options.scratch_root).expanduser().resolve()
    workdir = Path(options.working_directory).expanduser().resolve()
    if not workdir.is_dir() or workdir.is_symlink():
        raise ExternalEvaluatorError("EXTERNAL_EVALUATOR_WORKDIR_INVALID")
    scratch.mkdir(mode=0o700, parents=True, exist_ok=True)
    if scratch.is_symlink():
        raise ExternalEvaluatorError("EXTERNAL_EVALUATOR_SCRATCH_INVALID")

    result_path = _inside_scratch(scratch, options.result_path)
    index_path = _inside_scratch(scratch, options.artifact_index_path)
    stdout_path = _inside_scratch(scratch, options.stdout_path)
    stderr_path = _inside_scratch(scratch, options.stderr_path)
    reserved = {result_path, index_path, stdout_path, stderr_path}
    specs = tuple(
        _parse_artifact_spec(spec, scratch=scratch) for spec in options.artifacts
    )
    if len({target for _, target in specs}) != len(specs):
        raise ExternalEvaluatorError("EXTERNAL_EVALUATOR_ARTIFACT_TARGET_DUPLICATE")
    if any(target in reserved for _, target in specs):
        raise ExternalEvaluatorError("EXTERNAL_EVALUATOR_ARTIFACT_TARGET_RESERVED")

    child_environment = dict(environment or os.environ)
    child_environment["PYTHONDONTWRITEBYTECODE"] = "1"
    child_environment["PYTHONPYCACHEPREFIX"] = str(scratch / "pycache")
    with (
        stdout_path.open("wb") as stdout_handle,
        stderr_path.open("wb") as stderr_handle,
    ):
        try:
            completed = subprocess.run(
                options.command,
                cwd=workdir,
                env=child_environment,
                stdin=subprocess.DEVNULL,
                stdout=stdout_handle,
                stderr=stderr_handle,
                check=False,
            )
            exit_code = int(completed.returncode)
            launch_error = None
        except OSError as exc:
            exit_code = 127
            launch_error = f"{type(exc).__name__}:{exc}"
            stderr_handle.write(launch_error.encode("utf-8", errors="replace"))

    materialized: list[dict[str, object]] = []
    artifact_errors: list[str] = []
    for pattern, target in specs:
        matches = _artifact_matches(scratch, pattern)
        if len(matches) != 1:
            artifact_errors.append(f"{pattern}:expected_one_found_{len(matches)}")
            continue
        source = matches[0]
        if source != target:
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        materialized.append(
            {
                "source_pattern": pattern,
                "source_path": source.relative_to(scratch).as_posix(),
                "target_path": target.relative_to(scratch).as_posix(),
                "size_bytes": target.stat().st_size,
                "sha256": _sha256_file(target),
            }
        )

    external_metrics: dict[str, float] = {}
    if options.metrics_path is not None:
        try:
            metrics_artifact = _inside_scratch(scratch, options.metrics_path)
            if metrics_artifact not in {target for _, target in specs}:
                raise ExternalEvaluatorError(
                    "EXTERNAL_EVALUATOR_METRICS_ARTIFACT_NOT_DECLARED"
                )
            external_metrics = _load_external_metrics(
                metrics_artifact,
                prefix=options.metric_prefix,
            )
        except ExternalEvaluatorError as exc:
            artifact_errors.append(
                f"{options.metrics_path}:metrics_invalid:{str(exc)}"
            )

    physical_index = _required_nonnegative_int(
        child_environment, "VERDIWM_PHYSICAL_GPU_INDEX"
    )
    expected_uuid = child_environment.get("VERDIWM_PHYSICAL_GPU_UUID", "")
    observed_gpu = (identity_provider or _physical_gpu_identity)(physical_index)
    observed_uuid = str(observed_gpu.get("gpu_uuid", ""))
    identity_matches = bool(expected_uuid and observed_uuid == expected_uuid)
    runtime_ready = exit_code == 0 and not artifact_errors and identity_matches
    artifact_index = {
        "schema_version": 1,
        "artifact_type": "verdiwm-external-evaluator-artifact-index",
        "state": "ready" if not artifact_errors else "invalid",
        "command_sha256": _command_sha256(options.command),
        "artifacts": materialized,
        "errors": artifact_errors,
    }
    _write_json_atomic(index_path, artifact_index)
    result = {
        "schema_version": 1,
        "artifact_type": "verdiwm-auto-experiment-result",
        "state": "ready" if runtime_ready else "invalid",
        "device": {
            "type": "cuda",
            "physical_index": physical_index,
            "gpu_uuid": observed_uuid,
            "name": str(observed_gpu.get("name", "unknown")),
        },
        "metrics": {
            **external_metrics,
            "runtime_ready": 1.0 if runtime_ready else 0.0,
            "external_exit_code": float(exit_code),
            "artifact_count": float(len(materialized)),
            "artifact_error_count": float(len(artifact_errors)),
            "gpu_identity_match": 1.0 if identity_matches else 0.0,
        },
        "workload": {
            "command_sha256": artifact_index["command_sha256"],
            "artifact_index_path": index_path.relative_to(scratch).as_posix(),
            "stdout_path": stdout_path.relative_to(scratch).as_posix(),
            "stderr_path": stderr_path.relative_to(scratch).as_posix(),
            "launch_error": launch_error,
        },
    }
    _write_json_atomic(result_path, result)
    _relay_tail(stdout_path, stream=sys.stdout)
    _relay_tail(stderr_path, stream=sys.stderr)
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0 if runtime_ready else exit_code or 2


def _parse_artifact_spec(spec: str, *, scratch: Path) -> tuple[str, Path]:
    if "=" not in spec:
        raise ExternalEvaluatorError("EXTERNAL_EVALUATOR_ARTIFACT_SPEC_INVALID")
    pattern, target_value = spec.split("=", 1)
    if not pattern or Path(pattern).is_absolute() or ".." in Path(pattern).parts:
        raise ExternalEvaluatorError("EXTERNAL_EVALUATOR_ARTIFACT_PATTERN_INVALID")
    return pattern, _inside_scratch(scratch, target_value)


def _artifact_matches(scratch: Path, pattern: str) -> list[Path]:
    matches = []
    for candidate in scratch.glob(pattern):
        resolved = candidate.resolve()
        if (
            candidate.is_file()
            and not candidate.is_symlink()
            and _is_inside(scratch, resolved)
        ):
            matches.append(resolved)
    return sorted(set(matches), key=lambda path: path.relative_to(scratch).as_posix())


def _inside_scratch(scratch: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute() or not value or value.endswith("/"):
        raise ExternalEvaluatorError("EXTERNAL_EVALUATOR_PATH_INVALID")
    resolved = (scratch / candidate).resolve()
    if not _is_inside(scratch, resolved):
        raise ExternalEvaluatorError("EXTERNAL_EVALUATOR_PATH_ESCAPE")
    return resolved


_METRIC_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{0,79}\Z")
_RESERVED_METRICS = {
    "runtime_ready",
    "external_exit_code",
    "artifact_count",
    "artifact_error_count",
    "gpu_identity_match",
}


def _load_external_metrics(path: Path, *, prefix: str) -> dict[str, float]:
    if not path.is_file() or path.is_symlink():
        raise ExternalEvaluatorError("EXTERNAL_EVALUATOR_METRICS_MISSING")
    if prefix and _METRIC_NAME.fullmatch(prefix) is None:
        raise ExternalEvaluatorError("EXTERNAL_EVALUATOR_METRIC_PREFIX_INVALID")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExternalEvaluatorError("EXTERNAL_EVALUATOR_METRICS_INVALID") from exc
    if not isinstance(payload, Mapping):
        raise ExternalEvaluatorError("EXTERNAL_EVALUATOR_METRICS_INVALID")
    metrics = payload.get("metrics")
    if payload.get("state") != "ready" or not isinstance(metrics, Mapping):
        raise ExternalEvaluatorError("EXTERNAL_EVALUATOR_METRICS_INVALID")
    normalized: dict[str, float] = {}
    for raw_name, raw_value in metrics.items():
        name = f"{prefix}{raw_name}"
        if _METRIC_NAME.fullmatch(name) is None or name in _RESERVED_METRICS:
            raise ExternalEvaluatorError("EXTERNAL_EVALUATOR_METRIC_NAME_INVALID")
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise ExternalEvaluatorError("EXTERNAL_EVALUATOR_METRIC_VALUE_INVALID")
        value = float(raw_value)
        if not math.isfinite(value):
            raise ExternalEvaluatorError("EXTERNAL_EVALUATOR_METRIC_VALUE_INVALID")
        normalized[name] = value
    if not normalized:
        raise ExternalEvaluatorError("EXTERNAL_EVALUATOR_METRICS_EMPTY")
    return normalized


def _is_inside(root: Path, candidate: Path) -> bool:
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def _required_nonnegative_int(environment: Mapping[str, str], name: str) -> int:
    raw = environment.get(name)
    try:
        value = int(raw) if raw is not None else -1
    except ValueError as exc:
        raise ExternalEvaluatorError(
            f"EXTERNAL_EVALUATOR_ENVIRONMENT_INVALID:{name}"
        ) from exc
    if value < 0:
        raise ExternalEvaluatorError(f"EXTERNAL_EVALUATOR_ENVIRONMENT_INVALID:{name}")
    return value


def _physical_gpu_identity(index: int) -> Mapping[str, str]:
    try:
        completed = subprocess.run(
            (
                "nvidia-smi",
                "--id",
                str(index),
                "--query-gpu=uuid,name",
                "--format=csv,noheader,nounits",
            ),
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ExternalEvaluatorError(
            "EXTERNAL_EVALUATOR_GPU_IDENTITY_UNAVAILABLE"
        ) from exc
    fields = [field.strip() for field in completed.stdout.strip().split(",", 1)]
    if completed.returncode != 0 or len(fields) != 2 or not fields[0]:
        raise ExternalEvaluatorError("EXTERNAL_EVALUATOR_GPU_IDENTITY_UNAVAILABLE")
    return {"gpu_uuid": fields[0], "name": fields[1]}


def _command_sha256(command: Sequence[str]) -> str:
    return hashlib.sha256(
        json.dumps(list(command), ensure_ascii=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(
                payload,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def _relay_tail(path: Path, *, stream: object, limit: int = 32_768) -> None:
    with path.open("rb") as handle:
        size = path.stat().st_size
        if size > limit:
            handle.seek(size - limit)
        payload = handle.read().decode("utf-8", errors="replace")
    if payload:
        print(payload, file=stream, end="" if payload.endswith("\n") else "\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", action="append", default=[])
    parser.add_argument("--result-path", default="result.json")
    parser.add_argument("--artifact-index-path", default="artifact-index.json")
    parser.add_argument("--stdout-path", default="external-stdout.log")
    parser.add_argument("--stderr-path", default="external-stderr.log")
    parser.add_argument("--metrics-artifact")
    parser.add_argument("--metric-prefix", default="")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = tuple(args.command[1:] if args.command[:1] == ["--"] else args.command)
    scratch_value = os.environ.get("VERDIWM_TRIAL_SCRATCH")
    if not scratch_value:
        print("EXTERNAL_EVALUATOR_SCRATCH_REQUIRED", file=sys.stderr)
        return 2
    try:
        return run_external_evaluator(
            ExternalEvaluatorOptions(
                command=command,
                scratch_root=Path(scratch_value),
                working_directory=Path.cwd(),
                artifacts=tuple(args.artifact),
                result_path=args.result_path,
                artifact_index_path=args.artifact_index_path,
                stdout_path=args.stdout_path,
                stderr_path=args.stderr_path,
                metrics_path=args.metrics_artifact,
                metric_prefix=args.metric_prefix,
            )
        )
    except ExternalEvaluatorError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
