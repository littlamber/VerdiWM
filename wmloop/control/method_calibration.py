"""Execute declared implementation checks for an isolated generated method.

Calibration runs only the candidate's declared checks in a copied worktree. It
is an implementation receipt; it does not run the target evaluator, allocate a
GPU lease, or establish a model-quality effect.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import sys
import time
from collections.abc import Mapping
from pathlib import Path

from wmloop.execute.open_method_runtime import OpenRuntimeError, load_compilation, run_local_command, resolve_runtime_python
from wmloop.storage import atomic_write, canonical_bytes, checked_path


class MethodCalibrationError(RuntimeError):
    """A calibration request or generated bundle is invalid."""


def run_method_calibration(
    *, compilation_root: Path, output_root: Path, timeout_seconds: float = 300.0,
    runtime_python: Path | str = sys.executable,
) -> dict[str, object]:
    """Run all declared checks against a copied candidate workspace.

    A fresh output directory is required. Failed checks retain their logs and
    receipt so a later research decision can distinguish implementation failure
    from an ineffective method.
    """
    try:
        runtime_python = resolve_runtime_python(runtime_python)
        compilation = checked_path(
            compilation_root, code="METHOD_CALIBRATION_INPUT_INVALID", error=MethodCalibrationError
        )
        method, _execution, plan = load_compilation(compilation)
    except (MethodCalibrationError, OpenRuntimeError, OSError, ValueError, json.JSONDecodeError) as exc:
        if isinstance(exc, MethodCalibrationError):
            raise
        raise MethodCalibrationError(f"METHOD_CALIBRATION_INPUT_INVALID:{exc}") from exc
    if plan.get("state") != "declared_not_executed":
        raise MethodCalibrationError("METHOD_CALIBRATION_NOT_READY")
    tests = plan.get("tests")
    if not isinstance(tests, list) or not tests:
        raise MethodCalibrationError("METHOD_CALIBRATION_TESTS_MISSING")
    if type(timeout_seconds) not in (int, float) or not math.isfinite(float(timeout_seconds)) or timeout_seconds <= 0:
        raise MethodCalibrationError("METHOD_CALIBRATION_TIMEOUT_INVALID")
    destination = checked_path(
        output_root, code="METHOD_CALIBRATION_OUTPUT_INVALID", error=MethodCalibrationError
    )
    if destination.exists() or destination.is_symlink():
        raise MethodCalibrationError("METHOD_CALIBRATION_OUTPUT_EXISTS")
    destination.mkdir(mode=0o700, parents=True)
    workspace = destination / "workspace"
    started = time.monotonic()
    rows: list[dict[str, object]] = []
    try:
        shutil.copytree(compilation / "candidate-workspace", workspace, symlinks=False)
        names: set[str] = set()
        for index, raw in enumerate(tests):
            if not isinstance(raw, Mapping):
                raise MethodCalibrationError("METHOD_CALIBRATION_TEST_INVALID")
            name = raw.get("name")
            command = raw.get("command")
            if not isinstance(name, str) or not name.strip() or name in names:
                raise MethodCalibrationError("METHOD_CALIBRATION_TEST_INVALID")
            names.add(name)
            if (
                not isinstance(command, list)
                or not command
                or any(not isinstance(token, str) or not token or "\x00" in token for token in command)
                or any(re.search(r"[;&|`\n\r]", token) for token in command)
            ):
                raise MethodCalibrationError("METHOD_CALIBRATION_TEST_INVALID")
            # Tests must be relative commands; a test can import candidate files
            # from the copied worktree but may not target an arbitrary host path.
            for token in command:
                if token.startswith("/") or ".." in Path(token).parts:
                    raise MethodCalibrationError("METHOD_CALIBRATION_TEST_PATH_INVALID")
            result = run_local_command(
                command=list(command),
                cwd=workspace,
                output_root=destination / "checks" / f"{index:03d}",
                timeout_seconds=float(timeout_seconds),
                runtime_python=runtime_python,
                gpu_devices=(),
            )
            rows.append(
                {
                    "name": name,
                    "command": list(command),
                    "state": "passed" if result["state"] == "passed" else "failed",
                    "returncode": result.get("returncode"),
                    "error": result.get("error"),
                    "stdout_sha256": result.get("stdout_sha256"),
                    "stderr_sha256": result.get("stderr_sha256"),
                    "process_receipt": f"checks/{index:03d}/process.json",
                }
            )
    except (OpenRuntimeError, OSError) as exc:
        raise MethodCalibrationError(f"METHOD_CALIBRATION_EXECUTION_FAILED:{exc}") from exc
    passed = bool(rows) and all(row["state"] == "passed" for row in rows)
    receipt = {
        "schema_version": 1,
        "artifact_type": "verdiwm-method-calibration-receipt",
        "state": "passed" if passed else "failed",
        "method_id": method["method_id"],
        "overlay_id": _load_overlay_id(compilation),
        "checks": rows,
        "elapsed_seconds": time.monotonic() - started,
        "authority": {"gpu_scheduling": False, "promotion": False},
        "claim_boundary": (
            "Implementation checks only; passing declarations do not establish "
            "target-side effect or scientific improvement."
        ),
    }
    atomic_write(destination / "calibration.json", canonical_bytes(receipt) + b"\n")
    return receipt


def _load_overlay_id(compilation: Path) -> str:
    path = compilation / "manifest.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise MethodCalibrationError("METHOD_CALIBRATION_INPUT_INVALID") from exc
    if not isinstance(payload, Mapping) or not isinstance(payload.get("overlay_id"), str):
        raise MethodCalibrationError("METHOD_CALIBRATION_INPUT_INVALID")
    return str(payload["overlay_id"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compilation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--runtime-python", type=Path, default=Path(sys.executable))
    args = parser.parse_args()
    result = run_method_calibration(
        compilation_root=args.compilation,
        output_root=args.output,
        timeout_seconds=args.timeout,
        runtime_python=args.runtime_python,
    )
    print(
        json.dumps(
            {"state": result["state"], "method_id": result["method_id"], "output": str(args.output)}
        )
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
