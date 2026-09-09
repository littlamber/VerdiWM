"""Execute declared implementation checks for an isolated generated method."""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
import argparse
import os
import sys
from collections.abc import Mapping
from pathlib import Path

from wmloop.storage import atomic_write, canonical_bytes, checked_path


class MethodCalibrationError(RuntimeError):
    """A calibration request or generated bundle is invalid."""


def run_method_calibration(*, compilation_root: Path, output_root: Path, timeout_seconds: float = 300.0) -> dict[str, object]:
    """Run the proposal's declared tests in its copied candidate workspace.

    This is an implementation calibration receipt, not an effect result. Tests
    run without GPU leases, evaluator access, network credentials, or source
    tree access; target-side training/confirmation remains a later stage.
    """
    compilation = checked_path(compilation_root, code="METHOD_CALIBRATION_INPUT_INVALID", error=MethodCalibrationError)
    manifest_path = compilation / "manifest.json"
    plan_path = compilation / "implementation-check-plan.json"
    workspace = compilation / "candidate-workspace"
    for path in (manifest_path, plan_path):
        if not path.is_file() or path.is_symlink():
            raise MethodCalibrationError("METHOD_CALIBRATION_INPUT_INVALID")
    manifest = _load(manifest_path)
    plan = _load(plan_path)
    if manifest.get("state") != "ready_for_calibration" or plan.get("state") != "declared_not_executed":
        raise MethodCalibrationError("METHOD_CALIBRATION_NOT_READY")
    tests = plan.get("tests")
    if not isinstance(tests, list) or not tests:
        raise MethodCalibrationError("METHOD_CALIBRATION_TESTS_MISSING")
    if not workspace.is_dir() or workspace.is_symlink():
        raise MethodCalibrationError("METHOD_CALIBRATION_WORKSPACE_INVALID")
    if not isinstance(timeout_seconds, (int, float)) or timeout_seconds <= 0:
        raise MethodCalibrationError("METHOD_CALIBRATION_TIMEOUT_INVALID")
    destination = checked_path(output_root, code="METHOD_CALIBRATION_OUTPUT_INVALID", error=MethodCalibrationError)
    if destination.exists():
        raise MethodCalibrationError("METHOD_CALIBRATION_OUTPUT_EXISTS")
    destination.mkdir(mode=0o700, parents=True)
    started = time.monotonic()
    rows: list[dict[str, object]] = []
    try:
        for test in tests:
            if not isinstance(test, Mapping) or not isinstance(test.get("name"), str) or not isinstance(test.get("command"), list):
                raise MethodCalibrationError("METHOD_CALIBRATION_TEST_INVALID")
            command = [str(token) for token in test["command"]]
            if not command or any(not token or "\x00" in token for token in command):
                raise MethodCalibrationError("METHOD_CALIBRATION_TEST_INVALID")
            try:
                completed = subprocess.run(command, cwd=workspace, env={"PATH":os.path.dirname(sys.executable)+":/usr/local/bin:/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE":"1"}, capture_output=True, text=True, timeout=float(timeout_seconds), check=False)
                state = "passed" if completed.returncode == 0 else "failed"
                row = {"name": test["name"], "command": command, "state": state, "returncode": completed.returncode, "stdout_sha256": _sha256(completed.stdout), "stderr_sha256": _sha256(completed.stderr)}
            except subprocess.TimeoutExpired:
                row = {"name": test["name"], "command": command, "state":"failed", "returncode":None, "timeout":True}
            rows.append(row)
    except BaseException:
        if destination.exists():
            import shutil
            shutil.rmtree(destination)
        raise
    passed = all(row["state"] == "passed" for row in rows)
    receipt = {"schema_version":1, "artifact_type":"verdiwm-method-calibration-receipt", "state":"passed" if passed else "failed", "method_id":manifest["method_id"], "overlay_id":manifest["overlay_id"], "checks":rows, "elapsed_seconds":time.monotonic()-started, "authority":{"gpu_scheduling":False,"promotion":False}, "claim_boundary":"Implementation checks only; passing declarations do not establish target-side effect or scientific improvement."}
    atomic_write(destination / "calibration.json", canonical_bytes(receipt) + b"\n")
    return receipt


def _load(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise MethodCalibrationError("METHOD_CALIBRATION_INPUT_INVALID") from exc
    if not isinstance(value, dict):
        raise MethodCalibrationError("METHOD_CALIBRATION_INPUT_INVALID")
    return value


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compilation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timeout', type=float, default=300.0)
    args = parser.parse_args()
    result = run_method_calibration(compilation_root=args.compilation, output_root=args.output, timeout_seconds=args.timeout)
    print(json.dumps({'state': result['state'], 'method_id': result['method_id'], 'output': str(args.output)}))
