"""M0 baseline launch guard smoke for checkpoint-step fail-closed behavior."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Mapping

from wmloop.archive.store import ArchiveStore, ContentAddressedStore
from wmloop.evaluate.launch import BaselineLaunchError, create_m0_baseline_launch


class LaunchGuardSmokeError(RuntimeError):
    """The launch guard smoke artifact could not be generated."""


def run_launch_guard_smoke(
    *,
    data_root: Path,
    dataset_freeze_path: Path,
    heldout_protocol_path: Path,
    checkpoint_root: Path,
    runtime_python: Path,
    output_root: Path,
    expected_checkpoint_step: int,
    expected_error_fragment: str = "BASELINE_LAUNCH_CHECKPOINT_STEP_MISMATCH",
    archive_db: Path | None = None,
    cas_root: Path | None = None,
) -> dict[str, object]:
    """Attempt strict M0 launch creation and require the expected guard failure."""

    if not isinstance(expected_checkpoint_step, int) or isinstance(expected_checkpoint_step, bool) or expected_checkpoint_step < 1:
        raise LaunchGuardSmokeError("LAUNCH_GUARD_SMOKE_EXPECTED_STEP_INVALID")
    if not expected_error_fragment:
        raise LaunchGuardSmokeError("LAUNCH_GUARD_SMOKE_EXPECTED_ERROR_INVALID")
    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise LaunchGuardSmokeError("LAUNCH_GUARD_SMOKE_OUTPUT_EXISTS")
    archive = ArchiveStore(archive_db) if archive_db is not None else None
    cas_storage_root = cas_root if cas_root is not None else (Path(archive_db).resolve().parent if archive_db is not None else destination.parent)
    cas = ContentAddressedStore(Path(cas_storage_root))
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    attempted_run_root = temporary / "attempted-launch-plan"
    report: dict[str, object]
    try:
        temporary.mkdir(mode=0o700, parents=True)
        caught_error: str | None = None
        try:
            create_m0_baseline_launch(
                data_root=data_root,
                dataset_freeze_path=dataset_freeze_path,
                heldout_protocol_path=heldout_protocol_path,
                checkpoint_root=checkpoint_root,
                runtime_python=runtime_python,
                run_root=attempted_run_root,
                expected_checkpoint_step=expected_checkpoint_step,
            )
        except BaselineLaunchError as exc:
            caught_error = str(exc)
        materialized = attempted_run_root.exists() or attempted_run_root.is_symlink()
        expected_error_seen = caught_error is not None and expected_error_fragment in caught_error
        strict_pass = expected_error_seen and not materialized
        if strict_pass:
            state = "blocked_as_expected"
        elif caught_error is None:
            state = "unexpected_launch_materialized"
        else:
            state = "unexpected_error"
        report = {
            "schema_version": 1,
            "artifact_type": "acwm-m0-baseline-launch-guard-smoke-report",
            "state": state,
            "strict_launch_guard_pass": strict_pass,
            "expected_checkpoint_step": expected_checkpoint_step,
            "expected_error_fragment": expected_error_fragment,
            "observed_error": caught_error,
            "expected_error_seen": expected_error_seen,
            "materialized_launch_plan": materialized,
            "gpu_execution_started": False,
            "attempted_run_root": str(destination / "attempted-launch-plan"),
            "inputs": {
                "data_root": str(Path(data_root).resolve()),
                "dataset_freeze_path": str(Path(dataset_freeze_path).resolve()),
                "heldout_protocol_path": str(Path(heldout_protocol_path).resolve()),
                "checkpoint_root": str(Path(checkpoint_root).resolve()),
                "runtime_python": str(Path(runtime_python).resolve()),
            },
            "limitations": [
                "This smoke exercises launch-plan creation only; it does not run upstream eval.py.",
                "gpu_execution_started is false because create_m0_baseline_launch materializes metadata before any evaluator command can run.",
            ],
        }
        manifest = _write_report_bundle(report=report, temporary_root=temporary, output_root=destination, cas=cas, archive=archive)
        os.replace(temporary, destination)
        return manifest
    except Exception:
        if temporary.is_symlink():
            temporary.unlink()
        elif temporary.exists():
            shutil.rmtree(temporary)
        raise


def _write_report_bundle(
    *,
    report: Mapping[str, object],
    temporary_root: Path,
    output_root: Path,
    cas: ContentAddressedStore,
    archive: ArchiveStore | None,
) -> dict[str, object]:
    report_bytes = _canonical_json_bytes(report)
    markdown_bytes = _render_markdown(report).encode("utf-8")
    _write_bytes_atomic(temporary_root / "launch-guard-smoke.json", report_bytes)
    _write_bytes_atomic(temporary_root / "launch-guard-smoke.md", markdown_bytes)
    report_ref = cas.put_bytes(report_bytes, media_type="application/json").uri
    markdown_ref = cas.put_bytes(markdown_bytes, media_type="text/markdown").uri
    if archive is not None:
        archive.record_artifact_reference(report_ref)
        archive.record_artifact_reference(markdown_ref)
    manifest = {
        "schema_version": 1,
        "artifact_type": "acwm-m0-baseline-launch-guard-smoke-manifest",
        "state": report["state"],
        "strict_launch_guard_pass": report["strict_launch_guard_pass"],
        "expected_checkpoint_step": report["expected_checkpoint_step"],
        "observed_error": report["observed_error"],
        "materialized_launch_plan": report["materialized_launch_plan"],
        "gpu_execution_started": report["gpu_execution_started"],
        "report_path": str(output_root / "launch-guard-smoke.json"),
        "markdown_path": str(output_root / "launch-guard-smoke.md"),
        "cas_refs": {
            "launch_guard_smoke_json": report_ref,
            "launch_guard_smoke_markdown": markdown_ref,
        },
    }
    _write_bytes_atomic(temporary_root / "manifest.json", _canonical_json_bytes(manifest))
    return manifest


def _render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# M0 Baseline Launch Guard Smoke",
        "",
        f"State: `{report['state']}`",
        f"Strict launch guard pass: `{report['strict_launch_guard_pass']}`",
        f"Expected checkpoint step: `{report['expected_checkpoint_step']}`",
        f"Observed error: `{report['observed_error']}`",
        f"Materialized launch plan: `{report['materialized_launch_plan']}`",
        f"GPU execution started: `{report['gpu_execution_started']}`",
        "",
        "## Inputs",
        "",
        "| Field | Value |",
        "|:--|:--|",
    ]
    for key, value in report["inputs"].items():
        lines.append(f"| {key} | `{value}` |")
    lines.extend(["", "## Limitations", ""])
    for item in report["limitations"]:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run M0 launch guard smoke")
    run.add_argument("--data-root", type=Path, required=True)
    run.add_argument("--dataset-freeze", type=Path, required=True)
    run.add_argument("--heldout-protocol", type=Path, required=True)
    run.add_argument("--checkpoint-root", type=Path, required=True)
    run.add_argument("--runtime-python", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--expected-checkpoint-step", type=int, required=True)
    run.add_argument("--expected-error-fragment", default="BASELINE_LAUNCH_CHECKPOINT_STEP_MISMATCH")
    run.add_argument("--archive-db", type=Path)
    run.add_argument("--cas-root", type=Path)
    args = parser.parse_args(argv)
    if args.command == "run":
        manifest = run_launch_guard_smoke(
            data_root=args.data_root,
            dataset_freeze_path=args.dataset_freeze,
            heldout_protocol_path=args.heldout_protocol,
            checkpoint_root=args.checkpoint_root,
            runtime_python=args.runtime_python,
            output_root=args.output_root,
            expected_checkpoint_step=args.expected_checkpoint_step,
            expected_error_fragment=args.expected_error_fragment,
            archive_db=args.archive_db,
            cas_root=args.cas_root,
        )
        print(json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
