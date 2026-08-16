#!/usr/bin/env python3
"""Wait for CBMA training, then run and aggregate the held-out screen."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


SOURCE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_ROOT))

from scripts.freeze_ctrl_world_cbma_heldout_plan import freeze


ROOT = Path(__file__).resolve().parents[1]


class CBMAHeldoutPipelineError(RuntimeError):
    """The post-training held-out pipeline failed closed."""


def _load(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise CBMAHeldoutPipelineError(f"CBMA_HELDOUT_PIPELINE_JSON_INVALID:{path}")
    return payload


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _run_logged(command: list[str], *, log_path: Path) -> int:
    with log_path.open("w", encoding="utf-8") as log:
        return subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False).returncode


def run_worker(record_path: Path) -> int:
    record_path = record_path.resolve(strict=True)
    record = dict(_load(record_path))
    sequence_path = Path(str(record["sequence_path"])).resolve(strict=True)
    record.update({"state": "waiting_for_training", "started_at": _utc_now(), "worker_pid": os.getpid()})
    _write(record_path, record)
    while True:
        sequence = _load(sequence_path)
        state = sequence.get("state")
        if state == "completed":
            break
        if state == "failed":
            record.update({"state": "failed", "failure": "training_sequence_failed", "completed_at": _utc_now()})
            _write(record_path, record)
            return 1
        time.sleep(float(record["poll_seconds"]))

    plan_path = Path(str(record["heldout_plan_path"]))
    evaluation_root = Path(str(record["evaluation_output_root"]))
    manifest = freeze(
        training_plan_path=Path(str(record["training_plan_path"])),
        sequence_path=sequence_path,
        output_path=plan_path,
        evaluation_output_root=evaluation_root,
    )
    record.update({"state": "running_action_sensitivity", "heldout_plan": manifest})
    _write(record_path, record)
    runtime = os.environ.get("VERDIWM_CTRL_WORLD_PYTHON", sys.executable)
    launcher = ROOT / "scripts" / "run_ctrl_world_fshc_heldout_evaluation.py"
    aggregator = ROOT / "scripts" / "aggregate_ctrl_world_fshc_heldout_evaluation.py"
    for phase in ("action_sensitivity", "routing"):
        record["state"] = f"running_{phase}"
        _write(record_path, record)
        return_code = _run_logged(
            [runtime, str(launcher), "--plan", str(plan_path), "--phase", phase],
            log_path=record_path.parent / f"heldout-{phase}-launcher.log",
        )
        record.setdefault("phases", {})[phase] = {"return_code": return_code, "completed_at": _utc_now()}
        _write(record_path, record)
        if return_code != 0:
            record.update({"state": "failed", "failed_phase": phase, "completed_at": _utc_now()})
            _write(record_path, record)
            return return_code
    report_path = evaluation_root / "heldout-evaluation-report.json"
    record["state"] = "aggregating"
    _write(record_path, record)
    return_code = _run_logged(
        [runtime, str(aggregator), "--plan", str(plan_path), "--output", str(report_path)],
        log_path=record_path.parent / "heldout-aggregation.log",
    )
    record.update(
        {
            "state": "completed" if return_code == 0 else "failed",
            "aggregation_return_code": return_code,
            "report_path": str(report_path),
            "completed_at": _utc_now(),
        }
    )
    _write(record_path, record)
    return return_code


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-plan", type=Path)
    parser.add_argument("--sequence", type=Path)
    parser.add_argument("--heldout-plan", type=Path)
    parser.add_argument("--evaluation-output-root", type=Path)
    parser.add_argument("--status", type=Path)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker-record", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker_record is not None:
        return run_worker(args.worker_record)
    required = (args.training_plan, args.sequence, args.heldout_plan, args.evaluation_output_root, args.status)
    if any(value is None for value in required):
        parser.error("--training-plan, --sequence, --heldout-plan, --evaluation-output-root and --status are required")
    status_path = args.status.resolve()
    if (status_path.exists() or status_path.is_symlink()) and not args.resume:
        raise CBMAHeldoutPipelineError("CBMA_HELDOUT_PIPELINE_STATUS_EXISTS")
    if args.resume:
        record = dict(_load(status_path.resolve(strict=True)))
        if record.get("state") not in ("worker_starting", "waiting_for_training"):
            raise CBMAHeldoutPipelineError("CBMA_HELDOUT_PIPELINE_RESUME_STATE_INVALID")
    else:
        status_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        record = {
            "schema_version": 1,
            "artifact_type": "ctrl-world-cbma-heldout-pipeline",
            "state": "worker_starting" if args.detach else "starting",
            "training_plan_path": str(args.training_plan.resolve(strict=True)),
            "sequence_path": str(args.sequence.resolve(strict=True)),
            "heldout_plan_path": str(args.heldout_plan.resolve()),
            "evaluation_output_root": str(args.evaluation_output_root.resolve()),
            "poll_seconds": args.poll_seconds,
            "created_at": _utc_now(),
        }
        _write(status_path, record)
    if not args.detach:
        return run_worker(status_path)
    worker = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "--worker-record", str(status_path)],
        cwd=ROOT,
        env=os.environ.copy(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    record["worker_pid"] = worker.pid
    _write(status_path, record)
    print(json.dumps(record, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (CBMAHeldoutPipelineError, OSError, subprocess.SubprocessError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
