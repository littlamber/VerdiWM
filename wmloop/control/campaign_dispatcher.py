"""Consume Campaign API dispatch manifests through existing VerdiWM daemons."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from wmloop.control.campaign_api import CampaignAPIError, CampaignStore


class CampaignDispatchError(RuntimeError):
    """A dispatch manifest could not be admitted or settled."""


Runner = Callable[[Mapping[str, Any]], Mapping[str, Any]]


@dataclass(frozen=True)
class DispatcherOptions:
    state_root: Path
    poll_seconds: float = 10.0
    max_cycles: int = 1
    max_parallel: int = 1


def run_dispatcher(
    options: DispatcherOptions,
    *,
    runner: Runner | None = None,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    if options.max_cycles < 1 or options.max_parallel != 1 or options.poll_seconds < 0:
        raise CampaignDispatchError("DISPATCHER_OPTIONS_INVALID")
    root = Path(options.state_root).expanduser().resolve()
    store = CampaignStore(root)
    dispatch_root = store.root / "dispatch"
    pending = dispatch_root / "pending"
    running = dispatch_root / "running"
    completed = dispatch_root / "completed"
    failed = dispatch_root / "failed"
    for path in (pending, running, completed, failed):
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
    execute = runner or _run_subprocess
    settled: list[str] = []
    failed_ids: list[str] = []
    for cycle in range(options.max_cycles):
        paths = sorted(path for path in pending.glob("*.json") if path.is_file() and not path.is_symlink())
        if not paths:
            if cycle + 1 < options.max_cycles:
                sleeper(options.poll_seconds)
            continue
        for source in paths[: options.max_parallel]:
            campaign_id = source.stem
            active = running / source.name
            try:
                source.replace(active)
                dispatch = _load_dispatch(active)
                store.record_dispatch_result(campaign_id, status="running")
                result = dict(execute(dispatch["execution"]))
                store.record_dispatch_result(campaign_id, status="completed", result=result)
                dispatch["state"] = "completed"
                dispatch["result"] = result
                _write_json(completed / source.name, dispatch)
                active.unlink(missing_ok=True)
                settled.append(campaign_id)
            except Exception as exc:
                error = {"type": type(exc).__name__, "message": str(exc)[:500]}
                try:
                    store.record_dispatch_result(campaign_id, status="failed", error=error)
                except CampaignAPIError:
                    pass
                try:
                    dispatch = _load_dispatch(active)
                except Exception:
                    dispatch = {"campaign_id": campaign_id, "schema_version": 1}
                dispatch["state"] = "failed"
                dispatch["error"] = error
                _write_json(failed / source.name, dispatch)
                active.unlink(missing_ok=True)
                failed_ids.append(campaign_id)
    return {
        "schema_version": 1,
        "artifact_type": "verdiwm-campaign-dispatcher-manifest",
        "state": "completed",
        "settled_campaign_ids": settled,
        "failed_campaign_ids": failed_ids,
        "pending_count": len(list(pending.glob("*.json"))),
    }


def _load_dispatch(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignDispatchError("DISPATCH_MANIFEST_INVALID") from exc
    if not isinstance(payload, dict) or payload.get("artifact_type") != "verdiwm-campaign-dispatch":
        raise CampaignDispatchError("DISPATCH_MANIFEST_CONTRACT_INVALID")
    if payload.get("state") != "pending" or not isinstance(payload.get("execution"), dict):
        raise CampaignDispatchError("DISPATCH_MANIFEST_NOT_PENDING")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _run_subprocess(execution: Mapping[str, Any]) -> Mapping[str, Any]:
    kind = execution.get("kind")
    if kind == "evolution":
        command = [
            sys.executable,
            "-m",
            "wmloop.execute.evolution_daemon",
            str(execution["repo_root"]),
            "--output-root",
            str(execution["output_root"]),
            "--state-root",
            str(execution["state_root"]),
            "--evaluator-contract",
            str(execution["evaluator_contract"]),
            "--total-budget-gpu-hours",
            str(execution["total_budget_gpu_hours"]),
        ]
        if execution.get("probe_contract"):
            command.extend(["--probe-contract", str(execution["probe_contract"])])
        if execution.get("max_iterations") is not None:
            command.extend(["--max-iterations", str(execution["max_iterations"])])
    elif kind == "campaign_queue":
        command = [
            sys.executable,
            "-m",
            "wmloop.execute.campaign_daemon",
            "--output-root",
            str(execution["output_root"]),
            "--workspace-root",
            str(execution["workspace_root"]),
            "--archive-db",
            str(execution["archive_db"]),
            "--cas-root",
            str(execution["cas_root"]),
        ]
        for queue in execution["queue_paths"]:
            command.extend(["--queue", str(queue)])
    else:
        raise CampaignDispatchError("EXECUTION_KIND_INVALID")
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    result = {
        "returncode": completed.returncode,
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-4000:],
        "command": command,
    }
    if completed.returncode != 0:
        raise CampaignDispatchError(f"DISPATCH_PROCESS_FAILED:{completed.returncode}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--max-cycles", type=int, default=1)
    args = parser.parse_args()
    print(
        json.dumps(
            run_dispatcher(
                DispatcherOptions(
                    state_root=args.state_root,
                    poll_seconds=args.poll_seconds,
                    max_cycles=args.max_cycles,
                )
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
