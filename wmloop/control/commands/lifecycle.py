"""Campaign status, cancellation, reproduction, and detached-job commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wmloop.control.campaign_api import CampaignStore
from wmloop.control.campaign_dispatcher import DispatcherOptions, run_dispatcher
from wmloop.control.commands.inputs import campaign_store
from wmloop.control.commands.io import emit_json, emit_text
from wmloop.execute.job_supervisor import (
    cancel_job,
    load_status,
    resume_job,
    submit_job,
    tail_job,
)
from wmloop.experiments.job_spec import JobSpec


def _store(state_root: Path | None) -> CampaignStore:
    return campaign_store(state_root)


def _dispatch(
    store: CampaignStore, campaign_id: str, max_parallel: int
) -> dict[str, object]:
    return run_dispatcher(
        DispatcherOptions(
            state_root=store.root,
            max_cycles=1,
            max_parallel=max_parallel,
            campaign_ids=(campaign_id,),
        )
    )


def status(args: argparse.Namespace) -> int:
    store = _store(args.state_root)
    if args.campaign_id:
        emit_json(store.get(args.campaign_id))
    else:
        emit_json(
            {
                "schema_version": 2,
                "artifact_type": "verdiwm-campaign-list",
                "items": store.list(status=args.status, limit=args.limit),
            }
        )
    return 0


def cancel(args: argparse.Namespace) -> int:
    emit_json(_store(args.state_root).cancel(args.campaign_id))
    return 0


def job(args: argparse.Namespace) -> int:
    if args.job_action == "submit":
        metadata = json.loads(args.metadata)
        if not isinstance(metadata, dict):
            raise ValueError("JOB_METADATA_INVALID")
        command = tuple(args.job_command)
        if command and command[0] == "--":
            command = command[1:]
        emit_json(
            submit_job(
                JobSpec(
                    command=command,
                    cwd=args.cwd,
                    job_root=args.job_root,
                    environment=dict(args.env or ()),
                    output_root=args.output_root,
                    timeout_seconds=args.timeout_seconds,
                    metadata=metadata,
                )
            )
        )
        return 0
    if args.job_action == "status":
        emit_json(load_status(args.job_root))
        return 0
    if args.job_action == "tail":
        emit_text(tail_job(args.job_root, lines=args.lines))
        return 0
    if args.job_action == "cancel":
        emit_json(cancel_job(args.job_root, grace_seconds=args.grace_seconds))
        return 0
    emit_json(resume_job(args.job_root))
    return 0


def reproduce(args: argparse.Namespace) -> int:
    store = _store(args.state_root)
    campaign = store.reproduce(args.campaign_id)
    if args.queue_only:
        emit_json(campaign)
        return 0
    dispatcher = _dispatch(store, str(campaign["campaign_id"]), args.max_parallel)
    reproduced = store.get(str(campaign["campaign_id"]))
    emit_json({"campaign": reproduced, "dispatcher": dispatcher})
    return 0 if reproduced.get("status") in {"completed", "cancelled"} else 2
