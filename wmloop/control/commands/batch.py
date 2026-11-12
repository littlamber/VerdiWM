"""Batch planning and status CLI commands."""

from __future__ import annotations

import argparse

from wmloop.control.commands.io import emit_json
from wmloop.control.model_batch import (
    compile_model_batch,
    run_model_batch,
    summarize_model_batch,
)


def plan(args: argparse.Namespace) -> int:
    result = compile_model_batch(
        request_path=args.manifest, output_root=args.output_root, root=args.repo_root
    )
    emit_json(result)
    return 0 if result["state"] == "ready_for_dispatch" else 2


def run(args: argparse.Namespace) -> int:
    result = run_model_batch(
        plan_path=args.plan,
        queue_only=args.queue_only,
        max_parallel=args.max_parallel,
        output_root=args.output_root,
        root=args.repo_root,
    )
    emit_json(result)
    return 0 if result["state"] in {"queued", "completed"} else 2


def status(args: argparse.Namespace) -> int:
    emit_json(summarize_model_batch(args.execution, root=args.repo_root))
    return 0
