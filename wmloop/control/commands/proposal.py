"""Proposal compilation and experiment-engineering CLI commands."""

from __future__ import annotations

import argparse
from pathlib import Path

from wmloop.control.commands.io import emit_json, repository_root
from wmloop.control.model_executor_bootstrap import bootstrap_request_template
from wmloop.control.research_proposal import (
    compile_proposal_to_experiment_manifest,
    write_compiled_experiment_manifest,
)
from wmloop.experiments.engineering import lint_experiment_manifest


def lint(args: argparse.Namespace) -> int:
    result = lint_experiment_manifest(
        args.manifest,
        repo_root=args.repo_root,
        root=(args.schema_root or Path.cwd()).expanduser().resolve(),
    )
    emit_json(result)
    return 0 if result["state"] == "ready" else 3


def compile(args: argparse.Namespace) -> int:
    result = compile_proposal_to_experiment_manifest(
        args.proposal,
        engineering_manifest_path=args.engineering_manifest,
        training_scale_plan_path=args.training_scale_plan,
        model_capability_ir_path=args.model_capability_ir,
        workflow_registry_path=args.workflow_registry,
        engineering_repo_root=args.engineering_repo_root,
        root=(args.schema_root or repository_root()).expanduser().resolve(),
    )
    if args.output:
        write_compiled_experiment_manifest(result, args.output)
    emit_json(result)
    return 0 if result["state"] == "ready" else 3


def bootstrap(args: argparse.Namespace) -> int:
    emit_json(
        bootstrap_request_template(
            model=str(args.model), data=str(args.data), goal=args.goal
        )
    )
    return 0
