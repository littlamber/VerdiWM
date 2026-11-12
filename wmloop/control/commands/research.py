"""Research plan and selector-evaluation CLI commands."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from wmloop.control.campaign_api import CampaignStore
from wmloop.control.commands.io import emit_json
from wmloop.control.project_config import ProjectConfigError, load_project_config
from wmloop.control.research_plan_history import write_default_research_plan
from wmloop.control.research_request import (
    ResearchRequestError,
    compile_research_plan,
    load_research_plan,
    run_research_plan,
    write_research_plan,
)
from wmloop.experiments.repair_effect_predictor import run_contextual_predictor_replay
from wmloop.experiments.selector_replay import run_selector_replay


def plan(args: argparse.Namespace) -> int:
    try:
        loaded = load_project_config(cwd=args.project_root or Path.cwd())
        configured: dict[str, Any] = loaded.values
    except ProjectConfigError as exc:
        raise ResearchRequestError(str(exc)) from exc
    project_root = args.project_root or (loaded.source.parent if loaded.source else Path.cwd())
    result = compile_research_plan(
        model=args.model or configured.get("model"),
        data=args.data or configured.get("data", configured.get("dataset")),
        source=args.source or configured.get("source"),
        goal=args.goal or configured.get("goal"),
        budget=args.budget or configured.get("budget", "1gpu-hour"),
        mode=args.mode or configured.get("mode", "hybrid"),
        adapter=args.adapter or configured.get("adapter", "auto"),
        evaluator_contract=args.evaluator_contract
        or configured.get("evaluator_contract"),
        runtime_python=args.runtime_python or configured.get("runtime_python"),
        adapter_profile=args.adapter_profile or configured.get("adapter_profile"),
        target_metrics=args.target_metrics
        or configured.get(
            "target_metrics", configured.get("metrics", configured.get("metric"))
        ),
        model_irg=args.model_irg,
        irg_protected_metrics=args.irg_protected_metric,
        state_root=args.state_root or configured.get("state_root"),
        project_root=project_root,
    )
    destination = (
        write_research_plan(result, args.output)
        if args.output is not None
        else write_default_research_plan(result, project_root)
    )
    emit_json({**result, "plan_path": str(destination)})
    return 0 if result["state"] in {"ready", "ready_with_deferred_discovery"} else 2


def run(args: argparse.Namespace) -> int:
    plan_document = load_research_plan(args.plan)
    state_root = args.state_root or Path(
        str(
            plan_document.get(
                "state_root",
                Path(str(plan_document.get("project_root", Path.cwd())))
                / ".verdiwm"
                / "state",
            )
        )
    )
    result = run_research_plan(
        plan_document,
        store=CampaignStore(Path(state_root).expanduser().resolve() / "campaigns"),
        confirm=args.confirm,
        queue_only=args.queue_only,
        max_parallel=args.max_parallel,
    )
    emit_json(result)
    return (
        0
        if result["state"]
        in {"awaiting_confirmation", "queued", "completed", "cancelled"}
        else 2
    )


def evaluate_irg(args: argparse.Namespace) -> int:
    result = run_selector_replay(
        plan_path=args.plan,
        projection_path=args.projections,
        effect_label_index=args.labels,
        primitive_probe_affinity=args.affinity,
        output_root=args.output,
    )
    emit_json(result)
    return 0 if result.get("state") in {"ready", "partial"} else 2


def evaluate_predictor(args: argparse.Namespace) -> int:
    result = run_contextual_predictor_replay(
        examples_path=args.examples,
        plan_path=args.plan,
        labels_path=args.labels,
        output_root=args.output,
    )
    emit_json(result)
    return 0 if result.get("state") == "ready" else 2
