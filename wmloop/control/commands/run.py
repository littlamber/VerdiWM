"""Campaign creation command service.

The public ``wmloop.cli`` parser remains a compatibility facade; this module
contains the stateful run flow and accepts its infrastructure dependencies
explicitly so it can be tested without importing the CLI module.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from wmloop.control.adapter_profiles import AdapterProfileError
from wmloop.control.adapter_repair import run_adapter_repair
from wmloop.control.campaign_api import CampaignAPIError, CampaignStore
from wmloop.control.commands.io import emit_json, repository_root
from wmloop.control.research_proposal import (
    ResearchProposalError,
    load_compiled_experiment_manifest,
)
from wmloop.experiments.engineering import (
    ExperimentEngineeringError,
    lint_experiment_manifest,
)
from wmloop.storage import sha256_bytes, sha256_file


def run(
    args: argparse.Namespace,
    *,
    resolve_inputs: Callable[[argparse.Namespace], dict[str, Any]],
    store_factory: Callable[[Path], CampaignStore],
    dispatch: Callable[..., dict[str, Any]],
    configured_llm_adapter: Callable[[Path | None], dict[str, object]],
    load_training_scale_plan: Callable[..., dict[str, object]],
    is_repairable_error: Callable[[BaseException], bool],
) -> int:
    inputs = resolve_inputs(args)
    args.state_root = inputs["state_root"]
    store = store_factory(args.state_root)
    engineering_lint = None
    compiled_manifest = None
    if args.compiled_manifest is not None:
        compiled_manifest = load_compiled_experiment_manifest(args.compiled_manifest)
        compiled_engineering = Path(
            str(compiled_manifest["engineering"]["manifest_path"])
        ).resolve()
        if args.engineering_manifest is not None:
            if (
                Path(args.engineering_manifest).expanduser().resolve()
                != compiled_engineering
            ):
                raise ResearchProposalError(
                    "COMPILED_EXPERIMENT_ENGINEERING_BINDING_MISMATCH"
                )
        else:
            args.engineering_manifest = compiled_engineering
    if args.engineering_manifest is not None:
        engineering_lint = lint_experiment_manifest(
            args.engineering_manifest,
            repo_root=args.engineering_repo_root,
            root=(args.schema_root or repository_root()).expanduser().resolve(),
        )
        if engineering_lint["state"] != "ready":
            raise ExperimentEngineeringError(
                "EXPERIMENT_ENGINEERING_BLOCKED:"
                + ",".join(str(item) for item in engineering_lint["blockers"])
            )
    payload: dict[str, Any] = {
        "goal": inputs["goal"],
        "model": inputs["model"],
        "dataset": inputs["data"],
        "budget": inputs["budget"],
        "adapter": inputs["adapter"],
    }
    if inputs.get("source"):
        payload["source"] = inputs["source"]
    if inputs.get("target_metrics") is not None:
        payload["target_metrics"] = inputs["target_metrics"]
    if inputs["mode"] is not None:
        payload["research_mode"] = inputs["mode"]
    if args.literature_query is not None:
        payload["literature_query"] = args.literature_query
    if getattr(args, "model_irg", None) is not None:
        payload["model_irg_path"] = str(args.model_irg.expanduser().resolve())
        payload["irg_protected_metrics"] = args.irg_protected_metric
    if args.cpbe_request is not None:
        payload["cpbe_request"] = str(args.cpbe_request.expanduser().resolve())
    if args.cpbe_history is not None:
        payload["cpbe_history"] = str(args.cpbe_history.expanduser().resolve())
    if engineering_lint is not None:
        payload["engineering_manifest"] = {
            "path": engineering_lint["manifest_path"],
            "manifest_sha256": engineering_lint["source"]["manifest_sha256"],
            "source_revision": engineering_lint["source"]["revision"],
            "source_dirty": engineering_lint["source"]["dirty"],
        }
    if compiled_manifest is not None:
        payload["compiled_manifest"] = {
            "path": str(Path(args.compiled_manifest).expanduser().resolve()),
            "sha256": sha256_file(Path(args.compiled_manifest).expanduser().resolve()),
        }
    if inputs["campaign_id"]:
        payload["campaign_id"] = inputs["campaign_id"]
    if inputs["adapter_profile"]:
        payload["adapter_profile_path"] = str(inputs["adapter_profile"])
    if inputs["runtime_python"]:
        payload["runtime_python"] = str(inputs["runtime_python"])
    if args.asset:
        payload["assets"] = dict(args.asset)
    if args.training_scale_plan is not None:
        payload["training_scale_plan"] = load_training_scale_plan(
            args.training_scale_plan,
            root=(args.schema_root or repository_root()),
        )
    try:
        created = store.create(payload)
    except CampaignAPIError as exc:
        repairable = is_repairable_error(exc)
        if not getattr(args, "auto_repair_adapter", True) or not repairable:
            raise
        base_profile = args.repair_base_profile
        if base_profile is None:
            from wmloop.control.adapter_profiles import select_repair_base_profile

            try:
                base_profile = select_repair_base_profile(
                    root=(args.schema_root or repository_root()),
                    model=Path(inputs["model"]),
                    goal=inputs["goal"],
                    adapter=inputs.get("adapter"),
                )
            except AdapterProfileError as profile_exc:
                raise CampaignAPIError(str(profile_exc)) from exc
        repair_id = sha256_bytes(
            json.dumps(
                {
                    "model": inputs["model"],
                    "data": inputs["data"],
                    "goal": inputs["goal"],
                },
                sort_keys=True,
            ).encode("utf-8")
        )[:24]
        repair = run_adapter_repair(
            model=Path(inputs["model"]),
            source=Path(inputs["source"]) if inputs.get("source") else None,
            data=Path(inputs["data"]),
            goal=inputs["goal"],
            budget=inputs["budget"],
            failure_code=str(exc),
            base_profile_path=base_profile,
            llm_adapter=configured_llm_adapter(getattr(args, "llm_config", None)),
            output_root=args.state_root / "adapter-repairs" / repair_id,
            project_root=(args.schema_root or repository_root()),
            runtime_python=inputs["runtime_python"],
            max_attempts=args.repair_attempts,
        )
        if repair["state"] != "ready":
            raise CampaignAPIError(
                "AUTO_REPAIR_BLOCKED:"
                + ",".join(
                    str(row.get("code"))
                    for row in repair["blockers"]
                    if isinstance(row, dict)
                )
            ) from exc
        payload["adapter"] = "auto"
        payload["adapter_profile_path"] = repair["adapter_profile_path"]
        repair_manifest_path = (
            args.state_root / "adapter-repairs" / repair_id / "manifest.json"
        )
        payload["adapter_repair"] = {
            "manifest_path": str(repair_manifest_path),
            "manifest_sha256": sha256_file(repair_manifest_path),
            "input_digest": repair["input_digest"],
            "assurance_level": repair["assurance_level"],
        }
        created = store.create(payload)
    queued = store.confirm(str(created["campaign_id"]))
    if args.queue_only:
        emit_json(queued)
        return 0
    dispatcher = dispatch(
        store,
        campaign_id=str(created["campaign_id"]),
        max_parallel=args.max_parallel,
    )
    campaign = store.get(str(created["campaign_id"]))
    emit_json({"campaign": campaign, "dispatcher": dispatcher})
    return 0 if campaign.get("status") in {"completed", "cancelled"} else 2
