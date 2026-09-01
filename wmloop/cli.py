"""User-facing VerdiWM campaign commands."""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any, Sequence

from wmloop.control.campaign_api import CampaignAPIError, CampaignStore
from wmloop.control.adapter_repair import AdapterRepairError, run_adapter_repair
from wmloop.control.model_executor_bootstrap import bootstrap_request_template
from wmloop.control.adapter_profiles import AdapterProfileError
from wmloop.control.campaign_dispatcher import (
    CampaignDispatchError,
    DispatcherOptions,
    run_dispatcher,
)
from wmloop.control.research_proposal import (
    ResearchProposalError,
    compile_proposal_to_experiment_manifest,
    load_compiled_experiment_manifest,
    write_compiled_experiment_manifest,
)
from wmloop.control.project_config import ProjectConfigError, load_project_config
from wmloop.control.first_contact import (
    FirstContactError,
    explain_blocker,
    initialize_project,
    inspect_project,
)
from wmloop.control.onboarding_assistant import build_onboarding_questionnaire, write_onboarding_questionnaire
from wmloop.evaluate.system_utility import (
    SystemUtilityAuditError,
    run_system_utility_audit,
)
from wmloop.experiments.engineering import (
    ExperimentEngineeringError,
    lint_experiment_manifest,
)
from wmloop.experiments.training_scale import (
    TrainingScaleError,
    build_training_scale_plan,
    write_training_scale_plan,
)
from wmloop.experiments.training_gain_attribution import (
    TrainingGainAttributionError,
    build_training_gain_attribution,
    write_training_gain_attribution,
)
from wmloop.retrieve.training_recipes import (
    TrainingRecipeError,
    find_training_recipe,
    load_training_recipe_registry,
    require_admitted_recipe,
    summarize_training_recipes,
)
from wmloop.contracts import ContractValidationError, validate_document
from wmloop.execute.configured_llm_broker import ConfiguredBrokerError, load_config


def _default_state_root() -> Path:
    configured = os.environ.get("VERDIWM_STATE_ROOT")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".local" / "state" / "verdiwm"


def _asset(value: str) -> tuple[str, str]:
    parameter, separator, path = value.partition("=")
    if not separator or not parameter.strip() or not path.strip():
        raise argparse.ArgumentTypeError("asset must be PARAM=PATH")
    normalized = parameter.strip()
    if not normalized.startswith("--"):
        normalized = f"--{normalized}"
    return normalized, path.strip()


def _store(state_root: Path) -> CampaignStore:
    return CampaignStore(Path(state_root).expanduser().resolve() / "campaigns")


def _dispatch(
    store: CampaignStore, *, campaign_id: str, max_parallel: int
) -> dict[str, Any]:
    return run_dispatcher(
        DispatcherOptions(
            state_root=store.root,
            max_cycles=1,
            max_parallel=max_parallel,
            campaign_ids=(campaign_id,),
        )
    )


def _print(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def _load_project_intent() -> dict[str, Any]:
    try:
        return load_project_config().values
    except ProjectConfigError as exc:
        raise CampaignAPIError(str(exc)) from exc


def _resolve_run_inputs(args: argparse.Namespace) -> dict[str, Any]:
    """Resolve human intent and project defaults into the legacy run shape."""

    configured = _load_project_intent()
    model = args.model or configured.get("model")
    data = args.data or configured.get("data", configured.get("dataset"))
    if model is None:
        candidate = Path.cwd() / "model"
        model = str(candidate) if candidate.is_dir() else None
    if data is None:
        for name in ("data", "dataset"):
            candidate = Path.cwd() / name
            if candidate.exists():
                data = str(candidate)
                break
    goal = args.goal or args.intent or configured.get("goal")
    if not isinstance(goal, str) or not goal.strip():
        raise CampaignAPIError("GOAL_REQUIRED")
    if not model:
        raise CampaignAPIError("MODEL_PATH_REQUIRED:configure project.model or pass --model")
    if not data:
        raise CampaignAPIError("DATA_PATH_REQUIRED:configure project.data or pass --data")
    return {
        "model": str(model),
        "data": str(data),
        "goal": goal.strip(),
        "target_metrics": (
            args.target_metrics
            if getattr(args, "target_metrics", None) is not None
            else configured.get("target_metrics", configured.get("metrics", configured.get("metric")))
        ),
        "budget": args.budget or configured.get("budget", "1gpu-hour"),
        "adapter": args.adapter if args.adapter != "auto" else configured.get("adapter", "auto"),
        "mode": args.mode or configured.get("mode"),
        "adapter_profile": args.adapter_profile or _path_configured(configured.get("adapter_profile")),
        "runtime_python": args.runtime_python or _path_configured(configured.get("runtime_python")),
        "state_root": args.state_root if args.state_root != _default_state_root() else Path(str(configured.get("state_root", args.state_root))),
        "campaign_id": args.campaign_id or configured.get("campaign_id"),
    }


def _path_configured(value: object) -> Path | None:
    return Path(str(value)).expanduser() if value is not None else None


def _doctor(args: argparse.Namespace) -> int:
    root = (args.repo_root or Path(__file__).resolve().parents[1]).expanduser().resolve()
    checks: list[dict[str, object]] = []

    python_supported = sys.version_info[:2] == (3, 10)
    checks.append(
        {
            "name": "python_3_10",
            "required": True,
            "state": "pass" if python_supported else "fail",
            "detail": platform.python_version(),
        }
    )
    required_files = (
        ("core_package", "wmloop/__init__.py", None),
        ("goal_schema", "configs/schemas/goal_spec.schema.json", None),
        (
            "adapter_profile",
            "configs/adapters/ctrl_world_predictive_v2.json",
            "verdiwm-adapter-profile",
        ),
        (
            "mechanism_ontology",
            "configs/retrieval/mechanism_tag_ontology_v1.json",
            "wmloop-mechanism-tag-ontology",
        ),
    )
    for name, relative, expected_type in required_files:
        path = root / relative
        state = "pass"
        detail = relative
        if not path.is_file() or path.is_symlink():
            state = "fail"
            detail = f"missing:{relative}"
        elif path.suffix == ".json":
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                state = "fail"
                detail = f"invalid_json:{relative}"
            else:
                if not isinstance(payload, dict) or (
                    expected_type is not None
                    and payload.get("artifact_type") != expected_type
                ):
                    state = "fail"
                    detail = f"invalid_contract:{relative}"
        checks.append(
            {"name": name, "required": True, "state": state, "detail": detail}
        )

    public_example = root / "examples" / "acwm_minimal_loop_cloth_next_forcing_v2"
    checks.append(
        {
            "name": "public_example",
            "required": False,
            "state": "available" if public_example.is_dir() else "not_installed",
            "detail": "examples/acwm_minimal_loop_cloth_next_forcing_v2",
        }
    )
    blocked = any(
        row["required"] is True and row["state"] != "pass" for row in checks
    )
    try:
        package_version = version("verdiwm")
    except PackageNotFoundError:
        package_version = "source"
    _print(
        {
            "schema_version": 1,
            "artifact_type": "verdiwm-doctor-report",
            "state": "blocked" if blocked else "ready",
            "version": package_version,
            "checks": checks,
        }
    )
    return 2 if blocked else 0


def _run(args: argparse.Namespace) -> int:
    inputs = _resolve_run_inputs(args)
    args.state_root = inputs["state_root"]
    store = _store(args.state_root)
    engineering_lint = None
    compiled_manifest = None
    if args.compiled_manifest is not None:
        compiled_manifest = load_compiled_experiment_manifest(args.compiled_manifest)
        compiled_engineering = Path(
            str(compiled_manifest["engineering"]["manifest_path"])
        ).resolve()
        if args.engineering_manifest is not None:
            if Path(args.engineering_manifest).expanduser().resolve() != compiled_engineering:
                raise ResearchProposalError("COMPILED_EXPERIMENT_ENGINEERING_BINDING_MISMATCH")
        else:
            args.engineering_manifest = compiled_engineering
    if args.engineering_manifest is not None:
        engineering_lint = lint_experiment_manifest(
            args.engineering_manifest,
            repo_root=args.engineering_repo_root,
            root=(args.schema_root or Path(__file__).resolve().parents[1]).expanduser().resolve(),
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
    if inputs.get("target_metrics") is not None:
        payload["target_metrics"] = inputs["target_metrics"]
    if inputs["mode"] is not None:
        payload["research_mode"] = inputs["mode"]
    if args.literature_query is not None:
        payload["literature_query"] = args.literature_query
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
            "sha256": hashlib.sha256(
                Path(args.compiled_manifest).expanduser().resolve().read_bytes()
            ).hexdigest(),
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
        payload["training_scale_plan"] = _load_training_scale_plan(
            args.training_scale_plan,
            root=(args.schema_root or Path(__file__).resolve().parents[1]),
        )
    try:
        created = store.create(payload)
    except CampaignAPIError as exc:
        repairable = _is_repairable_campaign_error(exc)
        if not getattr(args, "auto_repair_adapter", True) or not repairable:
            raise
        base_profile = args.repair_base_profile
        if base_profile is None:
            from wmloop.control.adapter_profiles import select_repair_base_profile
            try:
                base_profile = select_repair_base_profile(
                    root=(args.schema_root or Path(__file__).resolve().parents[1]),
                    model=Path(inputs["model"]),
                    goal=inputs["goal"],
                    adapter=inputs.get("adapter"),
                )
            except AdapterProfileError as profile_exc:
                raise CampaignAPIError(str(profile_exc)) from exc
        repair_id = hashlib.sha256(
            json.dumps(
            {"model": inputs["model"], "data": inputs["data"], "goal": inputs["goal"]},
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:24]
        repair = run_adapter_repair(
            model=Path(inputs["model"]),
            data=Path(inputs["data"]),
            goal=inputs["goal"],
            budget=inputs["budget"],
            failure_code=str(exc),
            base_profile_path=base_profile,
            llm_adapter=_configured_llm_adapter(getattr(args, "llm_config", None)),
            output_root=args.state_root / "adapter-repairs" / repair_id,
            project_root=(args.schema_root or Path(__file__).resolve().parents[1]),
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
        repair_manifest_path = args.state_root / "adapter-repairs" / repair_id / "manifest.json"
        payload["adapter_repair"] = {
            "manifest_path": str(repair_manifest_path),
            "manifest_sha256": hashlib.sha256(repair_manifest_path.read_bytes()).hexdigest(),
            "input_digest": repair["input_digest"],
            "assurance_level": repair["assurance_level"],
        }
        created = store.create(payload)
    queued = store.confirm(str(created["campaign_id"]))
    if args.queue_only:
        _print(queued)
        return 0
    dispatcher = _dispatch(
        store,
        campaign_id=str(created["campaign_id"]),
        max_parallel=args.max_parallel,
    )
    campaign = store.get(str(created["campaign_id"]))
    _print({"campaign": campaign, "dispatcher": dispatcher})
    return 0 if campaign.get("status") in {"completed", "cancelled"} else 2


def _status(args: argparse.Namespace) -> int:
    store = _store(args.state_root)
    if args.campaign_id:
        _print(store.get(args.campaign_id))
    else:
        _print(
            {
                "schema_version": 2,
                "artifact_type": "verdiwm-campaign-list",
                "items": store.list(status=args.status, limit=args.limit),
            }
        )
    return 0


def _cancel(args: argparse.Namespace) -> int:
    _print(_store(args.state_root).cancel(args.campaign_id))
    return 0


def _reproduce(args: argparse.Namespace) -> int:
    store = _store(args.state_root)
    campaign = store.reproduce(args.campaign_id)
    if args.queue_only:
        _print(campaign)
        return 0
    dispatcher = _dispatch(
        store,
        campaign_id=str(campaign["campaign_id"]),
        max_parallel=args.max_parallel,
    )
    reproduced = store.get(str(campaign["campaign_id"]))
    _print({"campaign": reproduced, "dispatcher": dispatcher})
    return 0 if reproduced.get("status") in {"completed", "cancelled"} else 2


def _import_settlements(args: argparse.Namespace) -> int:
    from wmloop.experiments.ctrl_world_settlement_import import (
        CtrlWorldSettlementImportError,
        import_ctrl_world_settlements,
    )

    try:
        manifest = import_ctrl_world_settlements(
            input_root=args.input_root,
            archive_db=args.archive_db,
            cas_root=args.cas_root,
            output_root=args.output_root,
            allowed_roots=tuple(args.allowed_root),
            goal_id=args.goal_id,
            dry_run=args.dry_run,
        )
    except CtrlWorldSettlementImportError as exc:
        raise CampaignDispatchError(str(exc)) from exc
    _print(manifest)
    return 0


def _audit(args: argparse.Namespace) -> int:
    root = (args.repo_root or Path.cwd()).expanduser().resolve()
    config = args.config or root / "configs" / "experiments" / "system_utility_audit_v1.json"
    manifest = run_system_utility_audit(
        config_path=config,
        repo_root=root,
        output_root=args.output_root,
    )
    _print(manifest)
    return 3 if args.require_effect and manifest["research_effect_state"] != "established" else 0


def _plan_training(args: argparse.Namespace) -> int:
    root = (
        args.schema_root
        or args.repo_root
        or Path(__file__).resolve().parents[1]
    ).expanduser().resolve()
    training_profile = None
    if args.training_recipe:
        registry = load_training_recipe_registry(
            args.training_recipe_registry,
            root=root,
        )
        training_profile = require_admitted_recipe(registry, args.training_recipe)
    plan = build_training_scale_plan(
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
        stage=args.stage,
        batch_size=args.batch_size,
        gradient_accumulation=args.gradient_accumulation,
        world_size=args.world_size,
        sequence_length=args.sequence_length,
        requested_seed_count=args.seed_count,
        training_profile=training_profile,
        root=root,
    )
    if args.output:
        write_training_scale_plan(plan, args.output)
    _print(plan)
    return 0 if plan["state"] == "ready" else 3


def _training_recipes(args: argparse.Namespace) -> int:
    root = (args.repo_root or Path(__file__).resolve().parents[1]).expanduser().resolve()
    registry_path = args.registry or root / "configs" / "retrieval" / "world_model_training_recipes_v1.json"
    registry = load_training_recipe_registry(registry_path, root=root)
    if args.recipe_id:
        # Keep full recipe details available for an explicit audit query while
        # the default output remains a compact ranking/index artifact.
        _print({"index": summarize_training_recipes(registry, recipe_id=args.recipe_id), "recipe": find_training_recipe(registry, args.recipe_id)})
    else:
        _print(summarize_training_recipes(registry))
    return 0


def _lint_experiment(args: argparse.Namespace) -> int:
    result = lint_experiment_manifest(
        args.manifest,
        repo_root=args.repo_root,
        root=(args.schema_root or Path.cwd()).expanduser().resolve(),
    )
    _print(result)
    return 0 if result["state"] == "ready" else 3


def _compile_proposal(args: argparse.Namespace) -> int:
    manifest = compile_proposal_to_experiment_manifest(
        args.proposal,
        engineering_manifest_path=args.engineering_manifest,
        training_scale_plan_path=args.training_scale_plan,
        model_capability_ir_path=args.model_capability_ir,
        workflow_registry_path=args.workflow_registry,
        engineering_repo_root=args.engineering_repo_root,
        root=(args.schema_root or Path(__file__).resolve().parents[1]).expanduser().resolve(),
    )
    if args.output:
        write_compiled_experiment_manifest(manifest, args.output)
    _print(manifest)
    return 0 if manifest["state"] == "ready" else 3


def _repair_adapter(args: argparse.Namespace) -> int:
    manifest = run_adapter_repair(
        model=args.model,
        source=args.source,
        data=args.data,
        goal=args.goal,
        budget=args.budget,
        failure_code=args.failure_code,
        base_profile_path=args.base_profile,
        llm_adapter=_configured_llm_adapter(args.llm_config),
        output_root=args.output_root,
        project_root=(args.schema_root or Path(__file__).resolve().parents[1]),
        runtime_python=args.runtime_python,
        max_attempts=args.max_attempts,
    )
    _print(manifest)
    return 0 if manifest["state"] == "ready" else 3


def _bootstrap_template(args: argparse.Namespace) -> int:
    _print(bootstrap_request_template(model=str(args.model), data=str(args.data), goal=args.goal))
    return 0


def _init_project(args: argparse.Namespace) -> int:
    result = initialize_project(
        root=Path.cwd(),
        model=args.model,
        source=args.source,
        data=args.data,
        goal=args.goal,
        budget=args.budget,
        mode=args.mode,
        target_metrics=args.target_metrics,
        evaluator_contract=str(args.evaluator_contract) if args.evaluator_contract else None,
        runtime_python=str(args.runtime_python) if args.runtime_python else None,
        project_file=args.project_file,
        force=args.force,
    )
    _print(result)
    return 0 if result["state"] == "ready" else 3


def _check_model(args: argparse.Namespace) -> int:
    configured: dict[str, Any] = {}
    try:
        configured = load_project_config(cwd=Path.cwd()).values
    except ProjectConfigError:
        configured = {}
    result = inspect_project(
        root=Path.cwd(),
        model=args.model or configured.get("model"),
        source=args.source or configured.get("source"),
        data=args.data or configured.get("data", configured.get("dataset")),
        evaluator_contract=(
            str(args.evaluator_contract)
            if args.evaluator_contract
            else configured.get("evaluator_contract")
        ),
        runtime_python=(
            str(args.runtime_python)
            if args.runtime_python
            else configured.get("runtime_python")
        ),
    )
    _print(result)
    return 0 if result["state"] == "ready_for_conformance" else 2


def _guide_model(args: argparse.Namespace) -> int:
    configured: dict[str, Any] = {}
    try:
        configured = load_project_config(cwd=Path.cwd()).values
    except ProjectConfigError:
        configured = {}
    questionnaire = build_onboarding_questionnaire(
        root=Path.cwd(),
        model=args.model or configured.get("model"),
        source=args.source or configured.get("source"),
        data=args.data or configured.get("data", configured.get("dataset")),
        goal=args.goal or configured.get("goal"),
        evaluator_contract=args.evaluator_contract or configured.get("evaluator_contract"),
        runtime_python=args.runtime_python or configured.get("runtime_python"),
    )
    if args.output is not None:
        questionnaire["questionnaire_path"] = str(write_onboarding_questionnaire(args.output, questionnaire))
    _print(questionnaire)
    return 0 if questionnaire["state"] == "awaiting_confirmation" else 2


def _diagnose_training_gain(args: argparse.Namespace) -> int:
    plan = build_training_gain_attribution(
        training_receipt_path=args.training_receipt,
        screen_settlement_path=args.screen_settlement,
        confirm_settlement_path=args.confirm_settlement,
        verifier_manifest_path=args.verifier_manifest,
    )
    if args.output is not None:
        write_training_gain_attribution(plan, args.output)
    _print(plan)
    return 0


def _configured_llm_adapter(config_path: Path | None) -> dict[str, object]:
    source = _resolve_llm_config(config_path)
    config = load_config(source)
    return {
        "command": [
            sys.executable,
            "-m",
            "wmloop.execute.configured_llm_broker",
            "{request_path}",
            "{response_path}",
            "--config",
            str(source),
        ],
        "timeout_seconds": float(config["timeout_seconds"]),
        "max_output_bytes": int(config["maximum_bytes"]),
        "provider_alias": "configured-openai-compatible",
        "model_alias": str(config["model"]),
        "credential_environment_keys": [],
    }


def _resolve_llm_config(config_path: Path | None) -> Path:
    if config_path is not None:
        return config_path.expanduser().resolve()
    configured = os.environ.get("VERDIWM_CONFIG")
    if configured:
        return Path(configured).expanduser().resolve()
    for name in ("verdiwm.toml", "verdiwm.config.toml", "config.toml"):
        candidate = Path.cwd() / name
        if candidate.is_file() and not candidate.is_symlink():
            try:
                load_config(candidate)
            except ConfiguredBrokerError:
                continue
            return candidate.resolve()
    return (Path.home() / ".config" / "verdiwm" / "config.toml").resolve()


def _is_repairable_campaign_error(error: BaseException) -> bool:
    """Follow the domain-error chain for a provider-neutral repair decision."""

    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, AdapterProfileError) and current.repairable:
            return True
        current = current.__cause__ or current.__context__
    return False


def _load_training_scale_plan(path: Path, *, root: Path) -> dict[str, object]:
    source = path.expanduser().resolve()
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingScaleError("TRAINING_SCALE_PLAN_INVALID") from exc
    if not isinstance(value, dict):
        raise TrainingScaleError("TRAINING_SCALE_PLAN_INVALID")
    try:
        validate_document("training_scale_plan", value, root=root.expanduser().resolve())
    except ContractValidationError as exc:
        raise TrainingScaleError("TRAINING_SCALE_PLAN_INVALID") from exc
    if value.get("state") != "ready":
        raise TrainingScaleError("TRAINING_SCALE_PLAN_NOT_READY")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="verdiwm", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    doctor = commands.add_parser(
        "doctor", help="verify the local CPU control-plane installation"
    )
    doctor.add_argument("--repo-root", type=Path)
    doctor.set_defaults(handler=_doctor)

    run = commands.add_parser("run", help="compile and run a model optimization campaign")
    run.add_argument("intent", nargs="?", help="plain-language research goal")
    run.add_argument("--model")
    run.add_argument("--data")
    run.add_argument("--goal")
    run.add_argument("--target-metrics", "--metrics", dest="target_metrics", nargs="+", help="metrics to improve; validated against the frozen evaluator catalog")
    run.add_argument("--budget", default=None)
    run.add_argument("--adapter", default="auto")
    run.add_argument(
        "--mode",
        choices=("quick-start", "causal-discovery", "hybrid"),
        help="research routing policy; omitted preserves the legacy pipeline",
    )
    run.add_argument("--literature-query")
    run.add_argument("--cpbe-request", type=Path)
    run.add_argument("--cpbe-history", type=Path)
    run.add_argument("--adapter-profile", type=Path)
    run.add_argument("--runtime-python", type=Path)
    run.add_argument("--engineering-manifest", type=Path)
    run.add_argument("--compiled-manifest", type=Path)
    run.add_argument("--engineering-repo-root", type=Path)
    run.add_argument("--schema-root", type=Path)
    run.add_argument("--asset", action="append", type=_asset, default=[])
    run.add_argument("--training-scale-plan", type=Path)
    repair_group = run.add_mutually_exclusive_group()
    repair_group.add_argument("--auto-repair-adapter", dest="auto_repair_adapter", action="store_true", help=argparse.SUPPRESS)
    repair_group.add_argument("--no-auto-repair-adapter", dest="auto_repair_adapter", action="store_false", help="disable automatic adapter completion")
    run.set_defaults(auto_repair_adapter=True)
    run.add_argument("--repair-base-profile", type=Path, help=argparse.SUPPRESS)
    run.add_argument("--repair-attempts", type=int, default=3)
    run.add_argument("--llm-config", type=Path)
    run.add_argument("--campaign-id")
    run.add_argument("--state-root", type=Path, default=_default_state_root())
    run.add_argument("--queue-only", action="store_true")
    run.add_argument("--max-parallel", type=int, default=1)
    run.set_defaults(handler=_run)

    status = commands.add_parser("status", help="show one campaign or list campaigns")
    status.add_argument("campaign_id", nargs="?")
    status.add_argument("--status")
    status.add_argument("--limit", type=int, default=100)
    status.add_argument("--state-root", type=Path, default=_default_state_root())
    status.set_defaults(handler=_status)

    cancel = commands.add_parser("cancel", help="cancel a queued or running campaign")
    cancel.add_argument("campaign_id")
    cancel.add_argument("--state-root", type=Path, default=_default_state_root())
    cancel.set_defaults(handler=_cancel)

    reproduce = commands.add_parser(
        "reproduce", help="create and dispatch an isolated reproduction campaign"
    )
    reproduce.add_argument("campaign_id")
    reproduce.add_argument("--state-root", type=Path, default=_default_state_root())
    reproduce.add_argument("--queue-only", action="store_true")
    reproduce.add_argument("--max-parallel", type=int, default=1)
    reproduce.set_defaults(handler=_reproduce)

    settlement_import = commands.add_parser(
        "import-settlements",
        help="import terminal Ctrl-World mechanism evidence into Archive",
    )
    settlement_import.add_argument("--input-root", type=Path, required=True)
    settlement_import.add_argument("--archive-db", type=Path, required=True)
    settlement_import.add_argument("--cas-root", type=Path, required=True)
    settlement_import.add_argument("--output-root", type=Path, required=True)
    settlement_import.add_argument(
        "--allowed-root", type=Path, action="append", default=[]
    )
    settlement_import.add_argument(
        "--goal-id", default="ctrl_world_predictive_quality_pilot_v2"
    )
    settlement_import.add_argument("--dry-run", action="store_true")
    settlement_import.set_defaults(handler=_import_settlements)

    audit = commands.add_parser(
        "audit",
        help="summarize operational usability and evidence-backed research utility",
    )
    audit.add_argument("--config", type=Path)
    audit.add_argument("--repo-root", type=Path)
    audit.add_argument("--output-root", type=Path, required=True)
    audit.add_argument(
        "--require-effect",
        action="store_true",
        help="return 3 unless the configured effect gates are established",
    )
    audit.set_defaults(handler=_audit)

    plan_training = commands.add_parser(
        "plan-training",
        help="derive a bounded world-model training scale from sample manifests",
    )
    plan_training.add_argument("--train-manifest", type=Path, required=True)
    plan_training.add_argument("--val-manifest", type=Path, required=True)
    plan_training.add_argument("--stage", choices=("smoke", "screen", "pilot", "confirm"), default="screen")
    plan_training.add_argument("--batch-size", type=int, default=1)
    plan_training.add_argument("--gradient-accumulation", type=int, default=1)
    plan_training.add_argument("--world-size", type=int, default=1)
    plan_training.add_argument("--sequence-length", type=int)
    plan_training.add_argument("--seed-count", type=int)
    plan_training.add_argument("--training-recipe", help="admitted local recipe id; shadow-only literature is rejected")
    plan_training.add_argument(
        "--training-recipe-registry",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "retrieval" / "world_model_training_recipes_v1.json",
    )
    plan_training.add_argument("--schema-root", type=Path)
    plan_training.add_argument("--repo-root", type=Path)
    plan_training.add_argument("--output", type=Path)
    plan_training.set_defaults(handler=_plan_training)

    lint_experiment = commands.add_parser(
        "lint-experiment",
        help="lint an experiment engineering manifest and owned source surface",
    )
    lint_experiment.add_argument("--manifest", type=Path, required=True)
    lint_experiment.add_argument("--repo-root", type=Path)
    lint_experiment.add_argument("--schema-root", type=Path)
    lint_experiment.set_defaults(handler=_lint_experiment)

    compile_proposal = commands.add_parser(
        "compile-proposal",
        help="bind an LLM research proposal to an experiment and scale plan without executing it",
    )
    compile_proposal.add_argument("--proposal", type=Path, required=True)
    compile_proposal.add_argument("--engineering-manifest", type=Path, required=True)
    compile_proposal.add_argument("--training-scale-plan", type=Path, required=True)
    compile_proposal.add_argument("--model-capability-ir", type=Path)
    compile_proposal.add_argument("--workflow-registry", type=Path)
    compile_proposal.add_argument("--engineering-repo-root", type=Path)
    compile_proposal.add_argument("--schema-root", type=Path)
    compile_proposal.add_argument("--output", type=Path)
    compile_proposal.set_defaults(handler=_compile_proposal)

    training_recipes = commands.add_parser(
        "training-recipes",
        help="list auditable world-model training recipes and their evidence tiers",
    )
    training_recipes.add_argument("--registry", type=Path)
    training_recipes.add_argument("--recipe-id")
    training_recipes.add_argument("--repo-root", type=Path)
    training_recipes.set_defaults(handler=_training_recipes)

    repair_adapter = commands.add_parser(
        "repair-adapter",
        help="use the configured LLM to repair a model-interface adapter in an isolated overlay",
    )
    repair_adapter.add_argument("--model", type=Path, required=True)
    repair_adapter.add_argument("--data", type=Path, required=True)
    repair_adapter.add_argument("--goal", required=True)
    repair_adapter.add_argument("--budget", required=True)
    repair_adapter.add_argument("--failure-code", default="ADAPTER_PROFILE_NOT_FOUND")
    repair_adapter.add_argument("--base-profile", type=Path, required=True)
    repair_adapter.add_argument("--llm-config", type=Path)
    repair_adapter.add_argument("--runtime-python", type=Path)
    repair_adapter.add_argument("--max-attempts", type=int, default=3)
    repair_adapter.add_argument("--output-root", type=Path, required=True)
    repair_adapter.add_argument("--schema-root", type=Path)
    repair_adapter.set_defaults(handler=_repair_adapter)

    bootstrap_template = commands.add_parser(
        "bootstrap-template",
        help="print a plain-language request template for first-contact model setup",
    )
    bootstrap_template.add_argument("--model", required=True)
    bootstrap_template.add_argument("--data", required=True)
    bootstrap_template.add_argument("--goal", required=True)
    bootstrap_template.set_defaults(handler=_bootstrap_template)

    init = commands.add_parser(
        "init",
        help="用模型、数据和一句目标描述创建首次接入项目",
    )
    init.add_argument("--model", help="模型目录；默认发现 ./model 或 ./models")
    init.add_argument("--source", help="模型源码目录；可与权重目录分离")
    init.add_argument("--data", help="数据目录；默认发现 ./data、./dataset 或 ./datasets")
    init.add_argument("--goal", help="想改善的能力，用一句话描述")
    init.add_argument("--budget", default="1gpu-hour")
    init.add_argument("--mode", choices=("quick-start", "causal-discovery", "hybrid"), default="hybrid")
    init.add_argument("--target-metrics", "--metrics", dest="target_metrics", nargs="+", default=[])
    init.add_argument("--evaluator-contract", type=Path, help="冻结评测契约；没有时只生成项目并列出下一步")
    init.add_argument("--runtime-python", type=Path, help="模型运行环境中的 Python；默认自动发现")
    init.add_argument("--project-file", type=Path)
    init.add_argument("--force", action="store_true", help="允许覆盖已有项目文件")
    init.set_defaults(handler=_init_project)

    check_model = commands.add_parser(
        "check-model",
        help="只读检查模型项目是否具备开始接入所需的信息",
    )
    check_model.add_argument("--model", help="模型目录；默认读取项目配置或发现 ./model")
    check_model.add_argument("--source", help="模型源码目录；可与权重目录分离")
    check_model.add_argument("--data", help="数据目录；默认读取项目配置或发现 ./data")
    check_model.add_argument("--evaluator-contract", type=Path, help="冻结评测契约")
    check_model.add_argument("--runtime-python", type=Path, help="模型环境中的 Python")
    check_model.set_defaults(handler=_check_model)

    guide_model = commands.add_parser(
        "guide-model",
        help="根据只读检查结果生成新模型接入问卷",
    )
    guide_model.add_argument("--model", help="模型目录；默认读取项目配置或发现 ./model")
    guide_model.add_argument("--source", help="模型源码目录；可与权重目录分离")
    guide_model.add_argument("--data", help="数据目录；默认读取项目配置或发现 ./data")
    guide_model.add_argument("--goal", help="研究目标")
    guide_model.add_argument("--evaluator-contract", type=Path, help="已有冻结评测契约")
    guide_model.add_argument("--runtime-python", type=Path, help="模型环境中的 Python")
    guide_model.add_argument("--output", type=Path, help="将问卷写到模型目录之外的文件")
    guide_model.set_defaults(handler=_guide_model)

    diagnose_gain = commands.add_parser(
        "diagnose-training-gain",
        help="plan experiments that distinguish optimization, data, capacity, and mechanism limits",
    )
    diagnose_gain.add_argument("--training-receipt", type=Path, required=True)
    diagnose_gain.add_argument("--screen-settlement", type=Path, required=True)
    diagnose_gain.add_argument("--confirm-settlement", type=Path, required=True)
    diagnose_gain.add_argument("--verifier-manifest", type=Path, required=True)
    diagnose_gain.add_argument("--output", type=Path)
    diagnose_gain.set_defaults(handler=_diagnose_training_gain)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (
        CampaignAPIError,
        CampaignDispatchError,
        SystemUtilityAuditError,
        TrainingScaleError,
        ExperimentEngineeringError,
        TrainingRecipeError,
        ResearchProposalError,
        AdapterRepairError,
        ConfiguredBrokerError,
        TrainingGainAttributionError,
        FirstContactError,
    ) as exc:
        message = explain_blocker(exc)
        print(f"{message['error']} [{message['code']}]", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
