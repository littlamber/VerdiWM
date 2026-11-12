"""User-facing VerdiWM campaign commands."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from wmloop.control.adapter_repair import AdapterRepairError, run_adapter_repair
from wmloop.control.campaign_api import CampaignAPIError
from wmloop.control.campaign_dispatcher import (
    CampaignDispatchError,
)
from wmloop.control.commands.audit import (
    _audit as _audit,
    _import_settlements as _import_settlements,
)
from wmloop.control.commands.batch import (
    plan as _batch_plan,
    run as _batch_run,
    status as _batch_status,
)
from wmloop.control.commands.community import (
    _community_export,
    _community_lifecycle,
    _community_publish,
    _community_verify,
    _evidence_project,
    _load_community_documents as _load_community_documents,
    _write_community_json as _write_community_json,
)
from wmloop.control.commands.configuration import (
    _configured_llm_adapter as _configured_llm_adapter,
    _is_repairable_campaign_error as _is_repairable_campaign_error,
    _load_training_scale_plan as _load_training_scale_plan,
    _resolve_llm_config as _resolve_llm_config,
)
from wmloop.control.commands.inputs import (
    _default_state_root as _default_state_root,
    _load_project_intent as _load_project_intent,
    _path_configured as _path_configured,
    _resolve_run_inputs as _resolve_run_inputs,
)
from wmloop.control.commands.io import emit_json as _print
from wmloop.control.commands.lifecycle import (
    _dispatch as _dispatch,
    _store as _store,
    cancel as _cancel,
    job as _job,
    reproduce as _reproduce,
    status as _status,
)
from wmloop.control.commands.parser import build_parser
from wmloop.control.commands.project import (
    check_model as _check_model,
    evaluator_discover as _evaluator_discover,
    evaluator_overlay as _evaluator_overlay,
    guide as _guide_model,
    init as _init_project,
    wan22_droid as _wan22_droid,
)
from wmloop.control.commands.proposal import (
    bootstrap as _bootstrap_template,
    compile as _compile_proposal,
    lint as _lint_experiment,
)
from wmloop.control.commands.research import (
    evaluate_irg as _research_evaluate_irg,
    evaluate_predictor as _research_evaluate_predictor,
    plan as _research_plan,
    run as _research_run,
)
from wmloop.control.commands.run import run as _run_command
from wmloop.control.commands.system import (
    check as _check,
    diagnose as _diagnose,
    doctor as _doctor,
    doctor_report as _doctor_report,  # noqa: F401 - legacy import
    installed_version as _installed_version,
    llm as _llm,
)
from wmloop.control.commands.training import (
    diagnose_gain as _diagnose_training_gain,
    ladder as _plan_training_ladder,
    plan as _plan_training,
    recipes as _training_recipes,
)
from wmloop.control.first_contact import (
    FirstContactError,
    explain_blocker,
)
from wmloop.control.model_batch import (
    ModelBatchError,
)
from wmloop.control.research_proposal import (
    ResearchProposalError,
)
from wmloop.control.research_request import (
    ResearchRequestError,
)
from wmloop.evaluate.system_utility import (
    SystemUtilityAuditError,
)
from wmloop.execute.configured_llm_broker import ConfiguredBrokerError
from wmloop.experiments.community_bundle import (
    CommunityBundleError,
)
from wmloop.experiments.community_export import (
    CommunityExportError,
)
from wmloop.experiments.engineering import (
    ExperimentEngineeringError,
)
from wmloop.experiments.repair_effect_predictor import RepairPredictorError
from wmloop.experiments.selector_replay import SelectorReplayError
from wmloop.experiments.training_gain_attribution import (
    TrainingGainAttributionError,
)
from wmloop.experiments.training_scale import (
    TrainingScaleError,
)
from wmloop.geometry.community_knowledge import (
    CommunityKnowledgeError,
)
from wmloop.retrieve.training_recipes import (
    TrainingRecipeError,
)


def _cli_program_name() -> str:
    """Use the short product name for the public alias, keep legacy help stable."""

    executable = Path(sys.argv[0]).name.casefold()
    return "verdi" if executable in {"verdi", "verdi.exe"} else "verdiwm"


def _asset(value: str) -> tuple[str, str]:
    parameter, separator, path = value.partition("=")
    if not separator or not parameter.strip() or not path.strip():
        raise argparse.ArgumentTypeError("asset must be PARAM=PATH")
    normalized = parameter.strip()
    if not normalized.startswith("--"):
        normalized = f"--{normalized}"
    return normalized, path.strip()


def _job_env(value: str) -> tuple[str, str]:
    key, separator, item = value.partition("=")
    if not separator or not key or not item:
        raise argparse.ArgumentTypeError("environment override must be KEY=VALUE")
    return key, item


def _setup(args: argparse.Namespace) -> int:
    return _init_project(args)


def _run(args: argparse.Namespace) -> int:
    return _run_command(
        args,
        resolve_inputs=_resolve_run_inputs,
        store_factory=_store,
        dispatch=_dispatch,
        configured_llm_adapter=_configured_llm_adapter,
        load_training_scale_plan=_load_training_scale_plan,
        is_repairable_error=_is_repairable_campaign_error,
    )


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


def _interactive(args: argparse.Namespace) -> int:
    """Enter the optional line-oriented shell from an explicit subcommand."""

    from wmloop.control.interactive_cli import run_interactive_session

    return run_interactive_session()


def _parser() -> argparse.ArgumentParser:
    return build_parser(
        handlers={
            "_audit": _audit,
            "_batch_plan": _batch_plan,
            "_batch_run": _batch_run,
            "_batch_status": _batch_status,
            "_bootstrap_template": _bootstrap_template,
            "_cancel": _cancel,
            "_check": _check,
            "_check_model": _check_model,
            "_community_export": _community_export,
            "_community_lifecycle": _community_lifecycle,
            "_community_publish": _community_publish,
            "_community_verify": _community_verify,
            "_compile_proposal": _compile_proposal,
            "_diagnose": _diagnose,
            "_diagnose_training_gain": _diagnose_training_gain,
            "_doctor": _doctor,
            "_evaluator_discover": _evaluator_discover,
            "_evaluator_overlay": _evaluator_overlay,
            "_evidence_project": _evidence_project,
            "_guide_model": _guide_model,
            "_import_settlements": _import_settlements,
            "_init_project": _init_project,
            "_interactive": _interactive,
            "_job": _job,
            "_lint_experiment": _lint_experiment,
            "_llm": _llm,
            "_plan_training": _plan_training,
            "_plan_training_ladder": _plan_training_ladder,
            "_repair_adapter": _repair_adapter,
            "_reproduce": _reproduce,
            "_research_evaluate_irg": _research_evaluate_irg,
            "_research_evaluate_predictor": _research_evaluate_predictor,
            "_research_plan": _research_plan,
            "_research_run": _research_run,
            "_run": _run,
            "_setup": _setup,
            "_status": _status,
            "_training_recipes": _training_recipes,
            "_wan22_droid": _wan22_droid,
        },
        program_name=_cli_program_name(),
        installed_version=_installed_version(),
        default_state_root=_default_state_root,
        asset_argument=_asset,
        job_environment=_job_env,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if getattr(args, "command", None) is None:
        # ``verdi`` is the interactive product entrypoint when attached to a
        # terminal.  Keep non-TTY invocations deterministic for scripts, CI,
        # pipes, and tests; the legacy ``verdiwm`` alias always keeps help
        # behavior on a bare invocation.
        if _cli_program_name() == "verdi":
            from wmloop.control.interactive_cli import (
                interactive_supported,
                run_interactive_session,
            )

            if interactive_supported(sys.stdin, sys.stdout):
                return run_interactive_session()
        parser.print_help()
        return 0
    try:
        return int(args.handler(args))
    except OSError as exc:
        sys.stderr.write(
            "本地文件操作失败，请检查输入、输出目录及读写权限。"
            f" [LOCAL_FILESYSTEM_ERROR:{type(exc).__name__}]\n"
        )
        return 2
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
        ModelBatchError,
        CommunityBundleError,
        CommunityExportError,
        CommunityKnowledgeError,
        ResearchRequestError,
        SelectorReplayError,
        RepairPredictorError,
    ) as exc:
        message = explain_blocker(exc)
        sys.stderr.write(f"{message['error']} [{message['code']}]\n")
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
