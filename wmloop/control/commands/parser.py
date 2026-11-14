"""Public CLI grammar; command handlers are supplied by the entry point."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from pathlib import Path

from wmloop.control.commands.io import repository_root
from wmloop.control.commands.run_parser import add_run_command
from wmloop.control.commands.training_parser import (
    add_diagnose_training_gain_command,
    add_training_plan_commands,
    add_training_recipes_command,
)


def _positive_rounds(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("轮数必须是正整数，例如 --rounds 2") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("轮数必须是正整数，例如 --rounds 2")
    return parsed


def build_parser(
    *,
    handlers: Mapping[str, Callable[[argparse.Namespace], int]],
    program_name: str,
    installed_version: str,
    default_state_root: Callable[[], Path],
    asset_argument: Callable[[str], tuple[str, str]],
    job_environment: Callable[[str], tuple[str, str]],
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=program_name,
        description="Verdi：面向世界模型的证据驱动研究工作台。",
        epilog=(
            "常用流程：verdi research plan → 检查计划 → "
            "verdi research run --plan PLAN --confirm。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {installed_version}"
    )
    # Showing the top-level guide on a bare ``verdi`` invocation is friendlier
    # for first contact.  Existing explicit commands and the legacy ``verdiwm``
    # alias keep their normal argparse behavior.
    commands = parser.add_subparsers(dest="command", required=False)

    chat = commands.add_parser(
        "chat",
        aliases=("shell",),
        help="进入交互式 Verdi 会话；输入 / 查看命令面板",
    )
    chat.set_defaults(handler=handlers["_interactive"])

    doctor = commands.add_parser(
        "doctor", help="verify the local CPU control-plane installation"
    )
    doctor.add_argument("--repo-root", type=Path)
    doctor.set_defaults(handler=handlers["_doctor"])

    llm = commands.add_parser(
        "llm", help="配置或检查用于研究助手、检索和方法生成的 LLM API"
    )
    llm.add_argument(
        "llm_action", nargs="?", choices=("status",), help="只读检查本地配置"
    )
    llm.add_argument(
        "--config", type=Path, help="配置文件路径；默认 ~/.config/verdiwm/config.toml"
    )
    llm.set_defaults(handler=handlers["_llm"])

    add_run_command(
        commands,
        handlers=handlers,
        asset_argument=asset_argument,
        positive_rounds=_positive_rounds,
    )

    research = commands.add_parser(
        "research", help="从一句研究目标生成计划，并在一次确认后进入完整闭环"
    )
    research_commands = research.add_subparsers(dest="research_command", required=True)
    research_plan = research_commands.add_parser(
        "plan", help="只读生成可审阅的研究计划；不会导入模型或启动 GPU"
    )
    research_plan.add_argument("--model", help="模型目录；默认读取 verdiwm.toml")
    research_plan.add_argument("--source", help="模型源码目录；可与权重目录分离")
    research_plan.add_argument(
        "--data", "--dataset-root", dest="data",
        help="数据集根目录；默认读取 verdiwm.toml",
    )
    research_plan.add_argument("--goal", help="想改善的能力，用一句话描述")
    research_plan.add_argument(
        "--target-metrics", "--metrics", dest="target_metrics", nargs="+"
    )
    research_plan.add_argument(
        "--protected-metrics",
        nargs="+",
        help="核对 evaluator 的 frozen_goal v2 中声明的完整正式保护指标集合",
    )
    research_plan.add_argument("--budget", default=None, help="例如 4gpu-hours")
    research_plan.add_argument("--rounds", type=_positive_rounds, default=1, help="最多研究轮数，默认 1；所有轮共享 --budget 总预算")
    research_plan.add_argument("--stop-on-success", action=argparse.BooleanOptionalAction, default=False, help="正式冻结 verifier 确认收益并 settled 后提前停止")
    research_plan.add_argument("--adapter", default=None, help="适配器 profile 或 auto")
    research_plan.add_argument(
        "--mode", choices=("quick-start", "causal-discovery", "hybrid")
    )
    research_plan.add_argument("--evaluator-contract", type=Path)
    research_plan.add_argument("--runtime-python", type=Path)
    research_plan.add_argument("--adapter-profile", type=Path)
    research_plan.add_argument("--model-irg", type=Path, help="已有 IRG/行为画像文件")
    research_plan.add_argument("--irg-protected-metric", action="append", default=[])
    research_plan.add_argument("--state-root", type=Path)
    research_plan.add_argument("--project-root", type=Path)
    research_plan.add_argument(
        "--output",
        type=Path,
        default=None,
        help="显式计划输出路径；默认保留版本并更新项目最新计划",
    )
    research_plan.set_defaults(handler=handlers["_research_plan"])

    research_run = research_commands.add_parser(
        "run", help="校验计划输入并在一次确认后创建/运行 campaign"
    )
    research_run.add_argument("--plan", type=Path, required=True)
    research_run.add_argument(
        "--confirm", action="store_true", help="确认后创建并调度 campaign"
    )
    research_run.add_argument(
        "--queue-only", action="store_true", help="只入队，不立即执行"
    )
    research_run.add_argument("--state-root", type=Path)
    research_run.add_argument("--max-parallel", type=int, default=1)
    research_run.add_argument("--rounds", type=_positive_rounds, help="派生新计划的最多轮数；保留旧计划和跨轮总预算")
    research_run.add_argument("--stop-on-success", action=argparse.BooleanOptionalAction, default=None, help="派生计划启用/关闭正式确认成功后提前停止")
    research_run.set_defaults(handler=handlers["_research_run"])

    research_evaluate_irg = research_commands.add_parser(
        "evaluate-irg",
        help="在冻结的 outer replay 上比较 IRG、raw response 和其他 selector",
    )
    research_evaluate_irg.add_argument(
        "--plan", type=Path, required=True, help="selector replay plan"
    )
    research_evaluate_irg.add_argument(
        "--projections",
        type=Path,
        required=True,
        help="按环境生成的 selector projection JSONL",
    )
    research_evaluate_irg.add_argument(
        "--labels", type=Path, required=True, help="非泄漏的 settled effect labels"
    )
    research_evaluate_irg.add_argument(
        "--affinity", type=Path, help="可选的 primitive/probe coverage contract"
    )
    research_evaluate_irg.add_argument(
        "--output", type=Path, required=True, help="输出 replay bundle 目录"
    )
    research_evaluate_irg.set_defaults(handler=handlers["_research_evaluate_irg"])

    research_evaluate_predictor = research_commands.add_parser(
        "evaluate-predictor",
        help="在 outer fold 上评估 target-label-hidden contextual repair predictor",
    )
    research_evaluate_predictor.add_argument(
        "--examples",
        type=Path,
        required=True,
        help="inner source settled repair examples",
    )
    research_evaluate_predictor.add_argument(
        "--plan", type=Path, required=True, help="outer fold candidate plan"
    )
    research_evaluate_predictor.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="settled labels, used only after prediction",
    )
    research_evaluate_predictor.add_argument(
        "--output", type=Path, required=True, help="输出 replay 目录"
    )
    research_evaluate_predictor.set_defaults(
        handler=handlers["_research_evaluate_predictor"]
    )

    job = commands.add_parser(
        "job", help="submit and control detached long-running model jobs"
    )
    job_actions = job.add_subparsers(dest="job_action", required=True)
    job_submit = job_actions.add_parser("submit", help="submit and return immediately")
    job_submit.add_argument("--job-root", type=Path, required=True)
    job_submit.add_argument("--cwd", type=Path, default=Path.cwd())
    job_submit.add_argument("--output-root", type=Path)
    job_submit.add_argument("--timeout-seconds", type=float)
    job_submit.add_argument("--env", action="append", type=job_environment, default=[])
    job_submit.add_argument("--metadata", default="{}")
    job_submit.add_argument("job_command", nargs=argparse.REMAINDER)
    for action in ("status", "tail", "cancel", "resume"):
        action_parser = job_actions.add_parser(action)
        action_parser.add_argument("--job-root", type=Path, required=True)
        if action == "tail":
            action_parser.add_argument("--lines", type=int, default=40)
        elif action == "cancel":
            action_parser.add_argument("--grace-seconds", type=float, default=10.0)
    job.set_defaults(handler=handlers["_job"])

    status = commands.add_parser("status", help="show one campaign or list campaigns")
    status.add_argument("campaign_id", nargs="?")
    status.add_argument("--status")
    status.add_argument("--limit", type=int, default=100)
    status.add_argument(
        "--format",
        choices=("json", "text"),
        default="json",
        help="输出机器兼容 JSON（默认）或精简的人类可读摘要",
    )
    status.add_argument("--state-root", type=Path, default=None)
    status.set_defaults(handler=handlers["_status"])

    cancel = commands.add_parser("cancel", help="cancel a queued or running campaign")
    cancel.add_argument("campaign_id")
    cancel.add_argument(
        "--format",
        choices=("json", "text"),
        default=None,
        help="输出精简的人类可读摘要，或完整机器兼容 JSON",
    )
    cancel.add_argument("--state-root", type=Path, default=None)
    cancel.set_defaults(handler=handlers["_cancel"])

    reproduce = commands.add_parser(
        "reproduce", help="create and dispatch an isolated reproduction campaign"
    )
    reproduce.add_argument("campaign_id")
    reproduce.add_argument("--state-root", type=Path, default=None)
    reproduce.add_argument("--queue-only", action="store_true")
    reproduce.add_argument("--max-parallel", type=int, default=1)
    reproduce.set_defaults(handler=handlers["_reproduce"])

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
    settlement_import.set_defaults(handler=handlers["_import_settlements"])

    audit = commands.add_parser(
        "audit",
        help="summarize operational usability and evidence-backed research utility",
    )
    audit.add_argument(
        "audit_command",
        nargs="?",
        choices=("local-smoke",),
        help="run the CPU-only first-contact maturity audit",
    )
    audit.add_argument("--config", type=Path)
    audit.add_argument("--repo-root", type=Path)
    audit.add_argument("--output-root", type=Path, required=True)
    audit.add_argument(
        "--require-effect",
        action="store_true",
        help="return 3 unless the configured effect gates are established",
    )
    audit.set_defaults(handler=handlers["_audit"])

    add_training_plan_commands(
        commands,
        handlers=handlers,
        repository_root=repository_root,
    )

    lint_experiment = commands.add_parser(
        "lint-experiment",
        help="lint an experiment engineering manifest and owned source surface",
    )
    lint_experiment.add_argument("--manifest", type=Path, required=True)
    lint_experiment.add_argument("--repo-root", type=Path)
    lint_experiment.add_argument("--schema-root", type=Path)
    lint_experiment.set_defaults(handler=handlers["_lint_experiment"])

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
    compile_proposal.set_defaults(handler=handlers["_compile_proposal"])

    add_training_recipes_command(commands, handlers=handlers)

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
    repair_adapter.set_defaults(handler=handlers["_repair_adapter"])

    bootstrap_template = commands.add_parser(
        "bootstrap-template",
        help="print a plain-language request template for first-contact model setup",
    )
    bootstrap_template.add_argument("--model", required=True)
    bootstrap_template.add_argument("--data", required=True)
    bootstrap_template.add_argument("--goal", required=True)
    bootstrap_template.set_defaults(handler=handlers["_bootstrap_template"])

    init = commands.add_parser(
        "init",
        help="用模型、数据和一句目标描述创建首次接入项目",
    )
    init.add_argument("--model", help="模型目录；默认发现 ./model 或 ./models")
    init.add_argument("--source", help="模型源码目录；可与权重目录分离")
    init.add_argument(
        "--data", help="数据目录；默认发现 ./data、./dataset 或 ./datasets"
    )
    init.add_argument("--goal", help="想改善的能力，用一句话描述")
    init.add_argument("--budget", default="1gpu-hour")
    init.add_argument(
        "--mode",
        choices=("quick-start", "causal-discovery", "hybrid"),
        default="hybrid",
    )
    init.add_argument(
        "--target-metrics", "--metrics", dest="target_metrics", nargs="+", default=[]
    )
    init.add_argument(
        "--evaluator-contract",
        type=Path,
        help="冻结评测契约；没有时只生成项目并列出下一步",
    )
    init.add_argument(
        "--runtime-python", type=Path, help="模型运行环境中的 Python；默认自动发现"
    )
    init.add_argument("--project-file", type=Path)
    init.add_argument("--force", action="store_true", help="允许覆盖已有项目文件")
    init.set_defaults(handler=handlers["_init_project"])

    setup = commands.add_parser(
        "setup",
        help="首次设置项目：绑定模型、数据和目标",
    )
    setup.add_argument("--model", help="模型目录；默认发现 ./model 或 ./models")
    setup.add_argument("--source", help="模型源码目录；可与权重目录分离")
    setup.add_argument(
        "--data", help="数据目录；默认发现 ./data、./dataset 或 ./datasets"
    )
    setup.add_argument("--goal", help="想改善的能力，用一句话描述")
    setup.add_argument("--budget", default="1gpu-hour")
    setup.add_argument(
        "--mode",
        choices=("quick-start", "causal-discovery", "hybrid"),
        default="hybrid",
    )
    setup.add_argument(
        "--target-metrics", "--metrics", dest="target_metrics", nargs="+", default=[]
    )
    setup.add_argument("--evaluator-contract", type=Path, help="冻结评测契约")
    setup.add_argument("--runtime-python", type=Path, help="模型环境中的 Python")
    setup.add_argument("--project-file", type=Path)
    setup.add_argument("--force", action="store_true", help="允许覆盖已有项目文件")
    setup.set_defaults(handler=handlers["_setup"])

    check_model = commands.add_parser(
        "check-model",
        help="只读检查模型项目是否具备开始接入所需的信息",
    )
    check_model.add_argument("--model", help="模型目录；默认读取项目配置或发现 ./model")
    check_model.add_argument("--source", help="模型源码目录；可与权重目录分离")
    check_model.add_argument("--data", help="数据目录；默认读取项目配置或发现 ./data")
    check_model.add_argument("--evaluator-contract", type=Path, help="冻结评测契约")
    check_model.add_argument("--runtime-python", type=Path, help="模型环境中的 Python")
    check_model.set_defaults(handler=handlers["_check_model"])

    check = commands.add_parser(
        "check",
        help="汇总项目接入和本地控制面的 readiness/blocker",
    )
    check.add_argument("--model", help="模型目录；默认读取项目配置或发现 ./model")
    check.add_argument("--source", help="模型源码目录；可与权重目录分离")
    check.add_argument("--data", help="数据目录；默认读取项目配置或发现 ./data")
    check.add_argument("--evaluator-contract", type=Path, help="冻结评测契约")
    check.add_argument("--runtime-python", type=Path, help="模型环境中的 Python")
    check.add_argument("--repo-root", type=Path, help="VerdiWM 安装目录")
    check.set_defaults(handler=handlers["_check"])

    diagnose = commands.add_parser(
        "diagnose",
        help="只读扫描模型入口、权重、运行环境和接入阻塞项",
    )
    diagnose.add_argument("--model", help="模型目录；默认读取项目配置或发现 ./model")
    diagnose.add_argument("--source", help="模型源码目录；可与权重目录分离")
    diagnose.add_argument("--data", help="数据目录；默认读取项目配置或发现 ./data")
    diagnose.add_argument("--evaluator-contract", type=Path, help="冻结评测契约")
    diagnose.add_argument("--runtime-python", type=Path, help="模型环境中的 Python")
    diagnose.set_defaults(handler=handlers["_diagnose"])

    wan22_droid = commands.add_parser(
        "wan22-droid",
        help="prepare or validate the model-independent WAN2.2-DROID 30-second ACWM contract",
    )
    wan22_droid.add_argument("phase", choices=("prepare", "conformance"))
    wan22_droid.add_argument("--data-root", type=Path)
    wan22_droid.add_argument("--output-root", type=Path)
    wan22_droid.add_argument("--train-manifest", type=Path)
    wan22_droid.add_argument("--validation-manifest", type=Path)
    wan22_droid.add_argument("--model", type=Path)
    wan22_droid.add_argument("--source", type=Path)
    wan22_droid.add_argument("--evaluator-contract", type=Path)
    wan22_droid.add_argument("--adapter", type=Path)
    wan22_droid.add_argument("--horizon-frames", type=int, default=150)
    wan22_droid.add_argument("--stride", type=int, default=30)
    wan22_droid.set_defaults(handler=handlers["_wan22_droid"])

    evaluator = commands.add_parser(
        "evaluator", help="发现、改写并冻结外部模型的评测候选"
    )
    evaluator_commands = evaluator.add_subparsers(
        dest="evaluator_command", required=True
    )
    evaluator_discover = evaluator_commands.add_parser(
        "discover", help="只读扫描源码中的评测候选（不会自动冻结）"
    )
    evaluator_discover.add_argument("--source-root", type=Path, required=True)
    evaluator_discover.set_defaults(handler=handlers["_evaluator_discover"])
    evaluator_overlay = evaluator_commands.add_parser(
        "materialize", help="把一个候选复制到 VERDI 输出目录并记录路径改写"
    )
    evaluator_overlay.add_argument("--candidate", type=Path, required=True)
    evaluator_overlay.add_argument("--source-root", type=Path, required=True)
    evaluator_overlay.add_argument("--model-root", type=Path, required=True)
    evaluator_overlay.add_argument("--data-root", type=Path, required=True)
    evaluator_overlay.add_argument("--output-root", type=Path, required=True)
    evaluator_overlay.add_argument(
        "--confirm", action="store_true", help="确认此 overlay 作为冻结评测契约"
    )
    evaluator_overlay.set_defaults(handler=handlers["_evaluator_overlay"])

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
    guide_model.set_defaults(handler=handlers["_guide_model"])

    add_diagnose_training_gain_command(commands, handlers=handlers)

    batch_plan = commands.add_parser(
        "batch-plan",
        help="validate heterogeneous model inputs and compile an immutable batch plan",
    )
    batch_plan.add_argument("--manifest", type=Path, required=True)
    batch_plan.add_argument("--output-root", type=Path, required=True)
    batch_plan.add_argument("--repo-root", type=Path)
    batch_plan.set_defaults(handler=handlers["_batch_plan"])

    batch_run = commands.add_parser(
        "batch-run",
        help="将批次计划接入独立 campaigns，并通过统一调度器执行",
    )
    batch_run.add_argument(
        "--plan", type=Path, required=True, help="batch-plan 输出的 plan.json"
    )
    batch_run.add_argument(
        "--output-root", type=Path, help="批次目录；默认使用 plan.json 所在目录"
    )
    batch_run.add_argument(
        "--queue-only", action="store_true", help="只创建并排队，不启动 dispatcher"
    )
    batch_run.add_argument("--max-parallel", type=int, default=1)
    batch_run.add_argument("--repo-root", type=Path)
    batch_run.set_defaults(handler=handlers["_batch_run"])
    batch_status = commands.add_parser(
        "batch-status", help="只读汇总批次中各模型的最新 campaign 状态"
    )
    batch_status.add_argument("--execution", type=Path, required=True)
    batch_status.add_argument("--repo-root", type=Path)
    batch_status.set_defaults(handler=handlers["_batch_status"])

    # Grouped aliases are easier to discover in help output while the
    # hyphenated commands remain stable for scripts and older examples.
    batch = commands.add_parser("batch", help="批量模型实验的计划和执行")
    batch_commands = batch.add_subparsers(dest="batch_command", required=True)
    batch_plan_group = batch_commands.add_parser("plan", help="编译不可变批次计划")
    batch_plan_group.add_argument("--manifest", type=Path, required=True)
    batch_plan_group.add_argument("--output-root", type=Path, required=True)
    batch_plan_group.add_argument("--repo-root", type=Path)
    batch_plan_group.set_defaults(handler=handlers["_batch_plan"])
    batch_run_group = batch_commands.add_parser("run", help="创建并运行批次 campaigns")
    batch_run_group.add_argument("--plan", type=Path, required=True)
    batch_run_group.add_argument("--output-root", type=Path)
    batch_run_group.add_argument("--queue-only", action="store_true")
    batch_run_group.add_argument("--max-parallel", type=int, default=1)
    batch_run_group.add_argument("--repo-root", type=Path)
    batch_run_group.set_defaults(handler=handlers["_batch_run"])
    batch_status_group = batch_commands.add_parser(
        "status", help="只读汇总批次中各模型的最新 campaign 状态"
    )
    batch_status_group.add_argument("--execution", type=Path, required=True)
    batch_status_group.add_argument("--repo-root", type=Path)
    batch_status_group.set_defaults(handler=handlers["_batch_status"])

    community = commands.add_parser("community", help="导出和验证社区共享知识包")
    community_commands = community.add_subparsers(
        dest="community_command", required=True
    )
    community_publish = community_commands.add_parser(
        "publish", help="签名并导出 path-free Evidence/knowledge bundle"
    )
    document_group = community_publish.add_mutually_exclusive_group(required=True)
    document_group.add_argument(
        "--document",
        type=Path,
        action="append",
        help="包含一个语义文档或文档数组的 JSON 文件；可重复指定",
    )
    document_group.add_argument(
        "--documents-dir",
        type=Path,
        help="读取目录下按文件名排序的 *.json 语义文档",
    )
    community_publish.add_argument("--output-root", type=Path, required=True)
    community_publish.add_argument("--publisher-id", required=True)
    community_publish.add_argument("--signing-key", type=Path, required=True)
    community_publish.add_argument(
        "--trust-state",
        choices=(
            "unverified",
            "locally_validated",
            "source_reproducible",
            "target_confirmed",
            "transfer_licensed",
            "community_reviewed",
            "revoked",
        ),
        default="locally_validated",
    )
    community_publish.add_argument(
        "--community-review-state",
        choices=("unreviewed", "reviewed", "withdrawn"),
        default="unreviewed",
    )
    community_publish.add_argument("--license-spdx-id")
    community_publish.add_argument(
        "--execution",
        type=Path,
        help="可选：绑定 batch run 生成的 execution.json；只提取 path-free 批次身份",
    )
    community_publish.set_defaults(handler=handlers["_community_publish"])
    community_verify = community_commands.add_parser(
        "verify", help="验证社区 bundle 的签名、成员 hash 和图重建"
    )
    community_verify.add_argument("--bundle-root", type=Path, required=True)
    community_verify.add_argument(
        "--public-key",
        type=Path,
        help="可选：独立提供的 Ed25519 公钥；默认使用 bundle 中嵌入的公钥",
    )
    community_verify.set_defaults(handler=handlers["_community_verify"])
    community_export = community_commands.add_parser(
        "export", help="从本地语义 JSON 只读发现并整理社区发布记录"
    )
    community_export.add_argument(
        "--source-root",
        type=Path,
        action="append",
        required=True,
        help="递归扫描语义 JSON 的目录；可重复指定",
    )
    community_export.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="导出目录；会以临时目录和原子替换写入",
    )
    community_export.add_argument(
        "--execution",
        type=Path,
        help="可选：绑定 batch execution 的 path-free 身份",
    )
    community_export.add_argument(
        "--max-files",
        type=int,
        default=10_000,
        help="最多扫描的 JSON 文件数",
    )
    community_export.add_argument(
        "--max-file-bytes",
        type=int,
        default=5 * 1024 * 1024,
        help="单个 JSON 文件的最大字节数；超出者会被忽略",
    )
    community_export.add_argument("--repo-root", type=Path, help=argparse.SUPPRESS)
    community_export.set_defaults(handler=handlers["_community_export"])
    community_lifecycle = community_commands.add_parser(
        "lifecycle", help="生成撤回、弃用或 supersession 的 append-only 生命周期记录"
    )
    community_lifecycle.add_argument(
        "--action", choices=("revocation", "deprecation", "supersession"), required=True
    )
    community_lifecycle.add_argument("--subject-kind", required=True)
    community_lifecycle.add_argument("--subject-id", required=True)
    community_lifecycle.add_argument("--reason", required=True)
    community_lifecycle.add_argument(
        "--authority-ref", required=True, help="权威记录的 cas://、urn: 或 sha256: 引用"
    )
    community_lifecycle.add_argument(
        "--evidence-ref", action="append", default=[], help="证据引用；可重复指定"
    )
    community_lifecycle.add_argument("--replacement-kind")
    community_lifecycle.add_argument("--replacement-id")
    community_lifecycle.add_argument("--output", type=Path, required=True)
    community_lifecycle.set_defaults(handler=handlers["_community_lifecycle"])
    evidence = commands.add_parser("evidence", help="Project settled archive evidence")
    evidence_commands = evidence.add_subparsers(dest="evidence_command", required=True)
    project = evidence_commands.add_parser(
        "project", help="Export settled Archive/CAS records"
    )
    project.add_argument("--archive", type=Path, required=True)
    project.add_argument("--cas", type=Path, required=True)
    project.add_argument("--batch", type=Path)
    project.add_argument("--output", type=Path, required=True)
    project.add_argument("--max-trials", type=int, default=10_000)
    project.set_defaults(handler=handlers["_evidence_project"])
    return parser
