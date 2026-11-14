"""Parser registration for the public campaign run command."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from pathlib import Path


def _positive_world_size(value: str) -> int:
    try:
        world_size = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "world size must be a positive integer"
        ) from exc
    if world_size < 1:
        raise argparse.ArgumentTypeError("world size must be a positive integer")
    return world_size


def add_run_command(
    commands: argparse._SubParsersAction[argparse.ArgumentParser],
    *,
    handlers: Mapping[str, Callable[[argparse.Namespace], int]],
    asset_argument: Callable[[str], tuple[str, str]],
    positive_rounds: Callable[[str], int],
) -> None:
    """Register ``verdi run`` and retain its existing flags and help text."""

    run = commands.add_parser(
        "run", help="compile and run a model optimization campaign"
    )
    run.add_argument("intent", nargs="?", help="plain-language research goal")
    run.add_argument("--model")
    run.add_argument(
        "--source", help="模型源码目录；省略时从 checkpoint 父目录自动推断"
    )
    run.add_argument(
        "--data", "--dataset-root", dest="data", help="训练数据文件或数据集根目录"
    )
    run.add_argument("--goal")
    run.add_argument(
        "--target-metrics",
        "--metrics",
        dest="target_metrics",
        nargs="+",
        help="metrics to improve; validated against the frozen evaluator catalog",
    )
    run.add_argument("--budget", default=None)
    run.add_argument(
        "--rounds",
        type=positive_rounds,
        default=None,
        help="最多研究轮数；所有轮共享总 GPU 预算",
    )
    run.add_argument(
        "--stop-on-success",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="正式冻结 verifier 确认并 settled 后提前停止",
    )
    run.add_argument("--adapter", default="auto")
    run.add_argument(
        "--mode",
        choices=("quick-start", "causal-discovery", "hybrid"),
        help="research routing policy; omitted preserves the legacy pipeline",
    )
    run.add_argument("--literature-query")
    run.add_argument(
        "--model-irg", type=Path, help="已有实测 IRG；诊断查询与用户目标联合用于检索"
    )
    run.add_argument(
        "--irg-protected-metric",
        action="append",
        default=[],
        help="IRG 方法生成时保护的指标，可重复指定",
    )
    run.add_argument("--cpbe-request", type=Path)
    run.add_argument("--cpbe-history", type=Path)
    run.add_argument("--adapter-profile", type=Path)
    run.add_argument("--runtime-python", type=Path)
    run.add_argument("--engineering-manifest", type=Path)
    run.add_argument("--compiled-manifest", type=Path)
    run.add_argument("--engineering-repo-root", type=Path)
    run.add_argument("--schema-root", type=Path)
    run.add_argument("--asset", action="append", type=asset_argument, default=[])
    run.add_argument("--training-scale-plan", type=Path)
    run.add_argument(
        "--world-size",
        type=_positive_world_size,
        default=None,
        help=(
            "GPU worker count for the auto-generated Agibot training scale plan; "
            "requires --training-config and cannot be combined with --training-scale-plan"
        ),
    )
    run.add_argument(
        "--training-config",
        type=Path,
        help=(
            "native Agibot training YAML; creates a bounded engineering diagnostic "
            "plan when --training-scale-plan is omitted"
        ),
    )
    repair_group = run.add_mutually_exclusive_group()
    repair_group.add_argument(
        "--auto-repair-adapter",
        dest="auto_repair_adapter",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    repair_group.add_argument(
        "--no-auto-repair-adapter",
        dest="auto_repair_adapter",
        action="store_false",
        help="disable automatic adapter completion",
    )
    run.set_defaults(auto_repair_adapter=True)
    run.add_argument("--repair-base-profile", type=Path, help=argparse.SUPPRESS)
    run.add_argument("--repair-attempts", type=int, default=3)
    run.add_argument("--llm-config", type=Path)
    run.add_argument("--campaign-id")
    run.add_argument("--state-root", type=Path, default=None)
    run.add_argument("--queue-only", action="store_true")
    run.add_argument("--max-parallel", type=int, default=1)
    run.add_argument(
        "--format",
        choices=("json", "text"),
        default=None,
        help="输出精简的人类可读摘要，或完整机器兼容 JSON",
    )
    run.set_defaults(handler=handlers["_run"])
