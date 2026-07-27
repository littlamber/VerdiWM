#!/usr/bin/env python3
"""Export the next campaign plan for 8-env uplift and transfer."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wmloop.execute.acwm_primitive_routes import INVALIDATED_QUALITY_PRIMITIVES


REGIMES = {
    "push_cube": "rigid_body",
    "stack_cube": "rigid_body",
    "push_rope": "deformable",
    "cloth_move": "deformable",
    "push_sand": "particle",
    "pour_water": "particle_fluid",
    "robot_arm": "kinematics",
    "reacher": "kinematics",
}

FAILURE_SIGNATURES = {
    "push_cube": ["contact_event", "rigid_pose_slip", "action_binding"],
    "stack_cube": ["support_relation", "contact_instability", "object_identity"],
    "push_rope": ["topology_change", "endpoint_path", "deformable_contact"],
    "cloth_move": ["surface_fold", "cloth_identity_drift", "deformable_memory"],
    "push_sand": ["granular_frontier", "mass_redistribution", "particle_boundary"],
    "pour_water": ["fluid_volume_transport", "container_boundary_leak", "free_surface"],
    "robot_arm": ["kinematic_action_binding", "smooth_motion_prior", "target_path"],
    "reacher": ["target_conditioning", "inverse_dynamics_confidence", "endpoint_control"],
}

NEXT_ACTIONS = {
    "push_cube": "stage contact/slip diagnostics, then test frontier_collection, mixture_reweight, inv_dyn_reward_finetune, next_forcing",
    "stack_cube": "stage support/contact diagnostics, then test frontier_collection, mixture_reweight, latent_spatial_memory, inv_dyn_reward_finetune",
    "push_rope": "stage topology/endpoint diagnostics, then test frontier_collection, mixture_reweight, next_forcing, self_forcing_finetune",
    "cloth_move": "treat official-current ckpt as paired-delta only; stage fold/identity diagnostics, then test first_frame_anchor and latent_spatial_memory",
    "push_sand": "stage granular/mass diagnostics, then test frontier_collection, mixture_reweight, dino_rep_injection, next_forcing",
    "pour_water": "drop drift_token_trim promotion after 3 negative canaries; stage volume/leak diagnostics and try next_forcing/self_forcing or new fluid-specific staging",
    "robot_arm": "retain as stable positive exemplar; generate retained visual evidence and formal cross-campaign summary",
    "reacher": "stage target/inv-dyn confidence diagnostics, then test inv_dyn_reward_finetune, cfg_guidance_schedule, next_forcing",
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_effects(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows[(row["environment"], row["primitive"])] = row
    return rows


def float_field(row: dict[str, Any], name: str) -> float | None:
    try:
        return float(row[name])
    except Exception:
        return None


def int_field(row: dict[str, Any], name: str) -> int | None:
    try:
        return int(float(row[name]))
    except Exception:
        return None


def best_by_environment(effects: dict[tuple[str, str], dict[str, Any]]) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for (environment, primitive), row in effects.items():
        mean_delta = float_field(row, "mean_delta")
        if mean_delta is None:
            continue
        current = best.get(environment)
        current_delta = float_field(current, "mean_delta") if current else None
        if current is None or current_delta is None or mean_delta > current_delta:
            best[environment] = row | {"primitive": primitive}
    return best


def stable_positive_environments(summary: dict[str, Any]) -> set[str]:
    primitive = str(summary.get("primitive") or "")
    environment = summary.get("environment")
    if (
        summary.get("cross_campaign_stable_positive") is True
        and isinstance(environment, str)
        and environment
        and primitive not in INVALIDATED_QUALITY_PRIMITIVES
    ):
        return {environment}
    return set()


def env_status(environment: str, row: dict[str, Any] | None, stable_positive_envs: set[str], rejected_pairs: set[tuple[str, str]]) -> dict[str, Any]:
    primitive = row.get("primitive") if row else "none"
    pair = (environment, str(primitive))
    stable_positive = environment in stable_positive_envs
    rejected = pair in rejected_pairs
    if stable_positive:
        status = "stable_positive_exemplar"
    elif rejected:
        status = "rejected_candidate_needs_new_diagnosis"
    else:
        status = "uplift_gap"
    next_action = NEXT_ACTIONS[environment]
    if environment == "robot_arm" and not stable_positive:
        next_action = "Treat latent_motion_prior evidence as audit-only; diagnose and stage a valid replacement primitive."
    return {
        "environment": environment,
        "regime": REGIMES[environment],
        "current_best_primitive": primitive,
        "current_status": status,
        "trial_count": int_field(row, "trial_count") if row else 0,
        "positive_trial_count": int_field(row, "positive_trial_count") if row else 0,
        "positive_rate": float_field(row, "positive_rate") if row else None,
        "mean_delta": float_field(row, "mean_delta") if row else None,
        "max_delta": float_field(row, "max_delta") if row else None,
        "failure_signatures_to_probe": FAILURE_SIGNATURES[environment],
        "next_action": next_action,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_env_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "environment",
        "regime",
        "current_best_primitive",
        "current_status",
        "trial_count",
        "positive_trial_count",
        "positive_rate",
        "mean_delta",
        "max_delta",
        "failure_signatures_to_probe",
        "next_action",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["failure_signatures_to_probe"] = ",".join(row["failure_signatures_to_probe"])
            writer.writerow(out)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# ACWM 8-Env Uplift and Transfer Plan R1",
        "",
        "## Thesis",
        "",
        "目标不是把 ACWM-Phys 当成唯一场景，而是把它当作八个物理失败签名的训练场：系统要学会从目标和数据出发，诊断失败、选择或生成原语、执行代码落地、验证收益，并把成功与失败都沉淀成可迁移机制卡。",
        "",
        "对 Cosmos3、Ctrl-World 或其它世界模型，用户理想输入应是 base model、数据、目标指标/评测器和预算；闭环系统负责把它编译成一次可审计战役：goal_spec、held-out split、probe registry、evaluator adapter、hook adapter、primitive registry、canary/formal protocol。",
        "",
        "## Current Evidence Boundary",
        "",
        f"- Stable positive environments: `{report['summary']['stable_positive_environment_count']}` / 8",
        f"- Uplift gaps: `{report['summary']['uplift_gap_count']}` / 8",
        f"- Rejected candidate pairs: `{report['summary']['rejected_candidate_count']}`",
        "- Verdict probes remain frozen during the active campaign; diagnostic probes and staging candidates may evolve behind an admission boundary.",
        "",
        "## Eight-Environment Plan",
        "",
        "| Env | Regime | Current best | Status | Trials | Positive | Mean delta | Next action |",
        "|:--|:--|:--|:--|--:|--:|--:|:--|",
    ]
    for row in report["environment_rows"]:
        mean_delta = row["mean_delta"]
        lines.append(
            "| {environment} | {regime} | {primitive} | `{status}` | {trials} | {positive} | {mean_delta} | {next_action} |".format(
                environment=row["environment"],
                regime=row["regime"],
                primitive=row["current_best_primitive"],
                status=row["current_status"],
                trials=row["trial_count"],
                positive=row["positive_trial_count"],
                mean_delta=f"{mean_delta:.3f}" if isinstance(mean_delta, float) else "n/a",
                next_action=row["next_action"],
            )
        )
    lines.extend(
        [
            "",
            "## System Upgrade",
            "",
            "1. 目标编译器：把用户给的 base model、dataset、target metrics 编译成 goal_spec、held-out split 和 budget contract。",
            "2. 探针规划器：从目标指标和失败模式生成 diagnostic probes；只有人类批准的新版本才能把 probe 升级为 verdict probe。",
            "3. 失败签名库：把 horizon/action/appearance/OoD 结果和环境专用探针投影成 model-agnostic failure signature。",
            "4. 原语检索与生成：先检索已有 mechanism cards 和文献/经验库，再把新机制写入 staging，过 schema、hook contract、canary 后才允许进入 registry。",
            "5. 代码落地执行器：LLM 不能直接改正式训练代码；它输出意图、机制、参数和可证伪预测，执行器用 adapter/templates 渲染 diff 并做沙盒测试。",
            "6. 经验沉淀：每个正/负结果都更新 mechanism card 的 transfer preconditions 和 anti-conditions，供 Cosmos3/Ctrl-World 等新实例排序使用。",
            "",
            "## Paper Story",
            "",
            "更强的创新点应表述为：VerdiWM 是一个可实例化的世界模型改进闭环，它在 ACWM-Phys 八环境上学习失败签名到干预原语的映射，并通过冻结验证器筛选可复现提升；跨模型泛化不是复用 ACWM-Phys 指标，而是复用诊断-原语-验证-记忆的系统结构。",
            "",
            "不能提前声称八环境都已提升。当前正确目标是把八环境提升作为下一阶段战役，把 robot_arm 作为已验证正例，把 pour_water drift_token_trim 作为系统筛掉错误候选的负例。",
            "",
            "## Outputs",
            "",
            "- `tables/environment_uplift_plan.csv`",
            "- `uplift-transfer-plan.json`",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-export", type=Path, required=True)
    parser.add_argument("--robot-summary", type=Path, required=True)
    parser.add_argument("--pour-summary", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    if output_root.exists():
        raise SystemExit("ACWM_8ENV_UPLIFT_TRANSFER_PLAN_OUTPUT_EXISTS")
    effects = read_effects(args.paper_export / "tables" / "primitive_effects_by_cell.csv")
    best = best_by_environment(effects)
    robot_summary = read_json(args.robot_summary)
    pour_summary = read_json(args.pour_summary)
    stable_positive_envs = stable_positive_environments(robot_summary)
    rejected_pairs = set()
    if pour_summary.get("cross_campaign_stable_positive") is False:
        rejected_pairs.add((str(pour_summary.get("environment")), str(pour_summary.get("primitive"))))
    environment_rows = [
        env_status(environment, best.get(environment), stable_positive_envs, rejected_pairs)
        for environment in REGIMES
    ]
    uplift_gap_count = sum(1 for row in environment_rows if row["current_status"] != "stable_positive_exemplar")
    report = {
        "artifact_type": "wmloop-acwm-8env-uplift-transfer-plan",
        "schema_version": 1,
        "state": "ready",
        "summary": {
            "stable_positive_environment_count": len(stable_positive_envs),
            "uplift_gap_count": uplift_gap_count,
            "rejected_candidate_count": len(rejected_pairs),
            "target_environment_count": 8,
            "invalidated_quality_primitive_count": len(INVALIDATED_QUALITY_PRIMITIVES),
        },
        "environment_rows": environment_rows,
        "instance_compiler_contract": {
            "user_supplied": ["base_model", "dataset_or_rollout_source", "target_metrics_or_evaluator", "budget"],
            "system_compiles": [
                "goal_spec",
                "heldout_split",
                "probe_registry",
                "evaluator_adapter",
                "hook_adapter",
                "primitive_registry_or_staging_orders",
                "canary_and_formal_eval_protocol",
            ],
            "not_allowed": [
                "mutate verdict probes during a campaign",
                "claim transfer without a frozen evaluator and held-out split",
                "promote generated mechanism code without staging/admission",
            ],
        },
        "claim_boundary": {
            "metric_passing_artifacts_retained": True,
            "invalidated_quality_primitives_are_audit_only": sorted(INVALIDATED_QUALITY_PRIMITIVES),
            "stable_positive_requires_non_invalidated_primitive": True,
        },
        "source_files": {
            "paper_export": str(args.paper_export),
            "robot_summary": str(args.robot_summary),
            "pour_summary": str(args.pour_summary),
        },
    }
    write_json(output_root / "uplift-transfer-plan.json", report)
    write_env_csv(output_root / "tables" / "environment_uplift_plan.csv", environment_rows)
    write_markdown(output_root / "uplift-transfer-plan.md", report)
    print(json.dumps({"state": "ready", "output_root": str(output_root)}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
