"""Frozen user-facing iteration limits shared by planning and campaign creation."""

from __future__ import annotations

from collections.abc import Mapping
import copy
from datetime import datetime, timezone
from pathlib import Path

from wmloop.storage import canonical_bytes, sha256_bytes


class ResearchExecutionPolicyError(ValueError):
    """A requested iteration or stopping policy cannot be frozen safely."""


def research_execution_policy(*, rounds: object = 1, stop_on_success: object = False) -> dict[str, object]:
    """Validate limits without coercing booleans, fractions, or strings."""

    if isinstance(rounds, bool) or not isinstance(rounds, int) or rounds < 1:
        raise ResearchExecutionPolicyError("RESEARCH_ROUNDS_INVALID")
    if not isinstance(stop_on_success, bool):
        raise ResearchExecutionPolicyError("RESEARCH_STOP_ON_SUCCESS_INVALID")
    return {"rounds": rounds, "stop_on_success": stop_on_success, "budget_scope": "all_rounds"}


def plan_execution_policy(plan: Mapping[str, object]) -> dict[str, object]:
    """Historical plans retain their original single-campaign behavior."""

    policy = plan.get("execution_policy")
    if policy is None:
        return research_execution_policy()
    if not isinstance(policy, Mapping) or set(policy) != {"rounds", "stop_on_success", "budget_scope"} or policy.get("budget_scope") != "all_rounds":
        raise ResearchExecutionPolicyError("RESEARCH_EXECUTION_POLICY_INVALID")
    return research_execution_policy(rounds=policy.get("rounds"), stop_on_success=policy.get("stop_on_success"))


def requires_evolution(policy: Mapping[str, object]) -> bool:
    return policy["rounds"] != 1 or policy["stop_on_success"] is True


def evolution_evaluator_template(policy: Mapping[str, object], evaluator: Path | None) -> Path | None:
    """Use the executor's public preflight before promising a multi-round plan."""

    if not requires_evolution(policy):
        return None
    if evaluator is None:
        raise ResearchExecutionPolicyError("EVOLUTION_EVALUATOR_TEMPLATE_REQUIRED")
    from wmloop.execute.evolution_daemon import validate_evolution_evaluator

    try:
        return validate_evolution_evaluator(evaluator)
    except (OSError, ValueError) as exc:
        raise ResearchExecutionPolicyError(str(exc)) from exc


def policy_input_paths(policy: Mapping[str, object], evaluator: Path | None) -> dict[str, Path]:
    """Bind separately stored template and formal goal alongside their evaluator."""

    template = evolution_evaluator_template(policy, evaluator)
    result = {"evolution_candidate_template": template} if template is not None else {}
    if evaluator is not None:
        from wmloop.verify.goal_success import resolve_frozen_goal

        try:
            goal = resolve_frozen_goal(evaluator)
        except (OSError, ValueError) as exc:
            raise ResearchExecutionPolicyError(str(exc)) from exc
        if goal is not None:
            result["frozen_goal"] = goal
    return result


def policy_warnings(policy: Mapping[str, object], paths: Mapping[str, Path]) -> list[str]:
    if policy["stop_on_success"] is True and "frozen_goal" not in paths:
        return ["尚未绑定 evaluator 的 frozen_goal；系统无法确认目标达成，将继续到轮数、总预算或其他停止条件。"]
    return []


def apply_execution_policy(
    execution: Mapping[str, object], *, policy: Mapping[str, object], total_gpu_hours: float
) -> dict[str, object]:
    """Adapt an existing pipeline to the existing evolution controller once."""

    normalized = plan_execution_policy({"execution_policy": policy})
    result = copy.deepcopy(dict(execution))
    result["research_execution_policy"] = normalized
    if not requires_evolution(normalized):
        return result
    if result.get("kind") not in {"pipeline", "evolution"}:
        raise ResearchExecutionPolicyError("RESEARCH_EVOLUTION_EXECUTION_UNSUPPORTED")
    evolution_evaluator_template(normalized, Path(str(result["evaluator_contract"])))
    output_root = Path(str(result["output_root"]))
    result.update(
        kind="evolution",
        state_root=str(output_root.parent / "evolution-state" / output_root.name),
        total_budget_gpu_hours=total_gpu_hours,
        max_iterations=normalized["rounds"],
        stop_on_success=normalized["stop_on_success"],
        max_no_information=0,
        poll_seconds=0,
    )
    # A single durable ledger is passed to every inner pipeline. The controller
    # receives the user's total, never a fresh allowance for each iteration.
    result.pop("budget_total_gpu_hours", None)
    return result


def derive_research_plan_policy(
    plan: Mapping[str, object], *, rounds: int | None = None, stop_on_success: bool | None = None
) -> dict[str, object]:
    """Preserve frozen inputs while producing a separately reviewable revision.

    This function does not write files or launch work. A changed ready plan
    verifies its existing bindings before adding the scheduler template binding;
    it never accepts source/model drift as part of a policy change.
    """

    from wmloop.control.research_request import (
        ResearchRequestError,
        bind_research_input,
        research_input_digest,
        validate_research_plan,
        verify_research_plan_inputs,
    )

    validate_research_plan(plan)
    previous = plan_execution_policy(plan)
    policy = research_execution_policy(
        rounds=previous["rounds"] if rounds is None else rounds,
        stop_on_success=previous["stop_on_success"] if stop_on_success is None else stop_on_success,
    )
    if policy == previous:
        return copy.deepcopy(dict(plan))
    ready = plan.get("state") in {"ready", "ready_with_deferred_discovery"}
    if not ready:
        return _recompile_blocked_policy(plan, policy)
    if ready and verify_research_plan_inputs(plan)["state"] != "verified":
        raise ResearchRequestError("RESEARCH_PLAN_INPUT_DIGEST_DRIFT")
    result = copy.deepcopy(dict(plan))
    result.update(
        schema_version=3,
        execution_policy=policy,
        parent_plan_digest=plan["plan_digest"],
        created_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    )
    blockers = [item for item in result["blockers"] if not str(item["code"]).startswith("EVOLUTION_")]
    try:
        paths = policy_input_paths(policy, Path(str(plan["evaluator_contract"])) if plan.get("evaluator_contract") else None)
    except ResearchExecutionPolicyError as exc:
        blockers.append({"code": str(exc).split(":", 1)[0], "message": "研究计划的评测模板或冻结目标无效。", "action": "检查 evaluator 的 scheduler_template 与 frozen_goal；模板应包含 screen、gate、confirm，目标应包含冻结的 v2 验收规则。", "detail": str(exc)})
        paths = {}
    result["warnings"] = policy_warnings(policy, paths)
    result["blockers"] = blockers
    if blockers or not ready:
        result.update(state="blocked", input_bindings={}, input_digest=None)
    else:
        bindings = result["input_bindings"]
        for name in ("evolution_candidate_template", "frozen_goal"):
            bindings.pop(name, None)
        bindings.update({name: bind_research_input(path) for name, path in paths.items()})
        result["input_digest"] = research_input_digest(bindings)
    result["plan_id"] = "research-plan-" + sha256_bytes(canonical_bytes({"parent": plan["plan_digest"], "execution_policy": policy}))[:24]
    result.pop("plan_digest", None)
    result["plan_digest"] = sha256_bytes(canonical_bytes(result))
    validate_research_plan(result)
    return result


def _recompile_blocked_policy(plan: Mapping[str, object], policy: Mapping[str, object]) -> dict[str, object]:
    """An unbound readiness report needs a fresh plan before gaining authority."""

    from wmloop.control.research_request import compile_research_plan, validate_research_plan

    fields = (
        "model", "data", "goal", "budget", "mode", "adapter", "source",
        "evaluator_contract", "runtime_python", "adapter_profile", "target_metrics",
        "model_irg", "irg_protected_metrics", "state_root", "project_root",
    )
    mode_plan = plan.get("research_mode_plan")
    context = mode_plan.get("evidence_context") if isinstance(mode_plan, Mapping) else None
    result = compile_research_plan(
        **{name: plan[name] for name in fields},
        protected_metrics=plan.get("protected_metrics_requested"),
        rounds=policy["rounds"], stop_on_success=policy["stop_on_success"],
        evidence_context=context if isinstance(context, Mapping) else None,
    )
    result["parent_plan_digest"] = plan["plan_digest"]
    result.pop("plan_digest")
    result["plan_digest"] = sha256_bytes(canonical_bytes(result))
    validate_research_plan(result)
    return result
