"""Present existing CLI result contracts without changing their authority."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable, Protocol, TextIO

from wmloop.control.terminal_ui import plain_text, terminal_size, wrap_lines


class ResultTheme(Protocol):
    def accent(self, value: str) -> str: ...
    def muted(self, value: str) -> str: ...
    def good(self, value: str) -> str: ...
    def warn(self, value: str) -> str: ...
    def bad(self, value: str) -> str: ...


def _value(value: object) -> str:
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return plain_text(value)


def _line(value: object, *, stdout: TextIO, painter: Callable[[str], str] | None = None) -> None:
    for part in wrap_lines(value, terminal_size(stdout).columns - 1):
        stdout.write((painter(part) if painter else part) + "\n")


def _row(label: str, value: object, *, stdout: TextIO, painter: Callable[[str], str] | None = None) -> None:
    _line(f"  {label:<10} {_value(value)}", stdout=stdout, painter=painter)


def _warnings(payload: dict[str, object], *, stdout: TextIO, painter: Callable[[str], str] | None = None) -> None:
    warnings = payload.get("warnings")
    if isinstance(warnings, list):
        for warning in warnings:
            _row("Warning", warning, stdout=stdout, painter=painter)


def render_execution_policy(payload: dict[str, object], *, stdout: TextIO) -> None:
    """Show the reviewed round bound and the verifier's stopping authority."""

    policy = payload.get("execution_policy")
    if not isinstance(policy, dict):
        policy = {"rounds": 1, "stop_on_success": False}
    _row("Rounds", f"最多 {policy.get('rounds', 1)} 轮", stdout=stdout)
    budget = payload.get("budget")
    if isinstance(budget, dict) and budget.get("gpu_hours") is not None:
        _row("Total GPUh", f"{budget['gpu_hours']}（所有轮次共享）", stdout=stdout)
    stops = "轮数上限、总预算耗尽、失败上限或用户取消"
    if policy.get("stop_on_success") is True:
        stops += "；冻结验证器正式确认收益后提前停止"
    else:
        stops += "；确认收益后仍继续，直到其他停止条件触发"
    _row("Stop", stops, stdout=stdout)
    _warnings(payload, stdout=stdout)


def _reason_text(reason: object, *, goal: bool = False) -> str:
    messages = {
        "MAX_ITERATIONS_REACHED": "达到设定轮数上限",
        "MAX_ITERATIONS_REACHED_WITH_FAILURES": "达到设定轮数上限，仍有未成功完成的实验，请检查失败详情",
        "GLOBAL_BUDGET_EXHAUSTED": "所有轮次共享的总预算已耗尽",
        "CONSECUTIVE_FAILURE_LIMIT": "达到连续失败上限",
        "NO_NEW_INFORMATION_LIMIT": "达到诊断签名重复上限",
        "GOAL_CONFIRMED": "目标确认停止条件触发，提前停止",
        "GOAL_CONFIRMATION_INVALIDATED": "恢复时正式达标证据未通过复核",
        "SIGNAL_RECEIVED": "收到取消请求或停止信号",
        "CONTROLLER_ERROR": "研究控制器遇到错误",
        "INNER_MAX_CYCLES_REACHED": "本轮内部调度达到轮询上限，可检查后恢复",
        "INNER_PIPELINE_STOPPED": "本轮执行流程已停止",
        "INNER_PIPELINE_INTERRUPTED": "本轮执行流程被中断",
        "FROZEN_GOAL_NOT_CONFIGURED": "未绑定 evaluator 的 frozen_goal，无法正式确认目标达成",
        "FROZEN_GOAL_POLICY_REQUIRED": "缺少有效的冻结目标判定策略",
        "CONFIRM_EVIDENCE_MISSING": "本轮缺少确认阶段证据",
        "SETTLED_EVIDENCE_MISSING": "缺少可复核的已结算证据库",
        "FORMAL_GOAL_EVIDENCE_NOT_CONFIRMED": "本轮证据尚未满足冻结目标的正式确认条件",
        "FROZEN_GOAL_CONFIRMED": "冻结验证器已正式确认目标收益",
        "CONFIRM_ITERATION_BINDING_INVALID": "确认阶段证据与当前轮次的绑定无效",
        "CONFIRM_PLAN_BINDING_INVALID": "确认阶段证据与计划的绑定无效",
        "GOAL_EVIDENCE_INVALID": "正式目标证据无效或无法读取",
    }
    code = str(reason)
    message = messages.get(code.split(":", 1)[0], "尚未正式确认目标收益" if goal else "执行已停止，请检查任务详情")
    return f"{message}（{code}）"


def render_execution_outcome(payload: dict[str, object], *, stdout: TextIO, theme: ResultTheme) -> None:
    """Show controller outcomes for status records and research-run results.

    Completion and stop reasons describe execution. Only an explicit verifier
    confirmation in the reported goal status is presented as a confirmed gain.
    """

    result = payload.get("execution_result")
    if not isinstance(result, dict):
        result = payload
    manifest = result.get("evolution_manifest")
    if not isinstance(manifest, dict):
        manifest = {}
    reason = result.get("stop_reason", manifest.get("stop_reason"))
    rounds = result.get("rounds_completed", manifest.get("settled_iterations"))
    if rounds is None and payload.get("artifact_type") == "verdiwm-evolution-daemon-manifest":
        rounds = payload.get("settled_iterations")
    if rounds is not None:
        _row("Finished", f"已完成 {rounds} 轮", stdout=stdout)
    if reason is not None:
        _row("Stopped", _reason_text(reason), stdout=stdout)
    goal = result.get("goal_success", manifest.get("goal_success"))
    if isinstance(goal, dict):
        confirmed = goal.get("state") == "confirmed"
        _row(
            "Goal", "冻结验证器已正式确认目标收益（confirmed）" if confirmed else "尚未正式确认目标收益（not_confirmed）",
            stdout=stdout, painter=theme.good if confirmed else theme.warn,
        )
        if goal.get("reason") is not None:
            _row("Goal why", _reason_text(goal["reason"], goal=True), stdout=stdout)
    elif reason is not None or rounds is not None:
        _row("Goal", "未报告正式目标确认状态；执行完成不代表模型达标", stdout=stdout, painter=theme.warn)
    if result is not payload:
        _warnings(result, stdout=stdout, painter=theme.warn)
    _warnings(manifest, stdout=stdout, painter=theme.warn)


def _campaign_failure_code(error: object) -> str:
    """Return a bounded stable code without exposing arbitrary exception text."""

    if isinstance(error, dict):
        for key in ("code", "error_code", "message"):
            value = error.get(key)
            if not isinstance(value, str):
                continue
            candidate = value.split(":", 1)[0]
            if re.fullmatch(r"[A-Z][A-Z0-9_]{2,95}", candidate):
                return candidate
    return "CAMPAIGN_EXECUTION_FAILED"


def _campaign_failure_stage(payload: dict[str, object], code: str) -> str:
    error = payload.get("execution_error")
    if isinstance(error, dict) and isinstance(error.get("stage"), str):
        return str(error["stage"])
    result = payload.get("execution_result")
    if isinstance(result, dict):
        stages = result.get("stages")
        if isinstance(stages, list):
            for stage in stages:
                if isinstance(stage, dict) and stage.get("returncode") not in (None, 0):
                    name = stage.get("stage")
                    if isinstance(name, str) and name:
                        return name
    match = re.search(r"_(TRAIN|EVALUATE|SCREEN|CONFIRM|ARCHIVE)_", code)
    if match:
        return match.group(1).lower()
    return "未记录（查看回执）"


def _campaign_receipt_path(payload: dict[str, object]) -> str | None:
    error = payload.get("execution_error")
    if isinstance(error, dict):
        for key in ("receipt_path", "execution_receipt", "execution_receipt_path"):
            value = error.get(key)
            if isinstance(value, str) and value.strip():
                return value
    result = payload.get("execution_result")
    if isinstance(result, dict):
        for key in ("receipt_path", "execution_receipt", "execution_receipt_path"):
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                return value
    execution = payload.get("execution")
    output_root = execution.get("output_root") if isinstance(execution, dict) else None
    if isinstance(output_root, str) and output_root.strip():
        return str(Path(output_root) / "execution-receipt.json")
    return None


def _campaign_receipt_stages(payload: dict[str, object]) -> list[dict[str, object]]:
    result = payload.get("execution_result")
    stages = result.get("stages") if isinstance(result, dict) else None
    if isinstance(stages, list):
        rows = [row for row in stages if isinstance(row, dict)]
        if rows:
            return rows

    execution = payload.get("execution")
    output_root = execution.get("output_root") if isinstance(execution, dict) else None
    if not isinstance(output_root, str) or not output_root.strip():
        return []
    receipt_path = Path(output_root) / "execution-receipt.json"
    try:
        if receipt_path.is_symlink() or not receipt_path.is_file():
            return []
        if receipt_path.stat().st_size > 2_000_000:
            return []
        document = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(document, dict) or document.get("artifact_type") != "verdiwm-model-run-execution-receipt":
        return []
    stages = document.get("stages")
    if not isinstance(stages, list):
        return []
    return [row for row in stages if isinstance(row, dict)]


def _safe_failure_cause(payload: dict[str, object]) -> str | None:
    stages = _campaign_receipt_stages(payload)
    failed_stage = _campaign_failure_stage(payload, _campaign_failure_code(payload.get("execution_error")))
    stage = next(
        (
            row for row in stages
            if row.get("stage") == failed_stage and row.get("returncode") not in (None, 0)
        ),
        next((row for row in stages if row.get("returncode") not in (None, 0)), None),
    )
    if stage is None:
        return None
    candidates = [stage.get(key) for key in ("stderr_tail", "stderr_head", "stdout_tail")]
    text = "\n".join(value for value in candidates if isinstance(value, str))
    lines = [plain_text(line).strip() for line in text.splitlines() if line.strip()]
    lines = [
        line for line in lines
        if "ChildFailedError" not in line
        and "error_file=N/A" not in line
        and "Root Cause:" not in line
        and not line.startswith("Traceback")
    ]
    exception_lines = [
        line for line in lines
        if re.search(r"(?:[A-Za-z_][\w.]*Error|Exception|AssertionError|SystemExit):\s*\S", line)
    ]
    if exception_lines:
        cause = exception_lines[-1]
    else:
        cause = next((line for line in reversed(lines) if not line.startswith("Torch Elastic worker:")), "")
    if not cause:
        return None
    cause = re.sub(
        r"(?i)\b([A-Za-z_][\w-]*[_-])?(api[_-]?key|token|password|secret)\b\s*[:=]\s*([^\s,;]+)",
        lambda match: f"{match.group(1) or ''}{match.group(2)}=[redacted]",
        cause,
    )
    cause = re.sub(r"(?i)\bBearer\s+\S+", "Bearer [redacted]", cause)
    cause = re.sub(r"\b(?:sk-[A-Za-z0-9_-]{12,}|gh[pousr]_[A-Za-z0-9_]{12,}|github_pat_[A-Za-z0-9_]{12,}|hf_[A-Za-z0-9]{12,})\b", "[redacted]", cause)
    cause = re.sub(r"(?<!\S)/(?:home|root|cs|tmp|mnt|private|workspace)/[^\s,:;]+", "[path]", cause)
    cause = re.sub(r"(?:[A-Za-z]:\\|\\\\)[^\s,:;]+", "[path]", cause)
    cause = re.sub(r"[\x00-\x1f\x7f]", " ", cause)
    return cause[:240]


def _render_campaign_failure(payload: dict[str, object], *, stdout: TextIO, theme: ResultTheme) -> None:
    """Show the smallest actionable failure summary available to a user."""

    if payload.get("status") != "failed" or not isinstance(payload.get("campaign_id"), str):
        return
    error = payload.get("execution_error")
    code = _campaign_failure_code(error)
    _row("Failed at", _campaign_failure_stage(payload, code), stdout=stdout, painter=theme.bad)
    _row("Error code", code, stdout=stdout, painter=theme.bad)
    root_cause = _safe_failure_cause(payload)
    _row(
        "Root cause",
        root_cause if root_cause else "未能从阶段摘要提取根因；请检查完整回执及 torch Elastic 子进程日志。",
        stdout=stdout,
        painter=theme.bad if root_cause else theme.warn,
    )
    receipt = _campaign_receipt_path(payload)
    if receipt is not None:
        _row("Receipt", receipt, stdout=stdout)
        next_step = "如根因摘要不足，查看完整回执中该阶段 stderr 和 logs/torch-elastic 子进程日志；修复后用 /run 创建新 campaign。"
    else:
        next_step = "查看 campaign 的 dispatch 记录定位回执，再修复根因后用 /run 创建新 campaign。"
    _line(f"  Next       {next_step}", stdout=stdout, painter=theme.accent)


def _issues(payload: dict[str, object], *, stdout: TextIO, theme: ResultTheme, shown_actions: set[str]) -> None:
    for key, label in (("blockers", "Blockers"), ("errors", "Errors")):
        issues = payload.get(key)
        if not isinstance(issues, list) or not issues:
            continue
        _line(f"  {label} ({len(issues)})", stdout=stdout, painter=theme.warn)
        for issue in issues:
            if not isinstance(issue, dict):
                _line(f"    - {_value(issue)}", stdout=stdout)
                continue
            code = issue.get("code", issue.get("field", "BLOCKED" if key == "blockers" else "ERROR"))
            message = issue.get("message", "")
            _line(f"    - {code}: {_value(message)}", stdout=stdout)
            detail = issue.get("detail")
            if detail is not None and detail != message:
                _line(f"      Detail  {_value(detail)}", stdout=stdout)
            if issue.get("action"):
                action = _value(issue["action"])
                if action not in shown_actions:
                    _line(f"      Action  {action}", stdout=stdout, painter=theme.accent)
                    shown_actions.add(action)


def _checks(payload: dict[str, object], *, stdout: TextIO, theme: ResultTheme) -> None:
    checks = payload.get("checks")
    if not isinstance(checks, list) or not checks:
        return
    passed = sum(isinstance(check, dict) and check.get("state") == "pass" for check in checks)
    _line(f"  Checks     {passed}/{len(checks)} 通过", stdout=stdout, painter=theme.muted)
    for check in checks:
        if not isinstance(check, dict):
            continue
        state = str(check.get("state", "unknown"))
        if state == "pass":
            continue
        name = check.get("name", "check")
        suffix = " (optional)" if check.get("required") is False else ""
        line = f"    {state:<9} {name}{suffix}"
        _line(line, stdout=stdout, painter=theme.good if state == "pass" else theme.warn)
        for key in ("detail", "path", "action"):
            if check.get(key) is not None:
                _line(f"      {key.capitalize():<7} {_value(check[key])}", stdout=stdout)


def _questions(payload: dict[str, object], *, stdout: TextIO, theme: ResultTheme) -> None:
    questions = payload.get("questions")
    if not isinstance(questions, list) or not questions:
        return
    _line(f"  接入问卷 ({len(questions)})", stdout=stdout, painter=theme.accent)
    for index, question in enumerate(questions, start=1):
        if not isinstance(question, dict):
            continue
        _line(f"    {index}. {question.get('title', question.get('id', '需要确认的信息'))}", stdout=stdout)
        for key, label in (("prompt", "问题"), ("why", "用途"), ("example", "示例")):
            if question.get(key):
                _line(f"       {label}  {_value(question[key])}", stdout=stdout)


def _next_steps(payload: dict[str, object], *, stdout: TextIO, theme: ResultTheme, shown_actions: set[str]) -> None:
    steps: list[object] = []
    if payload.get("action"):
        steps.append(payload["action"])
    if payload.get("next_step"):
        steps.append(payload["next_step"])
    if isinstance(payload.get("next_steps"), list):
        steps.extend(payload["next_steps"])
    for step in steps:
        text = _value(step)
        if not text or text in shown_actions:
            continue
        shown_actions.add(text)
        _line(f"  Next       {text}", stdout=stdout, painter=theme.accent)


def render_result(raw: str, *, stdout: TextIO, theme: ResultTheme) -> None:
    """Render actionable fields from public command JSON; keep text help intact."""

    text = raw.strip()
    if not text:
        return
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        lines = text.splitlines()
        try:
            payload = json.loads(lines[-1])
        except json.JSONDecodeError:
            stdout.write(text + "\n")
            return
        if len(lines) > 1:
            stdout.write("\n".join(lines[:-1]) + "\n")
    if not isinstance(payload, dict):
        stdout.write(text + "\n")
        return
    state = payload.get("state", payload.get("status"))
    if payload.get("artifact_type") == "verdiwm-first-contact-project" and state == "ready":
        _row("Config", "配置已保存", stdout=stdout, painter=theme.good)
    elif state is not None:
        state_text = str(state)
        painter = theme.good if state_text in {"ready", "queued", "completed", "verified"} else theme.warn if state_text in {"blocked", "needs_input", "awaiting_confirmation", "awaiting_answers", "running"} else theme.bad
        _row("State", state_text, stdout=stdout, painter=painter)
    fields = (
        ("plan_path", "Plan"), ("project_file", "Project"),
        ("questionnaire_path", "Questions"), ("campaign_id", "Campaign"),
        ("job_id", "Job"), ("output", "Output"), ("output_root", "Output"),
        ("config_path", "Config"), ("provider", "Provider"),
        ("model", "Model"), ("source", "Source"), ("data", "Data"),
        ("runtime_python", "Runtime"), ("evaluator_contract", "Evaluator"),
        ("api_style", "API style"), ("error", "Error"),
    )
    is_campaign_record = isinstance(payload.get("campaign_id"), str) and isinstance(payload.get("status"), str)
    for key, label in fields:
        if key == "model" and is_campaign_record:
            # Campaign records retain the bound model in machine JSON. A TTY
            # status view should not print a potentially long private path.
            continue
        if payload.get(key) is not None:
            _row(label, payload[key], stdout=stdout)
    if isinstance(payload.get("execution_policy"), dict):
        render_execution_policy(payload, stdout=stdout)
    else:
        _warnings(payload, stdout=stdout, painter=theme.warn)
    render_execution_outcome(payload, stdout=stdout, theme=theme)
    _render_campaign_failure(payload, stdout=stdout, theme=theme)
    shown_actions: set[str] = set()
    _issues(payload, stdout=stdout, theme=theme, shown_actions=shown_actions)
    _checks(payload, stdout=stdout, theme=theme)
    readiness = payload.get("readiness")
    if isinstance(readiness, dict):
        _row("Readiness", readiness.get("state", "unknown"), stdout=stdout)
        _issues(readiness, stdout=stdout, theme=theme, shown_actions=shown_actions)
        _checks(readiness, stdout=stdout, theme=theme)
        _next_steps(readiness, stdout=stdout, theme=theme, shown_actions=shown_actions)
    _questions(payload, stdout=stdout, theme=theme)
    if payload.get("questions") and not payload.get("questionnaire_path"):
        _line("  问卷尚未保存；输入 /guide --output .verdiwm/onboarding-questions.json 保存。", stdout=stdout, painter=theme.accent)
    items = payload.get("items")
    if isinstance(items, list):
        _row("Campaigns", len(items), stdout=stdout)
        for item in items[:8]:
            if isinstance(item, dict):
                _line(f"    {item.get('campaign_id', '?')}  {item.get('status', item.get('state', '?'))}", stdout=stdout)
                render_execution_outcome(item, stdout=stdout, theme=theme)
                _render_campaign_failure(item, stdout=stdout, theme=theme)
    campaign = payload.get("campaign")
    if isinstance(campaign, dict):
        _row("Campaign", campaign.get("campaign_id", "?"), stdout=stdout)
        _row("Status", campaign.get("status", "unknown"), stdout=stdout)
        render_execution_outcome(campaign, stdout=stdout, theme=theme)
        _render_campaign_failure(campaign, stdout=stdout, theme=theme)
    candidates = payload.get("candidates")
    if isinstance(candidates, list):
        _line(f"  Evaluators {len(candidates)} candidates (none frozen)", stdout=stdout)
        for candidate in candidates[:8]:
            if not isinstance(candidate, dict):
                continue
            horizon = candidate.get("horizon_seconds")
            horizon_text = f"{horizon}s" if horizon is not None else "horizon?"
            marker = "minute-ready" if candidate.get("minute_level_supported") else "short/unknown"
            _line(f"    {candidate.get('candidate_id', '?')}  {horizon_text}  {marker}  {candidate.get('relative_path', '')}", stdout=stdout)
        hint = payload.get("selection_hint")
        if isinstance(hint, dict) and hint.get("candidate_id"):
            _line(f"  Hint       优先审阅 {hint['candidate_id']}；仍需明确确认。", stdout=stdout, painter=theme.accent)
    stages = payload.get("stages")
    if isinstance(stages, list):
        _row("Stages", len(stages), stdout=stdout)
    credential = payload.get("credential")
    if isinstance(credential, dict):
        # Only status metadata is presented. Credential contents never belong
        # in the terminal renderer, regardless of what another producer emits.
        _row("Credential", "configured" if credential.get("configured") else "missing/invalid", stdout=stdout)
        if credential.get("problem"):
            _row("Problem", credential["problem"], stdout=stdout)
    _next_steps(payload, stdout=stdout, theme=theme, shown_actions=shown_actions)
    known = {key for key, _ in fields} | {
        "state", "status", "blockers", "errors", "checks", "readiness", "questions",
        "items", "campaign", "candidates", "stages", "credential", "action", "next_step", "next_steps",
        "execution_policy", "execution_result", "execution_error", "evolution_manifest", "stop_reason", "rounds_completed", "goal_success", "warnings",
    }
    if not known.intersection(payload):
        stdout.write(text + "\n")
