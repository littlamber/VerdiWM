"""Present existing CLI result contracts without changing their authority."""

from __future__ import annotations

import json
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
    for key, label in fields:
        if payload.get(key) is not None:
            _row(label, payload[key], stdout=stdout)
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
    campaign = payload.get("campaign")
    if isinstance(campaign, dict):
        _row("Campaign", campaign.get("campaign_id", "?"), stdout=stdout)
        _row("Status", campaign.get("status", "unknown"), stdout=stdout)
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
    }
    if not known.intersection(payload):
        stdout.write(text + "\n")
