"""Campaign status, cancellation, reproduction, and detached-job commands."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Mapping

from wmloop.control.campaign_api import CampaignStore
from wmloop.control.campaign_dispatcher import DispatcherOptions, run_dispatcher
from wmloop.control.commands.inputs import campaign_store
from wmloop.control.commands.io import emit_json, emit_text
from wmloop.control.terminal_results import _safe_failure_cause
from wmloop.execute.job_supervisor import (
    cancel_job,
    load_status,
    resume_job,
    submit_job,
    tail_job,
)
from wmloop.experiments.job_spec import JobSpec


def _store(state_root: Path | None) -> CampaignStore:
    return campaign_store(state_root)


def _dispatch(
    store: CampaignStore, campaign_id: str, max_parallel: int
) -> dict[str, object]:
    return run_dispatcher(
        DispatcherOptions(
            state_root=store.root,
            max_cycles=1,
            max_parallel=max_parallel,
            campaign_ids=(campaign_id,),
        )
    )


def status(args: argparse.Namespace) -> int:
    store = _store(args.state_root)
    if args.campaign_id:
        record = store.get(args.campaign_id)
        if getattr(args, "format", "json") == "text":
            emit_text(_campaign_status_text(record))
        else:
            emit_json(record)
    else:
        records = store.list(status=args.status, limit=args.limit)
        if getattr(args, "format", "json") == "text":
            emit_text(_campaign_list_text(records))
        else:
            emit_json(
                {
                    "schema_version": 2,
                    "artifact_type": "verdiwm-campaign-list",
                    "items": records,
                }
            )
    return 0


_STATUS_TEXT_MAX_CAMPAIGNS = 20
_STATUS_TEXT_MAX_CHARS = 16_000
_STATUS_TEXT_MAX_GOAL = 140
_STATUS_TEXT_MAX_FAILURE_CAUSE = 220
_STATUS_TEXT_MAX_PATH = 180
_ABSOLUTE_PATH = re.compile(
    r"(?<![\w.])(?:[A-Za-z]:[\\/]|/)(?:[^\s<>\"']+[\\/])*[^\s<>\"']*"
)
_ERROR_CODE = re.compile(r"[A-Z][A-Z0-9_]{2,79}")
_STAGE_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{0,47}")


def _bounded_status_value(value: object, *, limit: int) -> str:
    """Render a short single-line value while removing path-shaped data."""

    if value is None:
        return ""
    text = str(value)
    text = re.sub(r"[\x00-\x1f\x7f]", " ", text)
    text = _ABSOLUTE_PATH.sub("[路径已隐藏]", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > limit:
        text = text[: max(0, limit - 1)].rstrip() + "…"
    return text


def _bounded_receipt_path(value: str) -> str:
    """Keep the requested receipt locator while making it single-line/bounded."""

    text = re.sub(r"[\x00-\x1f\x7f]", " ", value)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > _STATUS_TEXT_MAX_PATH:
        text = text[: _STATUS_TEXT_MAX_PATH - 1].rstrip() + "…"
    return text


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _campaign_rounds_and_stop(record: Mapping[str, Any]) -> tuple[str, str]:
    result = _mapping(record.get("execution_result"))
    manifest = _mapping(result.get("evolution_manifest"))
    rounds = result.get("rounds_completed", manifest.get("settled_iterations"))
    if isinstance(rounds, int) and not isinstance(rounds, bool) and rounds >= 0:
        rounds_text = str(rounds) if rounds <= 999_999 else "999999+"
    else:
        rounds_text = "尚无结果"
    reason = result.get("stop_reason", manifest.get("stop_reason"))
    if reason is not None:
        stop_text = _bounded_status_value(reason, limit=72)
    else:
        # A campaign without an execution result has not reached the
        # controller's stop-policy stage yet.  Cancellation is already a
        # terminal user decision, so describing it as "not stopped" is
        # misleading when the record is viewed immediately after cancel.
        status = str(record.get("status", ""))
        stop_text = {
            "cancelled": "用户取消",
            "created": "尚未开始",
            "confirmed": "尚未开始",
            "queued": "尚未开始",
            "running": "执行中",
        }.get(status, "尚未停止")
    return rounds_text, stop_text


def _campaign_failure(record: Mapping[str, Any]) -> tuple[str, str]:
    error = _mapping(record.get("execution_error"))
    result = _mapping(record.get("execution_result"))
    code = ""
    for key in ("code", "error_code", "message"):
        raw = error.get(key)
        if isinstance(raw, str):
            candidate = raw.split(":", 1)[0].strip()
            if _ERROR_CODE.fullmatch(candidate):
                code = candidate
                break
    stage = error.get("stage")
    if not isinstance(stage, str) or not _STAGE_NAME.fullmatch(stage):
        stages = result.get("stages")
        stage = None
        if isinstance(stages, list):
            for item in stages[:100]:
                row = _mapping(item)
                if row.get("returncode") not in (None, 0):
                    candidate = row.get("stage")
                    if isinstance(candidate, str) and _STAGE_NAME.fullmatch(candidate):
                        stage = candidate
                        break
        if stage is None and code:
            match = re.search(r"_(TRAIN|EVALUATE|SCREEN|CONFIRM|ARCHIVE)_", code)
            stage = match.group(1).lower() if match else None
    return (
        _bounded_status_value(stage, limit=48) if isinstance(stage, str) else "未记录",
        code or "CAMPAIGN_EXECUTION_FAILED",
    )


def _receipt_path(record: Mapping[str, Any]) -> str | None:
    error = _mapping(record.get("execution_error"))
    result = _mapping(record.get("execution_result"))
    receipt: str | None = None
    for source in (error, result):
        for key in ("receipt_path", "execution_receipt", "execution_receipt_path"):
            value = source.get(key)
            if isinstance(value, str) and value.strip():
                receipt = value.strip()
                break
        if receipt is not None:
            break
    if receipt is None:
        execution = _mapping(record.get("execution"))
        output_root = execution.get("output_root")
        if isinstance(output_root, str) and output_root.strip():
            receipt = str(Path(output_root) / "execution-receipt.json")
    if receipt is None:
        return None

    # A receipt path nested under the model or dataset is not a useful public
    # locator and can reveal private input layout. Keep the summary safe.
    try:
        receipt_path = Path(receipt).expanduser().resolve()
        for key in ("model", "dataset"):
            root = record.get(key)
            if not isinstance(root, str) or not root.strip():
                continue
            try:
                receipt_path.relative_to(Path(root).expanduser().resolve())
            except ValueError:
                continue
            return "[模型/数据目录内的回执，路径已隐藏]"
        receipt = str(receipt_path)
    except (OSError, RuntimeError, ValueError):
        return "[回执路径已隐藏]"
    return _bounded_receipt_path(receipt)


def _campaign_status_lines(
    record: Mapping[str, Any], *, include_receipt: bool = True
) -> list[str]:
    campaign_id = _bounded_status_value(record.get("campaign_id", "?"), limit=56) or "?"
    state = _bounded_status_value(record.get("status", "unknown"), limit=24) or "unknown"
    goal = _bounded_status_value(record.get("goal", "未记录"), limit=_STATUS_TEXT_MAX_GOAL) or "未记录"
    rounds, stop = _campaign_rounds_and_stop(record)
    lines = [
        f"Campaign: {campaign_id}",
        f"  状态: {state}",
        f"  目标: {goal}",
        f"  轮数: {rounds} · 停止原因: {stop}",
    ]
    if state == "failed":
        stage, code = _campaign_failure(record)
        lines.append(f"  失败阶段: {stage} · 错误码: {code}")
        cause = _safe_failure_cause(dict(record))
        if cause:
            lines.append(
                "  根因: "
                + _bounded_status_value(cause, limit=_STATUS_TEXT_MAX_FAILURE_CAUSE)
            )
        else:
            lines.append("  根因: 未能从阶段摘要提取；请检查完整回执及训练器子进程日志。")
    if include_receipt:
        receipt = _receipt_path(record)
        if receipt is not None:
            lines.append(f"  回执: {receipt}")
    return lines


def _campaign_status_text(
    record: Mapping[str, Any], *, include_receipt: bool = True
) -> str:
    return "\n".join(
        _campaign_status_lines(record, include_receipt=include_receipt)
    ) + "\n"


def _campaign_list_text(records: list[dict[str, Any]]) -> str:
    if not records:
        return "没有找到符合条件的 campaign。\n"
    shown = records[:_STATUS_TEXT_MAX_CAMPAIGNS]
    lines: list[str] = []
    for index, record in enumerate(shown):
        if index:
            lines.append("")
        lines.extend(_campaign_status_lines(record))
    remaining = len(records) - len(shown)
    if remaining > 0:
        lines.append(f"\n另有 {remaining} 个 campaign 未显示；可用 --limit 调整查询范围。")
    output = "\n".join(lines) + "\n"
    if len(output) > _STATUS_TEXT_MAX_CHARS:
        output = output[:_STATUS_TEXT_MAX_CHARS - 2].rstrip() + "…\n"
    return output


def campaign_result_text(payload: Mapping[str, Any]) -> str:
    """Render a bounded public summary for ``run`` and ``cancel``.

    The machine response remains the complete campaign record.  The default
    terminal response deliberately omits receipt locators and nested execution
    evidence so a normal command does not dump private paths or sample IDs.
    """

    campaign = payload.get("campaign")
    if not isinstance(campaign, Mapping):
        campaign = payload
    if not isinstance(campaign, Mapping):
        return "命令已完成，但没有可显示的 campaign 摘要。\n"
    lines = _campaign_status_lines(campaign, include_receipt=False)
    campaign_id = campaign.get("campaign_id")
    if isinstance(campaign_id, str) and campaign_id:
        lines.append(f"  查看: verdi status {campaign_id} --format text")
    return "\n".join(lines) + "\n"


def cancel(args: argparse.Namespace) -> int:
    result = _store(args.state_root).cancel(args.campaign_id)
    if getattr(args, "format", None) == "text":
        emit_text(campaign_result_text(result))
    else:
        emit_json(result)
    return 0


def job(args: argparse.Namespace) -> int:
    if args.job_action == "submit":
        metadata = json.loads(args.metadata)
        if not isinstance(metadata, dict):
            raise ValueError("JOB_METADATA_INVALID")
        command = tuple(args.job_command)
        if command and command[0] == "--":
            command = command[1:]
        emit_json(
            submit_job(
                JobSpec(
                    command=command,
                    cwd=args.cwd,
                    job_root=args.job_root,
                    environment=dict(args.env or ()),
                    output_root=args.output_root,
                    timeout_seconds=args.timeout_seconds,
                    metadata=metadata,
                )
            )
        )
        return 0
    if args.job_action == "status":
        emit_json(load_status(args.job_root))
        return 0
    if args.job_action == "tail":
        emit_text(tail_job(args.job_root, lines=args.lines))
        return 0
    if args.job_action == "cancel":
        emit_json(cancel_job(args.job_root, grace_seconds=args.grace_seconds))
        return 0
    emit_json(resume_job(args.job_root))
    return 0


def reproduce(args: argparse.Namespace) -> int:
    store = _store(args.state_root)
    campaign = store.reproduce(args.campaign_id)
    if args.queue_only:
        emit_json(campaign)
        return 0
    dispatcher = _dispatch(store, str(campaign["campaign_id"]), args.max_parallel)
    reproduced = store.get(str(campaign["campaign_id"]))
    emit_json({"campaign": reproduced, "dispatcher": dispatcher})
    return 0 if reproduced.get("status") in {"completed", "cancelled"} else 2
