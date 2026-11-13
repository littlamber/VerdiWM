"""Read-only system, configuration, and first-contact CLI commands."""

from __future__ import annotations

import argparse
import json
import platform
from importlib.metadata import PackageNotFoundError, metadata, version
from pathlib import Path
from typing import Any, Mapping

from packaging.specifiers import InvalidSpecifier, SpecifierSet

from wmloop.control.commands.io import emit_json, emit_text, repository_root
from wmloop.control.first_contact import inspect_project
from wmloop.control.llm_setup import inspect_llm_config, render_llm_guide
from wmloop.control.project_config import ProjectConfigError, load_project_config
from wmloop.control.research_request import ResearchRequestError


def installed_version() -> str:
    try:
        return version("verdiwm")
    except PackageNotFoundError:
        return "source"


def doctor_report(args: argparse.Namespace) -> dict[str, object]:
    root = (
        (getattr(args, "repo_root", None) or repository_root()).expanduser().resolve()
    )
    checks: list[dict[str, object]] = []
    try:
        python_requirement = metadata("verdiwm").get("Requires-Python")
        python_supported = bool(python_requirement) and SpecifierSet(python_requirement).contains(platform.python_version())
    except (PackageNotFoundError, InvalidSpecifier):
        python_requirement = None
        python_supported = False
    checks.append(
        {
            "name": "python_supported",
            "required": True,
            "state": "pass" if python_supported else "fail",
            "detail": platform.python_version(),
            "expected": python_requirement,
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
        state, detail = "pass", relative
        if not path.is_file() or path.is_symlink():
            state, detail = "fail", f"missing:{relative}"
        elif path.suffix == ".json":
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                state, detail = "fail", f"invalid_json:{relative}"
            else:
                if not isinstance(payload, dict) or (
                    expected_type is not None
                    and payload.get("artifact_type") != expected_type
                ):
                    state, detail = "fail", f"invalid_contract:{relative}"
        checks.append(
            {"name": name, "required": True, "state": state, "detail": detail}
        )
    onboarding_demo = root / "examples" / "portrait_first_minimal_loop_v1"
    checks.append(
        {
            "name": "onboarding_demo",
            "required": False,
            "state": "available" if onboarding_demo.is_dir() else "not_installed",
            "detail": "examples/portrait_first_minimal_loop_v1",
        }
    )
    blocked = any(row["required"] is True and row["state"] != "pass" for row in checks)
    return {
        "schema_version": 1,
        "artifact_type": "verdiwm-doctor-report",
        "state": "blocked" if blocked else "ready",
        "version": installed_version(),
        "checks": checks,
    }


def doctor(args: argparse.Namespace) -> int:
    report = doctor_report(args)
    emit_json(report)
    return 2 if report["state"] == "blocked" else 0


def llm(args: argparse.Namespace) -> int:
    """Show first-contact LLM guidance or inspect local configuration."""

    config_path = getattr(args, "config", None)
    if getattr(args, "llm_action", None) == "status":
        report = inspect_llm_config(config_path)
        emit_json(report)
        return 0 if report.get("state") == "ready" else 2
    emit_text(render_llm_guide(config_path=config_path) + "\n")
    return 0


def _unexpected_readiness_blocker(readiness_state: object) -> dict[str, object]:
    return {
        "code": "PROJECT_READINESS_STATE_UNEXPECTED",
        "message": "模型接入状态无法确认，系统已停止推进。",
        "action": "重新运行 verdiwm check；如果问题仍存在，请检查模型接入报告。",
        "detail": readiness_state,
    }


def check(args: argparse.Namespace) -> int:
    configured: dict[str, Any] = {}
    try:
        configured = load_project_config(cwd=Path.cwd()).values
    except ProjectConfigError as exc:
        raise ResearchRequestError(str(exc)) from exc
    readiness = inspect_project(
        root=Path.cwd(),
        model=args.model or configured.get("model"),
        source=args.source or configured.get("source"),
        data=args.data or configured.get("data", configured.get("dataset")),
        evaluator_contract=str(args.evaluator_contract)
        if args.evaluator_contract
        else configured.get("evaluator_contract"),
        runtime_python=str(args.runtime_python)
        if args.runtime_python
        else configured.get("runtime_python"),
    )
    doctor = doctor_report(args)
    checks = list(readiness.get("checks", []))
    checks.extend(
        {
            "name": f"control_plane.{row.get('name')}",
            "state": row.get("state"),
            "required": row.get("required"),
            "detail": row.get("detail"),
        }
        for row in doctor.get("checks", [])
        if isinstance(row, Mapping)
    )
    completed = [
        row.get("name")
        for row in checks
        if isinstance(row, Mapping) and row.get("state") in {"pass", "available"}
    ]
    blockers = list(readiness.get("blockers", []))
    for row in doctor.get("checks", []):
        if (
            isinstance(row, Mapping)
            and row.get("required") is True
            and row.get("state") != "pass"
        ):
            blockers.append(
                {
                    "code": f"CONTROL_PLANE_{str(row.get('name', 'CHECK')).upper()}",
                    "message": "本地控制面还没有准备好。",
                    "action": "安装 Python 3.12 和缺失的项目文件后再次运行 verdiwm check。",
                    "detail": row.get("detail"),
                }
            )
    deduped: list[dict[str, object]] = []
    seen: set[str] = set()
    for row in blockers:
        if isinstance(row, Mapping):
            code = str(row.get("code", "UNKNOWN"))
            if code not in seen:
                deduped.append(dict(row))
                seen.add(code)
    readiness_state = readiness.get("state")
    if deduped:
        state = "blocked"
    elif readiness_state in {"needs_confirmation", "ready_for_conformance"}:
        state = str(readiness_state)
    else:
        state = "blocked"
        deduped.append(_unexpected_readiness_blocker(readiness_state))
    if state == "needs_confirmation":
        next_step = str(
            readiness.get("next_step")
            or "确认评测方法和运行方式后重新运行 verdiwm check。"
        )
    elif state == "ready_for_conformance":
        next_step = (
            "运行 `verdiwm research plan` 生成并审阅计划；仅当计划状态允许时，"
            "再用 `verdiwm research run --plan <计划路径> --confirm` 明确启动。"
        )
    elif deduped:
        next_step = "按 blockers 中的 action 补齐信息，然后重新运行 verdiwm check。"
    else:
        next_step = str(readiness.get("next_step") or "重新运行 verdiwm check。")
    report = {
        "schema_version": 1,
        "artifact_type": "verdiwm-readiness-report",
        "state": state,
        "completed": completed,
        "blockers": deduped,
        "checks": checks,
        "project": readiness,
        "control_plane": doctor,
        "next_step": next_step,
    }
    emit_json(report)
    return 0 if state == "ready_for_conformance" else 2


def diagnose(args: argparse.Namespace) -> int:
    """Produce a read-only model diagnosis before any experiment is created."""

    try:
        configured = load_project_config(cwd=Path.cwd()).values
    except ProjectConfigError as exc:
        raise ResearchRequestError(str(exc)) from exc
    readiness = inspect_project(
        root=Path.cwd(),
        model=args.model or configured.get("model"),
        source=args.source or configured.get("source"),
        data=args.data or configured.get("data", configured.get("dataset")),
        evaluator_contract=str(args.evaluator_contract)
        if args.evaluator_contract
        else configured.get("evaluator_contract"),
        runtime_python=str(args.runtime_python)
        if args.runtime_python
        else configured.get("runtime_python"),
    )
    discovered = readiness.get("discovered", {})
    blockers = readiness.get("blockers", [])
    readiness_state = readiness.get("state")
    if blockers:
        state = "blocked"
    elif readiness_state == "needs_confirmation":
        state = "needs_confirmation"
    elif readiness_state == "ready_for_conformance":
        state = "ready_for_diagnostic_campaign"
    else:
        state = "blocked"
        blockers = [_unexpected_readiness_blocker(readiness_state)]
    report = {
        "schema_version": 1,
        "artifact_type": "verdiwm-model-diagnosis",
        "state": state,
        "readiness_state": readiness_state,
        "model": readiness.get("model"),
        "source": readiness.get("source"),
        "data": readiness.get("data"),
        "entrypoints": discovered.get("entrypoints", [])
        if isinstance(discovered, Mapping)
        else [],
        "assets": discovered.get("assets", [])
        if isinstance(discovered, Mapping)
        else [],
        "runtime": discovered.get("runtime", {})
        if isinstance(discovered, Mapping)
        else {},
        "source_revision": discovered.get("source_revision")
        if isinstance(discovered, Mapping)
        else None,
        "blockers": blockers,
        "next_step": (
            str(
                readiness.get("next_step")
                or "确认评测方法和运行方式后重新运行 verdiwm diagnose。"
            )
            if state == "needs_confirmation"
            else "按 blockers 中的 action 补齐信息，然后重新运行 verdiwm check。"
            if state == "blocked"
            else "运行 `verdiwm research plan` 生成并审阅计划；仅当计划状态允许时，"
            "再用 `verdiwm research run --plan <计划路径> --confirm` 明确启动。"
        ),
        "side_effects": {
            "model_import_executed": False,
            "gpu_execution_started": False,
        },
    }
    emit_json(report)
    return 0 if state == "ready_for_diagnostic_campaign" else 2
