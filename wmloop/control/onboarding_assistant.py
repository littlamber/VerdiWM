"""Human-facing onboarding questions for unfamiliar model repositories.

This module is deliberately provider-neutral. An agent such as Codex may use
the questionnaire to inspect a repository and draft answers, while VerdiWM
keeps the answers explicit and requires confirmation before execution.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from wmloop.control.first_contact import inspect_project


def build_onboarding_questionnaire(
    *,
    root: Path,
    model: str | None = None,
    data: str | None = None,
    goal: str | None = None,
    evaluator_contract: str | None = None,
    runtime_python: str | None = None,
) -> dict[str, Any]:
    """Build a deterministic question set from read-only project evidence."""
    readiness = inspect_project(
        root=root,
        model=model,
        data=data,
        evaluator_contract=evaluator_contract,
        runtime_python=runtime_python,
    )
    questions: list[dict[str, Any]] = []
    seen: set[str] = set()
    templates = {
        "MODEL_PATH_REQUIRED": {
            "title": "模型代码在哪里？",
            "prompt": "请提供包含模型代码的目录路径。",
            "answer_type": "path",
            "example": "./model",
        },
        "DATA_PATH_REQUIRED": {
            "title": "数据集在哪里？",
            "prompt": "请提供训练或评测数据所在的目录或文件路径。",
            "answer_type": "path",
            "example": "./data",
        },
        "RUNTIME_UNREADY": {
            "title": "模型用哪个 Python 环境运行？",
            "prompt": "请提供已经安装模型依赖的 Python 可执行文件路径。",
            "answer_type": "path",
            "example": "./model/.venv/bin/python",
        },
        "CHECKPOINT_MISSING": {
            "title": "模型权重在哪里？",
            "prompt": "请提供 checkpoint 或权重文件路径；如果模型不需要权重，请说明原因。",
            "answer_type": "path_or_text",
            "example": "./weights/checkpoint.pt",
        },
        "EVALUATION_ENTRYPOINT_MISSING": {
            "title": "怎样评测一次模型？",
            "prompt": "请提供评测命令，或说明哪个脚本负责评测以及它输出什么文件。",
            "answer_type": "text",
            "example": "python evaluate.py --checkpoint {checkpoint} --output {output}",
        },
        "EVALUATOR_CONTRACT_REQUIRED": {
            "title": "怎样判断模型变好了？",
            "prompt": "请提供已有评测契约，或确认评测命令、指标名称和指标方向。系统不会替你猜测成功标准。",
            "answer_type": "path_or_text",
            "example": "./contracts/evaluator.json；指标 success_rate，越高越好",
        },
        "SOURCE_REVISION_UNBOUND": {
            "title": "模型代码使用哪个版本？",
            "prompt": "请在模型仓库提交一个 Git 版本，或明确固定一个源码版本。",
            "answer_type": "text",
            "example": "git commit abc1234",
        },
        "MODEL_ASSET_BINDING_REQUIRED": {
            "title": "评测还需要哪些文件？",
            "prompt": "请补充评测命令所需的模型、数据或依赖文件路径。",
            "answer_type": "path_or_text",
            "example": "./models/encoder；./data/stats.json",
        },
    }
    for blocker in readiness.get("blockers", []):
        if not isinstance(blocker, Mapping):
            continue
        code = str(blocker.get("code", ""))
        if code in seen:
            continue
        seen.add(code)
        template = templates.get(code, {
            "title": "还需要确认一项接入信息",
            "prompt": str(blocker.get("message") or "请补充检查结果中缺少的信息。"),
            "answer_type": "text",
            "example": "请按模型项目的实际情况填写",
        })
        questions.append({
            "id": code.lower(),
            "code": code,
            "required": True,
            **template,
            "why": str(blocker.get("action") or "这项信息用于生成可复现的运行配置。"),
        })
    if not questions:
        questions.append({
            "id": "confirm_launch_contract",
            "code": "CONFIRM_LAUNCH_CONTRACT",
            "title": "确认接入配置",
            "prompt": "模型入口、运行环境和评测方法已发现。请确认这些信息后再进行 CPU 安全检查。",
            "answer_type": "confirmation",
            "required": True,
            "why": "确认只会固化已发现的信息，不会修改模型代码。",
        })
    return {
        "schema_version": 1,
        "artifact_type": "verdiwm-onboarding-questionnaire",
        "state": "awaiting_answers" if readiness.get("blockers") else "awaiting_confirmation",
        "project": {
            "root": str(Path(root).expanduser().resolve()),
            "model": readiness.get("model"),
            "data": readiness.get("data"),
            "goal": goal.strip() if isinstance(goal, str) else None,
        },
        "readiness": readiness,
        "questions": questions,
        "agent_guidance": {
            "can_inspect_source": True,
            "can_draft_files": True,
            "must_request_confirmation_for": ["evaluation semantics", "metric thresholds", "source changes", "GPU launch"],
            "credentials": "Never request or store API keys in this questionnaire.",
        },
        "next_step": "请回答上面的问题；回答会生成草稿，评测标准仍需明确确认。",
    }


def write_onboarding_questionnaire(path: Path, questionnaire: Mapping[str, object]) -> Path:
    """Write a questionnaire atomically, keeping it outside model source by policy."""
    destination = Path(path).expanduser().resolve()
    project = questionnaire.get("project")
    model_root = (
        Path(str(project.get("model"))).expanduser().resolve()
        if isinstance(project, Mapping) and project.get("model")
        else None
    )
    if model_root is not None and (destination == model_root or model_root in destination.parents):
        raise ValueError("ONBOARDING_QUESTIONNAIRE_INSIDE_MODEL")
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(dict(questionnaire), handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return destination
