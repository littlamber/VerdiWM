"""Human-facing project initialization for first-contact model research."""

from __future__ import annotations

import os
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping


class FirstContactError(ValueError):
    """The first-contact project inputs are missing or unsafe."""


def explain_blocker(error: BaseException | str) -> dict[str, str]:
    """Translate stable internal blocker codes into actionable user language."""
    detail = str(error)
    code = detail.split(":", 1)[0]
    exact = {
        "GOAL_REQUIRED": "请填写你想改善的能力。",
        "MODEL_REQUIRED": "还没有可用的模型目录，请先完成首次设置。",
        "MODEL_PATH_REQUIRED": "还没有可用的模型目录，请先完成首次设置。",
        "DATA_REQUIRED": "还没有可用的数据目录，请先完成首次设置。",
        "DATASET_REQUIRED": "还没有可用的数据目录，请先完成首次设置。",
        "PROJECT_CONFIG_INVALID": "项目配置无法读取，请检查 verdiwm.toml。",
        "PROJECT_CONFIG_NOT_FOUND": "还没有项目配置，请先完成首次设置。",
    }
    message = exact.get(code)
    if message is None and any(token in detail for token in ("TARGET_METRIC", "METRIC_")):
        message = "填写的成功指标不在当前评测清单中。请使用模型原有评测指标，或先补充评测方法。"
    if message is None and any(token in detail for token in ("ADAPTER", "PROFILE")):
        message = "这个模型还没有可用的运行连接器。系统已安全停止且没有占用 GPU；请确认模型平时如何启动和评测。"
    if message is None and any(token in detail for token in ("CONFIG_NOT_FOUND", "PROVIDER", "LLM_")):
        message = "当前安装尚未配置生成模型连接器所需的代码服务。请联系部署管理员，不要在页面粘贴密钥。"
    if message is None:
        message = "当前输入还不能安全开始实验。请检查模型、数据、评测方法和预算。"
    return {"code": code, "error": message, "detail": detail}


def discover_project_inputs(root: Path) -> tuple[Path | None, Path | None]:
    """Find conventional model and dataset directories without importing them."""
    base = Path(root).expanduser().resolve()
    model = next((base / name for name in ("model", "models") if (base / name).is_dir()), None)
    data = next((base / name for name in ("data", "dataset", "datasets") if (base / name).exists()), None)
    return model, data


def initialize_project(
    *,
    root: Path,
    model: str | None = None,
    data: str | None = None,
    goal: str | None = None,
    budget: str = "1gpu-hour",
    mode: str = "hybrid",
    target_metrics: list[str] | None = None,
    project_file: Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Create a minimal project file from user concepts, not internal contracts."""
    base = Path(root).expanduser().resolve()
    discovered_model, discovered_data = discover_project_inputs(base)
    model_path = Path(model).expanduser().resolve() if model else discovered_model
    data_path = Path(data).expanduser().resolve() if data else discovered_data
    errors: list[dict[str, str]] = []
    if model_path is None:
        errors.append({"field": "model", "message": "找不到模型目录；请用 --model 指定模型所在目录。"})
    elif not model_path.is_dir() or model_path.is_symlink():
        errors.append({"field": "model", "message": f"模型目录不可用：{model_path}"})
    if data_path is None:
        errors.append({"field": "data", "message": "找不到数据目录；请用 --data 指定数据所在目录。"})
    elif not data_path.exists() or data_path.is_symlink():
        errors.append({"field": "data", "message": f"数据路径不可用：{data_path}"})
    if not isinstance(goal, str) or not goal.strip():
        errors.append({"field": "goal", "message": "还需要一句话说明你想改善什么，例如：提升长时域预测稳定性。"})
    if errors:
        return {
            "schema_version": 1,
            "artifact_type": "verdiwm-first-contact-check",
            "state": "needs_input",
            "root": str(base),
            "discovered": {"model": str(discovered_model) if discovered_model else None, "data": str(discovered_data) if discovered_data else None},
            "missing": errors,
            "next_step": "补齐上面标出的信息后再次运行 verdiwm init。",
        }
    destination = (project_file or (base / "verdiwm.toml")).expanduser().resolve()
    if destination.exists() and not force:
        raise FirstContactError(f"PROJECT_FILE_EXISTS:{destination}:use --force to replace it")
    def relative(path: Path) -> str:
        try:
            return os.path.relpath(path, destination.parent)
        except ValueError:
            return str(path)
    runtime = next(
        (
            candidate
            for candidate in (
                model_path / ".venv" / "bin" / "python",
                model_path / "venv" / "bin" / "python",
            )
            if candidate.is_file() and not candidate.is_symlink()
        ),
        None,
    )
    values: list[tuple[str, object]] = [
        ("model", relative(model_path).replace(chr(92), "/")),
        ("data", relative(data_path).replace(chr(92), "/")),
        ("goal", goal.strip()),
        ("budget", budget),
        ("mode", mode),
        ("state_root", ".verdiwm/state"),
    ]
    metrics = [item.strip() for item in (target_metrics or []) if item.strip()]
    if metrics:
        values.append(("target_metrics", metrics))
    if runtime is not None:
        values.append(("runtime_python", relative(runtime).replace(chr(92), "/")))
    body = (
        "# VerdiWM project settings. Model and data paths are local and never uploaded.\n"
        "[project]\n"
        + "".join(f"{key} = {json.dumps(value, ensure_ascii=False)}\n" for key, value in values)
    )
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return {
        "schema_version": 1,
        "artifact_type": "verdiwm-first-contact-project",
        "state": "ready",
        "project_file": str(destination),
        "model": str(model_path),
        "data": str(data_path),
        "goal": goal.strip(),
        "runtime_python": str(runtime) if runtime is not None else None,
        "target_metrics": metrics,
        "next_steps": [
            "运行 verdiwm doctor 检查本地控制面。",
            "运行 verdiwm run 或打开 verdiwm-workbench 开始受边界实验。",
            "如果系统缺少模型运行入口，会明确列出需要补充的信息，不会擅自修改模型。",
        ],
    }
