"""Human-facing project initialization for first-contact model research."""

from __future__ import annotations

import os
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

from wmloop.control.onboarding import OnboardingError, OnboardingOptions, scan_repository


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
        "DATA_PATH_REQUIRED": "还没有可用的数据目录，请先完成首次设置。",
        "PROJECT_CONFIG_INVALID": "项目配置无法读取，请检查 verdiwm.toml。",
        "PROJECT_CONFIG_NOT_FOUND": "还没有项目配置，请先完成首次设置。",
        "PROJECT_FILE_EXISTS": "项目配置已经存在；如需替换请明确使用覆盖选项。",
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


def inspect_project(
    *,
    root: Path,
    model: str | None = None,
    data: str | None = None,
    evaluator_contract: str | None = None,
    runtime_python: str | None = None,
) -> dict[str, Any]:
    """Run a bounded, read-only readiness check for a first-contact project.

    This deliberately reports what can be proven from files. It never imports
    the model, starts a command, installs dependencies, or allocates a GPU.
    """
    base = Path(root).expanduser().resolve()
    discovered_model, discovered_data = discover_project_inputs(base)
    def resolve_input(value: str) -> Path:
        path = Path(value).expanduser()
        return (path if path.is_absolute() else base / path).resolve()

    model_path = resolve_input(model) if model else discovered_model
    data_path = resolve_input(data) if data else discovered_data
    checks: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []

    def path_check(name: str, path: Path | None, *, directory: bool) -> None:
        valid = path is not None and ((path.is_dir() if directory else path.exists()) and not path.is_symlink())
        checks.append({"name": name, "state": "pass" if valid else "missing", "path": str(path) if path else None})
        if not valid:
            blockers.append({
                "code": "MODEL_PATH_REQUIRED" if name == "model" else "DATA_PATH_REQUIRED",
                "message": "还没有找到模型目录。请填写模型所在目录。" if name == "model" else "还没有找到数据目录。请填写数据所在目录。",
                "action": "在首次设置中选择模型目录。" if name == "model" else "在首次设置中选择数据目录。",
            })

    path_check("model", model_path, directory=True)
    path_check("data", data_path, directory=False)
    report: dict[str, Any] | None = None
    if model_path is not None and model_path.is_dir() and not model_path.is_symlink():
        try:
            report = scan_repository(
                OnboardingOptions(
                    repo_root=model_path,
                    runtime_python=resolve_input(runtime_python) if runtime_python else None,
                    evaluator_contract=resolve_input(evaluator_contract) if evaluator_contract else None,
                    probe_imports=False,
                )
            )
        except (OnboardingError, OSError, ValueError) as exc:
            blockers.append({
                "code": "MODEL_SCAN_FAILED",
                "message": "模型目录可以读取，但系统无法完成只读检查。",
                "action": "确认目录权限后再次检查。",
                "detail": str(exc),
            })
    if report is not None:
        for item in report.get("blockers", []):
            if not isinstance(item, Mapping):
                continue
            code = str(item.get("code", "MODEL_ONBOARDING_BLOCKED"))
            if code in {"SOURCE_REVISION_UNBOUND", "CHECKPOINT_MISSING", "CAPABILITY_NOT_DISCOVERED"}:
                message = {
                    "SOURCE_REVISION_UNBOUND": "模型目录没有可追溯的版本记录。",
                    "CHECKPOINT_MISSING": "没有发现权重或 checkpoint 文件。",
                    "CAPABILITY_NOT_DISCOVERED": "还无法从文件中确认这是可运行的模型项目。",
                }[code]
                action = {
                    "SOURCE_REVISION_UNBOUND": "在模型仓库提交一个 Git 版本，或明确固定源码版本。",
                    "CHECKPOINT_MISSING": "确认权重文件路径；如果模型不需要 checkpoint，请在适配器中明确声明。",
                    "CAPABILITY_NOT_DISCOVERED": "补充训练、推理、评测或 rollout 的入口说明。",
                }[code]
            elif code == "EVALUATOR_CONTRACT_REQUIRED":
                message = "还没有确认如何判断模型变好。"
                action = "提供一个冻结的评测契约，或请管理员绑定已有评测方法。"
            elif code == "EVALUATION_ENTRYPOINT_MISSING":
                message = "发现了模型代码，但没有发现可用于评测的入口。"
                action = "说明评测命令和输出指标；训练入口不能代替评测。"
            elif code == "RUNTIME_UNREADY":
                message = "模型运行环境还没有准备好。"
                action = "选择模型自己的 Python 环境，并确认依赖可以导入。"
            elif code == "MODEL_ASSET_BINDING_REQUIRED":
                message = "评测需要的模型或数据文件还没有绑定。"
                action = "在设置中补充对应文件路径。"
            else:
                message = "模型接入还缺少一项可验证信息。"
                action = "打开检查详情，按列出的项目补齐。"
            blockers.append({"code": code, "message": message, "action": action, "detail": item.get("detail")})
    # Keep the list deterministic and avoid repeating the same blocker from a
    # missing top-level path and the onboarding scanner.
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for blocker in blockers:
        key = str(blocker.get("code"))
        if key not in seen:
            unique.append(blocker)
            seen.add(key)
    blockers = unique
    if blockers:
        state = "needs_input"
    elif report is not None and report.get("state") == "ready_for_conformance_smoke":
        state = "ready_for_conformance"
    else:
        state = "needs_confirmation"
    return {
        "schema_version": 1,
        "artifact_type": "verdiwm-first-contact-readiness",
        "state": state,
        "model": str(model_path) if model_path else None,
        "data": str(data_path) if data_path else None,
        "checks": checks,
        "blockers": blockers,
        "discovered": {
            "entrypoints": report.get("entrypoints", []) if report else [],
            "assets": report.get("assets", []) if report else [],
            "runtime": report.get("runtime", {}) if report else {},
            "source_revision": report.get("source_revision") if report else None,
            "evaluator": report.get("evaluator_contract", {}) if report else {},
        },
        "next_step": (
            "先补齐上面的信息，再运行检查。"
            if blockers
            else "运行 CPU 安全接入检查；通过后才可以创建正式实验。"
            if state == "ready_for_conformance"
            else "请确认评测方法和运行方式，系统不会自行猜测。"
        ),
        "side_effects": {"model_import_executed": False, "gpu_execution_started": False, "source_modified": False},
    }


def initialize_project(
    *,
    root: Path,
    model: str | None = None,
    data: str | None = None,
    goal: str | None = None,
    budget: str = "1gpu-hour",
    mode: str = "hybrid",
    target_metrics: list[str] | None = None,
    evaluator_contract: str | None = None,
    runtime_python: str | None = None,
    project_file: Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Create a minimal project file from user concepts, not internal contracts."""
    base = Path(root).expanduser().resolve()
    discovered_model, discovered_data = discover_project_inputs(base)
    def resolve_input(value: str) -> Path:
        path = Path(value).expanduser()
        return (path if path.is_absolute() else base / path).resolve()

    model_path = resolve_input(model) if model else discovered_model
    data_path = resolve_input(data) if data else discovered_data
    errors: list[dict[str, str]] = []
    if model_path is None:
        errors.append({"field": "model", "message": "找不到模型目录；请用 --model 指定模型所在目录。"})
    elif not model_path.is_dir() or model_path.is_symlink():
        errors.append({"field": "model", "message": f"模型目录不可用：{model_path}"})
    if data_path is None:
        errors.append({"field": "data", "message": "找不到数据目录；请用 --data 指定数据所在目录。"})
    elif not data_path.exists() or data_path.is_symlink():
        errors.append({"field": "data", "message": f"数据路径不可用：{data_path}"})
    explicit_runtime = resolve_input(runtime_python) if runtime_python else None
    if explicit_runtime is not None and (not explicit_runtime.is_file() or not os.access(explicit_runtime, os.X_OK)):
        errors.append({"field": "runtime_python", "message": f"Python 运行环境不可用：{explicit_runtime}"})
    evaluator_path = resolve_input(evaluator_contract) if evaluator_contract else None
    if evaluator_path is not None and (not evaluator_path.is_file() or evaluator_path.is_symlink()):
        errors.append({"field": "evaluator_contract", "message": f"评测契约不可用：{evaluator_path}"})
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
    runtime = explicit_runtime or next(
        (
            candidate
            for candidate in (
                model_path / ".venv" / "bin" / "python",
                model_path / "venv" / "bin" / "python",
            )
            if candidate.is_file() and os.access(candidate, os.X_OK)
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
    if evaluator_path is not None:
        values.append(("evaluator_contract", relative(evaluator_path).replace(chr(92), "/")))
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
        "readiness": inspect_project(
            root=base,
            model=str(model_path),
            data=str(data_path),
            evaluator_contract=evaluator_contract,
            runtime_python=runtime_python or (str(runtime) if runtime else None),
        ),
        "next_steps": [
            "运行 verdiwm doctor 检查本地控制面。",
            "运行 verdiwm run 或打开 verdiwm-workbench 开始受边界实验。",
            "如果系统缺少模型运行入口，会明确列出需要补充的信息，不会擅自修改模型。",
        ],
    }
