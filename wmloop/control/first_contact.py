"""Human-facing project initialization for first-contact model research."""

from __future__ import annotations

import os
import json
import sys
from pathlib import Path
from typing import Any, Mapping

from wmloop.control.adapter_profiles import resolve_adapter_asset_bindings
from wmloop.control.onboarding import OnboardingError, OnboardingOptions, scan_repository
from wmloop.control.evaluator_discovery import discover_evaluator_candidates
from wmloop.runtime_env import runtime_python_path
from wmloop.storage import atomic_write, checked_path, exclusive_file_lock


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
        "VERDIWM_CONFIG_INVALID": "研究 LLM 配置文件无法读取；请运行 verdi llm 查看示例，并用 verdi llm status 检查。",
        "VERDIWM_CONFIG_LLM_SECTION_MISSING": "研究 LLM 配置缺少 [llm] 区块；请运行 verdi llm 查看可复制示例。",
        "VERDIWM_CONFIG_ENDPOINT_INVALID": "研究 LLM 的 endpoint/base_url 配置无效；请运行 verdi llm status 查看检查结果。",
        "VERDIWM_CONFIG_MODEL_INVALID": "研究 LLM 没有配置 model；请运行 verdi llm 查看示例。",
        "VERDIWM_CONFIG_API_STYLE_INVALID": "研究 LLM 的 api_style 只能是 responses 或 chat_completions；请运行 verdi llm 查看示例。",
        "VERDIWM_CONFIG_TOKEN_FILE_INVALID": "研究 LLM 的 token_file 配置无效；请使用权限为 600 的普通文件，或改用环境变量。",
        "VERDIWM_CONFIG_TOKEN_KEY_INVALID": "研究 LLM 的 token_environment_key 无效；请使用合法的环境变量名。",
        "VERDIWM_CONFIG_AUTH_REQUIRED_INVALID": "研究 LLM 的 auth_required 必须是 true 或 false；请运行 verdi llm 查看示例。",
        "OPENAI_BROKER_TOKEN_MISSING": "研究 LLM 尚未找到密钥；请运行 verdi llm 查看教程，再运行 verdi llm status 检查。",
        "OPENAI_BROKER_TOKEN_FILE_PERMISSIONS": "研究 LLM 密钥文件权限不安全；请运行 chmod 600 PATH_TO_AUTH_FILE 后重试。",
        "EVALUATOR_CONTRACT_REQUIRED": "还没有确认如何判断模型变好，请先绑定冻结的评测方法。",
        "EVALUATOR_CONFIRMATION_REQUIRED": "发现了模型自带的评测候选，但它们尚未冻结；请确认评测切分、时域和 verifier 后再开始实验。",
        "RUNTIME_PYTHON_INVALID": "你指定的 Python 运行环境不可执行；请检查路径和权限。",
        "EVALUATION_ENTRYPOINT_MISSING": "没有发现可用于评测的入口，请说明评测命令和输出指标。",
        "CHECKPOINT_MISSING": "没有发现权重或 checkpoint 文件，请补充权重路径。",
        "RUNTIME_UNREADY": "模型运行环境还没有准备好，请选择正确的 Python 环境并检查依赖。",
        "COMMUNITY_BUNDLE_CRYPTOGRAPHY_REQUIRED": "当前环境缺少社区签名依赖，请运行 uv sync --group dev 后重试。",
        "COMMUNITY_LIFECYCLE_OUTPUT_CONFLICT": "生命周期文件已经存在且内容不同，请换一个输出路径或先检查已有记录。",
        "COMMUNITY_LIFECYCLE_OUTPUT_INVALID": "生命周期输出路径不可用，请选择普通文件路径。",
        "RESEARCH_PLAN_BLOCKED": "研究计划还有硬性阻塞项，系统没有启动实验。请先补齐计划中的评测器、运行环境或适配器绑定。",
        "RESEARCH_PLAN_INPUT_DIGEST_DRIFT": "研究计划生成后输入文件发生了变化，系统已停止以避免把结果绑定到错误的模型或评测器。请重新生成计划。",
        "RESEARCH_PLAN_DIGEST_MISMATCH": "研究计划文件的内容摘要不一致，请重新生成计划。",
        "RESEARCH_PLAN_INPUT_BINDINGS_MISSING": "研究计划没有锁定完整的输入摘要，请重新生成计划。",
        "RESEARCH_PLAN_OUTPUT_CONFLICT": "研究计划输出文件已经存在且内容不同，请换一个路径或先检查已有计划。",
        "RESEARCH_PLAN_OUTPUT_INSIDE_INPUT": "研究计划不能写入模型、源码或数据目录内部。",
        "RESEARCH_PLAN_ADAPTER_PREVIEW_MISSING": "研究计划没有保存可复现的适配器预览，请重新生成计划。",
        "TRAINING_SCALE_AGIBOT_DATASET_ROOT_INVALID": "Agibot 数据目录无效。请提供同时包含 `train.index.jsonl` 和 `val.index.jsonl` 的数据集根目录。",
        "TRAINING_SCALE_TRAIN_MANIFEST_INVALID": "Agibot 训练索引无法读取。请检查数据目录中的 `train.index.jsonl` 是否存在且每行都是有效 JSON。",
        "TRAINING_SCALE_VAL_MANIFEST_INVALID": "Agibot 验证索引无法读取。请将与训练集分离的 `val.index.jsonl` 放在数据集根目录中，再重试。",
        "TRAINING_SCALE_VAL_MANIFEST_REQUIRED": "还没有验证数据索引。Agibot 工程诊断需要独立的 `val.index.jsonl`，不能从训练集临时切分。",
        "TRAINING_SCALE_AGIBOT_NATIVE_INDEX_ENGINEERING_DIAGNOSTIC_ONLY": "原生 Agibot 索引当前只支持工程诊断阶段。请使用 `--stage engineering_diagnostic`；它验证训练链路，不判定模型效果。",
        "MODEL_RUN_AGIBOT_NATIVE_CONFIG_INVALID": "原生训练 YAML 无法解析。请确认文件包含 `world_model.action_num_domains` 和 `wm_training.action_num_domains` 两个正整数。",
        "MODEL_RUN_AGIBOT_NATIVE_CONFIG_DOMAIN_CAPACITY_TOO_SMALL": "训练配置的动作域容量小于 Agibot 数据中的最大 domain ID。请同时提高 YAML 中的 `world_model.action_num_domains` 和 `wm_training.action_num_domains`，使它们至少为最大 domain ID 加 1。",
        "MODEL_RUN_PROTOCOL_V3_NATIVE_AGIBOT_PLAN_REQUIRED": "这条训练入口需要 Agibot 原生 `train.index.jsonl`/`val.index.jsonl` 数据，以及工程诊断训练计划。",
        "MODEL_RUN_PROTOCOL_V3_ENGINEERING_DIAGNOSTIC_POLICY_INVALID": "原生 Agibot 训练目前只接受有界的工程诊断计划（最多 8 步、单 seed、全量索引支持）。请重新生成 `engineering_diagnostic` 计划。",
        "MODEL_RUN_AGIBOT_SUPPORT_INVALID": "Agibot 原生数据无法绑定。请检查 `manifest.json`、两个索引文件、统计文件和视频 shard 引用是否完整。",
        "MODEL_RUN_AGIBOT_SUPPORT_REQUIRED": "Agibot 原生训练缺少数据支持绑定。请通过 `verdi run --training-config` 创建诊断任务，或补齐适配器的 support loader。",
        "MODEL_RUN_BUDGET_ESTIMATE_EXCEEDS_TOTAL": "按训练和验证阶段的最长运行时间、以及计划使用的 GPU 数量估算，这次诊断需要的预算高于你声明的上限。请提高 `--budget`，或在自定义 Agibot profile 中合理调整阶段超时；系统已在入队前停止。",
    }
    message = exact.get(code)
    lower_detail = detail.casefold()
    if code == "MODEL_RUN_AGIBOT_SUPPORT_INVALID" and "agibot_support_val_index_invalid" in lower_detail:
        message = "Agibot 验证索引无法读取。请检查数据目录中的 `val.index.jsonl` 是否存在且与训练集分离。"
    if message is None:
        if "agibot_support_val_index_invalid" in lower_detail:
            message = "Agibot 验证索引无法读取。请检查数据目录中的 `val.index.jsonl` 是否存在且与训练集分离。"
        elif "agibot_support_data_root_invalid" in lower_detail:
            message = "Agibot 数据目录无效。请提供包含 `manifest.json`、`train.index.jsonl`、`val.index.jsonl` 和 `stats/` 的数据集根目录。"
        elif "agibot_support_shard" in lower_detail:
            message = "Agibot 索引引用的视频 shard 缺失或路径无效。请检查索引中的 shard 路径，并确认对应文件都已下载。"
        elif "agibot_support_native_source_invalid" in lower_detail:
            message = "模型源码缺少 Agibot 原生训练接口。请把 `--source` 指向包含 `sa_wm/data/` 与 `sa_wm/world_model/` 的源码目录。"
        elif "domain_capacity_too_small" in lower_detail:
            message = "训练配置中的动作域容量不足。请提高 `world_model.action_num_domains` 和 `wm_training.action_num_domains`，至少覆盖数据中最大的 domain ID 加 1。"
    if message is None and any(token in detail for token in ("TARGET_METRIC", "METRIC_")):
        message = "填写的成功指标不在当前评测清单中。请使用模型原有评测指标，或先补充评测方法。"
    if message is None and any(token in detail for token in ("ADAPTER", "PROFILE")):
        message = "这个模型还没有可用的运行连接器。系统已安全停止且没有占用 GPU；请确认模型平时如何启动和评测。"
    if message is None and any(token in detail for token in ("CONFIG_NOT_FOUND", "PROVIDER", "LLM_")):
        message = "研究 LLM 尚未配置或配置无效。请运行 verdi llm 查看教程，再运行 verdi llm status 检查；不要在页面粘贴密钥。"
    if message is None and code.startswith("COMMUNITY_BUNDLE_"):
        message = "社区 Bundle 输入或验证失败，请检查语义文档、签名密钥、执行摘要和内容地址。"
    if message is None and code.startswith("COMMUNITY_EXPORT_"):
        message = "社区语义导出失败，请检查输入目录中的语义记录、文件大小限制和导出目录状态。"
    if message is None and code.startswith("KNOWLEDGE_LIFECYCLE_"):
        message = "生命周期记录不符合社区知识契约，请检查 action、subject、authority 和 evidence 引用。"
    if message is None:
        message = "当前输入还不能安全开始实验。请检查模型、数据、评测方法和预算。"
    return {"code": code, "error": message, "detail": detail}


def discover_project_inputs(root: Path) -> tuple[Path | None, Path | None]:
    """Find conventional model and dataset directories without importing them."""
    base = Path(root).expanduser().resolve()
    model = next((base / name for name in ("model", "models") if (base / name).is_dir()), None)
    data = next((base / name for name in ("data", "dataset", "datasets") if (base / name).exists()), None)
    return model, data


def infer_source_root(model_path: Path | None) -> Path | None:
    """Infer a nearby source checkout from a checkpoint-only model path.

    Checkpoints are often nested several levels below the repository that
    owns the training/evaluation entrypoints.  We only accept a parent with
    conventional source markers, and never inspect or import model code while
    making this inference.
    """

    if model_path is None:
        return None
    model = Path(model_path).expanduser().absolute()
    if not model.exists() or model.is_symlink():
        return None
    current = model if model.is_dir() else model.parent
    for depth, candidate in enumerate((current, *current.parents)):
        if depth > 6:
            break
        if any((candidate / marker).is_file() for marker in ("pyproject.toml", "setup.py", "setup.cfg")):
            return candidate
        if (candidate / "configs").is_dir() and (candidate / "scripts").is_dir():
            return candidate
    return None


def inspect_project(
    *,
    root: Path,
    model: str | None = None,
    source: str | None = None,
    data: str | None = None,
    evaluator_contract: str | None = None,
    runtime_python: str | None = None,
    asset_bindings: Mapping[str, object] | None = None,
    adapter_profile: str | None = None,
    adapter: str | None = None,
    goal: str | None = None,
) -> dict[str, Any]:
    """Run a bounded, read-only readiness check for a first-contact project.

    This deliberately reports what can be proven from files. It never imports
    the model, starts a command, installs dependencies, or allocates a GPU.
    """
    base = Path(root).expanduser().resolve()
    discovered_model, discovered_data = discover_project_inputs(base)
    def resolve_input(value: str, *, preserve_symlink: bool = False) -> Path:
        path = Path(value).expanduser()
        candidate = path if path.is_absolute() else base / path
        # A virtualenv's ``bin/python`` is commonly a symlink.  Resolving it
        # here silently switches the selected interpreter to the base Python,
        # so retain the launcher spelling while still making it absolute.
        return runtime_python_path(candidate) if preserve_symlink else candidate.resolve()

    model_path = resolve_input(model) if model else discovered_model
    source_path = resolve_input(source) if source else (infer_source_root(model_path) or model_path)
    data_path = resolve_input(data) if data else discovered_data
    checks: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []

    def path_check(name: str, path: Path | None, *, directory: bool) -> None:
        valid = path is not None and ((path.is_dir() if directory else path.exists()) and not path.is_symlink())
        checks.append({"name": name, "state": "pass" if valid else "missing", "path": str(path) if path else None})
        if not valid:
            code = {
                "model": "MODEL_PATH_REQUIRED",
                "source": "MODEL_SOURCE_PATH_REQUIRED",
                "data": "DATA_PATH_REQUIRED",
            }[name]
            blockers.append({
                "code": code,
                "message": (
                    "还没有找到模型目录。请填写模型所在目录。"
                    if name == "model"
                    else "还没有找到模型源码目录。请填写包含训练、推理或评测入口的源码目录。"
                    if name == "source"
                    else "还没有找到数据目录。请填写数据所在目录。"
                ),
                "action": (
                    "在首次设置中选择模型目录。"
                    if name == "model"
                    else "在首次设置中选择模型源码目录。"
                    if name == "source"
                    else "在首次设置中选择数据目录。"
                ),
            })

    path_check("model", model_path, directory=True)
    if source is not None:
        path_check("source", source_path, directory=True)
        if source_path is not None and source_path == model_path:
            blockers.append({
                "code": "MODEL_SOURCE_NOT_SEPARATE",
                "message": "源码目录与权重目录相同；请确认模型源码是否已提供。",
                "action": "提供包含训练、推理或评测入口的源码目录。",
            })
    path_check("data", data_path, directory=False)
    report: dict[str, Any] | None = None
    evaluator_discovery: dict[str, Any] = {"state": "none", "candidates": []}
    model_assets_available = False
    scan_path = source_path
    if scan_path is not None and scan_path.is_dir() and not scan_path.is_symlink():
        try:
            if (
                asset_bindings is None and (adapter_profile is not None or adapter is not None)
                and model_path is not None and data_path is not None
            ):
                asset_bindings = resolve_adapter_asset_bindings(
                    model=model_path,
                    source_root=source_path,
                    data=data_path,
                    adapter_profile_path=base / Path(adapter_profile).expanduser() if adapter_profile else None,
                    adapter=adapter,
                    goal=goal or "",
                )
            evaluator_discovery = discover_evaluator_candidates(scan_path)
            report = scan_repository(
                OnboardingOptions(
                    repo_root=scan_path,
                    runtime_python=resolve_input(runtime_python, preserve_symlink=True) if runtime_python else None,
                    evaluator_contract=resolve_input(evaluator_contract) if evaluator_contract else None,
                    probe_imports=False,
                    asset_bindings=tuple(
                        (parameter, checked_path(base / Path(str(path)).expanduser(), code="ASSET_BINDING_PATH_INVALID", error=OnboardingError))
                        for parameter, path in (asset_bindings or {}).items()
                    ),
                    additional_asset_parameters=tuple(asset_bindings or ()),
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
        # We scan executable source separately from the weight directory.  A
        # source checkout normally contains no checkpoint, while a model
        # export normally contains no entrypoint; combine only the discovered
        # asset evidence and keep the two roots explicit in the report.
        if model_path is not None and model_path != scan_path:
            try:
                model_assets = scan_repository(
                    OnboardingOptions(repo_root=model_path, probe_imports=False)
                ).get("assets", [])
                report["assets"] = [*model_assets, *report.get("assets", [])]
                model_assets_available = any(
                    isinstance(asset, Mapping) and asset.get("kind") == "checkpoint"
                    for asset in model_assets
                )
            except (OnboardingError, OSError, ValueError):
                # Source readiness remains useful even when a model export is
                # too unusual for the generic asset scanner.
                pass
        for item in report.get("blockers", []):
            if not isinstance(item, Mapping):
                continue
            code = str(item.get("code", "MODEL_ONBOARDING_BLOCKED"))
            if code == "CHECKPOINT_MISSING" and model_assets_available:
                continue
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
    if evaluator_contract is None and evaluator_discovery.get("state") == "candidates_available":
        # A discovered file is useful evidence, but silently treating it as a
        # frozen verifier would invalidate every downstream quality claim.
        blockers = [item for item in blockers if item.get("code") != "EVALUATOR_CONTRACT_REQUIRED"]
        blockers.append({
            "code": "EVALUATOR_CONFIRMATION_REQUIRED",
            "message": "发现了评测候选，但它们尚未冻结。",
            "action": "确认一个候选；系统随后会在 VERDI 输出目录生成带摘要和路径改写的评测 overlay。",
        })
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
    elif (
        report is not None
        and report.get("runtime", {}).get("state") == "ready"
        and report.get("evaluator_contract", {}).get("state") == "ready"
    ):
        # The source-only scan may have been blocked by a checkpoint that
        # the separate model directory or explicit profile has now supplied.
        # Reconcile readiness from the combined evidence and remaining gates.
        state = "ready_for_conformance"
    else:
        state = "needs_confirmation"
    return {
        "schema_version": 1,
        "artifact_type": "verdiwm-first-contact-readiness",
        "state": state,
        "model": str(model_path) if model_path else None,
        "source": str(source_path) if source_path else None,
        "data": str(data_path) if data_path else None,
        "checks": checks,
        "blockers": blockers,
        "discovered": {
            "entrypoints": report.get("entrypoints", []) if report else [],
            "assets": report.get("assets", []) if report else [],
            "asset_bindings": report.get("connector", {}).get("asset_bindings", []) if report else [],
            "runtime": report.get("runtime", {}) if report else {},
            "source_revision": report.get("source_revision") if report else None,
            "evaluator": report.get("evaluator_contract", {}) if report else {},
            "evaluator_discovery": evaluator_discovery,
        },
        "next_step": (
            "先补齐上面的信息，再运行检查。"
            if blockers
            else "运行 CPU 安全接入检查；通过后才可以创建正式实验。"
            if state == "ready_for_conformance"
            else (
                "检查发现的评测候选，确认切分、指标和 verifier；同时确认模型运行入口或适配器，"
                "然后再次运行 verdiwm check。"
            )
        ),
        "side_effects": {"model_import_executed": False, "gpu_execution_started": False, "source_modified": False},
    }


def initialize_project(
    *,
    root: Path,
    model: str | None = None,
    source: str | None = None,
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
    source_path = resolve_input(source) if source else infer_source_root(model_path)
    data_path = resolve_input(data) if data else discovered_data
    errors: list[dict[str, str]] = []
    if model_path is None:
        errors.append({"field": "model", "message": "找不到模型目录；请用 --model 指定模型所在目录。"})
    elif not model_path.is_dir() or model_path.is_symlink():
        errors.append({"field": "model", "message": f"模型目录不可用：{model_path}"})
    if source_path is not None and (not source_path.is_dir() or source_path.is_symlink()):
        errors.append({"field": "source", "message": f"模型源码目录不可用：{source_path}"})
    if data_path is None:
        errors.append({"field": "data", "message": "找不到数据目录；请用 --data 指定数据所在目录。"})
    elif not data_path.exists() or data_path.is_symlink():
        errors.append({"field": "data", "message": f"数据路径不可用：{data_path}"})
    explicit_runtime = runtime_python_path(runtime_python, base=base) if runtime_python else None
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
    destination = checked_path(project_file or (base / "verdiwm.toml"), code="PROJECT_FILE_INVALID", error=FirstContactError)
    def relative(path: Path) -> str:
        try:
            return os.path.relpath(path, destination.parent)
        except ValueError:
            return str(path)
    # Keep the source checkout's virtualenv ahead of any checkpoint-local
    # environment and preserve its launcher symlink spelling.
    runtime = explicit_runtime or next(
        (
            candidate
            for path in (source_path, model_path)
            if path is not None
            for candidate in (
                path / ".venv" / "bin" / "python",
                path / "venv" / "bin" / "python",
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
    if source_path is not None:
        values.insert(1, ("source", relative(source_path).replace(chr(92), "/")))
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
    with exclusive_file_lock(destination.with_name(f".{destination.name}.lock")):
        if destination.exists() and not force:
            raise FirstContactError(f"PROJECT_FILE_EXISTS:{destination}:use --force to replace it")
        atomic_write(destination, body.encode("utf-8"))
    readiness = inspect_project(
        root=base,
        model=str(model_path),
        source=str(source_path) if source_path else None,
        data=str(data_path),
        evaluator_contract=evaluator_contract,
        runtime_python=runtime_python or (str(runtime) if runtime else None),
    )
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
        "readiness": readiness,
        "next_steps": _setup_next_steps(readiness),
    }


def _setup_next_steps(readiness: Mapping[str, Any]) -> list[str]:
    """Suggest only actions allowed by the authoritative project readiness."""
    steps = ["运行 verdiwm doctor 检查本地控制面。"]
    state = readiness.get("state")
    if state == "needs_confirmation":
        steps.append(
            str(
                readiness.get("next_step")
                or "确认评测方法和模型运行方式，然后再次运行 verdiwm check。"
            )
        )
    elif state == "ready_for_conformance":
        steps.append(
            "运行 `verdiwm research plan` 生成并审阅计划；仅当计划状态允许时，"
            "再用 `verdiwm research run --plan <计划路径> --confirm` 明确启动。"
        )
    elif state in {"blocked", "needs_input"}:
        blockers = readiness.get("blockers", [])
        actions = (
            [
                str(item["action"])
                for item in blockers
                if isinstance(item, Mapping) and item.get("action")
            ]
            if isinstance(blockers, list)
            else []
        )
        steps.extend(actions)
        steps.append(
            str(
                readiness.get("next_step")
                or "解决上面的接入阻塞项后，再次运行 verdiwm check。"
            )
        )
    else:
        steps.append(
            "接入状态未能确认；请运行 verdiwm check 并按报告中的 action 处理。"
        )
    steps.append("模型源码和权重不会因首次设置而被修改。")
    return steps
