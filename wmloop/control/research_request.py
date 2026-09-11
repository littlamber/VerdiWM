"""Compile and execute a user-level research request safely.

This module is the narrow product boundary between a natural research request
and the existing campaign control plane.  Planning is deliberately read-only:
it scans files, resolves a trusted adapter when possible, records digests, and
reports blockers.  Only ``run_research_plan`` may create a campaign, and it
requires an explicit confirmation plus a fresh digest check.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from wmloop.contracts import ContractValidationError, validate_document
from wmloop.control.adapter_profiles import (
    AdapterProfileError,
    compile_adapter_execution,
    parse_gpu_budget,
)
from wmloop.control.first_contact import inspect_project
from wmloop.control.project_config import ProjectConfigError, load_project_config
from wmloop.control.research_modes import (
    ResearchModeError,
    compile_research_mode_plan,
    normalize_research_mode,
)


class ResearchRequestError(ValueError):
    """A research request or its immutable plan cannot be used safely."""


_SKIP_DIRS = frozenset({".git", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache", ".verdiwm"})
_OPEN_METHOD_POLICY: dict[str, object] = {
    "enabled": True,
    "mode": "evidence_grounded",
    "max_methods": 2,
    "require_four_arm_study": True,
    "require_target_side_validation": True,
    "authority": "four_arm_study_only",
}


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_bytes(chunks: Sequence[bytes]) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk)
    return digest.hexdigest()


def _path_digest(path: Path) -> dict[str, object]:
    """Digest a file or directory without following internal symlinks."""

    path = path.expanduser().resolve()
    if path.is_symlink() or not path.exists():
        raise ResearchRequestError(f"INPUT_PATH_INVALID:{path}")
    if path.is_file():
        digest = hashlib.sha256()
        size = 0
        try:
            with path.open("rb") as handle:
                while True:
                    chunk = handle.read(1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    digest.update(chunk)
        except OSError as exc:
            raise ResearchRequestError(f"INPUT_PATH_UNREADABLE:{path}") from exc
        return {"path": str(path), "kind": "file", "sha256": digest.hexdigest(), "size_bytes": size, "file_count": 1}

    digest = hashlib.sha256()
    file_count = 0
    byte_count = 0
    try:
        entries = sorted(path.rglob("*"), key=lambda item: item.relative_to(path).as_posix())
    except OSError as exc:
        raise ResearchRequestError(f"INPUT_PATH_UNREADABLE:{path}") from exc
    for entry in entries:
        relative = entry.relative_to(path)
        if any(part in _SKIP_DIRS for part in relative.parts):
            continue
        relative_text = relative.as_posix()
        if entry.is_symlink():
            target = os.readlink(entry)
            digest.update(f"link:{relative_text}:{target}\n".encode("utf-8"))
            continue
        if not entry.is_file():
            continue
        file_digest = hashlib.sha256()
        size = 0
        try:
            with entry.open("rb") as handle:
                while True:
                    chunk = handle.read(1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    file_digest.update(chunk)
        except OSError as exc:
            raise ResearchRequestError(f"INPUT_PATH_UNREADABLE:{entry}") from exc
        digest.update(f"file:{relative_text}:{size}:".encode("utf-8"))
        digest.update(file_digest.digest())
        file_count += 1
        byte_count += size
    return {
        "path": str(path),
        "kind": "directory",
        "sha256": digest.hexdigest(),
        "size_bytes": byte_count,
        "file_count": file_count,
    }


def _resolve(value: object, *, base: Path) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, (str, Path)) or not str(value).strip():
        return None
    path = Path(str(value)).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _blocker(code: str, message: str, action: str, detail: object | None = None) -> dict[str, object]:
    result: dict[str, object] = {"code": code, "message": message, "action": action}
    if detail is not None:
        result["detail"] = str(detail)
    return result


def _error_code(error: BaseException | str) -> str:
    return str(error).split(":", 1)[0]


def _metric_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _adapter_blocker(error: BaseException) -> dict[str, object]:
    code = _error_code(error)
    return _blocker(
        code,
        "还没有找到可以安全运行这个模型的版本化适配器。",
        "提供模型源码/入口、明确的适配器 profile，或先完成一次隔离适配器修复。",
        str(error),
    )


def _dedupe_blockers(items: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    seen: set[str] = set()
    for item in items:
        code = str(item.get("code", "UNKNOWN"))
        if code not in seen:
            result.append(dict(item))
            seen.add(code)
    return result


def _write_json(path: Path, payload: Mapping[str, object]) -> Path:
    path = path.expanduser().resolve()
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ResearchRequestError("RESEARCH_PLAN_OUTPUT_INVALID")
    encoded = json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.is_file():
        try:
            if path.read_text(encoding="utf-8") == encoded:
                return path
        except OSError as exc:
            raise ResearchRequestError("RESEARCH_PLAN_OUTPUT_INVALID") from exc
        raise ResearchRequestError("RESEARCH_PLAN_OUTPUT_CONFLICT")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()
    return path


def _ensure_outside_inputs(output: Path, paths: Sequence[Path | None]) -> None:
    output = output.expanduser().resolve()
    for candidate in paths:
        if candidate is None:
            continue
        candidate = candidate.expanduser().resolve()
        if candidate == output or candidate in output.parents:
            raise ResearchRequestError("RESEARCH_PLAN_OUTPUT_INSIDE_INPUT")


def _project_values(base: Path) -> dict[str, object]:
    try:
        return dict(load_project_config(cwd=base).values)
    except ProjectConfigError as exc:
        raise ResearchRequestError(str(exc)) from exc


def _find_profile(root: Path, profile_id: str) -> Path | None:
    candidate = root / "configs" / "adapters" / f"{profile_id}.json"
    return candidate if candidate.is_file() else None


def _input_bindings(
    *,
    model: Path,
    data: Path,
    source: Path | None,
    evaluator: Path | None,
    runtime_python: Path | None,
    adapter_profile: Path | None,
    model_irg: Path | None,
    asset_bindings: Mapping[str, object] | None,
) -> dict[str, dict[str, object]]:
    values: dict[str, Path] = {"model": model, "data": data}
    optional = {
        "source": source,
        "evaluator_contract": evaluator,
        "runtime_python": runtime_python,
        "adapter_profile": adapter_profile,
        "model_irg": model_irg,
    }
    values.update({name: path for name, path in optional.items() if path is not None})
    if asset_bindings:
        for parameter, raw_path in sorted(asset_bindings.items()):
            values[f"asset:{parameter}"] = Path(str(raw_path)).expanduser().resolve()
    result: dict[str, dict[str, object]] = {}
    for name, path in sorted(values.items()):
        result[name] = _path_digest(path)
    return result


def _input_digest(bindings: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical(bindings).encode("utf-8")).hexdigest()


def compile_research_plan(
    *,
    model: str | Path | None = None,
    data: str | Path | None = None,
    goal: str | None = None,
    budget: object = "1gpu-hour",
    mode: str = "hybrid",
    adapter: str = "auto",
    source: str | Path | None = None,
    evaluator_contract: str | Path | None = None,
    runtime_python: str | Path | None = None,
    adapter_profile: str | Path | None = None,
    target_metrics: Sequence[str] | None = None,
    model_irg: str | Path | None = None,
    irg_protected_metrics: Sequence[str] | None = None,
    state_root: str | Path | None = None,
    project_root: str | Path | None = None,
    evidence_context: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Compile a deterministic plan without importing the model or running it."""

    base = Path(project_root or Path.cwd()).expanduser().resolve()
    configured = _project_values(base)
    model_path = _resolve(model or configured.get("model"), base=base)
    data_path = _resolve(data or configured.get("data", configured.get("dataset")), base=base)
    source_path = _resolve(source or configured.get("source"), base=base)
    evaluator_path = _resolve(evaluator_contract or configured.get("evaluator_contract"), base=base)
    runtime_path = _resolve(runtime_python or configured.get("runtime_python"), base=base)
    profile_path = _resolve(adapter_profile or configured.get("adapter_profile"), base=base)
    irg_path = _resolve(model_irg, base=base)
    normalized_goal = goal or configured.get("goal")
    normalized_mode = normalize_research_mode(mode or configured.get("mode", "hybrid"))
    normalized_budget = budget or configured.get("budget", "1gpu-hour")
    try:
        budget_hours = parse_gpu_budget(normalized_budget)
    except AdapterProfileError as exc:
        raise ResearchRequestError(str(exc)) from exc
    if model_path is None:
        raise ResearchRequestError("MODEL_PATH_REQUIRED")
    if data_path is None:
        raise ResearchRequestError("DATA_PATH_REQUIRED")
    if not isinstance(normalized_goal, str) or not normalized_goal.strip():
        raise ResearchRequestError("GOAL_REQUIRED")
    if not model_path.is_dir() or model_path.is_symlink():
        raise ResearchRequestError("MODEL_PATH_INVALID")
    if not data_path.exists() or data_path.is_symlink():
        raise ResearchRequestError("DATA_PATH_INVALID")
    _ensure_outside_inputs(base / ".verdiwm" / "research-plan.json", (model_path, data_path, source_path))

    repo_root = Path(__file__).resolve().parents[2]
    plan_seed = hashlib.sha256(_canonical({
        "model": str(model_path), "data": str(data_path), "goal": normalized_goal.strip(),
        "budget": budget_hours, "mode": normalized_mode, "adapter": adapter,
    }).encode("utf-8")).hexdigest()[:24]
    campaign_root = (Path(state_root).expanduser().resolve() if state_root is not None else base / ".verdiwm" / "state") / "campaigns"
    blockers: list[dict[str, object]] = []
    preview_execution: dict[str, object] | None = None
    selected_profile = profile_path
    preview_error: BaseException | None = None
    try:
        resolved = compile_adapter_execution(
            campaign_id=f"plan-{plan_seed}",
            model=model_path,
            data=data_path,
            goal=normalized_goal.strip(),
            budget=budget_hours,
            campaign_root=campaign_root,
            adapter=adapter,
            adapter_profile_path=profile_path,
            runtime_python=runtime_path,
            project_root=repo_root,
        )
        preview_execution = dict(resolved.execution)
        selected_profile = selected_profile or _find_profile(repo_root, resolved.profile_id)
        if selected_profile is None:
            blockers.append(_blocker("ADAPTER_PROFILE_BINDING_MISSING", "系统选中了适配器，但计划没有锁定它的文件。", "显式提供 adapter profile。"))
        if evaluator_path is None:
            evaluator_path = _resolve(preview_execution.get("evaluator_contract"), base=repo_root)
        if runtime_path is None:
            runtime_path = _resolve(preview_execution.get("runtime_python"), base=repo_root)
    except (AdapterProfileError, OSError, ValueError) as exc:
        preview_error = exc
        blockers.append(_adapter_blocker(exc))

    readiness = inspect_project(
        root=base,
        model=str(model_path),
        source=str(source_path) if source_path is not None else None,
        data=str(data_path),
        evaluator_contract=str(evaluator_path) if evaluator_path is not None else None,
        runtime_python=str(runtime_path) if runtime_path is not None else None,
    )
    blockers.extend(item for item in readiness.get("blockers", []) if isinstance(item, Mapping))
    if evaluator_path is None or not evaluator_path.is_file() or evaluator_path.is_symlink():
        blockers.append(_blocker(
            "EVALUATOR_CONTRACT_REQUIRED",
            "还没有锁定一个冻结的目标侧评测契约，系统不能判断方法是否真的有效。",
            "提供 --evaluator-contract，或让适配器 profile 绑定一个存在的评测契约。",
        ))
    if runtime_path is None or not runtime_path.is_file() or not os.access(runtime_path, os.X_OK):
        blockers.append(_blocker(
            "RUNTIME_PYTHON_NOT_FOUND",
            "还没有找到可以运行目标模型的 Python 环境。",
            "提供 --runtime-python，或在模型目录中放置可执行的 .venv/bin/python。",
        ))
    if irg_path is not None and (not irg_path.is_file() or irg_path.is_symlink()):
        blockers.append(_blocker("MODEL_IRG_INVALID", "提供的 IRG 文件不可读取。", "确认 --model-irg 是普通文件。"))

    mode_plan: dict[str, object] | None = None
    if preview_execution is not None:
        try:
            mode_plan = compile_research_mode_plan(
                mode=normalized_mode,
                goal=normalized_goal.strip(),
                execution=preview_execution,
                evidence_context=evidence_context,
                root=repo_root,
            )
            if mode_plan.get("state") == "blocked":
                blockers.extend(_blocker("RESEARCH_MODE_PREREQUISITES_MISSING", "当前研究模式缺少必要的诊断输入。", "补充 probe/CPBE 等契约，或改用 hybrid/quick-start。", ",".join(str(item) for item in mode_plan.get("blockers", []))))
        except ResearchModeError as exc:
            blockers.append(_blocker(_error_code(exc), "研究模式无法编译。", "修正研究模式和诊断输入后重新生成计划。", str(exc)))
    elif normalized_mode == "causal_discovery":
        blockers.append(_blocker("RESEARCH_MODE_PREREQUISITES_MISSING", "因果发现模式需要先绑定可冻结的诊断输入。", "补充适配器 profile、probe contract 和 CPBE 输入。"))

    asset_bindings = preview_execution.get("asset_bindings") if preview_execution else None
    if isinstance(asset_bindings, Mapping):
        try:
            bindings = _input_bindings(
                model=model_path, data=data_path, source=source_path,
                evaluator=evaluator_path, runtime_python=runtime_path,
                adapter_profile=selected_profile, model_irg=irg_path,
                asset_bindings=asset_bindings,
            )
        except ResearchRequestError as exc:
            blockers.append(_blocker(_error_code(exc), "研究计划无法锁定所有输入文件。", "确认模型、数据、评测器和适配器资源都存在且可读取。", str(exc)))
            bindings = {}
    else:
        try:
            bindings = _input_bindings(
                model=model_path, data=data_path, source=source_path,
                evaluator=evaluator_path, runtime_python=runtime_path,
                adapter_profile=selected_profile, model_irg=irg_path,
                asset_bindings=None,
            )
        except ResearchRequestError as exc:
            blockers.append(_blocker(_error_code(exc), "研究计划无法锁定所有输入文件。", "确认模型、数据和评测器都存在且可读取。", str(exc)))
            bindings = {}
    input_digest = _input_digest(bindings) if bindings else None
    state = "blocked" if blockers else ("ready_with_deferred_discovery" if mode_plan and mode_plan.get("state") == "ready_with_deferred_discovery" else "ready")
    request_id = "research-request-" + hashlib.sha256(_canonical({"goal": normalized_goal.strip(), "model": str(model_path), "data": str(data_path), "input_digest": input_digest}).encode("utf-8")).hexdigest()[:24]
    plan: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "verdiwm-research-plan",
        "plan_id": "research-plan-" + plan_seed,
        "request_id": request_id,
        "state": state,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "project_root": str(base),
        "state_root": str(campaign_root.parent),
        "goal": normalized_goal.strip(),
        "model": str(model_path),
        "data": str(data_path),
        "source": str(source_path) if source_path is not None else None,
        "target_metrics": _metric_list(target_metrics),
        "budget": {"gpu_hours": budget_hours},
        "mode": normalized_mode,
        "adapter": adapter,
        "adapter_profile": str(selected_profile) if selected_profile is not None else None,
        "evaluator_contract": str(evaluator_path) if evaluator_path is not None else None,
        "runtime_python": str(runtime_path) if runtime_path is not None else None,
        "model_irg": str(irg_path) if irg_path is not None else None,
        "irg_protected_metrics": _metric_list(irg_protected_metrics),
        "input_bindings": bindings,
        "input_digest": input_digest,
        "adapter_preview": preview_execution or {},
        "readiness": readiness,
        "research_mode_plan": mode_plan or {},
        "stages": [
            {"order": 1, "stage": "onboarding", "state": "active", "purpose": "只读接入和契约检查"},
            {"order": 2, "stage": "diagnostic_probe", "state": "pending", "purpose": "建立目标模型的 IRG/行为画像"},
            {"order": 3, "stage": "retrieval_and_synthesis", "state": "pending", "purpose": "从本地证据和跨域资料提出受约束的机制假设"},
            {"order": 4, "stage": "method_materialization", "state": "pending", "purpose": "将假设物化为隔离候选实现并做接口校准"},
            {"order": 5, "stage": "paired_screen", "state": "pending", "purpose": "目标侧配对筛选，失败和负结果同样入证据链"},
            {"order": 6, "stage": "heldout_confirmation", "state": "pending", "purpose": "冻结 verifier 后在独立切分上确认"},
            {"order": 7, "stage": "knowledge_deposition", "state": "pending", "purpose": "沉淀带上下文、边界和不确定性的证据"},
        ],
        "open_method_policy": dict(_OPEN_METHOD_POLICY),
        "confirmation_required": True,
        "side_effects": {"model_import_executed": False, "gpu_execution_started": False, "source_modified": False},
        "blockers": _dedupe_blockers(blockers),
        "claim_boundary": "本计划只证明输入已绑定、执行路径可编译和门禁是否齐全；任何真实提升都必须由目标侧冻结 verifier、配对试验和 held-out confirmation 建立，检索、代码生成和校准不能单独形成科学结论。",
    }
    plan["plan_digest"] = hashlib.sha256(_canonical(plan).encode("utf-8")).hexdigest()
    try:
        validate_document("research_plan", plan, root=repo_root)
    except ContractValidationError as exc:
        raise ResearchRequestError(f"RESEARCH_PLAN_INVALID:{exc}") from exc
    return plan


def write_research_plan(plan: Mapping[str, object], output: str | Path) -> Path:
    """Validate and atomically write a plan artifact."""

    payload = dict(plan)
    try:
        validate_document("research_plan", payload, root=Path(__file__).resolve().parents[2])
    except ContractValidationError as exc:
        raise ResearchRequestError(f"RESEARCH_PLAN_INVALID:{exc}") from exc
    _ensure_outside_inputs(Path(output), tuple(_resolve(payload.get(name), base=Path(payload.get("project_root", Path.cwd()))) for name in ("model", "data", "source")))
    return _write_json(Path(output), payload)


def load_research_plan(path: str | Path) -> dict[str, object]:
    source = Path(path).expanduser().resolve()
    if source.is_symlink() or not source.is_file():
        raise ResearchRequestError("RESEARCH_PLAN_NOT_FOUND")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResearchRequestError("RESEARCH_PLAN_INVALID") from exc
    if not isinstance(payload, dict):
        raise ResearchRequestError("RESEARCH_PLAN_INVALID")
    try:
        validate_document("research_plan", payload, root=Path(__file__).resolve().parents[2])
    except ContractValidationError as exc:
        raise ResearchRequestError(f"RESEARCH_PLAN_INVALID:{exc}") from exc
    digest = payload.get("plan_digest")
    unsigned = dict(payload)
    unsigned.pop("plan_digest", None)
    expected = hashlib.sha256(_canonical(unsigned).encode("utf-8")).hexdigest()
    if digest != expected:
        raise ResearchRequestError("RESEARCH_PLAN_DIGEST_MISMATCH")
    return payload


def verify_research_plan_inputs(plan: Mapping[str, object]) -> dict[str, object]:
    bindings = plan.get("input_bindings")
    if not isinstance(bindings, Mapping) or not bindings:
        raise ResearchRequestError("RESEARCH_PLAN_INPUT_BINDINGS_MISSING")
    current: dict[str, dict[str, object]] = {}
    drift: list[dict[str, object]] = []
    for name, expected in sorted(bindings.items()):
        if not isinstance(expected, Mapping) or not isinstance(expected.get("path"), str):
            raise ResearchRequestError("RESEARCH_PLAN_INPUT_BINDING_INVALID")
        try:
            observed = _path_digest(Path(str(expected["path"])))
        except ResearchRequestError as exc:
            drift.append({"name": name, "code": _error_code(exc), "detail": str(exc)})
            continue
        current[str(name)] = observed
        if observed.get("sha256") != expected.get("sha256") or observed.get("kind") != expected.get("kind"):
            drift.append({"name": name, "code": "INPUT_DIGEST_DRIFT", "expected": expected.get("sha256"), "observed": observed.get("sha256")})
    observed_digest = _input_digest(current)
    if observed_digest != plan.get("input_digest"):
        drift.append({"name": "input_bindings", "code": "INPUT_DIGEST_DRIFT", "expected": plan.get("input_digest"), "observed": observed_digest})
    return {"state": "verified" if not drift else "blocked", "input_digest": observed_digest, "drift": drift}


def plan_to_campaign_payload(plan: Mapping[str, object]) -> dict[str, object]:
    if plan.get("state") not in {"ready", "ready_with_deferred_discovery"}:
        raise ResearchRequestError("RESEARCH_PLAN_BLOCKED")
    verification = verify_research_plan_inputs(plan)
    if verification["state"] != "verified":
        raise ResearchRequestError("RESEARCH_PLAN_INPUT_DIGEST_DRIFT")
    preview = plan.get("adapter_preview")
    if not isinstance(preview, Mapping):
        raise ResearchRequestError("RESEARCH_PLAN_ADAPTER_PREVIEW_MISSING")
    assets = preview.get("asset_bindings")
    payload: dict[str, object] = {
        "goal": str(plan["goal"]),
        "model": str(plan["model"]),
        "dataset": str(plan["data"]),
        "budget": plan["budget"],
        "adapter": str(plan.get("adapter") or "auto"),
        "adapter_profile_path": str(plan["adapter_profile"]) if plan.get("adapter_profile") else None,
        "runtime_python": str(plan["runtime_python"]) if plan.get("runtime_python") else None,
        "research_mode": str(plan["mode"]),
        "literature_query": str(plan["goal"]),
        "open_method_policy": dict(plan.get("open_method_policy") or _OPEN_METHOD_POLICY),
        "plan_binding": {"plan_id": plan.get("plan_id"), "plan_digest": plan.get("plan_digest"), "input_digest": plan.get("input_digest")},
    }
    if isinstance(assets, Mapping):
        payload["assets"] = {str(key): str(value) for key, value in assets.items()}
    metrics = list(plan.get("target_metrics") or ())
    if metrics:
        payload["target_metrics"] = metrics
    if plan.get("model_irg"):
        payload["model_irg_path"] = str(plan["model_irg"])
        payload["irg_protected_metrics"] = list(plan.get("irg_protected_metrics") or ())
    return {key: value for key, value in payload.items() if value is not None}


def run_research_plan(
    plan: Mapping[str, object],
    *,
    store: Any,
    confirm: bool,
    queue_only: bool = False,
    max_parallel: int = 1,
) -> dict[str, object]:
    """Create/confirm/dispatch a plan through CampaignStore after one confirmation."""

    if not confirm:
        return {"state": "awaiting_confirmation", "plan_id": plan.get("plan_id"), "plan_digest": plan.get("plan_digest"), "next_step": "再次运行 verdiwm research run --plan PLAN --confirm。"}
    payload = plan_to_campaign_payload(plan)
    created = store.create(payload)
    queued = store.confirm(str(created["campaign_id"]))
    if queue_only:
        return {"state": "queued", "campaign": queued, "plan_id": plan.get("plan_id")}
    from wmloop.control.campaign_dispatcher import DispatcherOptions, run_dispatcher

    dispatcher = run_dispatcher(DispatcherOptions(state_root=store.root, max_cycles=1, max_parallel=max_parallel, campaign_ids=(str(created["campaign_id"]),)))
    campaign = store.get(str(created["campaign_id"]))
    return {"state": str(campaign.get("status")), "campaign": campaign, "dispatcher": dispatcher, "plan_id": plan.get("plan_id")}
