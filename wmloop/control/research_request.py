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
import logging
import os
from pathlib import Path
import time
from typing import Any, Callable

from wmloop.contracts import (
    ContractValidationError,
    parse_yaml_document_text,
    validate_document,
)
from wmloop.control.adapter_profiles import (
    AdapterProfileError,
    compile_adapter_execution,
    parse_gpu_budget,
)
from wmloop.control.first_contact import infer_source_root, inspect_project
from wmloop.control.project_config import ProjectConfigError, load_project_config
from wmloop.control.research_execution_policy import (
    ResearchExecutionPolicyError,
    derive_research_plan_policy,
    plan_execution_policy,
    policy_input_paths,
    policy_warnings,
    research_execution_policy,
)
from wmloop.control.research_modes import (
    ResearchModeError,
    compile_research_mode_plan,
    normalize_research_mode,
)
from wmloop.storage import atomic_write, checked_path, exclusive_file_lock
from wmloop.verify.acceptance_policy import verification_policy_from_goal


class ResearchRequestError(ValueError):
    """A research request or its immutable plan cannot be used safely."""


_SKIP_DIRS = frozenset({".git", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache", ".verdiwm"})
# Source checkouts often contain local training artifacts beside the code.  A
# source binding should capture the checkout that defines the adapter, while
# avoiding an accidental walk through multi-gigabyte checkpoints and latent
# caches that are already bound separately as model/data inputs.
_SOURCE_SKIP_DIRS = frozenset({"checkpoints", "outputs", "runs", "wandb", "logs", "artifacts"})
_SOURCE_SKIP_PREFIXES = ("checkpoint", "video_latent", "latent_cache")
# Model repositories commonly contain multi-gigabyte checkpoints and cached
# videos.  Reading those files during a read-only plan makes first contact
# needlessly slow.  Small files still receive a complete content digest;
# large files use a deterministic metadata plus head/tail fingerprint.
_LARGE_FILE_FAST_BYTES = 256 * 1024 * 1024
_FAST_SAMPLE_BYTES = 1024 * 1024
_PROGRESS_INTERVAL_SECONDS = 2.0
_OPEN_METHOD_POLICY: dict[str, object] = {
    "enabled": True,
    "mode": "evidence_grounded",
    "max_methods": 2,
    "require_four_arm_study": True,
    "require_target_side_validation": True,
    "authority": "four_arm_study_only",
}

ResearchProgressCallback = Callable[[Mapping[str, object]], None]


def _report_progress(
    callback: ResearchProgressCallback | None,
    phase: str,
    state: str,
    **details: object,
) -> None:
    """Send best-effort status to a caller without affecting plan evidence."""

    if callback is None:
        return
    try:
        callback({"phase": phase, "state": state, **details})
    except Exception:
        # A broken display callback must not alter a plan or its evidence.
        logging.getLogger(__name__).debug("Research progress callback failed", exc_info=True)


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_bytes(chunks: Sequence[bytes]) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk)
    return digest.hexdigest()


def _path_digest(
    path: Path,
    *,
    source_tree: bool = False,
    progress: ResearchProgressCallback | None = None,
    input_name: str = "input",
) -> dict[str, object]:
    """Digest a file or directory without following internal symlinks."""

    started = time.monotonic()
    last_progress = started
    file_count = 0
    byte_count = 0

    def report(state: str, **details: object) -> None:
        _report_progress(
            progress,
            "input_digest",
            state,
            input=input_name,
            files_hashed=file_count,
            input_bytes=byte_count,
            elapsed_seconds=round(time.monotonic() - started, 1),
            **details,
        )

    report("started")
    path = path.expanduser().resolve()
    if path.is_symlink() or not path.exists():
        raise ResearchRequestError(f"INPUT_PATH_INVALID:{path}")
    if path.is_file():
        try:
            stat = path.stat()
            size = int(stat.st_size)
            with path.open("rb") as handle:
                digest = _file_content_digest(handle, size=size, stat=stat)
        except OSError as exc:
            raise ResearchRequestError(f"INPUT_PATH_UNREADABLE:{path}") from exc
        file_count = 1
        byte_count = size
        result = {"path": str(path), "kind": "file", "sha256": digest, "size_bytes": size, "file_count": 1}
        report("completed")
        return result

    digest = hashlib.sha256()
    excluded = set(_SKIP_DIRS)
    if source_tree:
        excluded.update(_SOURCE_SKIP_DIRS)
    try:
        # ``os.walk`` lets us prune ignored artifact directories before the
        # filesystem enumerates their contents.  This matters for source
        # checkouts that colocate tens or hundreds of gigabytes of artifacts.
        for current, directories, filenames in os.walk(path, topdown=True, followlinks=False):
            current_path = Path(current)
            kept_directories: list[str] = []
            for name in sorted(directories):
                numeric_artifact = source_tree and current_path == path and name.isdigit() and len(name) >= 4
                if name in excluded or (source_tree and (name.casefold().startswith(_SOURCE_SKIP_PREFIXES) or numeric_artifact)):
                    continue
                child = current_path / name
                if child.is_symlink():
                    relative_text = child.relative_to(path).as_posix()
                    digest.update(f"link:{relative_text}:{os.readlink(child)}\n".encode("utf-8"))
                    continue
                kept_directories.append(name)
            directories[:] = kept_directories
            for name in sorted(filenames):
                entry = current_path / name
                relative_text = entry.relative_to(path).as_posix()
                if entry.is_symlink():
                    digest.update(f"link:{relative_text}:{os.readlink(entry)}\n".encode("utf-8"))
                    continue
                if not entry.is_file():
                    continue
                size = 0
                try:
                    stat = entry.stat()
                    size = int(stat.st_size)
                    with entry.open("rb") as handle:
                        file_digest = _file_content_digest(handle, size=size, stat=stat)
                except OSError as exc:
                    raise ResearchRequestError(f"INPUT_PATH_UNREADABLE:{entry}") from exc
                digest.update(f"file:{relative_text}:{size}:".encode("utf-8"))
                digest.update(bytes.fromhex(file_digest))
                file_count += 1
                byte_count += size
                now = time.monotonic()
                if progress is not None and (
                    file_count % 1000 == 0 or now - last_progress >= _PROGRESS_INTERVAL_SECONDS
                ):
                    report("running")
                    last_progress = now
    except OSError as exc:
        raise ResearchRequestError(f"INPUT_PATH_UNREADABLE:{path}") from exc
    result = {
        "path": str(path),
        "kind": "directory",
        "sha256": digest.hexdigest(),
        "size_bytes": byte_count,
        "file_count": file_count,
}
    report("completed")
    return result


def _file_content_digest(handle: Any, *, size: int, stat: os.stat_result) -> str:
    """Hash a file fully when small and by stable head/tail samples when huge."""

    digest = hashlib.sha256()
    if size <= _LARGE_FILE_FAST_BYTES:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    else:
        # Include metadata and both ends of the file.  This is intentionally
        # deterministic and is also used by the pre-dispatch drift check;
        # conformance receipts remain responsible for any full-content
        # verification required by a publication workflow.
        digest.update(f"fast:{size}:{stat.st_mtime_ns}:{stat.st_mode}\n".encode("ascii"))
        digest.update(handle.read(_FAST_SAMPLE_BYTES))
        handle.seek(max(0, size - _FAST_SAMPLE_BYTES))
        digest.update(handle.read(_FAST_SAMPLE_BYTES))
    return digest.hexdigest()


def _resolve(value: object, *, base: Path, preserve_symlink: bool = False) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, (str, Path)) or not str(value).strip():
        return None
    path = Path(str(value)).expanduser()
    candidate = path if path.is_absolute() else base / path
    # Runtime launchers are often virtualenv symlinks.  Keep the spelling the
    # user supplied so the child process receives the intended environment;
    # ordinary inputs remain fully resolved for stable digest bindings.
    return candidate.absolute() if preserve_symlink else candidate.resolve()


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
        if any(not isinstance(item, str) or not item.strip() for item in value):
            raise ResearchRequestError("TARGET_METRICS_INVALID")
        return [item.strip() for item in value]
    raise ResearchRequestError("TARGET_METRICS_INVALID")


def _adapter_blocker(error: BaseException) -> dict[str, object]:
    code = _error_code(error)
    if isinstance(error, AdapterProfileError) and error.execution_support is not None:
        support = error.execution_support
        return _blocker(
            code,
            f"该适配器仍需扩展执行接口（{support['state']}）：{support['reason']}",
            f"{support['action']} 参见 {support['documentation']}",
            str(error),
        )
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
    path = checked_path(path, code="RESEARCH_PLAN_OUTPUT_INVALID", error=ResearchRequestError)
    encoded = json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with exclusive_file_lock(path.with_name(f".{path.name}.lock")):
        if path.exists() and not path.is_file():
            raise ResearchRequestError("RESEARCH_PLAN_OUTPUT_INVALID")
        if path.is_file():
            if path.read_text(encoding="utf-8") == encoded:
                return path
            raise ResearchRequestError("RESEARCH_PLAN_OUTPUT_CONFLICT")
        atomic_write(path, encoded.encode("utf-8"))
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
    if candidate.is_file():
        return candidate
    for path in sorted((root / "configs" / "adapters").glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, Mapping) and payload.get("profile_id") == profile_id:
            return path
    return None


def _is_source_binding(name: str) -> bool:
    return name == "source" or (
        name.startswith("asset:") and any(token in name.casefold() for token in ("source", "repo"))
    )


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
    progress: ResearchProgressCallback | None = None,
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
    cache: dict[tuple[str, bool], dict[str, object]] = {}
    for name, path in sorted(values.items()):
        source_binding = _is_source_binding(name)
        cache_key = (str(path), source_binding)
        if cache_key not in cache:
            cache[cache_key] = _path_digest(
                path,
                source_tree=source_binding,
                progress=progress,
                input_name=name,
            )
        result[name] = dict(cache[cache_key])
    return result


def _input_digest(bindings: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical(bindings).encode("utf-8")).hexdigest()


# Public binding operations also serve separately reviewable policy revisions.
bind_research_input = _path_digest
research_input_digest = _input_digest


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
    protected_metrics: Sequence[str] | None = None,
    model_irg: str | Path | None = None,
    irg_protected_metrics: Sequence[str] | None = None,
    state_root: str | Path | None = None,
    project_root: str | Path | None = None,
    evidence_context: Mapping[str, object] | None = None,
    rounds: int = 1,
    stop_on_success: bool = False,
    progress_callback: ResearchProgressCallback | None = None,
) -> dict[str, object]:
    """Compile a deterministic plan without importing the model or running it."""

    _report_progress(progress_callback, "plan_compile", "started")
    try:
        execution_policy = research_execution_policy(rounds=rounds, stop_on_success=stop_on_success)
    except ResearchExecutionPolicyError as exc:
        raise ResearchRequestError(str(exc)) from exc
    base = Path(project_root or Path.cwd()).expanduser().resolve()
    configured = _project_values(base)
    model_path = _resolve(model or configured.get("model"), base=base)
    data_path = _resolve(data or configured.get("data", configured.get("dataset")), base=base)
    source_path = _resolve(source or configured.get("source"), base=base)
    if source_path is None:
        # Checkpoint-only requests are common.  Infer a nearby source checkout
        # so the user does not have to repeat the repository path on every
        # campaign, while leaving explicit bindings authoritative.
        source_path = infer_source_root(model_path)
    # Keep the distinction between a user-frozen evaluator and a profile
    # default.  A profile may carry a verifier template, but a model checkout
    # can expose a more appropriate candidate; silently freezing that file
    # would make the first-contact evidence boundary meaningless.
    requested_evaluator = evaluator_contract or configured.get("evaluator_contract")
    evaluator_path = _resolve(requested_evaluator, base=base)
    evaluator_was_explicit = evaluator_path is not None
    runtime_value = runtime_python if runtime_python is not None else configured.get("runtime_python")
    runtime_path = _resolve(runtime_value, base=base, preserve_symlink=True)
    profile_path = _resolve(adapter_profile or configured.get("adapter_profile"), base=base)
    irg_path = _resolve(model_irg, base=base)
    normalized_goal = goal or configured.get("goal")
    try:
        normalized_mode = normalize_research_mode(mode or configured.get("mode", "hybrid"))
    except ResearchModeError as exc:
        raise ResearchRequestError(str(exc)) from exc
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
    try:
        requested_protected_metrics = _metric_list(protected_metrics)
    except ResearchRequestError as exc:
        raise ResearchRequestError(f"PROTECTED_METRICS_INVALID:{exc}") from exc
    if protected_metrics is not None and len(set(requested_protected_metrics)) != len(requested_protected_metrics):
        raise ResearchRequestError("PROTECTED_METRICS_DUPLICATE")
    if not model_path.is_dir() or model_path.is_symlink():
        raise ResearchRequestError("MODEL_PATH_INVALID")
    if not data_path.exists() or data_path.is_symlink():
        raise ResearchRequestError("DATA_PATH_INVALID")
    repo_root = Path(__file__).resolve().parents[2]
    plan_seed = hashlib.sha256(_canonical({
        "model": str(model_path), "data": str(data_path), "goal": normalized_goal.strip(),
        "budget": budget_hours, "mode": normalized_mode, "adapter": adapter,
        "execution_policy": execution_policy,
    }).encode("utf-8")).hexdigest()[:24]
    campaign_root = (Path(state_root).expanduser().resolve() if state_root is not None else base / ".verdiwm" / "state") / "campaigns"
    blockers: list[dict[str, object]] = []
    preview_execution: dict[str, object] | None = None
    selected_profile = profile_path
    preview_error: BaseException | None = None
    _report_progress(progress_callback, "adapter_preview", "started")
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
            source_root=source_path,
        )
        preview_execution = dict(resolved.execution)
        selected_profile = selected_profile or _find_profile(repo_root, resolved.profile_id)
        if selected_profile is None:
            blockers.append(_blocker("ADAPTER_PROFILE_BINDING_MISSING", "系统选中了适配器，但计划没有锁定它的文件。", "显式提供 adapter profile。"))
        if evaluator_path is None:
            evaluator_path = _resolve(preview_execution.get("evaluator_contract"), base=repo_root)
        else:
            # Campaign creation applies the user-frozen evaluator over the
            # profile default; preview the same contract before confirmation.
            preview_execution["evaluator_contract"] = str(evaluator_path)
        if runtime_path is None:
            runtime_path = _resolve(preview_execution.get("runtime_python"), base=repo_root, preserve_symlink=True)
    except (AdapterProfileError, OSError, ValueError) as exc:
        preview_error = exc
        blockers.append(_adapter_blocker(exc))
    _report_progress(
        progress_callback,
        "adapter_preview",
        "failed" if preview_error is not None else "completed",
    )

    _report_progress(progress_callback, "onboarding_scan", "started")
    readiness = inspect_project(
        root=base,
        model=str(model_path),
        source=str(source_path) if source_path is not None else None,
        data=str(data_path),
        evaluator_contract=(str(evaluator_path) if evaluator_was_explicit and evaluator_path is not None else None),
        runtime_python=str(runtime_path) if runtime_path is not None else None,
        asset_bindings=preview_execution.get("asset_bindings") if preview_execution else None,
    )
    _report_progress(progress_callback, "onboarding_scan", "completed")
    blockers.extend(item for item in readiness.get("blockers", []) if isinstance(item, Mapping))
    discovered_candidates = readiness.get("discovered", {}).get("evaluator_discovery", {})
    has_candidates = (
        isinstance(discovered_candidates, Mapping)
        and discovered_candidates.get("state") == "candidates_available"
        and bool(discovered_candidates.get("candidates"))
    )
    if not evaluator_was_explicit and has_candidates:
        # Hold the profile default in the preview only.  The campaign cannot
        # be scheduled until one candidate is explicitly overlaid and frozen.
        evaluator_path = None
        blockers = [item for item in blockers if str(item.get("code")) != "EVALUATOR_CONTRACT_REQUIRED"]
        blockers.append(_blocker(
            "EVALUATOR_CONFIRMATION_REQUIRED",
            "发现了模型源码中的评测候选，但它们尚未冻结。",
            "确认一个候选；系统会在 VERDI 输出目录生成带摘要和路径改写的 overlay，再重新生成计划。",
        ))
    elif evaluator_path is None or not evaluator_path.is_file() or evaluator_path.is_symlink():
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

    policy_inputs: dict[str, Path] = {}
    try:
        policy_inputs = policy_input_paths(execution_policy, evaluator_path)
    except ResearchExecutionPolicyError as exc:
        blockers.append(_blocker(_error_code(exc), "研究计划的评测模板或冻结目标无效。", "检查 evaluator 的 scheduler_template 与 frozen_goal；模板应包含 screen、gate、confirm，目标应包含冻结的 v2 验收规则。", str(exc)))

    frozen_protected_metrics: list[str] = []
    frozen_goal_digest: str | None = None
    frozen_goal_path = policy_inputs.get("frozen_goal")
    frozen_policy_resolved = False
    if frozen_goal_path is not None:
        try:
            goal_bytes = frozen_goal_path.read_bytes()
            frozen_goal_digest = hashlib.sha256(goal_bytes).hexdigest()
            frozen_goal = parse_yaml_document_text(
                goal_bytes.decode("utf-8"), source=str(frozen_goal_path)
            )
            verification_policy = verification_policy_from_goal(frozen_goal)
            if verification_policy is None:
                raise ResearchRequestError("FROZEN_GOAL_POLICY_REQUIRED")
            frozen_protected_metrics = list(verification_policy.protected_metrics)
            frozen_policy_resolved = True
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            blockers.append(_blocker(
                "FROZEN_GOAL_POLICY_INVALID",
                "冻结目标中的正式保护指标无法读取。",
                "检查 evaluator 绑定的 frozen_goal v2 验收规则。",
                str(exc),
            ))
    if protected_metrics is not None:
        if frozen_goal_path is None:
            blockers.append(_blocker(
                "PROTECTED_METRICS_REQUIRE_FROZEN_GOAL",
                "正式保护指标必须来自 evaluator 绑定的冻结目标。",
                "先在 evaluator 中绑定有效的 frozen_goal v2，再核对 --protected-metrics。",
            ))
        elif frozen_policy_resolved and set(requested_protected_metrics) != set(frozen_protected_metrics):
            missing = sorted(set(frozen_protected_metrics) - set(requested_protected_metrics))
            extra = sorted(set(requested_protected_metrics) - set(frozen_protected_metrics))
            blockers.append(_blocker(
                "PROTECTED_METRICS_FROZEN_GOAL_MISMATCH",
                "--protected-metrics 必须与冻结目标中的完整保护指标集合一致。",
                "按 evaluator 绑定的 frozen_goal v2 修改参数，或省略参数以直接预览冻结集合。",
                f"missing={missing}; extra={extra}",
            ))

    # A blocked plan is a reviewable readiness report, not frozen execution
    # evidence. Avoid traversing large assets until the user resolves its
    # prerequisites; a fresh ready plan will bind and verify them in full.
    bindings: dict[str, dict[str, object]] = {}
    if blockers:
        _report_progress(progress_callback, "input_digest", "deferred", reason="readiness_blocked")
    else:
        asset_bindings = preview_execution.get("asset_bindings") if preview_execution else None
        _report_progress(progress_callback, "input_digest", "started")
        try:
            bindings = _input_bindings(
                model=model_path, data=data_path, source=source_path,
                evaluator=evaluator_path, runtime_python=runtime_path,
                adapter_profile=selected_profile, model_irg=irg_path,
                asset_bindings=asset_bindings if isinstance(asset_bindings, Mapping) else None,
                progress=progress_callback,
            )
            bindings.update({name: bind_research_input(path) for name, path in policy_inputs.items()})
            if (
                frozen_goal_digest is not None
                and bindings.get("frozen_goal", {}).get("sha256") != frozen_goal_digest
            ):
                raise ResearchRequestError("FROZEN_GOAL_CHANGED_DURING_PLAN")
        except ResearchRequestError as exc:
            bindings = {}
            blockers.append(_blocker(_error_code(exc), "研究计划无法锁定所有输入文件。", "确认模型、数据、评测器和适配器资源都存在且可读取。", str(exc)))
        _report_progress(progress_callback, "input_digest", "completed" if bindings else "blocked")
    input_digest = _input_digest(bindings) if bindings else None
    state = "blocked" if blockers else ("ready_with_deferred_discovery" if mode_plan and mode_plan.get("state") == "ready_with_deferred_discovery" else "ready")
    request_id = "research-request-" + hashlib.sha256(_canonical({"goal": normalized_goal.strip(), "model": str(model_path), "data": str(data_path), "input_digest": input_digest}).encode("utf-8")).hexdigest()[:24]
    plan: dict[str, object] = {
        "schema_version": 3,
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
        "protected_metrics": frozen_protected_metrics,
        "protected_metrics_requested": (
            requested_protected_metrics if protected_metrics is not None else None
        ),
        "budget": {"gpu_hours": budget_hours},
        "execution_policy": execution_policy,
        "warnings": policy_warnings(execution_policy, policy_inputs),
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
        "claim_boundary": "阻塞计划只报告接入状态，输入尚未冻结；就绪计划只证明输入已绑定、执行路径可编译和门禁齐全。任何真实提升都必须由目标侧冻结 verifier、配对试验和 held-out confirmation 建立，检索、代码生成和校准不能单独形成科学结论。",
    }
    plan["plan_digest"] = hashlib.sha256(_canonical(plan).encode("utf-8")).hexdigest()
    try:
        validate_document("research_plan", plan, root=repo_root)
    except ContractValidationError as exc:
        raise ResearchRequestError(f"RESEARCH_PLAN_INVALID:{exc}") from exc
    _report_progress(progress_callback, "plan_compile", "completed")
    return plan


def validate_research_plan(payload: Mapping[str, object]) -> None:
    """Apply the same artifact and integrity boundary before saving or using it."""

    try:
        validate_document("research_plan", payload, root=Path(__file__).resolve().parents[2])
    except ContractValidationError as exc:
        raise ResearchRequestError(f"RESEARCH_PLAN_INVALID:{exc}") from exc
    unsigned = dict(payload)
    unsigned.pop("plan_digest", None)
    expected = hashlib.sha256(_canonical(unsigned).encode("utf-8")).hexdigest()
    if payload.get("plan_digest") != expected:
        raise ResearchRequestError("RESEARCH_PLAN_DIGEST_MISMATCH")


def write_research_plan(plan: Mapping[str, object], output: str | Path) -> Path:
    """Validate and atomically write a plan artifact."""

    payload = dict(plan)
    validate_research_plan(payload)
    _ensure_outside_inputs(Path(output), tuple(_resolve(payload.get(name), base=Path(payload.get("project_root", Path.cwd()))) for name in ("model", "data", "source")))
    try:
        return _write_json(Path(output), payload)
    except OSError as exc:
        raise ResearchRequestError("RESEARCH_PLAN_OUTPUT_INVALID") from exc


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
    validate_research_plan(payload)
    return payload


def verify_research_plan_inputs(
    plan: Mapping[str, object], *, progress_callback: ResearchProgressCallback | None = None
) -> dict[str, object]:
    bindings = plan.get("input_bindings")
    if not isinstance(bindings, Mapping) or not bindings:
        raise ResearchRequestError("RESEARCH_PLAN_INPUT_BINDINGS_MISSING")
    current: dict[str, dict[str, object]] = {}
    drift: list[dict[str, object]] = []
    _report_progress(progress_callback, "input_verification", "started")
    for name, expected in sorted(bindings.items()):
        if not isinstance(expected, Mapping) or not isinstance(expected.get("path"), str):
            raise ResearchRequestError("RESEARCH_PLAN_INPUT_BINDING_INVALID")
        try:
            observed = _path_digest(
                Path(str(expected["path"])),
                source_tree=_is_source_binding(str(name)),
                progress=progress_callback,
                input_name=str(name),
            )
        except ResearchRequestError as exc:
            drift.append({"name": name, "code": _error_code(exc), "detail": str(exc)})
            continue
        current[str(name)] = observed
        if observed.get("sha256") != expected.get("sha256") or observed.get("kind") != expected.get("kind"):
            drift.append({"name": name, "code": "INPUT_DIGEST_DRIFT", "expected": expected.get("sha256"), "observed": observed.get("sha256")})
    observed_digest = _input_digest(current)
    if observed_digest != plan.get("input_digest"):
        drift.append({"name": "input_bindings", "code": "INPUT_DIGEST_DRIFT", "expected": plan.get("input_digest"), "observed": observed_digest})
    _report_progress(progress_callback, "input_verification", "blocked" if drift else "completed")
    return {"state": "verified" if not drift else "blocked", "input_digest": observed_digest, "drift": drift}


def _verify_protected_metrics_preview(plan: Mapping[str, object]) -> None:
    """Ensure a ready plan's display fields still match its bound frozen goal."""

    if "protected_metrics" not in plan:
        return  # Historical schema versions did not carry this preview.
    bindings = plan.get("input_bindings")
    frozen_binding = bindings.get("frozen_goal") if isinstance(bindings, Mapping) else None
    expected: list[str] = []
    if isinstance(frozen_binding, Mapping):
        path_value = frozen_binding.get("path")
        if not isinstance(path_value, str) or not path_value:
            raise ResearchRequestError("PROTECTED_METRICS_FROZEN_GOAL_BINDING_INVALID")
        try:
            goal = parse_yaml_document_text(
                Path(path_value).read_bytes().decode("utf-8"), source=path_value
            )
            policy = verification_policy_from_goal(goal)
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            raise ResearchRequestError("PROTECTED_METRICS_FROZEN_GOAL_INVALID") from exc
        if policy is None:
            raise ResearchRequestError("PROTECTED_METRICS_FROZEN_GOAL_POLICY_REQUIRED")
        expected = list(policy.protected_metrics)
    if plan.get("protected_metrics") != expected:
        raise ResearchRequestError("PROTECTED_METRICS_PLAN_BINDING_MISMATCH")
    requested = plan.get("protected_metrics_requested")
    if requested is not None and (
        not isinstance(requested, Sequence)
        or isinstance(requested, (str, bytes, bytearray))
        or set(requested) != set(expected)
    ):
        raise ResearchRequestError("PROTECTED_METRICS_PLAN_REQUEST_MISMATCH")


def plan_to_campaign_payload(
    plan: Mapping[str, object], *, progress_callback: ResearchProgressCallback | None = None
) -> dict[str, object]:
    if plan.get("state") not in {"ready", "ready_with_deferred_discovery"}:
        raise ResearchRequestError("RESEARCH_PLAN_BLOCKED")
    validate_research_plan(plan)
    verification = verify_research_plan_inputs(plan, progress_callback=progress_callback)
    if verification["state"] != "verified":
        raise ResearchRequestError("RESEARCH_PLAN_INPUT_DIGEST_DRIFT")
    _verify_protected_metrics_preview(plan)
    preview = plan.get("adapter_preview")
    if not isinstance(preview, Mapping):
        raise ResearchRequestError("RESEARCH_PLAN_ADAPTER_PREVIEW_MISSING")
    assets = preview.get("asset_bindings")
    payload: dict[str, object] = {
        "goal": str(plan["goal"]),
        "model": str(plan["model"]),
        "dataset": str(plan["data"]),
        "source": str(plan["source"]) if plan.get("source") else None,
        "budget": plan["budget"],
        "adapter": str(plan.get("adapter") or "auto"),
        "adapter_profile_path": str(plan["adapter_profile"]) if plan.get("adapter_profile") else None,
        "runtime_python": str(plan["runtime_python"]) if plan.get("runtime_python") else None,
        "research_mode": str(plan["mode"]),
        "literature_query": str(plan["goal"]),
        "open_method_policy": dict(plan.get("open_method_policy") or _OPEN_METHOD_POLICY),
        "execution_policy": plan_execution_policy(plan),
        "plan_binding": {"plan_id": plan.get("plan_id"), "plan_digest": plan.get("plan_digest"), "input_digest": plan.get("input_digest")},
    }
    if plan.get("schema_version") == 3:
        payload["campaign_id"] = "research-" + str(plan["plan_digest"])[:24]
    if isinstance(assets, Mapping):
        payload["assets"] = {str(key): str(value) for key, value in assets.items()}
    if plan.get("evaluator_contract"):
        payload["evaluator_contract"] = str(plan["evaluator_contract"])
    if plan.get("runtime_python"):
        payload["runtime_python"] = str(plan["runtime_python"])
    metrics = list(plan.get("target_metrics") or ())
    if metrics:
        payload["target_metrics"] = metrics
    if plan.get("model_irg"):
        payload["model_irg_path"] = str(plan["model_irg"])
        payload["irg_protected_metrics"] = list(plan.get("irg_protected_metrics") or ())
    return {key: value for key, value in payload.items() if value is not None}


def terminal_research_result(campaign: Mapping[str, object]) -> dict[str, object] | None:
    """Present a durable terminal attempt without confirming or rerunning it."""

    status = str(campaign.get("status"))
    if status not in {"completed", "blocked", "failed", "cancelled"}:
        return None
    if status in {"blocked", "failed"}:
        next_step = (
            "本次尝试已经终止。查看 campaign 中的 execution_error / execution_result 处理根因；"
            "源码或策略修复后可用原计划再次 /run，由系统创建新 revision。"
            "若要在相同输入下创建新尝试，重新 /plan，再审阅并 /run。"
            "脚本入口对应 verdi research plan 与 verdi research run --plan PLAN --confirm。"
        )
    else:
        next_step = "本次尝试已结束。需要新实验时重新 /plan，审阅新计划后 /run；旧结果与证据保留。"
    return {"state": status, "campaign": dict(campaign), "reused_terminal_result": True, "next_step": next_step}


def matching_terminal_research_result(
    plan: Mapping[str, object], campaigns: Sequence[Mapping[str, object]],
) -> dict[str, object] | None:
    """Find a terminal attempt bound to this exact plan, for read-only UI."""

    digest = plan.get("plan_digest")
    if not isinstance(digest, str) or not digest:
        return None
    for campaign in campaigns:
        execution = campaign.get("execution")
        binding = execution.get("research_plan_binding") if isinstance(execution, Mapping) else None
        bound_digest = binding.get("plan_digest") if isinstance(binding, Mapping) else None
        if bound_digest == digest or (
            binding is None
            and plan.get("schema_version") == 3
            and campaign.get("campaign_id") == "research-" + digest[:24]
        ):
            return terminal_research_result(campaign)
    return None


def run_research_plan(
    plan: Mapping[str, object],
    *,
    store: Any,
    confirm: bool,
    queue_only: bool = False,
    max_parallel: int = 1,
    rounds: int | None = None,
    stop_on_success: bool | None = None,
    progress_callback: ResearchProgressCallback | None = None,
) -> dict[str, object]:
    """Create/confirm/dispatch a plan through CampaignStore after one confirmation."""

    original_digest = plan.get("plan_digest")
    try:
        selected = derive_research_plan_policy(plan, rounds=rounds, stop_on_success=stop_on_success)
    except ResearchExecutionPolicyError as exc:
        raise ResearchRequestError(str(exc)) from exc
    metadata: dict[str, object] = {"execution_policy": plan_execution_policy(selected), "budget": selected["budget"], "policy_changed": selected.get("plan_digest") != original_digest}
    if metadata["policy_changed"]:
        from wmloop.control.research_plan_history import write_default_research_plan

        destination = write_default_research_plan(selected, Path(str(selected["project_root"])))
        metadata.update(plan_path=str(destination), derived_plan=selected)
    plan = selected
    if not confirm:
        return {"state": "awaiting_confirmation", "plan": plan, "plan_id": plan.get("plan_id"), "plan_digest": plan.get("plan_digest"), "next_step": "检查轮数、停止条件和跨轮总预算后，运行 verdi research run --plan PLAN --confirm。", **metadata}
    payload = plan_to_campaign_payload(plan, progress_callback=progress_callback)
    created = store.create(payload)
    terminal = terminal_research_result(created)
    if terminal is not None:
        return {**terminal, "plan_id": plan.get("plan_id"), **metadata}
    queued = store.confirm(str(created["campaign_id"]))
    if queue_only:
        return {"state": "queued", "campaign": queued, "plan_id": plan.get("plan_id"), **metadata}
    from wmloop.control.campaign_dispatcher import DispatcherOptions, run_dispatcher

    dispatcher = run_dispatcher(DispatcherOptions(state_root=store.root, max_cycles=1, max_parallel=max_parallel, campaign_ids=(str(created["campaign_id"]),)))
    campaign = store.get(str(created["campaign_id"]))
    return {"state": str(campaign.get("status")), "campaign": campaign, "dispatcher": dispatcher, "plan_id": plan.get("plan_id"), **metadata}
