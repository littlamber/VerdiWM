"""First-contact bootstrap for a model-family executor.

The bootstrap owns the retry policy around adapter discovery and repair.  It
does not contain model-specific logic: a reviewed base adapter and the existing
bounded adapter-repair provider are the extension points.  Every outcome is a
durable, explicit state so an unknown model cannot silently run with a proxy.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from wmloop.control.adapter_profiles import (
    AdapterProfileError,
    ResolvedAdapter,
    compile_adapter_execution,
)


class ModelExecutorBootstrapError(RuntimeError):
    """The first-contact executor bootstrap could not be completed."""


AdapterRepairRunner = Callable[..., Mapping[str, object]]


def bootstrap_model_executor(
    *,
    model: Path,
    data: Path,
    goal: str,
    budget: object,
    campaign_root: Path,
    project_root: Path,
    adapter: str | None = "auto",
    adapter_profile_path: Path | None = None,
    runtime_python: Path | None = None,
    asset_overrides: Mapping[str, object] | None = None,
    base_profile_path: Path | None = None,
    llm_adapter: Mapping[str, object] | None = None,
    repair_output_root: Path | None = None,
    max_attempts: int = 3,
    repair_runner: AdapterRepairRunner | None = None,
) -> dict[str, object]:
    """Resolve an executor, triggering bounded repair on first contact.

    ``repair_runner`` is injectable for tests and deployments that wrap the
    standard :func:`run_adapter_repair`; production callers normally omit it.
    The returned manifest never grants GPU or promotion authority.
    """

    root = Path(project_root).expanduser().resolve()
    model_path = Path(model).expanduser().resolve()
    data_path = Path(data).expanduser().resolve()
    try:
        resolved = compile_adapter_execution(
            campaign_id=Path(campaign_root).name,
            model=model_path,
            data=data_path,
            goal=goal,
            budget=budget,
            campaign_root=Path(campaign_root),
            adapter=adapter,
            adapter_profile_path=adapter_profile_path,
            runtime_python=runtime_python,
            asset_overrides=asset_overrides,
            project_root=root,
        )
        return _ready_manifest(resolved, source="existing_profile")
    except AdapterProfileError as exc:
        initial_error = str(exc)
        if not exc.repairable:
            return _blocked_manifest("adapter_resolution", initial_error)

    if base_profile_path is None or llm_adapter is None:
        return _blocked_manifest(
            "adapter_repair",
            "ADAPTER_REPAIR_INPUTS_REQUIRED",
            detail="Provide a trusted base profile and configured repair provider for first-contact bootstrap.",
            required_inputs=[
                {
                    "key": "base_profile_path",
                    "kind": "trusted_artifact",
                    "action": "Provide or approve a versioned adapter profile whose evaluator and constitution are trusted.",
                },
                {
                    "key": "llm_adapter",
                    "kind": "bounded_provider",
                    "action": "Configure the approved provider used only for adapter-and-test generation.",
                },
            ],
        )
    runner = repair_runner
    if runner is None:
        from wmloop.control.adapter_repair import run_adapter_repair

        runner = run_adapter_repair
    destination = Path(repair_output_root or (Path(campaign_root) / "adapter-repair"))
    try:
        repair = dict(
            runner(
                model=model_path,
                data=data_path,
                goal=goal,
                budget=budget,
                failure_code="ADAPTER_PROFILE_NOT_FOUND",
                base_profile_path=Path(base_profile_path).expanduser().resolve(),
                llm_adapter=llm_adapter,
                output_root=destination,
                project_root=root,
                runtime_python=runtime_python,
                max_attempts=max_attempts,
            )
        )
    except Exception as exc:
        return _blocked_manifest(
            "adapter_repair",
            "ADAPTER_REPAIR_FAILED",
            detail=f"{type(exc).__name__}: {str(exc)[:400]}",
        )
    if repair.get("state") != "ready" or not isinstance(repair.get("adapter_profile_path"), str):
        return _blocked_manifest(
            "adapter_repair",
            "ADAPTER_REPAIR_NOT_READY",
            detail="The bounded repair provider did not produce a conformance-ready profile.",
            repair=repair,
        )
    try:
        resolved = compile_adapter_execution(
            campaign_id=Path(campaign_root).name,
            model=model_path,
            data=data_path,
            goal=goal,
            budget=budget,
            campaign_root=Path(campaign_root),
            adapter_profile_path=Path(str(repair["adapter_profile_path"])),
            runtime_python=runtime_python,
            asset_overrides=asset_overrides,
            project_root=root,
        )
    except AdapterProfileError as exc:
        return _blocked_manifest(
            "adapter_recompile",
            "ADAPTER_REPAIR_PROFILE_INVALID",
            detail=str(exc),
            repair=repair,
        )
    return _ready_manifest(resolved, source="repaired_profile", repair=repair)


def write_bootstrap_manifest(path: Path, manifest: Mapping[str, object]) -> Path:
    """Persist one bootstrap outcome without mutating model or evaluator files."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(manifest, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return destination


def _ready_manifest(
    resolved: ResolvedAdapter,
    *,
    source: str,
    repair: Mapping[str, object] | None = None,
) -> dict[str, object]:
    body: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "verdiwm-model-executor-bootstrap",
        "state": "ready",
        "source": source,
        "profile_id": resolved.profile_id,
        "model_family": resolved.model_family,
        "capability_level": resolved.capability_level,
        "execution": resolved.execution,
        "constitution_freeze": resolved.constitution_freeze,
        "repair_manifest": dict(repair) if repair is not None else None,
        "authority": {"gpu_scheduling": False, "promotion": False},
        "claim_boundary": "Executor interface readiness only; model-quality evidence still requires the frozen evaluator.",
    }
    body["bootstrap_digest"] = _digest(body)
    return body


def _blocked_manifest(
    stage: str,
    code: str,
    *,
    detail: str | None = None,
    repair: Mapping[str, object] | None = None,
    required_inputs: list[Mapping[str, object]] | None = None,
) -> dict[str, object]:
    body: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "verdiwm-model-executor-bootstrap",
        "state": "blocked",
        "stage": stage,
        "blocker": {"code": code, **({"detail": detail} if detail else {})},
        "repair_manifest": dict(repair) if repair is not None else None,
        "required_inputs": [dict(item) for item in (required_inputs or [])],
        "authority": {"gpu_scheduling": False, "promotion": False},
        "claim_boundary": "No executor is available; no training, evaluation, or promotion may start.",
    }
    body["bootstrap_digest"] = _digest(body)
    return body


def _digest(value: Mapping[str, object]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")).hexdigest()
