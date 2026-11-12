"""Read-only onboarding scan components: readiness."""

from __future__ import annotations

import json
from typing import Iterable, Mapping, Sequence


def _build_blockers(
    *,
    source_revision: Mapping[str, object],
    runtime: Mapping[str, object],
    entrypoints: Sequence[Mapping[str, object]],
    assets: Sequence[Mapping[str, object]],
    capabilities: Sequence[Mapping[str, object]],
    evaluator_contract: Mapping[str, object],
    asset_bindings: Sequence[Mapping[str, object]],
    adapter_contract_ready: bool = False,
) -> list[dict[str, object]]:
    blockers: list[dict[str, object]] = []
    if source_revision.get("state") != "bound":
        blockers.append(
            {
                "code": "SOURCE_REVISION_UNBOUND",
                "detail": source_revision.get("detail", "source revision is not bound"),
            }
        )
    if runtime.get("state") != "ready":
        blockers.append(
            {
                "code": "RUNTIME_UNREADY",
                "detail": "selected Python runtime is missing required imports or has dependency conflicts",
            }
        )
        for probe in runtime.get("import_probes", []):
            if probe.get("state") != "ok":
                blockers.append(
                    {
                        "code": (
                            "RUNTIME_DEPENDENCY_MISSING"
                            if probe.get("state") == "failed"
                            else "RUNTIME_DEPENDENCY_PROBE_FAILED"
                        ),
                        "package": probe.get("package"),
                        "detail": probe.get("detail", ""),
                    }
                )
        if runtime.get("pip_check", {}).get("state") == "failed":
            blockers.append(
                {
                    "code": "RUNTIME_DEPENDENCY_CONFLICT",
                    "detail": runtime["pip_check"].get("detail", ""),
                }
            )  # type: ignore[index]
    if not entrypoints:
        blockers.append(
            {
                "code": "ENTRYPOINT_NOT_DISCOVERED",
                "detail": "no train, evaluation, inference, or rollout entrypoint was discovered",
            }
        )
    if not adapter_contract_ready and not any(
        asset.get("kind") == "checkpoint" for asset in assets
    ):
        blockers.append(
            {
                "code": "CHECKPOINT_MISSING",
                "detail": "no checkpoint or weight asset was discovered",
            }
        )
    if (
        not adapter_contract_ready
        and evaluator_contract.get("state") != "ready"
        and not any(
            entrypoint.get("executable") is True
            and "evaluation" in entrypoint.get("kinds", [])
            for entrypoint in entrypoints
        )
    ):
        blockers.append(
            {
                "code": "EVALUATION_ENTRYPOINT_MISSING",
                "detail": "no evaluation entrypoint was discovered",
            }
        )
    if evaluator_contract.get("state") != "ready":
        blockers.append(
            {
                "code": "EVALUATOR_CONTRACT_REQUIRED",
                "detail": "a frozen evaluator contract is required before scheduling",
            }
        )
    for binding in asset_bindings:
        if (
            binding.get("required_for_evaluator") is True
            and binding.get("state") != "discovered"
        ):
            blockers.append(
                {
                    "code": "MODEL_ASSET_BINDING_REQUIRED",
                    "parameter": binding.get("parameter"),
                    "detail": f"no repository asset matches {binding.get('parameter')}",
                }
            )
    if not any(item.get("state") == "discovered" for item in capabilities):
        blockers.append(
            {
                "code": "CAPABILITY_NOT_DISCOVERED",
                "detail": "no supported model capability was evidenced",
            }
        )
    return _dedupe_blockers(blockers)


def _dedupe_blockers(
    blockers: Iterable[Mapping[str, object]],
) -> list[dict[str, object]]:
    rows = [dict(item) for item in blockers]
    rows.sort(
        key=lambda item: (
            str(item.get("code", "")),
            str(item.get("package", "")),
            str(item.get("detail", "")),
        )
    )
    unique: list[dict[str, object]] = []
    seen: set[str] = set()
    for row in rows:
        key = json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        if key not in seen:
            unique.append(row)
            seen.add(key)
    return unique


def _state(
    blockers: Sequence[Mapping[str, object]],
    *,
    runtime: Mapping[str, object],
    evaluator_contract: Mapping[str, object],
) -> str:
    if blockers:
        return "blocked"
    if runtime.get("state") == "ready" and evaluator_contract.get("state") == "ready":
        return "ready_for_conformance_smoke"
    return "binding_required"


def _conformance_report() -> dict[str, object]:
    return {
        "state": "not_run",
        "gpu_execution_started": False,
        "checks": [
            {"name": "runtime_import", "state": "pending"},
            {"name": "entrypoint_help", "state": "pending"},
            {"name": "evaluator_contract", "state": "pending"},
            {"name": "output_isolation", "state": "pending"},
        ],
        "receipt_path": None,
    }


def _next_actions(state: str, blockers: Sequence[Mapping[str, object]]) -> list[str]:
    actions: list[str] = []
    codes = {str(item.get("code")) for item in blockers}
    if "RUNTIME_UNREADY" in codes:
        actions.append(
            "Select or provision a Python runtime satisfying the discovered dependency files."
        )
    if "SOURCE_REVISION_UNBOUND" in codes:
        actions.append(
            "Bind the source to a Git commit before publishing reusable experiment evidence."
        )
    if "EVALUATOR_CONTRACT_REQUIRED" in codes:
        actions.append(
            "Provide a frozen evaluator contract with inputs, outputs, metrics, verifier, and command."
        )
    if "EVALUATION_ENTRYPOINT_MISSING" in codes:
        actions.append(
            "Declare an evaluation entrypoint; training or rollout alone cannot authorize optimization."
        )
    if state == "ready_for_conformance_smoke":
        actions.append(
            "Run the CPU-safe conformance smoke, then promote only its passing receipt to the scheduler."
        )
    if not actions:
        actions.append(
            "Review the generated connector and run conformance before any GPU experiment."
        )
    return actions
