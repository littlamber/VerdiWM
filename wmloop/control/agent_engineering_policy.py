"""Shared engineering policy for agent-authored VerdiWM code changes."""

from __future__ import annotations

from collections.abc import Mapping


def engineering_policy() -> dict[str, object]:
    """Return the shared Google-style engineering discipline for code-writing agents."""

    return {
        "policy_id": "verdiwm_agent_engineering_policy_v1",
        "basis": "Google-style software engineering discipline adapted to VerdiWM authority boundaries.",
        "principles": [
            "design_before_code",
            "small_reviewable_changes",
            "hermetic_harness",
            "ci_first_for_control_plane",
            "readability_and_stable_errors",
            "no_hidden_coupling",
            "no_silent_compromise",
            "evidence_over_assertion",
            "backward_compatible_contracts",
            "security_and_reproducibility",
        ],
        "required_practices": [
            "State configuration intent, exact code touchpoints, input/output contract, and failure semantics before editing.",
            "Keep patches narrowly scoped to allowed mutation paths and avoid unrelated refactors.",
            "Add or update offline fixture/unit tests for every behavior that can be tested without GPU or network.",
            "Add runtime smoke only as a separate receipt when real GPU/model execution is required.",
            "Use stable explicit error codes for fail-closed blockers and contract violations.",
            "Preserve frozen evaluator, data split, goal, verdict probe, and registry authority boundaries.",
            "Do not substitute a weaker proxy, stub, disabled hook, or evaluator-side shortcut for the requested method.",
            "Report exact validation commands and artifacts; readiness claims require receipts, manifests, or tests.",
        ],
        "required_ci_harness_receipts": [
            "py_compile_or_import_check_for_changed_python",
            "narrow_pytest_for_changed_contracts",
            "schema_validation_for_new_json_outputs",
            "fixture_smoke_for_new_probe_or_adapter",
            "negative_test_for_forbidden_surface_or_disabled_hook",
        ],
        "forbidden_shortcuts": [
            "modify_frozen_evaluator_or_data_to_pass",
            "change_goal_or_verdict_probe_during_active_campaign",
            "skip_tests_because_change_is_small",
            "hide_implementation_compromise_inside_runtime_defaults",
            "claim_gpu_or_quality_success_from_cpu_only_checks",
            "write_secrets_or_untracked_external_state_into_artifacts",
        ],
    }


def render_engineering_policy(policy: Mapping[str, object] | None = None) -> str:
    """Render a compact prompt section for agent instruction packets."""

    payload = dict(policy or engineering_policy())
    lines = [
        "Engineering practice policy:",
        f"- Policy: {payload['policy_id']}",
        f"- Basis: {payload['basis']}",
        "- Required practices:",
    ]
    lines.extend(f"  - {item}" for item in _strings(payload.get("required_practices")))
    lines.append("- Required CI/harness receipts:")
    lines.extend(f"  - {item}" for item in _strings(payload.get("required_ci_harness_receipts")))
    lines.append("- Forbidden shortcuts:")
    lines.extend(f"  - {item}" for item in _strings(payload.get("forbidden_shortcuts")))
    return "\n".join(lines)


def _strings(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item]
