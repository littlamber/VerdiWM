"""Shared engineering policy for agent-authored VerdiWM code changes."""

from __future__ import annotations

from collections.abc import Mapping


def engineering_policy() -> dict[str, object]:
    """Return the shared Google-style engineering discipline for code-writing agents."""

    return {
        "policy_id": "verdiwm_agent_engineering_policy_v2",
        "authority_document": "AGENTS.md",
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
            "Read AGENTS.md and docs/code_conventions.md; preserve VerdiWM naming, the wmloop package, and the locked Python 3.12 environment.",
            "Follow docs/PROJECT_LAYOUT.md: extracted components must not import their compatibility facade or form cycles; reuse wmloop.storage while preserving historical receipt bytes and keep component files within 800 lines.",
            "Before adding code, find the owning service, existing callers, schemas and tests; use the shared capability index in docs/code_conventions.md section 6 and explain any necessary new interface.",
            "Fix shared behavior in its owning service; keep entrypoints as orchestration and model adapters as command/input/output translation, with no duplicate budget, lease, verdict or recovery state machine.",
            "Prefer explicit functions, contracts and dependencies; add abstractions only for concrete shared semantics, avoid expanding context dictionaries or boolean switches, and keep compatibility wrappers as delegation only.",
            "Resolve configurable resource paths at call time, keep package exports lazy, and never require a developer checkout path or sys.path/PYTHONPATH patch.",
            "Reuse wmloop.process_lifecycle for CPU process cancellation and retirement; shared helpers must not make retrieval or CPU evaluation indirectly load execute or torch.",
            "State configuration intent, exact code touchpoints, input/output contract, and failure semantics before editing.",
            "Keep patches narrowly scoped to allowed mutation paths and avoid unrelated refactors.",
            "Add or update offline fixture/unit tests for every behavior that can be tested without GPU or network.",
            "Add runtime smoke only as a separate receipt when real GPU/model execution is required.",
            "Use stable explicit error codes for fail-closed blockers and contract violations.",
            "Catch exceptions only to add domain context, clean up owned resources or follow an explicit recovery protocol; never hide failures behind empty results, default success or bypass flags.",
            "Preserve frozen evaluator, data split, goal, verdict probe, and registry authority boundaries.",
            "Do not substitute a weaker proxy, stub, disabled hook, or evaluator-side shortcut for the requested method.",
            "Report exact validation commands and artifacts; readiness claims require receipts, manifests, or tests.",
        ],
        "method_boundaries": [
            "Characterize: use matched contexts and seeds for reversible inference-only probes; retain response uncertainty, support masks and adapter hooks without updating weights or the goal.",
            "Retrieve: compatible IRG neighbors and cross-domain literature produce ranking-only hypotheses; preserve source provenance and abstain on insufficient support.",
            "Validate: only paired target experiments and the frozen verifier settle effects; positive gain requires its lower confidence bound above the goal threshold and every protected check passing.",
            "Accumulate: retain confirmed, null, harmful and jointly measured interaction outcomes with support, uncertainty, cost, operator version and implementation receipts; never overwrite source evidence.",
            "Probe evolution: comparable nearby opposing settled effects motivate a proposal; promotion requires held-out selection improvement, calibration checks and a later version, never an isolated failure alone.",
            "Keep operational failures, incomplete screens and diagnostic receipts distinct from confirmed operator-effect evidence.",
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
    if payload.get("method_boundaries"):
        lines.append("- VERDI method boundaries:")
        lines.extend(f"  - {item}" for item in _strings(payload.get("method_boundaries")))
    lines.append("- Required CI/harness receipts:")
    lines.extend(f"  - {item}" for item in _strings(payload.get("required_ci_harness_receipts")))
    lines.append("- Forbidden shortcuts:")
    lines.extend(f"  - {item}" for item in _strings(payload.get("forbidden_shortcuts")))
    return "\n".join(lines)


def _strings(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item]
