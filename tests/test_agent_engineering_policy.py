from __future__ import annotations

from wmloop.control.agent_engineering_policy import engineering_policy, render_engineering_policy


def test_engineering_policy_contains_ci_harness_and_no_compromise_rules() -> None:
    policy = engineering_policy()

    assert policy["policy_id"] == "verdiwm_agent_engineering_policy_v1"
    assert "hermetic_harness" in policy["principles"]
    assert "ci_first_for_control_plane" in policy["principles"]
    assert "no_silent_compromise" in policy["principles"]
    assert "narrow_pytest_for_changed_contracts" in policy["required_ci_harness_receipts"]
    assert "modify_frozen_evaluator_or_data_to_pass" in policy["forbidden_shortcuts"]


def test_rendered_policy_is_prompt_ready() -> None:
    rendered = render_engineering_policy()

    assert "Google-style software engineering discipline" in rendered
    assert "Required CI/harness receipts" in rendered
    assert "Forbidden shortcuts" in rendered
    assert "Do not substitute" in rendered
