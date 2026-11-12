from __future__ import annotations

from wmloop.control.agent_engineering_policy import engineering_policy, render_engineering_policy


def test_engineering_policy_contains_ci_harness_and_no_compromise_rules() -> None:
    policy = engineering_policy()

    assert policy["policy_id"] == "verdiwm_agent_engineering_policy_v2"
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
    for stage in ("Characterize", "Retrieve", "Validate", "Accumulate", "Probe evolution"):
        assert stage + ":" in rendered
    assert "AGENTS.md" in rendered
    assert "ranking-only" in rendered
    assert "never overwrite source evidence" in rendered
    assert "compatibility facade" in rendered
    assert "wmloop.storage" in rendered
    assert "call time" in rendered


def test_historical_policy_packets_remain_renderable() -> None:
    policy = engineering_policy()
    policy.pop("method_boundaries")
    policy["policy_id"] = "verdiwm_agent_engineering_policy_v1"
    rendered = render_engineering_policy(policy)
    assert "verdiwm_agent_engineering_policy_v1" in rendered
    assert "VERDI method boundaries" not in rendered
