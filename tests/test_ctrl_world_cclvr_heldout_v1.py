from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scripts.aggregate_ctrl_world_cclvr_heldout_v1 import (
    CCLVRHeldoutAggregationError,
    local_error,
    routing_report,
)
from scripts.evaluate_ctrl_world_cclvr_heldout_v1 import CCLVRRouteProbe
from scripts.run_ctrl_world_cclvr_heldout_v1 import (
    CCLVRHeldoutLaunchError,
    PLAN_V2,
    _validate_adapter_source,
)


class FakeCorrector(torch.nn.Module):
    def __init__(self, unforced_arms: list[int], *, normalized_dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.local_arm_value_head = object()
        self.multiscale_side_adapter = object()
        self.local_arm_normalized_dose = 0.99
        self.unforced_arms = list(unforced_arms)
        self.normalized_dtype = normalized_dtype
        self.unforced_index = 0
        self.received_overrides: list[float | None] = []

    def forward(self, *_args: object, adapter_scale_override=None, **_kwargs: object) -> object:
        override = None if adapter_scale_override is None else float(adapter_scale_override)
        self.received_overrides.append(override)
        if override is None:
            arm = self.unforced_arms[self.unforced_index]
            self.unforced_index += 1
            dose = (-0.99, 0.0, 0.99)[arm]
        else:
            arm = 1
            dose = override
        residual = torch.full((1, 1, 1, 1, 1), abs(dose))
        return SimpleNamespace(
            selected_arm=torch.tensor([arm]),
            normalized_adapter_dose=torch.tensor([dose], dtype=self.normalized_dtype),
            runtime_features=torch.ones(1, 1, 2),
            down_block_additional_residuals=(residual,),
            mid_block_additional_residual=residual,
        )

    def predict_local_arm_policy(self, _features: torch.Tensor, *, hard: bool = False) -> object:
        assert hard is False
        return SimpleNamespace(
            probabilities=torch.tensor([[0.1, 0.2, 0.7]]),
            values=torch.tensor([[-0.1, 0.0, 0.3]]),
            adjusted_values=torch.tensor([[-0.1, 0.0, 0.3]]),
        )


def test_v2_plan_binds_the_loaded_ctrl_world_adapter_source(tmp_path: Path) -> None:
    ctrl_world = tmp_path / "Ctrl-World"
    adapter = ctrl_world / "models" / "multiscale_history_adapter.py"
    adapter.parent.mkdir(parents=True)
    adapter.write_text("REPAIRED = True\n", encoding="utf-8")
    digest = hashlib.sha256(adapter.read_bytes()).hexdigest()
    dependencies = {
        "ctrl_world_root": str(ctrl_world),
        "adapter_source": str(adapter),
        "adapter_source_sha256": digest,
    }

    _validate_adapter_source(PLAN_V2, dependencies)

    dependencies["adapter_source_sha256"] = "0" * 64
    with pytest.raises(CCLVRHeldoutLaunchError, match="ADAPTER_SOURCE_HASH_MISMATCH"):
        _validate_adapter_source(PLAN_V2, dependencies)


def _call(probe: CCLVRRouteProbe, corrector: FakeCorrector, interaction: int, count: int = 2) -> None:
    probe.set_interaction(interaction)
    for _ in range(count):
        corrector.forward(torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1))


def test_interaction_cached_route_selects_once_and_reuses_exact_dose() -> None:
    corrector = FakeCorrector([2, 0])
    model = SimpleNamespace(history_corrector=corrector)
    with CCLVRRouteProbe(model, mode="learned_cached", route_scope="interaction") as probe:
        _call(probe, corrector, 0)
        _call(probe, corrector, 1)
    audit = probe.result()
    assert corrector.received_overrides == [None, 0.99, None, -0.99]
    assert [row["selected_arm"] for row in audit["decision_records"]] == [2, 0]
    assert [row["applied_dose"] for row in audit["invocations"]] == pytest.approx(
        [0.99, 0.99, -0.99, -0.99]
    )


def test_episode_cached_route_is_reused_across_interactions() -> None:
    corrector = FakeCorrector([2])
    model = SimpleNamespace(history_corrector=corrector)
    with CCLVRRouteProbe(model, mode="learned_cached", route_scope="episode") as probe:
        _call(probe, corrector, 0)
        _call(probe, corrector, 1)
    audit = probe.result()
    assert corrector.received_overrides == [None, 0.99, 0.99, 0.99]
    assert len(audit["decision_records"]) == 1
    assert {row["cache_key"] for row in audit["invocations"]} == {0}


def test_bfloat16_selected_dose_is_canonicalized_before_caching() -> None:
    corrector = FakeCorrector([2], normalized_dtype=torch.bfloat16)
    model = SimpleNamespace(history_corrector=corrector)

    with CCLVRRouteProbe(model, mode="learned_cached", route_scope="episode") as probe:
        _call(probe, corrector, 0)

    assert float(torch.tensor(0.99, dtype=torch.bfloat16)) == pytest.approx(0.98828125)
    assert corrector.received_overrides == [None, 0.99]
    assert probe.result()["decision_records"][0]["selected_dose"] == 0.99


def test_fixed_endpoint_uses_direct_override_only_at_target_interaction() -> None:
    corrector = FakeCorrector([])
    model = SimpleNamespace(history_corrector=corrector)
    with CCLVRRouteProbe(
        model,
        mode="fixed",
        route_scope="interaction",
        fixed_dose=-0.99,
        target_interaction=1,
    ) as probe:
        _call(probe, corrector, 0)
        _call(probe, corrector, 1)
    audit = probe.result()
    assert corrector.received_overrides == [0.0, 0.0, -0.99, -0.99]
    assert audit["decision_records"] == []


def _measurement(means: list[float], *, decisions: list[dict[str, object]] | None = None) -> dict[str, object]:
    interactions = [
        {
            "interaction": index,
            "mean_l1": value,
            "final_l1": value,
            "mean_psnr": 20.0,
            "final_psnr": 20.0,
        }
        for index, value in enumerate(means)
    ]
    slope = float(np.polyfit(np.arange(4), np.asarray(means), 1)[0])
    return {
        "interactions": interactions,
        "outcomes": {
            "negative_mean_l1": -float(np.mean(means)),
            "negative_final_interaction_l1": -means[-1],
            "negative_horizon_l1_slope": -slope,
            "mean_psnr": 20.0,
            "final_psnr": 20.0,
        },
        "route_audit": {"decision_records": decisions or []},
    }


def test_local_error_matches_training_return_definition() -> None:
    row = _measurement([1.0, 2.0, 4.0, 7.0])
    result = local_error(
        row,
        1,
        suffix_weight=1.0,
        terminal_weight=2.0,
        slope_weight=2.0,
    )
    assert result["suffix_mean_l1"] == pytest.approx(13.0 / 3.0)
    assert result["terminal_interaction_l1"] == 7.0
    assert result["suffix_horizon_l1_slope"] == pytest.approx(2.5)
    assert result["loss"] == pytest.approx(13.0 / 3.0 + 14.0 + 5.0)


def test_routing_report_scores_soft_brier_harm_and_counterfactual_value() -> None:
    zero = _measurement([10.0, 10.0, 10.0, 10.0])
    endpoints = {}
    decisions = []
    for interaction in range(4):
        negative = [10.0] * 4
        positive = [10.0] * 4
        negative[interaction:] = [9.0] * (4 - interaction)
        positive[interaction:] = [11.0] * (4 - interaction)
        endpoints[(interaction, -0.99)] = _measurement(negative)
        endpoints[(interaction, 0.99)] = _measurement(positive)
        decisions.append(
            {
                "decision_interaction": interaction,
                "selected_arm": 0,
                "soft_probabilities": [0.9, 0.05, 0.05],
            }
        )
    learned = _measurement([9.0, 9.0, 9.0, 9.0], decisions=decisions)
    report = routing_report(
        {101: {"zero": zero, "learned": learned, "endpoints": endpoints}},
        route_scope="interaction",
        suffix_weight=1.0,
        terminal_weight=2.0,
        slope_weight=2.0,
        minimum_benefit=0.01,
    )
    routing = report["routing"]
    assert routing["policy_brier"] == pytest.approx(0.015)
    assert routing["harmful_routing_rate"] == 0.0
    assert routing["counterfactual_policy_value"] > 0.0
    assert routing["active_coverage"] == 1.0


def test_routing_report_rejects_endpoint_that_changes_pre_target_prefix() -> None:
    zero = _measurement([10.0, 10.0, 10.0, 10.0])
    endpoints = {}
    decisions = []
    for interaction in range(4):
        negative = [10.0] * 4
        positive = [10.0] * 4
        if interaction == 2:
            negative[0] = 9.5
        endpoints[(interaction, -0.99)] = _measurement(negative)
        endpoints[(interaction, 0.99)] = _measurement(positive)
        decisions.append(
            {
                "decision_interaction": interaction,
                "selected_arm": 1,
                "soft_probabilities": [0.1, 0.8, 0.1],
            }
        )
    learned = _measurement([10.0, 10.0, 10.0, 10.0], decisions=decisions)
    with pytest.raises(CCLVRHeldoutAggregationError, match="CHANGED_PREFIX"):
        routing_report(
            {101: {"zero": zero, "learned": learned, "endpoints": endpoints}},
            route_scope="interaction",
            suffix_weight=1.0,
            terminal_weight=2.0,
            slope_weight=2.0,
            minimum_benefit=0.01,
        )
