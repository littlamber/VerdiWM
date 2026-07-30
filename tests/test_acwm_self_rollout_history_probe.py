from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest

import pytest

torch = pytest.importorskip("torch")

from scripts.run_acwm_fingerprint_probe import (
    ActionDimensionAnisotropyDose,
    ActionDimensionInteractionDose,
    ActionEmbeddingEventAlignmentDose,
    ActionEmbeddingTemporalMixDose,
    AutoregressiveHistoryDose,
    AutoregressiveHistoryTemporalMixDose,
    AutoregressiveLatestFeedbackDose,
    AutoregressiveTeacherHorizonRecoveryDose,
    AutoregressiveTeacherRecoveryDose,
    AutoregressiveMotionDose,
    AutoregressiveMotionEventAlignmentDose,
    AutoregressiveMotionEventPhaseCurvatureDose,
    AutoregressiveMotionEventPhaseLagDose,
    AutoregressiveMotionRegionDose,
    _dose_context,
)
from wmloop.experiments.acwm_fingerprint import compile_probe_receipt, load_campaign


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "configs" / "experiments" / "acwm_phys_self_rollout_history_probe_pilot_v1.json"
MOTION_CAMPAIGN = ROOT / "configs" / "experiments" / "acwm_phys_motion_history_probe_pilot_v1.json"
ACTION_TEMPORAL_MIX_CAMPAIGN = (
    ROOT / "configs" / "experiments" / "acwm_phys_action_temporal_mix_probe_pilot_v1.json"
)
ACTION_DIMENSION_ANISOTROPY_CAMPAIGN = (
    ROOT / "configs" / "experiments" / "acwm_phys_action_dimension_anisotropy_probe_pilot_v1.json"
)
ACTION_DIMENSION_INTERACTION_CAMPAIGN = (
    ROOT / "configs" / "experiments" / "acwm_phys_action_dimension_interaction_probe_pilot_v1.json"
)
ACTION_EVENT_ALIGNMENT_CAMPAIGN = (
    ROOT / "configs" / "experiments" / "acwm_phys_action_temporal_alignment_probe_pilot_v1.json"
)
MOTION_REGION_CAMPAIGN = ROOT / "configs" / "experiments" / "acwm_phys_motion_region_probe_pilot_v1.json"
MOTION_EVENT_CAMPAIGN = (
    ROOT / "configs" / "experiments" / "acwm_phys_motion_region_event_alignment_probe_pilot_v1.json"
)
MOTION_EVENT_PHASE_CAMPAIGN = (
    ROOT / "configs" / "experiments" / "acwm_phys_motion_region_event_phase_lag_probe_pilot_v1.json"
)
MOTION_EVENT_CURVATURE_CAMPAIGN = (
    ROOT / "configs" / "experiments" / "acwm_phys_motion_region_event_phase_curvature_probe_pilot_v1.json"
)
SELF_TEMPORAL_MIX_CAMPAIGN = (
    ROOT / "configs" / "experiments" / "acwm_phys_self_rollout_temporal_mix_probe_pilot_v1.json"
)
SELF_LATEST_FEEDBACK_CAMPAIGN = (
    ROOT / "configs" / "experiments" / "acwm_phys_self_rollout_latest_feedback_probe_pilot_v1.json"
)
SELF_TEACHER_RECOVERY_CAMPAIGN = (
    ROOT / "configs" / "experiments" / "acwm_phys_self_rollout_teacher_recovery_probe_pilot_v1.json"
)
SELF_HORIZON_RECOVERY_CAMPAIGN = (
    ROOT / "configs" / "experiments" / "acwm_phys_self_rollout_horizon_recovery_probe_pilot_v1.json"
)


class _IdentityDiT(torch.nn.Module):
    def forward(self, z, t, action):
        return z


class _IdentityActionEmbedder(torch.nn.Module):
    def forward(self, action):
        return action


class AcwmSelfRolloutHistoryProbeTests(unittest.TestCase):
    def test_campaign_compiles_a_typed_inference_only_receipt(self) -> None:
        campaign = load_campaign(CAMPAIGN)
        receipt = compile_probe_receipt(campaign, dose=0.05)

        self.assertEqual(campaign["probe"]["probe_id"], "self_rollout_history_scale")
        self.assertEqual(campaign["probe"]["generation_mode"], "autoregressive")
        self.assertTrue(receipt["compiled"])
        self.assertEqual(receipt["hook_type"], "H2")
        self.assertEqual(receipt["dose_unit"], "relative_generated_history_deviation_scale")

    def test_history_dose_changes_only_prior_generated_latents(self) -> None:
        dit = _IdentityDiT()
        dynamics = SimpleNamespace(model=dit)
        z = torch.tensor([[[1.0], [2.0], [5.0]]])
        t = torch.zeros(1, 3)
        action = torch.zeros(1, 3, 1)

        with AutoregressiveHistoryDose(dynamics, 0.5):
            observed = dit(z, t, action)

        self.assertTrue(torch.equal(observed[:, :1], z[:, :1]))
        self.assertTrue(torch.equal(observed[:, -1:], z[:, -1:]))
        self.assertTrue(torch.equal(observed[:, 1:2], torch.tensor([[[2.5]]])))
        self.assertTrue(torch.equal(dit(z, t, action), z))

    def test_runtime_factory_requires_autoregressive_mode(self) -> None:
        campaign = load_campaign(CAMPAIGN)
        dynamics = SimpleNamespace(model=_IdentityDiT())

        context = _dose_context(dynamics, campaign, 0.0)
        self.assertIsInstance(context, AutoregressiveHistoryDose)

        broken = {**campaign, "probe": {**campaign["probe"], "generation_mode": "parallel"}}
        with self.assertRaisesRegex(RuntimeError, "AUTOREGRESSIVE_MODE"):
            _dose_context(dynamics, broken, 0.0)

    def test_motion_dose_scales_history_increments_only(self) -> None:
        dit = _IdentityDiT()
        dynamics = SimpleNamespace(model=dit)
        z = torch.tensor([[[1.0], [2.0], [4.0], [9.0]]])
        t = torch.zeros(1, 4)
        action = torch.zeros(1, 4, 1)

        with AutoregressiveMotionDose(dynamics, 0.5):
            observed = dit(z, t, action)

        self.assertTrue(torch.equal(observed[:, :1], z[:, :1]))
        self.assertTrue(torch.equal(observed[:, -1:], z[:, -1:]))
        self.assertTrue(torch.equal(observed[:, 1:3], torch.tensor([[[2.5], [5.5]]])))
        campaign = load_campaign(MOTION_CAMPAIGN)
        self.assertIsInstance(_dose_context(dynamics, campaign, 0.0), AutoregressiveMotionDose)

    def test_action_temporal_mix_preserves_mean_and_is_reversible(self) -> None:
        embedder = _IdentityActionEmbedder()
        dynamics = SimpleNamespace(model=SimpleNamespace(action_embedder=embedder))
        action = torch.tensor([[[1.0], [3.0], [5.0]]])

        with ActionEmbeddingTemporalMixDose(dynamics, 0.5):
            observed = embedder(action)

        self.assertTrue(torch.equal(observed, torch.tensor([[[2.0], [3.0], [4.0]]])))
        self.assertTrue(torch.equal(observed.mean(dim=1), action.mean(dim=1)))
        self.assertTrue(torch.equal(embedder(action), action))
        campaign = load_campaign(ACTION_TEMPORAL_MIX_CAMPAIGN)
        self.assertIsInstance(_dose_context(dynamics, campaign, 0.0), ActionEmbeddingTemporalMixDose)

    def test_action_dimension_anisotropy_preserves_per_dimension_temporal_mean(self) -> None:
        embedder = _IdentityActionEmbedder()
        dynamics = SimpleNamespace(model=SimpleNamespace(action_embedder=embedder))
        action = torch.tensor([[[0.0, 2.0], [1.0, 2.0], [2.0, 2.0]]])

        with ActionDimensionAnisotropyDose(dynamics, 0.5):
            observed = embedder(action)

        self.assertTrue(
            torch.allclose(observed, torch.tensor([[[-0.25, 2.0], [1.0, 2.0], [2.25, 2.0]]]))
        )
        self.assertTrue(torch.allclose(observed.mean(dim=1), action.mean(dim=1)))
        self.assertTrue(torch.equal(embedder(action), action))
        campaign = load_campaign(ACTION_DIMENSION_ANISOTROPY_CAMPAIGN)
        self.assertIsInstance(_dose_context(dynamics, campaign, 0.0), ActionDimensionAnisotropyDose)

    def test_action_dimension_interaction_preserves_mean_energy_and_unpaired_dimension(self) -> None:
        embedder = _IdentityActionEmbedder()
        dynamics = SimpleNamespace(model=SimpleNamespace(action_embedder=embedder))
        action = torch.tensor(
            [[[0.0, 2.0, 7.0], [1.0, 4.0, 8.0], [2.0, 6.0, 9.0]]]
        )

        with ActionDimensionInteractionDose(dynamics, 0.05):
            observed = embedder(action)

        centered_before = action - action.mean(dim=1, keepdim=True)
        centered_after = observed - observed.mean(dim=1, keepdim=True)
        self.assertFalse(torch.equal(observed[..., :2], action[..., :2]))
        self.assertTrue(torch.allclose(observed.mean(dim=1), action.mean(dim=1)))
        self.assertTrue(
            torch.allclose(
                centered_after[..., :2].square().sum(),
                centered_before[..., :2].square().sum(),
            )
        )
        self.assertTrue(torch.equal(observed[..., 2], action[..., 2]))
        self.assertTrue(torch.equal(embedder(action), action))
        campaign = load_campaign(ACTION_DIMENSION_INTERACTION_CAMPAIGN)
        self.assertIsInstance(
            _dose_context(dynamics, campaign, 0.0), ActionDimensionInteractionDose
        )

    def test_action_dimension_interaction_zero_dose_is_exact_identity(self) -> None:
        embedder = _IdentityActionEmbedder()
        dynamics = SimpleNamespace(model=SimpleNamespace(action_embedder=embedder))
        action = torch.randn(2, 5, 7)

        with ActionDimensionInteractionDose(dynamics, 0.0):
            observed = embedder(action)

        self.assertTrue(torch.equal(observed, action))

    def test_action_event_alignment_preserves_embedding_mean(self) -> None:
        embedder = _IdentityActionEmbedder()
        dynamics = SimpleNamespace(model=SimpleNamespace(action_embedder=embedder))
        action = torch.tensor([[[0.0], [1.0], [1.0], [3.0]]])

        with ActionEmbeddingEventAlignmentDose(dynamics, 0.5):
            observed = embedder(action)

        self.assertFalse(torch.equal(observed, action))
        self.assertTrue(torch.allclose(observed.mean(dim=1), action.mean(dim=1)))
        self.assertTrue(torch.equal(embedder(action), action))
        campaign = load_campaign(ACTION_EVENT_ALIGNMENT_CAMPAIGN)
        self.assertIsInstance(_dose_context(dynamics, campaign, 0.0), ActionEmbeddingEventAlignmentDose)

    def test_action_event_alignment_is_identity_without_action_transition(self) -> None:
        embedder = _IdentityActionEmbedder()
        dynamics = SimpleNamespace(model=SimpleNamespace(action_embedder=embedder))
        action = torch.ones(1, 4, 2)

        with ActionEmbeddingEventAlignmentDose(dynamics, 0.5):
            observed = embedder(action)

        self.assertTrue(torch.equal(observed, action))

    def test_motion_region_dose_changes_dynamic_pixels_only(self) -> None:
        dit = _IdentityDiT()
        dynamics = SimpleNamespace(model=dit)
        z = torch.tensor(
            [[[[[0.0], [0.0]]], [[[1.0], [0.0]]], [[[2.0], [0.0]]], [[[9.0], [9.0]]]]]
        )
        t = torch.zeros(1, 4)
        action = torch.zeros(1, 4, 1)

        with AutoregressiveMotionRegionDose(dynamics, 0.5):
            observed = dit(z, t, action)

        self.assertTrue(torch.equal(observed[:, :1], z[:, :1]))
        self.assertTrue(torch.equal(observed[:, -1:], z[:, -1:]))
        self.assertTrue(torch.equal(observed[0, 1:3, 0, 0, 0], torch.tensor([1.5, 3.0])))
        self.assertTrue(torch.equal(observed[0, 1:3, 0, 1, 0], torch.tensor([0.0, 0.0])))
        campaign = load_campaign(MOTION_REGION_CAMPAIGN)
        self.assertIsInstance(_dose_context(dynamics, campaign, 0.0), AutoregressiveMotionRegionDose)

    def test_motion_event_alignment_requires_spatial_motion_and_action_transition(self) -> None:
        dit = _IdentityDiT()
        dynamics = SimpleNamespace(model=dit)
        z = torch.tensor(
            [[[[[0.0], [0.0]]], [[[1.0], [0.0]]], [[[2.0], [0.0]]], [[[9.0], [9.0]]]]]
        )
        t = torch.zeros(1, 4)
        action = torch.tensor([[[0.0], [1.0], [1.0], [1.0]]])

        with AutoregressiveMotionEventAlignmentDose(dynamics, 0.5):
            observed = dit(z, t, action)

        self.assertTrue(torch.equal(observed[:, :1], z[:, :1]))
        self.assertTrue(torch.equal(observed[:, -1:], z[:, -1:]))
        self.assertTrue(torch.equal(observed[0, 1:3, 0, 0, 0], torch.tensor([1.5, 2.5])))
        self.assertTrue(torch.equal(observed[0, 1:3, 0, 1, 0], torch.tensor([0.0, 0.0])))
        self.assertTrue(torch.equal(dit(z, t, action), z))
        campaign = load_campaign(MOTION_EVENT_CAMPAIGN)
        self.assertIsInstance(
            _dose_context(dynamics, campaign, 0.0), AutoregressiveMotionEventAlignmentDose
        )

    def test_motion_event_alignment_is_identity_without_action_transition(self) -> None:
        dit = _IdentityDiT()
        dynamics = SimpleNamespace(model=dit)
        z = torch.tensor([[[1.0], [2.0], [4.0], [9.0]]])
        t = torch.zeros(1, 4)
        action = torch.ones(1, 4, 2)

        with AutoregressiveMotionEventAlignmentDose(dynamics, 0.5):
            observed = dit(z, t, action)

        self.assertTrue(torch.equal(observed, z))

    def test_motion_event_phase_curvature_is_event_local_and_reversible(self) -> None:
        dit = _IdentityDiT()
        dynamics = SimpleNamespace(model=dit)
        z = torch.tensor([[[1.0], [2.0], [4.0], [7.0], [9.0]]])
        t = torch.zeros(1, 5)
        action = torch.tensor([[[0.0], [0.0], [2.0], [2.0], [2.0]]])

        with AutoregressiveMotionEventPhaseCurvatureDose(dynamics, 0.05):
            observed = dit(z, t, action)

        self.assertTrue(torch.equal(observed[:, :1], z[:, :1]))
        self.assertTrue(torch.equal(observed[:, -1:], z[:, -1:]))
        self.assertFalse(torch.equal(observed[:, 1:-1], z[:, 1:-1]))
        self.assertTrue(torch.equal(dit(z, t, action), z))
        campaign = load_campaign(MOTION_EVENT_CURVATURE_CAMPAIGN)
        self.assertIsInstance(
            _dose_context(dynamics, campaign, 0.0),
            AutoregressiveMotionEventPhaseCurvatureDose,
        )

    def test_motion_event_phase_curvature_is_identity_without_action_transition(self) -> None:
        dit = _IdentityDiT()
        dynamics = SimpleNamespace(model=dit)
        z = torch.tensor([[[1.0], [2.0], [4.0], [7.0], [9.0]]])
        t = torch.zeros(1, 5)
        action = torch.ones(1, 5, 2)

        with AutoregressiveMotionEventPhaseCurvatureDose(dynamics, 0.05):
            observed = dit(z, t, action)

        self.assertTrue(torch.equal(observed, z))

    def test_motion_event_phase_lag_changes_following_motion_only(self) -> None:
        dit = _IdentityDiT()
        dynamics = SimpleNamespace(model=dit)
        z = torch.tensor(
            [[[[[0.0], [0.0]]], [[[1.0], [0.0]]], [[[2.0], [0.0]]], [[[9.0], [9.0]]]]]
        )
        t = torch.zeros(1, 4)
        action = torch.tensor([[[0.0], [1.0], [1.0], [1.0]]])

        with AutoregressiveMotionEventPhaseLagDose(dynamics, 0.5):
            observed = dit(z, t, action)

        self.assertTrue(torch.equal(observed[:, :1], z[:, :1]))
        self.assertTrue(torch.equal(observed[:, -1:], z[:, -1:]))
        self.assertTrue(torch.equal(observed[0, 1:3, 0, 0, 0], torch.tensor([1.0, 2.5])))
        self.assertTrue(torch.equal(observed[0, 1:3, 0, 1, 0], torch.tensor([0.0, 0.0])))
        self.assertTrue(torch.equal(dit(z, t, action), z))
        campaign = load_campaign(MOTION_EVENT_PHASE_CAMPAIGN)
        self.assertIsInstance(
            _dose_context(dynamics, campaign, 0.0), AutoregressiveMotionEventPhaseLagDose
        )

    def test_self_rollout_temporal_mix_changes_each_history_state_locally(self) -> None:
        dit = _IdentityDiT()
        dynamics = SimpleNamespace(model=dit)
        z = torch.tensor([[[1.0], [3.0], [7.0], [9.0]]])
        t = torch.zeros(1, 4)
        action = torch.zeros(1, 4, 1)

        with AutoregressiveHistoryTemporalMixDose(dynamics, 0.5):
            observed = dit(z, t, action)

        self.assertTrue(torch.equal(observed[:, :1], z[:, :1]))
        self.assertTrue(torch.equal(observed[:, -1:], z[:, -1:]))
        self.assertTrue(torch.equal(observed[:, 1:3], torch.tensor([[[2.0], [5.0]]])))
        campaign = load_campaign(SELF_TEMPORAL_MIX_CAMPAIGN)
        self.assertIsInstance(_dose_context(dynamics, campaign, 0.0), AutoregressiveHistoryTemporalMixDose)

    def test_self_rollout_latest_feedback_changes_only_latest_history_state(self) -> None:
        dit = _IdentityDiT()
        dynamics = SimpleNamespace(model=dit)
        z = torch.tensor([[[1.0], [3.0], [7.0], [9.0]]])
        t = torch.zeros(1, 4)
        action = torch.zeros(1, 4, 1)

        with AutoregressiveLatestFeedbackDose(dynamics, 0.5):
            observed = dit(z, t, action)

        self.assertTrue(torch.equal(observed[:, :2], z[:, :2]))
        self.assertTrue(torch.equal(observed[:, 2:3], torch.tensor([[[5.0]]])))
        self.assertTrue(torch.equal(observed[:, -1:], z[:, -1:]))
        self.assertTrue(torch.equal(dit(z, t, action), z))
        campaign = load_campaign(SELF_LATEST_FEEDBACK_CAMPAIGN)
        self.assertIsInstance(_dose_context(dynamics, campaign, 0.0), AutoregressiveLatestFeedbackDose)

    def test_self_rollout_teacher_recovery_preserves_anchor_and_noisy_state(self) -> None:
        dit = _IdentityDiT()
        dynamics = SimpleNamespace(model=dit)
        z = torch.tensor([[[1.0], [3.0], [7.0], [9.0]]])
        teacher = torch.tensor([[[1.0], [2.0], [5.0], [8.0]]])
        t = torch.zeros(1, 4)
        action = torch.zeros(1, 4, 1)

        with AutoregressiveTeacherRecoveryDose(dynamics, 0.5, teacher):
            observed = dit(z, t, action)

        self.assertTrue(torch.equal(observed[:, :1], z[:, :1]))
        self.assertTrue(torch.equal(observed[:, -1:], z[:, -1:]))
        self.assertTrue(torch.equal(observed[:, 1:3], torch.tensor([[[2.5], [6.0]]])))
        self.assertTrue(torch.equal(dit(z, t, action), z))
        campaign = load_campaign(SELF_TEACHER_RECOVERY_CAMPAIGN)
        context = _dose_context(dynamics, campaign, 0.0, teacher_history=teacher)
        self.assertIsInstance(context, AutoregressiveTeacherRecoveryDose)

    def test_self_rollout_teacher_recovery_fails_without_reference(self) -> None:
        dynamics = SimpleNamespace(model=_IdentityDiT())
        campaign = load_campaign(SELF_TEACHER_RECOVERY_CAMPAIGN)

        with self.assertRaisesRegex(RuntimeError, "REFERENCE_MISSING"):
            _dose_context(dynamics, campaign, 0.0)

    def test_self_rollout_horizon_recovery_increases_with_history_age(self) -> None:
        dit = _IdentityDiT()
        dynamics = SimpleNamespace(model=dit)
        z = torch.tensor([[[1.0], [3.0], [7.0], [9.0]]])
        teacher = torch.tensor([[[1.0], [2.0], [5.0], [8.0]]])
        t = torch.zeros(1, 4)
        action = torch.zeros(1, 4, 1)

        with AutoregressiveTeacherHorizonRecoveryDose(dynamics, 0.5, teacher):
            observed = dit(z, t, action)

        self.assertTrue(torch.equal(observed[:, :1], z[:, :1]))
        self.assertTrue(torch.equal(observed[:, -1:], z[:, -1:]))
        self.assertTrue(torch.equal(observed[:, 1:3], torch.tensor([[[2.75], [6.0]]])))
        campaign = load_campaign(SELF_HORIZON_RECOVERY_CAMPAIGN)
        context = _dose_context(dynamics, campaign, 0.0, teacher_history=teacher)
        self.assertIsInstance(context, AutoregressiveTeacherHorizonRecoveryDose)


if __name__ == "__main__":
    unittest.main()
