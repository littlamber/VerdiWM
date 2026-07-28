from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest

import pytest

torch = pytest.importorskip("torch")

from scripts.run_acwm_fingerprint_probe import (
    ActionEmbeddingTemporalMixDose,
    AutoregressiveHistoryDose,
    AutoregressiveHistoryTemporalMixDose,
    AutoregressiveMotionDose,
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
MOTION_REGION_CAMPAIGN = ROOT / "configs" / "experiments" / "acwm_phys_motion_region_probe_pilot_v1.json"
SELF_TEMPORAL_MIX_CAMPAIGN = (
    ROOT / "configs" / "experiments" / "acwm_phys_self_rollout_temporal_mix_probe_pilot_v1.json"
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


if __name__ == "__main__":
    unittest.main()
