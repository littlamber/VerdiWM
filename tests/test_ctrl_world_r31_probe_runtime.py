from __future__ import annotations

from types import SimpleNamespace
import unittest

import pytest

torch = pytest.importorskip("torch")

from wmloop.experiments.ctrl_world_fingerprint import (
    CtrlWorldCPBEActionEmbeddingDeltaDose,
    CtrlWorldFingerprintError,
)


class _Encoder(torch.nn.Module):
    def forward(self, action: torch.Tensor, *_args: object, **_kwargs: object) -> torch.Tensor:
        return torch.cat((action, action.square()), dim=-1)


def _model() -> SimpleNamespace:
    return SimpleNamespace(action_encoder=_Encoder())


def _probe() -> dict[str, object]:
    return {
        "probe_id": "cpbe_residual_63f088b0d5",
        "hook_type": "H2",
        "signal_source": "action_embedding_delta",
        "spatial_mask": "all_action_embedding",
        "temporal_basis": "event_phase_tangent",
        "contrast_operator": "signed_mean_preserving_phase",
        "aggregation": "goal_outcome_vector",
        "diagnostic_only": True,
        "reversible": True,
        "doses": [-0.05, -0.025, 0.0, 0.025, 0.05],
    }


class CtrlWorldR31RuntimeTests(unittest.TestCase):
    def test_exact_program_changes_embedding_and_preserves_temporal_mean(self) -> None:
        model = _model()
        action = torch.tensor(
            [[[0.0, 0.2], [0.5, -0.1], [0.2, 0.8], [1.0, 0.4], [0.7, 1.1]]],
            dtype=torch.float64,
        )
        baseline = model.action_encoder(action)
        original = model.action_encoder.forward
        with CtrlWorldCPBEActionEmbeddingDeltaDose(
            model, 0.05, probe=_probe()
        ) as context:
            observed = model.action_encoder(action, "ignored")
            self.assertGreater(context.audit["invocation_count"], 0)
        self.assertEqual(model.action_encoder.forward, original)
        self.assertFalse(torch.equal(observed, baseline))
        torch.testing.assert_close(
            observed.mean(dim=1), baseline.mean(dim=1), rtol=1e-12, atol=1e-12
        )
        self.assertLessEqual(context.audit["maximum_temporal_mean_abs_error"], 1e-12)

    def test_zero_dose_is_bitwise_identity(self) -> None:
        model = _model()
        action = torch.randn(2, 5, 3)
        baseline = model.action_encoder(action)
        with CtrlWorldCPBEActionEmbeddingDeltaDose(model, 0.0, probe=_probe()):
            observed = model.action_encoder(action)
        self.assertTrue(torch.equal(observed, baseline))

    def test_rejects_nearby_but_different_program(self) -> None:
        probe = _probe()
        probe["signal_source"] = "raw_action_sequence"
        with self.assertRaisesRegex(CtrlWorldFingerprintError, "PROGRAM_MISMATCH"):
            CtrlWorldCPBEActionEmbeddingDeltaDose(_model(), 0.05, probe=probe)


if __name__ == "__main__":
    unittest.main()
