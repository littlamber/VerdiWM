from __future__ import annotations

from types import SimpleNamespace
import unittest

import pytest

torch = pytest.importorskip("torch")

from scripts.run_acwm_fingerprint_probe import _dose_context
from wmloop.diagnose.probes.cpbe_residual_027381736e import apply_contrast_dose
from wmloop.diagnose.probes.cpbe_residual_397450ddec import apply_multi_scale_event_phase
from wmloop.diagnose.probes.cpbe_residual_eeecb3d726 import apply_phase_curvature_dose


PROGRAMS = {
    "cpbe_residual_027381736e": (
        "event_phase_tangent",
        "signed_mean_preserving_scale",
        apply_contrast_dose,
    ),
    "cpbe_residual_397450ddec": (
        "multi_scale_event_phase",
        "signed_mean_preserving_phase",
        apply_multi_scale_event_phase,
    ),
    "cpbe_residual_eeecb3d726": (
        "event_phase_curvature",
        "signed_mean_preserving_phase",
        apply_phase_curvature_dose,
    ),
}


class _IdentityEmbedder(torch.nn.Module):
    def forward(self, action: torch.Tensor) -> torch.Tensor:
        return action


class _DownsampledEmbedder(torch.nn.Module):
    def forward(self, action: torch.Tensor) -> torch.Tensor:
        return action[:, ::2]


def _campaign(probe_id: str) -> dict[str, object]:
    temporal_basis, contrast_operator, _ = PROGRAMS[probe_id]
    return {
        "probe": {
            "probe_id": probe_id,
            "hook_type": "H2",
            "signal_source": "raw_action_sequence",
            "spatial_mask": "all_action_embedding",
            "temporal_basis": temporal_basis,
            "contrast_operator": contrast_operator,
            "aggregation": "goal_outcome_vector",
            "diagnostic_only": True,
            "reversible": True,
            "doses": [-0.05, -0.025, 0.0, 0.025, 0.05],
        }
    }


def _model() -> SimpleNamespace:
    return SimpleNamespace(model=SimpleNamespace(action_embedder=_IdentityEmbedder()))


class ACWMCPBERuntimeAdapterTests(unittest.TestCase):
    def test_runtime_adapter_matches_frozen_offline_transform(self) -> None:
        action = torch.tensor(
            [[[0.0, 0.2], [0.5, -0.1], [0.2, 0.8], [1.0, 0.4], [0.7, 1.1]]],
            dtype=torch.float64,
        )
        for probe_id in sorted(PROGRAMS):
            with self.subTest(probe_id=probe_id):
                model = _model()
                _, _, offline_transform = PROGRAMS[probe_id]
                expected = torch.tensor(
                    offline_transform(
                        action_sequence=action[0].tolist(),
                        action_embeddings=action[0].tolist(),
                        dose=0.05,
                    ),
                    dtype=action.dtype,
                ).unsqueeze(0)

                with _dose_context(model, _campaign(probe_id), 0.05):
                    observed = model.model.action_embedder(action)

                torch.testing.assert_close(observed, expected, rtol=1e-12, atol=1e-12)
                torch.testing.assert_close(
                    observed.mean(dim=1), action.mean(dim=1), rtol=1e-12, atol=1e-12
                )

    def test_zero_dose_is_exact_identity_and_hook_is_restored(self) -> None:
        action = torch.randn(2, 5, 3)
        for probe_id in sorted(PROGRAMS):
            with self.subTest(probe_id=probe_id):
                model = _model()
                original = model.model.action_embedder.forward
                with _dose_context(model, _campaign(probe_id), 0.0):
                    observed = model.model.action_embedder(action)
                    self.assertTrue(torch.equal(observed, action))
                self.assertEqual(model.model.action_embedder.forward, original)

    def test_runtime_adapter_rejects_dsl_intent_drift(self) -> None:
        model = _model()
        campaign = _campaign("cpbe_residual_397450ddec")
        campaign["probe"]["temporal_basis"] = "event_phase_tangent"  # type: ignore[index]
        with self.assertRaisesRegex(RuntimeError, "ACWM_CPBE_PROGRAM_MISMATCH"):
            _dose_context(model, campaign, 0.05)

    def test_runtime_adapter_lowers_action_basis_to_embedding_token_grid(self) -> None:
        action = torch.tensor(
            [[[0.0, 0.2], [0.5, -0.1], [0.2, 0.8], [1.0, 0.4], [0.7, 1.1]]],
            dtype=torch.float64,
        )
        for probe_id in sorted(PROGRAMS):
            with self.subTest(probe_id=probe_id):
                model = SimpleNamespace(
                    model=SimpleNamespace(action_embedder=_DownsampledEmbedder())
                )
                baseline = model.model.action_embedder(action)
                with _dose_context(model, _campaign(probe_id), 0.05):
                    observed = model.model.action_embedder(action)
                self.assertEqual(observed.shape, baseline.shape)
                torch.testing.assert_close(
                    observed.mean(dim=1), baseline.mean(dim=1), rtol=1e-12, atol=1e-12
                )
                if probe_id == "cpbe_residual_027381736e":
                    expected = torch.tensor(
                        apply_contrast_dose(
                            action_sequence=action[0].tolist(),
                            action_embeddings=baseline[0].tolist(),
                            dose=0.05,
                        ),
                        dtype=action.dtype,
                    ).unsqueeze(0)
                    torch.testing.assert_close(observed, expected, rtol=1e-12, atol=1e-12)

    def test_runtime_adapter_rejects_unfrozen_dose(self) -> None:
        model = _model()
        with self.assertRaisesRegex(RuntimeError, "ACWM_CPBE_DOSE_GRID_MISMATCH"):
            _dose_context(model, _campaign("cpbe_residual_eeecb3d726"), 0.1)

    def test_runtime_adapter_lowers_new_single_axis_aggregation_program(self) -> None:
        campaign = _campaign("cpbe_residual_027381736e")
        campaign["probe"].update(  # type: ignore[index]
            probe_id="cpbe_residual_new_source_margin",
            aggregation="source_sign_margin",
        )
        model = _model()
        action = torch.randn(2, 5, 3)
        with _dose_context(model, campaign, 0.05):
            observed = model.model.action_embedder(action)
        self.assertEqual(observed.shape, action.shape)
        torch.testing.assert_close(observed.mean(dim=1), action.mean(dim=1))


if __name__ == "__main__":
    unittest.main()
