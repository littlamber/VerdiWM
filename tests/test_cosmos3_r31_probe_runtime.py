from __future__ import annotations

from types import SimpleNamespace
import unittest

import pytest

torch = pytest.importorskip("torch")

from wmloop.primitives.adapters.cosmos3_hooks import (
    Cosmos3CPBEActionEmbeddingDeltaDose,
    apply_action_probe,
)
from scripts.integrations.run_cosmos3_inference_with_r31_probe import _eager_official_args


class Cosmos3R31RuntimeTests(unittest.TestCase):
    def test_embedding_hook_is_exactly_mean_preserving_and_reversible(self) -> None:
        torch.manual_seed(7)
        model = SimpleNamespace(action2llm=torch.nn.Linear(3, 8, bias=False))
        tokens = torch.randn(5, 3)
        baseline = model.action2llm(tokens)
        original = model.action2llm.forward
        with Cosmos3CPBEActionEmbeddingDeltaDose(
            model, dose=0.05, expected_token_count=5
        ) as context:
            observed = model.action2llm(tokens)
        self.assertEqual(model.action2llm.forward, original)
        self.assertFalse(torch.equal(observed, baseline))
        torch.testing.assert_close(
            observed.mean(dim=0), baseline.mean(dim=0), rtol=1e-6, atol=1e-6
        )
        self.assertLessEqual(context.audit["maximum_temporal_mean_abs_error"], 1e-6)
        self.assertLessEqual(
            context.audit["maximum_temporal_mean_abs_error"],
            context.audit["maximum_temporal_mean_tolerance"],
        )
        self.assertEqual(context.audit["invocation_count"], 1)

    def test_bfloat16_mean_gate_uses_dtype_resolution(self) -> None:
        model = SimpleNamespace(action2llm=torch.nn.Linear(3, 8, bias=False).bfloat16())
        tokens = torch.randn(16, 3, dtype=torch.bfloat16)
        with Cosmos3CPBEActionEmbeddingDeltaDose(
            model, dose=-0.05, expected_token_count=16
        ) as context:
            model.action2llm(tokens)
        self.assertLessEqual(
            context.audit["maximum_temporal_mean_abs_error"],
            context.audit["maximum_temporal_mean_tolerance"],
        )

    def test_zero_dose_is_identity_and_raw_action_probe_is_not_changed(self) -> None:
        model = SimpleNamespace(action2llm=torch.nn.Linear(3, 8, bias=False))
        tokens = torch.randn(5, 3)
        baseline = model.action2llm(tokens)
        with Cosmos3CPBEActionEmbeddingDeltaDose(
            model, dose=0.0, expected_token_count=5
        ):
            observed = model.action2llm(tokens)
        self.assertTrue(torch.equal(observed, baseline))
        actions = [[float(index), 1.0] for index in range(5)]
        self.assertEqual(
            apply_action_probe(actions, probe_id="cpbe_residual_63f088b0d5", dose=0.05),
            actions,
        )

    def test_rejects_multi_trajectory_token_shape(self) -> None:
        model = SimpleNamespace(action2llm=torch.nn.Linear(3, 8, bias=False))
        with self.assertRaisesRegex(ValueError, "TOKEN_COUNT_INVALID"):
            Cosmos3CPBEActionEmbeddingDeltaDose(model, dose=0.05, expected_token_count=1)

    def test_inference_shim_disables_fullgraph_compile(self) -> None:
        self.assertEqual(
            _eager_official_args(["--", "-i", "sample.jsonl"]),
            ["--no-use-torch-compile", "-i", "sample.jsonl"],
        )
        self.assertEqual(
            _eager_official_args(["--no-use-torch-compile", "-i", "sample.jsonl"]),
            ["--no-use-torch-compile", "-i", "sample.jsonl"],
        )
        with self.assertRaisesRegex(ValueError, "TORCH_COMPILE_UNSUPPORTED"):
            _eager_official_args(["--use-torch-compile", "-i", "sample.jsonl"])
