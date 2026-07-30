from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.run_cosmos3_fingerprint_campaign import (
    _foreign_compute_pids,
    _pid_is_descendant,
    _preflight_runtime,
)
from wmloop.primitives.adapters.cosmos3_hooks import apply_action_probe, materialize_action_json


class Cosmos3FingerprintCampaignRunnerTests(unittest.TestCase):
    def test_dimension_interaction_is_mean_and_energy_preserving_reversible_rotation(self) -> None:
        actions = [
            [
                float(step),
                float(step) * 0.5,
                float(step) * -0.25,
                float(step % 3),
                float((step + 1) % 5),
                float(step) * 0.125,
                float(step % 4),
                float(step) * -0.75,
                0.25,
                -1.0,
            ]
            for step in range(16)
        ]
        rotated = apply_action_probe(
            actions,
            probe_id="action_dimension_interaction",
            dose=0.05,
        )
        recovered = apply_action_probe(
            rotated,
            probe_id="action_dimension_interaction",
            dose=-0.05,
        )
        self.assertNotEqual(rotated, actions)
        for index in range(10):
            self.assertAlmostEqual(
                sum(row[index] for row in rotated),
                sum(row[index] for row in actions),
                places=11,
            )
        self.assertAlmostEqual(_centered_energy(rotated), _centered_energy(actions), places=10)
        for source_row, recovered_row in zip(actions, recovered, strict=True):
            for source, value in zip(source_row, recovered_row, strict=True):
                self.assertAlmostEqual(source, value, places=11)
        for source_row, rotated_row in zip(actions, rotated, strict=True):
            self.assertEqual(rotated_row[8:], source_row[8:])

    def test_dimension_interaction_receipt_proves_runtime_invariants(self) -> None:
        actions = [
            [float(step + index) for index in range(9)] + [float(step % 2)]
            for step in range(16)
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.json"
            source.write_text(json.dumps(actions), encoding="utf-8")
            zero = materialize_action_json(
                source=source,
                destination=root / "zero.json",
                mode="action_dimension_interaction",
                dose=0.0,
            )
            positive = materialize_action_json(
                source=source,
                destination=root / "positive.json",
                mode="action_dimension_interaction",
                dose=0.05,
            )
            self.assertEqual((root / "zero.json").read_bytes(), source.read_bytes())
        self.assertTrue(zero["zero_dose_byte_identity"])
        self.assertEqual(positive["dose_unit"], "radians_action_dimension_coupling")
        self.assertEqual(positive["coupling_pairs"], [[0, 3], [1, 4], [2, 5], [6, 7]])
        self.assertLessEqual(positive["temporal_mean_max_abs_error"], 1e-12)
        self.assertLessEqual(positive["action_dimension_centered_energy_relative_error"], 1e-12)
        self.assertEqual(positive["unchanged_uncoupled_max_abs_error"], 0.0)

    def test_dimension_anisotropy_probe_preserves_means_and_balances_energy(self) -> None:
        actions = [
            [float(step), float(step) * 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
            for step in range(16)
        ]
        balanced = apply_action_probe(
            actions,
            probe_id="action_dimension_anisotropy",
            dose=0.1,
        )
        amplified = apply_action_probe(
            actions,
            probe_id="action_dimension_anisotropy",
            dose=-0.1,
        )
        for index in range(10):
            self.assertAlmostEqual(
                sum(row[index] for row in balanced),
                sum(row[index] for row in actions),
            )
        source_ratio = _contrast_rms(actions, 0) / _contrast_rms(actions, 1)
        balanced_ratio = _contrast_rms(balanced, 0) / _contrast_rms(balanced, 1)
        amplified_ratio = _contrast_rms(amplified, 0) / _contrast_rms(amplified, 1)
        self.assertLess(balanced_ratio, source_ratio)
        self.assertGreater(amplified_ratio, source_ratio)

    def test_dimension_anisotropy_receipt_proves_zero_identity_and_energy_change(self) -> None:
        actions = [
            [float(step), float(step) * 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
            for step in range(16)
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.json"
            source.write_text(json.dumps(actions), encoding="utf-8")
            zero = materialize_action_json(
                source=source,
                destination=root / "zero.json",
                mode="action_dimension_anisotropy",
                dose=0.0,
            )
            positive = materialize_action_json(
                source=source,
                destination=root / "positive.json",
                mode="action_dimension_anisotropy",
                dose=0.1,
            )
            self.assertEqual((root / "zero.json").read_bytes(), source.read_bytes())
        self.assertTrue(zero["zero_dose_byte_identity"])
        self.assertEqual(zero["dose_unit"], "relative_action_dimension_contrast_balance")
        self.assertEqual(len(positive["gains"]), 10)
        self.assertLessEqual(positive["temporal_mean_max_abs_error"], 1e-12)
        self.assertLess(
            positive["action_dimension_log_energy_spread_after"],
            positive["action_dimension_log_energy_spread_before"],
        )

    def test_temporal_mix_dispatch_preserves_per_dimension_mean(self) -> None:
        actions = [[1.0, 4.0], [3.0, 0.0], [2.0, 2.0]]
        mixed = apply_action_probe(
            actions,
            probe_id="action_embedding_temporal_mix",
            dose=0.1,
        )
        self.assertNotEqual(mixed, actions)
        for index in range(2):
            self.assertAlmostEqual(
                sum(row[index] for row in mixed),
                sum(row[index] for row in actions),
            )

    def test_translation_scale_dispatch_preserves_rotation_and_gripper(self) -> None:
        actions = [
            [1.0, -2.0, 3.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, -1.0],
            [2.0, -1.0, 4.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0],
        ]
        scaled = apply_action_probe(
            actions,
            probe_id="action_translation_scale",
            dose=0.05,
        )
        for source, transformed in zip(actions, scaled, strict=True):
            self.assertEqual(transformed[3:], source[3:])
            for index in range(3):
                self.assertAlmostEqual(transformed[index], source[index] * 1.05)

    def test_translation_scale_receipt_proves_nontranslation_identity(self) -> None:
        actions = [
            [1.0, -2.0, 3.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, -1.0],
            [2.0, -1.0, 4.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0],
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.json"
            source.write_text(json.dumps(actions), encoding="utf-8")
            receipt = materialize_action_json(
                source=source,
                destination=root / "transformed.json",
                mode="action_translation_scale",
                dose=-0.025,
            )
        self.assertEqual(receipt["dose_unit"], "relative_translation_action_scale")
        self.assertEqual(receipt["unchanged_nontranslation_max_abs_error"], 0.0)

    def test_gpu_exclusivity_accepts_only_descendant_compute_pids(self) -> None:
        parents = {31: 20, 20: 10, 77: 1}
        reader = parents.get
        rows = [("GPU-target", 31), ("GPU-other", 77)]
        self.assertEqual(_foreign_compute_pids("GPU-target", rows, 10, reader), [])
        self.assertTrue(_pid_is_descendant(31, 10, reader))

    def test_gpu_exclusivity_rejects_foreign_pid_on_target_uuid(self) -> None:
        parents = {31: 20, 20: 10, 77: 1}
        rows = [("GPU-target", 31), ("GPU-target", 77)]
        self.assertEqual(_foreign_compute_pids("GPU-target", rows, 10, parents.get), [77])

    def test_gpu_exclusivity_retains_a_previously_observed_child_during_exit(self) -> None:
        parents = {31: 1}
        rows = [("GPU-target", 31)]
        self.assertEqual(
            _foreign_compute_pids(
                "GPU-target",
                rows,
                10,
                parents.get,
                trusted_pids={31},
            ),
            [],
        )

    def test_runtime_preflight_accepts_available_modules(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            _preflight_runtime(
                Path(sys.executable),
                modules=("json", "pathlib"),
                code="PREFLIGHT_FAILED",
                cwd=Path(temporary),
                env=os.environ.copy(),
            )

    def test_runtime_preflight_reports_missing_dependency_before_campaign(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "PREFLIGHT_FAILED:1:.*ModuleNotFoundError"):
                _preflight_runtime(
                    Path(sys.executable),
                    modules=("verdiwm_module_that_does_not_exist",),
                    code="PREFLIGHT_FAILED",
                    cwd=Path(temporary),
                    env=os.environ.copy(),
                )


def _contrast_rms(actions: list[list[float]], index: int) -> float:
    mean = sum(row[index] for row in actions) / len(actions)
    return (sum((row[index] - mean) ** 2 for row in actions) / len(actions)) ** 0.5


def _centered_energy(actions: list[list[float]]) -> float:
    means = [sum(row[index] for row in actions) / len(actions) for index in range(len(actions[0]))]
    return sum(
        (row[index] - means[index]) ** 2
        for row in actions
        for index in range(len(row))
    )


if __name__ == "__main__":
    unittest.main()
