from __future__ import annotations

import json
import unittest
from pathlib import Path

from wmloop.contracts import validate_document
from wmloop.diagnose.probes.cpbe_residual_eeecb3d726 import (
    AGGREGATION,
    CONTRAST_OPERATOR,
    DOSE_SCHEDULE,
    TEMPORAL_BASIS,
    CpbeResidualPhaseCurvatureProbeError,
    apply_phase_curvature_dose,
    measure_phase_curvature_fixture,
)


ROOT = Path(__file__).resolve().parents[1]
DESCRIPTOR = ROOT / "configs" / "probes" / "staging" / "cpbe_residual_eeecb3d726.json"


class CpbeResidualPhaseCurvatureProbeTests(unittest.TestCase):
    def test_fixture_output_is_schema_valid_and_not_verdict_exposed(self) -> None:
        output = measure_phase_curvature_fixture(
            action_sequence=_actions(),
            action_embeddings=_embeddings(),
            goal_outcome_vectors=_goal_outcomes(),
            evidence_refs=["cas://sha256/" + "a" * 64],
        )

        validate_document("diagnostic_probe_output", output)
        self.assertEqual(output["probe_id"], "cpbe_residual_eeecb3d726")
        self.assertEqual(output["role"], "diagnostic")
        self.assertFalse(output["verdict_exposure_allowed"])
        self.assertFalse(_contains_key(output, "verdict_evidence"))
        self.assertEqual(output["metrics"]["aggregation"], "goal_outcome_vector")
        self.assertEqual(output["metrics"]["goal_outcome_vectors"], _goal_outcomes())

    def test_zero_dose_is_exact_identity_and_all_doses_preserve_temporal_mean(self) -> None:
        baseline = _embeddings()
        zero = apply_phase_curvature_dose(
            action_sequence=_actions(), action_embeddings=baseline, dose=0.0
        )
        self.assertEqual(zero, baseline)
        self.assertIsNot(zero, baseline)

        baseline_mean = _temporal_mean(baseline)
        for dose in DOSE_SCHEDULE:
            observed = apply_phase_curvature_dose(
                action_sequence=_actions(), action_embeddings=baseline, dose=dose
            )
            self.assertEqual(len(observed), len(baseline))
            self.assertEqual(len(observed[0]), len(baseline[0]))
            self.assertSequenceAlmostEqual(_temporal_mean(observed), baseline_mean)

    def test_signed_doses_are_antisymmetric_around_baseline(self) -> None:
        baseline = _embeddings()
        positive = apply_phase_curvature_dose(
            action_sequence=_actions(), action_embeddings=baseline, dose=0.05
        )
        negative = apply_phase_curvature_dose(
            action_sequence=_actions(), action_embeddings=baseline, dose=-0.05
        )

        self.assertNotEqual(positive, baseline)
        for negative_row, baseline_row, positive_row in zip(negative, baseline, positive):
            for left, center, right in zip(negative_row, baseline_row, positive_row):
                self.assertAlmostEqual(left - center, -(right - center), places=12)

    def test_no_action_transition_is_identity_and_flagged(self) -> None:
        actions = [[1.0, -1.0]] * len(_embeddings())
        output = measure_phase_curvature_fixture(
            action_sequence=actions,
            action_embeddings=_embeddings(),
            goal_outcome_vectors=_goal_outcomes(),
        )

        self.assertIn("no_action_transition", output["flags"])
        self.assertEqual(output["metrics"]["active_curvature_step_count"], 0)
        self.assertEqual(output["metrics"]["mean_absolute_embedding_delta_by_dose"], [0.0] * 5)

    def test_invalid_boundaries_fail_closed_with_stable_codes(self) -> None:
        with self.assertRaisesRegex(CpbeResidualPhaseCurvatureProbeError, "TIMESTEPS_INSUFFICIENT"):
            apply_phase_curvature_dose(
                action_sequence=_actions()[:2], action_embeddings=_embeddings()[:2], dose=0.05
            )
        with self.assertRaisesRegex(CpbeResidualPhaseCurvatureProbeError, "TEMPORAL_LENGTH_MISMATCH"):
            apply_phase_curvature_dose(
                action_sequence=_actions(), action_embeddings=_embeddings()[:-1], dose=0.05
            )
        with self.assertRaisesRegex(CpbeResidualPhaseCurvatureProbeError, "DOSE_GRID_MISMATCH"):
            apply_phase_curvature_dose(
                action_sequence=_actions(), action_embeddings=_embeddings(), dose=0.01
            )
        with self.assertRaisesRegex(CpbeResidualPhaseCurvatureProbeError, "VALUE_INVALID"):
            apply_phase_curvature_dose(
                action_sequence=_actions(),
                action_embeddings=[[float("nan"), 0.0], *_embeddings()[1:]],
                dose=0.05,
            )
        with self.assertRaisesRegex(CpbeResidualPhaseCurvatureProbeError, "DOSE_GRID_MISMATCH"):
            measure_phase_curvature_fixture(
                action_sequence=_actions(),
                action_embeddings=_embeddings(),
                goal_outcome_vectors=_goal_outcomes()[:-1],
            )
        with self.assertRaisesRegex(CpbeResidualPhaseCurvatureProbeError, "EVIDENCE_REFS_INVALID"):
            measure_phase_curvature_fixture(
                action_sequence=_actions(),
                action_embeddings=_embeddings(),
                goal_outcome_vectors=_goal_outcomes(),
                evidence_refs=["results/local.json"],
            )

    def test_staged_descriptor_matches_implementation_contract(self) -> None:
        descriptor = json.loads(DESCRIPTOR.read_text(encoding="utf-8"))

        self.assertEqual(descriptor["probe_id"], "cpbe_residual_eeecb3d726")
        self.assertEqual(descriptor["role"], "diagnostic")
        self.assertFalse(descriptor["verdict_exposure_allowed"])
        self.assertEqual(descriptor["module"], "wmloop.diagnose.probes.cpbe_residual_eeecb3d726")
        self.assertEqual(descriptor["callable"], "measure_phase_curvature_fixture")
        self.assertEqual(descriptor["program"], _expected_program())
        self.assertEqual(descriptor["program"]["dose_schedule"], list(DOSE_SCHEDULE))
        self.assertEqual(descriptor["program"]["temporal_basis"], TEMPORAL_BASIS)
        self.assertEqual(descriptor["program"]["contrast_operator"], CONTRAST_OPERATOR)
        self.assertEqual(descriptor["program"]["aggregation"], AGGREGATION)
        self.assertNotIn("verdict_evidence", descriptor["output_contract"])

    def assertSequenceAlmostEqual(self, first: list[float], second: list[float]) -> None:
        self.assertEqual(len(first), len(second))
        for left, right in zip(first, second):
            self.assertAlmostEqual(left, right, places=12)


def _actions() -> list[list[float]]:
    return [
        [0.0, 0.0],
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 2.0],
    ]


def _embeddings() -> list[list[float]]:
    return [
        [0.0, 1.0],
        [1.0, 2.0],
        [3.0, 1.0],
        [4.0, -1.0],
        [2.0, 2.0],
    ]


def _goal_outcomes() -> list[list[float]]:
    return [
        [0.10, -0.05],
        [0.08, -0.02],
        [0.00, 0.00],
        [-0.04, 0.03],
        [-0.09, 0.07],
    ]


def _expected_program() -> dict[str, object]:
    return {
        "probe_id": "cpbe_residual_eeecb3d726",
        "signal_source": "raw_action_sequence",
        "hook_type": "H2",
        "spatial_mask": "all_action_embedding",
        "temporal_basis": "event_phase_curvature",
        "contrast_operator": "signed_mean_preserving_phase",
        "dose_schedule": [-0.05, -0.025, 0.0, 0.025, 0.05],
        "aggregation": "goal_outcome_vector",
        "invariants": [
            "same_checkpoint",
            "same_input_action",
            "same_trajectory",
            "same_seed",
            "same_evaluator",
            "per_trajectory_action_embedding_temporal_mean_preserved",
            "dose_grid_matches_action_temporal_alignment_reference",
        ],
        "required_capabilities": [
            "action_embedding_hook",
            "action_sequence_hook",
            "paired_seed_control",
        ],
        "estimated_gpu_hours": 0.011611821701388888,
        "origin": "residual",
        "parent_probe_ids": ["action_temporal_alignment_phase"],
        "diagnostic_only": True,
        "reversible": True,
        "rationale": (
            "Change temporal_basis from event_phase_tangent to event_phase_curvature. "
            "Residual weight=0.320000."
        ),
    }


def _temporal_mean(matrix: list[list[float]]) -> list[float]:
    return [
        sum(row[column] for row in matrix) / len(matrix)
        for column in range(len(matrix[0]))
    ]


def _contains_key(value: object, target: str) -> bool:
    if isinstance(value, dict):
        return target in value or any(_contains_key(item, target) for item in value.values())
    if isinstance(value, list):
        return any(_contains_key(item, target) for item in value)
    return False


if __name__ == "__main__":
    unittest.main()
