from __future__ import annotations

import json
import unittest
from pathlib import Path

from wmloop.contracts import validate_document
from wmloop.diagnose.probes.cpbe_residual_397450ddec import (
    DOSE_SCHEDULE,
    INVARIANTS,
    CpbeResidual397450ddecError,
    apply_multi_scale_event_phase,
    measure_cpbe_residual,
)


ROOT = Path(__file__).resolve().parents[1]
DESCRIPTOR_PATH = (
    ROOT / "configs" / "probes" / "staging" / "cpbe_residual_397450ddec.json"
)


class CpbeResidual397450ddecTests(unittest.TestCase):
    def test_zero_dose_is_exact_identity(self) -> None:
        fixture = _trajectory()

        transformed = apply_multi_scale_event_phase(
            action_sequence=fixture["action_sequence"],
            action_embeddings=fixture["action_embeddings"],
            dose=0.0,
        )

        self.assertEqual(transformed, fixture["action_embeddings"])

    def test_multi_scale_boundary_is_deterministic_and_mean_preserving(self) -> None:
        fixture = _trajectory()

        transformed = apply_multi_scale_event_phase(
            action_sequence=fixture["action_sequence"],
            action_embeddings=fixture["action_embeddings"],
            dose=0.05,
        )

        self.assertNotEqual(transformed[0], fixture["action_embeddings"][0])
        self.assertNotEqual(transformed[-1], fixture["action_embeddings"][-1])
        for column in range(len(transformed[0])):
            baseline_mean = sum(row[column] for row in fixture["action_embeddings"]) / len(transformed)
            transformed_mean = sum(row[column] for row in transformed) / len(transformed)
            self.assertAlmostEqual(transformed_mean, baseline_mean, places=14)

    def test_schema_valid_aggregation_matches_frozen_program(self) -> None:
        output = measure_cpbe_residual(
            trajectories=[_trajectory("trajectory-a", offset=0.0), _trajectory("trajectory-b", offset=0.2)],
            evidence_refs=["cas://sha256/" + "a" * 64],
        )

        validate_document("diagnostic_probe_output", output)
        self.assertEqual(output["probe_id"], "cpbe_residual_397450ddec")
        self.assertEqual(output["environment"], "push_cube")
        self.assertEqual(
            output["signature"],
            "mixed_source_sign_positive_prediction_vs_negative_target",
        )
        self.assertEqual(output["metrics"]["dose_schedule"], list(DOSE_SCHEDULE))
        self.assertEqual(output["metrics"]["temporal_scales"], [1, 2])
        self.assertEqual(output["metrics"]["invariants_checked"], list(INVARIANTS))
        self.assertLessEqual(output["metrics"]["max_temporal_mean_abs_error"], 1e-14)
        self.assertEqual(output["metrics"]["max_zero_dose_abs_error"], 0.0)
        self.assertEqual(output["metrics"]["zero_dose_goal_outcome_vector"], [0.7, -0.2])

    def test_dose_and_fixture_boundaries_fail_closed(self) -> None:
        fixture = _trajectory()
        with self.assertRaisesRegex(CpbeResidual397450ddecError, "DOSE_OUTSIDE_FROZEN_GRID"):
            apply_multi_scale_event_phase(
                action_sequence=fixture["action_sequence"],
                action_embeddings=fixture["action_embeddings"],
                dose=0.01,
            )
        with self.assertRaisesRegex(CpbeResidual397450ddecError, "SEQUENCE_TOO_SHORT"):
            apply_multi_scale_event_phase(
                action_sequence=[[0.0], [1.0]],
                action_embeddings=[[0.0], [1.0]],
                dose=0.05,
            )
        broken = _trajectory()
        broken["goal_outcomes"] = broken["goal_outcomes"][:-1]
        with self.assertRaisesRegex(CpbeResidual397450ddecError, "DOSE_GRID_MISMATCH"):
            measure_cpbe_residual(trajectories=[broken])

    def test_invariant_and_numeric_violations_fail_closed(self) -> None:
        duplicate = _trajectory()
        with self.assertRaisesRegex(CpbeResidual397450ddecError, "TRAJECTORY_DUPLICATE"):
            measure_cpbe_residual(trajectories=[duplicate, duplicate])

        nonfinite = _trajectory()
        nonfinite["action_sequence"][2][0] = float("nan")
        with self.assertRaisesRegex(CpbeResidual397450ddecError, "NONFINITE"):
            measure_cpbe_residual(trajectories=[nonfinite])

        mismatched = _trajectory("trajectory-a")
        second = _trajectory("trajectory-b")
        second["goal_outcomes"][0]["vector"] = [0.1]
        with self.assertRaisesRegex(CpbeResidual397450ddecError, "OUTCOME_DIMENSION_MISMATCH"):
            measure_cpbe_residual(trajectories=[mismatched, second])

    def test_descriptor_and_output_cannot_expose_verdict_evidence(self) -> None:
        descriptor = json.loads(DESCRIPTOR_PATH.read_text(encoding="utf-8"))
        output = measure_cpbe_residual(trajectories=[_trajectory()])

        self.assertEqual(descriptor["role"], "diagnostic")
        self.assertFalse(descriptor["verdict_exposure_allowed"])
        self.assertEqual(descriptor["program"]["diagnostic_only"], True)
        self.assertEqual(descriptor["program"]["temporal_basis"], "multi_scale_event_phase")
        self.assertEqual(descriptor["program"]["aggregation"], "goal_outcome_vector")
        self.assertNotIn("verdict_evidence", descriptor)
        self.assertFalse(output["verdict_exposure_allowed"])
        self.assertNotIn("verdict_evidence", output)


def _trajectory(
    trajectory_id: str = "trajectory-a", *, offset: float = 0.0
) -> dict[str, object]:
    return {
        "trajectory_id": trajectory_id,
        "checkpoint_id": "checkpoint-frozen",
        "input_action_id": f"input-{trajectory_id}",
        "seed": 17,
        "evaluator_id": "offline-fixture-evaluator",
        "action_sequence": [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ],
        "action_embeddings": [
            [0.0, 1.0],
            [1.0, 2.0],
            [2.0, 1.0],
            [1.0, 0.0],
            [0.0, -1.0],
        ],
        "goal_outcomes": [
            {"dose": -0.05, "vector": [0.40 + offset, -0.05]},
            {"dose": -0.025, "vector": [0.50 + offset, -0.10]},
            {"dose": 0.0, "vector": [0.60 + offset, -0.20]},
            {"dose": 0.025, "vector": [0.75 + offset, -0.25]},
            {"dose": 0.05, "vector": [0.90 + offset, -0.30]},
        ],
    }


if __name__ == "__main__":
    unittest.main()
