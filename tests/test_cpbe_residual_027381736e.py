from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import unittest

from wmloop.contracts import validate_document
from wmloop.diagnose.probes.cpbe_residual_027381736e import (
    CPBEResidual027381736eError,
    DOSE_SCHEDULE,
    apply_contrast_dose,
    measure_probe,
    program_contract,
)


ROOT = Path(__file__).resolve().parents[1]
DESCRIPTOR = ROOT / "configs/probes/staging/cpbe_residual_027381736e.json"


class CPBEResidual027381736eTests(unittest.TestCase):
    def test_offline_fixture_is_schema_valid_and_diagnostic_only(self) -> None:
        output = measure_probe(
            fixture=_fixture(),
            evidence_refs=["cas://sha256/" + "a" * 64],
        )

        self.assertEqual(output["probe_id"], "cpbe_residual_027381736e")
        self.assertEqual(output["role"], "diagnostic")
        self.assertEqual(output["state"], "measured")
        self.assertFalse(output["verdict_exposure_allowed"])
        self.assertNotIn("verdict_evidence", _all_keys(output))
        for observed, expected in zip(
            output["metrics"]["signed_goal_outcome_response_vector"],
            [2.0, -1.0],
        ):
            self.assertAlmostEqual(observed, expected, places=12)
        self.assertEqual(output["flags"], [])
        validate_document("diagnostic_probe_output", output)

    def test_zero_dose_and_signed_scale_preserve_input_and_temporal_mean(self) -> None:
        fixture = _fixture()
        original_fixture = deepcopy(fixture)
        embeddings = fixture["action_embeddings"]
        transformed = {
            dose: apply_contrast_dose(
                action_sequence=fixture["action_sequence"],
                action_embeddings=embeddings,
                dose=dose,
            )
            for dose in DOSE_SCHEDULE
        }

        self.assertEqual(transformed[0.0], embeddings)
        self.assertEqual(fixture, original_fixture)
        for dose in DOSE_SCHEDULE:
            for column in range(len(embeddings[0])):
                before = sum(row[column] for row in embeddings) / len(embeddings)
                after = sum(row[column] for row in transformed[dose]) / len(transformed[dose])
                self.assertAlmostEqual(before, after, places=12)
        for original_row, negative_row, positive_row in zip(
            embeddings, transformed[-0.05], transformed[0.05]
        ):
            for original, negative, positive in zip(original_row, negative_row, positive_row):
                self.assertAlmostEqual(negative + positive, 2.0 * original, places=12)

    def test_zero_transition_sequence_is_an_invariant_boundary(self) -> None:
        embeddings = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        transformed = apply_contrast_dose(
            action_sequence=[[0.5], [0.5], [0.5]],
            action_embeddings=embeddings,
            dose=0.05,
        )

        self.assertEqual(transformed, embeddings)

    def test_action_phase_weights_resample_to_embedding_timeline(self) -> None:
        embeddings = [[0.0], [1.0], [2.0], [3.0], [4.0]]
        transformed = apply_contrast_dose(
            action_sequence=[[0.0], [1.0], [1.0]],
            action_embeddings=embeddings,
            dose=0.05,
        )

        self.assertEqual(len(transformed), len(embeddings))
        self.assertNotEqual(transformed, embeddings)
        self.assertAlmostEqual(
            sum(row[0] for row in transformed),
            sum(row[0] for row in embeddings),
            places=12,
        )

    def test_frozen_dose_grid_and_paired_context_fail_closed(self) -> None:
        missing_dose = _fixture()
        missing_dose["dose_observations"] = missing_dose["dose_observations"][:-1]
        with self.assertRaisesRegex(CPBEResidual027381736eError, "DOSE_SCHEDULE_MISMATCH"):
            measure_probe(fixture=missing_dose)

        reordered = _fixture()
        reordered["dose_observations"][0], reordered["dose_observations"][1] = (
            reordered["dose_observations"][1],
            reordered["dose_observations"][0],
        )
        with self.assertRaisesRegex(CPBEResidual027381736eError, "DOSE_SCHEDULE_MISMATCH"):
            measure_probe(fixture=reordered)

        mismatched_context = _fixture()
        mismatched_context["dose_observations"][-1]["seed"] = 202
        with self.assertRaisesRegex(CPBEResidual027381736eError, "PAIRED_CONTEXT_MISMATCH"):
            measure_probe(fixture=mismatched_context)

        with self.assertRaisesRegex(CPBEResidual027381736eError, "DOSE_OUTSIDE_FROZEN_SCHEDULE"):
            apply_contrast_dose(
                action_sequence=_fixture()["action_sequence"],
                action_embeddings=_fixture()["action_embeddings"],
                dose=0.1,
            )

    def test_invalid_schema_boundaries_fail_closed(self) -> None:
        invalid = _fixture()
        invalid["action_embeddings"][1][0] = float("nan")
        with self.assertRaisesRegex(CPBEResidual027381736eError, "VALUE_INVALID"):
            measure_probe(fixture=invalid)

        invalid = _fixture()
        invalid["dose_observations"][0]["goal_outcome_vector"] = [1.0]
        with self.assertRaisesRegex(CPBEResidual027381736eError, "GOAL_OUTCOME_VECTOR_INVALID"):
            measure_probe(fixture=invalid)

        with self.assertRaisesRegex(CPBEResidual027381736eError, "EVIDENCE_REFS_INVALID"):
            measure_probe(fixture=_fixture(), evidence_refs=["results/not-cas.json"])

    def test_staging_descriptor_matches_code_and_withholds_routing(self) -> None:
        descriptor = json.loads(DESCRIPTOR.read_text(encoding="utf-8"))

        self.assertEqual(descriptor["probe_id"], "cpbe_residual_027381736e")
        self.assertEqual(descriptor["role"], "diagnostic")
        self.assertEqual(descriptor["program"], program_contract())
        self.assertEqual(descriptor["required_stages"], ["static", "offline", "canary", "expanded"])
        self.assertIn("diagnostic-only", descriptor["signal_contract"])
        self.assertFalse(descriptor["verdict_exposure_allowed"])
        self.assertFalse(descriptor["routing_allowed"])
        self.assertFalse(descriptor["gpu_execution_required"])
        self.assertEqual(descriptor["admission"]["runtime_smoke_on_dev_split"], "pending")
        self.assertEqual(descriptor["admission"]["locality_and_nonredundancy_canary_passed"], "pending")
        self.assertEqual(descriptor["admission"]["selector_regret_or_coverage_gain_observed"], "pending")
        self.assertNotIn("verdict_evidence", _all_keys(descriptor))


def _fixture() -> dict[str, object]:
    observations = []
    for dose in DOSE_SCHEDULE:
        observations.append(
            {
                "dose": dose,
                "goal_outcome_vector": [10.0 + 2.0 * dose, 0.5 - dose],
                "checkpoint_id": "checkpoint-fixture",
                "trajectory_id": "trajectory-fixture",
                "seed": 101,
                "evaluator_id": "offline-fixture-evaluator",
            }
        )
    return {
        "action_sequence": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        "action_embeddings": [[1.0, 0.0], [2.0, 1.0], [4.0, 1.0], [5.0, 2.0]],
        "outcome_names": ["quality", "error"],
        "dose_observations": observations,
    }


def _all_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value).union(*(_all_keys(item) for item in value.values()))
    if isinstance(value, list):
        return set().union(*(_all_keys(item) for item in value)) if value else set()
    return set()


if __name__ == "__main__":
    unittest.main()
