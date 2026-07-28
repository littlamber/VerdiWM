from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import unittest

from jsonschema import Draft202012Validator

from wmloop.geometry.assets import IRGChartSource, compose_irg_asset, validate_irg_asset
from wmloop.geometry.types import GeometryValidationError


class AcwmUnifiedIRGAssetTests(unittest.TestCase):
    def test_checked_in_eight_environment_bundle_matches_static_and_dynamic_contracts(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        bundle_root = repo_root / "examples" / "acwm_unified_irg_assets_v1"
        schema = json.loads(
            (repo_root / "configs" / "schemas" / "irg_asset.schema.json").read_text(
                encoding="utf-8"
            )
        )
        validator = Draft202012Validator(schema)
        index = json.loads((bundle_root / "index.json").read_text(encoding="utf-8"))
        self.assertEqual(index["environment_count"], 8)
        self.assertEqual(index["routing_ready_environment_count"], 8)
        self.assertEqual(index["transfer_ready_environment_count"], 0)
        for relative in index["asset_paths"].values():
            asset = json.loads((bundle_root / relative).read_text(encoding="utf-8"))
            validator.validate(asset)
            validate_irg_asset(asset)
            self.assertEqual(
                asset["dimensions"],
                {
                    "outcome_count": 4,
                    "probe_path_count": 7,
                    "response_coordinate_count": 28,
                },
            )

    def test_compatible_sources_form_joint_covariance_and_transfer_ready_asset(self) -> None:
        left = self._source("left", "probe_a", baseline_shift=0.0)
        right = self._source("right", "probe_b", baseline_shift=0.0)

        asset = self._compose((left, right))

        self.assertEqual(asset["dimensions"], {
            "outcome_count": 2,
            "probe_path_count": 2,
            "response_coordinate_count": 4,
        })
        self.assertEqual(asset["symbol_table"]["J_X"], "jacobian")
        self.assertEqual(asset["covariance_contract"]["joint_baseline_group_count"], 1)
        self.assertEqual(asset["transfer_state"], "ready")
        self.assertEqual(asset["transfer_blockers"], [])
        self.assertNotEqual(asset["response_covariance"][0][1], 0.0)
        validate_irg_asset(asset)

    def test_baseline_mismatch_zero_fills_cross_group_covariance_and_abstains(self) -> None:
        left = self._source("left", "probe_a", baseline_shift=0.0)
        right = self._source("right", "probe_b", baseline_shift=1.0)

        asset = self._compose((left, right))

        self.assertEqual(asset["covariance_contract"]["joint_baseline_group_count"], 2)
        self.assertEqual(asset["transfer_state"], "abstain")
        self.assertEqual(asset["transfer_blockers"], ["joint_baseline_frame_mismatch"])
        self.assertEqual(asset["response_covariance"][0][1], 0.0)
        self.assertEqual(asset["response_covariance"][2][3], 0.0)
        validate_irg_asset(asset)

    def test_unsupported_path_is_preserved_raw_and_masked_from_effective_assets(self) -> None:
        source = self._source("unsupported", "probe_a", baseline_shift=0.0, locality=0.9)

        asset = self._compose((source,))

        self.assertNotEqual(asset["raw_jacobian"][0][0], 0.0)
        self.assertEqual(asset["jacobian"], [[0.0], [0.0]])
        self.assertEqual(asset["repair_metric"], [[0.0]])
        self.assertEqual(asset["response_coordinate"], [0.0, 0.0])
        self.assertEqual(asset["routing_state"], "abstain")
        validate_irg_asset(asset)

    def test_dynamic_validator_rejects_malformed_matrix_and_mask_leakage(self) -> None:
        asset = self._compose((self._source("source", "probe_a", baseline_shift=0.0),))
        malformed = copy.deepcopy(asset)
        malformed["jacobian"][0].append(1.0)
        with self.assertRaisesRegex(GeometryValidationError, "IRG_ASSET_JACOBIAN_INVALID"):
            validate_irg_asset(malformed)

        unsupported = self._compose((self._source(
            "unsupported", "probe_a", baseline_shift=0.0, locality=0.9
        ),))
        unsupported["jacobian"][0][0] = 1.0
        with self.assertRaisesRegex(
            GeometryValidationError, "IRG_ASSET_UNSUPPORTED_JACOBIAN_NONZERO"
        ):
            validate_irg_asset(unsupported)

    def _compose(self, sources: tuple[IRGChartSource, ...]) -> dict[str, object]:
        return compose_irg_asset(
            asset_id="fixture:environment:r1",
            environment="fixture_environment",
            backbone_family="fixture_backbone",
            capability_class="fixture_capability",
            backbone_instance_ref="configs/backbones/fixture.json",
            sources=sources,
            locality_threshold=0.5,
        )

    def _source(
        self,
        source_id: str,
        path_name: str,
        *,
        baseline_shift: float,
        locality: float = 0.1,
    ) -> IRGChartSource:
        repeats = (
            ((1.0,), (2.0,)),
            ((2.0,), (4.0,)),
            ((3.0,), (6.0,)),
        )
        weights = (1.0, 4.0)
        jacobian = self._mean(repeats)
        coordinates = tuple(
            tuple(math.sqrt(weights[outcome]) * value for outcome, row in enumerate(matrix) for value in row)
            for matrix in repeats
        )
        covariance = self._covariance(coordinates)
        baselines = tuple(
            (20.0 + baseline_shift + seed_offset, 0.8 + baseline_shift)
            for seed_offset in (0.0, 0.1, 0.2)
        )
        return IRGChartSource(
            source_id=source_id,
            campaign_id=f"campaign:{source_id}",
            path_names=(path_name,),
            outcome_names=("psnr", "ssim"),
            outcome_weights=weights,
            seeds=(101, 202, 303),
            jacobian=jacobian,
            covariance=covariance,
            locality_residuals={path_name: locality},
            repeat_jacobians=repeats,
            baseline_vectors=baselines,
            checkpoint_step=1000,
            provenance={"chart_sha256": "0" * 64},
        )

    @staticmethod
    def _mean(matrices: tuple[tuple[tuple[float, ...], ...], ...]) -> tuple[tuple[float, ...], ...]:
        return tuple(
            tuple(sum(matrix[row][column] for matrix in matrices) / len(matrices) for column in range(len(matrices[0][0])))
            for row in range(len(matrices[0]))
        )

    @staticmethod
    def _covariance(vectors: tuple[tuple[float, ...], ...]) -> tuple[tuple[float, ...], ...]:
        means = tuple(sum(row[index] for row in vectors) / len(vectors) for index in range(len(vectors[0])))
        return tuple(
            tuple(
                sum((row[i] - means[i]) * (row[j] - means[j]) for row in vectors) / (len(vectors) - 1)
                for j in range(len(means))
            )
            for i in range(len(means))
        )


if __name__ == "__main__":
    unittest.main()
