from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from wmloop.experiments.selector_replay import run_selector_replay


class ACWMSelectorCPUReplayTests(unittest.TestCase):
    def test_irg_abstains_and_emits_work_orders_when_candidate_probe_coverage_is_incomplete(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            environments = ("a", "b", "c")
            plan, projections = self._write_two_path_inputs(root, environments)
            labels = root / "labels.json"
            labels.write_text(
                json.dumps(
                    {
                        "labels": [
                            self._label(environment, primitive, positive, 1.0 if positive else -1.0)
                            for environment in environments
                            for primitive, positive in (("covered", True), ("missing", False))
                        ]
                    }
                ),
                encoding="utf-8",
            )
            affinity = root / "affinity.json"
            affinity.write_text(
                json.dumps(
                    self._affinity(
                        {
                            "covered": {
                                "coverage_state": "covered",
                                "required_probe_paths": ["amplification"],
                            },
                            "missing": {
                                "coverage_state": "probe_missing",
                                "required_probe_paths": ["exposure_recovery"],
                                "successor_probe_axis": "exposure_recovery",
                            },
                        }
                    )
                ),
                encoding="utf-8",
            )

            manifest = run_selector_replay(
                plan_path=plan,
                projection_path=projections,
                effect_label_index=labels,
                primitive_probe_affinity=affinity,
                output_root=root / "output",
            )

            self.assertEqual(manifest["state"], "partial")
            self.assertEqual(manifest["evaluated_cell_count"], 9)
            self.assertEqual(manifest["abstained_cell_count"], 3)
            self.assertFalse(manifest["probe_coverage_ready"])
            self.assertEqual(manifest["probe_evolution_work_order_count"], 3)
            report = json.loads((root / "output" / "selector-replay.json").read_text())
            irg_cells = [row for row in report["cells"] if row["selector"] == "irg"]
            self.assertTrue(all(row["abstention_reason"] == "candidate_probe_coverage_incomplete" for row in irg_cells))
            self.assertEqual(
                {row["successor_probe_axis"] for row in report["probe_evolution_work_orders"]},
                {"exposure_recovery"},
            )

    def test_candidate_conditioned_irg_can_create_auditable_selector_divergence(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            environments = ("a", "b", "target")
            plan, projections = self._write_two_path_inputs(
                root,
                environments,
                irg_coordinates={"a": (0.0, 0.0), "b": (10.0, 10.0), "target": (4.0, 10.0)},
            )
            labels = root / "labels.json"
            labels.write_text(
                json.dumps(
                    {
                        "labels": [
                            self._label("a", "amp_candidate", True, 1.0),
                            self._label("a", "z_att_candidate", False, -1.0),
                            self._label("b", "amp_candidate", False, -1.0),
                            self._label("b", "z_att_candidate", True, 1.0),
                            self._label("target", "amp_candidate", False, -1.0),
                            self._label("target", "z_att_candidate", True, 1.0),
                        ]
                    }
                ),
                encoding="utf-8",
            )
            affinity = root / "affinity.json"
            affinity.write_text(
                json.dumps(
                    self._affinity(
                        {
                            "amp_candidate": {
                                "coverage_state": "covered",
                                "required_probe_paths": ["amplification"],
                            },
                            "z_att_candidate": {
                                "coverage_state": "covered",
                                "required_probe_paths": ["attenuation"],
                            },
                        }
                    )
                ),
                encoding="utf-8",
            )

            manifest = run_selector_replay(
                plan_path=plan,
                projection_path=projections,
                effect_label_index=labels,
                primitive_probe_affinity=affinity,
                output_root=root / "output",
            )

            self.assertTrue(manifest["formal_comparison_ready"])
            self.assertTrue(manifest["selection_discrimination_ready"])
            report = json.loads((root / "output" / "selector-replay.json").read_text())
            target_cells = {
                row["selector"]: row
                for row in report["cells"]
                if row["target_environment"] == "target"
            }
            self.assertEqual(target_cells["environment_label"]["selected_primitive"], "amp_candidate")
            self.assertEqual(target_cells["irg"]["selected_primitive"], "z_att_candidate")
            self.assertEqual(target_cells["irg"]["top1_positive_hit"], 1.0)

    def test_irg_transfer_certificate_rejects_single_source_extrapolation(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            environments = ("a", "b", "target")
            plan, projections = self._write_two_path_inputs(root, environments)
            labels = root / "labels.json"
            labels.write_text(
                json.dumps(
                    {
                        "labels": [
                            self._label("a", "single_source", True, 1.0),
                            self._label("a", "supported_negative", False, -1.0),
                            self._label("b", "supported_negative", False, -1.0),
                            self._label("target", "single_source", False, -1.0),
                            self._label("target", "supported_negative", False, -1.0),
                        ]
                    }
                ),
                encoding="utf-8",
            )
            affinity_payload = self._affinity(
                {
                    primitive: {
                        "coverage_state": "covered",
                        "required_probe_paths": ["amplification"],
                    }
                    for primitive in ("single_source", "supported_negative")
                }
            )
            affinity_payload["transfer_certificate"] = {
                "minimum_nonleaking_source_environments": 2,
                "minimum_selected_positive_probability": 0.75,
                "require_unanimous_positive_sources": True,
                "distance_support_policy": "source_effect_leave_one_out_max_nearest",
                "failure_policy": "abstain_fold",
            }
            affinity = root / "affinity.json"
            affinity.write_text(json.dumps(affinity_payload), encoding="utf-8")

            run_selector_replay(
                plan_path=plan,
                projection_path=projections,
                effect_label_index=labels,
                primitive_probe_affinity=affinity,
                output_root=root / "output",
            )

            report = json.loads((root / "output" / "selector-replay.json").read_text())
            target = next(
                row
                for row in report["cells"]
                if row["target_environment"] == "target" and row["selector"] == "irg"
            )
            self.assertEqual(target["abstention_reason"], "transfer_certificate_failed")
            self.assertEqual(target["selected_primitive_before_certificate"], "single_source")
            self.assertFalse(
                target["transfer_certificate"]["checks"]["nonleaking_source_environment_count"]["passed"]
            )

    def test_irg_transfer_certificate_rejects_mixed_source_effect_signs(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            environments = ("a", "b", "target")
            plan, projections = self._write_two_path_inputs(root, environments)
            labels = root / "labels.json"
            labels.write_text(
                json.dumps(
                    {
                        "labels": [
                            self._label(source, primitive, positive, 1.0 if positive else -1.0)
                            for source, primitive, positive in (
                                ("a", "mixed", True),
                                ("b", "mixed", False),
                                ("a", "other", False),
                                ("b", "other", False),
                                ("target", "mixed", False),
                                ("target", "other", False),
                            )
                        ]
                    }
                ),
                encoding="utf-8",
            )
            affinity_payload = self._affinity(
                {
                    primitive: {
                        "coverage_state": "covered",
                        "required_probe_paths": ["amplification"],
                    }
                    for primitive in ("mixed", "other")
                }
            )
            affinity_payload["transfer_certificate"] = {
                "minimum_nonleaking_source_environments": 2,
                "minimum_selected_positive_probability": 0.75,
                "require_unanimous_positive_sources": True,
                "distance_support_policy": "source_effect_leave_one_out_max_nearest",
                "failure_policy": "abstain_fold",
            }
            affinity = root / "affinity.json"
            affinity.write_text(json.dumps(affinity_payload), encoding="utf-8")

            run_selector_replay(
                plan_path=plan,
                projection_path=projections,
                effect_label_index=labels,
                primitive_probe_affinity=affinity,
                output_root=root / "output",
            )

            report = json.loads((root / "output" / "selector-replay.json").read_text())
            target = next(
                row
                for row in report["cells"]
                if row["target_environment"] == "target" and row["selector"] == "irg"
            )
            self.assertEqual(target["abstention_reason"], "transfer_certificate_failed")
            self.assertFalse(
                target["transfer_certificate"]["checks"]["source_effect_sign_consistency"]["passed"]
            )
            self.assertTrue(
                any(row["reason"] == "source_effect_sign_consistency" for row in report["transfer_work_orders"])
            )

    def test_replays_supported_folds_and_abstains_without_source_support(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            selectors = ("environment_label", "static_probe", "raw_response", "irg")
            environments = ("a", "b", "c")
            trials = []
            sequence = 0
            for target in environments:
                for selector in selectors:
                    sequence += 1
                    trials.append(
                        {
                            "trial_id": f"{target}-{selector}",
                            "fold_id": f"holdout-{target}",
                            "target_environment": target,
                            "source_environments": [value for value in environments if value != target],
                            "selector": selector,
                            "seed": 101,
                        }
                    )
            plan = root / "plan.json"
            plan.write_text(json.dumps({"trials": trials}), encoding="utf-8")
            projections = root / "projections.jsonl"
            projections.write_text(
                "".join(
                    json.dumps(
                        {
                            "environment": environment,
                            "selector": selector,
                            "features": [float(index), float(index + 1)],
                        }
                    )
                    + "\n"
                    for index, environment in enumerate(environments)
                    for selector in selectors
                ),
                encoding="utf-8",
            )
            labels = root / "labels.json"
            labels.write_text(
                json.dumps(
                    {
                        "labels": [
                            self._label("a", "shared", True, 1.0),
                            self._label("b", "shared", True, 0.5),
                            self._label("c", "target_only", False, -1.0),
                        ]
                    }
                ),
                encoding="utf-8",
            )
            manifest = run_selector_replay(
                plan_path=plan,
                projection_path=projections,
                effect_label_index=labels,
                output_root=root / "output",
            )
            self.assertEqual(manifest["state"], "partial")
            self.assertEqual(manifest["evaluated_cell_count"], 8)
            self.assertEqual(manifest["abstained_cell_count"], 4)
            report = json.loads((root / "output" / "selector-replay.json").read_text())
            self.assertEqual(report["evaluated_environment_count"], 2)
            self.assertFalse(report["formal_comparison_ready"])
            self.assertFalse(report["selection_discrimination_ready"])
            self.assertTrue(all(row["top1_positive_hit"] == 1.0 for row in report["selectors"]))

    def test_complete_identical_choices_are_not_selection_discriminative(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            selectors = ("environment_label", "static_probe", "raw_response", "irg")
            environments = ("a", "b")
            plan = root / "plan.json"
            plan.write_text(
                json.dumps(
                    {
                        "trials": [
                            {
                                "trial_id": f"{target}-{selector}",
                                "fold_id": f"holdout-{target}",
                                "target_environment": target,
                                "source_environments": [source for source in environments if source != target],
                                "selector": selector,
                                "seed": 101,
                            }
                            for target in environments
                            for selector in selectors
                        ]
                    }
                ),
                encoding="utf-8",
            )
            projections = root / "projections.jsonl"
            projections.write_text(
                "".join(
                    json.dumps({"environment": environment, "selector": selector, "features": [0.0]}) + "\n"
                    for environment in environments
                    for selector in selectors
                ),
                encoding="utf-8",
            )
            labels = root / "labels.json"
            labels.write_text(
                json.dumps(
                    {
                        "labels": [
                            self._label(environment, primitive, positive, 1.0 if positive else -1.0)
                            for environment in environments
                            for primitive, positive in (("positive", True), ("negative", False))
                        ]
                    }
                ),
                encoding="utf-8",
            )

            manifest = run_selector_replay(
                plan_path=plan,
                projection_path=projections,
                effect_label_index=labels,
                output_root=root / "output",
            )

            self.assertTrue(manifest["formal_comparison_ready"])
            self.assertFalse(manifest["selection_discrimination_ready"])

    def test_ambiguous_source_effect_does_not_support_transfer(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            selectors = ("environment_label", "static_probe", "raw_response", "irg")
            environments = ("source", "target")
            plan = root / "plan.json"
            plan.write_text(
                json.dumps(
                    {
                        "trials": [
                            {
                                "trial_id": f"target-{selector}",
                                "fold_id": "holdout-target",
                                "target_environment": "target",
                                "source_environments": ["source"],
                                "selector": selector,
                                "seed": 101,
                            }
                            for selector in selectors
                        ]
                    }
                ),
                encoding="utf-8",
            )
            projections = root / "projections.jsonl"
            projections.write_text(
                "".join(
                    json.dumps(
                        {
                            "environment": environment,
                            "selector": selector,
                            "features": [0.0],
                        }
                    )
                    + "\n"
                    for environment in environments
                    for selector in selectors
                ),
                encoding="utf-8",
            )
            labels = root / "labels.json"
            labels.write_text(
                json.dumps(
                    {
                        "labels": [
                            self._label("source", "mixed", True, 0.2),
                            self._label("source", "mixed", False, -0.1),
                            self._label("target", "mixed", True, 0.3),
                        ]
                    }
                ),
                encoding="utf-8",
            )

            manifest = run_selector_replay(
                plan_path=plan,
                projection_path=projections,
                effect_label_index=labels,
                output_root=root / "output",
            )

            self.assertEqual(manifest["state"], "partial")
            self.assertEqual(manifest["evaluated_cell_count"], 0)
            self.assertEqual(manifest["abstained_cell_count"], 4)

    @staticmethod
    def _label(environment: str, primitive: str, positive: bool, psnr: float) -> dict[str, object]:
        return {
            "environment": environment,
            "primitive": primitive,
            "settled": True,
            "positive": positive,
            "delta_candidate_minus_baseline": {"psnr": psnr},
        }

    def _write_two_path_inputs(
        self,
        root: Path,
        environments: tuple[str, ...],
        *,
        irg_coordinates: dict[str, tuple[float, float]] | None = None,
    ) -> tuple[Path, Path]:
        selectors = ("environment_label", "static_probe", "raw_response", "irg")
        plan = root / "plan.json"
        plan.write_text(
            json.dumps(
                {
                    "trials": [
                        {
                            "trial_id": f"{target}-{selector}",
                            "fold_id": f"holdout-{target}",
                            "target_environment": target,
                            "source_environments": [source for source in environments if source != target],
                            "selector": selector,
                            "seed": 101,
                        }
                        for target in environments
                        for selector in selectors
                    ]
                }
            ),
            encoding="utf-8",
        )
        coordinates = irg_coordinates or {
            environment: (float(index), float(index))
            for index, environment in enumerate(environments)
        }
        rows = []
        for environment in environments:
            for selector in selectors:
                if selector == "irg":
                    amp, attenuation = coordinates[environment]
                    rows.append(
                        {
                            "environment": environment,
                            "selector": selector,
                            "feature_names": [
                                "response_coordinate:0",
                                "response_coordinate:1",
                                "covariance_diagonal:0",
                                "covariance_diagonal:1",
                                "locality:amplification",
                                "locality:attenuation",
                                "path_supported:amplification",
                                "path_supported:attenuation",
                            ],
                            "features": [amp, attenuation, 0.0, 0.0, 0.1, 0.1, 1.0, 1.0],
                        }
                    )
                else:
                    rows.append(
                        {
                            "environment": environment,
                            "selector": selector,
                            "feature_names": ["constant"],
                            "features": [0.0],
                        }
                    )
        projections = root / "projections.jsonl"
        projections.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return plan, projections

    @staticmethod
    def _affinity(primitives: dict[str, dict[str, object]]) -> dict[str, object]:
        return {
            "schema_version": 1,
            "artifact_type": "verdiwm-primitive-probe-affinity",
            "projection_path_order": ["amplification", "attenuation"],
            "minimum_covered_candidates_per_fold": 2,
            "require_full_candidate_coverage": True,
            "primitives": primitives,
        }


if __name__ == "__main__":
    unittest.main()
