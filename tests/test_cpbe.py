from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from wmloop.experiments.cpbe import (
    CPBEError,
    build_cpbe_plan,
    publish_cpbe_plan,
    publish_cpbe_settlement,
    settle_cpbe_plan,
)


def _probe(
    probe_id: str,
    *,
    origin: str = "retrieval",
    signal_source: str = "latent_delta",
    hook_type: str = "action_conditioning",
    spatial_mask: str = "global",
    temporal_basis: str = "latest",
    contrast_operator: str = "signed",
    aggregation: str = "mean",
    cost: float = 0.5,
) -> dict[str, object]:
    return {
        "probe_id": probe_id,
        "signal_source": signal_source,
        "hook_type": hook_type,
        "spatial_mask": spatial_mask,
        "temporal_basis": temporal_basis,
        "contrast_operator": contrast_operator,
        "dose_schedule": [-0.05, 0.0, 0.05],
        "aggregation": aggregation,
        "invariants": ["same_seed", "same_trajectory", "same_evaluator"],
        "required_capabilities": ["paired_seed_control"],
        "estimated_gpu_hours": cost,
        "origin": origin,
        "diagnostic_only": True,
        "reversible": True,
    }


def _request() -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_type": "verdiwm-cpbe-request",
        "experiment_id": "cpbe-push-sand-v1",
        "evidence_class": "synthetic_fixture",
        "context": {
            "collision_id": "motion-region-cloth-vs-sand",
            "target_id": "push_sand",
            "backbone_family": "acwm_phys",
            "capability_class": "latent_dit_action_conditioned",
            "failure_signature": "granular_frontier",
            "primitive": "motion_region_reweight",
            "available_hooks": ["action_conditioning", "latent_state"],
            "capabilities": ["paired_seed_control"],
            "unexplained_residual": {
                "signal_source": 0.1,
                "hook_type": 0.05,
                "spatial_mask": 0.25,
                "temporal_basis": 0.5,
                "contrast_operator": 0.05,
                "aggregation": 0.05
            },
            "residual_evidence_refs": ["fixture://collision/residual-attribution"],
            "max_canaries": 3
        },
        "current_probes": [_probe("motion_magnitude")],
        "grammar": {
            "signal_source": ["latent_delta", "contact_event", "boundary_flux"],
            "hook_type": ["action_conditioning", "latent_state", "unsupported_hook"],
            "spatial_mask": ["global", "motion_roi", "boundary_roi"],
            "temporal_basis": ["latest", "phase_lag", "curvature"],
            "contrast_operator": ["signed", "one_sided_positive", "orthogonal_rotation"],
            "aggregation": ["mean", "horizon_weighted", "endpoint"]
        },
        "retrieval_candidates": [
            _probe(
                "retrieved_boundary_phase",
                signal_source="boundary_flux",
                spatial_mask="boundary_roi",
                temporal_basis="phase_lag",
                aggregation="endpoint",
            )
        ],
        "llm_candidates": [
            _probe(
                "llm_contact_curvature",
                origin="llm",
                signal_source="contact_event",
                hook_type="latent_state",
                spatial_mask="motion_roi",
                temporal_basis="curvature",
                contrast_operator="orthogonal_rotation",
                aggregation="horizon_weighted",
            )
        ],
    }


def _history() -> list[dict[str, object]]:
    return [
        {
            "trial_id": "accepted-phase",
            "evidence_class": "synthetic_fixture",
            "context": {
                "backbone_family": "acwm_phys",
                "capability_class": "latent_dit_action_conditioned",
                "failure_signature": "granular_frontier",
                "primitive": "motion_region_reweight",
            },
            "probe": _probe("historical_phase", temporal_basis="phase_lag"),
            "outcomes": {
                "locality_pass": True,
                "nonredundant": True,
                "collision_resolved": True,
                "regret_reduction": 0.4,
                "coverage_gain": 0.25,
                "gpu_hours": 0.4,
            },
            "evidence_refs": ["fixture://history/accepted-phase"],
        },
        {
            "trial_id": "rejected-global",
            "evidence_class": "synthetic_fixture",
            "context": {
                "backbone_family": "acwm_phys",
                "capability_class": "latent_dit_action_conditioned",
                "failure_signature": "granular_frontier",
                "primitive": "motion_region_reweight",
            },
            "probe": _probe("historical_global"),
            "outcomes": {
                "locality_pass": False,
                "nonredundant": False,
                "collision_resolved": False,
                "regret_reduction": 0.0,
                "coverage_gain": 0.0,
                "gpu_hours": 0.5,
            },
            "evidence_refs": ["fixture://history/rejected-global"],
        },
    ]


class CPBEPlannerTests(unittest.TestCase):
    def test_uses_four_sources_and_keeps_llm_under_same_gate(self) -> None:
        plan = build_cpbe_plan(request=_request(), history=_history())
        origins = {row["origin"] for row in plan["ranking"]}
        self.assertEqual(origins, {"residual", "mutation", "retrieval", "llm"})
        self.assertEqual(plan["state"], "ready")
        self.assertEqual(len(plan["selected_work_orders"]), 3)
        for work_order in plan["selected_work_orders"]:
            self.assertFalse(work_order["verdict_exposure_allowed"])
            self.assertEqual(work_order["required_stages"], ["static", "offline", "canary", "expanded"])
            self.assertIn("selection_evidence", work_order)

    def test_multi_axis_llm_hypothesis_pays_an_intervention_complexity_penalty(self) -> None:
        plan = build_cpbe_plan(request=_request(), history=_history())
        llm = next(row for row in plan["ranking"] if row["origin"] == "llm")
        one_axis = next(row for row in plan["ranking"] if row["edit_count"] == 1)
        self.assertGreater(llm["edit_count"], 1)
        self.assertGreater(llm["complexity_penalty"], 0.0)
        self.assertEqual(one_axis["complexity_penalty"], 0.0)

    def test_residual_axis_and_history_change_ranking(self) -> None:
        plan = build_cpbe_plan(request=_request(), history=_history())
        phase_rows = [
            row
            for row in plan["ranking"]
            if row["program"]["temporal_basis"] == "phase_lag"
            and row["program"]["signal_source"] == "latent_delta"
        ]
        latest_rows = [
            row
            for row in plan["ranking"]
            if row["program"]["temporal_basis"] == "latest"
            and row["program"]["signal_source"] == "latent_delta"
        ]
        self.assertTrue(phase_rows)
        self.assertTrue(latest_rows)
        self.assertGreater(
            max(row["acquisition_score"] for row in phase_rows),
            max(row["acquisition_score"] for row in latest_rows),
        )

    def test_capability_filter_blocks_unsupported_hook(self) -> None:
        plan = build_cpbe_plan(request=_request(), history=_history())
        blocked = [row for row in plan["ranking"] if row["program"]["hook_type"] == "unsupported_hook"]
        self.assertTrue(blocked)
        self.assertTrue(all("hook_unavailable:unsupported_hook" in row["blockers"] for row in blocked))
        self.assertTrue(all(not row["selected_for_canary"] for row in blocked))

    def test_rejects_llm_program_outside_frozen_grammar(self) -> None:
        request = _request()
        request["llm_candidates"][0]["signal_source"] = "invented_unbounded_signal"
        with self.assertRaisesRegex(CPBEError, "CPBE_PROBE_OUTSIDE_GRAMMAR"):
            build_cpbe_plan(request=request, history=_history())

    def test_live_plan_does_not_train_surrogate_on_synthetic_history(self) -> None:
        request = _request()
        request["evidence_class"] = "live"
        plan = build_cpbe_plan(request=request, history=_history())
        self.assertEqual(plan["candidate_generation"]["history_trial_count"], 2)
        self.assertEqual(plan["candidate_generation"]["history_trial_count_used"], 0)
        self.assertEqual(plan["candidate_generation"]["synthetic_history_excluded"], 2)

    def test_history_schema_rejects_empty_probe_before_runtime_parsing(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            request_path = root / "request.json"
            history_path = root / "history.jsonl"
            request_path.write_text(json.dumps(_request()), encoding="utf-8")
            history = _history()[0]
            history["probe"] = {}
            history_path.write_text(json.dumps(history) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(CPBEError, "CPBE_HISTORY_CONTRACT_INVALID"):
                publish_cpbe_plan(
                    request_path=request_path,
                    history_path=history_path,
                    output_root=root / "plan",
                )

    def test_unobserved_history_outcomes_are_not_treated_as_failures_or_zero_gain(self) -> None:
        request = _request()
        request["evidence_class"] = "live"
        history = _history()
        history[0]["evidence_class"] = "historical_replay"
        history[0]["outcomes"].update(
            collision_resolved=None,
            regret_reduction=None,
            coverage_gain=None,
        )
        plan = build_cpbe_plan(request=request, history=[history[0]])
        matching = [
            row for row in plan["ranking"] if "accepted-phase" in row["history_trial_ids"]
        ]
        self.assertTrue(matching)
        self.assertTrue(all(row["p_collision_resolved"] == 0.5 for row in matching))
        self.assertTrue(all(row["expected_regret_reduction"] == 0.0 for row in matching))
        self.assertTrue(all(row["expected_coverage_gain"] == 0.0 for row in matching))

    def test_publishes_path_safe_bundle_and_work_orders(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            request_path = root / "request.json"
            history_path = root / "history.jsonl"
            request_path.write_text(json.dumps(_request()), encoding="utf-8")
            history_path.write_text("".join(json.dumps(row) + "\n" for row in _history()), encoding="utf-8")
            manifest = publish_cpbe_plan(
                request_path=request_path,
                history_path=history_path,
                output_root=root / "plan",
            )
            self.assertEqual(manifest["state"], "ready")
            self.assertTrue((root / "plan/cpbe-plan.json").is_file())
            self.assertEqual(len(list((root / "plan/work-orders").glob("*.json"))), 3)


class CPBESettlementTests(unittest.TestCase):
    def _plan(self) -> dict[str, object]:
        request = _request()
        request["context"]["max_canaries"] = 1
        return build_cpbe_plan(request=request, history=_history())

    @staticmethod
    def _receipt(probe_id: str, stage: str, *, passed: bool = True, metrics: dict[str, float] | None = None):
        return {
            "schema_version": 1,
            "artifact_type": "verdiwm-cpbe-stage-receipt",
            "probe_id": probe_id,
            "stage": stage,
            "passed": passed,
            "metrics": metrics or {},
            "evidence_refs": [f"receipt://{probe_id}/{stage}"],
        }

    def test_admits_only_after_all_stages_and_measured_gain(self) -> None:
        plan = self._plan()
        probe_id = plan["selected_work_orders"][0]["probe_id"]
        receipts = [
            self._receipt(probe_id, "static"),
            self._receipt(probe_id, "offline"),
            self._receipt(
                probe_id,
                "canary",
                metrics={"locality_residual": 0.2, "redundancy_cosine": 0.4, "collision_separation": 0.3},
            ),
            self._receipt(probe_id, "expanded", metrics={"regret_reduction": 0.1, "coverage_gain": 0.0}),
        ]
        settlement = settle_cpbe_plan(plan=plan, receipts=receipts)
        self.assertEqual(settlement["state"], "settled")
        self.assertEqual(settlement["admitted_count"], 1)
        self.assertEqual(settlement["candidates"][0]["state"], "settled_admitted")

    def test_eliminates_nonlocal_canary(self) -> None:
        plan = self._plan()
        probe_id = plan["selected_work_orders"][0]["probe_id"]
        receipts = [
            self._receipt(probe_id, "static"),
            self._receipt(probe_id, "offline"),
            self._receipt(
                probe_id,
                "canary",
                metrics={"locality_residual": 0.8, "redundancy_cosine": 0.1, "collision_separation": 0.3},
            ),
        ]
        settlement = settle_cpbe_plan(plan=plan, receipts=receipts)
        candidate = settlement["candidates"][0]
        self.assertEqual(candidate["state"], "eliminated_canary")
        self.assertIn("locality_residual_exceeded", candidate["blockers"])

    def test_rejects_out_of_order_receipt(self) -> None:
        plan = self._plan()
        probe_id = plan["selected_work_orders"][0]["probe_id"]
        with self.assertRaisesRegex(CPBEError, "CPBE_STAGE_RECEIPT_OUT_OF_ORDER"):
            settle_cpbe_plan(
                plan=plan,
                receipts=[
                    self._receipt(
                        probe_id,
                        "canary",
                        metrics={"locality_residual": 0.2, "redundancy_cosine": 0.2, "collision_separation": 0.2},
                    )
                ],
            )

    def test_live_cli_settlement_requires_hash_verified_artifacts(self) -> None:
        plan = self._plan()
        plan["evidence_class"] = "live"
        probe_id = plan["selected_work_orders"][0]["probe_id"]
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan_path = root / "plan.json"
            receipts_path = root / "receipts.jsonl"
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            receipts_path.write_text(
                json.dumps(self._receipt(probe_id, "static")) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(CPBEError, "LIVE_RECEIPT_ARTIFACTS_REQUIRED"):
                publish_cpbe_settlement(
                    plan_path=plan_path,
                    receipts_path=receipts_path,
                    output_root=root / "settlement",
                )

    def test_live_cli_settlement_verifies_bound_artifact_bytes(self) -> None:
        plan = self._plan()
        plan["evidence_class"] = "live"
        probe_id = plan["selected_work_orders"][0]["probe_id"]
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            evidence = root / "evidence.json"
            payload = b'{"measured":true}\n'
            evidence.write_bytes(payload)
            receipt = self._receipt(probe_id, "static")
            receipt["evidence_artifacts"] = [
                {
                    "path": evidence.name,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "size_bytes": len(payload),
                }
            ]
            plan_path = root / "plan.json"
            receipts_path = root / "receipts.jsonl"
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            receipts_path.write_text(json.dumps(receipt) + "\n", encoding="utf-8")

            manifest = publish_cpbe_settlement(
                plan_path=plan_path,
                receipts_path=receipts_path,
                output_root=root / "settlement",
            )
            self.assertEqual(manifest["state"], "partial")

            receipt["evidence_artifacts"][0]["sha256"] = "0" * 64
            receipts_path.write_text(json.dumps(receipt) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(CPBEError, "LIVE_RECEIPT_ARTIFACT_SHA256_MISMATCH"):
                publish_cpbe_settlement(
                    plan_path=plan_path,
                    receipts_path=receipts_path,
                    output_root=root / "tampered",
                )


if __name__ == "__main__":
    unittest.main()
