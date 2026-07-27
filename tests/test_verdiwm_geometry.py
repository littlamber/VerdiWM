from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from wmloop.geometry import (
    AtlasPoint,
    CapabilityProfile,
    EffectContext,
    EffectMemory,
    EffectRecord,
    GeometryValidationError,
    InterventionDescriptor,
    ProbeCandidate,
    compile_intervention,
    detect_repair_collisions,
    estimate_response_chart,
    evaluate_transfer_certificate,
    irg_distance,
    rank_probe_candidates,
)
from wmloop.geometry.evolution import EffectEstimate


class TypedInterventionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.capabilities = CapabilityProfile(
            backbone_family="acwm_phys",
            capability_class="latent_dit_action_conditioned",
            capabilities=frozenset({"sampler_callback", "paired_seed_control"}),
            hook_types=frozenset({"H4"}),
        )
        self.descriptor = InterventionDescriptor(
            name="action_scale_probe",
            kind="probe_path",
            hook_type="H4",
            transformation="multiply action conditioning by 1+d",
            scope="inference_only",
            dose_unit="relative_action_scale",
            schedule="constant",
            preconditions=("action conditioning is exposed",),
            invariants=("same_seed", "same_context"),
            prediction="action-following response changes monotonically near zero",
            required_capabilities=frozenset({"sampler_callback", "paired_seed_control"}),
            inference_only=True,
            reversible=True,
        )

    def test_compile_passes_only_exact_semantic_contract(self) -> None:
        receipt = compile_intervention(
            self.descriptor,
            self.capabilities,
            invariant_checks={"same_seed": True, "same_context": True},
            dose_direction=0.1,
        )
        self.assertTrue(receipt.compiled)
        self.assertEqual(receipt.blockers, ())

        blocked = compile_intervention(
            self.descriptor,
            CapabilityProfile(
                backbone_family="other",
                capability_class="unknown",
                capabilities=frozenset(),
                hook_types=frozenset(),
            ),
            invariant_checks={"same_seed": True},
            dose_direction=0.1,
        )
        self.assertFalse(blocked.compiled)
        self.assertIn("hook_unavailable:H4", blocked.blockers)
        self.assertIn("invariant_unchecked:same_context", blocked.blockers)


class ResponseGeometryTests(unittest.TestCase):
    def _chart(self, scale: float = 1.0):
        return estimate_response_chart(
            chart_id=f"chart-{scale}",
            goal_schema="pixel-and-action-v1",
            outcome_names=("psnr", "action_following"),
            outcome_weights=(1.0, 4.0),
            baseline_repeats=((10.0, 0.5), (10.2, 0.55)),
            dose_observations={
                "action_scale": {
                    -0.1: ((10.0 - 0.2 * scale, 0.5 - 0.1 * scale), (10.2 - 0.2 * scale, 0.55 - 0.1 * scale)),
                    0.1: ((10.0 + 0.2 * scale, 0.5 + 0.1 * scale), (10.2 + 0.2 * scale, 0.55 + 0.1 * scale)),
                },
                "context_retention": {
                    0.1: ((10.0 + 0.1 * scale, 0.5), (10.2 + 0.1 * scale, 0.55)),
                    0.2: ((10.0 + 0.2 * scale, 0.5), (10.2 + 0.2 * scale, 0.55)),
                },
            },
        )

    def test_estimates_central_and_one_sided_directions(self) -> None:
        chart = self._chart()
        self.assertEqual(chart.intervention_names, ("action_scale", "context_retention"))
        self.assertAlmostEqual(chart.jacobian[0][0], 2.0)
        self.assertAlmostEqual(chart.jacobian[1][0], 1.0)
        self.assertAlmostEqual(chart.jacobian[0][1], 1.0)
        self.assertAlmostEqual(chart.locality_residuals["context_retention"], 0.0, places=8)
        self.assertEqual(chart.repeat_count, 2)

    def test_distance_is_zero_for_identical_response(self) -> None:
        left = self._chart()
        right = self._chart()
        self.assertAlmostEqual(irg_distance(left, right), 0.0)
        self.assertGreater(irg_distance(left, self._chart(1.5)), 0.0)


class TransferAndMemoryTests(unittest.TestCase):
    def _receipt(self):
        descriptor = InterventionDescriptor(
            name="anchor",
            kind="repair",
            hook_type="H4",
            transformation="blend first-frame latent",
            scope="inference",
            dose_unit="blend_fraction",
            schedule="constant",
            preconditions=("first frame available",),
            invariants=("same_split",),
            prediction="late appearance drift decreases",
        )
        return compile_intervention(
            descriptor,
            CapabilityProfile("acwm_phys", "dit", frozenset(), frozenset({"H4"})),
            invariant_checks={"same_split": True},
            dose_direction=0.2,
        )

    def test_certificate_licenses_or_abstains_term_by_term(self) -> None:
        certificate = evaluate_transfer_certificate(
            receipt=self._receipt(),
            support_overlap=0.8,
            effective_sample_size=12,
            alignment_error=0.1,
            sign_agreement=0.9,
            calibrated_lower_bound=0.3,
            minimum_overlap=0.5,
            minimum_effective_sample_size=8,
            maximum_alignment_error=0.2,
            minimum_sign_agreement=0.8,
            goal_threshold=0.1,
        )
        self.assertTrue(certificate.licensed)

        abstained = evaluate_transfer_certificate(
            receipt=self._receipt(),
            support_overlap=0.2,
            effective_sample_size=12,
            alignment_error=0.1,
            sign_agreement=0.9,
            calibrated_lower_bound=0.3,
            minimum_overlap=0.5,
            minimum_effective_sample_size=8,
            maximum_alignment_error=0.2,
            minimum_sign_agreement=0.8,
            goal_threshold=0.1,
        )
        self.assertFalse(abstained.licensed)
        self.assertEqual(abstained.abstention_reasons, ("overlap",))

    def test_memory_retains_confirmed_and_rejected_effects(self) -> None:
        context = EffectContext(
            campaign_id="cloth-next-forcing-s2802",
            backbone_family="acwm_phys",
            capability_class="latent_dit_action_conditioned",
            goal_schema="g1",
            outcome_schema="official-pixel-gate-v1",
            chart_id="cloth-local-chart",
            data_regime="ind_test",
            horizons=(16, 32, 48),
        )
        confirmed = EffectRecord(
            record_id="confirmed-1",
            primitive="next_forcing",
            context=context,
            status="confirmed",
            mean_effect=0.72,
            standard_error=0.12,
            lower_bound=0.48,
            goal_threshold=0.0,
            validity_gates={"psnr": True, "ssim": True, "mse": True, "masked_mse": True},
            replication_count=2,
            evidence_refs=("official-gate.json", "confirmation-gate.json"),
        )
        rejected = EffectRecord(
            record_id="rejected-1",
            primitive="mixture_reweight",
            context=context,
            status="rejected",
            mean_effect=-0.4,
            standard_error=0.1,
            lower_bound=-0.6,
            goal_threshold=0.0,
            validity_gates={"psnr": False},
            replication_count=1,
            evidence_refs=("screen.json",),
        )
        memory = EffectMemory((confirmed, rejected))
        self.assertEqual(memory.query(status="confirmed"), (confirmed,))
        self.assertEqual(memory.query(primitive="mixture_reweight"), (rejected,))
        with tempfile.TemporaryDirectory() as temp:
            path = memory.write_jsonl(Path(temp) / "effects.jsonl")
            self.assertEqual(len(path.read_text().splitlines()), 2)

        with self.assertRaises(GeometryValidationError):
            EffectRecord(
                record_id="bad",
                primitive="next_forcing",
                context=context,
                status="confirmed",
                mean_effect=0.2,
                standard_error=0.1,
                lower_bound=-0.1,
                goal_threshold=0.0,
                validity_gates={"psnr": True},
                replication_count=2,
                evidence_refs=("gate.json",),
            )


class AtlasEvolutionTests(unittest.TestCase):
    def test_detects_confident_opposing_effects_and_ranks_probe_lcb(self) -> None:
        points = (
            AtlasPoint(
                "soft-body",
                "shared",
                (0.0, 0.0),
                {"anchor": EffectEstimate(0.8, 0.4, 1.2, 0.01)},
            ),
            AtlasPoint(
                "rigid-body",
                "shared",
                (0.1, 0.1),
                {"anchor": EffectEstimate(-0.7, -1.0, -0.3, 0.02)},
            ),
        )
        collisions = detect_repair_collisions(
            points,
            distance_threshold=0.5,
            minimum_effect=0.1,
            fdr_alpha=0.05,
        )
        self.assertEqual(len(collisions), 1)
        self.assertEqual(collisions[0].primitive, "anchor")

        ranked = rank_probe_candidates(
            (
                ProbeCandidate("contact", 0.5, 0.1, 2.0, 0.0, True),
                ProbeCandidate("appearance", 0.1, 0.2, 0.5, 0.0, True),
            )
        )
        self.assertEqual(ranked[0]["name"], "contact")
        self.assertTrue(ranked[0]["promoted"])
        self.assertFalse(ranked[1]["promoted"])


if __name__ == "__main__":
    unittest.main()
