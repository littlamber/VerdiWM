from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from scripts.aggregate_ctrl_world_local_fingerprint import (
    LocalFingerprintAggregationError,
    aggregate,
)
from scripts.merge_ctrl_world_local_fingerprint_shards import (
    LocalFingerprintShardMergeError,
    merge,
)
from scripts.run_ctrl_world_local_fingerprint_probe import (
    FSHCInteractionLocalGain,
    FSHCSignedHistoryGain,
    _prediction_response,
)


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "configs" / "experiments" / "ctrl_world_context_local_fingerprint_v1.json"
OUTCOMES = (
    "negative_mean_l1",
    "negative_final_interaction_l1",
    "negative_horizon_l1_slope",
    "mean_psnr",
    "final_psnr",
)
PROBES = (
    "action_conditioning_scale",
    "history_retention_gain",
    "first_frame_anchor_blend",
    "sampler_initial_noise_gain",
)


class CtrlWorldLocalFingerprintTests(unittest.TestCase):
    def _result(self, *, probe_id: str, seed_count: int = 3, zero_offset: float = 0.0) -> dict[str, object]:
        measurements = []
        for dose in (-0.025, 0.0, 0.025):
            for seed in (101, 202, 303)[:seed_count]:
                base = 0.01 * seed
                outcomes = {name: base + zero_offset for name in OUTCOMES}
                if dose:
                    outcomes = {
                        name: value + dose * (index + 1) for index, (name, value) in enumerate(outcomes.items())
                    }
                measurements.append(
                    {
                        "dose": dose,
                        "identity": {
                            "context_id": "ctx-1",
                            "episode_id": "899",
                            "start_idx": 8,
                            "seed": seed,
                        },
                        "outcomes": outcomes,
                    }
                )
        return {
            "artifact_type": "verdiwm-ctrl-world-local-fingerprint-probe-result",
            "state": "ready",
            "campaign_id": "ctrl_world_context_local_fingerprint_v1",
            "probe_id": probe_id,
            "base_probe_family": probe_id,
            "outcome_names": list(OUTCOMES),
            "zero_identity_checks": [{"state": "passed"}],
            "hook_activation": {"state": "passed"},
            "measurements": measurements,
        }

    def _write(self, root: Path, payload: dict[str, object], name: str) -> Path:
        path = root / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_aggregates_all_base_paths_with_seed_covariance(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            result_paths = [
                self._write(root, self._result(probe_id=probe), f"{probe}.json") for probe in PROBES
            ]
            manifest = aggregate(
                campaign_path=CAMPAIGN,
                result_paths=result_paths,
                output_root=root / "atlas",
            )
            self.assertEqual(manifest["context_count"], 1)
            report = json.loads((root / "atlas" / "context-local-fingerprint-atlas.json").read_text())
            chart = report["contexts"][0]["chart"]
            self.assertEqual(chart["repeat_count"], 3)
            self.assertEqual(set(chart["intervention_names"]), set(PROBES))
            self.assertEqual(report["routing_readiness"]["state"], "not_licensed")

    def test_rejects_single_seed_even_when_every_dose_is_present(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            result_paths = [
                self._write(root, self._result(probe_id=probe, seed_count=1), f"{probe}.json")
                for probe in PROBES
            ]
            with self.assertRaisesRegex(LocalFingerprintAggregationError, "SEED_REPEATS_INSUFFICIENT"):
                aggregate(campaign_path=CAMPAIGN, result_paths=result_paths, output_root=root / "atlas")

    def test_rejects_cross_probe_zero_dose_contamination(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            result_paths = []
            for probe in PROBES:
                offset = 1e-3 if probe == "sampler_initial_noise_gain" else 0.0
                result_paths.append(
                    self._write(root, self._result(probe_id=probe, zero_offset=offset), f"{probe}.json")
                )
            with self.assertRaisesRegex(LocalFingerprintAggregationError, "ZERO_DOSE_CROSS_PROBE_MISMATCH"):
                aggregate(campaign_path=CAMPAIGN, result_paths=result_paths, output_root=root / "atlas")

    def test_fshc_probe_uses_exact_corrector_override_and_restores_hook(self) -> None:
        class Corrector(nn.Module):
            def runtime_features(self, history, current, action, noise, *, rollout_consistency=None):
                del current, action, noise, rollout_consistency
                return torch.zeros(history.shape[0], history.shape[1], 21)

            def predict_gate(self, features):
                shape = features.shape[:2]
                return type(
                    "Gate",
                    (),
                    {
                        "signed_gain": torch.full(shape, 0.2),
                        "confidence": torch.full(shape, 0.8),
                        "use_probability": torch.full(shape, 0.5),
                    },
                )()

            def forward(self, history, current, action, noise, *, correction_gain_override=None):
                del action, noise
                gain = float(correction_gain_override or 0.0)
                features = torch.zeros(history.shape[0], history.shape[1], 21)
                return type(
                    "Output",
                    (),
                    {
                        "history": history + gain * (current[:, None] - history),
                        "runtime_features": features,
                    },
                )()

        model = type("Model", (), {"history_corrector": Corrector()})()
        history = torch.zeros(1, 2, 1, 2, 2)
        current = torch.ones(1, 1, 2, 2)
        action = torch.zeros(1, 3, 1)
        noise = torch.zeros(1)
        original = model.history_corrector.forward

        with FSHCSignedHistoryGain(model, 0.025) as probe:
            output = model.history_corrector(history, current, action, noise)
            self.assertGreater(float(output.history.abs().sum()), 0.0)
            self.assertGreater(probe.result()["maximum_abs_tensor_delta"], 0.0)
            self.assertEqual(len(probe.runtime_feature_records()), 1)
            self.assertEqual(len(probe.runtime_feature_records()[0]), 2)
            self.assertEqual(len(probe.learned_gate_records()), 1)
            for value in probe.learned_gate_records()[0]["signed_gain"]:
                self.assertAlmostEqual(value, 0.2)

        self.assertEqual(model.history_corrector.forward.__func__, original.__func__)

    def test_fshc_probe_scales_complete_multiscale_residual_symmetrically(self) -> None:
        class Corrector(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.multiscale_side_adapter = object()
                self.received_scale = None

            def runtime_features(self, history, current, action, noise, *, rollout_consistency=None):
                del current, action, noise, rollout_consistency
                return torch.zeros(history.shape[0], history.shape[1], 21)

            def predict_gate(self, features):
                values = torch.zeros(features.shape[:2])
                return type("Gate", (), {"signed_gain": values, "confidence": values, "use_probability": values})()

            def forward(self, history, current, action, noise, *, adapter_scale_override=None):
                del current, action, noise
                self.received_scale = adapter_scale_override
                residual = torch.full((history.shape[0] * history.shape[1], 1, 1, 1), 0.25)
                return type(
                    "Output",
                    (),
                    {
                        "history": history,
                        "runtime_features": torch.zeros(history.shape[0], history.shape[1], 21),
                        "down_block_additional_residuals": (residual, residual, residual, residual),
                        "mid_block_additional_residual": residual,
                    },
                )()

        model = type("Model", (), {"history_corrector": Corrector()})()
        history = torch.zeros(1, 2, 1, 2, 2)
        for dose in (-0.5, 0.0, 0.5):
            with FSHCSignedHistoryGain(model, dose, normalized_mechanism=True) as probe:
                output = model.history_corrector(
                    history,
                    torch.ones(1, 1, 2, 2),
                    torch.zeros(1, 3, 1),
                    torch.zeros(1),
                )
                self.assertIsNone(model.history_corrector.received_scale)
                torch.testing.assert_close(
                    output.mid_block_additional_residual,
                    torch.full_like(output.mid_block_additional_residual, 0.25 * dose),
                )
                if dose == 0.0:
                    self.assertEqual(probe.result()["maximum_abs_tensor_delta"], 0.0)
                else:
                    self.assertGreater(probe.result()["maximum_abs_tensor_delta"], 0.0)

    def test_fshc_interaction_local_probe_activates_only_selected_interaction(self) -> None:
        class Corrector(nn.Module):
            def runtime_features(self, history, current, action, noise, *, rollout_consistency=None):
                del current, action, noise, rollout_consistency
                return torch.zeros(history.shape[0], history.shape[1], 21)

            def predict_gate(self, features):
                values = torch.zeros(features.shape[:2])
                return type("Gate", (), {"signed_gain": values, "confidence": values, "use_probability": values})()

            def forward(self, history, current, action, noise, *, correction_gain_override=None):
                del action, noise
                gain = float(correction_gain_override or 0.0)
                return type(
                    "Output",
                    (),
                    {
                        "history": history + gain * (current[:, None] - history),
                        "runtime_features": torch.zeros(history.shape[0], history.shape[1], 21),
                    },
                )()

        model = type("Model", (), {"history_corrector": Corrector()})()
        history = torch.zeros(1, 2, 1, 2, 2)
        current = torch.ones(1, 1, 2, 2)
        action = torch.zeros(1, 3, 1)
        noise = torch.zeros(1)

        with FSHCInteractionLocalGain(model, 0.25, target_interaction=1) as probe:
            probe.set_interaction(0)
            inactive = model.history_corrector(history, current, action, noise)
            probe.set_interaction(1)
            active = model.history_corrector(history, current, action, noise)

        torch.testing.assert_close(inactive.history, history)
        self.assertGreater(float(active.history.abs().sum()), 0.0)
        self.assertEqual(probe.result()["active_invocation_count"], 1)
        self.assertEqual(probe.result()["target_interaction"], 1)

    def test_prediction_response_preserves_counterfactual_output_changes(self) -> None:
        baseline = torch.zeros(2, 3, 8, 8, 3, dtype=torch.uint8).numpy()
        changed = baseline.copy()
        changed[:, :, :4, :4, 0] = 255
        left = _prediction_response(baseline)
        right = _prediction_response(changed)
        self.assertEqual(left["shape"], [2, 3, 3, 4, 4])
        self.assertNotEqual(left["source_sha256"], right["source_sha256"])
        self.assertGreater(
            float(torch.tensor(right["values"]).sub(torch.tensor(left["values"])).abs().mean()),
            0.0,
        )

    def test_merges_complete_identity_shards_and_rejects_duplicate(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            contexts = {
                "artifact_type": "verdiwm-ctrl-world-local-context-set",
                "contexts": [
                    {"context_id": "ctx-1", "episode_id": "899", "start_idx": 8, "seeds": [101, 202]}
                ],
            }
            contexts_path = self._write(root, contexts, "contexts.json")
            checkpoint = self._write(root, {}, "checkpoint.pt")
            campaign = {
                "artifact_type": "verdiwm-ctrl-world-local-fingerprint-campaign",
                "campaign_id": "merge-test",
                "checkpoint": str(checkpoint),
                "protocol": {"interact_num": 4, "num_inference_steps": 2},
                "probe_paths": [{"probe_id": "fshc_signed_history_gain", "doses": [-0.025, 0.0, 0.025]}],
                "claim_scope": "test only",
            }
            campaign_path = self._write(root, campaign, "campaign.json")
            context_hash = __import__("hashlib").sha256(contexts_path.read_bytes()).hexdigest()

            def shard(seed: int) -> dict[str, object]:
                identity = {"context_id": "ctx-1", "episode_id": "899", "start_idx": 8, "seed": seed}
                return {
                    "artifact_type": "verdiwm-ctrl-world-local-fingerprint-probe-result",
                    "state": "ready",
                    "campaign_id": "merge-test",
                    "probe_id": "fshc_signed_history_gain",
                    "base_probe_family": "trainable_signed_history_correction",
                    "outcome_names": list(OUTCOMES),
                    "input": {
                        "checkpoint": str(checkpoint),
                        "contexts_sha256": context_hash,
                        "doses": [0.0, -0.025, 0.025],
                        "interact_num": 4,
                        "num_inference_steps": 2,
                    },
                    "runtime": {"cuda_visible_devices": str(seed)},
                    "unwrapped_references": [{"identity": identity}],
                    "measurements": [
                        {"identity": identity, "dose": dose, "outcomes": {name: 0.0 for name in OUTCOMES}}
                        for dose in (0.0, -0.025, 0.025)
                    ],
                    "zero_identity_checks": [{"identity": identity, "state": "passed"}],
                    "hook_activation": {"state": "passed"},
                    "artifacts": {"zero_dose_rollout": "zero-dose-rollout.mp4"},
                }

            paths = []
            for seed in (101, 202):
                shard_root = root / str(seed)
                shard_root.mkdir()
                (shard_root / "zero-dose-rollout.mp4").write_bytes(b"video")
                paths.append(self._write(shard_root, shard(seed), "result.json"))
            manifest = merge(
                campaign_path=campaign_path,
                contexts_path=contexts_path,
                shard_paths=paths,
                output_root=root / "merged",
            )
            self.assertEqual(manifest["identity_count"], 2)
            self.assertEqual(manifest["measurement_count"], 6)
            with self.assertRaisesRegex(LocalFingerprintShardMergeError, "IDENTITY_DUPLICATE"):
                merge(
                    campaign_path=campaign_path,
                    contexts_path=contexts_path,
                    shard_paths=[paths[0], paths[0]],
                    output_root=root / "duplicate",
                )

    def test_merges_multi_identity_shard(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            seeds = (101, 202, 303)
            contexts = {
                "artifact_type": "verdiwm-ctrl-world-local-context-set",
                "contexts": [
                    {"context_id": "ctx-1", "episode_id": "899", "start_idx": 8, "seeds": list(seeds)}
                ],
            }
            contexts_path = self._write(root, contexts, "contexts.json")
            checkpoint = self._write(root, {}, "checkpoint.pt")
            campaign = {
                "artifact_type": "verdiwm-ctrl-world-local-fingerprint-campaign",
                "campaign_id": "multi-identity-merge-test",
                "checkpoint": str(checkpoint),
                "protocol": {"interact_num": 4, "num_inference_steps": 2},
                "probe_paths": [{"probe_id": "fshc_signed_history_gain", "doses": [-0.025, 0.0, 0.025]}],
                "claim_scope": "test only",
            }
            campaign_path = self._write(root, campaign, "campaign.json")
            context_hash = __import__("hashlib").sha256(contexts_path.read_bytes()).hexdigest()
            identities = [
                {"context_id": "ctx-1", "episode_id": "899", "start_idx": 8, "seed": seed}
                for seed in seeds
            ]
            shard = {
                "artifact_type": "verdiwm-ctrl-world-local-fingerprint-probe-result",
                "state": "ready",
                "campaign_id": "multi-identity-merge-test",
                "probe_id": "fshc_signed_history_gain",
                "base_probe_family": "trainable_signed_history_correction",
                "outcome_names": list(OUTCOMES),
                "input": {
                    "checkpoint": str(checkpoint),
                    "contexts_sha256": context_hash,
                    "doses": [0.0, -0.025, 0.025],
                    "interact_num": 4,
                    "num_inference_steps": 2,
                },
                "runtime": {"cuda_visible_devices": "0"},
                "unwrapped_references": [{"identity": identity} for identity in identities],
                "measurements": [
                    {"identity": identity, "dose": dose, "outcomes": {name: 0.0 for name in OUTCOMES}}
                    for identity in identities
                    for dose in (0.0, -0.025, 0.025)
                ],
                "zero_identity_checks": [
                    {"identity": identity, "state": "passed"} for identity in identities
                ],
                "hook_activation": {"state": "passed"},
                "artifacts": {"zero_dose_rollout": "zero-dose-rollout.mp4"},
            }
            shard_root = root / "multi"
            shard_root.mkdir()
            (shard_root / "zero-dose-rollout.mp4").write_bytes(b"video")
            shard_path = self._write(shard_root, shard, "result.json")

            manifest = merge(
                campaign_path=campaign_path,
                contexts_path=contexts_path,
                shard_paths=[shard_path],
                output_root=root / "merged",
            )

            self.assertEqual(manifest["identity_count"], 3)
            self.assertEqual(manifest["measurement_count"], 9)
            result = json.loads((root / "merged" / "result.json").read_text())
            receipt = result["input"]["shard_receipts"][0]
            self.assertNotIn("identity", receipt)
            self.assertEqual(receipt["identities"], [
                {"context_id": "ctx-1", "seed": seed} for seed in seeds
            ])


if __name__ == "__main__":
    unittest.main()
