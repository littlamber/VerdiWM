from __future__ import annotations

import json
import hashlib
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts.export.acwm_formal_visualization import (
    AcwmFormalVisualizationError,
    _eval_command,
    _checkpoint_transform_provenance,
    _load_candidate_runtime_hook_receipt,
    _official_quality_gate,
    _parse_eval_results,
    rank_baseline_hard_cases,
    run_export,
)
from wmloop.runtime_contract import runtime_tree_sha256


class AcwmFormalVisualizationTests(unittest.TestCase):
    def test_checkpoint_delta_provenance_verifies_source_gate_and_scaled_sha(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = root / "alpha_0.25.pt"
            candidate.write_bytes(b"scaled")
            candidate_sha = hashlib.sha256(candidate.read_bytes()).hexdigest()
            source_candidate_sha = "a" * 64
            source_baseline_sha = "b" * 64
            transform = root / "transform.json"
            transform.write_text(
                json.dumps(
                    {
                        "artifact_type": "wmloop-checkpoint-delta-scaling",
                        "state": "ready",
                        "environment": "reacher",
                        "source_primitive": "next_forcing",
                        "seed": 41,
                        "candidate_sha256": source_candidate_sha,
                        "baseline_sha256": source_baseline_sha,
                        "rule": "theta_scaled = theta_baseline + alpha * delta",
                        "outputs": [
                            {
                                "alpha": 0.25,
                                "path": str(candidate),
                                "sha256": candidate_sha,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            source_gate = root / "source-gate.json"
            source_gate.write_text(
                json.dumps(
                    {
                        "artifact_type": "wmloop-acwm-formal-visualization-export",
                        "state": "ready",
                        "environment": "reacher",
                        "primitive": "next_forcing",
                        "seed": 41,
                        "candidate_checkpoint_sha256": source_candidate_sha,
                        "baseline_checkpoint_sha256": source_baseline_sha,
                        "official_quality_gate": {"pass": False},
                    }
                ),
                encoding="utf-8",
            )

            provenance = _checkpoint_transform_provenance(
                primitive="checkpoint_delta_scaling",
                environment="reacher",
                seed=42,
                candidate_checkpoint=candidate,
                transform_manifest=transform,
                delta_alpha=0.25,
                source_primitive="next_forcing",
                source_official_gate_manifest=source_gate,
                dry_run=False,
            )

            assert provenance is not None
            self.assertEqual(provenance["state"], "verified")
            self.assertEqual(provenance["alpha"], 0.25)
            self.assertEqual(provenance["source_seed"], 41)
            self.assertEqual(provenance["eval_seed"], 42)
            self.assertEqual(provenance["scaled_checkpoint_sha256"], candidate_sha)

    def test_candidate_eval_wrapper_records_actual_hook_invocation(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime_root = root / "runtime"
            hook_root = runtime_root / "acwm" / "wmloop_hooks"
            hook_root.mkdir(parents=True)
            (runtime_root / "numpy.py").write_text(
                "class _Random:\n    @staticmethod\n    def seed(value):\n        pass\nrandom = _Random()\n",
                encoding="utf-8",
            )
            (runtime_root / "torch.py").write_text(
                "def manual_seed(value):\n    pass\n"
                "class _Cuda:\n    @staticmethod\n    def manual_seed_all(value):\n        pass\n"
                "cuda = _Cuda()\n",
                encoding="utf-8",
            )
            (runtime_root / "acwm" / "__init__.py").write_text("", encoding="utf-8")
            (hook_root / "__init__.py").write_text("", encoding="utf-8")
            (hook_root / "demo.py").write_text(
                "def apply_demo(value):\n    return value + 1\n",
                encoding="utf-8",
            )
            sidecar_root = runtime_root / "wmloop_interventions"
            sidecar_root.mkdir()
            (sidecar_root / "demo.json").write_text(
                json.dumps(
                    {
                        "runtime_hook": {
                            "module": "acwm.wmloop_hooks.demo",
                            "function": "apply_demo",
                        }
                    }
                ),
                encoding="utf-8",
            )
            (runtime_root / "wmloop-runtime-manifest.json").write_text(
                json.dumps({"rendered_primitives": [{"name": "demo"}]}),
                encoding="utf-8",
            )
            fake_eval = root / "fake_eval.py"
            fake_eval.write_text(
                "from acwm.wmloop_hooks.demo import apply_demo\n"
                "assert apply_demo(2) == 3\n",
                encoding="utf-8",
            )
            receipt_path = root / "hook-receipts.jsonl"
            command = _eval_command(
                runtime=Path(sys.executable),
                env_arg="push_rope",
                config_path=root / "config.yaml",
                checkpoint_path=root / "checkpoint.pt",
                output_root=root / "output",
                runtime_root=runtime_root,
                steps=1,
                split="ind_test",
                max_trajs=1,
                max_saved_vids=1,
                batch_size=1,
                num_workers=0,
                test_cuts=1,
                eval_seed=7,
                hook_receipt_path=receipt_path,
            )
            command[6] = str(fake_eval)

            completed = subprocess.run(command, cwd=runtime_root, check=False, capture_output=True, text=True)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            receipt = _load_candidate_runtime_hook_receipt(receipt_path, expected_primitives={"demo"})
            self.assertEqual(receipt["call_count"], 1)

    def test_candidate_runtime_hook_receipt_requires_every_rendered_primitive(self) -> None:
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "receipts.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "primitive": "latent_spatial_memory",
                        "module": "acwm.wmloop_hooks.latent_spatial_memory",
                        "function": "apply_latent_spatial_memory",
                        "pid": 1,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            receipt = _load_candidate_runtime_hook_receipt(
                path,
                expected_primitives={"latent_spatial_memory"},
            )

            self.assertEqual(receipt["call_count"], 1)
            self.assertEqual(receipt["observed_primitives"], ["latent_spatial_memory"])

    def test_candidate_runtime_hook_receipt_fails_when_hook_was_not_called(self) -> None:
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "missing.jsonl"
            with self.assertRaisesRegex(AcwmFormalVisualizationError, "RUNTIME_HOOK_NOT_INVOKED"):
                _load_candidate_runtime_hook_receipt(
                    path,
                    expected_primitives={"latent_spatial_memory"},
                )

    def test_candidate_runtime_contract_checks_shared_tree_hash(self) -> None:
        from scripts.export.acwm_formal_visualization import _candidate_runtime_contract

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "model.py").write_text("model\n", encoding="utf-8")
            (root / "wmloop-runtime-manifest.json").write_text(
                json.dumps(
                    {
                        "artifact_type": "wmloop-materialized-candidate-runtime",
                        "state": "ready",
                        "tree_sha256": runtime_tree_sha256(root),
                    }
                ),
                encoding="utf-8",
            )
            contract = _candidate_runtime_contract(root)
            self.assertEqual(contract["tree_sha256"], runtime_tree_sha256(root))
    def test_hard_case_ranking_uses_baseline_error_not_candidate_gain(self) -> None:
        selected = rank_baseline_hard_cases(
            [
                {"sample_index": 0, "baseline_video_mse": 0.10, "candidate_video_mse": 0.01},
                {"sample_index": 1, "baseline_video_mse": 0.30, "candidate_video_mse": 0.29},
                {"sample_index": 2, "baseline_video_mse": 0.20, "candidate_video_mse": 0.50},
            ],
            top_k=2,
        )

        self.assertEqual([record["sample_index"] for record in selected], [1, 2])

    def test_official_results_gate_rejects_visual_protocol_regression(self) -> None:
        with TemporaryDirectory() as temporary:
            results = Path(temporary) / "results.md"
            results.write_text(
                "| Env | Split | Steps | MSE | Masked-MSE | PSNR | SSIM |\n"
                "|:--|:--|--:|--:|--:|--:|--:|\n"
                "| reacher | ind_test | 50 | 0.000857 | 0.022856 | 30.68 | 0.9863 |\n",
                encoding="utf-8",
            )
            candidate = _parse_eval_results(results, environment="reacher", split="ind_test")
            gate = _official_quality_gate(
                baseline={"steps": 50.0, "mse": 0.000274, "masked_mse": 0.006161, "psnr": 35.66, "ssim": 0.9917},
                candidate=candidate,
            )

            self.assertFalse(gate["pass"])
            self.assertEqual(gate["state"], "fail")
            self.assertAlmostEqual(gate["delta_candidate_minus_baseline"]["psnr"], -4.98)

    def test_official_results_accepts_clothmove_vendor_alias(self) -> None:
        with TemporaryDirectory() as temporary:
            results = Path(temporary) / "results.md"
            results.write_text(
                "| clothmove | ind_test | 50 | 0.001 | 0.010 | 30.0 | 0.98 |\n",
                encoding="utf-8",
            )

            metrics = _parse_eval_results(results, environment="cloth_move", split="ind_test")

            self.assertEqual(metrics["psnr"], 30.0)

    def test_dry_run_writes_official_eval_video_commands(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            vendor_root = _fake_vendor_root(root)
            runtime = root / "env" / "bin" / "python"
            runtime.parent.mkdir(parents=True)
            runtime.write_text("#!/bin/sh\n", encoding="utf-8")
            data_root = root / "data"
            data_root.mkdir()
            checkpoint_root = root / "checkpoints"
            official = checkpoint_root / "VideoDiT_S_robot_arm_240x240" / "latest.pt"
            official.parent.mkdir(parents=True)
            official.write_bytes(b"official")
            (checkpoint_root / "Wan2.1_VAE.pth").write_bytes(b"vae")
            candidate = root / "candidate" / "latest.pt"
            candidate.parent.mkdir()
            candidate.write_bytes(b"candidate")
            candidate_runtime = root / "candidate-runtime"
            candidate_runtime.mkdir()
            dataset_freeze, heldout_protocol = _fake_protocol_files(root)

            with patch("scripts.export.acwm_formal_visualization.VENDOR_ROOT", vendor_root):
                manifest = run_export(
                    output_root=root / "visual-export",
                    training_seed=4101,
                    runtime_python=runtime,
                    data_root=data_root,
                    checkpoint_root=checkpoint_root,
                    dataset_freeze=dataset_freeze,
                    heldout_protocol=heldout_protocol,
                    candidate_checkpoint=candidate,
                    candidate_runtime_root=candidate_runtime,
                    require_candidate_runtime_hook=True,
                    dry_run=True,
                    max_trajs=1,
                    max_saved_vids=1,
                )

            stored = json.loads((root / "visual-export" / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["state"], "planned")
            self.assertEqual(stored["environment"], "robot_arm")
            self.assertIn("--save_videos", stored["baseline_command"])
            self.assertEqual(stored["eval_seed"], 621)
            self.assertEqual(stored["training_seed"], 4101)
            self.assertEqual(stored["paired_randomness_policy"], "common Python, NumPy, Torch, and CUDA seed for baseline and candidate")
            self.assertIn("621", stored["baseline_command"])
            self.assertIn("621", stored["candidate_command"])
            self.assertEqual(stored["candidate_checkpoint"], str(candidate.resolve()))
            self.assertEqual(stored["candidate_runtime_root"], str(candidate_runtime.resolve()))
            self.assertEqual(len(stored["baseline_runtime_sha256"]), 64)
            self.assertEqual(len(stored["candidate_runtime_sha256"]), 64)
            self.assertIn(str(candidate_runtime.resolve()), stored["candidate_command"])
            self.assertEqual(stored["baseline_command"][5], "")
            self.assertIn(str((root / "visual-export" / "logs" / "candidate-runtime-hook-receipts.jsonl")), stored["candidate_command"])
            self.assertIn(str(vendor_root / "eval.py"), stored["candidate_command"])
            self.assertEqual(len(stored["candidate_checkpoint_sha256"]), 64)
            self.assertEqual(len(stored["baseline_checkpoint_sha256"]), 64)
            provenance = stored["protocol_provenance"]
            self.assertEqual(len(provenance["eval_script_sha256"]), 64)
            self.assertEqual(len(provenance["eval_config_sha256"]), 64)
            self.assertEqual(len(provenance["dataset_freeze_sha256"]), 64)
            self.assertEqual(len(provenance["heldout_protocol_sha256"]), 64)
            self.assertEqual(stored["eval_config_path"], str(root / "visual-export" / "robot_arm-eval-config.yaml"))

    def test_dry_run_can_plan_before_candidate_checkpoint_exists(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            vendor_root = _fake_vendor_root(root)
            runtime = root / "env" / "bin" / "python"
            runtime.parent.mkdir(parents=True)
            runtime.write_text("#!/bin/sh\n", encoding="utf-8")
            data_root = root / "data"
            data_root.mkdir()
            checkpoint_root = root / "checkpoints"
            official = checkpoint_root / "VideoDiT_S_robot_arm_240x240" / "latest.pt"
            official.parent.mkdir(parents=True)
            official.write_bytes(b"official")
            (checkpoint_root / "Wan2.1_VAE.pth").write_bytes(b"vae")
            candidate = root / "candidate" / "latest.pt"
            dataset_freeze, heldout_protocol = _fake_protocol_files(root)

            with patch("scripts.export.acwm_formal_visualization.VENDOR_ROOT", vendor_root):
                manifest = run_export(
                    output_root=root / "visual-export",
                    runtime_python=runtime,
                    data_root=data_root,
                    checkpoint_root=checkpoint_root,
                    dataset_freeze=dataset_freeze,
                    heldout_protocol=heldout_protocol,
                    candidate_checkpoint=candidate,
                    dry_run=True,
                    max_trajs=1,
                    max_saved_vids=1,
                )

            self.assertEqual(manifest["state"], "planned")
            self.assertEqual(manifest["candidate_checkpoint"], str(candidate.resolve()))


def _fake_vendor_root(root: Path) -> Path:
    vendor = root / "vendor" / "ACWM-Phys"
    (vendor / "configs/envs").mkdir(parents=True)
    (vendor / "configs/model").mkdir(parents=True)
    (vendor / "eval.py").write_text("# synthetic test evaluator\n", encoding="utf-8")
    (vendor / "configs/envs/robot_arm.yaml").write_text(
        "model_type: dit_s\nmodel_config: {}\n",
        encoding="utf-8",
    )
    (vendor / "configs/model/dit_s.yaml").write_text("model_config: {}\n", encoding="utf-8")
    return vendor


def _fake_protocol_files(root: Path) -> tuple[Path, Path]:
    dataset_freeze = root / "dataset-freeze.json"
    heldout_protocol = root / "heldout-protocol.json"
    dataset_freeze.write_text("{}\n", encoding="utf-8")
    heldout_protocol.write_text("{}\n", encoding="utf-8")
    return dataset_freeze, heldout_protocol


if __name__ == "__main__":
    unittest.main()
