from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.export.cosmos3_forward_dynamics_public_bundle import (
    export_cosmos3_forward_dynamics_public_bundle,
)


class Cosmos3ForwardDynamicsPublicBundleTests(unittest.TestCase):
    def test_export_is_path_safe_and_preserves_claim_boundary(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            smoke = root / "smoke"
            audit = root / "audit"
            capability = root / "capability"
            smoke.mkdir()
            audit.mkdir()
            capability.mkdir()
            gpu_runtime = root / "gpu-runtime"
            gpu_runtime.mkdir()
            (smoke / "smoke.json").write_text(json.dumps(_smoke()))
            (audit / "backbone-instantiation.json").write_text(json.dumps(_audit()))
            (capability / "backbone-capability-matrix.json").write_text(json.dumps(_capability()))
            (gpu_runtime / "runtime-receipt.json").write_text(json.dumps(_gpu_runtime()))
            output = root / "public"
            bundle = export_cosmos3_forward_dynamics_public_bundle(
                smoke_root=smoke,
                instance_audit_root=audit,
                capability_root=capability,
                output_root=output,
                gpu_runtime_root=gpu_runtime,
            )
            self.assertFalse(bundle["formal_launch_allowed"])
            self.assertFalse(bundle["model_quality_evidence_included"])
            self.assertEqual(bundle["gpu_runtime_state"], "ready")
            self.assertIn("Runtime proof only", bundle["claim_boundary"])
            self.assertTrue((output / "gpu-runtime-summary.json").is_file())
            self.assertTrue((output / "MANIFEST.sha256").is_file())
            for path in output.rglob("*"):
                if path.is_file():
                    text = path.read_text(encoding="utf-8")
                    self.assertNotIn("/" + "mnt" + "/", text)


def _smoke() -> dict[str, object]:
    return {
        "state": "ready",
        "model_family": "cosmos3",
        "model_mode": "forward_dynamics",
        "split": "dev",
        "identity": {"sample_index": 0, "seed": 101},
        "action_shape": [16, 10],
        "dataset_audit": {"state": "verified", "file_count": 7},
        "hook_audit": {"available_hooks": ["H1", "H2", "H3", "H4", "H5"]},
        "zero_dose_receipt": {"zero_dose_byte_identity": True},
        "action_dimension_balancing_receipt": {"shape": [16, 10]},
        "side_effects": {"gpu_execution_started": False},
        "claim_boundary": "CPU smoke only; no model-quality or transfer claim.",
    }


def _audit() -> dict[str, object]:
    return {
        "instance_id": "cosmos3-v1",
        "closed_loop_instance_ready": True,
        "formal_verdict_instance_ready": True,
        "instance_formal_launch_allowed": False,
    }


def _capability() -> dict[str, object]:
    return {
        "instance_id": "cosmos3-v1",
        "state": "pilot_draft",
        "campaign_state": "pilot_draft",
        "capability_summary": {
            "available_hooks": ["H1", "H2", "H3", "H4", "H5"],
            "primitive_count": 17,
            "eligible_primitive_count": 1,
            "blocked_primitive_count": 16,
        },
        "primitive_matrix": [
            {"primitive": "action_dimension_balancing", "status": "eligible_for_instance_canary"}
        ],
    }


def _gpu_runtime() -> dict[str, object]:
    return {
        "state": "ready",
        "model_mode": "forward_dynamics",
        "identity": {"sample_index": 0, "seed": 101},
        "action_shape": [16, 10],
        "physical_gpu": {"observed_gpu_uuids": ["GPU-test"], "active_sample_count": 2},
        "video": {"frame_count": 17, "duration_seconds": 1.134},
        "runtime_seconds": 34.7,
        "sha256": {"action_input": "a", "sample_outputs": "b", "vision_video": "c"},
        "claim_boundary": "Runtime proof only; no quality claim.",
    }


if __name__ == "__main__":
    unittest.main()
