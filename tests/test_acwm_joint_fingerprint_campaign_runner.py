from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts.run_acwm_joint_fingerprint_campaign import run


ROOT = Path(__file__).resolve().parents[1]


class AcwmJointFingerprintCampaignRunnerTests(unittest.TestCase):
    def test_rejects_duplicate_gpu_ids_before_spawning_workers(self) -> None:
        with TemporaryDirectory() as temp:
            args = Namespace(
                joint_campaign=ROOT / "configs/experiments/acwm_phys_joint_irg_autoregressive_pilot_v1.json",
                protocol="smoke",
                runtime_python=Path("/bin/true"),
                gpu=[0, 0],
                environment=["push_cube"],
                vendor_root=ROOT / "vendor/ACWM-Phys",
                data_root=ROOT,
                checkpoint_root=ROOT,
                vae_path=ROOT / "README.md",
                output_root=Path(temp) / "output",
            )
            with self.assertRaisesRegex(ValueError, "JOINT_FINGERPRINT_GPU_SET_INVALID"):
                run(args)

    def test_dynamic_queue_reuses_first_available_gpu(self) -> None:
        calls: list[tuple[str, str]] = []

        class ImmediateProcess:
            returncode = 0

        def fake_run(command, **_kwargs):
            environment = command[command.index("--environment") + 1]
            gpu = command[command.index("--physical-gpu") + 1]
            calls.append((environment, gpu))
            return ImmediateProcess()

        with TemporaryDirectory() as temp, patch(
            "scripts.run_acwm_joint_fingerprint_campaign.subprocess.run",
            side_effect=fake_run,
        ):
            args = Namespace(
                joint_campaign=ROOT / "configs/experiments/acwm_phys_joint_irg_autoregressive_pilot_v1.json",
                protocol="smoke",
                runtime_python=Path("/bin/true"),
                gpu=[0, 1, 2],
                environment=["push_cube", "stack_cube", "push_rope", "cloth_move"],
                vendor_root=ROOT / "vendor/ACWM-Phys",
                data_root=ROOT,
                checkpoint_root=ROOT,
                vae_path=ROOT / "README.md",
                output_root=Path(temp) / "output",
            )
            status = run(args)

        self.assertEqual(status["state"], "ready")
        self.assertEqual({environment for environment, _gpu in calls}, set(args.environment))
        self.assertEqual(len(calls), 4)


if __name__ == "__main__":
    unittest.main()
