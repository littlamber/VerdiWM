from __future__ import annotations

import unittest

from scripts.run_acwm_fingerprint_campaign import build_gpu_assignments


class AcwmFingerprintCampaignRunnerTests(unittest.TestCase):
    def test_assigns_one_serial_environment_queue_per_physical_gpu(self) -> None:
        environments = [f"env_{index}" for index in range(8)]
        assignments = build_gpu_assignments(environments, [0, 1, 2])

        self.assertEqual(assignments[0], ["env_0", "env_3", "env_6"])
        self.assertEqual(assignments[1], ["env_1", "env_4", "env_7"])
        self.assertEqual(assignments[2], ["env_2", "env_5"])
        self.assertEqual(
            sorted(environment for values in assignments.values() for environment in values),
            environments,
        )

    def test_rejects_duplicate_gpu_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "GPU_SET_INVALID"):
            build_gpu_assignments(["env"], [0, 0])


if __name__ == "__main__":
    unittest.main()
