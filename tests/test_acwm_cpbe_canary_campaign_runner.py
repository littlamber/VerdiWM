from __future__ import annotations

import unittest

from scripts.run_acwm_cpbe_canary_campaigns import build_candidate_gpu_assignments


class ACWMCPBECanaryCampaignRunnerTests(unittest.TestCase):
    def test_assigns_one_candidate_per_gpu_deterministically(self) -> None:
        campaigns = [
            {"probe_id": "probe_c"},
            {"probe_id": "probe_a"},
            {"probe_id": "probe_b"},
        ]
        assignments = build_candidate_gpu_assignments(campaigns, [2, 0, 1])
        self.assertEqual(
            [(row["probe_id"], gpu) for row, gpu in assignments],
            [("probe_a", 2), ("probe_b", 0), ("probe_c", 1)],
        )

    def test_rejects_candidate_gpu_cardinality_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "CANDIDATE_GPU_FRAME_INVALID"):
            build_candidate_gpu_assignments([{"probe_id": "probe"}], [0, 1])


if __name__ == "__main__":
    unittest.main()
