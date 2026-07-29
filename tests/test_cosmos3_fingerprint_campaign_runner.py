from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.run_cosmos3_fingerprint_campaign import (
    _foreign_compute_pids,
    _pid_is_descendant,
    _preflight_runtime,
)
from wmloop.primitives.adapters.cosmos3_hooks import apply_action_probe


class Cosmos3FingerprintCampaignRunnerTests(unittest.TestCase):
    def test_temporal_mix_dispatch_preserves_per_dimension_mean(self) -> None:
        actions = [[1.0, 4.0], [3.0, 0.0], [2.0, 2.0]]
        mixed = apply_action_probe(
            actions,
            probe_id="action_embedding_temporal_mix",
            dose=0.1,
        )
        self.assertNotEqual(mixed, actions)
        for index in range(2):
            self.assertAlmostEqual(
                sum(row[index] for row in mixed),
                sum(row[index] for row in actions),
            )

    def test_gpu_exclusivity_accepts_only_descendant_compute_pids(self) -> None:
        parents = {31: 20, 20: 10, 77: 1}
        reader = parents.get
        rows = [("GPU-target", 31), ("GPU-other", 77)]
        self.assertEqual(_foreign_compute_pids("GPU-target", rows, 10, reader), [])
        self.assertTrue(_pid_is_descendant(31, 10, reader))

    def test_gpu_exclusivity_rejects_foreign_pid_on_target_uuid(self) -> None:
        parents = {31: 20, 20: 10, 77: 1}
        rows = [("GPU-target", 31), ("GPU-target", 77)]
        self.assertEqual(_foreign_compute_pids("GPU-target", rows, 10, parents.get), [77])

    def test_gpu_exclusivity_retains_a_previously_observed_child_during_exit(self) -> None:
        parents = {31: 1}
        rows = [("GPU-target", 31)]
        self.assertEqual(
            _foreign_compute_pids(
                "GPU-target",
                rows,
                10,
                parents.get,
                trusted_pids={31},
            ),
            [],
        )

    def test_runtime_preflight_accepts_available_modules(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            _preflight_runtime(
                Path(sys.executable),
                modules=("json", "pathlib"),
                code="PREFLIGHT_FAILED",
                cwd=Path(temporary),
                env=os.environ.copy(),
            )

    def test_runtime_preflight_reports_missing_dependency_before_campaign(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "PREFLIGHT_FAILED:1:.*ModuleNotFoundError"):
                _preflight_runtime(
                    Path(sys.executable),
                    modules=("verdiwm_module_that_does_not_exist",),
                    code="PREFLIGHT_FAILED",
                    cwd=Path(temporary),
                    env=os.environ.copy(),
                )


if __name__ == "__main__":
    unittest.main()
