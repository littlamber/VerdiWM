from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from wmloop.experiments.progressive_fidelity import (
    ProgressiveFidelityError,
    build_progressive_fidelity_report,
)


class ProgressiveFidelityTests(unittest.TestCase):
    def test_computes_recall_rejection_cost_and_rank_metrics(self) -> None:
        candidates = [
            _candidate("a", 1, screen_score=3.0, screen_positive=True, gate_score=1.0, gate_positive=True),
            _candidate("b", 2, screen_score=-1.0, screen_positive=False, gate_score=2.0, gate_positive=True),
            _candidate("c", 3, screen_score=-2.0, screen_positive=False, gate_score=-1.0, gate_positive=False),
        ]
        confirmations = [
            _confirmation("a", 1, score=0.5, positive=True),
            _confirmation("c", 3, score=-0.5, positive=False),
        ]

        report = build_progressive_fidelity_report(
            candidates=candidates,
            confirmations=confirmations,
        )

        self.assertEqual(report["state"], "ready")
        self.assertEqual(report["candidate_count"], 3)
        self.assertEqual(report["confirm_pair_count"], 2)
        self.assertEqual(report["metrics"]["positive_recall"], 0.5)
        self.assertEqual(report["metrics"]["false_rejection_rate"], 0.5)
        self.assertEqual(report["metrics"]["screen_confirm_rank_correlation"], 1.0)
        self.assertAlmostEqual(report["metrics"]["progressive_fidelity_gpu_hours"], 11.7)
        self.assertAlmostEqual(report["metrics"]["confirm_every_candidate_projected_gpu_hours"], 12.6)
        self.assertAlmostEqual(report["metrics"]["gpu_hour_reduction"], 0.9 / 12.6)
        self.assertEqual(report["counterfactual_projected_candidate_count"], 1)

    def test_rejects_duplicate_candidate_identity(self) -> None:
        candidates = [_candidate("a", 1), _candidate("a", 1)]
        confirmations = [_confirmation("a", 1)]
        with self.assertRaisesRegex(ProgressiveFidelityError, "CANDIDATE_DUPLICATE"):
            build_progressive_fidelity_report(candidates=candidates, confirmations=confirmations)

    def test_cli_help_is_cwd_independent(self) -> None:
        import subprocess
        import sys

        script = Path(__file__).resolve().parents[1] / "scripts/export/progressive_fidelity_efficiency.py"
        with tempfile.TemporaryDirectory() as temporary:
            completed = subprocess.run(
                [sys.executable, str(script), "--help"],
                cwd=temporary,
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("usage:", completed.stdout)


def _candidate(
    primitive: str,
    seed: int,
    *,
    screen_score: float = 1.0,
    screen_positive: bool = True,
    gate_score: float = 1.0,
    gate_positive: bool = True,
) -> dict[str, object]:
    return {
        "environment": "push_cube",
        "primitive": primitive,
        "seed": seed,
        "screen_score": screen_score,
        "screen_positive": screen_positive,
        "gate_score": gate_score,
        "gate_positive": gate_positive,
        "screen_gpu_hours": 1.0,
        "gate_gpu_hours": 0.1,
        "screen_manifest": f"screen-{primitive}.json",
        "gate_manifest": f"gate-{primitive}.json",
    }


def _confirmation(
    primitive: str,
    seed: int,
    *,
    score: float = 1.0,
    positive: bool = True,
) -> dict[str, object]:
    return {
        "environment": "push_cube",
        "primitive": primitive,
        "seed": seed,
        "confirm_score": score,
        "confirm_positive": positive,
        "confirm_gpu_hours": 4.0,
        "confirm_gate_gpu_hours": 0.2,
        "confirmation_manifest": f"confirm-{primitive}.json",
    }


if __name__ == "__main__":
    unittest.main()
