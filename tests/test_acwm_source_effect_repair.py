from __future__ import annotations

import unittest

from wmloop.experiments.source_effect_repair import summarize_source_effect_repair


class ACWMSourceEffectRepairTests(unittest.TestCase):
    def test_summarizes_sign_consistency_per_training_checkpoint(self) -> None:
        rows = [
            _row("cloth_move", 1, 101, "a", True),
            _row("cloth_move", 1, 202, "a", True),
            _row("push_rope", 2, 101, "b", False),
            _row("push_rope", 2, 202, "b", True),
        ]
        groups = summarize_source_effect_repair(rows=rows)
        by_environment = {row["environment"]: row for row in groups}
        self.assertEqual(by_environment["cloth_move"]["classification"], "consistently_positive")
        self.assertEqual(by_environment["cloth_move"]["distinct_eval_seed_count"], 2)
        self.assertEqual(by_environment["push_rope"]["classification"], "sign_inconsistent")


def _row(
    environment: str,
    training_seed: int,
    eval_seed: int,
    checkpoint: str,
    positive: bool,
) -> dict[str, object]:
    return {
        "environment": environment,
        "training_seed": training_seed,
        "eval_seed": eval_seed,
        "candidate_checkpoint_sha256": checkpoint,
        "positive": positive,
    }


if __name__ == "__main__":
    unittest.main()
