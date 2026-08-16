from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")

from scripts.run_ctrl_world_anchor_repair_screen import (
    FirstFrameConditioningDecay,
    RecencyAnchorBalance,
    run,
)
from scripts.settle_ctrl_world_anchor_repair_screen import settle


OUTCOMES = (
    "negative_mean_l1",
    "negative_final_interaction_l1",
    "negative_horizon_l1_slope",
    "mean_psnr",
    "final_psnr",
)


class CtrlWorldAnchorRepairTests(unittest.TestCase):
    def test_screen_constructs_unmodified_baseline_runtime(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            inputs = {
                name: root / name
                for name in (
                    "ctrl-world",
                    "dataset",
                    "data-stat.json",
                    "svd-model",
                    "clip-model",
                    "checkpoint.pt",
                    "contexts.json",
                )
            }
            inputs["ctrl-world"].mkdir()
            inputs["dataset"].mkdir()
            inputs["svd-model"].mkdir()
            inputs["clip-model"].mkdir()
            for name in ("data-stat.json", "checkpoint.pt", "contexts.json"):
                inputs[name].touch()

            def capture_runtime_args(**kwargs: object) -> object:
                self.assertFalse(kwargs["enable_signed_history_correction"])
                self.assertFalse(kwargs["unsigned_history_gate"])
                self.assertFalse(kwargs["enable_multiscale_history_adapter"])
                self.assertFalse(kwargs["multiscale_history_always_on"])
                raise RuntimeError("runtime-args-captured")

            args = type(
                "Args",
                (),
                {
                    "output_root": root / "output",
                    "primitive_id": "first_frame_conditioning_decay",
                    "strengths": [0.0, 0.025],
                    "interact_num": 4,
                    "ctrl_world_root": inputs["ctrl-world"],
                    "contexts_json": inputs["contexts.json"],
                    "seeds": [],
                    "context_ids": [],
                    "dataset_root": inputs["dataset"],
                    "data_stat": inputs["data-stat.json"],
                    "svd_model_path": inputs["svd-model"],
                    "clip_model_path": inputs["clip-model"],
                    "ckpt_path": inputs["checkpoint.pt"],
                    "num_inference_steps": 2,
                },
            )()
            with (
                patch(
                    "scripts.run_ctrl_world_anchor_repair_screen.load_contexts",
                    return_value=[{"context_id": "ctx", "episode_id": "1", "start_idx": 0, "seed": 1}],
                ),
                patch(
                    "scripts.run_ctrl_world_anchor_repair_screen._load_rollout_module",
                    return_value=object(),
                ),
                patch(
                    "scripts.run_ctrl_world_anchor_repair_screen._runtime_args",
                    side_effect=capture_runtime_args,
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "runtime-args-captured"):
                    run(args)

    def test_decay_is_identity_at_zero_and_increases_with_interaction(self) -> None:
        current = torch.ones((1, 4, 2, 2))
        first = torch.zeros_like(current)
        history = torch.zeros((1, 6, 4, 2, 2))
        zero = FirstFrameConditioningDecay(0.0)
        _, unchanged = zero.transform_conditions(
            history=history, current=current, first=first, interaction=3, interaction_count=4
        )
        self.assertTrue(torch.equal(unchanged, current))
        repair = FirstFrameConditioningDecay(0.2)
        _, early = repair.transform_conditions(
            history=history, current=current, first=first, interaction=0, interaction_count=4
        )
        _, late = repair.transform_conditions(
            history=history, current=current, first=first, interaction=3, interaction_count=4
        )
        self.assertTrue(torch.equal(early, current))
        self.assertGreater(float((late - current).abs().mean()), 0.0)

    def test_recency_balance_changes_oldest_not_newest_history_slot(self) -> None:
        history = torch.zeros((1, 3, 4, 2, 2))
        current = torch.ones((1, 4, 2, 2))
        repair = RecencyAnchorBalance(0.2)
        transformed, unchanged_current = repair.transform_conditions(
            history=history,
            current=current,
            first=torch.zeros_like(current),
            interaction=1,
            interaction_count=4,
        )
        self.assertTrue(torch.equal(unchanged_current, current))
        self.assertTrue(torch.allclose(transformed[:, 0], torch.full_like(current, 0.2)))
        self.assertTrue(torch.equal(transformed[:, -1], history[:, -1]))

    def _result(self, primitive_id: str, effect: float) -> dict[str, object]:
        measurements = []
        references = []
        for strength in (0.0, 0.025):
            for seed in (101, 202, 303):
                outcomes = {name: float(seed) for name in OUTCOMES}
                if strength:
                    outcomes = {name: value + effect for name, value in outcomes.items()}
                row = {
                    "strength": strength,
                    "identity": {"context_id": "ctx", "episode_id": "1", "start_idx": 0, "seed": seed},
                    "outcomes": outcomes,
                    "hook_audit": {},
                }
                measurements.append(row)
                if strength == 0.0:
                    references.append(row)
        return {
            "artifact_type": "verdiwm-ctrl-world-anchor-repair-result",
            "state": "ready",
            "primitive_id": primitive_id,
            "primitive_type": "inference_conditioning",
            "target_hook": "current_image_condition",
            "outcome_names": list(OUTCOMES),
            "input": {"checkpoint": "/checkpoint", "contexts_sha256": "abc"},
            "unwrapped_references": references,
            "measurements": measurements,
            "zero_identity_checks": [{"state": "passed"}],
            "hook_activation": {"state": "passed"},
        }

    def test_settlement_freezes_positive_top_candidate(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            paths = []
            for primitive_id, effect in (("decay", 0.2), ("recency", 0.1)):
                path = root / f"{primitive_id}.json"
                path.write_text(json.dumps(self._result(primitive_id, effect)), encoding="utf-8")
                paths.append(path)
            manifest = settle(result_paths=paths, output_root=root / "settled", confidence_z=0.0)
            self.assertEqual(manifest["selector_action"], "execute")
            selector = json.loads((root / "settled" / "selector.json").read_text())
            self.assertEqual(selector["selected_candidate"]["primitive_id"], "decay")


if __name__ == "__main__":
    unittest.main()
