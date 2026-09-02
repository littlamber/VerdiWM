import json
import tempfile
import unittest
from pathlib import Path

from wmloop.wan22_droid import Wan22DroidError, build_sample_manifest, validate_contract
from experiments.wan22_droid_acwm_v1.wan22_droid_runner import _assert_episode_disjoint, _conditioning_for_mode


class Wan22DroidContractTests(unittest.TestCase):
    def _dataset(self, root: Path) -> None:
        for split in ("train", "val"):
            (root / "annotation" / split).mkdir(parents=True)
            (root / "videos" / split / "episode-a").mkdir(parents=True)
            (root / "latent_videos" / split / "episode-a").mkdir(parents=True)
            (root / "videos" / split / "episode-a" / "wrist.mp4").write_bytes(b"v")
            (root / "latent_videos" / split / "episode-a" / "wrist.pt").write_bytes(b"l")
            payload = {
                "episode_id": "episode-a", "split": split, "video_length": 151,
                "video_path": f"videos/{split}/episode-a/wrist.mp4",
                "latent_path": f"latent_videos/{split}/episode-a/wrist.pt",
                "processed_fps": 5, "action": [[0.0] * 7 for _ in range(151)],
                "proprio": [[0.0] * 14 for _ in range(151)],
            }
            (root / "annotation" / split / "episode-a.json").write_text(json.dumps(payload), encoding="utf-8")

    def test_manifest_windows_are_temporally_bound(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); self._dataset(root)
            manifest = build_sample_manifest(root, "val")
            self.assertEqual(manifest["record_count"], 1)
            self.assertEqual(manifest["records"][0]["horizon_frames"], 150)
            self.assertEqual(manifest["records"][0]["action_dim"], 7)

    def test_short_episode_is_excluded_and_empty_split_blocks(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); self._dataset(root)
            payload = json.loads((root / "annotation" / "val" / "episode-a.json").read_text())
            payload["video_length"] = 20; payload["action"] = payload["action"][:20]; payload["proprio"] = payload["proprio"][:20]
            (root / "annotation" / "val" / "episode-a.json").write_text(json.dumps(payload))
            with self.assertRaises(Wan22DroidError): build_sample_manifest(root, "val")

    def test_contract_requires_real_paths(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); self._dataset(root)
            train = root / "train.json"; val = root / "val.json"
            train.write_text(json.dumps(build_sample_manifest(root, "train")))
            val.write_text(json.dumps(build_sample_manifest(root, "val")))
            report = validate_contract(train_manifest=train, validation_manifest=val, model=root / "model", source=root / "source", evaluator_contract=root / "eval.json", adapter=root / "wan22_droid_adapter.py")
            self.assertEqual(report["state"], "blocked")
            self.assertIn("MODEL_PATH_INVALID", report["blockers"])

    def test_adapter_contract_accepts_explicit_wan22_droid_name(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); self._dataset(root)
            adapter = root / "wan22_droid_adapter.py"; adapter.write_text("# contract\n")
            model = root / "model"; model.mkdir()
            source = root / "source"; (source / "wan" / "modules").mkdir(parents=True)
            (source / "wan" / "modules" / "model.py").write_text("# entrypoint\n")
            evaluator = root / "eval.json"; evaluator.write_text("{}")
            train = root / "train.json"; val = root / "val.json"
            train.write_text(json.dumps(build_sample_manifest(root, "train")))
            val.write_text(json.dumps(build_sample_manifest(root, "val")))
            report = validate_contract(train_manifest=train, validation_manifest=val, model=model, source=source, evaluator_contract=evaluator, adapter=adapter)
            self.assertNotIn("WAN22_ADAPTER_IDENTITY_INVALID", report["blockers"])

    def test_runner_rejects_train_validation_episode_overlap(self):
        train = {"records": [{"episode_id": "episode-a"}]}
        validation = {"records": [{"episode_id": "episode-a"}]}
        with self.assertRaisesRegex(ValueError, "WAN22_DROID_EPISODE_SPLIT_OVERLAP"):
            _assert_episode_disjoint(train, validation)

    def test_conditioning_arms_are_explicit_and_history_is_causal(self):
        import numpy as np
        actions = np.asarray([[1.0] * 7, [3.0] * 7], dtype=np.float32)
        proprio = np.asarray([[2.0] * 14, [4.0] * 14], dtype=np.float32)
        zero_a, zero_p = _conditioning_for_mode(actions, proprio, "visual_anchor_only")
        self.assertTrue(np.all(zero_a == 0) and np.all(zero_p == 0))
        action_a, action_p = _conditioning_for_mode(actions, proprio, "action")
        self.assertTrue(np.array_equal(action_a, actions) and np.all(action_p == 0))
        hist_a, hist_p = _conditioning_for_mode(actions, proprio, "action_proprio_history")
        self.assertEqual(float(hist_a[0, 0]), 1.0)
        self.assertEqual(float(hist_a[1, 0]), 2.0)
        self.assertEqual(float(hist_p[1, 0]), 3.0)


if __name__ == "__main__":
    unittest.main()
