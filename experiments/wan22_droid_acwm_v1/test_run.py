import json
import tempfile
import unittest
from pathlib import Path

from wmloop.wan22_droid import Wan22DroidError, build_sample_manifest, validate_contract


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


if __name__ == "__main__":
    unittest.main()
