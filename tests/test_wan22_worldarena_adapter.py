import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.evaluate_wan22_worldarena import _extract_metrics, _validate_assets, _validate_prepared_root, _validate_run
from scripts.fetch_worldarena_assets import main as fetch_assets


class _Reader:
    def __init__(self, frames=150, fps=5.0):
        self.frames = frames
        self.metadata = {"fps": fps}

    def count_frames(self):
        return self.frames

    def get_meta_data(self):
        return self.metadata

    def close(self):
        return None


class Wan22WorldArenaAdapterTests(unittest.TestCase):
    def test_extract_metrics_preserves_aggregate_and_rows(self):
        result = _extract_metrics({"subject_consistency": [0.2, [{"video_results": 0.7}]], "empty": 3})
        self.assertEqual(result["subject_consistency"]["aggregate_reported"], 0.2)
        self.assertEqual(result["subject_consistency"]["per_video"][0]["video_results"], 0.7)
        self.assertEqual(result["empty"], {"raw": 3})

    def test_validate_run_rejects_non_150_frame_rollout(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            for name in ("generated_150f.mp4", "ground_truth_150f.mp4", "worldarena_input.json", "worldarena_summary.json"):
                (root / name).write_bytes(b"x")
            with patch("scripts.evaluate_wan22_worldarena.imageio.get_reader", return_value=_Reader(frames=149)):
                with self.assertRaisesRegex(ValueError, "WAN22_WORLD_ARENA_VIDEO_INVALID"):
                    _validate_run(root)

    def test_validate_assets_is_hash_bound(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            asset = root / "asset.bin"
            asset.write_bytes(b"ok")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"assets": [{"id": "a", "local_path": "asset.bin", "sha256": "2689367b205c16ce2f8f6f6f9f8f4f3f5f7f0f6f3f6c6f6f6f6f6f6f6f6f6f6f6", "bytes": 2}]}))
            with self.assertRaisesRegex(ValueError, "WORLD_ARENA_ASSET_MISMATCH"):
                _validate_assets(manifest, root)

    def test_fetch_verify_only_writes_receipt(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "asset.bin").write_bytes(b"ok")
            import hashlib
            digest = hashlib.sha256(b"ok").hexdigest()
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"assets": [{"id": "a", "local_path": "asset.bin", "sha256": digest, "bytes": 2}]}))
            receipt = root / "receipt.json"
            self.assertEqual(fetch_assets(["--manifest", str(manifest), "--output-root", str(root), "--verify-only", "--receipt", str(receipt)]), 0)
            payload = json.loads(receipt.read_text())
            self.assertEqual(payload["state"], "verified")

    def test_action_following_rejects_single_generated_gid(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            video = root / "generated_dataset" / "droid" / "episode" / "1" / "video"
            video.mkdir(parents=True)
            for index in range(150):
                (video / f"frame_{index:05d}.png").write_bytes(b"x")
            with self.assertRaisesRegex(ValueError, "REQUIRES_MULTIPLE_GIDS"):
                _validate_prepared_root(root, ["action_following"])


if __name__ == "__main__":
    unittest.main()
