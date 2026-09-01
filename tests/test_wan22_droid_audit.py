import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_wan22_droid_subset import audit


class Wan22DroidAuditTests(unittest.TestCase):
    def test_audit_checks_pairs_and_target_horizon(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            for split in ("train", "val"):
                for part in ("annotation", "videos", "latent_videos"):
                    (root / part / split).mkdir(parents=True)
                episode = "episode-a"
                payload = {
                    "split": split,
                    "video_length": 150,
                    "action": [[0.0] * 7 for _ in range(150)],
                    "proprio": [[0.0] * 14 for _ in range(150)],
                }
                (root / "annotation" / split / f"{episode}.json").write_text(json.dumps(payload))
                (root / "videos" / split / episode).mkdir()
                (root / "latent_videos" / split / episode).mkdir()
                (root / "videos" / split / episode / "wrist.mp4").write_bytes(b"video")
                (root / "latent_videos" / split / episode / "wrist.pt").write_bytes(b"latent")
            report = audit(root)
            self.assertEqual(report["splits"]["val"]["episodes_at_target"], 1)
            self.assertEqual(report["splits"]["val"]["missing_count"], 0)


if __name__ == "__main__":
    unittest.main()
