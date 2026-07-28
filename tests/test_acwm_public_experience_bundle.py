from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

from scripts.export.acwm_public_experience_bundle import (
    PublicExperienceBundleError,
    export_public_experience_bundle,
    validate_public_experience_bundle,
)


class PublicExperienceBundleTests(unittest.TestCase):
    def test_exports_path_safe_positive_and_negative_experience(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            screen = self._screen_summary(root)
            showcase = self._showcase(root)
            experience = self._experience_map(root)
            output = root / "public"

            report = export_public_experience_bundle(
                screen_summary_root=screen,
                showcase_root=showcase,
                experience_map_paths=[experience],
                output_root=output,
            )

            self.assertEqual(report["screen_trial_count"], 2)
            self.assertEqual(report["positive_screen_count"], 1)
            self.assertEqual(report["negative_screen_count"], 1)
            self.assertEqual(report["showcase_case_count"], 4)
            atlas = json.loads((output / "experience-atlas.json").read_text(encoding="utf-8"))
            self.assertEqual(atlas["record_count"], 2)
            self.assertEqual(atlas["causal_edge_count"], 0)
            self.assertNotIn("/" + "mnt" + "/", (output / "showcase" / "manifest.json").read_text(encoding="utf-8"))

    def test_rejects_mutated_public_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "public"
            export_public_experience_bundle(
                screen_summary_root=self._screen_summary(root),
                showcase_root=self._showcase(root),
                experience_map_paths=[self._experience_map(root)],
                output_root=output,
            )
            target = output / "screen-summary.json"
            target.write_text(target.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(PublicExperienceBundleError, "SHA256_MISMATCH"):
                validate_public_experience_bundle(output)

    def _screen_summary(self, root: Path) -> Path:
        screen = root / "screen"
        tables = screen / "tables"
        tables.mkdir(parents=True)
        rows = [
            self._screen_row("promote_to_confirmation", "1.0"),
            self._screen_row("reject_or_revise", "-2.0"),
        ]
        self._csv(tables / "screen-trials.csv", rows)
        self._csv(tables / "horizon-metrics.csv", [{field: "1" for field in self._horizon_fields()}])
        self._csv(tables / "best-by-environment.csv", [{field: "x" for field in self._best_fields()}])
        (screen / "manifest.json").write_text(json.dumps({
            "artifact_type": "wmloop-acwm-screen-summary",
            "state": "ready",
            "campaign_count": 2,
            "completed_row_count": 2,
            "horizon_row_count": 1,
            "positive_screen_count": 1,
            "negative_screen_count": 1,
            "action_gate_fail_count": 0,
            "limitations": ["screen only"],
        }), encoding="utf-8")
        return screen

    def _showcase(self, root: Path) -> Path:
        showcase = root / "showcase"
        (showcase / "videos").mkdir(parents=True)
        (showcase / "posters").mkdir(parents=True)
        records = []
        for index in range(4):
            video = showcase / "videos" / f"case-{index}.mp4"
            poster = showcase / "posters" / f"case-{index}.png"
            video.write_bytes(b"video" + bytes([index]))
            poster.write_bytes(b"poster" + bytes([index]))
            records.append({
                "id": f"case-{index}",
                "environment": f"env-{index}",
                "primitive": "repair",
                "evidence_type": "aggregate_official_gate_pass",
                "aggregate_official_gate_pass": True,
                "independent_confirmation_pass": True,
                "psnr_delta": 1.0,
                "ssim_delta": 0.1,
                "mse_delta": -0.1,
                "masked_mse_delta": -0.1,
                "video_path": str(video),
                "poster_path": str(poster),
                "source_video": "/" + "mnt" + "/private/source.mp4",
                "official_gate_manifest": "/" + "mnt" + "/private/gate.json",
                "official_gate_manifest_sha256": "a" * 64,
                "confirmation_evidence": {"confirmation_manifest": "/" + "mnt" + "/private/confirm.json"},
            })
        (showcase / "manifest.json").write_text(json.dumps({
            "artifact_type": "wmloop-acwm-projectpage-showcase-bundle",
            "state": "ready",
            "case_count": 4,
            "confirmed_case_count": 4,
            "claim_boundary": "official gate",
            "records": records,
            "source_spec": "/" + "mnt" + "/private/spec.json",
        }), encoding="utf-8")
        return showcase

    def _experience_map(self, root: Path) -> Path:
        path = root / "map" / "horizon-experience-map.json"
        path.parent.mkdir()
        edge = {
            "environment": "env-0",
            "primitive": "repair",
            "effect_scope": "aggregate_long_horizon_positive",
            "candidate_checkpoint": "/" + "mnt" + "/private/model.pt",
            "causal_credit_eligible": False,
        }
        path.write_text(json.dumps({
            "artifact_type": "wmloop-acwm-horizon-experience-map",
            "observational_edges": [edge],
            "routing_priors": [edge],
            "anti_conditions": [{**edge, "effect_scope": "negative"}],
            "causal_edges": [],
        }), encoding="utf-8")
        return path

    def _screen_row(self, decision: str, delta: str) -> dict[str, str]:
        fields = (
            "campaign_id", "environment", "primitive", "seed", "train_steps", "state",
            "verdict", "screen_decision", "primary_metric", "delta_primary_metric",
            "baseline_primary_metric", "candidate_primary_metric", "action_following_enabled",
            "action_following_pass", "action_following_observed", "latest_checkpoint_step",
            "candidate_checkpoint_retained", "candidate_checkpoint_sha256",
            "official_visual_asset_count",
        )
        row = {field: "x" for field in fields}
        row.update({"campaign_id": decision, "screen_decision": decision, "delta_primary_metric": delta})
        return row

    def _horizon_fields(self) -> tuple[str, ...]:
        return (
            "campaign_id", "environment", "primitive", "seed", "train_steps", "horizon",
            "baseline_psnr", "candidate_psnr", "delta_psnr", "baseline_ssim", "candidate_ssim",
            "delta_ssim", "baseline_masked_mse", "candidate_masked_mse", "delta_masked_mse",
        )

    def _best_fields(self) -> tuple[str, ...]:
        return (
            "environment", "primitive", "campaign_id", "seed", "train_steps", "screen_decision",
            "delta_primary_metric", "action_following_pass", "official_visual_asset_count",
        )

    def _csv(self, path: Path, rows: list[dict[str, str]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
