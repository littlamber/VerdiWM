"""Tests for the artifact convention lint and the clean graph projection."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from wmloop.experiments.artifact_lint import (
    lint_payload,
    lint_root,
    make_compliance_filter,
)
from wmloop.experiments.evidence_graph import build_evidence_graph


def _codes(issues: list[dict[str, object]]) -> set[str]:
    return {str(issue["code"]) for issue in issues}


class ArtifactLintPayloadTests(unittest.TestCase):
    def test_marker_payload_without_artifact_type_is_an_error(self) -> None:
        issues = lint_payload(
            {"model_ref": "cas://sha256/" + "ab" * 32, "goal_id": "g1"},
            source="memory:0",
        )
        self.assertIn("MISSING_ARTIFACT_TYPE", _codes(issues))

    def test_plain_config_without_markers_is_not_an_artifact(self) -> None:
        # lint_payload itself is agnostic; root-level detection is what
        # decides whether a payload counts as an artifact.
        issues = lint_payload({"learning_rate": 0.001}, source="memory:0")
        self.assertIn("MISSING_ARTIFACT_TYPE", _codes(issues))

    def test_cas_model_ref_without_family_is_an_error(self) -> None:
        issues = lint_payload(
            {
                "artifact_type": "verdiwm-eval-failure-report",
                "model_ref": "cas://sha256/" + "cd" * 32,
            },
            source="memory:0",
        )
        self.assertIn("MODEL_REF_WITHOUT_NAME", _codes(issues))

    def test_bare_hex_model_ref_without_family_is_an_error(self) -> None:
        issues = lint_payload(
            {
                "artifact_type": "verdiwm-eval-failure-report",
                "model_ref": "caa256cd",
            },
            source="memory:0",
        )
        self.assertIn("MODEL_REF_WITHOUT_NAME", _codes(issues))

    def test_named_model_ref_is_compliant(self) -> None:
        issues = lint_payload(
            {
                "artifact_type": "verdiwm-eval-failure-report",
                "model_ref": "cas://sha256/" + "ef" * 32,
                "model_family": "ctrl-world",
                "claim_boundary": "only covers cloth_move",
                "state": "settled",
            },
            source="memory:0",
        )
        self.assertEqual(
            [issue for issue in issues if issue["severity"] == "error"], []
        )

    def test_semantic_identity_with_trailing_digest_is_compliant(self) -> None:
        issues = lint_payload(
            {
                "artifact_type": "verdiwm-research-campaign",
                "campaign_id": "cloth-move-dose-ladder-a1b2c3d4",
                "claim_boundary": "x",
                "state": "open",
            },
            source="memory:0",
        )
        self.assertNotIn("IDENTITY_EMBEDS_BARE_HASH", _codes(issues))

    def test_final_style_suffix_is_an_error(self) -> None:
        issues = lint_payload(
            {
                "artifact_type": "verdiwm-research-campaign",
                "campaign_id": "dose-ladder-final",
                "claim_boundary": "x",
                "state": "open",
            },
            source="memory:0",
        )
        self.assertIn("IDENTITY_BAD_SUFFIX", _codes(issues))


class ArtifactLintRootTests(unittest.TestCase):
    def test_root_lints_marker_payloads_missing_artifact_type(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "failure_report.json").write_text(
                json.dumps({"model_ref": "cas://sha256/" + "ab" * 32, "goal_id": "g1"}),
                encoding="utf-8",
            )
            (root / "config.json").write_text(
                json.dumps({"learning_rate": 0.001}), encoding="utf-8"
            )
            report = lint_root(root)
        self.assertEqual(report["checked"], 1)
        self.assertEqual(report["error_count"], 2)
        self.assertIn("MISSING_ARTIFACT_TYPE", report["code_counts"])
        self.assertIn("MODEL_REF_WITHOUT_NAME", report["code_counts"])


class CleanProjectionTests(unittest.TestCase):
    def test_compliance_filter_excludes_error_payloads_from_graph(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            bad = {
                "model_ref": "cas://sha256/" + "cd" * 32,
                "goal_id": "g1_long_horizon",
            }
            good = {
                "artifact_type": "verdiwm-eval-horizon-profile",
                "record_id": "cloth-move-horizon-01234567",
                "model_ref": "cas://sha256/" + "ef" * 32,
                "model_family": "ctrl-world",
                "claim_boundary": "cloth_move only",
                "state": "settled",
            }
            (root / "failure_report.json").write_text(
                json.dumps(bad), encoding="utf-8"
            )
            (root / "horizon.json").write_text(json.dumps(good), encoding="utf-8")

            full = build_evidence_graph(root)
            include = make_compliance_filter(root)
            clean = build_evidence_graph(root, include_payload=include)

        full_models = [n for n in full["nodes"] if n.get("kind") == "model"]
        clean_models = [n for n in clean["nodes"] if n.get("kind") == "model"]
        # The non-compliant payload's bare model node disappears; the named
        # model from the compliant artifact stays.
        self.assertEqual(len(full_models), 2)
        self.assertEqual(len(clean_models), 1)
        self.assertEqual(clean_models[0].get("family"), "ctrl-world")
        self.assertLess(clean["node_count"], full["node_count"])

    def test_warning_only_payloads_stay_in_clean_projection(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            warning_only = {
                "artifact_type": "wmloop-legacy-note",
                "record_id": "legacy-note-89abcdef",
            }
            (root / "note.json").write_text(
                json.dumps(warning_only), encoding="utf-8"
            )
            report = lint_root(root)
            include = make_compliance_filter(root)
            clean = build_evidence_graph(root, include_payload=include)

        self.assertEqual(report["error_count"], 0)
        self.assertGreater(report["warning_count"], 0)
        self.assertEqual(clean["source_count"], 1)


if __name__ == "__main__":
    unittest.main()
