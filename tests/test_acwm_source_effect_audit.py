from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from wmloop.experiments.acwm_fingerprint import sha256_file
from wmloop.experiments.source_effect_audit import (
    SourceEffectAuditError,
    build_source_effect_audit,
)


class ACWMSourceEffectAuditTests(unittest.TestCase):
    def test_separates_stable_negative_underreplicated_and_reproduction_conflict(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            reports = root / "reports"
            labels = []
            for environment, seeds, signs in (
                ("stable", (1, 2, 3), (False, False, False)),
                ("under", (4,), (True,)),
                ("conflict", (5, 5), (True, False)),
            ):
                for ordinal, (seed, positive) in enumerate(zip(seeds, signs), start=1):
                    receipt = reports / f"acwm-effect-label-gate-{environment}-self_forcing_finetune-s{seed}-r{ordinal}"
                    receipt.mkdir(parents=True)
                    payload = _receipt(environment=environment, seed=seed, positive=positive)
                    (receipt / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
                    if ordinal == 1:
                        evidence = receipt / "manifest.json"
                        labels.append(
                            {
                                "label_id": receipt.name,
                                "environment": environment,
                                "primitive": "self_forcing_finetune",
                                "seed": seed,
                                "settled": True,
                                "positive": positive,
                                "label_source": "retained_checkpoint_completion_gate",
                                "evidence_ref": str(evidence),
                                "evidence_sha256": sha256_file(evidence),
                            }
                        )
            index = root / "effect-label-index.json"
            index.write_text(
                json.dumps(
                    {
                        "artifact_type": "verdiwm-settled-effect-label-index",
                        "expected_environments": ["stable", "under", "conflict", "empty"],
                        "labels": labels,
                    }
                ),
                encoding="utf-8",
            )
            build_source_effect_audit(
                effect_label_index_path=index,
                reports_root=reports,
                output_root=root / "audit",
                primitive="self_forcing_finetune",
            )
            report = json.loads((root / "audit" / "source-effect-audit.json").read_text())
            by_environment = {row["environment"]: row for row in report["environment_audits"]}
            self.assertEqual(by_environment["stable"]["classification"], "stable_negative")
            self.assertEqual(by_environment["under"]["classification"], "positive_underreplicated")
            self.assertEqual(
                by_environment["conflict"]["classification"],
                "same_protocol_reproduction_conflict",
            )
            self.assertEqual(by_environment["empty"]["classification"], "no_official_gate_evidence")
            self.assertEqual(report["collision_verdict"], "mixed_effects_with_underreplicated_positive_sources")

    def test_fails_when_indexed_evidence_sha_changes(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            receipt = root / "reports" / "acwm-effect-label-gate-env-self_forcing_finetune-s1-r1"
            receipt.mkdir(parents=True)
            evidence = receipt / "manifest.json"
            evidence.write_text(json.dumps(_receipt(environment="env", seed=1, positive=True)), encoding="utf-8")
            index = root / "index.json"
            index.write_text(
                json.dumps(
                    {
                        "artifact_type": "verdiwm-settled-effect-label-index",
                        "expected_environments": ["env"],
                        "labels": [
                            {
                                "environment": "env",
                                "primitive": "self_forcing_finetune",
                                "seed": 1,
                                "settled": True,
                                "positive": True,
                                "evidence_ref": str(evidence),
                                "evidence_sha256": "0" * 64,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(SourceEffectAuditError, "INDEX_SHA_MISMATCH"):
                build_source_effect_audit(
                    effect_label_index_path=index,
                    reports_root=root / "reports",
                    output_root=root / "audit",
                    primitive="self_forcing_finetune",
                )

    def test_includes_archived_formal_training_seed_gates(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            reports = root / "reports"
            reports.mkdir()
            archive = root / "archive-gates"
            for training_seed in (4101, 4202, 4303):
                for eval_seed in (1101, 2202, 3303):
                    receipt = archive / (
                        "acwm-formal-trainseed-gate-cloth_move-self_forcing_finetune-"
                        f"ts{training_seed}-es{eval_seed}-r1"
                    )
                    receipt.mkdir(parents=True)
                    payload = _receipt(environment="cloth_move", seed=eval_seed, positive=True)
                    payload["training_seed"] = training_seed
                    payload["candidate_checkpoint_sha256"] = f"{training_seed:064x}"
                    (receipt / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
            index = root / "index.json"
            index.write_text(
                json.dumps(
                    {
                        "artifact_type": "verdiwm-settled-effect-label-index",
                        "expected_environments": ["cloth_move"],
                        "labels": [],
                    }
                ),
                encoding="utf-8",
            )
            build_source_effect_audit(
                effect_label_index_path=index,
                reports_root=reports,
                additional_receipt_roots=[archive],
                output_root=root / "audit",
                primitive="self_forcing_finetune",
            )
            report = json.loads((root / "audit" / "source-effect-audit.json").read_text())
            row = report["environment_audits"][0]
            self.assertEqual(row["classification"], "stable_positive")
            self.assertEqual(row["protocol_complete_receipt_count"], 9)
            self.assertEqual(row["independent_training_seed_count"], 3)
            self.assertEqual(row["eval_stable_positive_training_seed_checkpoint_count"], 3)
            self.assertEqual(row["eval_stable_positive_training_seeds"], [4101, 4202, 4303])

    def test_deduplicates_a_receipt_copied_into_an_archive(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            reports = root / "reports"
            archive = root / "archive-gates"
            receipt_id = "acwm-formal-trainseed-gate-env-self_forcing_finetune-ts1-es2-r1"
            payload = _receipt(environment="env", seed=2, positive=True)
            payload["training_seed"] = 1
            for receipt_root in (reports, archive):
                receipt = receipt_root / receipt_id
                receipt.mkdir(parents=True)
                (receipt / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
            index = root / "index.json"
            index.write_text(
                json.dumps(
                    {
                        "artifact_type": "verdiwm-settled-effect-label-index",
                        "expected_environments": ["env"],
                        "labels": [],
                    }
                ),
                encoding="utf-8",
            )
            build_source_effect_audit(
                effect_label_index_path=index,
                reports_root=reports,
                additional_receipt_roots=[archive],
                output_root=root / "audit",
                primitive="self_forcing_finetune",
            )
            report = json.loads((root / "audit" / "source-effect-audit.json").read_text())
            self.assertEqual(report["discovered_official_gate_receipt_count"], 1)
            self.assertEqual(report["environment_audits"][0]["discovered_receipt_count"], 1)


def _receipt(*, environment: str, seed: int, positive: bool) -> dict[str, object]:
    return {
        "state": "ready",
        "environment": environment,
        "primitive": "self_forcing_finetune",
        "seed": seed,
        "eval_seed": seed,
        "steps": 50,
        "split": "ind_test",
        "max_trajs": 3,
        "training_seed": seed,
        "candidate_checkpoint_sha256": f"{seed:064x}",
        "protocol_provenance": {
            "dataset_freeze_sha256": "a" * 64,
            "heldout_protocol_sha256": "b" * 64,
            "eval_script_sha256": "c" * 64,
            "eval_config_sha256": "d" * 64,
        },
        "official_quality_gate": {
            "state": "pass" if positive else "fail",
            "pass": positive,
            "checks": {"psnr": positive},
            "delta_candidate_minus_baseline": {"psnr": 0.2 if positive else -0.2},
        },
    }


if __name__ == "__main__":
    unittest.main()
