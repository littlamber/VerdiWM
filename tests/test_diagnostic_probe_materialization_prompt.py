from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from wmloop.archive.store import ArchiveStore
from wmloop.diagnose.diagnostic_probe_materialization_prompt import (
    DiagnosticProbeMaterializationPromptError,
    main,
    run_diagnostic_probe_materialization_prompt_batch,
)


ROOT = Path(__file__).resolve().parents[1]


class DiagnosticProbeMaterializationPromptTests(unittest.TestCase):
    def test_batch_generates_guarded_probe_prompts(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            bank = _write_bank(root)

            manifest = run_diagnostic_probe_materialization_prompt_batch(
                repo_root=ROOT,
                failure_signature_bank_manifest=bank,
                output_root=root / "reports" / "autoprobe-prompts",
                archive_db=root / "results" / "archive.db",
                cas_root=root / "results",
            )

            self.assertEqual(manifest["state"], "ready")
            self.assertEqual(manifest["prompt_count"], 2)
            self.assertFalse(manifest["side_effects"]["verdict_probe_mutated"])
            self.assertFalse(manifest["side_effects"]["gpu_execution_started"])
            self.assertGreaterEqual(ArchiveStore(root / "results" / "archive.db").archive_statistics()["artifacts"], 3)

            report = json.loads(Path(manifest["report_path"]).read_text(encoding="utf-8"))
            self.assertIn("verdict_evidence", report["alignment_contract"]["must_not_change"])
            self.assertEqual(
                report["alignment_contract"]["engineering_practice_policy"]["policy_id"],
                "verdiwm_agent_engineering_policy_v1",
            )
            first_prompt = Path(report["records"][0]["prompt_path"]).read_text(encoding="utf-8")
            self.assertIn("diagnostic-only", first_prompt)
            self.assertIn("do not expose its output to verdict_evidence", first_prompt)
            self.assertIn("Do not start GPU jobs", first_prompt)
            self.assertIn("Google-style software engineering discipline", first_prompt)
            self.assertIn("Required CI/harness receipts", first_prompt)

    def test_filter_unknown_probe_fails_closed(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            bank = _write_bank(root)

            with self.assertRaisesRegex(DiagnosticProbeMaterializationPromptError, "UNKNOWN_PROBE"):
                run_diagnostic_probe_materialization_prompt_batch(
                    repo_root=ROOT,
                    failure_signature_bank_manifest=bank,
                    probe_ids=("missing_probe",),
                    output_root=root / "reports" / "autoprobe-prompts",
                )

            self.assertFalse((root / "reports" / "autoprobe-prompts").exists())

    def test_non_diagnostic_work_order_is_rejected(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            bank = _write_bank(root, non_diagnostic=True)

            with self.assertRaisesRegex(DiagnosticProbeMaterializationPromptError, "NOT_DIAGNOSTIC"):
                run_diagnostic_probe_materialization_prompt_batch(
                    repo_root=ROOT,
                    failure_signature_bank_manifest=bank,
                    output_root=root / "reports" / "autoprobe-prompts",
                )

    def test_cli_prints_manifest(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            bank = _write_bank(root)
            output = io.StringIO()

            with redirect_stdout(output):
                code = main(
                    [
                        "batch",
                        "--repo-root",
                        str(ROOT),
                        "--failure-signature-bank-manifest",
                        str(bank),
                        "--probe-id",
                        "probe_a",
                        "--output-root",
                        str(root / "reports" / "autoprobe-prompts"),
                    ]
                )

            self.assertEqual(code, 0)
            payload = json.loads(output.getvalue())
            self.assertEqual(payload["artifact_type"], "wmloop-diagnostic-probe-materialization-prompt-batch-manifest")
            self.assertEqual(payload["prompt_count"], 1)
            self.assertEqual(payload["probe_ids"], ["probe_a"])


def _write_bank(root: Path, *, non_diagnostic: bool = False) -> Path:
    orders = root / "bank" / "diagnostic-probe-work-orders"
    orders.mkdir(parents=True)
    first = orders / "probe_a.json"
    second = orders / "probe_b.json"
    _write_json(first, _work_order("probe_a", non_diagnostic=non_diagnostic))
    _write_json(second, _work_order("probe_b"))
    manifest = {
        "schema_version": 1,
        "artifact_type": "wmloop-failure-signature-bank-manifest",
        "state": "ready",
        "summary": {"p0_probe_order_count": 2},
        "report_path": str(root / "bank" / "failure-signature-bank.json"),
        "markdown_path": str(root / "bank" / "failure-signature-bank.md"),
        "probe_work_order_paths": {"probe_a": str(first), "probe_b": str(second)},
        "tables": {},
        "cas_refs": {},
        "side_effects": {},
        "limitations": [],
    }
    return _write_json(root / "bank" / "manifest.json", manifest)


def _work_order(probe_id: str, *, non_diagnostic: bool = False) -> dict[str, object]:
    return {
        "work_order_id": f"{probe_id}_work_order",
        "environment": "pour_water",
        "signature": "fluid_volume_transport",
        "probe_id": probe_id,
        "role": "verdict" if non_diagnostic else "diagnostic",
        "priority": "P0_rediagnose",
        "signal_contract": "Measure fluid transport as diagnostic-only routing evidence.",
        "allowed_mutation_paths": [f"wmloop/diagnose/probes/{probe_id}.py", f"tests/test_{probe_id}.py"],
        "forbidden_surfaces": ["configs/goal/", "verdict_evidence", "frozen evaluator code"],
        "admission_gates": ["schema_valid_diagnostic_probe_output", "no_verdict_evidence_exposure"],
        "verdict_exposure_allowed": non_diagnostic,
    }


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    unittest.main()
