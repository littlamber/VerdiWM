from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from wmloop.control.onboarding import (
    OnboardingError,
    OnboardingOptions,
    run_onboarding,
    scan_repository,
)


class ModelOnboardingTests(unittest.TestCase):
    def test_sidecar_is_outside_source_and_scan_is_read_only(self) -> None:
        with TemporaryDirectory() as temporary:
            parent = Path(temporary)
            repo = _model_repo(parent / "model")
            before = _snapshot(repo)

            manifest = run_onboarding(
                OnboardingOptions(
                    repo_root=repo,
                    runtime_python=Path(sys.executable),
                    probe_imports=False,
                )
            )

            sidecar = parent / "model.verdiwm-instance"
            self.assertEqual(Path(manifest["sidecar_root"]), sidecar)
            self.assertTrue(
                (sidecar / "generated_connector" / "connector.json").is_file()
            )
            self.assertTrue((sidecar / "onboarding-report.md").is_file())
            self.assertEqual(_snapshot(repo), before)
            self.assertFalse(manifest["optimization_launch_allowed"])

            report = json.loads(
                (sidecar / "onboarding-report.json").read_text(encoding="utf-8")
            )
            self.assertFalse(report["side_effects"]["source_modified"])
            self.assertFalse(report["side_effects"]["gpu_execution_started"])
            self.assertIn("rollout", _discovered_capabilities(report))
            self.assertIn("action_conditioned", _discovered_capabilities(report))
            self.assertEqual(
                report["connector"]["asset_bindings"][0]["state"], "discovered"
            )

    def test_output_inside_source_fails_closed(self) -> None:
        with TemporaryDirectory() as temporary:
            repo = _model_repo(Path(temporary) / "model")

            with self.assertRaisesRegex(
                OnboardingError, "ONBOARDING_OUTPUT_INSIDE_SOURCE"
            ):
                run_onboarding(
                    OnboardingOptions(
                        repo_root=repo,
                        output_root=repo / ".verdiwm-instance",
                        probe_imports=False,
                    )
                )

    def test_missing_runtime_dependency_is_a_stable_blocker(self) -> None:
        with TemporaryDirectory() as temporary:
            repo = _model_repo(Path(temporary) / "model")
            (repo / "requirements.txt").write_text(
                "verdiwm-package-that-does-not-exist-10291==1\n", encoding="utf-8"
            )

            report = scan_repository(
                OnboardingOptions(
                    repo_root=repo,
                    runtime_python=Path(sys.executable),
                    probe_imports=True,
                )
            )

            codes = [blocker["code"] for blocker in report["blockers"]]
            self.assertIn("RUNTIME_UNREADY", codes)
            self.assertIn("RUNTIME_DEPENDENCY_MISSING", codes)
            probe = report["runtime"]["import_probes"][0]
            self.assertEqual(probe["state"], "failed")
            self.assertLessEqual(len(probe["detail"]), 600)

    def test_default_runtime_skips_absent_repository_virtualenvs(self) -> None:
        with TemporaryDirectory() as temporary:
            repo = _model_repo(Path(temporary) / "model")

            report = scan_repository(
                OnboardingOptions(repo_root=repo, probe_imports=False)
            )

            self.assertEqual(
                Path(report["runtime"]["selected_python"]),
                Path(sys.executable).resolve(),
            )
            self.assertEqual(report["runtime"]["state"], "ready")

    def test_declared_candidate_asset_is_bound_without_relaxing_unknown_flags(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            parent = Path(temporary)
            repo = _model_repo(parent / "model")
            data_stat = parent / "stat.json"
            data_stat.write_text('{"count": 1}\n', encoding="utf-8")

            report = scan_repository(
                OnboardingOptions(
                    repo_root=repo,
                    runtime_python=Path(sys.executable),
                    asset_bindings=(("--data-stat-path", data_stat),),
                    additional_asset_parameters=("--data-stat-path",),
                    probe_imports=False,
                )
            )

            binding = next(
                row
                for row in report["connector"]["asset_bindings"]
                if row["parameter"] == "--data-stat-path"
            )
            self.assertEqual(binding["state"], "discovered")
            self.assertEqual(binding["resolved_path"], str(data_stat))
            self.assertFalse(binding["required_for_evaluator"])

            with self.assertRaisesRegex(
                OnboardingError,
                "ASSET_BINDING_PARAMETER_UNKNOWN:--undeclared-path",
            ):
                scan_repository(
                    OnboardingOptions(
                        repo_root=repo,
                        asset_bindings=(("--undeclared-path", data_stat),),
                        additional_asset_parameters=("--data-stat-path",),
                        probe_imports=False,
                    )
                )

    def test_complete_bindings_stop_at_conformance_boundary(self) -> None:
        with TemporaryDirectory() as temporary:
            parent = Path(temporary)
            repo = _model_repo(parent / "model")
            _git_commit(repo)
            evaluator = parent / "evaluator.json"
            evaluator.write_text(
                json.dumps(
                    {
                        "evaluator_id": "frozen_eval_v1",
                        "command": ["python", "scripts/eval_rollout.py"],
                        "input_artifacts": ["checkpoint"],
                        "output_artifacts": ["receipt.json"],
                        "metrics": ["success_rate"],
                        "verifier": "receipt_schema_v1",
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )

            report = scan_repository(
                OnboardingOptions(
                    repo_root=repo,
                    runtime_python=Path(sys.executable),
                    evaluator_contract=evaluator,
                    probe_imports=False,
                )
            )

            self.assertEqual(report["blockers"], [])
            self.assertEqual(report["state"], "ready_for_conformance_smoke")
            self.assertFalse(report["optimization_launch_allowed"])
            self.assertEqual(report["conformance"]["state"], "not_run")
            self.assertEqual(report["connector"]["state"], "ready")

    def test_invalid_evaluator_contract_fails_closed(self) -> None:
        with TemporaryDirectory() as temporary:
            parent = Path(temporary)
            repo = _model_repo(parent / "model")
            evaluator = parent / "evaluator.json"
            evaluator.write_text(
                json.dumps(
                    {
                        "evaluator_id": "eval_v1",
                        "command": "python scripts/eval_rollout.py",
                        "input_artifacts": [],
                        "output_artifacts": ["receipt.json"],
                        "metrics": ["success_rate"],
                        "verifier": "receipt_v1",
                    }
                ),
                encoding="utf-8",
            )

            report = scan_repository(
                OnboardingOptions(
                    repo_root=repo, evaluator_contract=evaluator, probe_imports=False
                )
            )

            self.assertEqual(report["evaluator_contract"]["state"], "blocked")
            self.assertIn("command", report["evaluator_contract"]["error"])
            self.assertIn("input_artifacts", report["evaluator_contract"]["error"])
            self.assertIn(
                "EVALUATOR_CONTRACT_REQUIRED",
                {item["code"] for item in report["blockers"]},
            )

    def test_report_is_deterministic_for_unchanged_repository(self) -> None:
        with TemporaryDirectory() as temporary:
            repo = _model_repo(Path(temporary) / "model")
            options = OnboardingOptions(
                repo_root=repo, runtime_python=Path(sys.executable), probe_imports=False
            )

            first = scan_repository(options)
            second = scan_repository(options)

            self.assertEqual(first, second)


def _model_repo(repo: Path) -> Path:
    (repo / "scripts").mkdir(parents=True)
    (repo / "requirements.txt").write_text("\n", encoding="utf-8")
    (repo / "checkpoint-10.pt").write_bytes(b"checkpoint")
    (repo / "scripts" / "train.py").write_text(
        "def main():\n    return 0\n\nif __name__ == '__main__':\n    raise SystemExit(main())\n",
        encoding="utf-8",
    )
    (repo / "scripts" / "eval_rollout.py").write_text(
        "import argparse\n\ndef evaluate(action):\n    return action\n\nif __name__ == '__main__':\n    parser = argparse.ArgumentParser()\n    parser.add_argument('--action-dim')\n    parser.add_argument('--ckpt-path')\n    evaluate('action')\n",
        encoding="utf-8",
    )
    return repo


def _snapshot(repo: Path) -> dict[str, bytes]:
    return {
        path.relative_to(repo).as_posix(): path.read_bytes()
        for path in sorted(repo.rglob("*"))
        if path.is_file()
    }


def _discovered_capabilities(report: dict[str, object]) -> set[str]:
    return {
        item["capability"]
        for item in report["capabilities"]
        if item["state"] == "discovered"
    }


def _git_commit(repo: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "tests@verdiwm.invalid"], cwd=repo, check=True
    )
    subprocess.run(
        ["git", "config", "user.name", "VerdiWM Tests"], cwd=repo, check=True
    )
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=repo, check=True)


if __name__ == "__main__":
    unittest.main()
