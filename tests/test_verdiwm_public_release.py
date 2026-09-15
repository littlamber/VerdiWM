from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.export.build_github_staging import (
    GithubStagingError,
    audit_release_tree,
    build_github_staging,
)
from scripts.export.validate_portrait_first_public_example import (
    validate_portrait_first_public_example,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
ONBOARDING_EXAMPLE = REPO_ROOT / "examples" / "portrait_first_minimal_loop_v1"


class GithubStagingTests(unittest.TestCase):
    def test_onboarding_example_is_cpu_only_and_valid(self) -> None:
        report = validate_portrait_first_public_example(ONBOARDING_EXAMPLE)
        self.assertEqual(report["state"], "ready")
        self.assertFalse(report["authority"]["gpu_authority"])

    def test_audit_rejects_generated_artifacts_and_local_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "bad.txt").write_text("/" + "mnt" + "/private/run", encoding="utf-8")
            (root / "model.pt").write_bytes(b"weight")
            (root / "examples" / "result").mkdir(parents=True)
            (root / "examples" / "result" / "receipt.json").write_text("{}", encoding="utf-8")
            findings = audit_release_tree(root)
            self.assertIn("local_path:bad.txt", findings)
            self.assertIn("blocked_extension:model.pt", findings)
            self.assertIn("unexpected_example:examples/result/receipt.json", findings)

    def test_builder_refuses_existing_destination(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(GithubStagingError, "OUTPUT_EXISTS"):
                build_github_staging(source_root=REPO_ROOT, output_root=Path(temp))

    def test_builder_emits_source_only_repository(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "release"
            audit = build_github_staging(source_root=REPO_ROOT, output_root=output)

            self.assertEqual(audit["state"], "ready")
            self.assertTrue(all(audit["checks"].values()))
            self.assertTrue((output / "pyproject.toml").is_file())
            self.assertTrue((output / "wmloop" / "cli.py").is_file())
            self.assertTrue((output / "configs" / "schemas" / "model_irg.schema.json").is_file())
            self.assertTrue((output / "docs" / "COMMUNITY_BUNDLES.md").is_file())
            self.assertTrue((output / "tests" / "test_model_irg.py").is_file())
            self.assertTrue((output / ONBOARDING_EXAMPLE.relative_to(REPO_ROOT) / "run.py").is_file())

            example_files = sorted(
                path.relative_to(output).as_posix()
                for path in (output / "examples").rglob("*")
                if path.is_file()
            )
            self.assertEqual(
                example_files,
                [
                    "examples/portrait_first_minimal_loop_v1/README.md",
                    "examples/portrait_first_minimal_loop_v1/run.py",
                ],
            )
            self.assertFalse((output / "figures").exists())
            self.assertFalse((output / ".audit-state").exists())
            self.assertFalse((output / "RELEASE_AUDIT.json").exists())
            self.assertFalse((output / "MANIFEST.sha256").exists())
            self.assertFalse((output / ".verdiwm-public-release").exists())
            self.assertEqual(audit_release_tree(output), [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
