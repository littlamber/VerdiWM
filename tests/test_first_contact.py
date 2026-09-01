from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from wmloop.control.first_contact import explain_blocker, initialize_project
from wmloop.control.project_config import load_project_config


class FirstContactTests(unittest.TestCase):
    def test_init_discovers_conventional_directories_and_writes_config(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "model").mkdir()
            (root / "dataset").mkdir()
            result = initialize_project(root=root, goal="提升长时域预测稳定性")
            self.assertEqual(result["state"], "ready")
            config = load_project_config(cwd=root)
            self.assertEqual(Path(config.values["model"]), root / "model")
            self.assertEqual(Path(config.values["data"]), root / "dataset")
            self.assertEqual(config.values["goal"], "提升长时域预测稳定性")

    def test_missing_inputs_are_human_readable_and_do_not_write(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            result = initialize_project(root=root)
            self.assertEqual(result["state"], "needs_input")
            self.assertTrue(result["missing"])
            self.assertFalse((root / "verdiwm.toml").exists())
            self.assertTrue(all("message" in item for item in result["missing"]))

    def test_existing_project_requires_explicit_force(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "model").mkdir()
            (root / "data").mkdir()
            initialize_project(root=root, goal="目标")
            with self.assertRaisesRegex(ValueError, "PROJECT_FILE_EXISTS"):
                initialize_project(root=root, goal="新目标")

    def test_internal_adapter_error_is_explained_in_user_language(self) -> None:
        result = explain_blocker("ADAPTER_PROFILE_NOT_FOUND:no exact match")
        self.assertEqual(result["code"], "ADAPTER_PROFILE_NOT_FOUND")
        self.assertIn("运行连接器", result["error"])
        self.assertIn("ADAPTER_PROFILE_NOT_FOUND", result["detail"])


if __name__ == "__main__":
    unittest.main()
