from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from wmloop.control.first_contact import explain_blocker, initialize_project, inspect_project
from wmloop.control.onboarding_assistant import build_onboarding_questionnaire, write_onboarding_questionnaire
from wmloop.control.project_config import load_project_config


class FirstContactTests(unittest.TestCase):
    def test_readiness_combines_separate_source_and_weight_directories(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            model = root / "weights"
            source = root / "source"
            data = root / "data"
            model.mkdir()
            source.mkdir()
            data.mkdir()
            (model / "model.safetensors").write_bytes(b"weights")
            (source / "train.py").write_text(
                "if __name__ == '__main__':\n    print('train')\n",
                encoding="utf-8",
            )
            (source / "evaluate.py").write_text(
                "if __name__ == '__main__':\n    print('evaluate')\n",
                encoding="utf-8",
            )
            report = inspect_project(
                root=root,
                model=str(model),
                source=str(source),
                data=str(data),
            )
            codes = {item["code"] for item in report["blockers"]}
            self.assertNotIn("CHECKPOINT_MISSING", codes)
            self.assertEqual(report["model"], str(model))
            self.assertEqual(report["source"], str(source))

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

    def test_readiness_requires_an_explicit_evaluator(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            model = root / "model"
            model.mkdir()
            (model / ".git").mkdir()
            (model / "infer.py").write_text("def infer(x):\n    return x\n", encoding="utf-8")
            (model / "evaluate.py").write_text("def evaluate(x):\n    return 1\n", encoding="utf-8")
            (model / "checkpoint.pt").write_bytes(b"fixture")
            (root / "data").mkdir()
            report = inspect_project(root=root)
        self.assertEqual(report["state"], "needs_input")
        codes = {item["code"] for item in report["blockers"]}
        self.assertIn("EVALUATOR_CONTRACT_REQUIRED", codes)
        self.assertFalse(report["side_effects"]["gpu_execution_started"])

    def test_init_persists_runtime_and_evaluator_bindings(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "model").mkdir()
            (root / "data").mkdir()
            evaluator = root / "evaluator.json"
            evaluator.write_text("{}", encoding="utf-8")
            initialize_project(
                root=root,
                goal="目标",
                evaluator_contract=str(evaluator),
            )
            config = load_project_config(cwd=root)
        self.assertEqual(Path(config.values["evaluator_contract"]), evaluator)

    def test_relative_api_style_inputs_are_resolved_from_project_root(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "model").mkdir()
            (root / "data").mkdir()
            report = inspect_project(root=root, model="./model", data="./data")
        self.assertEqual(report["model"], str(root / "model"))
        self.assertEqual(report["data"], str(root / "data"))
        self.assertNotIn("MODEL_PATH_REQUIRED", {item["code"] for item in report["blockers"]})

    def test_onboarding_questionnaire_is_actionable_and_stays_outside_model(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            model = root / "model"
            model.mkdir()
            (root / "data").mkdir()
            questionnaire = build_onboarding_questionnaire(root=root)
            output = write_onboarding_questionnaire(root / ".verdiwm" / "questions.json", questionnaire)
            self.assertTrue(output.is_file())
            self.assertTrue(questionnaire["questions"])
            with self.assertRaisesRegex(ValueError, "INSIDE_MODEL"):
                write_onboarding_questionnaire(model / "questions.json", questionnaire)


if __name__ == "__main__":
    unittest.main()
