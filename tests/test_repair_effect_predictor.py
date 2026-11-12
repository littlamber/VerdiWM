from __future__ import annotations

import json
from pathlib import Path

import pytest

from wmloop.experiments.repair_effect_predictor import (
    RepairCandidate,
    RepairExample,
    RepairPredictorError,
    RidgeRepairEffectPredictor,
    TrainingScope,
    run_contextual_predictor_replay,
)


def test_ridge_predictor_is_deterministic_and_emits_scope_receipt() -> None:
    examples = [
        RepairExample("a", "p", {"response": 1.0, "compat": 1.0}, 1.0, "positive"),
        RepairExample("a", "n", {"response": -1.0, "compat": 1.0}, -1.0, "harmful"),
        RepairExample("b", "p", {"response": 0.8, "compat": 1.0}, 0.8, "positive"),
        RepairExample("b", "n", {"response": -0.8, "compat": 1.0}, -0.8, "null"),
    ]
    predictor = RidgeRepairEffectPredictor(ridge=0.01)
    receipt = predictor.fit(examples, scope=TrainingScope(("a", "b"), target_environment="target", split_id="outer-1-inner"))
    assert receipt["target_environment"] == "target"
    assert receipt["target_labels_used_for_fit"] is False
    prediction = predictor.predict([RepairCandidate("target", "p", {"response": 0.9, "compat": 1.0})])[0]
    assert prediction.support_state == "in_support"
    assert prediction.expected_delta > 0
    assert prediction.decision in {"screen", "abstain"}


def test_predictor_rejects_outer_target_examples() -> None:
    predictor = RidgeRepairEffectPredictor()
    with pytest.raises(RepairPredictorError, match="EXAMPLE_OUTSIDE_SOURCE_SCOPE"):
        predictor.fit(
            [RepairExample("target", "p", {"x": 1.0}, 1.0, "positive")],
            scope=TrainingScope(("source",), target_environment="target"),
        )


def test_predictor_abstains_outside_training_support() -> None:
    predictor = RidgeRepairEffectPredictor(ridge=0.1, interval_z=0.01)
    predictor.fit(
        [
            RepairExample("a", "p", {"x": 0.0}, 0.1, "positive"),
            RepairExample("b", "p", {"x": 1.0}, 0.2, "positive"),
        ],
        scope=TrainingScope(("a", "b")),
    )
    prediction = predictor.predict([RepairCandidate("target", "p", {"x": 10.0})])[0]
    assert prediction.support_state == "out_of_support"
    assert prediction.decision == "abstain"


def _interaction_examples() -> list[RepairExample]:
    return [
        RepairExample(environment, primitive, {"x": x}, sign * x)
        for environment, x in (("a", -1.0), ("b", 0.0), ("c", 1.0))
        for primitive, sign in (("p", 1), ("n", -1))
    ]


def test_primitive_context_interaction_reverses_preferred_repair() -> None:
    predictor = RidgeRepairEffectPredictor(ridge=0.01)
    predictor.fit(_interaction_examples(), scope=TrainingScope(("a", "b", "c")))
    for x, preferred in ((0.75, "p"), (-0.75, "n")):
        predictions = predictor.predict([
            RepairCandidate("target", primitive, {"x": x}) for primitive in ("p", "n")
        ])
        assert [p.primitive for p in predictions if p.decision == "screen"] == [preferred]
        assert max(predictions, key=lambda p: p.expected_delta).primitive == preferred


@pytest.mark.parametrize("features", [{}, {"other": 1.0}, {"x": 1.0, "extra": 2.0}])
def test_prediction_rejects_missing_and_extra_features(features) -> None:
    predictor = RidgeRepairEffectPredictor()
    predictor.fit(_interaction_examples(), scope=TrainingScope(("a", "b", "c")))
    with pytest.raises(RepairPredictorError, match="FEATURE_SCHEMA"):
        predictor.predict([RepairCandidate("target", "p", features)])


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, 0, True])
def test_invalid_hyperparameters(value) -> None:
    with pytest.raises(RepairPredictorError):
        RidgeRepairEffectPredictor(ridge=value)


def test_source_error_bands_detect_environment_conflict() -> None:
    predictor = RidgeRepairEffectPredictor(ridge=0.01)
    predictor.fit([
        RepairExample("a", "p", {"x": 0.0}, 1.0),
        RepairExample("b", "p", {"x": 0.0}, -1.0),
    ], scope=TrainingScope(("a", "b")))
    prediction = predictor.predict([RepairCandidate("target", "p", {"x": 0.0})])[0]
    assert prediction.interval[0] < -1
    assert prediction.interval[1] > 1
    assert prediction.decision == "abstain"


def test_unseen_primitive_and_single_environment_abstain() -> None:
    predictor = RidgeRepairEffectPredictor()
    predictor.fit([RepairExample("a", "p", {"x": 1.0}, 1.0)], scope=TrainingScope(("a",)))
    predictions = predictor.predict([
        RepairCandidate("target", primitive, {"x": 1.0}) for primitive in ("p", "new")
    ])
    assert [p.support_state for p in predictions] == ["insufficient_source_environments", "unseen_primitive"]
    assert all(p.decision == "abstain" for p in predictions)


def test_failed_refit_cannot_use_stale_model() -> None:
    predictor = RidgeRepairEffectPredictor()
    predictor.fit(_interaction_examples(), scope=TrainingScope(("a", "b", "c")))
    with pytest.raises(RepairPredictorError):
        predictor.fit([], scope=TrainingScope(("a",)))
    with pytest.raises(RepairPredictorError, match="NOT_FIT"):
        predictor.predict([RepairCandidate("target", "p", {"x": 0.5})])


FIXTURE = Path(__file__).parent / "fixtures" / "contextual_predictor"


def _replay(tmp_path, *, plan=None, examples=None, labels=None, name="run"):
    paths = {}
    for key, payload in (("plan", plan), ("examples", examples), ("labels", labels)):
        paths[key] = FIXTURE / f"{key}.json"
        if payload is not None:
            paths[key] = tmp_path / f"{name}-{key}.json"
            paths[key].write_text(json.dumps(payload), encoding="utf-8")
    result = run_contextual_predictor_replay(
        examples_path=paths["examples"], plan_path=paths["plan"],
        labels_path=paths["labels"], output_root=tmp_path / name,
    )
    return json.loads(Path(result["report_path"]).read_text()), result


def _fixture(name):
    return json.loads((FIXTURE / f"{name}.json").read_text())


def test_replay_selects_context_specific_repair_and_writes_bundle(tmp_path) -> None:
    report, result = _replay(tmp_path)
    assert report["top1_positive_hit"] == 1.0
    assert report["selection_regret"] == 0.0
    assert report["random_baseline"]["top1_positive"] == 0.5
    assert [row["selected_primitive"] for row in report["folds"]] == ["p", "n"]
    assert report["target_labels_used_for_fit"] is False
    assert all(r["target_labels_used_for_fit"] is False for r in report["predictor_receipts"])
    manifest = json.loads(Path(result["manifest_path"]).read_text())
    assert "predictor-replay.json" in manifest["cas_refs"]
    assert len(report["prediction_digest_before_scoring"]) == 64


def test_target_labels_cannot_change_predictions_or_artifacts(tmp_path) -> None:
    before, _ = _replay(tmp_path, name="before")
    labels = _fixture("labels")
    for row in labels["labels"]:
        row["positive"] = not row["positive"]
        row["delta_candidate_minus_baseline"] *= -1
    after, _ = _replay(tmp_path, labels=labels, name="after")
    assert before["prediction_digest_before_scoring"] == after["prediction_digest_before_scoring"]
    assert before["predictor_receipts"] == after["predictor_receipts"]
    assert after["top1_positive_hit"] == 0.0


def test_outer_examples_are_excluded_and_each_fold_refits(tmp_path) -> None:
    baseline, _ = _replay(tmp_path, name="baseline")
    examples = _fixture("examples")
    examples["examples"].append({
        "environment": "target-plus", "primitive": "p", "features": {"x": 999}, "effect": 999,
    })
    changed, _ = _replay(tmp_path, examples=examples, name="changed")
    assert changed["predictor_receipts"] == baseline["predictor_receipts"]
    plan = _fixture("plan")
    plan["folds"][1]["source_environments"] = ["b", "c"]
    subset, _ = _replay(tmp_path, plan=plan, name="subset")
    receipts = subset["predictor_receipts"]
    assert receipts[0]["example_count"] == 6
    assert receipts[1]["example_count"] == 4
    assert receipts[0]["training_examples_digest"] != receipts[1]["training_examples_digest"]


def test_all_abstentions_do_not_select_a_candidate_or_hide_regret(tmp_path) -> None:
    plan = _fixture("plan")
    for fold in plan["folds"]:
        for candidate in fold["candidates"]:
            candidate["features"]["x"] = 100
    report, _ = _replay(tmp_path, plan=plan)
    assert report["accepted_coverage"] == 0
    assert report["evaluated_fold_count"] == 0
    assert report["selection_regret"] == 1
    assert report["harmful_selection"] is None
    assert all(row["selected_primitive"] is None for row in report["folds"])


@pytest.mark.parametrize("case,code", [
    ("target_source", "TARGET_LEAK"), ("source_string", "PLAN_INVALID"),
    ("empty_folds", "REPLAY_INPUT_INVALID"), ("duplicate_fold", "DUPLICATE_FOLD"),
    ("duplicate_candidate", "DUPLICATE_CANDIDATE"), ("missing_source", "SOURCE_EXAMPLES_MISSING"),
])
def test_invalid_plans_fail_closed(tmp_path, case, code) -> None:
    plan = _fixture("plan")
    fold = plan["folds"][0]
    if case == "target_source":
        fold["source_environments"].append(fold["target_environment"])
    elif case == "source_string":
        fold["source_environments"] = "abc"
    elif case == "empty_folds":
        plan["folds"] = []
    elif case == "duplicate_fold":
        plan["folds"].append(fold)
    elif case == "duplicate_candidate":
        fold["candidates"].append(fold["candidates"][0])
    elif case == "missing_source":
        fold["source_environments"].append("missing")
    with pytest.raises(RepairPredictorError, match=code):
        _replay(tmp_path, plan=plan)
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("case,code", [
    ("missing", "TARGET_LABEL_MISSING"), ("unsettled", "LABEL_NOT_SETTLED"),
    ("duplicate", "DUPLICATE_LABEL"), ("nonfinite", "NONFINITE_INPUT"),
])
def test_invalid_labels_fail_closed(tmp_path, case, code) -> None:
    labels = _fixture("labels")
    if case == "missing":
        labels["labels"].pop(1)  # Even an unselected candidate needs a label.
    elif case == "unsettled":
        labels["labels"][0]["settled"] = False
    elif case == "duplicate":
        labels["labels"].append(labels["labels"][0])
    elif case == "nonfinite":
        labels["labels"][0]["delta_candidate_minus_baseline"] = float("nan")
    with pytest.raises(RepairPredictorError, match=code):
        _replay(tmp_path, labels=labels)


def test_cli_executes_replay(tmp_path, capsys) -> None:
    from wmloop.cli import main

    assert main([
        "research", "evaluate-predictor", "--examples", str(FIXTURE / "examples.json"),
        "--plan", str(FIXTURE / "plan.json"), "--labels", str(FIXTURE / "labels.json"),
        "--output", str(tmp_path / "cli"),
    ]) == 0
    assert json.loads(capsys.readouterr().out)["fold_count"] == 2
