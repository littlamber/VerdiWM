"""Source-only contextual repair prediction and held-out CPU replay.

Intervals use leave-one-source-environment-out errors. They are empirical error
bands, not calibrated confidence guarantees under a new environment shift.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from wmloop.experiments._artifacts import canonical_json, write_bundle


class RepairPredictorError(ValueError):
    """Input violates the predictor or split contract."""


def _digest(payload: object) -> str:
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def _identifier(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RepairPredictorError("REPAIR_PREDICTOR_IDENTIFIER_INVALID")
    return value


def _number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RepairPredictorError("REPAIR_PREDICTOR_NUMBER_INVALID")
    if not np.isfinite(value):
        raise RepairPredictorError("REPAIR_PREDICTOR_NONFINITE_INPUT")
    return float(value)


def _features(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping) or not value:
        raise RepairPredictorError("REPAIR_PREDICTOR_FEATURE_SCHEMA_EMPTY")
    return {_identifier(key): _number(number) for key, number in value.items()}


@dataclass(frozen=True)
class TrainingScope:
    source_environments: tuple[str, ...]
    target_environment: str | None = None
    split_id: str = "inner-source"

    def validate(self) -> None:
        if (
            not isinstance(self.source_environments, tuple)
            or not self.source_environments
            or len(set(self.source_environments)) != len(self.source_environments)
        ):
            raise RepairPredictorError("REPAIR_PREDICTOR_SOURCE_SCOPE_INVALID")
        for environment in self.source_environments:
            _identifier(environment)
        _identifier(self.split_id)
        if self.target_environment is not None:
            _identifier(self.target_environment)
            if self.target_environment in self.source_environments:
                raise RepairPredictorError("REPAIR_PREDICTOR_TARGET_LEAK")


@dataclass(frozen=True)
class RepairExample:
    environment: str
    primitive: str
    features: Mapping[str, float]
    effect: float
    effect_class: str = "settled"


@dataclass(frozen=True)
class RepairCandidate:
    environment: str
    primitive: str
    features: Mapping[str, float]


@dataclass(frozen=True)
class RepairPrediction:
    primitive: str
    expected_delta: float
    interval: tuple[float, float]
    support_state: str
    decision: str


class RidgeRepairEffectPredictor:
    """Ridge with primitive indicators and primitive/context interactions.

    Features must have exactly the same explicit schema; missing measurements
    are never silently treated as zero. Only positive lower bounds may screen.
    """

    def __init__(
        self, *, ridge: float = 1.0, support_z: float = 3.0, interval_z: float = 1.96
    ) -> None:
        self.ridge, self.support_z, self.interval_z = map(
            _number, (ridge, support_z, interval_z)
        )
        if min(self.ridge, self.support_z, self.interval_z) <= 0:
            raise RepairPredictorError("REPAIR_PREDICTOR_HYPERPARAMETER_INVALID")
        self.feature_names: tuple[str, ...] = ()
        self.primitive_names: tuple[str, ...] = ()
        self.receipt: dict[str, Any] | None = None

    def _design(self, matrix: np.ndarray, primitives: Sequence[str]) -> np.ndarray:
        indicators = np.asarray(
            [[float(value == name) for name in self.primitive_names] for value in primitives]
        )
        interactions = (indicators[:, :, None] * matrix[:, None, :]).reshape(len(matrix), -1)
        return np.column_stack([np.ones(len(matrix)), matrix, indicators, interactions])

    def _solve(
        self, matrix: np.ndarray, primitives: Sequence[str], target: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mean, scale = matrix.mean(axis=0), matrix.std(axis=0)
        scale[scale < 1e-12] = 1.0
        design = self._design((matrix - mean) / scale, primitives)
        regularizer = np.eye(design.shape[1]) * self.ridge
        regularizer[0, 0] = 0.0
        coef = np.linalg.solve(design.T @ design + regularizer, design.T @ target)
        return mean, scale, coef

    def fit(self, examples: Sequence[RepairExample], *, scope: TrainingScope) -> dict[str, Any]:
        # A failed refit must not leave an earlier model usable under a new scope.
        self.receipt = None
        scope.validate()
        if not examples:
            raise RepairPredictorError("REPAIR_PREDICTOR_NO_EXAMPLES")
        if any(example.environment not in scope.source_environments for example in examples):
            raise RepairPredictorError("REPAIR_PREDICTOR_EXAMPLE_OUTSIDE_SOURCE_SCOPE")
        if set(scope.source_environments) != {example.environment for example in examples}:
            raise RepairPredictorError("REPAIR_PREDICTOR_SOURCE_EXAMPLES_MISSING")
        for example in examples:
            _identifier(example.primitive)
            _features(example.features)
            _number(example.effect)
            if example.effect_class not in {"settled", "positive", "null", "harmful"}:
                raise RepairPredictorError("REPAIR_PREDICTOR_EFFECT_NOT_SETTLED")
        names = tuple(sorted(examples[0].features))
        if any(set(example.features) != set(names) for example in examples):
            raise RepairPredictorError("REPAIR_PREDICTOR_FEATURE_SCHEMA_MISMATCH")
        self.feature_names = names
        self.primitive_names = tuple(sorted({example.primitive for example in examples}))
        matrix = np.asarray([[example.features[name] for name in names] for example in examples])
        target = np.asarray([example.effect for example in examples])
        primitives = [example.primitive for example in examples]
        self._mean, self._scale, self._coef = self._solve(matrix, primitives, target)
        self._support: dict[str, dict[str, Any]] = {}
        for primitive in self.primitive_names:
            subset = matrix[[value == primitive for value in primitives]]
            scale = subset.std(axis=0)
            # Constant source coordinates supply no evidence for extrapolation.
            scale[scale < 1e-12] = 0.0
            self._support[primitive] = {
                "low": subset.min(axis=0).tolist(),
                "high": subset.max(axis=0).tolist(),
                "scale": scale.tolist(),
                "environment_count": len({
                    example.environment for example in examples if example.primitive == primitive
                }),
            }

        # Refit preprocessing and coefficients inside each source-only fold.
        errors: dict[str, list[float]] = {primitive: [] for primitive in self.primitive_names}
        for held_out in sorted(scope.source_environments):
            train = [i for i, example in enumerate(examples) if example.environment != held_out]
            valid = [i for i, example in enumerate(examples) if example.environment == held_out]
            if not train:
                continue
            train_primitives = [primitives[i] for i in train]
            mean, scale, coef = self._solve(matrix[train], train_primitives, target[train])
            estimates = self._design(
                (matrix[valid] - mean) / scale, [primitives[i] for i in valid]
            ) @ coef
            for index, estimate in zip(valid, estimates, strict=True):
                if primitives[index] in train_primitives:
                    errors[primitives[index]].append(float(target[index] - estimate))
        self._error_scales = {
            primitive: float(np.sqrt(np.mean(np.square(values)))) if values else 0.0
            for primitive, values in errors.items()
        }
        self.receipt = {
            "schema_version": 1,
            "artifact_type": "verdiwm-repair-effect-predictor",
            "predictor": "ridge-primitive-context-interactions-v1",
            "ridge": self.ridge,
            "support_z": self.support_z,
            "interval_z": self.interval_z,
            "feature_names": list(names),
            "primitive_names": list(self.primitive_names),
            "example_count": len(examples),
            "source_environments": sorted(scope.source_environments),
            "target_environment": scope.target_environment,
            "split_id": scope.split_id,
            "training_scope_digest": _digest(asdict(scope)),
            "source_environments_digest": _digest(sorted(scope.source_environments)),
            "training_examples_digest": _digest([asdict(example) for example in examples]),
            "feature_schema_digest": _digest(names),
            "target_labels_used_for_fit": False,
            "source_labels_used_for_fit": True,
            "interval_method": "leave-one-source-environment-out RMSE by primitive",
            "interval_coverage_guarantee": False,
            "calibration_example_counts": {key: len(value) for key, value in errors.items()},
        }
        return dict(self.receipt)

    def predict(self, candidates: Sequence[RepairCandidate]) -> list[RepairPrediction]:
        if self.receipt is None:
            raise RepairPredictorError("REPAIR_PREDICTOR_NOT_FIT")
        predictions = []
        for candidate in candidates:
            _identifier(candidate.environment)
            _identifier(candidate.primitive)
            features = _features(candidate.features)
            if set(features) != set(self.feature_names):
                raise RepairPredictorError("REPAIR_PREDICTOR_FEATURE_SCHEMA_MISMATCH")
            if self.receipt["target_environment"] not in {None, candidate.environment}:
                raise RepairPredictorError("REPAIR_PREDICTOR_TARGET_SCOPE_MISMATCH")
            vector = np.asarray([features[name] for name in self.feature_names])
            design = self._design(((vector - self._mean) / self._scale)[None, :], [candidate.primitive])
            estimate = float((design @ self._coef)[0])
            support = self._support.get(candidate.primitive)
            if support is None:
                state = "unseen_primitive"
            elif support["environment_count"] < 2:
                state = "insufficient_source_environments"
            else:
                margin = self.support_z * np.asarray(support["scale"]) + 1e-12
                outside = np.any(vector < np.asarray(support["low"]) - margin) or np.any(
                    vector > np.asarray(support["high"]) + margin
                )
                state = "out_of_support" if outside else "in_support"
            half = self.interval_z * max(self._error_scales.get(candidate.primitive, 0.0), 1e-12)
            interval = (estimate - half, estimate + half)
            if not np.isfinite([estimate, *interval]).all():
                raise RepairPredictorError("REPAIR_PREDICTOR_NONFINITE_PREDICTION")
            decision = "screen" if state == "in_support" and interval[0] > 0 else "abstain"
            predictions.append(RepairPrediction(candidate.primitive, estimate, interval, state, decision))
        return predictions

    def artifact(self) -> dict[str, Any]:
        if self.receipt is None:
            raise RepairPredictorError("REPAIR_PREDICTOR_NOT_FIT")
        return {
            **self.receipt,
            "mean": self._mean.tolist(),
            "scale": self._scale.tolist(),
            "coef": self._coef.tolist(),
            "design_columns": "intercept, context, primitive indicators, primitive-major interactions",
            "error_scales": self._error_scales,
            "support": self._support,
        }


def run_contextual_predictor_replay(
    *, examples_path: str | Path, plan_path: str | Path,
    labels_path: str | Path, output_root: str | Path,
) -> dict[str, Any]:
    """Freeze predictions for every outer fold before loading evaluation labels."""
    examples_payload = _load_mapping(examples_path, "REPAIR_PREDICTOR_EXAMPLES_INVALID")
    plan = _load_mapping(plan_path, "REPAIR_PREDICTOR_PLAN_INVALID")
    raw_examples, folds = examples_payload.get("examples"), plan.get("folds")
    if not isinstance(raw_examples, list) or not isinstance(folds, list) or not folds:
        raise RepairPredictorError("REPAIR_PREDICTOR_REPLAY_INPUT_INVALID")
    examples = [_example_from_mapping(row) for row in raw_examples]
    groups = plan.get("baseline_feature_groups", {})
    if not isinstance(groups, Mapping) or set(groups) - {"metadata", "raw_response", "irg"}:
        raise RepairPredictorError("REPAIR_PREDICTOR_BASELINE_FEATURE_GROUP_INVALID")
    for names in groups.values():
        if (
            not isinstance(names, list) or not names
            or any(not isinstance(name, str) for name in names)
            or len(set(names)) != len(names)
        ):
            raise RepairPredictorError("REPAIR_PREDICTOR_BASELINE_FEATURE_GROUP_INVALID")
    rows: list[dict[str, Any]] = []
    artifacts = []
    baseline_rows: dict[str, list[dict[str, Any]]] = {name: [] for name in groups}
    baseline_artifacts: dict[str, list[dict[str, Any]]] = {name: [] for name in groups}
    fold_ids: set[str] = set()
    for fold in folds:
        if not isinstance(fold, Mapping):
            raise RepairPredictorError("REPAIR_PREDICTOR_PLAN_INVALID")
        target = _identifier(fold.get("target_environment"))
        sources = fold.get("source_environments")
        candidates_raw = fold.get("candidates")
        if not isinstance(sources, list) or not isinstance(candidates_raw, list) or not candidates_raw:
            raise RepairPredictorError("REPAIR_PREDICTOR_PLAN_INVALID")
        scope = TrainingScope(tuple(sources), target, _identifier(fold.get("fold_id", target)))
        scope.validate()
        if scope.split_id in fold_ids:
            raise RepairPredictorError("REPAIR_PREDICTOR_DUPLICATE_FOLD")
        fold_ids.add(scope.split_id)
        source_examples = [example for example in examples if example.environment in sources]
        candidates = [_candidate_from_mapping(row, target) for row in candidates_raw]
        if len({candidate.primitive for candidate in candidates}) != len(candidates):
            raise RepairPredictorError("REPAIR_PREDICTOR_DUPLICATE_CANDIDATE")
        row, artifact = _predict_fold(source_examples, candidates, scope, fold)
        artifacts.append(artifact)
        rows.append(row)
        for name, feature_names in groups.items():
            if any(
                not set(feature_names) <= set(item.features)
                for item in [*source_examples, *candidates]
            ):
                raise RepairPredictorError("REPAIR_PREDICTOR_BASELINE_FEATURE_MISSING")
            subset_examples = [RepairExample(
                example.environment, example.primitive,
                {key: example.features[key] for key in feature_names},
                example.effect, example.effect_class,
            ) for example in source_examples]
            subset_candidates = [RepairCandidate(
                candidate.environment, candidate.primitive,
                {key: candidate.features[key] for key in feature_names},
            ) for candidate in candidates]
            baseline_row, baseline_artifact = _predict_fold(
                subset_examples, subset_candidates, scope, fold
            )
            baseline_row["candidate_pool_digest"] = row["candidate_pool_digest"]
            baseline_rows[name].append(baseline_row)
            baseline_artifacts[name].append(baseline_artifact)

    prediction_digest = _digest({"contextual": rows, **baseline_rows})
    labels_payload = _load_mapping(labels_path, "REPAIR_PREDICTOR_LABELS_INVALID")
    labels = _label_map(labels_payload.get("labels"))
    for arm_rows in [rows, *baseline_rows.values()]:
        for row in arm_rows:
            _score_fold(row, labels)
    evaluated = [row for row in rows if row["state"] == "evaluated"]
    report = {
        "schema_version": 1,
        "artifact_type": "verdiwm-contextual-repair-predictor-replay",
        "state": "ready" if len(evaluated) == len(rows) else "partial",
        "claim_boundary": (
            "Source-only CPU replay. Empirical source error bands have no coverage guarantee. "
            "Outer labels score frozen decisions only; this run cannot authorize deployment or tune routing."
        ),
        "fold_count": len(rows),
        "evaluated_fold_count": len(evaluated),
        "abstained_fold_count": len(rows) - len(evaluated),
        "accepted_coverage": len(evaluated) / len(rows),
        "top1_positive_hit": float(np.mean([row["top1_positive_hit"] for row in rows])),
        "selection_regret": float(np.mean([row["selection_regret"] for row in rows])),
        "negative_selection": float(np.mean([row["negative_selection"] for row in evaluated])) if evaluated else None,
        "harmful_selection": float(np.mean([row["harmful_selection"] for row in evaluated])) if evaluated else None,
        "metric_contract": "Top1/regret include abstentions as no hit; negative/harmful rates condition on selection.",
        "random_baseline": {
            name: float(np.mean([row[f"random_{name}_expectation"] for row in rows]))
            for name in ("top1_positive", "selection_regret", "harmful_selection")
        },
        "prediction_digest_before_scoring": prediction_digest,
        "input_digests": {"examples": _digest(examples_payload), "plan": _digest(plan), "labels": _digest(labels_payload)},
        "target_labels_used_for_fit": False,
        "target_labels_used_for_selection": False,
        "routing_recommendation": "insufficient_evidence",
        "fixture_only": any(payload.get("fixture_only") is True for payload in (examples_payload, plan, labels_payload)),
        "comparison_contract": (
            "Feature ablations of the same ridge learner, source examples, outer folds and candidate pool. "
            "Each arm recommends at most one screen per fold. No GPU runtime or first-positive cost measured. "
            "raw_response/irg names denote supplied feature groups, not the legacy nearest-neighbor selectors."
        ),
        "comparison": {
            name: _aggregate_arm(arm_rows)
            for name, arm_rows in {"contextual": rows, **baseline_rows}.items()
        },
        "baseline_folds": baseline_rows,
        "baseline_predictor_receipts": baseline_artifacts,
        "folds": rows,
        "predictor_receipts": artifacts,
    }
    destination = Path(output_root).resolve()
    return write_bundle(
        output_root=destination,
        files={
            "predictor-replay.json": canonical_json(report),
            "inputs/examples.json": canonical_json(examples_payload),
            "inputs/plan.json": canonical_json(plan),
            "inputs/labels.json": canonical_json(labels_payload),
            "predictor-replay.md": _markdown_report(report).encode("utf-8"),
        },
        manifest_fields={
            "artifact_type": "verdiwm-contextual-repair-predictor-replay-manifest",
            "state": report["state"],
            "report_path": str(destination / "predictor-replay.json"),
            "fold_count": len(rows),
            "top1_positive_hit": report["top1_positive_hit"],
            "selection_regret": report["selection_regret"],
        },
    )


def _predict_fold(
    examples: Sequence[RepairExample], candidates: Sequence[RepairCandidate],
    scope: TrainingScope, config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    predictor = RidgeRepairEffectPredictor(
        ridge=config.get("ridge", 1.0), support_z=config.get("support_z", 3.0),
        interval_z=config.get("interval_z", 1.96),
    )
    predictor.fit(examples, scope=scope)
    predictions = predictor.predict(candidates)
    eligible = sorted(
        (prediction for prediction in predictions if prediction.decision == "screen"),
        key=lambda prediction: (-prediction.interval[0], prediction.primitive),
    )
    artifact = predictor.artifact()
    selected = eligible[0] if eligible else None
    return {
        "fold_id": scope.split_id,
        "target_environment": scope.target_environment,
        "state": "evaluated" if selected else "abstained",
        "selected_primitive": selected.primitive if selected else None,
        "selected_decision": "screen" if selected else "abstain",
        "abstention_reason": None if selected else "no_supported_positive_lower_bound",
        "candidate_count": len(candidates),
        "candidate_pool_digest": _digest([asdict(candidate) for candidate in candidates]),
        "training_examples_digest": artifact["training_examples_digest"],
        "predictor_artifact_digest": _digest(artifact),
        "predictions": [asdict(prediction) for prediction in predictions],
    }, artifact


def _score_fold(row: dict[str, Any], labels: Mapping[tuple[str, str], Any]) -> None:
    target = row["target_environment"]
    pool = [prediction["primitive"] for prediction in row["predictions"]]
    if any((target, primitive) not in labels for primitive in pool):
        raise RepairPredictorError("REPAIR_PREDICTOR_TARGET_LABEL_MISSING")
    effects = [labels[(target, primitive)] for primitive in pool]
    selected = labels.get((target, row["selected_primitive"]))
    any_positive = any(effect["positive"] for effect in effects)
    hit = bool(selected and selected["positive"])
    covered = [
        prediction["interval"][0] <= labels[(target, prediction["primitive"])]["effect"]
        <= prediction["interval"][1] for prediction in row["predictions"]
        if prediction["support_state"] == "in_support"
    ]
    row.update({
        "selected_target_positive": selected["positive"] if selected else None,
        "top1_positive_hit": float(hit),
        "selection_regret": float(any_positive and not hit),
        "negative_selection": float(not selected["positive"]) if selected else None,
        "harmful_selection": float(selected["effect"] < 0) if selected else None,
        "random_top1_positive_expectation": float(np.mean([effect["positive"] for effect in effects])),
        "random_selection_regret_expectation": float(np.mean([
            any_positive and not effect["positive"] for effect in effects
        ])),
        "random_harmful_selection_expectation": float(np.mean([effect["effect"] < 0 for effect in effects])),
        "interval_coverage": float(np.mean(covered)) if covered else None,
    })


def _aggregate_arm(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = {}
    for metric in ("top1_positive_hit", "selection_regret", "negative_selection", "harmful_selection", "interval_coverage"):
        values = [row[metric] for row in rows if row[metric] is not None]
        metrics[metric] = float(np.mean(values)) if values else None
    return {
        **metrics, "fold_count": len(rows),
        "accepted_coverage": sum(row["state"] == "evaluated" for row in rows) / len(rows),
    }


def _markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Contextual Repair Predictor Replay", "",
        "Synthetic fixture only." if report["fixture_only"] else report["claim_boundary"], "",
        report["comparison_contract"], "",
        "| Arm | Top-1 | Regret | Accepted coverage | Harmful selection |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, metrics in report["comparison"].items():
        values = [
            f"{metrics[key]:.4f}" if metrics[key] is not None else "n/a"
            for key in ("top1_positive_hit", "selection_regret", "accepted_coverage", "harmful_selection")
        ]
        lines.append("| " + " | ".join([name, *values]) + " |")
    random = report["random_baseline"]
    lines.append(
        f"| random expectation | {random['top1_positive']:.4f} | "
        f"{random['selection_regret']:.4f} | 1.0000 | {random['harmful_selection']:.4f} |"
    )
    lines.extend(["", report["metric_contract"], "", "Routing: insufficient_evidence.", ""])
    return "\n".join(lines)


def _load_mapping(path: str | Path, code: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RepairPredictorError(code) from exc
    if not isinstance(payload, Mapping):
        raise RepairPredictorError(code)
    return payload


def _example_from_mapping(row: Any) -> RepairExample:
    if not isinstance(row, Mapping):
        raise RepairPredictorError("REPAIR_PREDICTOR_EXAMPLE_INVALID")
    return RepairExample(
        _identifier(row.get("environment")), _identifier(row.get("primitive")),
        _features(row.get("features")), _number(row.get("effect")), row.get("effect_class", "settled"),
    )


def _candidate_from_mapping(row: Any, target: str) -> RepairCandidate:
    if not isinstance(row, Mapping):
        raise RepairPredictorError("REPAIR_PREDICTOR_CANDIDATE_INVALID")
    if row.get("environment", target) != target:
        raise RepairPredictorError("REPAIR_PREDICTOR_TARGET_SCOPE_MISMATCH")
    return RepairCandidate(target, _identifier(row.get("primitive")), _features(row.get("features")))


def _label_map(rows: Any) -> dict[tuple[str, str], dict[str, Any]]:
    if not isinstance(rows, list):
        raise RepairPredictorError("REPAIR_PREDICTOR_LABELS_INVALID")
    result = {}
    for row in rows:
        if not isinstance(row, Mapping) or row.get("settled") is not True or not isinstance(row.get("positive"), bool):
            raise RepairPredictorError("REPAIR_PREDICTOR_LABEL_NOT_SETTLED")
        key = (_identifier(row.get("environment")), _identifier(row.get("primitive")))
        if key in result:
            raise RepairPredictorError("REPAIR_PREDICTOR_DUPLICATE_LABEL")
        effect = _number(row.get("delta_candidate_minus_baseline"))
        if row["positive"] and effect <= 0:
            raise RepairPredictorError("REPAIR_PREDICTOR_LABEL_SIGN_CONFLICT")
        result[key] = {"positive": row["positive"], "effect": effect}
    return result
