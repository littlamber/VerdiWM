"""Plan experiments that identify why a verified training gain is small."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path


class TrainingGainAttributionError(ValueError):
    """Training evidence is missing or inconsistent."""


def build_training_gain_attribution(
    *,
    training_receipt_path: Path,
    screen_settlement_path: Path,
    confirm_settlement_path: Path,
    verifier_manifest_path: Path,
) -> dict[str, object]:
    training = _load(training_receipt_path, "GAIN_ATTRIBUTION_TRAINING_RECEIPT_INVALID")
    screen = _load(screen_settlement_path, "GAIN_ATTRIBUTION_SCREEN_INVALID")
    confirm = _load(confirm_settlement_path, "GAIN_ATTRIBUTION_CONFIRM_INVALID")
    verifier = _load(verifier_manifest_path, "GAIN_ATTRIBUTION_VERIFIER_INVALID")
    candidate_id = training.get("candidate_id")
    if (
        training.get("artifact_type") != "verdiwm-masked-adapter-training-receipt"
        or training.get("state") != "ready_for_evaluation"
        or screen.get("candidate", {}).get("candidate_id") != candidate_id
        or confirm.get("candidate", {}).get("candidate_id") != candidate_id
        or verifier.get("candidate_id") != candidate_id
        or verifier.get("state") != "verified"
        or verifier.get("decision") != "confirmed_positive"
    ):
        raise TrainingGainAttributionError("GAIN_ATTRIBUTION_EVIDENCE_BINDING_INVALID")
    optimizer = training.get("optimizer_receipt")
    parameters = training.get("parameters")
    if not isinstance(optimizer, Mapping) or not isinstance(parameters, Mapping):
        raise TrainingGainAttributionError("GAIN_ATTRIBUTION_TRAINING_RECEIPT_INVALID")
    steps = optimizer.get("steps")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise TrainingGainAttributionError("GAIN_ATTRIBUTION_TRAINING_RECEIPT_INVALID")
    data_receipt = training.get("data_receipt")
    data_observed = isinstance(data_receipt, Mapping)
    screen_effects = _effects(screen)
    confirm_effects = _effects(confirm)
    base_budget = steps
    medium_budget = max(1024, base_budget * 3)
    high_budget = max(medium_budget * 3, base_budget * 9)
    body: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "verdiwm-training-gain-attribution-plan",
        "plan_id": "",
        "candidate_id": candidate_id,
        "evidence": {
            "training_receipt_sha256": _sha256(Path(training_receipt_path)),
            "screen_settlement_sha256": _sha256(Path(screen_settlement_path)),
            "confirm_settlement_sha256": _sha256(Path(confirm_settlement_path)),
            "verifier_manifest_sha256": _sha256(Path(verifier_manifest_path)),
            "screen_relative_improvements_percent": screen_effects,
            "confirm_relative_improvements_percent": confirm_effects,
        },
        "diagnosis": {
            "state": "not_identifiable_from_single_training_point",
            "effect_established": True,
            "observed_training_budget_count": 1,
            "observed_data_scale_count": 1,
            "observed_mechanism_ablation_count": 0,
            "undertraining_plausible": steps < 1024,
            "data_limitation_plausible": True,
            "capacity_limit_plausible": (
                int(parameters.get("hidden_dim", 0)) <= 64
                or float(parameters.get("max_residual", 1.0)) <= 0.1
            ),
            "data_receipt_complete": data_observed,
            "missing_evidence": [
                value
                for value, missing in (
                    ("training_budget_curve", True),
                    ("data_scale_curve", True),
                    ("mechanism_ablation", True),
                    ("dataset_examples_and_examples_seen", not data_observed),
                    ("multi_seed_effect_distribution", True),
                )
                if missing
            ],
            "interpretation": (
                "The positive effect is real under the frozen contract, but one 300-step-style "
                "training point cannot identify whether optimization, data coverage, or mechanism "
                "capacity is limiting the effect."
            ),
        },
        "factorial_screen": {
            "common_controls": {
                "same_backbone_checkpoint": True,
                "same_train_validation_split": True,
                "same_frozen_evaluator": True,
                "same_seed_set": [20260830, 20260831, 20260832],
                "protected_regressions_allowed": 0,
            },
            "budget_axis": [
                {"arm_id": "budget-base", "steps": base_budget},
                {"arm_id": "budget-medium", "steps": medium_budget},
                {"arm_id": "budget-high", "steps": high_budget},
            ],
            "data_axis": [
                {"arm_id": "data-25", "train_fraction": 0.25, "steps": medium_budget},
                {"arm_id": "data-50", "train_fraction": 0.50, "steps": medium_budget},
                {"arm_id": "data-100", "train_fraction": 1.0, "steps": medium_budget},
            ],
            "mechanism_axis": [
                {"arm_id": "learned-mask-full", "learned_action_mask": True, "hidden_dim": 64, "max_residual": 0.1},
                {"arm_id": "constant-mask-control", "learned_action_mask": False, "hidden_dim": 64, "max_residual": 0.1},
                {"arm_id": "capacity-128", "learned_action_mask": True, "hidden_dim": 128, "max_residual": 0.1},
                {"arm_id": "residual-020", "learned_action_mask": True, "hidden_dim": 64, "max_residual": 0.2},
            ],
        },
        "decision_rules": [
            {"cause": "optimization_limited", "criterion": "gain increases monotonically across the fixed-data budget axis"},
            {"cause": "data_limited", "criterion": "gain increases across data fractions at fixed optimizer updates"},
            {"cause": "capacity_limited", "criterion": "a larger adapter improves confirm after budget and data curves saturate"},
            {"cause": "mechanism_limited", "criterion": "budget, data, and capacity increases saturate below the preregistered minimum effect"},
            {"cause": "mask_not_causal", "criterion": "the constant-mask control matches or exceeds the learned-mask arm"},
        ],
        "innovation_admission": {
            "required_origin": "target_side_causal_discovery",
            "requirements": [
                "candidate is generated from a frozen target-side response collision or counterexample",
                "novelty search finds no mechanism-equivalent method in the versioned literature atlas",
                "the proposal predicts a discriminating intervention before training",
                "ablation defeats the nearest retrieved mechanism and the strongest current local baseline",
                "screen and independent multi-seed confirm improve the primary metric with zero protected regressions",
                "the frozen verifier returns confirmed_positive",
            ],
            "forbidden_claims": [
                "renaming a retrieved paper mechanism as autonomous discovery",
                "calling an untested LLM proposal innovative",
                "using screen-only or same-split evidence as confirmation",
            ],
        },
        "claim_boundary": (
            "This artifact plans causal attribution and innovation admission. It does not "
            "attribute the current effect or establish a novel mechanism before its arms settle."
        ),
    }
    body["plan_id"] = "gain-attribution-" + _digest(body)[:24]
    return body


def write_training_gain_attribution(plan: Mapping[str, object], path: Path) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(plan, handle, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, target)
    finally:
        if os.path.exists(name):
            os.unlink(name)
    return target


def _effects(settlement: Mapping[str, object]) -> dict[str, float]:
    metrics = settlement.get("metrics")
    deltas = settlement.get("metric_deltas")
    if not isinstance(metrics, Mapping) or not isinstance(deltas, Mapping):
        raise TrainingGainAttributionError("GAIN_ATTRIBUTION_SETTLEMENT_METRICS_INVALID")
    effects: dict[str, float] = {}
    for name, raw_delta in deltas.items():
        raw_value = metrics.get(name)
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise TrainingGainAttributionError("GAIN_ATTRIBUTION_SETTLEMENT_METRICS_INVALID")
        if isinstance(raw_delta, bool) or not isinstance(raw_delta, (int, float)):
            raise TrainingGainAttributionError("GAIN_ATTRIBUTION_SETTLEMENT_METRICS_INVALID")
        baseline = float(raw_value) + float(raw_delta)
        effects[str(name)] = round(100.0 * float(raw_delta) / baseline, 6) if baseline else 0.0
    return effects


def _load(path: Path, code: str) -> dict[str, object]:
    source = Path(path).expanduser()
    if source.is_symlink() or not source.is_file():
        raise TrainingGainAttributionError(code)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingGainAttributionError(code) from exc
    if not isinstance(value, dict):
        raise TrainingGainAttributionError(code)
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("utf-8")).hexdigest()
