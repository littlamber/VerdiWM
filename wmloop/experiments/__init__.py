"""Reproducible cross-backbone experiment control plane.

Public exports are lazy so importing a component does not initialize workflows.
"""

from __future__ import annotations

from importlib import import_module
from types import MappingProxyType
from typing import Any

_EXPORTS = MappingProxyType(
    {
        "ARMS": ("wmloop.experiments.spec", "ARMS"),
        "SELECTORS": ("wmloop.experiments.spec", "SELECTORS"),
        "STAGES": ("wmloop.experiments.spec", "STAGES"),
        "ExperimentSpecError": ("wmloop.experiments.spec", "ExperimentSpecError"),
        "FingerprintAtlasError": (
            "wmloop.experiments.fingerprint_atlas",
            "FingerprintAtlasError",
        ),
        "EffectLabelIndexError": (
            "wmloop.experiments.effect_labels",
            "EffectLabelIndexError",
        ),
        "CtrlWorldFingerprintError": (
            "wmloop.experiments.ctrl_world_fingerprint",
            "CtrlWorldFingerprintError",
        ),
        "CPBEError": ("wmloop.experiments.cpbe", "CPBEError"),
        "ACWMCPBEBootstrapError": (
            "wmloop.experiments.acwm_cpbe_bootstrap",
            "ACWMCPBEBootstrapError",
        ),
        "CPBEStageRunnerError": (
            "wmloop.experiments.cpbe_stage_runner",
            "CPBEStageRunnerError",
        ),
        "ACWMCPBECanaryError": (
            "wmloop.experiments.acwm_cpbe_canary",
            "ACWMCPBECanaryError",
        ),
        "CPBECounterexampleError": (
            "wmloop.experiments.cpbe_counterexample",
            "CPBECounterexampleError",
        ),
        "CPBEMaterializerError": (
            "wmloop.experiments.cpbe_materializer",
            "CPBEMaterializerError",
        ),
        "RepairCandidate": (
            "wmloop.experiments.repair_effect_predictor",
            "RepairCandidate",
        ),
        "RepairExample": (
            "wmloop.experiments.repair_effect_predictor",
            "RepairExample",
        ),
        "RepairPrediction": (
            "wmloop.experiments.repair_effect_predictor",
            "RepairPrediction",
        ),
        "RepairPredictorError": (
            "wmloop.experiments.repair_effect_predictor",
            "RepairPredictorError",
        ),
        "RidgeRepairEffectPredictor": (
            "wmloop.experiments.repair_effect_predictor",
            "RidgeRepairEffectPredictor",
        ),
        "TrainingScope": (
            "wmloop.experiments.repair_effect_predictor",
            "TrainingScope",
        ),
        "SelectorAblationError": (
            "wmloop.experiments.selector_ablation",
            "SelectorAblationError",
        ),
        "build_fingerprint_atlas": (
            "wmloop.experiments.fingerprint_atlas",
            "build_fingerprint_atlas",
        ),
        "build_effect_label_index": (
            "wmloop.experiments.effect_labels",
            "build_effect_label_index",
        ),
        "evaluate_ctrl_world_fingerprint": (
            "wmloop.experiments.ctrl_world_fingerprint",
            "evaluate_ctrl_world_fingerprint",
        ),
        "build_selector_ablation_plan": (
            "wmloop.experiments.selector_ablation",
            "build_selector_ablation_plan",
        ),
        "build_cpbe_plan": ("wmloop.experiments.cpbe", "build_cpbe_plan"),
        "build_acwm_cpbe_bootstrap": (
            "wmloop.experiments.acwm_cpbe_bootstrap",
            "build_acwm_cpbe_bootstrap",
        ),
        "publish_static_offline_receipts": (
            "wmloop.experiments.cpbe_stage_runner",
            "publish_static_offline_receipts",
        ),
        "evaluate_acwm_cpbe_canary": (
            "wmloop.experiments.acwm_cpbe_canary",
            "evaluate_acwm_cpbe_canary",
        ),
        "prepare_acwm_cpbe_canary_bundle": (
            "wmloop.experiments.acwm_cpbe_canary",
            "prepare_acwm_cpbe_canary_bundle",
        ),
        "build_counterexample_round": (
            "wmloop.experiments.cpbe_counterexample",
            "build_counterexample_round",
        ),
        "publish_cpbe_materialization": (
            "wmloop.experiments.cpbe_materializer",
            "publish_cpbe_materialization",
        ),
        "load_experiment_spec": ("wmloop.experiments.spec", "load_experiment_spec"),
        "settle_cpbe_plan": ("wmloop.experiments.cpbe", "settle_cpbe_plan"),
    }
)

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module, attribute = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    return getattr(import_module(module), attribute)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
