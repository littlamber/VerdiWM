"""Reproducible cross-backbone experiment control plane."""

from wmloop.experiments.spec import (
    ARMS,
    SELECTORS,
    STAGES,
    ExperimentSpecError,
    load_experiment_spec,
)
from wmloop.experiments.fingerprint_atlas import FingerprintAtlasError, build_fingerprint_atlas
from wmloop.experiments.selector_ablation import SelectorAblationError, build_selector_ablation_plan
from wmloop.experiments.effect_labels import EffectLabelIndexError, build_effect_label_index
from wmloop.experiments.ctrl_world_fingerprint import (
    CtrlWorldFingerprintError,
    evaluate_ctrl_world_fingerprint,
)
from wmloop.experiments.cpbe import (
    CPBEError,
    build_cpbe_plan,
    settle_cpbe_plan,
)
from wmloop.experiments.acwm_cpbe_bootstrap import (
    ACWMCPBEBootstrapError,
    build_acwm_cpbe_bootstrap,
)
from wmloop.experiments.cpbe_stage_runner import (
    CPBEStageRunnerError,
    publish_static_offline_receipts,
)
from wmloop.experiments.acwm_cpbe_canary import (
    ACWMCPBECanaryError,
    evaluate_acwm_cpbe_canary,
    prepare_acwm_cpbe_canary_bundle,
)
from wmloop.experiments.cpbe_counterexample import (
    CPBECounterexampleError,
    build_counterexample_round,
)
from wmloop.experiments.cpbe_materializer import (
    CPBEMaterializerError,
    publish_cpbe_materialization,
)

__all__ = [
    "ARMS",
    "SELECTORS",
    "STAGES",
    "ExperimentSpecError",
    "FingerprintAtlasError",
    "EffectLabelIndexError",
    "CtrlWorldFingerprintError",
    "CPBEError",
    "ACWMCPBEBootstrapError",
    "CPBEStageRunnerError",
    "ACWMCPBECanaryError",
    "CPBECounterexampleError",
    "CPBEMaterializerError",
    "SelectorAblationError",
    "build_fingerprint_atlas",
    "build_effect_label_index",
    "evaluate_ctrl_world_fingerprint",
    "build_selector_ablation_plan",
    "build_cpbe_plan",
    "build_acwm_cpbe_bootstrap",
    "publish_static_offline_receipts",
    "evaluate_acwm_cpbe_canary",
    "prepare_acwm_cpbe_canary_bundle",
    "build_counterexample_round",
    "publish_cpbe_materialization",
    "load_experiment_spec",
    "settle_cpbe_plan",
]
