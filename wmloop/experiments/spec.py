"""Contracts and semantic validation for cross-backbone experiments.

The JSON schema catches malformed documents.  The semantic checks here enforce
the research claims that a generic schema cannot express: exact LOBO folds,
non-leaking source sets, and distinct warm/cold/random arm semantics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from wmloop.contracts import ContractValidationError, validate_document


ARMS = ("warm_start", "cold_start", "random_search")
SELECTORS = ("environment_label", "static_probe", "raw_response", "irg")
STAGES = ("screen", "gate", "confirm")


class ExperimentSpecError(ValueError):
    """A cross-backbone experiment contract is invalid."""


def load_experiment_spec(path: Path) -> dict[str, Any]:
    """Load and validate a cross-backbone experiment specification."""

    source = Path(path).resolve()
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentSpecError(f"EXPERIMENT_SPEC_READ_FAILED:{source}") from exc
    if not isinstance(document, dict):
        raise ExperimentSpecError("EXPERIMENT_SPEC_ROOT_OBJECT_REQUIRED")
    try:
        validate_document("cross_backbone_experiment", document)
    except ContractValidationError as exc:
        raise ExperimentSpecError(str(exc)) from exc
    _validate_semantics(document)
    document["_spec_path"] = str(source)
    return document


def _validate_semantics(document: Mapping[str, Any]) -> None:
    backbones = document["backbones"]
    backbone_ids = [str(item["backbone_id"]) for item in backbones]
    if len(set(backbone_ids)) != len(backbone_ids):
        raise ExperimentSpecError("EXPERIMENT_BACKBONE_IDS_NOT_UNIQUE")
    if len(backbone_ids) < 2:
        raise ExperimentSpecError("EXPERIMENT_NEEDS_TWO_BACKBONES")

    arm_ids = [str(item["arm"]) for item in document["arms"]]
    if tuple(arm_ids) != ARMS:
        raise ExperimentSpecError(f"EXPERIMENT_ARMS_MUST_BE_ORDERED:{','.join(ARMS)}")
    for item in document["arms"]:
        arm = str(item["arm"])
        expected = arm == "warm_start"
        if bool(item["uses_source_experience"]) is not expected:
            raise ExperimentSpecError(f"EXPERIMENT_ARM_SOURCE_SEMANTICS_INVALID:{arm}")

    selector_ids = [str(item["selector"]) for item in document["selectors"]]
    if tuple(selector_ids) != SELECTORS:
        raise ExperimentSpecError(f"EXPERIMENT_SELECTORS_MUST_BE_ORDERED:{','.join(SELECTORS)}")
    stage_ids = [str(item["stage"]) for item in document["stages"]]
    if tuple(stage_ids) != STAGES:
        raise ExperimentSpecError(f"EXPERIMENT_STAGES_MUST_BE_ORDERED:{','.join(STAGES)}")
    seeds = [int(value) for value in document["seeds"]]
    if len(set(seeds)) != len(seeds):
        raise ExperimentSpecError("EXPERIMENT_SEEDS_NOT_UNIQUE")

    known = set(backbone_ids)
    fold_ids: set[str] = set()
    for fold in document["folds"]:
        fold_id = str(fold["fold_id"])
        if fold_id in fold_ids:
            raise ExperimentSpecError("EXPERIMENT_FOLD_IDS_NOT_UNIQUE")
        fold_ids.add(fold_id)
        target = str(fold["target_backbone"])
        sources = [str(value) for value in fold["source_backbones"]]
        expected_sources = known - {target}
        if target not in known:
            raise ExperimentSpecError(f"EXPERIMENT_FOLD_TARGET_UNKNOWN:{target}")
        if len(set(sources)) != len(sources) or set(sources) != expected_sources:
            raise ExperimentSpecError(f"EXPERIMENT_LOBO_SOURCE_LEAK_OR_OMISSION:{fold_id}")
        if target in sources:
            raise ExperimentSpecError(f"EXPERIMENT_TARGET_LEAK:{fold_id}")
        scenarios = [str(value) for value in fold["scenarios"]]
        if len(set(scenarios)) != len(scenarios):
            raise ExperimentSpecError(f"EXPERIMENT_SCENARIOS_NOT_UNIQUE:{fold_id}")

    metric = document["metric_contract"]
    if float(metric["positive_delta_threshold"]) < 0.0:
        raise ExperimentSpecError("EXPERIMENT_POSITIVE_THRESHOLD_INVALID")


def backbone_map(spec: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """Return a stable backbone-id lookup for planners and reporters."""

    return {str(item["backbone_id"]): item for item in spec["backbones"]}


def stage_map(spec: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """Return stage configuration keyed by the frozen stage order."""

    return {str(item["stage"]): item for item in spec["stages"]}
