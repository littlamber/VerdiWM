"""Build exact, GPU-neutral ACWM baseline evaluation plans.

The upstream evaluator defaults to a sampled, whole-split run.  This module
does not run it.  It makes the controller's required selections explicit so a
later read-only data-view executor has no authority to silently widen a held
out cohort or change the 3 + 3 + 2 environment schedule.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from wmloop.acwm_data import AcwmEnvironmentSpec, CANONICAL_ACWM_ENVIRONMENTS
from wmloop.freeze import dataset_freeze_sha256, verify_acwm_heldout_protocol


class BaselinePlanError(ValueError):
    """The immutable protocol cannot be scheduled as an exact baseline."""


_COHORTS = ("ind_dev", "ind_accept", "ood_accept")


@dataclass(frozen=True)
class EvaluationSelection:
    environment: str
    vendor_environment: str
    cohort: str
    trajectory_ids: tuple[str, ...]


@dataclass(frozen=True)
class BaselineEvaluationPlan:
    dataset_freeze_sha256: str
    heldout_protocol_sha256: str
    waves: tuple[tuple[str, ...], ...]
    selections: tuple[EvaluationSelection, ...]

    def selection(self, environment: str, cohort: str) -> EvaluationSelection:
        for item in self.selections:
            if item.environment == environment and item.cohort == cohort:
                return item
        raise KeyError((environment, cohort))


def build_baseline_evaluation_plan(
    dataset_freeze: Mapping[str, Any],
    heldout_protocol: Mapping[str, Any],
    *,
    environment_specs: Iterable[AcwmEnvironmentSpec] = CANONICAL_ACWM_ENVIRONMENTS,
    gpus: tuple[int, ...] = (0, 1, 2),
) -> BaselineEvaluationPlan:
    """Create a 3 + 3 + 2 plan without launching a subprocess or touching data."""

    specs = tuple(environment_specs)
    if len(gpus) != 3 or len(set(gpus)) != 3:
        raise BaselinePlanError("BASELINE_PLAN_GPU_PROFILE_INVALID")
    try:
        verify_acwm_heldout_protocol(dataset_freeze, heldout_protocol, environment_specs=specs)
    except ValueError as exc:
        raise BaselinePlanError(str(exc)) from exc
    raw_partitions = heldout_protocol.get("environment_partitions")
    if not isinstance(raw_partitions, Mapping):
        raise BaselinePlanError("BASELINE_PLAN_PARTITIONS_INVALID")
    expected_environments = tuple(spec.environment for spec in specs)
    if tuple(sorted(raw_partitions)) != tuple(sorted(expected_environments)):
        raise BaselinePlanError("BASELINE_PLAN_ENVIRONMENT_MISMATCH")
    selections: list[EvaluationSelection] = []
    for spec in specs:
        partition = raw_partitions.get(spec.environment)
        if not isinstance(partition, Mapping):
            raise BaselinePlanError(f"BASELINE_PLAN_PARTITION_INVALID:{spec.environment}")
        for cohort in _COHORTS:
            values = partition.get(cohort)
            if not isinstance(values, list) or not values or any(not isinstance(item, str) or not item for item in values):
                raise BaselinePlanError(f"BASELINE_PLAN_COHORT_INVALID:{spec.environment}:{cohort}")
            selections.append(
                EvaluationSelection(
                    environment=spec.environment,
                    vendor_environment=_vendor_environment(spec),
                    cohort=cohort,
                    trajectory_ids=tuple(values),
                )
            )
    waves = tuple(
        expected_environments[index : index + len(gpus)] for index in range(0, len(expected_environments), len(gpus))
    )
    return BaselineEvaluationPlan(
        dataset_freeze_sha256=dataset_freeze_sha256(dataset_freeze),
        heldout_protocol_sha256=_canonical_sha256(heldout_protocol),
        waves=waves,
        selections=tuple(selections),
    )


def _vendor_environment(spec: AcwmEnvironmentSpec) -> str:
    return "clothmove" if spec.environment == "cloth_move" else spec.environment


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
