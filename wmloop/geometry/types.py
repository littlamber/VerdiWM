"""Typed intervention descriptors and semantics-preserving compilation."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping


class GeometryValidationError(ValueError):
    """A VerdiWM geometry or intervention contract is malformed."""


@dataclass(frozen=True)
class CapabilityProfile:
    """Capabilities exposed by one instantiated backbone adapter."""

    backbone_family: str
    capability_class: str
    capabilities: frozenset[str]
    hook_types: frozenset[str]

    def __post_init__(self) -> None:
        _require_text(self.backbone_family, "CAPABILITY_BACKBONE_EMPTY")
        _require_text(self.capability_class, "CAPABILITY_CLASS_EMPTY")
        if any(not value for value in self.capabilities):
            raise GeometryValidationError("CAPABILITY_VALUE_EMPTY")
        if any(not value for value in self.hook_types):
            raise GeometryValidationError("CAPABILITY_HOOK_EMPTY")


@dataclass(frozen=True)
class InterventionDescriptor:
    """A semantic repair or dose-calibrated probe path.

    ``kind=probe_path`` is deliberately stricter than ``kind=repair``: probe
    paths define IRG coordinates and therefore must be inference-only,
    reversible, and measurable on both sides of zero.  A repair may instead be
    a training loss or data intervention, but it still needs an exact hook,
    preconditions, invariants, and a falsifiable prediction.
    """

    name: str
    kind: str
    hook_type: str
    transformation: str
    scope: str
    dose_unit: str
    schedule: str
    preconditions: tuple[str, ...]
    invariants: tuple[str, ...]
    prediction: str
    required_capabilities: frozenset[str] = field(default_factory=frozenset)
    inference_only: bool = False
    reversible: bool = False

    def __post_init__(self) -> None:
        for value, code in (
            (self.name, "INTERVENTION_NAME_EMPTY"),
            (self.hook_type, "INTERVENTION_HOOK_EMPTY"),
            (self.transformation, "INTERVENTION_TRANSFORM_EMPTY"),
            (self.scope, "INTERVENTION_SCOPE_EMPTY"),
            (self.dose_unit, "INTERVENTION_DOSE_UNIT_EMPTY"),
            (self.schedule, "INTERVENTION_SCHEDULE_EMPTY"),
            (self.prediction, "INTERVENTION_PREDICTION_EMPTY"),
        ):
            _require_text(value, code)
        if self.kind not in {"probe_path", "repair"}:
            raise GeometryValidationError("INTERVENTION_KIND_INVALID")
        if not self.preconditions:
            raise GeometryValidationError("INTERVENTION_PRECONDITIONS_EMPTY")
        if not self.invariants:
            raise GeometryValidationError("INTERVENTION_INVARIANTS_EMPTY")
        if self.kind == "probe_path" and (not self.inference_only or not self.reversible):
            raise GeometryValidationError("PROBE_PATH_MUST_BE_INFERENCE_ONLY_AND_REVERSIBLE")


@dataclass(frozen=True)
class CompileReceipt:
    """Auditable result of compiling one descriptor for one backbone."""

    descriptor_name: str
    backbone_family: str
    capability_class: str
    compiled: bool
    hook_type: str
    dose_unit: str
    dose_direction: float | None
    semantic_obligations: tuple[str, ...]
    invariant_checks: Mapping[str, bool]
    blockers: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.compiled and self.blockers:
            raise GeometryValidationError("COMPILE_RECEIPT_PASS_WITH_BLOCKERS")
        if not self.compiled and not self.blockers:
            raise GeometryValidationError("COMPILE_RECEIPT_FAIL_WITHOUT_BLOCKER")
        if self.compiled and not all(self.invariant_checks.values()):
            raise GeometryValidationError("COMPILE_RECEIPT_FAILED_INVARIANT")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "artifact_type": "verdiwm-typed-compile-receipt",
            "descriptor_name": self.descriptor_name,
            "backbone_family": self.backbone_family,
            "capability_class": self.capability_class,
            "compiled": self.compiled,
            "hook_type": self.hook_type,
            "dose_unit": self.dose_unit,
            "dose_direction": self.dose_direction,
            "semantic_obligations": list(self.semantic_obligations),
            "invariant_checks": dict(self.invariant_checks),
            "blockers": list(self.blockers),
        }


def compile_intervention(
    descriptor: InterventionDescriptor,
    capabilities: CapabilityProfile,
    *,
    invariant_checks: Mapping[str, bool],
    dose_direction: float | None = None,
) -> CompileReceipt:
    """Compile a descriptor or return a fail-closed receipt.

    The compiler never substitutes a nearby hook or silently drops an
    invariant.  That is the concrete guard against configuration intent and
    runtime behavior drifting apart.
    """

    blockers: list[str] = []
    if descriptor.hook_type not in capabilities.hook_types:
        blockers.append(f"hook_unavailable:{descriptor.hook_type}")
    for capability in sorted(descriptor.required_capabilities - capabilities.capabilities):
        blockers.append(f"capability_unavailable:{capability}")
    missing_checks = sorted(set(descriptor.invariants) - set(invariant_checks))
    blockers.extend(f"invariant_unchecked:{name}" for name in missing_checks)
    blockers.extend(
        f"invariant_failed:{name}"
        for name, passed in sorted(invariant_checks.items())
        if name in descriptor.invariants and passed is not True
    )
    if descriptor.kind == "probe_path":
        if dose_direction is None or not math.isfinite(dose_direction) or dose_direction == 0.0:
            blockers.append("probe_dose_direction_invalid")
    elif dose_direction is not None and not math.isfinite(dose_direction):
        blockers.append("repair_dose_direction_invalid")

    obligations = (
        f"apply_exact_hook:{descriptor.hook_type}",
        f"preserve_scope:{descriptor.scope}",
        f"measure_dose_in:{descriptor.dose_unit}",
        *descriptor.invariants,
    )
    return CompileReceipt(
        descriptor_name=descriptor.name,
        backbone_family=capabilities.backbone_family,
        capability_class=capabilities.capability_class,
        compiled=not blockers,
        hook_type=descriptor.hook_type,
        dose_unit=descriptor.dose_unit,
        dose_direction=dose_direction,
        semantic_obligations=obligations,
        invariant_checks=dict(invariant_checks),
        blockers=tuple(blockers),
    )


def _require_text(value: str, code: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise GeometryValidationError(code)
