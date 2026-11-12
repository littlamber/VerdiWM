"""Core, backbone-agnostic objects from the VerdiWM method.

The GPU campaign machinery lives elsewhere in :mod:`wmloop`.  This package
contains the small deterministic layer that makes campaign evidence portable:
typed intervention contracts, response geometry, selective transfer, effect
memory, and counterexample-driven atlas evolution.

Public exports are lazy so importing a component does not initialize workflows.
"""

from __future__ import annotations

from importlib import import_module
from types import MappingProxyType
from typing import Any

_EXPORTS = MappingProxyType(
    {
        "AtlasPoint": ("wmloop.geometry.evolution", "AtlasPoint"),
        "CapabilityProfile": ("wmloop.geometry.types", "CapabilityProfile"),
        "CompileReceipt": ("wmloop.geometry.types", "CompileReceipt"),
        "EffectContext": ("wmloop.geometry.memory", "EffectContext"),
        "EffectMemory": ("wmloop.geometry.memory", "EffectMemory"),
        "EffectRecord": ("wmloop.geometry.memory", "EffectRecord"),
        "MechanismRelation": (
            "wmloop.geometry.mechanism_relations",
            "MechanismRelation",
        ),
        "RELATION_TYPES": ("wmloop.geometry.mechanism_relations", "RELATION_TYPES"),
        "COMPOSITION_OPERATORS": (
            "wmloop.geometry.mechanism_relations",
            "COMPOSITION_OPERATORS",
        ),
        "VERIFICATION_STATES": (
            "wmloop.geometry.mechanism_relations",
            "VERIFICATION_STATES",
        ),
        "build_transferable_experience": (
            "wmloop.geometry.memory",
            "build_transferable_experience",
        ),
        "build_mechanism_relation": (
            "wmloop.geometry.mechanism_relations",
            "build_mechanism_relation",
        ),
        "propose_mechanism_relation": (
            "wmloop.geometry.mechanism_relations",
            "propose_mechanism_relation",
        ),
        "settle_mechanism_relation": (
            "wmloop.geometry.mechanism_relations",
            "settle_mechanism_relation",
        ),
        "classify_interaction": (
            "wmloop.geometry.mechanism_relations",
            "classify_interaction",
        ),
        "interaction_effect": (
            "wmloop.geometry.mechanism_relations",
            "interaction_effect",
        ),
        "relation_from_dict": (
            "wmloop.geometry.mechanism_relations",
            "relation_from_dict",
        ),
        "build_evidence_ir": ("wmloop.geometry.evidence_ir", "build_evidence_ir"),
        "build_mechanism_contract": (
            "wmloop.geometry.portable_transfer_knowledge",
            "build_mechanism_contract",
        ),
        "build_method_embodiment": (
            "wmloop.geometry.portable_transfer_knowledge",
            "build_method_embodiment",
        ),
        "build_portable_experience": (
            "wmloop.geometry.portable_experience",
            "build_portable_experience",
        ),
        "build_probe_fingerprint_summary": (
            "wmloop.geometry.portable_transfer_knowledge",
            "build_probe_fingerprint_summary",
        ),
        "build_transfer_boundary": (
            "wmloop.geometry.portable_transfer_knowledge",
            "build_transfer_boundary",
        ),
        "GeometryValidationError": ("wmloop.geometry.types", "GeometryValidationError"),
        "InterventionDescriptor": ("wmloop.geometry.types", "InterventionDescriptor"),
        "IRGChartSource": ("wmloop.geometry.assets", "IRGChartSource"),
        "ProbeCandidate": ("wmloop.geometry.evolution", "ProbeCandidate"),
        "RepairCollision": ("wmloop.geometry.evolution", "RepairCollision"),
        "ResponseChart": ("wmloop.geometry.irg", "ResponseChart"),
        "TransferCertificate": ("wmloop.geometry.transfer", "TransferCertificate"),
        "compile_intervention": ("wmloop.geometry.types", "compile_intervention"),
        "compose_irg_asset": ("wmloop.geometry.assets", "compose_irg_asset"),
        "detect_repair_collisions": (
            "wmloop.geometry.evolution",
            "detect_repair_collisions",
        ),
        "estimate_response_chart": ("wmloop.geometry.irg", "estimate_response_chart"),
        "evaluate_transfer_certificate": (
            "wmloop.geometry.transfer",
            "evaluate_transfer_certificate",
        ),
        "irg_distance": ("wmloop.geometry.irg", "irg_distance"),
        "rank_probe_candidates": ("wmloop.geometry.evolution", "rank_probe_candidates"),
        "validate_irg_asset": ("wmloop.geometry.assets", "validate_irg_asset"),
        "validate_evidence_ir": ("wmloop.geometry.evidence_ir", "validate_evidence_ir"),
        "validate_mechanism_contract": (
            "wmloop.geometry.portable_transfer_knowledge",
            "validate_mechanism_contract",
        ),
        "validate_method_embodiment": (
            "wmloop.geometry.portable_transfer_knowledge",
            "validate_method_embodiment",
        ),
        "validate_probe_fingerprint_summary": (
            "wmloop.geometry.portable_transfer_knowledge",
            "validate_probe_fingerprint_summary",
        ),
        "ModelIRGError": ("wmloop.geometry.model_irg", "ModelIRGError"),
        "build_model_irg": ("wmloop.geometry.model_irg", "build_model_irg"),
        "validate_model_irg": ("wmloop.geometry.model_irg", "validate_model_irg"),
        "model_irg_distance": ("wmloop.geometry.model_irg", "model_irg_distance"),
        "rank_method_effects_by_irg": (
            "wmloop.geometry.model_irg",
            "rank_method_effects_by_irg",
        ),
        "detect_model_irg_collisions": (
            "wmloop.geometry.model_irg",
            "detect_model_irg_collisions",
        ),
        "validate_portable_experience": (
            "wmloop.geometry.portable_experience",
            "validate_portable_experience",
        ),
        "validate_mechanism_relation": (
            "wmloop.geometry.mechanism_relations",
            "validate_mechanism_relation",
        ),
        "validate_transfer_boundary": (
            "wmloop.geometry.portable_transfer_knowledge",
            "validate_transfer_boundary",
        ),
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
