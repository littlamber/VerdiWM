"""Core, backbone-agnostic objects from the VerdiWM method.

The GPU campaign machinery lives elsewhere in :mod:`wmloop`.  This package
contains the small deterministic layer that makes campaign evidence portable:
typed intervention contracts, response geometry, selective transfer, effect
memory, and counterexample-driven atlas evolution.
"""

from wmloop.geometry.evolution import (
    AtlasPoint,
    ProbeCandidate,
    RepairCollision,
    detect_repair_collisions,
    rank_probe_candidates,
)
from wmloop.geometry.assets import IRGChartSource, compose_irg_asset, validate_irg_asset
from wmloop.geometry.irg import ResponseChart, estimate_response_chart, irg_distance
from wmloop.geometry.evidence_ir import build_evidence_ir, validate_evidence_ir
from wmloop.geometry.memory import (
    EffectContext,
    EffectMemory,
    EffectRecord,
    build_transferable_experience,
)
from wmloop.geometry.mechanism_relations import (
    COMPOSITION_OPERATORS,
    RELATION_TYPES,
    VERIFICATION_STATES,
    MechanismRelation,
    build_mechanism_relation,
    propose_mechanism_relation,
    classify_interaction,
    interaction_effect,
    relation_from_dict,
    validate_mechanism_relation,
)
from wmloop.geometry.portable_experience import (
    build_portable_experience,
    validate_portable_experience,
)
from wmloop.geometry.portable_transfer_knowledge import (
    build_mechanism_contract,
    build_method_embodiment,
    build_probe_fingerprint_summary,
    build_transfer_boundary,
    validate_mechanism_contract,
    validate_method_embodiment,
    validate_probe_fingerprint_summary,
    validate_transfer_boundary,
)
from wmloop.geometry.transfer import TransferCertificate, evaluate_transfer_certificate
from wmloop.geometry.types import (
    CapabilityProfile,
    CompileReceipt,
    GeometryValidationError,
    InterventionDescriptor,
    compile_intervention,
)

__all__ = [
    "AtlasPoint",
    "CapabilityProfile",
    "CompileReceipt",
    "EffectContext",
    "EffectMemory",
    "EffectRecord",
    "MechanismRelation",
    "RELATION_TYPES",
    "COMPOSITION_OPERATORS",
    "VERIFICATION_STATES",
    "build_transferable_experience",
    "build_mechanism_relation",
    "propose_mechanism_relation",
    "classify_interaction",
    "interaction_effect",
    "relation_from_dict",
    "build_evidence_ir",
    "build_mechanism_contract",
    "build_method_embodiment",
    "build_portable_experience",
    "build_probe_fingerprint_summary",
    "build_transfer_boundary",
    "GeometryValidationError",
    "InterventionDescriptor",
    "IRGChartSource",
    "ProbeCandidate",
    "RepairCollision",
    "ResponseChart",
    "TransferCertificate",
    "compile_intervention",
    "compose_irg_asset",
    "detect_repair_collisions",
    "estimate_response_chart",
    "evaluate_transfer_certificate",
    "irg_distance",
    "rank_probe_candidates",
    "validate_irg_asset",
    "validate_evidence_ir",
    "validate_mechanism_contract",
    "validate_method_embodiment",
    "validate_probe_fingerprint_summary",
    "validate_portable_experience",
    "validate_mechanism_relation",
    "validate_transfer_boundary",
]
