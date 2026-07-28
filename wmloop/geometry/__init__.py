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
from wmloop.geometry.memory import EffectContext, EffectMemory, EffectRecord
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
]
