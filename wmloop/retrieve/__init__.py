"""Evidence-bound retrieval for settled diagnostic probe experiences.

Public exports are lazy so importing a component does not initialize workflows.
"""

from __future__ import annotations

from importlib import import_module
from types import MappingProxyType
from typing import Any

_EXPORTS = MappingProxyType(
    {
        "EvidenceCapsuleError": (
            "wmloop.retrieve.evidence_capsule",
            "EvidenceCapsuleError",
        ),
        "build_evidence_capsule": (
            "wmloop.retrieve.evidence_capsule",
            "build_evidence_capsule",
        ),
        "write_evidence_capsule": (
            "wmloop.retrieve.evidence_capsule",
            "write_evidence_capsule",
        ),
        "ProbeExperience": ("wmloop.retrieve.index", "ProbeExperience"),
        "ProbeRetrievalError": ("wmloop.retrieve.index", "ProbeRetrievalError"),
        "index_probe_experience": ("wmloop.retrieve.index", "index_probe_experience"),
        "retrieve_probe_experiences": (
            "wmloop.retrieve.index",
            "retrieve_probe_experiences",
        ),
        "AgibotPSNRMemory": (
            "wmloop.retrieve.agibot_psnr_memory",
            "AgibotPSNRMemory",
        ),
        "AgibotPSNRMemoryError": (
            "wmloop.retrieve.agibot_psnr_memory",
            "AgibotPSNRMemoryError",
        ),
        "retrieve_agibot_psnr_memories": (
            "wmloop.retrieve.agibot_psnr_memory",
            "retrieve_agibot_psnr_memories",
        ),
        "build_agibot_psnr_memory_context": (
            "wmloop.retrieve.agibot_psnr_memory",
            "build_agibot_psnr_memory_context",
        ),
        "LiteratureRecord": ("wmloop.retrieve.literature", "LiteratureRecord"),
        "LiteratureRetrievalError": (
            "wmloop.retrieve.literature",
            "LiteratureRetrievalError",
        ),
        "search_arxiv": ("wmloop.retrieve.literature", "search_arxiv"),
        "run_literature_retrieval": (
            "wmloop.retrieve.literature",
            "run_literature_retrieval",
        ),
        "stage_literature_results": (
            "wmloop.retrieve.literature",
            "stage_literature_results",
        ),
        "HeuristicMethodSynthesisClient": (
            "wmloop.retrieve.method_staging",
            "HeuristicMethodSynthesisClient",
        ),
        "LiteratureMethodStagingError": (
            "wmloop.retrieve.method_staging",
            "LiteratureMethodStagingError",
        ),
        "MethodSynthesisClient": (
            "wmloop.retrieve.method_staging",
            "MethodSynthesisClient",
        ),
        "run_literature_method_prompt_batch": (
            "wmloop.retrieve.method_staging",
            "run_literature_method_prompt_batch",
        ),
        "run_literature_method_staging": (
            "wmloop.retrieve.method_staging",
            "run_literature_method_staging",
        ),
        "AnnotationMechanismExtractor": (
            "wmloop.retrieve.mechanism_discovery",
            "AnnotationMechanismExtractor",
        ),
        "DiscoveryRequest": ("wmloop.retrieve.mechanism_discovery", "DiscoveryRequest"),
        "EvidenceOnlyMechanismExtractor": (
            "wmloop.retrieve.mechanism_discovery",
            "EvidenceOnlyMechanismExtractor",
        ),
        "MechanismDiscoveryError": (
            "wmloop.retrieve.mechanism_discovery",
            "MechanismDiscoveryError",
        ),
        "build_multiview_queries": (
            "wmloop.retrieve.mechanism_discovery",
            "build_multiview_queries",
        ),
        "compare_mechanism_signature": (
            "wmloop.retrieve.mechanism_discovery",
            "compare_mechanism_signature",
        ),
        "run_mechanism_discovery": (
            "wmloop.retrieve.mechanism_discovery",
            "run_mechanism_discovery",
        ),
        "IRGDiscoveryError": (
            "wmloop.retrieve.irg_guided_discovery",
            "IRGDiscoveryError",
        ),
        "derive_irg_bottlenecks": (
            "wmloop.retrieve.irg_guided_discovery",
            "derive_irg_bottlenecks",
        ),
        "build_irg_discovery_request": (
            "wmloop.retrieve.irg_guided_discovery",
            "build_irg_discovery_request",
        ),
        "run_irg_guided_mechanism_discovery": (
            "wmloop.retrieve.irg_guided_discovery",
            "run_irg_guided_mechanism_discovery",
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
