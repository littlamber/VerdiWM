"""Evidence-bound retrieval for settled diagnostic probe experiences."""

from wmloop.retrieve.index import (
    ProbeExperience,
    ProbeRetrievalError,
    index_probe_experience,
    retrieve_probe_experiences,
)
from wmloop.retrieve.evidence_capsule import (
    EvidenceCapsuleError,
    build_evidence_capsule,
    write_evidence_capsule,
)
from wmloop.retrieve.literature import (
    LiteratureRecord,
    LiteratureRetrievalError,
    search_arxiv,
    run_literature_retrieval,
    stage_literature_results,
)
from wmloop.retrieve.method_staging import (
    HeuristicMethodSynthesisClient,
    LiteratureMethodStagingError,
    MethodSynthesisClient,
    run_literature_method_prompt_batch,
    run_literature_method_staging,
)
from wmloop.retrieve.mechanism_discovery import (
    AnnotationMechanismExtractor,
    DiscoveryRequest,
    EvidenceOnlyMechanismExtractor,
    MechanismDiscoveryError,
    build_multiview_queries,
    compare_mechanism_signature,
    run_mechanism_discovery,
)
from wmloop.retrieve.irg_guided_discovery import (
    IRGDiscoveryError,
    build_irg_discovery_request,
    derive_irg_bottlenecks,
    run_irg_guided_mechanism_discovery,
)

__all__ = [
    "EvidenceCapsuleError",
    "build_evidence_capsule",
    "write_evidence_capsule",
    "ProbeExperience",
    "ProbeRetrievalError",
    "index_probe_experience",
    "retrieve_probe_experiences",
    "LiteratureRecord",
    "LiteratureRetrievalError",
    "search_arxiv",
    "run_literature_retrieval",
    "stage_literature_results",
    "HeuristicMethodSynthesisClient",
    "LiteratureMethodStagingError",
    "MethodSynthesisClient",
    "run_literature_method_prompt_batch",
    "run_literature_method_staging",
    "AnnotationMechanismExtractor",
    "DiscoveryRequest",
    "EvidenceOnlyMechanismExtractor",
    "MechanismDiscoveryError",
    "build_multiview_queries",
    "compare_mechanism_signature",
    "run_mechanism_discovery",
    "IRGDiscoveryError",
    "derive_irg_bottlenecks",
    "build_irg_discovery_request",
    "run_irg_guided_mechanism_discovery",
]
