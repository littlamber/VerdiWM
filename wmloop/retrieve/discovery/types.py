"""Evidence-bound mechanism discovery components: types."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol


class MechanismDiscoveryError(RuntimeError):
    """A mechanism-discovery transaction failed closed."""


_AXES = (
    "state_representation",
    "conditioning_path",
    "update_operator",
    "reliability_routing",
    "training_distribution",
    "learning_signal",
    "gradient_path",
    "inference_transition",
)


_ARXIV_ID = re.compile(r"(?<![0-9])([0-9]{4}\.[0-9]{4,5})(?:v([0-9]+))?(?![0-9])")


_COMPLEXITY_BUDGETS = {
    "light": {
        "max_papers": 12,
        "max_results_per_view": 3,
        "max_query_views": 4,
    },
    "full": {
        "max_papers": 200,
        "max_results_per_view": 20,
        "max_query_views": 100,
    },
}


@dataclass(frozen=True)
class DiscoveryRequest:
    """Diagnostic context used to plan retrieval without naming a method."""

    symptom_description: str
    failure_signatures: tuple[str, ...]
    target_metrics: tuple[str, ...]
    protected_metrics: tuple[str, ...]
    available_hooks: tuple[str, ...]
    model_family: str
    seed_arxiv_ids: tuple[str, ...] = ()
    cross_domain_lenses: tuple[str, ...] = ()


@dataclass(frozen=True)
class DiscoveryPaper:
    """Bounded paper metadata and evidence text."""

    arxiv_id: str
    title: str
    abstract: str
    source_url: str
    full_text_url: str
    full_text: str
    full_text_state: str
    citation_depth: int
    discovered_from: tuple[str, ...]
    local_source_path: str | None = None
    local_source_sha256: str | None = None


class MechanismExtractionClient(Protocol):
    """Semantic boundary for converting evidence into typed mechanism axes."""

    def extract(
        self,
        paper: DiscoveryPaper,
        *,
        request: DiscoveryRequest,
    ) -> Mapping[str, Any]:
        """Return evidence-bound axes; no executable fields are permitted."""
