"""Portable knowledge graph components: types."""

from __future__ import annotations

import re


class PortableKnowledgeGraphError(ValueError):
    """A portable-knowledge record or graph request was invalid."""


_SIMILARITY_FIELDS = {
    "available_capabilities",
    "available_hooks",
    "capabilities",
    "context_class",
    "dose_values",
    "horizons",
    "invariants",
    "model_family",
    "protected_fields",
    "provides",
    "requires",
    "semantic_dimensions",
    "source_semantics",
    "target_semantics",
}


_SEMANTIC_COUNT_KEY = re.compile(r"^[a-z][a-z0-9_]*$")


_CAS_SHA256 = re.compile(r"^cas://sha256/[0-9a-f]{64}$")
