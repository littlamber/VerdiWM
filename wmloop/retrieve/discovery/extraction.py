"""Evidence-bound mechanism discovery components: extraction."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from wmloop.retrieve.discovery.common import (
    _bounded_excerpt,
    _canonical_arxiv_id,
    _load_mapping,
    _normalise,
    _safe_id,
    _sha256,
    _valid_tag,
)
from wmloop.retrieve.discovery.types import (
    _AXES,
    DiscoveryPaper,
    DiscoveryRequest,
    MechanismDiscoveryError,
    MechanismExtractionClient,
)


def extractor_identity(extractor: MechanismExtractionClient) -> str:
    """Keep built-in identities stable in pre-existing atlas input digests."""
    kind = type(extractor)
    if kind in (EvidenceOnlyMechanismExtractor, AnnotationMechanismExtractor):
        return f"wmloop.retrieve.mechanism_discovery.{kind.__qualname__}"
    return f"{kind.__module__}.{kind.__qualname__}"


class EvidenceOnlyMechanismExtractor:
    """Conservative fallback when no semantic extractor is configured.

    It records evidence but leaves operator semantics unresolved.  In
    particular, it does not infer novelty from an unfamiliar title.
    """

    def extract(
        self,
        paper: DiscoveryPaper,
        *,
        request: DiscoveryRequest,
    ) -> Mapping[str, Any]:
        del request
        excerpt = _bounded_excerpt(paper.full_text or paper.abstract, paper.abstract)
        return {
            "axes": {axis: [] for axis in _AXES},
            "evidence_excerpts": [excerpt] if excerpt else [],
            "requirements": [],
            "failure_boundaries": [],
            "extraction_state": "unresolved",
            "extraction_rationale": (
                "No evidence-bound semantic extractor was configured; unfamiliar "
                "terminology is not treated as novelty evidence."
            ),
        }


class AnnotationMechanismExtractor:
    """Use reviewed semantic annotations keyed by exact arXiv revision."""

    def __init__(self, annotations: Mapping[str, Mapping[str, Any]]) -> None:
        self._annotations = {
            _canonical_arxiv_id(key): dict(value) for key, value in annotations.items()
        }

    @classmethod
    def from_path(cls, path: Path) -> "AnnotationMechanismExtractor":
        payload = _load_mapping(path, "MECHANISM_ANNOTATIONS_INVALID")
        annotations = payload.get("annotations")
        if not isinstance(annotations, Mapping):
            raise MechanismDiscoveryError("MECHANISM_ANNOTATIONS_INVALID")
        typed: dict[str, Mapping[str, Any]] = {}
        for key, value in annotations.items():
            if not isinstance(key, str) or not isinstance(value, Mapping):
                raise MechanismDiscoveryError("MECHANISM_ANNOTATIONS_INVALID")
            typed[key] = value
        return cls(typed)

    def extract(
        self,
        paper: DiscoveryPaper,
        *,
        request: DiscoveryRequest,
    ) -> Mapping[str, Any]:
        del request
        exact = self._annotations.get(_canonical_arxiv_id(paper.arxiv_id))
        if exact is None:
            return EvidenceOnlyMechanismExtractor().extract(
                paper,
                request=DiscoveryRequest("unresolved", (), (), (), (), "unknown"),
            )
        result = dict(exact)
        _validate_extraction(result, paper=paper)
        return result


def _atlas_entry(
    paper: DiscoveryPaper,
    extraction: Mapping[str, Any],
    comparison: Mapping[str, Any],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_type": "wmloop-mechanism-atlas-entry",
        "candidate_id": "mechanism-" + _safe_id(paper.arxiv_id),
        "source": {
            "arxiv_id": paper.arxiv_id,
            "title": paper.title,
            "source_url": paper.source_url,
            "full_text_url": paper.full_text_url,
            "full_text_state": paper.full_text_state,
            "full_text_sha256": _sha256(paper.full_text.encode("utf-8"))
            if paper.full_text
            else None,
            "citation_depth": paper.citation_depth,
            "discovered_from": list(paper.discovered_from),
            "local_source_path": paper.local_source_path,
            "local_source_sha256": paper.local_source_sha256,
        },
        "mechanism": extraction,
        "comparison": comparison,
        "execution_authority": "shadow_only",
    }


def _validate_extraction(
    extraction: Mapping[str, Any], *, paper: DiscoveryPaper
) -> None:
    axes = extraction.get("axes")
    if not isinstance(axes, Mapping) or set(axes) != set(_AXES):
        raise MechanismDiscoveryError("MECHANISM_EXTRACTION_AXES_INVALID")
    for value in axes.values():
        if not isinstance(value, list) or any(
            not isinstance(tag, str) or not _valid_tag(tag) for tag in value
        ):
            raise MechanismDiscoveryError("MECHANISM_EXTRACTION_TAG_INVALID")
    state = extraction.get("extraction_state")
    if state not in {"unresolved", "evidence_supported"}:
        raise MechanismDiscoveryError("MECHANISM_EXTRACTION_STATE_INVALID")
    excerpts = extraction.get("evidence_excerpts")
    if not isinstance(excerpts, list) or any(
        not isinstance(value, str) or not value.strip() for value in excerpts
    ):
        raise MechanismDiscoveryError("MECHANISM_EXTRACTION_EVIDENCE_INVALID")
    evidence = _normalise(" ".join((paper.abstract, paper.full_text)))
    for excerpt in excerpts:
        if _normalise(excerpt) not in evidence:
            raise MechanismDiscoveryError("MECHANISM_EXTRACTION_EVIDENCE_UNBOUND")
    for field in ("requirements", "failure_boundaries"):
        value = extraction.get(field)
        if not isinstance(value, list) or any(
            not isinstance(item, str) or not item for item in value
        ):
            raise MechanismDiscoveryError("MECHANISM_EXTRACTION_BOUNDARY_INVALID")
    rationale = extraction.get("extraction_rationale")
    if not isinstance(rationale, str) or len(rationale) < 12:
        raise MechanismDiscoveryError("MECHANISM_EXTRACTION_RATIONALE_INVALID")
