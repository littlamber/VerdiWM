"""Evidence-bound mechanism discovery beyond method-name matching.

Keyword search is useful for collecting seed papers, but a missing method name
is not novelty evidence.  This module expands seed papers through their cited
arXiv neighbourhood, captures bounded full-text evidence, and compares typed
mechanism signatures against typed registry profiles.  It never grants
experiment or source-mutation authority.
"""

from __future__ import annotations

import hashlib
import html
import heapq
import json
import re
import shutil
import subprocess
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ElementTree
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Protocol

from wmloop.contracts import ContractValidationError, validate_document
from wmloop.primitives.registry import PrimitiveRegistry
from wmloop.retrieve.literature import LiteratureRetrievalError, search_arxiv


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


def build_multiview_queries(request: DiscoveryRequest) -> tuple[dict[str, str], ...]:
    """Build diagnostic, operator, and cross-domain queries.

    These queries only collect candidates.  They are deliberately not used to
    label equivalence or novelty.
    """

    symptom = _query_terms(
        " ".join((request.symptom_description, *request.failure_signatures)),
        maximum=7,
    )
    metrics = _query_terms(" ".join(request.target_metrics), maximum=5)
    domain = ("video", "diffusion", "world", "model")
    rows = [
        {
            "view": "diagnostic_symptom",
            "query": _join_query(domain, ("history", "conditioning", "long", "horizon", "drift"), symptom[:4]),
        },
        {
            "view": "state_update_operator",
            "query": _join_query(
                domain,
                ("adaptive", "history", "memory", "reliability", "update"),
                metrics,
            ),
        },
        {
            "view": "training_distribution",
            "query": _join_query(
                domain,
                ("autoregressive", "generated", "history", "training", "inference", "rollout"),
            ),
        },
        {
            "view": "architecture_hook",
            "query": _join_query(
                domain,
                ("temporal", "latent", "conditioning", "memory", "action"),
            ),
        },
    ]
    lenses = request.cross_domain_lenses or (
        "belief state estimation confidence update",
        "adaptive memory retention forgetting",
        "online system identification uncertainty",
    )
    for index, lens in enumerate(lenses):
        rows.append(
            {
                "view": f"cross_domain_{index + 1}",
                "query": _join_query(
                    ("sequence", "prediction", "temporal", "memory"),
                    _query_terms(lens, maximum=8),
                ),
            }
        )
    deduplicated: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        query = row["query"].strip()
        if query and query not in seen:
            deduplicated.append(row)
            seen.add(query)
    return tuple(deduplicated)


def run_mechanism_discovery(
    *,
    request: DiscoveryRequest,
    seed_records: Sequence[Mapping[str, Any]],
    output_root: Path,
    repo_root: Path,
    annotations_path: Path | None = None,
    citation_depth: int = 1,
    max_papers: int = 12,
    timeout_seconds: float = 20.0,
    full_text_fetcher: Any | None = None,
    metadata_fetcher: Any | None = None,
    search_client: Any | None = None,
    search_results_per_view: int = 3,
    complexity_budget: str = "light",
    extraction_client: MechanismExtractionClient | None = None,
    local_sources_root: Path | None = None,
    reference_atlas_paths: Sequence[Path] = (),
    reference_profiles_paths: Sequence[Path] = (),
) -> dict[str, object]:
    """Create an immutable evidence-bound mechanism atlas.

    Seed collection may come from lexical or semantic search.  Novelty is
    assessed only from typed mechanism axes backed by evidence excerpts.
    """

    budget = _complexity_budget(complexity_budget)
    if citation_depth < 0 or citation_depth > 2:
        raise MechanismDiscoveryError("MECHANISM_CITATION_DEPTH_INVALID")
    if max_papers < 1 or max_papers > int(budget["max_papers"]):
        raise MechanismDiscoveryError("MECHANISM_MAX_PAPERS_INVALID")
    if timeout_seconds <= 0 or timeout_seconds > 60:
        raise MechanismDiscoveryError("MECHANISM_TIMEOUT_INVALID")
    if search_results_per_view < 0 or search_results_per_view > int(budget["max_results_per_view"]):
        raise MechanismDiscoveryError("MECHANISM_SEARCH_RESULTS_INVALID")
    root = Path(repo_root).resolve(strict=True)
    destination = Path(output_root).resolve()
    local_root = Path(local_sources_root).resolve() if local_sources_root is not None else None
    if local_root is not None:
        local_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if local_root.is_symlink() or not local_root.is_dir():
            raise MechanismDiscoveryError("MECHANISM_LOCAL_SOURCES_INVALID")
    if destination.exists() or destination.is_symlink():
        raise MechanismDiscoveryError("MECHANISM_OUTPUT_EXISTS")
    profiles_path = root / "configs" / "retrieval" / "primitive_mechanism_profiles_v1.json"
    ontology_path = root / "configs" / "retrieval" / "mechanism_tag_ontology_v1.json"
    profiles = _load_profiles(profiles_path)
    registry_names = set(PrimitiveRegistry.from_root(root).names())
    if set(profiles) != registry_names:
        missing = ",".join(sorted(registry_names - set(profiles)))
        extra = ",".join(sorted(set(profiles) - registry_names))
        raise MechanismDiscoveryError(f"MECHANISM_PROFILE_COVERAGE_MISMATCH:{missing}:{extra}")
    ontology = _load_ontology(ontology_path)
    reference_kinds = {name: "registered_primitive" for name in profiles}
    reference_sources: list[dict[str, str]] = []
    for raw_path in reference_atlas_paths:
        path = Path(raw_path).resolve(strict=True)
        loaded = _load_reference_atlas_profiles(path)
        _merge_profiles(profiles, loaded, source=path)
        reference_kinds.update({name: "evidence_supported_literature" for name in loaded})
        reference_sources.append({"kind": "atlas", "path": str(path), "sha256": _sha256(path.read_bytes())})
    for raw_path in reference_profiles_paths:
        path = Path(raw_path).resolve(strict=True)
        loaded, kinds = _load_reference_profiles(path)
        _merge_profiles(profiles, loaded, source=path)
        reference_kinds.update(kinds)
        reference_sources.append(
            {"kind": "reference_profiles", "path": str(path), "sha256": _sha256(path.read_bytes())}
        )
    if extraction_client is not None and annotations_path is not None:
        raise MechanismDiscoveryError("MECHANISM_EXTRACTOR_AMBIGUOUS")
    extractor: MechanismExtractionClient = extraction_client or (
        AnnotationMechanismExtractor.from_path(annotations_path)
        if annotations_path is not None
        else EvidenceOnlyMechanismExtractor()
    )
    fetch_text = full_text_fetcher or _fetch_ar5iv_text
    fetch_metadata = metadata_fetcher or _fetch_arxiv_metadata
    all_queries = build_multiview_queries(request)
    queries = all_queries[: int(budget["max_query_views"])]
    searched_records, search_states = _collect_query_seeds(
        queries,
        max_results=search_results_per_view,
        timeout_seconds=timeout_seconds,
        search_client=search_client,
    )
    combined_seeds = _deduplicate_seed_records((*seed_records, *searched_records))
    input_payload = {
        "request": _request_dict(request),
        "seed_records": [dict(row) for row in combined_seeds],
        "citation_depth": citation_depth,
        "max_papers": max_papers,
        "search_results_per_view": search_results_per_view,
        "complexity_budget": complexity_budget,
        "query_view_count": len(queries),
        "profiles_sha256": _sha256(profiles_path.read_bytes()),
        "ontology_sha256": _sha256(ontology_path.read_bytes()),
        "reference_sources": reference_sources,
        "annotations_sha256": (
            _sha256(Path(annotations_path).resolve(strict=True).read_bytes())
            if annotations_path is not None
            else None
        ),
        "extraction_client": f"{type(extractor).__module__}.{type(extractor).__qualname__}",
        "local_sources": _local_source_inventory(local_root),
    }
    papers, raw_missing_sources = _expand_papers(
        seed_records=combined_seeds,
        seed_ids=request.seed_arxiv_ids,
        citation_depth=citation_depth,
        max_papers=max_papers,
        timeout_seconds=timeout_seconds,
        fetch_text=fetch_text,
        fetch_metadata=fetch_metadata,
        local_sources_root=local_root,
    )
    missing_sources = list(raw_missing_sources)
    materials: list[tuple[DiscoveryPaper, dict[str, Any]]] = []
    for paper in papers:
        try:
            extraction = dict(extractor.extract(paper, request=request))
        except MechanismDiscoveryError as exc:
            if str(exc) != "MECHANISM_EXTRACTION_EVIDENCE_UNBOUND":
                raise
            extraction = dict(EvidenceOnlyMechanismExtractor().extract(paper, request=request))
            extraction["failure_boundaries"] = [
                "A reviewed annotation exists, but its quoted evidence could not be bound "
                "to the abstract or full text available in this run."
            ]
            extraction["extraction_rationale"] = (
                "The reviewed mechanism annotation was withheld because its evidence quote "
                "was unavailable in the fetched source. The candidate remains unresolved "
                "until the exact revision is supplied or the annotation is reviewed."
            )
            _merge_missing_source(
                missing_sources,
                _missing_source_record(
                    arxiv_id=paper.arxiv_id,
                    title=paper.title,
                    reasons=("annotation_evidence_unbound",),
                    local_sources_root=local_root,
                ),
            )
        _validate_extraction(extraction, paper=paper)
        materials.append((paper, extraction))

    peer_profiles = {
        f"candidate_pool:{paper.arxiv_id}": _validated_profile_axes(
            extraction["axes"],
            code="MECHANISM_EXTRACTION_AXES_INVALID",
        )
        for paper, extraction in materials
        if extraction.get("extraction_state") == "evidence_supported"
    }
    entries: list[dict[str, object]] = []
    for paper, extraction in materials:
        comparison_profiles = dict(profiles)
        comparison_kinds = dict(reference_kinds)
        for reference_id, peer_profile in peer_profiles.items():
            if reference_id == f"candidate_pool:{paper.arxiv_id}":
                continue
            comparison_profiles[reference_id] = peer_profile
            comparison_kinds[reference_id] = "evidence_supported_candidate_pool"
        comparison = compare_mechanism_signature(
            extraction,
            comparison_profiles,
            ontology=ontology,
            reference_kinds=comparison_kinds,
        )
        entry = _atlas_entry(paper, extraction, comparison)
        try:
            validate_document("mechanism_atlas_entry", entry, root=root)
        except ContractValidationError as exc:
            raise MechanismDiscoveryError(f"MECHANISM_ATLAS_SCHEMA_INVALID:{exc}") from exc
        entries.append(entry)
    report = {
        "schema_version": 1,
        "artifact_type": "wmloop-mechanism-discovery-atlas",
        "state": "ready",
        "input_hash": _sha256(_canonical_json(input_payload)),
        "request": _request_dict(request),
        "query_plan": [
            {**query, "source_state": search_states.get(query["view"], "disabled")}
            for query in queries
        ],
        "complexity_budget": {
            "name": complexity_budget,
            "max_papers": int(budget["max_papers"]),
            "max_results_per_view": int(budget["max_results_per_view"]),
            "max_query_views": int(budget["max_query_views"]),
            "query_view_count": len(queries),
            "query_views_truncated": len(queries) < len(all_queries),
        },
        "seed_count": len(combined_seeds) + len(request.seed_arxiv_ids),
        "paper_count": len(entries),
        "full_text_supported_count": sum(
            entry["source"]["full_text_state"] in {"network", "cached", "local"}  # type: ignore[index]
            for entry in entries
        ),
        "novelty_resolved_count": sum(
            entry["comparison"]["novelty_state"] != "unresolved" for entry in entries  # type: ignore[index]
        ),
        "missing_source_count": len(missing_sources),
        "local_sources_root": str(local_root) if local_root is not None else None,
        "comparison_ontology": str(ontology_path),
        "comparison_ontology_sha256": _sha256(ontology_path.read_bytes()),
        "reference_profile_count": len(profiles),
        "candidate_pool_reference_count": len(peer_profiles),
        "reference_sources": reference_sources,
        "entries": entries,
        "claim_boundary": (
            "The atlas supports research ranking only. A missing lexical match is "
            "never novelty evidence; unresolved, equivalent, extension, composition, "
            "and structural-candidate labels require separate target-side validation."
        ),
        "execution_authority": "shadow_only",
    }
    destination.mkdir(mode=0o700, parents=True)
    _write_json(destination / "mechanism-atlas.json", report)
    missing_report = {
        "schema_version": 1,
        "artifact_type": "wmloop-mechanism-missing-sources",
        "state": "action_required" if missing_sources else "complete",
        "local_sources_root": str(local_root) if local_root is not None else None,
        "missing_source_count": len(missing_sources),
        "records": missing_sources,
        "instructions": (
            "Place one accepted file for each arXiv revision in local_sources_root "
            "and rerun into a new output directory. Exact-revision filenames are preferred."
        ),
    }
    _write_json(destination / "missing-sources.json", missing_report)
    (destination / "MISSING_SOURCES.md").write_text(
        _render_missing_sources_markdown(missing_report),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "artifact_type": "wmloop-mechanism-discovery-manifest",
        "state": "ready",
        "input_hash": report["input_hash"],
        "atlas_path": str(destination / "mechanism-atlas.json"),
        "atlas_sha256": _sha256((destination / "mechanism-atlas.json").read_bytes()),
        "paper_count": len(entries),
        "missing_source_count": len(missing_sources),
        "missing_sources_path": str(destination / "missing-sources.json"),
        "missing_sources_sha256": _sha256((destination / "missing-sources.json").read_bytes()),
        "missing_sources_markdown_path": str(destination / "MISSING_SOURCES.md"),
        "execution_authority": "shadow_only",
        "claim_boundary": report["claim_boundary"],
    }
    _write_json(destination / "manifest.json", manifest)
    return manifest


def _collect_query_seeds(
    queries: Sequence[Mapping[str, str]],
    *,
    max_results: int,
    timeout_seconds: float,
    search_client: Any | None,
) -> tuple[tuple[Mapping[str, Any], ...], dict[str, str]]:
    if max_results == 0:
        return (), {str(row["view"]): "disabled" for row in queries}
    search = search_client or search_arxiv
    records: list[Mapping[str, Any]] = []
    states: dict[str, str] = {}
    for row in queries:
        view = str(row["view"])
        try:
            found, state = search(
                str(row["query"]),
                max_results=max_results,
                timeout_seconds=timeout_seconds,
            )
        except (LiteratureRetrievalError, OSError):
            found, state = (), "offline"
        states[view] = str(state)
        for record in found:
            if hasattr(record, "to_dict"):
                records.append(record.to_dict())
            elif isinstance(record, Mapping):
                records.append(dict(record))
    return _deduplicate_seed_records(records), states


def _complexity_budget(name: str) -> Mapping[str, int]:
    budget = _COMPLEXITY_BUDGETS.get(name)
    if budget is None:
        raise MechanismDiscoveryError("MECHANISM_COMPLEXITY_BUDGET_INVALID")
    return budget


def _deduplicate_seed_records(
    records: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    deduplicated: dict[str, Mapping[str, Any]] = {}
    for row in records:
        raw_id = str(row.get("arxiv_id") or "")
        if not raw_id:
            continue
        try:
            key = _base_arxiv_id(raw_id)
        except MechanismDiscoveryError as exc:
            if str(exc) != "MECHANISM_ARXIV_ID_INVALID":
                raise
            continue
        current = deduplicated.get(key)
        if current is None or len(str(row.get("abstract") or "")) > len(str(current.get("abstract") or "")):
            deduplicated[key] = dict(row)
    return tuple(deduplicated.values())


def compare_mechanism_signature(
    extraction: Mapping[str, Any],
    profiles: Mapping[str, Mapping[str, Sequence[str]]],
    *,
    ontology: Mapping[str, Mapping[str, Sequence[str]]] | None = None,
    reference_kinds: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Compare controlled mechanism concepts, never titles or method names."""

    axes = extraction.get("axes")
    if extraction.get("extraction_state") != "evidence_supported" or not isinstance(axes, Mapping):
        return {
            "novelty_state": "unresolved",
            "nearest_primitives": [],
            "novel_axes": [],
            "novelty_basis": "Typed mechanism semantics are incomplete or not evidence-supported.",
        }
    candidate = {
        axis: _canonical_concepts(axis, _tag_set(axes.get(axis, [])), ontology)
        for axis in _AXES
    }
    if sum(bool(tags) for tags in candidate.values()) < 4:
        return {
            "novelty_state": "unresolved",
            "nearest_primitives": [],
            "novel_axes": [],
            "novelty_basis": "Fewer than four mechanism axes are populated.",
        }
    scored: list[dict[str, object]] = []
    known_tags: dict[str, set[str]] = {axis: set() for axis in _AXES}
    for primitive, profile in sorted(profiles.items()):
        overlaps: list[str] = []
        differences: list[str] = []
        matched_concepts: dict[str, list[str]] = {}
        intersection = 0
        union = 0
        for axis in _AXES:
            reference = _canonical_concepts(
                axis,
                _tag_set(profile.get(axis, [])),
                ontology,
            )
            known_tags[axis].update(reference)
            current = candidate[axis]
            shared = current & reference
            if shared:
                overlaps.append(axis)
                matched_concepts[axis] = sorted(shared)
                intersection += 1
            if current != reference:
                differences.append(axis)
            if current or reference:
                union += 1
        score = intersection / union if union else 0.0
        scored.append(
            {
                "primitive": primitive,
                "reference_kind": str(
                    (reference_kinds or {}).get(primitive, "registered_primitive")
                ),
                "structural_similarity": round(score, 6),
                "matching_axes": overlaps,
                "differing_axes": differences,
                "matched_concepts": matched_concepts,
            }
        )
    scored.sort(key=lambda row: (-float(row["structural_similarity"]), str(row["primitive"])))
    nearest = scored[:3]
    best = float(nearest[0]["structural_similarity"]) if nearest else 0.0
    novel_axes = [
        axis for axis in _AXES if candidate[axis] and not candidate[axis].issubset(known_tags[axis])
    ]
    if best >= 0.85 and not novel_axes:
        state = "equivalent"
    elif best >= 0.45 and novel_axes:
        state = "extension"
    elif len([row for row in nearest if float(row["structural_similarity"]) >= 0.25]) >= 2:
        state = "composition"
    else:
        state = "structural_candidate"
    return {
        "novelty_state": state,
        "nearest_primitives": nearest,
        "novel_axes": novel_axes,
        "novelty_basis": (
            "Classification uses ontology-canonicalized operator concepts, evidence-backed "
            "annotations, registered primitives, and supplied prior-mechanism references; "
            "it is a screening label, not a publication-level novelty claim."
        ),
    }


def _canonical_concepts(
    axis: str,
    tags: set[str],
    ontology: Mapping[str, Mapping[str, Sequence[str]]] | None,
) -> set[str]:
    if ontology is None:
        return set(tags)
    concepts = ontology.get(axis, {})
    aliases = {
        str(alias): str(concept)
        for concept, values in concepts.items()
        for alias in values
    }
    return {aliases.get(tag, f"unmapped:{tag}") for tag in tags}


def _expand_papers(
    *,
    seed_records: Sequence[Mapping[str, Any]],
    seed_ids: Sequence[str],
    citation_depth: int,
    max_papers: int,
    timeout_seconds: float,
    fetch_text: Any,
    fetch_metadata: Any,
    local_sources_root: Path | None,
) -> tuple[tuple[DiscoveryPaper, ...], tuple[dict[str, object], ...]]:
    metadata: dict[str, dict[str, str]] = {}
    queue: list[tuple[int, int, str, int, tuple[str, ...]]] = []
    order = 0
    for row in seed_records:
        raw_id = str(row.get("arxiv_id") or "")
        if not raw_id:
            continue
        arxiv_id = _canonical_arxiv_id(raw_id)
        metadata[arxiv_id] = {
            "arxiv_id": arxiv_id,
            "title": str(row.get("title") or "Unknown title"),
            "abstract": str(row.get("abstract") or row.get("mechanism_summary") or ""),
            "source_url": str(row.get("pdf_url") or row.get("source_url") or f"https://arxiv.org/abs/{arxiv_id}"),
        }
        heapq.heappush(queue, (20, order, arxiv_id, 0, ("query_or_seed_record",)))
        order += 1
    for raw_id in seed_ids:
        arxiv_id = _canonical_arxiv_id(raw_id)
        heapq.heappush(queue, (0, order, arxiv_id, 0, ("seed_request",)))
        order += 1
    papers: list[DiscoveryPaper] = []
    missing_sources: list[dict[str, object]] = []
    seen: set[str] = set()
    while queue and len(papers) < max_papers:
        batch: list[tuple[int, int, str, int, tuple[str, ...]]] = []
        while queue and len(batch) < 4 and len(papers) + len(batch) < max_papers:
            item = heapq.heappop(queue)
            base_id = _base_arxiv_id(item[2])
            if base_id in seen:
                continue
            seen.add(base_id)
            batch.append(item)
        if not batch:
            continue
        with ThreadPoolExecutor(max_workers=len(batch)) as executor:
            loaded = list(
                executor.map(
                    lambda item: _load_discovery_paper(
                        item,
                        metadata=metadata,
                        timeout_seconds=timeout_seconds,
                        fetch_text=fetch_text,
                        fetch_metadata=fetch_metadata,
                        local_sources_root=local_sources_root,
                    ),
                    batch,
                )
            )
        for item, result in zip(batch, loaded):
            paper, cited_ids, missing = result
            papers.append(paper)
            if missing is not None:
                missing_sources.append(missing)
            priority = item[0]
            if paper.citation_depth < citation_depth:
                citation_priority = 5 if priority < 20 else 25
                for cited_id in cited_ids:
                    if _base_arxiv_id(cited_id) == _base_arxiv_id(paper.arxiv_id):
                        continue
                    heapq.heappush(
                        queue,
                        (
                            citation_priority,
                            order,
                            str(cited_id),
                            paper.citation_depth + 1,
                            (paper.arxiv_id,),
                        ),
                    )
                    order += 1
    return tuple(papers), tuple(missing_sources)


def _load_discovery_paper(
    item: tuple[int, int, str, int, tuple[str, ...]],
    *,
    metadata: Mapping[str, Mapping[str, str]],
    timeout_seconds: float,
    fetch_text: Any,
    fetch_metadata: Any,
    local_sources_root: Path | None,
) -> tuple[DiscoveryPaper, tuple[str, ...], dict[str, object] | None]:
    _, _, arxiv_id, depth, parents = item
    row = metadata.get(arxiv_id) or metadata.get(_base_arxiv_id(arxiv_id))
    metadata_state = "seed"
    if row is None:
        try:
            fetched = fetch_metadata(arxiv_id, timeout_seconds=timeout_seconds)
        except Exception:
            fetched = None
        if isinstance(fetched, Mapping):
            row = {key: str(value) for key, value in fetched.items()}
            arxiv_id = _canonical_arxiv_id(row.get("arxiv_id") or arxiv_id)
            metadata_state = "network"
        else:
            row = {
                "arxiv_id": arxiv_id,
                "title": f"Metadata unavailable for {arxiv_id}",
                "abstract": "",
                "source_url": f"https://arxiv.org/abs/{arxiv_id}",
            }
            metadata_state = "unavailable"
    full_text_url = f"https://ar5iv.labs.arxiv.org/html/{_base_arxiv_id(arxiv_id)}"
    local_path: Path | None = None
    local_sha256: str | None = None
    local_error: str | None = None
    local_result = _load_local_source(arxiv_id, local_sources_root)
    if local_result is not None:
        full_text, cited_ids, state, local_path, local_error = local_result
        local_sha256 = _sha256(local_path.read_bytes()) if local_path is not None else None
    else:
        try:
            full_text, cited_ids, state = fetch_text(
                arxiv_id,
                timeout_seconds=timeout_seconds,
            )
        except Exception:
            full_text, cited_ids, state = "", (), "unavailable"
    paper = DiscoveryPaper(
        arxiv_id=arxiv_id,
        title=str(row.get("title") or "Unknown title"),
        abstract=str(row.get("abstract") or ""),
        source_url=str(row.get("source_url") or f"https://arxiv.org/abs/{arxiv_id}"),
        full_text_url=full_text_url,
        full_text=str(full_text)[:1_000_000],
        full_text_state=str(state),
        citation_depth=depth,
        discovered_from=parents,
        local_source_path=str(local_path) if local_path is not None else None,
        local_source_sha256=local_sha256,
    )
    missing = None
    if not full_text:
        reasons = []
        if metadata_state == "unavailable":
            reasons.append("metadata_api_unavailable")
        reasons.append(local_error or "full_text_network_unavailable")
        missing = _missing_source_record(
            arxiv_id=arxiv_id,
            title=paper.title,
            reasons=reasons,
            local_sources_root=local_sources_root,
        )
    return paper, tuple(str(value) for value in cited_ids), missing


def _fetch_ar5iv_text(
    arxiv_id: str,
    *,
    timeout_seconds: float,
) -> tuple[str, tuple[str, ...], str]:
    url = f"https://ar5iv.labs.arxiv.org/html/{_base_arxiv_id(arxiv_id)}"
    request = urllib.request.Request(url, headers={"User-Agent": "verdiwm/0.1 mechanism-discovery"})
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            payload = response.read(4_000_001)
    except (OSError, urllib.error.URLError) as exc:
        raise MechanismDiscoveryError("MECHANISM_FULL_TEXT_UNAVAILABLE") from exc
    if len(payload) > 4_000_000:
        raise MechanismDiscoveryError("MECHANISM_FULL_TEXT_TOO_LARGE")
    parser = _BoundedHTMLTextParser()
    parser.feed(payload.decode("utf-8", errors="replace"))
    text = parser.text()
    cited = tuple(dict.fromkeys(_base_arxiv_id(value) for value in parser.arxiv_ids()))
    return text, cited, "network"


def _fetch_arxiv_metadata(arxiv_id: str, *, timeout_seconds: float) -> Mapping[str, str]:
    url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
        {"id_list": _base_arxiv_id(arxiv_id), "start": 0, "max_results": 1}
    )
    request = urllib.request.Request(url, headers={"User-Agent": "verdiwm/0.1 mechanism-discovery"})
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            payload = response.read(2_000_001)
        if len(payload) > 2_000_000:
            raise MechanismDiscoveryError("MECHANISM_METADATA_TOO_LARGE")
        root = ElementTree.fromstring(payload)
    except (OSError, urllib.error.URLError, ElementTree.ParseError) as exc:
        raise MechanismDiscoveryError("MECHANISM_METADATA_UNAVAILABLE") from exc
    namespace = {"atom": "http://www.w3.org/2005/Atom"}
    entry = root.find("atom:entry", namespace)
    if entry is None:
        raise MechanismDiscoveryError("MECHANISM_METADATA_UNAVAILABLE")
    identifier = _node_text(entry.find("atom:id", namespace))
    match = _ARXIV_ID.search(identifier)
    return {
        "arxiv_id": match.group(0) if match else arxiv_id,
        "title": " ".join(_node_text(entry.find("atom:title", namespace)).split()),
        "abstract": " ".join(_node_text(entry.find("atom:summary", namespace)).split()),
        "source_url": identifier,
    }


def _load_local_source(
    arxiv_id: str,
    root: Path | None,
) -> tuple[str, tuple[str, ...], str, Path, str | None] | None:
    if root is None:
        return None
    candidates: list[Path] = []
    for stem in (_canonical_arxiv_id(arxiv_id), _base_arxiv_id(arxiv_id)):
        for suffix in (".txt", ".html", ".htm", ".pdf"):
            path = root / f"{stem}{suffix}"
            if path.is_file() and not path.is_symlink():
                candidates.append(path)
    if not candidates:
        return None
    path = candidates[0].resolve()
    if root not in path.parents:
        raise MechanismDiscoveryError("MECHANISM_LOCAL_SOURCE_OUTSIDE_ROOT")
    if path.stat().st_size > 50_000_000:
        raise MechanismDiscoveryError("MECHANISM_LOCAL_SOURCE_TOO_LARGE")
    suffix = path.suffix.lower()
    if suffix == ".txt":
        text = path.read_text(encoding="utf-8", errors="replace")[:1_000_000]
        return text, _arxiv_ids_from_text(text), "local", path, None
    if suffix in {".html", ".htm"}:
        parser = _BoundedHTMLTextParser()
        parser.feed(path.read_text(encoding="utf-8", errors="replace"))
        return parser.text(), parser.arxiv_ids(), "local", path, None
    converter = shutil.which("pdftotext")
    if converter is None:
        return "", (), "unavailable", path, "local_pdf_parser_unavailable"
    try:
        completed = subprocess.run(
            [converter, str(path), "-"],
            check=True,
            capture_output=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return "", (), "unavailable", path, "local_pdf_parse_failed"
    text = completed.stdout.decode("utf-8", errors="replace")[:1_000_000]
    return text, _arxiv_ids_from_text(text), "local", path, None


def _local_source_inventory(root: Path | None) -> list[dict[str, object]]:
    if root is None:
        return []
    records: list[dict[str, object]] = []
    for path in sorted(root.iterdir()):
        if (
            path.is_symlink()
            or not path.is_file()
            or path.suffix.lower() not in {".txt", ".html", ".htm", ".pdf"}
        ):
            continue
        records.append(
            {
                "path": str(path.resolve()),
                "sha256": _sha256(path.read_bytes()),
                "size_bytes": path.stat().st_size,
            }
        )
    return records


def _missing_source_record(
    *,
    arxiv_id: str,
    title: str,
    reasons: Sequence[str],
    local_sources_root: Path | None,
) -> dict[str, object]:
    revision = _canonical_arxiv_id(arxiv_id)
    return {
        "arxiv_id": revision,
        "title": title,
        "reasons": list(dict.fromkeys(str(value) for value in reasons if value)),
        "pdf_url": f"https://arxiv.org/pdf/{revision}",
        "html_url": f"https://ar5iv.labs.arxiv.org/html/{_base_arxiv_id(revision)}",
        "accepted_filenames": [
            f"{revision}.txt",
            f"{revision}.html",
            f"{revision}.pdf",
            f"{_base_arxiv_id(revision)}.txt",
            f"{_base_arxiv_id(revision)}.html",
            f"{_base_arxiv_id(revision)}.pdf",
        ],
        "destination_root": str(local_sources_root) if local_sources_root is not None else None,
        "integrity_rule": "The rerun records SHA256 and size for every local source file.",
    }


def _merge_missing_source(
    records: list[dict[str, object]],
    candidate: Mapping[str, object],
) -> None:
    arxiv_id = str(candidate.get("arxiv_id", ""))
    for record in records:
        if str(record.get("arxiv_id", "")) != arxiv_id:
            continue
        record["reasons"] = list(
            dict.fromkeys(
                [
                    *(str(value) for value in record.get("reasons", [])),
                    *(str(value) for value in candidate.get("reasons", [])),
                ]
            )
        )
        return
    records.append(dict(candidate))


def _render_missing_sources_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Missing Literature Sources",
        "",
        f"Local source directory: `{report.get('local_sources_root') or 'not configured'}`",
        "",
        "Download one TXT, HTML, or PDF file per record using an accepted filename, then rerun into a new output directory.",
        "",
    ]
    records = report.get("records")
    if (
        not isinstance(records, Sequence)
        or isinstance(records, (str, bytes))
        or not records
    ):
        lines.append("No missing sources.")
        return "\n".join(lines) + "\n"
    for row in records:
        if not isinstance(row, Mapping):
            continue
        lines.extend(
            [
                f"## {row.get('arxiv_id')} - {row.get('title')}",
                "",
                f"Reason: `{', '.join(str(value) for value in row.get('reasons', []))}`",
                "",
                f"PDF: {row.get('pdf_url')}",
                "",
                f"HTML: {row.get('html_url')}",
                "",
                "Accepted filenames: " + ", ".join(
                    f"`{value}`" for value in row.get("accepted_filenames", [])
                ),
                "",
            ]
        )
    return "\n".join(lines)


def _arxiv_ids_from_text(value: str) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(_base_arxiv_id(match.group(0)) for match in _ARXIV_ID.finditer(value))
    )


class _BoundedHTMLTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._text: list[str] = []
        self._hrefs: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "svg", "noscript"}:
            self._ignored_depth += 1
        if tag == "a":
            for key, value in attrs:
                if key == "href" and value:
                    self._hrefs.append(value)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "svg", "noscript"} and self._ignored_depth:
            self._ignored_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth and data.strip():
            self._text.append(data.strip())

    def text(self) -> str:
        return " ".join(" ".join(self._text).split())[:1_000_000]

    def arxiv_ids(self) -> tuple[str, ...]:
        values: list[str] = []
        for value in self._hrefs:
            if "arxiv" not in value.lower():
                continue
            values.extend(match.group(0) for match in _ARXIV_ID.finditer(value))
        return tuple(values)


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
            "full_text_sha256": _sha256(paper.full_text.encode("utf-8")) if paper.full_text else None,
            "citation_depth": paper.citation_depth,
            "discovered_from": list(paper.discovered_from),
            "local_source_path": paper.local_source_path,
            "local_source_sha256": paper.local_source_sha256,
        },
        "mechanism": extraction,
        "comparison": comparison,
        "execution_authority": "shadow_only",
    }


def _validate_extraction(extraction: Mapping[str, Any], *, paper: DiscoveryPaper) -> None:
    axes = extraction.get("axes")
    if not isinstance(axes, Mapping) or set(axes) != set(_AXES):
        raise MechanismDiscoveryError("MECHANISM_EXTRACTION_AXES_INVALID")
    for value in axes.values():
        if not isinstance(value, list) or any(not isinstance(tag, str) or not _valid_tag(tag) for tag in value):
            raise MechanismDiscoveryError("MECHANISM_EXTRACTION_TAG_INVALID")
    state = extraction.get("extraction_state")
    if state not in {"unresolved", "evidence_supported"}:
        raise MechanismDiscoveryError("MECHANISM_EXTRACTION_STATE_INVALID")
    excerpts = extraction.get("evidence_excerpts")
    if not isinstance(excerpts, list) or any(not isinstance(value, str) or not value.strip() for value in excerpts):
        raise MechanismDiscoveryError("MECHANISM_EXTRACTION_EVIDENCE_INVALID")
    evidence = _normalise(" ".join((paper.abstract, paper.full_text)))
    for excerpt in excerpts:
        if _normalise(excerpt) not in evidence:
            raise MechanismDiscoveryError("MECHANISM_EXTRACTION_EVIDENCE_UNBOUND")
    for field in ("requirements", "failure_boundaries"):
        value = extraction.get(field)
        if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
            raise MechanismDiscoveryError("MECHANISM_EXTRACTION_BOUNDARY_INVALID")
    rationale = extraction.get("extraction_rationale")
    if not isinstance(rationale, str) or len(rationale) < 12:
        raise MechanismDiscoveryError("MECHANISM_EXTRACTION_RATIONALE_INVALID")


def _load_profiles(path: Path) -> dict[str, Mapping[str, Sequence[str]]]:
    payload = _load_mapping(path, "MECHANISM_PROFILES_INVALID")
    profiles = payload.get("profiles")
    if not isinstance(profiles, Mapping) or not profiles:
        raise MechanismDiscoveryError("MECHANISM_PROFILES_INVALID")
    result: dict[str, Mapping[str, Sequence[str]]] = {}
    for name, profile in profiles.items():
        if not isinstance(name, str) or not isinstance(profile, Mapping) or set(profile) != set(_AXES):
            raise MechanismDiscoveryError("MECHANISM_PROFILES_INVALID")
        for values in profile.values():
            if not isinstance(values, list) or any(not isinstance(tag, str) or not _valid_tag(tag) for tag in values):
                raise MechanismDiscoveryError("MECHANISM_PROFILES_INVALID")
        result[name] = profile  # type: ignore[assignment]
    return result


def _load_ontology(path: Path) -> dict[str, Mapping[str, Sequence[str]]]:
    payload = _load_mapping(path, "MECHANISM_ONTOLOGY_INVALID")
    if payload.get("artifact_type") != "wmloop-mechanism-tag-ontology":
        raise MechanismDiscoveryError("MECHANISM_ONTOLOGY_INVALID")
    axes = payload.get("axes")
    if not isinstance(axes, Mapping) or set(axes) != set(_AXES):
        raise MechanismDiscoveryError("MECHANISM_ONTOLOGY_INVALID")
    result: dict[str, Mapping[str, Sequence[str]]] = {}
    for axis, concepts in axes.items():
        if not isinstance(concepts, Mapping) or not concepts:
            raise MechanismDiscoveryError("MECHANISM_ONTOLOGY_INVALID")
        seen: set[str] = set()
        typed: dict[str, Sequence[str]] = {}
        for concept, aliases in concepts.items():
            if (
                not isinstance(concept, str)
                or not _valid_tag(concept)
                or not isinstance(aliases, list)
                or not aliases
                or any(not isinstance(alias, str) or not _valid_tag(alias) for alias in aliases)
            ):
                raise MechanismDiscoveryError("MECHANISM_ONTOLOGY_INVALID")
            duplicate = seen & set(aliases)
            if duplicate:
                raise MechanismDiscoveryError("MECHANISM_ONTOLOGY_ALIAS_DUPLICATE")
            seen.update(aliases)
            typed[concept] = tuple(aliases)
        result[str(axis)] = typed
    return result


def _load_reference_atlas_profiles(
    path: Path,
) -> dict[str, Mapping[str, Sequence[str]]]:
    payload = _load_mapping(path, "MECHANISM_REFERENCE_ATLAS_INVALID")
    entries = payload.get("entries")
    if (
        payload.get("artifact_type") != "wmloop-mechanism-discovery-atlas"
        or payload.get("state") != "ready"
        or not isinstance(entries, list)
    ):
        raise MechanismDiscoveryError("MECHANISM_REFERENCE_ATLAS_INVALID")
    profiles: dict[str, Mapping[str, Sequence[str]]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise MechanismDiscoveryError("MECHANISM_REFERENCE_ATLAS_INVALID")
        mechanism = entry.get("mechanism")
        source = entry.get("source")
        if not isinstance(mechanism, Mapping) or not isinstance(source, Mapping):
            raise MechanismDiscoveryError("MECHANISM_REFERENCE_ATLAS_INVALID")
        if mechanism.get("extraction_state") != "evidence_supported":
            continue
        axes = _validated_profile_axes(
            mechanism.get("axes"),
            code="MECHANISM_REFERENCE_ATLAS_INVALID",
        )
        arxiv_id = str(source.get("arxiv_id") or "")
        if not arxiv_id:
            raise MechanismDiscoveryError("MECHANISM_REFERENCE_ATLAS_INVALID")
        profiles[f"literature:{arxiv_id}"] = axes
    return profiles


def _load_reference_profiles(
    path: Path,
) -> tuple[dict[str, Mapping[str, Sequence[str]]], dict[str, str]]:
    payload = _load_mapping(path, "MECHANISM_REFERENCE_PROFILES_INVALID")
    rows = payload.get("profiles")
    if (
        payload.get("artifact_type") != "wmloop-mechanism-reference-profiles"
        or not isinstance(rows, Mapping)
        or not rows
    ):
        raise MechanismDiscoveryError("MECHANISM_REFERENCE_PROFILES_INVALID")
    profiles: dict[str, Mapping[str, Sequence[str]]] = {}
    kinds: dict[str, str] = {}
    for name, row in rows.items():
        if not isinstance(name, str) or not isinstance(row, Mapping):
            raise MechanismDiscoveryError("MECHANISM_REFERENCE_PROFILES_INVALID")
        kind = row.get("reference_kind")
        if not isinstance(kind, str) or not kind:
            raise MechanismDiscoveryError("MECHANISM_REFERENCE_PROFILES_INVALID")
        reference_id = f"settled:{name}"
        profiles[reference_id] = _validated_profile_axes(
            row.get("axes"),
            code="MECHANISM_REFERENCE_PROFILES_INVALID",
        )
        kinds[reference_id] = kind
    return profiles, kinds


def _validated_profile_axes(
    value: object,
    *,
    code: str,
) -> Mapping[str, Sequence[str]]:
    if not isinstance(value, Mapping) or set(value) != set(_AXES):
        raise MechanismDiscoveryError(code)
    result: dict[str, Sequence[str]] = {}
    for axis, tags in value.items():
        if (
            not isinstance(tags, list)
            or any(not isinstance(tag, str) or not _valid_tag(tag) for tag in tags)
        ):
            raise MechanismDiscoveryError(code)
        result[str(axis)] = tuple(tags)
    return result


def _merge_profiles(
    destination: dict[str, Mapping[str, Sequence[str]]],
    incoming: Mapping[str, Mapping[str, Sequence[str]]],
    *,
    source: Path,
) -> None:
    duplicate = set(destination) & set(incoming)
    if duplicate:
        names = ",".join(sorted(duplicate))
        raise MechanismDiscoveryError(f"MECHANISM_REFERENCE_DUPLICATE:{source}:{names}")
    destination.update(incoming)


def _request_dict(request: DiscoveryRequest) -> dict[str, object]:
    return {
        "symptom_description": request.symptom_description,
        "failure_signatures": list(request.failure_signatures),
        "target_metrics": list(request.target_metrics),
        "protected_metrics": list(request.protected_metrics),
        "available_hooks": list(request.available_hooks),
        "model_family": request.model_family,
        "seed_arxiv_ids": list(request.seed_arxiv_ids),
        "cross_domain_lenses": list(request.cross_domain_lenses),
    }


def _query_terms(value: str, *, maximum: int) -> tuple[str, ...]:
    stop = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "in", "is", "it", "of", "on", "or", "that", "the", "this", "to",
        "with", "model", "failure", "metric", "hook",
    }
    values = [token for token in re.findall(r"[a-z0-9]+", value.lower()) if len(token) > 1 and token not in stop]
    return tuple(dict.fromkeys(values))[:maximum]


def _join_query(*parts: Sequence[str]) -> str:
    return " ".join(dict.fromkeys(token for part in parts for token in part))


def _tag_set(value: object) -> set[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return set()
    return {str(tag) for tag in value if isinstance(tag, str)}


def _valid_tag(value: str) -> bool:
    return re.fullmatch(r"[a-z][a-z0-9_]{1,63}", value) is not None


def _bounded_excerpt(full_text: str, abstract: str) -> str:
    source = " ".join((full_text or abstract).split())
    return source[:1000]


def _canonical_arxiv_id(value: str) -> str:
    match = _ARXIV_ID.search(value)
    if match is None:
        raise MechanismDiscoveryError("MECHANISM_ARXIV_ID_INVALID")
    version = f"v{match.group(2)}" if match.group(2) else ""
    return f"{match.group(1)}{version}"


def _base_arxiv_id(value: str) -> str:
    return _canonical_arxiv_id(value).split("v", 1)[0]


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")[:90] or "unknown"


def _normalise(value: str) -> str:
    return " ".join(html.unescape(value).lower().split())


def _node_text(node: ElementTree.Element | None) -> str:
    return (node.text or "").strip() if node is not None else ""


def _load_mapping(path: Path, code: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).resolve(strict=True).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MechanismDiscoveryError(code) from exc
    if not isinstance(value, dict):
        raise MechanismDiscoveryError(code)
    return value


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.write_bytes(_canonical_json(value))


def recompare_mechanism_atlas(
    *,
    input_atlas: Path,
    output_root: Path,
    repo_root: Path,
    reference_atlas_paths: Sequence[Path] = (),
    reference_profiles_paths: Sequence[Path] = (),
) -> dict[str, object]:
    """Rebuild comparison labels without refetching or changing paper evidence."""

    root = Path(repo_root).resolve(strict=True)
    source_path = Path(input_atlas).resolve(strict=True)
    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise MechanismDiscoveryError("MECHANISM_OUTPUT_EXISTS")
    source = _load_mapping(source_path, "MECHANISM_REFERENCE_ATLAS_INVALID")
    entries = source.get("entries")
    if (
        source.get("artifact_type") != "wmloop-mechanism-discovery-atlas"
        or source.get("state") != "ready"
        or not isinstance(entries, list)
    ):
        raise MechanismDiscoveryError("MECHANISM_REFERENCE_ATLAS_INVALID")

    profiles_path = root / "configs" / "retrieval" / "primitive_mechanism_profiles_v1.json"
    ontology_path = root / "configs" / "retrieval" / "mechanism_tag_ontology_v1.json"
    profiles = _load_profiles(profiles_path)
    registry_names = set(PrimitiveRegistry.from_root(root).names())
    if set(profiles) != registry_names:
        raise MechanismDiscoveryError("MECHANISM_PROFILE_COVERAGE_MISMATCH")
    ontology = _load_ontology(ontology_path)
    reference_kinds = {name: "registered_primitive" for name in profiles}
    reference_sources: list[dict[str, str]] = []
    for raw_path in reference_atlas_paths:
        path = Path(raw_path).resolve(strict=True)
        loaded = _load_reference_atlas_profiles(path)
        _merge_profiles(profiles, loaded, source=path)
        reference_kinds.update({name: "evidence_supported_literature" for name in loaded})
        reference_sources.append({"kind": "atlas", "path": str(path), "sha256": _sha256(path.read_bytes())})
    for raw_path in reference_profiles_paths:
        path = Path(raw_path).resolve(strict=True)
        loaded, kinds = _load_reference_profiles(path)
        _merge_profiles(profiles, loaded, source=path)
        reference_kinds.update(kinds)
        reference_sources.append(
            {"kind": "reference_profiles", "path": str(path), "sha256": _sha256(path.read_bytes())}
        )

    peer_profiles: dict[str, Mapping[str, Sequence[str]]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise MechanismDiscoveryError("MECHANISM_REFERENCE_ATLAS_INVALID")
        mechanism = entry.get("mechanism")
        source_row = entry.get("source")
        if not isinstance(mechanism, Mapping) or not isinstance(source_row, Mapping):
            raise MechanismDiscoveryError("MECHANISM_REFERENCE_ATLAS_INVALID")
        if mechanism.get("extraction_state") != "evidence_supported":
            continue
        arxiv_id = str(source_row.get("arxiv_id") or "")
        peer_profiles[f"candidate_pool:{arxiv_id}"] = _validated_profile_axes(
            mechanism.get("axes"),
            code="MECHANISM_REFERENCE_ATLAS_INVALID",
        )

    rebuilt_entries: list[dict[str, object]] = []
    for entry in entries:
        mechanism = entry["mechanism"]
        source_row = entry["source"]
        arxiv_id = str(source_row["arxiv_id"])
        comparison_profiles = dict(profiles)
        comparison_kinds = dict(reference_kinds)
        for reference_id, peer_profile in peer_profiles.items():
            if reference_id == f"candidate_pool:{arxiv_id}":
                continue
            comparison_profiles[reference_id] = peer_profile
            comparison_kinds[reference_id] = "evidence_supported_candidate_pool"
        comparison = compare_mechanism_signature(
            mechanism,
            comparison_profiles,
            ontology=ontology,
            reference_kinds=comparison_kinds,
        )
        rebuilt = {**entry, "comparison": comparison}
        try:
            validate_document("mechanism_atlas_entry", rebuilt, root=root)
        except ContractValidationError as exc:
            raise MechanismDiscoveryError(f"MECHANISM_ATLAS_SCHEMA_INVALID:{exc}") from exc
        rebuilt_entries.append(rebuilt)

    input_payload = {
        "input_atlas_sha256": _sha256(source_path.read_bytes()),
        "ontology_sha256": _sha256(ontology_path.read_bytes()),
        "profiles_sha256": _sha256(profiles_path.read_bytes()),
        "reference_sources": reference_sources,
    }
    report = {
        **source,
        "input_hash": _sha256(_canonical_json(input_payload)),
        "entries": rebuilt_entries,
        "novelty_resolved_count": sum(
            entry["comparison"]["novelty_state"] != "unresolved"
            for entry in rebuilt_entries
        ),
        "comparison_ontology": str(ontology_path),
        "comparison_ontology_sha256": _sha256(ontology_path.read_bytes()),
        "reference_profile_count": len(profiles),
        "candidate_pool_reference_count": len(peer_profiles),
        "reference_sources": reference_sources,
        "supersedes_atlas": str(source_path),
        "supersedes_atlas_sha256": _sha256(source_path.read_bytes()),
        "recomparison_only": True,
    }
    destination.mkdir(mode=0o700, parents=True)
    atlas_path = destination / "mechanism-atlas.json"
    _write_json(atlas_path, report)

    source_missing_path = source_path.parent / "missing-sources.json"
    if source_missing_path.is_file() and not source_missing_path.is_symlink():
        missing_report = _load_mapping(source_missing_path, "MECHANISM_MISSING_SOURCES_INVALID")
    else:
        missing_report = {
            "schema_version": 1,
            "artifact_type": "wmloop-mechanism-missing-sources",
            "state": "complete",
            "local_sources_root": report.get("local_sources_root"),
            "missing_source_count": int(report.get("missing_source_count", 0)),
            "records": [],
            "instructions": "No source refetch was performed during comparison-only rebuild.",
        }
    _write_json(destination / "missing-sources.json", missing_report)
    (destination / "MISSING_SOURCES.md").write_text(
        _render_missing_sources_markdown(missing_report),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "artifact_type": "wmloop-mechanism-discovery-manifest",
        "state": "ready",
        "input_hash": report["input_hash"],
        "atlas_path": str(atlas_path),
        "atlas_sha256": _sha256(atlas_path.read_bytes()),
        "paper_count": len(rebuilt_entries),
        "missing_source_count": int(missing_report.get("missing_source_count", 0)),
        "missing_sources_path": str(destination / "missing-sources.json"),
        "missing_sources_sha256": _sha256((destination / "missing-sources.json").read_bytes()),
        "missing_sources_markdown_path": str(destination / "MISSING_SOURCES.md"),
        "execution_authority": "shadow_only",
        "claim_boundary": report["claim_boundary"],
        "recomparison_only": True,
    }
    _write_json(destination / "manifest.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    """CLI for a bounded mechanism-discovery transaction."""

    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--seed-records", type=Path, action="append", default=[])
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--citation-depth", type=int, default=1)
    parser.add_argument("--max-papers", type=int, default=12)
    parser.add_argument("--search-results-per-view", type=int, default=3)
    parser.add_argument("--complexity-budget", choices=("light", "full"), default="light")
    parser.add_argument("--timeout-seconds", type=float, default=20.0)
    parser.add_argument("--local-sources-root", type=Path)
    parser.add_argument("--reference-atlas", type=Path, action="append", default=[])
    parser.add_argument("--reference-profiles", type=Path, action="append", default=[])
    args = parser.parse_args(argv)
    try:
        request_payload = _load_mapping(args.request, "MECHANISM_REQUEST_INVALID")
        request = _request_from_mapping(request_payload)
        seeds: list[Mapping[str, Any]] = []
        for path in args.seed_records:
            payload = _load_mapping(path, "MECHANISM_SEEDS_INVALID")
            records = payload.get("records")
            if not isinstance(records, list) or any(not isinstance(row, Mapping) for row in records):
                raise MechanismDiscoveryError("MECHANISM_SEEDS_INVALID")
            seeds.extend(records)
        manifest = run_mechanism_discovery(
            request=request,
            seed_records=seeds,
            output_root=args.output_root,
            repo_root=args.repo_root,
            annotations_path=args.annotations,
            citation_depth=args.citation_depth,
            max_papers=args.max_papers,
            timeout_seconds=args.timeout_seconds,
            search_results_per_view=args.search_results_per_view,
            complexity_budget=args.complexity_budget,
            local_sources_root=args.local_sources_root,
            reference_atlas_paths=args.reference_atlas,
            reference_profiles_paths=args.reference_profiles,
        )
    except MechanismDiscoveryError as exc:
        print(str(exc), file=__import__("sys").stderr)
        return 2
    print(json.dumps(manifest, ensure_ascii=True, sort_keys=True))
    return 0


def _request_from_mapping(payload: Mapping[str, Any]) -> DiscoveryRequest:
    try:
        request = DiscoveryRequest(
            symptom_description=_nonempty_string(payload, "symptom_description"),
            failure_signatures=_string_tuple(payload, "failure_signatures", minimum=1),
            target_metrics=_string_tuple(payload, "target_metrics", minimum=1),
            protected_metrics=_string_tuple(payload, "protected_metrics", minimum=1),
            available_hooks=_string_tuple(payload, "available_hooks", minimum=1),
            model_family=_nonempty_string(payload, "model_family"),
            seed_arxiv_ids=_string_tuple(payload, "seed_arxiv_ids", minimum=0),
            cross_domain_lenses=_string_tuple(payload, "cross_domain_lenses", minimum=0),
        )
        for arxiv_id in request.seed_arxiv_ids:
            _canonical_arxiv_id(arxiv_id)
        return request
    except (KeyError, TypeError, ValueError) as exc:
        raise MechanismDiscoveryError("MECHANISM_REQUEST_INVALID") from exc


def _nonempty_string(payload: Mapping[str, Any], field: str) -> str:
    value = payload[field]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(field)
    return value.strip()


def _string_tuple(
    payload: Mapping[str, Any],
    field: str,
    *,
    minimum: int,
) -> tuple[str, ...]:
    value = payload.get(field, [])
    if not isinstance(value, list) or len(value) < minimum:
        raise ValueError(field)
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(field)
    return tuple(dict.fromkeys(item.strip() for item in value))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
