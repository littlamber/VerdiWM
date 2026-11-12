"""Evidence-bound mechanism discovery beyond method-name matching.

Keyword search is useful for collecting seed papers, but a missing method name
is not novelty evidence.  This module expands seed papers through their cited
arXiv neighbourhood, captures bounded full-text evidence, and compares typed
mechanism signatures against typed registry profiles.  It never grants
experiment or source-mutation authority.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from wmloop.contracts import ContractValidationError, validate_document
from wmloop.primitives.registry import PrimitiveRegistry
from wmloop.retrieve.discovery.common import (
    _base_arxiv_id as _base_arxiv_id,
    _bounded_excerpt as _bounded_excerpt,
    _canonical_arxiv_id as _canonical_arxiv_id,
    _canonical_json as _canonical_json,
    _join_query as _join_query,
    _load_mapping as _load_mapping,
    _node_text as _node_text,
    _normalise as _normalise,
    _query_terms as _query_terms,
    _request_dict as _request_dict,
    _safe_id as _safe_id,
    _sha256 as _sha256,
    _tag_set as _tag_set,
    _valid_tag as _valid_tag,
    _write_json as _write_json,
)
from wmloop.retrieve.discovery.comparison import (
    _canonical_concepts as _canonical_concepts,
    compare_mechanism_signature as compare_mechanism_signature,
)
from wmloop.retrieve.discovery.extraction import (
    AnnotationMechanismExtractor as AnnotationMechanismExtractor,
    EvidenceOnlyMechanismExtractor as EvidenceOnlyMechanismExtractor,
    _atlas_entry as _atlas_entry,
    _validate_extraction as _validate_extraction,
    extractor_identity,
)
from wmloop.retrieve.discovery.profiles import (
    _load_ontology as _load_ontology,
    _load_profiles as _load_profiles,
    _load_reference_atlas_profiles as _load_reference_atlas_profiles,
    _load_reference_profiles as _load_reference_profiles,
    _merge_profiles as _merge_profiles,
    _validated_profile_axes as _validated_profile_axes,
)
from wmloop.retrieve.discovery.queries import (
    _collect_query_seeds as _collect_query_seeds,
    _complexity_budget as _complexity_budget,
    _deduplicate_seed_records as _deduplicate_seed_records,
    build_multiview_queries as build_multiview_queries,
)
from wmloop.retrieve.discovery.requests import (
    _nonempty_string as _nonempty_string,
    _request_from_mapping as _request_from_mapping,
    _string_tuple as _string_tuple,
)
from wmloop.retrieve.discovery.sources import (
    _arxiv_ids_from_text as _arxiv_ids_from_text,
    _BoundedHTMLTextParser as _BoundedHTMLTextParser,
    _expand_papers as _expand_papers,
    _fetch_ar5iv_text as _fetch_ar5iv_text,
    _fetch_arxiv_metadata as _fetch_arxiv_metadata,
    _load_discovery_paper as _load_discovery_paper,
    _load_local_source as _load_local_source,
    _local_source_inventory as _local_source_inventory,
    _merge_missing_source as _merge_missing_source,
    _missing_source_record as _missing_source_record,
    _render_missing_sources_markdown as _render_missing_sources_markdown,
)
from wmloop.retrieve.discovery.types import (
    _ARXIV_ID as _ARXIV_ID,
    _AXES as _AXES,
    _COMPLEXITY_BUDGETS as _COMPLEXITY_BUDGETS,
    DiscoveryPaper as DiscoveryPaper,
    DiscoveryRequest as DiscoveryRequest,
    MechanismDiscoveryError as MechanismDiscoveryError,
    MechanismExtractionClient as MechanismExtractionClient,
)


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
    if search_results_per_view < 0 or search_results_per_view > int(
        budget["max_results_per_view"]
    ):
        raise MechanismDiscoveryError("MECHANISM_SEARCH_RESULTS_INVALID")
    root = Path(repo_root).resolve(strict=True)
    destination = Path(output_root).resolve()
    local_root = (
        Path(local_sources_root).resolve() if local_sources_root is not None else None
    )
    if local_root is not None:
        local_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if local_root.is_symlink() or not local_root.is_dir():
            raise MechanismDiscoveryError("MECHANISM_LOCAL_SOURCES_INVALID")
    if destination.exists() or destination.is_symlink():
        raise MechanismDiscoveryError("MECHANISM_OUTPUT_EXISTS")
    profiles_path = (
        root / "configs" / "retrieval" / "primitive_mechanism_profiles_v1.json"
    )
    ontology_path = root / "configs" / "retrieval" / "mechanism_tag_ontology_v1.json"
    profiles = _load_profiles(profiles_path)
    registry_names = set(PrimitiveRegistry.from_root(root).names())
    if set(profiles) != registry_names:
        missing = ",".join(sorted(registry_names - set(profiles)))
        extra = ",".join(sorted(set(profiles) - registry_names))
        raise MechanismDiscoveryError(
            f"MECHANISM_PROFILE_COVERAGE_MISMATCH:{missing}:{extra}"
        )
    ontology = _load_ontology(ontology_path)
    reference_kinds = {name: "registered_primitive" for name in profiles}
    reference_sources: list[dict[str, str]] = []
    for raw_path in reference_atlas_paths:
        path = Path(raw_path).resolve(strict=True)
        loaded = _load_reference_atlas_profiles(path)
        _merge_profiles(profiles, loaded, source=path)
        reference_kinds.update(
            {name: "evidence_supported_literature" for name in loaded}
        )
        reference_sources.append(
            {"kind": "atlas", "path": str(path), "sha256": _sha256(path.read_bytes())}
        )
    for raw_path in reference_profiles_paths:
        path = Path(raw_path).resolve(strict=True)
        loaded, kinds = _load_reference_profiles(path)
        _merge_profiles(profiles, loaded, source=path)
        reference_kinds.update(kinds)
        reference_sources.append(
            {
                "kind": "reference_profiles",
                "path": str(path),
                "sha256": _sha256(path.read_bytes()),
            }
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
        "extraction_client": extractor_identity(extractor),
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
            extraction = dict(
                EvidenceOnlyMechanismExtractor().extract(paper, request=request)
            )
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
            raise MechanismDiscoveryError(
                f"MECHANISM_ATLAS_SCHEMA_INVALID:{exc}"
            ) from exc
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
            entry["comparison"]["novelty_state"] != "unresolved"
            for entry in entries  # type: ignore[index]
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
        "missing_sources_sha256": _sha256(
            (destination / "missing-sources.json").read_bytes()
        ),
        "missing_sources_markdown_path": str(destination / "MISSING_SOURCES.md"),
        "execution_authority": "shadow_only",
        "claim_boundary": report["claim_boundary"],
    }
    _write_json(destination / "manifest.json", manifest)
    return manifest


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

    profiles_path = (
        root / "configs" / "retrieval" / "primitive_mechanism_profiles_v1.json"
    )
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
        reference_kinds.update(
            {name: "evidence_supported_literature" for name in loaded}
        )
        reference_sources.append(
            {"kind": "atlas", "path": str(path), "sha256": _sha256(path.read_bytes())}
        )
    for raw_path in reference_profiles_paths:
        path = Path(raw_path).resolve(strict=True)
        loaded, kinds = _load_reference_profiles(path)
        _merge_profiles(profiles, loaded, source=path)
        reference_kinds.update(kinds)
        reference_sources.append(
            {
                "kind": "reference_profiles",
                "path": str(path),
                "sha256": _sha256(path.read_bytes()),
            }
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
            raise MechanismDiscoveryError(
                f"MECHANISM_ATLAS_SCHEMA_INVALID:{exc}"
            ) from exc
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
        missing_report = _load_mapping(
            source_missing_path, "MECHANISM_MISSING_SOURCES_INVALID"
        )
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
        "missing_sources_sha256": _sha256(
            (destination / "missing-sources.json").read_bytes()
        ),
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
    parser.add_argument(
        "--complexity-budget", choices=("light", "full"), default="light"
    )
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
            if not isinstance(records, list) or any(
                not isinstance(row, Mapping) for row in records
            ):
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


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
