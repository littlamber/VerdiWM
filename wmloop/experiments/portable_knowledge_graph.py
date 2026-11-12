"""Build a path-free, shareable transfer-knowledge projection.

Local Evidence Graph artifacts preserve receipt locations for audit. This module
accepts only validated semantic records and therefore produces a graph that can
move between checkouts without retaining a runtime path or source filename.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

from wmloop.contracts import ContractValidationError, validate_document
from wmloop.experiments.knowledge.common import (
    _canonical as _canonical,
    _digest as _digest,
    _is_cas_sha256 as _is_cas_sha256,
    _node_id as _node_id,
    _reject_path_bound_value as _reject_path_bound_value,
    _unique_documents as _unique_documents,
)
from wmloop.experiments.knowledge.contracts import (
    _project_evidence_ir as _project_evidence_ir,
    _project_knowledge_lifecycle as _project_knowledge_lifecycle,
    _project_module_composition as _project_module_composition,
    _project_protocol_contract as _project_protocol_contract,
    _project_transformation_contract as _project_transformation_contract,
)
from wmloop.experiments.knowledge.graph import (
    _PortableKnowledgeGraph as _PortableKnowledgeGraph,
)
from wmloop.experiments.knowledge.methods import (
    _project_method_interface_extension as _project_method_interface_extension,
    _project_method_ir as _project_method_ir,
    _project_model_capability as _project_model_capability,
)
from wmloop.experiments.knowledge.models import (
    _project_model_irg as _project_model_irg,
    _project_model_portrait as _project_model_portrait,
    _project_portrait_transition as _project_portrait_transition,
)
from wmloop.experiments.knowledge.projection import (
    _project_document as _project_document,
)
from wmloop.experiments.knowledge.query import (
    _canonical_scalar as _canonical_scalar,
    _load_graph as _load_graph,
    _normalize_similarity as _normalize_similarity,
    _semantic_similarity as _semantic_similarity,
)
from wmloop.experiments.knowledge.transfer import (
    _project_embodiment as _project_embodiment,
    _project_fingerprint as _project_fingerprint,
    _project_mechanism as _project_mechanism,
    _project_mechanism_relation as _project_mechanism_relation,
    _project_portable_experience as _project_portable_experience,
    _project_transfer_boundary as _project_transfer_boundary,
)
from wmloop.experiments.knowledge.types import (
    _CAS_SHA256 as _CAS_SHA256,
    _SEMANTIC_COUNT_KEY as _SEMANTIC_COUNT_KEY,
    _SIMILARITY_FIELDS as _SIMILARITY_FIELDS,
    PortableKnowledgeGraphError as PortableKnowledgeGraphError,
)
from wmloop.experiments.knowledge.validation import (
    _composition_module_node as _composition_module_node,
    _validate_community_document as _validate_community_document,
    _validate_module_composition_receipt as _validate_module_composition_receipt,
)


def build_portable_knowledge_graph(
    documents: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Project path-free validated records into a deterministic graph."""

    graph = _PortableKnowledgeGraph()
    for document in _unique_documents(documents):
        _project_document(graph, document)
    return graph.document()


def write_portable_knowledge_graph(
    *,
    documents: Sequence[Mapping[str, object]],
    output_root: Path,
) -> dict[str, object]:
    """Atomically write a portable graph without recording the output location."""

    destination = Path(output_root).expanduser().resolve()
    destination.mkdir(mode=0o700, parents=True, exist_ok=True)
    report = build_portable_knowledge_graph(documents)
    audit = audit_portable_knowledge_graph(documents=documents, graph=report)
    temporary = Path(tempfile.mkdtemp(prefix=".portable-knowledge-", dir=destination))
    try:
        (temporary / "graph.json").write_bytes(_canonical(report) + b"\n")
        (temporary / "quality-audit.json").write_bytes(_canonical(audit) + b"\n")
        manifest = {
            name: report[name]
            for name in (
                "schema_version",
                "artifact_type",
                "state",
                "document_count",
                "node_count",
                "edge_count",
                "claim_boundary",
            )
        }
        manifest["graph_digest"] = audit["graph_digest"]
        manifest["quality_audit_id"] = audit["audit_id"]
        (temporary / "manifest.json").write_bytes(_canonical(manifest) + b"\n")
        for name in ("graph.json", "manifest.json", "quality-audit.json"):
            os.replace(temporary / name, destination / name)
    finally:
        try:
            temporary.rmdir()
        except OSError:
            pass
    return {
        **report,
        "quality_audit_id": audit["audit_id"],
        "quality_audit_state": audit["state"],
    }


def stage_portable_knowledge_graph(
    *, documents: Sequence[Mapping[str, object]], output_root: Path
) -> dict[str, object]:
    """One-way semantic staging alias used by community export workflows."""

    return write_portable_knowledge_graph(documents=documents, output_root=output_root)


def stage_portable_knowledge_records(
    *, documents: Sequence[Mapping[str, object]], output_root: Path
) -> dict[str, object]:
    """Idempotently stage validated semantic records without source locations."""

    unique_documents = _unique_documents(documents)
    graph = build_portable_knowledge_graph(unique_documents)
    audit = audit_portable_knowledge_graph(documents=unique_documents, graph=graph)
    raw = Path(output_root).expanduser()
    if raw.is_symlink():
        raise PortableKnowledgeGraphError("PORTABLE_KNOWLEDGE_RECORD_OUTPUT_INVALID")
    destination = raw.resolve()
    destination.mkdir(mode=0o700, parents=True, exist_ok=True)
    record_ids = []
    for document in unique_documents:
        record_id = "portable-record-" + _digest(document)
        path = destination / f"{record_id}.json"
        payload = _canonical(document) + b"\n"
        if path.exists():
            if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
                raise PortableKnowledgeGraphError(
                    "PORTABLE_KNOWLEDGE_RECORD_WRITE_CONFLICT"
                )
        else:
            temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
            try:
                with temporary.open("xb") as handle:
                    handle.write(payload)
                os.replace(temporary, path)
            except Exception:
                temporary.unlink(missing_ok=True)
                raise
        record_ids.append(record_id)
    return {
        "schema_version": 1,
        "artifact_type": "verdiwm-portable-knowledge-record-staging",
        "state": "ready",
        "record_count": len(record_ids),
        "record_ids": sorted(record_ids),
        "quality_audit_id": audit["audit_id"],
        "claim_boundary": (
            "Only validated semantic documents were copied. Source locations and local "
            "execution authority are not part of the staged records."
        ),
    }


def audit_portable_knowledge_graph(
    *,
    documents: Sequence[Mapping[str, object]],
    graph: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Audit path freedom, frozen promotions, licensing, and deterministic inputs."""

    unique_documents = _unique_documents(documents)
    payload = dict(graph or build_portable_knowledge_graph(unique_documents))
    _reject_path_bound_value(payload)
    document_digests = ["sha256:" + _digest(document) for document in unique_documents]
    promoted = [
        row
        for row in payload.get("edges", [])
        if isinstance(row, Mapping) and row.get("claim_scope") == "community"
    ]
    for row in promoted:
        evidence = row.get("evidence")
        if (
            row.get("frozen") is not True
            or not isinstance(evidence, list)
            or not any(_is_cas_sha256(value) for value in evidence)
        ):
            raise PortableKnowledgeGraphError(
                "PORTABLE_KNOWLEDGE_PROMOTION_EVIDENCE_INVALID"
            )
    licensed = 0
    for row in payload.get("nodes", []):
        if (
            not isinstance(row, Mapping)
            or row.get("redistributable_content") is not True
        ):
            continue
        if not isinstance(row.get("license_spdx_id"), str):
            raise PortableKnowledgeGraphError("PORTABLE_KNOWLEDGE_LICENSE_REQUIRED")
        licensed += 1
    body: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "verdiwm-portable-knowledge-quality-audit",
        "state": "ready",
        "graph_digest": "sha256:" + _digest(payload),
        "document_count": len(unique_documents),
        "document_digests": document_digests,
        "path_free": True,
        "semantic_documents_only": True,
        "promoted_relation_count": len(promoted),
        "promoted_relations_frozen_and_cas_bound": True,
        "redistributable_content_count": licensed,
        "licensed_content_only": True,
        "claim_boundary": (
            "The audit covers the one-way semantic projection only. Local ledgers, "
            "runtime files, private artifacts, and target-side verdict authority remain out of scope."
        ),
    }
    body["audit_id"] = "portable-knowledge-audit-" + _digest(body)[:24]
    try:
        validate_document("portable_knowledge_quality_audit", body)
    except ContractValidationError as exc:
        raise PortableKnowledgeGraphError(
            f"PORTABLE_KNOWLEDGE_QUALITY_AUDIT_INVALID:{exc}"
        ) from exc
    expected = build_portable_knowledge_graph(unique_documents)
    if _canonical(payload) != _canonical(expected):
        raise PortableKnowledgeGraphError("PORTABLE_KNOWLEDGE_GRAPH_DOCUMENT_MISMATCH")
    return body


def query_portable_knowledge_graph(
    graph: Mapping[str, object] | Path,
    *,
    entity: str,
    filters: Mapping[str, str] | None = None,
    similarity: Mapping[str, object] | None = None,
    minimum_similarity: float = 0.0,
    target_portrait_id: str | None = None,
) -> dict[str, object]:
    """Query exact fields and bounded semantic set similarity for ranking only."""

    payload = _load_graph(graph)
    if entity not in {"nodes", "edges"}:
        raise PortableKnowledgeGraphError("PORTABLE_KNOWLEDGE_GRAPH_ENTITY_INVALID")
    rows = payload.get(entity)
    if not isinstance(rows, list):
        raise PortableKnowledgeGraphError("PORTABLE_KNOWLEDGE_GRAPH_ROWS_INVALID")
    supplied = filters or {}
    normalized = {
        str(key): str(value)
        for key, value in supplied.items()
        if key not in {"limit", "offset"}
    }
    if (
        isinstance(minimum_similarity, bool)
        or not isinstance(minimum_similarity, (int, float))
        or not 0.0 <= float(minimum_similarity) <= 1.0
    ):
        raise PortableKnowledgeGraphError(
            "PORTABLE_KNOWLEDGE_GRAPH_SIMILARITY_THRESHOLD_INVALID"
        )
    normalized_similarity = _normalize_similarity(similarity or {})
    if target_portrait_id is not None and (
        not isinstance(target_portrait_id, str) or not target_portrait_id
    ):
        raise PortableKnowledgeGraphError(
            "PORTABLE_KNOWLEDGE_GRAPH_TARGET_PORTRAIT_INVALID"
        )
    try:
        offset = max(0, int(supplied.get("offset", "0")))
        limit = min(1000, max(1, int(supplied.get("limit", "100"))))
    except (TypeError, ValueError) as exc:
        raise PortableKnowledgeGraphError(
            "PORTABLE_KNOWLEDGE_GRAPH_PAGING_INVALID"
        ) from exc
    selected = []
    for row in rows:
        if not isinstance(row, Mapping) or not all(
            str(row.get(key)) == value for key, value in normalized.items()
        ):
            continue
        score = _semantic_similarity(row, normalized_similarity)
        if normalized_similarity and score < float(minimum_similarity):
            continue
        item = dict(row)
        if normalized_similarity:
            item["similarity"] = round(score, 6)
        if target_portrait_id is not None:
            source_portrait_id = row.get("portrait_id")
            item["usage_scope"] = (
                "target_context"
                if source_portrait_id == target_portrait_id
                else "prior_only"
            )
        selected.append(item)
    if normalized_similarity:
        selected.sort(
            key=lambda row: (-float(row["similarity"]), str(row.get("id", "")))
        )
    return {
        "schema_version": 1,
        "artifact_type": "verdiwm-portable-knowledge-graph-query",
        "entity": entity,
        "filters": normalized,
        "similarity": normalized_similarity,
        "minimum_similarity": float(minimum_similarity),
        "target_portrait_id": target_portrait_id,
        "total": len(selected),
        "offset": offset,
        "limit": limit,
        "items": selected[offset : offset + limit],
        "claim_boundary": (
            "Similarity is a bounded semantic ranking prior. Cross-portrait results "
            "never establish a target verdict and require target-side frozen verification."
        ),
    }
