"""Build a path-free, shareable transfer-knowledge projection.

Local Evidence Graph artifacts preserve receipt locations for audit. This module
accepts only validated semantic records and therefore produces a graph that can
move between checkouts without retaining a runtime path or source filename.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from wmloop.contracts import ContractValidationError, validate_document
from wmloop.control.intermediate_ir import (
    IntermediateRepresentationError,
    validate_model_capability_ir,
)
from wmloop.control.model_portrait import ModelPortraitError, validate_model_portrait
from wmloop.control.interface_extension import (
    MethodInterfaceExtensionError,
    validate_method_interface_extension,
)
from wmloop.control.open_method_ir import OpenMethodIRError, validate_method_ir
from wmloop.control.module_composition import module_composition_receipt_digest
from wmloop.geometry.community_knowledge import (
    CommunityKnowledgeError,
    validate_knowledge_lifecycle_record,
    validate_portrait_transition,
    validate_protocol_contract,
    validate_transformation_contract,
)
from wmloop.geometry.evidence_ir import validate_evidence_ir
from wmloop.geometry.portable_experience import validate_portable_experience
from wmloop.geometry.portable_transfer_knowledge import (
    validate_mechanism_contract,
    validate_method_embodiment,
    validate_probe_fingerprint_summary,
    validate_transfer_boundary,
)
from wmloop.geometry.types import GeometryValidationError


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
        raise PortableKnowledgeGraphError(
            "PORTABLE_KNOWLEDGE_RECORD_OUTPUT_INVALID"
        )
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
    document_digests = [
        "sha256:" + _digest(document) for document in unique_documents
    ]
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
        if not isinstance(row, Mapping) or row.get("redistributable_content") is not True:
            continue
        if not isinstance(row.get("license_spdx_id"), str):
            raise PortableKnowledgeGraphError(
                "PORTABLE_KNOWLEDGE_LICENSE_REQUIRED"
            )
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
        raise PortableKnowledgeGraphError(
            "PORTABLE_KNOWLEDGE_GRAPH_DOCUMENT_MISMATCH"
        )
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
    if isinstance(minimum_similarity, bool) or not isinstance(
        minimum_similarity, (int, float)
    ) or not 0.0 <= float(minimum_similarity) <= 1.0:
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
        raise PortableKnowledgeGraphError("PORTABLE_KNOWLEDGE_GRAPH_PAGING_INVALID") from exc
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


class _PortableKnowledgeGraph:
    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, object]] = {}
        self.edges: dict[str, dict[str, object]] = {}
        self.document_count = 0

    def node(self, kind: str, key: str, **attributes: object) -> str:
        identifier = _node_id(kind, key)
        row = self.nodes.setdefault(identifier, {"id": identifier, "kind": kind, "key": key})
        for name, value in attributes.items():
            if value is not None:
                if name in row and row[name] != value:
                    raise PortableKnowledgeGraphError(
                        "PORTABLE_KNOWLEDGE_NODE_CONFLICT:" + identifier + ":" + name
                    )
                row[name] = value
        return identifier

    def edge(
        self,
        source: str,
        relation: str,
        target: str,
        *,
        evidence: str | Sequence[str] | None = None,
        **attributes: object,
    ) -> None:
        key = f"{source}|{relation}|{target}"
        row = self.edges.setdefault(
            key,
            {
                "id": _node_id("edge", key),
                "source": source,
                "relation": relation,
                "target": target,
            },
        )
        for name, value in attributes.items():
            if value is not None:
                if name in row and row[name] != value:
                    raise PortableKnowledgeGraphError(
                        "PORTABLE_KNOWLEDGE_EDGE_CONFLICT:" + str(row["id"]) + ":" + name
                    )
                row[name] = value
        if evidence is not None:
            values = [evidence] if isinstance(evidence, str) else list(evidence)
            row.setdefault("evidence", set()).update(str(value) for value in values)

    def document(self) -> dict[str, object]:
        nodes = [dict(value) for value in sorted(self.nodes.values(), key=lambda item: str(item["id"]))]
        edges = []
        for value in sorted(self.edges.values(), key=lambda item: str(item["id"])):
            row = dict(value)
            if isinstance(row.get("evidence"), set):
                row["evidence"] = sorted(row["evidence"])
            edges.append(row)
        return {
            "schema_version": 1,
            "artifact_type": "verdiwm-portable-knowledge-graph",
            "state": "ready",
            "document_count": self.document_count,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "node_kind_counts": dict(sorted(Counter(str(row["kind"]) for row in nodes).items())),
            "relation_counts": dict(sorted(Counter(str(row["relation"]) for row in edges).items())),
            "claim_boundary": (
                "This is a semantic, path-free projection. It ranks or describes "
                "transfer evidence but does not replace target-side frozen verification."
            ),
            "nodes": nodes,
            "edges": edges,
        }


def _project_document(graph: _PortableKnowledgeGraph, document: Mapping[str, object]) -> None:
    artifact = document.get("artifact_type")
    if artifact == "verdiwm-model-capability-ir":
        try:
            validate_model_capability_ir(document)
        except IntermediateRepresentationError as exc:
            raise PortableKnowledgeGraphError(
                f"PORTABLE_KNOWLEDGE_MODEL_CAPABILITY_INVALID:{exc}"
            ) from exc
        _project_model_capability(graph, document)
    elif artifact == "verdiwm-model-portrait":
        try:
            validate_model_portrait(document)
        except ModelPortraitError as exc:
            raise PortableKnowledgeGraphError(
                f"PORTABLE_KNOWLEDGE_MODEL_PORTRAIT_INVALID:{exc}"
            ) from exc
        _project_model_portrait(graph, document)
    elif artifact == "verdiwm-portrait-transition":
        _validate_community_document(validate_portrait_transition, document)
        _project_portrait_transition(graph, document)
    elif artifact == "verdiwm-module-composition-receipt":
        _validate_module_composition_receipt(document)
        _project_module_composition(graph, document)
    elif artifact == "verdiwm-evidence-ir":
        try:
            validate_evidence_ir(document)
        except GeometryValidationError as exc:
            raise PortableKnowledgeGraphError(
                f"PORTABLE_KNOWLEDGE_EVIDENCE_INVALID:{exc}"
            ) from exc
        _project_evidence_ir(graph, document)
    elif artifact == "verdiwm-protocol-contract":
        _validate_community_document(validate_protocol_contract, document)
        _project_protocol_contract(graph, document)
    elif artifact == "verdiwm-transformation-contract":
        _validate_community_document(validate_transformation_contract, document)
        _project_transformation_contract(graph, document)
    elif artifact == "verdiwm-knowledge-lifecycle":
        _validate_community_document(validate_knowledge_lifecycle_record, document)
        _project_knowledge_lifecycle(graph, document)
    elif artifact == "verdiwm-transferable-experience":
        validate_portable_experience(document)
        _project_portable_experience(graph, document)
    elif artifact == "verdiwm-mechanism-contract":
        validate_mechanism_contract(document)
        _project_mechanism(graph, document)
    elif artifact == "verdiwm-method-embodiment":
        validate_method_embodiment(document)
        _project_embodiment(graph, document)
    elif artifact == "verdiwm-probe-fingerprint-summary":
        validate_probe_fingerprint_summary(document)
        _project_fingerprint(graph, document)
    elif artifact == "verdiwm-transfer-boundary":
        validate_transfer_boundary(document)
        _project_transfer_boundary(graph, document)
    elif artifact == "verdiwm-method-ir":
        try:
            validate_method_ir(document)
        except OpenMethodIRError as exc:
            raise PortableKnowledgeGraphError(
                f"PORTABLE_KNOWLEDGE_METHOD_IR_INVALID:{exc}"
            ) from exc
        _project_method_ir(graph, document)
    elif artifact == "verdiwm-method-interface-extension":
        try:
            validate_method_interface_extension(document)
        except MethodInterfaceExtensionError as exc:
            raise PortableKnowledgeGraphError(
                f"PORTABLE_KNOWLEDGE_METHOD_INTERFACE_INVALID:{exc}"
            ) from exc
        _project_method_interface_extension(graph, document)
    else:
        raise PortableKnowledgeGraphError("PORTABLE_KNOWLEDGE_DOCUMENT_UNSUPPORTED")
    graph.document_count += 1


def _project_method_ir(
    graph: _PortableKnowledgeGraph, document: Mapping[str, object]
) -> None:
    mapping = document["target_mapping"]
    training = document["training"]
    falsification = document["falsification"]
    mechanism = document["mechanism"]
    root = graph.node(
        "open_method",
        str(document["method_id"]),
        method_id=document["method_id"],
        mechanism_summary=mechanism["summary"],
        transformations=mechanism["transformations"],
        mapping_state=mapping["mapping_state"],
        training_mode=training["mode"],
        training_objective=training["objective"],
        required_capabilities=mapping["required_capabilities"],
        primary_metrics=falsification["primary_metrics"],
        protected_metrics=falsification["protected_metrics"],
        source_evidence_digest=document.get("source_evidence_digest"),
        claim_scope="ranking_only",
    )
    for capability in mapping["required_capabilities"]:
        graph.edge(
            root,
            "requires_capability",
            graph.node("capability", str(capability)),
        )
    portrait = document.get("target_portrait_binding")
    if isinstance(portrait, Mapping):
        graph.edge(
            root,
            "targets_portrait",
            graph.node(
                "model_portrait",
                str(portrait["portrait_id"]),
                portrait_id=portrait["portrait_id"],
            ),
            evidence=str(portrait["portrait_digest"]),
            claim_scope="ranking_only",
            frozen=False,
        )
    probe = document.get("probe_binding")
    if isinstance(probe, Mapping):
        for fingerprint_id in probe["fingerprint_ids"]:
            graph.edge(
                root,
                "informed_by_fingerprint",
                graph.node("probe_fingerprint", str(fingerprint_id)),
                evidence=str(probe["binding_digest"]),
                claim_scope="ranking_only",
                frozen=False,
            )
    for source in document["source_evidence"]:
        source_node = graph.node(
            "source_evidence",
            str(source["source_digest"]),
            source_id=source["source_id"],
            source_digest=source["source_digest"],
            claim=source["claim"],
        )
        graph.edge(root, "grounded_in", source_node, evidence=str(source["source_digest"]))


def _project_method_interface_extension(
    graph: _PortableKnowledgeGraph, document: Mapping[str, object]
) -> None:
    root = graph.node(
        "method_interface_extension",
        str(document["extension_id"]),
        extension_id=document["extension_id"],
        requested_surface=document["requested_surface"],
        semantic_role=document["semantic_role"],
        typed_inputs=document["typed_inputs"],
        typed_outputs=document["typed_outputs"],
        side_effect_class=document["side_effect_class"],
        state=document["state"],
        claim_scope="ranking_only",
    )
    method = graph.node("open_method", str(document["method_id"]), method_id=document["method_id"])
    graph.edge(method, "proposes_interface", root, claim_scope="ranking_only", frozen=False)


def _project_model_capability(
    graph: _PortableKnowledgeGraph, document: Mapping[str, object]
) -> None:
    capabilities = [
        str(row["capability"])
        for row in document["capabilities"]
        if row["state"] == "available"
    ]
    hooks = [
        str(row["hook"])
        for row in document["hooks"]
        if row["state"] == "available"
    ]
    root = graph.node(
        "model_capability_profile",
        str(document["capability_id"]),
        model_capability_id=document["capability_id"],
        model_family=document["model_family"],
        available_capabilities=sorted(capabilities),
        available_hooks=sorted(hooks),
        source_revision=document["source_revision"],
    )
    graph.edge(
        root,
        "profiles_model_family",
        graph.node("model_family", str(document["model_family"])),
    )
    for row in document["capabilities"]:
        graph.edge(
            root,
            "has_capability",
            graph.node("capability", str(row["capability"])),
            state=row["state"],
        )
    for row in document["execution_interfaces"]:
        graph.edge(
            root,
            "has_execution_interface",
            graph.node("interface_contract", str(row["contract_id"])),
            state=row["state"],
        )
    for row in document["hooks"]:
        graph.edge(
            root,
            "has_hook",
            graph.node(
                "semantic_hook",
                str(row["hook"]),
                semantic_role=row["semantic_role"],
                binding_contract=row["binding_contract"],
            ),
            state=row["state"],
        )


def _project_model_portrait(
    graph: _PortableKnowledgeGraph, document: Mapping[str, object]
) -> None:
    coverage = document["coverage"]
    root = graph.node(
        "model_portrait",
        str(document["portrait_id"]),
        portrait_id=document["portrait_id"],
        model_capability_id=document["model_capability_id"],
        model_family=document["model_family"],
        available_capabilities=coverage["available_capabilities"],
        available_hooks=coverage["available_hooks"],
        observed_probe_keys=coverage["observed_probe_keys"],
        unknown_operational_metrics=coverage["unknown_operational_metrics"],
    )
    capability = graph.node(
        "model_capability_profile",
        str(document["model_capability_id"]),
        model_capability_id=document["model_capability_id"],
        model_family=document["model_family"],
    )
    graph.edge(root, "describes_capability_profile", capability)
    for capability_name in coverage["available_capabilities"]:
        graph.edge(
            root,
            "has_capability",
            graph.node("capability", str(capability_name)),
            state="available",
        )
    for hook in coverage["available_hooks"]:
        graph.edge(
            root,
            "has_hook",
            graph.node("semantic_hook", str(hook)),
            state="available",
        )
    for fingerprint in document["behavioral_fingerprints"]:
        child = graph.node(
            "probe_fingerprint",
            str(fingerprint["fingerprint_id"]),
            model_capability_id=fingerprint["model_capability_id"],
            diagnostic_role=fingerprint["diagnostic_role"],
            context_class=fingerprint["context_class"],
            split=fingerprint["split"],
            horizons=fingerprint["horizons"],
            dose_values=fingerprint["dose_values"],
            replication_count=fingerprint["replication_count"],
            response_digest=fingerprint["response_digest"],
        )
        graph.edge(root, "has_fingerprint", child, state=fingerprint["state"])
    parent = document.get("parent_portrait_id")
    transition = document.get("transition_ref")
    if isinstance(parent, str) and isinstance(transition, str):
        graph.edge(
            root,
            "derived_from",
            graph.node("model_portrait", parent, portrait_id=parent),
            evidence=transition,
            claim_scope="ranking_only",
            frozen=False,
        )
    for reference in document["evidence_refs"]:
        graph.edge(
            root,
            "cites_evidence",
            graph.node("evidence", str(reference)),
            evidence=str(reference),
        )


def _project_portrait_transition(
    graph: _PortableKnowledgeGraph, document: Mapping[str, object]
) -> None:
    state = str(document["outcome_state"])
    frozen = state != "admitted"
    root = graph.node(
        "portrait_transition",
        str(document["transition_id"]),
        outcome_state=state,
        claim_scope=document["claim_scope"],
    )
    parent = graph.node(
        "model_portrait",
        str(document["parent_portrait_id"]),
        portrait_id=document["parent_portrait_id"],
    )
    portrait = graph.node(
        "model_portrait",
        str(document["portrait_id"]),
        portrait_id=document["portrait_id"],
    )
    embodiment = graph.node("embodiment", str(document["embodiment_id"]))
    graph.edge(root, "updates_portrait", portrait)
    graph.edge(
        portrait,
        "derived_from",
        parent,
        evidence=document["transition_ref"],
        claim_scope="ranking_only",
        frozen=False,
    )
    graph.edge(
        portrait,
        "changed_by",
        embodiment,
        evidence=document["evidence_refs"],
        claim_scope=document["claim_scope"],
        frozen=frozen,
    )
    verdict = document.get("verdict_ref")
    if isinstance(verdict, str):
        verdict_node = graph.node("verdict", verdict)
        graph.edge(
            verdict_node,
            "establishes",
            root,
            evidence=document["evidence_refs"],
            claim_scope=document["claim_scope"],
            frozen=True,
        )


def _project_module_composition(
    graph: _PortableKnowledgeGraph, document: Mapping[str, object]
) -> None:
    root = graph.node(
        "module_composition",
        str(document["composition_id"]),
        requested_capabilities=document["requested_capabilities"],
        effective_authority_level=document["effective_authority_level"],
    )
    for row in document["selected_modules"]:
        module = graph.node(
            "module_manifest",
            f"{row['module_id']}:{row['abi_id']}",
            module_id=row["module_id"],
            abi_id=row["abi_id"],
            abi_version=row["abi_version"],
            implementation_ref="sha256:" + str(row["abi_digest"]),
            provides=row["provides"],
            requires=row["requires"],
            authority_level=row["authority_level"],
            side_effect_class=row["side_effect_class"],
            license_spdx_id=row["license_spdx_id"],
            redistributable_content=True,
        )
        graph.edge(root, "contains_module", module)
        for capability in row["provides"]:
            graph.edge(
                module,
                "provides_capability",
                graph.node("capability", str(capability)),
            )
        for capability in row["requires"]:
            graph.edge(
                module,
                "requires_capability",
                graph.node("capability", str(capability)),
            )
    for edge in document["dependency_edges"]:
        consumer = _composition_module_node(graph, document, str(edge["consumer_module_id"]))
        provider = _composition_module_node(graph, document, str(edge["provider_module_id"]))
        graph.edge(
            consumer,
            "requires_module",
            provider,
            capability=edge["required_capability"],
            contract=edge["contract"],
        )


def _project_evidence_ir(
    graph: _PortableKnowledgeGraph, document: Mapping[str, object]
) -> None:
    context = document["context"]
    status = document["status"]
    authority = document["authority"]
    state = str(status["state"])
    community = state == "community_promoted"
    if community and (
        authority["claim_scope"] != "community"
        or not any(_is_cas_sha256(value) for value in document["evidence_refs"])
    ):
        raise PortableKnowledgeGraphError(
            "PORTABLE_KNOWLEDGE_PROMOTED_EVIDENCE_INVALID"
        )
    root = graph.node(
        "evidence_record",
        str(document["evidence_id"]),
        model_family=context["model_family"],
        capabilities=[context["capability_class"]],
        goal_protocol=context["goal_protocol"],
        outcome_protocol=context["outcome_protocol"],
        data_regime=context["data_regime"],
        horizons=context["horizons"],
        outcome_label=document["outcome"]["label"],
        status=state,
        claim_scope=authority["claim_scope"],
    )
    primitive = graph.node(
        "primitive", str(document["intervention"]["primitive_id"])
    )
    graph.edge(
        root,
        "evaluates_intervention",
        primitive,
        evidence=document["evidence_refs"],
        claim_scope=authority["claim_scope"],
        frozen=community,
    )
    graph.edge(
        root,
        "observed_on_model_family",
        graph.node("model_family", str(context["model_family"])),
    )
    boundary_kind = (
        "negative_boundary"
        if document["outcome"]["label"] in {"harmful", "null"}
        else "applicability_boundary"
    )
    boundary = graph.node(
        boundary_kind,
        str(document["evidence_id"]),
        applicability=document["validity_region"],
        outcome_label=document["outcome"]["label"],
    )
    graph.edge(
        root,
        "establishes_boundary",
        boundary,
        evidence=document["evidence_refs"],
        claim_scope=authority["claim_scope"],
        frozen=community,
    )


def _project_protocol_contract(
    graph: _PortableKnowledgeGraph, document: Mapping[str, object]
) -> None:
    key = f"{document['protocol_kind']}:{document['protocol_id']}:{document['protocol_version']}"
    root = graph.node(
        "protocol_contract",
        key,
        contract_id=document["contract_id"],
        protocol_kind=document["protocol_kind"],
        protocol_id=document["protocol_id"],
        protocol_version=document["protocol_version"],
        semantic_dimensions=document["semantic_dimensions"],
        protected_fields=document["protected_fields"],
        contract_ref=document["contract_ref"],
        license_spdx_id=document["license_spdx_id"],
        redistributable_content=True,
    )
    for dimension in document["semantic_dimensions"]:
        graph.edge(
            root,
            "measures_semantic_dimension",
            graph.node("semantic_dimension", str(dimension)),
        )
    for reference in document["evidence_refs"]:
        graph.edge(
            root,
            "cites_evidence",
            graph.node("evidence", str(reference)),
            evidence=str(reference),
        )


def _project_transformation_contract(
    graph: _PortableKnowledgeGraph, document: Mapping[str, object]
) -> None:
    root = graph.node(
        "transformation_contract",
        str(document["transform_id"]),
        source_semantics=document["source_semantics"],
        target_semantics=document["target_semantics"],
        invariants=document["invariants"],
        loss_policy=document["loss_policy"],
        implementation_ref=document["implementation_ref"],
        verification_ref=document["verification_ref"],
        license_spdx_id=document["license_spdx_id"],
        redistributable_content=True,
    )
    for value in document["source_semantics"]:
        graph.edge(
            root,
            "maps_from",
            graph.node("data_semantics", str(value)),
        )
    for value in document["target_semantics"]:
        graph.edge(
            root,
            "maps_to",
            graph.node("data_semantics", str(value)),
        )
    for invariant in document["invariants"]:
        graph.edge(
            root,
            "preserves_invariant",
            graph.node("semantic_invariant", str(invariant)),
        )
    for reference in document["evidence_refs"]:
        graph.edge(
            root,
            "cites_evidence",
            graph.node("evidence", str(reference)),
            evidence=str(reference),
        )


def _project_knowledge_lifecycle(
    graph: _PortableKnowledgeGraph, document: Mapping[str, object]
) -> None:
    root = graph.node(
        "knowledge_lifecycle",
        str(document["lifecycle_id"]),
        action=document["action"],
        reason=document["reason"],
    )
    subject = document["subject"]
    subject_node = graph.node(str(subject["kind"]), str(subject["id"]))
    action = str(document["action"])
    relation = {"revocation": "revokes", "deprecation": "deprecates"}.get(action)
    if relation is not None:
        graph.edge(
            root,
            relation,
            subject_node,
            evidence=document["evidence_refs"],
            claim_scope="community",
            frozen=True,
        )
    else:
        replacement = document["replacement"]
        replacement_node = graph.node(
            str(replacement["kind"]), str(replacement["id"])
        )
        graph.edge(root, "records_supersession", replacement_node)
        graph.edge(
            replacement_node,
            "supersedes",
            subject_node,
            evidence=document["evidence_refs"],
            claim_scope="community",
            frozen=True,
        )


def _composition_module_node(
    graph: _PortableKnowledgeGraph,
    document: Mapping[str, object],
    module_id: str,
) -> str:
    rows = [row for row in document["selected_modules"] if row["module_id"] == module_id]
    if len(rows) != 1:
        raise PortableKnowledgeGraphError(
            "PORTABLE_KNOWLEDGE_COMPOSITION_MODULE_INVALID:" + module_id
        )
    row = rows[0]
    return graph.node("module_manifest", f"{row['module_id']}:{row['abi_id']}")


def _validate_module_composition_receipt(document: Mapping[str, object]) -> None:
    try:
        validate_document("module_composition_receipt", document)
    except ContractValidationError as exc:
        raise PortableKnowledgeGraphError(
            f"PORTABLE_KNOWLEDGE_COMPOSITION_INVALID:{exc}"
        ) from exc
    _reject_path_bound_value(document)
    if document.get("receipt_digest") != module_composition_receipt_digest(document):
        raise PortableKnowledgeGraphError(
            "PORTABLE_KNOWLEDGE_COMPOSITION_DIGEST_MISMATCH"
        )
    identity = {
        "registry_digest": document["registry_digest"],
        "requested_capabilities": document["requested_capabilities"],
        "eligible_abi_ids": document["eligible_abi_ids"],
        "maximum_authority_level": document["maximum_authority_level"],
        "selected_modules": [
            {
                "module_id": row["module_id"],
                "abi_id": row["abi_id"],
                "abi_version": row["abi_version"],
                "abi_digest": row["abi_digest"],
            }
            for row in document["selected_modules"]
        ],
        "capability_bindings": document["capability_bindings"],
        "dependency_edges": document["dependency_edges"],
        "external_bindings": document["external_bindings"],
    }
    expected = "composition-" + _digest(identity)[:24]
    if document.get("composition_id") != expected:
        raise PortableKnowledgeGraphError(
            "PORTABLE_KNOWLEDGE_COMPOSITION_ID_MISMATCH"
        )


def _validate_community_document(validator: Any, document: Mapping[str, object]) -> None:
    try:
        validator(document)
    except CommunityKnowledgeError as exc:
        raise PortableKnowledgeGraphError(
            f"PORTABLE_KNOWLEDGE_COMMUNITY_RECORD_INVALID:{exc}"
        ) from exc


def _project_portable_experience(graph: _PortableKnowledgeGraph, document: Mapping[str, object]) -> None:
    identity = str(document.get("portable_experience_id") or _digest(document))
    root = graph.node("portable_experience", identity)
    knowledge = document["portable_knowledge"]
    assert isinstance(knowledge, Mapping)
    for field, kind, relation in (
        ("model_family", "model_family", "observed_on_model_family"),
        ("capability_class", "capability_class", "requires_capability_class"),
        ("goal_protocol", "goal_protocol", "evaluated_under_goal"),
        ("outcome_protocol", "outcome_protocol", "measured_by_outcome"),
        ("dataset_regime", "dataset_regime", "observed_in_data_regime"),
        ("primitive", "primitive", "tests_primitive"),
    ):
        value = knowledge[field]
        child = graph.node(kind, str(value))
        graph.edge(root, relation, child)
    for anti_condition in document["anti_conditions"]:
        child = graph.node("anti_condition", str(anti_condition))
        graph.edge(root, "bounded_by", child)
    for reference in document["evidence_refs"]:
        child = graph.node("evidence", str(reference))
        graph.edge(root, "cites_evidence", child, evidence=str(reference))


def _project_mechanism(graph: _PortableKnowledgeGraph, document: Mapping[str, object]) -> None:
    root = graph.node(
        "mechanism",
        str(document["mechanism_id"]),
        causal_claim=document["causal_claim"],
        intervention_semantics=document["intervention_semantics"],
    )
    for capability in document["required_capabilities"]:
        graph.edge(root, "requires_capability", graph.node("capability", str(capability)))
    for capability in document["optional_capabilities"]:
        graph.edge(root, "optionally_uses_capability", graph.node("capability", str(capability)))
    for interface in document["target_interface_requirements"]:
        graph.edge(root, "requires_interface", graph.node("interface_contract", str(interface)))
    for condition in document["known_anti_conditions"]:
        graph.edge(root, "bounded_by", graph.node("anti_condition", str(condition)))
    for reference in document["source_evidence_refs"]:
        evidence = graph.node("evidence", str(reference))
        graph.edge(root, "derived_from_evidence", evidence, evidence=str(reference))


def _project_embodiment(graph: _PortableKnowledgeGraph, document: Mapping[str, object]) -> None:
    root = graph.node(
        "embodiment",
        str(document["embodiment_id"]),
        materialization_class=document["materialization_class"],
        implementation_state=document["implementation_state"],
        implementation_revision=document["implementation_revision"],
    )
    graph.edge(root, "embodies_mechanism", graph.node("mechanism", str(document["mechanism_id"])))
    for interface in document["interface_contracts"]:
        graph.edge(root, "uses_interface", graph.node("interface_contract", str(interface)))
    for reference in document["evidence_refs"]:
        evidence = graph.node("evidence", str(reference))
        graph.edge(root, "cites_evidence", evidence, evidence=str(reference))


def _project_fingerprint(graph: _PortableKnowledgeGraph, document: Mapping[str, object]) -> None:
    root = graph.node(
        "probe_fingerprint",
        str(document["fingerprint_id"]),
        model_capability_id=document["model_capability_id"],
        model_family=document["model_family"],
        diagnostic_role=document["diagnostic_role"],
        context_class=document["context_class"],
        split=document["split"],
        horizons=document["horizons"],
        dose_values=document["dose_values"],
        replication_count=document["replication_count"],
        response_digest=document["response_digest"],
        uncertainty_summary=document["uncertainty_summary"],
    )
    graph.edge(root, "profiles_model_capability", graph.node("model_capability", str(document["model_capability_id"])))
    graph.edge(root, "profiles_model_family", graph.node("model_family", str(document["model_family"])))
    protocol = graph.node(
        "protocol_contract",
        f"probe:{document['probe_protocol_id']}:{document['probe_protocol_version']}",
        protocol_kind="probe",
        protocol_id=document["probe_protocol_id"],
        protocol_version=document["probe_protocol_version"],
    )
    graph.edge(root, "uses_probe_protocol", protocol)
    graph.edge(root, "measured_by", protocol)
    response = graph.node("response_artifact", str(document["response_digest"]))
    graph.edge(root, "summarizes_response", response, evidence=str(document["response_digest"]))
    for reference in document["evidence_refs"]:
        evidence = graph.node("evidence", str(reference))
        graph.edge(root, "cites_evidence", evidence, evidence=str(reference))


def _project_transfer_boundary(graph: _PortableKnowledgeGraph, document: Mapping[str, object]) -> None:
    root = graph.node(
        "transfer_boundary",
        str(document["boundary_id"]),
        outcome_state=document["outcome_state"],
        claim_scope=document["claim_scope"],
        boundary_statement=document["boundary_statement"],
        capabilities=document["required_capabilities"],
        anti_conditions=document["anti_conditions"],
    )
    semantic_kind = (
        "negative_boundary"
        if document["outcome_state"] == "verified_negative_boundary"
        else "applicability_boundary"
    )
    semantic = graph.node(
        semantic_kind,
        str(document["boundary_id"]),
        outcome_state=document["outcome_state"],
        claim_scope=document["claim_scope"],
        capabilities=document["required_capabilities"],
        anti_conditions=document["anti_conditions"],
    )
    graph.edge(root, "projects_boundary_as", semantic)
    graph.edge(root, "constrains_mechanism", graph.node("mechanism", str(document["mechanism_id"])))
    graph.edge(root, "applies_to_embodiment", graph.node("embodiment", str(document["embodiment_id"])))
    graph.edge(root, "compares_source_fingerprint", graph.node("probe_fingerprint", str(document["source_fingerprint_id"])))
    graph.edge(root, "compares_target_fingerprint", graph.node("probe_fingerprint", str(document["target_fingerprint_id"])))
    graph.edge(root, "targets_model_capability", graph.node("model_capability", str(document["target_model_capability_id"])))
    for capability in document["required_capabilities"]:
        graph.edge(root, "requires_capability", graph.node("capability", str(capability)))
    for condition in document["anti_conditions"]:
        graph.edge(root, "bounded_by", graph.node("anti_condition", str(condition)))
    for field, relation in (("evaluator_binding", "bound_to_evaluator"), ("verdict_ref", "supported_by_verdict")):
        value = document[field]
        if isinstance(value, str):
            graph.edge(root, relation, graph.node("evidence", value), evidence=value)
    verdict = document.get("verdict_ref")
    if isinstance(verdict, str):
        graph.edge(
            graph.node("verdict", verdict),
            "establishes",
            semantic,
            evidence=document["evidence_refs"],
            claim_scope=document["claim_scope"],
            frozen=True,
        )
    for reference in document["evidence_refs"]:
        evidence = graph.node("evidence", str(reference))
        graph.edge(root, "cites_evidence", evidence, evidence=str(reference))


def _load_graph(graph: Mapping[str, object] | Path) -> Mapping[str, object]:
    if isinstance(graph, Mapping):
        payload = graph
    else:
        path = Path(graph).expanduser().resolve()
        if path.is_dir():
            path = path / "graph.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PortableKnowledgeGraphError("PORTABLE_KNOWLEDGE_GRAPH_NOT_FOUND") from exc
    if payload.get("artifact_type") != "verdiwm-portable-knowledge-graph":
        raise PortableKnowledgeGraphError("PORTABLE_KNOWLEDGE_GRAPH_CONTRACT_INVALID")
    _reject_path_bound_value(payload)
    return payload


def _normalize_similarity(values: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(values, Mapping):
        raise PortableKnowledgeGraphError(
            "PORTABLE_KNOWLEDGE_GRAPH_SIMILARITY_INVALID"
        )
    normalized: dict[str, object] = {}
    for key, value in values.items():
        name = str(key)
        if name not in _SIMILARITY_FIELDS:
            raise PortableKnowledgeGraphError(
                "PORTABLE_KNOWLEDGE_GRAPH_SIMILARITY_FIELD_INVALID:" + name
            )
        if isinstance(value, (str, int, float)) and not isinstance(value, bool):
            normalized[name] = value
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            sequence = list(value)
            if not sequence or any(
                isinstance(item, (Mapping, list, bool))
                or not isinstance(item, (str, int, float))
                for item in sequence
            ):
                raise PortableKnowledgeGraphError(
                    "PORTABLE_KNOWLEDGE_GRAPH_SIMILARITY_VALUE_INVALID:" + name
                )
            normalized[name] = sorted(set(sequence), key=lambda item: str(item))
        else:
            raise PortableKnowledgeGraphError(
                "PORTABLE_KNOWLEDGE_GRAPH_SIMILARITY_VALUE_INVALID:" + name
            )
    return dict(sorted(normalized.items()))


def _semantic_similarity(
    row: Mapping[str, object], requested: Mapping[str, object]
) -> float:
    if not requested:
        return 1.0
    scores = []
    for field, target in requested.items():
        observed = row.get(field)
        if isinstance(target, list):
            target_set = {_canonical_scalar(value) for value in target}
            observed_values = observed if isinstance(observed, list) else [observed]
            observed_set = {
                _canonical_scalar(value)
                for value in observed_values
                if isinstance(value, (str, int, float)) and not isinstance(value, bool)
            }
            union = target_set | observed_set
            scores.append(len(target_set & observed_set) / len(union) if union else 1.0)
        else:
            scores.append(1.0 if observed == target else 0.0)
    return sum(scores) / len(scores)


def _canonical_scalar(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _is_cas_sha256(value: object) -> bool:
    return isinstance(value, str) and _CAS_SHA256.fullmatch(value) is not None


def _reject_path_bound_value(value: object) -> None:
    try:
        from wmloop.geometry.evidence_ir import reject_runtime_bindings

        if isinstance(value, Mapping) and value.get("artifact_type") == (
            "verdiwm-portable-knowledge-graph"
        ):
            sanitized = dict(value)
            for field in ("node_kind_counts", "relation_counts"):
                counts = sanitized.pop(field, None)
                if not isinstance(counts, Mapping) or any(
                    not isinstance(key, str)
                    or _SEMANTIC_COUNT_KEY.fullmatch(key) is None
                    or isinstance(count, bool)
                    or not isinstance(count, int)
                    or count < 0
                    for key, count in counts.items()
                ):
                    raise PortableKnowledgeGraphError(
                        "PORTABLE_KNOWLEDGE_GRAPH_COUNTS_INVALID"
                    )
            reject_runtime_bindings(sanitized)
        else:
            reject_runtime_bindings(value)
    except GeometryValidationError as exc:
        raise PortableKnowledgeGraphError("PORTABLE_KNOWLEDGE_GRAPH_PATH_BOUND") from exc


def _node_id(kind: str, key: str) -> str:
    digest = hashlib.sha256(f"{kind}:{key}".encode("utf-8")).hexdigest()[:16]
    safe_key = "".join(character if character.isalnum() or character in "-_.:" else "_" for character in key)
    return f"{kind}:{safe_key[:240]}:{digest}"


def _unique_documents(
    documents: Sequence[Mapping[str, object]],
) -> list[Mapping[str, object]]:
    by_digest: dict[str, Mapping[str, object]] = {}
    for document in documents:
        if not isinstance(document, Mapping):
            raise PortableKnowledgeGraphError(
                "PORTABLE_KNOWLEDGE_DOCUMENT_INVALID"
            )
        by_digest.setdefault(_digest(document), document)
    return [by_digest[digest] for digest in sorted(by_digest)]


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PortableKnowledgeGraphError(
            "PORTABLE_KNOWLEDGE_CANONICAL_INVALID"
        ) from exc


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()
