"""Portable knowledge graph components: validation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from wmloop.contracts import ContractValidationError, validate_document
from wmloop.control.module_composition import module_composition_receipt_digest
from wmloop.experiments.knowledge.common import _digest, _reject_path_bound_value
from wmloop.experiments.knowledge.graph import _PortableKnowledgeGraph
from wmloop.experiments.knowledge.types import PortableKnowledgeGraphError
from wmloop.geometry.community_knowledge import CommunityKnowledgeError


def _composition_module_node(
    graph: _PortableKnowledgeGraph,
    document: Mapping[str, object],
    module_id: str,
) -> str:
    rows = [
        row for row in document["selected_modules"] if row["module_id"] == module_id
    ]
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
        raise PortableKnowledgeGraphError("PORTABLE_KNOWLEDGE_COMPOSITION_ID_MISMATCH")


def _validate_community_document(
    validator: Any, document: Mapping[str, object]
) -> None:
    try:
        validator(document)
    except CommunityKnowledgeError as exc:
        raise PortableKnowledgeGraphError(
            f"PORTABLE_KNOWLEDGE_COMMUNITY_RECORD_INVALID:{exc}"
        ) from exc
