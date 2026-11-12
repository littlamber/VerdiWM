"""Portable knowledge graph components: common."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence

from wmloop.experiments.knowledge.types import (
    _CAS_SHA256,
    _SEMANTIC_COUNT_KEY,
    PortableKnowledgeGraphError,
)
from wmloop.geometry.types import GeometryValidationError
from wmloop.storage import canonical_bytes, sha256_bytes


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
        raise PortableKnowledgeGraphError(
            "PORTABLE_KNOWLEDGE_GRAPH_PATH_BOUND"
        ) from exc


def _node_id(kind: str, key: str) -> str:
    digest = hashlib.sha256(f"{kind}:{key}".encode("utf-8")).hexdigest()[:16]
    safe_key = "".join(
        character if character.isalnum() or character in "-_.:" else "_"
        for character in key
    )
    return f"{kind}:{safe_key[:240]}:{digest}"


def _unique_documents(
    documents: Sequence[Mapping[str, object]],
) -> list[Mapping[str, object]]:
    by_digest: dict[str, Mapping[str, object]] = {}
    for document in documents:
        if not isinstance(document, Mapping):
            raise PortableKnowledgeGraphError("PORTABLE_KNOWLEDGE_DOCUMENT_INVALID")
        by_digest.setdefault(_digest(document), document)
    return [by_digest[digest] for digest in sorted(by_digest)]


def _canonical(value: object) -> bytes:
    try:
        return canonical_bytes(value)
    except (TypeError, ValueError) as exc:
        raise PortableKnowledgeGraphError(
            "PORTABLE_KNOWLEDGE_CANONICAL_INVALID"
        ) from exc


def _digest(value: object) -> str:
    return sha256_bytes(_canonical(value))
