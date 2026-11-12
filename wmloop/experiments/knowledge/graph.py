"""Portable knowledge graph components: graph."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from wmloop.experiments.knowledge.common import _node_id
from wmloop.experiments.knowledge.types import PortableKnowledgeGraphError


class _PortableKnowledgeGraph:
    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, object]] = {}
        self.edges: dict[str, dict[str, object]] = {}
        self.document_count = 0

    def node(self, kind: str, key: str, **attributes: object) -> str:
        identifier = _node_id(kind, key)
        row = self.nodes.setdefault(
            identifier, {"id": identifier, "kind": kind, "key": key}
        )
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
                        "PORTABLE_KNOWLEDGE_EDGE_CONFLICT:"
                        + str(row["id"])
                        + ":"
                        + name
                    )
                row[name] = value
        if evidence is not None:
            values = [evidence] if isinstance(evidence, str) else list(evidence)
            row.setdefault("evidence", set()).update(str(value) for value in values)

    def document(self) -> dict[str, object]:
        nodes = [
            dict(value)
            for value in sorted(self.nodes.values(), key=lambda item: str(item["id"]))
        ]
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
            "node_kind_counts": dict(
                sorted(Counter(str(row["kind"]) for row in nodes).items())
            ),
            "relation_counts": dict(
                sorted(Counter(str(row["relation"]) for row in edges).items())
            ),
            "claim_boundary": (
                "This is a semantic, path-free projection. It ranks or describes "
                "transfer evidence but does not replace target-side frozen verification."
            ),
            "nodes": nodes,
            "edges": edges,
        }
