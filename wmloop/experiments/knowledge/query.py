"""Portable knowledge graph components: query."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from wmloop.experiments.knowledge.common import _reject_path_bound_value
from wmloop.experiments.knowledge.types import (
    _SIMILARITY_FIELDS,
    PortableKnowledgeGraphError,
)


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
            raise PortableKnowledgeGraphError(
                "PORTABLE_KNOWLEDGE_GRAPH_NOT_FOUND"
            ) from exc
    if payload.get("artifact_type") != "verdiwm-portable-knowledge-graph":
        raise PortableKnowledgeGraphError("PORTABLE_KNOWLEDGE_GRAPH_CONTRACT_INVALID")
    _reject_path_bound_value(payload)
    return payload


def _normalize_similarity(values: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(values, Mapping):
        raise PortableKnowledgeGraphError("PORTABLE_KNOWLEDGE_GRAPH_SIMILARITY_INVALID")
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
