"""Evidence-bound mechanism discovery components: common."""

from __future__ import annotations

import html
import json
import re
import xml.etree.ElementTree as ElementTree
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from wmloop.retrieve.discovery.types import (
    _ARXIV_ID,
    DiscoveryRequest,
    MechanismDiscoveryError,
)
from wmloop.retrieve.query_terms import query_terms as _query_terms
from wmloop.storage import atomic_write, canonical_json_bytes, sha256_bytes


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
    return canonical_json_bytes(value)


def _sha256(value: bytes) -> str:
    return sha256_bytes(value)


def _write_json(path: Path, value: object) -> None:
    atomic_write(path, _canonical_json(value))
