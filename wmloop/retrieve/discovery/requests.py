"""Evidence-bound mechanism discovery components: requests."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from wmloop.retrieve.discovery.common import _canonical_arxiv_id
from wmloop.retrieve.discovery.types import DiscoveryRequest, MechanismDiscoveryError


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
            cross_domain_lenses=_string_tuple(
                payload, "cross_domain_lenses", minimum=0
            ),
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
