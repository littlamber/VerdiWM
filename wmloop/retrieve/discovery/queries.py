"""Evidence-bound mechanism discovery components: queries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from wmloop.retrieve.discovery.common import _base_arxiv_id, _join_query, _query_terms
from wmloop.retrieve.discovery.types import (
    _COMPLEXITY_BUDGETS,
    DiscoveryRequest,
    MechanismDiscoveryError,
)
from wmloop.retrieve.literature import LiteratureRetrievalError, search_arxiv


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
            "query": _join_query(
                domain,
                ("history", "conditioning", "long", "horizon", "drift"),
                symptom[:4],
            ),
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
                (
                    "autoregressive",
                    "generated",
                    "history",
                    "training",
                    "inference",
                    "rollout",
                ),
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
        if current is None or len(str(row.get("abstract") or "")) > len(
            str(current.get("abstract") or "")
        ):
            deduplicated[key] = dict(row)
    return tuple(deduplicated.values())
