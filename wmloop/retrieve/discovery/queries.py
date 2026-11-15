"""Evidence-bound mechanism discovery components: queries."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from wmloop.retrieve.discovery.common import _base_arxiv_id, _join_query, _query_terms
from wmloop.retrieve.discovery.types import (
    _COMPLEXITY_BUDGETS,
    DiscoveryRequest,
    MechanismDiscoveryError,
)
from wmloop.retrieve.literature import LiteratureRetrievalError, search_arxiv
from wmloop.retrieve.query_terms import query_terms


_LOW_SIGNAL_SYMPTOM_TERMS = frozenset({
    "across", "change", "changes", "effect", "effects", "otherwise", "similar",
    "comparable", "old", "different", "difference",
})


def build_multiview_queries(request: DiscoveryRequest) -> tuple[dict[str, object], ...]:
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
    model_family_terms = tuple(
        term
        for term in query_terms(request.model_family, maximum=8, intent=True)
        if term not in {"world", "model", "video", "diffusion", "prediction"}
    )
    domain_search = tuple(dict.fromkeys(
        ("video", "prediction", "world-model", "diffusion", *model_family_terms)
    ))
    symptom_search = ("history", "context", "conditioning", "drift", "mismatch", "rollout")
    target_domain = tuple(dict.fromkeys(
        ("video", "world-model", "diffusion", *model_family_terms)
    ))
    symptom_description_terms = tuple(
        term
        for term in query_terms(request.symptom_description, maximum=24, intent=True)
        if term not in _LOW_SIGNAL_SYMPTOM_TERMS
    )[:8]
    symptom_signature_terms = tuple(
        term
        for term in query_terms(" ".join(request.failure_signatures), maximum=12, intent=True)
        if term not in _LOW_SIGNAL_SYMPTOM_TERMS
    )[:6]
    target_search = query_terms(" ".join(request.target_metrics), maximum=8, intent=True)
    hook_search = query_terms(" ".join(request.available_hooks), maximum=8, intent=True)
    rows = [
        {
            "view": "diagnostic_symptom",
            "query": _join_query(
                domain,
                ("history", "conditioning", "long", "horizon", "drift"),
                symptom[:4],
            ),
            "search_expression": _search_expression(
                domain_search,
                ("history", "context", "conditioning"),
                (
                    *symptom_search,
                    *symptom_description_terms,
                    *symptom_signature_terms,
                ),
            ),
            "relevance_groups": _relevance_groups(
                ("target_domain", target_domain),
                ("task_context", ("history", "context", "conditioning", "rollout")),
                ("current_symptom", (*symptom_description_terms, *symptom_signature_terms)),
            ),
            "required_relevance_groups": ["target_domain", "task_context"],
        },
        {
            "view": "state_update_operator",
            "query": _join_query(
                domain,
                ("adaptive", "history", "memory", "reliability", "update"),
                metrics,
            ),
            "search_expression": _search_expression(
                domain_search,
                ("history", "memory", "state"),
                ("adaptive", "update", "confidence", "reliability", "filter"),
                target_search,
            ),
            "relevance_groups": _relevance_groups(
                ("target_domain", target_domain),
                ("state_memory", ("history", "memory")),
                ("update_operator", ("adaptive", "update", "confidence", "reliability", "filter")),
                ("target_metric", target_search),
            ),
            "required_relevance_groups": ["target_domain", "state_memory", "update_operator"],
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
            "search_expression": _search_expression(
                domain_search,
                ("autoregressive", "rollout", "generated", "free-running"),
                ("training", "inference", "distribution", "mismatch"),
            ),
            "relevance_groups": _relevance_groups(
                ("target_domain", target_domain),
                ("autoregressive_rollout", ("autoregressive", "rollout", "generated", "free-running")),
                ("training_inference", ("training", "inference", "distribution", "mismatch")),
            ),
            "required_relevance_groups": [
                "target_domain", "autoregressive_rollout", "training_inference"
            ],
        },
        {
            "view": "architecture_hook",
            "query": _join_query(
                domain,
                ("temporal", "latent", "conditioning", "memory", "action"),
            ),
            "search_expression": _search_expression(
                domain_search,
                ("temporal", "latent", "conditioning", "memory"),
                hook_search,
            ),
            "relevance_groups": _relevance_groups(
                ("target_domain", target_domain),
                ("temporal_conditioning", ("temporal", "latent", "conditioning", "memory")),
                ("available_hook", hook_search),
            ),
            "required_relevance_groups": ["target_domain", "temporal_conditioning"],
        },
    ]
    lenses = request.cross_domain_lenses or (
        "belief state estimation confidence update",
        "adaptive memory retention forgetting",
        "online system identification uncertainty",
    )
    for index, lens in enumerate(lenses):
        lens_terms = query_terms(lens, maximum=8, intent=True)
        rows.append(
            {
                "view": f"cross_domain_{index + 1}",
                "query": _join_query(
                    ("sequence", "prediction", "temporal", "memory"),
                    _query_terms(lens, maximum=8),
                ),
                "search_expression": _search_expression(
                    ("sequence", "prediction", "video", "temporal"),
                    ("memory", "belief", "state", "confidence", "update"),
                    lens_terms,
                ),
                "relevance_groups": _relevance_groups(
                    ("target_domain", target_domain),
                    ("state_mechanism", ("memory", "belief", "state", "confidence", "update")),
                    ("cross_domain_lens", lens_terms),
                ),
                "required_relevance_groups": list(
                    ("target_domain", "state_mechanism", "cross_domain_lens")
                    if lens_terms
                    else ("target_domain", "state_mechanism")
                ),
            }
        )
    deduplicated: list[dict[str, object]] = []
    seen: set[str] = set()
    for row in rows:
        query = row["query"].strip()
        if query and query not in seen:
            deduplicated.append(row)
            seen.add(query)
    return tuple(deduplicated)


def _collect_query_seeds(
    queries: Sequence[Mapping[str, Any]],
    *,
    max_results: int,
    timeout_seconds: float,
    search_client: Any | None,
) -> tuple[tuple[Mapping[str, Any], ...], dict[str, dict[str, Any]]]:
    if max_results == 0:
        return (), {
            str(row["view"]): {
                "source_state": "disabled",
                "candidate_count": 0,
                "accepted_seed_count": 0,
                "candidates": [],
            }
            for row in queries
        }
    search = search_client or search_arxiv
    records: list[Mapping[str, Any]] = []
    results: dict[str, dict[str, Any]] = {}
    for row in queries:
        view = str(row["view"])
        try:
            found, state = search(
                str(row.get("search_expression") or row["query"]),
                max_results=max_results,
                timeout_seconds=timeout_seconds,
            )
        except (LiteratureRetrievalError, OSError):
            found, state = (), "offline"
        if len(found) > max_results:
            raise MechanismDiscoveryError("MECHANISM_SEARCH_RESULT_LIMIT_EXCEEDED")
        candidates: list[dict[str, Any]] = []
        anchors = tuple(dict.fromkeys(
            re.findall(r'all:"([^"]+)"', str(row.get("search_expression") or ""))
        ))
        accepted_count = 0
        for record in found:
            if hasattr(record, "to_dict"):
                candidate = record.to_dict()
            elif isinstance(record, Mapping):
                candidate = dict(record)
            else:
                raise MechanismDiscoveryError("MECHANISM_SEARCH_RESULT_INVALID")
            title = str(candidate.get("title") or "")[:1000]
            abstract = " ".join(str(candidate.get("abstract") or "").split())[:1200]
            candidate_terms = set(query_terms(f"{title} {abstract}", maximum=128, intent=True))
            matched_groups = [
                str(group["name"])
                for group in row.get("relevance_groups", ())
                if candidate_terms.intersection(str(term) for term in group["terms"])
            ]
            required_groups = tuple(str(value) for value in row.get("required_relevance_groups", ()))
            relevance_state = (
                "anchor_gate_passed"
                if required_groups and all(group in matched_groups for group in required_groups)
                else "low_relevance"
            )
            if relevance_state == "anchor_gate_passed":
                accepted_count += 1
                records.append(candidate)
            candidates.append({
                "arxiv_id": str(candidate.get("arxiv_id") or "")[:64],
                "title": title,
                "abstract_excerpt": abstract,
                "matched_anchor_terms": [term for term in anchors if term in candidate_terms],
                "matched_anchor_groups": matched_groups,
                "relevance_state": relevance_state,
            })
        results[view] = {
            "source_state": str(state),
            "candidate_count": len(found),
            "accepted_seed_count": accepted_count,
            "candidates": candidates,
        }
    return _deduplicate_seed_records(records), results


def _search_expression(*groups: Sequence[str]) -> str:
    clauses = []
    for group in groups:
        terms = query_terms(" ".join(group), maximum=16, intent=True)
        if terms:
            clauses.append("(" + " OR ".join(f'all:"{term}"' for term in terms) + ")")
    return " AND ".join(clauses)


def _relevance_groups(
    *groups: tuple[str, Sequence[str]],
) -> list[dict[str, object]]:
    return [
        {"name": name, "terms": list(dict.fromkeys(terms))}
        for name, terms in groups
        if terms
    ]


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
