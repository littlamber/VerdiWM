"""Versioned, data-driven query planning for mechanism discovery."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


class QueryPolicyError(ValueError):
    """A discovery query policy is missing or malformed."""


@dataclass(frozen=True)
class QueryView:
    name: str
    static_terms: tuple[str, ...]
    dynamic_sources: tuple[str, ...]
    max_dynamic_terms: int


@dataclass(frozen=True)
class DomainRule:
    name: str
    match_terms: tuple[str, ...]
    lenses: tuple[str, ...]


@dataclass(frozen=True)
class DiscoveryQueryPolicy:
    policy_id: str
    path: Path
    sha256: str
    base_terms: tuple[str, ...]
    query_views: tuple[QueryView, ...]
    cross_domain_base_terms: tuple[str, ...]
    fallback_cross_domain_lenses: tuple[str, ...]
    domain_rules: tuple[DomainRule, ...]

    def domain_tags(self, values: Sequence[object]) -> tuple[str, ...]:
        text = " ".join(str(value) for value in values).lower()
        matched = tuple(
            rule.name
            for rule in self.domain_rules
            if rule.name != "default" and any(term in text for term in rule.match_terms)
        )
        return matched or ("default",)

    def lenses_for(self, tags: Sequence[str]) -> tuple[str, ...]:
        by_name = {rule.name: rule for rule in self.domain_rules}
        lenses = [
            lens
            for tag in tags
            for lens in by_name.get(tag, by_name["default"]).lenses
        ]
        return tuple(dict.fromkeys(lenses))

    def build_queries(self, request: object) -> tuple[dict[str, str], ...]:
        values = {
            "symptom": (str(getattr(request, "symptom_description")),),
            "failure_signatures": tuple(getattr(request, "failure_signatures")),
            "target_metrics": tuple(getattr(request, "target_metrics")),
            "available_hooks": tuple(getattr(request, "available_hooks")),
            "model_family": (str(getattr(request, "model_family")),),
        }
        rows: list[dict[str, str]] = []
        for view in self.query_views:
            dynamic_text = " ".join(
                str(value)
                for source in view.dynamic_sources
                for value in values.get(source, ())
            )
            dynamic = _query_terms(dynamic_text, maximum=view.max_dynamic_terms)
            rows.append(
                {
                    "view": view.name,
                    "query": _join_query(self.base_terms, view.static_terms, dynamic),
                }
            )
        lenses = (
            tuple(getattr(request, "cross_domain_lenses"))
            or self.fallback_cross_domain_lenses
        )
        for index, lens in enumerate(lenses, start=1):
            rows.append(
                {
                    "view": f"cross_domain_{index}",
                    "query": _join_query(
                        self.cross_domain_base_terms,
                        _query_terms(str(lens), maximum=8),
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


def load_discovery_query_policy(
    path: Path | None = None,
    *,
    root: Path | None = None,
) -> DiscoveryQueryPolicy:
    repository = Path(root or Path(__file__).resolve().parents[2]).resolve()
    source = Path(path).expanduser().resolve() if path is not None else (
        repository / "configs" / "retrieval" / "discovery_query_policy_v1.json"
    )
    try:
        raw = source.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise QueryPolicyError("DISCOVERY_QUERY_POLICY_INVALID") from exc
    if not isinstance(payload, Mapping):
        raise QueryPolicyError("DISCOVERY_QUERY_POLICY_INVALID")
    try:
        if payload["artifact_type"] != "verdiwm-discovery-query-policy":
            raise ValueError
        views = tuple(
            QueryView(
                name=_text(row, "view"),
                static_terms=_strings(row, "static_terms"),
                dynamic_sources=_strings(row, "dynamic_sources", allow_empty=True),
                max_dynamic_terms=_integer(row, "max_dynamic_terms", minimum=0, maximum=20),
            )
            for row in _mappings(payload, "query_views")
        )
        rules = tuple(
            DomainRule(
                name=_text(row, "domain"),
                match_terms=_strings(row, "match_terms", allow_empty=True),
                lenses=_strings(row, "lenses"),
            )
            for row in _mappings(payload, "domain_rules")
        )
        if len({view.name for view in views}) != len(views):
            raise ValueError
        if len({rule.name for rule in rules}) != len(rules) or "default" not in {
            rule.name for rule in rules
        }:
            raise ValueError
        allowed_sources = {
            "symptom",
            "failure_signatures",
            "target_metrics",
            "available_hooks",
            "model_family",
        }
        if any(set(view.dynamic_sources) - allowed_sources for view in views):
            raise ValueError
        return DiscoveryQueryPolicy(
            policy_id=_text(payload, "policy_id"),
            path=source,
            sha256=hashlib.sha256(raw).hexdigest(),
            base_terms=_strings(payload, "base_terms"),
            query_views=views,
            cross_domain_base_terms=_strings(payload, "cross_domain_base_terms"),
            fallback_cross_domain_lenses=_strings(payload, "fallback_cross_domain_lenses"),
            domain_rules=rules,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise QueryPolicyError("DISCOVERY_QUERY_POLICY_INVALID") from exc


def _mappings(value: Mapping[str, object], key: str) -> tuple[Mapping[str, object], ...]:
    rows = value[key]
    if not isinstance(rows, list) or not rows or any(not isinstance(row, Mapping) for row in rows):
        raise ValueError(key)
    return tuple(rows)


def _strings(
    value: Mapping[str, object], key: str, *, allow_empty: bool = False
) -> tuple[str, ...]:
    rows = value[key]
    if not isinstance(rows, list) or (not rows and not allow_empty):
        raise ValueError(key)
    cleaned = tuple(
        str(row).strip().lower()
        for row in rows
        if isinstance(row, str) and row.strip()
    )
    if len(cleaned) != len(rows) or len(set(cleaned)) != len(cleaned):
        raise ValueError(key)
    return cleaned


def _text(value: Mapping[str, object], key: str) -> str:
    item = value[key]
    if not isinstance(item, str) or not item.strip():
        raise ValueError(key)
    return item.strip()


def _integer(value: Mapping[str, object], key: str, *, minimum: int, maximum: int) -> int:
    item = value[key]
    if isinstance(item, bool) or not isinstance(item, int) or not minimum <= item <= maximum:
        raise ValueError(key)
    return item


def _query_terms(value: str, *, maximum: int) -> tuple[str, ...]:
    if maximum <= 0:
        return ()
    ignored = {"and", "for", "from", "into", "model", "the", "this", "with"}
    tokens = [
        token
        for token in re.findall(r"[a-z0-9]+", value.lower())
        if len(token) > 2 and token not in ignored
    ]
    return tuple(dict.fromkeys(tokens))[:maximum]


def _join_query(*groups: Sequence[str]) -> str:
    return " ".join(dict.fromkeys(term for group in groups for term in group if term))
