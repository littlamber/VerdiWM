"""Bounded, data-only external research providers shared by all workflows."""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ElementTree
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol


FetchBytes = Callable[[str, float, int], bytes]


class ResearchProviderError(RuntimeError):
    """An external metadata provider failed or returned malformed data."""


@dataclass(frozen=True)
class ExternalResearchRecord:
    source: str
    record_id: str
    title: str
    summary: str
    source_url: str
    query: str
    published: str = ""


class ResearchProvider(Protocol):
    name: str

    def search(
        self,
        query: str,
        *,
        limit: int,
        timeout_seconds: float,
        byte_limit: int,
        fetch: FetchBytes,
    ) -> tuple[ExternalResearchRecord, ...]: ...


_ATOM = {"atom": "http://www.w3.org/2005/Atom"}
_ARXIV_ID = re.compile(r"arxiv.org/abs/([^/?]+)")


class ArxivProvider:
    name = "arxiv"

    def search(
        self,
        query: str,
        *,
        limit: int,
        timeout_seconds: float,
        byte_limit: int,
        fetch: FetchBytes,
    ) -> tuple[ExternalResearchRecord, ...]:
        url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
            {
                "search_query": f"all:{query}",
                "start": 0,
                "max_results": limit,
                "sortBy": "relevance",
            }
        )
        try:
            root = ElementTree.fromstring(fetch(url, timeout_seconds, byte_limit))
        except (OSError, ElementTree.ParseError) as exc:
            raise ResearchProviderError("RESEARCH_PROVIDER_ARXIV_FAILED") from exc
        records = []
        for entry in root.findall("atom:entry", _ATOM):
            raw_id = _text(entry.find("atom:id", _ATOM))
            match = _ARXIV_ID.search(raw_id)
            identity = match.group(1) if match else raw_id.rsplit("/", 1)[-1]
            title = _compact(_text(entry.find("atom:title", _ATOM)))
            summary = _compact(_text(entry.find("atom:summary", _ATOM)))
            published = _text(entry.find("atom:published", _ATOM))
            pdf_url = next(
                (
                    str(link.attrib.get("href") or "")
                    for link in entry.findall("atom:link", _ATOM)
                    if link.attrib.get("title") == "pdf"
                ),
                "",
            )
            if identity and title and summary:
                records.append(
                    ExternalResearchRecord(
                        source=self.name,
                        record_id=identity,
                        title=title,
                        summary=summary,
                        source_url=pdf_url or raw_id or f"https://arxiv.org/abs/{identity}",
                        query=query,
                        published=published,
                    )
                )
        return tuple(records)


class GithubProvider:
    name = "github"

    def search(
        self,
        query: str,
        *,
        limit: int,
        timeout_seconds: float,
        byte_limit: int,
        fetch: FetchBytes,
    ) -> tuple[ExternalResearchRecord, ...]:
        url = "https://api.github.com/search/repositories?" + urllib.parse.urlencode(
            {"q": query, "per_page": limit, "sort": "updated", "order": "desc"}
        )
        payload = _json_response(
            fetch, url, timeout_seconds, byte_limit, "RESEARCH_PROVIDER_GITHUB_FAILED"
        )
        items = payload.get("items")
        if not isinstance(items, list):
            raise ResearchProviderError("RESEARCH_PROVIDER_GITHUB_RESPONSE_INVALID")
        records = []
        for item in items[:limit]:
            if not isinstance(item, Mapping):
                continue
            identity = str(item.get("full_name") or "").strip()
            title = str(item.get("name") or identity).strip()
            summary = _compact(str(item.get("description") or ""))
            source_url = str(item.get("html_url") or "").strip()
            if identity and title and summary and source_url:
                records.append(
                    ExternalResearchRecord(
                        source=self.name,
                        record_id=identity,
                        title=title,
                        summary=summary,
                        source_url=source_url,
                        query=query,
                    )
                )
        return tuple(records)


class OpenAlexProvider:
    name = "openalex"

    def search(
        self,
        query: str,
        *,
        limit: int,
        timeout_seconds: float,
        byte_limit: int,
        fetch: FetchBytes,
    ) -> tuple[ExternalResearchRecord, ...]:
        url = "https://api.openalex.org/works?" + urllib.parse.urlencode(
            {"search": query, "per-page": limit, "sort": "relevance_score:desc"}
        )
        payload = _json_response(
            fetch, url, timeout_seconds, byte_limit, "RESEARCH_PROVIDER_OPENALEX_FAILED"
        )
        items = payload.get("results")
        if not isinstance(items, list):
            raise ResearchProviderError("RESEARCH_PROVIDER_OPENALEX_RESPONSE_INVALID")
        records = []
        for item in items[:limit]:
            if not isinstance(item, Mapping):
                continue
            raw_id = str(item.get("id") or "").strip()
            identity = raw_id.rsplit("/", 1)[-1]
            title = str(item.get("display_name") or "").strip()
            summary = _openalex_abstract(item.get("abstract_inverted_index"))
            source_url = str(
                item.get("landing_page_url") or item.get("doi") or raw_id
            ).strip()
            published = str(item.get("publication_date") or "").strip()
            if identity and title and summary and source_url:
                records.append(
                    ExternalResearchRecord(
                        source=self.name,
                        record_id=identity,
                        title=title,
                        summary=summary,
                        source_url=source_url,
                        query=query,
                        published=published,
                    )
                )
        return tuple(records)


_PROVIDERS: dict[str, ResearchProvider] = {
    provider.name: provider
    for provider in (ArxivProvider(), GithubProvider(), OpenAlexProvider())
}


def search_provider(
    source: str,
    query: str,
    *,
    limit: int,
    timeout_seconds: float,
    byte_limit: int,
    fetch: FetchBytes | None = None,
) -> tuple[ExternalResearchRecord, ...]:
    """Search one registered provider under common resource limits."""

    if not query.strip() or not 1 <= limit <= 50:
        raise ResearchProviderError("RESEARCH_PROVIDER_QUERY_INVALID")
    if not 0 < timeout_seconds <= 60 or not 1 <= byte_limit <= 20_000_000:
        raise ResearchProviderError("RESEARCH_PROVIDER_LIMIT_INVALID")
    provider = _PROVIDERS.get(source)
    if provider is None:
        raise ResearchProviderError("RESEARCH_PROVIDER_UNSUPPORTED")
    return provider.search(
        query.strip(),
        limit=limit,
        timeout_seconds=timeout_seconds,
        byte_limit=byte_limit,
        fetch=fetch or fetch_url,
    )


def fetch_url(url: str, timeout_seconds: float, byte_limit: int) -> bytes:
    """Fetch bounded metadata bytes without importing or executing source code."""

    request = urllib.request.Request(url, headers={"User-Agent": "verdiwm/1 research-provider"})
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            payload = response.read(byte_limit + 1)
    except OSError as exc:
        raise ResearchProviderError("RESEARCH_PROVIDER_NETWORK_UNAVAILABLE") from exc
    if len(payload) > byte_limit:
        raise ResearchProviderError("RESEARCH_PROVIDER_RESPONSE_TOO_LARGE")
    return payload


def _json_response(
    fetch: FetchBytes,
    url: str,
    timeout_seconds: float,
    byte_limit: int,
    code: str,
) -> Mapping[str, object]:
    try:
        payload = json.loads(fetch(url, timeout_seconds, byte_limit))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResearchProviderError(code) from exc
    if not isinstance(payload, Mapping):
        raise ResearchProviderError(code)
    return payload


def _openalex_abstract(value: object) -> str:
    if not isinstance(value, Mapping):
        return ""
    indexed: dict[int, str] = {}
    for token, positions in value.items():
        if not isinstance(token, str) or not isinstance(positions, list):
            continue
        for position in positions:
            if isinstance(position, int) and position >= 0:
                indexed.setdefault(position, token)
    return _compact(" ".join(indexed[index] for index in sorted(indexed)))


def _text(node: ElementTree.Element | None) -> str:
    return (node.text or "").strip() if node is not None else ""


def _compact(value: str) -> str:
    return " ".join(value.split())[:4000]
