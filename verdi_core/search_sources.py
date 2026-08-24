"""Public literature/code search backends and a fan-out combiner."""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from dataclasses import dataclass

from .retrieval import SearchHit, SearchBackend


def _get_json(url: str, *, timeout: float = 20.0) -> object:
    request = urllib.request.Request(url, headers={"User-Agent": "VerdiWM/0.1 research client", "Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


class ArxivBackend:
    def search(self, query: str) -> list[SearchHit]:
        params = urllib.parse.urlencode({"search_query": f"all:{query}", "start": 0, "max_results": 10})
        url = "https://export.arxiv.org/api/query?" + params
        try:
            import xml.etree.ElementTree as ET
            with urllib.request.urlopen(url, timeout=20) as response:
                root = ET.fromstring(response.read())
            ns = {"a": "http://www.w3.org/2005/Atom"}
            hits = []
            for entry in root.findall("a:entry", ns):
                link = next((v.attrib.get("href") for v in entry.findall("a:link", ns) if v.attrib.get("type") == "text/html"), None)
                pdf = next((v.attrib.get("href") for v in entry.findall("a:link", ns) if v.attrib.get("title") == "pdf"), None)
                if link:
                    hits.append(SearchHit(link, entry.findtext("a:title", "", ns).strip(), "arxiv", entry.findtext("a:summary", "", ns).strip(), pdf))
            return hits
        except Exception:
            return []


class CrossrefBackend:
    def search(self, query: str) -> list[SearchHit]:
        url = "https://api.crossref.org/works?" + urllib.parse.urlencode({"query": query, "rows": 10, "select": "title,URL,abstract"})
        try:
            items = _get_json(url)["message"]["items"]
            return [SearchHit(str(item.get("URL")), str((item.get("title") or [""])[0]), "crossref", str(item.get("abstract", ""))) for item in items if item.get("URL")]
        except Exception:
            return []


class SemanticScholarBackend:
    def search(self, query: str) -> list[SearchHit]:
        url = "https://api.semanticscholar.org/graph/v1/paper/search?" + urllib.parse.urlencode({"query": query, "limit": 10, "fields": "title,url,abstract,openAccessPdf"})
        try:
            rows = _get_json(url).get("data", [])
            hits = []
            for row in rows:
                pdf = (row.get("openAccessPdf") or {}).get("url")
                hits.append(SearchHit(str(row.get("url") or pdf or ""), str(row.get("title", "")), "semantic_scholar", str(row.get("abstract", "") or ""), pdf))
            return [hit for hit in hits if hit.url]
        except Exception:
            return []


class GitHubBackend:
    """Public repository/code search using the GitHub REST API."""

    def __init__(self, token: str | None = None):
        self.token = token

    def search(self, query: str) -> list[SearchHit]:
        url = "https://api.github.com/search/repositories?" + urllib.parse.urlencode({"q": query, "per_page": 10})
        request = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json", "User-Agent": "VerdiWM/0.1"})
        if self.token:
            request.add_header("Authorization", "Bearer " + self.token)
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                rows = json.loads(response.read().decode("utf-8")).get("items", [])
            return [SearchHit(str(row.get("html_url")), str(row.get("full_name", "")), "github", str(row.get("description", ""))) for row in rows if row.get("html_url")]
        except Exception:
            return []


@dataclass
class FanoutBackend:
    backends: tuple[SearchBackend, ...]

    def search(self, query: str) -> list[SearchHit]:
        hits: list[SearchHit] = []
        seen: set[str] = set()
        for backend in self.backends:
            for hit in backend.search(query):
                if hit.url and hit.url not in seen:
                    seen.add(hit.url)
                    hits.append(hit)
        return hits
