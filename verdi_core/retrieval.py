"""Pluggable web/literature retrieval with deterministic human handoff."""

from __future__ import annotations

import json
import mimetypes
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from .contracts import canonical_digest


@dataclass(frozen=True)
class RetrievalRequest:
    objective: str
    portrait: dict[str, Any]
    constraints: tuple[str, ...] = ()


@dataclass(frozen=True)
class SearchHit:
    url: str
    title: str = ""
    source: str = "web"
    snippet: str = ""
    pdf_url: str | None = None


@dataclass(frozen=True)
class AcquiredDocument:
    status: str
    url: str
    local_path: str | None
    content_type: str | None
    error: str | None = None


class SearchPlanner(Protocol):
    def plan(self, request: RetrievalRequest) -> list[str]: ...


class SearchBackend(Protocol):
    def search(self, query: str) -> list[SearchHit]: ...


class OnlineRetriever:
    """Searches arbitrary backends; fetches HTML first and stages PDF fallback."""

    def __init__(self, backend: SearchBackend, *, state_root: Path, timeout: float = 15.0):
        self.backend = backend
        self.state_root = Path(state_root)
        self.timeout = timeout
        self.inbox = self.state_root / "retrieval" / "inbox"
        self.inbox.mkdir(parents=True, exist_ok=True)

    def retrieve(self, queries: list[str], *, limit: int = 20) -> list[AcquiredDocument]:
        hits: list[SearchHit] = []
        seen: set[str] = set()
        for query in queries:
            for hit in self.backend.search(query):
                if hit.url not in seen:
                    seen.add(hit.url)
                    hits.append(hit)
                if len(hits) >= limit:
                    break
            if len(hits) >= limit:
                break
        return [self._acquire(hit) for hit in hits]

    def _acquire(self, hit: SearchHit) -> AcquiredDocument:
        last_error: Exception | None = None
        candidates = [hit.url] + ([hit.pdf_url] if hit.pdf_url else [])
        for candidate in candidates:
            if not candidate:
                continue
            request = urllib.request.Request(candidate, headers={"User-Agent": "VerdiWM/0.1"})
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = response.read()
                    content_type = response.headers.get_content_type()
                    suffix = ".pdf" if content_type == "application/pdf" or candidate.lower().split("?")[0].endswith(".pdf") else ".html"
                    target = self.inbox / ("document-" + canonical_digest(candidate)[7:23] + suffix)
                    target.write_bytes(body)
                    return AcquiredDocument("acquired_pdf" if suffix == ".pdf" else "acquired_html", candidate, str(target), content_type)
            except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
                last_error = exc
        return AcquiredDocument("human_download", hit.url, str(self.inbox), None, str(last_error) if last_error else "no URL")


class RetrievalLedger:
    def __init__(self, state_root: Path):
        self.path = Path(state_root) / "retrieval" / "ledger.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, documents: list[AcquiredDocument]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            for document in documents:
                handle.write(json.dumps(asdict(document), sort_keys=True) + "\n")
