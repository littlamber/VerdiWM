"""Source acquisition and text extraction for HTML, PDF, and code."""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:  # Optional parsers keep the CPU fixture usable before extras are installed.
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover - exercised only in minimal installs
    BeautifulSoup = None
try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover
    PdfReader = None
try:
    import trafilatura
except ImportError:  # pragma: no cover
    trafilatura = None


@dataclass(frozen=True)
class IngestedDocument:
    document_id: str
    status: str
    source_url: str | None
    local_path: str
    content_type: str
    text: str
    title: str
    metadata: dict[str, Any]


class DocumentIngestor:
    def __init__(self, inbox: Path):
        self.inbox = Path(inbox)
        self.inbox.mkdir(parents=True, exist_ok=True)

    def ingest_file(self, path: Path, *, source_url: str | None = None, metadata: dict[str, Any] | None = None) -> IngestedDocument:
        path = Path(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        metadata = dict(metadata or {})
        metadata["sha256"] = digest
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return self._pdf(path, source_url, metadata, digest)
        if suffix in {".html", ".htm"}:
            return self._html(path, source_url, metadata, digest)
        if suffix in {".py", ".js", ".ts", ".java", ".cpp", ".c", ".md", ".txt", ".json", ".yaml", ".yml"}:
            text = path.read_text(encoding="utf-8", errors="replace")
            return IngestedDocument(digest[:24], "acquired_code", source_url, str(path), "text/plain", text, path.name, metadata)
        return IngestedDocument(digest[:24], "unsupported_type", source_url, str(path), "application/octet-stream", "", path.name, metadata)

    def scan_inbox(self) -> list[IngestedDocument]:
        documents = []
        for path in sorted(self.inbox.iterdir()):
            if path.is_file() and not path.name.startswith("."):
                documents.append(self.ingest_file(path))
        return documents

    def clone_repository(self, url: str, target: Path, *, revision: str | None = None) -> list[IngestedDocument]:
        target = Path(target)
        command = ["git", "clone", "--depth", "1"] + (["--branch", revision] if revision else []) + [url, str(target)]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True, timeout=180)
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            return [IngestedDocument("repo-" + hashlib.sha256(url.encode()).hexdigest()[:20], "human_download", url, str(self.inbox), "text/plain", "", url, {"error": str(exc)})]
        documents = []
        for path in sorted(target.rglob("*")):
            if path.is_file() and path.stat().st_size < 1_000_000 and path.suffix.lower() in {".py", ".js", ".ts", ".java", ".cpp", ".c", ".md", ".txt", ".json", ".yaml", ".yml"}:
                documents.append(self.ingest_file(path, source_url=url, metadata={"repository": url}))
        return documents

    def _html(self, path: Path, source_url: str | None, metadata: dict[str, Any], digest: str) -> IngestedDocument:
        raw = path.read_text(encoding="utf-8", errors="replace")
        if trafilatura:
            extracted = trafilatura.extract(raw, include_links=True, include_tables=True)
        else:
            extracted = None
        if BeautifulSoup:
            soup = BeautifulSoup(raw, "html.parser")
            extracted = extracted or soup.get_text("\n", strip=True)
            title = soup.title.get_text(" ", strip=True) if soup.title else path.name
        else:
            import re
            extracted = extracted or re.sub(r"<[^>]+>", " ", raw)
            title = path.name
        status = "acquired_html" if extracted.strip() else "needs_human_review"
        return IngestedDocument(digest[:24], status, source_url, str(path), "text/html", extracted, title, metadata)

    def _pdf(self, path: Path, source_url: str | None, metadata: dict[str, Any], digest: str) -> IngestedDocument:
        try:
            if PdfReader is None:
                return IngestedDocument(digest[:24], "needs_dependency", source_url, str(path), "application/pdf", "", path.name, {**metadata, "missing": "pypdf"})
            reader = PdfReader(str(path))
            text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
        except Exception as exc:  # malformed/encrypted/scanned PDFs are surfaced, not hidden
            return IngestedDocument(digest[:24], "needs_ocr", source_url, str(path), "application/pdf", "", path.name, {**metadata, "error": str(exc)})
        status = "acquired_pdf" if text else "needs_ocr"
        return IngestedDocument(digest[:24], status, source_url, str(path), "application/pdf", text, path.name, metadata)
