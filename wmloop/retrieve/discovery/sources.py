"""Evidence-bound mechanism discovery components: sources."""

from __future__ import annotations

import heapq
import shutil
import subprocess
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ElementTree
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from wmloop.retrieve.discovery.common import (
    _base_arxiv_id,
    _canonical_arxiv_id,
    _node_text,
    _sha256,
)
from wmloop.retrieve.discovery.types import (
    _ARXIV_ID,
    DiscoveryPaper,
    MechanismDiscoveryError,
)


def _expand_papers(
    *,
    seed_records: Sequence[Mapping[str, Any]],
    seed_ids: Sequence[str],
    citation_depth: int,
    max_papers: int,
    timeout_seconds: float,
    fetch_text: Any,
    fetch_metadata: Any,
    local_sources_root: Path | None,
) -> tuple[tuple[DiscoveryPaper, ...], tuple[dict[str, object], ...]]:
    metadata: dict[str, dict[str, str]] = {}
    queue: list[tuple[int, int, str, int, tuple[str, ...]]] = []
    order = 0
    for row in seed_records:
        raw_id = str(row.get("arxiv_id") or "")
        if not raw_id:
            continue
        arxiv_id = _canonical_arxiv_id(raw_id)
        metadata[arxiv_id] = {
            "arxiv_id": arxiv_id,
            "title": str(row.get("title") or "Unknown title"),
            "abstract": str(row.get("abstract") or row.get("mechanism_summary") or ""),
            "source_url": str(
                row.get("pdf_url")
                or row.get("source_url")
                or f"https://arxiv.org/abs/{arxiv_id}"
            ),
        }
        heapq.heappush(queue, (20, order, arxiv_id, 0, ("query_or_seed_record",)))
        order += 1
    for raw_id in seed_ids:
        arxiv_id = _canonical_arxiv_id(raw_id)
        heapq.heappush(queue, (0, order, arxiv_id, 0, ("seed_request",)))
        order += 1
    papers: list[DiscoveryPaper] = []
    missing_sources: list[dict[str, object]] = []
    seen: set[str] = set()
    while queue and len(papers) < max_papers:
        batch: list[tuple[int, int, str, int, tuple[str, ...]]] = []
        while queue and len(batch) < 4 and len(papers) + len(batch) < max_papers:
            item = heapq.heappop(queue)
            base_id = _base_arxiv_id(item[2])
            if base_id in seen:
                continue
            seen.add(base_id)
            batch.append(item)
        if not batch:
            continue
        with ThreadPoolExecutor(max_workers=len(batch)) as executor:
            loaded = list(
                executor.map(
                    lambda item: _load_discovery_paper(
                        item,
                        metadata=metadata,
                        timeout_seconds=timeout_seconds,
                        fetch_text=fetch_text,
                        fetch_metadata=fetch_metadata,
                        local_sources_root=local_sources_root,
                    ),
                    batch,
                )
            )
        for item, result in zip(batch, loaded):
            paper, cited_ids, missing = result
            papers.append(paper)
            if missing is not None:
                missing_sources.append(missing)
            priority = item[0]
            if paper.citation_depth < citation_depth:
                citation_priority = 5 if priority < 20 else 25
                for cited_id in cited_ids:
                    if _base_arxiv_id(cited_id) == _base_arxiv_id(paper.arxiv_id):
                        continue
                    heapq.heappush(
                        queue,
                        (
                            citation_priority,
                            order,
                            str(cited_id),
                            paper.citation_depth + 1,
                            (paper.arxiv_id,),
                        ),
                    )
                    order += 1
    return tuple(papers), tuple(missing_sources)


def _load_discovery_paper(
    item: tuple[int, int, str, int, tuple[str, ...]],
    *,
    metadata: Mapping[str, Mapping[str, str]],
    timeout_seconds: float,
    fetch_text: Any,
    fetch_metadata: Any,
    local_sources_root: Path | None,
) -> tuple[DiscoveryPaper, tuple[str, ...], dict[str, object] | None]:
    _, _, arxiv_id, depth, parents = item
    row = metadata.get(arxiv_id) or metadata.get(_base_arxiv_id(arxiv_id))
    metadata_state = "seed"
    if row is None:
        try:
            fetched = fetch_metadata(arxiv_id, timeout_seconds=timeout_seconds)
        except Exception:
            fetched = None
        if isinstance(fetched, Mapping):
            row = {key: str(value) for key, value in fetched.items()}
            arxiv_id = _canonical_arxiv_id(row.get("arxiv_id") or arxiv_id)
            metadata_state = "network"
        else:
            row = {
                "arxiv_id": arxiv_id,
                "title": f"Metadata unavailable for {arxiv_id}",
                "abstract": "",
                "source_url": f"https://arxiv.org/abs/{arxiv_id}",
            }
            metadata_state = "unavailable"
    full_text_url = f"https://ar5iv.labs.arxiv.org/html/{_base_arxiv_id(arxiv_id)}"
    local_path: Path | None = None
    local_sha256: str | None = None
    local_error: str | None = None
    local_result = _load_local_source(arxiv_id, local_sources_root)
    if local_result is not None:
        full_text, cited_ids, state, local_path, local_error = local_result
        local_sha256 = (
            _sha256(local_path.read_bytes()) if local_path is not None else None
        )
    else:
        try:
            full_text, cited_ids, state = fetch_text(
                arxiv_id,
                timeout_seconds=timeout_seconds,
            )
        except Exception:
            full_text, cited_ids, state = "", (), "unavailable"
    paper = DiscoveryPaper(
        arxiv_id=arxiv_id,
        title=str(row.get("title") or "Unknown title"),
        abstract=str(row.get("abstract") or ""),
        source_url=str(row.get("source_url") or f"https://arxiv.org/abs/{arxiv_id}"),
        full_text_url=full_text_url,
        full_text=str(full_text)[:1_000_000],
        full_text_state=str(state),
        citation_depth=depth,
        discovered_from=parents,
        local_source_path=str(local_path) if local_path is not None else None,
        local_source_sha256=local_sha256,
    )
    missing = None
    if not full_text:
        reasons = []
        if metadata_state == "unavailable":
            reasons.append("metadata_api_unavailable")
        reasons.append(local_error or "full_text_network_unavailable")
        missing = _missing_source_record(
            arxiv_id=arxiv_id,
            title=paper.title,
            reasons=reasons,
            local_sources_root=local_sources_root,
        )
    return paper, tuple(str(value) for value in cited_ids), missing


def _fetch_ar5iv_text(
    arxiv_id: str,
    *,
    timeout_seconds: float,
) -> tuple[str, tuple[str, ...], str]:
    url = f"https://ar5iv.labs.arxiv.org/html/{_base_arxiv_id(arxiv_id)}"
    request = urllib.request.Request(
        url, headers={"User-Agent": "verdiwm/0.1 mechanism-discovery"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            payload = response.read(4_000_001)
    except (OSError, urllib.error.URLError) as exc:
        raise MechanismDiscoveryError("MECHANISM_FULL_TEXT_UNAVAILABLE") from exc
    if len(payload) > 4_000_000:
        raise MechanismDiscoveryError("MECHANISM_FULL_TEXT_TOO_LARGE")
    parser = _BoundedHTMLTextParser()
    parser.feed(payload.decode("utf-8", errors="replace"))
    text = parser.text()
    cited = tuple(dict.fromkeys(_base_arxiv_id(value) for value in parser.arxiv_ids()))
    return text, cited, "network"


def _fetch_arxiv_metadata(
    arxiv_id: str, *, timeout_seconds: float
) -> Mapping[str, str]:
    url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
        {"id_list": _base_arxiv_id(arxiv_id), "start": 0, "max_results": 1}
    )
    request = urllib.request.Request(
        url, headers={"User-Agent": "verdiwm/0.1 mechanism-discovery"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            payload = response.read(2_000_001)
        if len(payload) > 2_000_000:
            raise MechanismDiscoveryError("MECHANISM_METADATA_TOO_LARGE")
        root = ElementTree.fromstring(payload)
    except (OSError, urllib.error.URLError, ElementTree.ParseError) as exc:
        raise MechanismDiscoveryError("MECHANISM_METADATA_UNAVAILABLE") from exc
    namespace = {"atom": "http://www.w3.org/2005/Atom"}
    entry = root.find("atom:entry", namespace)
    if entry is None:
        raise MechanismDiscoveryError("MECHANISM_METADATA_UNAVAILABLE")
    identifier = _node_text(entry.find("atom:id", namespace))
    match = _ARXIV_ID.search(identifier)
    return {
        "arxiv_id": match.group(0) if match else arxiv_id,
        "title": " ".join(_node_text(entry.find("atom:title", namespace)).split()),
        "abstract": " ".join(_node_text(entry.find("atom:summary", namespace)).split()),
        "source_url": identifier,
    }


def _load_local_source(
    arxiv_id: str,
    root: Path | None,
) -> tuple[str, tuple[str, ...], str, Path, str | None] | None:
    if root is None:
        return None
    candidates: list[Path] = []
    for stem in (_canonical_arxiv_id(arxiv_id), _base_arxiv_id(arxiv_id)):
        for suffix in (".txt", ".html", ".htm", ".pdf"):
            path = root / f"{stem}{suffix}"
            if path.is_file() and not path.is_symlink():
                candidates.append(path)
    if not candidates:
        return None
    path = candidates[0].resolve()
    if root not in path.parents:
        raise MechanismDiscoveryError("MECHANISM_LOCAL_SOURCE_OUTSIDE_ROOT")
    if path.stat().st_size > 50_000_000:
        raise MechanismDiscoveryError("MECHANISM_LOCAL_SOURCE_TOO_LARGE")
    suffix = path.suffix.lower()
    if suffix == ".txt":
        text = path.read_text(encoding="utf-8", errors="replace")[:1_000_000]
        return text, _arxiv_ids_from_text(text), "local", path, None
    if suffix in {".html", ".htm"}:
        parser = _BoundedHTMLTextParser()
        parser.feed(path.read_text(encoding="utf-8", errors="replace"))
        return parser.text(), parser.arxiv_ids(), "local", path, None
    converter = shutil.which("pdftotext")
    if converter is None:
        return "", (), "unavailable", path, "local_pdf_parser_unavailable"
    try:
        completed = subprocess.run(
            [converter, str(path), "-"],
            check=True,
            capture_output=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return "", (), "unavailable", path, "local_pdf_parse_failed"
    text = completed.stdout.decode("utf-8", errors="replace")[:1_000_000]
    return text, _arxiv_ids_from_text(text), "local", path, None


def _local_source_inventory(root: Path | None) -> list[dict[str, object]]:
    if root is None:
        return []
    records: list[dict[str, object]] = []
    for path in sorted(root.iterdir()):
        if (
            path.is_symlink()
            or not path.is_file()
            or path.suffix.lower() not in {".txt", ".html", ".htm", ".pdf"}
        ):
            continue
        records.append(
            {
                "path": str(path.resolve()),
                "sha256": _sha256(path.read_bytes()),
                "size_bytes": path.stat().st_size,
            }
        )
    return records


def _missing_source_record(
    *,
    arxiv_id: str,
    title: str,
    reasons: Sequence[str],
    local_sources_root: Path | None,
) -> dict[str, object]:
    revision = _canonical_arxiv_id(arxiv_id)
    return {
        "arxiv_id": revision,
        "title": title,
        "reasons": list(dict.fromkeys(str(value) for value in reasons if value)),
        "pdf_url": f"https://arxiv.org/pdf/{revision}",
        "html_url": f"https://ar5iv.labs.arxiv.org/html/{_base_arxiv_id(revision)}",
        "accepted_filenames": [
            f"{revision}.txt",
            f"{revision}.html",
            f"{revision}.pdf",
            f"{_base_arxiv_id(revision)}.txt",
            f"{_base_arxiv_id(revision)}.html",
            f"{_base_arxiv_id(revision)}.pdf",
        ],
        "destination_root": str(local_sources_root)
        if local_sources_root is not None
        else None,
        "integrity_rule": "The rerun records SHA256 and size for every local source file.",
    }


def _merge_missing_source(
    records: list[dict[str, object]],
    candidate: Mapping[str, object],
) -> None:
    arxiv_id = str(candidate.get("arxiv_id", ""))
    for record in records:
        if str(record.get("arxiv_id", "")) != arxiv_id:
            continue
        record["reasons"] = list(
            dict.fromkeys(
                [
                    *(str(value) for value in record.get("reasons", [])),
                    *(str(value) for value in candidate.get("reasons", [])),
                ]
            )
        )
        return
    records.append(dict(candidate))


def _render_missing_sources_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Missing Literature Sources",
        "",
        f"Local source directory: `{report.get('local_sources_root') or 'not configured'}`",
        "",
        "Download one TXT, HTML, or PDF file per record using an accepted filename, then rerun into a new output directory.",
        "",
    ]
    records = report.get("records")
    if (
        not isinstance(records, Sequence)
        or isinstance(records, (str, bytes))
        or not records
    ):
        lines.append("No missing sources.")
        return "\n".join(lines) + "\n"
    for row in records:
        if not isinstance(row, Mapping):
            continue
        lines.extend(
            [
                f"## {row.get('arxiv_id')} - {row.get('title')}",
                "",
                f"Reason: `{', '.join(str(value) for value in row.get('reasons', []))}`",
                "",
                f"PDF: {row.get('pdf_url')}",
                "",
                f"HTML: {row.get('html_url')}",
                "",
                "Accepted filenames: "
                + ", ".join(
                    f"`{value}`" for value in row.get("accepted_filenames", [])
                ),
                "",
            ]
        )
    return "\n".join(lines)


def _arxiv_ids_from_text(value: str) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            _base_arxiv_id(match.group(0)) for match in _ARXIV_ID.finditer(value)
        )
    )


class _BoundedHTMLTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._text: list[str] = []
        self._hrefs: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "svg", "noscript"}:
            self._ignored_depth += 1
        if tag == "a":
            for key, value in attrs:
                if key == "href" and value:
                    self._hrefs.append(value)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "svg", "noscript"} and self._ignored_depth:
            self._ignored_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth and data.strip():
            self._text.append(data.strip())

    def text(self) -> str:
        return " ".join(" ".join(self._text).split())[:1_000_000]

    def arxiv_ids(self) -> tuple[str, ...]:
        values: list[str] = []
        for value in self._hrefs:
            if "arxiv" not in value.lower():
                continue
            values.extend(match.group(0) for match in _ARXIV_ID.finditer(value))
        return tuple(values)
