"""Bounded, read-only literature retrieval for cold-start campaigns.

Network results are treated as untrusted data.  They are cached and staged as
paper candidates only; no title, abstract, PDF, or repository is imported or
executed by this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from wmloop.propose.prior_library import PriorLibraryError, stage_literature_candidate


class LiteratureRetrievalError(RuntimeError):
    """A bounded literature lookup or staging operation failed."""


@dataclass(frozen=True)
class LiteratureRecord:
    """A minimal, source-linked paper record suitable for staging."""

    arxiv_id: str
    title: str
    abstract: str
    pdf_url: str
    published: str

    def to_dict(self) -> dict[str, str]:
        return {
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "abstract": self.abstract,
            "pdf_url": self.pdf_url,
            "published": self.published,
        }


def search_arxiv(
    query: str,
    *,
    max_results: int = 8,
    timeout_seconds: float = 10.0,
    cache_path: Path | None = None,
) -> tuple[tuple[LiteratureRecord, ...], str]:
    """Search arXiv with strict bounds, returning ``(records, source_state)``."""

    if not query.strip() or max_results < 1 or max_results > 50:
        raise LiteratureRetrievalError("LITERATURE_QUERY_INVALID")
    if timeout_seconds <= 0 or timeout_seconds > 60:
        raise LiteratureRetrievalError("LITERATURE_TIMEOUT_INVALID")
    cache = Path(cache_path).resolve() if cache_path is not None else None
    url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
        {"search_query": f"all:{query.strip()}", "start": 0, "max_results": max_results, "sortBy": "relevance"}
    )
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "verdiwm/0.1 literature-retrieval"})
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            payload = response.read(2_000_000 + 1)
        if len(payload) > 2_000_000:
            raise LiteratureRetrievalError("LITERATURE_RESPONSE_TOO_LARGE")
        records = _parse_atom(payload)
        if cache is not None:
            _write_json(cache, {"query": query.strip(), "records": [row.to_dict() for row in records]})
        return records, "network"
    except (OSError, ElementTree.ParseError, LiteratureRetrievalError):
        cached = _load_cache(cache, query=query.strip())
        if cached:
            return cached, "cached"
        return (), "offline"


def stage_literature_results(
    records: Sequence[LiteratureRecord],
    *,
    staging_root: Path,
    query: str,
) -> tuple[dict[str, object], ...]:
    """Persist safe data-only paper candidates for later promotion."""

    rows: list[dict[str, object]] = []
    for record in records:
        candidate_id = "lit-" + _safe_id(record.arxiv_id)
        candidate = {
            "candidate_id": candidate_id,
            "arxiv_id": record.arxiv_id,
            "title": record.title,
            "mechanism_summary": _summary(record.abstract),
            "proposed_manifest": {
                "state": "staged",
                "source": "arxiv",
                "source_url": record.pdf_url,
                "query": query,
                "execution_authority": "shadow_only",
            },
        }
        try:
            path = stage_literature_candidate(candidate, staging_root=Path(staging_root))
        except PriorLibraryError as exc:
            rows.append({"candidate_id": candidate_id, "state": "blocked", "reason": str(exc)})
            continue
        rows.append(
            {
                "candidate_id": candidate_id,
                "arxiv_id": record.arxiv_id,
                "title": record.title,
                "state": "staged",
                "path": str(path),
                "execution_authority": "shadow_only",
            }
        )
    return tuple(rows)


def run_literature_retrieval(
    *,
    query: str,
    output_root: Path,
    max_results: int = 8,
    timeout_seconds: float = 10.0,
) -> dict[str, object]:
    """Run or resume one bounded literature retrieval transaction."""

    destination = Path(output_root).resolve()
    input_hash = hashlib.sha256(
        json.dumps(
            {"query": query, "max_results": max_results, "timeout_seconds": timeout_seconds},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    manifest_path = destination / "manifest.json"
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or not destination.is_dir():
            raise LiteratureRetrievalError("LITERATURE_OUTPUT_INVALID")
        if manifest_path.is_file() and not manifest_path.is_symlink():
            existing = _load_mapping(manifest_path)
            if existing.get("input_hash") != input_hash:
                raise LiteratureRetrievalError("LITERATURE_INPUT_MISMATCH")
            return existing
        if any(destination.iterdir()):
            raise LiteratureRetrievalError("LITERATURE_OUTPUT_UNBOUND")
    records, source_state = search_arxiv(
        query,
        max_results=max_results,
        timeout_seconds=timeout_seconds,
        cache_path=destination / "cache.json",
    )
    staged = stage_literature_results(records, staging_root=destination / "candidates", query=query)
    _write_json(destination / "records.json", {"query": query, "records": [item.to_dict() for item in records]})
    manifest = {
        "schema_version": 1,
        "artifact_type": "verdiwm-literature-retrieval-manifest",
        "state": source_state,
        "input_hash": input_hash,
        "query": query,
        "record_count": len(records),
        "staged_count": sum(row.get("state") == "staged" for row in staged),
        "records_path": str(destination / "records.json"),
        "staging_root": str(destination / "candidates"),
        "rows": list(staged),
        "claim_boundary": "Literature records are untrusted, data-only cold-start candidates. They require a typed executable contract and local screen before any experiment can run.",
    }
    _write_json(manifest_path, manifest)
    return manifest


def _parse_atom(payload: bytes) -> tuple[LiteratureRecord, ...]:
    root = ElementTree.fromstring(payload)
    namespace = {"atom": "http://www.w3.org/2005/Atom"}
    records: list[LiteratureRecord] = []
    for entry in root.findall("atom:entry", namespace):
        identifier = _text(entry.find("atom:id", namespace))
        match = re.search(r"arxiv.org/abs/([^/?]+)", identifier)
        arxiv_id = match.group(1) if match else identifier.rsplit("/", 1)[-1]
        title = " ".join(_text(entry.find("atom:title", namespace)).split())
        abstract = " ".join(_text(entry.find("atom:summary", namespace)).split())
        published = _text(entry.find("atom:published", namespace))
        pdf_url = ""
        for link in entry.findall("atom:link", namespace):
            if link.attrib.get("title") == "pdf":
                pdf_url = str(link.attrib.get("href") or "")
                break
        if arxiv_id and title and abstract:
            records.append(LiteratureRecord(arxiv_id, title, abstract, pdf_url, published))
    return tuple(records)


def _load_cache(path: Path | None, *, query: str) -> tuple[LiteratureRecord, ...]:
    if path is None or not path.is_file() or path.is_symlink():
        return ()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("query") != query or not isinstance(payload.get("records"), list):
            return ()
        return tuple(
            LiteratureRecord(
                arxiv_id=str(row["arxiv_id"]),
                title=str(row["title"]),
                abstract=str(row["abstract"]),
                pdf_url=str(row.get("pdf_url") or ""),
                published=str(row.get("published") or ""),
            )
            for row in payload["records"]
            if isinstance(row, dict) and row.get("arxiv_id") and row.get("title") and row.get("abstract")
        )
    except (OSError, TypeError, KeyError, json.JSONDecodeError):
        return ()


def _load_mapping(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LiteratureRetrievalError("LITERATURE_MANIFEST_INVALID") from exc
    if not isinstance(value, dict):
        raise LiteratureRetrievalError("LITERATURE_MANIFEST_INVALID")
    return value


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n", encoding="utf-8")


def _text(node: ElementTree.Element | None) -> str:
    return (node.text or "").strip() if node is not None else ""


def _summary(value: str) -> str:
    return " ".join(value.split())[:4000] or "No abstract supplied."


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")[:100] or "unknown"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-results", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    args = parser.parse_args(argv)
    try:
        manifest = run_literature_retrieval(
            query=args.query,
            output_root=args.output_root,
            max_results=args.max_results,
            timeout_seconds=args.timeout_seconds,
        )
    except LiteratureRetrievalError as exc:
        print(str(exc), file=__import__("sys").stderr)
        return 2
    print(json.dumps(manifest, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
