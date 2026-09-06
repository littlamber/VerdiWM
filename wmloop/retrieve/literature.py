"""Bounded, read-only literature retrieval for cold-start campaigns.

Network results are treated as untrusted data.  They are cached and staged as
paper candidates only; no title, abstract, PDF, or repository is imported or
executed by this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from wmloop.propose.prior_library import PriorLibraryError, stage_literature_candidate
from wmloop.retrieve.providers import ResearchProviderError, search_provider


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
    try:
        external = search_provider(
            "arxiv",
            query,
            limit=max_results,
            timeout_seconds=timeout_seconds,
            byte_limit=2_000_000,
        )
        records = tuple(
            LiteratureRecord(
                arxiv_id=row.record_id,
                title=row.title,
                abstract=row.summary,
                pdf_url=row.source_url,
                published=row.published,
            )
            for row in external
        )
        if cache is not None:
            _write_json(cache, {"query": query.strip(), "records": [row.to_dict() for row in records]})
        return records, "network"
    except (OSError, ResearchProviderError, LiteratureRetrievalError):
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

    return run_literature_retrieval_batch(
        queries=(query,),
        output_root=output_root,
        max_results=max_results,
        timeout_seconds=timeout_seconds,
    )


def run_literature_retrieval_batch(
    *,
    queries: Sequence[str],
    output_root: Path,
    max_results: int = 8,
    timeout_seconds: float = 10.0,
) -> dict[str, object]:
    """Run or resume bounded multi-view retrieval with cross-query deduplication."""

    destination = Path(output_root).resolve()
    normalized_queries = tuple(
        dict.fromkeys(str(query).strip() for query in queries if str(query).strip())
    )
    if not normalized_queries or max_results < 1 or max_results > 50:
        raise LiteratureRetrievalError("LITERATURE_QUERY_INVALID")
    if timeout_seconds <= 0 or timeout_seconds > 60:
        raise LiteratureRetrievalError("LITERATURE_TIMEOUT_INVALID")
    input_contract: dict[str, object] = (
        {
            "query": normalized_queries[0],
            "max_results": max_results,
            "timeout_seconds": timeout_seconds,
        }
        if len(normalized_queries) == 1
        else {
            "queries": normalized_queries,
            "max_results": max_results,
            "timeout_seconds": timeout_seconds,
        }
    )
    input_hash = hashlib.sha256(
        json.dumps(
            input_contract,
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
    records_by_id: dict[str, LiteratureRecord] = {}
    query_states: list[dict[str, object]] = []
    selected_queries = normalized_queries[:max_results]
    per_query_limit = max(1, math.ceil(max_results / len(selected_queries)))
    for index, query in enumerate(selected_queries, start=1):
        records, source_state = search_arxiv(
            query,
            max_results=per_query_limit,
            timeout_seconds=timeout_seconds,
            cache_path=destination / "cache" / f"query-{index:03d}.json",
        )
        for record in records:
            records_by_id.setdefault(record.arxiv_id, record)
        query_states.append(
            {
                "query": query,
                "state": source_state,
                "record_count": len(records),
            }
        )
    records = tuple(records_by_id.values())[:max_results]
    source_state = _aggregate_source_state(query_states)
    query_label = " | ".join(normalized_queries)
    staged = stage_literature_results(
        records,
        staging_root=destination / "candidates",
        query=query_label,
    )
    _write_json(
        destination / "records.json",
        {
            "queries": list(normalized_queries),
            "queried": list(selected_queries),
            "queries_truncated": len(selected_queries) < len(normalized_queries),
            "query_states": query_states,
            "records": [item.to_dict() for item in records],
        },
    )
    manifest = {
        "schema_version": 1,
        "artifact_type": "verdiwm-literature-retrieval-manifest",
        "state": source_state,
        "input_hash": input_hash,
        "query": normalized_queries[0],
        "queries": list(normalized_queries),
        "queried": list(selected_queries),
        "queries_truncated": len(selected_queries) < len(normalized_queries),
        "query_states": query_states,
        "record_count": len(records),
        "staged_count": sum(row.get("state") == "staged" for row in staged),
        "records_path": str(destination / "records.json"),
        "staging_root": str(destination / "candidates"),
        "rows": list(staged),
        "claim_boundary": "Literature records are untrusted, data-only cold-start candidates. They require a typed executable contract and local screen before any experiment can run.",
    }
    _write_json(manifest_path, manifest)
    return manifest


def _aggregate_source_state(rows: Sequence[dict[str, object]]) -> str:
    states = {str(row.get("state")) for row in rows}
    if "network" in states:
        return "network"
    if "cached" in states:
        return "cached"
    return "offline"


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
