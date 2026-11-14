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
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from wmloop.contracts import ContractValidationError, validate_document
from wmloop.propose.prior_library import PriorLibraryError, stage_literature_candidate
from wmloop.retrieve.query_terms import arxiv_clause, query_terms
from wmloop.storage import atomic_write, canonical_bytes, canonical_json_bytes, checked_path, exclusive_file_lock, sha256_bytes


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

    records, source_state, _ = _search_arxiv(
        query, max_results=max_results, timeout_seconds=timeout_seconds, cache_path=cache_path
    )
    return records, source_state


def _search_arxiv(
    query: str,
    *,
    max_results: int,
    timeout_seconds: float,
    cache_path: Path | None,
) -> tuple[tuple[LiteratureRecord, ...], str, dict[str, object]]:
    if not isinstance(query, str) or not query.strip() or len(query) > 4096 or type(max_results) is not int or not 1 <= max_results <= 50:
        raise LiteratureRetrievalError("LITERATURE_QUERY_INVALID")
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)) or not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 60:
        raise LiteratureRetrievalError("LITERATURE_TIMEOUT_INVALID")
    cache = checked_path(cache_path, code="LITERATURE_CACHE_INVALID", error=LiteratureRetrievalError) if cache_path is not None else None
    expression = query.strip() if re.search(r"\b(?:all|ti|au|abs|cat|id):", query) else arxiv_clause(query_terms(query, maximum=16, intent=True), operator="AND")
    if not expression:
        raise LiteratureRetrievalError("LITERATURE_QUERY_NO_SEARCHABLE_TERMS")
    url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
        {"search_query": expression, "start": 0, "max_results": max_results, "sortBy": "relevance"}
    )
    attempt: dict[str, object] = {
        "query": query, "search_expression": expression, "url": url,
        "fresh_request": True, "source_state": "offline", "error_code": None,
        "http_status": None, "response_bytes": None, "response_sha256": None,
        "record_count": 0,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
    }
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "verdiwm/0.1 literature-retrieval"})
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            attempt["http_status"] = response.getcode()
            payload = response.read(2_000_000 + 1)
        attempt["response_bytes"] = len(payload)
        attempt["response_sha256"] = sha256_bytes(payload)
        if len(payload) > 2_000_000:
            raise LiteratureRetrievalError("LITERATURE_RESPONSE_TOO_LARGE")
        records = _parse_atom(payload)
    except urllib.error.HTTPError as exc:
        attempt.update(error_code=f"LITERATURE_HTTP_{exc.code}", http_status=exc.code)
    except (TimeoutError, urllib.error.URLError, OSError) as exc:
        reason = exc.reason if isinstance(exc, urllib.error.URLError) else exc
        attempt["error_code"] = "LITERATURE_TIMEOUT" if isinstance(reason, TimeoutError) else "LITERATURE_NETWORK_ERROR"
    except ElementTree.ParseError:
        attempt["error_code"] = "LITERATURE_ATOM_INVALID"
    except LiteratureRetrievalError as exc:
        attempt["error_code"] = str(exc)
    else:
        # Persistence is outside the recoverable network boundary: a broken
        # output filesystem is not an offline literature provider.
        if cache is not None:
            _write_json(cache, {"query": query.strip(), "records": [row.to_dict() for row in records]})
        attempt.update(source_state="network", record_count=len(records), completed_at=datetime.now(timezone.utc).isoformat())
        return records, "network", attempt
    cached = _load_cache(cache, query=query.strip())
    attempt["completed_at"] = datetime.now(timezone.utc).isoformat()
    if cached is not None:
        attempt.update(source_state="cached", record_count=len(cached))
        return cached, "cached", attempt
    return (), "offline", attempt


def stage_literature_results(
    records: Sequence[LiteratureRecord],
    *,
    staging_root: Path,
    query: str,
    reuse_identical: bool = False,
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
            path = stage_literature_candidate(candidate, staging_root=Path(staging_root), reuse_identical=reuse_identical)
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
    query: str | None,
    output_root: Path,
    max_results: int = 8,
    timeout_seconds: float = 10.0,
    query_plan: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Run or resume one bounded literature retrieval transaction."""

    destination = checked_path(output_root, code="LITERATURE_OUTPUT_INVALID", error=LiteratureRetrievalError)
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with exclusive_file_lock(destination.parent / f".{destination.name}.literature.lock"):
        return _run_literature_retrieval_locked(
            query=query, destination=destination, max_results=max_results,
            timeout_seconds=timeout_seconds, query_plan=query_plan,
        )


def _run_literature_retrieval_locked(
    *, query: str | None, destination: Path, max_results: int,
    timeout_seconds: float, query_plan: Mapping[str, object] | None,
) -> dict[str, object]:
    if query_plan is not None:
        validate_document("literature_query_plan", query_plan)
        if query_plan.get("query") != query:
            raise LiteratureRetrievalError("LITERATURE_QUERY_PLAN_MISMATCH")
        if query_plan["state"] == "planned" and not query_plan["queries"]:
            raise LiteratureRetrievalError("LITERATURE_QUERY_PLAN_EMPTY")
        if query_plan["state"] == "skipped" and (query is not None or query_plan["queries"]):
            raise LiteratureRetrievalError("LITERATURE_QUERY_PLAN_MISMATCH")
    if query is None and (query_plan is None or query_plan.get("state") != "skipped"):
        raise LiteratureRetrievalError("LITERATURE_QUERY_INVALID")
    inputs: dict[str, object] = {"query": query, "max_results": max_results, "timeout_seconds": timeout_seconds}
    if query_plan is not None:
        inputs["query_plan_sha256"] = sha256_bytes(canonical_bytes(query_plan))
    input_hash = hashlib.sha256(
        json.dumps(
            inputs,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    binding = {"schema_version": 1, "artifact_type": "verdiwm-literature-input-binding",
               "input_hash": input_hash, "inputs": inputs}
    try:
        validate_document("literature_input_binding", binding)
    except ContractValidationError as exc:
        raise LiteratureRetrievalError("LITERATURE_INPUT_INVALID") from exc
    manifest_path = destination / "manifest.json"
    binding_path = destination / "input-binding.json"
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or not destination.is_dir():
            raise LiteratureRetrievalError("LITERATURE_OUTPUT_INVALID")
        if manifest_path.is_file() and not manifest_path.is_symlink():
            existing = _load_mapping(manifest_path)
            if existing.get("input_hash") != input_hash:
                raise LiteratureRetrievalError("LITERATURE_INPUT_MISMATCH")
            # Historical evidence is immutable. A resume returns a read-only
            # delivery projection rather than rewriting its original trace.
            return {**existing, "delivery": {"mode": "resume", "fresh_request": False}}
        if binding_path.exists():
            existing_binding = _load_checked_artifact(binding_path, "literature_input_binding")
            if existing_binding != binding:
                raise LiteratureRetrievalError("LITERATURE_INPUT_MISMATCH")
        elif any(destination.iterdir()):
            raise LiteratureRetrievalError("LITERATURE_OUTPUT_UNBOUND")
    if not binding_path.exists():
        _write_json(binding_path, binding)
    if query is None:
        manifest = {
            "schema_version": 1, "artifact_type": "verdiwm-literature-retrieval-manifest",
            "state": "skipped", "input_hash": input_hash, "query": None,
            "record_count": 0, "staged_count": 0, "rows": [],
            "reason": query_plan["reason"], "suggested_action": query_plan["suggested_action"],
            "delivery": {"mode": "fresh", "fresh_request": False},
            "claim_boundary": query_plan["claim_boundary"],
        }
        _write_json(manifest_path, manifest)
        return manifest
    checkpoint_path = destination / "search-result.json"
    recovered = checkpoint_path.exists()
    if recovered:
        checkpoint = _load_checked_artifact(checkpoint_path, "literature_search_result")
        if checkpoint["input_hash"] != input_hash:
            raise LiteratureRetrievalError("LITERATURE_INPUT_MISMATCH")
        trace = checkpoint["search_trace"]
        validate_document("literature_search_trace", trace)
        records = tuple(LiteratureRecord(**row) for row in checkpoint["records"])
        if trace["attempts"][-1]["record_count"] != len(records) or trace["query_plan_sha256"] != inputs.get("query_plan_sha256"):
            raise LiteratureRetrievalError("LITERATURE_SEARCH_RESULT_INVALID")
        source_state = trace["source_state"]
    else:
        searches = query_plan["queries"] if query_plan is not None else [{"view": "explicit", "query": query}]
        attempts = []
        for index, search in enumerate(searches):
            records, source_state, attempt = _search_arxiv(
                search["query"], max_results=max_results, timeout_seconds=timeout_seconds,
                cache_path=destination / ("cache.json" if index == 0 else f"cache-{index + 1}.json"),
            )
            attempts.append({"view": search["view"], **attempt})
            if records or source_state != "network":
                break
        trace = {
            "schema_version": 1, "artifact_type": "verdiwm-literature-search-trace",
            "authority": "ranking_only", "fresh_request": True,
            "source_state": source_state, "effective_query": attempts[-1]["query"],
            "query_plan_sha256": inputs.get("query_plan_sha256"), "attempts": attempts,
            "claim_boundary": "Provider responses and relevance are literature candidates only; network access, cached records and diagnostic outcomes do not confirm target model improvement.",
        }
        validate_document("literature_search_trace", trace)
        checkpoint = {"schema_version": 1, "artifact_type": "verdiwm-literature-search-result",
                      "input_hash": input_hash, "records": [row.to_dict() for row in records], "search_trace": trace}
        validate_document("literature_search_result", checkpoint)
        _write_json(checkpoint_path, checkpoint)
    trace_path = destination / "search-trace.json"
    _write_json(trace_path, trace)
    effective_query = str(trace["effective_query"])
    staged = stage_literature_results(records, staging_root=destination / "candidates", query=effective_query, reuse_identical=True)
    _write_json(destination / "records.json", {"query": effective_query, "records": [item.to_dict() for item in records]})
    manifest = {
        "schema_version": 1,
        "artifact_type": "verdiwm-literature-retrieval-manifest",
        "state": source_state,
        "input_hash": input_hash,
        "query": query,
        "effective_query": effective_query,
        "record_count": len(records),
        "staged_count": sum(row.get("state") == "staged" for row in staged),
        "records_path": str(destination / "records.json"),
        "staging_root": str(destination / "candidates"),
        "rows": list(staged),
        "search_trace_path": str(trace_path),
        "search_trace": trace,
        "delivery": {"mode": "recovery" if recovered else "fresh", "fresh_request": not recovered},
        "claim_boundary": "Literature records are untrusted, data-only cold-start candidates. They require a typed executable contract and local screen before any experiment can run.",
    }
    _write_json(manifest_path, manifest)
    return manifest


def _parse_atom(payload: bytes) -> tuple[LiteratureRecord, ...]:
    root = ElementTree.fromstring(payload)
    if root.tag != "{http://www.w3.org/2005/Atom}feed":
        raise LiteratureRetrievalError("LITERATURE_ATOM_FEED_INVALID")
    namespace = {"atom": "http://www.w3.org/2005/Atom"}
    records: list[LiteratureRecord] = []
    for entry in root.findall("atom:entry", namespace):
        identifier = _text(entry.find("atom:id", namespace))
        if "arxiv.org/api/errors" in identifier:
            raise LiteratureRetrievalError("LITERATURE_API_ERROR")
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


def _load_cache(path: Path | None, *, query: str) -> tuple[LiteratureRecord, ...] | None:
    if path is None or not path.is_file() or path.is_symlink():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("query") != query or not isinstance(payload.get("records"), list):
            return None
        if len(payload["records"]) > 50 or any(
            not isinstance(row, dict) or not all(isinstance(row.get(key), str) and row[key] for key in ("arxiv_id", "title", "abstract"))
            or not all(isinstance(row.get(key, ""), str) for key in ("pdf_url", "published"))
            for row in payload["records"]
        ):
            return None
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
        return None


def _load_mapping(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LiteratureRetrievalError("LITERATURE_MANIFEST_INVALID") from exc
    if not isinstance(value, dict):
        raise LiteratureRetrievalError("LITERATURE_MANIFEST_INVALID")
    return value


def _load_checked_artifact(path: Path, schema_name: str) -> dict[str, object]:
    checked_path(path, code="LITERATURE_OUTPUT_INVALID", error=LiteratureRetrievalError)
    value = _load_mapping(path)
    try:
        validate_document(schema_name, value)
    except ContractValidationError as exc:
        raise LiteratureRetrievalError("LITERATURE_RECOVERY_ARTIFACT_INVALID") from exc
    return value


def _write_json(path: Path, payload: object) -> None:
    atomic_write(checked_path(path, code="LITERATURE_OUTPUT_INVALID", error=LiteratureRetrievalError), canonical_json_bytes(payload))


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
