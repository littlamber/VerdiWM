"""Safe arXiv metadata monitoring for the frozen-prior staging boundary.

The monitor is deliberately a data-ingestion utility, not an autonomous code
path.  It fetches Atom metadata, records only papers inside a fixed lookback
window, and may stage a separately supplied candidate only when its arXiv ID
was observed in that scan.  No paper text is imported, executed, or promoted
to the frozen primitive registry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
from urllib.parse import urlencode
from urllib.request import urlopen
from xml.etree import ElementTree

from wmloop.propose.prior_library import PriorLibraryError, stage_literature_candidate


class LiteratureMonitorError(ValueError):
    """A source, time window, or staged-candidate invariant failed."""


_ATOM = "{http://www.w3.org/2005/Atom}"
_ARXIV_ID_FROM_URL = re.compile(r"/(?:abs|pdf)/((?:arXiv:)?\d{4}\.\d{4,5}(?:v\d+)?)$")
_SAFE_RECORD_STEM = re.compile(r"[^A-Za-z0-9._-]+")
_DEFAULT_API = "https://export.arxiv.org/api/query"

FetchBytes = Callable[[str], bytes]


@dataclass(frozen=True)
class LiteratureRecord:
    """Normalized untrusted metadata from one arXiv Atom entry."""

    arxiv_id: str
    title: str
    abstract: str
    published_at: datetime
    source_url: str
    source_digest: str

    def to_document(self) -> dict[str, str]:
        return {
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "abstract": self.abstract,
            "published_at": self.published_at.isoformat(),
            "source_url": self.source_url,
            "source_digest": self.source_digest,
        }


@dataclass(frozen=True)
class LiteratureMonitorReport:
    query_terms: tuple[str, ...]
    cutoff: datetime
    fetched_records: int
    retained_records: tuple[LiteratureRecord, ...]
    record_paths: tuple[Path, ...]
    staged_candidate_paths: tuple[Path, ...]

    def to_document(self) -> dict[str, Any]:
        return {
            "query_terms": list(self.query_terms),
            "cutoff": self.cutoff.isoformat(),
            "fetched_records": self.fetched_records,
            "retained_arxiv_ids": [record.arxiv_id for record in self.retained_records],
            "record_paths": [str(path) for path in self.record_paths],
            "staged_candidate_paths": [str(path) for path in self.staged_candidate_paths],
        }


def run_literature_monitor(
    *,
    query_terms: Sequence[str],
    staging_root: Path,
    now: datetime | None = None,
    lookback_days: int = 7,
    max_results_per_query: int = 50,
    candidate_documents: Iterable[Mapping[str, Any]] = (),
    fetch_bytes: FetchBytes | None = None,
) -> LiteratureMonitorReport:
    """Collect recent metadata and stage only source-verified candidates.

    ``candidate_documents`` are intentionally supplied out of band (for
    example by a constrained distiller).  The monitor does not generate code
    or candidate mechanisms from untrusted paper text.
    """

    terms = _normalize_query_terms(query_terms)
    if lookback_days < 1 or max_results_per_query < 1:
        raise LiteratureMonitorError("LIT_MONITOR_CONFIGURATION_INVALID")
    instant = _coerce_utc(now or datetime.now(timezone.utc))
    cutoff = instant - timedelta(days=lookback_days)
    fetch = fetch_bytes or _fetch_url

    fetched: list[LiteratureRecord] = []
    for term in terms:
        url = _query_url(term, max_results=max_results_per_query)
        fetched.extend(_parse_atom_records(fetch(url)))
    retained = _deduplicate_recent(fetched, cutoff=cutoff)
    record_paths = tuple(_persist_record(record, Path(staging_root)) for record in retained)
    observed_ids = {record.arxiv_id for record in retained}

    staged: list[Path] = []
    for candidate in candidate_documents:
        arxiv_id = _canonical_arxiv_id(candidate.get("arxiv_id"))
        if arxiv_id not in observed_ids:
            raise LiteratureMonitorError("LIT_MONITOR_ARXIV_UNVERIFIED")
        try:
            staged.append(stage_literature_candidate(candidate, staging_root=Path(staging_root) / "candidates"))
        except PriorLibraryError as exc:
            raise LiteratureMonitorError(str(exc)) from exc
    return LiteratureMonitorReport(
        query_terms=terms,
        cutoff=cutoff,
        fetched_records=len(fetched),
        retained_records=retained,
        record_paths=record_paths,
        staged_candidate_paths=tuple(staged),
    )


def _normalize_query_terms(query_terms: Sequence[str]) -> tuple[str, ...]:
    terms = tuple(sorted({term.strip() for term in query_terms if isinstance(term, str) and term.strip()}))
    if not terms:
        raise LiteratureMonitorError("LIT_MONITOR_QUERY_EMPTY")
    return terms


def _query_url(term: str, *, max_results: int) -> str:
    return _DEFAULT_API + "?" + urlencode(
        {"search_query": f'all:"{term}"', "start": 0, "max_results": max_results, "sortBy": "submittedDate", "sortOrder": "descending"}
    )


def _fetch_url(url: str) -> bytes:
    try:
        with urlopen(url, timeout=30) as response:  # nosec B310 -- fixed HTTPS endpoint
            return response.read()
    except OSError as exc:
        raise LiteratureMonitorError("LIT_MONITOR_FETCH_FAILED") from exc


def _parse_atom_records(payload: bytes) -> tuple[LiteratureRecord, ...]:
    try:
        root = ElementTree.fromstring(payload)
    except ElementTree.ParseError as exc:
        raise LiteratureMonitorError("LIT_MONITOR_ATOM_INVALID") from exc
    records: list[LiteratureRecord] = []
    for entry in root.findall(f"{_ATOM}entry"):
        raw_url = _required_text(entry, f"{_ATOM}id")
        arxiv_id = _arxiv_id_from_url(raw_url)
        title = _required_text(entry, f"{_ATOM}title")
        abstract = _required_text(entry, f"{_ATOM}summary")
        published = _parse_timestamp(_required_text(entry, f"{_ATOM}published"))
        digest = hashlib.sha256(
            json.dumps(
                {"arxiv_id": arxiv_id, "title": title, "abstract": abstract, "published_at": published.isoformat(), "source_url": raw_url},
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()
        records.append(
            LiteratureRecord(
                arxiv_id=arxiv_id,
                title=title,
                abstract=abstract,
                published_at=published,
                source_url=raw_url,
                source_digest=digest,
            )
        )
    return tuple(records)


def _deduplicate_recent(records: Iterable[LiteratureRecord], *, cutoff: datetime) -> tuple[LiteratureRecord, ...]:
    latest: dict[str, LiteratureRecord] = {}
    for record in records:
        if record.published_at < cutoff:
            continue
        previous = latest.get(record.arxiv_id)
        if previous is None or record.published_at > previous.published_at:
            latest[record.arxiv_id] = record
    return tuple(sorted(latest.values(), key=lambda record: (record.published_at, record.arxiv_id), reverse=True))


def _persist_record(record: LiteratureRecord, staging_root: Path) -> Path:
    records_root = Path(staging_root).resolve() / "records"
    records_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    stem = _SAFE_RECORD_STEM.sub("_", record.arxiv_id)
    destination = records_root / f"{stem}.json"
    serialized = json.dumps(record.to_document(), sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    if destination.exists():
        if destination.read_text(encoding="utf-8") != serialized:
            raise LiteratureMonitorError("LIT_MONITOR_RECORD_CONFLICT")
        return destination
    destination.write_text(serialized, encoding="utf-8")
    return destination


def _required_text(entry: ElementTree.Element, field: str) -> str:
    value = (entry.findtext(field) or "").strip()
    if not value:
        raise LiteratureMonitorError("LIT_MONITOR_ATOM_FIELD_INVALID")
    return " ".join(value.split())


def _arxiv_id_from_url(value: str) -> str:
    matched = _ARXIV_ID_FROM_URL.search(value.strip())
    if matched is None:
        raise LiteratureMonitorError("LIT_MONITOR_ARXIV_ID_INVALID")
    return _canonical_arxiv_id(matched.group(1))


def _canonical_arxiv_id(value: Any) -> str:
    if not isinstance(value, str):
        raise LiteratureMonitorError("LIT_MONITOR_ARXIV_ID_INVALID")
    normalized = value.removeprefix("arXiv:")
    if not re.fullmatch(r"\d{4}\.\d{4,5}(?:v\d+)?", normalized):
        raise LiteratureMonitorError("LIT_MONITOR_ARXIV_ID_INVALID")
    return normalized


def _parse_timestamp(value: str) -> datetime:
    try:
        return _coerce_utc(datetime.fromisoformat(value.replace("Z", "+00:00")))
    except ValueError as exc:
        raise LiteratureMonitorError("LIT_MONITOR_TIMESTAMP_INVALID") from exc


def _coerce_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise LiteratureMonitorError("LIT_MONITOR_TIMESTAMP_TZ_REQUIRED")
    return value.astimezone(timezone.utc)


def _read_candidate(path: Path) -> Mapping[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LiteratureMonitorError("LIT_MONITOR_CANDIDATE_FILE_INVALID") from exc
    if not isinstance(document, Mapping):
        raise LiteratureMonitorError("LIT_MONITOR_CANDIDATE_FILE_INVALID")
    return document


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    if target.exists() or target.is_symlink():
        raise LiteratureMonitorError("LIT_MONITOR_REPORT_OUTPUT_EXISTS")
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def _parse_now(value: str | None) -> datetime | None:
    if value is None:
        return None
    try:
        return _coerce_utc(datetime.fromisoformat(value.replace("Z", "+00:00")))
    except ValueError as exc:
        raise LiteratureMonitorError("LIT_MONITOR_NOW_INVALID") from exc


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect recent arXiv metadata into untrusted wm-loop staging.")
    parser.add_argument("--query", action="append", required=True, help="Search phrase; repeat for multiple phrases.")
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--lookback-days", type=int, default=7)
    parser.add_argument("--max-results-per-query", type=int, default=50)
    parser.add_argument("--candidate-json", action="append", type=Path, default=[])
    parser.add_argument("--fixture-atom", type=Path, help="Read a local Atom fixture instead of calling arXiv; for deterministic smoke runs.")
    parser.add_argument("--now", help="UTC timestamp for deterministic lookback windows.")
    parser.add_argument("--report-output", type=Path, help="Write the CLI report JSON to this path.")
    args = parser.parse_args(argv)
    fixture_payload = None
    if args.fixture_atom is not None:
        fixture = Path(args.fixture_atom)
        if fixture.is_symlink() or not fixture.is_file():
            raise LiteratureMonitorError("LIT_MONITOR_FIXTURE_INVALID")
        fixture_payload = fixture.read_bytes()
    fetch_bytes = (lambda _url: fixture_payload) if fixture_payload is not None else None
    report = run_literature_monitor(
        query_terms=args.query,
        staging_root=args.staging_root,
        now=_parse_now(args.now),
        lookback_days=args.lookback_days,
        max_results_per_query=args.max_results_per_query,
        candidate_documents=tuple(_read_candidate(path) for path in args.candidate_json),
        fetch_bytes=fetch_bytes,
    )
    document = report.to_document()
    if args.report_output is not None:
        _write_json_atomic(args.report_output, document)
    print(json.dumps(document, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by CLI invocation
    raise SystemExit(main())
