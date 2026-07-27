"""Frozen prior retrieval and untrusted literature staging for v2.0.

Only the frozen library can affect a proposal.  Literature ingestion creates
data-only staging records; it never imports, executes, or appends a mechanism
to the primitive registry.  Promotion is an explicit four-gate boundary.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from wmloop.contracts import ContractValidationError, load_yaml_document, validate_document
from wmloop.primitives.registry import PrimitiveRegistry


class PriorLibraryError(ValueError):
    """Frozen-prior or candidate-stage invariant failed."""


_ARXIV_ID = re.compile(r"^(?:arXiv:)?\d{4}\.\d{4,5}(?:v\d+)?$")
_INSTRUCTION_MARKERS = ("ignore previous", "system prompt", "developer message", "<script", "```")


@dataclass(frozen=True)
class PriorEntry:
    failure: str
    primitive_names: tuple[str, ...]
    rationale: str
    sources: tuple[str, ...]


@dataclass(frozen=True)
class FrozenPriorLibrary:
    version: str
    registry_digest: str
    entries: tuple[PriorEntry, ...]

    @classmethod
    def from_path(cls, path: Path, *, registry: PrimitiveRegistry) -> "FrozenPriorLibrary":
        try:
            raw = load_yaml_document(path)
            version = _nonempty(raw, "library_version")
            registry_digest = _nonempty(raw, "registry_digest")
            rows = raw["entries"]
        except (ContractValidationError, KeyError, TypeError, ValueError) as exc:
            raise PriorLibraryError("PRIOR_LIBRARY_INVALID") from exc
        if registry_digest != registry.digest() or not isinstance(rows, list):
            raise PriorLibraryError("PRIOR_LIBRARY_REGISTRY_MISMATCH")
        entries: list[PriorEntry] = []
        for row in rows:
            if not isinstance(row, Mapping):
                raise PriorLibraryError("PRIOR_LIBRARY_ENTRY_INVALID")
            try:
                names = tuple(str(name) for name in row["primitive_names"])
                if not names or any(registry.manifest(name) is None for name in names):
                    raise ValueError("primitive_names")
                entry = PriorEntry(
                    failure=_nonempty(row, "failure"),
                    primitive_names=names,
                    rationale=_nonempty(row, "rationale"),
                    sources=tuple(str(source) for source in row["sources"]),
                )
            except (KeyError, TypeError, ValueError, PriorLibraryError) as exc:
                raise PriorLibraryError("PRIOR_LIBRARY_ENTRY_INVALID") from exc
            if not entry.sources or any(not _ARXIV_ID.fullmatch(source) for source in entry.sources):
                raise PriorLibraryError("PRIOR_LIBRARY_ENTRY_INVALID")
            entries.append(entry)
        return cls(version=version, registry_digest=registry_digest, entries=tuple(entries))

    def retrieve(self, failure: str) -> tuple[PriorEntry, ...]:
        return tuple(entry for entry in self.entries if entry.failure == failure)


def stage_literature_candidate(candidate: Mapping[str, Any], *, staging_root: Path) -> Path:
    """Persist a data-only candidate after rejecting instruction-like content."""

    try:
        validate_document("literature_candidate", candidate)
    except ContractValidationError as exc:
        raise PriorLibraryError("PRIOR_STAGING_SCHEMA_INVALID") from exc
    if not isinstance(candidate["proposed_manifest"], Mapping):
        raise PriorLibraryError("PRIOR_STAGING_SCHEMA_INVALID")
    if not _ARXIV_ID.fullmatch(str(candidate["arxiv_id"])):
        raise PriorLibraryError("PRIOR_STAGING_ARXIV_ID_INVALID")
    serialized = json.dumps(candidate, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    if any(marker in serialized.lower() for marker in _INSTRUCTION_MARKERS):
        raise PriorLibraryError("PRIOR_STAGING_INSTRUCTION_CONTENT")
    candidate_id = str(candidate["candidate_id"])
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", candidate_id):
        raise PriorLibraryError("PRIOR_STAGING_ID_INVALID")
    destination = Path(staging_root).resolve() / f"{candidate_id}.json"
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if destination.exists():
        raise PriorLibraryError("PRIOR_STAGING_ALREADY_EXISTS")
    destination.write_text(serialized + "\n", encoding="utf-8")
    return destination


def promotion_allowed(
    *,
    schema_valid: bool,
    clean_diff: bool,
    smoke_passed: bool,
    human_approved_version_boundary: bool,
) -> bool:
    """The sole four-gate admission predicate for a staged mechanism."""

    return all((schema_valid, clean_diff, smoke_passed, human_approved_version_boundary))


def _nonempty(raw: Mapping[str, Any], field: str) -> str:
    value = raw[field]
    if not isinstance(value, str) or not value:
        raise ValueError(field)
    return value
