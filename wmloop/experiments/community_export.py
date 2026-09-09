"""Read-only discovery and staging of path-free community knowledge records.

The exporter scans local JSON artifacts for the semantic document types already
accepted by :mod:`wmloop.experiments.portable_knowledge_graph`. Unrelated
runtime manifests are counted and ignored. A recognized semantic document must
pass its full contract, path-freedom audit, and deterministic graph projection
before it is copied into the export.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

from wmloop.contracts import ContractValidationError, validate_document
from wmloop.storage import checked_path
from wmloop.experiments.community_files import CommunityExportError, semantic_files, bounded_read
from wmloop.control.model_batch import (
    ModelBatchError,
    build_model_batch_execution_binding,
)
from wmloop.experiments.portable_knowledge_graph import (
    audit_portable_knowledge_graph,
    build_portable_knowledge_graph,
)


SUPPORTED_ARTIFACT_TYPES = frozenset(
    {
        "verdiwm-settled-evidence",
        "verdiwm-model-capability-ir",
        "verdiwm-model-portrait",
        "verdiwm-model-irg",
        "verdiwm-portrait-transition",
        "verdiwm-module-composition-receipt",
        "verdiwm-evidence-ir",
        "verdiwm-protocol-contract",
        "verdiwm-transformation-contract",
        "verdiwm-knowledge-lifecycle",
        "verdiwm-transferable-experience",
        "verdiwm-mechanism-contract",
        "verdiwm-mechanism-relation",
        "verdiwm-method-embodiment",
        "verdiwm-probe-fingerprint-summary",
        "verdiwm-transfer-boundary",
        "verdiwm-method-ir",
        "verdiwm-method-interface-extension",
    }
)


def export_community_knowledge(
    *,
    source_roots: Sequence[Path],
    output_root: Path,
    execution_manifest: Path | None = None,
    max_files: int = 10_000,
    max_file_bytes: int = 5 * 1024 * 1024,
    root: Path | None = None,
) -> dict[str, object]:
    """Discover, validate, and atomically stage portable semantic documents."""

    if (
        isinstance(max_files, bool)
        or not isinstance(max_files, int)
        or max_files < 1
        or max_files > 1_000_000
    ):
        raise CommunityExportError("COMMUNITY_EXPORT_MAX_FILES_INVALID")
    if (
        isinstance(max_file_bytes, bool)
        or not isinstance(max_file_bytes, int)
        or max_file_bytes < 1024
        or max_file_bytes > 1024 * 1024 * 1024
    ):
        raise CommunityExportError("COMMUNITY_EXPORT_MAX_FILE_BYTES_INVALID")
    if not isinstance(source_roots, Sequence) or isinstance(source_roots, (str, bytes)):
        raise CommunityExportError("COMMUNITY_EXPORT_SOURCE_ROOTS_INVALID")
    sources: list[Path] = []
    for raw in source_roots:
        source = checked_path(raw, code="COMMUNITY_EXPORT_SOURCE_ROOT_INVALID", error=CommunityExportError)
        if source.is_symlink() or not source.is_dir():
            raise CommunityExportError("COMMUNITY_EXPORT_SOURCE_ROOT_INVALID")
        if source not in sources:
            sources.append(source)
    if not sources:
        raise CommunityExportError("COMMUNITY_EXPORT_SOURCE_ROOTS_EMPTY")

    destination = checked_path(output_root, code="COMMUNITY_EXPORT_OUTPUT_INVALID", error=CommunityExportError)
    documents: dict[str, Mapping[str, object]] = {}
    artifact_counts: Counter[str] = Counter()
    ignored_counts: Counter[str] = Counter()
    scanned_files = 0
    candidate_documents = 0
    for path in semantic_files(sources, destination, max_files=max_files):
        if path.is_symlink() or not path.is_file() or _is_within(path, destination):
            continue
        scanned_files += 1
        if scanned_files > max_files:
            raise CommunityExportError("COMMUNITY_EXPORT_FILE_LIMIT_EXCEEDED")
        try:
            size = path.stat().st_size
        except OSError as exc:
            raise CommunityExportError("COMMUNITY_EXPORT_SOURCE_READ_FAILED") from exc
        if size > max_file_bytes:
            ignored_counts["oversized_json"] += 1
            continue
        try:
            content = bounded_read(path, max_file_bytes)
            if len(content) > max_file_bytes:
                ignored_counts["oversized_json"] += 1
                continue
            value = ([json.loads(line) for line in content.splitlines() if line.strip()]
                     if path.suffix == ".jsonl" else json.loads(content))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            ignored_counts["invalid_json"] += 1
            continue
        rows: list[object]
        if isinstance(value, Mapping):
            rows = [value]
        elif isinstance(value, list):
            rows = list(value)
        else:
            ignored_counts["non_document_json"] += 1
            continue
        for row in rows:
            candidate_documents += 1
            if not isinstance(row, Mapping):
                ignored_counts["non_object_document"] += 1
                continue
            artifact = row.get("artifact_type")
            if not isinstance(artifact, str) or artifact not in SUPPORTED_ARTIFACT_TYPES:
                ignored_counts["unsupported_artifact_type"] += 1
                continue
            document = dict(row)
            try:
                single_graph = build_portable_knowledge_graph([document])
                audit_portable_knowledge_graph(
                    documents=[document], graph=single_graph
                )
            except Exception as exc:
                raise CommunityExportError(
                    f"COMMUNITY_EXPORT_DOCUMENT_INVALID:{artifact}:{exc}"
                ) from exc
            digest = _digest(document)
            documents.setdefault(digest, document)
            artifact_counts[artifact] += 1

    if not documents:
        raise CommunityExportError("COMMUNITY_EXPORT_NO_PORTABLE_DOCUMENTS")
    records = [documents[digest] for digest in sorted(documents)]
    graph = build_portable_knowledge_graph(records)
    audit = audit_portable_knowledge_graph(documents=records, graph=graph)
    source_execution = None
    if execution_manifest is not None:
        try:
            source_execution = build_model_batch_execution_binding(
                execution_manifest, root=root
            )
        except ModelBatchError as exc:
            raise CommunityExportError(
                f"COMMUNITY_EXPORT_EXECUTION_INVALID:{exc}"
            ) from exc

    manifest: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "verdiwm-community-semantic-export",
        "state": "ready",
        "source_root_count": len(sources),
        "scanned_file_count": scanned_files,
        "candidate_document_count": candidate_documents,
        "record_count": len(records),
        "record_digests": ["sha256:" + digest for digest in sorted(documents)],
        "artifact_type_counts": dict(sorted(artifact_counts.items())),
        "ignored_document_count": sum(ignored_counts.values()),
        "ignored_reason_counts": dict(sorted(ignored_counts.items())),
        "graph_digest": audit["graph_digest"],
        "quality_audit_id": audit["audit_id"],
        "source_execution": source_execution,
        "claim_boundary": (
            "This read-only export contains validated path-free semantic records. "
            "It does not promote orchestration state, local files, or unsettled results into evidence."
        ),
    }
    manifest["export_id"] = "verdiwm-community-export-" + _digest(manifest)[:24]
    try:
        validate_document(
            "community_export",
            manifest,
            root=(root or Path(__file__).resolve().parents[2]).resolve(),
        )
    except ContractValidationError as exc:
        raise CommunityExportError(f"COMMUNITY_EXPORT_CONTRACT_INVALID:{exc}") from exc

    payloads: dict[str, bytes] = {
        "export.json": _canonical(manifest) + b"\n",
        "graph.json": _canonical(graph) + b"\n",
        "quality-audit.json": _canonical(audit) + b"\n",
    }
    for digest, document in sorted(documents.items()):
        payloads[f"records/{digest}.json"] = _canonical(document) + b"\n"
    _write_export(destination, payloads=payloads, export_id=str(manifest["export_id"]))
    return manifest


def _write_export(
    destination: Path, *, payloads: Mapping[str, bytes], export_id: str
) -> None:
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or not destination.is_dir():
            raise CommunityExportError("COMMUNITY_EXPORT_OUTPUT_INVALID")
        manifest_path = destination / "export.json"
        if manifest_path.is_file() and not manifest_path.is_symlink():
            try:
                prior = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise CommunityExportError("COMMUNITY_EXPORT_OUTPUT_INVALID") from exc
            if isinstance(prior, Mapping) and prior.get("export_id") == export_id:
                for name, payload in payloads.items():
                    path = destination / name
                    if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
                        raise CommunityExportError("COMMUNITY_EXPORT_OUTPUT_CONFLICT")
                actual_files: set[str] = set()
                actual_directories: set[str] = set()
                for path in destination.rglob("*"):
                    if path.is_symlink():
                        raise CommunityExportError("COMMUNITY_EXPORT_OUTPUT_CONFLICT")
                    relative = path.relative_to(destination).as_posix()
                    if path.is_file():
                        actual_files.add(relative)
                    elif path.is_dir():
                        actual_directories.add(relative)
                    else:
                        raise CommunityExportError("COMMUNITY_EXPORT_OUTPUT_CONFLICT")
                expected_directories = {
                    str(Path(name).parent).replace(os.sep, "/")
                    for name in payloads
                    if Path(name).parent != Path(".")
                }
                if actual_files != set(payloads) or actual_directories != expected_directories:
                    raise CommunityExportError("COMMUNITY_EXPORT_OUTPUT_CONFLICT")
                return
        if any(destination.iterdir()):
            raise CommunityExportError("COMMUNITY_EXPORT_OUTPUT_BOUND")
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        for name, payload in payloads.items():
            target = temporary / name
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            target.write_bytes(payload)
        if destination.exists() and destination.is_dir() and not any(destination.iterdir()):
            destination.rmdir()
        os.replace(temporary, destination)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root)
    except ValueError:
        return False
    return True


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CommunityExportError("COMMUNITY_EXPORT_CANONICAL_INVALID") from exc


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()
