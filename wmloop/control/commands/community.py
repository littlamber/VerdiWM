"""Community and settled-evidence CLI command implementations."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

from wmloop.contracts import ContractValidationError
from wmloop.control.commands.io import emit_json
from wmloop.experiments.community_bundle import (
    CommunityBundleError,
    publish_community_bundle,
    verify_community_bundle,
)
from wmloop.experiments.community_export import (
    CommunityExportError,
    export_community_knowledge,
)
from wmloop.storage import checked_path
from wmloop.geometry.community_knowledge import (
    CommunityKnowledgeError,
    build_knowledge_lifecycle_record,
)


def _load_community_documents(
    paths: Sequence[Path] | None, directory: Path | None = None
) -> list[Mapping[str, object]]:
    documents: list[Mapping[str, object]] = []
    candidates = [Path(path) for path in (paths or ())]
    if directory is not None:
        source_dir = checked_path(directory, code="COMMUNITY_DOCUMENT_DIRECTORY_INVALID", error=CommunityBundleError)
        if source_dir.is_symlink() or not source_dir.is_dir():
            raise CommunityBundleError("COMMUNITY_DOCUMENT_DIRECTORY_INVALID")
        candidates.extend(sorted(source_dir.glob("*.json")))
    if not candidates:
        raise CommunityBundleError("COMMUNITY_DOCUMENTS_EMPTY")
    for raw_path in candidates:
        path = checked_path(raw_path, code="COMMUNITY_DOCUMENT_FILE_INVALID", error=CommunityBundleError)
        if path.is_symlink() or not path.is_file():
            raise CommunityBundleError("COMMUNITY_DOCUMENT_FILE_INVALID")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CommunityBundleError("COMMUNITY_DOCUMENT_FILE_INVALID") from exc
        if isinstance(payload, Mapping):
            documents.append(payload)
        elif isinstance(payload, list) and all(
            isinstance(item, Mapping) for item in payload
        ):
            documents.extend(payload)
        else:
            raise CommunityBundleError("COMMUNITY_DOCUMENT_FILE_INVALID")
    return documents


def _community_publish(args: argparse.Namespace) -> int:
    documents = _load_community_documents(args.document, args.documents_dir)
    manifest = publish_community_bundle(
        documents=documents,
        output_root=args.output_root,
        publisher_id=args.publisher_id,
        signing_key=args.signing_key,
        trust_state=args.trust_state,
        license_spdx_id=args.license_spdx_id,
        community_review_state=args.community_review_state,
        execution_manifest=args.execution,
    )
    emit_json(manifest)
    return 0


def _community_verify(args: argparse.Namespace) -> int:
    result = verify_community_bundle(args.bundle_root, public_key=args.public_key)
    emit_json(result)
    return 0 if result["state"] in {"verified", "revoked"} else 2


def _community_export(args: argparse.Namespace) -> int:
    """Discover and stage validated semantic records without publishing them."""

    report = export_community_knowledge(
        source_roots=args.source_root,
        output_root=args.output_root,
        execution_manifest=args.execution,
        max_files=args.max_files,
        max_file_bytes=args.max_file_bytes,
        root=args.repo_root,
    )
    emit_json(report)
    return 0


def _evidence_project(args: argparse.Namespace) -> int:
    from wmloop.archive.community_projection import (
        EvidenceProjectionError,
        project_archive_evidence,
    )

    try:
        report = project_archive_evidence(
            archive=args.archive,
            cas=args.cas,
            output_root=args.output,
            execution_manifest=args.batch,
            max_trials=args.max_trials,
        )
    except (EvidenceProjectionError, ContractValidationError) as exc:
        raise CommunityExportError(str(exc)) from exc
    emit_json(report)
    return 0


def _community_lifecycle(args: argparse.Namespace) -> int:
    record = build_knowledge_lifecycle_record(
        action=args.action,
        subject_kind=args.subject_kind,
        subject_id=args.subject_id,
        reason=args.reason,
        authority_ref=args.authority_ref,
        evidence_refs=tuple(args.evidence_ref or ()),
        replacement_kind=args.replacement_kind,
        replacement_id=args.replacement_id,
    )
    destination = Path(args.output).expanduser().resolve()
    _write_community_json(destination, record)
    emit_json({**record, "output": str(destination)})
    return 0


def _write_community_json(destination: Path, payload: Mapping[str, object]) -> None:
    """Persist one path-free community document without partial writes."""

    if destination.is_symlink() or (destination.exists() and not destination.is_file()):
        raise CommunityKnowledgeError("COMMUNITY_LIFECYCLE_OUTPUT_INVALID")
    encoded = (
        json.dumps(
            dict(payload), ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False
        )
        + "\n"
    )
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if destination.is_file():
        try:
            if destination.read_text(encoding="utf-8") == encoded:
                return
        except OSError as exc:
            raise CommunityKnowledgeError("COMMUNITY_LIFECYCLE_OUTPUT_INVALID") from exc
        raise CommunityKnowledgeError("COMMUNITY_LIFECYCLE_OUTPUT_CONFLICT")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(encoded)
        os.replace(temporary, destination)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()
