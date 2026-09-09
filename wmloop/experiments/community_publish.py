"""Community publish implementation."""

from __future__ import annotations
import hashlib
import base64
from collections.abc import Mapping, Sequence
from pathlib import Path
from wmloop.contracts import ContractValidationError, validate_document
from wmloop.storage import checked_path
from wmloop.experiments.portable_knowledge_graph import audit_portable_knowledge_graph, build_portable_knowledge_graph
from .community_signing import CommunityBundleError, _ALGORITHM, _read_private_key, _require_cryptography, serialization
from .community_manifest import _canonical, _digest, _load_execution_context, _unique_documents, _validate_publisher_id, _validate_review_state, _validate_state
from .community_bundle_io import _write_bundle


def publish_community_bundle(
    *,
    documents: Sequence[Mapping[str, object]],
    output_root: Path,
    publisher_id: str,
    signing_key: Path,
    trust_state: str = "locally_validated",
    license_spdx_id: str | None = None,
    community_review_state: str = "unreviewed",
    execution_manifest: Path | None = None,
) -> dict[str, object]:
    """Build and sign an immutable community bundle.

    ``documents`` are passed through the existing portable graph validator. The
    function writes only path-free semantic records and their derived graph;
    local execution state is never copied into the bundle.
    """

    publisher = _validate_publisher_id(publisher_id)
    _require_cryptography()
    state = _validate_state(trust_state)
    review = _validate_review_state(community_review_state)
    if license_spdx_id is not None and (
        not isinstance(license_spdx_id, str) or not license_spdx_id.strip()
    ):
        raise CommunityBundleError("COMMUNITY_BUNDLE_LICENSE_INVALID")
    private_key = _read_private_key(signing_key)
    public_bytes = private_key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    key_id = "key-" + hashlib.sha256(public_bytes).hexdigest()[:16]
    graph = build_portable_knowledge_graph(documents)
    audit = audit_portable_knowledge_graph(documents=documents, graph=graph)
    records = _unique_documents(documents)
    source_execution = _load_execution_context(execution_manifest) if execution_manifest is not None else None
    record_payloads = {
        f"records/{_digest(document)[:32]}.json": _canonical(document) + b"\n"
        for document in records
    }
    files_payloads: dict[str, bytes] = {
        "graph.json": _canonical(graph) + b"\n",
        "quality-audit.json": _canonical(audit) + b"\n",
        **record_payloads,
    }
    file_hashes = {
        name: hashlib.sha256(payload).hexdigest()
        for name, payload in sorted(files_payloads.items())
    }
    manifest_body: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "verdiwm-community-evidence-bundle",
        "state": "ready",
        "trust_state": state,
        "community_review_state": review,
        "publisher": {"publisher_id": publisher, "signing_key_id": key_id},
        "license_spdx_id": license_spdx_id,
        "graph_digest": audit["graph_digest"],
        "quality_audit_id": audit["audit_id"],
        "record_count": len(records),
        "files": file_hashes,
        "claim_boundary": (
            "This signed bundle transports a path-free semantic projection. "
            "It does not replace Archive/CAS receipts, frozen evaluators, or target-side verification."
        ),
    }
    if source_execution is not None:
        manifest_body["source_execution"] = source_execution
    manifest_body["bundle_id"] = "verdiwm-bundle-" + _digest(manifest_body)[:24]
    manifest_bytes = _canonical(manifest_body) + b"\n"
    signature = {
        "schema_version": 1,
        "artifact_type": "verdiwm-community-bundle-signature",
        "algorithm": _ALGORITHM,
        "publisher_id": publisher,
        "signing_key_id": key_id,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "public_key": base64.b64encode(public_bytes).decode("ascii"),
        "signature": base64.b64encode(private_key.sign(manifest_bytes)).decode("ascii"),
        "claim_boundary": "The signature authenticates the bundle manifest and member hashes only.",
    }
    try:
        validate_document("community_bundle", manifest_body)
        validate_document("community_bundle_signature", signature)
    except ContractValidationError as exc:
        raise CommunityBundleError(f"COMMUNITY_BUNDLE_CONTRACT_INVALID:{exc}") from exc
    destination = checked_path(Path(output_root).expanduser(), code="COMMUNITY_BUNDLE_OUTPUT_INVALID", error=CommunityBundleError)
    _write_bundle(
        destination,
        payloads={**files_payloads, "bundle.json": manifest_bytes, "signature.json": _canonical(signature) + b"\n"},
        bundle_id=str(manifest_body["bundle_id"]),
    )
    return {**manifest_body, "signature": signature}
