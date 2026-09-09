"""Community verification implementation."""

from __future__ import annotations
import hashlib
import base64
from collections.abc import Mapping
from pathlib import Path
from wmloop.contracts import ContractValidationError, validate_document
from wmloop.control.model_batch import ModelBatchError
from wmloop.storage import checked_path, require_exact_tree
from wmloop.control.batch_state import validate_model_ids
from wmloop.experiments.portable_knowledge_graph import build_portable_knowledge_graph, audit_portable_knowledge_graph
from .community_signing import CommunityBundleError, Ed25519PublicKey, InvalidSignature, _ALGORITHM, _read_public_key, _require_cryptography, serialization
from .community_manifest import _canonical, _digest, _validate_publisher_id
from .community_bundle_io import _read_json, _validate_member_name


def verify_community_bundle(
    bundle_root: Path,
    *,
    public_key: Path | None = None,
) -> dict[str, object]:
    """Verify manifest, member hashes, semantic graph, and publisher signature."""

    root = checked_path(Path(bundle_root).expanduser(), code="COMMUNITY_BUNDLE_ROOT_INVALID", error=CommunityBundleError)
    _require_cryptography()
    if root.is_symlink() or not root.is_dir():
        raise CommunityBundleError("COMMUNITY_BUNDLE_ROOT_INVALID")
    manifest = _read_json(root / "bundle.json", "COMMUNITY_BUNDLE_MANIFEST_INVALID")
    signature = _read_json(root / "signature.json", "COMMUNITY_BUNDLE_SIGNATURE_INVALID")
    try:
        validate_document("community_bundle", manifest)
        validate_document("community_bundle_signature", signature)
    except ContractValidationError as exc:
        raise CommunityBundleError(f"COMMUNITY_BUNDLE_CONTRACT_INVALID:{exc}") from exc
    if signature.get("algorithm") != _ALGORITHM:
        raise CommunityBundleError("COMMUNITY_BUNDLE_SIGNATURE_ALGORITHM_UNSUPPORTED")
    publisher = manifest.get("publisher")
    if not isinstance(publisher, Mapping) or signature.get("publisher_id") != publisher.get("publisher_id") or signature.get("signing_key_id") != publisher.get("signing_key_id"):
        raise CommunityBundleError("COMMUNITY_BUNDLE_SIGNER_BINDING_INVALID")
    try:
        _validate_publisher_id(str(publisher.get("publisher_id")))
    except CommunityBundleError as exc:
        raise CommunityBundleError("COMMUNITY_BUNDLE_PUBLISHER_ID_INVALID") from exc
    bundle_id = manifest.get("bundle_id")
    manifest_without_id = dict(manifest)
    manifest_without_id.pop("bundle_id", None)
    expected_bundle_id = "verdiwm-bundle-" + _digest(manifest_without_id)[:24]
    if bundle_id != expected_bundle_id:
        raise CommunityBundleError("COMMUNITY_BUNDLE_ID_MISMATCH")
    manifest_bytes = _canonical(manifest) + b"\n"
    if signature.get("manifest_sha256") != hashlib.sha256(manifest_bytes).hexdigest():
        raise CommunityBundleError("COMMUNITY_BUNDLE_MANIFEST_DIGEST_MISMATCH")
    encoded_public = signature.get("public_key")
    if not isinstance(encoded_public, str):
        raise CommunityBundleError("COMMUNITY_BUNDLE_PUBLIC_KEY_INVALID")
    try:
        public_bytes = base64.b64decode(encoded_public, validate=True)
        if len(public_bytes) != 32:
            raise ValueError("Ed25519 public keys are exactly 32 bytes")
        verifier = Ed25519PublicKey.from_public_bytes(public_bytes)
    except (ValueError, TypeError) as exc:
        raise CommunityBundleError("COMMUNITY_BUNDLE_PUBLIC_KEY_INVALID") from exc
    if public_key is not None:
        supplied = _read_public_key(public_key)
        supplied_bytes = supplied.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        if supplied_bytes != public_bytes:
            raise CommunityBundleError("COMMUNITY_BUNDLE_PUBLIC_KEY_MISMATCH")
        verifier = supplied
    expected_key_id = "key-" + hashlib.sha256(public_bytes).hexdigest()[:16]
    if publisher.get("signing_key_id") != expected_key_id:
        raise CommunityBundleError("COMMUNITY_BUNDLE_SIGNING_KEY_ID_MISMATCH")
    encoded_signature = signature.get("signature")
    if not isinstance(encoded_signature, str):
        raise CommunityBundleError("COMMUNITY_BUNDLE_SIGNATURE_INVALID")
    try:
        raw_signature = base64.b64decode(encoded_signature, validate=True)
        if len(raw_signature) != 64:
            raise ValueError("Ed25519 signatures are exactly 64 bytes")
        verifier.verify(raw_signature, manifest_bytes)
    except (InvalidSignature, ValueError, TypeError):
        raise CommunityBundleError("COMMUNITY_BUNDLE_SIGNATURE_INVALID")
    file_hashes = manifest.get("files")
    if not isinstance(file_hashes, Mapping):
        raise CommunityBundleError("COMMUNITY_BUNDLE_FILES_INVALID")
    required_members = {"graph.json", "quality-audit.json"}
    record_members = sorted(
        str(member) for member in file_hashes if str(member).startswith("records/")
    )
    if not required_members.issubset({str(member) for member in file_hashes}):
        raise CommunityBundleError("COMMUNITY_BUNDLE_REQUIRED_MEMBER_MISSING")
    if not record_members:
        raise CommunityBundleError("COMMUNITY_BUNDLE_RECORDS_EMPTY")
    if set(file_hashes) != required_members | set(record_members):
        raise CommunityBundleError("COMMUNITY_BUNDLE_MEMBER_TYPE_INVALID")
    if manifest.get("record_count") != len(record_members):
        raise CommunityBundleError("COMMUNITY_BUNDLE_RECORD_COUNT_MISMATCH")
    for member, expected in file_hashes.items():
        name = str(member)
        _validate_member_name(name)
        path = root / name
        if not path.is_file() or path.is_symlink():
            raise CommunityBundleError("COMMUNITY_BUNDLE_MEMBER_MISSING:" + name)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise CommunityBundleError("COMMUNITY_BUNDLE_MEMBER_DIGEST_MISMATCH:" + name)
    expected_files = {"bundle.json", "signature.json", *map(str, file_hashes)}
    actual_files: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise CommunityBundleError("COMMUNITY_BUNDLE_SYMLINK_FORBIDDEN")
        if path.is_file():
            actual_files.add(path.relative_to(root).as_posix())
    if actual_files != expected_files:
        raise CommunityBundleError("COMMUNITY_BUNDLE_DIRECTORY_CONTENT_MISMATCH")
    require_exact_tree(root, expected_files, code="COMMUNITY_BUNDLE_DIRECTORY_CONTENT_MISMATCH", error=CommunityBundleError)
    graph = _read_json(root / "graph.json", "COMMUNITY_BUNDLE_GRAPH_INVALID")
    audit = _read_json(root / "quality-audit.json", "COMMUNITY_BUNDLE_AUDIT_INVALID")
    try:
        validate_document("portable_knowledge_quality_audit", audit)
    except ContractValidationError as exc:
        raise CommunityBundleError(f"COMMUNITY_BUNDLE_AUDIT_CONTRACT_INVALID:{exc}") from exc
    if "graph_digest" not in manifest or manifest["graph_digest"] != audit.get("graph_digest"):
        raise CommunityBundleError("COMMUNITY_BUNDLE_GRAPH_BINDING_INVALID")
    if manifest.get("quality_audit_id") != audit.get("audit_id"):
        raise CommunityBundleError("COMMUNITY_BUNDLE_AUDIT_BINDING_INVALID")
    if _digest(graph) != str(audit.get("graph_digest", "")).removeprefix("sha256:"):
        raise CommunityBundleError("COMMUNITY_BUNDLE_GRAPH_DIGEST_INVALID")
    records = []
    for member in record_members:
        records.append(_read_json(root / member, "COMMUNITY_BUNDLE_RECORD_INVALID"))
        expected_record_name = f"records/{_digest(records[-1])[:32]}.json"
        if member != expected_record_name:
            raise CommunityBundleError("COMMUNITY_BUNDLE_RECORD_NAME_MISMATCH:" + member)
    rebuilt = build_portable_knowledge_graph(records)
    if _canonical(rebuilt) != _canonical(graph):
        raise CommunityBundleError("COMMUNITY_BUNDLE_GRAPH_REBUILD_MISMATCH")
    if graph.get("document_count") != len(records):
        raise CommunityBundleError("COMMUNITY_BUNDLE_GRAPH_DOCUMENT_COUNT_MISMATCH")
    if audit.get("document_count") != len(records):
        raise CommunityBundleError("COMMUNITY_BUNDLE_AUDIT_DOCUMENT_COUNT_MISMATCH")
    expected_digests = sorted("sha256:" + _digest(record) for record in records)
    if audit.get("document_digests") != expected_digests:
        raise CommunityBundleError("COMMUNITY_BUNDLE_AUDIT_RECORD_BINDING_MISMATCH")
    if _canonical(audit_portable_knowledge_graph(documents=records, graph=graph)) != _canonical(audit):
        raise CommunityBundleError("COMMUNITY_BUNDLE_AUDIT_REBUILD_MISMATCH")
    source_execution = manifest.get("source_execution")
    if source_execution is not None:
        if not isinstance(source_execution, Mapping):
            raise CommunityBundleError("COMMUNITY_BUNDLE_EXECUTION_BINDING_INVALID")
        model_ids = source_execution.get("model_ids")
        try:
            valid_ids = validate_model_ids(model_ids)
        except ModelBatchError as exc:
            raise CommunityBundleError("COMMUNITY_BUNDLE_EXECUTION_BINDING_INVALID") from exc
        if model_ids != valid_ids:
            raise CommunityBundleError("COMMUNITY_BUNDLE_EXECUTION_BINDING_INVALID")
    return {
        "schema_version": 1,
        "artifact_type": "verdiwm-community-bundle-verification",
        "state": "revoked" if manifest.get("trust_state") == "revoked" else "verified",
        "bundle_id": manifest.get("bundle_id"),
        "publisher": dict(publisher),
        "trust_state": manifest.get("trust_state"),
        "community_review_state": manifest.get("community_review_state"),
        "source_execution": dict(source_execution) if isinstance(source_execution, Mapping) else None,
        "record_count": len(records),
        "file_count": len(file_hashes),
        "signature_algorithm": _ALGORITHM,
        "claim_boundary": "Verification establishes bundle integrity and signer binding; it does not establish a target model-quality claim.",
    }
