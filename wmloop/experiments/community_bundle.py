"""Signed, path-free bundles for community knowledge exchange.

The bundle is a transport envelope around the existing portable knowledge
projection.  It contains no model source, checkpoint, dataset path, runtime
command, or local database.  Archive/CAS receipts remain authoritative; a
bundle is only a signed, verifiable publication artifact for a registry.

The bundle uses Ed25519. Publishers keep the private key local; a registry can
verify authorship with the public key embedded in the signature envelope.
"""

from __future__ import annotations

import hashlib
import base64
import json
import os
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

try:  # Keep base CLI/help usable before runtime dependencies are installed.
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
        Ed25519PublicKey,
    )
except ModuleNotFoundError:  # pragma: no cover - exercised on a bare install
    InvalidSignature = type("InvalidSignature", (Exception,), {})
    Ed25519PrivateKey = None  # type: ignore[assignment,misc]
    Ed25519PublicKey = None  # type: ignore[assignment,misc]
    serialization = None  # type: ignore[assignment]
from wmloop.contracts import ContractValidationError, validate_document
from wmloop.control.model_batch import (
    ModelBatchError,
    build_model_batch_execution_binding,
)
from wmloop.experiments.portable_knowledge_graph import (
    audit_portable_knowledge_graph,
    build_portable_knowledge_graph,
)


class CommunityBundleError(ValueError):
    """A community bundle failed validation or an immutable write invariant."""


_STATES = {
    "unverified",
    "locally_validated",
    "source_reproducible",
    "target_confirmed",
    "transfer_licensed",
    "community_reviewed",
    "revoked",
}
_ALGORITHM = "ed25519-v1"
_KEY_ID = re.compile(r"^key-[0-9a-f]{16}$")
_PUBLISHER_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@/-]{2,127}$")
_SAFE_MEMBER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*(?:/[A-Za-z0-9][A-Za-z0-9_.-]*)*$")


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
    destination = Path(output_root).expanduser().resolve()
    _write_bundle(
        destination,
        payloads={**files_payloads, "bundle.json": manifest_bytes, "signature.json": _canonical(signature) + b"\n"},
        bundle_id=str(manifest_body["bundle_id"]),
    )
    return {**manifest_body, "signature": signature}


def verify_community_bundle(
    bundle_root: Path,
    *,
    public_key: Path | None = None,
) -> dict[str, object]:
    """Verify manifest, member hashes, semantic graph, and publisher signature."""

    root = Path(bundle_root).expanduser().resolve()
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
    source_execution = manifest.get("source_execution")
    if source_execution is not None:
        if not isinstance(source_execution, Mapping):
            raise CommunityBundleError("COMMUNITY_BUNDLE_EXECUTION_BINDING_INVALID")
        model_ids = source_execution.get("model_ids")
        if (
            not isinstance(model_ids, list)
            or model_ids != sorted(model_ids)
            or len(set(model_ids)) != len(model_ids)
            or any(not isinstance(model_id, str) or not model_id for model_id in model_ids)
        ):
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


def _load_execution_context(path: Path) -> dict[str, object]:
    """Extract a path-free binding from a batch execution manifest.

    The execution manifest is deliberately not copied into the bundle: it is an
    orchestration record and contains local campaign, budget, and runtime paths.
    Only its validated identity and model identifiers cross the publication
    boundary.
    """

    try:
        return build_model_batch_execution_binding(path)
    except ModelBatchError as exc:
        code = str(exc).split(":", 1)[0]
        if code == "MODEL_BATCH_EXECUTION_DIGEST_MISMATCH":
            raise CommunityBundleError("COMMUNITY_BUNDLE_EXECUTION_DIGEST_MISMATCH") from exc
        raise CommunityBundleError(f"COMMUNITY_BUNDLE_EXECUTION_INVALID:{exc}") from exc


def _unique_documents(documents: Sequence[Mapping[str, object]]) -> list[Mapping[str, object]]:
    if not isinstance(documents, Sequence) or isinstance(documents, (str, bytes)):
        raise CommunityBundleError("COMMUNITY_BUNDLE_DOCUMENTS_INVALID")
    rows: dict[str, Mapping[str, object]] = {}
    for document in documents:
        if not isinstance(document, Mapping):
            raise CommunityBundleError("COMMUNITY_BUNDLE_DOCUMENT_INVALID")
        rows[_digest(document)] = document
    if not rows:
        raise CommunityBundleError("COMMUNITY_BUNDLE_DOCUMENTS_EMPTY")
    return [rows[key] for key in sorted(rows)]


def _read_private_key(path: Path) -> Ed25519PrivateKey:
    _require_cryptography()
    source = Path(path).expanduser().resolve()
    if source.is_symlink() or not source.is_file():
        raise CommunityBundleError("COMMUNITY_BUNDLE_SIGNING_KEY_INVALID")
    try:
        key = serialization.load_pem_private_key(source.read_bytes(), password=None)
    except (OSError, ValueError, TypeError) as exc:
        raise CommunityBundleError("COMMUNITY_BUNDLE_SIGNING_KEY_INVALID") from exc
    if not isinstance(key, Ed25519PrivateKey):
        raise CommunityBundleError("COMMUNITY_BUNDLE_SIGNING_KEY_INVALID")
    return key


def _read_public_key(path: Path) -> Ed25519PublicKey:
    _require_cryptography()
    source = Path(path).expanduser().resolve()
    if source.is_symlink() or not source.is_file():
        raise CommunityBundleError("COMMUNITY_BUNDLE_PUBLIC_KEY_INVALID")
    try:
        key = serialization.load_pem_public_key(source.read_bytes())
    except (OSError, ValueError, TypeError) as exc:
        raise CommunityBundleError("COMMUNITY_BUNDLE_PUBLIC_KEY_INVALID") from exc
    if not isinstance(key, Ed25519PublicKey):
        raise CommunityBundleError("COMMUNITY_BUNDLE_PUBLIC_KEY_INVALID")
    return key


def _validate_publisher_id(value: str) -> str:
    if not isinstance(value, str) or _PUBLISHER_ID.fullmatch(value) is None:
        raise CommunityBundleError("COMMUNITY_BUNDLE_PUBLISHER_ID_INVALID")
    return value


def _require_cryptography() -> None:
    if Ed25519PrivateKey is None or Ed25519PublicKey is None or serialization is None:
        raise CommunityBundleError(
            "COMMUNITY_BUNDLE_CRYPTOGRAPHY_REQUIRED:run 'uv sync --group dev' to install the signed-community dependency"
        )


def _validate_state(value: str) -> str:
    if value not in _STATES:
        raise CommunityBundleError("COMMUNITY_BUNDLE_TRUST_STATE_INVALID")
    return value


def _validate_review_state(value: str) -> str:
    if value not in {"unreviewed", "reviewed", "withdrawn"}:
        raise CommunityBundleError("COMMUNITY_BUNDLE_REVIEW_STATE_INVALID")
    return value


def _validate_member_name(value: str) -> None:
    if _SAFE_MEMBER.fullmatch(value) is None or value.startswith("/") or ".." in value.split("/"):
        raise CommunityBundleError("COMMUNITY_BUNDLE_MEMBER_NAME_INVALID")


def _read_json(path: Path, code: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise CommunityBundleError(code)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CommunityBundleError(code) from exc
    if not isinstance(payload, dict):
        raise CommunityBundleError(code)
    return payload


def _write_bundle(destination: Path, *, payloads: Mapping[str, bytes], bundle_id: str) -> None:
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or not destination.is_dir():
            raise CommunityBundleError("COMMUNITY_BUNDLE_OUTPUT_INVALID")
        existing = destination / "bundle.json"
        if existing.is_file() and not existing.is_symlink():
            prior = _read_json(existing, "COMMUNITY_BUNDLE_OUTPUT_INVALID")
            if prior.get("bundle_id") == bundle_id:
                for name, payload in payloads.items():
                    candidate = destination / name
                    if (
                        candidate.is_symlink()
                        or not candidate.is_file()
                        or candidate.read_bytes() != payload
                    ):
                        raise CommunityBundleError("COMMUNITY_BUNDLE_OUTPUT_CONFLICT")
                return
        if any(destination.iterdir()):
            raise CommunityBundleError("COMMUNITY_BUNDLE_OUTPUT_BOUND")
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        for name, payload in payloads.items():
            _validate_member_name(name)
            target = temporary / name
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            target.write_bytes(payload)
        # ``os.replace`` cannot replace an existing directory.  An empty
        # destination is safe to claim atomically after removing that marker.
        if destination.exists() and destination.is_dir() and not any(destination.iterdir()):
            destination.rmdir()
        os.replace(temporary, destination)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()
