"""Community manifest implementation."""

from __future__ import annotations
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from wmloop.control.model_batch import ModelBatchError, build_model_batch_execution_binding
from .community_signing import CommunityBundleError


_STATES = {
    "unverified",
    "locally_validated",
    "source_reproducible",
    "target_confirmed",
    "transfer_licensed",
    "community_reviewed",
    "revoked",
}


_KEY_ID = re.compile(r"^key-[0-9a-f]{16}$")


_PUBLISHER_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@/-]{2,127}$")


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


def _validate_publisher_id(value: str) -> str:
    if not isinstance(value, str) or _PUBLISHER_ID.fullmatch(value) is None:
        raise CommunityBundleError("COMMUNITY_BUNDLE_PUBLISHER_ID_INVALID")
    return value


def _validate_state(value: str) -> str:
    if value not in _STATES:
        raise CommunityBundleError("COMMUNITY_BUNDLE_TRUST_STATE_INVALID")
    return value


def _validate_review_state(value: str) -> str:
    if value not in {"unreviewed", "reviewed", "withdrawn"}:
        raise CommunityBundleError("COMMUNITY_BUNDLE_REVIEW_STATE_INVALID")
    return value


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()
