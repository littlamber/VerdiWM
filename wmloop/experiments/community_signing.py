"""Community signing implementation."""

from __future__ import annotations
from pathlib import Path
from wmloop.storage import checked_path


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


_ALGORITHM = "ed25519-v1"


class CommunityBundleError(ValueError):
    """A community bundle failed validation or an immutable write invariant."""


def _read_private_key(path: Path) -> Ed25519PrivateKey:
    _require_cryptography()
    source = checked_path(path, code="COMMUNITY_BUNDLE_SIGNING_KEY_INVALID", error=CommunityBundleError)
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
    source = checked_path(path, code="COMMUNITY_BUNDLE_PUBLIC_KEY_INVALID", error=CommunityBundleError)
    if source.is_symlink() or not source.is_file():
        raise CommunityBundleError("COMMUNITY_BUNDLE_PUBLIC_KEY_INVALID")
    try:
        key = serialization.load_pem_public_key(source.read_bytes())
    except (OSError, ValueError, TypeError) as exc:
        raise CommunityBundleError("COMMUNITY_BUNDLE_PUBLIC_KEY_INVALID") from exc
    if not isinstance(key, Ed25519PublicKey):
        raise CommunityBundleError("COMMUNITY_BUNDLE_PUBLIC_KEY_INVALID")
    return key


def _require_cryptography() -> None:
    if Ed25519PrivateKey is None or Ed25519PublicKey is None or serialization is None:
        raise CommunityBundleError(
            "COMMUNITY_BUNDLE_CRYPTOGRAPHY_REQUIRED:run 'uv sync --group dev' to install the signed-community dependency"
        )
