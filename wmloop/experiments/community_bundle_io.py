"""Community bundle io implementation."""

from __future__ import annotations
import json
import os
import re
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from wmloop.storage import require_exact_tree
from .community_signing import CommunityBundleError


_SAFE_MEMBER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*(?:/[A-Za-z0-9][A-Za-z0-9_.-]*)*$")


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
                    if (candidate.is_symlink() or not candidate.is_file() or candidate.read_bytes() != payload):
                        raise CommunityBundleError("COMMUNITY_BUNDLE_OUTPUT_CONFLICT")
                require_exact_tree(destination, set(payloads), code="COMMUNITY_BUNDLE_OUTPUT_CONFLICT", error=CommunityBundleError)
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
