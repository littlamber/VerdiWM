#!/usr/bin/env python3
"""Validate an exported VerdiWM example without trusting its manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


class PublicExampleValidationError(RuntimeError):
    """The public example is incomplete, altered, or unsafe to publish."""


def validate_public_example(example_root: Path) -> dict[str, object]:
    root = Path(example_root).resolve(strict=True)
    if not root.is_dir():
        raise PublicExampleValidationError("PUBLIC_EXAMPLE_NOT_DIRECTORY")
    proof_path = root / "minimal-loop-proof.json"
    proof = _load_json(proof_path)
    if proof.get("artifact_type") != "verdiwm-minimal-loop-proof" or proof.get("state") != "ready":
        raise PublicExampleValidationError("PUBLIC_EXAMPLE_PROOF_INVALID")
    if proof.get("operational_minimal_loop_pass") is not True:
        raise PublicExampleValidationError("PUBLIC_EXAMPLE_OPERATIONAL_LOOP_NOT_PROVEN")

    artifacts = proof.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise PublicExampleValidationError("PUBLIC_EXAMPLE_ARTIFACTS_MISSING")
    paths: set[str] = set()
    roles: set[str] = set()
    verified_bytes = 0
    for raw in artifacts:
        if not isinstance(raw, Mapping):
            raise PublicExampleValidationError("PUBLIC_EXAMPLE_ARTIFACT_INVALID")
        relative = raw.get("path")
        role = raw.get("role")
        digest = raw.get("sha256")
        size = raw.get("size_bytes")
        if not isinstance(relative, str) or not relative or not isinstance(role, str) or not role:
            raise PublicExampleValidationError("PUBLIC_EXAMPLE_ARTIFACT_IDENTITY_INVALID")
        if relative in paths or role in roles:
            raise PublicExampleValidationError("PUBLIC_EXAMPLE_ARTIFACT_DUPLICATE")
        paths.add(relative)
        roles.add(role)
        target = _contained_path(root, relative)
        if not target.is_file() or target.is_symlink():
            raise PublicExampleValidationError(f"PUBLIC_EXAMPLE_ARTIFACT_MISSING:{relative}")
        actual_size = target.stat().st_size
        if not isinstance(size, int) or size != actual_size:
            raise PublicExampleValidationError(f"PUBLIC_EXAMPLE_SIZE_MISMATCH:{relative}")
        if not isinstance(digest, str) or digest != _sha256(target):
            raise PublicExampleValidationError(f"PUBLIC_EXAMPLE_SHA256_MISMATCH:{relative}")
        verified_bytes += actual_size

    chain = proof.get("chain_checks")
    if not isinstance(chain, Mapping) or not chain or not all(value is True for value in chain.values()):
        raise PublicExampleValidationError("PUBLIC_EXAMPLE_CHAIN_INVALID")
    independence = proof.get("confirmation_independence")
    if not isinstance(independence, Mapping):
        raise PublicExampleValidationError("PUBLIC_EXAMPLE_INDEPENDENCE_MISSING")
    if independence.get("evaluation_seed_independent") is False and proof.get("paper_confirmed_effect") is not False:
        raise PublicExampleValidationError("PUBLIC_EXAMPLE_SHARED_SEED_OVERCLAIM")

    _scan_public_text(root)
    return {
        "schema_version": 1,
        "artifact_type": "verdiwm-public-example-validation",
        "state": "ready",
        "example": root.name,
        "artifact_count": len(artifacts),
        "verified_bytes": verified_bytes,
        "operational_minimal_loop_pass": True,
        "paper_confirmed_effect": bool(proof.get("paper_confirmed_effect")),
    }


def _contained_path(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise PublicExampleValidationError(f"PUBLIC_EXAMPLE_PATH_UNSAFE:{relative}")
    target = (root / candidate).resolve()
    if target != root and root not in target.parents:
        raise PublicExampleValidationError(f"PUBLIC_EXAMPLE_PATH_UNSAFE:{relative}")
    return target


def _scan_public_text(root: Path) -> None:
    host_prefixes = ("/" + "mnt" + "/", "/" + "root" + "/")
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".json", ".md", ".txt", ".yaml", ".yml"}:
            continue
        text = path.read_text(encoding="utf-8")
        if any(prefix in text for prefix in host_prefixes):
            raise PublicExampleValidationError(f"PUBLIC_EXAMPLE_LOCAL_PATH_LEAK:{path.relative_to(root)}")


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PublicExampleValidationError(f"PUBLIC_EXAMPLE_JSON_INVALID:{path.name}") from exc
    if not isinstance(payload, Mapping):
        raise PublicExampleValidationError(f"PUBLIC_EXAMPLE_JSON_INVALID:{path.name}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("example_root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    print(json.dumps(validate_public_example(args.example_root), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

