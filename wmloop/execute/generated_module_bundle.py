#!/usr/bin/env python3
"""Apply one trusted automatic-module bundle inside an isolated snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path, PurePosixPath


class GeneratedModuleBundleError(RuntimeError):
    """A generated module bundle failed its immutable binding."""


def apply_bundle(
    *,
    bundle_path: Path,
    workspace: Path,
    descriptor_path: Path,
    candidate_id: str,
    idea_id: str,
) -> None:
    bundle = _load(bundle_path)
    if bundle.get("artifact_type") != "verdiwm-automatic-module-bundle":
        raise GeneratedModuleBundleError("GENERATED_MODULE_BUNDLE_TYPE_INVALID")
    if bundle.get("bundle_digest") != _digest(bundle, excluded="bundle_digest"):
        raise GeneratedModuleBundleError("GENERATED_MODULE_BUNDLE_DIGEST_MISMATCH")
    if bundle.get("candidate_id") != candidate_id or bundle.get("idea_id") != idea_id:
        raise GeneratedModuleBundleError("GENERATED_MODULE_BUNDLE_BINDING_MISMATCH")
    expected_descriptor = _member(workspace, str(bundle["descriptor_path"]))
    if descriptor_path.resolve() != expected_descriptor:
        raise GeneratedModuleBundleError("GENERATED_MODULE_DESCRIPTOR_PATH_MISMATCH")
    module_source = str(bundle["module_source"])
    test_source = str(bundle["test_source"])
    if hashlib.sha256(module_source.encode("utf-8")).hexdigest() != bundle.get(
        "module_source_sha256"
    ):
        raise GeneratedModuleBundleError("GENERATED_MODULE_SOURCE_HASH_MISMATCH")
    if hashlib.sha256(test_source.encode("utf-8")).hexdigest() != bundle.get(
        "test_source_sha256"
    ):
        raise GeneratedModuleBundleError("GENERATED_MODULE_TEST_HASH_MISMATCH")
    targets = {
        _member(workspace, str(bundle["module_path"])): module_source,
        _member(workspace, str(bundle["test_path"])): test_source,
        expected_descriptor: json.dumps(
            bundle["descriptor"], sort_keys=True, ensure_ascii=True, separators=(",", ":")
        )
        + "\n",
    }
    if any(path.exists() or path.is_symlink() for path in targets):
        raise GeneratedModuleBundleError("GENERATED_MODULE_TARGET_EXISTS")
    for path, source in targets.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")


def _member(workspace: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or str(pure) != relative:
        raise GeneratedModuleBundleError("GENERATED_MODULE_MEMBER_INVALID")
    root = workspace.resolve()
    path = (root / relative).resolve()
    if root not in path.parents:
        raise GeneratedModuleBundleError("GENERATED_MODULE_MEMBER_INVALID")
    return path


def _load(path: Path) -> dict[str, object]:
    source = path.expanduser().resolve()
    if source.is_symlink() or not source.is_file():
        raise GeneratedModuleBundleError("GENERATED_MODULE_BUNDLE_INVALID")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GeneratedModuleBundleError("GENERATED_MODULE_BUNDLE_INVALID") from exc
    if not isinstance(payload, dict):
        raise GeneratedModuleBundleError("GENERATED_MODULE_BUNDLE_INVALID")
    return payload


def _digest(document: Mapping[str, object], *, excluded: str) -> str:
    payload = {key: value for key, value in document.items() if key != excluded}
    body = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--descriptor-path", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--idea-id", required=True)
    args = parser.parse_args()
    apply_bundle(
        bundle_path=args.bundle,
        workspace=args.workspace,
        descriptor_path=args.descriptor_path,
        candidate_id=args.candidate_id,
        idea_id=args.idea_id,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
