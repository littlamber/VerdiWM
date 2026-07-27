"""Constitutional freeze manifest for verifier-affecting wm-loop state."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from wmloop.contracts import ContractValidationError, load_yaml_document, validate_document
from wmloop.diagnose.probe_registry import ProbeRegistryError, load_probe_registry


class ConstitutionalFreezeError(ValueError):
    """A verifier-affecting freeze file is missing, mutable or mismatched."""


def build_constitutional_freeze(config_path: Path, *, root: Path | None = None) -> dict[str, object]:
    base = (root or Path(__file__).resolve().parents[1]).resolve()
    config_file = _resolve_inside(base, config_path)
    config = json.loads(config_file.read_text(encoding="utf-8"))
    try:
        validate_document("constitutional_config", config, root=base)
    except ContractValidationError as exc:
        raise ConstitutionalFreezeError(f"CONSTITUTION_CONFIG_INVALID:{exc}") from exc
    registry = load_probe_registry(Path(str(config["probe_registry"])), root=base)
    verdict_probe_ids = tuple(str(value) for value in config["verdict_probe_ids"])
    if verdict_probe_ids != registry.verdict_probe_ids:
        raise ConstitutionalFreezeError("CONSTITUTION_VERDICT_PROBES_MISMATCH")
    goal = load_yaml_document(_resolve_inside(base, Path(str(config["goal_spec"]))))
    try:
        validate_document("goal_spec", goal, root=base)
    except ContractValidationError as exc:
        raise ConstitutionalFreezeError(f"CONSTITUTION_GOAL_INVALID:{exc}") from exc
    entries = [
        _freeze_entry(base, "constitution_config", config_file.relative_to(base)),
        _freeze_entry(base, "goal_spec", Path(str(config["goal_spec"]))),
        _freeze_entry(base, "dataset_freeze", Path(str(config["dataset_freeze"]))),
        _freeze_entry(base, "heldout_protocol", Path(str(config["heldout_protocol"]))),
        _freeze_entry(base, "evaluator_freeze", Path(str(config["evaluator_freeze"]))),
        _freeze_entry(base, "probe_registry", Path(str(config["probe_registry"]))),
    ]
    if "horizon_ladder" in config:
        entries.append(_freeze_entry(base, "horizon_ladder", Path(str(config["horizon_ladder"]))))
    frozen_code = config["frozen_code"]
    if not isinstance(frozen_code, Mapping) or not frozen_code:
        raise ConstitutionalFreezeError("CONSTITUTION_FROZEN_CODE_INVALID")
    for component, paths in sorted(frozen_code.items()):
        if not isinstance(paths, list) or not paths:
            raise ConstitutionalFreezeError("CONSTITUTION_FROZEN_CODE_INVALID")
        for raw_path in paths:
            entries.append(_freeze_entry(base, f"code:{component}", Path(str(raw_path))))
    manifest: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "wmloop-constitutional-freeze",
        "constitution_id": str(config["constitution_id"]),
        "goal_id": str(goal["goal_id"]),
        "verdict_probe_ids": list(verdict_probe_ids),
        "entries": entries,
    }
    manifest["constitution_sha256"] = _document_sha256(manifest)
    try:
        validate_document("constitutional_manifest", manifest, root=base)
    except ContractValidationError as exc:
        raise ConstitutionalFreezeError(f"CONSTITUTION_MANIFEST_INVALID:{exc}") from exc
    return manifest


def verify_constitutional_freeze(manifest: Mapping[str, Any], *, root: Path | None = None) -> None:
    base = (root or Path(__file__).resolve().parents[1]).resolve()
    try:
        validate_document("constitutional_manifest", manifest, root=base)
    except ContractValidationError as exc:
        raise ConstitutionalFreezeError(f"CONSTITUTION_MANIFEST_INVALID:{exc}") from exc
    expected_identity = str(manifest["constitution_sha256"])
    without_identity = {key: value for key, value in manifest.items() if key != "constitution_sha256"}
    if _document_sha256(without_identity) != expected_identity:
        raise ConstitutionalFreezeError("CONSTITUTION_IDENTITY_MISMATCH")
    seen: set[str] = set()
    for raw in manifest["entries"]:
        entry = dict(raw)
        path = str(entry["path"])
        if path in seen:
            raise ConstitutionalFreezeError("CONSTITUTION_ENTRY_DUPLICATE")
        seen.add(path)
        actual = _freeze_entry(base, str(entry["component"]), Path(path))
        if actual != entry:
            raise ConstitutionalFreezeError(f"CONSTITUTION_ENTRY_MISMATCH:{path}")


def write_constitutional_freeze(path: Path, manifest: Mapping[str, Any]) -> None:
    verify_constitutional_freeze(manifest, root=Path(__file__).resolve().parents[1])
    _atomic_write_json(path, manifest)


def _freeze_entry(root: Path, component: str, relative_path: Path) -> dict[str, object]:
    path = _resolve_inside(root, relative_path)
    metadata = os.lstat(path)
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ConstitutionalFreezeError(f"CONSTITUTION_ENTRY_UNSAFE:{relative_path}")
    payload = path.read_bytes()
    return {
        "component": component,
        "path": str(path.relative_to(root)),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size": len(payload),
    }


def _resolve_inside(root: Path, path: Path) -> Path:
    resolved = (root / path).resolve() if not path.is_absolute() else path.resolve()
    if root not in resolved.parents and resolved != root:
        raise ConstitutionalFreezeError("CONSTITUTION_PATH_OUTSIDE_ROOT")
    if not resolved.is_file():
        raise ConstitutionalFreezeError(f"CONSTITUTION_PATH_MISSING:{path}")
    return resolved


def _document_sha256(document: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in document.items() if key != "constitution_sha256"}
    content = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create", help="create a deterministic constitutional freeze manifest")
    create.add_argument("--config", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("verify", help="verify an existing constitutional freeze manifest")
    verify.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "create":
        manifest = build_constitutional_freeze(args.config)
        write_constitutional_freeze(args.output, manifest)
        print(json.dumps({"ready": True, "manifest": str(args.output), "constitution_sha256": manifest["constitution_sha256"]}, sort_keys=True))
        return 0
    if args.command == "verify":
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        verify_constitutional_freeze(manifest)
        print(json.dumps({"ready": True, "manifest": str(args.manifest)}, sort_keys=True))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
