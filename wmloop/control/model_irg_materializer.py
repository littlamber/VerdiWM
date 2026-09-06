"""Materialize one immutable model-conditioned IRG from measured probe assets."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from wmloop.experiments._artifacts import (
    ExperimentArtifactError,
    canonical_json,
    write_bundle,
)
from wmloop.geometry.model_irg import ModelIRGError, build_model_irg, validate_model_irg


class ModelIRGMaterializationError(RuntimeError):
    """The portrait/measurement binding cannot be materialized or resumed."""


def materialize_model_irg(
    *,
    portrait_path: Path,
    irg_asset_path: Path,
    output_root: Path,
    diagnostic_axes_path: Path | None = None,
    method_effects_path: Path | None = None,
    collision_refs: Sequence[str] = (),
    evolution_refs: Sequence[str] = (),
    evidence_refs: Sequence[str] = (),
    repo_root: Path | None = None,
    archive_db: Path | None = None,
    cas_root: Path | None = None,
) -> dict[str, object]:
    """Bind real probe responses to a model portrait with hash-locked resume."""

    root = Path(repo_root or Path(__file__).resolve().parents[2]).resolve()
    portrait_file = _file(portrait_path, "MODEL_IRG_PORTRAIT_INPUT_INVALID")
    asset_file = _file(irg_asset_path, "MODEL_IRG_ASSET_INPUT_INVALID")
    axes_file = _optional_file(
        diagnostic_axes_path, "MODEL_IRG_DIAGNOSTIC_AXES_INPUT_INVALID"
    )
    effects_file = _optional_file(
        method_effects_path, "MODEL_IRG_METHOD_EFFECTS_INPUT_INVALID"
    )
    portrait = _mapping(portrait_file, "MODEL_IRG_PORTRAIT_INPUT_INVALID")
    asset = _mapping(asset_file, "MODEL_IRG_ASSET_INPUT_INVALID")
    axes = _rows(axes_file, "diagnostic_axes", "MODEL_IRG_DIAGNOSTIC_AXES_INPUT_INVALID")
    effects = _rows(effects_file, "method_effects", "MODEL_IRG_METHOD_EFFECTS_INPUT_INVALID")
    normalized_refs = {
        "collision_refs": _strings(collision_refs, "MODEL_IRG_COLLISION_REF_INVALID"),
        "evolution_refs": _strings(evolution_refs, "MODEL_IRG_EVOLUTION_REF_INVALID"),
        "evidence_refs": _strings(evidence_refs, "MODEL_IRG_EVIDENCE_REF_INVALID"),
    }
    input_lock: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "verdiwm-model-irg-materialization-input-lock",
        "state": "frozen",
        "portrait_path": str(portrait_file),
        "portrait_sha256": _sha256_file(portrait_file),
        "irg_asset_path": str(asset_file),
        "irg_asset_sha256": _sha256_file(asset_file),
        "diagnostic_axes_path": str(axes_file) if axes_file is not None else None,
        "diagnostic_axes_sha256": _sha256_file(axes_file) if axes_file is not None else None,
        "method_effects_path": str(effects_file) if effects_file is not None else None,
        "method_effects_sha256": _sha256_file(effects_file) if effects_file is not None else None,
        **normalized_refs,
        "claim_boundary": (
            "This lock freezes materialization inputs only; it is not a model-quality "
            "or probe-admission result."
        ),
    }
    input_hash = _digest(input_lock)
    input_lock["input_hash"] = input_hash
    destination = Path(output_root).expanduser().resolve()
    resumed = _resume(destination, input_hash=input_hash, root=root)
    if resumed is not None:
        return resumed
    try:
        model_irg = build_model_irg(
            portrait=portrait,
            asset=asset,
            diagnostic_axes=axes,
            method_effects=effects,
            collision_refs=normalized_refs["collision_refs"],
            evolution_refs=normalized_refs["evolution_refs"],
            evidence_refs=normalized_refs["evidence_refs"],
            root=root,
        )
    except ModelIRGError as exc:
        raise ModelIRGMaterializationError(f"MODEL_IRG_BUILD_INVALID:{exc}") from exc
    try:
        return write_bundle(
            output_root=destination,
            files={
                "input-lock.json": canonical_json(input_lock),
                "model-irg.json": canonical_json(model_irg),
            },
            manifest_fields={
                "artifact_type": "verdiwm-model-irg-materialization-manifest",
                "state": "ready",
                "input_hash": input_hash,
                "irg_id": model_irg["irg_id"],
                "model_family": model_irg["portrait_binding"]["model_family"],
                "routing_state": model_irg["routing_state"],
                "model_irg_path": str(destination / "model-irg.json"),
                "authority": "diagnostic_ranking_only",
                "claim_boundary": model_irg["claim_boundary"],
            },
            archive_db=archive_db,
            cas_root=cas_root,
        )
    except ExperimentArtifactError as exc:
        raise ModelIRGMaterializationError(f"MODEL_IRG_OUTPUT_INVALID:{exc}") from exc


def _resume(
    destination: Path, *, input_hash: str, root: Path
) -> dict[str, object] | None:
    if not destination.exists() and not destination.is_symlink():
        return None
    if destination.is_symlink() or not destination.is_dir():
        raise ModelIRGMaterializationError("MODEL_IRG_OUTPUT_INVALID")
    lock = _mapping(
        destination / "input-lock.json", "MODEL_IRG_INPUT_LOCK_INVALID"
    )
    if lock.get("input_hash") != input_hash:
        raise ModelIRGMaterializationError("MODEL_IRG_INPUT_LOCK_MISMATCH")
    manifest = _mapping(
        destination / "manifest.json", "MODEL_IRG_MANIFEST_INVALID"
    )
    if manifest.get("input_hash") != input_hash or manifest.get("state") != "ready":
        raise ModelIRGMaterializationError("MODEL_IRG_MANIFEST_INVALID")
    irg = _mapping(destination / "model-irg.json", "MODEL_IRG_OUTPUT_INVALID")
    try:
        validate_model_irg(irg, root=root)
    except ModelIRGError as exc:
        raise ModelIRGMaterializationError("MODEL_IRG_OUTPUT_INVALID") from exc
    if manifest.get("irg_id") != irg.get("irg_id"):
        raise ModelIRGMaterializationError("MODEL_IRG_MANIFEST_BINDING_MISMATCH")
    return manifest


def _rows(path: Path | None, key: str, code: str) -> tuple[dict[str, object], ...]:
    if path is None:
        return ()
    payload = _json(path, code)
    value = payload.get(key) if isinstance(payload, Mapping) else payload
    if not isinstance(value, list) or any(not isinstance(row, Mapping) for row in value):
        raise ModelIRGMaterializationError(code)
    return tuple(dict(row) for row in value)


def _strings(values: Sequence[str], code: str) -> tuple[str, ...]:
    normalized = tuple(dict.fromkeys(str(value).strip() for value in values))
    if any(not value for value in normalized):
        raise ModelIRGMaterializationError(code)
    return normalized


def _mapping(path: Path, code: str) -> dict[str, object]:
    payload = _json(path, code)
    if not isinstance(payload, dict):
        raise ModelIRGMaterializationError(code)
    return payload


def _json(path: Path, code: str) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelIRGMaterializationError(code) from exc


def _file(path: Path, code: str) -> Path:
    value = Path(path).expanduser().resolve()
    if value.is_symlink() or not value.is_file():
        raise ModelIRGMaterializationError(code)
    return value


def _optional_file(path: Path | None, code: str) -> Path | None:
    return None if path is None else _file(path, code)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(payload: object) -> str:
    return hashlib.sha256(canonical_json(payload).rstrip(b"\n")).hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--portrait", type=Path, required=True)
    parser.add_argument("--irg-asset", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--diagnostic-axes", type=Path)
    parser.add_argument("--method-effects", type=Path)
    parser.add_argument("--collision-ref", action="append", default=[])
    parser.add_argument("--evolution-ref", action="append", default=[])
    parser.add_argument("--evidence-ref", action="append", default=[])
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--archive-db", type=Path)
    parser.add_argument("--cas-root", type=Path)
    args = parser.parse_args(argv)
    try:
        manifest = materialize_model_irg(
            portrait_path=args.portrait,
            irg_asset_path=args.irg_asset,
            output_root=args.output_root,
            diagnostic_axes_path=args.diagnostic_axes,
            method_effects_path=args.method_effects,
            collision_refs=args.collision_ref,
            evolution_refs=args.evolution_ref,
            evidence_refs=args.evidence_ref,
            repo_root=args.repo_root,
            archive_db=args.archive_db,
            cas_root=args.cas_root,
        )
    except ModelIRGMaterializationError as exc:
        print(str(exc), file=__import__("sys").stderr)
        return 2
    print(json.dumps(manifest, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
