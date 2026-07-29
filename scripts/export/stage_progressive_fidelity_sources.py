#!/usr/bin/env python3
"""Stage the JSON evidence closure referenced by an S6 efficiency report."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


class ProgressiveFidelitySourceStageError(ValueError):
    """The report references missing, unsafe, or inconsistent JSON evidence."""


def stage_progressive_fidelity_sources(
    *, report_path: Path, output_root: Path
) -> dict[str, object]:
    report_file = Path(report_path).resolve(strict=True)
    report = _load_json(report_file)
    if report.get("artifact_type") != "verdiwm-progressive-fidelity-efficiency":
        raise ProgressiveFidelitySourceStageError("S6_SOURCE_REPORT_INVALID")
    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise ProgressiveFidelitySourceStageError("S6_SOURCE_OUTPUT_EXISTS")
    references: list[dict[str, object]] = []
    rows = report.get("candidate_rows")
    if not isinstance(rows, list) or not rows:
        raise ProgressiveFidelitySourceStageError("S6_SOURCE_ROWS_INVALID")
    for row in rows:
        if not isinstance(row, Mapping):
            raise ProgressiveFidelitySourceStageError("S6_SOURCE_ROW_INVALID")
        identity = _identity(row)
        screen = _required_json(row, "screen_manifest")
        gate = _required_json(row, "gate_manifest")
        _add(references, identity, "screen_manifest", screen)
        screen_doc = _load_json(screen)
        _add(references, identity, "screen_report", _json_path(screen_doc, "report_path"))
        _add_if_file(references, identity, "screen_status", screen.parents[2] / "status.json")
        _add(references, identity, "screen_gate_manifest", gate)

        finalizer_value = row.get("confirmation_manifest")
        if not isinstance(finalizer_value, str) or not finalizer_value:
            continue
        finalizer = Path(finalizer_value).resolve(strict=True)
        _add(references, identity, "checkpoint_finalization", finalizer)
        _add_if_file(references, identity, "checkpoint_finalization_manifest", finalizer.parent / "manifest.json")
        finalizer_doc = _load_json(finalizer)
        checkpoint_manifest = _json_path(finalizer_doc, "checkpoint_manifest")
        _add(references, identity, "confirm_checkpoint_manifest", checkpoint_manifest)
        confirm_env_manifest = checkpoint_manifest.parents[1] / "manifest.json"
        _add(references, identity, "confirm_environment_manifest", confirm_env_manifest)
        confirm_env = _load_json(confirm_env_manifest)
        _add(references, identity, "confirm_report", _json_path(confirm_env, "report_path"))
        _add_if_file(references, identity, "confirm_status", confirm_env_manifest.parents[2] / "status.json")
        records = finalizer_doc.get("selection", {}).get("records", [])
        for record in records:
            if not isinstance(record, Mapping):
                continue
            step = int(record.get("checkpoint_step", 0))
            if step >= 800:
                _add(
                    references,
                    identity,
                    f"confirm_gate_step_{step}",
                    _json_path(record, "official_manifest_path"),
                )

    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        unique: dict[str, Path] = {}
        for reference in references:
            source = Path(str(reference["source_path"]))
            digest = str(reference["sha256"])
            target = temporary / "objects" / digest[:2] / f"{digest}.json"
            if digest not in unique:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                unique[digest] = target
            reference["object_ref"] = str(Path("objects") / digest[:2] / f"{digest}.json")
        source_map = {
            "schema_version": 1,
            "artifact_type": "verdiwm-progressive-fidelity-source-closure",
            "state": "ready",
            "study_id": report["study_id"],
            "candidate_count": report["candidate_count"],
            "reference_count": len(references),
            "unique_object_count": len(unique),
            "references": references,
            "exclusions": [
                "Checkpoint tensors are excluded from this JSON closure; their paths and SHA-256 identities remain in the copied receipts.",
                "Videos are excluded because this closure supports cost and transition recomputation, not qualitative claims.",
            ],
        }
        _write_json(temporary / "source-map.json", source_map)
        manifest = {
            "schema_version": 1,
            "artifact_type": "verdiwm-progressive-fidelity-source-closure-manifest",
            "state": "ready",
            "reference_count": len(references),
            "unique_object_count": len(unique),
            "source_map_sha256": _sha256(temporary / "source-map.json"),
        }
        _write_json(temporary / "manifest.json", manifest)
        os.replace(temporary, destination)
        return {**manifest, "manifest_path": str(destination / "manifest.json")}
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _identity(row: Mapping[str, Any]) -> dict[str, object]:
    values = {key: row.get(key) for key in ("environment", "primitive", "seed")}
    if (
        not isinstance(values["environment"], str)
        or not isinstance(values["primitive"], str)
        or not isinstance(values["seed"], int)
    ):
        raise ProgressiveFidelitySourceStageError("S6_SOURCE_IDENTITY_INVALID")
    return values


def _required_json(row: Mapping[str, Any], field: str) -> Path:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise ProgressiveFidelitySourceStageError(f"S6_SOURCE_PATH_MISSING:{field}")
    return Path(value).resolve(strict=True)


def _json_path(row: Mapping[str, Any], field: str) -> Path:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise ProgressiveFidelitySourceStageError(f"S6_SOURCE_PATH_MISSING:{field}")
    path = Path(value).resolve(strict=True)
    if path.suffix != ".json":
        raise ProgressiveFidelitySourceStageError(f"S6_SOURCE_PATH_NOT_JSON:{field}")
    return path


def _add(
    references: list[dict[str, object]], identity: Mapping[str, object], role: str, path: Path
) -> None:
    source = Path(path).resolve(strict=True)
    if source.suffix != ".json" or source.is_symlink() or not source.is_file():
        raise ProgressiveFidelitySourceStageError(f"S6_SOURCE_OBJECT_INVALID:{role}")
    references.append(
        {
            **identity,
            "role": role,
            "source_path": str(source),
            "size_bytes": source.stat().st_size,
            "sha256": _sha256(source),
        }
    )


def _add_if_file(
    references: list[dict[str, object]], identity: Mapping[str, object], role: str, path: Path
) -> None:
    if path.is_file() and not path.is_symlink():
        _add(references, identity, role, path)


def _load_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ProgressiveFidelitySourceStageError(f"S6_SOURCE_JSON_INVALID:{path}")
    return payload


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            stage_progressive_fidelity_sources(
                report_path=args.report,
                output_root=args.output_root,
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
