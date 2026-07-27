"""Torch runtime CLI that writes one exact ACWM evaluation cohort view."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Sequence

from wmloop.acwm_data import CANONICAL_ACWM_ENVIRONMENTS
from wmloop.evaluate.cohort_view import (
    CohortViewError,
    canonical_report_sha256,
    cohort_split,
    project_cohort_metadata_rows,
    trusted_source_directory,
    trusted_source_file,
    with_metadata_digest,
)
from wmloop.evaluate.plan import EvaluationSelection


def create_cohort_view(
    *,
    data_root: Path,
    output_root: Path,
    environment: str,
    cohort: str,
    trajectory_ids: Sequence[str],
    dataset_freeze_sha256: str,
) -> dict[str, Any]:
    """Create an atomic metadata-only view that upstream ``eval.py`` can read."""

    if not _is_sha256(dataset_freeze_sha256):
        raise CohortViewError("COHORT_VIEW_DATASET_FREEZE_INVALID")
    spec = next((item for item in CANONICAL_ACWM_ENVIRONMENTS if item.environment == environment), None)
    if spec is None:
        raise CohortViewError("COHORT_VIEW_ENVIRONMENT_UNKNOWN")
    source = trusted_source_directory(Path(data_root))
    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise CohortViewError("COHORT_VIEW_OUTPUT_EXISTS")
    selection = EvaluationSelection(
        environment=spec.environment,
        vendor_environment="clothmove" if spec.environment == "cloth_move" else spec.environment,
        cohort=cohort,
        trajectory_ids=tuple(trajectory_ids),
    )
    split = cohort_split(cohort)
    source_metadata = trusted_source_file(source / spec.dataset_relative_path / split / "metadata.pt")
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    try:
        temporary.mkdir(mode=0o700, parents=True)
        import torch

        rows = torch.load(source_metadata, map_location="cpu", weights_only=False)
        if not isinstance(rows, list):
            raise CohortViewError("COHORT_VIEW_METADATA_ROOT_INVALID")
        projected, report = project_cohort_metadata_rows(
            rows,
            selection=selection,
            environment_spec=spec,
            source_data_root=source,
        )
        report = with_metadata_digest(report, source_metadata)
        output_metadata = temporary / spec.dataset_relative_path / split / "metadata.pt"
        output_metadata.parent.mkdir(mode=0o700, parents=True)
        torch.save(projected, output_metadata)
        manifest = {
            "schema_version": 1,
            "artifact_type": "acwm-evaluation-cohort-view",
            "source_data_root": str(source),
            "dataset_freeze_sha256": dataset_freeze_sha256,
            "metadata_paths_are": "absolute_readonly_source_paths",
            "report": report.to_document(),
            "report_sha256": canonical_report_sha256(report),
        }
        (temporary / "cohort-view-manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n", encoding="utf-8"
        )
        os.replace(temporary, destination)
        return manifest
    except Exception:
        if temporary.is_symlink():
            temporary.unlink()
        elif temporary.exists():
            shutil.rmtree(temporary)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("create", nargs="?")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--cohort", choices=("ind_dev", "ind_accept", "ood_accept"), required=True)
    parser.add_argument("--trajectory-ids-json", type=Path, required=True)
    parser.add_argument("--dataset-freeze-sha256", required=True)
    args = parser.parse_args(argv)
    try:
        ids = json.loads(args.trajectory_ids_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CohortViewError("COHORT_VIEW_TRAJECTORY_IDS_INVALID") from exc
    if not isinstance(ids, list) or any(not isinstance(item, str) for item in ids):
        raise CohortViewError("COHORT_VIEW_TRAJECTORY_IDS_INVALID")
    manifest = create_cohort_view(
        data_root=args.data_root,
        output_root=args.output_root,
        environment=args.environment,
        cohort=args.cohort,
        trajectory_ids=ids,
        dataset_freeze_sha256=args.dataset_freeze_sha256,
    )
    print(json.dumps({"ready": True, "output_root": str(args.output_root.resolve()), "report_sha256": manifest["report_sha256"]}, sort_keys=True))
    return 0


def _is_sha256(value: str) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
