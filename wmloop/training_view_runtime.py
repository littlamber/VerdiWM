"""Runtime CLI that materializes metadata-only ACWM training views.

This module is run with the dedicated ACWM Python environment because it must
deserialize and serialize upstream ``metadata.pt`` files.  The project base
environment deliberately has no Torch dependency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Sequence

from wmloop.acwm_data import CANONICAL_ACWM_ENVIRONMENTS
from wmloop.training_view import (
    SplitProjectionReport,
    TrainingViewError,
    canonical_report_sha256,
    project_metadata_rows,
    with_metadata_digest,
)


def create_training_view(*, data_root: Path, output_root: Path, source_revision: str) -> dict[str, Any]:
    """Build an atomic metadata projection over the immutable public release."""

    if len(source_revision) != 40 or any(char not in "0123456789abcdef" for char in source_revision):
        raise TrainingViewError("TRAINING_VIEW_SOURCE_REVISION_INVALID")
    source = Path(data_root).resolve(strict=True)
    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise TrainingViewError("TRAINING_VIEW_OUTPUT_EXISTS")
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    reports: list[SplitProjectionReport] = []
    try:
        temporary.mkdir(mode=0o700)
        import torch

        for spec in CANONICAL_ACWM_ENVIRONMENTS:
            for split, _ in spec.split_sizes:
                split_root = source / spec.dataset_relative_path / split
                metadata_path = split_root / "metadata.pt"
                rows = torch.load(metadata_path, map_location="cpu", weights_only=False)
                if not isinstance(rows, list):
                    raise TrainingViewError("TRAINING_VIEW_METADATA_ROOT_INVALID")
                projected, report = project_metadata_rows(
                    rows,
                    split_root=split_root,
                    allow_missing_references=(spec.environment == "reacher" and split == "ind_train"),
                )
                report = with_metadata_digest(report, metadata_path)
                reports.append(report)
                output_metadata = temporary / spec.dataset_relative_path / split / "metadata.pt"
                output_metadata.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                torch.save(projected, output_metadata)
        manifest = {
            "schema_version": 1,
            "artifact_type": "acwm-derived-training-view",
            "source_data_root": str(source),
            "source_revision": source_revision,
            "metadata_paths_are": "absolute_readonly_source_paths",
            "reports": [report.to_document() for report in reports],
            "reports_sha256": canonical_report_sha256(reports),
        }
        manifest_path = temporary / "training-view-manifest.json"
        manifest_path.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(temporary, destination)
        return manifest
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("create", nargs="?")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    args = parser.parse_args(argv)
    manifest = create_training_view(
        data_root=args.data_root,
        output_root=args.output_root,
        source_revision=args.source_revision,
    )
    print(json.dumps({"ready": True, "reports_sha256": manifest["reports_sha256"], "output_root": str(args.output_root.resolve())}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI only
    raise SystemExit(main())
