"""Smoke the archive cell projection without publishing model-quality trials."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Mapping

from wmloop.archive.store import ArchiveStore, ContentAddressedStore, SettledTrialRecord
from wmloop.propose.scheduler import InterventionCell


class CellProjectionSmokeError(RuntimeError):
    """The archive cell projection smoke failed closed."""


def run_cell_projection_smoke(
    *,
    output_root: Path,
    archive_db: Path | None = None,
    cas_root: Path | None = None,
) -> dict[str, object]:
    """Write evidence that settled single-factor trials update archive cells."""

    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise CellProjectionSmokeError("CELL_PROJECTION_SMOKE_OUTPUT_EXISTS")
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    archive = ArchiveStore(archive_db) if archive_db is not None else None
    cas = ContentAddressedStore(cas_root if cas_root is not None else destination.parent)
    try:
        temporary.mkdir(mode=0o700)
        smoke_archive = ArchiveStore(temporary / "cell-projection.db")
        cell = InterventionCell("push_cube", "L3", "latent_motion_prior", "weight=0.2")
        smoke_archive.record_settled_trial(_record("cell-smoke-trial-1", "cell-smoke-proposal-1", cell=cell, gain=2.0))
        smoke_archive.record_settled_trial(_record("cell-smoke-trial-2", "cell-smoke-proposal-2", cell=cell, gain=-1.0))
        smoke_archive.record_settled_trial(
            _record("cell-smoke-trial-3", "cell-smoke-proposal-3", cell=cell, gain=10.0, exploratory=True)
        )
        cells = [record.to_dict() for record in smoke_archive.list_cells()]
        if len(cells) != 1 or cells[0]["stats"]["visits"] != 2 or cells[0]["stats"]["mean_verified_improvement"] != 0.5:
            raise CellProjectionSmokeError("CELL_PROJECTION_SMOKE_UNEXPECTED_STATS")
        report = {
            "schema_version": 1,
            "artifact_type": "wmloop-archive-cell-projection-smoke-report",
            "state": "ready",
            "smoke_archive_path": str(destination / "cell-projection-smoke.db"),
            "published_to_main_archive": False,
            "settled_trials_in_smoke_archive": smoke_archive.visible_settled_trials(),
            "cell_records": cells,
            "limitations": [
                "This smoke uses an isolated archive database, so it does not change main archive settled-trial counts.",
                "It verifies single-factor cell aggregation only; exploratory multi-axis trials are intentionally excluded.",
            ],
        }
        report_bytes = _canonical_json_bytes(report)
        markdown_bytes = _render_markdown(report).encode("utf-8")
        _write_bytes_atomic(temporary / "cell-projection-smoke.json", report_bytes)
        _write_bytes_atomic(temporary / "cell-projection-smoke.md", markdown_bytes)
        shutil.copy2(temporary / "cell-projection.db", temporary / "cell-projection-smoke.db")
        report_ref = cas.put_bytes(report_bytes, media_type="application/json").uri
        markdown_ref = cas.put_bytes(markdown_bytes, media_type="text/markdown").uri
        db_ref = cas.put_bytes((temporary / "cell-projection-smoke.db").read_bytes(), media_type="application/x-sqlite3").uri
        if archive is not None:
            for ref in (report_ref, markdown_ref, db_ref):
                archive.record_artifact_reference(ref)
        manifest = {
            "schema_version": 1,
            "artifact_type": "wmloop-archive-cell-projection-smoke-manifest",
            "state": "ready",
            "published_to_main_archive": False,
            "settled_trial_count": 3,
            "projected_cell_count": len(cells),
            "report_path": str(destination / "cell-projection-smoke.json"),
            "markdown_path": str(destination / "cell-projection-smoke.md"),
            "smoke_archive_path": str(destination / "cell-projection-smoke.db"),
            "cas_refs": {
                "cell_projection_smoke_json": report_ref,
                "cell_projection_smoke_markdown": markdown_ref,
                "cell_projection_smoke_db": db_ref,
            },
            "limitations": report["limitations"],
        }
        _write_bytes_atomic(temporary / "manifest.json", _canonical_json_bytes(manifest))
        os.replace(temporary, destination)
        return manifest
    except Exception:
        if temporary.is_symlink():
            temporary.unlink()
        elif temporary.exists():
            shutil.rmtree(temporary)
        raise


def _record(
    trial_id: str,
    proposal_id: str,
    *,
    cell: InterventionCell,
    gain: float,
    exploratory: bool = False,
) -> SettledTrialRecord:
    receipt_hash = _sha256_text(f"{trial_id}:receipt")
    return SettledTrialRecord(
        trial_id=trial_id,
        proposal_id=proposal_id,
        goal_id="g1_long_horizon",
        library_version="v1.0",
        failure_context_ref="cas://sha256/" + _sha256_text(f"{trial_id}:failure"),
        verdict_ref="cas://sha256/" + _sha256_text(f"{trial_id}:verdict"),
        receipt_ref="cas://sha256/" + receipt_hash,
        gpu_hours=0.0,
        hypothesis_hash=_sha256_text(f"{proposal_id}:hypothesis"),
        impl_diff_hash=_sha256_text(f"{trial_id}:diff"),
        evaluator_hash=_sha256_text("cell-projection-smoke:evaluator"),
        settlement_state="settled",
        receipt_hash=receipt_hash,
        cell=cell,
        verified_gain=gain,
        exploratory=exploratory,
    )


def _render_markdown(report: Mapping[str, object]) -> str:
    rows = []
    for item in report["cell_records"]:
        cell = item["cell"]
        stats = item["stats"]
        rows.append(
            f"| {cell['environment']} | {cell['layer']} | {cell['primitive_family']} | "
            f"{cell['parameter_bucket']} | {stats['visits']} | {stats['mean_verified_improvement']} |"
        )
    return "\n".join(
        [
            "# Archive Cell Projection Smoke",
            "",
            f"State: `{report['state']}`",
            f"Published to main archive: `{report['published_to_main_archive']}`",
            "",
            "| Env | Layer | Primitive Family | Bucket | Visits | Mean Verified Gain |",
            "|:--|:--|:--|:--|--:|--:|",
            *rows,
            "",
            "## Limitations",
            "",
            *[f"- {item}" for item in report["limitations"]],
            "",
        ]
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise CellProjectionSmokeError("CELL_PROJECTION_SMOKE_OUTPUT_EXISTS")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run archive cell projection smoke")
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--archive-db", type=Path)
    run.add_argument("--cas-root", type=Path)
    args = parser.parse_args(argv)
    if args.command == "run":
        manifest = run_cell_projection_smoke(
            output_root=args.output_root,
            archive_db=args.archive_db,
            cas_root=args.cas_root,
        )
        print(json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
        return 0
    raise CellProjectionSmokeError("CELL_PROJECTION_SMOKE_COMMAND_INVALID")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
