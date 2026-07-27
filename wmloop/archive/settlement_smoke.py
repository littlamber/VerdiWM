"""M2/M3 archive settlement smoke through the public ArchiveStore API."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Mapping

from wmloop.archive.store import ArchiveStore, ContentAddressedStore, SettledTrialRecord


class ArchiveSettlementSmokeError(RuntimeError):
    """The archive settlement smoke could not publish a trial safely."""


def run_archive_settlement_smoke(
    *,
    output_root: Path,
    archive_db: Path,
    cas_root: Path | None = None,
    trial_id: str = "archive-settlement-smoke-r1",
    proposal_id: str = "archive-settlement-smoke-proposal-r1",
) -> dict[str, object]:
    """Publish one smoke settled trial and write an auditable receipt."""

    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise ArchiveSettlementSmokeError("ARCHIVE_SETTLEMENT_SMOKE_OUTPUT_EXISTS")
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    archive = ArchiveStore(archive_db)
    cas = ContentAddressedStore(cas_root if cas_root is not None else Path(archive_db).resolve().parent)
    before = archive.archive_statistics()
    try:
        temporary.mkdir(mode=0o700)
        failure_ref = cas.put_bytes(_canonical_json_bytes(_failure_context(trial_id)), media_type="application/json")
        verdict_ref = cas.put_bytes(_canonical_json_bytes(_verdict(trial_id)), media_type="application/json")
        receipt_payload = _receipt(trial_id)
        receipt_ref = cas.put_bytes(_canonical_json_bytes(receipt_payload), media_type="application/json")
        record = SettledTrialRecord(
            trial_id=trial_id,
            proposal_id=proposal_id,
            goal_id="g1_long_horizon",
            library_version="v1.0",
            failure_context_ref=failure_ref.uri,
            verdict_ref=verdict_ref.uri,
            receipt_ref=receipt_ref.uri,
            gpu_hours=0.0,
            hypothesis_hash=_sha256_text(f"{proposal_id}:hypothesis"),
            impl_diff_hash=_sha256_text(f"{trial_id}:no-diff"),
            evaluator_hash=_sha256_text("archive-settlement-smoke:evaluator"),
            settlement_state="settled",
            receipt_hash=receipt_ref.sha256,
        )
        archive.record_settled_trial(record)
        after_settlement = archive.archive_statistics()
        if trial_id not in archive.visible_settled_trials():
            raise ArchiveSettlementSmokeError("ARCHIVE_SETTLEMENT_SMOKE_TRIAL_NOT_VISIBLE")
        report = {
            "schema_version": 1,
            "artifact_type": "wmloop-m2m3-archive-settlement-smoke-report",
            "state": "ready",
            "trial_id": trial_id,
            "proposal_id": proposal_id,
            "archive_db": str(Path(archive_db).resolve()),
            "cas_root": str((cas_root if cas_root is not None else Path(archive_db).resolve().parent).resolve()),
            "archive_statistics_before": before,
            "archive_statistics_after_settlement": after_settlement,
            "visible_settled_trials": list(archive.visible_settled_trials()),
            "settled_trial_record": record.to_dict(),
            "cas_refs": {
                "failure_context": failure_ref.uri,
                "verdict": verdict_ref.uri,
                "receipt": receipt_ref.uri,
            },
            "limitations": [
                "This is a smoke settled trial for archive table linkage; it is not a model-quality experiment.",
            ],
        }
        report_bytes = _canonical_json_bytes(report)
        markdown_bytes = _render_markdown(report).encode("utf-8")
        report_ref = cas.put_bytes(report_bytes, media_type="application/json")
        markdown_ref = cas.put_bytes(markdown_bytes, media_type="text/markdown")
        archive.record_artifact_reference(report_ref.uri)
        archive.record_artifact_reference(markdown_ref.uri)
        after_report_index = archive.archive_statistics()
        _write_bytes_atomic(temporary / "archive-settlement-smoke.json", report_bytes)
        _write_bytes_atomic(temporary / "archive-settlement-smoke.md", markdown_bytes)
        manifest = {
            "schema_version": 1,
            "artifact_type": "wmloop-m2m3-archive-settlement-smoke-manifest",
            "state": "ready",
            "trial_id": trial_id,
            "proposal_id": proposal_id,
            "settled_trials_before": before["settled_trials"],
            "settled_trials_after": after_report_index["settled_trials"],
            "artifacts_after": after_report_index["artifacts"],
            "report_path": str(destination / "archive-settlement-smoke.json"),
            "markdown_path": str(destination / "archive-settlement-smoke.md"),
            "cas_refs": {
                **report["cas_refs"],
                "archive_settlement_report_json": report_ref.uri,
                "archive_settlement_report_markdown": markdown_ref.uri,
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
            import shutil

            shutil.rmtree(temporary)
        raise


def _failure_context(trial_id: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_type": "wmloop-archive-settlement-smoke-failure-context",
        "trial_id": trial_id,
        "goal_id": "g1_long_horizon",
        "source": "archive_settlement_smoke",
    }


def _verdict(trial_id: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_type": "wmloop-archive-settlement-smoke-verdict",
        "trial_id": trial_id,
        "verdict": "REJECT",
        "gates": {"archive_linkage_smoke": "pass"},
    }


def _receipt(trial_id: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_type": "wmloop-archive-settlement-smoke-receipt",
        "trial_id": trial_id,
        "settlement_state": "settled",
        "gpu_hours": 0.0,
    }


def _render_markdown(report: Mapping[str, object]) -> str:
    before = report["archive_statistics_before"]
    after = report["archive_statistics_after_settlement"]
    return "\n".join(
        [
            "# Archive Settlement Smoke",
            "",
            f"State: `{report['state']}`",
            f"Trial: `{report['trial_id']}`",
            f"Settled trials: `{before['settled_trials']} -> {after['settled_trials']}`",
            f"Artifacts after settlement: `{after['artifacts']}`",
            "",
            "## Limitation",
            "",
            "- This is a smoke settled trial for archive table linkage; it is not a model-quality experiment.",
        ]
    ) + "\n"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise ArchiveSettlementSmokeError("ARCHIVE_SETTLEMENT_SMOKE_OUTPUT_EXISTS")
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
    run = commands.add_parser("run", help="run archive settlement smoke")
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--archive-db", type=Path, required=True)
    run.add_argument("--cas-root", type=Path)
    run.add_argument("--trial-id", default="archive-settlement-smoke-r1")
    run.add_argument("--proposal-id", default="archive-settlement-smoke-proposal-r1")
    args = parser.parse_args(argv)
    if args.command == "run":
        manifest = run_archive_settlement_smoke(
            output_root=args.output_root,
            archive_db=args.archive_db,
            cas_root=args.cas_root,
            trial_id=args.trial_id,
            proposal_id=args.proposal_id,
        )
        print(json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
