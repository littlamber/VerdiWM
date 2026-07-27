"""Audit whether dataset trajectories can support requested horizon probes."""

from __future__ import annotations

import argparse
import json
import math
import os
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from wmloop.acwm_data import CANONICAL_ACWM_ENVIRONMENTS
from wmloop.archive.store import ArchiveStore, ContentAddressedStore
from wmloop.diagnose.horizon_runtime import _model_output_length_for_horizon


class HorizonAvailabilityError(RuntimeError):
    """A horizon availability audit failed closed."""


def summarize_horizon_availability(
    *,
    environment: str,
    split: str,
    horizons: Sequence[int],
    entry_lengths: Sequence[int],
    action_counts: Sequence[int],
    temporal_compress_rate: int,
) -> dict[str, object]:
    """Summarize support for each requested horizon from lengths only."""

    if (
        not environment
        or not split
        or not horizons
        or len(entry_lengths) != len(action_counts)
        or not entry_lengths
        or temporal_compress_rate < 1
        or any(horizon < 1 for horizon in horizons)
        or any(length < 0 for length in entry_lengths)
        or any(count < 0 for count in action_counts)
    ):
        raise HorizonAvailabilityError("HORIZON_AVAILABILITY_INPUT_INVALID")
    available_steps = [min(entry, action) if entry > 0 else action for entry, action in zip(entry_lengths, action_counts)]
    required_output_lengths = {
        str(horizon): _model_output_length_for_horizon(int(horizon), temporal_compress_rate)
        for horizon in horizons
    }
    support_counts = {
        str(horizon): sum(1 for steps in available_steps if steps >= required_output_lengths[str(horizon)])
        for horizon in horizons
    }
    unsupported = [str(horizon) for horizon in horizons if support_counts[str(horizon)] == 0]
    supported = [str(horizon) for horizon in horizons if support_counts[str(horizon)] > 0]
    return {
        "environment": environment,
        "split": split,
        "state": "ready" if not unsupported else "limited",
        "trajectory_count": len(available_steps),
        "temporal_compress_rate": temporal_compress_rate,
        "entry_length_min": min(entry_lengths),
        "entry_length_max": max(entry_lengths),
        "action_count_min": min(action_counts),
        "action_count_max": max(action_counts),
        "available_steps_min": min(available_steps),
        "available_steps_max": max(available_steps),
        "required_output_lengths": required_output_lengths,
        "support_counts": support_counts,
        "supported_horizons": supported,
        "unsupported_horizons": unsupported,
    }


def generate_horizon_availability_report(
    *,
    repo_root: Path,
    data_root: Path,
    output_root: Path,
    horizons: Sequence[int] = (16, 32, 48, 64),
    splits: Sequence[str] = ("ind_test", "ood_test"),
    archive_db: Path | None = None,
    cas_root: Path | None = None,
) -> dict[str, object]:
    """Scan ACWM metadata and write a horizon availability report."""

    if not splits or any(not split for split in splits):
        raise HorizonAvailabilityError("HORIZON_AVAILABILITY_SPLITS_INVALID")
    repo = Path(repo_root).resolve(strict=True)
    data = Path(data_root).resolve(strict=True)
    vendor = repo / "vendor" / "ACWM-Phys"
    records: list[dict[str, object]] = []
    for split in splits:
        for spec in CANONICAL_ACWM_ENVIRONMENTS:
            temporal_rate = _temporal_rate(vendor, spec.environment)
            entry_lengths, action_counts, metadata_path = _metadata_lengths(data, spec.dataset_relative_path, split)
            record = summarize_horizon_availability(
                environment=spec.environment,
                split=split,
                horizons=horizons,
                entry_lengths=entry_lengths,
                action_counts=action_counts,
                temporal_compress_rate=temporal_rate,
            )
            record["metadata_path"] = str(metadata_path)
            records.append(record)
    report = {
        "schema_version": 1,
        "artifact_type": "wmloop-horizon-availability-report",
        "state": "ready" if all(record["state"] == "ready" for record in records) else "limited",
        "data_root": str(data),
        "repo_root": str(repo),
        "horizons": [int(item) for item in horizons],
        "splits": list(splits),
        "environment_count": len(CANONICAL_ACWM_ENVIRONMENTS),
        "record_count": len(records),
        "limited_records": [
            {
                "environment": record["environment"],
                "split": record["split"],
                "unsupported_horizons": record["unsupported_horizons"],
                "available_steps_max": record["available_steps_max"],
            }
            for record in records
            if record["state"] != "ready"
        ],
        "records": records,
    }
    markdown = _render_markdown(report)
    return _write_report_bundle(report=report, markdown=markdown, output_root=output_root, archive_db=archive_db, cas_root=cas_root)


def _temporal_rate(vendor_root: Path, environment: str) -> int:
    vendor_environment = "clothmove" if environment == "cloth_move" else environment
    config_path = vendor_root / "configs" / "envs" / f"{vendor_environment}.yaml"
    if config_path.is_symlink() or not config_path.is_file():
        raise HorizonAvailabilityError("HORIZON_AVAILABILITY_CONFIG_MISSING")
    import yaml  # type: ignore[import-not-found]

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise HorizonAvailabilityError("HORIZON_AVAILABILITY_CONFIG_INVALID")
    model_config = payload.get("model_config")
    if not isinstance(model_config, Mapping):
        raise HorizonAvailabilityError("HORIZON_AVAILABILITY_CONFIG_INVALID")
    value = model_config.get("temporal_compress_rate", 4)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise HorizonAvailabilityError("HORIZON_AVAILABILITY_CONFIG_INVALID")
    return value


def _metadata_lengths(data_root: Path, dataset_relative_path: str, split: str) -> tuple[list[int], list[int], Path]:
    metadata_path = data_root / dataset_relative_path / split / "metadata.pt"
    if metadata_path.is_symlink() or not metadata_path.is_file():
        raise HorizonAvailabilityError("HORIZON_AVAILABILITY_METADATA_MISSING")
    import torch  # type: ignore[import-not-found]

    rows = torch.load(metadata_path, map_location="cpu", weights_only=False)
    if not isinstance(rows, list) or not rows:
        raise HorizonAvailabilityError("HORIZON_AVAILABILITY_METADATA_INVALID")
    entry_lengths: list[int] = []
    action_counts: list[int] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise HorizonAvailabilityError("HORIZON_AVAILABILITY_METADATA_INVALID")
        entry_lengths.append(_entry_length(row))
        action_counts.append(_action_count(row))
    return entry_lengths, action_counts, metadata_path


def _entry_length(entry: Mapping[str, Any]) -> int:
    value = entry.get("length")
    if isinstance(value, int) and value >= 0:
        return value
    return 0


def _action_count(entry: Mapping[str, Any]) -> int:
    raw = entry.get("actions")
    if raw is None:
        raw = entry.get("commands")
    if isinstance(raw, Mapping):
        linear = raw.get("linear_velocity")
        if hasattr(linear, "shape") and linear.shape:
            return int(linear.shape[0])
        return 0
    if hasattr(raw, "shape") and raw.shape:
        return int(raw.shape[0])
    return 0


def _render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Horizon Availability Report",
        "",
        f"State: `{report['state']}`",
        "",
        "| Environment | Split | Available Max | Unsupported Horizons | Support Counts |",
        "|:--|:--|--:|:--|:--|",
    ]
    for record in report["records"]:
        counts = ",".join(f"{key}:{value}" for key, value in record["support_counts"].items())
        unsupported = ",".join(record["unsupported_horizons"]) or "none"
        lines.append(
            f"| {record['environment']} | {record['split']} | {record['available_steps_max']} | {unsupported} | {counts} |"
        )
    return "\n".join(lines) + "\n"


def _write_report_bundle(
    *,
    report: Mapping[str, object],
    markdown: str,
    output_root: Path,
    archive_db: Path | None,
    cas_root: Path | None,
) -> dict[str, object]:
    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise HorizonAvailabilityError("HORIZON_AVAILABILITY_OUTPUT_EXISTS")
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    report_bytes = _canonical_json_bytes(report)
    markdown_bytes = markdown.encode("utf-8")
    cas_refs: dict[str, str] = {}
    try:
        temporary.mkdir(mode=0o700)
        _write_bytes_atomic(temporary / "horizon-availability.json", report_bytes)
        _write_bytes_atomic(temporary / "horizon-availability.md", markdown_bytes)
        archive = ArchiveStore(archive_db) if archive_db is not None else None
        if archive is not None or cas_root is not None:
            root = cas_root if cas_root is not None else Path(archive_db).resolve().parent  # type: ignore[union-attr]
            cas = ContentAddressedStore(Path(root))
            for name, payload, media_type in (
                ("horizon_availability_json", report_bytes, "application/json"),
                ("horizon_availability_markdown", markdown_bytes, "text/markdown"),
            ):
                ref = cas.put_bytes(payload, media_type=media_type).uri
                cas_refs[name] = ref
                if archive is not None:
                    archive.record_artifact_reference(ref)
        manifest = {
            "schema_version": 1,
            "artifact_type": "wmloop-horizon-availability-manifest",
            "state": report["state"],
            "report_path": str(destination / "horizon-availability.json"),
            "markdown_path": str(destination / "horizon-availability.md"),
            "cas_refs": cas_refs,
            "limited_records": report["limited_records"],
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


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise HorizonAvailabilityError("HORIZON_AVAILABILITY_OUTPUT_EXISTS")
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
    generate = commands.add_parser("generate", help="generate horizon availability report")
    generate.add_argument("--repo-root", type=Path, default=Path("."))
    generate.add_argument("--data-root", type=Path, required=True)
    generate.add_argument("--output-root", type=Path, required=True)
    generate.add_argument("--horizons", type=int, nargs="+", default=[16, 32, 48, 64])
    generate.add_argument("--splits", nargs="+", default=["ind_test", "ood_test"])
    generate.add_argument("--archive-db", type=Path)
    generate.add_argument("--cas-root", type=Path)
    args = parser.parse_args(argv)
    if args.command == "generate":
        result = generate_horizon_availability_report(
            repo_root=args.repo_root,
            data_root=args.data_root,
            output_root=args.output_root,
            horizons=tuple(args.horizons),
            splits=tuple(args.splits),
            archive_db=args.archive_db,
            cas_root=args.cas_root,
        )
        print(json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
