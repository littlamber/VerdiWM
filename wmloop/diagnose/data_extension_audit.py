"""Audit local ACWM data candidates for preserving the frozen G1 horizon."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from wmloop.acwm_data import CANONICAL_ACWM_ENVIRONMENTS
from wmloop.archive.store import ArchiveStore, ContentAddressedStore


class DataExtensionAuditError(RuntimeError):
    """A data-extension audit failed closed."""


MetadataLoader = Callable[[Path], Sequence[Mapping[str, Any]]]


def generate_data_extension_audit(
    *,
    data_root: Path,
    availability_report_path: Path,
    output_root: Path,
    archive_db: Path | None = None,
    cas_root: Path | None = None,
    metadata_loader: MetadataLoader | None = None,
) -> dict[str, object]:
    """Scan local metadata and write a human-review packet.

    The audit does not mutate the active goal/protocol.  It only answers
    whether currently present non-canonical data could plausibly unblock the
    frozen 16/32/48/64 objective after a human-approved refreeze.
    """

    data = Path(data_root).resolve(strict=True)
    availability_path = Path(availability_report_path).resolve(strict=True)
    availability = json.loads(availability_path.read_text(encoding="utf-8"))
    report = build_data_extension_audit(
        data_root=data,
        availability_report=availability,
        candidate_summaries=scan_metadata_candidates(data, metadata_loader=metadata_loader),
        availability_report_path=availability_path,
    )
    markdown = _render_markdown(report)
    return _write_report_bundle(report=report, markdown=markdown, output_root=output_root, archive_db=archive_db, cas_root=cas_root)


def build_data_extension_audit(
    *,
    data_root: Path,
    availability_report: Mapping[str, Any],
    candidate_summaries: Sequence[Mapping[str, Any]],
    availability_report_path: Path,
) -> dict[str, object]:
    horizons = _int_list(availability_report.get("horizons"), "DATA_EXTENSION_HORIZONS_INVALID")
    required_horizon = max(horizons)
    splits = _str_list(availability_report.get("splits"), "DATA_EXTENSION_SPLITS_INVALID")
    records = availability_report.get("records")
    if not isinstance(records, list) or not records:
        raise DataExtensionAuditError("DATA_EXTENSION_AVAILABILITY_INVALID")

    canonical_paths = {spec.environment: spec.dataset_relative_path for spec in CANONICAL_ACWM_ENVIRONMENTS}
    canonical_path_set = set(canonical_paths.values())
    required_steps = _required_steps_from_availability(records, required_horizon=required_horizon)
    blocked = _blocked_environments(records, required_horizon=required_horizon, splits=splits)
    groups = _group_candidate_summaries(
        candidate_summaries,
        required_horizon=required_horizon,
        required_steps=required_steps,
        required_splits=splits,
    )
    complete_groups = [group for group in groups if group["supports_required_splits"]]
    noncanonical_complete = [group for group in complete_groups if group["dataset_relative_path"] not in canonical_path_set]

    fill_opportunities: list[dict[str, object]] = []
    for environment in blocked:
        for group in noncanonical_complete:
            if _inferred_environment(group["dataset_relative_path"], canonical_paths) != environment:
                continue
            fill_opportunities.append(
                {
                    "environment": environment,
                    "candidate_dataset_relative_path": group["dataset_relative_path"],
                    "candidate_splits": group["splits"],
                    "required_horizon": required_horizon,
                    "required_steps": required_steps,
                    "requires_human_refreeze": True,
                    "runtime_validation_required": True,
                }
            )
    filled = {item["environment"] for item in fill_opportunities}
    unfilled = [environment for environment in blocked if environment not in filled]
    return {
        "schema_version": 1,
        "artifact_type": "wmloop-m1-data-extension-audit",
        "state": "partial_candidate_found" if fill_opportunities else "blocked",
        "active_protocol_changed": False,
        "human_approval_required": True,
        "availability_report_path": str(availability_report_path),
        "data_root": str(data_root),
        "required_horizon": required_horizon,
        "required_steps": required_steps,
        "required_splits": splits,
        "canonical_dataset_paths": canonical_paths,
        "blocked_environments": blocked,
        "fill_opportunities": fill_opportunities,
        "unfilled_blocked_environments": unfilled,
        "preserve_original_g1_feasible_with_current_data": not unfilled and bool(blocked),
        "candidate_group_count": len(groups),
        "complete_candidate_group_count": len(complete_groups),
        "noncanonical_complete_candidate_groups": noncanonical_complete,
        "candidate_groups": groups,
        "notes": [
            "This audit is metadata-only and does not alter the frozen active protocol.",
            "Any noncanonical candidate requires human approval, a new freeze, and runtime validation before formal claims.",
        ],
    }


def scan_metadata_candidates(data_root: Path, *, metadata_loader: MetadataLoader | None = None) -> list[dict[str, object]]:
    loader = metadata_loader or _load_metadata_rows
    root = Path(data_root).resolve(strict=True)
    summaries: list[dict[str, object]] = []
    for metadata_path in sorted(root.glob("*/*/*/metadata.pt")):
        relative = metadata_path.relative_to(root)
        if len(relative.parts) != 4:
            continue
        dataset_relative_path = str(Path(*relative.parts[:2]))
        split = relative.parts[2]
        rows = loader(metadata_path)
        summaries.append(
            summarize_metadata_rows(
                dataset_relative_path=dataset_relative_path,
                split=split,
                metadata_path=metadata_path,
                rows=rows,
            )
        )
    return summaries


def summarize_metadata_rows(
    *,
    dataset_relative_path: str,
    split: str,
    metadata_path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    if not dataset_relative_path or not split or not rows:
        raise DataExtensionAuditError("DATA_EXTENSION_METADATA_INVALID")
    lengths: list[int] = []
    action_counts: list[int] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise DataExtensionAuditError("DATA_EXTENSION_METADATA_INVALID")
        lengths.append(_entry_length(row))
        action_counts.append(_action_count(row))
    if len(lengths) != len(action_counts) or not lengths:
        raise DataExtensionAuditError("DATA_EXTENSION_METADATA_INVALID")
    available_steps = [min(length, action) if length > 0 else action for length, action in zip(lengths, action_counts)]
    return {
        "dataset_relative_path": dataset_relative_path.replace(os.sep, "/"),
        "split": split,
        "metadata_path": str(metadata_path),
        "trajectory_count": len(rows),
        "entry_length_min": min(lengths),
        "entry_length_max": max(lengths),
        "action_count_min": min(action_counts),
        "action_count_max": max(action_counts),
        "available_steps_min": min(available_steps),
        "available_steps_max": max(available_steps),
    }


def _group_candidate_summaries(
    summaries: Sequence[Mapping[str, Any]],
    *,
    required_horizon: int,
    required_steps: int,
    required_splits: Sequence[str],
) -> list[dict[str, object]]:
    groups: dict[str, dict[str, Any]] = {}
    for summary in summaries:
        dataset_path = summary.get("dataset_relative_path")
        split = summary.get("split")
        available_max = summary.get("available_steps_max")
        trajectory_count = summary.get("trajectory_count")
        if not isinstance(dataset_path, str) or not isinstance(split, str) or not isinstance(available_max, int):
            raise DataExtensionAuditError("DATA_EXTENSION_SUMMARY_INVALID")
        group = groups.setdefault(
            dataset_path,
            {
                "dataset_relative_path": dataset_path,
                "supports_required_splits": False,
                "splits": {},
            },
        )
        group["splits"][split] = {
            "available_steps_max": available_max,
            "trajectory_count": trajectory_count,
            "supports_required_horizon": available_max >= required_steps,
            "additional_steps_needed": max(0, required_steps - available_max),
        }
    result: list[dict[str, object]] = []
    for dataset_path, group in sorted(groups.items()):
        split_map = group["splits"]
        group["supports_required_splits"] = all(
            split in split_map and bool(split_map[split]["supports_required_horizon"]) for split in required_splits
        )
        group["inferred_environment"] = _inferred_environment(
            dataset_path,
            {spec.environment: spec.dataset_relative_path for spec in CANONICAL_ACWM_ENVIRONMENTS},
        )
        result.append(group)
    return result


def _required_steps_from_availability(records: Sequence[Mapping[str, Any]], *, required_horizon: int) -> int:
    horizon_key = str(required_horizon)
    values: list[int] = []
    for record in records:
        required_output_lengths = record.get("required_output_lengths")
        if isinstance(required_output_lengths, Mapping):
            value = required_output_lengths.get(horizon_key)
            if isinstance(value, int) and value >= required_horizon:
                values.append(value)
    return max(values) if values else required_horizon + 1


def _blocked_environments(records: Sequence[Mapping[str, Any]], *, required_horizon: int, splits: Sequence[str]) -> list[str]:
    by_env: dict[str, dict[str, bool]] = {}
    horizon_key = str(required_horizon)
    for record in records:
        environment = record.get("environment")
        split = record.get("split")
        support_counts = record.get("support_counts")
        if not isinstance(environment, str) or not isinstance(split, str):
            raise DataExtensionAuditError("DATA_EXTENSION_AVAILABILITY_INVALID")
        supported = False
        if isinstance(support_counts, Mapping):
            count = support_counts.get(horizon_key)
            supported = isinstance(count, int) and count > 0
        else:
            supported_horizons = record.get("supported_horizons")
            if isinstance(supported_horizons, list):
                supported = required_horizon in {int(item) for item in supported_horizons}
        by_env.setdefault(environment, {})[split] = supported
    return sorted(environment for environment, split_map in by_env.items() if not all(split_map.get(split, False) for split in splits))


def _inferred_environment(dataset_relative_path: str, canonical_paths: Mapping[str, str]) -> str | None:
    leaf = Path(dataset_relative_path).name
    normalized_leaf = _normalize_name(leaf)
    for environment, canonical_path in canonical_paths.items():
        aliases = {_normalize_name(environment), _normalize_name(Path(canonical_path).name)}
        if normalized_leaf in aliases:
            return environment
    return None


def _entry_length(entry: Mapping[str, Any]) -> int:
    value = entry.get("length")
    return int(value) if isinstance(value, int) and value >= 0 else 0


def _action_count(entry: Mapping[str, Any]) -> int:
    raw = entry.get("actions")
    if raw is None:
        raw = entry.get("commands")
    if isinstance(raw, Mapping):
        raw = raw.get("linear_velocity")
    if hasattr(raw, "shape") and raw.shape:
        return int(raw.shape[0])
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        return len(raw)
    return 0


def _load_metadata_rows(path: Path) -> Sequence[Mapping[str, Any]]:
    import torch  # type: ignore[import-not-found]

    rows = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(rows, list):
        raise DataExtensionAuditError("DATA_EXTENSION_METADATA_INVALID")
    return rows


def _normalize_name(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


def _int_list(value: object, error: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise DataExtensionAuditError(error)
    result = [int(item) for item in value]
    if any(item < 1 for item in result):
        raise DataExtensionAuditError(error)
    return result


def _str_list(value: object, error: str) -> list[str]:
    if not isinstance(value, list) or not value or any(not isinstance(item, str) or not item for item in value):
        raise DataExtensionAuditError(error)
    return list(value)


def _render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# M1 Data Extension Audit",
        "",
        f"State: `{report['state']}`",
        f"Active protocol changed: `{report['active_protocol_changed']}`",
        f"Preserve original G1 feasible with current data: `{report['preserve_original_g1_feasible_with_current_data']}`",
        "",
        "## Blocked Environments",
        "",
        ", ".join(report["blocked_environments"]) or "none",
        "",
        "## Fill Opportunities",
        "",
        "| Environment | Candidate Path | Splits | Human Refreeze | Runtime Validation |",
        "|:--|:--|:--|:--|:--|",
    ]
    for item in report["fill_opportunities"]:
        split_names = ",".join(sorted(item["candidate_splits"].keys()))
        lines.append(
            f"| {item['environment']} | `{item['candidate_dataset_relative_path']}` | {split_names} | "
            f"{item['requires_human_refreeze']} | {item['runtime_validation_required']} |"
        )
    if not report["fill_opportunities"]:
        lines.append("| none | none | none | n/a | n/a |")
    lines.extend(
        [
            "",
            "## Unfilled Blocked Environments",
            "",
            ", ".join(report["unfilled_blocked_environments"]) or "none",
            "",
            "## Noncanonical Complete Candidate Groups",
            "",
            "| Candidate Path | Inferred Environment | Required Splits Supported |",
            "|:--|:--|:--|",
        ]
    )
    for group in report["noncanonical_complete_candidate_groups"]:
        lines.append(
            f"| `{group['dataset_relative_path']}` | {group['inferred_environment'] or 'none'} | "
            f"{group['supports_required_splits']} |"
        )
    if not report["noncanonical_complete_candidate_groups"]:
        lines.append("| none | none | false |")
    lines.extend(
        [
            "",
            "Notes:",
            "- This audit is metadata-only and does not alter the frozen active protocol.",
            "- Any noncanonical candidate requires human approval, a new freeze, and runtime validation before formal claims.",
        ]
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
        raise DataExtensionAuditError("DATA_EXTENSION_OUTPUT_EXISTS")
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    report_bytes = _canonical_json_bytes(report)
    markdown_bytes = markdown.encode("utf-8")
    cas_refs: dict[str, str] = {}
    try:
        temporary.mkdir(mode=0o700)
        _write_bytes_atomic(temporary / "data-extension-audit.json", report_bytes)
        _write_bytes_atomic(temporary / "data-extension-audit.md", markdown_bytes)
        archive = ArchiveStore(archive_db) if archive_db is not None else None
        if archive is not None or cas_root is not None:
            root = cas_root if cas_root is not None else Path(archive_db).resolve().parent  # type: ignore[union-attr]
            cas = ContentAddressedStore(Path(root))
            for name, payload, media_type in (
                ("data_extension_audit_json", report_bytes, "application/json"),
                ("data_extension_audit_markdown", markdown_bytes, "text/markdown"),
            ):
                ref = cas.put_bytes(payload, media_type=media_type).uri
                cas_refs[name] = ref
                if archive is not None:
                    archive.record_artifact_reference(ref)
        manifest = {
            "schema_version": 1,
            "artifact_type": "wmloop-m1-data-extension-audit-manifest",
            "state": report["state"],
            "active_protocol_changed": report["active_protocol_changed"],
            "human_approval_required": report["human_approval_required"],
            "preserve_original_g1_feasible_with_current_data": report["preserve_original_g1_feasible_with_current_data"],
            "report_path": str(destination / "data-extension-audit.json"),
            "markdown_path": str(destination / "data-extension-audit.md"),
            "cas_refs": cas_refs,
            "fill_opportunity_count": len(report["fill_opportunities"]),  # type: ignore[arg-type]
            "unfilled_blocked_environments": report["unfilled_blocked_environments"],
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
        raise DataExtensionAuditError("DATA_EXTENSION_OUTPUT_EXISTS")
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate", help="scan local metadata for G1-preserving data extension candidates")
    generate.add_argument("--data-root", type=Path, required=True)
    generate.add_argument("--availability-report", type=Path, required=True)
    generate.add_argument("--output-root", type=Path, required=True)
    generate.add_argument("--archive-db", type=Path)
    generate.add_argument("--cas-root", type=Path)
    args = parser.parse_args(argv)
    if args.command == "generate":
        manifest = generate_data_extension_audit(
            data_root=args.data_root,
            availability_report_path=args.availability_report,
            output_root=args.output_root,
            archive_db=args.archive_db,
            cas_root=args.cas_root,
        )
        print(json.dumps(manifest, sort_keys=True))
        return 0
    raise DataExtensionAuditError("DATA_EXTENSION_COMMAND_INVALID")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
