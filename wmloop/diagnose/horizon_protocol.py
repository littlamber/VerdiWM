"""Freeze M1 horizon availability decisions for fail-closed diagnostics."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from wmloop.archive.store import ArchiveStore, ContentAddressedStore
from wmloop.contracts import ContractValidationError, load_yaml_document, validate_document


class HorizonProtocolDecisionError(RuntimeError):
    """A horizon protocol decision report could not be produced safely."""


def generate_horizon_protocol_decision(
    *,
    availability_report_path: Path,
    output_root: Path,
    goal_config: Path = Path("configs/goal/long_horizon_v1.yaml"),
    archive_db: Path | None = None,
    cas_root: Path | None = None,
) -> dict[str, object]:
    """Materialize supported/unavailable horizon decisions from availability."""

    goal = _load_goal(goal_config)
    availability = _load_availability_report(availability_report_path)
    envs = tuple(str(item) for item in goal["envs"])
    horizons = tuple(int(item) for item in goal["horizons"])
    records_by_key = _availability_records_by_key(availability)
    splits = _splits(availability, records_by_key)
    missing: list[dict[str, str]] = []
    records: list[dict[str, object]] = []
    for split in splits:
        for environment in envs:
            availability_record = records_by_key.get((environment, split))
            if availability_record is None:
                missing.append({"environment": environment, "split": split})
                continue
            records.append(_decision_record(environment=environment, split=split, horizons=horizons, availability_record=availability_record))
    unavailable_records = [
        {
            "environment": record["environment"],
            "split": record["split"],
            "unavailable_horizons": record["unavailable_horizons"],
        }
        for record in records
        if record["unavailable_horizons"]
    ]
    unavailable_horizon_count = sum(len(record["unavailable_horizons"]) for record in records)
    report = {
        "schema_version": 1,
        "artifact_type": "wmloop-m1-horizon-protocol-decision",
        "decision_id": "m1-horizon-unavailable-dataset-length-v1",
        "state": "ready" if not missing else "incomplete",
        "goal_id": goal["goal_id"],
        "primary_objective": goal["primary_objective"],
        "required_horizons": list(horizons),
        "availability_report_path": str(Path(availability_report_path).resolve()),
        "goal_config": str(Path(goal_config).resolve()),
        "splits": list(splits),
        "environment_count": len(envs),
        "record_count": len(records),
        "missing_records": missing,
        "unavailable_record_count": len(unavailable_records),
        "unavailable_horizon_count": unavailable_horizon_count,
        "unavailable_records": unavailable_records,
        "policy": {
            "failure_report_contract": "fail_closed_when_required_horizon_unavailable",
            "rerun_policy": "do_not_rerun_dataset_length_limited_horizons_without_new_data_or_goal_revision",
            "formal_auc_policy": "do_not_emit_auc_psnr_16_64_for records with unavailable required horizons",
            "schema_policy": "do_not_relax_failure_report_schema_in_this_decision",
        },
        "records": records,
        "next_actions": _next_actions(unavailable_records, missing),
    }
    markdown = _render_markdown(report)
    return _write_report_bundle(report=report, markdown=markdown, output_root=output_root, archive_db=archive_db, cas_root=cas_root)


def _load_goal(path: Path) -> Mapping[str, Any]:
    try:
        payload = load_yaml_document(Path(path))
        validate_document("goal_spec", payload)
    except (OSError, ContractValidationError) as exc:
        raise HorizonProtocolDecisionError("HORIZON_PROTOCOL_GOAL_INVALID") from exc
    if not isinstance(payload.get("envs"), list) or not isinstance(payload.get("horizons"), list):
        raise HorizonProtocolDecisionError("HORIZON_PROTOCOL_GOAL_INVALID")
    return payload


def _load_availability_report(path: Path) -> Mapping[str, Any]:
    candidate = Path(path)
    try:
        if candidate.is_symlink() or not candidate.is_file():
            raise HorizonProtocolDecisionError("HORIZON_PROTOCOL_AVAILABILITY_INVALID")
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HorizonProtocolDecisionError("HORIZON_PROTOCOL_AVAILABILITY_INVALID") from exc
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1 or payload.get("artifact_type") != "wmloop-horizon-availability-report":
        raise HorizonProtocolDecisionError("HORIZON_PROTOCOL_AVAILABILITY_INVALID")
    if not isinstance(payload.get("records"), list):
        raise HorizonProtocolDecisionError("HORIZON_PROTOCOL_AVAILABILITY_INVALID")
    return payload


def _availability_records_by_key(report: Mapping[str, Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
    records: dict[tuple[str, str], Mapping[str, Any]] = {}
    for item in report["records"]:
        if not isinstance(item, Mapping) or not isinstance(item.get("environment"), str) or not isinstance(item.get("split"), str):
            raise HorizonProtocolDecisionError("HORIZON_PROTOCOL_AVAILABILITY_INVALID")
        key = (str(item["environment"]), str(item["split"]))
        if key in records:
            raise HorizonProtocolDecisionError(f"HORIZON_PROTOCOL_AVAILABILITY_DUPLICATE:{key[0]}:{key[1]}")
        records[key] = item
    return records


def _splits(report: Mapping[str, Any], records_by_key: Mapping[tuple[str, str], Mapping[str, Any]]) -> tuple[str, ...]:
    raw = report.get("splits")
    if isinstance(raw, list) and raw and all(isinstance(item, str) and item for item in raw):
        return tuple(str(item) for item in raw)
    return tuple(sorted({split for _, split in records_by_key}))


def _decision_record(
    *,
    environment: str,
    split: str,
    horizons: Sequence[int],
    availability_record: Mapping[str, Any],
) -> dict[str, object]:
    unsupported = _string_set(availability_record.get("unsupported_horizons"), "HORIZON_PROTOCOL_UNSUPPORTED_INVALID")
    support_counts = _string_int_mapping(availability_record.get("support_counts"), "HORIZON_PROTOCOL_SUPPORT_COUNTS_INVALID")
    required_lengths = _string_int_mapping(availability_record.get("required_output_lengths"), "HORIZON_PROTOCOL_REQUIRED_LENGTHS_INVALID")
    horizon_decisions: list[dict[str, object]] = []
    for horizon in horizons:
        key = str(horizon)
        status = "unavailable_dataset_length" if key in unsupported else "supported"
        horizon_decisions.append(
            {
                "horizon": horizon,
                "status": status,
                "support_count": support_counts.get(key, 0),
                "required_output_length": required_lengths.get(key),
            }
        )
    unavailable = [str(item["horizon"]) for item in horizon_decisions if item["status"] != "supported"]
    return {
        "environment": environment,
        "split": split,
        "availability_state": availability_record.get("state", "unknown"),
        "available_steps_max": availability_record.get("available_steps_max"),
        "unavailable_horizons": unavailable,
        "horizon_decisions": horizon_decisions,
        "failure_report_contract_effect": "blocked" if unavailable else "horizon_fields_supported",
        "rerun_recommendation": "protocol_or_data_revision_required" if unavailable else "raw_probe_measurement_allowed",
    }


def _string_set(value: object, code: str) -> set[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise HorizonProtocolDecisionError(code)
    return set(value)


def _string_int_mapping(value: object, code: str) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise HorizonProtocolDecisionError(code)
    parsed: dict[str, int] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, int) or isinstance(item, bool) or item < 0:
            raise HorizonProtocolDecisionError(code)
        parsed[key] = item
    return parsed


def _next_actions(unavailable_records: Sequence[Mapping[str, object]], missing: Sequence[Mapping[str, str]]) -> list[str]:
    actions: list[str] = []
    if missing:
        actions.append("Regenerate horizon availability so every goal environment/split has a decision record.")
    if unavailable_records:
        actions.extend(
            [
                "Do not rerun dataset-length-limited horizons as a GPU job; the required output length is not supported by current metadata/actions.",
                "Choose a human protocol revision or keep affected formal failure_report records blocked.",
                "Use this decision report as the raw-probe coverage availability authority.",
            ]
        )
    else:
        actions.append("Proceed with raw horizon probe measurement for all required horizons.")
    return actions


def _render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# M1 Horizon Protocol Decision",
        "",
        f"State: `{report['state']}`",
        f"Decision: `{report['decision_id']}`",
        f"Unavailable horizon decisions: `{report['unavailable_horizon_count']}`",
        "",
        "| Environment | Split | Effect | Unavailable Horizons | Rerun Recommendation |",
        "|:--|:--|:--|:--|:--|",
    ]
    for record in report["records"]:
        unavailable = ",".join(record["unavailable_horizons"]) or "none"
        lines.append(
            f"| {record['environment']} | {record['split']} | {record['failure_report_contract_effect']} | {unavailable} | {record['rerun_recommendation']} |"
        )
    lines.extend(["", "## Policy", ""])
    for key, value in report["policy"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Next Actions", ""])
    for action in report["next_actions"]:
        lines.append(f"- {action}")
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
        raise HorizonProtocolDecisionError("HORIZON_PROTOCOL_OUTPUT_EXISTS")
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    report_bytes = _canonical_json_bytes(report)
    markdown_bytes = markdown.encode("utf-8")
    cas_refs: dict[str, str] = {}
    try:
        temporary.mkdir(mode=0o700)
        _write_bytes_atomic(temporary / "horizon-protocol-decision.json", report_bytes)
        _write_bytes_atomic(temporary / "horizon-protocol-decision.md", markdown_bytes)
        archive = ArchiveStore(archive_db) if archive_db is not None else None
        if archive is not None or cas_root is not None:
            root = cas_root if cas_root is not None else Path(archive_db).resolve().parent  # type: ignore[union-attr]
            cas = ContentAddressedStore(Path(root))
            for name, payload, media_type in (
                ("horizon_protocol_decision_json", report_bytes, "application/json"),
                ("horizon_protocol_decision_markdown", markdown_bytes, "text/markdown"),
            ):
                ref = cas.put_bytes(payload, media_type=media_type).uri
                cas_refs[name] = ref
                if archive is not None:
                    archive.record_artifact_reference(ref)
        manifest = {
            "schema_version": 1,
            "artifact_type": "wmloop-m1-horizon-protocol-decision-manifest",
            "state": report["state"],
            "decision_id": report["decision_id"],
            "report_path": str(destination / "horizon-protocol-decision.json"),
            "markdown_path": str(destination / "horizon-protocol-decision.md"),
            "cas_refs": cas_refs,
            "unavailable_record_count": report["unavailable_record_count"],
            "unavailable_horizon_count": report["unavailable_horizon_count"],
            "missing_records": report["missing_records"],
            "next_actions": report["next_actions"],
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
        raise HorizonProtocolDecisionError("HORIZON_PROTOCOL_OUTPUT_EXISTS")
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
    generate = commands.add_parser("generate", help="generate horizon protocol decision")
    generate.add_argument("--availability-report", type=Path, required=True)
    generate.add_argument("--output-root", type=Path, required=True)
    generate.add_argument("--goal-config", type=Path, default=Path("configs/goal/long_horizon_v1.yaml"))
    generate.add_argument("--archive-db", type=Path)
    generate.add_argument("--cas-root", type=Path)
    args = parser.parse_args(argv)
    if args.command == "generate":
        result = generate_horizon_protocol_decision(
            availability_report_path=args.availability_report,
            output_root=args.output_root,
            goal_config=args.goal_config,
            archive_db=args.archive_db,
            cas_root=args.cas_root,
        )
        print(json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
