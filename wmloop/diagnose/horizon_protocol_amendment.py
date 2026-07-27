"""Draft human-reviewable horizon protocol amendment candidates."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from wmloop.archive.store import ArchiveStore, ContentAddressedStore
from wmloop.contracts import ContractValidationError, load_yaml_document, validate_document


class HorizonProtocolAmendmentError(RuntimeError):
    """Horizon protocol amendment candidates could not be produced."""


def generate_horizon_protocol_amendment(
    *,
    availability_report_path: Path,
    output_root: Path,
    goal_config: Path = Path("configs/goal/long_horizon_v1.yaml"),
    raw_failure_batch_manifest: Path | None = None,
    strict_required_reports: int = 3,
    archive_db: Path | None = None,
    cas_root: Path | None = None,
) -> dict[str, object]:
    """Write amendment candidates without mutating the active goal protocol."""

    if strict_required_reports < 1:
        raise HorizonProtocolAmendmentError("HORIZON_AMENDMENT_REQUIRED_COUNT_INVALID")
    goal = _load_goal(goal_config)
    availability = _load_availability_report(availability_report_path)
    raw_failure = _load_optional_raw_failure(raw_failure_batch_manifest)
    envs = tuple(str(item) for item in goal["envs"])
    horizons = tuple(int(item) for item in goal["horizons"])
    records_by_key = _records_by_key(availability)
    split_names = _splits(availability, records_by_key)
    support = {
        environment: _environment_support(environment, split_names=split_names, records_by_key=records_by_key)
        for environment in envs
    }
    options = _options(
        envs=envs,
        horizons=horizons,
        support=support,
        strict_required_reports=strict_required_reports,
    )
    report = {
        "schema_version": 1,
        "artifact_type": "wmloop-m1-horizon-protocol-amendment-candidates",
        "state": "ready",
        "active_protocol_changed": False,
        "human_approval_required": True,
        "goal_id": goal["goal_id"],
        "primary_objective": goal["primary_objective"],
        "goal_config_path": str(Path(goal_config).resolve()),
        "availability_report_path": str(Path(availability_report_path).resolve()),
        "raw_failure_batch_manifest_path": str(Path(raw_failure_batch_manifest).resolve()) if raw_failure_batch_manifest is not None else None,
        "strict_required_reports": strict_required_reports,
        "current_raw_failure_report_state": raw_failure,
        "environment_support": support,
        "candidate_options": options,
        "recommendations": _recommendations(options),
        "limitations": [
            "This artifact is a human-review draft; it does not modify the frozen goal config or active verdict protocol.",
            "Any option that changes horizons changes the claim scope and must not be mixed with current auc_psnr_16_64 results.",
            "The data-extension option is the only option that preserves the current 16/32/48/64 G1 metric definition.",
        ],
    }
    return _write_report_bundle(report=report, output_root=output_root, archive_db=archive_db, cas_root=cas_root)


def _options(
    *,
    envs: Sequence[str],
    horizons: Sequence[int],
    support: Mapping[str, Mapping[str, Any]],
    strict_required_reports: int,
) -> list[dict[str, object]]:
    current_supported = _supported_envs(envs, horizons, support)
    common_all = _common_supported_horizons(envs, support)
    longest_t31 = _longest_prefix_meeting_count(
        envs=envs,
        horizons=horizons,
        support=support,
        minimum_count=strict_required_reports,
    )
    options = [
        {
            "option_id": "keep_current_frozen_protocol",
            "human_action": "keep current G1 horizon protocol unchanged",
            "active_protocol_changed": False,
            "horizons": list(horizons),
            "primary_objective": f"auc_psnr_{min(horizons)}_{max(horizons)}",
            "supported_environment_count": len(current_supported),
            "supported_environments": current_supported,
            "blocked_environments": [env for env in envs if env not in current_supported],
            "strict_t3_1_input_feasible": len(current_supported) >= strict_required_reports,
            "claim_scope": "original_g1_long_horizon",
            "risk": "strict T3.1 remains blocked if fewer than the required real failure reports are available",
        },
        {
            "option_id": "common_horizon_all_environments",
            "human_action": "revise goal horizons to the longest common supported horizon set for all environments",
            "active_protocol_changed": True,
            "horizons": common_all,
            "primary_objective": _auc_name(common_all),
            "supported_environment_count": len(envs) if common_all else 0,
            "supported_environments": list(envs) if common_all else [],
            "blocked_environments": [] if common_all else list(envs),
            "strict_t3_1_input_feasible": bool(common_all) and len(envs) >= strict_required_reports,
            "claim_scope": "revised_shorter_horizon_all_envs",
            "risk": "max horizon is shorter than G1; results are not comparable to auc_psnr_16_64",
        },
        {
            "option_id": "longest_common_prefix_meeting_t3_1",
            "human_action": "revise horizons to the longest original-prefix horizon set that supports at least the T3.1 report count",
            "active_protocol_changed": True,
            "horizons": longest_t31["horizons"],
            "primary_objective": _auc_name(longest_t31["horizons"]),
            "supported_environment_count": longest_t31["supported_environment_count"],
            "supported_environments": longest_t31["supported_environments"],
            "blocked_environments": [env for env in envs if env not in longest_t31["supported_environments"]],
            "strict_t3_1_input_feasible": longest_t31["supported_environment_count"] >= strict_required_reports,
            "claim_scope": "revised_shorter_horizon_subset",
            "risk": "can unblock proposal-generation count faster, but does not preserve 8-env G1 long-horizon coverage",
        },
        {
            "option_id": "per_environment_max_horizon",
            "human_action": "revise protocol to use each environment's maximum supported original-prefix horizon set",
            "active_protocol_changed": True,
            "horizons_by_environment": {env: support[env]["supported_horizons_all_splits"] for env in envs},
            "supported_environment_count": len([env for env in envs if support[env]["supported_horizons_all_splits"]]),
            "supported_environments": [env for env in envs if support[env]["supported_horizons_all_splits"]],
            "blocked_environments": [env for env in envs if not support[env]["supported_horizons_all_splits"]],
            "strict_t3_1_input_feasible": len([env for env in envs if support[env]["supported_horizons_all_splits"]]) >= strict_required_reports,
            "claim_scope": "nonuniform_metric_by_environment",
            "risk": "requires schema/reporting changes because a single auc_psnr_16_64 objective no longer describes every environment",
        },
        {
            "option_id": "data_extension_preserve_g1",
            "human_action": "collect or locate longer trajectories so every split supports the current required horizons",
            "active_protocol_changed": False,
            "horizons": list(horizons),
            "primary_objective": f"auc_psnr_{min(horizons)}_{max(horizons)}",
            "required_data_extensions": _required_data_extensions(envs=envs, horizons=horizons, support=support),
            "strict_t3_1_input_feasible": True,
            "claim_scope": "original_g1_long_horizon_after_new_data",
            "risk": "requires new data or metadata, then freeze and rerun M0/M1 evidence before formal claims",
        },
    ]
    return options


def _environment_support(
    environment: str,
    *,
    split_names: Sequence[str],
    records_by_key: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    per_split: dict[str, dict[str, object]] = {}
    split_sets: list[set[int]] = []
    for split in split_names:
        record = records_by_key.get((environment, split))
        if record is None:
            per_split[split] = {
                "state": "missing",
                "supported_horizons": [],
                "unsupported_horizons": [],
                "available_steps_max": None,
                "required_output_lengths": {},
            }
            split_sets.append(set())
            continue
        supported = _int_list(record.get("supported_horizons"), "HORIZON_AMENDMENT_AVAILABILITY_INVALID")
        unsupported = _int_list(record.get("unsupported_horizons"), "HORIZON_AMENDMENT_AVAILABILITY_INVALID")
        per_split[split] = {
            "state": record.get("state"),
            "supported_horizons": supported,
            "unsupported_horizons": unsupported,
            "available_steps_max": record.get("available_steps_max"),
            "required_output_lengths": record.get("required_output_lengths", {}),
        }
        split_sets.append(set(supported))
    common = sorted(set.intersection(*split_sets)) if split_sets else []
    return {
        "splits": per_split,
        "supported_horizons_all_splits": common,
        "max_supported_horizon_all_splits": max(common) if common else None,
    }


def _supported_envs(envs: Sequence[str], horizons: Sequence[int], support: Mapping[str, Mapping[str, Any]]) -> list[str]:
    required = set(int(item) for item in horizons)
    return [
        env
        for env in envs
        if required <= set(int(item) for item in support[env]["supported_horizons_all_splits"])
    ]


def _common_supported_horizons(envs: Sequence[str], support: Mapping[str, Mapping[str, Any]]) -> list[int]:
    sets = [set(int(item) for item in support[env]["supported_horizons_all_splits"]) for env in envs]
    return sorted(set.intersection(*sets)) if sets else []


def _longest_prefix_meeting_count(
    *,
    envs: Sequence[str],
    horizons: Sequence[int],
    support: Mapping[str, Mapping[str, Any]],
    minimum_count: int,
) -> dict[str, Any]:
    best = {"horizons": [], "supported_environment_count": 0, "supported_environments": []}
    for end in range(1, len(horizons) + 1):
        candidate = list(horizons[:end])
        supported = _supported_envs(envs, candidate, support)
        if len(supported) >= minimum_count and len(candidate) >= len(best["horizons"]):
            best = {
                "horizons": candidate,
                "supported_environment_count": len(supported),
                "supported_environments": supported,
            }
    return best


def _required_data_extensions(
    *,
    envs: Sequence[str],
    horizons: Sequence[int],
    support: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, object]]:
    required_lengths_by_horizon: dict[int, int] = {}
    for env in envs:
        for split_record in support[env]["splits"].values():
            raw = split_record.get("required_output_lengths", {})
            if not isinstance(raw, Mapping):
                continue
            for horizon in horizons:
                value = raw.get(str(horizon))
                if isinstance(value, int) and not isinstance(value, bool):
                    required_lengths_by_horizon[horizon] = max(required_lengths_by_horizon.get(horizon, 0), value)
    max_required_length = max(required_lengths_by_horizon.values()) if required_lengths_by_horizon else max(horizons)
    rows = []
    for env in envs:
        split_rows = []
        for split, split_record in support[env]["splits"].items():
            available = split_record.get("available_steps_max")
            if isinstance(available, int) and not isinstance(available, bool) and available < max_required_length:
                split_rows.append(
                    {
                        "split": split,
                        "available_steps_max": available,
                        "required_steps_min": max_required_length,
                        "additional_steps_min": max_required_length - available,
                    }
                )
        if split_rows:
            rows.append({"environment": env, "splits": split_rows})
    return rows


def _recommendations(options: Sequence[Mapping[str, object]]) -> dict[str, object]:
    by_id = {str(option["option_id"]): option for option in options}
    return {
        "preserve_original_g1": "data_extension_preserve_g1",
        "fastest_t3_1_unblock": "longest_common_prefix_meeting_t3_1"
        if by_id["longest_common_prefix_meeting_t3_1"]["strict_t3_1_input_feasible"]
        else "common_horizon_all_environments",
        "do_not_auto_apply": True,
    }


def _load_goal(path: Path) -> Mapping[str, Any]:
    try:
        payload = load_yaml_document(Path(path))
        validate_document("goal_spec", payload)
    except (OSError, ContractValidationError) as exc:
        raise HorizonProtocolAmendmentError("HORIZON_AMENDMENT_GOAL_INVALID") from exc
    if not isinstance(payload.get("envs"), list) or not isinstance(payload.get("horizons"), list):
        raise HorizonProtocolAmendmentError("HORIZON_AMENDMENT_GOAL_INVALID")
    return payload


def _load_availability_report(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HorizonProtocolAmendmentError("HORIZON_AMENDMENT_AVAILABILITY_INVALID") from exc
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema_version") != 1
        or payload.get("artifact_type") != "wmloop-horizon-availability-report"
        or not isinstance(payload.get("records"), list)
    ):
        raise HorizonProtocolAmendmentError("HORIZON_AMENDMENT_AVAILABILITY_INVALID")
    return payload


def _load_optional_raw_failure(path: Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HorizonProtocolAmendmentError("HORIZON_AMENDMENT_RAW_FAILURE_INVALID") from exc
    if not isinstance(payload, Mapping) or payload.get("artifact_type") != "wmloop-m1-raw-failure-report-batch":
        raise HorizonProtocolAmendmentError("HORIZON_AMENDMENT_RAW_FAILURE_INVALID")
    return {
        "state": payload.get("state"),
        "report_count": payload.get("report_count"),
        "blocked_count": payload.get("blocked_count"),
        "blocked_records": payload.get("blocked_records", []),
    }


def _records_by_key(report: Mapping[str, Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
    records: dict[tuple[str, str], Mapping[str, Any]] = {}
    for item in report["records"]:
        if not isinstance(item, Mapping) or not isinstance(item.get("environment"), str) or not isinstance(item.get("split"), str):
            raise HorizonProtocolAmendmentError("HORIZON_AMENDMENT_AVAILABILITY_INVALID")
        key = (str(item["environment"]), str(item["split"]))
        if key in records:
            raise HorizonProtocolAmendmentError(f"HORIZON_AMENDMENT_AVAILABILITY_DUPLICATE:{key[0]}:{key[1]}")
        records[key] = item
    return records


def _splits(report: Mapping[str, Any], records_by_key: Mapping[tuple[str, str], Mapping[str, Any]]) -> tuple[str, ...]:
    raw = report.get("splits")
    if isinstance(raw, list) and raw and all(isinstance(item, str) and item for item in raw):
        return tuple(str(item) for item in raw)
    return tuple(sorted({split for _, split in records_by_key}))


def _int_list(value: object, code: str) -> list[int]:
    if not isinstance(value, list):
        raise HorizonProtocolAmendmentError(code)
    parsed: list[int] = []
    for item in value:
        if isinstance(item, int) and not isinstance(item, bool):
            parsed.append(item)
        elif isinstance(item, str) and item.isdigit():
            parsed.append(int(item))
        else:
            raise HorizonProtocolAmendmentError(code)
    return sorted(parsed)


def _auc_name(horizons: Sequence[int]) -> str | None:
    if len(horizons) < 2:
        return None
    return f"auc_psnr_{min(horizons)}_{max(horizons)}"


def _write_report_bundle(
    *,
    report: Mapping[str, object],
    output_root: Path,
    archive_db: Path | None,
    cas_root: Path | None,
) -> dict[str, object]:
    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise HorizonProtocolAmendmentError("HORIZON_AMENDMENT_OUTPUT_EXISTS")
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    report_bytes = _canonical_json_bytes(report)
    markdown_bytes = _render_markdown(report).encode("utf-8")
    cas_refs: dict[str, str] = {}
    try:
        temporary.mkdir(mode=0o700, parents=True)
        _write_bytes_atomic(temporary / "horizon-protocol-amendment.json", report_bytes)
        _write_bytes_atomic(temporary / "horizon-protocol-amendment.md", markdown_bytes)
        archive = ArchiveStore(archive_db) if archive_db is not None else None
        if archive is not None or cas_root is not None:
            root = cas_root if cas_root is not None else Path(archive_db).resolve().parent  # type: ignore[union-attr]
            cas = ContentAddressedStore(Path(root))
            for name, payload, media_type in (
                ("horizon_protocol_amendment_json", report_bytes, "application/json"),
                ("horizon_protocol_amendment_markdown", markdown_bytes, "text/markdown"),
            ):
                ref = cas.put_bytes(payload, media_type=media_type).uri
                cas_refs[name] = ref
                if archive is not None:
                    archive.record_artifact_reference(ref)
        manifest = {
            "schema_version": 1,
            "artifact_type": "wmloop-m1-horizon-protocol-amendment-candidates-manifest",
            "state": report["state"],
            "active_protocol_changed": report["active_protocol_changed"],
            "human_approval_required": report["human_approval_required"],
            "report_path": str(destination / "horizon-protocol-amendment.json"),
            "markdown_path": str(destination / "horizon-protocol-amendment.md"),
            "cas_refs": cas_refs,
            "recommendations": report["recommendations"],
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


def _render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Horizon Protocol Amendment Candidates",
        "",
        f"State: `{report['state']}`",
        f"Active protocol changed: `{report['active_protocol_changed']}`",
        f"Human approval required: `{report['human_approval_required']}`",
        "",
        "| Option | Protocol Changed | Supported Envs | T3.1 Feasible | Claim Scope | Risk |",
        "|:--|:--|--:|:--|:--|:--|",
    ]
    for option in report["candidate_options"]:
        lines.append(
            "| {option} | {changed} | {count} | {feasible} | {scope} | {risk} |".format(
                option=option["option_id"],
                changed=option["active_protocol_changed"],
                count=option.get("supported_environment_count", "n/a"),
                feasible=option["strict_t3_1_input_feasible"],
                scope=option["claim_scope"],
                risk=option["risk"],
            )
        )
    lines.extend(["", "## Recommendations", ""])
    for key, value in report["recommendations"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Limitations", ""])
    for limitation in report["limitations"]:
        lines.append(f"- {limitation}")
    return "\n".join(lines) + "\n"


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise HorizonProtocolAmendmentError("HORIZON_AMENDMENT_OUTPUT_EXISTS")
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
    generate = commands.add_parser("generate", help="generate protocol amendment candidates")
    generate.add_argument("--availability-report", type=Path, required=True)
    generate.add_argument("--output-root", type=Path, required=True)
    generate.add_argument("--goal-config", type=Path, default=Path("configs/goal/long_horizon_v1.yaml"))
    generate.add_argument("--raw-failure-batch-manifest", type=Path)
    generate.add_argument("--strict-required-reports", type=int, default=3)
    generate.add_argument("--archive-db", type=Path)
    generate.add_argument("--cas-root", type=Path)
    args = parser.parse_args(argv)
    if args.command == "generate":
        manifest = generate_horizon_protocol_amendment(
            availability_report_path=args.availability_report,
            output_root=args.output_root,
            goal_config=args.goal_config,
            raw_failure_batch_manifest=args.raw_failure_batch_manifest,
            strict_required_reports=args.strict_required_reports,
            archive_db=args.archive_db,
            cas_root=args.cas_root,
        )
        print(json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
