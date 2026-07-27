"""Generate per-environment horizon ladder protocol artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

from wmloop.archive.store import ArchiveStore, ContentAddressedStore
from wmloop.contracts import ContractValidationError, load_yaml_document, validate_document


class HorizonLadderError(RuntimeError):
    """A horizon ladder protocol artifact could not be generated safely."""


def build_horizon_ladder(
    *,
    availability_report_path: Path,
    goal_config: Path,
    protocol_id: str = "g1_acwm_phys_horizon_ladder_v1",
    goal_id: str = "g1_long_horizon_ladder_v1",
    source_availability_report: str | None = None,
) -> dict[str, object]:
    """Build a validated per-environment max-horizon ladder from an audit report."""

    if not protocol_id or not goal_id:
        raise HorizonLadderError("HORIZON_LADDER_ID_INVALID")
    goal = _load_goal(goal_config)
    availability = _load_availability_report(availability_report_path)
    base_horizons = _positive_ints(goal.get("horizons"), "HORIZON_LADDER_GOAL_HORIZONS_INVALID")
    envs = _strings(goal.get("envs"), "HORIZON_LADDER_GOAL_ENVS_INVALID")
    records_by_key = _records_by_key(availability)
    splits = _splits(availability, records_by_key)
    horizons_by_environment = {
        environment: _environment_horizon_prefix(
            environment,
            splits=splits,
            records_by_key=records_by_key,
            base_horizons=base_horizons,
        )
        for environment in envs
    }
    if any(not horizons for horizons in horizons_by_environment.values()):
        raise HorizonLadderError("HORIZON_LADDER_EMPTY_ENVIRONMENT_HORIZONS")
    common_horizons = _common_horizons(horizons_by_environment.values())
    if not common_horizons:
        raise HorizonLadderError("HORIZON_LADDER_COMMON_HORIZONS_EMPTY")
    max_by_environment = {environment: max(horizons) for environment, horizons in horizons_by_environment.items()}
    max_groups = {
        str(horizon): [environment for environment in envs if max_by_environment[environment] == horizon]
        for horizon in base_horizons
    }
    claim_envs = {
        str(horizon): [environment for environment in envs if horizon in horizons_by_environment[environment]]
        for horizon in base_horizons
    }
    ladder: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "wmloop-horizon-ladder",
        "protocol_id": protocol_id,
        "mode": "per_environment_max_horizon",
        "goal_id": goal_id,
        "source_availability_report": source_availability_report or str(Path(availability_report_path).resolve()),
        "base_horizons": base_horizons,
        "splits": list(splits),
        "common_horizons_all_envs": common_horizons,
        "cross_environment_comparison_horizons": common_horizons,
        "horizons_by_environment": horizons_by_environment,
        "max_horizon_by_environment": max_by_environment,
        "max_horizon_groups": max_groups,
        "claim_envs_by_horizon": claim_envs,
        "long_horizon_64_envs": claim_envs.get("64", []),
        "reporting_policy": {
            "cross_environment": "Compare environments only on common_horizons_all_envs.",
            "long_horizon_64": "Report horizon-64 claims only for long_horizon_64_envs.",
            "environment_metric": "Use ladder_auc_psnr_envmax as the per-environment AUC over its declared horizon set.",
        },
        "m4_launch_allowed": False,
        "formal_training_allowed": False,
    }
    try:
        validate_document("horizon_ladder", ladder)
    except ContractValidationError as exc:
        raise HorizonLadderError("HORIZON_LADDER_SCHEMA_INVALID") from exc
    return ladder


def build_ladder_goal(
    *,
    active_goal_config: Path,
    ladder: Mapping[str, Any],
    ladder_path: str,
) -> dict[str, object]:
    """Build a goal spec that references a validated horizon ladder artifact."""

    if not ladder_path:
        raise HorizonLadderError("HORIZON_LADDER_GOAL_PATH_INVALID")
    try:
        validate_document("horizon_ladder", ladder)
    except ContractValidationError as exc:
        raise HorizonLadderError("HORIZON_LADDER_SCHEMA_INVALID") from exc
    goal = deepcopy(_load_goal(active_goal_config))
    goal["goal_id"] = str(ladder["goal_id"])
    goal["primary_objective"] = "ladder_auc_psnr_envmax"
    goal["horizons"] = [int(value) for value in ladder["base_horizons"]]
    protocol = dict(goal["eval_protocol"])
    protocol["mode"] = "per_environment_horizon_ladder"
    protocol["horizon_ladder_path"] = ladder_path
    protocol["cross_environment_comparison_horizons"] = [
        int(value) for value in ladder["cross_environment_comparison_horizons"]
    ]
    protocol["long_horizon_claim_policy"] = "horizon_64_claims_only_for_ladder_long_horizon_64_envs"
    goal["eval_protocol"] = protocol
    try:
        validate_document("goal_spec", goal)
    except ContractValidationError as exc:
        raise HorizonLadderError("HORIZON_LADDER_GOAL_SCHEMA_INVALID") from exc
    return goal


def generate_horizon_ladder_bundle(
    *,
    availability_report_path: Path,
    goal_config: Path,
    output_root: Path,
    protocol_id: str = "g1_acwm_phys_horizon_ladder_v1",
    goal_id: str = "g1_long_horizon_ladder_v1",
    ladder_path_for_goal: str | None = None,
    source_availability_report: str | None = None,
    archive_db: Path | None = None,
    cas_root: Path | None = None,
) -> dict[str, object]:
    """Write a horizon-ladder report bundle without mutating active configs."""

    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise HorizonLadderError("HORIZON_LADDER_OUTPUT_EXISTS")
    ladder = build_horizon_ladder(
        availability_report_path=availability_report_path,
        goal_config=goal_config,
        protocol_id=protocol_id,
        goal_id=goal_id,
        source_availability_report=source_availability_report,
    )
    goal = build_ladder_goal(
        active_goal_config=goal_config,
        ladder=ladder,
        ladder_path=ladder_path_for_goal or str(destination / "horizon-ladder.yaml"),
    )
    report = {
        "schema_version": 1,
        "artifact_type": "wmloop-horizon-ladder-generation-report",
        "state": "pre_m4_protocol_calibration_ready",
        "active_old_goal_mutated": False,
        "m4_launch_allowed": False,
        "formal_training_allowed": False,
        "availability_report_path": str(Path(availability_report_path).resolve()),
        "availability_report_sha256": _sha256_file(Path(availability_report_path).resolve()),
        "active_goal_config_path": str(Path(goal_config).resolve()),
        "active_goal_sha256": _sha256_file(Path(goal_config).resolve()),
        "ladder": ladder,
        "candidate_goal": goal,
        "next_actions": [
            "Promote the ladder goal only as a new version boundary; do not overwrite the original G1 goal.",
            "Regenerate M1 raw failure reports under the ladder goal before using them for M3/M4.",
            "Regenerate constitutional freeze/audit and M4 phase gate before any formal M4 launch.",
        ],
    }
    return _write_report_bundle(
        report=report,
        output_root=destination,
        archive_db=archive_db,
        cas_root=cas_root,
    )


def _load_goal(path: Path) -> Mapping[str, Any]:
    try:
        payload = load_yaml_document(Path(path))
        validate_document("goal_spec", payload)
    except (OSError, ContractValidationError) as exc:
        raise HorizonLadderError("HORIZON_LADDER_GOAL_INVALID") from exc
    return payload


def _load_availability_report(path: Path) -> Mapping[str, Any]:
    try:
        candidate = Path(path).resolve(strict=True)
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HorizonLadderError("HORIZON_LADDER_AVAILABILITY_INVALID") from exc
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema_version") != 1
        or payload.get("artifact_type") != "wmloop-horizon-availability-report"
        or not isinstance(payload.get("records"), list)
    ):
        raise HorizonLadderError("HORIZON_LADDER_AVAILABILITY_INVALID")
    return payload


def _records_by_key(report: Mapping[str, Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
    records: dict[tuple[str, str], Mapping[str, Any]] = {}
    for item in report["records"]:
        if not isinstance(item, Mapping) or not isinstance(item.get("environment"), str) or not isinstance(item.get("split"), str):
            raise HorizonLadderError("HORIZON_LADDER_AVAILABILITY_INVALID")
        key = (str(item["environment"]), str(item["split"]))
        if key in records:
            raise HorizonLadderError(f"HORIZON_LADDER_AVAILABILITY_DUPLICATE:{key[0]}:{key[1]}")
        records[key] = item
    return records


def _splits(report: Mapping[str, Any], records_by_key: Mapping[tuple[str, str], Mapping[str, Any]]) -> tuple[str, ...]:
    raw = report.get("splits")
    if isinstance(raw, list) and raw and all(isinstance(item, str) and item for item in raw):
        return tuple(str(item) for item in raw)
    return tuple(sorted({split for _, split in records_by_key}))


def _environment_horizon_prefix(
    environment: str,
    *,
    splits: Sequence[str],
    records_by_key: Mapping[tuple[str, str], Mapping[str, Any]],
    base_horizons: Sequence[int],
) -> list[int]:
    split_sets: list[set[int]] = []
    for split in splits:
        record = records_by_key.get((environment, split))
        if record is None:
            raise HorizonLadderError(f"HORIZON_LADDER_AVAILABILITY_MISSING:{environment}:{split}")
        split_sets.append(set(_positive_ints(record.get("supported_horizons"), "HORIZON_LADDER_SUPPORTED_INVALID")))
    common = set.intersection(*split_sets) if split_sets else set()
    prefix: list[int] = []
    for horizon in base_horizons:
        if horizon not in common:
            break
        prefix.append(horizon)
    return prefix


def _common_horizons(values: Sequence[Sequence[int]]) -> list[int]:
    sets = [set(int(item) for item in value) for value in values]
    return sorted(set.intersection(*sets)) if sets else []


def _positive_ints(value: object, code: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise HorizonLadderError(code)
    parsed: list[int] = []
    for item in value:
        if isinstance(item, int) and not isinstance(item, bool) and item > 0:
            parsed.append(item)
        elif isinstance(item, str) and item.isdigit() and int(item) > 0:
            parsed.append(int(item))
        else:
            raise HorizonLadderError(code)
    if len(set(parsed)) != len(parsed):
        raise HorizonLadderError(code)
    return parsed


def _strings(value: object, code: str) -> list[str]:
    if not isinstance(value, list) or not value or any(not isinstance(item, str) or not item for item in value):
        raise HorizonLadderError(code)
    parsed = [str(item) for item in value]
    if len(set(parsed)) != len(parsed):
        raise HorizonLadderError(code)
    return parsed


def _write_report_bundle(
    *,
    report: Mapping[str, object],
    output_root: Path,
    archive_db: Path | None,
    cas_root: Path | None,
) -> dict[str, object]:
    destination = Path(output_root).resolve()
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    ladder = report["ladder"]
    candidate_goal = report["candidate_goal"]
    report_bytes = _canonical_json_bytes(report)
    ladder_json_bytes = _canonical_json_bytes(ladder)  # type: ignore[arg-type]
    ladder_yaml_bytes = _yaml_bytes(ladder)
    candidate_goal_bytes = _canonical_json_bytes(candidate_goal)  # type: ignore[arg-type]
    markdown_bytes = _render_markdown(report).encode("utf-8")
    cas_refs: dict[str, str] = {}
    try:
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary.mkdir(mode=0o700)
        _write_bytes_atomic(temporary / "horizon-ladder-generation.json", report_bytes)
        _write_bytes_atomic(temporary / "horizon-ladder-generation.md", markdown_bytes)
        _write_bytes_atomic(temporary / "horizon-ladder.json", ladder_json_bytes)
        _write_bytes_atomic(temporary / "horizon-ladder.yaml", ladder_yaml_bytes)
        _write_bytes_atomic(temporary / "ladder-goal.json", candidate_goal_bytes)
        archive = ArchiveStore(archive_db) if archive_db is not None else None
        if archive is not None or cas_root is not None:
            root = cas_root if cas_root is not None else Path(archive_db).resolve().parent  # type: ignore[union-attr]
            cas = ContentAddressedStore(Path(root))
            for name, payload, media_type in (
                ("horizon_ladder_generation_json", report_bytes, "application/json"),
                ("horizon_ladder_generation_markdown", markdown_bytes, "text/markdown"),
                ("horizon_ladder_json", ladder_json_bytes, "application/json"),
                ("horizon_ladder_yaml", ladder_yaml_bytes, "application/x-yaml"),
                ("horizon_ladder_goal", candidate_goal_bytes, "application/json"),
            ):
                ref = cas.put_bytes(payload, media_type=media_type).uri
                cas_refs[name] = ref
                if archive is not None:
                    archive.record_artifact_reference(ref)
        manifest = {
            "schema_version": 1,
            "artifact_type": "wmloop-horizon-ladder-generation-manifest",
            "state": report["state"],
            "active_old_goal_mutated": report["active_old_goal_mutated"],
            "m4_launch_allowed": report["m4_launch_allowed"],
            "formal_training_allowed": report["formal_training_allowed"],
            "protocol_id": ladder["protocol_id"],  # type: ignore[index]
            "goal_id": candidate_goal["goal_id"],  # type: ignore[index]
            "ladder_path": str(destination / "horizon-ladder.yaml"),
            "ladder_json_path": str(destination / "horizon-ladder.json"),
            "candidate_goal_path": str(destination / "ladder-goal.json"),
            "report_path": str(destination / "horizon-ladder-generation.json"),
            "markdown_path": str(destination / "horizon-ladder-generation.md"),
            "common_horizons_all_envs": ladder["common_horizons_all_envs"],  # type: ignore[index]
            "horizons_by_environment": ladder["horizons_by_environment"],  # type: ignore[index]
            "long_horizon_64_envs": ladder["long_horizon_64_envs"],  # type: ignore[index]
            "cas_refs": cas_refs,
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


def _render_markdown(report: Mapping[str, Any]) -> str:
    ladder = report["ladder"]
    lines = [
        "# Horizon Ladder Generation",
        "",
        f"State: `{report['state']}`",
        f"Active old goal mutated: `{report['active_old_goal_mutated']}`",
        f"M4 launch allowed: `{report['m4_launch_allowed']}`",
        "",
        "| Environment | Horizons | Max |",
        "|:--|:--|--:|",
    ]
    for environment, horizons in ladder["horizons_by_environment"].items():
        lines.append(f"| {environment} | {','.join(str(item) for item in horizons)} | {max(horizons)} |")
    lines.extend(
        [
            "",
            f"Common cross-environment horizons: `{','.join(str(item) for item in ladder['common_horizons_all_envs'])}`",
            f"Horizon-64 claim environments: `{','.join(str(item) for item in ladder['long_horizon_64_envs'])}`",
            "",
            "## Next Actions",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in report["next_actions"])
    return "\n".join(lines) + "\n"


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"


def _yaml_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise HorizonLadderError("HORIZON_LADDER_OUTPUT_EXISTS")
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


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate", help="generate a per-environment horizon ladder bundle")
    generate.add_argument("--availability-report", type=Path, required=True)
    generate.add_argument("--goal-config", type=Path, default=Path("configs/goal/long_horizon_v1.yaml"))
    generate.add_argument("--output-root", type=Path, required=True)
    generate.add_argument("--protocol-id", default="g1_acwm_phys_horizon_ladder_v1")
    generate.add_argument("--goal-id", default="g1_long_horizon_ladder_v1")
    generate.add_argument("--ladder-path-for-goal")
    generate.add_argument("--source-availability-report")
    generate.add_argument("--archive-db", type=Path)
    generate.add_argument("--cas-root", type=Path)
    args = parser.parse_args(argv)
    if args.command == "generate":
        manifest = generate_horizon_ladder_bundle(
            availability_report_path=args.availability_report,
            goal_config=args.goal_config,
            output_root=args.output_root,
            protocol_id=args.protocol_id,
            goal_id=args.goal_id,
            ladder_path_for_goal=args.ladder_path_for_goal,
            source_availability_report=args.source_availability_report,
            archive_db=args.archive_db,
            cas_root=args.cas_root,
        )
        print(json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
