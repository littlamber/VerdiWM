"""Backbone/goal instantiation audit for portable wm-loop campaigns."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from wmloop.archive.store import ArchiveStore, ContentAddressedStore
from wmloop.constitution import ConstitutionalFreezeError, verify_constitutional_freeze
from wmloop.contracts import ContractValidationError, load_yaml_document, validate_document
from wmloop.diagnose.probe_registry import ProbeRegistryError, load_probe_registry


class BackboneInstantiationError(RuntimeError):
    """A backbone instance packet could not be produced safely."""


READY_STATUSES = {"ready", "external_ready"}


def run_backbone_instantiation_audit(
    *,
    instance_config: Path,
    output_root: Path,
    repo_root: Path | None = None,
    archive_db: Path | None = None,
    cas_root: Path | None = None,
) -> dict[str, object]:
    """Write a read-only audit of one backbone/goal instance.

    The audit is intentionally about instantiation readiness only.  It does not
    launch training, does not mutate configs, and does not grant a phase gate.
    """

    root = Path(repo_root or Path(__file__).resolve().parents[2]).resolve()
    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise BackboneInstantiationError("BACKBONE_INSTANCE_OUTPUT_EXISTS")
    config_path = Path(instance_config).resolve(strict=True)
    config = _load_instance_config(config_path, root=root)
    surfaces = [_audit_surface(surface, root=root) for surface in config["surfaces"]]
    contract_checks = _contract_checks(config=config, surfaces=surfaces, root=root)
    blockers = _blockers(surfaces=surfaces, contract_checks=contract_checks)
    closed_loop_ready = _all_required_ready(surfaces, "required_for_closed_loop") and not _has_contract_blocker(
        contract_checks,
        purpose="closed_loop",
    )
    formal_ready = _all_required_ready(surfaces, "required_for_formal_verdict") and not _has_contract_blocker(
        contract_checks,
        purpose="formal_verdict",
    )
    state = _state(config=config, closed_loop_ready=closed_loop_ready, formal_ready=formal_ready, blockers=blockers)
    report = {
        "schema_version": 1,
        "artifact_type": "wmloop-backbone-instantiation-audit",
        "state": state,
        "instance_id": config["instance_id"],
        "backbone_family": config["backbone_family"],
        "goal_id": config["goal_id"],
        "campaign_state": config["campaign_state"],
        "claim_scope": config["claim_scope"],
        "closed_loop_instance_ready": closed_loop_ready,
        "formal_verdict_instance_ready": formal_ready,
        "instance_formal_launch_allowed": formal_ready and config["campaign_state"] == "active_ready",
        "surface_count": len(surfaces),
        "ready_surface_count": sum(1 for surface in surfaces if surface["ready_for_declared_status"]),
        "missing_or_draft_surface_count": sum(1 for surface in surfaces if surface["status"] not in READY_STATUSES),
        "surfaces": surfaces,
        "contract_checks": contract_checks,
        "blockers": blockers,
        "invariants": list(config["invariants"]),
        "next_actions": list(config["next_actions"]),
        "side_effects": {
            "config_mutated": False,
            "constitution_changed": False,
            "registry_changed": False,
            "gpu_execution_started": False,
            "phase_gate_granted": False,
        },
        "limitations": [
            "This audit checks whether a backbone/goal instance is wired for wm-loop; it does not run training or evaluation.",
            "Pilot or draft surfaces can be useful for planning, but they cannot enter a formal verifier until promoted and frozen.",
            "External-ready paths are observed local assets; re-check them before any live run because external repos can drift.",
        ],
    }
    return _write_bundle(
        report=report,
        config_path=config_path,
        output_root=destination,
        archive_db=archive_db,
        cas_root=cas_root,
    )


def _load_instance_config(config_path: Path, *, root: Path) -> Mapping[str, Any]:
    try:
        if config_path.suffix == ".json":
            config = json.loads(config_path.read_text(encoding="utf-8"))
        else:
            config = load_yaml_document(config_path)
        validate_document("backbone_instance", config, root=root)
    except (OSError, json.JSONDecodeError, ContractValidationError) as exc:
        raise BackboneInstantiationError("BACKBONE_INSTANCE_CONFIG_INVALID") from exc
    return config


def _audit_surface(surface: Mapping[str, Any], *, root: Path) -> dict[str, object]:
    artifact_ref = str(surface["artifact_ref"])
    path = Path(artifact_ref)
    resolved = path if path.is_absolute() else root / path
    status = str(surface["status"])
    exists = resolved.exists()
    ready_requires_path = status in READY_STATUSES
    ready_for_declared_status = exists if ready_requires_path else True
    return {
        "surface_id": str(surface["surface_id"]),
        "role": str(surface["role"]),
        "status": status,
        "artifact_ref": artifact_ref,
        "resolved_path": str(resolved),
        "path_exists": exists,
        "ready_for_declared_status": ready_for_declared_status,
        "required_for_closed_loop": bool(surface["required_for_closed_loop"]),
        "required_for_formal_verdict": bool(surface["required_for_formal_verdict"]),
        "notes": str(surface["notes"]),
    }


def _contract_checks(
    *,
    config: Mapping[str, Any],
    surfaces: list[Mapping[str, object]],
    root: Path,
) -> list[dict[str, object]]:
    by_id = {str(surface["surface_id"]): surface for surface in surfaces}
    checks: list[dict[str, object]] = []
    if "goal_spec" in by_id:
        checks.append(_check_goal(by_id["goal_spec"], root=root, expected_goal_id=str(config["goal_id"])))
    if "probe_registry" in by_id:
        checks.append(_check_probe_registry(by_id["probe_registry"], root=root))
    if "constitution_config" in by_id:
        checks.append(_check_constitution_config(by_id["constitution_config"], root=root))
    if "constitution_freeze" in by_id:
        checks.append(_check_constitution_freeze(by_id["constitution_freeze"], root=root))
    return checks


def _check_goal(surface: Mapping[str, object], *, root: Path, expected_goal_id: str) -> dict[str, object]:
    path = Path(str(surface["resolved_path"]))
    if not path.exists():
        return _check_row(surface, "goal_spec", "missing", False, "closed_loop", "goal file is absent")
    try:
        payload = load_yaml_document(path)
        validate_document("goal_spec", payload, root=root)
    except (OSError, ContractValidationError) as exc:
        return _check_row(surface, "goal_spec", "invalid", False, "closed_loop", str(exc))
    if str(payload.get("goal_id")) != expected_goal_id:
        return _check_row(surface, "goal_spec", "mismatch", False, "closed_loop", "goal_id differs from instance")
    return _check_row(surface, "goal_spec", "ready", True, "closed_loop", "goal_spec validates")


def _check_probe_registry(surface: Mapping[str, object], *, root: Path) -> dict[str, object]:
    path = Path(str(surface["resolved_path"]))
    if not path.exists():
        return _check_row(surface, "probe_registry", "missing", False, "formal_verdict", "probe registry is absent")
    try:
        registry = load_probe_registry(path, root=root)
    except (OSError, ProbeRegistryError) as exc:
        return _check_row(surface, "probe_registry", "invalid", False, "formal_verdict", str(exc))
    return _check_row(
        surface,
        "probe_registry",
        "ready",
        True,
        "formal_verdict",
        f"verdict_probes={','.join(registry.verdict_probe_ids)}",
    )


def _check_constitution_config(surface: Mapping[str, object], *, root: Path) -> dict[str, object]:
    path = Path(str(surface["resolved_path"]))
    if not path.exists():
        return _check_row(surface, "constitution_config", "missing", False, "formal_verdict", "constitution config is absent")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        validate_document("constitutional_config", payload, root=root)
    except (OSError, json.JSONDecodeError, ContractValidationError) as exc:
        return _check_row(surface, "constitution_config", "invalid", False, "formal_verdict", str(exc))
    return _check_row(surface, "constitution_config", "ready", True, "formal_verdict", "constitutional config validates")


def _check_constitution_freeze(surface: Mapping[str, object], *, root: Path) -> dict[str, object]:
    path = Path(str(surface["resolved_path"]))
    if not path.exists():
        return _check_row(surface, "constitution_freeze", "missing", False, "formal_verdict", "freeze manifest is absent")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        verify_constitutional_freeze(manifest, root=root)
    except (OSError, json.JSONDecodeError, ConstitutionalFreezeError) as exc:
        return _check_row(surface, "constitution_freeze", "invalid", False, "formal_verdict", str(exc))
    return _check_row(surface, "constitution_freeze", "ready", True, "formal_verdict", "constitutional freeze verifies")


def _check_row(
    surface: Mapping[str, object],
    contract: str,
    state: str,
    passed: bool,
    purpose: str,
    detail: str,
) -> dict[str, object]:
    return {
        "surface_id": surface["surface_id"],
        "contract": contract,
        "state": state,
        "passed": passed,
        "purpose": purpose,
        "detail": detail,
    }


def _all_required_ready(surfaces: list[Mapping[str, object]], key: str) -> bool:
    required = [surface for surface in surfaces if surface[key] is True]
    return bool(required) and all(surface["status"] in READY_STATUSES and surface["ready_for_declared_status"] is True for surface in required)


def _has_contract_blocker(checks: list[Mapping[str, object]], *, purpose: str) -> bool:
    return any(check["passed"] is not True and check["purpose"] == purpose for check in checks)


def _blockers(
    *,
    surfaces: list[Mapping[str, object]],
    contract_checks: list[Mapping[str, object]],
) -> list[dict[str, object]]:
    blockers: list[dict[str, object]] = []
    for surface in surfaces:
        if surface["status"] in READY_STATUSES and surface["ready_for_declared_status"] is not True:
            blockers.append(
                {
                    "code": "declared_ready_path_missing",
                    "surface_id": surface["surface_id"],
                    "artifact_ref": surface["artifact_ref"],
                }
            )
        if surface["status"] not in READY_STATUSES and (
            surface["required_for_closed_loop"] is True or surface["required_for_formal_verdict"] is True
        ):
            blockers.append(
                {
                    "code": f"surface_{surface['status']}",
                    "surface_id": surface["surface_id"],
                    "required_for_closed_loop": surface["required_for_closed_loop"],
                    "required_for_formal_verdict": surface["required_for_formal_verdict"],
                }
            )
    for check in contract_checks:
        if check["passed"] is not True:
            blockers.append(
                {
                    "code": "contract_check_failed",
                    "surface_id": check["surface_id"],
                    "contract": check["contract"],
                    "detail": check["detail"],
                }
            )
    return blockers


def _state(
    *,
    config: Mapping[str, Any],
    closed_loop_ready: bool,
    formal_ready: bool,
    blockers: list[Mapping[str, object]],
) -> str:
    if formal_ready and config["campaign_state"] == "active_ready" and not blockers:
        return "ready"
    if config["campaign_state"] == "pilot_draft":
        return "pilot_draft"
    return "blocked" if blockers else "ready"


def _write_bundle(
    *,
    report: Mapping[str, object],
    config_path: Path,
    output_root: Path,
    archive_db: Path | None,
    cas_root: Path | None,
) -> dict[str, object]:
    destination = Path(output_root).resolve()
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    report_bytes = _canonical_json_bytes(report)
    markdown_bytes = _render_markdown(report).encode("utf-8")
    config_bytes = config_path.read_bytes()
    cas_storage_root = (
        Path(cas_root).resolve()
        if cas_root is not None
        else (Path(archive_db).resolve().parent if archive_db is not None else destination.parent)
    )
    cas = ContentAddressedStore(cas_storage_root)
    archive = ArchiveStore(archive_db) if archive_db is not None else None
    try:
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary.mkdir(mode=0o700)
        refs = {
            "backbone_instantiation_json": cas.put_bytes(report_bytes, media_type="application/json").uri,
            "backbone_instantiation_markdown": cas.put_bytes(markdown_bytes, media_type="text/markdown").uri,
            "instance_config": cas.put_bytes(config_bytes, media_type="application/json").uri,
        }
        if archive is not None:
            for ref in refs.values():
                archive.record_artifact_reference(ref)
        _write_bytes_atomic(temporary / "backbone-instantiation.json", report_bytes)
        _write_bytes_atomic(temporary / "backbone-instantiation.md", markdown_bytes)
        _write_bytes_atomic(temporary / "input-config.json", config_bytes)
        manifest = {
            "schema_version": 1,
            "artifact_type": "wmloop-backbone-instantiation-manifest",
            "state": report["state"],
            "instance_id": report["instance_id"],
            "backbone_family": report["backbone_family"],
            "goal_id": report["goal_id"],
            "closed_loop_instance_ready": report["closed_loop_instance_ready"],
            "formal_verdict_instance_ready": report["formal_verdict_instance_ready"],
            "instance_formal_launch_allowed": report["instance_formal_launch_allowed"],
            "blocker_count": len(report["blockers"]),  # type: ignore[arg-type]
            "report_path": str(destination / "backbone-instantiation.json"),
            "markdown_path": str(destination / "backbone-instantiation.md"),
            "input_config_path": str(destination / "input-config.json"),
            "cas_refs": refs,
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


def _render_markdown(report: Mapping[str, object]) -> str:
    lines = [
        "# Backbone Instantiation Audit",
        "",
        f"Instance: `{report['instance_id']}`",
        f"Backbone: `{report['backbone_family']}`",
        f"Goal: `{report['goal_id']}`",
        f"State: `{report['state']}`",
        f"Closed-loop instance ready: `{report['closed_loop_instance_ready']}`",
        f"Formal verdict instance ready: `{report['formal_verdict_instance_ready']}`",
        f"Instance formal launch allowed: `{report['instance_formal_launch_allowed']}`",
        "",
        "## Surfaces",
        "",
        "| Surface | Role | Status | Exists | Closed-loop | Formal | Notes |",
        "|:--|:--|:--|:--|:--|:--|:--|",
    ]
    for surface in report["surfaces"]:  # type: ignore[index]
        item = dict(surface)
        lines.append(
            "| {surface_id} | {role} | {status} | {path_exists} | {required_for_closed_loop} | {required_for_formal_verdict} | {notes} |".format(
                **item
            )
        )
    lines.extend(["", "## Blockers", ""])
    blockers = list(report["blockers"])  # type: ignore[arg-type]
    if blockers:
        for blocker in blockers:
            lines.append(f"- `{dict(blocker).get('code')}`: {json.dumps(blocker, ensure_ascii=False, sort_keys=True)}")
    else:
        lines.append("- none")
    lines.extend(["", "## Next Actions", ""])
    for action in report["next_actions"]:  # type: ignore[index]
        lines.append(f"- {action}")
    lines.extend(["", "## Limitations", ""])
    for limitation in report["limitations"]:  # type: ignore[index]
        lines.append(f"- {limitation}")
    lines.append("")
    return "\n".join(lines)


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
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
    parser.add_argument("--instance-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--archive-db", type=Path)
    parser.add_argument("--cas-root", type=Path)
    args = parser.parse_args(argv)
    manifest = run_backbone_instantiation_audit(
        instance_config=args.instance_config,
        output_root=args.output_root,
        repo_root=args.repo_root,
        archive_db=args.archive_db,
        cas_root=args.cas_root,
    )
    print(json.dumps(manifest, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
