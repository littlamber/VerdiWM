"""Export a read-only primitive/hook capability matrix for one backbone instance."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from wmloop.contracts import ContractValidationError, load_yaml_document, validate_document
from wmloop.primitives.adapters.backbone_registry import (
    BackbonePrimitiveRegistryError,
    load_backbone_primitive_registry,
)
from wmloop.primitives.registry import PrimitiveRegistry, PrimitiveValidationError


class BackboneCapabilityMatrixError(RuntimeError):
    """A capability matrix could not be produced safely."""


READY_STATUSES = {"ready", "external_ready"}
HOOKS = ("H1", "H2", "H3", "H4", "H5")
DEFAULT_FAMILY_HOOKS = {
    "acwm_phys": HOOKS,
    "ctrl_world": HOOKS,
    "cosmos3": HOOKS,
    "wam": HOOKS,
    "generic": (),
}


def run_backbone_capability_matrix(
    *,
    instance_config: Path,
    output_root: Path,
    repo_root: Path | None = None,
) -> dict[str, object]:
    """Write a matrix of instance surfaces, hook availability, and primitive eligibility.

    This is a control-plane artifact only. It does not mutate the instance,
    launch training, start evaluation, or grant formal campaign permission.
    """

    root = Path(repo_root or Path(__file__).resolve().parents[2]).resolve()
    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise BackboneCapabilityMatrixError("BACKBONE_CAPABILITY_MATRIX_OUTPUT_EXISTS")
    config = _load_instance_config(Path(instance_config).resolve(strict=True), root=root)
    surfaces = [_surface_row(surface, root=root) for surface in _mappings(config, "surfaces")]
    registry = _load_primitive_registry(root)
    primitive_registry_ready = _surface_ready(surfaces, "primitive_registry")
    bound_primitives, registry_error = _bound_primitive_states(config=config, surfaces=surfaces, root=root)
    hook_adapter_ready = _hook_adapter_ready(config=config, surfaces=surfaces)
    available_hooks = _available_hooks(config=config, hook_adapter_ready=hook_adapter_ready)
    hook_rows = _hook_rows(available_hooks=available_hooks, hook_adapter_ready=hook_adapter_ready)
    primitive_rows = [
        _primitive_row(
            manifest=registry.manifest(name),
            primitive_registry_ready=primitive_registry_ready,
            bound_primitives=bound_primitives,
            hook_adapter_ready=hook_adapter_ready,
            available_hooks=available_hooks,
        )
        for name in registry.names()
    ]
    closed_loop_surface_ready = _all_required_ready(surfaces, "required_for_closed_loop")
    formal_surface_ready = _all_required_ready(surfaces, "required_for_formal_verdict")
    blockers = _blockers(
        surfaces=surfaces,
        primitive_rows=primitive_rows,
        primitive_registry_ready=primitive_registry_ready,
        registry_error=registry_error,
        hook_adapter_ready=hook_adapter_ready,
    )
    eligible_count = sum(row["status"] == "eligible_for_instance_canary" for row in primitive_rows)
    state = _state(
        campaign_state=str(config["campaign_state"]),
        blockers=blockers,
        closed_loop_surface_ready=closed_loop_surface_ready,
        eligible_count=eligible_count,
    )
    report = {
        "schema_version": 1,
        "artifact_type": "wmloop-backbone-capability-matrix",
        "state": state,
        "instance_id": str(config["instance_id"]),
        "backbone_family": str(config["backbone_family"]),
        "goal_id": str(config["goal_id"]),
        "campaign_state": str(config["campaign_state"]),
        "claim_scope": str(config["claim_scope"]),
        "capability_summary": {
            "surface_count": len(surfaces),
            "ready_surface_count": sum(1 for surface in surfaces if surface["ready_for_declared_status"]),
            "available_hooks": list(available_hooks),
            "primitive_count": len(primitive_rows),
            "eligible_primitive_count": eligible_count,
            "blocked_primitive_count": len(primitive_rows) - eligible_count,
            "primitive_registry_ready": primitive_registry_ready,
            "hook_adapter_ready": hook_adapter_ready,
            "closed_loop_surface_ready": closed_loop_surface_ready,
            "formal_surface_ready": formal_surface_ready,
        },
        "surface_matrix": surfaces,
        "hook_matrix": hook_rows,
        "primitive_matrix": primitive_rows,
        "blockers": blockers,
        "side_effects": {
            "source_code_mutated": False,
            "goal_config_mutated": False,
            "protocol_changed": False,
            "registry_changed": False,
            "gpu_execution_started": False,
            "formal_verdict_mutated": False,
        },
        "limitations": [
            "This matrix checks wiring capability only; it does not prove that a primitive is already materialized or beneficial.",
            "A primitive marked eligible still needs primitive materialization gates, runtime smoke, canary, and frozen verdict evidence.",
            "Backbone families without an explicit hook adapter are blocked even if their source repository is present.",
        ],
    }
    try:
        validate_document("backbone_capability_matrix", report, root=root)
    except ContractValidationError as exc:
        raise BackboneCapabilityMatrixError(f"BACKBONE_CAPABILITY_MATRIX_CONTRACT_INVALID:{exc}") from exc
    return _write_bundle(report=report, output_root=destination)


def _load_instance_config(path: Path, *, root: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8")) if path.suffix == ".json" else load_yaml_document(path)
        validate_document("backbone_instance", payload, root=root)
    except (OSError, json.JSONDecodeError, ContractValidationError) as exc:
        raise BackboneCapabilityMatrixError("BACKBONE_CAPABILITY_MATRIX_INSTANCE_INVALID") from exc
    return payload


def _load_primitive_registry(root: Path) -> PrimitiveRegistry:
    try:
        return PrimitiveRegistry.from_root(root)
    except PrimitiveValidationError as exc:
        raise BackboneCapabilityMatrixError("BACKBONE_CAPABILITY_MATRIX_PRIMITIVE_REGISTRY_INVALID") from exc


def _bound_primitive_states(
    *,
    config: Mapping[str, Any],
    surfaces: list[Mapping[str, object]],
    root: Path,
) -> tuple[dict[str, str] | None, str | None]:
    """Return explicit backbone bindings, or None for the native ACWM registry."""
    if str(config["backbone_family"]) == "acwm_phys":
        return None, None
    surface = next((row for row in surfaces if row["surface_id"] == "primitive_registry"), None)
    if surface is None or not bool(surface["ready_for_declared_status"]):
        return {}, "primitive_registry_missing_or_not_ready"
    try:
        registry = load_backbone_primitive_registry(Path(str(surface["artifact_ref"])), root=root)
    except BackbonePrimitiveRegistryError as exc:
        return {}, str(exc)
    return {binding.primitive: binding.runtime_state for binding in registry.bindings}, None


def _surface_row(surface: Mapping[str, Any], *, root: Path) -> dict[str, object]:
    artifact_ref = str(surface["artifact_ref"])
    path = Path(artifact_ref)
    resolved = path if path.is_absolute() else root / path
    status = str(surface["status"])
    ready_for_declared_status = resolved.exists() if status in READY_STATUSES else True
    return {
        "surface_id": str(surface["surface_id"]),
        "role": str(surface["role"]),
        "status": status,
        "artifact_ref": artifact_ref,
        "path_exists": resolved.exists(),
        "ready_for_declared_status": ready_for_declared_status,
        "required_for_closed_loop": bool(surface["required_for_closed_loop"]),
        "required_for_formal_verdict": bool(surface["required_for_formal_verdict"]),
    }


def _hook_adapter_ready(*, config: Mapping[str, Any], surfaces: list[Mapping[str, object]]) -> bool:
    if str(config["backbone_family"]) == "acwm_phys":
        return _surface_ready(surfaces, "hook_adapter") or _surface_ready(surfaces, "executor_adapter")
    return _surface_ready(surfaces, "hook_adapter")


def _surface_ready(surfaces: list[Mapping[str, object]], surface_id: str) -> bool:
    return any(
        surface["surface_id"] == surface_id
        and surface["status"] in READY_STATUSES
        and surface["ready_for_declared_status"] is True
        for surface in surfaces
    )


def _available_hooks(*, config: Mapping[str, Any], hook_adapter_ready: bool) -> tuple[str, ...]:
    if not hook_adapter_ready:
        return ()
    return tuple(DEFAULT_FAMILY_HOOKS.get(str(config["backbone_family"]), ()))


def _hook_rows(*, available_hooks: tuple[str, ...], hook_adapter_ready: bool) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for hook in HOOKS:
        available = hook in available_hooks
        rows.append(
            {
                "hook": hook,
                "available": available,
                "reason": "declared_by_instance_adapter" if available else _hook_blocker_reason(hook_adapter_ready),
            }
        )
    return rows


def _hook_blocker_reason(hook_adapter_ready: bool) -> str:
    return "hook_adapter_missing_or_not_ready" if not hook_adapter_ready else "not_declared_for_backbone_family"


def _primitive_row(
    *,
    manifest: Any,
    primitive_registry_ready: bool,
    bound_primitives: Mapping[str, str] | None,
    hook_adapter_ready: bool,
    available_hooks: tuple[str, ...],
) -> dict[str, object]:
    blockers: list[str] = []
    if not primitive_registry_ready:
        blockers.append("primitive_registry_missing_or_not_ready")
    if bound_primitives is not None:
        state = bound_primitives.get(manifest.name)
        if state is None:
            blockers.append("primitive_not_bound_for_backbone")
        elif state == "quarantined":
            blockers.append("primitive_quarantined_for_backbone")
    if not hook_adapter_ready:
        blockers.append("hook_adapter_missing_or_not_ready")
    missing_hooks = sorted(set(manifest.hooks) - set(available_hooks))
    blockers.extend(f"hook_unavailable:{hook}" for hook in missing_hooks)
    status = "eligible_for_instance_canary" if not blockers else "blocked"
    return {
        "primitive": manifest.name,
        "layer": manifest.layer,
        "family": manifest.family,
        "cost_class": manifest.cost_class,
        "hooks": list(manifest.hooks),
        "targets_failures": list(manifest.targets_failures),
        "status": status,
        "blockers": blockers,
    }


def _all_required_ready(surfaces: list[Mapping[str, object]], required_key: str) -> bool:
    required = [surface for surface in surfaces if surface[required_key] is True]
    return bool(required) and all(
        surface["status"] in READY_STATUSES and surface["ready_for_declared_status"] is True for surface in required
    )


def _blockers(
    *,
    surfaces: list[Mapping[str, object]],
    primitive_rows: list[Mapping[str, object]],
    primitive_registry_ready: bool,
    registry_error: str | None,
    hook_adapter_ready: bool,
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
    if not primitive_registry_ready:
        blockers.append({"code": "primitive_registry_missing_or_not_ready"})
    if registry_error is not None:
        blockers.append({"code": "primitive_registry_invalid", "detail": registry_error})
    if not hook_adapter_ready:
        blockers.append({"code": "hook_adapter_missing_or_not_ready"})
    return blockers


def _state(*, campaign_state: str, blockers: list[Mapping[str, object]], closed_loop_surface_ready: bool, eligible_count: int) -> str:
    if campaign_state == "pilot_draft":
        return "pilot_draft"
    if blockers or not closed_loop_surface_ready or eligible_count == 0:
        return "blocked"
    return "ready"


def _write_bundle(*, report: Mapping[str, object], output_root: Path) -> dict[str, object]:
    temporary = output_root.parent / f".{output_root.name}.{uuid.uuid4().hex}.tmp"
    try:
        output_root.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary.mkdir(mode=0o700)
        _write_bytes_atomic(temporary / "backbone-capability-matrix.json", _canonical_json_bytes(report))
        _write_bytes_atomic(temporary / "backbone-capability-matrix.md", _render_markdown(report).encode("utf-8"))
        manifest = {
            "schema_version": 1,
            "artifact_type": "wmloop-backbone-capability-matrix-manifest",
            "state": report["state"],
            "instance_id": report["instance_id"],
            "backbone_family": report["backbone_family"],
            "goal_id": report["goal_id"],
            "eligible_primitive_count": report["capability_summary"]["eligible_primitive_count"],
            "blocked_primitive_count": report["capability_summary"]["blocked_primitive_count"],
            "available_hooks": report["capability_summary"]["available_hooks"],
            "report_path": str(output_root / "backbone-capability-matrix.json"),
            "markdown_path": str(output_root / "backbone-capability-matrix.md"),
            "side_effects": report["side_effects"],
            "limitations": report["limitations"],
        }
        _write_bytes_atomic(temporary / "manifest.json", _canonical_json_bytes(manifest))
        os.replace(temporary, output_root)
        return manifest
    except Exception:
        if temporary.is_symlink():
            temporary.unlink()
        elif temporary.exists():
            shutil.rmtree(temporary)
        raise


def _render_markdown(report: Mapping[str, object]) -> str:
    summary = report["capability_summary"]
    lines = [
        "# Backbone Capability Matrix",
        "",
        f"State: `{report['state']}`",
        f"Instance: `{report['instance_id']}`",
        f"Backbone: `{report['backbone_family']}`",
        f"Goal: `{report['goal_id']}`",
        "",
        "## Summary",
        "",
        f"- available_hooks: `{','.join(summary['available_hooks']) or 'none'}`",
        f"- primitive_count: `{summary['primitive_count']}`",
        f"- eligible_primitive_count: `{summary['eligible_primitive_count']}`",
        f"- blocked_primitive_count: `{summary['blocked_primitive_count']}`",
        f"- primitive_registry_ready: `{summary['primitive_registry_ready']}`",
        f"- hook_adapter_ready: `{summary['hook_adapter_ready']}`",
        f"- closed_loop_surface_ready: `{summary['closed_loop_surface_ready']}`",
        f"- formal_surface_ready: `{summary['formal_surface_ready']}`",
        "",
        "## Hooks",
        "",
        "| Hook | Available | Reason |",
        "|:--|:--|:--|",
    ]
    for row in report["hook_matrix"]:
        lines.append(f"| `{row['hook']}` | `{row['available']}` | {row['reason']} |")
    lines.extend(
        [
            "",
            "## Primitives",
            "",
            "| Primitive | Layer | Hooks | Status | Blockers |",
            "|:--|:--|:--|:--|:--|",
        ]
    )
    for row in report["primitive_matrix"]:
        hooks = ",".join(row["hooks"])
        blockers = "; ".join(row["blockers"]) or "none"
        lines.append(f"| `{row['primitive']}` | `{row['layer']}` | `{hooks}` | `{row['status']}` | {blockers} |")
    lines.extend(
        [
            "",
            "## Limitations",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in report["limitations"])
    return "\n".join(lines) + "\n"


def _mappings(payload: Mapping[str, Any], key: str) -> list[Mapping[str, Any]]:
    value = payload.get(key)
    if not isinstance(value, list) or any(not isinstance(item, Mapping) for item in value):
        raise BackboneCapabilityMatrixError(f"BACKBONE_CAPABILITY_MATRIX_FIELD_INVALID:{key}")
    return value


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n"


def _write_bytes_atomic(path: Path, content: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path)
    args = parser.parse_args(argv)
    manifest = run_backbone_capability_matrix(
        instance_config=args.instance_config,
        output_root=args.output_root,
        repo_root=args.repo_root,
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
