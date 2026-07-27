"""Audit the frozen constitutional layer and probe-role separation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from wmloop.archive.store import ArchiveStore, ContentAddressedStore
from wmloop.constitution import ConstitutionalFreezeError, verify_constitutional_freeze
from wmloop.contracts import ContractValidationError, load_yaml_document, validate_document
from wmloop.diagnose.probe_registry import ProbeRegistry, ProbeRegistryError, load_probe_registry


class ConstitutionalAuditError(RuntimeError):
    """A constitutional-audit input or output invariant failed."""


SURFACE_DEFINITIONS: tuple[dict[str, object], ...] = (
    {
        "surface_id": "evaluation_code_and_data",
        "document_component": "evaluation_code_and_data",
        "required_components": ("dataset_freeze", "heldout_protocol", "evaluator_freeze"),
        "description": "Official evaluation code, dataset freeze and held-out protocol are frozen before verifier use.",
    },
    {
        "surface_id": "metric_collection_protocol",
        "document_component": "metric_collection_protocol",
        "required_components": ("goal_spec",),
        "required_goal_fields": ("horizons", "primary_objective", "eval_protocol"),
        "description": "Horizon set, primary objective and collection protocol are bound by the frozen goal spec.",
    },
    {
        "surface_id": "behavior_validity_gate",
        "document_component": "behavior_validity_gate",
        "required_components": ("goal_spec", "probe_registry"),
        "required_goal_fields": ("action_following_gate",),
        "required_verdict_probes": ("action_following",),
        "description": "Behavior-validity threshold and verdict-facing action-following probe are frozen.",
    },
    {
        "surface_id": "attribution_policy",
        "document_component": "attribution_policy",
        "required_components": ("code:attribution_policy", "probe_registry"),
        "requires_diagnostic_probes": True,
        "description": "Dominant-failure attribution logic and diagnostic probe roles are auditable but not verdict-visible.",
    },
    {
        "surface_id": "four_gate_verdict_rules",
        "document_component": "four_gate_verdict_rules",
        "required_components": ("code:verdict_policy", "code:verdict_evidence_projection", "code:schema_contracts"),
        "requires_verdict_projection": True,
        "description": "Verdict policy, verdict-evidence projection and schemas are frozen as verifier-facing rules.",
    },
)


def run_constitutional_audit(
    *,
    constitution_manifest: Path,
    output_root: Path,
    repo_root: Path | None = None,
    archive_db: Path | None = None,
    cas_root: Path | None = None,
) -> dict[str, object]:
    """Write a read-only audit bundle for the §6.6 constitutional layer."""

    base = (repo_root or Path(__file__).resolve().parents[2]).resolve()
    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise ConstitutionalAuditError("CONSTITUTIONAL_AUDIT_OUTPUT_EXISTS")
    manifest, manifest_bytes, manifest_path = _read_json_mapping(
        constitution_manifest,
        "CONSTITUTIONAL_AUDIT_MANIFEST_INVALID",
    )
    blockers: list[dict[str, object]] = []
    freeze_state = _verify_freeze(manifest, root=base, blockers=blockers)
    entries = _entries(manifest, blockers=blockers)
    entry_observations = [_audit_entry(entry, root=base) for entry in entries]
    for observation in entry_observations:
        blockers.extend(observation["blockers"])
    entries_by_component = _entries_by_component(entry_observations)
    config = _load_constitution_config(entries_by_component, root=base, blockers=blockers)
    goal = _load_goal(config, root=base, blockers=blockers)
    registry = _load_registry(config, root=base, blockers=blockers)
    verdict_probe_ids = _verdict_probe_ids(manifest=manifest, config=config, registry=registry, blockers=blockers)
    diagnostic_probe_ids = _diagnostic_probe_ids(registry, blockers=blockers)
    _audit_probe_role_contracts(
        manifest=manifest,
        config=config,
        registry=registry,
        verdict_probe_ids=verdict_probe_ids,
        diagnostic_probe_ids=diagnostic_probe_ids,
        blockers=blockers,
    )
    surfaces = [
        _audit_surface(
            definition,
            entries_by_component=entries_by_component,
            config=config,
            goal=goal,
            root=base,
            verdict_probe_ids=verdict_probe_ids,
            diagnostic_probe_ids=diagnostic_probe_ids,
        )
        for definition in SURFACE_DEFINITIONS
    ]
    for surface in surfaces:
        blockers.extend(surface["blockers"])
    state = "ready" if not blockers and len(surfaces) == 5 else "blocked"
    report = {
        "schema_version": 1,
        "artifact_type": "wmloop-constitutional-audit",
        "state": state,
        "source_manifest_path": str(manifest_path),
        "source_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "repo_root": str(base),
        "constitution_id": manifest.get("constitution_id"),
        "goal_id": manifest.get("goal_id"),
        "constitution_sha256": manifest.get("constitution_sha256"),
        "freeze_verification": freeze_state,
        "entry_count": len(entry_observations),
        "ready_entry_count": sum(1 for observation in entry_observations if observation["state"] == "ready"),
        "component_counts": dict(sorted(Counter(str(item.get("component")) for item in entry_observations).items())),
        "frozen_components": sorted({str(item.get("component")) for item in entry_observations}),
        "entry_observations": entry_observations,
        "verdict_probe_ids": verdict_probe_ids,
        "diagnostic_probe_ids": diagnostic_probe_ids,
        "probe_role_counts": _probe_role_counts(registry),
        "verdict_probe_ids_frozen": _has_ready_component(entries_by_component, "probe_registry"),
        "diagnostic_probe_ids_enter_verdict_evidence": False,
        "surface_count": len(surfaces),
        "ready_surface_count": sum(1 for surface in surfaces if surface["state"] == "ready"),
        "constitutional_surfaces": surfaces,
        "blocker_count": len(blockers),
        "blockers": blockers,
        "active_constitution_mutated": False,
        "m4_launch_allowed": False,
        "formal_training_allowed": False,
        "limitations": [
            "This audit is read-only and does not update the constitutional freeze manifest, goal config, probe registry or vendor checkout.",
            "A ready constitutional audit is necessary evidence for §6.6, but it is not M4 launch authorization.",
            "Diagnostic probes remain proposal-routing signals only; verifier-facing verdict evidence is limited to frozen verdict-role probes.",
        ],
    }
    return _write_report_bundle(report=report, output_root=destination, archive_db=archive_db, cas_root=cas_root)


def _verify_freeze(
    manifest: Mapping[str, Any],
    *,
    root: Path,
    blockers: list[dict[str, object]],
) -> dict[str, object]:
    expected_identity = manifest.get("constitution_sha256")
    actual_identity = _document_sha256(manifest) if isinstance(expected_identity, str) else None
    identity_ready = actual_identity == expected_identity
    if not identity_ready:
        blockers.append(
            _blocker(
                surface="freeze_manifest",
                detail="constitution_sha256 identity mismatch",
                component="constitution_sha256",
            )
        )
    try:
        verify_constitutional_freeze(manifest, root=root)
    except ConstitutionalFreezeError as exc:
        blockers.append(_blocker(surface="freeze_manifest", detail=str(exc), component="constitutional_freeze"))
        return {
            "state": "blocked",
            "identity_ready": identity_ready,
            "expected_constitution_sha256": expected_identity,
            "actual_constitution_sha256": actual_identity,
            "error": str(exc),
        }
    return {
        "state": "ready",
        "identity_ready": True,
        "expected_constitution_sha256": expected_identity,
        "actual_constitution_sha256": actual_identity,
        "error": None,
    }


def _entries(manifest: Mapping[str, Any], *, blockers: list[dict[str, object]]) -> list[Mapping[str, Any]]:
    raw_entries = manifest.get("entries")
    if not isinstance(raw_entries, list):
        blockers.append(_blocker(surface="freeze_manifest", detail="entries missing or invalid", component="entries"))
        return []
    entries: list[Mapping[str, Any]] = []
    for item in raw_entries:
        if isinstance(item, Mapping):
            entries.append(item)
        else:
            blockers.append(_blocker(surface="freeze_manifest", detail="entry is not an object", component="entries"))
    return entries


def _audit_entry(entry: Mapping[str, Any], *, root: Path) -> dict[str, object]:
    component = entry.get("component")
    path = entry.get("path")
    expected_sha256 = entry.get("sha256")
    expected_size = entry.get("size")
    blockers: list[dict[str, object]] = []
    if not isinstance(component, str) or not component:
        blockers.append(_blocker(surface="freeze_entry", detail="component invalid", component=component))
    if not isinstance(path, str) or not path:
        blockers.append(_blocker(surface="freeze_entry", detail="path invalid", component=component))
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        blockers.append(_blocker(surface="freeze_entry", detail="sha256 invalid", component=component, path=path))
    if not isinstance(expected_size, int) or expected_size < 0:
        blockers.append(_blocker(surface="freeze_entry", detail="size invalid", component=component, path=path))
    actual_sha256: str | None = None
    actual_size: int | None = None
    safe_regular_file = False
    if isinstance(path, str) and path:
        try:
            resolved = _resolve_inside(root, Path(path))
            metadata = os.lstat(resolved)
            safe_regular_file = stat.S_ISREG(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode)
            if not safe_regular_file:
                blockers.append(_blocker(surface="freeze_entry", detail="path is not a safe regular file", component=component, path=path))
            else:
                payload = resolved.read_bytes()
                actual_sha256 = hashlib.sha256(payload).hexdigest()
                actual_size = len(payload)
        except ConstitutionalAuditError as exc:
            blockers.append(_blocker(surface="freeze_entry", detail=str(exc), component=component, path=path))
        except OSError as exc:
            blockers.append(_blocker(surface="freeze_entry", detail=f"path unreadable: {exc}", component=component, path=path))
    if actual_sha256 is not None and isinstance(expected_sha256, str) and actual_sha256 != expected_sha256:
        blockers.append(_blocker(surface="freeze_entry", detail="sha256 mismatch", component=component, path=path))
    if actual_size is not None and isinstance(expected_size, int) and actual_size != expected_size:
        blockers.append(_blocker(surface="freeze_entry", detail="size mismatch", component=component, path=path))
    return {
        "component": component,
        "path": path,
        "state": "ready" if not blockers else "blocked",
        "safe_regular_file": safe_regular_file,
        "expected_sha256": expected_sha256,
        "actual_sha256": actual_sha256,
        "expected_size": expected_size,
        "actual_size": actual_size,
        "blockers": blockers,
    }


def _entries_by_component(entry_observations: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for observation in entry_observations:
        component = observation.get("component")
        if isinstance(component, str):
            grouped.setdefault(component, []).append(observation)
    return grouped


def _load_constitution_config(
    entries_by_component: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    root: Path,
    blockers: list[dict[str, object]],
) -> Mapping[str, Any] | None:
    entries = entries_by_component.get("constitution_config") or ()
    ready_entries = [entry for entry in entries if entry.get("state") == "ready" and isinstance(entry.get("path"), str)]
    if len(ready_entries) != 1:
        blockers.append(_blocker(surface="constitution_config", detail="ready constitution_config entry missing or not unique", component="constitution_config"))
        return None
    try:
        return _read_json_mapping(_resolve_inside(root, Path(str(ready_entries[0]["path"]))), "CONSTITUTIONAL_AUDIT_CONFIG_INVALID")[0]
    except ConstitutionalAuditError as exc:
        blockers.append(_blocker(surface="constitution_config", detail=str(exc), component="constitution_config", path=ready_entries[0].get("path")))
        return None


def _load_goal(
    config: Mapping[str, Any] | None,
    *,
    root: Path,
    blockers: list[dict[str, object]],
) -> Mapping[str, Any] | None:
    if config is None:
        return None
    raw_path = config.get("goal_spec")
    if not isinstance(raw_path, str) or not raw_path:
        blockers.append(_blocker(surface="goal_spec", detail="goal_spec path missing", component="goal_spec"))
        return None
    try:
        goal = load_yaml_document(_resolve_inside(root, Path(raw_path)))
    except Exception as exc:
        blockers.append(_blocker(surface="goal_spec", detail=f"goal_spec unreadable: {exc}", component="goal_spec", path=raw_path))
        return None
    if not isinstance(goal, Mapping):
        blockers.append(_blocker(surface="goal_spec", detail="goal_spec is not a mapping", component="goal_spec", path=raw_path))
        return None
    return goal


def _load_registry(
    config: Mapping[str, Any] | None,
    *,
    root: Path,
    blockers: list[dict[str, object]],
) -> ProbeRegistry | None:
    if config is None:
        return None
    raw_path = config.get("probe_registry")
    if not isinstance(raw_path, str) or not raw_path:
        blockers.append(_blocker(surface="probe_registry", detail="probe_registry path missing", component="probe_registry"))
        return None
    try:
        return load_probe_registry(Path(raw_path), root=root)
    except ProbeRegistryError as exc:
        blockers.append(_blocker(surface="probe_registry", detail=str(exc), component="probe_registry", path=raw_path))
        return None


def _verdict_probe_ids(
    *,
    manifest: Mapping[str, Any],
    config: Mapping[str, Any] | None,
    registry: ProbeRegistry | None,
    blockers: list[dict[str, object]],
) -> list[str]:
    ids = _string_list(manifest.get("verdict_probe_ids"))
    if ids is None:
        blockers.append(_blocker(surface="probe_registry", detail="manifest verdict_probe_ids invalid", component="verdict_probe_ids"))
        ids = []
    config_ids = _string_list(config.get("verdict_probe_ids")) if config is not None else None
    if config_ids is not None and ids and ids != config_ids:
        blockers.append(_blocker(surface="probe_registry", detail="manifest/config verdict_probe_ids mismatch", component="verdict_probe_ids"))
    if registry is not None and ids and tuple(ids) != registry.verdict_probe_ids:
        blockers.append(_blocker(surface="probe_registry", detail="manifest/registry verdict_probe_ids mismatch", component="verdict_probe_ids"))
    return ids


def _diagnostic_probe_ids(
    registry: ProbeRegistry | None,
    *,
    blockers: list[dict[str, object]],
) -> list[str]:
    if registry is None:
        return []
    ids = [probe.probe_id for probe in registry.probes if probe.role == "diagnostic"]
    if not ids:
        blockers.append(_blocker(surface="probe_registry", detail="diagnostic probe set is empty", component="probe_registry"))
    return ids


def _audit_probe_role_contracts(
    *,
    manifest: Mapping[str, Any],
    config: Mapping[str, Any] | None,
    registry: ProbeRegistry | None,
    verdict_probe_ids: Sequence[str],
    diagnostic_probe_ids: Sequence[str],
    blockers: list[dict[str, object]],
) -> None:
    if config is not None and _string_list(config.get("verdict_probe_ids")) is None:
        blockers.append(_blocker(surface="probe_registry", detail="config verdict_probe_ids invalid", component="verdict_probe_ids"))
    if registry is None:
        return
    overlap = sorted(set(verdict_probe_ids).intersection(diagnostic_probe_ids))
    if overlap:
        blockers.append(_blocker(surface="probe_registry", detail="probe cannot be both verdict and diagnostic: " + ",".join(overlap), component="probe_registry"))
    if set(verdict_probe_ids) != set(registry.verdict_probe_ids):
        blockers.append(_blocker(surface="probe_registry", detail="verdict probe freeze set does not match registry verdict role set", component="probe_registry"))
    manifest_ids = _string_list(manifest.get("verdict_probe_ids")) or []
    if set(diagnostic_probe_ids).intersection(manifest_ids):
        blockers.append(_blocker(surface="probe_registry", detail="diagnostic probe included in frozen verdict set", component="probe_registry"))


def _audit_surface(
    definition: Mapping[str, object],
    *,
    entries_by_component: Mapping[str, Sequence[Mapping[str, Any]]],
    config: Mapping[str, Any] | None,
    goal: Mapping[str, Any] | None,
    root: Path,
    verdict_probe_ids: Sequence[str],
    diagnostic_probe_ids: Sequence[str],
) -> dict[str, object]:
    surface_id = str(definition["surface_id"])
    blockers: list[dict[str, object]] = []
    required_components = tuple(str(item) for item in definition.get("required_components", ()))
    component_states: dict[str, object] = {}
    for component in required_components:
        entries = list(entries_by_component.get(component) or ())
        ready_count = sum(1 for entry in entries if entry.get("state") == "ready")
        component_states[component] = {"entry_count": len(entries), "ready_count": ready_count}
        if ready_count < 1:
            blockers.append(_blocker(surface=surface_id, detail="required frozen component missing or blocked", component=component))
    required_goal_fields = tuple(str(item) for item in definition.get("required_goal_fields", ()))
    goal_fields: dict[str, object] = {}
    for field in required_goal_fields:
        value = _nested_value(goal, field)
        ready = value is not None
        goal_fields[field] = {"present": ready, "value": value}
        if not ready:
            blockers.append(_blocker(surface=surface_id, detail="required goal field missing", component="goal_spec", path=field))
    if surface_id == "metric_collection_protocol":
        _audit_horizon_ladder_binding(
            config=config,
            goal=goal,
            entries_by_component=entries_by_component,
            component_states=component_states,
            root=root,
            blockers=blockers,
            surface_id=surface_id,
        )
    required_verdict_probes = tuple(str(item) for item in definition.get("required_verdict_probes", ()))
    for probe_id in required_verdict_probes:
        if probe_id not in verdict_probe_ids:
            blockers.append(_blocker(surface=surface_id, detail="required verdict probe missing", component="probe_registry", path=probe_id))
    if definition.get("requires_diagnostic_probes") is True and not diagnostic_probe_ids:
        blockers.append(_blocker(surface=surface_id, detail="diagnostic probes missing", component="probe_registry"))
    projection_ready = None
    if definition.get("requires_verdict_projection") is True:
        projection_ready = "code:verdict_evidence_projection" in component_states and component_states["code:verdict_evidence_projection"]["ready_count"] >= 1  # type: ignore[index]
        if not verdict_probe_ids:
            blockers.append(_blocker(surface=surface_id, detail="verdict projection has no frozen verdict probes", component="probe_registry"))
    return {
        "surface_id": surface_id,
        "document_component": definition["document_component"],
        "state": "ready" if not blockers else "blocked",
        "description": definition["description"],
        "required_components": list(required_components),
        "component_states": component_states,
        "required_goal_fields": list(required_goal_fields),
        "goal_fields": goal_fields,
        "required_verdict_probes": list(required_verdict_probes),
        "verdict_probe_ids": list(verdict_probe_ids),
        "diagnostic_probe_ids": list(diagnostic_probe_ids),
        "verdict_projection_ready": projection_ready,
        "blockers": blockers,
    }


def _audit_horizon_ladder_binding(
    *,
    config: Mapping[str, Any] | None,
    goal: Mapping[str, Any] | None,
    entries_by_component: Mapping[str, Sequence[Mapping[str, Any]]],
    component_states: dict[str, object],
    root: Path,
    blockers: list[dict[str, object]],
    surface_id: str,
) -> None:
    protocol = goal.get("eval_protocol") if isinstance(goal, Mapping) else None
    if not isinstance(protocol, Mapping):
        return
    ladder_path = protocol.get("horizon_ladder_path")
    if ladder_path is None:
        return
    if protocol.get("mode") != "per_environment_horizon_ladder":
        blockers.append(_blocker(surface=surface_id, detail="horizon ladder path present without ladder mode", component="goal_spec", path="eval_protocol.mode"))
    if not isinstance(ladder_path, str) or not ladder_path:
        blockers.append(_blocker(surface=surface_id, detail="horizon_ladder_path invalid", component="goal_spec", path="eval_protocol.horizon_ladder_path"))
        return
    config_ladder = config.get("horizon_ladder") if isinstance(config, Mapping) else None
    if config_ladder != ladder_path:
        blockers.append(_blocker(surface=surface_id, detail="constitution horizon_ladder path does not match goal eval_protocol", component="horizon_ladder", path=ladder_path))
    entries = list(entries_by_component.get("horizon_ladder") or ())
    ready_entries = [entry for entry in entries if entry.get("state") == "ready" and entry.get("path") == ladder_path]
    component_states["horizon_ladder"] = {"entry_count": len(entries), "ready_count": len(ready_entries)}
    if len(ready_entries) != 1:
        blockers.append(_blocker(surface=surface_id, detail="ready horizon_ladder entry missing or path mismatch", component="horizon_ladder", path=ladder_path))
        return
    try:
        ladder = load_yaml_document(_resolve_inside(root, Path(ladder_path)))
        validate_document("horizon_ladder", ladder, root=root)
    except (OSError, ContractValidationError, ConstitutionalAuditError) as exc:
        blockers.append(_blocker(surface=surface_id, detail=f"horizon_ladder invalid: {exc}", component="horizon_ladder", path=ladder_path))
        return
    if ladder.get("goal_id") != goal.get("goal_id"):
        blockers.append(_blocker(surface=surface_id, detail="horizon_ladder goal_id does not match goal_spec", component="horizon_ladder", path=ladder_path))
    expected_common = protocol.get("cross_environment_comparison_horizons")
    if expected_common is not None and ladder.get("cross_environment_comparison_horizons") != expected_common:
        blockers.append(_blocker(surface=surface_id, detail="horizon_ladder common horizons do not match goal eval_protocol", component="horizon_ladder", path=ladder_path))


def _probe_role_counts(registry: ProbeRegistry | None) -> dict[str, int]:
    if registry is None:
        return {}
    counts = Counter(probe.role for probe in registry.probes)
    return dict(sorted(counts.items()))


def _has_ready_component(entries_by_component: Mapping[str, Sequence[Mapping[str, Any]]], component: str) -> bool:
    return any(entry.get("state") == "ready" for entry in entries_by_component.get(component, ()))


def _nested_value(document: Mapping[str, Any] | None, dotted: str) -> object | None:
    if document is None:
        return None
    current: object = document
    for part in dotted.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _string_list(value: object) -> list[str] | None:
    if not isinstance(value, list):
        return None
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item:
            return None
        result.append(item)
    return result


def _resolve_inside(root: Path, path: Path) -> Path:
    resolved = (root / path).resolve() if not path.is_absolute() else path.resolve()
    if root not in resolved.parents and resolved != root:
        raise ConstitutionalAuditError("CONSTITUTIONAL_AUDIT_PATH_OUTSIDE_ROOT")
    if not resolved.is_file():
        raise ConstitutionalAuditError(f"CONSTITUTIONAL_AUDIT_PATH_MISSING:{path}")
    return resolved


def _read_json_mapping(path: Path, error_code: str) -> tuple[Mapping[str, Any], bytes, Path]:
    try:
        resolved = Path(path).resolve(strict=True)
        payload_bytes = resolved.read_bytes()
        payload = json.loads(payload_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise ConstitutionalAuditError(f"{error_code}:{path}") from exc
    if not isinstance(payload, Mapping):
        raise ConstitutionalAuditError(f"{error_code}:{resolved}")
    return payload, payload_bytes, resolved


def _write_report_bundle(
    *,
    report: Mapping[str, object],
    output_root: Path,
    archive_db: Path | None,
    cas_root: Path | None,
) -> dict[str, object]:
    temporary = output_root.parent / f".{output_root.name}.{uuid.uuid4().hex}.tmp"
    report_bytes = _canonical_json_bytes(report)
    markdown_bytes = _render_markdown(report).encode("utf-8")
    try:
        output_root.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary.mkdir(mode=0o700)
        _write_bytes_atomic(temporary / "constitutional-audit.json", report_bytes)
        _write_bytes_atomic(temporary / "constitutional-audit.md", markdown_bytes)
        cas_refs: dict[str, str] = {}
        archive = ArchiveStore(archive_db) if archive_db is not None else None
        if archive is not None or cas_root is not None:
            root = cas_root if cas_root is not None else Path(archive_db).resolve().parent  # type: ignore[union-attr]
            cas = ContentAddressedStore(Path(root))
            for name, payload, media_type in (
                ("constitutional_audit_json", report_bytes, "application/json"),
                ("constitutional_audit_markdown", markdown_bytes, "text/markdown"),
            ):
                ref = cas.put_bytes(payload, media_type=media_type).uri
                cas_refs[name] = ref
                if archive is not None:
                    archive.record_artifact_reference(ref)
        manifest = {
            "schema_version": 1,
            "artifact_type": "wmloop-constitutional-audit-manifest",
            "state": report["state"],
            "source_manifest_path": report["source_manifest_path"],
            "source_manifest_sha256": report["source_manifest_sha256"],
            "constitution_id": report["constitution_id"],
            "goal_id": report["goal_id"],
            "constitution_sha256": report["constitution_sha256"],
            "entry_count": report["entry_count"],
            "ready_entry_count": report["ready_entry_count"],
            "surface_count": report["surface_count"],
            "ready_surface_count": report["ready_surface_count"],
            "verdict_probe_ids": report["verdict_probe_ids"],
            "diagnostic_probe_ids": report["diagnostic_probe_ids"],
            "blocker_count": report["blocker_count"],
            "blockers": report["blockers"],
            "active_constitution_mutated": report["active_constitution_mutated"],
            "m4_launch_allowed": report["m4_launch_allowed"],
            "formal_training_allowed": report["formal_training_allowed"],
            "report_path": str(output_root / "constitutional-audit.json"),
            "markdown_path": str(output_root / "constitutional-audit.md"),
            "cas_refs": cas_refs,
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


def _render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Constitutional Audit",
        "",
        f"State: `{report['state']}`",
        f"Constitution: `{report['constitution_id']}`",
        f"Goal: `{report['goal_id']}`",
        f"Freeze verification: `{report['freeze_verification']['state']}`",
        f"Entries: `{report['ready_entry_count']}/{report['entry_count']}` ready",
        f"Surfaces: `{report['ready_surface_count']}/{report['surface_count']}` ready",
        f"Blockers: `{report['blocker_count']}`",
        "",
        "| Surface | State | Required Components | Blockers |",
        "|:--|:--|:--|--:|",
    ]
    for surface in report["constitutional_surfaces"]:
        lines.append(
            f"| `{surface['surface_id']}` | `{surface['state']}` | "
            f"`{','.join(surface['required_components'])}` | {len(surface['blockers'])} |"
        )
    lines.extend(
        [
            "",
            "## Probe Roles",
            "",
            f"- Verdict probes: `{','.join(report['verdict_probe_ids'])}`",
            f"- Diagnostic probes: `{','.join(report['diagnostic_probe_ids'])}`",
            f"- Diagnostic probes enter verdict evidence: `{report['diagnostic_probe_ids_enter_verdict_evidence']}`",
        ]
    )
    if report["blockers"]:
        lines.extend(["", "## Blockers", ""])
        for blocker in report["blockers"]:
            component = blocker.get("component")
            path = blocker.get("path")
            suffix = ""
            if component is not None:
                suffix += f" component={component}"
            if path is not None:
                suffix += f" path={path}"
            lines.append(f"- {blocker.get('surface')}: {blocker.get('detail')}{suffix}")
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in report["limitations"])
    return "\n".join(lines) + "\n"


def _document_sha256(document: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in document.items() if key != "constitution_sha256"}
    content = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise ConstitutionalAuditError("CONSTITUTIONAL_AUDIT_OUTPUT_EXISTS")
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


def _blocker(
    *,
    surface: str,
    detail: str,
    component: object = None,
    path: object = None,
) -> dict[str, object]:
    blocker: dict[str, object] = {"surface": surface, "detail": detail}
    if component is not None:
        blocker["component"] = component
    if path is not None:
        blocker["path"] = path
    return blocker


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="audit a frozen constitutional manifest")
    run.add_argument("--constitution-manifest", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--repo-root", type=Path)
    run.add_argument("--archive-db", type=Path)
    run.add_argument("--cas-root", type=Path)
    args = parser.parse_args(argv)
    if args.command == "run":
        manifest = run_constitutional_audit(
            constitution_manifest=args.constitution_manifest,
            output_root=args.output_root,
            repo_root=args.repo_root,
            archive_db=args.archive_db,
            cas_root=args.cas_root,
        )
        print(json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
