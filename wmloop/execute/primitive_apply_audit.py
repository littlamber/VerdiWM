"""Audit every registered primitive apply template in an isolated worktree."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from wmloop.archive.store import ArchiveStore, ContentAddressedStore
from wmloop.execute.primitive_smoke import _default_hook_ios
from wmloop.execute.sandbox import SandboxLease, WorktreeSandbox
from wmloop.primitives.registry import PrimitiveManifest, PrimitiveRegistry
from wmloop.primitives.render import PrimitiveRenderer
from wmloop.vendor import verify_vendor_checkout


class PrimitiveApplyAuditError(RuntimeError):
    """The all-primitive apply audit failed closed."""


def run_primitive_apply_audit(
    *,
    repo_root: Path,
    output_root: Path,
    archive_db: Path | None = None,
    cas_root: Path | None = None,
) -> dict[str, object]:
    """Render and apply every primitive individually, then write a receipt."""

    root = Path(repo_root).resolve()
    vendor_root = root / "vendor" / "ACWM-Phys"
    source_revision = verify_vendor_checkout(root)
    registry = PrimitiveRegistry.from_root(root)
    renderer = PrimitiveRenderer(registry)
    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise PrimitiveApplyAuditError("PRIMITIVE_APPLY_AUDIT_OUTPUT_EXISTS")
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    sandbox_root = destination.parent / f".{destination.name}.sandbox.{uuid.uuid4().hex}"
    sandbox = WorktreeSandbox(vendor_root=vendor_root, runs_root=sandbox_root)
    lease: SandboxLease | None = None
    worktree_removed = False
    try:
        records: list[dict[str, object]] = []
        final_changed_path_set: set[str] = set()
        status_sections: list[str] = []
        last_worktree = ""
        for name in registry.names():
            lease = sandbox.create(trial_id=f"primitive-apply-{name.replace('_', '-')}", expected_revision=source_revision)
            last_worktree = str(lease.worktree)
            before_paths = set(_git_changed_paths(lease.worktree))
            manifest = registry.manifest(name)
            params = sample_params(manifest)
            rendered = renderer.render_checked(
                worktree=lease.worktree,
                interventions=({"primitive": name, "params": params},),
                hook_ios=_default_hook_ios(),
            )[0]
            _apply_diff(lease.worktree, rendered.diff)
            sidecar_path = Path("wmloop_interventions") / f"{name}.json"
            sidecar = _read_sidecar(lease.worktree / sidecar_path, expected_name=name)
            changed_paths = _git_changed_paths(lease.worktree)
            if sidecar_path.as_posix() not in changed_paths:
                raise PrimitiveApplyAuditError(f"PRIMITIVE_APPLY_AUDIT_SIDECAR_MISSING:{name}")
            changed_from_apply = sorted((set(changed_paths) - before_paths) | set(_paths_from_diff(rendered.diff)))
            final_changed_path_set.update(changed_paths)
            status_sections.append(f"## {name}\n{_git_status_short(lease.worktree)}")
            records.append(
                {
                    "primitive": name,
                    "layer": manifest.layer,
                    "hook": manifest.hooks[0],
                    "params": params,
                    "diff_sha256": rendered.sha256,
                    "sidecar_path": sidecar_path.as_posix(),
                    "changed_paths_from_apply": changed_from_apply,
                    "changed_paths_after_apply": changed_paths,
                    "materialization_state": sidecar["materialization_state"],
                    "runtime_hook_paths": list(sidecar.get("runtime_hook_paths", ())),
                    "intent_to_code_contract": dict(sidecar["intent_to_code_contract"]),
                }
            )
            sandbox.remove(lease)
            lease = None
        worktree_removed = True
        final_changed_paths = sorted(final_changed_path_set)
        expected_paths = [f"wmloop_interventions/{name}.json" for name in registry.names()]
        if not set(expected_paths).issubset(set(final_changed_paths)):
            raise PrimitiveApplyAuditError("PRIMITIVE_APPLY_AUDIT_CHANGED_PATHS_INVALID")
        status_short = "\n".join(status_sections)
        report = {
            "schema_version": 1,
            "artifact_type": "wmloop-m2-primitive-apply-all-report",
            "state": "ready",
            "source_revision": source_revision,
            "registry_digest": registry.digest(),
            "primitive_count": len(records),
            "ready_count": len(records),
            "worktree": last_worktree,
            "worktree_removed": worktree_removed,
            "records": records,
            "final_changed_paths": final_changed_paths,
            "sidecar_paths": expected_paths,
            "git_status_short": status_short,
            "limitations": [
                "apply-all proves template rendering and clean git application, not closed-loop training effectiveness",
                "closed-loop eligibility is decided by the primitive materialization gate after runtime/training evidence is attached",
            ],
        }
        return _write_report_bundle(report=report, output_root=destination, archive_db=archive_db, cas_root=cas_root)
    except Exception:
        if lease is not None and not worktree_removed:
            try:
                sandbox.remove(lease)
            except Exception:
                pass
        raise
    finally:
        if sandbox_root.exists() or sandbox_root.is_symlink():
            shutil.rmtree(sandbox_root, ignore_errors=True)


def sample_params(manifest: PrimitiveManifest) -> dict[str, object]:
    """Build minimal valid params from the primitive's JSON Schema subset."""

    schema = manifest.params_schema
    required = schema.get("required")
    properties = schema.get("properties")
    if not isinstance(required, list) or not isinstance(properties, Mapping):
        raise PrimitiveApplyAuditError(f"PRIMITIVE_APPLY_AUDIT_SCHEMA_INVALID:{manifest.name}")
    params: dict[str, object] = {}
    for name in required:
        if not isinstance(name, str) or name not in properties:
            raise PrimitiveApplyAuditError(f"PRIMITIVE_APPLY_AUDIT_SCHEMA_INVALID:{manifest.name}")
        prop = properties[name]
        if not isinstance(prop, Mapping):
            raise PrimitiveApplyAuditError(f"PRIMITIVE_APPLY_AUDIT_SCHEMA_INVALID:{manifest.name}")
        params[name] = _sample_value(prop)
    return params


def _sample_value(schema: Mapping[str, Any]) -> object:
    kind = schema.get("type")
    if kind == "integer":
        value = int(schema.get("minimum", 1))
        maximum = schema.get("maximum")
        if isinstance(maximum, int) and value > maximum:
            value = maximum
        return value
    if kind == "number":
        if "exclusiveMinimum" in schema:
            value = float(schema["exclusiveMinimum"]) + 0.1
        else:
            value = float(schema.get("minimum", 0.0))
        maximum = schema.get("maximum")
        if isinstance(maximum, (int, float)) and math.isfinite(float(maximum)) and value > float(maximum):
            value = float(maximum)
        return value
    if kind == "string":
        return "smoke"
    raise PrimitiveApplyAuditError("PRIMITIVE_APPLY_AUDIT_SCHEMA_TYPE_UNSUPPORTED")


def _apply_diff(worktree: Path, diff: str) -> None:
    completed = subprocess.run(
        ["git", "-C", str(worktree), "apply", "--whitespace=error-all", "-"],
        input=diff,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise PrimitiveApplyAuditError("PRIMITIVE_APPLY_AUDIT_GIT_APPLY_FAILED")


def _read_sidecar(path: Path, *, expected_name: str) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise PrimitiveApplyAuditError(f"PRIMITIVE_APPLY_AUDIT_SIDECAR_INVALID:{expected_name}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PrimitiveApplyAuditError(f"PRIMITIVE_APPLY_AUDIT_SIDECAR_INVALID:{expected_name}") from exc
    if (
        not isinstance(payload, Mapping)
        or payload.get("artifact_type") != "wmloop-materialized-primitive-smoke"
        or payload.get("primitive") != expected_name
        or payload.get("materialization_state") not in {"smoke_sidecar_only", "acwm_runtime_hook_smoke"}
    ):
        raise PrimitiveApplyAuditError(f"PRIMITIVE_APPLY_AUDIT_SIDECAR_INVALID:{expected_name}")
    _validate_intent_contract(payload.get("intent_to_code_contract"), expected_name=expected_name)
    return payload


def _validate_intent_contract(value: object, *, expected_name: str) -> None:
    if not isinstance(value, Mapping):
        raise PrimitiveApplyAuditError(f"PRIMITIVE_APPLY_AUDIT_INTENT_CONTRACT_MISSING:{expected_name}")
    required_strings = ("method_intent", "runtime_behavior", "declared_proxy")
    for field in required_strings:
        item = value.get(field)
        if not isinstance(item, str) or not item.strip():
            raise PrimitiveApplyAuditError(f"PRIMITIVE_APPLY_AUDIT_INTENT_CONTRACT_INVALID:{expected_name}:{field}")
    not_claimed = value.get("not_claimed")
    if (
        not isinstance(not_claimed, list)
        or not not_claimed
        or any(not isinstance(item, str) or not item.strip() for item in not_claimed)
    ):
        raise PrimitiveApplyAuditError(f"PRIMITIVE_APPLY_AUDIT_INTENT_CONTRACT_INVALID:{expected_name}:not_claimed")


def _git_changed_paths(worktree: Path) -> list[str]:
    paths: list[str] = []
    for line in _git_status_short(worktree):
        if len(line) < 4:
            raise PrimitiveApplyAuditError("PRIMITIVE_APPLY_AUDIT_GIT_STATUS_INVALID")
        path = line[3:].strip()
        if " -> " in path:
            path = path.rsplit(" -> ", 1)[1]
        if path:
            paths.append(path)
    return sorted(paths)


def _paths_from_diff(diff: str) -> list[str]:
    paths: set[str] = set()
    for line in diff.splitlines():
        if not line.startswith(("--- ", "+++ ")):
            continue
        raw = line[4:].split("\t", 1)[0].strip()
        if raw == "/dev/null":
            continue
        normalized = raw.removeprefix("a/").removeprefix("b/")
        if normalized:
            paths.add(normalized)
    return sorted(paths)


def _git_status_short(worktree: Path) -> list[str]:
    completed = subprocess.run(
        ["git", "-C", str(worktree), "status", "--short", "--untracked-files=all"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise PrimitiveApplyAuditError("PRIMITIVE_APPLY_AUDIT_GIT_STATUS_FAILED")
    return [line for line in completed.stdout.splitlines() if line]


def _write_report_bundle(
    *,
    report: Mapping[str, object],
    output_root: Path,
    archive_db: Path | None,
    cas_root: Path | None,
) -> dict[str, object]:
    destination = Path(output_root).resolve()
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    report_bytes = _canonical_json_bytes(report)
    markdown_bytes = _render_markdown(report).encode("utf-8")
    cas_refs: dict[str, str] = {}
    try:
        temporary.mkdir(mode=0o700)
        _write_bytes_atomic(temporary / "primitive-apply-all.json", report_bytes)
        _write_bytes_atomic(temporary / "primitive-apply-all.md", markdown_bytes)
        archive = ArchiveStore(archive_db) if archive_db is not None else None
        if archive is not None or cas_root is not None:
            root = cas_root if cas_root is not None else Path(archive_db).resolve().parent  # type: ignore[union-attr]
            cas = ContentAddressedStore(Path(root))
            for name, payload, media_type in (
                ("primitive_apply_all_json", report_bytes, "application/json"),
                ("primitive_apply_all_markdown", markdown_bytes, "text/markdown"),
            ):
                ref = cas.put_bytes(payload, media_type=media_type).uri
                cas_refs[name] = ref
                if archive is not None:
                    archive.record_artifact_reference(ref)
        manifest = {
            "schema_version": 1,
            "artifact_type": "wmloop-m2-primitive-apply-all-manifest",
            "state": report["state"],
            "primitive_count": report["primitive_count"],
            "ready_count": report["ready_count"],
            "report_path": str(destination / "primitive-apply-all.json"),
            "markdown_path": str(destination / "primitive-apply-all.md"),
            "cas_refs": cas_refs,
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


def _render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# M2 Primitive Apply-All Audit",
        "",
        f"State: `{report['state']}`",
        f"Ready primitives: `{report['ready_count']}/{report['primitive_count']}`",
        "",
        "| Primitive | Layer | Hook | Materialization | Sidecar |",
        "|:--|:--|:--|:--|:--|",
    ]
    for record in report["records"]:
        lines.append(
            f"| {record['primitive']} | {record['layer']} | {record['hook']} | "
            f"{record['materialization_state']} | {record['sidecar_path']} |"
        )
    lines.extend(["", "## Limitations", ""])
    for item in report["limitations"]:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise PrimitiveApplyAuditError("PRIMITIVE_APPLY_AUDIT_OUTPUT_EXISTS")
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
    run = commands.add_parser("run", help="run all-primitive apply audit")
    run.add_argument("--repo-root", type=Path, default=Path("."))
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--archive-db", type=Path)
    run.add_argument("--cas-root", type=Path)
    args = parser.parse_args(argv)
    if args.command == "run":
        result = run_primitive_apply_audit(
            repo_root=args.repo_root,
            output_root=args.output_root,
            archive_db=args.archive_db,
            cas_root=args.cas_root,
        )
        print(json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
