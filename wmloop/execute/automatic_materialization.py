"""Receipt-bound automatic materialization in an isolated Git snapshot."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from wmloop.archive.store import ArchiveInvariantError, ContentAddressedStore
from wmloop.contracts import ContractValidationError, validate_document
from wmloop.control.module_manufacturing import (
    ModuleManufacturingError,
    load_module_manufacturing_work_order,
)
from wmloop.execute.agent_staging import AgentRepairSession, AgentStagingError, CommandReceipt


class AutomaticMaterializationError(RuntimeError):
    """An automatic implementation transaction failed a trust boundary."""


_SENSITIVE_INHERITED_ENVIRONMENT_PREFIXES = (
    "ANTHROPIC_",
    "CODEX_",
    "LARK_",
    "LARKSUITE_",
    "OPENAI_",
)
_SENSITIVE_INHERITED_ENVIRONMENT_SUFFIXES = (
    "_API_KEY",
    "_PASSWORD",
    "_SECRET",
    "_TOKEN",
)


def materialization_plan_digest(plan: Mapping[str, object]) -> str:
    payload = {key: value for key, value in plan.items() if key != "plan_digest"}
    return _sha256(_canonical_json(payload))


def validate_automatic_materialization_plan(
    plan: Mapping[str, object], *, root: Path | None = None
) -> None:
    try:
        validate_document("automatic_materialization_plan", plan, root=root)
    except ContractValidationError as exc:
        raise AutomaticMaterializationError(
            f"AUTOMATIC_MATERIALIZATION_PLAN_INVALID:{exc}"
        ) from exc
    if plan.get("plan_digest") != materialization_plan_digest(plan):
        raise AutomaticMaterializationError("AUTOMATIC_MATERIALIZATION_PLAN_DIGEST_MISMATCH")
    checks = plan.get("fixed_checks")
    assert isinstance(checks, list)
    labels = [str(row["label"]) for row in checks if isinstance(row, Mapping)]
    if len(labels) != len(checks) or len(labels) != len(set(labels)):
        raise AutomaticMaterializationError("AUTOMATIC_MATERIALIZATION_CHECK_LABEL_INVALID")
    if "agent_materialization" in labels:
        raise AutomaticMaterializationError("AUTOMATIC_MATERIALIZATION_CHECK_LABEL_RESERVED")
    if plan.get("preserve_codex_auth") is not False:
        raise AutomaticMaterializationError("AUTOMATIC_MATERIALIZATION_CODEX_AUTH_FORBIDDEN")
    inherited = [str(value) for value in plan["inherit_environment_keys"]]
    if any(_is_sensitive_environment_key(key) for key in inherited):
        raise AutomaticMaterializationError(
            "AUTOMATIC_MATERIALIZATION_SENSITIVE_ENVIRONMENT_INHERIT_FORBIDDEN"
        )
    manufacturing_fields = (
        "manufacturing_mode",
        "manufacturing_work_order_id",
        "manufacturing_work_order_digest",
        "manufacturing_work_order_sha256",
        "manufacturing_request_id",
        "manufacturing_abi_id",
        "dependency_abi_ids",
        "portfolio_entry_ids",
        "protected_metrics",
        "evaluator_binding",
    )
    present = [field for field in manufacturing_fields if field in plan]
    if present and len(present) != len(manufacturing_fields):
        raise AutomaticMaterializationError(
            "AUTOMATIC_MATERIALIZATION_MANUFACTURING_BINDING_INCOMPLETE"
        )
    if present and plan.get("manufacturing_mode") != "intervention":
        raise AutomaticMaterializationError(
            "AUTOMATIC_MATERIALIZATION_MANUFACTURING_MODE_INVALID"
        )


def run_automatic_materialization(
    *,
    plan_path: Path,
    work_order_path: Path,
    idea_path: Path,
    source_root: Path,
    output_root: Path,
    project_root: Path | None = None,
    manufacturing_work_order_path: Path | None = None,
) -> dict[str, object]:
    """Run one resumable-at-terminal materialization transaction."""

    schema_root = Path(project_root).resolve() if project_root is not None else None
    plan_path = _require_file(plan_path, "AUTOMATIC_MATERIALIZATION_PLAN_FILE_INVALID")
    work_order_path = _require_file(
        work_order_path, "AUTOMATIC_MATERIALIZATION_WORK_ORDER_FILE_INVALID"
    )
    idea_path = _require_file(idea_path, "AUTOMATIC_MATERIALIZATION_IDEA_FILE_INVALID")
    source = _require_directory(source_root, "AUTOMATIC_MATERIALIZATION_SOURCE_ROOT_INVALID")
    plan = _load(plan_path, "AUTOMATIC_MATERIALIZATION_PLAN_FILE_INVALID")
    work_order = _load(
        work_order_path, "AUTOMATIC_MATERIALIZATION_WORK_ORDER_FILE_INVALID"
    )
    idea = _load(idea_path, "AUTOMATIC_MATERIALIZATION_IDEA_FILE_INVALID")
    validate_automatic_materialization_plan(plan, root=schema_root)
    _validate_research_binding(plan=plan, work_order=work_order, idea=idea)
    manufacturing_file = None
    manufacturing_order = None
    if manufacturing_work_order_path is not None:
        manufacturing_file = _require_file(
            manufacturing_work_order_path,
            "AUTOMATIC_MATERIALIZATION_MANUFACTURING_WORK_ORDER_FILE_INVALID",
        )
        try:
            manufacturing_order = load_module_manufacturing_work_order(
                manufacturing_file,
                expected_sha256=str(plan.get("manufacturing_work_order_sha256") or ""),
                root=schema_root,
            )
        except ModuleManufacturingError as exc:
            raise AutomaticMaterializationError(
                f"AUTOMATIC_MATERIALIZATION_MANUFACTURING_WORK_ORDER_INVALID:{exc}"
            ) from exc
        _validate_manufacturing_binding(plan=plan, order=manufacturing_order)
    elif plan.get("manufacturing_work_order_id") is not None:
        raise AutomaticMaterializationError(
            "AUTOMATIC_MATERIALIZATION_MANUFACTURING_WORK_ORDER_REQUIRED"
        )

    destination = Path(output_root).expanduser()
    if destination.is_symlink():
        raise AutomaticMaterializationError("AUTOMATIC_MATERIALIZATION_OUTPUT_INVALID")
    destination = destination.resolve()
    if destination == source or source in destination.parents:
        raise AutomaticMaterializationError("AUTOMATIC_MATERIALIZATION_OUTPUT_INSIDE_SOURCE")
    input_lock = _input_lock(
        plan_path=plan_path,
        plan=plan,
        work_order_path=work_order_path,
        idea_path=idea_path,
        source=source,
        manufacturing_work_order_path=manufacturing_file,
    )
    resumed = _resume_terminal(destination, input_lock=input_lock)
    if resumed is not None:
        return resumed
    destination.mkdir(mode=0o700, parents=True, exist_ok=False)
    _write_json(destination / "input-lock.json", input_lock)

    source_status_before = _git_output(source, ("status", "--porcelain=v1", "-z"))
    source_revision = _git_revision(source)
    workspace = destination / "workspace"
    snapshot = _create_source_snapshot(
        source=source,
        workspace=workspace,
        plan=plan,
        source_revision=source_revision,
    )
    _write_json(destination / "source-snapshot-receipt.json", snapshot)
    prompt = _render_prompt(
        plan=plan,
        work_order=work_order,
        idea=idea,
        manufacturing_order=manufacturing_order,
    )
    prompt_path = destination / "materialization-prompt.txt"
    _write_bytes(prompt_path, prompt.encode("utf-8"))

    fixed_checks = plan["fixed_checks"]
    assert isinstance(fixed_checks, list)
    required_labels = [str(row["label"]) for row in fixed_checks if isinstance(row, Mapping)]
    environment = _agent_environment(destination=destination, plan=plan)
    session = AgentRepairSession(
        worktree=workspace,
        staging_root=destination / "staging",
        candidate_id=str(plan["candidate_id"]),
        source_revision=str(snapshot["snapshot_revision"]),
        registry_digest=str(plan["plan_digest"]),
        required_check_labels=required_labels,
        environment=environment,
        max_command_timeout_seconds=max(
            float(plan["agent_timeout_seconds"]),
            *(float(row["timeout_seconds"]) for row in fixed_checks if isinstance(row, Mapping)),
        ),
    )
    placeholders = _placeholders(
        workspace=workspace,
        destination=destination,
        prompt=prompt,
        prompt_path=prompt_path,
        plan=plan,
        idea=idea,
    )
    receipts: list[CommandReceipt] = []
    blockers: list[dict[str, object]] = []
    agent_receipt = session.run(
        label="agent_materialization",
        argv=_expand_argv(plan["agent_command"], placeholders),
        timeout_seconds=float(plan["agent_timeout_seconds"]),
    )
    receipts.append(agent_receipt)
    if not agent_receipt.passed:
        blockers.append({"code": "AGENT_MATERIALIZATION_FAILED"})
    for raw in fixed_checks:
        assert isinstance(raw, Mapping)
        receipt = session.run(
            label=str(raw["label"]),
            argv=_expand_argv(raw["argv"], placeholders),
            timeout_seconds=float(raw["timeout_seconds"]),
        )
        receipts.append(receipt)
        if not receipt.passed:
            blockers.append(
                {
                    "code": "FIXED_CHECK_FAILED",
                    "label": receipt.label,
                    "exit_code": receipt.exit_code,
                    "timed_out": receipt.timed_out,
                }
            )

    staged = None
    try:
        staged = session.seal()
    except AgentStagingError as exc:
        blockers.append({"code": "STAGING_SEAL_FAILED", "detail": str(exc)})
    changed_paths: list[str] = []
    implementation_revision = None
    descriptor = None
    descriptor_path = workspace / str(plan["descriptor_path"])
    if staged is not None:
        changed_paths = list(staged.changed_paths)
        blockers.extend(_path_blockers(changed_paths, plan=plan))
        try:
            descriptor = _load_descriptor(
                descriptor_path,
                workspace=workspace,
                candidate_id=str(plan["candidate_id"]),
                idea_id=str(idea["idea_id"]),
                changed_paths=changed_paths,
                root=schema_root,
            )
        except AutomaticMaterializationError as exc:
            blockers.append({"code": "MATERIALIZATION_DESCRIPTOR_INVALID", "detail": str(exc)})
        if descriptor is not None:
            blockers.extend(
                _source_component_mapping_blockers(
                    descriptor=descriptor,
                    idea=idea,
                    work_order=work_order,
                )
            )
        if descriptor is not None and descriptor.get("declared_compromises"):
            blockers.append({"code": "DECLARED_COMPROMISE_REQUIRES_ABSTENTION"})
        if not any(blocker["code"] in {"STAGING_SEAL_FAILED", "FORBIDDEN_PATH_CHANGED"} for blocker in blockers):
            implementation_revision = _commit_implementation(
                workspace, candidate_id=str(plan["candidate_id"])
            )

    if plan.get("candidate_template") is None:
        blockers.append({"code": "CANDIDATE_TEMPLATE_MISSING"})
    source_status_after = _git_output(source, ("status", "--porcelain=v1", "-z"))
    if source_status_after != source_status_before or _git_revision(source) != source_revision:
        raise AutomaticMaterializationError("AUTOMATIC_MATERIALIZATION_SOURCE_MUTATED")

    state = "ready_for_candidate_compilation" if not blockers else "blocked"
    cas = ContentAddressedStore(destination)
    evidence_refs = _archive_transaction_inputs(
        cas=cas,
        plan_path=plan_path,
        work_order_path=work_order_path,
        idea_path=idea_path,
        manufacturing_work_order_path=manufacturing_file,
        snapshot_path=destination / "source-snapshot-receipt.json",
        prompt_path=prompt_path,
        staged=staged,
        descriptor_path=descriptor_path if descriptor is not None else None,
    )
    receipt = {
        "schema_version": 1,
        "artifact_type": "verdiwm-automatic-materialization-receipt",
        "state": state,
        "candidate_id": plan["candidate_id"],
        "idea_id": idea["idea_id"],
        "plan_id": plan["plan_id"],
        "plan_digest": plan["plan_digest"],
        "input_sha256": input_lock["input_sha256"],
        "source_revision": source_revision,
        "source_status_sha256": _sha256(source_status_before),
        "snapshot_revision": snapshot["snapshot_revision"],
        "implementation_revision": implementation_revision,
        "worktree_diff_sha256": staged.worktree_diff_sha256 if staged is not None else None,
        "changed_paths": changed_paths,
        "required_check_labels": required_labels,
        "command_receipts": [row.to_document() for row in receipts],
        "descriptor_sha256": _sha256(descriptor_path.read_bytes()) if descriptor is not None else None,
        "blockers": blockers,
        "evidence_refs": sorted(evidence_refs),
        "side_effects": {
            "source_workspace_mutated": False,
            "isolated_snapshot_created": True,
            "isolated_implementation_committed": implementation_revision is not None,
            "gpu_execution_started": False,
            "candidate_compilation_authority": state == "ready_for_candidate_compilation",
            "gpu_scheduling_authority": False,
            "promotion_authority": False,
        },
        "claim_boundary": (
            "A ready receipt grants candidate-compilation authority only. GPU scheduling "
            "and promotion still require their independent admission and frozen verifier."
        ),
    }
    if manufacturing_order is not None:
        receipt.update(
            {
                "manufacturing_mode": manufacturing_order["manufacturing_mode"],
                "manufacturing_work_order_id": manufacturing_order["work_order_id"],
                "manufacturing_work_order_digest": manufacturing_order[
                    "work_order_digest"
                ],
                "manufacturing_request_id": manufacturing_order[
                    "manufacturing_request"
                ]["request_id"],
                "manufacturing_abi_id": manufacturing_order["target_abi"]["abi_id"],
                "portfolio_entry_ids": list(
                    manufacturing_order["portfolio_entry_ids"]
                ),
                "protected_metrics": list(manufacturing_order["protected_metrics"]),
            }
        )
    _validate_document("automatic_materialization_receipt", receipt, root=schema_root)
    receipt_path = destination / "receipt.json"
    _write_json(receipt_path, receipt)
    receipt_sha256 = _sha256(receipt_path.read_bytes())
    catalog_path = None
    catalog_sha256 = None
    if state == "ready_for_candidate_compilation":
        assert descriptor is not None
        catalog = _candidate_catalog(
            plan=plan,
            idea=idea,
            descriptor=descriptor,
            workspace=workspace,
            receipt_path=receipt_path,
            receipt_sha256=receipt_sha256,
        )
        _validate_document("method_candidate_catalog", catalog, root=schema_root)
        catalog_path = destination / "candidate-catalog.json"
        _write_json(catalog_path, catalog)
        catalog_sha256 = _sha256(catalog_path.read_bytes())
    capability_gap_path = None
    capability_gap_sha256 = None
    if state == "blocked":
        capability_gap = {
            "schema_version": 1,
            "artifact_type": "verdiwm-materialization-capability-gap",
            "candidate_id": plan["candidate_id"],
            "idea_id": idea["idea_id"],
            "state": "capability_gap",
            "implementation_revision": implementation_revision,
            "blockers": blockers,
            "evidence_refs": sorted(evidence_refs),
            "claim_boundary": (
                "The implementation is not executable research authority. Resolve every "
                "recorded blocker in a new versioned transaction before compilation."
            ),
        }
        capability_gap_path = destination / "capability-gap.json"
        _write_json(capability_gap_path, capability_gap)
        capability_gap_sha256 = _sha256(capability_gap_path.read_bytes())
    manifest = {
        "schema_version": 1,
        "artifact_type": "verdiwm-automatic-materialization-manifest",
        "state": state,
        "candidate_id": plan["candidate_id"],
        "input_sha256": input_lock["input_sha256"],
        "receipt_path": str(receipt_path),
        "receipt_sha256": receipt_sha256,
        "candidate_catalog_path": str(catalog_path) if catalog_path is not None else None,
        "candidate_catalog_sha256": catalog_sha256,
        "capability_gap_path": (
            str(capability_gap_path) if capability_gap_path is not None else None
        ),
        "capability_gap_sha256": capability_gap_sha256,
        "workspace_path": str(workspace),
        "implementation_revision": implementation_revision,
        "blocker_count": len(blockers),
    }
    if manufacturing_order is not None:
        manifest.update(
            {
                "manufacturing_mode": manufacturing_order["manufacturing_mode"],
                "manufacturing_work_order_id": manufacturing_order["work_order_id"],
                "manufacturing_request_id": manufacturing_order[
                    "manufacturing_request"
                ]["request_id"],
            }
        )
    _write_json(destination / "manifest.json", manifest)
    return manifest


def _create_source_snapshot(
    *,
    source: Path,
    workspace: Path,
    plan: Mapping[str, object],
    source_revision: str,
) -> dict[str, object]:
    completed = subprocess.run(
        ["git", "clone", "--no-hardlinks", "--no-checkout", str(source), str(workspace)],
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise AutomaticMaterializationError("AUTOMATIC_MATERIALIZATION_CLONE_FAILED")
    _git_run(workspace, ("checkout", "--detach", source_revision))
    tracked_patch = _git_output(source, ("diff", "--binary", "--full-index", "HEAD"))
    if tracked_patch:
        applied = subprocess.run(
            ["git", "-C", str(workspace), "apply", "--binary", "-"],
            input=tracked_patch,
            capture_output=True,
            check=False,
        )
        if applied.returncode != 0:
            raise AutomaticMaterializationError("AUTOMATIC_MATERIALIZATION_TRACKED_OVERLAY_FAILED")
    included, excluded = _copy_untracked_overlay(source=source, workspace=workspace, plan=plan)
    _git_run(workspace, ("add", "-A"))
    _git_run(
        workspace,
        (
            "-c",
            "user.name=VerdiWM Materializer",
            "-c",
            "user.email=materializer@invalid",
            "commit",
            "--allow-empty",
            "-m",
            "verdiwm isolated source snapshot",
        ),
    )
    snapshot_revision = _git_revision(workspace)
    return {
        "schema_version": 1,
        "artifact_type": "verdiwm-isolated-source-snapshot-receipt",
        "source_revision": source_revision,
        "source_status_sha256": _sha256(
            _git_output(source, ("status", "--porcelain=v1", "-z"))
        ),
        "tracked_diff_sha256": _sha256(tracked_patch),
        "included_untracked": included,
        "excluded_untracked": excluded,
        "snapshot_revision": snapshot_revision,
    }


def _copy_untracked_overlay(
    *, source: Path, workspace: Path, plan: Mapping[str, object]
) -> tuple[list[dict[str, object]], list[str]]:
    policy = plan["source_overlay"]
    assert isinstance(policy, Mapping)
    include = [str(value) for value in policy["include_untracked_globs"]]
    maximum_file = int(policy["max_file_bytes"])
    maximum_total = int(policy["max_total_bytes"])
    names = _parse_nul_paths(
        _git_output(source, ("ls-files", "--others", "--exclude-standard", "-z"))
    )
    included: list[dict[str, object]] = []
    excluded: list[str] = []
    total = 0
    for relative in names:
        if not any(fnmatch.fnmatchcase(relative, pattern) for pattern in include):
            excluded.append(relative)
            continue
        source_file = source / relative
        if source_file.is_symlink() or not source_file.is_file():
            raise AutomaticMaterializationError("AUTOMATIC_MATERIALIZATION_UNTRACKED_MEMBER_INVALID")
        size = source_file.stat().st_size
        total += size
        if size > maximum_file or total > maximum_total:
            raise AutomaticMaterializationError("AUTOMATIC_MATERIALIZATION_UNTRACKED_LIMIT_EXCEEDED")
        destination = workspace / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, destination, follow_symlinks=False)
        included.append({"path": relative, "size": size, "sha256": _sha256(source_file.read_bytes())})
    return included, excluded


def _candidate_catalog(
    *,
    plan: Mapping[str, object],
    idea: Mapping[str, object],
    descriptor: Mapping[str, object],
    workspace: Path,
    receipt_path: Path,
    receipt_sha256: str,
) -> dict[str, object]:
    required_files = []
    for relative in descriptor["implementation_files"]:
        path = workspace / str(relative)
        required_files.append(
            {
                "name": Path(str(relative)).stem,
                "path": str(path),
                "sha256": _sha256(path.read_bytes()),
            }
        )
    source_reference = _source_reference(idea.get("source"))
    return {
        "schema_version": 1,
        "artifact_type": "verdiwm-method-candidate-catalog",
        "catalog_id": f"materialized-{plan['candidate_id']}",
        "model_family": plan["model_family"],
        "candidates": [
            {
                "candidate_id": plan["candidate_id"],
                "primitive_reference": None,
                "source": source_reference,
                "mechanism_hypothesis": idea["hypothesis"],
                "required_hooks": list(plan["required_hooks"]),
                "failure_signatures": _candidate_failure_signatures(idea),
                "applicability_conditions": list(descriptor["applicability_conditions"]),
                "failure_boundaries": list(descriptor["failure_boundaries"]),
                "estimated_gpu_hours": plan["estimated_gpu_hours"],
                "historical_candidate_ids": [],
                "required_files": required_files,
                "materialization_receipt_path": str(receipt_path),
                "materialization_receipt_sha256": receipt_sha256,
                "candidate_template": dict(plan["candidate_template"]),
            }
        ],
        "capability_gaps": [],
        "claim_boundary": (
            "This catalog is bound to an isolated materialization receipt. It grants "
            "candidate compilation only; GPU and promotion authority remain external."
        ),
    }


def _source_reference(raw: object) -> str:
    if not isinstance(raw, Mapping):
        raise AutomaticMaterializationError(
            "AUTOMATIC_MATERIALIZATION_SOURCE_REFERENCE_INVALID"
        )
    source_url = raw.get("source_url")
    if isinstance(source_url, str) and source_url:
        return source_url
    source_id = raw.get("source_id")
    source_digest = raw.get("source_digest")
    assessment_digest = raw.get("assessment_digest")
    if not all(
        isinstance(value, str) and value
        for value in (source_id, source_digest, assessment_digest)
    ):
        raise AutomaticMaterializationError(
            "AUTOMATIC_MATERIALIZATION_SOURCE_REFERENCE_INVALID"
        )
    return (
        f"{source_id}@sha256:{source_digest}"
        f"#assessment-sha256:{assessment_digest}"
    )


def _load_descriptor(
    path: Path,
    *,
    workspace: Path,
    candidate_id: str,
    idea_id: str,
    changed_paths: Sequence[str],
    root: Path | None,
) -> dict[str, object]:
    descriptor_path = _require_file(path, "AUTOMATIC_MATERIALIZATION_DESCRIPTOR_MISSING")
    descriptor = _load(descriptor_path, "AUTOMATIC_MATERIALIZATION_DESCRIPTOR_INVALID")
    _validate_document("materialized_method_descriptor", descriptor, root=root)
    if descriptor.get("candidate_id") != candidate_id or descriptor.get("idea_id") != idea_id:
        raise AutomaticMaterializationError("AUTOMATIC_MATERIALIZATION_DESCRIPTOR_BINDING_MISMATCH")
    implementation_files = descriptor.get("implementation_files")
    assert isinstance(implementation_files, list)
    changed = set(changed_paths)
    if not set(str(value) for value in implementation_files).issubset(changed):
        raise AutomaticMaterializationError("AUTOMATIC_MATERIALIZATION_DESCRIPTOR_FILE_MISMATCH")
    for relative in implementation_files:
        _require_file(workspace / str(relative), "AUTOMATIC_MATERIALIZATION_IMPLEMENTATION_FILE_MISSING")
    return descriptor


def _path_blockers(
    changed_paths: Sequence[str], *, plan: Mapping[str, object]
) -> list[dict[str, object]]:
    allowed = [str(value) for value in plan["allowed_changed_paths"]]
    forbidden = [str(value) for value in plan["forbidden_changed_paths"]]
    blockers = []
    for path in changed_paths:
        if any(fnmatch.fnmatchcase(path, pattern) for pattern in forbidden):
            blockers.append({"code": "FORBIDDEN_PATH_CHANGED", "path": path})
        elif not any(fnmatch.fnmatchcase(path, pattern) for pattern in allowed):
            blockers.append({"code": "CHANGED_PATH_NOT_ALLOWED", "path": path})
    return blockers


def _commit_implementation(workspace: Path, *, candidate_id: str) -> str:
    _git_run(workspace, ("add", "-A"))
    _git_run(
        workspace,
        (
            "-c",
            "user.name=VerdiWM Materializer",
            "-c",
            "user.email=materializer@invalid",
            "commit",
            "-m",
            f"materialize {candidate_id}",
        ),
    )
    return _git_revision(workspace)


def _agent_environment(
    *, destination: Path, plan: Mapping[str, object]
) -> dict[str, str]:
    cache_root = destination / "check-cache"
    environment = {
        "CUDA_VISIBLE_DEVICES": "",
        "HYPOTHESIS_STORAGE_DIRECTORY": str(cache_root / "hypothesis"),
        "NVIDIA_VISIBLE_DEVICES": "none",
        "PYTHONUNBUFFERED": "1",
        "PYTHONPYCACHEPREFIX": str(cache_root / "python"),
        "XDG_CACHE_HOME": str(cache_root / "xdg"),
    }
    for key in plan["inherit_environment_keys"]:
        value = os.environ.get(str(key))
        if value is not None:
            environment[str(key)] = value
    environment["VERDIWM_MATERIALIZATION_ROOT"] = str(destination)
    return environment


def _candidate_failure_signatures(idea: Mapping[str, object]) -> list[str]:
    values = [
        str(value).strip()
        for value in idea.get("failure_context", [])
        if str(value).strip()
    ]
    if values:
        return sorted(set(values))
    track = str(idea.get("track") or "predictive_quality").strip().lower()
    safe_track = re.sub(r"[^a-z0-9]+", "_", track).strip("_")
    return [f"fresh_{safe_track or 'predictive_quality'}_baseline_gap"]


def _is_sensitive_environment_key(key: str) -> bool:
    upper = key.upper()
    return upper.startswith(_SENSITIVE_INHERITED_ENVIRONMENT_PREFIXES) or upper.endswith(
        _SENSITIVE_INHERITED_ENVIRONMENT_SUFFIXES
    )


def _placeholders(
    *,
    workspace: Path,
    destination: Path,
    prompt: str,
    prompt_path: Path,
    plan: Mapping[str, object],
    idea: Mapping[str, object],
) -> dict[str, str]:
    return {
        "{workspace}": str(workspace),
        "{output_root}": str(destination),
        "{prompt_text}": prompt,
        "{prompt_path}": str(prompt_path),
        "{descriptor_path}": str(workspace / str(plan["descriptor_path"])),
        "{candidate_id}": str(plan["candidate_id"]),
        "{idea_id}": str(idea["idea_id"]),
        "{python}": sys.executable,
    }


def _expand_argv(value: object, placeholders: Mapping[str, str]) -> list[str]:
    if not isinstance(value, list) or not value:
        raise AutomaticMaterializationError("AUTOMATIC_MATERIALIZATION_COMMAND_INVALID")
    command = []
    for raw in value:
        if not isinstance(raw, str) or not raw:
            raise AutomaticMaterializationError("AUTOMATIC_MATERIALIZATION_COMMAND_INVALID")
        expanded = raw
        for key, replacement in placeholders.items():
            expanded = expanded.replace(key, replacement)
        if re.search(r"\{[A-Za-z][A-Za-z0-9_]*\}", expanded):
            raise AutomaticMaterializationError(
                "AUTOMATIC_MATERIALIZATION_COMMAND_PLACEHOLDER_UNBOUND"
            )
        command.append(expanded)
    return command


def _render_prompt(
    *,
    plan: Mapping[str, object],
    work_order: Mapping[str, object],
    idea: Mapping[str, object],
    manufacturing_order: Mapping[str, object] | None,
) -> str:
    lines = [
            "Implement exactly one Ctrl-World research candidate in this isolated snapshot.",
            "Do not weaken, substitute, disable, or proxy the stated method intent.",
            "Do not run GPU workloads. Do not modify files outside the allowed paths.",
            "If faithful implementation is impossible, write the descriptor with a declared compromise and stop.",
            f"Candidate id: {plan['candidate_id']}",
            f"Descriptor path: {plan['descriptor_path']}",
            "Allowed changed paths: " + json.dumps(plan["allowed_changed_paths"], sort_keys=True),
            "Forbidden changed paths: " + json.dumps(plan["forbidden_changed_paths"], sort_keys=True),
            "Fixed checks are controller-owned and cannot be changed by this task.",
            "Implementation contract: "
            + json.dumps(plan.get("implementation_contract", {}), sort_keys=True),
            "Work order:",
            json.dumps(work_order, sort_keys=True, ensure_ascii=False),
            "Research idea:",
            json.dumps(idea, sort_keys=True, ensure_ascii=False),
        ]
    if manufacturing_order is not None:
        lines.extend(
            [
                "Module manufacturing work order:",
                json.dumps(manufacturing_order, sort_keys=True, ensure_ascii=False),
            ]
        )
    return "\n".join(lines)


def _validate_manufacturing_binding(
    *, plan: Mapping[str, object], order: Mapping[str, object]
) -> None:
    expected = {
        "manufacturing_mode": order["manufacturing_mode"],
        "manufacturing_work_order_id": order["work_order_id"],
        "manufacturing_work_order_digest": order["work_order_digest"],
        "manufacturing_request_id": order["manufacturing_request"]["request_id"],
        "manufacturing_abi_id": order["target_abi"]["abi_id"],
        "dependency_abi_ids": order["manufacturing_request"]["dependency_abi_ids"],
        "portfolio_entry_ids": order["portfolio_entry_ids"],
        "protected_metrics": order["protected_metrics"],
        "evaluator_binding": order["evaluator_binding"],
    }
    if any(plan.get(field) != value for field, value in expected.items()):
        raise AutomaticMaterializationError(
            "AUTOMATIC_MATERIALIZATION_MANUFACTURING_BINDING_MISMATCH"
        )
    if order.get("manufacturing_mode") != "intervention":
        raise AutomaticMaterializationError(
            "AUTOMATIC_MATERIALIZATION_MANUFACTURING_MODE_INVALID"
        )


def _validate_research_binding(
    *, plan: Mapping[str, object], work_order: Mapping[str, object], idea: Mapping[str, object]
) -> None:
    if work_order.get("artifact_type") != "verdiwm-acwm-research-materialization-work-order":
        raise AutomaticMaterializationError("AUTOMATIC_MATERIALIZATION_WORK_ORDER_INVALID")
    if idea.get("artifact_type") != "verdiwm-acwm-research-idea":
        raise AutomaticMaterializationError("AUTOMATIC_MATERIALIZATION_IDEA_INVALID")
    if work_order.get("idea_id") != idea.get("idea_id") or plan.get("idea_id") != idea.get("idea_id"):
        raise AutomaticMaterializationError("AUTOMATIC_MATERIALIZATION_IDEA_BINDING_MISMATCH")
    if work_order.get("idea_sha256") != _sha256(_canonical_json(idea)):
        raise AutomaticMaterializationError("AUTOMATIC_MATERIALIZATION_IDEA_HASH_MISMATCH")
    if work_order.get("execution_authority") != "none_until_materialized_and_compiled":
        raise AutomaticMaterializationError("AUTOMATIC_MATERIALIZATION_WORK_ORDER_AUTHORITY_INVALID")
    if int(work_order.get("schema_version", 1)) >= 2:
        if work_order.get("source_grounding") != idea.get("source"):
            raise AutomaticMaterializationError(
                "AUTOMATIC_MATERIALIZATION_SOURCE_GROUNDING_MISMATCH"
            )
        if work_order.get("materialization_contract") != idea.get(
            "materialization_contract"
        ):
            raise AutomaticMaterializationError(
                "AUTOMATIC_MATERIALIZATION_COMPONENT_CONTRACT_MISMATCH"
            )
    template = plan.get("candidate_template")
    if isinstance(template, Mapping) and template.get("candidate_id") != plan.get("candidate_id"):
        raise AutomaticMaterializationError("AUTOMATIC_MATERIALIZATION_TEMPLATE_CANDIDATE_MISMATCH")


def _source_component_mapping_blockers(
    *,
    descriptor: Mapping[str, object],
    idea: Mapping[str, object],
    work_order: Mapping[str, object],
) -> list[dict[str, object]]:
    """Require a one-to-one source component mapping for v2 work orders."""

    if int(work_order.get("schema_version", 1)) < 2:
        return []
    contract = idea.get("materialization_contract")
    if not isinstance(contract, Mapping):
        return [{"code": "SOURCE_COMPONENT_CONTRACT_MISSING"}]
    raw_required = contract.get("preserved_components")
    if not isinstance(raw_required, list) or not raw_required:
        return [{"code": "SOURCE_COMPONENT_CONTRACT_MISSING"}]
    required = [str(value) for value in raw_required]
    rows = descriptor.get("intent_to_code")
    if not isinstance(rows, list):
        return [{"code": "SOURCE_COMPONENT_MAPPING_INVALID"}]
    mapped = []
    invalid_rows = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            invalid_rows.append(index)
            continue
        component_id = row.get("source_component_id")
        touchpoint = row.get("touchpoint")
        if not isinstance(component_id, str) or not component_id:
            invalid_rows.append(index)
            continue
        if not isinstance(touchpoint, str) or not touchpoint:
            invalid_rows.append(index)
            continue
        mapped.append(component_id)
    blockers: list[dict[str, object]] = []
    if invalid_rows:
        blockers.append(
            {"code": "SOURCE_COMPONENT_MAPPING_INVALID", "row_indices": invalid_rows}
        )
    duplicates = sorted({value for value in mapped if mapped.count(value) > 1})
    missing = sorted(set(required) - set(mapped))
    unexpected = sorted(set(mapped) - set(required))
    if duplicates or missing or unexpected:
        blockers.append(
            {
                "code": "SOURCE_COMPONENT_MAPPING_INCOMPLETE",
                "required": required,
                "mapped": mapped,
                "missing": missing,
                "unexpected": unexpected,
                "duplicates": duplicates,
            }
        )
    return blockers


def _input_lock(
    *,
    plan_path: Path,
    plan: Mapping[str, object],
    work_order_path: Path,
    idea_path: Path,
    source: Path,
    manufacturing_work_order_path: Path | None,
) -> dict[str, object]:
    payload = {
        "schema_version": 1,
        "artifact_type": "verdiwm-automatic-materialization-input-lock",
        "plan_id": plan["plan_id"],
        "plan_digest": plan["plan_digest"],
        "plan_path": str(plan_path),
        "plan_file_sha256": _sha256(plan_path.read_bytes()),
        "work_order_path": str(work_order_path),
        "work_order_sha256": _sha256(work_order_path.read_bytes()),
        "idea_path": str(idea_path),
        "idea_sha256": _sha256(idea_path.read_bytes()),
        "source_root": str(source),
        "source_revision": _git_revision(source),
        "source_status_sha256": _sha256(
            _git_output(source, ("status", "--porcelain=v1", "-z"))
        ),
    }
    if manufacturing_work_order_path is not None:
        payload.update(
            {
                "manufacturing_work_order_path": str(manufacturing_work_order_path),
                "manufacturing_work_order_sha256": _sha256(
                    manufacturing_work_order_path.read_bytes()
                ),
            }
        )
    payload["input_sha256"] = _sha256(_canonical_json(payload))
    return payload


def _resume_terminal(
    destination: Path, *, input_lock: Mapping[str, object]
) -> dict[str, object] | None:
    if not destination.exists() and not destination.is_symlink():
        return None
    if destination.is_symlink() or not destination.is_dir():
        raise AutomaticMaterializationError("AUTOMATIC_MATERIALIZATION_OUTPUT_INVALID")
    existing = _load(
        destination / "input-lock.json", "AUTOMATIC_MATERIALIZATION_INPUT_LOCK_INVALID"
    )
    if existing.get("input_sha256") != input_lock.get("input_sha256"):
        raise AutomaticMaterializationError("AUTOMATIC_MATERIALIZATION_INPUT_LOCK_MISMATCH")
    manifest_path = destination / "manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise AutomaticMaterializationError("AUTOMATIC_MATERIALIZATION_INCOMPLETE_ATTEMPT")
    return _load(manifest_path, "AUTOMATIC_MATERIALIZATION_MANIFEST_INVALID")


def _archive_transaction_inputs(
    *,
    cas: ContentAddressedStore,
    plan_path: Path,
    work_order_path: Path,
    idea_path: Path,
    manufacturing_work_order_path: Path | None,
    snapshot_path: Path,
    prompt_path: Path,
    staged: object,
    descriptor_path: Path | None,
) -> list[str]:
    paths = [plan_path, work_order_path, idea_path, snapshot_path, prompt_path]
    if manufacturing_work_order_path is not None:
        paths.append(manufacturing_work_order_path)
    if staged is not None:
        paths.extend([staged.manifest_path, staged.diff_path])
    if descriptor_path is not None:
        paths.append(descriptor_path)
    refs = []
    for path in paths:
        media_type = "application/json" if path.suffix == ".json" else "text/plain"
        try:
            refs.append(cas.put_bytes(path.read_bytes(), media_type=media_type).uri)
        except (OSError, ArchiveInvariantError) as exc:
            raise AutomaticMaterializationError("AUTOMATIC_MATERIALIZATION_ARCHIVE_FAILED") from exc
    return refs


def _validate_document(
    schema_name: str, payload: Mapping[str, object], *, root: Path | None
) -> None:
    try:
        validate_document(schema_name, payload, root=root)
    except ContractValidationError as exc:
        raise AutomaticMaterializationError(
            f"AUTOMATIC_MATERIALIZATION_SCHEMA_INVALID:{schema_name}:{exc}"
        ) from exc


def _parse_nul_paths(payload: bytes) -> list[str]:
    result = []
    for raw in filter(None, payload.split(b"\0")):
        try:
            relative = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise AutomaticMaterializationError("AUTOMATIC_MATERIALIZATION_PATH_INVALID") from exc
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or ".git" in path.parts:
            raise AutomaticMaterializationError("AUTOMATIC_MATERIALIZATION_PATH_INVALID")
        result.append(relative)
    return result


def _git_revision(root: Path) -> str:
    revision = _git_output(root, ("rev-parse", "HEAD")).decode("ascii").strip()
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise AutomaticMaterializationError("AUTOMATIC_MATERIALIZATION_SOURCE_NOT_GIT")
    return revision


def _git_output(root: Path, arguments: Sequence[str]) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments], capture_output=True, check=False
    )
    if completed.returncode != 0:
        raise AutomaticMaterializationError("AUTOMATIC_MATERIALIZATION_GIT_COMMAND_FAILED")
    return completed.stdout


def _git_run(root: Path, arguments: Sequence[str]) -> None:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments], capture_output=True, check=False
    )
    if completed.returncode != 0:
        raise AutomaticMaterializationError("AUTOMATIC_MATERIALIZATION_GIT_COMMAND_FAILED")


def _require_file(path: Path, code: str) -> Path:
    raw = Path(path).expanduser()
    if raw.is_symlink():
        raise AutomaticMaterializationError(code)
    source = raw.resolve()
    if not source.is_file():
        raise AutomaticMaterializationError(code)
    return source


def _require_directory(path: Path, code: str) -> Path:
    raw = Path(path).expanduser()
    if raw.is_symlink():
        raise AutomaticMaterializationError(code)
    source = raw.resolve()
    if not source.is_dir():
        raise AutomaticMaterializationError(code)
    return source


def _load(path: Path, code: str) -> dict[str, object]:
    source = _require_file(path, code)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AutomaticMaterializationError(code) from exc
    if not isinstance(payload, dict):
        raise AutomaticMaterializationError(code)
    return payload


def _canonical_json(payload: object) -> bytes:
    try:
        return (
            json.dumps(
                payload,
                sort_keys=True,
                ensure_ascii=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise AutomaticMaterializationError("AUTOMATIC_MATERIALIZATION_PAYLOAD_INVALID") from exc


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    _write_bytes(path, _canonical_json(payload))


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise AutomaticMaterializationError("AUTOMATIC_MATERIALIZATION_IMMUTABLE_WRITE_CONFLICT")
        return
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--work-order", type=Path, required=True)
    parser.add_argument("--idea", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = run_automatic_materialization(
            plan_path=args.plan,
            work_order_path=args.work_order,
            idea_path=args.idea,
            source_root=args.source_root,
            output_root=args.output_root,
        )
    except AutomaticMaterializationError as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
