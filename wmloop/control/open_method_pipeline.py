"""Compile unrestricted LLM method proposals into isolated overlay receipts.

The LLM chooses the mechanism and candidate file layout.  The kernel derives
identities, rejects path escapes, keeps generated files outside the source
tree, and grants only calibration eligibility.  Evaluation and promotion stay
outside this pipeline.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath

from wmloop.contracts import ContractValidationError, validate_document
from wmloop.control.interface_extension import build_method_interface_extension
from wmloop.control.candidate_execution import (
    build_candidate_execution_contract,
    candidate_execution_digest,
)
from wmloop.control.open_method_compiler import compile_candidate_overlay
from wmloop.control.open_method_ir import build_method_ir, validate_method_ir


class OpenMethodPipelineError(RuntimeError):
    """An open proposal failed normalization or isolation."""


OPEN_METHOD_PROMPT = """Propose a source-grounded method for the bound target portrait.
Do not select from a fixed ABI or paper-family list. Describe the mechanism as Method IR,
then provide only candidate-local implementation, training, data-transform, configuration,
and test files needed to falsify it. If a semantic surface is missing, return an interface
extension proposal instead of weakening the method. Never modify the source tree, evaluator,
active metrics, data split, GPU policy, verifier, or promotion policy.
"""


def build_open_method_request(
    *,
    source_evidence: Sequence[Mapping[str, object]],
    target_portrait: Mapping[str, object],
    probe_fingerprints: Sequence[Mapping[str, object]],
    failure_context: Sequence[str],
) -> dict[str, object]:
    """Build a provider-neutral LLM request with no predefined method ABI."""

    seed = {
        "source_evidence": [dict(row) for row in source_evidence],
        "portrait_id": target_portrait.get("portrait_id"),
        "probe_ids": sorted(str(row.get("fingerprint_id")) for row in probe_fingerprints),
        "failure_context": sorted(set(str(value) for value in failure_context)),
    }
    task_id = "open-method-" + _digest(seed)[:24]
    return {
        "schema_version": 1,
        "artifact_type": "verdiwm-llm-research-task",
        "task_id": task_id,
        "task_type": "open_method_generation",
        "prompt_template_digest": hashlib.sha256(OPEN_METHOD_PROMPT.encode()).hexdigest(),
        "output_schema": "open_method_proposal",
        "input": {
            "instructions": OPEN_METHOD_PROMPT,
            "source_evidence": [dict(row) for row in source_evidence],
            "target_portrait": dict(target_portrait),
            "probe_fingerprints": [dict(row) for row in probe_fingerprints],
            "failure_context": sorted(set(str(value) for value in failure_context)),
        },
    }


def compile_open_method_proposal(
    *,
    proposal: Mapping[str, object],
    base_revision: Mapping[str, object],
    output_root: Path,
    project_root: Path,
    expected_portrait_binding: Mapping[str, object] | None = None,
    expected_probe_binding: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Normalize one LLM proposal and emit an immutable local compilation record."""

    root = Path(project_root).expanduser().resolve()
    destination = Path(output_root).expanduser().resolve()
    if destination == root or root in destination.parents:
        raise OpenMethodPipelineError("OPEN_METHOD_OUTPUT_INSIDE_SOURCE")
    if destination.exists() or destination.is_symlink():
        raise OpenMethodPipelineError("OPEN_METHOD_OUTPUT_EXISTS")
    try:
        validate_document("open_method_proposal", proposal, root=root)
    except ContractValidationError as exc:
        raise OpenMethodPipelineError(f"OPEN_METHOD_PROPOSAL_INVALID:{exc}") from exc

    method = _normalize_method_ir(_mapping(proposal, "method_ir"), root=root)
    if expected_portrait_binding is not None and method.get("target_portrait_binding") != dict(
        expected_portrait_binding
    ):
        raise OpenMethodPipelineError("OPEN_METHOD_PORTRAIT_BINDING_MISMATCH")
    if expected_probe_binding is not None and method.get("probe_binding") != dict(
        expected_probe_binding
    ):
        raise OpenMethodPipelineError("OPEN_METHOD_PROBE_BINDING_MISMATCH")
    extensions = [
        _normalize_extension(row, method=method, root=root)
        for row in _mapping_rows(proposal.get("interface_extensions"))
    ]
    state = str(proposal["state"])
    files = _mapping_rows(proposal.get("files"))
    tests = _mapping_rows(proposal.get("tests"))
    blockers = _mapping_rows(proposal.get("blockers"))
    execution = None
    raw_execution = proposal.get("execution_contract")
    if raw_execution is not None:
        if not isinstance(raw_execution, Mapping):
            raise OpenMethodPipelineError("OPEN_METHOD_EXECUTION_CONTRACT_INVALID")
        execution = build_candidate_execution_contract(
            method_ir=method,
            entrypoints=_mapping(raw_execution, "entrypoints"),
            inputs=_mapping_rows(raw_execution.get("inputs")),
            outputs=_mapping_rows(raw_execution.get("outputs")),
            root=root,
        )
    if state == "candidate_ready":
        if not files or not tests or blockers or extensions or execution is None:
            raise OpenMethodPipelineError("OPEN_METHOD_READY_STATE_INCONSISTENT")
    elif files or execution is not None:
        raise OpenMethodPipelineError("OPEN_METHOD_BLOCKED_FILES_FORBIDDEN")
    if state == "interface_extension_required" and not extensions:
        raise OpenMethodPipelineError("OPEN_METHOD_INTERFACE_EXTENSION_REQUIRED")

    temporary = destination.parent / f".{destination.name}.tmp-{os.getpid()}"
    if temporary.exists() or temporary.is_symlink():
        raise OpenMethodPipelineError("OPEN_METHOD_TEMPORARY_EXISTS")
    try:
        temporary.mkdir(mode=0o700, parents=True)
        _write_json(temporary / "method-ir.json", method)
        if execution is not None:
            _write_json(temporary / "candidate-execution.json", execution)
        for extension in extensions:
            _write_json(
                temporary / "interface-extensions" / f"{extension['extension_id']}.json",
                extension,
            )
        overlay = None
        if state == "candidate_ready":
            workspace = temporary / "candidate-workspace"
            workspace.mkdir(mode=0o700)
            roles = _write_candidate_files(workspace, files)
            overlay = compile_candidate_overlay(
                method_ir=method,
                base_revision=base_revision,
                workspace=workspace,
                output_path=temporary / "candidate-overlay.json",
                tests=tests,
                project_root=root,
                file_roles=roles,
                execution_contract_binding={
                    "execution_id": execution["execution_id"],
                    "contract_digest": candidate_execution_digest(execution),
                },
            )
        manifest = {
            "schema_version": 1,
            "artifact_type": "verdiwm-open-method-compilation",
            "state": "ready_for_calibration" if overlay else "blocked",
            "proposal_state": state,
            "method_id": method["method_id"],
            "method_ir_digest": _method_digest(method),
            "overlay_id": overlay.get("overlay_id") if overlay else None,
            "execution_id": execution.get("execution_id") if execution else None,
            "interface_extension_ids": [row["extension_id"] for row in extensions],
            "blockers": [dict(row) for row in blockers],
            "authority": {
                "calibration_eligible": overlay is not None,
                "gpu_scheduling": False,
                "evaluator_mutation": False,
                "promotion": False,
            },
            "claim_boundary": (
                "Compilation creates an isolated candidate only. Calibration, GPU leases, "
                "evaluation, verification, and promotion require separate kernel receipts."
            ),
        }
        _write_json(temporary / "manifest.json", manifest)
        os.replace(temporary, destination)
        return manifest
    except Exception:
        if temporary.exists() and not temporary.is_symlink():
            shutil.rmtree(temporary)
        raise


def _normalize_method_ir(raw: Mapping[str, object], *, root: Path) -> dict[str, object]:
    required = ("source_evidence", "mechanism", "target_mapping", "training", "falsification")
    if any(not isinstance(raw.get(key), (list, Mapping)) for key in required):
        raise OpenMethodPipelineError("OPEN_METHOD_IR_FIELDS_INVALID")
    method = build_method_ir(
        source_evidence=_mapping_rows(raw["source_evidence"]),
        mechanism=_mapping(raw, "mechanism"),
        target_mapping=_mapping(raw, "target_mapping"),
        training=_mapping(raw, "training"),
        falsification=_mapping(raw, "falsification"),
        source_evidence_digest=(str(raw["source_evidence_digest"]) if raw.get("source_evidence_digest") else None),
        target_portrait_binding=(
            _mapping(raw, "target_portrait_binding") if raw.get("target_portrait_binding") else None
        ),
        probe_binding=_mapping(raw, "probe_binding") if raw.get("probe_binding") else None,
        interface_extension_refs=(
            [str(value) for value in raw.get("interface_extension_refs", [])]
            if isinstance(raw.get("interface_extension_refs", []), list)
            else []
        ),
        training_resource_binding=(
            _mapping(raw, "training_resource_binding")
            if raw.get("training_resource_binding")
            else None
        ),
        state=str(raw.get("state") or "draft"),
        claim_boundary=str(
            raw.get("claim_boundary")
            or "This Method IR is a research proposal, not an execution or promotion decision."
        ),
    )
    validate_method_ir(method, root=root)
    return method


def _normalize_extension(
    raw: Mapping[str, object], *, method: Mapping[str, object], root: Path
) -> dict[str, object]:
    return build_method_interface_extension(
        method_ir=method,
        requested_surface=str(raw.get("requested_surface") or ""),
        semantic_role=str(raw.get("semantic_role") or ""),
        typed_inputs=_mapping_rows(raw.get("typed_inputs")),
        typed_outputs=_mapping_rows(raw.get("typed_outputs")),
        side_effect_class=str(raw.get("side_effect_class") or ""),
        conformance_tests=_string_rows(raw.get("conformance_tests")),
        negative_tests=_string_rows(raw.get("negative_tests")),
        root=root,
    )


def _write_candidate_files(
    workspace: Path, rows: Sequence[Mapping[str, object]]
) -> dict[str, str]:
    roles: dict[str, str] = {}
    for row in rows:
        relative = str(row.get("relative_path") or "")
        pure = PurePosixPath(relative)
        if not relative or pure.is_absolute() or ".." in pure.parts or str(pure) != relative:
            raise OpenMethodPipelineError("OPEN_METHOD_FILE_PATH_INVALID")
        if relative in roles:
            raise OpenMethodPipelineError("OPEN_METHOD_FILE_DUPLICATE")
        target = workspace.joinpath(*pure.parts)
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        target.write_text(str(row.get("content_utf8") or ""), encoding="utf-8")
        roles[relative] = str(row.get("role") or "")
    return roles


def _mapping(value: Mapping[str, object], key: str) -> Mapping[str, object]:
    row = value.get(key)
    if not isinstance(row, Mapping):
        raise OpenMethodPipelineError("OPEN_METHOD_MAPPING_INVALID:" + key)
    return row


def _mapping_rows(value: object) -> list[Mapping[str, object]]:
    if not isinstance(value, list) or any(not isinstance(row, Mapping) for row in value):
        raise OpenMethodPipelineError("OPEN_METHOD_ROWS_INVALID")
    return [row for row in value if isinstance(row, Mapping)]


def _string_rows(value: object) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(row, str) or not row for row in value):
        raise OpenMethodPipelineError("OPEN_METHOD_STRINGS_INVALID")
    return [str(row) for row in value]


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, ensure_ascii=True, indent=2) + "\n")


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode()
    ).hexdigest()


def _method_digest(method: Mapping[str, object]) -> str:
    return _digest({key: value for key, value in method.items() if key != "method_id"})
