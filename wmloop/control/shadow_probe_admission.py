"""Compile generated observation probes into shadow-only admission receipts."""

from __future__ import annotations

import ast
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from wmloop.contracts import ContractValidationError, validate_document
from wmloop.control.adaptive_observation import (
    AdaptiveObservationError,
    load_observation_abi_registry,
    validate_adaptive_probe_plan,
)
from wmloop.control.module_manufacturing import (
    ModuleManufacturingError,
    load_module_manufacturing_work_order,
)
from wmloop.execute.llm_task_adapter import LLMTaskAdapterError, run_llm_task


class ShadowProbeAdmissionError(RuntimeError):
    """A generated observation probe crossed its shadow-only boundary."""


_PROMPT = """Implement one diagnostic probe draft for the supplied shadow-only ABI.
The module must define exactly one top-level function named run_shadow_probe(request).
It may transform only the request value and return a JSON-shaped shadow diagnostic.
Do not import modules, access files, networks, processes, environment variables, GPUs,
evaluators, protected metrics, verdicts, promotion policy, or the target source tree.
The result is a shadow candidate only and cannot satisfy active portrait coverage.
"""
_PROMPT_DIGEST = hashlib.sha256(_PROMPT.encode("ascii")).hexdigest()
_ALLOWED_CALLS = {
    "abs",
    "all",
    "any",
    "bool",
    "enumerate",
    "float",
    "int",
    "len",
    "list",
    "max",
    "min",
    "range",
    "round",
    "sorted",
    "str",
    "sum",
    "tuple",
}
_ALLOWED_METHODS = {
    "append",
    "copy",
    "count",
    "get",
    "items",
    "keys",
    "setdefault",
    "values",
}
_FORBIDDEN_NAMES = {
    "__builtins__",
    "breakpoint",
    "compile",
    "eval",
    "evaluator",
    "exec",
    "globals",
    "input",
    "locals",
    "metric",
    "open",
    "os",
    "pathlib",
    "promote",
    "promotion",
    "socket",
    "subprocess",
    "sys",
    "verdict",
}


def compile_shadow_probe_admission(
    *,
    manufacturing_work_order_path: Path,
    adaptive_probe_plan_path: Path,
    observation_registry_path: Path,
    probe_requirement: Mapping[str, object],
    adapter: Mapping[str, object],
    output_root: Path,
    project_root: Path,
) -> dict[str, object]:
    """Generate and statically admit one probe for shadow execution only."""

    root = Path(project_root).resolve()
    work_path = _require_file(
        manufacturing_work_order_path,
        "SHADOW_PROBE_MANUFACTURING_WORK_ORDER_INVALID",
    )
    plan_path = _require_file(
        adaptive_probe_plan_path, "SHADOW_PROBE_PLAN_INVALID"
    )
    registry_path = _require_file(
        observation_registry_path, "SHADOW_PROBE_REGISTRY_INVALID"
    )
    try:
        work_order = load_module_manufacturing_work_order(work_path, root=root)
        plan = _load(plan_path, "SHADOW_PROBE_PLAN_INVALID")
        validate_adaptive_probe_plan(plan, root=root)
        registry = load_observation_abi_registry(registry_path, root=root)
    except (ModuleManufacturingError, AdaptiveObservationError) as exc:
        raise ShadowProbeAdmissionError(
            f"SHADOW_PROBE_INPUT_INVALID:{exc}"
        ) from exc
    task, abi = _validate_bindings(
        work_order=work_order,
        plan=plan,
        registry=registry,
        probe_requirement=probe_requirement,
    )
    destination = Path(output_root).expanduser().resolve()
    input_lock = {
        "manufacturing_work_order_sha256": _sha256(work_path.read_bytes()),
        "adaptive_probe_plan_sha256": _sha256(plan_path.read_bytes()),
        "observation_registry_sha256": _sha256(registry_path.read_bytes()),
        "probe_requirement_sha256": _sha256(_canonical_bytes(probe_requirement)),
        "adapter": _adapter_identity(adapter),
    }
    input_digest = _sha256(_canonical_bytes(input_lock))
    resumed = _resume(destination, input_digest=input_digest)
    if resumed is not None:
        return resumed
    if destination.exists() or destination.is_symlink():
        raise ShadowProbeAdmissionError("SHADOW_PROBE_OUTPUT_INVALID")
    destination.mkdir(mode=0o700, parents=True)
    _write_json(
        destination / "input-lock.json",
        {**input_lock, "input_digest": input_digest},
    )
    request = _request(
        work_order=work_order,
        plan=plan,
        task=task,
        abi=abi,
        probe_requirement=probe_requirement,
    )
    try:
        llm_manifest = run_llm_task(
            request=request,
            adapter=adapter,
            output_root=destination / "llm-task",
            project_root=root,
        )
    except LLMTaskAdapterError as exc:
        raise ShadowProbeAdmissionError(
            f"SHADOW_PROBE_LLM_ADAPTER_INVALID:{exc}"
        ) from exc
    blockers: list[dict[str, object]] = []
    candidate: dict[str, object] | None = None
    candidate_path: Path | None = None
    candidate_sha256: str | None = None
    module_source_sha256: str | None = None
    conformance_checks: list[dict[str, object]] = []
    negative_checks: list[dict[str, object]] = []
    if llm_manifest.get("state") != "completed":
        blockers.append({"code": "SHADOW_PROBE_LLM_TASK_BLOCKED"})
    else:
        response = _load(
            _require_file(
                Path(str(llm_manifest.get("response_path") or "")),
                "SHADOW_PROBE_RESPONSE_INVALID",
            ),
            "SHADOW_PROBE_RESPONSE_INVALID",
        )
        raw_candidate = response.get("output")
        if not isinstance(raw_candidate, Mapping):
            raise ShadowProbeAdmissionError("SHADOW_PROBE_CANDIDATE_INVALID")
        candidate = dict(raw_candidate)
        _validate_candidate_binding(
            candidate,
            task=task,
            abi=abi,
            probe_requirement=probe_requirement,
        )
        conformance_checks, negative_checks, blockers = _static_checks(
            str(candidate["module_source"])
        )
        candidate_path = destination / "candidate.json"
        _write_json(candidate_path, candidate)
        candidate_sha256 = _sha256(candidate_path.read_bytes())
        module_source_sha256 = _sha256(
            str(candidate["module_source"]).encode("utf-8")
        )
    candidate_id = (
        "shadow-probe-"
        + _sha256(
            _canonical_bytes(
                {
                    "work_order_id": work_order["work_order_id"],
                    "task_id": task["task_id"],
                    "candidate_sha256": candidate_sha256,
                }
            )
        )[:24]
        if candidate_sha256 is not None
        else None
    )
    state = "admitted_for_shadow_execution" if not blockers else "blocked"
    llm_receipt_sha256 = str(llm_manifest["receipt_sha256"])
    evidence_refs = {"sha256:" + llm_receipt_sha256}
    if candidate_sha256 is not None:
        evidence_refs.add("sha256:" + candidate_sha256)
    if module_source_sha256 is not None:
        evidence_refs.add("sha256:" + module_source_sha256)
    receipt = {
        "schema_version": 1,
        "artifact_type": "verdiwm-shadow-probe-admission-receipt",
        "state": state,
        "candidate_id": candidate_id,
        "work_order_id": work_order["work_order_id"],
        "work_order_digest": work_order["work_order_digest"],
        "plan_id": plan["plan_id"],
        "plan_digest": _digest(plan),
        "observation_task_id": task["task_id"],
        "abi_id": abi["abi_id"],
        "probe_coverage_key": probe_requirement["coverage_key"],
        "llm_task_receipt_sha256": llm_receipt_sha256,
        "candidate_sha256": candidate_sha256,
        "module_source_sha256": module_source_sha256,
        "conformance_checks": conformance_checks,
        "negative_checks": negative_checks,
        "blockers": blockers,
        "evidence_refs": sorted(evidence_refs),
        "authority": {
            "shadow_execution": state == "admitted_for_shadow_execution",
            "active_portrait_coverage": False,
            "source_mutation": False,
            "gpu_scheduling": False,
            "metric_selection": False,
            "evaluator_selection": False,
            "verdict_exposure": False,
            "promotion": False,
        },
        "claim_boundary": (
            "This receipt may admit one generated probe to isolated shadow execution. "
            "It cannot satisfy active portrait coverage or select metrics, evaluators, "
            "verdicts, GPU resources, or promotion policy."
        ),
    }
    _validate("shadow_probe_admission_receipt", receipt, root=root)
    if receipt["authority"]["shadow_execution"] != (
        state == "admitted_for_shadow_execution"
    ):
        raise ShadowProbeAdmissionError("SHADOW_PROBE_AUTHORITY_INVALID")
    receipt_path = destination / "receipt.json"
    _write_json(receipt_path, receipt)
    manifest = {
        "schema_version": 1,
        "artifact_type": "verdiwm-shadow-probe-admission-manifest",
        "state": state,
        "candidate_id": candidate_id,
        "candidate_path": str(candidate_path) if candidate_path is not None else None,
        "candidate_sha256": candidate_sha256,
        "receipt_path": str(receipt_path),
        "receipt_sha256": _sha256(receipt_path.read_bytes()),
        "llm_task_receipt_path": llm_manifest["receipt_path"],
        "llm_task_receipt_sha256": llm_receipt_sha256,
        "blockers": blockers,
    }
    _write_json(destination / "manifest.json", manifest)
    return manifest


def _validate_bindings(
    *,
    work_order: Mapping[str, object],
    plan: Mapping[str, object],
    registry: Mapping[str, object],
    probe_requirement: Mapping[str, object],
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    if work_order.get("manufacturing_mode") != "observation":
        raise ShadowProbeAdmissionError("SHADOW_PROBE_WORK_ORDER_MODE_INVALID")
    bindings = _mapping(work_order, "bindings")
    if (
        bindings.get("observation_plan_id") != plan.get("plan_id")
        or bindings.get("observation_plan_digest") != _digest(plan)
    ):
        raise ShadowProbeAdmissionError("SHADOW_PROBE_PLAN_BINDING_MISMATCH")
    task_id = bindings.get("observation_task_id")
    tasks = [
        row
        for row in _mapping_rows(plan.get("tasks"))
        if row.get("task_id") == task_id
    ]
    if len(tasks) != 1:
        raise ShadowProbeAdmissionError("SHADOW_PROBE_TASK_BINDING_INVALID")
    task = tasks[0]
    if (
        task.get("task_type") != "manufacture_shadow_probe"
        or task.get("execution_authority") != "shadow_only"
        or task.get("probe_coverage_key") != probe_requirement.get("coverage_key")
    ):
        raise ShadowProbeAdmissionError("SHADOW_PROBE_TASK_AUTHORITY_INVALID")
    rows = [
        row
        for row in _mapping_rows(registry.get("abis"))
        if row.get("abi_id") == task.get("abi_id")
    ]
    if len(rows) != 1:
        raise ShadowProbeAdmissionError("SHADOW_PROBE_ABI_INVALID")
    abi = rows[0]
    if (
        abi.get("admission_state") != "shadow_template"
        or abi.get("execution_mode") != "shadow_via_controller"
        or work_order.get("target_abi", {}).get("abi_id") != abi.get("abi_id")
    ):
        raise ShadowProbeAdmissionError("SHADOW_PROBE_ABI_AUTHORITY_INVALID")
    return task, abi


def _request(
    *,
    work_order: Mapping[str, object],
    plan: Mapping[str, object],
    task: Mapping[str, object],
    abi: Mapping[str, object],
    probe_requirement: Mapping[str, object],
) -> dict[str, object]:
    task_id = "shadow-probe-" + _sha256(
        _canonical_bytes(
            {
                "work_order_id": work_order["work_order_id"],
                "plan_id": plan["plan_id"],
                "observation_task_id": task["task_id"],
            }
        )
    )[:24]
    return {
        "schema_version": 1,
        "artifact_type": "verdiwm-llm-research-task",
        "task_id": task_id,
        "task_type": "module_generation",
        "prompt_template_digest": _PROMPT_DIGEST,
        "output_schema": "shadow_probe_candidate",
        "input": {
            "instructions": _PROMPT,
            "module_manufacturing_work_order": dict(work_order),
            "adaptive_probe_plan": dict(plan),
            "observation_task": dict(task),
            "probe_requirement": dict(probe_requirement),
            "shadow_abi": dict(abi),
        },
    }


def _validate_candidate_binding(
    candidate: Mapping[str, object],
    *,
    task: Mapping[str, object],
    abi: Mapping[str, object],
    probe_requirement: Mapping[str, object],
) -> None:
    try:
        validate_document("shadow_probe_candidate", candidate)
    except ContractValidationError as exc:
        raise ShadowProbeAdmissionError(
            f"SHADOW_PROBE_CANDIDATE_INVALID:{exc}"
        ) from exc
    if (
        candidate.get("abi_id") != abi.get("abi_id")
        or candidate.get("observation_task_id") != task.get("task_id")
        or candidate.get("probe_protocol_id")
        != probe_requirement.get("probe_protocol_id")
        or candidate.get("probe_protocol_version")
        != probe_requirement.get("probe_protocol_version")
    ):
        raise ShadowProbeAdmissionError("SHADOW_PROBE_CANDIDATE_BINDING_MISMATCH")


def _static_checks(
    source: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    conformance = [
        {"check_id": "syntax_valid", "passed": False},
        {"check_id": "single_entrypoint_signature", "passed": False},
        {"check_id": "bounded_source_size", "passed": len(source.encode("utf-8")) <= 12000},
    ]
    negative = [
        {"check_id": "no_imports", "passed": False},
        {"check_id": "no_io_network_or_process", "passed": False},
        {"check_id": "no_evaluator_metric_verdict_or_promotion", "passed": False},
        {"check_id": "bounded_calls_only", "passed": False},
    ]
    blockers: list[dict[str, object]] = []
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as exc:
        blockers.append({"code": "SHADOW_PROBE_SYNTAX_INVALID", "detail": str(exc)})
        return conformance, negative, blockers
    conformance[0]["passed"] = True
    top_level = [node for node in tree.body if not isinstance(node, ast.Expr)]
    functions = [node for node in top_level if isinstance(node, ast.FunctionDef)]
    signature_ok = (
        len(top_level) == 1
        and len(functions) == 1
        and functions[0].name == "run_shadow_probe"
        and not functions[0].decorator_list
        and len(functions[0].args.args) == 1
        and functions[0].args.args[0].arg == "request"
        and functions[0].args.vararg is None
        and functions[0].args.kwarg is None
        and not functions[0].args.kwonlyargs
    )
    conformance[1]["passed"] = signature_ok
    imports_ok = not any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(tree))
    negative[0]["passed"] = imports_ok
    names = {
        str(node.id).lower()
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
    }
    attributes = {
        str(node.attr).lower()
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
    }
    authority_ok = not bool((names | attributes) & _FORBIDDEN_NAMES)
    negative[1]["passed"] = authority_ok
    negative[2]["passed"] = authority_ok
    calls_ok = True
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            calls_ok = calls_ok and node.func.id in _ALLOWED_CALLS
        elif isinstance(node.func, ast.Attribute):
            calls_ok = calls_ok and node.func.attr in _ALLOWED_METHODS
        else:
            calls_ok = False
    negative[3]["passed"] = calls_ok
    if not signature_ok:
        blockers.append({"code": "SHADOW_PROBE_ENTRYPOINT_INVALID"})
    if not conformance[2]["passed"]:
        blockers.append({"code": "SHADOW_PROBE_SOURCE_TOO_LARGE"})
    if not imports_ok:
        blockers.append({"code": "SHADOW_PROBE_IMPORT_FORBIDDEN"})
    if not authority_ok:
        blockers.append({"code": "SHADOW_PROBE_AUTHORITY_SYMBOL_FORBIDDEN"})
    if not calls_ok:
        blockers.append({"code": "SHADOW_PROBE_CALL_FORBIDDEN"})
    return conformance, negative, blockers


def _adapter_identity(adapter: Mapping[str, object]) -> dict[str, object]:
    return {
        "command": list(adapter.get("command", [])),
        "provider_alias": adapter.get("provider_alias"),
        "model_alias": adapter.get("model_alias"),
        "timeout_seconds": adapter.get("timeout_seconds"),
        "max_output_bytes": adapter.get("max_output_bytes"),
        "credential_environment_keys": list(
            adapter.get("credential_environment_keys", [])
        ),
    }


def _resume(destination: Path, *, input_digest: str) -> dict[str, object] | None:
    if not destination.exists() and not destination.is_symlink():
        return None
    if destination.is_symlink() or not destination.is_dir():
        raise ShadowProbeAdmissionError("SHADOW_PROBE_OUTPUT_INVALID")
    lock = _load(destination / "input-lock.json", "SHADOW_PROBE_INPUT_LOCK_INVALID")
    if lock.get("input_digest") != input_digest:
        raise ShadowProbeAdmissionError("SHADOW_PROBE_INPUT_LOCK_MISMATCH")
    manifest = _load(destination / "manifest.json", "SHADOW_PROBE_MANIFEST_INVALID")
    if manifest.get("state") not in {"admitted_for_shadow_execution", "blocked"}:
        raise ShadowProbeAdmissionError("SHADOW_PROBE_MANIFEST_INVALID")
    receipt_path = _require_file(
        Path(str(manifest.get("receipt_path") or "")),
        "SHADOW_PROBE_RECEIPT_INVALID",
    )
    receipt_sha = manifest.get("receipt_sha256")
    if (
        not isinstance(receipt_sha, str)
        or _sha256(receipt_path.read_bytes()) != receipt_sha
    ):
        raise ShadowProbeAdmissionError("SHADOW_PROBE_RECEIPT_HASH_MISMATCH")
    candidate_path_value = manifest.get("candidate_path")
    candidate_sha = manifest.get("candidate_sha256")
    if candidate_path_value is None:
        if candidate_sha is not None:
            raise ShadowProbeAdmissionError("SHADOW_PROBE_CANDIDATE_BINDING_INVALID")
    else:
        candidate_path = _require_file(
            Path(str(candidate_path_value)), "SHADOW_PROBE_CANDIDATE_INVALID"
        )
        if (
            not isinstance(candidate_sha, str)
            or _sha256(candidate_path.read_bytes()) != candidate_sha
        ):
            raise ShadowProbeAdmissionError("SHADOW_PROBE_CANDIDATE_HASH_MISMATCH")
    return manifest


def _validate(name: str, document: Mapping[str, object], *, root: Path) -> None:
    try:
        validate_document(name, document, root=root)
    except ContractValidationError as exc:
        raise ShadowProbeAdmissionError(
            f"SHADOW_PROBE_CONTRACT_INVALID:{name}:{exc}"
        ) from exc


def _mapping(value: Mapping[str, object], name: str) -> Mapping[str, object]:
    row = value.get(name)
    if not isinstance(row, Mapping):
        raise ShadowProbeAdmissionError("SHADOW_PROBE_MAPPING_INVALID:" + name)
    return row


def _mapping_rows(value: object) -> list[Mapping[str, object]]:
    if not isinstance(value, list):
        return []
    return [row for row in value if isinstance(row, Mapping)]


def _require_file(path: Path, code: str) -> Path:
    raw = Path(path).expanduser()
    if raw.is_symlink():
        raise ShadowProbeAdmissionError(code)
    resolved = raw.resolve()
    if not resolved.is_file():
        raise ShadowProbeAdmissionError(code)
    return resolved


def _load(path: Path, code: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ShadowProbeAdmissionError(code) from exc
    if not isinstance(value, dict):
        raise ShadowProbeAdmissionError(code)
    return value


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    body = _canonical_bytes(value)
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != body:
            raise ShadowProbeAdmissionError("SHADOW_PROBE_IMMUTABLE_WRITE_CONFLICT")
        return
    path.write_bytes(body)


def _digest(value: Mapping[str, object]) -> str:
    # Cross-module bindings use the repository's compact JSON digest convention,
    # while transaction files retain a trailing newline for stable file hashes.
    return _sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
        .encode("ascii")
    )


def _canonical_bytes(value: Mapping[str, object]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
        + "\n"
    ).encode("ascii")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()
