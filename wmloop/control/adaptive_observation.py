"""Plan bounded observations from portrait-readiness blockers.

The planner reuses admitted probes before selecting a shadow-only generation
template.  It may propose a typed read-only interface extension, but it never
grants intervention GPU, evaluator, metric, verdict, or promotion authority.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any

from wmloop.contracts import ContractValidationError, validate_document
from wmloop.geometry.evidence_ir import reject_runtime_bindings
from wmloop.geometry.types import GeometryValidationError


class AdaptiveObservationError(ValueError):
    """An observation registry, blocker, or generated plan is invalid."""


def load_observation_abi_registry(
    path: Path, *, root: Path | None = None
) -> dict[str, object]:
    source = Path(path).expanduser().resolve()
    if not source.is_file() or source.is_symlink():
        raise AdaptiveObservationError("OBSERVATION_ABI_REGISTRY_FILE_INVALID")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AdaptiveObservationError("OBSERVATION_ABI_REGISTRY_FILE_INVALID") from exc
    if not isinstance(payload, dict):
        raise AdaptiveObservationError("OBSERVATION_ABI_REGISTRY_FILE_INVALID")
    try:
        validate_document("observation_module_abi_registry", payload, root=root)
    except ContractValidationError as exc:
        raise AdaptiveObservationError(
            f"OBSERVATION_ABI_REGISTRY_SCHEMA_INVALID:{exc}"
        ) from exc
    abis = _mapping_rows(payload.get("abis"))
    identities = [str(row["abi_id"]) for row in abis]
    if len(identities) != len(set(identities)):
        raise AdaptiveObservationError("OBSERVATION_ABI_REGISTRY_DUPLICATE")
    body = dict(payload)
    received = body.pop("registry_digest", None)
    if received != _canonical_digest(body):
        raise AdaptiveObservationError("OBSERVATION_ABI_REGISTRY_DIGEST_MISMATCH")
    for row in abis:
        if any(
            row.get(field) is not False
            for field in (
                "verdict_exposure_allowed",
                "active_metric_mutation_allowed",
                "active_evaluator_mutation_allowed",
            )
        ):
            raise AdaptiveObservationError("OBSERVATION_ABI_AUTHORITY_INVALID")
    return payload


def build_adaptive_probe_plan(
    *,
    observation_work_order: Mapping[str, object],
    registry: Mapping[str, object],
    root: Path | None = None,
) -> tuple[dict[str, object], tuple[dict[str, object], ...]]:
    """Compile readiness blockers into deterministic observation tasks."""

    try:
        validate_document(
            "portrait_observation_work_order", observation_work_order, root=root
        )
        validate_document("observation_module_abi_registry", registry, root=root)
    except ContractValidationError as exc:
        raise AdaptiveObservationError(f"ADAPTIVE_PROBE_INPUT_SCHEMA_INVALID:{exc}") from exc
    _reject_runtime(observation_work_order)
    _validate_registry_mapping(registry)
    requirements = _mapping(observation_work_order, "requirements")
    probe_requirements = {
        str(row.get("coverage_key")): row
        for row in _mapping_rows(requirements.get("probe_coverage"))
    }
    blockers = observation_work_order.get("blockers")
    if not isinstance(blockers, list) or not blockers:
        raise AdaptiveObservationError("ADAPTIVE_PROBE_BLOCKERS_INVALID")
    tasks: list[dict[str, object]] = []
    extensions: dict[str, dict[str, object]] = {}
    seen_task_bodies: set[str] = set()
    for blocker_value in sorted(str(value) for value in blockers):
        for task, extension in _tasks_for_blocker(
            blocker_value,
            probe_requirements=probe_requirements,
            registry=registry,
            root=root,
        ):
            task_body = _canonical_json(task)
            if task_body in seen_task_bodies:
                continue
            seen_task_bodies.add(task_body)
            task["task_id"] = _stable_id("observation-task", task)
            tasks.append(task)
            if extension is not None:
                extensions[str(extension["extension_id"])] = extension
    tasks.sort(key=lambda row: str(row["task_id"]))
    blocked_types = {
        "requires_evaluator_binding",
        "missing_data_regime",
        "architecture_bound",
    }
    state = (
        "blocked"
        if any(str(task["task_type"]) in blocked_types for task in tasks)
        else "ready_for_observation"
    )
    body: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "verdiwm-adaptive-probe-plan",
        "observation_id": observation_work_order["observation_id"],
        "portrait_id": observation_work_order["portrait_id"],
        "readiness_id": observation_work_order["readiness_id"],
        "goal_binding": observation_work_order["goal_binding"],
        "registry_id": registry["registry_id"],
        "registry_digest": registry["registry_digest"],
        "state": state,
        "tasks": tasks,
        "authority": {
            "intervention_gpu_authority": False,
            "promotion_authority": False,
            "active_metric_mutation": False,
            "active_evaluator_mutation": False,
            "novel_probe_authority": "shadow_only",
        },
        "claim_boundary": (
            "This plan may schedule read-only or shadow observations only. It grants no "
            "intervention, evaluator, active-metric, verdict, or promotion authority."
        ),
    }
    body["plan_id"] = _stable_id("adaptive-probe-plan", body)
    validate_adaptive_probe_plan(body, root=root)
    return body, tuple(extensions[key] for key in sorted(extensions))


def validate_adaptive_probe_plan(
    document: Mapping[str, object], *, root: Path | None = None
) -> None:
    _reject_runtime(document)
    try:
        validate_document("adaptive_probe_plan", document, root=root)
    except ContractValidationError as exc:
        raise AdaptiveObservationError(f"ADAPTIVE_PROBE_PLAN_SCHEMA_INVALID:{exc}") from exc
    tasks = _mapping_rows(document.get("tasks"))
    identities = [str(row["task_id"]) for row in tasks]
    if len(identities) != len(set(identities)):
        raise AdaptiveObservationError("ADAPTIVE_PROBE_TASK_DUPLICATE")
    for row in tasks:
        body = dict(row)
        received = body.pop("task_id", None)
        if received != _stable_id("observation-task", body):
            raise AdaptiveObservationError("ADAPTIVE_PROBE_TASK_ID_MISMATCH")
        if row.get("task_type") == "manufacture_shadow_probe" and row.get(
            "execution_authority"
        ) != "shadow_only":
            raise AdaptiveObservationError("ADAPTIVE_PROBE_SHADOW_AUTHORITY_INVALID")
    authority = _mapping(document, "authority")
    if authority != {
        "intervention_gpu_authority": False,
        "promotion_authority": False,
        "active_metric_mutation": False,
        "active_evaluator_mutation": False,
        "novel_probe_authority": "shadow_only",
    }:
        raise AdaptiveObservationError("ADAPTIVE_PROBE_AUTHORITY_INVALID")
    body = dict(document)
    received = body.pop("plan_id", None)
    if received != _stable_id("adaptive-probe-plan", body):
        raise AdaptiveObservationError("ADAPTIVE_PROBE_PLAN_ID_MISMATCH")


def build_interface_extension_spec(
    *,
    requested_surface: str,
    semantic_role: str,
    root: Path | None = None,
) -> dict[str, object]:
    """Describe a narrow read-only observation hook without mutating source."""

    body: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "verdiwm-interface-extension-spec",
        "requested_surface": _required_text(
            requested_surface, "INTERFACE_EXTENSION_SURFACE_INVALID"
        ),
        "semantic_role": _required_text(
            semantic_role, "INTERFACE_EXTENSION_ROLE_INVALID"
        ),
        "typed_inputs": [
            {"name": "model_state", "contract": "read-only-semantic-model-state-v1"}
        ],
        "typed_outputs": [
            {"name": "observation", "contract": "finite-diagnostic-observation-v1"}
        ],
        "side_effect_class": "read_only_observation_hook",
        "authority_level": "L1",
        "conformance_tests": [
            "exact_typed_signature",
            "read_only_source_and_state",
            "deterministic_fixture_output",
        ],
        "negative_tests": [
            "reject_source_mutation",
            "reject_evaluator_mutation",
            "reject_active_metric_registration",
            "reject_verdict_exposure",
        ],
        "source_mutation_allowed": False,
        "active_metric_mutation_allowed": False,
        "active_evaluator_mutation_allowed": False,
        "execution_state": "proposed",
        "claim_boundary": (
            "This proposal exposes one read-only observation surface. It is not an "
            "implementation, evaluator change, intervention, or scientific verdict."
        ),
    }
    body["extension_id"] = _stable_id("interface-extension", body)
    validate_interface_extension_spec(body, root=root)
    return body


def validate_interface_extension_spec(
    document: Mapping[str, object], *, root: Path | None = None
) -> None:
    _reject_runtime(document)
    try:
        validate_document("interface_extension_spec", document, root=root)
    except ContractValidationError as exc:
        raise AdaptiveObservationError(
            f"INTERFACE_EXTENSION_SCHEMA_INVALID:{exc}"
        ) from exc
    body = dict(document)
    received = body.pop("extension_id", None)
    if received != _stable_id("interface-extension", body):
        raise AdaptiveObservationError("INTERFACE_EXTENSION_ID_MISMATCH")


def _tasks_for_blocker(
    blocker: str,
    *,
    probe_requirements: Mapping[str, Mapping[str, object]],
    registry: Mapping[str, object],
    root: Path | None,
) -> list[tuple[dict[str, object], dict[str, object] | None]]:
    if blocker.startswith(
        ("PORTRAIT_PROBE_COVERAGE_MISSING:", "PORTRAIT_CONFLICTING_FINGERPRINT:")
    ):
        key = blocker.rsplit(":", 1)[1]
        requirement = probe_requirements.get(key)
        if requirement is None:
            raise AdaptiveObservationError("ADAPTIVE_PROBE_REQUIREMENT_BINDING_MISSING")
        return [(_probe_task(blocker, requirement=requirement, registry=registry), None)]
    if blocker.startswith("PORTRAIT_STALE_FINGERPRINT:"):
        if not probe_requirements:
            return [(_read_only_task(blocker, registry=registry), None)]
        return [
            (_probe_task(blocker, requirement=requirement, registry=registry), None)
            for _, requirement in sorted(probe_requirements.items())
        ]
    if blocker.startswith(
        (
            "PORTRAIT_CAPABILITY_UNKNOWN:",
            "PORTRAIT_INTERFACE_UNKNOWN:",
            "PORTRAIT_HOOK_UNKNOWN:",
            "PORTRAIT_OPERATIONAL_UNKNOWN:",
        )
    ):
        return [(_read_only_task(blocker, registry=registry), None)]
    if blocker.startswith(
        (
            "PORTRAIT_CAPABILITY_UNAVAILABLE:",
            "PORTRAIT_INTERFACE_UNAVAILABLE:",
            "PORTRAIT_HOOK_UNAVAILABLE:",
        )
    ):
        surface = blocker.rsplit(":", 1)[1]
        extension = build_interface_extension_spec(
            requested_surface=surface,
            semantic_role=_interface_role(blocker),
            root=root,
        )
        return [
            (
                {
                    "task_type": "generate_interface_extension",
                    "blocker": blocker,
                    "abi_id": None,
                    "probe_coverage_key": None,
                    "interface_extension_id": extension["extension_id"],
                    "execution_authority": "read_only",
                    "reason": "A required observation surface is confirmed unavailable.",
                },
                extension,
            )
        ]
    if blocker.startswith("PORTRAIT_EVALUATOR_"):
        return [(_blocked_task("requires_evaluator_binding", blocker), None)]
    if blocker.startswith("MISSING_DATA_REGIME:"):
        return [(_blocked_task("missing_data_regime", blocker), None)]
    if blocker.startswith("ARCHITECTURE_BOUND:"):
        return [(_blocked_task("architecture_bound", blocker), None)]
    raise AdaptiveObservationError("ADAPTIVE_PROBE_BLOCKER_UNSUPPORTED:" + blocker)


def _probe_task(
    blocker: str,
    *,
    requirement: Mapping[str, object],
    registry: Mapping[str, object],
) -> dict[str, object]:
    protocol = (
        str(requirement["probe_protocol_id"])
        + "@"
        + str(requirement["probe_protocol_version"])
    )
    key = str(requirement["coverage_key"])
    admitted = [
        row
        for row in _mapping_rows(registry.get("abis"))
        if row.get("module_kind") == "diagnostic_probe_extension"
        and row.get("admission_state") == "admitted"
        and protocol in row.get("supported_probe_protocols", [])
    ]
    if admitted:
        selected = sorted(admitted, key=lambda row: str(row["abi_id"]))[0]
        return {
            "task_type": "reuse_existing_probe",
            "blocker": blocker,
            "abi_id": selected["abi_id"],
            "probe_coverage_key": key,
            "interface_extension_id": None,
            "execution_authority": "shadow_only",
            "reason": "An admitted probe ABI exactly covers the required protocol.",
        }
    templates = [
        row
        for row in _mapping_rows(registry.get("abis"))
        if row.get("module_kind") == "diagnostic_probe_extension"
        and row.get("admission_state") == "shadow_template"
    ]
    if not templates:
        raise AdaptiveObservationError("ADAPTIVE_PROBE_SHADOW_TEMPLATE_MISSING")
    selected = sorted(templates, key=lambda row: str(row["abi_id"]))[0]
    return {
        "task_type": "manufacture_shadow_probe",
        "blocker": blocker,
        "abi_id": selected["abi_id"],
        "probe_coverage_key": key,
        "interface_extension_id": None,
        "execution_authority": "shadow_only",
        "reason": "No admitted probe covers the protocol; generate a shadow-only candidate.",
    }


def _read_only_task(
    blocker: str, *, registry: Mapping[str, object]
) -> dict[str, object]:
    adapters = [
        row
        for row in _mapping_rows(registry.get("abis"))
        if row.get("module_kind") == "read_only_adapter"
        and row.get("admission_state") == "admitted"
    ]
    if not adapters:
        raise AdaptiveObservationError("ADAPTIVE_PROBE_READ_ONLY_ADAPTER_MISSING")
    selected = sorted(adapters, key=lambda row: str(row["abi_id"]))[0]
    return {
        "task_type": "run_read_only_adapter",
        "blocker": blocker,
        "abi_id": selected["abi_id"],
        "probe_coverage_key": None,
        "interface_extension_id": None,
        "execution_authority": "read_only",
        "reason": "The structural or operational state is unknown and requires read-only inspection.",
    }


def _blocked_task(task_type: str, blocker: str) -> dict[str, object]:
    return {
        "task_type": task_type,
        "blocker": blocker,
        "abi_id": None,
        "probe_coverage_key": None,
        "interface_extension_id": None,
        "execution_authority": "none",
        "reason": "The missing authority, data regime, or architecture cannot be generated by observation code.",
    }


def _interface_role(blocker: str) -> str:
    if blocker.startswith("PORTRAIT_HOOK_UNAVAILABLE:"):
        return "diagnostic_hook"
    if blocker.startswith("PORTRAIT_INTERFACE_UNAVAILABLE:"):
        return "execution_interface"
    return "capability_observation_surface"


def _validate_registry_mapping(registry: Mapping[str, object]) -> None:
    body = dict(registry)
    received = body.pop("registry_digest", None)
    if received != _canonical_digest(body):
        raise AdaptiveObservationError("OBSERVATION_ABI_REGISTRY_DIGEST_MISMATCH")


def _mapping(value: Mapping[str, object], name: str) -> Mapping[str, Any]:
    result = value.get(name)
    if not isinstance(result, Mapping):
        raise AdaptiveObservationError("ADAPTIVE_PROBE_MAPPING_REQUIRED:" + name)
    return result


def _mapping_rows(value: object) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(row, Mapping) for row in value):
        raise AdaptiveObservationError("ADAPTIVE_PROBE_MAPPING_ROWS_INVALID")
    return list(value)


def _required_text(value: object, code: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AdaptiveObservationError(code)
    text = value.strip()
    _reject_runtime(text)
    return text


def _reject_runtime(value: object) -> None:
    try:
        reject_runtime_bindings(value)
    except GeometryValidationError as exc:
        raise AdaptiveObservationError(
            f"ADAPTIVE_PROBE_RUNTIME_BINDING_FORBIDDEN:{exc}"
        ) from exc


def _stable_id(prefix: str, body: Mapping[str, object]) -> str:
    return prefix + "-" + _canonical_digest(body)[:24]


def _canonical_digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise AdaptiveObservationError("ADAPTIVE_PROBE_CANONICALIZATION_INVALID") from exc
