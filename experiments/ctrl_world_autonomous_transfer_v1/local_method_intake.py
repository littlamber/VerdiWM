"""Offline, source-addressed method intake for the Ctrl-World loop.

The external research intake is intentionally allowed to fail when the host has
no network.  This module turns the repository's reviewed mechanism profiles
into the same immutable assessment/idea/work-order contract.  It only admits
profiles that have an explicitly configured materializer; the remaining local
methods are recorded as knowledge-only candidates by the caller.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path

from wmloop.contracts import ContractValidationError, validate_document
from wmloop.execute.acwm_primitive_routes import primitive_execution_role
from wmloop.primitives.registry import PrimitiveRegistry


class LocalMethodIntakeError(RuntimeError):
    """A local method could not be compiled without weakening its contract."""


def run_local_method_intake(
    *,
    config: Mapping[str, object],
    output_root: Path,
    project_root: Path,
    failure_context: Sequence[str] = (),
) -> dict[str, object]:
    """Compile configured, reviewed local profiles into a durable intake bundle."""

    root = Path(project_root).expanduser().resolve()
    settings = config.get("local_method_discovery")
    if not isinstance(settings, Mapping) or settings.get("enabled") is not True:
        return {
            "state": "disabled",
            "assessment_paths": [],
            "idea_paths": [],
            "work_order_paths": [],
            "local_method_count": 0,
        }
    intake_path = _file(settings.get("intake_config"), "LOCAL_METHOD_INTAKE_CONFIG_INVALID")
    intake = _load(intake_path, "LOCAL_METHOD_INTAKE_CONFIG_INVALID")
    contract_path = _file(config.get("paths", {}).get("contract") if isinstance(config.get("paths"), Mapping) else None, "LOCAL_METHOD_CONTRACT_INVALID")
    contract = _load(contract_path, "LOCAL_METHOD_CONTRACT_INVALID")
    materializers = config.get("materializers")
    if not isinstance(materializers, Mapping):
        raise LocalMethodIntakeError("LOCAL_METHOD_MATERIALIZERS_INVALID")
    requested = settings.get("materializer_profiles")
    if requested is None:
        profile_ids = sorted(str(key) for key in materializers)
    elif isinstance(requested, list) and all(isinstance(value, str) and value for value in requested):
        profile_ids = sorted(set(str(value) for value in requested))
    else:
        raise LocalMethodIntakeError("LOCAL_METHOD_PROFILE_SELECTION_INVALID")
    unknown = [profile_id for profile_id in profile_ids if profile_id not in materializers]
    if unknown:
        raise LocalMethodIntakeError("LOCAL_METHOD_MATERIALIZER_PROFILE_UNKNOWN:" + ",".join(unknown))
    profiles = intake.get("mechanism_profiles")
    if not isinstance(profiles, list):
        raise LocalMethodIntakeError("LOCAL_METHOD_PROFILES_INVALID")
    by_id = {str(row.get("profile_id")): row for row in profiles if isinstance(row, Mapping)}
    selected = [by_id[profile_id] for profile_id in profile_ids if profile_id in by_id]
    maximum = settings.get("max_methods", len(selected))
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1:
        raise LocalMethodIntakeError("LOCAL_METHOD_MAX_INVALID")
    selected = sorted(
        selected,
        key=lambda row: (-int(row.get("priority", 0)), str(row.get("profile_id"))),
    )[:maximum]
    primitive_registry = PrimitiveRegistry.from_root(root)
    primitive_inventory = [
        {
            "primitive": name,
            "execution_role": primitive_execution_role(name),
            "execution_state": "knowledge_only",
            "reason": (
                "No closed-loop materialization receipt is configured for this primitive; "
                "retain it as a method/probe candidate until the materialization gate passes."
            ),
        }
        for name in primitive_registry.names()
    ]
    destination = Path(output_root).expanduser().resolve()
    if destination == root or root in destination.parents:
        raise LocalMethodIntakeError("LOCAL_METHOD_OUTPUT_INSIDE_SOURCE")
    input_lock = {
        "schema_version": 1,
        "artifact_type": "verdiwm-local-method-intake-lock",
        "intake_config": str(intake_path),
        "intake_config_sha256": _sha256(intake_path.read_bytes()),
        "contract_sha256": _sha256(contract_path.read_bytes()),
        "materializer_profiles": profile_ids,
        "primitive_registry_digest": primitive_registry.digest(),
        "failure_context": sorted(set(str(value).strip() for value in failure_context if str(value).strip())),
    }
    input_lock["input_sha256"] = _digest(input_lock)
    resumed = _resume(destination, input_lock)
    if resumed is not None:
        return resumed
    if destination.exists() or destination.is_symlink():
        raise LocalMethodIntakeError("LOCAL_METHOD_OUTPUT_EXISTS")
    temporary = destination.parent / f".{destination.name}.tmp-{os.getpid()}"
    temporary.mkdir(mode=0o700, parents=True, exist_ok=False)
    assessments: list[str] = []
    ideas: list[str] = []
    work_orders: list[str] = []
    try:
        _write(temporary / "input-lock.json", input_lock)
        for profile in selected:
            assessment = _assessment(
                intake=intake,
                profile=profile,
                input_lock=input_lock,
                failure_context=failure_context,
            )
            idea = _idea(intake, assessment)
            work_order = _work_order(intake, contract, idea)
            _validate("acwm_source_transfer_assessment_v2", assessment, root)
            _validate("acwm_research_idea_v2", idea, root)
            _validate("acwm_research_work_order_v2", work_order, root)
            assessment_path = temporary / "assessments" / f"{assessment['source_id']}.json"
            idea_path = temporary / "ideas" / f"{idea['idea_id']}.json"
            work_order_path = temporary / "work-orders" / f"{idea['idea_id']}.json"
            _write(assessment_path, assessment)
            _write(idea_path, idea)
            _write(work_order_path, work_order)
            assessments.append(str(destination / assessment_path.relative_to(temporary)))
            ideas.append(str(destination / idea_path.relative_to(temporary)))
            work_orders.append(str(destination / work_order_path.relative_to(temporary)))
        manifest = {
            "schema_version": 1,
            "artifact_type": "verdiwm-local-method-intake-manifest",
            "state": "ready_for_materialization" if selected else "empty",
            "source_type": "local_reviewed_mechanism_profile",
            "input_sha256": input_lock["input_sha256"],
            "local_method_count": len(selected),
            "assessment_paths": assessments,
            "idea_paths": ideas,
            "work_order_paths": work_orders,
            "knowledge_only_profiles": sorted(
                profile_id for profile_id in by_id if profile_id not in profile_ids
            ),
            "primitive_registry_digest": primitive_registry.digest(),
            "primitive_inventory": primitive_inventory,
            "claim_boundary": (
                "Local profile grounding is repository-owned method evidence, not an external "
                "paper claim. GPU and promotion authority remain downstream."
            ),
        }
        _write(temporary / "manifest.json", manifest)
        os.replace(temporary, destination)
        return manifest
    except Exception:
        if temporary.exists() and not temporary.is_symlink():
            import shutil

            shutil.rmtree(temporary)
        raise


def _assessment(*, intake: Mapping[str, object], profile: Mapping[str, object], input_lock: Mapping[str, object], failure_context: Sequence[str]) -> dict[str, object]:
    profile_id = str(profile.get("profile_id") or "")
    if not profile_id or profile.get("source_role") != "transferable_optimizer":
        raise LocalMethodIntakeError("LOCAL_METHOD_PROFILE_NOT_EXECUTABLE:" + profile_id)
    components = [row for row in profile.get("components", []) if isinstance(row, Mapping)]
    if not components:
        raise LocalMethodIntakeError("LOCAL_METHOD_COMPONENTS_MISSING:" + profile_id)
    source_id = "local-profile-" + _safe_id(profile_id)
    source_digest = _digest({"profile": profile, "input": input_lock["intake_config_sha256"]})
    evidence = [
        {
            "component_id": str(row["component_id"]),
            "description": str(row["description"]),
            "evidence_type": "local_reviewed_profile",
            "evidence_ref": "sha256:" + source_digest,
            "evidence_snippet": str(row["description"]),
        }
        for row in components
    ]
    body: dict[str, object] = {
        "schema_version": 2,
        "artifact_type": "verdiwm-acwm-source-transfer-assessment",
        "source_id": source_id,
        "source_digest": source_digest,
        "source_title": "Local reviewed mechanism profile: " + profile_id,
        "profile_id": profile_id,
        "source_role": "transferable_optimizer",
        "track": str(profile.get("track") or "local_reviewed"),
        "transfer_mode": "mechanism_transfer",
        "execution_state": "materialization_required",
        "failure_context": [str(value) for value in failure_context if str(value).strip()],
        "source_evidence": evidence,
        "preserved_components": [str(row["component_id"]) for row in components],
        "required_target_capabilities": [str(value) for value in profile.get("required_capabilities", [])],
        "missing_target_capabilities": [],
        "target_intervention": str(profile.get("target_intervention") or ""),
        "forbidden_substitutions": [str(value) for value in profile.get("forbidden_substitutions", [])],
        "falsification_criterion": str(profile.get("falsification_criterion") or "Reject unless the frozen verifier supports the declared improvement."),
        "claim_boundary": "This local profile is an executable hypothesis seed, not a scientific result or external source claim.",
    }
    body["assessment_digest"] = _digest(body)
    return body


def _idea(intake: Mapping[str, object], assessment: Mapping[str, object]) -> dict[str, object]:
    profile_id = str(assessment["profile_id"])
    idea_id = "idea-local-" + _safe_id(profile_id) + "-v1"
    return {
        "schema_version": 2,
        "artifact_type": "verdiwm-acwm-research-idea",
        "idea_id": idea_id,
        "source": {key: assessment[key] for key in ("source_id", "source_digest", "assessment_digest")},
        "track": assessment["track"],
        "transfer_mode": "mechanism_transfer",
        "failure_context": list(assessment["failure_context"]),
        "source_evidence": list(assessment["source_evidence"]),
        "materialization_contract": {
            "target_model_family": intake["target_model_family"],
            "target_intervention": assessment["target_intervention"],
            "preserved_components": list(assessment["preserved_components"]),
            "required_target_capabilities": list(assessment["required_target_capabilities"]),
            "forbidden_substitutions": list(assessment["forbidden_substitutions"]),
        },
        "objective": f"Test the locally reviewed {profile_id} mechanism on {intake['target_model_family']} under the frozen ACWM contract.",
        "hypothesis": "Preserving " + ", ".join(str(value) for value in assessment["preserved_components"]) + " improves the primary paired-prediction metric without a protected regression.",
        "falsification_criterion": assessment["falsification_criterion"],
        "required_protocol": ["screen", "confirm", "frozen_verifier"],
        "synthesis_method": "source_component_grounding_v2",
        "execution_state": "materialization_required",
        "claim_boundary": "This is a locally grounded mechanism-transfer hypothesis; it is not a claim of external paper reproduction.",
    }


def _work_order(intake: Mapping[str, object], contract: Mapping[str, object], idea: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema_version": 2,
        "artifact_type": "verdiwm-acwm-research-materialization-work-order",
        "idea_id": idea["idea_id"],
        "idea_sha256": _canonical_sha256(idea),
        "policy_id": intake["policy_id"],
        "contract": {"contract_id": contract["contract_id"], "contract_digest": contract["contract_digest"]},
        "source_grounding": idea["source"],
        "materialization_contract": idea["materialization_contract"],
        "required_before_gpu": ["isolated implementation receipt with source revision", "complete source-component-to-code mapping", "zero declared compromises", "immutable candidate batch bound to the implementation receipt", "independent screen and confirm baselines"],
        "required_execution_stages": ["screen", "confirm", "frozen_verifier"],
        "forbidden_actions": ["execute external source code", "replace a preserved component with a proxy", "mutate the frozen evaluator or data split", "reuse confirm measurements as a screen baseline", "promote a model from screen or confirm evidence"],
        "execution_authority": "none_until_materialized_and_compiled",
        "claim_boundary": "This work order grants no GPU or promotion authority. The materializer must preserve every declared component or abstain.",
    }


def _resume(destination: Path, input_lock: Mapping[str, object]) -> dict[str, object] | None:
    if not destination.exists():
        return None
    if destination.is_symlink() or not destination.is_dir():
        raise LocalMethodIntakeError("LOCAL_METHOD_OUTPUT_INVALID")
    lock = _load(destination / "input-lock.json", "LOCAL_METHOD_INPUT_LOCK_INVALID")
    if lock.get("input_sha256") != input_lock.get("input_sha256"):
        raise LocalMethodIntakeError("LOCAL_METHOD_INPUT_LOCK_MISMATCH")
    return _load(destination / "manifest.json", "LOCAL_METHOD_MANIFEST_INVALID")


def _validate(name: str, payload: Mapping[str, object], root: Path) -> None:
    try:
        validate_document(name, payload, root=root)
    except ContractValidationError as exc:
        raise LocalMethodIntakeError(f"LOCAL_METHOD_SCHEMA_INVALID:{name}:{exc}") from exc


def _load(path: Path, code: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LocalMethodIntakeError(code) from exc
    if not isinstance(payload, dict):
        raise LocalMethodIntakeError(code)
    return payload


def _file(value: object, code: str) -> Path:
    path = Path(str(value or "")).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise LocalMethodIntakeError(code)
    return path


def _write(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _digest(payload: Mapping[str, object]) -> str:
    body = {key: value for key, value in payload.items() if key not in {"assessment_digest", "input_sha256"}}
    encoded = (
        json.dumps(body, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_sha256(payload: Mapping[str, object]) -> str:
    encoded = (
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_id(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "-" for char in value.lower()).strip("-") or "method"
