"""Deterministic, path-free model portraits and goal-relative readiness.

The portrait combines structural Capability IR with evidence-bound behavioral
fingerprints.  It records observation coverage but carries no experiment or
claim authority.  A separate readiness receipt decides whether one Goal IR has
enough evidence to enter capability-gap planning.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import copy
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from wmloop.contracts import ContractValidationError, validate_document
from wmloop.control.intermediate_ir import (
    IntermediateRepresentationError,
    ir_digest,
    validate_model_capability_ir,
)
from wmloop.geometry.evidence_ir import is_content_addressed, reject_runtime_bindings
from wmloop.geometry.portable_transfer_knowledge import (
    validate_probe_fingerprint_summary,
)
from wmloop.geometry.types import GeometryValidationError


class ModelPortraitError(ValueError):
    """A portrait source, identity, or readiness decision is invalid."""


_HEX_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_READINESS_PRIORITY = (
    "stale_portrait",
    "conflicting_evidence",
    "requires_evaluator_binding",
    "requires_static_onboarding",
    "requires_interface_extension",
    "requires_probe_coverage",
)


def build_model_portrait(
    *,
    model_capability: Mapping[str, object],
    fingerprints: Sequence[Mapping[str, object]] = (),
    operational_observations: Sequence[Mapping[str, object]] = (),
    parent_portrait: Mapping[str, object] | None = None,
    transition_ref: str | None = None,
    evidence_refs: Sequence[str] = (),
    root: Path | None = None,
) -> dict[str, object]:
    """Build an append-only semantic portrait from validated observations."""

    try:
        validate_model_capability_ir(model_capability, root=root)
    except IntermediateRepresentationError as exc:
        raise ModelPortraitError(f"MODEL_PORTRAIT_CAPABILITY_INVALID:{exc}") from exc
    fingerprint_rows = _validated_fingerprints(fingerprints)
    operational_rows = _normalized_operational_observations(operational_observations)
    parent_id: str | None = None
    if parent_portrait is not None:
        validate_model_portrait(parent_portrait, root=root)
        parent_id = str(parent_portrait["portrait_id"])
        if not isinstance(transition_ref, str) or not is_content_addressed(transition_ref):
            raise ModelPortraitError("MODEL_PORTRAIT_TRANSITION_REF_REQUIRED")
    elif transition_ref is not None:
        raise ModelPortraitError("MODEL_PORTRAIT_PARENT_REQUIRED")

    capability_id = str(model_capability["capability_id"])
    behavioral = [
        _portrait_fingerprint(row, current_capability_id=capability_id)
        for row in fingerprint_rows
    ]
    behavioral.sort(key=lambda row: (str(row["coverage_key"]), str(row["fingerprint_id"])))
    structural = {
        "capabilities": _sorted_mapping_rows(model_capability["capabilities"], "capability"),
        "execution_interfaces": _sorted_mapping_rows(
            model_capability["execution_interfaces"], "kind"
        ),
        "hooks": _sorted_mapping_rows(model_capability["hooks"], "hook"),
        "asset_classes": sorted(str(value) for value in model_capability["asset_classes"]),
        "evaluator": dict(_mapping(model_capability, "evaluator")),
    }
    coverage = _portrait_coverage(structural, behavioral, operational_rows)
    refs = {
        "sha256:" + ir_digest(model_capability),
        *(_validated_refs(evidence_refs, "MODEL_PORTRAIT_EVIDENCE_REF_INVALID")),
    }
    for fingerprint in fingerprint_rows:
        refs.add(str(fingerprint["response_digest"]))
        refs.update(str(value) for value in fingerprint["evidence_refs"])
    for observation in operational_rows:
        refs.update(str(value) for value in observation["evidence_refs"])
    if transition_ref is not None:
        refs.add(transition_ref)

    source_revision = dict(_mapping(model_capability, "source_revision"))
    body: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "verdiwm-model-portrait",
        "model_capability_id": capability_id,
        "model_capability_digest": "sha256:" + ir_digest(model_capability),
        "model_family": str(model_capability["model_family"]),
        "source_revision": source_revision,
        "parent_portrait_id": parent_id,
        "transition_ref": transition_ref,
        "structural_profile": structural,
        "behavioral_fingerprints": behavioral,
        "operational_profile": operational_rows,
        "coverage": coverage,
        "evidence_refs": sorted(refs),
    }
    body["portrait_id"] = _stable_id("model-portrait", body)
    validate_model_portrait(body, root=root)
    return body


def derive_model_portrait(
    *,
    parent_portrait: Mapping[str, object],
    transition_ref: str,
    evidence_refs: Sequence[str],
    root: Path | None = None,
) -> dict[str, object]:
    """Create a conservative child portrait after an admitted intervention."""

    validate_model_portrait(parent_portrait, root=root)
    if not isinstance(transition_ref, str) or not is_content_addressed(
        transition_ref
    ):
        raise ModelPortraitError("MODEL_PORTRAIT_TRANSITION_REF_INVALID")
    behavioral = []
    for value in _mapping_rows(parent_portrait.get("behavioral_fingerprints")):
        row = dict(value)
        row["state"] = "stale"
        behavioral.append(row)
    operational = []
    for value in _mapping_rows(parent_portrait.get("operational_profile")):
        row = dict(value)
        if row.get("state") == "observed":
            row["state"] = "unknown"
            row["summary"] = (
                "Re-observation required after admitted intervention; prior evidence: "
                + str(row["summary"])
            )
        operational.append(row)
    structural = dict(_mapping(parent_portrait, "structural_profile"))
    refs = {
        *(
            _validated_refs(
                evidence_refs, "MODEL_PORTRAIT_EVIDENCE_REF_INVALID"
            )
        ),
        *(
            str(value)
            for value in parent_portrait.get("evidence_refs", [])
        ),
        transition_ref,
        "sha256:" + _canonical_digest(parent_portrait),
    }
    body: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "verdiwm-model-portrait",
        "model_capability_id": parent_portrait["model_capability_id"],
        "model_capability_digest": parent_portrait["model_capability_digest"],
        "model_family": parent_portrait["model_family"],
        "source_revision": dict(_mapping(parent_portrait, "source_revision")),
        "parent_portrait_id": parent_portrait["portrait_id"],
        "transition_ref": transition_ref,
        "structural_profile": structural,
        "behavioral_fingerprints": behavioral,
        "operational_profile": operational,
        "coverage": _portrait_coverage(structural, behavioral, operational),
        "evidence_refs": sorted(refs),
    }
    body["portrait_id"] = _stable_id("model-portrait", body)
    validate_model_portrait(body, root=root)
    return body


def update_model_portrait_from_observation(
    *,
    parent_portrait: Mapping[str, object],
    transition_ref: str,
    fingerprints: Sequence[Mapping[str, object]] = (),
    structural_observation: Mapping[str, object] | None = None,
    evidence_refs: Sequence[str] = (),
    root: Path | None = None,
) -> dict[str, object]:
    """Append validated read-only observations to one immutable portrait."""

    validate_model_portrait(parent_portrait, root=root)
    if not isinstance(transition_ref, str) or not is_content_addressed(
        transition_ref
    ):
        raise ModelPortraitError("MODEL_PORTRAIT_OBSERVATION_TRANSITION_INVALID")
    fingerprint_rows = _validated_fingerprints(fingerprints)
    if any(
        row.get("model_capability_id") != parent_portrait.get("model_capability_id")
        or row.get("model_family") != parent_portrait.get("model_family")
        for row in fingerprint_rows
    ):
        raise ModelPortraitError("MODEL_PORTRAIT_OBSERVATION_FINGERPRINT_MISMATCH")
    structural = copy.deepcopy(dict(_mapping(parent_portrait, "structural_profile")))
    operational = [
        copy.deepcopy(dict(row))
        for row in _mapping_rows(parent_portrait.get("operational_profile"))
    ]
    observation_refs = _validated_refs(
        evidence_refs, "MODEL_PORTRAIT_OBSERVATION_EVIDENCE_INVALID"
    )
    if transition_ref not in observation_refs:
        observation_refs.append(transition_ref)
    changed = False
    if structural_observation is not None:
        changed = _apply_structural_observation(
            structural,
            operational,
            structural_observation,
            evidence_refs=observation_refs,
        )

    behavioral = [
        copy.deepcopy(dict(row))
        for row in _mapping_rows(parent_portrait.get("behavioral_fingerprints"))
    ]
    existing_ids = {str(row["fingerprint_id"]) for row in behavioral}
    for fingerprint in fingerprint_rows:
        if str(fingerprint["fingerprint_id"]) in existing_ids:
            continue
        behavioral.append(
            _portrait_fingerprint(
                fingerprint,
                current_capability_id=str(parent_portrait["model_capability_id"]),
            )
        )
        existing_ids.add(str(fingerprint["fingerprint_id"]))
        changed = True
    if not changed:
        raise ModelPortraitError("MODEL_PORTRAIT_OBSERVATION_EMPTY")
    behavioral.sort(
        key=lambda row: (str(row["coverage_key"]), str(row["fingerprint_id"]))
    )
    operational.sort(key=lambda row: str(row["metric"]))
    refs = {
        *(str(value) for value in parent_portrait.get("evidence_refs", [])),
        *observation_refs,
        "sha256:" + _canonical_digest(parent_portrait),
    }
    for fingerprint in fingerprint_rows:
        refs.add(str(fingerprint["response_digest"]))
        refs.update(str(value) for value in fingerprint["evidence_refs"])
    body: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "verdiwm-model-portrait",
        "model_capability_id": parent_portrait["model_capability_id"],
        "model_capability_digest": parent_portrait["model_capability_digest"],
        "model_family": parent_portrait["model_family"],
        "source_revision": copy.deepcopy(
            dict(_mapping(parent_portrait, "source_revision"))
        ),
        "parent_portrait_id": parent_portrait["portrait_id"],
        "transition_ref": transition_ref,
        "structural_profile": structural,
        "behavioral_fingerprints": behavioral,
        "operational_profile": operational,
        "coverage": _portrait_coverage(structural, behavioral, operational),
        "evidence_refs": sorted(refs),
    }
    body["portrait_id"] = _stable_id("model-portrait", body)
    validate_model_portrait(body, root=root)
    return body


def validate_model_portrait(
    document: Mapping[str, object], *, root: Path | None = None
) -> None:
    if not isinstance(document, Mapping):
        raise ModelPortraitError("MODEL_PORTRAIT_DOCUMENT_INVALID")
    _reject_runtime(document)
    try:
        validate_document("model_portrait", document, root=root)
    except ContractValidationError as exc:
        raise ModelPortraitError(f"MODEL_PORTRAIT_SCHEMA_INVALID:{exc}") from exc
    _validate_refs(document.get("evidence_refs"), "MODEL_PORTRAIT_EVIDENCE_REF_INVALID")
    transition = document.get("transition_ref")
    parent = document.get("parent_portrait_id")
    if (parent is None) != (transition is None):
        raise ModelPortraitError("MODEL_PORTRAIT_TRANSITION_PAIR_INVALID")
    if isinstance(transition, str) and not is_content_addressed(transition):
        raise ModelPortraitError("MODEL_PORTRAIT_TRANSITION_REF_INVALID")
    _require_unique(document, "behavioral_fingerprints", "fingerprint_id")
    _require_unique(document, "operational_profile", "metric")
    structural = _mapping(document, "structural_profile")
    _require_unique(structural, "capabilities", "capability")
    _require_unique(structural, "execution_interfaces", "kind")
    _require_unique(structural, "hooks", "hook")
    assets = structural.get("asset_classes")
    if not isinstance(assets, list) or len(assets) != len(set(assets)):
        raise ModelPortraitError("MODEL_PORTRAIT_ASSET_CLASS_DUPLICATE")
    operational = _mapping_rows(document.get("operational_profile"))
    for observation in operational:
        _validate_refs(
            observation.get("evidence_refs"),
            "MODEL_PORTRAIT_OPERATIONAL_EVIDENCE_REF_INVALID",
        )
    expected_coverage = _portrait_coverage(
        structural,
        _mapping_rows(document.get("behavioral_fingerprints")),
        operational,
    )
    if document.get("coverage") != expected_coverage:
        raise ModelPortraitError("MODEL_PORTRAIT_COVERAGE_MISMATCH")
    body = dict(document)
    received = body.pop("portrait_id", None)
    if received != _stable_id("model-portrait", body):
        raise ModelPortraitError("MODEL_PORTRAIT_ID_MISMATCH")


def build_portrait_readiness_receipt(
    *,
    portrait: Mapping[str, object],
    goal_binding: str,
    coverage_policy_id: str,
    required_capabilities: Sequence[str] = (),
    required_execution_interfaces: Sequence[Mapping[str, object]] = (),
    required_hooks: Sequence[str] = (),
    required_probe_coverage: Sequence[Mapping[str, object]] = (),
    required_operational_metrics: Sequence[str] = (),
    evaluator_required: bool = True,
    root: Path | None = None,
) -> dict[str, object]:
    """Evaluate goal-relative portrait coverage without granting execution."""

    validate_model_portrait(portrait, root=root)
    if not isinstance(goal_binding, str) or not is_content_addressed(goal_binding):
        raise ModelPortraitError("PORTRAIT_READINESS_GOAL_BINDING_INVALID")
    policy_id = _required_text(
        coverage_policy_id, "PORTRAIT_READINESS_POLICY_ID_INVALID"
    )
    requirements = {
        "capabilities": _unique_texts(required_capabilities, "PORTRAIT_READINESS_CAPABILITY_INVALID"),
        "execution_interfaces": _normalized_interface_requirements(
            required_execution_interfaces
        ),
        "hooks": _unique_texts(required_hooks, "PORTRAIT_READINESS_HOOK_INVALID"),
        "probe_coverage": _normalized_probe_requirements(required_probe_coverage),
        "operational_metrics": _unique_texts(
            required_operational_metrics,
            "PORTRAIT_READINESS_OPERATIONAL_METRIC_INVALID",
        ),
        "evaluator_required": bool(evaluator_required),
    }
    blockers, coverage, evaluator_binding = _evaluate_readiness(portrait, requirements)
    state = _readiness_state(blockers)
    portrait_digest = "sha256:" + _canonical_digest(portrait)
    refs = {goal_binding, portrait_digest, str(portrait["model_capability_digest"])}
    refs.update(str(value) for value in portrait["evidence_refs"])
    body: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "verdiwm-portrait-readiness-receipt",
        "portrait_id": str(portrait["portrait_id"]),
        "portrait_digest": portrait_digest,
        "model_capability_id": str(portrait["model_capability_id"]),
        "model_capability_digest": str(portrait["model_capability_digest"]),
        "goal_binding": goal_binding,
        "coverage_policy_id": policy_id,
        "requirements": requirements,
        "coverage": coverage,
        "evaluator_binding": evaluator_binding,
        "state": state,
        "blockers": blockers,
        "evidence_refs": sorted(refs),
    }
    body["readiness_id"] = _stable_id("portrait-readiness", body)
    validate_portrait_readiness_receipt(body, root=root)
    return body


def validate_portrait_readiness_receipt(
    document: Mapping[str, object], *, root: Path | None = None
) -> None:
    if not isinstance(document, Mapping):
        raise ModelPortraitError("PORTRAIT_READINESS_DOCUMENT_INVALID")
    _reject_runtime(document)
    try:
        validate_document("portrait_readiness_receipt", document, root=root)
    except ContractValidationError as exc:
        raise ModelPortraitError(f"PORTRAIT_READINESS_SCHEMA_INVALID:{exc}") from exc
    _validate_refs(document.get("evidence_refs"), "PORTRAIT_READINESS_EVIDENCE_REF_INVALID")
    goal = document.get("goal_binding")
    evaluator = document.get("evaluator_binding")
    if not isinstance(goal, str) or not is_content_addressed(goal):
        raise ModelPortraitError("PORTRAIT_READINESS_GOAL_BINDING_INVALID")
    if evaluator is not None and (
        not isinstance(evaluator, str) or not is_content_addressed(evaluator)
    ):
        raise ModelPortraitError("PORTRAIT_READINESS_EVALUATOR_BINDING_INVALID")
    blockers = document.get("blockers")
    if not isinstance(blockers, list):
        raise ModelPortraitError("PORTRAIT_READINESS_BLOCKERS_INVALID")
    expected_state = _readiness_state([str(value) for value in blockers])
    if document.get("state") != expected_state:
        raise ModelPortraitError("PORTRAIT_READINESS_STATE_MISMATCH")
    body = dict(document)
    received = body.pop("readiness_id", None)
    if received != _stable_id("portrait-readiness", body):
        raise ModelPortraitError("PORTRAIT_READINESS_ID_MISMATCH")


def probe_coverage_key(value: Mapping[str, object]) -> str:
    """Return the semantic key shared by a fingerprint and its requirement."""

    body = {
        "probe_protocol_id": _required_text(
            value.get("probe_protocol_id"), "MODEL_PORTRAIT_PROBE_PROTOCOL_INVALID"
        ),
        "probe_protocol_version": _required_text(
            value.get("probe_protocol_version"), "MODEL_PORTRAIT_PROBE_VERSION_INVALID"
        ),
        "diagnostic_role": _required_text(
            value.get("diagnostic_role"), "MODEL_PORTRAIT_PROBE_ROLE_INVALID"
        ),
        "context_class": _required_text(
            value.get("context_class"), "MODEL_PORTRAIT_PROBE_CONTEXT_INVALID"
        ),
        "split": _required_text(value.get("split"), "MODEL_PORTRAIT_PROBE_SPLIT_INVALID"),
        "horizons": _integer_sequence(
            value.get("horizons"), "MODEL_PORTRAIT_PROBE_HORIZONS_INVALID"
        ),
        "dose_values": _number_sequence(
            value.get("dose_values"), "MODEL_PORTRAIT_PROBE_DOSES_INVALID"
        ),
    }
    return _stable_id("probe-coverage", body)


def _validated_fingerprints(
    fingerprints: Sequence[Mapping[str, object]],
) -> list[Mapping[str, object]]:
    rows: list[Mapping[str, object]] = []
    seen: set[str] = set()
    for row in fingerprints:
        try:
            validate_probe_fingerprint_summary(row)
        except GeometryValidationError as exc:
            raise ModelPortraitError(f"MODEL_PORTRAIT_FINGERPRINT_INVALID:{exc}") from exc
        identity = str(row["fingerprint_id"])
        if identity in seen:
            raise ModelPortraitError("MODEL_PORTRAIT_FINGERPRINT_DUPLICATE")
        seen.add(identity)
        rows.append(row)
    return rows


def _portrait_fingerprint(
    row: Mapping[str, object], *, current_capability_id: str
) -> dict[str, object]:
    return {
        "fingerprint_id": str(row["fingerprint_id"]),
        "model_capability_id": str(row["model_capability_id"]),
        "probe_protocol_id": str(row["probe_protocol_id"]),
        "probe_protocol_version": str(row["probe_protocol_version"]),
        "diagnostic_role": str(row["diagnostic_role"]),
        "context_class": str(row["context_class"]),
        "split": str(row["split"]),
        "horizons": list(row["horizons"]),
        "dose_values": list(row["dose_values"]),
        "replication_count": int(row["replication_count"]),
        "response_digest": str(row["response_digest"]),
        "coverage_key": probe_coverage_key(row),
        "state": (
            "current"
            if row["model_capability_id"] == current_capability_id
            else "stale"
        ),
    }


def _apply_structural_observation(
    structural: dict[str, object],
    operational: list[dict[str, object]],
    observation: Mapping[str, object],
    *,
    evidence_refs: Sequence[str],
) -> bool:
    allowed = {
        "capabilities",
        "execution_interfaces",
        "hooks",
        "operational_metrics",
    }
    if set(observation) != allowed:
        raise ModelPortraitError("MODEL_PORTRAIT_STRUCTURAL_OBSERVATION_INVALID")
    changed = False
    capabilities = [
        dict(row) for row in _mapping_rows(structural.get("capabilities"))
    ]
    by_capability = _by_identity(capabilities, "capability")
    for update in _mapping_rows(observation.get("capabilities")):
        identity = str(update.get("capability") or "")
        row = by_capability.get(identity)
        state = update.get("state")
        if row is None or state not in {"available", "unavailable"}:
            raise ModelPortraitError("MODEL_PORTRAIT_CAPABILITY_OBSERVATION_INVALID")
        if row.get("state") not in {"unknown", state}:
            raise ModelPortraitError("MODEL_PORTRAIT_CAPABILITY_OBSERVATION_CONFLICT")
        row["state"] = state
        row["evidence_count"] = int(row.get("evidence_count") or 0) + 1
        changed = True
    structural["capabilities"] = sorted(
        capabilities, key=lambda row: str(row["capability"])
    )

    interfaces = [
        dict(row) for row in _mapping_rows(structural.get("execution_interfaces"))
    ]
    by_interface = {
        (str(row["kind"]), str(row["contract_id"])): row for row in interfaces
    }
    for update in _mapping_rows(observation.get("execution_interfaces")):
        identity = (str(update.get("kind") or ""), str(update.get("contract_id") or ""))
        row = by_interface.get(identity)
        state = update.get("state")
        if row is None or state not in {"available", "unavailable"}:
            raise ModelPortraitError("MODEL_PORTRAIT_INTERFACE_OBSERVATION_INVALID")
        if row.get("state") not in {"unknown", state}:
            raise ModelPortraitError("MODEL_PORTRAIT_INTERFACE_OBSERVATION_CONFLICT")
        row["state"] = state
        changed = True
    structural["execution_interfaces"] = sorted(
        interfaces, key=lambda row: (str(row["kind"]), str(row["contract_id"]))
    )

    hooks = [dict(row) for row in _mapping_rows(structural.get("hooks"))]
    by_hook = _by_identity(hooks, "hook")
    for update in _mapping_rows(observation.get("hooks")):
        identity = str(update.get("hook") or "")
        row = by_hook.get(identity)
        state = update.get("state")
        if row is None or state not in {"available", "unavailable"}:
            raise ModelPortraitError("MODEL_PORTRAIT_HOOK_OBSERVATION_INVALID")
        if row.get("state") not in {"unknown", state}:
            raise ModelPortraitError("MODEL_PORTRAIT_HOOK_OBSERVATION_CONFLICT")
        row["state"] = state
        changed = True
    structural["hooks"] = sorted(hooks, key=lambda row: str(row["hook"]))

    by_metric = _by_identity(operational, "metric")
    for update in _mapping_rows(observation.get("operational_metrics")):
        identity = str(update.get("metric") or "")
        state = update.get("state")
        summary = update.get("summary")
        if (
            not identity
            or state not in {"observed", "unavailable"}
            or not isinstance(summary, str)
            or not summary.strip()
        ):
            raise ModelPortraitError("MODEL_PORTRAIT_OPERATIONAL_OBSERVATION_INVALID")
        row = by_metric.get(identity)
        if row is not None and row.get("state") not in {"unknown", state}:
            raise ModelPortraitError("MODEL_PORTRAIT_OPERATIONAL_OBSERVATION_CONFLICT")
        replacement = {
            "metric": identity,
            "state": state,
            "summary": summary.strip(),
            "evidence_refs": sorted(set(str(value) for value in evidence_refs)),
        }
        if row is None:
            operational.append(replacement)
            by_metric[identity] = replacement
        else:
            row.clear()
            row.update(replacement)
        changed = True
    return changed


def _portrait_coverage(
    structural: Mapping[str, object],
    behavioral: Sequence[Mapping[str, object]],
    operational: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    capabilities = _state_buckets(
        _mapping_rows(structural.get("capabilities")), "capability"
    )
    interfaces = _state_buckets(
        _mapping_rows(structural.get("execution_interfaces")), "kind"
    )
    hooks = _state_buckets(_mapping_rows(structural.get("hooks")), "hook")
    current_by_key: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    stale_ids = []
    for row in behavioral:
        if row.get("state") == "stale":
            stale_ids.append(str(row["fingerprint_id"]))
        else:
            current_by_key[str(row["coverage_key"])].append(row)
    conflicts = []
    for key, rows in sorted(current_by_key.items()):
        response_digests = {str(row["response_digest"]) for row in rows}
        if len(response_digests) > 1:
            conflicts.append(
                {
                    "coverage_key": key,
                    "fingerprint_ids": sorted(str(row["fingerprint_id"]) for row in rows),
                }
            )
    unknown_operational = sorted(
        str(row["metric"]) for row in operational if row.get("state") == "unknown"
    )
    return {
        "available_capabilities": capabilities["available"],
        "unknown_capabilities": capabilities["unknown"],
        "unavailable_capabilities": capabilities["unavailable"],
        "available_execution_interfaces": interfaces["available"],
        "unknown_execution_interfaces": interfaces["unknown"],
        "unavailable_execution_interfaces": interfaces["unavailable"],
        "available_hooks": hooks["available"],
        "unknown_hooks": hooks["unknown"],
        "unavailable_hooks": hooks["unavailable"],
        "observed_probe_keys": sorted(current_by_key),
        "stale_fingerprint_ids": sorted(stale_ids),
        "conflicts": conflicts,
        "unknown_operational_metrics": unknown_operational,
    }


def _state_buckets(
    rows: Sequence[Mapping[str, object]], identity: str
) -> dict[str, list[str]]:
    result = {"available": [], "unknown": [], "unavailable": []}
    for row in rows:
        state = str(row.get("state"))
        if state not in result:
            raise ModelPortraitError("MODEL_PORTRAIT_STRUCTURAL_STATE_INVALID")
        result[state].append(str(row[identity]))
    return {state: sorted(values) for state, values in result.items()}


def _normalized_operational_observations(
    values: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    rows = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, Mapping):
            raise ModelPortraitError("MODEL_PORTRAIT_OPERATIONAL_OBSERVATION_INVALID")
        metric = _required_text(
            value.get("metric"), "MODEL_PORTRAIT_OPERATIONAL_METRIC_INVALID"
        )
        if metric in seen:
            raise ModelPortraitError("MODEL_PORTRAIT_OPERATIONAL_METRIC_DUPLICATE")
        seen.add(metric)
        state = str(value.get("state"))
        if state not in {"observed", "unknown", "unavailable"}:
            raise ModelPortraitError("MODEL_PORTRAIT_OPERATIONAL_STATE_INVALID")
        refs = _validated_refs(
            value.get("evidence_refs", ()),
            "MODEL_PORTRAIT_OPERATIONAL_EVIDENCE_REF_INVALID",
        )
        if state == "observed" and not refs:
            raise ModelPortraitError("MODEL_PORTRAIT_OPERATIONAL_EVIDENCE_REQUIRED")
        row = {
            "metric": metric,
            "state": state,
            "summary": _required_text(
                value.get("summary"), "MODEL_PORTRAIT_OPERATIONAL_SUMMARY_INVALID"
            ),
            "evidence_refs": refs,
        }
        _reject_runtime(row)
        rows.append(row)
    return sorted(rows, key=lambda row: str(row["metric"]))


def _normalized_interface_requirements(
    values: Sequence[Mapping[str, object]],
) -> list[dict[str, str]]:
    rows = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, Mapping):
            raise ModelPortraitError("PORTRAIT_READINESS_INTERFACE_INVALID")
        kind = _required_text(
            value.get("kind"), "PORTRAIT_READINESS_INTERFACE_INVALID"
        )
        contract = _required_text(
            value.get("contract_id"), "PORTRAIT_READINESS_INTERFACE_INVALID"
        )
        if kind in seen:
            raise ModelPortraitError("PORTRAIT_READINESS_INTERFACE_DUPLICATE")
        seen.add(kind)
        rows.append({"kind": kind, "contract_id": contract})
    return sorted(rows, key=lambda row: row["kind"])


def _normalized_probe_requirements(
    values: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    rows = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, Mapping):
            raise ModelPortraitError("PORTRAIT_READINESS_PROBE_REQUIREMENT_INVALID")
        row: dict[str, object] = {
            "probe_protocol_id": _required_text(
                value.get("probe_protocol_id"),
                "PORTRAIT_READINESS_PROBE_REQUIREMENT_INVALID",
            ),
            "probe_protocol_version": _required_text(
                value.get("probe_protocol_version"),
                "PORTRAIT_READINESS_PROBE_REQUIREMENT_INVALID",
            ),
            "diagnostic_role": _required_text(
                value.get("diagnostic_role"),
                "PORTRAIT_READINESS_PROBE_REQUIREMENT_INVALID",
            ),
            "context_class": _required_text(
                value.get("context_class"),
                "PORTRAIT_READINESS_PROBE_REQUIREMENT_INVALID",
            ),
            "split": _required_text(
                value.get("split"), "PORTRAIT_READINESS_PROBE_REQUIREMENT_INVALID"
            ),
            "horizons": _integer_sequence(
                value.get("horizons"), "PORTRAIT_READINESS_PROBE_REQUIREMENT_INVALID"
            ),
            "dose_values": _number_sequence(
                value.get("dose_values"), "PORTRAIT_READINESS_PROBE_REQUIREMENT_INVALID"
            ),
            "minimum_replication_count": _positive_integer(
                value.get("minimum_replication_count"),
                "PORTRAIT_READINESS_PROBE_REQUIREMENT_INVALID",
            ),
        }
        row["coverage_key"] = probe_coverage_key(row)
        key = str(row["coverage_key"])
        if key in seen:
            raise ModelPortraitError("PORTRAIT_READINESS_PROBE_REQUIREMENT_DUPLICATE")
        seen.add(key)
        rows.append(row)
    return sorted(rows, key=lambda row: str(row["coverage_key"]))


def _evaluate_readiness(
    portrait: Mapping[str, object], requirements: Mapping[str, object]
) -> tuple[list[str], dict[str, object], str | None]:
    structural = _mapping(portrait, "structural_profile")
    capability_rows = _by_identity(
        _mapping_rows(structural.get("capabilities")), "capability"
    )
    interface_rows = _by_identity(
        _mapping_rows(structural.get("execution_interfaces")), "kind"
    )
    hook_rows = _by_identity(_mapping_rows(structural.get("hooks")), "hook")
    fingerprints = _mapping_rows(portrait.get("behavioral_fingerprints"))
    operational = _by_identity(
        _mapping_rows(portrait.get("operational_profile")), "metric"
    )
    portrait_coverage = _mapping(portrait, "coverage")

    blockers: list[str] = []
    if portrait_coverage.get("stale_fingerprint_ids"):
        blockers.extend(
            "PORTRAIT_STALE_FINGERPRINT:" + str(value)
            for value in portrait_coverage["stale_fingerprint_ids"]
        )
    if portrait_coverage.get("conflicts"):
        blockers.extend(
            "PORTRAIT_CONFLICTING_FINGERPRINT:" + str(row["coverage_key"])
            for row in _mapping_rows(portrait_coverage["conflicts"])
        )

    satisfied_capabilities = []
    for capability in requirements["capabilities"]:
        row = capability_rows.get(str(capability))
        if row is not None and row.get("state") == "available":
            satisfied_capabilities.append(str(capability))
        elif row is None or row.get("state") == "unknown":
            blockers.append("PORTRAIT_CAPABILITY_UNKNOWN:" + str(capability))
        else:
            blockers.append("PORTRAIT_CAPABILITY_UNAVAILABLE:" + str(capability))

    satisfied_interfaces = []
    for requirement in _mapping_rows(requirements["execution_interfaces"]):
        kind = str(requirement["kind"])
        row = interface_rows.get(kind)
        if (
            row is not None
            and row.get("state") == "available"
            and row.get("contract_id") == requirement.get("contract_id")
        ):
            satisfied_interfaces.append(kind)
        elif row is None or row.get("state") == "unknown":
            blockers.append("PORTRAIT_INTERFACE_UNKNOWN:" + kind)
        else:
            blockers.append("PORTRAIT_INTERFACE_UNAVAILABLE:" + kind)

    satisfied_hooks = []
    for hook in requirements["hooks"]:
        row = hook_rows.get(str(hook))
        if row is not None and row.get("state") == "available":
            satisfied_hooks.append(str(hook))
        elif row is None or row.get("state") == "unknown":
            blockers.append("PORTRAIT_HOOK_UNKNOWN:" + str(hook))
        else:
            blockers.append("PORTRAIT_HOOK_UNAVAILABLE:" + str(hook))

    current_fingerprints = [row for row in fingerprints if row.get("state") == "current"]
    by_coverage: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in current_fingerprints:
        by_coverage[str(row["coverage_key"])].append(row)
    satisfied_probe_keys = []
    satisfied_fingerprint_ids = []
    for requirement in _mapping_rows(requirements["probe_coverage"]):
        key = str(requirement["coverage_key"])
        matches = [
            row
            for row in by_coverage.get(key, [])
            if int(row["replication_count"])
            >= int(requirement["minimum_replication_count"])
        ]
        if matches:
            satisfied_probe_keys.append(key)
            satisfied_fingerprint_ids.extend(str(row["fingerprint_id"]) for row in matches)
        else:
            blockers.append("PORTRAIT_PROBE_COVERAGE_MISSING:" + key)

    satisfied_operational = []
    for metric in requirements["operational_metrics"]:
        row = operational.get(str(metric))
        if row is not None and row.get("state") == "observed":
            satisfied_operational.append(str(metric))
        else:
            blockers.append("PORTRAIT_OPERATIONAL_UNKNOWN:" + str(metric))

    evaluator_binding = _evaluator_binding(_mapping(structural, "evaluator"))
    if requirements["evaluator_required"] and evaluator_binding is None:
        blockers.append("PORTRAIT_EVALUATOR_BINDING_REQUIRED")

    coverage = {
        "satisfied_capabilities": sorted(satisfied_capabilities),
        "satisfied_execution_interfaces": sorted(satisfied_interfaces),
        "satisfied_hooks": sorted(satisfied_hooks),
        "satisfied_probe_keys": sorted(satisfied_probe_keys),
        "satisfied_fingerprint_ids": sorted(set(satisfied_fingerprint_ids)),
        "satisfied_operational_metrics": sorted(satisfied_operational),
    }
    return sorted(set(blockers)), coverage, evaluator_binding


def _readiness_state(blockers: Sequence[str]) -> str:
    if not blockers:
        return "ready_for_gap_planning"
    classes = set()
    for blocker in blockers:
        if blocker.startswith("PORTRAIT_STALE_"):
            classes.add("stale_portrait")
        elif blocker.startswith("PORTRAIT_CONFLICTING_"):
            classes.add("conflicting_evidence")
        elif blocker.startswith("PORTRAIT_EVALUATOR_"):
            classes.add("requires_evaluator_binding")
        elif blocker.startswith(("PORTRAIT_CAPABILITY_UNKNOWN", "PORTRAIT_INTERFACE_UNKNOWN", "PORTRAIT_HOOK_UNKNOWN")):
            classes.add("requires_static_onboarding")
        elif blocker.startswith(("PORTRAIT_CAPABILITY_UNAVAILABLE", "PORTRAIT_INTERFACE_UNAVAILABLE", "PORTRAIT_HOOK_UNAVAILABLE")):
            classes.add("requires_interface_extension")
        elif blocker.startswith(("PORTRAIT_PROBE_", "PORTRAIT_OPERATIONAL_")):
            classes.add("requires_probe_coverage")
        else:
            raise ModelPortraitError("PORTRAIT_READINESS_BLOCKER_UNKNOWN:" + blocker)
    for state in _READINESS_PRIORITY:
        if state in classes:
            return state
    raise ModelPortraitError("PORTRAIT_READINESS_STATE_UNRESOLVED")


def _evaluator_binding(evaluator: Mapping[str, object]) -> str | None:
    if evaluator.get("state") != "ready":
        return None
    digest = evaluator.get("contract_digest")
    if isinstance(digest, str) and is_content_addressed(digest):
        return digest
    if isinstance(digest, str) and _HEX_DIGEST.fullmatch(digest):
        return "sha256:" + digest
    return None


def _sorted_mapping_rows(value: object, identity: str) -> list[dict[str, object]]:
    return sorted(
        (dict(row) for row in _mapping_rows(value)),
        key=lambda row: str(row[identity]),
    )


def _by_identity(
    rows: Sequence[Mapping[str, object]], identity: str
) -> dict[str, Mapping[str, object]]:
    return {str(row[identity]): row for row in rows}


def _require_unique(document: Mapping[str, object], field: str, identity: str) -> None:
    rows = _mapping_rows(document.get(field))
    values = [str(row.get(identity)) for row in rows]
    if len(values) != len(set(values)):
        raise ModelPortraitError("MODEL_PORTRAIT_DUPLICATE:" + field)


def _mapping(value: Mapping[str, object], name: str) -> Mapping[str, Any]:
    result = value.get(name)
    if not isinstance(result, Mapping):
        raise ModelPortraitError("MODEL_PORTRAIT_MAPPING_REQUIRED:" + name)
    return result


def _mapping_rows(value: object) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(row, Mapping) for row in value):
        raise ModelPortraitError("MODEL_PORTRAIT_MAPPING_ROWS_INVALID")
    return list(value)


def _unique_texts(values: Sequence[str], code: str) -> list[str]:
    normalized = [_required_text(value, code) for value in values]
    if len(normalized) != len(set(normalized)):
        raise ModelPortraitError(code + ":DUPLICATE")
    return sorted(normalized)


def _required_text(value: object, code: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ModelPortraitError(code)
    text = value.strip()
    _reject_runtime(text)
    return text


def _integer_sequence(value: object, code: str) -> list[int]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or not value
        or any(not isinstance(item, int) or isinstance(item, bool) or item < 1 for item in value)
    ):
        raise ModelPortraitError(code)
    return sorted(set(int(item) for item in value))


def _number_sequence(value: object, code: str) -> list[float]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or not value
        or any(not isinstance(item, (int, float)) or isinstance(item, bool) for item in value)
    ):
        raise ModelPortraitError(code)
    return sorted(set(float(item) for item in value))


def _positive_integer(value: object, code: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ModelPortraitError(code)
    return value


def _validated_refs(value: object, code: str) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ModelPortraitError(code)
    refs = [str(item) for item in value]
    if any(not isinstance(item, str) or not is_content_addressed(item) for item in value):
        raise ModelPortraitError(code)
    return sorted(set(refs))


def _validate_refs(value: object, code: str) -> None:
    _validated_refs(value, code)


def _reject_runtime(value: object) -> None:
    try:
        reject_runtime_bindings(value)
    except GeometryValidationError as exc:
        raise ModelPortraitError(f"MODEL_PORTRAIT_RUNTIME_BINDING_FORBIDDEN:{exc}") from exc


def _stable_id(prefix: str, body: Mapping[str, object]) -> str:
    return prefix + "-" + _canonical_digest(body)[:24]


def _canonical_digest(value: object) -> str:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ModelPortraitError("MODEL_PORTRAIT_CANONICALIZATION_INVALID") from exc
    return hashlib.sha256(payload).hexdigest()
