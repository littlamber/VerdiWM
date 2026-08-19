"""Open method representations and isolated candidate overlay contracts.

The representation is intentionally mechanism-oriented rather than a registry
of known paper families. It lets an LLM describe a novel method while keeping
execution, evaluator, and promotion authority outside the generated artifact.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath

from wmloop.contracts import ContractValidationError, validate_document


class OpenMethodIRError(ValueError):
    """An open method or overlay crossed a contract boundary."""


_SHA256 = re.compile(r"^(?:sha256:)?[0-9a-f]{64}$")


def method_ir_digest(document: Mapping[str, object]) -> str:
    return _digest(document, excluded="method_id")


def overlay_digest(document: Mapping[str, object]) -> str:
    return _digest(document, excluded="overlay_id")


def build_method_ir(
    *,
    source_evidence: Sequence[Mapping[str, object]],
    mechanism: Mapping[str, object],
    target_mapping: Mapping[str, object],
    training: Mapping[str, object],
    falsification: Mapping[str, object],
    source_evidence_digest: str | None = None,
    target_portrait_binding: Mapping[str, object] | None = None,
    probe_binding: Mapping[str, object] | None = None,
    interface_extension_refs: Sequence[str] = (),
    training_resource_binding: Mapping[str, object] | None = None,
    state: str = "draft",
    claim_boundary: str = "This Method IR is a research proposal, not an execution or promotion decision.",
) -> dict[str, object]:
    body: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "verdiwm-method-ir",
        "method_id": "",
        "source_evidence": [dict(row) for row in source_evidence],
        "mechanism": dict(mechanism),
        "target_mapping": dict(target_mapping),
        "training": dict(training),
        "falsification": dict(falsification),
        "source_evidence_digest": source_evidence_digest or _evidence_digest(source_evidence),
        "interface_extension_refs": list(interface_extension_refs),
        "state": state,
        "claim_boundary": claim_boundary,
    }
    if target_portrait_binding:
        body["target_portrait_binding"] = dict(target_portrait_binding)
    if probe_binding:
        body["probe_binding"] = dict(probe_binding)
    if training_resource_binding:
        body["training_resource_binding"] = dict(training_resource_binding)
    body["method_id"] = "method-ir-" + method_ir_digest(body)[:24]
    validate_method_ir(body)
    return body


def validate_method_ir(document: Mapping[str, object], *, root: Path | None = None) -> None:
    try:
        validate_document("method_ir", document, root=root)
    except ContractValidationError as exc:
        raise OpenMethodIRError(f"METHOD_IR_SCHEMA_INVALID:{exc}") from exc
    method_id = document.get("method_id")
    if method_id != "method-ir-" + method_ir_digest(document)[:24]:
        raise OpenMethodIRError("METHOD_IR_DIGEST_MISMATCH")
    _validate_digests(document.get("source_evidence"))
    evidence_digest = document.get("source_evidence_digest")
    if evidence_digest is not None and evidence_digest != _evidence_digest(document["source_evidence"]):
        raise OpenMethodIRError("METHOD_IR_SOURCE_EVIDENCE_DIGEST_MISMATCH")
    mapping = document["target_mapping"]
    if mapping["mapping_state"] == "direct_candidate" and mapping["missing_capabilities"]:
        raise OpenMethodIRError("METHOD_IR_DIRECT_MAPPING_HAS_MISSING_CAPABILITIES")
    if document["training"]["mode"] == "training" and not document["training"]["trainable_scope"]:
        raise OpenMethodIRError("METHOD_IR_TRAINING_SCOPE_REQUIRED")
    refs = document.get("interface_extension_refs", [])
    if not isinstance(refs, list) or len(refs) != len(set(str(value) for value in refs)):
        raise OpenMethodIRError("METHOD_IR_INTERFACE_EXTENSION_REFS_INVALID")
    probe = document.get("probe_binding")
    if probe is not None:
        if not isinstance(probe, Mapping):
            raise OpenMethodIRError("METHOD_IR_PROBE_BINDING_INVALID")
        expected = _binding_digest(probe, excluded="binding_digest")
        if probe.get("binding_digest") not in {expected, "sha256:" + expected}:
            raise OpenMethodIRError("METHOD_IR_PROBE_BINDING_DIGEST_MISMATCH")


def build_candidate_overlay(
    *,
    method_ir: Mapping[str, object],
    base_revision: Mapping[str, object],
    files: Sequence[Mapping[str, object]],
    tests: Sequence[Mapping[str, object]],
    execution_contract_binding: Mapping[str, object] | None = None,
    target_portrait_binding: Mapping[str, object] | None = None,
    probe_binding: Mapping[str, object] | None = None,
    interface_extension_refs: Sequence[str] = (),
    training_resource_binding: Mapping[str, object] | None = None,
    state: str = "proposed",
    claim_boundary: str = "This overlay grants no evaluator, metric, GPU, or promotion authority.",
) -> dict[str, object]:
    validate_method_ir(method_ir)
    body: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "verdiwm-candidate-overlay",
        "overlay_id": "",
        "method_id": method_ir["method_id"],
        "method_ir_digest": method_ir_digest(method_ir),
        "base_revision": dict(base_revision),
        "files": [dict(row) for row in files],
        "tests": [dict(row) for row in tests],
        "interface_extension_refs": list(
            interface_extension_refs or method_ir.get("interface_extension_refs", [])
        ),
        "authority": {
            "source_mutation_allowed": False,
            "evaluator_mutation_allowed": False,
            "metric_mutation_allowed": False,
            "gpu_authority": False,
        },
        "state": state,
        "claim_boundary": claim_boundary,
    }
    portrait = target_portrait_binding if target_portrait_binding is not None else method_ir.get(
        "target_portrait_binding"
    )
    probe = probe_binding if probe_binding is not None else method_ir.get("probe_binding")
    resource = (
        training_resource_binding
        if training_resource_binding is not None
        else method_ir.get("training_resource_binding")
    )
    if portrait is not None:
        body["target_portrait_binding"] = dict(portrait)
    if execution_contract_binding is not None:
        body["execution_contract_binding"] = dict(execution_contract_binding)
    if probe is not None:
        body["probe_binding"] = dict(probe)
    if resource is not None:
        body["training_resource_binding"] = dict(resource)
    body["overlay_id"] = "overlay-" + overlay_digest(body)[:24]
    validate_candidate_overlay(body)
    return body


def validate_candidate_overlay(document: Mapping[str, object], *, root: Path | None = None) -> None:
    try:
        validate_document("candidate_overlay", document, root=root)
    except ContractValidationError as exc:
        raise OpenMethodIRError(f"CANDIDATE_OVERLAY_SCHEMA_INVALID:{exc}") from exc
    if document.get("overlay_id") != "overlay-" + overlay_digest(document)[:24]:
        raise OpenMethodIRError("CANDIDATE_OVERLAY_DIGEST_MISMATCH")
    for row in document["files"]:
        relative = str(row["relative_path"])
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts or str(pure) != relative:
            raise OpenMethodIRError("CANDIDATE_OVERLAY_PATH_INVALID")
        if not _SHA256.fullmatch(str(row["sha256"])):
            raise OpenMethodIRError("CANDIDATE_OVERLAY_FILE_DIGEST_INVALID")
    method_digest = document.get("method_ir_digest")
    # The full Method IR is validated by the compiler before this contract is
    # built; the overlay keeps its digest so later stages can verify the binding.
    if method_digest is not None and not _SHA256.fullmatch(str(method_digest)):
        raise OpenMethodIRError("CANDIDATE_OVERLAY_METHOD_DIGEST_INVALID")
    if any(row["state"] == "passed" for row in document["tests"]) and document["state"] == "proposed":
        raise OpenMethodIRError("CANDIDATE_OVERLAY_STATE_INCONSISTENT")


def _validate_digests(value: object) -> None:
    if not isinstance(value, list):
        raise OpenMethodIRError("METHOD_IR_SOURCE_EVIDENCE_INVALID")
    for row in value:
        if not isinstance(row, Mapping) or not _SHA256.fullmatch(str(row["source_digest"])):
            raise OpenMethodIRError("METHOD_IR_SOURCE_DIGEST_INVALID")


def _evidence_digest(value: Sequence[Mapping[str, object]]) -> str:
    return hashlib.sha256(
        json.dumps([dict(row) for row in value], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _binding_digest(value: Mapping[str, object], *, excluded: str) -> str:
    return hashlib.sha256(
        json.dumps(
            {key: item for key, item in value.items() if key != excluded},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _digest(document: Mapping[str, object], *, excluded: str) -> str:
    body = {key: value for key, value in document.items() if key != excluded}
    encoded = json.dumps(body, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
