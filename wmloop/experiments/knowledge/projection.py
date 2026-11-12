"""Portable knowledge graph components: projection."""

from __future__ import annotations

from collections.abc import Mapping

from wmloop.control.interface_extension import (
    MethodInterfaceExtensionError,
    validate_method_interface_extension,
)
from wmloop.control.intermediate_ir import (
    IntermediateRepresentationError,
    validate_model_capability_ir,
)
from wmloop.control.model_portrait import ModelPortraitError, validate_model_portrait
from wmloop.control.open_method_ir import OpenMethodIRError, validate_method_ir
from wmloop.experiments.knowledge.contracts import (
    _project_evidence_ir,
    _project_knowledge_lifecycle,
    _project_module_composition,
    _project_protocol_contract,
    _project_transformation_contract,
)
from wmloop.experiments.knowledge.graph import _PortableKnowledgeGraph
from wmloop.experiments.knowledge.methods import (
    _project_method_interface_extension,
    _project_method_ir,
    _project_model_capability,
)
from wmloop.experiments.knowledge.models import (
    _project_model_irg,
    _project_model_portrait,
    _project_portrait_transition,
)
from wmloop.experiments.knowledge.transfer import (
    _project_embodiment,
    _project_fingerprint,
    _project_mechanism,
    _project_mechanism_relation,
    _project_portable_experience,
    _project_transfer_boundary,
)
from wmloop.experiments.knowledge.types import PortableKnowledgeGraphError
from wmloop.experiments.knowledge.validation import (
    _validate_community_document,
    _validate_module_composition_receipt,
)
from wmloop.geometry.community_knowledge import (
    validate_knowledge_lifecycle_record,
    validate_portrait_transition,
    validate_protocol_contract,
    validate_transformation_contract,
)
from wmloop.geometry.evidence_ir import validate_evidence_ir
from wmloop.geometry.mechanism_relations import validate_mechanism_relation
from wmloop.geometry.model_irg import ModelIRGError, validate_model_irg
from wmloop.geometry.portable_experience import validate_portable_experience
from wmloop.geometry.portable_transfer_knowledge import (
    validate_mechanism_contract,
    validate_method_embodiment,
    validate_probe_fingerprint_summary,
    validate_transfer_boundary,
)
from wmloop.geometry.settled_evidence import validate_settled_evidence
from wmloop.geometry.types import GeometryValidationError


def _project_document(
    graph: _PortableKnowledgeGraph, document: Mapping[str, object]
) -> None:
    artifact = document.get("artifact_type")
    if artifact == "verdiwm-settled-evidence":
        validate_settled_evidence(document)
        identifier = graph.node(
            "settled_evidence",
            str(document["evidence_id"]),
            decision=document["decision"],
            evidence_scope=document["evidence_scope"],
            claim_scope="settlement_only",
        )
        for name in ("receipt_ref", "verdict_ref", "failure_context_ref"):
            reference = str(document[name])
            graph.edge(
                identifier,
                name.removesuffix("_ref"),
                graph.node("evidence", reference),
                evidence=reference,
            )
        graph.document_count += 1
        return
    if artifact == "verdiwm-model-capability-ir":
        try:
            validate_model_capability_ir(document)
        except IntermediateRepresentationError as exc:
            raise PortableKnowledgeGraphError(
                f"PORTABLE_KNOWLEDGE_MODEL_CAPABILITY_INVALID:{exc}"
            ) from exc
        _project_model_capability(graph, document)
    elif artifact == "verdiwm-model-portrait":
        try:
            validate_model_portrait(document)
        except ModelPortraitError as exc:
            raise PortableKnowledgeGraphError(
                f"PORTABLE_KNOWLEDGE_MODEL_PORTRAIT_INVALID:{exc}"
            ) from exc
        _project_model_portrait(graph, document)
    elif artifact == "verdiwm-model-irg":
        try:
            validate_model_irg(document)
        except ModelIRGError as exc:
            raise PortableKnowledgeGraphError(
                f"PORTABLE_KNOWLEDGE_MODEL_IRG_INVALID:{exc}"
            ) from exc
        _project_model_irg(graph, document)
    elif artifact == "verdiwm-portrait-transition":
        _validate_community_document(validate_portrait_transition, document)
        _project_portrait_transition(graph, document)
    elif artifact == "verdiwm-module-composition-receipt":
        _validate_module_composition_receipt(document)
        _project_module_composition(graph, document)
    elif artifact == "verdiwm-evidence-ir":
        try:
            validate_evidence_ir(document)
        except GeometryValidationError as exc:
            raise PortableKnowledgeGraphError(
                f"PORTABLE_KNOWLEDGE_EVIDENCE_INVALID:{exc}"
            ) from exc
        _project_evidence_ir(graph, document)
    elif artifact == "verdiwm-protocol-contract":
        _validate_community_document(validate_protocol_contract, document)
        _project_protocol_contract(graph, document)
    elif artifact == "verdiwm-transformation-contract":
        _validate_community_document(validate_transformation_contract, document)
        _project_transformation_contract(graph, document)
    elif artifact == "verdiwm-knowledge-lifecycle":
        _validate_community_document(validate_knowledge_lifecycle_record, document)
        _project_knowledge_lifecycle(graph, document)
    elif artifact == "verdiwm-transferable-experience":
        validate_portable_experience(document)
        _project_portable_experience(graph, document)
    elif artifact == "verdiwm-mechanism-contract":
        validate_mechanism_contract(document)
        _project_mechanism(graph, document)
    elif artifact == "verdiwm-mechanism-relation":
        validate_mechanism_relation(document)
        _project_mechanism_relation(graph, document)
    elif artifact == "verdiwm-method-embodiment":
        validate_method_embodiment(document)
        _project_embodiment(graph, document)
    elif artifact == "verdiwm-probe-fingerprint-summary":
        validate_probe_fingerprint_summary(document)
        _project_fingerprint(graph, document)
    elif artifact == "verdiwm-transfer-boundary":
        validate_transfer_boundary(document)
        _project_transfer_boundary(graph, document)
    elif artifact == "verdiwm-method-ir":
        try:
            validate_method_ir(document)
        except OpenMethodIRError as exc:
            raise PortableKnowledgeGraphError(
                f"PORTABLE_KNOWLEDGE_METHOD_IR_INVALID:{exc}"
            ) from exc
        _project_method_ir(graph, document)
    elif artifact == "verdiwm-method-interface-extension":
        try:
            validate_method_interface_extension(document)
        except MethodInterfaceExtensionError as exc:
            raise PortableKnowledgeGraphError(
                f"PORTABLE_KNOWLEDGE_METHOD_INTERFACE_INVALID:{exc}"
            ) from exc
        _project_method_interface_extension(graph, document)
    else:
        raise PortableKnowledgeGraphError("PORTABLE_KNOWLEDGE_DOCUMENT_UNSUPPORTED")
    graph.document_count += 1
