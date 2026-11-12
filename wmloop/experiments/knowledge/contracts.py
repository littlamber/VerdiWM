"""Portable knowledge graph components: contracts."""

from __future__ import annotations

from collections.abc import Mapping

from wmloop.experiments.knowledge.common import _is_cas_sha256
from wmloop.experiments.knowledge.graph import _PortableKnowledgeGraph
from wmloop.experiments.knowledge.types import PortableKnowledgeGraphError
from wmloop.experiments.knowledge.validation import _composition_module_node


def _project_module_composition(
    graph: _PortableKnowledgeGraph, document: Mapping[str, object]
) -> None:
    root = graph.node(
        "module_composition",
        str(document["composition_id"]),
        requested_capabilities=document["requested_capabilities"],
        effective_authority_level=document["effective_authority_level"],
    )
    for row in document["selected_modules"]:
        module = graph.node(
            "module_manifest",
            f"{row['module_id']}:{row['abi_id']}",
            module_id=row["module_id"],
            abi_id=row["abi_id"],
            abi_version=row["abi_version"],
            implementation_ref="sha256:" + str(row["abi_digest"]),
            provides=row["provides"],
            requires=row["requires"],
            authority_level=row["authority_level"],
            side_effect_class=row["side_effect_class"],
            license_spdx_id=row["license_spdx_id"],
            redistributable_content=True,
        )
        graph.edge(root, "contains_module", module)
        for capability in row["provides"]:
            graph.edge(
                module,
                "provides_capability",
                graph.node("capability", str(capability)),
            )
        for capability in row["requires"]:
            graph.edge(
                module,
                "requires_capability",
                graph.node("capability", str(capability)),
            )
    for edge in document["dependency_edges"]:
        consumer = _composition_module_node(
            graph, document, str(edge["consumer_module_id"])
        )
        provider = _composition_module_node(
            graph, document, str(edge["provider_module_id"])
        )
        graph.edge(
            consumer,
            "requires_module",
            provider,
            capability=edge["required_capability"],
            contract=edge["contract"],
        )


def _project_evidence_ir(
    graph: _PortableKnowledgeGraph, document: Mapping[str, object]
) -> None:
    context = document["context"]
    status = document["status"]
    authority = document["authority"]
    state = str(status["state"])
    community = state == "community_promoted"
    if community and (
        authority["claim_scope"] != "community"
        or not any(_is_cas_sha256(value) for value in document["evidence_refs"])
    ):
        raise PortableKnowledgeGraphError(
            "PORTABLE_KNOWLEDGE_PROMOTED_EVIDENCE_INVALID"
        )
    root = graph.node(
        "evidence_record",
        str(document["evidence_id"]),
        model_family=context["model_family"],
        capabilities=[context["capability_class"]],
        goal_protocol=context["goal_protocol"],
        outcome_protocol=context["outcome_protocol"],
        data_regime=context["data_regime"],
        horizons=context["horizons"],
        outcome_label=document["outcome"]["label"],
        status=state,
        claim_scope=authority["claim_scope"],
    )
    primitive = graph.node("primitive", str(document["intervention"]["primitive_id"]))
    graph.edge(
        root,
        "evaluates_intervention",
        primitive,
        evidence=document["evidence_refs"],
        claim_scope=authority["claim_scope"],
        frozen=community,
    )
    graph.edge(
        root,
        "observed_on_model_family",
        graph.node("model_family", str(context["model_family"])),
    )
    boundary_kind = (
        "negative_boundary"
        if document["outcome"]["label"] in {"harmful", "null"}
        else "applicability_boundary"
    )
    boundary = graph.node(
        boundary_kind,
        str(document["evidence_id"]),
        applicability=document["validity_region"],
        outcome_label=document["outcome"]["label"],
    )
    graph.edge(
        root,
        "establishes_boundary",
        boundary,
        evidence=document["evidence_refs"],
        claim_scope=authority["claim_scope"],
        frozen=community,
    )


def _project_protocol_contract(
    graph: _PortableKnowledgeGraph, document: Mapping[str, object]
) -> None:
    key = f"{document['protocol_kind']}:{document['protocol_id']}:{document['protocol_version']}"
    root = graph.node(
        "protocol_contract",
        key,
        contract_id=document["contract_id"],
        protocol_kind=document["protocol_kind"],
        protocol_id=document["protocol_id"],
        protocol_version=document["protocol_version"],
        semantic_dimensions=document["semantic_dimensions"],
        protected_fields=document["protected_fields"],
        contract_ref=document["contract_ref"],
        license_spdx_id=document["license_spdx_id"],
        redistributable_content=True,
    )
    for dimension in document["semantic_dimensions"]:
        graph.edge(
            root,
            "measures_semantic_dimension",
            graph.node("semantic_dimension", str(dimension)),
        )
    for reference in document["evidence_refs"]:
        graph.edge(
            root,
            "cites_evidence",
            graph.node("evidence", str(reference)),
            evidence=str(reference),
        )


def _project_transformation_contract(
    graph: _PortableKnowledgeGraph, document: Mapping[str, object]
) -> None:
    root = graph.node(
        "transformation_contract",
        str(document["transform_id"]),
        source_semantics=document["source_semantics"],
        target_semantics=document["target_semantics"],
        invariants=document["invariants"],
        loss_policy=document["loss_policy"],
        implementation_ref=document["implementation_ref"],
        verification_ref=document["verification_ref"],
        license_spdx_id=document["license_spdx_id"],
        redistributable_content=True,
    )
    for value in document["source_semantics"]:
        graph.edge(
            root,
            "maps_from",
            graph.node("data_semantics", str(value)),
        )
    for value in document["target_semantics"]:
        graph.edge(
            root,
            "maps_to",
            graph.node("data_semantics", str(value)),
        )
    for invariant in document["invariants"]:
        graph.edge(
            root,
            "preserves_invariant",
            graph.node("semantic_invariant", str(invariant)),
        )
    for reference in document["evidence_refs"]:
        graph.edge(
            root,
            "cites_evidence",
            graph.node("evidence", str(reference)),
            evidence=str(reference),
        )


def _project_knowledge_lifecycle(
    graph: _PortableKnowledgeGraph, document: Mapping[str, object]
) -> None:
    root = graph.node(
        "knowledge_lifecycle",
        str(document["lifecycle_id"]),
        action=document["action"],
        reason=document["reason"],
    )
    subject = document["subject"]
    subject_node = graph.node(str(subject["kind"]), str(subject["id"]))
    action = str(document["action"])
    relation = {"revocation": "revokes", "deprecation": "deprecates"}.get(action)
    if relation is not None:
        graph.edge(
            root,
            relation,
            subject_node,
            evidence=document["evidence_refs"],
            claim_scope="community",
            frozen=True,
        )
    else:
        replacement = document["replacement"]
        replacement_node = graph.node(str(replacement["kind"]), str(replacement["id"]))
        graph.edge(root, "records_supersession", replacement_node)
        graph.edge(
            replacement_node,
            "supersedes",
            subject_node,
            evidence=document["evidence_refs"],
            claim_scope="community",
            frozen=True,
        )
