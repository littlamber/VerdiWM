"""Portable knowledge graph components: transfer."""

from __future__ import annotations

from collections.abc import Mapping

from wmloop.experiments.knowledge.common import _digest
from wmloop.experiments.knowledge.graph import _PortableKnowledgeGraph


def _project_portable_experience(
    graph: _PortableKnowledgeGraph, document: Mapping[str, object]
) -> None:
    identity = str(document.get("portable_experience_id") or _digest(document))
    root = graph.node("portable_experience", identity)
    knowledge = document["portable_knowledge"]
    assert isinstance(knowledge, Mapping)
    for field, kind, relation in (
        ("model_family", "model_family", "observed_on_model_family"),
        ("capability_class", "capability_class", "requires_capability_class"),
        ("goal_protocol", "goal_protocol", "evaluated_under_goal"),
        ("outcome_protocol", "outcome_protocol", "measured_by_outcome"),
        ("dataset_regime", "dataset_regime", "observed_in_data_regime"),
        ("primitive", "primitive", "tests_primitive"),
    ):
        value = knowledge[field]
        child = graph.node(kind, str(value))
        graph.edge(root, relation, child)
    for anti_condition in document["anti_conditions"]:
        child = graph.node("anti_condition", str(anti_condition))
        graph.edge(root, "bounded_by", child)
    for reference in document["evidence_refs"]:
        child = graph.node("evidence", str(reference))
        graph.edge(root, "cites_evidence", child, evidence=str(reference))


def _project_mechanism(
    graph: _PortableKnowledgeGraph, document: Mapping[str, object]
) -> None:
    root = graph.node(
        "mechanism",
        str(document["mechanism_id"]),
        causal_claim=document["causal_claim"],
        intervention_semantics=document["intervention_semantics"],
    )
    for capability in document["required_capabilities"]:
        graph.edge(
            root, "requires_capability", graph.node("capability", str(capability))
        )
    for capability in document["optional_capabilities"]:
        graph.edge(
            root,
            "optionally_uses_capability",
            graph.node("capability", str(capability)),
        )
    for interface in document["target_interface_requirements"]:
        graph.edge(
            root, "requires_interface", graph.node("interface_contract", str(interface))
        )
    for condition in document["known_anti_conditions"]:
        graph.edge(root, "bounded_by", graph.node("anti_condition", str(condition)))
    for reference in document["source_evidence_refs"]:
        evidence = graph.node("evidence", str(reference))
        graph.edge(root, "derived_from_evidence", evidence, evidence=str(reference))


def _project_mechanism_relation(
    graph: _PortableKnowledgeGraph, document: Mapping[str, object]
) -> None:
    """Project a pairwise claim while retaining its evidence and boundary."""

    root = graph.node(
        "mechanism_relation",
        str(document["relation_id"]),
        relation_type=document["relation_type"],
        composition_operator=document["composition_operator"],
        interaction_effect=document["interaction_effect"],
        uncertainty=document["uncertainty"],
        verification_state=document["verification_state"],
        claim_scope=document["claim_scope"],
    )
    graph.edge(
        root,
        "relates_source",
        graph.node("mechanism", str(document["source_mechanism_id"])),
    )
    graph.edge(
        root,
        "relates_target",
        graph.node("mechanism", str(document["target_mechanism_id"])),
    )
    for condition in document["condition_set"]:
        graph.edge(root, "requires_condition", graph.node("condition", str(condition)))
    for anti_condition in document["anti_conditions"]:
        graph.edge(
            root, "bounded_by", graph.node("anti_condition", str(anti_condition))
        )
    for ablation in document["required_ablations"]:
        graph.edge(root, "requires_ablation", graph.node("ablation", str(ablation)))
    for reference in document["evidence_refs"]:
        evidence = graph.node("evidence", str(reference))
        graph.edge(root, "cites_evidence", evidence, evidence=str(reference))


def _project_embodiment(
    graph: _PortableKnowledgeGraph, document: Mapping[str, object]
) -> None:
    root = graph.node(
        "embodiment",
        str(document["embodiment_id"]),
        materialization_class=document["materialization_class"],
        implementation_state=document["implementation_state"],
        implementation_revision=document["implementation_revision"],
    )
    graph.edge(
        root,
        "embodies_mechanism",
        graph.node("mechanism", str(document["mechanism_id"])),
    )
    executable = document.get("executable_binding")
    if isinstance(executable, Mapping) and isinstance(executable.get("primitive"), str):
        graph.edge(
            root,
            "executes_primitive",
            graph.node("primitive", str(executable["primitive"])),
            implementation_revision=str(
                executable.get("implementation_revision") or ""
            ),
        )
    for interface in document["interface_contracts"]:
        graph.edge(
            root, "uses_interface", graph.node("interface_contract", str(interface))
        )
    for reference in document["evidence_refs"]:
        evidence = graph.node("evidence", str(reference))
        graph.edge(root, "cites_evidence", evidence, evidence=str(reference))


def _project_fingerprint(
    graph: _PortableKnowledgeGraph, document: Mapping[str, object]
) -> None:
    root = graph.node(
        "probe_fingerprint",
        str(document["fingerprint_id"]),
        model_capability_id=document["model_capability_id"],
        model_family=document["model_family"],
        diagnostic_role=document["diagnostic_role"],
        context_class=document["context_class"],
        split=document["split"],
        horizons=document["horizons"],
        dose_values=document["dose_values"],
        replication_count=document["replication_count"],
        response_digest=document["response_digest"],
        uncertainty_summary=document["uncertainty_summary"],
    )
    graph.edge(
        root,
        "profiles_model_capability",
        graph.node("model_capability", str(document["model_capability_id"])),
    )
    graph.edge(
        root,
        "profiles_model_family",
        graph.node("model_family", str(document["model_family"])),
    )
    protocol = graph.node(
        "protocol_contract",
        f"probe:{document['probe_protocol_id']}:{document['probe_protocol_version']}",
        protocol_kind="probe",
        protocol_id=document["probe_protocol_id"],
        protocol_version=document["probe_protocol_version"],
    )
    graph.edge(root, "uses_probe_protocol", protocol)
    graph.edge(root, "measured_by", protocol)
    response = graph.node("response_artifact", str(document["response_digest"]))
    graph.edge(
        root, "summarizes_response", response, evidence=str(document["response_digest"])
    )
    for reference in document["evidence_refs"]:
        evidence = graph.node("evidence", str(reference))
        graph.edge(root, "cites_evidence", evidence, evidence=str(reference))


def _project_transfer_boundary(
    graph: _PortableKnowledgeGraph, document: Mapping[str, object]
) -> None:
    root = graph.node(
        "transfer_boundary",
        str(document["boundary_id"]),
        outcome_state=document["outcome_state"],
        claim_scope=document["claim_scope"],
        boundary_statement=document["boundary_statement"],
        capabilities=document["required_capabilities"],
        anti_conditions=document["anti_conditions"],
    )
    semantic_kind = (
        "negative_boundary"
        if document["outcome_state"] == "verified_negative_boundary"
        else "applicability_boundary"
    )
    semantic = graph.node(
        semantic_kind,
        str(document["boundary_id"]),
        outcome_state=document["outcome_state"],
        claim_scope=document["claim_scope"],
        capabilities=document["required_capabilities"],
        anti_conditions=document["anti_conditions"],
    )
    graph.edge(root, "projects_boundary_as", semantic)
    graph.edge(
        root,
        "constrains_mechanism",
        graph.node("mechanism", str(document["mechanism_id"])),
    )
    graph.edge(
        root,
        "applies_to_embodiment",
        graph.node("embodiment", str(document["embodiment_id"])),
    )
    graph.edge(
        root,
        "compares_source_fingerprint",
        graph.node("probe_fingerprint", str(document["source_fingerprint_id"])),
    )
    graph.edge(
        root,
        "compares_target_fingerprint",
        graph.node("probe_fingerprint", str(document["target_fingerprint_id"])),
    )
    graph.edge(
        root,
        "targets_model_capability",
        graph.node("model_capability", str(document["target_model_capability_id"])),
    )
    for capability in document["required_capabilities"]:
        graph.edge(
            root, "requires_capability", graph.node("capability", str(capability))
        )
    for condition in document["anti_conditions"]:
        graph.edge(root, "bounded_by", graph.node("anti_condition", str(condition)))
    for field, relation in (
        ("evaluator_binding", "bound_to_evaluator"),
        ("verdict_ref", "supported_by_verdict"),
    ):
        value = document[field]
        if isinstance(value, str):
            graph.edge(root, relation, graph.node("evidence", value), evidence=value)
    verdict = document.get("verdict_ref")
    if isinstance(verdict, str):
        graph.edge(
            graph.node("verdict", verdict),
            "establishes",
            semantic,
            evidence=document["evidence_refs"],
            claim_scope=document["claim_scope"],
            frozen=True,
        )
    for reference in document["evidence_refs"]:
        evidence = graph.node("evidence", str(reference))
        graph.edge(root, "cites_evidence", evidence, evidence=str(reference))
