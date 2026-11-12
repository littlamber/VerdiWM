"""Portable knowledge graph components: methods."""

from __future__ import annotations

from collections.abc import Mapping

from wmloop.experiments.knowledge.graph import _PortableKnowledgeGraph


def _project_method_ir(
    graph: _PortableKnowledgeGraph, document: Mapping[str, object]
) -> None:
    mapping = document["target_mapping"]
    training = document["training"]
    falsification = document["falsification"]
    mechanism = document["mechanism"]
    root = graph.node(
        "open_method",
        str(document["method_id"]),
        method_id=document["method_id"],
        mechanism_summary=mechanism["summary"],
        transformations=mechanism["transformations"],
        mapping_state=mapping["mapping_state"],
        training_mode=training["mode"],
        training_objective=training["objective"],
        required_capabilities=mapping["required_capabilities"],
        primary_metrics=falsification["primary_metrics"],
        protected_metrics=falsification["protected_metrics"],
        source_evidence_digest=document.get("source_evidence_digest"),
        claim_scope="ranking_only",
    )
    for capability in mapping["required_capabilities"]:
        graph.edge(
            root,
            "requires_capability",
            graph.node("capability", str(capability)),
        )
    portrait = document.get("target_portrait_binding")
    if isinstance(portrait, Mapping):
        graph.edge(
            root,
            "targets_portrait",
            graph.node(
                "model_portrait",
                str(portrait["portrait_id"]),
                portrait_id=portrait["portrait_id"],
            ),
            evidence=str(portrait["portrait_digest"]),
            claim_scope="ranking_only",
            frozen=False,
        )
    probe = document.get("probe_binding")
    if isinstance(probe, Mapping):
        for fingerprint_id in probe["fingerprint_ids"]:
            graph.edge(
                root,
                "informed_by_fingerprint",
                graph.node("probe_fingerprint", str(fingerprint_id)),
                evidence=str(probe["binding_digest"]),
                claim_scope="ranking_only",
                frozen=False,
            )
    for source in document["source_evidence"]:
        source_node = graph.node(
            "source_evidence",
            str(source["source_digest"]),
            source_id=source["source_id"],
            source_digest=source["source_digest"],
            claim=source["claim"],
        )
        graph.edge(
            root, "grounded_in", source_node, evidence=str(source["source_digest"])
        )


def _project_method_interface_extension(
    graph: _PortableKnowledgeGraph, document: Mapping[str, object]
) -> None:
    root = graph.node(
        "method_interface_extension",
        str(document["extension_id"]),
        extension_id=document["extension_id"],
        requested_surface=document["requested_surface"],
        semantic_role=document["semantic_role"],
        typed_inputs=document["typed_inputs"],
        typed_outputs=document["typed_outputs"],
        side_effect_class=document["side_effect_class"],
        state=document["state"],
        claim_scope="ranking_only",
    )
    method = graph.node(
        "open_method", str(document["method_id"]), method_id=document["method_id"]
    )
    graph.edge(
        method, "proposes_interface", root, claim_scope="ranking_only", frozen=False
    )


def _project_model_capability(
    graph: _PortableKnowledgeGraph, document: Mapping[str, object]
) -> None:
    capabilities = [
        str(row["capability"])
        for row in document["capabilities"]
        if row["state"] == "available"
    ]
    hooks = [
        str(row["hook"]) for row in document["hooks"] if row["state"] == "available"
    ]
    root = graph.node(
        "model_capability_profile",
        str(document["capability_id"]),
        model_capability_id=document["capability_id"],
        model_family=document["model_family"],
        available_capabilities=sorted(capabilities),
        available_hooks=sorted(hooks),
        source_revision=document["source_revision"],
    )
    graph.edge(
        root,
        "profiles_model_family",
        graph.node("model_family", str(document["model_family"])),
    )
    for row in document["capabilities"]:
        graph.edge(
            root,
            "has_capability",
            graph.node("capability", str(row["capability"])),
            state=row["state"],
        )
    for row in document["execution_interfaces"]:
        graph.edge(
            root,
            "has_execution_interface",
            graph.node("interface_contract", str(row["contract_id"])),
            state=row["state"],
        )
    for row in document["hooks"]:
        graph.edge(
            root,
            "has_hook",
            graph.node(
                "semantic_hook",
                str(row["hook"]),
                semantic_role=row["semantic_role"],
                binding_contract=row["binding_contract"],
            ),
            state=row["state"],
        )
