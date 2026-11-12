"""Portable knowledge graph components: models."""

from __future__ import annotations

from collections.abc import Mapping

from wmloop.experiments.knowledge.graph import _PortableKnowledgeGraph


def _project_model_portrait(
    graph: _PortableKnowledgeGraph, document: Mapping[str, object]
) -> None:
    coverage = document["coverage"]
    root = graph.node(
        "model_portrait",
        str(document["portrait_id"]),
        portrait_id=document["portrait_id"],
        model_capability_id=document["model_capability_id"],
        model_family=document["model_family"],
        available_capabilities=coverage["available_capabilities"],
        available_hooks=coverage["available_hooks"],
        observed_probe_keys=coverage["observed_probe_keys"],
        unknown_operational_metrics=coverage["unknown_operational_metrics"],
    )
    capability = graph.node(
        "model_capability_profile",
        str(document["model_capability_id"]),
        model_capability_id=document["model_capability_id"],
        model_family=document["model_family"],
    )
    graph.edge(root, "describes_capability_profile", capability)
    for capability_name in coverage["available_capabilities"]:
        graph.edge(
            root,
            "has_capability",
            graph.node("capability", str(capability_name)),
            state="available",
        )
    for hook in coverage["available_hooks"]:
        graph.edge(
            root,
            "has_hook",
            graph.node("semantic_hook", str(hook)),
            state="available",
        )
    for fingerprint in document["behavioral_fingerprints"]:
        child = graph.node(
            "probe_fingerprint",
            str(fingerprint["fingerprint_id"]),
            model_capability_id=fingerprint["model_capability_id"],
            diagnostic_role=fingerprint["diagnostic_role"],
            context_class=fingerprint["context_class"],
            split=fingerprint["split"],
            horizons=fingerprint["horizons"],
            dose_values=fingerprint["dose_values"],
            replication_count=fingerprint["replication_count"],
            response_digest=fingerprint["response_digest"],
        )
        graph.edge(root, "has_fingerprint", child, state=fingerprint["state"])
    parent = document.get("parent_portrait_id")
    transition = document.get("transition_ref")
    if isinstance(parent, str) and isinstance(transition, str):
        graph.edge(
            root,
            "derived_from",
            graph.node("model_portrait", parent, portrait_id=parent),
            evidence=transition,
            claim_scope="ranking_only",
            frozen=False,
        )
    for reference in document["evidence_refs"]:
        graph.edge(
            root,
            "cites_evidence",
            graph.node("evidence", str(reference)),
            evidence=str(reference),
        )


def _project_model_irg(
    graph: _PortableKnowledgeGraph, document: Mapping[str, object]
) -> None:
    """Project the model-conditioned IRG without granting verdict authority."""

    portrait = document["portrait_binding"]
    asset = document["asset_binding"]
    root = graph.node(
        "model_irg",
        str(document["irg_id"]),
        irg_id=document["irg_id"],
        model_family=portrait["model_family"],
        environment=asset["environment"],
        backbone_family=asset["backbone_family"],
        goal_schema=asset["goal_schema"],
        coordinate_names=document["coordinate_names"],
        response_vector=document["response_vector"],
        support_mask=document["support_mask"],
        available_hooks=document["available_hooks"],
        routing_state=document["routing_state"],
        claim_scope="ranking_only",
    )
    portrait_node = graph.node(
        "model_portrait",
        str(portrait["portrait_id"]),
        portrait_id=portrait["portrait_id"],
        model_family=portrait["model_family"],
    )
    graph.edge(
        root,
        "conditioned_on_portrait",
        portrait_node,
        evidence=str(portrait["portrait_digest"]),
        claim_scope="ranking_only",
        frozen=False,
    )
    asset_node = graph.node(
        "irg_asset",
        str(asset["asset_id"]),
        asset_id=asset["asset_id"],
        environment=asset["environment"],
        backbone_family=asset["backbone_family"],
    )
    graph.edge(
        root,
        "measured_by",
        asset_node,
        evidence=str(asset["asset_digest"]),
        claim_scope="ranking_only",
        frozen=False,
    )
    for axis in document["diagnostic_axes"]:
        axis_node = graph.node(
            "irg_diagnostic_axis",
            f"{document['irg_id']}:{axis['axis']}",
            axis=axis["axis"],
            probe_id=axis["probe_id"],
            outcome=axis["outcome"],
            magnitude=axis["magnitude"],
            support_state=axis["support_state"],
            diagnosis=axis["diagnosis"],
        )
        graph.edge(
            root,
            "has_diagnostic_axis",
            axis_node,
            evidence=list(axis["evidence_refs"]),
            claim_scope="ranking_only",
            frozen=False,
        )
    for effect in document["method_effects"]:
        method_node = graph.node(
            "open_method",
            str(effect["method_id"]),
            method_id=effect["method_id"],
            claim_scope="ranking_only",
        )
        graph.edge(
            root,
            "binds_method_effect",
            method_node,
            primitive=effect["primitive"],
            effect_status=effect["effect_status"],
            mean_effect=effect["mean_effect"],
            lower_bound=effect["lower_bound"],
            upper_bound=effect["upper_bound"],
            transfer_state=effect["transfer_state"],
            evidence=list(effect["evidence_refs"]),
            claim_scope="ranking_only",
            frozen=False,
        )
    for reference in document["collision_refs"]:
        graph.edge(
            root,
            "retains_collision",
            graph.node("counterexample", str(reference)),
            evidence=str(reference),
            claim_scope="ranking_only",
            frozen=False,
        )
    for reference in document["evolution_refs"]:
        graph.edge(
            root,
            "evolved_from_counterexample",
            graph.node("probe_evolution", str(reference)),
            evidence=str(reference),
            claim_scope="ranking_only",
            frozen=False,
        )


def _project_portrait_transition(
    graph: _PortableKnowledgeGraph, document: Mapping[str, object]
) -> None:
    state = str(document["outcome_state"])
    frozen = state != "admitted"
    root = graph.node(
        "portrait_transition",
        str(document["transition_id"]),
        outcome_state=state,
        claim_scope=document["claim_scope"],
    )
    parent = graph.node(
        "model_portrait",
        str(document["parent_portrait_id"]),
        portrait_id=document["parent_portrait_id"],
    )
    portrait = graph.node(
        "model_portrait",
        str(document["portrait_id"]),
        portrait_id=document["portrait_id"],
    )
    embodiment = graph.node("embodiment", str(document["embodiment_id"]))
    graph.edge(root, "updates_portrait", portrait)
    graph.edge(
        portrait,
        "derived_from",
        parent,
        evidence=document["transition_ref"],
        claim_scope="ranking_only",
        frozen=False,
    )
    graph.edge(
        portrait,
        "changed_by",
        embodiment,
        evidence=document["evidence_refs"],
        claim_scope=document["claim_scope"],
        frozen=frozen,
    )
    verdict = document.get("verdict_ref")
    if isinstance(verdict, str):
        verdict_node = graph.node("verdict", verdict)
        graph.edge(
            verdict_node,
            "establishes",
            root,
            evidence=document["evidence_refs"],
            claim_scope=document["claim_scope"],
            frozen=True,
        )
