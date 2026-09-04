from __future__ import annotations

import json
from pathlib import Path

from wmloop.control.intermediate_ir import build_model_capability_ir
from wmloop.control.model_portrait import build_model_portrait
from wmloop.experiments.portable_knowledge_graph import build_portable_knowledge_graph
from wmloop.geometry.model_irg import (
    build_model_irg,
    detect_model_irg_collisions,
    model_irg_distance,
    rank_method_effects_by_irg,
    validate_model_irg,
)
from wmloop.retrieve.irg_guided_discovery import (
    build_irg_discovery_request,
    derive_irg_bottlenecks,
)


ROOT = Path(__file__).resolve().parents[1]


def _portrait() -> dict[str, object]:
    report = {
        "schema_version": 1,
        "artifact_type": "wmloop-model-onboarding-report",
        "repo_name": "model-irg-fixture",
        "source_revision": {"kind": "source_tree_sha256", "revision": "1" * 64},
        "capabilities": [
            {"capability": "inference", "state": "discovered", "evidence": ["fixture"]}
        ],
        "connector": {
            "entrypoints_by_kind": {"inference": ["inference"]},
            "asset_bindings": [{"kind": "model_asset"}],
        },
        "evaluator_contract": {
            "state": "ready",
            "evaluator_id": "fixture-evaluator",
            "contract_sha256": "2" * 64,
            "verifier": "fixture-verifier",
        },
    }
    capability = build_model_capability_ir(report, model_family="fixture", root=ROOT)
    return build_model_portrait(model_capability=capability, root=ROOT)


def _asset() -> dict[str, object]:
    return json.loads(
        (ROOT / "examples/acwm_unified_irg_assets_v1/assets/reacher.json").read_text(
            encoding="utf-8"
        )
    )


def _effect(mean: float, effect_id: str) -> dict[str, object]:
    return {
        "effect_id": effect_id,
        "method_id": "method-action-scale",
        "primitive": "action_conditioning_scale",
        "effect_status": "confirmed",
        "mean_effect": mean,
        "lower_bound": mean - 0.5 if mean > 0 else mean - 0.5,
        "upper_bound": mean + 0.5 if mean > 0 else mean + 0.5,
        "sign_q_value": 0.01,
        "transfer_state": "local_only",
        "evidence_refs": ["sha256:" + "a" * 64],
    }


def test_model_irg_binds_vector_to_portrait_and_projects_graph() -> None:
    portrait = _portrait()
    asset = _asset()
    binding = build_model_irg(portrait=portrait, asset=asset, root=ROOT)

    validate_model_irg(binding, portrait=portrait, asset=asset, root=ROOT)
    assert binding["dimensions"]["coordinate_count"] == len(asset["response_coordinate"])
    assert len(binding["diagnostic_axes"]) == len(asset["response_coordinate"])
    graph = build_portable_knowledge_graph([binding])
    assert graph["node_kind_counts"]["model_irg"] == 1
    assert graph["relation_counts"]["conditioned_on_portrait"] == 1


def test_model_irg_distance_and_effect_ranking_are_ranking_only() -> None:
    portrait = _portrait()
    asset = _asset()
    target = build_model_irg(portrait=portrait, asset=asset, root=ROOT)
    source = build_model_irg(
        portrait=portrait,
        asset=asset,
        method_effects=[_effect(1.0, "effect-source")],
        root=ROOT,
    )
    assert model_irg_distance(target, source) == 0.0
    ranked = rank_method_effects_by_irg(target, [source])
    assert ranked[0]["method_id"] == "method-action-scale"
    assert ranked[0]["claim_scope"] == "ranking_only"


def test_similar_irg_with_opposite_effects_is_a_collision() -> None:
    portrait = _portrait()
    asset = _asset()
    positive = build_model_irg(
        portrait=portrait,
        asset=asset,
        method_effects=[_effect(1.0, "effect-positive")],
        root=ROOT,
    )
    negative = build_model_irg(
        portrait=portrait,
        asset=asset,
        method_effects=[_effect(-1.0, "effect-negative")],
        root=ROOT,
    )
    collisions = detect_model_irg_collisions(
        [positive, negative],
        distance_threshold=0.1,
        minimum_effect=0.1,
        fdr_alpha=0.05,
    )
    assert len(collisions) == 1
    assert collisions[0].primitive == "action_conditioning_scale"


def test_irg_guides_bottleneck_hypotheses_and_cross_domain_queries() -> None:
    portrait = _portrait()
    asset = _asset()
    irg = build_model_irg(portrait=portrait, asset=asset, root=ROOT)
    bottlenecks = derive_irg_bottlenecks(irg, top_k=4)
    assert len(bottlenecks) == 4
    assert all(row["limitation_type"] == "local_sensitivity_bottleneck" for row in bottlenecks)
    request, plan = build_irg_discovery_request(
        irg,
        protected_metrics=("frozen_primary_metric",),
        top_k=4,
    )
    assert request.model_family == "fixture"
    assert request.failure_signatures
    assert request.target_metrics
    assert request.cross_domain_lenses
    assert plan["authority"] == "shadow_only"
    assert "global model ceiling" in plan["claim_boundary"]
