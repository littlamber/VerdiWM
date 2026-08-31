from pathlib import Path

import pytest

from wmloop.geometry import (
    EffectMemory,
    GeometryValidationError,
    build_mechanism_relation,
    classify_interaction,
    interaction_effect,
    propose_mechanism_relation,
    settle_mechanism_relation,
    relation_from_dict,
)
from wmloop.experiments.portable_knowledge_graph import build_portable_knowledge_graph
from wmloop.control.experiment_portfolio import build_relation_hypothesis_batch, validate_hypothesis_batch
from wmloop.geometry.portable_transfer_knowledge import build_mechanism_contract


REF = "cas://sha256/" + "a" * 64


def _relation(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "source_mechanism_id": "mechanism-a",
        "target_mechanism_id": "mechanism-b",
        "relation_type": "positive_synergy",
        "composition_operator": "parallel",
        "baseline_effect": 0.1,
        "source_effect": 0.3,
        "target_effect": 0.2,
        "combined_effect": 0.7,
        "uncertainty": 0.05,
        "replication_count": 3,
        "required_ablations": ["a-only", "b-only"],
        "evidence_refs": [REF],
        "validity_gates": {"frozen_evaluator": True, "heldout": True},
        "verification_state": "confirmed",
    }
    values.update(overrides)
    return build_mechanism_relation(**values)  # type: ignore[arg-type]


def test_interaction_contrast_and_classification() -> None:
    assert interaction_effect(baseline=0.1, source=0.3, target=0.2, combined=0.7) == pytest.approx(0.3)
    assert classify_interaction(baseline=0.1, source=0.3, target=0.2, combined=0.7, uncertainty=0.05) == (
        "positive_synergy",
        pytest.approx(0.3),
    )
    assert classify_interaction(baseline=0.1, source=0.3, target=0.2, combined=0.31, uncertainty=0.2)[0] == "abstained"


def test_relation_is_validated_and_queryable(tmp_path: Path) -> None:
    relation = relation_from_dict(_relation())
    memory = EffectMemory(relations=(relation,))
    assert memory.query_relations(mechanism_id="mechanism-a", relation_type="positive_synergy") == (relation,)
    output = memory.write_relation_jsonl(tmp_path / "relations.jsonl")
    assert '"relation_type": "positive_synergy"' in output.read_text(encoding="utf-8")


def test_authoritative_relation_requires_gates_and_ablations() -> None:
    with pytest.raises(GeometryValidationError, match="GATE_FAILED"):
        _relation(validity_gates={"heldout": False})
    with pytest.raises(GeometryValidationError, match="ABLATION_MISSING"):
        _relation(required_ablations=[])


def test_relation_projects_into_portable_graph() -> None:
    graph = build_portable_knowledge_graph([_relation()])
    relations = [edge["relation"] for edge in graph["edges"]]
    assert "relates_source" in relations
    assert "relates_target" in relations
    assert graph["relation_counts"]["cites_evidence"] == 1


def test_proposal_classifies_without_promoting_claim() -> None:
    relation = propose_mechanism_relation(
        source_mechanism_id="mechanism-a",
        target_mechanism_id="mechanism-b",
        composition_operator="parallel",
        baseline_effect=0.1,
        source_effect=0.3,
        target_effect=0.2,
        combined_effect=0.7,
        uncertainty=0.05,
        replication_count=1,
        required_ablations=["a-only", "b-only"],
        evidence_refs=[REF],
    )
    assert relation["relation_type"] == "positive_synergy"
    assert relation["verification_state"] == "candidate"
    assert relation["claim_scope"] == "ranking_only"


def test_relation_adapter_creates_composition_and_component_ablations() -> None:
    source = build_mechanism_contract(
        causal_claim="A claim", intervention_semantics="A", required_capabilities=["cap-a"],
        required_ablations=["a-component"], falsification_criterion="A fails", source_evidence_refs=[REF],
    )
    target = build_mechanism_contract(
        causal_claim="B claim", intervention_semantics="B", required_capabilities=["cap-b"],
        required_ablations=["b-component"], falsification_criterion="B fails", source_evidence_refs=[REF],
    )
    relation = build_mechanism_relation(
        source_mechanism_id=source["mechanism_id"], target_mechanism_id=target["mechanism_id"],
        relation_type="positive_synergy", composition_operator="parallel",
        baseline_effect=0.1, source_effect=0.3, target_effect=0.2, combined_effect=0.7,
        uncertainty=0.05, replication_count=2, required_ablations=["pair-control"], evidence_refs=[REF],
        verification_state="candidate",
    )
    batch = build_relation_hypothesis_batch(
        relation=relation, source_mechanism=source, target_mechanism=target,
        expected_portrait_changes=["quality_gain"], required_module_capabilities=["cap-a", "cap-b"],
        information_gain=0.8, uncertainty=0.4, estimated_screen_gpu_hours=1.0,
    )
    validate_hypothesis_batch(batch)
    mechanism = batch["candidates"][0]["mechanism_contract"]
    assert set(mechanism["required_ablations"]) >= {
        f"remove:{source['mechanism_id']}", f"remove:{target['mechanism_id']}"
    }


def test_settlement_closes_effect_records_to_confirmed_relation() -> None:
    from wmloop.geometry import EffectContext, EffectRecord

    context = EffectContext(
        campaign_id="campaign", backbone_family="model", capability_class="capability",
        goal_schema="goal", outcome_schema="outcome", chart_id="chart",
        data_regime="heldout", horizons=(16,),
    )

    def record(record_id: str, primitive: str, effect: float) -> EffectRecord:
        return EffectRecord(
            record_id=record_id, primitive=primitive, context=context, status="confirmed",
            mean_effect=effect, standard_error=0.01, lower_bound=0.1, goal_threshold=0.0,
            validity_gates={"frozen": True}, replication_count=2, evidence_refs=(REF,),
        )

    relation = settle_mechanism_relation(
        baseline=record("baseline", "baseline", 0.1),
        source=record("source", "mechanism-a", 0.3),
        target=record("target", "mechanism-b", 0.2),
        combined=record("combined", "composition", 0.7),
        source_mechanism_id="mechanism-a", target_mechanism_id="mechanism-b",
        composition_operator="parallel", required_ablations=["remove:mechanism-a", "remove:mechanism-b"],
    )
    assert relation["verification_state"] == "confirmed"
    assert relation["relation_type"] == "positive_synergy"


def test_memory_settle_relation_persists_result() -> None:
    from wmloop.geometry import EffectContext, EffectRecord

    context = EffectContext(
        campaign_id="campaign", backbone_family="model", capability_class="capability",
        goal_schema="goal", outcome_schema="outcome", chart_id="chart",
        data_regime="heldout", horizons=(16,),
    )

    def record(record_id: str, primitive: str, effect: float) -> EffectRecord:
        return EffectRecord(
            record_id=record_id, primitive=primitive, context=context, status="confirmed",
            mean_effect=effect, standard_error=0.01, lower_bound=0.1, goal_threshold=0.0,
            validity_gates={"frozen": True}, replication_count=2, evidence_refs=(REF,),
        )

    memory = EffectMemory((
        record("baseline", "baseline", 0.1), record("source", "mechanism-a", 0.3),
        record("target", "mechanism-b", 0.2), record("combined", "composition", 0.7),
    ))
    relation = memory.settle_relation(
        baseline=memory.query(primitive="baseline")[0],
        source=memory.query(primitive="mechanism-a")[0],
        target=memory.query(primitive="mechanism-b")[0],
        combined=memory.query(primitive="composition")[0],
        source_mechanism_id="mechanism-a", target_mechanism_id="mechanism-b",
        composition_operator="parallel", required_ablations=["remove:mechanism-a", "remove:mechanism-b"],
    )
    assert relation in memory.relations()
