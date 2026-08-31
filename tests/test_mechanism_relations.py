from pathlib import Path
from collections.abc import Mapping

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
from wmloop.control.mechanism_composition import (
    MechanismCompositionError,
    compile_mechanism_composition,
    discover_mechanism_compositions,
    discover_from_memory,
    execute_mechanism_composition,
    binding_from_embodiment,
)
from wmloop.control.adapter_profiles import AdapterProfileError, ResolvedAdapter
from wmloop.control.model_executor_bootstrap import bootstrap_model_executor, bootstrap_request_template
from wmloop.control.campaign_api import CampaignStore
from wmloop.primitives.registry import PrimitiveRegistry
from wmloop.geometry.portable_transfer_knowledge import build_mechanism_contract
from wmloop.geometry.portable_transfer_knowledge import build_method_embodiment


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


def test_composition_compiler_derives_four_cells_without_pair_specific_runner() -> None:
    registry = PrimitiveRegistry.from_root(Path(__file__).resolve().parents[1])
    source = {"mechanism_id": "mechanism-a", "primitive": "drift_token_trim", "params": {"keep_tokens": 8}}
    target = {"mechanism_id": "mechanism-b", "primitive": "cfg_guidance_schedule", "params": {"guidance_start": 1.0, "guidance_end": 0.5}}
    plan = compile_mechanism_composition(registry=registry, source=source, target=target)
    assert [row["kind"] for row in plan["cells"]] == ["baseline", "source_only", "target_only", "combined"]
    assert plan["cells"][3]["interventions"] == [plan["source"], plan["target"]]


def test_composition_executor_settles_generic_effect_records() -> None:
    from wmloop.geometry import EffectContext, EffectRecord

    registry = PrimitiveRegistry.from_root(Path(__file__).resolve().parents[1])
    plan = compile_mechanism_composition(
        registry=registry,
        source={"mechanism_id": "mechanism-a", "primitive": "drift_token_trim", "params": {"keep_tokens": 8}},
        target={"mechanism_id": "mechanism-b", "primitive": "cfg_guidance_schedule", "params": {"guidance_start": 1.0, "guidance_end": 0.5}},
    )
    context = EffectContext("campaign", "model", "capability", "goal", "outcome", "chart", "heldout", (16,))
    effects = {"baseline": 0.1, "source_only": 0.3, "target_only": 0.2, "combined": 0.7}

    def executor(cell: object) -> EffectRecord:
        row = cell
        assert isinstance(row, Mapping)
        kind = str(row["kind"])
        primitive = {"baseline": "baseline", "source_only": "mechanism-a", "target_only": "mechanism-b", "combined": "composition"}[kind]
        value = effects[kind]
        return EffectRecord(f"{kind}-effect", primitive, context, "confirmed", value, 0.01, 0.05, 0.0, {"frozen": True}, 2, (REF,))

    result = execute_mechanism_composition(plan=plan, executor=executor)
    assert result["relation"]["verification_state"] == "confirmed"


def test_composition_rejects_registry_conflicts() -> None:
    registry = PrimitiveRegistry.from_root(Path(__file__).resolve().parents[1])
    with pytest.raises(MechanismCompositionError):
        compile_mechanism_composition(
            registry=registry,
            source={"mechanism_id": "a", "primitive": "next_forcing", "params": {"probability": 0.5}},
            target={"mechanism_id": "b", "primitive": "self_forcing_finetune", "params": {"probability": 0.5}},
        )


def test_discovery_selects_compatible_confirmed_methods_from_memory() -> None:
    from wmloop.geometry import EffectContext, EffectRecord

    registry = PrimitiveRegistry.from_root(Path(__file__).resolve().parents[1])
    context = EffectContext("campaign", "model", "capability", "goal", "outcome", "chart", "heldout", (16,))
    effects = [
        EffectRecord("a", "drift_token_trim", context, "confirmed", 0.3, 0.02, 0.2, 0.0, {"frozen": True}, 2, (REF,)),
        EffectRecord("b", "cfg_guidance_schedule", context, "confirmed", 0.2, 0.02, 0.1, 0.0, {"frozen": True}, 2, (REF,)),
    ]
    candidates = discover_mechanism_compositions(
        registry=registry,
        effect_records=effects,
        executable_bindings=[
            {"mechanism_id": "mechanism-a", "primitive": "drift_token_trim", "params": {"keep_tokens": 8}},
            {"mechanism_id": "mechanism-b", "primitive": "cfg_guidance_schedule", "params": {"guidance_start": 1.0, "guidance_end": 0.5}},
        ],
    )
    assert len(candidates) == 1
    assert candidates[0]["rationale"]["registry_compatible"] is True
    assert [row["kind"] for row in candidates[0]["plan"]["cells"]] == ["baseline", "source_only", "target_only", "combined"]


def test_embodiment_carries_its_reusable_execution_binding() -> None:
    registry = PrimitiveRegistry.from_root(Path(__file__).resolve().parents[1])
    mechanism = build_mechanism_contract(
        causal_claim="A claim", intervention_semantics="A", required_capabilities=["cap-a"],
        required_ablations=["a-component"], falsification_criterion="A fails", source_evidence_refs=[REF],
    )
    embodiment = build_method_embodiment(
        mechanism_id=mechanism["mechanism_id"], materialization_class="derived_embodiment",
        implementation_revision="rev-1", interface_contracts=["capability:cap-a"],
        implementation_state="confirmed", claim_boundary="target local", evidence_refs=[REF],
        executable_binding={"primitive": "drift_token_trim", "params": {"keep_tokens": 8}, "implementation_revision": "rev-1"},
    )
    binding = binding_from_embodiment(registry=registry, embodiment=embodiment)
    assert binding["primitive"] == "drift_token_trim"
    assert binding["mechanism_id"] == mechanism["mechanism_id"]


def test_discovery_can_consume_deposited_embodiments_directly() -> None:
    from wmloop.geometry import EffectContext, EffectRecord

    root = Path(__file__).resolve().parents[1]
    registry = PrimitiveRegistry.from_root(root)
    context = EffectContext("campaign", "model", "capability", "goal", "outcome", "chart", "heldout", (16,))
    effects = [
        EffectRecord("a", "drift_token_trim", context, "confirmed", 0.3, 0.02, 0.2, 0.0, {"frozen": True}, 2, (REF,)),
        EffectRecord("b", "cfg_guidance_schedule", context, "confirmed", 0.2, 0.02, 0.1, 0.0, {"frozen": True}, 2, (REF,)),
    ]
    embodiments = [
        {"mechanism_id": "mechanism-a", "executable_binding": {"primitive": "drift_token_trim", "params": {"keep_tokens": 8}, "implementation_revision": "r1"}},
        {"mechanism_id": "mechanism-b", "executable_binding": {"primitive": "cfg_guidance_schedule", "params": {"guidance_start": 1.0, "guidance_end": 0.5}, "implementation_revision": "r1"}},
    ]
    assert len(discover_from_memory(registry=registry, effect_records=effects, embodiments=embodiments)) == 1


def test_unknown_model_bootstrap_blocks_without_repair_authority(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fail(**_: object) -> object:
        raise AdapterProfileError("ADAPTER_PROFILE_NOT_FOUND")

    monkeypatch.setattr("wmloop.control.model_executor_bootstrap.compile_adapter_execution", fail)
    result = bootstrap_model_executor(
        model=tmp_path / "model", data=tmp_path / "data", goal="goal", budget=1.0,
        campaign_root=tmp_path / "campaign", project_root=Path(__file__).resolve().parents[1],
    )
    assert result["state"] == "blocked"
    assert result["blocker"]["code"] == "ADAPTER_REPAIR_INPUTS_REQUIRED"
    assert {item["key"] for item in result["required_inputs"]} == {"base_profile_path", "llm_adapter"}


def test_bootstrap_retries_compile_after_bounded_repair(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = {"compile": 0}
    resolved = ResolvedAdapter("profile", "family", "L1", {"kind": "pipeline"}, "freeze")

    def compile_stub(**_: object) -> ResolvedAdapter:
        calls["compile"] += 1
        if calls["compile"] == 1:
            raise AdapterProfileError("ADAPTER_PROFILE_NOT_FOUND")
        return resolved

    monkeypatch.setattr("wmloop.control.model_executor_bootstrap.compile_adapter_execution", compile_stub)
    repair_profile = tmp_path / "repaired-profile.json"
    repair_profile.write_text("{}", encoding="utf-8")

    def repair_stub(**_: object) -> Mapping[str, object]:
        return {"state": "ready", "adapter_profile_path": str(repair_profile), "assurance_level": "process_guarded_local"}

    result = bootstrap_model_executor(
        model=tmp_path / "model", data=tmp_path / "data", goal="goal", budget=1.0,
        campaign_root=tmp_path / "campaign", project_root=Path(__file__).resolve().parents[1],
        base_profile_path=tmp_path / "base.json", llm_adapter={"kind": "test"}, repair_runner=repair_stub,
    )
    assert result["state"] == "ready"
    assert result["source"] == "repaired_profile"
    assert calls["compile"] == 2


def test_campaign_store_routes_first_contact_through_bootstrap(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    execution = {
        "kind": "pipeline",
        "repo_root": str(tmp_path / "model"),
        "output_root": str(tmp_path / "runs"),
        "evaluator_contract": str(tmp_path / "evaluator.json"),
        "budget_total_gpu_hours": 1.0,
        "asset_bindings": {"--config": str(tmp_path / "config.json")},
        "probe_imports": False,
    }
    bootstrap = {
        "state": "ready",
        "source": "repaired_profile",
        "profile_id": "family-profile",
        "model_family": "new-family",
        "capability_level": "L1",
        "execution": execution,
        "constitution_freeze": "freeze",
        "bootstrap_digest": "d" * 64,
    }
    monkeypatch.setattr("wmloop.control.campaign_api.bootstrap_model_executor", lambda **_: bootstrap)
    store = CampaignStore(tmp_path / "campaigns", project_root=tmp_path)
    record = store.create(
        {
            "campaign_id": "first-contact",
            "model": str(tmp_path / "model"),
            "dataset": str(tmp_path / "data.jsonl"),
            "goal": "improve quality",
            "budget": 1.0,
            "executor_bootstrap": {"llm_adapter": {"provider": "test"}},
        }
    )
    assert record["executor_bootstrap"]["source"] == "repaired_profile"
    assert record["adapter_profile"] == "family-profile"


def test_bootstrap_request_template_is_human_readable_and_credential_free() -> None:
    template = bootstrap_request_template(model="/models/new", data="/data", goal="improve quality")
    assert template["artifact_type"] == "verdiwm-executor-bootstrap-request"
    assert template["approval"]["base_profile_path"] == ""
    assert "api_key" not in str(template).lower()
