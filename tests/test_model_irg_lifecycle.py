from __future__ import annotations

import json
from pathlib import Path

import pytest

from wmloop.control.intermediate_ir import build_model_capability_ir
from wmloop.control.model_irg_materializer import (
    ModelIRGMaterializationError,
    materialize_model_irg,
)
from wmloop.control.model_portrait import build_model_portrait
from wmloop.control.probe_evolution_cycle import plan_probe_evolution_cycle
from wmloop.execute.pipeline_irg import (
    PipelineIRGError,
    resolve_pipeline_irg_inputs,
)
from wmloop.retrieve.coordinator import prepare_irg_discovery


ROOT = Path(__file__).resolve().parents[1]


def _portrait() -> dict[str, object]:
    report = {
        "schema_version": 1,
        "artifact_type": "wmloop-model-onboarding-report",
        "repo_name": "model-irg-lifecycle-fixture",
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


def _effect(mean: float, effect_id: str) -> dict[str, object]:
    return {
        "effect_id": effect_id,
        "method_id": "method-action-scale",
        "primitive": "action_conditioning_scale",
        "effect_status": "confirmed",
        "mean_effect": mean,
        "lower_bound": mean - 0.25,
        "upper_bound": mean + 0.25,
        "sign_q_value": 0.01,
        "transfer_state": "local_only",
        "evidence_refs": ["sha256:" + "a" * 64],
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_materializer_resumes_and_discovery_plans_without_network(tmp_path: Path) -> None:
    portrait = tmp_path / "portrait.json"
    asset = tmp_path / "asset.json"
    _write_json(portrait, _portrait())
    _write_json(
        asset,
        json.loads(
            (ROOT / "examples/acwm_unified_irg_assets_v1/assets/reacher.json").read_text(
                encoding="utf-8"
            )
        ),
    )
    output = tmp_path / "model-irg"
    first = materialize_model_irg(
        portrait_path=portrait,
        irg_asset_path=asset,
        output_root=output,
        repo_root=ROOT,
    )
    second = materialize_model_irg(
        portrait_path=portrait,
        irg_asset_path=asset,
        output_root=output,
        repo_root=ROOT,
    )
    assert first["irg_id"] == second["irg_id"]
    context = prepare_irg_discovery(
        model_irg_path=output / "model-irg.json",
        protected_metrics=("frozen_primary_metric",),
        output_root=tmp_path / "unused-network-output",
        control_root=ROOT,
        enable_external_discovery=False,
        max_results=8,
        timeout_seconds=2.0,
    )
    assert context.manifest["state"] == "planned"
    assert context.literature_queries
    assert context.failure_signatures


def test_materializer_fails_closed_when_resume_inputs_drift(tmp_path: Path) -> None:
    portrait = tmp_path / "portrait.json"
    asset = tmp_path / "asset.json"
    _write_json(portrait, _portrait())
    _write_json(
        asset,
        json.loads(
            (ROOT / "examples/acwm_unified_irg_assets_v1/assets/reacher.json").read_text(
                encoding="utf-8"
            )
        ),
    )
    output = tmp_path / "model-irg"
    materialize_model_irg(
        portrait_path=portrait,
        irg_asset_path=asset,
        output_root=output,
        repo_root=ROOT,
    )
    portrait.write_text(portrait.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ModelIRGMaterializationError, match="INPUT_LOCK_MISMATCH"):
        materialize_model_irg(
            portrait_path=portrait,
            irg_asset_path=asset,
            output_root=output,
            repo_root=ROOT,
        )


def test_opposite_effects_emit_generic_probe_evolution_work_order(tmp_path: Path) -> None:
    portrait = tmp_path / "portrait.json"
    asset = tmp_path / "asset.json"
    positive_effect = tmp_path / "positive.json"
    negative_effect = tmp_path / "negative.json"
    _write_json(portrait, _portrait())
    _write_json(
        asset,
        json.loads(
            (ROOT / "examples/acwm_unified_irg_assets_v1/assets/reacher.json").read_text(
                encoding="utf-8"
            )
        ),
    )
    _write_json(positive_effect, {"method_effects": [_effect(1.0, "positive")]})
    _write_json(negative_effect, {"method_effects": [_effect(-1.0, "negative")]})
    irg_paths = []
    for name, effects in (("positive", positive_effect), ("negative", negative_effect)):
        output = tmp_path / name
        manifest = materialize_model_irg(
            portrait_path=portrait,
            irg_asset_path=asset,
            method_effects_path=effects,
            output_root=output,
            repo_root=ROOT,
        )
        irg_paths.append(Path(str(manifest["model_irg_path"])))
    cycle = plan_probe_evolution_cycle(
        model_irg_paths=irg_paths,
        output_root=tmp_path / "probe-evolution",
        distance_threshold=0.1,
        minimum_effect=0.1,
        fdr_alpha=0.05,
        repo_root=ROOT,
    )
    assert cycle["state"] == "proposal_ready"
    assert cycle["collision_count"] == 1
    report = json.loads(
        (tmp_path / "probe-evolution/probe-evolution-cycle.json").read_text(
            encoding="utf-8"
        )
    )
    work_order = report["work_orders"][0]
    assert work_order["collision"]["primitive"] == "action_conditioning_scale"
    assert work_order["execution_authority"] == "none_until_materialized_and_settled"
    assert work_order["candidate_generation_contract"]["target_axes"]


def test_pipeline_irg_inputs_are_model_agnostic_and_fail_closed(tmp_path: Path) -> None:
    materialized = tmp_path / "model-irg.json"
    portrait = tmp_path / "portrait.json"
    asset = tmp_path / "asset.json"
    for path in (materialized, portrait, asset):
        path.write_text("{}", encoding="utf-8")
    empty = {
        "irg_diagnostic_axes_path": None,
        "irg_method_effects_path": None,
        "comparison_model_irg_paths": (),
    }
    with pytest.raises(PipelineIRGError, match="IRG_INPUTS_CONFLICT"):
        resolve_pipeline_irg_inputs(
            **empty,
            model_irg_path=materialized,
            model_portrait_path=portrait,
            irg_asset_path=asset,
        )
    with pytest.raises(PipelineIRGError, match="IRG_INPUTS_INCOMPLETE"):
        resolve_pipeline_irg_inputs(
            **empty,
            model_irg_path=None,
            model_portrait_path=portrait,
            irg_asset_path=None,
        )
    resolved = resolve_pipeline_irg_inputs(
        **empty,
        model_irg_path=None,
        model_portrait_path=portrait,
        irg_asset_path=asset,
    )
    assert resolved.enabled
    assert resolved.portrait == portrait.resolve()
