from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from wmloop.retrieve.mechanism_discovery import (
    DiscoveryPaper,
    DiscoveryRequest,
    MechanismDiscoveryError,
    _deduplicate_seed_records,
    _load_ontology,
    _load_profiles,
    _validate_extraction,
    build_multiview_queries,
    compare_mechanism_signature,
    run_mechanism_discovery,
)


ROOT = Path(__file__).resolve().parents[1]
AXES = (
    "state_representation",
    "conditioning_path",
    "update_operator",
    "reliability_routing",
    "training_distribution",
    "learning_signal",
    "gradient_path",
    "inference_transition",
)


def test_query_plan_uses_diagnostic_and_cross_domain_views() -> None:
    queries = build_multiview_queries(_request())

    assert {row["view"] for row in queries} >= {
        "diagnostic_symptom",
        "state_update_operator",
        "training_distribution",
        "architecture_hook",
        "cross_domain_1",
    }
    assert all("self forcing" not in row["query"] for row in queries)
    assert any("belief" in row["query"] for row in queries)


def test_light_discovery_budget_keeps_the_default_route_small(tmp_path: Path) -> None:
    manifest = run_mechanism_discovery(
        request=_request(),
        seed_records=(),
        output_root=tmp_path / "atlas",
        repo_root=ROOT,
        citation_depth=0,
        max_papers=1,
        search_results_per_view=0,
    )
    report = json.loads(Path(manifest["atlas_path"]).read_text(encoding="utf-8"))
    assert report["complexity_budget"]["name"] == "light"
    assert report["complexity_budget"]["query_view_count"] == 4
    assert report["complexity_budget"]["query_views_truncated"] is True


def test_light_discovery_rejects_unbounded_paper_request(tmp_path: Path) -> None:
    with pytest.raises(MechanismDiscoveryError, match="MECHANISM_MAX_PAPERS_INVALID"):
        run_mechanism_discovery(
            request=_request(),
            seed_records=(),
            output_root=tmp_path / "atlas",
            repo_root=ROOT,
            max_papers=13,
            search_results_per_view=0,
        )


def test_query_seed_collection_ignores_unsupported_arxiv_ids() -> None:
    records = _deduplicate_seed_records(
        (
            {"arxiv_id": "cs", "title": "Legacy identifier parser artifact"},
            {"arxiv_id": "2401.01234v1", "title": "Supported paper"},
        )
    )

    assert [row["arxiv_id"] for row in records] == ["2401.01234v1"]


def test_unfamiliar_title_is_not_novelty_evidence() -> None:
    profiles = _load_profiles(ROOT / "configs/retrieval/primitive_mechanism_profiles_v1.json")
    result = compare_mechanism_signature(
        {
            "axes": {axis: [] for axis in AXES},
            "extraction_state": "unresolved",
        },
        profiles,
    )

    assert result["novelty_state"] == "unresolved"
    assert result["nearest_primitives"] == []


def test_operator_equivalence_does_not_require_method_name() -> None:
    profiles = _load_profiles(ROOT / "configs/retrieval/primitive_mechanism_profiles_v1.json")
    extraction = _extraction(
        profiles["self_forcing_finetune"],
        excerpt="The model conditions on its own generated outputs during training.",
    )

    result = compare_mechanism_signature(extraction, profiles)

    assert result["novelty_state"] == "equivalent"
    assert result["nearest_primitives"][0]["primitive"] == "self_forcing_finetune"
    assert result["nearest_primitives"][0]["structural_similarity"] == 1.0


def test_new_reliability_operator_is_an_extension_not_a_name_match() -> None:
    profiles = _load_profiles(ROOT / "configs/retrieval/primitive_mechanism_profiles_v1.json")
    axes = {axis: list(values) for axis, values in profiles["self_forcing_finetune"].items()}
    axes["reliability_routing"] = ["state_dependent_signed_reliability"]
    extraction = _extraction(
        axes,
        excerpt="The controller predicts whether old observations should help or oppose the update.",
    )

    result = compare_mechanism_signature(extraction, profiles)

    assert result["novelty_state"] == "extension"
    assert result["nearest_primitives"][0]["primitive"] == "self_forcing_finetune"
    assert "reliability_routing" in result["novel_axes"]


def test_ontology_matches_semantic_aliases_without_exact_tag_identity() -> None:
    profiles = _load_profiles(ROOT / "configs/retrieval/primitive_mechanism_profiles_v1.json")
    ontology = _load_ontology(ROOT / "configs/retrieval/mechanism_tag_ontology_v1.json")
    axes = {
        "state_representation": ["self_generated_history"],
        "conditioning_path": ["autoregressive_self_conditioning"],
        "update_operator": ["self_forced_video_rollout"],
        "reliability_routing": ["fixed_chunk_schedule"],
        "training_distribution": ["free_running_generated_history"],
        "learning_signal": ["holistic_video_level_loss"],
        "gradient_path": ["truncated_autoregressive_rollout_gradient", "world_model"],
        "inference_transition": ["rolling_cache_video_extrapolation"],
    }

    result = compare_mechanism_signature(
        _extraction(axes, excerpt="The model trains on generated video rollouts."),
        profiles,
        ontology=ontology,
    )

    assert result["novelty_state"] == "equivalent"
    assert result["nearest_primitives"][0]["primitive"] == "self_forcing_finetune"
    assert result["nearest_primitives"][0]["structural_similarity"] == 1.0


def test_prior_mechanism_reference_participates_in_novelty_comparison() -> None:
    ontology = _load_ontology(ROOT / "configs/retrieval/mechanism_tag_ontology_v1.json")
    reference = {
        "literature:2405.20222v3": {
            "state_representation": ["multiscale_image_features", "dense_motion_flow"],
            "conditioning_path": ["multiscale_guided_feature_injection"],
            "update_operator": ["sparse_to_dense_motion_warp"],
            "reliability_routing": ["domain_specific_adapter_composition"],
            "training_distribution": ["sparse_control_video_pairs"],
            "learning_signal": ["control_fidelity_objective"],
            "gradient_path": ["motion_adapters"],
            "inference_transition": ["motion_field_guided_video_diffusion"],
        }
    }
    candidate = {
        "state_representation": ["target_backbone_features", "corrective_control_features"],
        "conditioning_path": ["high_frequency_high_bandwidth_feedback"],
        "update_operator": ["iterative_corrective_feedback"],
        "reliability_routing": ["mixture_of_experts_router"],
        "training_distribution": ["image_video_sparse_frame_controls"],
        "learning_signal": ["generation_quality_and_control_fidelity"],
        "gradient_path": ["compact_control_network"],
        "inference_transition": ["dense_interleaved_feature_correction"],
    }

    result = compare_mechanism_signature(
        _extraction(candidate, excerpt="A compact side path applies corrective features."),
        reference,
        ontology=ontology,
        reference_kinds={"literature:2405.20222v3": "evidence_supported_literature"},
    )

    assert result["nearest_primitives"][0]["primitive"] == "literature:2405.20222v3"
    assert result["nearest_primitives"][0]["reference_kind"] == "evidence_supported_literature"
    assert result["nearest_primitives"][0]["structural_similarity"] >= 0.75


def test_discovery_expands_citations_and_keeps_unknown_paper_unresolved(tmp_path: Path) -> None:
    seed_text = (
        "The model conditions on its own generated outputs during training. "
        "We use a sequence generation loss and a truncated rollout gradient."
    )
    cited_text = "A behavioural discriminator aligns teacher and free-running dynamics."
    annotations = tmp_path / "annotations.json"
    annotations.write_text(
        json.dumps(
            {
                "annotations": {
                    "2401.12345v2": _extraction(
                        _load_profiles(ROOT / "configs/retrieval/primitive_mechanism_profiles_v1.json")[
                            "self_forcing_finetune"
                        ],
                        excerpt="The model conditions on its own generated outputs during training.",
                    )
                }
            }
        ),
        encoding="utf-8",
    )

    def fetch_text(arxiv_id: str, *, timeout_seconds: float):
        del timeout_seconds
        if arxiv_id.startswith("2401.12345"):
            return seed_text, ("1610.09038",), "network"
        return cited_text, (), "network"

    def fetch_metadata(arxiv_id: str, *, timeout_seconds: float) -> Mapping[str, str]:
        del timeout_seconds
        assert arxiv_id == "1610.09038"
        return {
            "arxiv_id": arxiv_id,
            "title": "Adversarial dynamics alignment",
            "abstract": cited_text,
            "source_url": f"https://arxiv.org/abs/{arxiv_id}",
        }

    manifest = run_mechanism_discovery(
        request=_request(),
        seed_records=(
            {
                "arxiv_id": "2401.12345v2",
                "title": "Opaque temporal generator",
                "abstract": seed_text,
                "source_url": "https://arxiv.org/abs/2401.12345v2",
            },
        ),
        output_root=tmp_path / "atlas",
        repo_root=ROOT,
        annotations_path=annotations,
        citation_depth=1,
        max_papers=4,
        full_text_fetcher=fetch_text,
        metadata_fetcher=fetch_metadata,
        search_results_per_view=0,
    )

    report = json.loads(Path(manifest["atlas_path"]).read_text(encoding="utf-8"))
    assert manifest["execution_authority"] == "shadow_only"
    assert report["paper_count"] == 2
    assert report["entries"][0]["comparison"]["novelty_state"] == "equivalent"
    assert report["entries"][0]["comparison"]["nearest_primitives"][0]["primitive"] == "self_forcing_finetune"
    assert report["entries"][1]["source"]["citation_depth"] == 1
    assert report["entries"][1]["comparison"]["novelty_state"] == "unresolved"


def test_semantic_annotation_must_quote_available_evidence() -> None:
    paper = DiscoveryPaper(
        arxiv_id="2401.12345",
        title="Opaque method name",
        abstract="The paper uses a bounded recurrent state.",
        source_url="https://arxiv.org/abs/2401.12345",
        full_text_url="https://ar5iv.labs.arxiv.org/html/2401.12345",
        full_text="The paper uses a bounded recurrent state.",
        full_text_state="network",
        citation_depth=0,
        discovered_from=("seed",),
    )
    extraction = _extraction(
        {axis: [f"axis_{index}"] for index, axis in enumerate(AXES)},
        excerpt="This sentence is not in the paper.",
    )

    with pytest.raises(MechanismDiscoveryError, match="MECHANISM_EXTRACTION_EVIDENCE_UNBOUND"):
        _validate_extraction(extraction, paper=paper)


def test_ctrl_world_router_horizon_annotations_are_evidence_dense() -> None:
    payload = json.loads(
        (
            ROOT
            / "configs"
            / "retrieval"
            / "ctrl_world_cbma_router_horizon_mechanism_annotations_v2.json"
        ).read_text(encoding="utf-8")
    )
    annotations = payload["annotations"]

    assert set(annotations) == {
        "1506.03099v3",
        "1702.02896v6",
        "1907.00208v2",
        "2202.03673v2",
        "2407.01392v4",
        "2408.15664v1",
        "2502.01459v2",
        "2505.10160v2",
    }
    for annotation in annotations.values():
        assert annotation["extraction_state"] == "evidence_supported"
        assert set(annotation["axes"]) == set(AXES)
        assert all(annotation["axes"][axis] for axis in AXES)
        assert len(annotation["evidence_excerpts"]) >= 2


def test_ctrl_world_local_value_router_keeps_confirmation_blocked() -> None:
    candidate = json.loads(
        (
            ROOT
            / "configs"
            / "primitives"
            / "ctrl_world_coverage_constrained_local_value_router_v1.json"
        ).read_text(encoding="utf-8")
    )

    assert candidate["state"] == "frozen_before_implementation"
    assert candidate["execution_authority"] == "design_only_no_training"
    assert candidate["screen_budget"]["gpus"] == 8
    assert candidate["screen_budget"]["steps"] == 64
    assert candidate["screen_budget"]["confirmation_steps"] == 512
    assert candidate["screen_budget"]["confirmation_authorized"] is False
    assert "1799" in candidate["data_admission_gate"]["heldout_episode_policy"]
    assert {row["id"] for row in candidate["screen_ablation"]} >= {
        "a0",
        "b3",
        "c1",
        "c3",
        "d1",
        "d2",
        "d3",
        "d4",
    }


def test_unbound_reviewed_annotation_is_listed_and_downgraded(tmp_path: Path) -> None:
    evidence = "The paper uses a bounded recurrent state."
    annotations = tmp_path / "annotations.json"
    annotations.write_text(
        json.dumps(
            {
                "annotations": {
                    "2401.12345v2": _extraction(
                        {axis: [f"axis_{index}"] for index, axis in enumerate(AXES)},
                        excerpt="This exact reviewed quote is unavailable.",
                    )
                }
            }
        ),
        encoding="utf-8",
    )

    manifest = run_mechanism_discovery(
        request=_request(),
        seed_records=(
            {
                "arxiv_id": "2401.12345v2",
                "title": "Opaque temporal mechanism",
                "abstract": evidence,
                "source_url": "https://arxiv.org/abs/2401.12345v2",
            },
        ),
        output_root=tmp_path / "atlas",
        repo_root=ROOT,
        annotations_path=annotations,
        citation_depth=0,
        max_papers=1,
        full_text_fetcher=lambda arxiv_id, timeout_seconds: (evidence, (), "network"),
        search_results_per_view=0,
        local_sources_root=tmp_path / "sources",
    )

    report = json.loads(Path(manifest["atlas_path"]).read_text(encoding="utf-8"))
    entry = report["entries"][0]
    assert entry["mechanism"]["extraction_state"] == "unresolved"
    assert entry["comparison"]["novelty_state"] == "unresolved"
    missing = json.loads(Path(manifest["missing_sources_path"]).read_text(encoding="utf-8"))
    assert missing["missing_source_count"] == 1
    assert missing["records"][0]["reasons"] == ["annotation_evidence_unbound"]


def test_missing_source_list_can_be_satisfied_by_local_text(tmp_path: Path) -> None:
    local_root = tmp_path / "sources"
    request = _request()

    def missing_text(arxiv_id: str, *, timeout_seconds: float):
        del arxiv_id, timeout_seconds
        raise OSError("network unavailable")

    first = run_mechanism_discovery(
        request=request,
        seed_records=(
            {
                "arxiv_id": "2401.12345v2",
                "title": "Offline temporal mechanism",
                "abstract": "A bounded temporal mechanism.",
                "source_url": "https://arxiv.org/abs/2401.12345v2",
            },
        ),
        output_root=tmp_path / "atlas-missing",
        repo_root=ROOT,
        citation_depth=0,
        max_papers=1,
        full_text_fetcher=missing_text,
        search_results_per_view=0,
        local_sources_root=local_root,
    )
    missing = json.loads(Path(first["missing_sources_path"]).read_text(encoding="utf-8"))
    assert missing["missing_source_count"] == 1
    assert missing["records"][0]["accepted_filenames"][0] == "2401.12345v2.txt"
    markdown = Path(first["missing_sources_markdown_path"]).read_text(encoding="utf-8")
    assert "2401.12345v2 - Offline temporal mechanism" in markdown
    assert "2401.12345v2.txt" in markdown

    local_text = "A bounded temporal mechanism with cited prior arXiv:1610.09038."
    (local_root / "2401.12345v2.txt").write_text(local_text, encoding="utf-8")
    second = run_mechanism_discovery(
        request=request,
        seed_records=(
            {
                "arxiv_id": "2401.12345v2",
                "title": "Offline temporal mechanism",
                "abstract": "A bounded temporal mechanism.",
                "source_url": "https://arxiv.org/abs/2401.12345v2",
            },
        ),
        output_root=tmp_path / "atlas-local",
        repo_root=ROOT,
        citation_depth=0,
        max_papers=1,
        full_text_fetcher=missing_text,
        search_results_per_view=0,
        local_sources_root=local_root,
    )
    report = json.loads(Path(second["atlas_path"]).read_text(encoding="utf-8"))
    assert second["missing_source_count"] == 0
    assert report["entries"][0]["source"]["full_text_state"] == "local"
    assert report["entries"][0]["source"]["local_source_sha256"]


def _request() -> DiscoveryRequest:
    return DiscoveryRequest(
        symptom_description=(
            "The effect of old visual history changes sign across otherwise comparable contexts"
        ),
        failure_signatures=("context_local_anchor_sign_flip", "train_infer_mismatch"),
        target_metrics=("long_horizon_prediction_quality", "action_following"),
        protected_metrics=("short_horizon_quality",),
        available_hooks=("history_latent_input", "action_cross_attention", "training_objective"),
        model_family="ctrl-world spatiotemporal unet diffusion",
        cross_domain_lenses=(
            "belief state estimation confidence update",
            "adaptive memory retention forgetting",
        ),
    )


def _extraction(
    axes: Mapping[str, Any],
    *,
    excerpt: str,
) -> dict[str, object]:
    return {
        "axes": {axis: list(axes[axis]) for axis in AXES},
        "evidence_excerpts": [excerpt],
        "requirements": ["A differentiable world-model training path is available."],
        "failure_boundaries": ["The mechanism remains subject to held-out evaluation."],
        "extraction_state": "evidence_supported",
        "extraction_rationale": "The operator axes are directly supported by the quoted evidence.",
    }
