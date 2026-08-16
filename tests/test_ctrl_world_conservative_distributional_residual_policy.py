from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROFILE = (
    ROOT
    / "configs"
    / "primitives"
    / "ctrl_world_conservative_distributional_residual_policy_v1.json"
)


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_candidate_is_design_only_and_removes_forced_coverage() -> None:
    profile = _load(PROFILE)

    assert profile["state"] == "frozen_before_implementation"
    assert profile["execution_authority"] == "design_only_no_training"
    loss_contract = profile["loss_contract"]
    assert isinstance(loss_contract, dict)
    assert loss_contract["coverage_loss"] == "forbidden"
    assert loss_contract["coverage_dual"] == "forbidden"
    assert profile["screen_budget"]["confirmation_authorized"] is False


def test_candidate_uses_distributional_training_not_only_inference_calibration() -> None:
    profile = _load(PROFILE)
    changes = profile["structural_changes"]
    assert isinstance(changes, dict)
    components = {row["component"] for row in changes["added"]}

    assert components == {
        "multi_objective_quantile_arm_value_ensemble",
        "zero_baseline_conservative_dominance",
        "risk_gated_adapter_gradient",
        "on_policy_counterfactual_refresh",
        "development_only_tail_risk_calibration",
    }
    trainable = profile["implementation_contract"]["trainable_modules"]
    assert trainable == [
        "history_corrector.multiscale_side_adapter",
        "history_corrector.bootstrap_quantile_value_heads",
    ]


def test_candidate_forbids_reuse_of_episode_1799() -> None:
    profile = _load(PROFILE)
    gate = profile["data_admission_gate"]
    assert isinstance(gate, dict)

    policy = gate["promotion_episode_1799_policy"]
    assert isinstance(policy, str)
    for forbidden_use in (
        "training",
        "label construction",
        "calibration",
        "method selection",
        "promotion evaluation",
    ):
        assert forbidden_use in policy
    assert profile["promotion_gate"]["episode_1799_forbidden"] is True
    assert profile["promotion_gate"]["new_promotion_context_required"] is True


def test_candidate_ablation_identifies_trained_residual_contribution() -> None:
    profile = _load(PROFILE)
    ablations = {row["id"]: row for row in profile["screen_ablation"]}

    assert set(ablations) == {"a0", "d4", "e1", "e2", "e3", "e4", "e5", "e6"}
    assert ablations["e5"]["candidate"] is True
    assert "trained residual improvement" in profile["promotion_gate"][
        "mechanism_ablation_rule"
    ]


def test_candidate_evidence_receipts_match_versioned_repo_inputs() -> None:
    profile = _load(PROFILE)
    evidence = profile["mechanism_evidence"]
    assert isinstance(evidence, dict)

    for path_key, hash_key in (
        ("broad_request", "broad_request_sha256"),
        ("targeted_request", "targeted_request_sha256"),
        ("settled_reference_profiles", "settled_reference_profiles_sha256"),
    ):
        path = ROOT / evidence[path_key]
        assert _sha256(path) == evidence[hash_key]


def test_selected_sources_have_reviewed_evidence_annotations() -> None:
    profile = _load(PROFILE)
    evidence = profile["mechanism_evidence"]
    selected = {row["arxiv_id"] for row in evidence["selected_sources"]}
    annotations: set[str] = set()
    for name in (
        "ctrl_world_cbma_router_horizon_mechanism_annotations_v2.json",
        "ctrl_world_cclvr_risk_calibrated_router_mechanism_annotations_v1.json",
        "ctrl_world_cclvr_risk_calibrated_router_mechanism_annotations_v2.json",
    ):
        payload = _load(ROOT / "configs" / "retrieval" / name)
        annotations.update(payload["annotations"])

    assert selected <= annotations
