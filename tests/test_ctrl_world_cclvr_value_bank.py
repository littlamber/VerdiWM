from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.materialize_ctrl_world_cclvr_value_bank import CCLVRValueBankError, materialize


def _write(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value).union(*( _keys(child) for child in value.values()))
    if isinstance(value, list):
        return set().union(*(_keys(child) for child in value)) if value else set()
    return set()


def _frame(root: Path, *, all_zero: bool = False, promotion_episode: bool = False):
    checkpoint = _write(root / "checkpoint.pt", {"checkpoint": 1})
    evaluator = _write(root / "evaluator.py", {"evaluator": 1})
    materializer_script = Path(__file__).resolve().parents[1] / "scripts" / "materialize_ctrl_world_cclvr_value_bank.py"
    contexts = []
    identities = []
    for index in range(3):
        episode = "1799" if promotion_episode and index == 2 else str(100 + index)
        context_id = f"context-{index}"
        contexts.append({"context_id": context_id, "episode_id": episode, "start_idx": index, "seeds": [101]})
        identities.append((context_id, episode, index, 101))
    contexts_path = _write(
        root / "contexts.json",
        {"artifact_type": "verdiwm-ctrl-world-local-context-set", "contexts": contexts},
    )
    campaign = _write(
        root / "campaign.json",
        {
            "artifact_type": "verdiwm-ctrl-world-cclvr-local-value-campaign",
            "state": "frozen_before_execution",
            "campaign_id": "test-cclvr",
            "checkpoint": {"path": str(checkpoint), "sha256": _sha256(checkpoint)},
            "contexts": {"path": str(contexts_path), "sha256": _sha256(contexts_path)},
            "dependencies": {
                "evaluator": str(evaluator),
                "evaluator_sha256": _sha256(evaluator),
                "materializer": str(materializer_script),
                "materializer_sha256": _sha256(materializer_script),
            },
            "execution": {"gpus": 1},
            "protocol": {
                "interact_num": 4,
                "num_inference_steps": 2,
                "target_interactions": [0, 1, 2, 3],
                "doses": [-0.99, 0.0, 0.99],
            },
            "local_return": {
                "suffix_mean_l1_weight": 1.0,
                "terminal_interaction_l1_weight": 2.0,
                "suffix_horizon_l1_slope_weight": 2.0,
                "minimum_benefit": 0.01,
            },
            "data_admission": {
                "prefix_identity_tolerance": 1e-9,
                "coverage_wilson_z": 1.96,
                "minimum_best_arm_support": {"negative": 4, "zero": 4, "positive": 4},
                "failure_action": "do_not_train",
            },
        },
    )
    measurements = []
    checks = []
    decision = 0
    for context_id, episode_id, start_idx, seed in identities:
        identity = {
            "context_id": context_id,
            "episode_id": episode_id,
            "start_idx": start_idx,
            "seed": seed,
        }
        zero_interactions = [
            {"interaction": index, "mean_l1": 10.0 + index, "final_l1": 10.5 + index}
            for index in range(4)
        ]
        measurements.append(
            {
                "identity": identity,
                "dose": 0.0,
                "target_interaction": None,
                "interactions": zero_interactions,
                "runtime_feature_records": [
                    [[float(index), 0.0, 1.0], [float(index), 1.0, 1.0]] for index in range(8)
                ],
            }
        )
        checks.append({"identity": identity, "target_interaction": None, "state": "passed"})
        for target in range(4):
            wanted = "zero" if all_zero else ("negative", "zero", "positive")[decision % 3]
            decision += 1
            for dose, name in ((-0.99, "negative"), (0.99, "positive")):
                interactions = [dict(row) for row in zero_interactions]
                delta = -1.0 if name == wanted else 1.0
                for index in range(target, 4):
                    interactions[index]["mean_l1"] += delta
                    interactions[index]["final_l1"] += delta
                measurements.append(
                    {
                        "identity": identity,
                        "dose": dose,
                        "target_interaction": target,
                        "interactions": interactions,
                        "runtime_feature_records": [],
                    }
                )
    shard = _write(
        root / "shard.json",
        {
            "artifact_type": "verdiwm-ctrl-world-local-fingerprint-probe-result",
            "state": "ready",
            "campaign_id": "test-cclvr",
            "probe_id": "fshc_interaction_local_gain",
            "hook_activation": {"state": "passed"},
            "input": {"checkpoint": str(checkpoint), "contexts_sha256": _sha256(contexts_path)},
            "measurements": measurements,
            "zero_identity_checks": checks,
        },
    )
    return campaign, shard


def test_materializes_anonymous_three_arm_local_values_and_coverage(tmp_path: Path) -> None:
    campaign, shard = _frame(tmp_path)

    manifest = materialize(campaign_path=campaign, shard_paths=[shard], output_root=tmp_path / "out")

    bank = json.loads(Path(manifest["training_path"]).read_text(encoding="utf-8"))
    assert manifest["state"] == "admitted"
    assert manifest["record_count"] == 12
    assert manifest["target_arm_counts"] == {"negative": 4, "zero": 4, "positive": 4}
    assert bank["feature_shape"] == [2, 2, 3]
    assert bank["coverage_band"]["successes"] == 8
    assert 0.0 < bank["coverage_band"]["lower"] < bank["coverage_band"]["upper"] < 1.0
    assert not {"context_id", "episode_id", "seed", "start_idx"}.intersection(_keys(bank))
    assert "1799" not in json.dumps(bank)
    assert all(len(row["arm_losses"]) == len(row["arm_values"]) == 3 for row in bank["records"])


def test_blocks_training_when_one_arm_lacks_support(tmp_path: Path) -> None:
    campaign, shard = _frame(tmp_path, all_zero=True)

    manifest = materialize(campaign_path=campaign, shard_paths=[shard], output_root=tmp_path / "out")

    assert manifest["state"] == "blocked"
    assert manifest["training_authorized"] is False
    settlement = json.loads(Path(manifest["settlement_path"]).read_text(encoding="utf-8"))
    assert settlement["support_checks"]["negative"]["state"] == "failed"
    assert settlement["support_checks"]["positive"]["state"] == "failed"


def test_rejects_promotion_episode_before_writing(tmp_path: Path) -> None:
    campaign, shard = _frame(tmp_path, promotion_episode=True)

    with pytest.raises(CCLVRValueBankError, match="PROMOTION_EPISODE_FORBIDDEN"):
        materialize(campaign_path=campaign, shard_paths=[shard], output_root=tmp_path / "out")

    assert not (tmp_path / "out").exists()
