from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.materialize_ctrl_world_fshc_feature_bank import FSHCFeatureBankError, materialize


def _write(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _frame(root: Path, *, promotion_episode: bool = False):
    campaigns = []
    selectors = []
    shards = []
    classifications = []
    receipts = []
    for campaign_index, target in enumerate((-1, 1), start=1):
        campaign_id = f"campaign-{campaign_index}"
        context_id = f"context-{campaign_index}"
        episode_id = "1799" if promotion_episode and campaign_index == 2 else str(campaign_index)
        campaign = _write(
            root / f"campaign-{campaign_index}.json",
            {
                "artifact_type": "verdiwm-ctrl-world-local-fingerprint-campaign",
                "campaign_id": campaign_id,
                "outcomes": [{"name": "quality", "weight": 1.0}],
            },
        )
        selected = {"dose": 0.025 * target}
        selector = _write(
            root / f"selector-{campaign_index}.json",
            {
                "artifact_type": "verdiwm-ctrl-world-frozen-directional-selector",
                "state": "frozen",
                "fingerprint_campaign_id": campaign_id,
                "contexts": [
                    {
                        "context": {
                            "context_id": context_id,
                            "episode_id": episode_id,
                            "start_idx": 4,
                            "seeds": [101, 202],
                        },
                        "selector_action": "execute",
                        "selected_candidate": selected,
                    }
                ],
            },
        )
        measurements = []
        checks = []
        for seed in (101, 202):
            identity = {
                "context_id": context_id,
                "episode_id": episode_id,
                "start_idx": 4,
                "seed": seed,
            }
            checks.append({"identity": identity, "state": "passed"})
            for dose in (-0.025, 0.0, 0.025):
                measurements.append(
                    {
                        "identity": identity,
                        "dose": dose,
                        "outcomes": {"quality": float(seed) + dose},
                        "runtime_feature_records": (
                            [[[float(seed), 0.0, 0.0], [float(seed), 1.0, 0.0]]]
                            if dose == 0.0
                            else []
                        ),
                    }
                )
        shard = _write(
            root / f"shard-{campaign_index}.json",
            {
                "artifact_type": "verdiwm-ctrl-world-local-fingerprint-probe-result",
                "state": "ready",
                "campaign_id": campaign_id,
                "probe_id": "fshc_signed_history_gain",
                "hook_activation": {"state": "passed"},
                "measurements": measurements,
                "zero_identity_checks": checks,
            },
        )
        context = {
            "context_id": context_id,
            "episode_id": episode_id,
            "start_idx": 4,
            "seeds": [101, 202],
        }
        classifications.append(
            {
                "campaign_id": campaign_id,
                "context": context,
                "target_class": target,
                "selector_action": "execute",
                "selected_candidate": selected,
            }
        )
        receipts.append(
            {
                "campaign": {"campaign_id": campaign_id, "sha256": _sha256(campaign)},
                "selector_sha256": _sha256(selector),
            }
        )
        campaigns.append(campaign)
        selectors.append(selector)
        shards.append(shard)
    admission = _write(
        root / "admission.json",
        {
            "artifact_type": "verdiwm-ctrl-world-directional-data-admission-settlement",
            "state": "admitted",
            "candidate_training_licensed": True,
            "input_receipts": receipts,
            "classifications": classifications,
        },
    )
    return admission, campaigns, selectors, shards


def test_materializes_all_identities_from_multi_identity_shards(tmp_path: Path) -> None:
    admission, campaigns, selectors, shards = _frame(tmp_path)

    manifest = materialize(
        admission_path=admission,
        campaign_paths=campaigns,
        selector_paths=selectors,
        shard_paths=shards,
        output_root=tmp_path / "out",
    )

    bank = json.loads(Path(manifest["training_path"]).read_text(encoding="utf-8"))
    assert manifest["identity_count"] == 4
    assert manifest["record_count"] == 4
    assert manifest["target_counts"] == {"-1": 2, "0": 0, "1": 2}
    assert all(set(record) == {"features", "target", "counterfactual_losses"} for record in bank["records"])
    assert "1799" not in json.dumps(bank)


def test_rejects_promotion_episode_before_writing_bank(tmp_path: Path) -> None:
    admission, campaigns, selectors, shards = _frame(tmp_path, promotion_episode=True)

    with pytest.raises(FSHCFeatureBankError, match="PROMOTION_EPISODE_FORBIDDEN|LABEL_INVALID"):
        materialize(
            admission_path=admission,
            campaign_paths=campaigns,
            selector_paths=selectors,
            shard_paths=shards,
            output_root=tmp_path / "out",
        )

    assert not (tmp_path / "out").exists()
