from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.materialize_ctrl_world_cclvr_ablation_banks import (
    CCLVRAblationBankError,
    materialize,
)


def _write(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _source(tmp_path: Path, *, episode_id: str = "199") -> tuple[Path, Path]:
    records = []
    interactions = []
    for interaction in range(4):
        features = [
            [
                [float(interaction * 10 + invocation), float(history), 1.0]
                for history in range(2)
            ]
            for invocation in range(2)
        ]
        suffix = [4.0 + interaction, 5.0 + interaction, 6.0 + interaction]
        full = [suffix[0] + 2.0, suffix[1] + 4.0, suffix[2] + 1.0]
        records.append(
            {
                "features": features,
                "interaction_index": interaction,
                "arm_doses": [-0.99, 0.0, 0.99],
                "arm_losses": full,
                "arm_values": [full[1] - value for value in full],
                "target_arm": 0,
                "target_dose": -0.99,
            }
        )
        interactions.append(
            {
                "interaction_index": interaction,
                "arm_components": [
                    {
                        "loss": full[arm],
                        "suffix_mean_l1": suffix[arm],
                        "terminal_interaction_l1": 1.0 + arm,
                        "suffix_horizon_l1_slope": 0.1 * arm,
                    }
                    for arm in range(3)
                ],
                "arm_losses": full,
                "arm_values": [full[1] - value for value in full],
                "target_arm": 0,
                "target_dose": -0.99,
            }
        )
    source = _write(
        tmp_path / "source.json",
        {
            "artifact_type": "verdiwm-ctrl-world-cclvr-anonymous-local-value-bank",
            "state": "ready",
            "feature_shape": [2, 2, 3],
            "record_count": 4,
            "arm_doses": [-0.99, 0.0, 0.99],
            "local_return": {
                "minimum_benefit": 0.01,
                "suffix_mean_l1_weight": 1.0,
                "terminal_interaction_l1_weight": 2.0,
                "suffix_horizon_l1_slope_weight": 2.0,
            },
            "records": records,
        },
    )
    audit = _write(
        tmp_path / "audit.json",
        {
            "artifact_type": "verdiwm-ctrl-world-cclvr-value-bank-audit",
            "state": "ready",
            "forbidden_promotion_episodes": ["1799"],
            "record_count": 4,
            "rows": [
                {
                    "identity": {
                        "context_id": "development",
                        "episode_id": episode_id,
                        "start_idx": 0,
                        "seed": 101,
                    },
                    "training_record_offset": 0,
                    "training_record_count": 4,
                    "interactions": interactions,
                }
            ],
        },
    )
    return source, audit


def test_materializes_anonymous_episode_and_local_return_ablations(tmp_path: Path) -> None:
    source, audit = _source(tmp_path)
    output = tmp_path / "out"

    manifest = materialize(source_bank_path=source, audit_path=audit, output_root=output)

    episode = json.loads((output / "episode_suffix_mean.json").read_text(encoding="utf-8"))
    local = json.loads((output / "interaction_suffix_mean.json").read_text(encoding="utf-8"))
    full = json.loads((output / "interaction_terminal_horizon.json").read_text(encoding="utf-8"))
    assert manifest["state"] == "ready"
    assert episode["record_count"] == 1
    assert episode["feature_shape"] == [2, 3]
    assert episode["records"][0]["arm_losses"] == [5.5, 6.5, 7.5]
    assert local["record_count"] == full["record_count"] == 4
    assert local["records"][0]["arm_losses"] == [4.0, 5.0, 6.0]
    assert full["records"][0]["arm_losses"] == [6.0, 9.0, 7.0]
    rendered = json.dumps({"manifest": manifest, "episode": episode, "local": local, "full": full})
    assert not {"context_id", "episode_id", "seed", "start_idx"}.intersection(
        key for payload in (episode, local, full) for record in payload["records"] for key in record
    )
    assert "1799" not in rendered


def test_rejects_promotion_episode_and_existing_output(tmp_path: Path) -> None:
    source, audit = _source(tmp_path, episode_id="1799")
    with pytest.raises(CCLVRAblationBankError, match="PROMOTION_EPISODE_FORBIDDEN"):
        materialize(source_bank_path=source, audit_path=audit, output_root=tmp_path / "out")

    source, audit = _source(tmp_path)
    output = tmp_path / "exists"
    output.mkdir()
    with pytest.raises(CCLVRAblationBankError, match="OUTPUT_EXISTS"):
        materialize(source_bank_path=source, audit_path=audit, output_root=output)
