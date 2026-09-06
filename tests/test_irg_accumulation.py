from __future__ import annotations

import json
from pathlib import Path

import pytest

from wmloop.execute.autonomous_pipeline import _probe_irg_batch
from wmloop.execute.irg_accumulation import (
    IRGAccumulationError,
    accumulate_irg_observations,
)


def _batch(*, state: str = "exploratory", suffix: str = "a") -> dict[str, object]:
    return {
        "model_family": "fixture-world-model",
        "capability_class": "predictive-video",
        "goal_schema": "fixture-protected-metrics-v1",
        "probe_id": "action-dose-probe-v1",
        "protocol_hash": "sha256:" + "1" * 64,
        "state": state,
        "outcome_names": ["quality", "action_following"],
        "outcome_weights": [1.0, 2.0],
        "baseline_repeats": [[1.0, 2.0], [1.2, 2.2]],
        "dose_observations": {
            "action_scale": {
                -0.5: [[0.5, 1.8], [0.7, 2.0]],
                0.5: [[1.5, 2.3], [1.7, 2.5]],
            }
        },
        "paired_identity": {"seeds": [7, 11], "horizons": [30, 30]},
        "failure_signatures": ["action_binding"],
        "anti_conditions": ["long_horizon_only"],
        "evidence_refs": ["sha256:" + suffix * 64],
    }


def test_first_round_builds_irg_without_prior_input(tmp_path: Path) -> None:
    result = accumulate_irg_observations(
        batch=_batch(), output_root=tmp_path / "round-1", cas_root=tmp_path / "cas"
    )
    assert result["sequence"] == 1
    assert result["routing_authority"] == "diagnostic_ranking_only"
    assert result["active_verdict_unchanged"] is True
    chart = json.loads((tmp_path / "round-1/response-chart.json").read_text())
    assert chart["artifact_type"] == "verdiwm-irg-response-chart"
    assert chart["repeat_count"] == 2


def test_second_round_appends_parent_lineage(tmp_path: Path) -> None:
    first = accumulate_irg_observations(
        batch=_batch(), output_root=tmp_path / "round-1", cas_root=tmp_path / "cas"
    )
    second = accumulate_irg_observations(
        batch=_batch(state="null", suffix="b"),
        output_root=tmp_path / "round-2",
        prior_manifest=Path(str(first["manifest_path"])),
        cas_root=tmp_path / "cas",
    )
    assert second["sequence"] == 2
    assert second["parent_accumulation_id"] == first["accumulation_id"]
    assert [row["state"] for row in second["observations"]] == ["exploratory", "null"]


def test_incomplete_paired_repeats_fail_closed(tmp_path: Path) -> None:
    batch = _batch()
    batch["dose_observations"] = {"action_scale": {0.5: [[1.5, 2.3]]}}
    with pytest.raises(IRGAccumulationError, match="PAIRED_IDENTITY_MISMATCH"):
        accumulate_irg_observations(batch=batch, output_root=tmp_path / "invalid")


@pytest.mark.parametrize("state", ["abstained", "disputed"])
def test_unsettled_evidence_cannot_become_active_routing(
    tmp_path: Path, state: str
) -> None:
    result = accumulate_irg_observations(
        batch=_batch(state=state), output_root=tmp_path / state
    )
    assert result["routing_state"] == "abstain"
    assert result["active_verdict_unchanged"] is True


def test_pipeline_adapter_uses_only_explicit_standard_observation() -> None:
    manifest = {
        "probe_id": "action-dose-probe-v1",
        "model_family": "fixture-world-model",
        "result_ref": "cas://sha256/" + "2" * 64,
        "receipt_ref": "cas://sha256/" + "3" * 64,
        "failure_signatures": ["action_binding"],
    }
    assert _probe_irg_batch(probe_manifest=manifest, probe_result={}) is None
    result = {"irg_observation": _batch()}
    projected = _probe_irg_batch(probe_manifest=manifest, probe_result=result)
    assert projected is not None
    assert projected["failure_signatures"] == ["action_binding"]
    assert len(projected["evidence_refs"]) == 3
