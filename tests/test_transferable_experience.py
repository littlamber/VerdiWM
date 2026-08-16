from __future__ import annotations

from pathlib import Path

import pytest

from wmloop.geometry import (
    EffectContext,
    EffectMemory,
    EffectRecord,
    GeometryValidationError,
    build_transferable_experience,
)


def _effect(status: str = "confirmed") -> EffectRecord:
    return EffectRecord(
        record_id=f"{status}-1",
        primitive="first_frame_anchor",
        context=EffectContext(
            campaign_id="ctrl-world-confirm-1",
            backbone_family="ctrl-world",
            capability_class="predictive-video",
            goal_schema="long-horizon-v1",
            outcome_schema="quality-v1",
            chart_id="anchor-chart-1",
            data_regime="heldout",
            horizons=(16, 32),
        ),
        status=status,
        mean_effect=0.4 if status == "confirmed" else -0.2,
        standard_error=0.05,
        lower_bound=0.2 if status == "confirmed" else -0.3,
        goal_threshold=0.0,
        validity_gates={"official": status == "confirmed"},
        replication_count=2 if status == "confirmed" else 1,
        evidence_refs=("cas://gate",),
    )


def _certificate(status: str = "licensed") -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_type": "verdiwm-transfer-certificate",
        "status": status,
        "terms": {"compile": True, "overlap": status == "licensed"},
    }


def test_only_fully_licensed_confirmed_effect_becomes_reusable_prior() -> None:
    licensed = build_transferable_experience(
        _effect(), transfer_certificate=_certificate(), anti_conditions=("short_horizon_shift",)
    )
    assert licensed["transfer_state"] == "licensed_prior"
    assert licensed["transfer_authority"] == "licensed_prior"
    assert licensed["anti_conditions"] == ["short_horizon_shift"]

    local = build_transferable_experience(_effect())
    assert local["transfer_state"] == "local_only"
    assert local["transfer_authority"] == "ranking_only"


def test_negative_effect_is_retained_without_becoming_positive_transfer() -> None:
    negative = build_transferable_experience(
        _effect("rejected"), transfer_certificate=_certificate("abstain")
    )
    assert negative["transfer_state"] == "abstained"
    assert negative["effect"]["status"] == "rejected"


def test_memory_writes_derived_transfer_view(tmp_path: Path) -> None:
    memory = EffectMemory((_effect(), _effect("rejected")))
    output = memory.write_transferable_jsonl(
        tmp_path / "transfer.jsonl",
        certificates={"confirmed-1": _certificate()},
    )
    rows = output.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 2
    assert '"transfer_state": "licensed_prior"' in rows[0]


def test_certificate_terms_must_be_boolean() -> None:
    with pytest.raises(GeometryValidationError, match="CERTIFICATE_TERMS_INVALID"):
        build_transferable_experience(
            _effect(),
            transfer_certificate={
                "artifact_type": "verdiwm-transfer-certificate",
                "status": "licensed",
                "terms": {"compile": "yes"},
            },
        )
