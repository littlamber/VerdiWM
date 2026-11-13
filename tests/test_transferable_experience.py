from __future__ import annotations

from pathlib import Path
import json

import pytest

from wmloop.geometry import (
    EffectContext,
    EffectMemory,
    EffectRecord,
    GeometryValidationError,
    build_transferable_experience,
    evaluate_transfer_certificate,
)
from wmloop.geometry.types import CompileReceipt


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


def _receipt() -> CompileReceipt:
    return CompileReceipt(
        descriptor_name="first_frame_anchor", backbone_family="target",
        capability_class="video", compiled=True, hook_type="H4", dose_unit="fraction",
        dose_direction=0.2, semantic_obligations=(), invariant_checks={"same_split": True},
        blockers=(),
    )


def _certificate(status: str = "licensed", **overrides: object) -> dict[str, object]:
    inputs = dict(
        receipt=_receipt(), support_overlap=0.8 if status == "licensed" else 0.2,
        effective_sample_size=12, alignment_error=0.1, sign_agreement=0.9,
        calibrated_lower_bound=0.3, minimum_overlap=0.5,
        minimum_effective_sample_size=8, maximum_alignment_error=0.2,
        minimum_sign_agreement=0.8, goal_threshold=0.1,
    )
    inputs.update(overrides)
    return evaluate_transfer_certificate(**inputs).to_dict()


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
    certificate = _certificate()
    certificate["terms"]["compile"] = "yes"
    with pytest.raises(GeometryValidationError, match="CERTIFICATE_TERMS_INVALID"):
        build_transferable_experience(_effect(), transfer_certificate=certificate)


@pytest.mark.parametrize("term", ["compile", "overlap", "effective_sample_size", "alignment", "sign_agreement", "effect_lower_bound"])
def test_certificate_cannot_omit_a_gate(term: str) -> None:
    certificate = _certificate()
    del certificate["terms"][term]
    with pytest.raises(GeometryValidationError, match="CERTIFICATE_INVALID"):
        build_transferable_experience(_effect(), transfer_certificate=certificate)


@pytest.mark.parametrize("field,value", [
    ("descriptor_name", "different_primitive"), ("compile_pass", False), ("schema_version", True),
    ("status", "abstain"), ("abstention_reasons", ["overlap"]),
    ("support_overlap", True), ("alignment_error", "0.1"),
    ("calibrated_lower_bound", -1), ("goal_threshold", float("nan")),
])
def test_inconsistent_certificate_is_rejected(field: str, value: object) -> None:
    certificate = _certificate()
    certificate[field] = value
    with pytest.raises(GeometryValidationError, match="CERTIFICATE_INVALID"):
        build_transferable_experience(_effect(), transfer_certificate=certificate)


def test_abstention_reasons_survive_sorted_json_and_reject_duplicates() -> None:
    certificate = _certificate("abstain")
    certificate["terms"]["alignment"] = False
    certificate["abstention_reasons"].append("alignment")
    reordered = json.loads(json.dumps(certificate, sort_keys=True))
    assert build_transferable_experience(_effect(), transfer_certificate=reordered)["transfer_state"] == "abstained"
    reordered["abstention_reasons"].append("alignment")
    with pytest.raises(GeometryValidationError, match="CERTIFICATE_INVALID"):
        build_transferable_experience(_effect(), transfer_certificate=reordered)


@pytest.mark.parametrize("value", [True, "0.8", float("nan"), float("inf")])
def test_certificate_evaluation_rejects_invalid_numbers(value: object) -> None:
    with pytest.raises(GeometryValidationError, match="TRANSFER_VALUE_INVALID"):
        _certificate(support_overlap=value)


def test_certificate_evaluation_rechecks_mutated_receipt() -> None:
    receipt = _receipt()
    receipt.invariant_checks["same_split"] = "yes"
    with pytest.raises(GeometryValidationError, match="TRANSFER_COMPILE_RECEIPT_INVALID"):
        _certificate(receipt=receipt)
