from __future__ import annotations

from pathlib import Path

from wmloop.contracts import validate_document
from wmloop.execute.agibot_psnr_contract import (
    AGIBOT_PSNR_ROLLOUT_BUILDER_UNAVAILABLE,
    build_agibot_psnr_execution_status,
)
from wmloop.execute.pipeline.retrieval import _runtime_retrieval_manifest


ROOT = Path(__file__).resolve().parents[1]


def test_agibot_psnr_execution_contract_is_explicitly_blocked_until_rollout_builder_exists():
    status = build_agibot_psnr_execution_status()

    validate_document("agibot_psnr_execution_contract", status, root=ROOT)
    assert status["state"] == "unavailable"
    assert status["reason"] == AGIBOT_PSNR_ROLLOUT_BUILDER_UNAVAILABLE
    assert status["quality_eligible"] is False
    assert "paired RGB rollouts" in status["claim_boundary"]


def test_runtime_retrieval_manifest_preserves_psnr_status_and_settled_memory():
    status = build_agibot_psnr_execution_status()
    manifest = _runtime_retrieval_manifest(
        {
            "state": "cold_start",
            "capsule": {"query": {}, "match_count": 0, "route": "bounded"},
            "agibot_psnr_execution": status,
            "agibot_psnr_memory": {
                "state": "matched",
                "match_count": 1,
                "matches": [{"settlement_state": "settled"}],
            },
        }
    )

    assert manifest["agibot_psnr_execution"] == status
    assert manifest["agibot_psnr_memory"]["state"] == "matched"
    assert manifest["agibot_psnr_memory"]["matches"][0]["settlement_state"] == "settled"
