"""CPU contract tests for settled Agibot PSNR memory retrieval."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from wmloop.archive.store import ArchiveStore, SettledTrialRecord
from wmloop.retrieve import agibot_psnr_memory as memory_module


def _digest(letter: str) -> str:
    return letter * 64


def _record(archive: ArchiveStore, trial: str, *, library: str = "agibot-psnr-settlement-v1") -> None:
    receipt_hash = _digest("a" if trial.endswith("1") else "b")
    archive.record_settled_trial(
        SettledTrialRecord(
            trial_id=trial,
            proposal_id=f"proposal-{trial}",
            goal_id="agibot-psnr-goal",
            library_version=library,
            failure_context_ref=f"cas://sha256/{_digest('c')}",
            verdict_ref=f"cas://sha256/{_digest('d')}",
            receipt_ref=f"cas://sha256/{receipt_hash}",
            gpu_hours=0.0,
            hypothesis_hash=_digest("e"),
            impl_diff_hash=_digest("f"),
            evaluator_hash=_digest("1"),
            settlement_state="settled",
            receipt_hash=receipt_hash,
        )
    )


def _verified(trial: str, *, accepted: bool, dataset: str = "2" * 64):
    return SimpleNamespace(
        trial_id=trial,
        proposal_id=f"proposal-{trial}",
        goal_id="agibot-psnr-goal",
        receipt_ref=f"cas://sha256/{_digest('a' if trial.endswith('1') else 'b')}",
        verdict_ref=f"cas://sha256/{_digest('d')}",
        measurement_ref=f"cas://sha256/{_digest('7')}",
        verdict={
            "verdict": "ACCEPT" if accepted else "REJECT",
            "reason": "PSNR_LOWER_BOUND_ABOVE_2_DB" if accepted else "PSNR_LOWER_BOUND_NOT_ABOVE_2_DB",
        },
        receipt={"effect_verified": accepted},
        measurement={
            "dataset_sha256": dataset,
            "support_sha256": "3" * 64,
            "evaluator_digest": "4" * 64,
            "checkpoint_sha256": {"baseline": "5" * 64, "candidate": "6" * 64},
            "statistics": {
                "mean_paired_delta_db": 3.0 if accepted else 1.0,
                "lower_bound_db": 2.5 if accepted else 1.5,
                "upper_bound_db": 3.5 if accepted else 2.0,
            },
            "report_only_metrics": {"ssim": {"mean_delta": 0.1}, "lpips": {"state": "not_measured"}},
        },
        context={"hypothesis": "memory hypothesis", "method": "memory method"},
    )


def test_retrieval_returns_only_verified_agibot_settlements(monkeypatch, tmp_path: Path):
    archive_db = tmp_path / "archive.db"
    archive = ArchiveStore(archive_db)
    _record(archive, "trial-1")
    _record(archive, "trial-2")
    _record(archive, "other", library="other-library")
    cas_root = tmp_path / "cas-store"
    (cas_root / "cas").mkdir(parents=True)

    monkeypatch.setattr(
        memory_module,
        "verify_agibot_psnr_settled_receipt",
        lambda **kwargs: _verified(kwargs["receipt_ref"].endswith("a" * 64) and "trial-1" or "trial-2", accepted=kwargs["receipt_ref"].endswith("a" * 64)),
    )
    rows = memory_module.retrieve_agibot_psnr_memories(
        archive_db=archive_db, cas_root=cas_root, replay_measurement=False
    )
    assert [row.trial_id for row in rows] == ["trial-1", "trial-2"]
    assert rows[0].effect_verified is True
    assert rows[1].effect_verified is False
    assert rows[0].to_dict()["evidence"]["measurement_ref"].startswith("cas://sha256/")


def test_retrieval_filter_and_empty_uninitialized_store(tmp_path: Path):
    assert memory_module.retrieve_agibot_psnr_memories(
        archive_db=tmp_path / "missing.db", cas_root=tmp_path / "missing-cas"
    ) == ()
    with pytest.raises(memory_module.AgibotPSNRMemoryError, match="FILTER_INVALID"):
        memory_module.retrieve_agibot_psnr_memories(
            archive_db=tmp_path / "missing.db", cas_root=tmp_path / "missing-cas", dataset_sha256="bad"
        )
