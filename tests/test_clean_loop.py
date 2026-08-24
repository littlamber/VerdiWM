from pathlib import Path

from adapters.fixture_world import FixtureWorldAdapter
from verdi_core.loop import run_loop


class AlternateFixtureAdapter:
    """Second implementation proving the Kernel does not depend on one adapter."""

    adapter_id = "alternate-fixture"
    version = "1"

    def inspect(self):
        return {
            "model_id": "alternate-world-v1",
            "revision": "r1",
            "capabilities": ["inference", "evaluation"],
            "hooks": [],
            "evaluator_id": "alternate-verifier-v1",
        }

    def probe(self, probe_id):
        return {"response_digest": "sha256:" + "2" * 64, "uncertainty": "bounded"}

    def intervene(self, hypothesis):
        return hypothesis

    def evaluate(self, intervention, split):
        return {"outcome": "abstain", "delta": 0.0, "protected_ok": True, "split": split}


def test_loop_retains_positive_null_and_harmful(tmp_path: Path) -> None:
    summary = run_loop(FixtureWorldAdapter(), state_root=tmp_path)
    assert summary["state"] == "settled"
    assert summary["outcomes"] == ["null", "confirmed_positive", "harmful"]
    assert len((tmp_path / "knowledge" / "knowledge.jsonl").read_text().splitlines()) == 3


def test_loop_is_idempotent_for_same_evidence(tmp_path: Path) -> None:
    adapter = FixtureWorldAdapter()
    run_loop(adapter, state_root=tmp_path)
    run_loop(adapter, state_root=tmp_path)
    assert len((tmp_path / "knowledge" / "knowledge.jsonl").read_text().splitlines()) == 3


def test_second_adapter_uses_same_kernel_contract(tmp_path: Path) -> None:
    summary = run_loop(AlternateFixtureAdapter(), state_root=tmp_path)
    assert summary["capability"]["model_id"] == "alternate-world-v1"
    assert summary["outcomes"] == ["abstain", "abstain", "abstain"]
