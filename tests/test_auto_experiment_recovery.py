"""Recovery must derive its authority from the settled receipt, not local JSON."""

import json
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from wmloop.archive.store import ArchiveStore, ContentAddressedStore, SettledTrialRecord
from wmloop.execute.auto_experiment import AutoExperimentError, _recover_archived_run, cleanup_auto_experiment_scratch
from wmloop.storage import canonical_json_bytes
from wmloop.execute import auto_experiment


@pytest.fixture
def recovery(tmp_path: Path) -> dict:
    cas = ContentAddressedStore(tmp_path / "cas")
    archive = ArchiveStore(tmp_path / "archive.db")
    plan = {"trial_id": "trial", "campaign_id": "campaign", "artifacts": [], "cleanup_policy": "retain"}
    core = {
        "artifact_type": "verdiwm-auto-experiment-receipt", "fencing_token": 2,
        "state": "terminal", "settlement_state": "settled", "trial_id": "trial",
        "campaign_id": "campaign", "archive_trial_id": "auto-trial", "plan_sha256": "p" * 64,
        "trial_signature": "s" * 64, "stage": "confirm", "verdict": {"verdict": "PASS"},
        "artifact_refs": {},
        "support_refs": {},
    }
    core["verdict_ref"] = cas.put_bytes(canonical_json_bytes(core["verdict"]), media_type="application/json").uri
    core["failure_context_ref"] = cas.put_bytes(b'{}', media_type="application/json").uri
    ref = cas.put_bytes(canonical_json_bytes(core), media_type="application/json")
    archive.record_settled_trial(SettledTrialRecord(
        trial_id="auto-trial", proposal_id="proposal", goal_id="campaign", library_version="v1",
        failure_context_ref=ref.uri, verdict_ref=ref.uri, receipt_ref=ref.uri, gpu_hours=0,
        hypothesis_hash="a" * 64, impl_diff_hash="b" * 64, evaluator_hash="c" * 64,
        settlement_state="settled", receipt_hash=ref.sha256,
    ))
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps({**core, "receipt_ref": ref.uri, "receipt_hash": ref.sha256}), encoding="utf-8")
    return dict(receipt_path=path, manifest_path=tmp_path / "manifest.json", destination=tmp_path,
                plan=plan, archive=archive, cas=cas, plan_sha256="p" * 64,
                trial_signature="s" * 64, archive_trial_id="auto-trial")


def test_recovery_accepts_receipted_projection(recovery: dict) -> None:
    assert _recover_archived_run(**recovery)["verdict"] == "PASS"


@pytest.mark.parametrize("field,value", [("verdict", {"verdict": "VOID"}), ("plan_sha256", "x" * 64)])
def test_recovery_rejects_changed_local_receipt(recovery: dict, field: str, value: object) -> None:
    path = recovery["receipt_path"]
    receipt = json.loads(path.read_text(encoding="utf-8"))
    receipt[field] = value
    path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(AutoExperimentError, match="RECOVERY_RECEIPT_MISMATCH"):
        _recover_archived_run(**recovery)
    assert not recovery["manifest_path"].exists()


def test_recovery_rejects_different_plan(recovery: dict) -> None:
    recovery["plan_sha256"] = "x" * 64
    with pytest.raises(AutoExperimentError, match="RECOVERY_RECEIPT_MISMATCH"):
        _recover_archived_run(**recovery)


def test_recovery_rejects_missing_cas(recovery: dict) -> None:
    receipt = json.loads(recovery["receipt_path"].read_text(encoding="utf-8"))
    digest = receipt["receipt_hash"]
    (recovery["cas"]._cas / digest[:2] / digest).unlink()
    with pytest.raises(AutoExperimentError, match="RECOVERY_RECEIPT_INVALID"):
        _recover_archived_run(**recovery)


def _scratch(recovery: dict, token: int, **overrides: object) -> Path:
    path = recovery["destination"] / "scratch" / "trial" / f"attempt-{token:04d}"
    path.mkdir(parents=True)
    receipt = json.loads(recovery["receipt_path"].read_text())
    marker = {"artifact_type": "verdiwm-auto-experiment-scratch", "campaign_id": "campaign",
              "trial_id": "trial", "fencing_token": token, "state": "settled",
              "cleanup_eligible": True, "archive_recorded": True,
              "receipt_ref": receipt["receipt_ref"], "required_artifact_count": 0,
              "archived_artifact_count": 0, **overrides}
    (path / ".verdiwm-scratch.json").write_text(json.dumps(marker))
    return path


def test_recovery_only_cleans_the_settled_attempt(recovery: dict) -> None:
    recovery["plan"]["cleanup_policy"] = "archive_then_delete"
    old = _scratch(recovery, 1)
    current = _scratch(recovery, 2, state="running")
    _recover_archived_run(**recovery)
    assert old.exists()
    assert not current.exists()


@pytest.mark.parametrize("override", [{"campaign_id": "other"}, {"trial_id": "other"}, {"fencing_token": 1}])
def test_recovery_retains_mismatched_scratch(recovery: dict, override: dict) -> None:
    recovery["plan"]["cleanup_policy"] = "archive_then_delete"
    current = _scratch(recovery, 2, **override)
    _recover_archived_run(**recovery)
    assert current.exists()


def test_periodic_cleanup_requires_matching_receipt_and_attempt(recovery: dict) -> None:
    old = _scratch(recovery, 1)
    current = _scratch(recovery, 2)
    report = cleanup_auto_experiment_scratch(run_root=recovery["destination"] / "scratch", older_than_hours=0,
        apply=True, archive_db=recovery["archive"]._path, cas_root=recovery["cas"]._root)
    assert old.exists()
    assert not current.exists()
    assert report["deleted_count"] == 1


def test_periodic_cleanup_rejects_unregistered_receipt(recovery: dict) -> None:
    path = _scratch(recovery, 2)
    marker_path = path / ".verdiwm-scratch.json"
    marker = json.loads(marker_path.read_text())
    core = json.loads(recovery["cas"].read_bytes(marker["receipt_ref"]))
    core["verdict"] = {"verdict": "different"}
    marker["receipt_ref"] = recovery["cas"].put_bytes(canonical_json_bytes(core), media_type="application/json").uri
    marker_path.write_text(json.dumps(marker))
    report = cleanup_auto_experiment_scratch(run_root=path.parent.parent, older_than_hours=0,
        apply=True, archive_db=recovery["archive"]._path, cas_root=recovery["cas"]._root)
    assert path.exists()
    assert report["retained"][0]["reason"] == "ARCHIVE_RECORD_MISSING"


def test_recovery_retains_missing_artifacts(recovery: dict) -> None:
    recovery["plan"].update(cleanup_policy="archive_then_delete", artifacts=["model.bin"])
    path = _scratch(recovery, 2, state="running")
    assert _recover_archived_run(**recovery)["cleanup"]["reason"] == "REQUIRED_ARTIFACTS_MISSING"
    assert path.exists()


def test_plan_digest_describes_the_validated_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "plan.json"
    original = b'{"trial_id": "first"}'
    path.write_bytes(original)
    monkeypatch.setattr(auto_experiment, "validate_document", lambda *_: None)
    def mutate_source(plan: dict, *, workspace_root: Path) -> None:
        assert plan["trial_id"] == "first"
        path.write_bytes(b'{"trial_id": "second"}')
    monkeypatch.setattr(auto_experiment, "_validate_plan_semantics", mutate_source)
    plan, digest = auto_experiment._load_plan(path, workspace_root=tmp_path)
    assert plan["trial_id"] == "first"
    assert digest == hashlib.sha256(original).hexdigest()


def test_output_lock_prevents_concurrent_start_and_releases_after_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    entered, release = threading.Event(), threading.Event()
    calls = []

    def run_locked(**kwargs: object) -> dict:
        calls.append(kwargs)
        entered.set()
        assert release.wait(timeout=5)
        raise RuntimeError("injected failure")

    monkeypatch.setattr(auto_experiment, "_run_auto_experiment_locked", run_locked)
    args = dict(plan_path=tmp_path / "plan.json", output_root=tmp_path / "nested" / "run",
                workspace_root=tmp_path, archive_db=tmp_path / "archive.db", cas_root=tmp_path / "cas")
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(auto_experiment.run_auto_experiment, **args)
        assert entered.wait(timeout=5)
        with pytest.raises(AutoExperimentError, match="AUTO_EXPERIMENT_ALREADY_RUNNING"):
            auto_experiment.run_auto_experiment(**args)
        assert len(calls) == 1
        release.set()
        with pytest.raises(RuntimeError, match="injected failure"):
            first.result(timeout=5)
    with pytest.raises(RuntimeError, match="injected failure"):
        auto_experiment.run_auto_experiment(**args)
    assert len(calls) == 2
