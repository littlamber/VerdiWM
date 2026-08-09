from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from wmloop.archive.store import ArchiveStore
from wmloop.execute import auto_experiment
from wmloop.execute.auto_experiment import AutoExperimentError
from wmloop.execute.budget import BudgetError, BudgetLedger, BudgetPolicy
from wmloop.execute.gpu_lease import GpuLeaseError, GpuLeaseManager
from wmloop.verify.auto_experiment import verify_auto_experiment_result


ROOT = Path(__file__).resolve().parents[1]


def _snapshot(
    *,
    memory: int = 0,
    utilization: int = 0,
    apps: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "gpus": [
            {
                "index": 4,
                "uuid": "GPU-test-4",
                "name": "Test GPU",
                "memory_used_mib": memory,
                "utilization_gpu_percent": utilization,
            }
        ],
        "compute_apps": apps or [],
    }


def test_gpu_lease_selects_physical_gpu_and_rejects_busy(tmp_path: Path) -> None:
    manager = GpuLeaseManager(
        lock_root=tmp_path / "locks", snapshot_provider=lambda: _snapshot()
    )
    with manager.acquire([4]) as lease:
        assert lease.index == 4
        assert lease.uuid == "GPU-test-4"
        assert lease.environment()["CUDA_VISIBLE_DEVICES"] == "4"
        held = GpuLeaseManager(
            lock_root=tmp_path / "locks", snapshot_provider=lambda: _snapshot()
        )
        with pytest.raises(GpuLeaseError, match="LEASE_HELD"):
            held.acquire([4])

    busy = GpuLeaseManager(
        lock_root=tmp_path / "busy-locks",
        snapshot_provider=lambda: _snapshot(memory=4096, utilization=90),
    )
    with pytest.raises(GpuLeaseError, match="MEMORY_BUSY"):
        busy.acquire([4])


def test_verifier_requires_matching_gpu_activity_and_metric_gates() -> None:
    sampling = {
        "samples": [
            {
                "status": "ready",
                "gpu_uuid": "GPU-test-4",
                "phase": "start",
                "memory_used_mib": 10,
                "utilization_gpu_percent": 0,
            },
            {
                "status": "ready",
                "gpu_uuid": "GPU-test-4",
                "phase": "during",
                "memory_used_mib": 80,
                "utilization_gpu_percent": 65,
            },
        ]
    }
    result = {
        "schema_version": 1,
        "artifact_type": "verdiwm-auto-experiment-result",
        "state": "ready",
        "device": {"type": "cuda", "gpu_uuid": "GPU-test-4"},
        "metrics": {"score": 0.9},
    }
    verdict = verify_auto_experiment_result(
        result=result,
        metric_gates=(
            {"metric": "score", "role": "primary", "operator": "gte", "threshold": 0.8},
        ),
        expected_gpu_uuid="GPU-test-4",
        gpu_sampling=sampling,
        stage="smoke",
    )
    assert verdict["verdict"] == "PASS"
    assert verdict["gpu_activity"]["verified"] is True

    mismatch = verify_auto_experiment_result(
        result={**result, "device": {"type": "cuda", "gpu_uuid": "GPU-other"}},
        metric_gates=(
            {"metric": "score", "role": "primary", "operator": "gte", "threshold": 0.8},
        ),
        expected_gpu_uuid="GPU-test-4",
        gpu_sampling=sampling,
        stage="smoke",
    )
    assert mismatch["verdict"] == "VOID"
    assert any(
        blocker["code"] == "RESULT_GPU_UUID_MISMATCH"
        for blocker in mismatch["blockers"]
    )


def test_long_trial_uses_configurable_resource_policy(tmp_path: Path) -> None:
    ledger = BudgetLedger(
        tmp_path / "long-trial-budget.db",
        BudgetPolicy(
            total_gpu_hours=800.0,
            max_trial_gpu_hours=500.0,
            high_trial_limit=4,
            require_high_cost_approval=False,
        ),
    )

    admission = ledger.admit(
        "long-training-method",
        cost_class="high",
        estimated_gpu_hours=300.0,
    )

    assert admission.state == "admitted"
    assert auto_experiment._cost_class(300.0) == "high"


def test_long_trial_policy_remains_resource_admission_only(
    tmp_path: Path,
) -> None:
    ledger = BudgetLedger(
        tmp_path / "bounded-long-trial.db",
        BudgetPolicy(
            total_gpu_hours=800.0,
            max_trial_gpu_hours=250.0,
            require_high_cost_approval=False,
        ),
    )

    with pytest.raises(BudgetError, match="TRIAL_COST_CAP_EXCEEDED"):
        ledger.admit(
            "too-large-for-declared-resource-policy",
            cost_class="high",
            estimated_gpu_hours=300.0,
        )


def test_runtime_placeholders_expand_in_environment() -> None:
    substitutions = {
        "scratch_dir": "/run/scratch",
        "workspace_root": "/model",
        "output_root": "/run",
        "gpu_index": "4",
        "gpu_uuid": "GPU-test-4",
    }

    assert (
        auto_experiment._expand_token("{scratch_dir}/cache/{gpu_uuid}", substitutions)
        == "/run/scratch/cache/GPU-test-4"
    )


def _plan(
    path: Path,
    *,
    trial_id: str = "control-test",
    command: list[str] | None = None,
    environment: dict[str, str] | None = None,
) -> Path:
    payload = {
        "schema_version": 1,
        "artifact_type": "verdiwm-auto-experiment-plan",
        "campaign_id": "control-plane-tests",
        "trial_id": trial_id,
        "objective": "Prove this bounded control-plane trial is worth its cost.",
        "hypothesis": "A deterministic command will produce a valid archived result.",
        "selection_reason": "The trial is the smallest useful test of the transaction boundary.",
        "falsification_criterion": "Any missing artifact, failed gate, or wrong GPU voids the trial.",
        "stage": "smoke",
        "command": command or ["test-command"],
        "working_directory": ".",
        "allowed_gpu_indices": [4],
        "estimated_gpu_hours": 0.01,
        "total_budget_gpu_hours": 0.05,
        "timeout_seconds": 10,
        "gpu_wait_seconds": 0,
        "sample_interval_seconds": 0.1,
        "result_path": "result.json",
        "artifacts": ["result.json"],
        "metric_gates": [
            {"metric": "score", "role": "primary", "operator": "gte", "threshold": 0.5}
        ],
        "environment": environment or {},
        "cleanup_policy": "retain",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class _FakeLease:
    index = 4
    uuid = "GPU-test-4"

    def __enter__(self) -> "_FakeLease":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def environment(self) -> dict[str, str]:
        return {
            "CUDA_VISIBLE_DEVICES": "4",
            "VERDIWM_PHYSICAL_GPU_INDEX": "4",
            "VERDIWM_PHYSICAL_GPU_UUID": self.uuid,
        }

    def to_document(self) -> dict[str, object]:
        return {"index": 4, "uuid": self.uuid, "name": "Test GPU", "lock_path": "fake"}


class _FakeLeaseManager:
    def __init__(self, **_: object) -> None:
        pass

    def acquire(self, *_: object, **__: object) -> _FakeLease:
        return _FakeLease()


class _UnavailableLeaseManager:
    def __init__(self, **_: object) -> None:
        pass

    def acquire(self, *_: object, **__: object) -> _FakeLease:
        raise GpuLeaseError("GPU_LEASE_UNAVAILABLE:4:COMPUTE_APP_PRESENT")


class _FakeSampler:
    def __init__(self, **_: object) -> None:
        pass

    def capture(self, *, label: str, callback):
        return callback()

    def to_document(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "artifact_type": "wmloop-gpu-sampling-curve",
            "state": "ready",
            "samples": [
                {
                    "status": "ready",
                    "gpu_uuid": "GPU-test-4",
                    "phase": "start",
                    "memory_used_mib": 0,
                    "utilization_gpu_percent": 0,
                },
                {
                    "status": "ready",
                    "gpu_uuid": "GPU-test-4",
                    "phase": "during",
                    "memory_used_mib": 128,
                    "utilization_gpu_percent": 80,
                },
            ],
        }


class _FakeBackend:
    def run(self, *, worktree: Path, command, environment, timeout_seconds: float):
        scratch = Path(environment["VERDIWM_TRIAL_SCRATCH"])
        result = {
            "schema_version": 1,
            "artifact_type": "verdiwm-auto-experiment-result",
            "state": "ready",
            "device": {"type": "cuda", "gpu_uuid": "GPU-test-4"},
            "metrics": {"score": 0.9},
        }
        (scratch / "result.json").write_text(json.dumps(result), encoding="utf-8")
        from wmloop.execute.backends import CommandExecutionResult

        return CommandExecutionResult(b"ok\n", b"", 0, False, 0.01)


class _TimeoutBackend:
    def run(self, **_: object):
        from wmloop.execute.backends import CommandExecutionResult

        return CommandExecutionResult(b"", b"timed out\n", -9, True, 10.0)


class _EnvironmentAssertingBackend(_FakeBackend):
    def run(self, *, worktree: Path, command, environment, timeout_seconds: float):
        scratch = Path(environment["VERDIWM_TRIAL_SCRATCH"])
        assert environment["RESULT_ROOT"] == str(scratch)
        assert environment["CACHE_ROOT"] == str(scratch / "cache" / "GPU-test-4" / "4")
        assert environment["BOUND_WORKSPACE"] == str(ROOT)
        assert environment["BOUND_OUTPUT"].endswith("/run")
        return super().run(
            worktree=worktree,
            command=command,
            environment=environment,
            timeout_seconds=timeout_seconds,
        )


def _patch_success_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(auto_experiment, "GpuLeaseManager", _FakeLeaseManager)
    monkeypatch.setattr(auto_experiment, "GpuSamplingRecorder", _FakeSampler)
    monkeypatch.setattr(auto_experiment, "LocalSubprocessBackend", _FakeBackend)
    monkeypatch.setattr(
        auto_experiment,
        "run_gpu_exclusivity_audit",
        lambda **kwargs: {
            "report_path": str(tmp_path / "audit" / "gpu-exclusivity-audit.json")
        },
    )
    monkeypatch.setattr(
        auto_experiment,
        "verify_gpu_exclusivity_ready",
        lambda *args, **kwargs: {"state": "ready", "m4_launch_allowed": False},
    )


def test_runtime_placeholders_reach_executed_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_success_runtime(monkeypatch, tmp_path)
    monkeypatch.setattr(
        auto_experiment, "LocalSubprocessBackend", _EnvironmentAssertingBackend
    )
    plan_path = _plan(
        tmp_path / "plan.json",
        trial_id="environment-expansion",
        environment={
            "RESULT_ROOT": "{scratch_dir}",
            "CACHE_ROOT": "{scratch_dir}/cache/{gpu_uuid}/{gpu_index}",
            "BOUND_WORKSPACE": "{workspace_root}",
            "BOUND_OUTPUT": "{output_root}",
        },
    )

    manifest = auto_experiment.run_auto_experiment(
        plan_path=plan_path,
        output_root=tmp_path / "run",
        workspace_root=ROOT,
        archive_db=tmp_path / "archive.db",
        cas_root=tmp_path / "store",
        lock_root=tmp_path / "locks",
        budget_total_gpu_hours=0.1,
    )

    assert manifest["verdict"] == "PASS"
    assert manifest["budget"]["ledger_total_gpu_hours"] == 0.1


def test_run_auto_experiment_settles_and_is_idempotent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_success_runtime(monkeypatch, tmp_path)
    plan_path = _plan(tmp_path / "plan.json")
    output = tmp_path / "run"
    archive_db = tmp_path / "archive.db"
    manifest = auto_experiment.run_auto_experiment(
        plan_path=plan_path,
        output_root=output,
        workspace_root=ROOT,
        archive_db=archive_db,
        cas_root=tmp_path / "store",
        lock_root=tmp_path / "locks",
        budget_total_gpu_hours=0.1,
    )
    assert manifest["settlement_state"] == "settled"
    assert manifest["verdict"] == "PASS"
    assert (output / "receipts" / "control-test.json").is_file()
    assert ArchiveStore(archive_db).archive_statistics()["settled_trials"] == 1
    (output / "manifest.json").unlink()
    marker_path = (
        output / "scratch" / "control-test" / "attempt-0001" / ".verdiwm-scratch.json"
    )
    marker = json.loads(marker_path.read_text())
    marker.update({"state": "running", "archive_recorded": False})
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    second = auto_experiment.run_auto_experiment(
        plan_path=plan_path,
        output_root=output,
        workspace_root=ROOT,
        archive_db=archive_db,
        cas_root=tmp_path / "store",
        lock_root=tmp_path / "locks",
        budget_total_gpu_hours=0.1,
    )
    assert second["receipt_ref"] == manifest["receipt_ref"]
    recovered_marker = json.loads(marker_path.read_text())
    assert recovered_marker["state"] == "settled"
    assert recovered_marker["archive_recorded"] is True


def test_execution_failure_is_archived_as_void(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(auto_experiment, "GpuLeaseManager", _FakeLeaseManager)
    monkeypatch.setattr(
        auto_experiment,
        "run_gpu_exclusivity_audit",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("busy")),
    )
    plan_path = _plan(tmp_path / "plan.json", trial_id="failed-control-test")
    output = tmp_path / "run"
    manifest = auto_experiment.run_auto_experiment(
        plan_path=plan_path,
        output_root=output,
        workspace_root=ROOT,
        archive_db=tmp_path / "archive.db",
        cas_root=tmp_path / "store",
        lock_root=tmp_path / "locks",
        budget_db=output / "budget.db",
    )
    assert manifest["verdict"] == "VOID"
    receipt = json.loads((output / "receipts" / "failed-control-test.json").read_text())
    assert any(
        blocker["code"] == "EXECUTION_ERROR"
        for blocker in receipt["verdict"]["blockers"]
    )
    assert any(
        blocker["code"] == "REQUIRED_ARTIFACTS_MISSING"
        for blocker in receipt["verdict"]["blockers"]
    )


def test_gpu_capacity_contention_is_deferred_without_receipt_or_budget_charge(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from wmloop.execute.budget import BudgetLedger, BudgetPolicy

    monkeypatch.setattr(
        auto_experiment,
        "GpuLeaseManager",
        _UnavailableLeaseManager,
    )
    plan_path = _plan(tmp_path / "plan.json", trial_id="deferred-control-test")
    output = tmp_path / "run"
    archive_db = tmp_path / "archive.db"
    budget_db = tmp_path / "budget.db"

    with pytest.raises(GpuLeaseError, match="GPU_LEASE_UNAVAILABLE"):
        auto_experiment.run_auto_experiment(
            plan_path=plan_path,
            output_root=output,
            workspace_root=ROOT,
            archive_db=archive_db,
            cas_root=tmp_path / "store",
            lock_root=tmp_path / "locks",
            budget_db=budget_db,
        )

    assert not (output / "receipts" / "deferred-control-test.json").exists()
    assert not (output / "manifest.json").exists()
    assert not list((output / "scratch" / "deferred-control-test").glob("attempt-*"))
    assert ArchiveStore(archive_db).archive_statistics()["settled_trials"] == 0
    ledger = BudgetLedger(budget_db, BudgetPolicy(total_gpu_hours=0.05))
    assert ledger.visible_settled_trial_ids() == ()
    assert ledger.admit(
        "replacement-trial",
        cost_class="very_low",
        estimated_gpu_hours=0.05,
    ).state == "admitted"


def test_timeout_and_command_failure_are_archived(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_success_runtime(monkeypatch, tmp_path)
    monkeypatch.setattr(auto_experiment, "LocalSubprocessBackend", _TimeoutBackend)
    plan_path = _plan(tmp_path / "plan.json", trial_id="timeout-control-test")
    output = tmp_path / "run"
    manifest = auto_experiment.run_auto_experiment(
        plan_path=plan_path,
        output_root=output,
        workspace_root=ROOT,
        archive_db=tmp_path / "archive.db",
        cas_root=tmp_path / "store",
        lock_root=tmp_path / "locks",
    )
    assert manifest["verdict"] == "VOID"
    receipt = json.loads(
        (output / "receipts" / "timeout-control-test.json").read_text()
    )
    codes = {blocker["code"] for blocker in receipt["verdict"]["blockers"]}
    assert {
        "EXECUTION_TIMED_OUT",
        "EXECUTION_FAILED",
        "REQUIRED_ARTIFACTS_MISSING",
    } <= codes


def test_stale_admission_is_taken_over_with_new_fencing_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from wmloop.execute.budget import BudgetLedger, BudgetPolicy

    _patch_success_runtime(monkeypatch, tmp_path)
    plan_path = _plan(tmp_path / "plan.json", trial_id="takeover-control-test")
    plan = auto_experiment._load_plan(plan_path, workspace_root=ROOT)
    output = tmp_path / "run"
    plan_sha256 = auto_experiment._sha256_bytes(plan_path.read_bytes())
    auto_experiment._initialize_output(
        destination=output, plan=plan, plan_sha256=plan_sha256
    )
    archive_trial_id = (
        "auto-" + auto_experiment._trial_signature(plan=plan, workspace_root=ROOT)[:32]
    )
    ledger = BudgetLedger(output / "budget.db", BudgetPolicy(total_gpu_hours=0.05))
    first = ledger.admit(
        archive_trial_id, cost_class="very_low", estimated_gpu_hours=0.01
    )
    assert first.fencing_token == 1
    auto_experiment.run_auto_experiment(
        plan_path=plan_path,
        output_root=output,
        workspace_root=ROOT,
        archive_db=tmp_path / "archive.db",
        cas_root=tmp_path / "store",
        lock_root=tmp_path / "locks",
        budget_db=output / "budget.db",
    )
    receipt = json.loads(
        (output / "receipts" / "takeover-control-test.json").read_text()
    )
    assert receipt["fencing_token"] == 2


def test_campaign_budget_is_shared_and_policy_is_immutable(tmp_path: Path) -> None:
    from wmloop.execute.budget import BudgetError, BudgetLedger, BudgetPolicy

    database = tmp_path / "campaign-budget.db"
    policy = BudgetPolicy(total_gpu_hours=0.015)
    ledger = BudgetLedger(database, policy)
    first = ledger.admit("trial-one", cost_class="very_low", estimated_gpu_hours=0.01)
    ledger.settle(
        "trial-one",
        fencing_token=first.fencing_token,
        actual_gpu_hours=0.01,
        receipt_ref="cas://sha256/" + "1" * 64,
    )
    reopened = BudgetLedger(database, policy)
    with pytest.raises(BudgetError, match="GLOBAL_BUDGET_EXHAUSTED"):
        reopened.admit("trial-two", cost_class="very_low", estimated_gpu_hours=0.01)
    with pytest.raises(BudgetError, match="BUDGET_POLICY_MISMATCH"):
        BudgetLedger(database, BudgetPolicy(total_gpu_hours=1.0))


def test_budget_release_requires_current_fencing_token(tmp_path: Path) -> None:
    from wmloop.execute.budget import BudgetError, BudgetLedger, BudgetPolicy

    ledger = BudgetLedger(
        tmp_path / "campaign-budget.db",
        BudgetPolicy(total_gpu_hours=0.05),
    )
    admission = ledger.admit(
        "deferred-trial",
        cost_class="very_low",
        estimated_gpu_hours=0.05,
    )
    with pytest.raises(BudgetError, match="STALE_FENCING_TOKEN"):
        ledger.release(
            "deferred-trial",
            fencing_token=admission.fencing_token + 1,
        )
    ledger.release(
        "deferred-trial",
        fencing_token=admission.fencing_token,
    )
    assert ledger.get("deferred-trial") is None


def test_cleanup_requires_independent_durable_proof(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    scratch = run_root / "scratch" / "trial" / "attempt-0001"
    scratch.mkdir(parents=True)
    marker = {
        "state": "settled",
        "cleanup_eligible": True,
        "archive_recorded": True,
        "receipt_ref": "cas://sha256/" + "a" * 64,
        "required_artifact_count": 1,
        "archived_artifact_count": 1,
    }
    (scratch / ".verdiwm-scratch.json").write_text(json.dumps(marker), encoding="utf-8")
    old = time.time() - 10 * 24 * 3600
    os.utime(scratch / ".verdiwm-scratch.json", (old, old))
    report = auto_experiment.cleanup_auto_experiment_scratch(
        run_root=run_root,
        older_than_hours=1,
        archive_db=tmp_path / "archive.db",
        cas_root=tmp_path / "store",
        apply=True,
    )
    assert report["deleted_count"] == 0
    assert scratch.is_dir()
    assert report["retained"][0]["reason"] == "RECEIPT_CAS_UNREADABLE"


def test_cleanup_deletes_only_after_cas_and_archive_proof(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_success_runtime(monkeypatch, tmp_path)
    plan_path = _plan(tmp_path / "plan.json", trial_id="cleanup-control-test")
    output = tmp_path / "run"
    archive_db = tmp_path / "archive.db"
    cas_root = tmp_path / "store"
    auto_experiment.run_auto_experiment(
        plan_path=plan_path,
        output_root=output,
        workspace_root=ROOT,
        archive_db=archive_db,
        cas_root=cas_root,
        lock_root=tmp_path / "locks",
    )
    scratch = output / "scratch" / "cleanup-control-test" / "attempt-0001"
    marker_path = scratch / ".verdiwm-scratch.json"
    marker = json.loads(marker_path.read_text())
    marker["cleanup_eligible"] = True
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    old = time.time() - 10 * 24 * 3600
    os.utime(marker_path, (old, old))
    dry_run = auto_experiment.cleanup_auto_experiment_scratch(
        run_root=output,
        older_than_hours=1,
        archive_db=archive_db,
        cas_root=cas_root,
    )
    assert dry_run["candidates"] == [str(scratch)]
    assert scratch.is_dir()
    applied = auto_experiment.cleanup_auto_experiment_scratch(
        run_root=output,
        older_than_hours=1,
        archive_db=archive_db,
        cas_root=cas_root,
        apply=True,
    )
    assert applied["deleted_count"] == 1
    assert not scratch.exists()


def test_short_rationale_is_rejected(tmp_path: Path) -> None:
    plan_path = _plan(tmp_path / "plan.json")
    payload = json.loads(plan_path.read_text())
    payload["hypothesis"] = "short"
    plan_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(AutoExperimentError, match="RATIONALE_TOO_SHORT"):
        auto_experiment._load_plan(plan_path, workspace_root=ROOT)
