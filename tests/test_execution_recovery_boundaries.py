"""CPU failure injection through real runner, verifier, ledgers and archive.

Only physical GPU execution/sampling and source identity are explicit fixtures.
Optional literature discovery is excluded from the recovery scenarios.
No assertions here establish model quality or physical GPU behavior.
"""

from __future__ import annotations

import copy
import json
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from wmloop.archive.store import ArchiveStore, ContentAddressedStore
from wmloop.execute import auto_experiment as auto
from wmloop.execute import autonomous_pipeline as pipeline
from wmloop.execute import experiment_scheduler as scheduler
from wmloop.execute import diagnostic_memory
from wmloop.execute.budget import BudgetError, BudgetLedger, BudgetPolicy
from wmloop.diagnose import probe_campaign
from wmloop.retrieve.index import ProbeRetrievalError, retrieve_probe_experiences
from wmloop.storage import sha256_bytes


SOURCE = Path(__file__).resolve().parents[1]


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def read(path: Path) -> dict:
    return json.loads(path.read_bytes())


def plan_file(root: Path, stage: str = "screen") -> Path:
    plan = read(SOURCE / "configs/smoke/auto_experiment_cuda_smoke_v1.json")
    plan.update(campaign_id="recovery-fixture", trial_id=f"recovery-{stage}",
                stage=stage, total_budget_gpu_hours=1.0, cleanup_policy="retain")
    plan["metric_gates"] = [{"metric": "runtime_ready", "role": "primary",
                             "operator": "gte", "threshold": 1.0}]
    path = root / f"{stage}.json"
    write(path, plan)
    return path


@pytest.fixture
def engine(monkeypatch):
    calls = []
    fail_stages = set()

    def execute(*, plan, scratch, fencing_token, **kwargs):
        calls.append((plan["stage"], fencing_token))
        samples = {"samples": [
            {"status": "ready", "gpu_uuid": "GPU-cpu-fixture", "phase": "before",
             "memory_used_mib": 0.0, "utilization_gpu_percent": 0.0},
            {"status": "ready", "gpu_uuid": "GPU-cpu-fixture", "phase": "during",
             "memory_used_mib": 64.0, "utilization_gpu_percent": 10.0},
        ]}
        write(scratch / "result.json", {
            "schema_version": 1, "artifact_type": "verdiwm-auto-experiment-result",
            "state": "ready", "device": {"type": "cuda", "gpu_uuid": "GPU-cpu-fixture"},
            "metrics": {"runtime_ready": 0.0 if plan["stage"] in fail_stages else 1.0},
            "claim_boundary": "CPU recovery fixture with synthetic GPU telemetry.",
        })
        write(scratch / "gpu-sampling.json", samples)
        if plan.get("diagnostic_result_path") is not None:
            write(scratch / plan["diagnostic_result_path"], {
                "schema_version": 1, "artifact_type": "verdiwm-diagnostic-probe-result",
                "state": "ready", "probe_id": f"probe-{plan['stage']}",
                "model_family": "cpu-fixture", "runtime_capability": "video",
                "failure_signatures": ["history"], "primitive": "anchor",
                "metrics": {"primary": 0.2},
                "claim_boundary": "CPU diagnostic routing fixture; no model effect claim.",
            })
        (scratch / "stdout.log").write_bytes(b"CPU recovery fixture\n")
        (scratch / "stderr.log").write_bytes(b"")
        return {"lease": {"uuid": "GPU-cpu-fixture"}, "gpu_sampling": samples,
                "timed_out": False, "exit_code": 0, "duration_seconds": 36.0}

    monkeypatch.setattr(auto, "_git_state", lambda _: {"revision": "cpu-fixed", "implementation_hash": "a" * 64})
    monkeypatch.setattr(auto, "_execute_trial", execute)
    return calls, fail_stages


def run_options(root: Path) -> dict:
    return dict(plan_path=plan_file(root), output_root=root / "run", workspace_root=SOURCE,
                archive_db=root / "archive.db", cas_root=root / "cas", lock_root=root / "locks",
                budget_db=root / "budget.db", budget_total_gpu_hours=1.0)


def rows(path: Path) -> list[dict]:
    with sqlite3.connect(path) as db:
        db.row_factory = sqlite3.Row
        return [dict(row) for row in db.execute("SELECT * FROM trials")]


@pytest.mark.parametrize("after_commit", [False, True])
def test_interrupted_budget_commit_recovers_original_receipt_and_cost(tmp_path, monkeypatch, engine, after_commit):
    args = run_options(tmp_path)
    settle = BudgetLedger.settle

    def interrupt(self, *positional, **kwargs):
        if after_commit:
            settle(self, *positional, **kwargs)
        raise RuntimeError("crash at ledger commit boundary")

    with monkeypatch.context() as fault:
        fault.setattr(BudgetLedger, "settle", interrupt)
        with pytest.raises(RuntimeError, match="ledger commit boundary"):
            auto.run_auto_experiment(**args)
    receipt_path = tmp_path / "run/receipts/recovery-screen.json"
    receipt = read(receipt_path)
    assert rows(args["budget_db"])[0]["state"] == ("settled" if after_commit else "admitted")
    first_ref = receipt["receipt_ref"]
    assert ContentAddressedStore(args["cas_root"]).read_bytes(first_ref)
    for _ in range(2):
        assert auto.run_auto_experiment(**args)["receipt_ref"] == first_ref
    assert engine[0] == [("screen", 1)]
    assert read(receipt_path)["fencing_token"] == 1
    row = rows(args["budget_db"])[0]
    assert row["actual_gpu_hours"] == 0.01
    assert row["receipt_ref"] == first_ref
    assert ArchiveStore(args["archive_db"]).visible_settled_trials() == (receipt["archive_trial_id"],)


def test_unknown_cost_without_receipt_keeps_reservation_and_never_reexecutes(tmp_path, monkeypatch, engine):
    args = run_options(tmp_path)

    def killed(**kwargs):
        engine[0].append(("interrupted", kwargs["fencing_token"]))
        raise KeyboardInterrupt("process died before receipt")

    with monkeypatch.context() as fault:
        fault.setattr(auto, "_execute_trial", killed)
        with pytest.raises(KeyboardInterrupt):
            auto.run_auto_experiment(**args)
    before = rows(args["budget_db"])
    with pytest.raises(auto.AutoExperimentError, match="REQUIRES_RECONCILIATION"):
        auto.run_auto_experiment(**args)
    assert rows(args["budget_db"]) == before
    assert engine[0] == [("interrupted", 1)]


@pytest.mark.parametrize("mutation", ["projection", "token", "reservation"])
def test_precommit_recovery_rejects_untrusted_bindings(tmp_path, monkeypatch, engine, mutation):
    args = run_options(tmp_path)
    with monkeypatch.context() as fault:
        fault.setattr(BudgetLedger, "settle", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("crash")))
        with pytest.raises(RuntimeError):
            auto.run_auto_experiment(**args)
    path = tmp_path / "run/receipts/recovery-screen.json"
    if mutation == "projection":
        receipt = read(path)
        receipt["verdict"]["verdict"] = "VOID"
        write(path, receipt)
    else:
        with sqlite3.connect(args["budget_db"]) as db:
            db.execute("UPDATE trials SET fencing_token = 2" if mutation == "token"
                       else "UPDATE trials SET plan_sha256 = ?", () if mutation == "token" else ("b" * 64,))
    before = rows(args["budget_db"])
    with pytest.raises(auto.AutoExperimentError, match="MISMATCH"):
        auto.run_auto_experiment(**args)
    assert rows(args["budget_db"]) == before
    assert engine[0] == [("screen", 1)]


def prepare_queue(root: Path) -> Path:
    stages = []
    for stage in ("screen", "gate", "confirm"):
        path = plan_file(root, stage)
        stages.append({"stage": stage, "plan_path": path.name, "plan_sha256": sha256_bytes(path.read_bytes())})
    path = root / "queue.json"
    write(path, {"artifact_type": "verdiwm-auto-experiment-queue", "state": "ready",
                 "campaign_id": "recovery-fixture", "batch_sha256": "a" * 64,
                 "selected": [{"candidate_id": "candidate", "stages": stages,
                               "promotion_policy": {"required_stage": "confirm",
                                                    "required_quality_metrics": ["runtime_ready"]}}]})
    return path


def test_scheduler_recovers_real_void_confirm_instead_of_forged_pass(tmp_path, engine):
    engine[1].add("confirm")
    args = dict(queue_path=prepare_queue(tmp_path), workspace_root=SOURCE,
                archive_db=tmp_path / "archive.db", cas_root=tmp_path / "cas", lock_root=tmp_path / "locks")
    initial = scheduler.run_selected_queue(**args)
    assert initial["candidate_states"] == {"candidate": "blocked"}
    forged = copy.deepcopy(initial)
    forged["results"]["candidate:confirm"].update(state="completed", verdict="PASS")
    write(tmp_path / "execution.json", forged)
    recovered = scheduler.run_selected_queue(**args)
    assert recovered["results"]["candidate:confirm"]["verdict"] == "VOID"
    assert recovered["candidate_states"] == {"candidate": "blocked"}
    assert recovered["promotion_decisions"]["candidate"]["state"] == "rejected"
    assert engine[0] == [("screen", 1), ("gate", 1), ("confirm", 1)]


def prepare_pipeline(root: Path, monkeypatch) -> pipeline.AutonomousPipelineOptions:
    output = root / "output"
    template = root / "template.json"
    evaluator = root / "evaluator.json"
    write(template, {"objective": "Validate recovery evidence boundaries",
                     "falsification_criterion": "Reject tampered or unavailable settled evidence"})
    write(evaluator, {"scheduler_template": str(template), "metrics": ["runtime_ready"]})
    monkeypatch.setattr(pipeline, "_input_document", lambda *args, **kwargs: {"cpu_fixture": "fixed"})
    monkeypatch.setattr(pipeline, "run_onboarding", lambda _: {"state": "ready_for_conformance_smoke"})
    monkeypatch.setattr(pipeline, "run_conformance", lambda _: {"verdict": "PASS"})
    # Probe signatures can start external literature discovery; it is outside
    # these receipt/CAS recovery scenarios and must not access the network.
    monkeypatch.setattr(pipeline, "build_literature_query", lambda **kwargs: None)
    monkeypatch.setattr(pipeline, "compile_and_plan", lambda **kwargs: {"queue_path": str(prepare_queue(output / "compiled/queue"))})
    return pipeline.AutonomousPipelineOptions(repo_root=SOURCE, output_root=output,
        evaluator_contract=evaluator, lock_root=root / "locks", budget_total_gpu_hours=1.0)


def prepare_diagnostic_queue(root: Path) -> Path:
    path = prepare_queue(root)
    queue = read(path)
    for stage in queue["selected"][0]["stages"]:
        plan_path = root / stage["plan_path"]
        plan = read(plan_path)
        plan.update(experiment_kind="diagnostic", diagnostic_result_path="diagnostic.json")
        plan["artifacts"].append("diagnostic.json")
        write(plan_path, plan)
        stage["plan_sha256"] = sha256_bytes(plan_path.read_bytes())
    write(path, queue)
    return path


def diagnostic_matches(root: Path) -> tuple:
    return retrieve_probe_experiences(
        database_path=root / "retrieval.db", archive_db=root / "archive.db",
        cas_root=root / "cas", model_family="cpu-fixture",
        runtime_capability="video", failure_signatures=("history",),
    )


@pytest.mark.parametrize("interrupt_after_index", [False, True])
def test_top_pipeline_accumulates_declared_diagnostics_and_resumes_idempotently(
    tmp_path, monkeypatch, engine, interrupt_after_index,
):
    options = prepare_pipeline(tmp_path, monkeypatch)
    monkeypatch.setattr(pipeline, "compile_and_plan", lambda **kwargs: {
        "queue_path": str(prepare_diagnostic_queue(options.output_root / "compiled/queue"))})
    if interrupt_after_index:
        index = diagnostic_memory.index_probe_experience

        def interrupt(**kwargs):
            index(**kwargs)
            raise RuntimeError("interruption after committed diagnostic indexing")

        with monkeypatch.context() as fault:
            fault.setattr(diagnostic_memory, "index_probe_experience", interrupt)
            with pytest.raises(RuntimeError, match="committed diagnostic"):
                pipeline.run_autonomous_pipeline(options)
        assert len(diagnostic_matches(options.output_root)) == 1
        assert engine[0] == [("screen", 1)]
    initial = pipeline.run_autonomous_pipeline(options)
    assert initial["state"] == "settled"
    assert {row.stage for row in diagnostic_matches(options.output_root)} == {"screen", "gate", "confirm"}
    assert pipeline.run_autonomous_pipeline(options) == initial
    assert len(diagnostic_matches(options.output_root)) == 3
    assert engine[0] == [("screen", 1), ("gate", 1), ("confirm", 1)]


def test_invalid_diagnostic_contract_cannot_create_memory(tmp_path, monkeypatch, engine):
    execute = auto._execute_trial

    def invalid(**kwargs):
        result = execute(**kwargs)
        write(kwargs["scratch"] / "diagnostic.json", {"state": "ready"})
        return result

    monkeypatch.setattr(auto, "_execute_trial", invalid)
    with pytest.raises(ProbeRetrievalError, match="RESULT_CONTRACT_INVALID"):
        scheduler.run_selected_queue(
            queue_path=prepare_diagnostic_queue(tmp_path), workspace_root=SOURCE,
            archive_db=tmp_path / "archive.db", cas_root=tmp_path / "cas",
            lock_root=tmp_path / "locks", retrieval_db=tmp_path / "retrieval.db",
        )
    assert not (tmp_path / "retrieval.db").exists()
    assert len(ArchiveStore(tmp_path / "archive.db").visible_settled_trials()) == 1
    assert engine[0] == [("screen", 1)]


@pytest.mark.parametrize("mutation", ["missing", "tampered"])
def test_settled_diagnostic_memory_requires_unchanged_cas(tmp_path, engine, mutation):
    args = dict(queue_path=prepare_diagnostic_queue(tmp_path), workspace_root=SOURCE,
                archive_db=tmp_path / "archive.db", cas_root=tmp_path / "cas",
                lock_root=tmp_path / "locks", retrieval_db=tmp_path / "retrieval.db")
    scheduler.run_selected_queue(**args)
    receipt = read(tmp_path / "runs/candidate/screen/receipts/recovery-screen.json")
    ref = receipt["artifact_refs"]["diagnostic.json"]
    cas = ContentAddressedStore(args["cas_root"])
    digest = ref.rsplit("/", 1)[-1]
    artifact = cas._cas / digest[:2] / digest
    if mutation == "missing":
        artifact.unlink()
    else:
        artifact.write_bytes(b"{}")
    assert all(row.stage != "screen" for row in diagnostic_matches(tmp_path))
    calls = list(engine[0])
    with pytest.raises(auto.AutoExperimentError, match="RECOVERY"):
        scheduler.run_selected_queue(**args)
    assert engine[0] == calls


@pytest.mark.parametrize("mutation", ["none", "receipt", "receipt_cas", "artifact", "manifest"])
def test_settled_pipeline_revalidates_probe_before_returning_or_scheduling(
    tmp_path, monkeypatch, engine, mutation,
):
    options = prepare_pipeline(tmp_path, monkeypatch)
    contract_path = tmp_path / "probe-contract.json"
    contract = {"probe_id": "probe-screen", "model_family": "cpu-fixture",
                "runtime_capability": "video", "failure_signature_fields": [],
                "diagnostic_result_path": "diagnostic.json"}
    write(contract_path, contract)
    options = replace(options, probe_contract=contract_path)
    report = b"{}"

    def onboard(value):
        value.output_root.mkdir(parents=True, exist_ok=True)
        (value.output_root / "onboarding-report.json").write_bytes(report)
        return {"state": "ready_for_conformance_smoke"}

    def conform(value):
        value.output_root.mkdir(parents=True, exist_ok=True)
        return {"verdict": "PASS"}

    def build_plan(*args, **kwargs):
        plan = read(plan_file(tmp_path / "probe-plan-source"))
        plan.update(trial_id="diagnostic-screen", campaign_id="diagnostic-fixture",
                    experiment_kind="diagnostic", diagnostic_result_path="diagnostic.json")
        plan["artifacts"].append("diagnostic.json")
        return plan

    monkeypatch.setattr(pipeline, "run_onboarding", onboard)
    monkeypatch.setattr(pipeline, "run_conformance", conform)
    monkeypatch.setattr(probe_campaign, "admission_from_manifest", lambda _: {
        "onboarding_report_sha256": sha256_bytes(report)})
    monkeypatch.setattr(probe_campaign, "_load_contract", lambda _: contract)
    monkeypatch.setattr(probe_campaign, "_build_plan", build_plan)
    initial = pipeline.run_autonomous_pipeline(options)
    assert initial["state"] == "settled"
    assert initial["diagnostic_probe"]["verdict"] == "PASS"
    root = options.output_root / "diagnostic-probe"
    receipt_path = root / "run/receipts/diagnostic-screen.json"
    receipt = read(receipt_path)
    if mutation == "receipt":
        receipt_path.unlink()
    elif mutation == "manifest":
        manifest = read(root / "manifest.json")
        manifest["failure_signatures"] = ["forged"]
        write(root / "manifest.json", manifest)
    elif mutation != "none":
        ref = receipt["receipt_ref"] if mutation == "receipt_cas" else receipt["artifact_refs"]["diagnostic.json"]
        cas = ContentAddressedStore(options.output_root / "cas")
        digest = ref.rsplit("/", 1)[-1]
        (cas._cas / digest[:2] / digest).unlink()
    calls = list(engine[0])
    if mutation in {"none", "manifest"}:
        assert pipeline.run_autonomous_pipeline(options) == initial
        assert read(root / "manifest.json")["failure_signatures"] == ["history"]
    else:
        with pytest.raises((probe_campaign.DiagnosticProbeError, auto.AutoExperimentError), match="RECOVERY"):
            pipeline.run_autonomous_pipeline(options)
    assert engine[0] == calls


@pytest.mark.parametrize("kind,artifact", [
    ("evaluation_only", "diagnostic.json"), ("diagnostic", "undeclared.json"),
    ("diagnostic", "../escape.json"),
])
def test_diagnostic_declaration_is_checked_before_admission(tmp_path, engine, kind, artifact):
    args = run_options(tmp_path)
    plan = read(args["plan_path"])
    plan.update(experiment_kind=kind, diagnostic_result_path=artifact)
    plan["artifacts"].append("diagnostic.json")
    write(args["plan_path"], plan)
    with pytest.raises(auto.AutoExperimentError, match="DIAGNOSTIC_RESULT"):
        auto.run_auto_experiment(**args)
    assert not args["budget_db"].exists()
    assert engine[0] == []


@pytest.mark.parametrize("value", [0, -1, 301, float("nan"), True, "30"])
def test_pipeline_runtime_timeout_is_checked_before_output(tmp_path, value):
    with pytest.raises(pipeline.OnboardingError, match="RUNTIME_TIMEOUT_INVALID"):
        pipeline.run_autonomous_pipeline(pipeline.AutonomousPipelineOptions(
            repo_root=SOURCE, output_root=tmp_path / "output",
            evaluator_contract=tmp_path / "missing.json", runtime_timeout_seconds=value,
        ))
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("mutation", ["none", "manifest", "execution", "receipt", "receipt_cas",
                                    "artifact", "support", "verdict", "context", "snapshot_cas"])
def test_settled_pipeline_revalidates_all_evidence_without_execution(tmp_path, monkeypatch, engine, mutation):
    options = prepare_pipeline(tmp_path, monkeypatch)
    initial = pipeline.run_autonomous_pipeline(options)
    assert initial["state"] == "settled"
    root = options.output_root
    path = root / "compiled/queue/runs/candidate/confirm/receipts/recovery-confirm.json"
    receipt = read(path)
    cas = ContentAddressedStore(root / "cas")
    if mutation == "manifest":
        manifest = read(root / "pipeline-manifest.json")
        manifest.update(verdict="BLOCKED", candidate_states={"forged": "completed"})
        write(root / "pipeline-manifest.json", manifest)
    elif mutation == "execution":
        execution = read(root / "compiled/queue/execution.json")
        execution["candidate_states"] = {"forged": "completed"}
        write(root / "compiled/queue/execution.json", execution)
    elif mutation == "receipt":
        receipt["verdict"]["verdict"] = "VOID"
        write(path, receipt)
    elif mutation != "none":
        refs = {"receipt_cas": receipt["receipt_ref"], "artifact": receipt["artifact_refs"]["result.json"],
                "support": receipt["support_refs"]["stdout"], "verdict": receipt["verdict_ref"],
                "context": receipt["failure_context_ref"]}
        if mutation == "snapshot_cas":
            refs[mutation] = "cas://sha256/" + sha256_bytes(auto._canonical_json(initial))
        digest = refs[mutation].rsplit("/", 1)[-1]
        (cas._cas / digest[:2] / digest).unlink()
    calls = list(engine[0])
    if mutation == "none":
        assert pipeline.run_autonomous_pipeline(options) == initial
    else:
        with pytest.raises((pipeline.AutonomousPipelineError, auto.AutoExperimentError), match="SETTLEMENT|RECOVERY"):
            pipeline.run_autonomous_pipeline(options)
    assert engine[0] == calls


@pytest.mark.parametrize("link_kind", ["plan", "output", "output_parent"])
def test_auto_rejects_original_symlink_paths_before_creating_outputs(tmp_path, engine, link_kind):
    args = run_options(tmp_path)
    if link_kind == "plan":
        link = tmp_path / "plan-link.json"
        link.symlink_to(args["plan_path"])
        args["plan_path"] = link
    elif link_kind == "output":
        target = tmp_path / "target"
        target.mkdir()
        args["output_root"].symlink_to(target, target_is_directory=True)
    else:
        target = tmp_path / "target"
        target.mkdir()
        link = tmp_path / "parent-link"
        link.symlink_to(target, target_is_directory=True)
        args["output_root"] = link / "run"
    with pytest.raises(auto.AutoExperimentError, match="PATH_INVALID|OUTPUT_ROOT_INVALID"):
        auto.run_auto_experiment(**args)
    assert engine[0] == []
    assert not args["budget_db"].exists()


def test_auto_recovery_rejects_symlinked_receipt_parent(tmp_path, engine):
    args = run_options(tmp_path)
    auto.run_auto_experiment(**args)
    receipts = args["output_root"] / "receipts"
    target = tmp_path / "external-receipts"
    receipts.rename(target)
    receipts.symlink_to(target, target_is_directory=True)
    with pytest.raises(auto.AutoExperimentError, match="RECOVERY_RECEIPT_INVALID"):
        auto.run_auto_experiment(**args)
    assert engine[0] == [("screen", 1)]


def test_settlement_is_idempotent_only_for_exact_binding_cost_and_receipt(tmp_path):
    ledger = BudgetLedger(tmp_path / "budget.db", BudgetPolicy(total_gpu_hours=1))
    ledger.admit("trial", cost_class="very_low", estimated_gpu_hours=0.01, plan_sha256="a" * 64)
    args = dict(fencing_token=1, actual_gpu_hours=0.01, receipt_ref="cas://sha256/" + "b" * 64,
                plan_sha256="a" * 64)
    assert ledger.settle("trial", **args) == ledger.settle("trial", **args)
    for delta in ({"fencing_token": 2}, {"actual_gpu_hours": 0.02},
                  {"receipt_ref": "cas://sha256/" + "c" * 64}, {"plan_sha256": "d" * 64}):
        with pytest.raises(BudgetError):
            ledger.settle("trial", **(args | delta))
    assert rows(tmp_path / "budget.db")[0]["actual_gpu_hours"] == 0.01
