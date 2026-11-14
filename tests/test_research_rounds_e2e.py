"""Public two-round engineering loop with real child processes and stores.

Only the hardware telemetry boundary is simulated: no GPU, model training, or
scientific improvement is claimed. The scheduler, lease, settlement, CAS,
Archive, retrieval, controller and dispatcher execute their production code.
"""

from __future__ import annotations

import copy
import hashlib
from io import StringIO
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import textwrap

import pytest

from wmloop.archive.store import ContentAddressedStore


ROOT = Path(__file__).resolve().parents[1]
SIMULATED_UUID = "GPU-CPU-FIXTURE-NO-HARDWARE"


def _write_json(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return path


def _run(command: list[str], *, cwd: Path, env: dict[str, str], expected: int = 0) -> subprocess.CompletedProcess:
    process = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=120)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
        raise
    result = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    assert result.returncode == expected, result.stdout + "\n" + result.stderr
    return result


@pytest.fixture
def public_rounds_project(tmp_path):
    source, model, data = (tmp_path / name for name in ("source", "model", "data"))
    for directory in (source, model, data):
        directory.mkdir()
    (model / "checkpoint.pt").write_bytes(b"CPU engineering fixture; not model weights\n")
    (data / "observations.json").write_text('[{"observation": 1}]\n')
    workload = source / "evaluate.py"
    workload.write_text(textwrap.dedent('''\
        """CPU engineering fixture only; no model or physical GPU claim."""
        import argparse
        import json
        import os
        from pathlib import Path
        import time

        def predict():
            return 1.0

        def train():
            return predict()

        def evaluate():
            parser = argparse.ArgumentParser()
            parser.add_argument("--checkpoint")
            parser.add_argument("--output", default=os.environ.get("VERDIWM_TRIAL_SCRATCH"))
            args = parser.parse_args()
            destination = Path(args.output)
            destination.mkdir(parents=True, exist_ok=True)
            time.sleep(0.2)
            result = {
                "schema_version": 1,
                "artifact_type": "verdiwm-auto-experiment-result",
                "state": "ready",
                "device": {"type": "cuda", "gpu_uuid": os.environ["VERDIWM_PHYSICAL_GPU_UUID"]},
                "metrics": {"quality": predict()},
                "claim_boundary": "CPU fixture with simulated hardware telemetry; no model effect or physical GPU evidence.",
            }
            (destination / "result.json").write_text(json.dumps(result))
            if os.environ.get("VERDIWM_PROBE_ID"):
                diagnostic = {
                    "schema_version": 1,
                    "artifact_type": "verdiwm-diagnostic-probe-result",
                    "state": "ready",
                    "probe_id": os.environ["VERDIWM_PROBE_ID"],
                    "model_family": "cpu_fixture",
                    "runtime_capability": "cpu_fixture",
                    "failure_signatures": ["fixture_symptom"],
                    "metrics": {"quality": predict()},
                    "claim_boundary": "CPU fixture symptom for retrieval validation only.",
                }
                (destination / "diagnostic.json").write_text(json.dumps(diagnostic))

        if __name__ == "__main__":
            evaluate()
    '''), encoding="utf-8")
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.update(PYTHONDONTWRITEBYTECODE="1", VERDIWM_LOCK_ROOT=str(tmp_path / "locks"))
    # Exercise existing network-failure degradation without contacting a real
    # service. This is neither a literature response fixture nor a pipeline mock.
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        env[name] = "http://127.0.0.1:1"
    env.update(NO_PROXY="", no_proxy="")
    hardware = tmp_path / "hardware-simulator"
    hardware.mkdir()
    telemetry = hardware / "nvidia-smi"
    telemetry.write_text(textwrap.dedent(f'''\
        #!{sys.executable}
        """Explicit CPU fixture hardware boundary, no real device access."""
        import sys
        query = " ".join(sys.argv[1:])
        if "--query-compute-apps=" in query:
            pass
        elif "temperature.gpu" in query:
            print("0, {SIMULATED_UUID}, CPU SIMULATION, 100, 50, 30, 1")
        elif "--query-gpu=" in query:
            print("0, {SIMULATED_UUID}, CPU SIMULATION, 0, 65536, 0")
        else:
            raise SystemExit("Unsupported CPU simulator query: " + query)
    '''), encoding="utf-8")
    telemetry.chmod(0o755)
    env["PATH"] = str(hardware) + os.pathsep + env.get("PATH", "")
    for git_args in (
        ["init", "--quiet"], ["add", "evaluate.py"],
        ["-c", "user.name=CPU fixture", "-c", "user.email=fixture@example.invalid", "commit", "--quiet", "-m", "CPU fixture"],
    ):
        _run(["git", *git_args], cwd=source, env=env)
    # A first-time pip user keeps every profile resource outside the install.
    contracts = tmp_path / "contracts"
    contracts.mkdir()
    evaluator = _write_json(contracts / "evaluator.json", {
        "evaluator_id": "cpu-fixture", "command": ["{python}", "evaluate.py", "--checkpoint", "{asset:--checkpoint}"],
        "input_artifacts": ["checkpoint"], "output_artifacts": ["result.json"],
        "metrics": ["quality"], "verifier": "CPU engineering fixture only",
        "working_directory": ".", "conformance_imports": [], "entrypoint_probe": "help",
        "scheduler_template": "candidate-batch.json",
    })
    runtime_stage = {
        "command": [sys.executable, "evaluate.py", "--output", "{scratch_dir}"],
        "working_directory": ".", "allowed_gpu_indices": [0],
        "estimated_gpu_hours": 0.0001, "timeout_seconds": 10,
        "gpu_wait_seconds": 0, "sample_interval_seconds": 0.02,
        "result_path": "result.json", "artifacts": ["result.json"],
        "metric_gates": [{"metric": "quality", "role": "primary", "operator": "gte", "threshold": 1}],
        "environment": {}, "cleanup_policy": "retain",
    }
    batch = json.loads((ROOT / "configs/smoke/auto_experiment_candidate_batch_cuda_v1.json").read_text())
    batch.update(campaign_id="cpu-fixture", total_budget_gpu_hours=0.1)
    candidate = batch["candidates"][0]
    candidate.update(candidate_id="cpu-fixture", hypothesis="CPU engineering fixture exercises settled stages without model effect claims.")
    candidate["retrieval_keys"] = {"failure_signatures": ["fixture_symptom"]}
    candidate["stages"] = [{**copy.deepcopy(runtime_stage), "stage": stage} for stage in ("screen", "gate", "confirm")]
    batch["candidates"] = [candidate]
    _write_json(contracts / "candidate-batch.json", batch)
    _write_json(contracts / "probe.json", {
        **runtime_stage, "schema_version": 1, "artifact_type": "verdiwm-diagnostic-probe-contract",
        "probe_id": "cpu-fixture", "objective": "Exercise CPU engineering retrieval through settled evidence.",
        "hypothesis": "A repeated CPU fixture symptom can retrieve the earlier settled observation.",
        "selection_reason": "Bounded fixture isolates engineering behavior from model performance.",
        "falsification_criterion": "Missing or mismatched settled evidence invalidates this engineering run.",
        "total_budget_gpu_hours": 0.1, "diagnostic_result_path": "diagnostic.json",
        "artifacts": ["result.json", "diagnostic.json"], "model_family": "cpu_fixture",
        "runtime_capability": "cpu_fixture", "failure_signature_fields": ["failure_signatures"],
    })
    frozen_path = _write_json(contracts / "fixture.json", {"claim_boundary": "CPU fixture only"})
    frozen_bytes = frozen_path.read_bytes()
    freeze = {
        "schema_version": 1, "artifact_type": "wmloop-constitutional-freeze",
        "constitution_id": "cpu-fixture", "goal_id": "cpu-fixture", "verdict_probe_ids": ["cpu-fixture"],
        "entries": [{"component": "fixture", "path": frozen_path.name,
                     "sha256": hashlib.sha256(frozen_bytes).hexdigest(), "size": len(frozen_bytes)}],
    }
    freeze["constitution_sha256"] = hashlib.sha256(json.dumps(freeze, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    _write_json(contracts / "constitution.freeze.json", freeze)
    profile = _write_json(contracts / "profile.json", {
        "schema_version": 1, "artifact_type": "verdiwm-adapter-profile", "profile_id": "cpu-fixture",
        "aliases": ["cpu_fixture"], "model_family": "cpu_fixture", "capability_level": "L1",
        "execution_kind": "pipeline", "repo_markers": ["evaluate.py"], "goal_keywords": ["improve"],
        "evaluator_contract": "evaluator.json", "probe_contract": "probe.json",
        "constitution_freeze": "constitution.freeze.json", "runtime_candidates": [sys.executable],
        "asset_bindings": [{"parameter": "--checkpoint", "candidates": ["{model}/checkpoint.pt"]}],
        "runner": {"protocol_version": 1, "adapter_id": "cpu-fixture", "requires_training": True,
                   "train": ["{python}", "evaluate.py"], "evaluate": ["{python}", "evaluate.py"],
                   "checkpoint_receipt": "receipts/checkpoint.json", "evaluation_receipt": "receipts/evaluation.json"},
        "probe_imports": False,
    })
    yield dict(root=tmp_path, source=source, model=model, data=data, evaluator=evaluator,
               profile=profile, env=env, state=tmp_path / "state")


def _plan(context: dict, *, budget: str = "0.1gpu-hours", rounds: int = 2) -> dict:
    command = [sys.executable, "-m", "wmloop.cli", "research", "plan",
               "--model", str(context["model"]), "--source", str(context["source"]),
               "--data", str(context["data"]), "--goal", "improve video prediction quality", "--budget", budget,
               "--rounds", str(rounds), "--stop-on-success", "--adapter-profile", str(context["profile"]),
               "--runtime-python", sys.executable, "--evaluator-contract", str(context["evaluator"]),
               "--project-root", str(context["root"]), "--state-root", str(context["state"]),
               "--output", str(context["root"] / "plan.json"), "--mode", "quick-start"]
    result = _run(command, cwd=ROOT, env=context["env"])
    return json.loads(result.stdout)


def _configured_readiness_project(context: dict, monkeypatch) -> Path:
    project_file = context["root"] / "verdiwm.toml"
    values = {
        "model": "model", "source": "source", "data": "data",
        "goal": "improve quality", "budget": "0.1gpu-hours",
        "adapter_profile": "contracts/profile.json",
        "evaluator_contract": "contracts/evaluator.json",
        "runtime_python": sys.executable, "state_root": "state",
    }
    project_file.write_text(
        "[project]\n" + "".join(f"{key} = {json.dumps(value)}\n" for key, value in values.items()),
        encoding="utf-8",
    )
    monkeypatch.setenv("VERDIWM_PROJECT_CONFIG", str(project_file))
    monkeypatch.delenv("VERDIWM_STATE_ROOT", raising=False)
    return project_file


@pytest.mark.parametrize("unrelated_cwd", [False, True])
def test_external_profile_assets_are_ready_in_welcome_check_and_start(
    public_rounds_project, monkeypatch, capsys, unrelated_cwd: bool,
):
    from wmloop.cli import main
    from wmloop.control import interactive_cli

    context = public_rounds_project
    _configured_readiness_project(context, monkeypatch)
    cwd = context["root"] / "unrelated" if unrelated_cwd else context["root"]
    cwd.mkdir(exist_ok=True)
    monkeypatch.chdir(cwd)
    config = interactive_cli._project_snapshot(cwd)
    readiness = interactive_cli._readiness_snapshot(config, cwd)
    assert readiness["state"] == "ready_for_conformance", readiness
    assert not readiness["blockers"]
    assert readiness["side_effects"] == {
        "model_import_executed": False, "gpu_execution_started": False, "source_modified": False,
    }
    welcome = StringIO()
    interactive_cli.render_welcome(stdout=welcome, project_root=cwd, color=False)
    assert "BLOCKED" not in welcome.getvalue()
    assert "模型或数据文件还没有绑定" not in welcome.getvalue()
    assert "/start" in welcome.getvalue()
    assert main(["check"]) == 0
    checked = json.loads(capsys.readouterr().out)
    assert checked["project"] == readiness
    assert main(["check-model"]) == 0
    assert json.loads(capsys.readouterr().out) == readiness
    assert main(["diagnose"]) == 0
    diagnosis = json.loads(capsys.readouterr().out)
    assert diagnosis["state"] == "ready_for_diagnostic_campaign"
    assert not diagnosis["blockers"]
    calls = []
    interactive_cli._dispatch_slash(
        "start", [], lambda argv: calls.append(argv) or 0,
        stdout=StringIO(), theme=interactive_cli._Theme(False),
    )
    assert calls == [["research", "plan", "--goal", "improve quality"]]
    assert not context["state"].exists(), "Readiness must not create a campaign or execution state."


@pytest.mark.parametrize("failure", ["missing_asset", "invalid_profile", "unconfirmed_evaluator", "missing_runtime"])
def test_profile_readiness_preserves_missing_and_invalid_inputs(
    public_rounds_project, monkeypatch, capsys, failure: str,
):
    from wmloop.cli import main
    from wmloop.control import interactive_cli

    context = public_rounds_project
    project_file = _configured_readiness_project(context, monkeypatch)
    monkeypatch.chdir(context["root"])
    if failure == "missing_asset":
        (context["model"] / "checkpoint.pt").unlink()
        reason = "ASSET_NOT_FOUND:--checkpoint"
    elif failure == "invalid_profile":
        context["profile"].write_text("{invalid profile", encoding="utf-8")
        reason = "ADAPTER_PROFILE_INVALID"
    elif failure == "unconfirmed_evaluator":
        project_file.write_text("\n".join(
            line for line in project_file.read_text().splitlines() if not line.startswith("evaluator_contract")
        ) + "\n", encoding="utf-8")
        reason = "EVALUATOR_CONTRACT_REQUIRED"
    else:
        project_file.write_text("\n".join(
            'runtime_python = "missing-python"' if line.startswith("runtime_python") else line
            for line in project_file.read_text().splitlines()
        ) + "\n", encoding="utf-8")
        reason = "RUNTIME_UNREADY"
    config = interactive_cli._project_snapshot(context["root"])
    readiness = interactive_cli._readiness_snapshot(config, context["root"])
    assert readiness["state"] == "needs_input"
    assert reason in json.dumps(readiness["blockers"])
    assert interactive_cli._next_step(config, readiness=readiness) == ("BLOCKED", "/check")
    assert main(["check"]) == 2
    assert json.loads(capsys.readouterr().out)["project"] == readiness
    assert main(["diagnose"]) == 2
    diagnosis = json.loads(capsys.readouterr().out)
    assert diagnosis["state"] == "blocked"
    assert reason in json.dumps(diagnosis["blockers"])
    assert not context["state"].exists()


def test_public_research_runs_two_real_pipeline_rounds_and_reads_settled_memory(public_rounds_project):
    context = public_rounds_project
    plan = _plan(context)
    assert plan["execution_policy"]["rounds"] == 2
    assert plan["warnings"], "No frozen goal means pipeline PASS cannot confirm scientific success."
    result = _run([sys.executable, "-m", "wmloop.cli", "research", "run", "--plan", plan["plan_path"], "--confirm"], cwd=ROOT, env=context["env"])
    report = json.loads(result.stdout)
    _write_json(context["root"] / "run-report.json", report)
    assert report["state"] == "completed"
    campaign = report["campaign"]
    assert campaign["execution"]["kind"] == "evolution"
    assert campaign["execution_result"]["rounds_completed"] == 2
    assert campaign["execution_result"]["stop_reason"] == "MAX_ITERATIONS_REACHED"
    execution = campaign["execution"]
    iterations_root = Path(execution["output_root"]) / "iterations"
    iterations = sorted(iterations_root.glob("iteration-*"))
    assert len(iterations) == 2
    pipelines = [json.loads((item / "pipeline/pipeline-manifest.json").read_text()) for item in iterations]
    assert all(item["verdict"] == "PASS" for item in pipelines)
    for pipeline in pipelines:
        assert pipeline["literature"]["state"] == "offline"
        assert pipeline["literature_method_materialization"]["candidate_selection"] == {
            "state": "degraded", "reason": "LITERATURE_NO_EXECUTABLE_CANDIDATES",
            "retained_source": "evaluator_template",
        }
    assert pipelines[0]["retrieval"]["route"] == "cold_start"
    assert pipelines[1]["retrieval"]["route"] == "reuse_settled"
    previous_probe = pipelines[0]["diagnostic_probe"]
    evidence = pipelines[1]["retrieval"]["capsule"]["selected_evidence"]
    assert evidence[0]["evidence"]["receipt_ref"] == previous_probe["receipt_ref"]
    assert evidence[0]["evidence"]["result_ref"] == previous_probe["result_ref"]
    assert evidence[0]["evidence"]["archive_trial_id"] == previous_probe["archive_trial_id"]
    cas = ContentAddressedStore(Path(execution["cas_root"]))
    previous_result = json.loads(cas.read_bytes(previous_probe["result_ref"]))
    assert previous_result["failure_signatures"] == ["fixture_symptom"]
    for iteration in iterations:
        queue = json.loads((iteration / "pipeline/compiled/queue/execution.json").read_text())
        assert queue["budget_db"] == execution["budget_db"]
        assert queue["budget_total_gpu_hours"] == 0.1
        assert queue["candidate_states"], "An empty candidate queue cannot demonstrate a closed loop."
    with sqlite3.connect(execution["archive_db"]) as connection:
        archived = connection.execute("SELECT trial_id,receipt_ref FROM trials").fetchall()
    with sqlite3.connect(execution["budget_db"]) as connection:
        charges = connection.execute("SELECT trial_id,state,actual_gpu_hours,receipt_ref FROM trials").fetchall()
        policy = connection.execute("SELECT total_gpu_hours FROM policy").fetchone()
    assert len(archived) == len(charges) == 8, "Two diagnostics and six candidate stages must settle."
    assert policy == (0.1,)
    assert {row[0] for row in archived} == {row[0] for row in charges}
    receipts = [json.loads(cas.read_bytes(row[1])) for row in archived]
    assert sorted(row["stage"] for row in receipts) == ["confirm", "confirm", "gate", "gate", "screen", "screen", "screen", "screen"]
    assert all(row[1] == "settled" and row[2] > 0 and row[3] for row in charges)
    assert sum(row[2] for row in charges) == pytest.approx(sum(item["cost"]["actual_gpu_hours"] for item in receipts))
    assert all(item["budget"]["ledger_path"] == execution["budget_db"] for item in receipts)
    assert all(item["budget"]["ledger_total_gpu_hours"] == 0.1 for item in receipts)
    assert not list(iterations_root.rglob("budget.db")), "No per-round fresh budget ledger is allowed."
    assert _run(["git", "status", "--porcelain"], cwd=context["source"], env=context["env"]).stdout == ""

    # A later independent user campaign shares settled memory, while owning
    # fresh execution identities and preserving the earlier frozen outputs.
    first_manifests = {item / "pipeline/pipeline-manifest.json": (item / "pipeline/pipeline-manifest.json").read_bytes() for item in iterations}
    later_workspace = context["root"] / "Later User Workspace"
    later_workspace.mkdir()
    later_plan = _plan({**context, "root": later_workspace}, rounds=1)
    later_run = _run([
        sys.executable, "-m", "wmloop.cli", "research", "run",
        "--plan", later_plan["plan_path"], "--confirm",
    ], cwd=later_workspace, env=context["env"])
    later_report = json.loads(later_run.stdout)
    assert later_report["state"] == "completed"
    later_execution = later_report["campaign"]["execution"]
    assert later_execution["archive_db"] == execution["archive_db"]
    assert later_execution["cas_root"] == execution["cas_root"]
    later_pipeline = json.loads((Path(later_execution["output_root"]) / "iterations/iteration-000001/pipeline/pipeline-manifest.json").read_text())
    assert later_pipeline["verdict"] == "PASS"
    assert later_pipeline["diagnostic_probe"]["archive_trial_id"] not in {row[0] for row in archived}
    assert later_pipeline["retrieval"]["route"] == "reuse_settled"
    later_evidence = later_pipeline["retrieval"]["capsule"]["selected_evidence"]
    assert any(item["evidence"]["archive_trial_id"] in {row[0] for row in archived} for item in later_evidence)
    with sqlite3.connect(execution["archive_db"]) as connection:
        later_archived = connection.execute("SELECT trial_id,receipt_ref FROM trials").fetchall()
    assert len(later_archived) == 12
    assert set(archived).issubset(later_archived)
    assert all(path.read_bytes() == payload for path, payload in first_manifests.items())


def test_top_level_run_rounds_routes_into_same_evolution_controller(public_rounds_project):
    context = public_rounds_project
    result = _run([
        sys.executable, "-m", "wmloop.cli", "run", "improve quality",
        "--model", str(context["model"]), "--source", str(context["source"]), "--data", str(context["data"]),
        "--budget", "0.1gpu-hours", "--rounds", "2", "--stop-on-success",
        "--adapter-profile", str(context["profile"]), "--runtime-python", sys.executable,
        "--state-root", str(context["state"]), "--format", "json",
    ], cwd=ROOT, env=context["env"])
    report = json.loads(result.stdout)
    _write_json(context["root"] / "run-report.json", report)
    campaign = report["campaign"]
    assert campaign["status"] == "completed"
    assert campaign["execution"]["kind"] == "evolution"
    assert campaign["execution"]["max_iterations"] == 2
    assert campaign["execution"]["stop_on_success"] is True
    assert campaign["execution"]["total_budget_gpu_hours"] == 0.1
    assert campaign["execution_result"]["rounds_completed"] == 2
    assert campaign["execution_result"]["stop_reason"] == "MAX_ITERATIONS_REACHED"


@pytest.mark.parametrize("total", [0.00047, 0.0009], ids=["candidate-stage", "next-diagnostic"])
def test_public_rounds_exhaust_shared_total_budget_without_resetting_it(public_rounds_project, total):
    context = public_rounds_project
    plan = _plan(context, budget=f"{total}gpu-hours", rounds=20)
    result = _run([
        sys.executable, "-m", "wmloop.cli", "research", "run",
        "--plan", plan["plan_path"], "--confirm",
    ], cwd=ROOT, env=context["env"], expected=2)
    report = json.loads(result.stdout)
    _write_json(context["root"] / "run-report.json", report)
    campaign = report["campaign"]
    execution = campaign["execution"]
    outcome = campaign["execution_result"]
    assert report["state"] == "blocked"
    assert outcome["stop_reason"] == "GLOBAL_BUDGET_EXHAUSTED"
    assert 1 <= outcome["rounds_completed"] < 20
    assert outcome["evolution_manifest"]["goal_success"]["state"] != "confirmed"
    iterations_root = Path(execution["output_root"]) / "iterations"
    queues = sorted(iterations_root.glob("iteration-*/pipeline/compiled/queue/execution.json"))
    assert len(queues) >= 2, "Budget exhaustion must occur after a completed earlier round."
    for path in queues:
        queue = json.loads(path.read_text())
        assert queue["budget_db"] == execution["budget_db"]
        assert queue["budget_total_gpu_hours"] == total
    with sqlite3.connect(execution["budget_db"]) as connection:
        policy = connection.execute("SELECT total_gpu_hours FROM policy").fetchone()
        charges = connection.execute("SELECT trial_id,state,actual_gpu_hours,receipt_ref FROM trials").fetchall()
    with sqlite3.connect(execution["archive_db"]) as connection:
        archived = connection.execute("SELECT trial_id,receipt_ref FROM trials").fetchall()
    assert policy == (total,)
    assert len(charges) >= 4
    assert {row[0] for row in charges} == {row[0] for row in archived}
    assert all(row[1] == "settled" and row[2] > 0 and row[3] for row in charges)
    assert sum(row[2] for row in charges) + 0.0001 > total
    assert not list(iterations_root.rglob("budget.db"))
