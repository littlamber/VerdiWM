"""Generic runtime results share one strict, extensible evidence contract."""

from __future__ import annotations

import json
import sqlite3
import sys

import pytest

from wmloop.archive.store import ArchiveStore, ContentAddressedStore
from wmloop.contracts import ContractValidationError, validate_document
from wmloop.execute import auto_experiment, cuda_smoke_workload, external_evaluator_workload
from wmloop.storage import canonical_bytes
from wmloop.verify.auto_experiment import verify_auto_experiment_result


@pytest.fixture
def result() -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_type": "verdiwm-auto-experiment-result",
        "state": "ready",
        "device": {"type": "cuda", "gpu_uuid": "GPU-fixture"},
        "metrics": {"runtime_ready": 1.0, "unused_quality": 0.5},
    }


def _sampling() -> dict[str, object]:
    return {"samples": [
        {"status": "ready", "gpu_uuid": "GPU-fixture", "phase": "before",
         "memory_used_mib": 0.0, "utilization_gpu_percent": 0.0},
        {"status": "ready", "gpu_uuid": "GPU-fixture", "phase": "during",
         "memory_used_mib": 64.0, "utilization_gpu_percent": 10.0},
    ]}


def _verify(result: dict[str, object]) -> dict[str, object]:
    return verify_auto_experiment_result(
        result=result,
        metric_gates=({"metric": "runtime_ready", "operator": "gte", "threshold": 1.0},),
        expected_gpu_uuid="GPU-fixture", gpu_sampling=_sampling(), stage="screen",
    )


def _reject_contract_and_verifier(result: dict[str, object]) -> None:
    with pytest.raises(ContractValidationError):
        validate_document("auto_experiment_result", result)
    verdict = _verify(result)
    assert verdict["verdict"] == "VOID"
    assert verdict["evidence_level"] == "void"
    assert any(item["code"] == "RESULT_CONTRACT_INVALID" for item in verdict["blockers"])
    # Invalid values must not escape into the terminal verdict artifact.
    canonical_bytes(verdict)


@pytest.mark.parametrize("value", [None, [], "invalid", 1])
def test_nonobject_result_root_is_void(value):
    _reject_contract_and_verifier(value)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"),
                                   True, False, "0.5", None, [], {}, 10**400])
def test_even_ungated_invalid_metrics_fail_contract_and_verifier(result, value):
    result["metrics"]["unused_quality"] = value
    _reject_contract_and_verifier(result)


@pytest.mark.parametrize("field", ["schema_version", "artifact_type", "state", "device", "metrics"])
def test_missing_required_fields_fail_contract_and_verifier(result, field):
    result.pop(field)
    _reject_contract_and_verifier(result)


@pytest.mark.parametrize("field,value", [
    ("schema_version", True), ("schema_version", 2), ("artifact_type", "other"),
    ("state", "partial"), ("state", float("nan")), ("metrics", {}),
    ("metrics", {"": 1.0}), ("metrics", []), ("device", []),
    ("workload", []), ("workload", {"command_sha256": True}),
    ("workload", {"matrix_size": False}), ("workload", {"dtype": 1}),
    ("claim_boundary", False), ("undeclared", 1.0),
])
def test_structure_and_types_fail_contract_and_verifier(result, field, value):
    result[field] = value
    _reject_contract_and_verifier(result)


@pytest.mark.parametrize("field,value", [
    ("type", "cpu"), ("gpu_uuid", ""), ("gpu_uuid", True),
    ("physical_index", False), ("physical_index", -1), ("logical_index", "0"),
    ("total_memory_bytes", True), ("name", False), ("torch_cuda_version", 12),
    ("undeclared", 1.0),
])
def test_device_fields_have_explicit_types(result, field, value):
    result["device"][field] = value
    _reject_contract_and_verifier(result)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), 10**400, {1, 2}])
def test_extensible_metadata_still_rejects_nonfinite_or_nonjson_values(result, value):
    result["workload"] = {"custom": {"nested": [value]}}
    _reject_contract_and_verifier(result)


def test_custom_nested_workload_and_minimal_device_survive_cas_unchanged(result, tmp_path):
    result["workload"] = {"custom_model": "adapter", "settings": {
        "schedule": [1, 2.5, {"enabled": True}], "optional": None,
    }}
    original = canonical_bytes(result)
    cas = ContentAddressedStore(tmp_path / "cas")
    ref = cas.put_bytes(original, media_type="application/json")
    restored = json.loads(cas.read_bytes(ref.uri))
    validate_document("auto_experiment_result", restored)
    assert canonical_bytes(restored) == original
    assert _verify(restored)["verdict"] == "PASS"


def test_valid_failure_and_wrong_uuid_remain_void(result):
    result["state"] = "invalid"
    validate_document("auto_experiment_result", result)
    assert _verify(result)["verdict"] == "VOID"
    result["state"] = "ready"
    result["device"]["gpu_uuid"] = "GPU-other"
    assert _verify(result)["verdict"] == "VOID"
    result["device"]["gpu_uuid"] = "GPU-fixture"
    assert verify_auto_experiment_result(
        result=result, metric_gates=({"metric": "runtime_ready", "operator": "gte", "threshold": 1.0},),
        expected_gpu_uuid="GPU-fixture", gpu_sampling={"samples": []}, stage="screen",
    )["verdict"] == "VOID"


@pytest.mark.parametrize("exit_code", [0, 3])
def test_external_producer_emits_schema_valid_cpu_fixture_results(tmp_path, exit_code, capsys):
    # A real CPU child tests the producer contract; it makes no GPU-use claim.
    status = external_evaluator_workload.run_external_evaluator(
        external_evaluator_workload.ExternalEvaluatorOptions(
            command=(sys.executable, "-c", f"raise SystemExit({exit_code})"),
            scratch_root=tmp_path, working_directory=tmp_path, artifacts=(),
        ),
        environment={"VERDIWM_PHYSICAL_GPU_INDEX": "0", "VERDIWM_PHYSICAL_GPU_UUID": "GPU-fixture"},
        identity_provider=lambda index: {"gpu_uuid": "GPU-fixture", "name": "CPU identity fixture"},
    )
    output = json.loads((tmp_path / "result.json").read_text())
    validate_document("auto_experiment_result", output)
    assert status == exit_code
    assert output["state"] == ("ready" if exit_code == 0 else "invalid")
    # The pre-existing stdout JSON interface is intentionally preserved.
    assert json.loads(capsys.readouterr().out) == output


def test_external_producer_checks_result_before_writing(tmp_path, monkeypatch):
    (tmp_path / "metrics.json").write_text('{"state":"ready","metrics":{"quality":1}}')
    monkeypatch.setattr(external_evaluator_workload, "_load_external_metrics", lambda *args, **kwargs: {"quality": True})
    with pytest.raises(external_evaluator_workload.ExternalEvaluatorError, match="RESULT_INVALID"):
        external_evaluator_workload.run_external_evaluator(
            external_evaluator_workload.ExternalEvaluatorOptions(
                command=(sys.executable, "-c", "pass"), scratch_root=tmp_path,
                working_directory=tmp_path, artifacts=("metrics.json=metrics.json",),
                metrics_path="metrics.json",
            ),
            environment={"VERDIWM_PHYSICAL_GPU_INDEX": "0", "VERDIWM_PHYSICAL_GPU_UUID": "GPU-fixture"},
            identity_provider=lambda index: {"gpu_uuid": "GPU-fixture", "name": "CPU identity fixture"},
        )
    assert not (tmp_path / "result.json").exists()


@pytest.mark.parametrize("value", [True, "1", None, float("nan"), float("inf"), 10**400])
def test_external_invalid_metrics_become_schema_valid_failure_results(tmp_path, value, capsys):
    (tmp_path / "metrics.json").write_text(json.dumps({"state": "ready", "metrics": {"quality": value}}))
    status = external_evaluator_workload.run_external_evaluator(
        external_evaluator_workload.ExternalEvaluatorOptions(
            command=(sys.executable, "-c", "pass"), scratch_root=tmp_path,
            working_directory=tmp_path, artifacts=("metrics.json=metrics.json",),
            metrics_path="metrics.json",
        ),
        environment={"VERDIWM_PHYSICAL_GPU_INDEX": "0", "VERDIWM_PHYSICAL_GPU_UUID": "GPU-fixture"},
        identity_provider=lambda index: {"gpu_uuid": "GPU-fixture", "name": "CPU identity fixture"},
    )
    output = json.loads((tmp_path / "result.json").read_bytes())
    validate_document("auto_experiment_result", output)
    assert status == 2
    assert output["state"] == "invalid"
    assert output["metrics"]["artifact_error_count"] == 1.0
    assert "quality" not in output["metrics"]
    assert json.loads(capsys.readouterr().out) == output


def test_cuda_producer_writer_checks_contract_without_loading_torch(result, tmp_path):
    result["device"].update(logical_index=0, physical_index=0, name="Fixture", total_memory_bytes=1024,
                            torch_cuda_version="12.4")
    result["workload"] = {"matrix_size": 16, "dtype": "float32", "minimum_seconds": 0.1}
    path = tmp_path / "cuda-result.json"
    cuda_smoke_workload._write_json_atomic(path, result)
    validate_document("auto_experiment_result", json.loads(path.read_bytes()))
    result["metrics"]["unused_quality"] = float("inf")
    with pytest.raises(ContractValidationError):
        cuda_smoke_workload._write_json_atomic(tmp_path / "invalid.json", result)
    assert not (tmp_path / "invalid.json").exists()


@pytest.mark.parametrize("damage", [False, True])
def test_settlement_archives_result_bytes_but_never_promotes_screen(result, tmp_path, damage):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    if damage:
        result["metrics"]["unused_quality"] = True
    original = json.dumps(result, indent=2).encode()
    (scratch / "result.json").write_bytes(original)
    for name in ("stdout.log", "stderr.log"):
        (scratch / name).write_bytes(b"")
    (scratch / "gpu-sampling.json").write_bytes(canonical_bytes(_sampling()))
    cas = ContentAddressedStore(tmp_path / "cas")
    archive = ArchiveStore(tmp_path / "archive.db")
    plan = {
        "campaign_id": "campaign", "trial_id": "trial", "objective": "Test screen settlement",
        "hypothesis": "A valid result can settle", "selection_reason": "Contract regression",
        "falsification_criterion": "Corrupt result passes", "stage": "screen", "result_path": "result.json",
        "artifacts": ["result.json"], "total_budget_gpu_hours": 1.0, "estimated_gpu_hours": 0.01,
        "metric_gates": [{"metric": "runtime_ready", "operator": "gte", "threshold": 1.0}],
    }
    receipt = auto_experiment._settle_trial(
        plan=plan, plan_sha256="a" * 64, trial_signature="b" * 64, archive_trial_id="trial",
        admission_token=1, execution={"lease": {"uuid": "GPU-fixture"}, "gpu_sampling": _sampling(),
            "timed_out": False, "exit_code": 0, "duration_seconds": 0.1},
        scratch=scratch, workspace_root=tmp_path, cas=cas, archive=archive,
        budget_db=tmp_path / "budget.db", budget_total_gpu_hours=1.0,
        budget_max_trial_gpu_hours=1.0, budget_high_trial_limit=1,
        budget_require_high_cost_approval=True, conformance_ref=None,
    )
    auto_experiment._record_archive_from_receipt(archive=archive, receipt=receipt)
    assert receipt["verdict"]["verdict"] == ("VOID" if damage else "PASS")
    assert cas.read_bytes(receipt["artifact_refs"]["result.json"]) == original
    assert archive.has_settled_receipt(
        trial_id="trial", receipt_ref=receipt["receipt_ref"], receipt_hash=receipt["receipt_hash"],
    )
    with sqlite3.connect(tmp_path / "archive.db") as connection:
        settlement = json.loads(connection.execute("SELECT settlement_json FROM trials").fetchone()[0])
    assert settlement["evidence_scope"] == "exploratory"
    assert archive.visible_settled_trials() == ("trial",)
    assert not archive.list_cells()
