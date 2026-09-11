import json
from pathlib import Path
import sys

from test_open_method_study import request as study_request
from wmloop.control.open_method_study import compile_open_method_study
from wmloop.execute.gpu_lease import GpuLeaseManager
from wmloop.execute.open_method_study_runner import (
    artifact_digest,
    build_open_method_verifier,
    execute_open_method_study,
)

ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _runner_fixture(tmp_path: Path, *, overlapping_splits: bool = False) -> dict[str, object]:
    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.write_bytes(b"checkpoint-v1")
    train = tmp_path / "train.json"
    selection = tmp_path / "selection.json"
    confirmation = tmp_path / "confirmation.json"
    _write_json(train, {"episode_ids": ["train-1", "train-2"]})
    _write_json(selection, {"episode_ids": ["selection-1", "selection-2"]})
    _write_json(
        confirmation,
        {"episode_ids": ["selection-1", "confirmation-2"]}
        if overlapping_splits
        else {"episode_ids": ["confirmation-1", "confirmation-2"]},
    )

    verifier_root = tmp_path / "verifier"
    verifier_root.mkdir()
    (verifier_root / "verify.py").write_text(
        """import json, sys
from pathlib import Path
input_doc = json.loads(Path(sys.argv[1]).read_text())
role = input_doc['role']
phase = input_doc['phase']
seed = input_doc['seed']
values = {'baseline': 1.0, 'source_only': 0.8, 'target_only': 0.7, 'combined': 0.4}
Path(sys.argv[2]).mkdir(parents=True, exist_ok=True)
Path(sys.argv[2], 'evaluation.json').write_text(json.dumps({
    'study_id': input_doc['study_id'], 'method_id': input_doc['method_id'],
    'role': role, 'phase': phase, 'seed': seed, 'verifier_id': input_doc['verifier_id'],
    'validity_gates': {'data_bound': True},
    'unit_metrics': [
        {'unit_id': 'episode-1', 'metrics': {'identity_error': values[role], 'action_fidelity': 1.0}},
        {'unit_id': 'episode-2', 'metrics': {'identity_error': values[role], 'action_fidelity': 1.0}},
    ],
    'claim_boundary': 'Paired verifier observation only; settlement is performed by the study runner.'
}))
""",
        encoding="utf-8",
    )
    verifier_spec = build_open_method_verifier(
        verifier_root=verifier_root,
        output_path=verifier_root / "verifier.json",
        command=["python", "verify.py", "{input_manifest}", "{output_root}"],
        implementation_files=["verify.py"],
        primary_metrics=["identity_error"],
        protected_metrics=["action_fidelity"],
        metric_policy={
            "identity_error": {"direction": "minimize", "minimum_effect": 0.05, "maximum_regression": 0.0},
            "action_fidelity": {"direction": "maximize", "minimum_effect": 0.0, "maximum_regression": 0.02},
        },
        required_validity_gates=["data_bound"],
        effect_context={
            "backbone_family": "fixture",
            "capability_class": "recurrent",
            "goal_schema": "long_horizon",
            "outcome_schema": "paired_metrics",
            "chart_id": "chart-fixture",
            "data_regime": "fixture",
            "horizons": [30, 60],
        },
    )
    req = study_request()
    req["experiment_binding"] = {
        "checkpoint_digest": artifact_digest(checkpoint),
        "train_split_digest": artifact_digest(train),
        "selection_split_digest": artifact_digest(selection),
        "confirmation_split_digest": artifact_digest(confirmation),
        "verifier_digest": artifact_digest(verifier_root / "verifier.json"),
    }
    study = compile_open_method_study(
        **req, output_root=tmp_path / "study", project_root=ROOT
    )
    manager = GpuLeaseManager(
        lock_root=tmp_path / "leases",
        snapshot_provider=lambda: {
            "gpus": [{
                "index": 0, "uuid": "fixture-gpu-0", "name": "fixture",
                "memory_used_mib": 0, "utilization_gpu_percent": 0,
            }],
            "compute_apps": [],
        },
    )
    return {
        "study": tmp_path / "study",
        "checkpoint": checkpoint,
        "train": train,
        "selection": selection,
        "confirmation": confirmation,
        "verifier": verifier_root / "verifier.json",
        "manager": manager,
        "study_id": study["study_id"],
    }


def test_study_runner_executes_calibration_paired_verification_and_memory(tmp_path):
    fixture = _runner_fixture(tmp_path)
    result = execute_open_method_study(
        study_root=fixture["study"],
        checkpoint=fixture["checkpoint"],
        train_split=fixture["train"],
        selection_split=fixture["selection"],
        confirmation_split=fixture["confirmation"],
        verifier=fixture["verifier"],
        output_root=tmp_path / "execution",
        runtime_python=Path(sys.executable),
        gpu_indices=[0],
        max_parallel=1,
        lease_manager=fixture["manager"],
    )
    assert result["state"] == "completed"
    assert result["split_validation"]["state"] == "verified"
    assert len(result["calibrations"]) == 4
    assert all(row["state"] == "passed" for row in result["calibrations"])
    assert len(result["trials"]) == 8
    assert all(row["state"] == "passed" for row in result["trials"])
    assert result["confirmation_settlement"]["state"] == "settled"
    assert result["knowledge"]["effect_record_count"] == 3
    assert result["authority"] == {"local_evidence": True, "community_projection": False, "promotion": False}
    assert (tmp_path / "execution" / "execution.json").is_file()
    assert (tmp_path / "execution" / "evidence-index.json").is_file()
    records = json.loads((tmp_path / "execution" / "knowledge" / "effect-records.json").read_text())
    assert {row["status"] for row in records["records"]} == {"confirmed"}


def test_study_runner_refuses_unproven_episode_disjointness(tmp_path):
    fixture = _runner_fixture(tmp_path, overlapping_splits=True)
    try:
        execute_open_method_study(
            study_root=fixture["study"], checkpoint=fixture["checkpoint"],
            train_split=fixture["train"], selection_split=fixture["selection"],
            confirmation_split=fixture["confirmation"], verifier=fixture["verifier"],
            output_root=tmp_path / "execution", runtime_python=Path(sys.executable),
            gpu_indices=[0], lease_manager=fixture["manager"],
        )
    except Exception as exc:
        assert "SPLIT_EPISODE_OVERLAP" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("overlapping split unexpectedly executed")
    assert not (tmp_path / "execution").exists()
