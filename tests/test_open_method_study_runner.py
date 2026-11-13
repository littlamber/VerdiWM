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
    _evaluation_gates_valid,
)

ROOT = Path(__file__).resolve().parents[1]


def test_raw_evidence_rejects_candidate_symlinks(tmp_path):
    import hashlib
    import pytest
    from wmloop.archive.store import ContentAddressedStore
    from wmloop.execute.open_method_study_runner import OpenMethodStudyExecutionError, _archive_raw_files

    root = tmp_path / "execution"
    root.mkdir()
    outside = tmp_path / "private.bin"
    outside.write_bytes(b"private")
    (root / "candidate-link").symlink_to(outside)
    cas = ContentAddressedStore(root / "evidence")
    with pytest.raises(OpenMethodStudyExecutionError, match="EVIDENCE_SYMLINK"):
        _archive_raw_files(root, cas)
    digest = hashlib.sha256(b"private").hexdigest()
    assert not (root / "evidence" / "cas" / digest[:2] / digest).exists()


def test_study_loader_rejects_linked_study_files(tmp_path):
    import pytest
    from wmloop.execute.open_method_study_runner import OpenMethodStudyExecutionError, _load_study

    fixture = _runner_fixture(tmp_path)
    root = fixture["study"]
    manifest = root / "study.json"
    study = json.loads(manifest.read_text())
    original = root / next(iter(study["file_sha256"]))
    moved = tmp_path / "original"
    original.rename(moved)
    original.symlink_to(moved)
    with pytest.raises(OpenMethodStudyExecutionError, match="STUDY_SYMLINK"):
        _load_study(root)


def test_study_loader_rejects_linked_manifest(tmp_path):
    import pytest
    from wmloop.execute.open_method_study_runner import OpenMethodStudyExecutionError, _load_study

    fixture = _runner_fixture(tmp_path)
    root = fixture["study"]
    manifest = root / "study.json"
    moved = tmp_path / "study-original.json"
    manifest.rename(moved)
    manifest.symlink_to(moved)
    with pytest.raises(OpenMethodStudyExecutionError, match="STUDY_PATH_INVALID"):
        _load_study(root)


def test_frozen_input_snapshot_rejects_changed_verifier(tmp_path):
    import pytest
    from wmloop.archive.store import ContentAddressedStore
    from wmloop.execute.open_method_study_runner import OpenMethodStudyExecutionError, _archive_frozen_inputs

    fixture = _runner_fixture(tmp_path)
    study = json.loads((fixture["study"] / "study.json").read_text())
    verifier = json.loads(fixture["verifier"].read_text())
    (fixture["verifier"].parent / "verify.py").write_text("changed", encoding="utf-8")
    paths = {"train_split": fixture["train"], "selection_split": fixture["selection"],
             "confirmation_split": fixture["confirmation"], "verifier": fixture["verifier"]}
    with pytest.raises(OpenMethodStudyExecutionError, match="VERIFIER_CHANGED"):
        _archive_frozen_inputs(fixture["study"], study, paths, verifier,
                               ContentAddressedStore(tmp_path / "cas-root"))


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
        {'unit_id': episode_id, 'metrics': {'identity_error': values[role], 'action_fidelity': 1.0}}
        for episode_id in json.loads(Path(input_doc['split']).read_text())['episode_ids']
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


def test_study_runner_retains_observations_without_claiming_independent_confirmation(tmp_path):
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
    assert result["state"] == "abstained"
    assert result["split_validation"]["state"] == "verified"
    assert len(result["calibrations"]) == 4
    assert all(row["state"] == "passed" for row in result["calibrations"])
    assert len(result["trials"]) == 16
    assert all(row["state"] == "passed" for row in result["trials"])
    assert result["confirmation_settlement"]["state"] == "abstained"
    assert result["confirmation_settlement"]["reason"] == "FORMAL_VERIFICATION_REQUIRED"
    assert result["confirmation_settlement"]["metrics"]
    selection = [row for row in result["trials"] if row["evaluations"][0]["phase"] == "selection"]
    confirmation = [row for row in result["trials"] if row["evaluations"][0]["phase"] == "confirmation"]
    assert {row["seed"] for row in selection} == {11, 22}
    assert {row["seed"] for row in confirmation} == {33, 44}
    for row in selection:
        manifest = next((tmp_path / "execution" / "trials" / "selection" / row["role"] / f"seed-{row['seed']}" / "selection").rglob("input.json"))
        assert str(fixture["confirmation"]) not in manifest.read_text()
    assert result["knowledge"]["effect_record_count"] == 0
    assert result["authority"] == {"local_evidence": False, "community_projection": False, "promotion": False}
    assert (tmp_path / "execution" / "execution.json").is_file()
    assert (tmp_path / "execution" / "evidence-index.json").is_file()
    from wmloop.archive.store import ContentAddressedStore
    cas = ContentAddressedStore(tmp_path / "execution" / "evidence")
    index = json.loads((tmp_path / "execution" / "evidence-index.json").read_text())
    assert json.loads(cas.read_bytes(result["raw_evidence_manifest_ref"])) == index["raw"]
    assert json.loads(cas.read_bytes(index["execution_receipt"])) == result
    frozen = json.loads(cas.read_bytes(result["frozen_inputs_manifest_ref"]))
    assert json.loads(cas.read_bytes(frozen["study_files"]["study.json"]))["study_id"] == fixture["study_id"]
    assert cas.read_bytes(frozen["inputs"]["confirmation_split"]) == fixture["confirmation"].read_bytes()
    assert cas.read_bytes(frozen["verifier_implementation"]["verify.py"]) == (fixture["verifier"].parent / "verify.py").read_bytes()
    assert frozen["checkpoint_digest"] == artifact_digest(fixture["checkpoint"])
    from wmloop.verify.open_method_evidence import OpenMethodEvidenceError, read_frozen_inputs
    reconstructed = read_frozen_inputs(cas, index["execution_receipt"])
    assert reconstructed["study"]["study_id"] == fixture["study_id"]
    assert reconstructed["splits"]["confirmation_split"]["episode_ids"] == ["confirmation-1", "confirmation-2"]
    import pytest
    digest = frozen["verifier_implementation"]["verify.py"].rsplit("/", 1)[-1]
    (tmp_path / "execution" / "evidence" / "cas" / digest[:2] / digest).write_bytes(b"corrupt")
    with pytest.raises(OpenMethodEvidenceError, match="MEMBER_INVALID"):
        read_frozen_inputs(cas, index["execution_receipt"])
    assert not (tmp_path / "execution" / "knowledge" / "effect-records.json").exists()


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


def test_verifier_rejects_linked_implementation_before_execution(tmp_path):
    import pytest
    from wmloop.execute.open_method_study_runner import OpenMethodStudyExecutionError

    fixture = _runner_fixture(tmp_path)
    implementation = fixture["verifier"].parent / "verify.py"
    moved = implementation.with_name("verify-original.py")
    implementation.rename(moved)
    implementation.symlink_to(moved)
    with pytest.raises(OpenMethodStudyExecutionError, match="VERIFIER_PATH_INVALID"):
        execute_open_method_study(
            study_root=fixture["study"], checkpoint=fixture["checkpoint"],
            train_split=fixture["train"], selection_split=fixture["selection"],
            confirmation_split=fixture["confirmation"], verifier=fixture["verifier"],
            output_root=tmp_path / "execution", runtime_python=Path(sys.executable),
            gpu_indices=[0], lease_manager=fixture["manager"],
        )
    assert not (tmp_path / "execution").exists()


def test_verifier_builder_rejects_linked_implementation_parent(tmp_path):
    import pytest
    from wmloop.execute.open_method_study_runner import OpenMethodStudyExecutionError

    fixture = _runner_fixture(tmp_path)
    root = fixture["verifier"].parent
    spec = json.loads(fixture["verifier"].read_text())
    (root / "linked").symlink_to(root, target_is_directory=True)
    with pytest.raises(OpenMethodStudyExecutionError, match="IMPLEMENTATION_PATH_INVALID"):
        build_open_method_verifier(
            verifier_root=root, output_path=root / "other.json", command=spec["command"],
            implementation_files=["linked/verify.py"], primary_metrics=spec["primary_metrics"],
            protected_metrics=spec["protected_metrics"], metric_policy=spec["metric_policy"],
            required_validity_gates=spec["required_validity_gates"],
            effect_context=spec["effect_context"],
        )
    assert not (root / "other.json").exists()


def test_verifier_builder_rejects_output_outside_implementation_root(tmp_path):
    import pytest
    from wmloop.execute.open_method_study_runner import OpenMethodStudyExecutionError

    fixture = _runner_fixture(tmp_path)
    root = fixture["verifier"].parent
    spec = json.loads(fixture["verifier"].read_text())
    output = tmp_path / "elsewhere" / "verifier.json"
    with pytest.raises(OpenMethodStudyExecutionError, match="OUTPUT_ROOT_MISMATCH"):
        build_open_method_verifier(
            verifier_root=root, output_path=output, command=spec["command"],
            implementation_files=["verify.py"], primary_metrics=spec["primary_metrics"],
            protected_metrics=spec["protected_metrics"], metric_policy=spec["metric_policy"],
            required_validity_gates=spec["required_validity_gates"],
            effect_context=spec["effect_context"],
        )
    assert not output.exists()


def test_verifier_builder_concurrent_writers_do_not_replace_each_other(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    from wmloop.execute.open_method_study_runner import OpenMethodStudyExecutionError

    fixture = _runner_fixture(tmp_path)
    root = fixture["verifier"].parent
    spec = json.loads(fixture["verifier"].read_text())
    barrier = Barrier(2)

    def build():
        barrier.wait()
        try:
            return build_open_method_verifier(
                verifier_root=root, output_path=root / "concurrent.json", command=spec["command"],
                implementation_files=["verify.py"], primary_metrics=spec["primary_metrics"],
                protected_metrics=spec["protected_metrics"], metric_policy=spec["metric_policy"],
                required_validity_gates=spec["required_validity_gates"],
                effect_context=spec["effect_context"],
            )["verifier_id"]
        except OpenMethodStudyExecutionError as exc:
            return str(exc)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: build(), range(2)))
    assert results.count("OPEN_VERIFIER_OUTPUT_EXISTS") == 1
    assert results.count(json.loads((root / "concurrent.json").read_text())["verifier_id"]) == 1


def test_paired_observations_must_cover_the_bound_split():
    from wmloop.execute.open_method_study_runner import _settle_phase

    roles = ("baseline", "source_only", "target_only", "combined")
    gates = {name: True for name in ("data_bound", "verifier_process", "evaluation_schema", "identity_binding")}
    trials = [
        {"role": role, "seed": seed, "state": "passed", "evaluations": [{
            "phase": "selection", "validity_gates": gates,
            "unit_metrics": [{"unit_id": unit, "metrics": {"quality": 1.0}} for unit in ("wrong-1", "wrong-2")],
        }]}
        for role in roles for seed in (1, 2)
    ]
    result = _settle_phase(
        {"paired_seeds": [1, 2]}, trials,
        {"required_validity_gates": ["data_bound"], "primary_metrics": ["quality"], "protected_metrics": [],
         "metric_policy": {"quality": {"direction": "maximize"}}},
        phase="selection", expected_units={"heldout-1", "heldout-2"},
    )
    assert result["state"] == "abstained"
    assert result["reason"] == "SPLIT_EPISODE_COVERAGE_MISMATCH"


def test_paired_settlement_rejects_failed_and_duplicate_trials():
    from wmloop.execute.open_method_study_runner import _settle_phase

    roles = ("baseline", "source_only", "target_only", "combined")
    gates = {name: True for name in ("data_bound", "verifier_process", "evaluation_schema", "identity_binding")}
    trials = [
        {"role": role, "seed": seed, "state": "passed", "evaluations": [{
            "phase": "selection", "validity_gates": gates,
            "unit_metrics": [{"unit_id": unit, "metrics": {"quality": 1.0}} for unit in ("a", "b")],
        }]}
        for role in roles for seed in (1, 2)
    ]
    study = {"paired_seeds": [1, 2]}
    verifier = {"required_validity_gates": ["data_bound"], "primary_metrics": ["quality"],
                "protected_metrics": [], "metric_policy": {"quality": {"direction": "maximize"}}}
    def settle(rows):
        return _settle_phase(study, rows, verifier, phase="selection", expected_units={"a", "b"})

    assert settle(trials)["state"] == "settled"
    assert settle([{**trials[0], "state": "failed"}, *trials[1:]])["reason"] == "PAIRED_TRIAL_FAILED"
    assert settle([*trials, trials[0]])["reason"] == "DUPLICATE_PAIRED_TRIAL"
    assert settle([{**trials[0], "evaluations": trials[0]["evaluations"] * 2}, *trials[1:]])["reason"] == "DUPLICATE_PAIRED_OBSERVATION"


def test_split_validation_rejects_empty_duplicate_and_blank_episode_ids(tmp_path):
    import pytest
    from wmloop.execute.open_method_study_runner import (
        OpenMethodStudyExecutionError, _validate_disjoint_splits,
    )

    paths = {name: tmp_path / f"{name}.json" for name in
             ("train_split", "selection_split", "confirmation_split")}
    for name, path in paths.items():
        _write_json(path, {"episode_ids": [name]})
    for invalid in ([], ["duplicate", "duplicate"], [" "]):
        _write_json(paths["selection_split"], {"episode_ids": invalid})
        with pytest.raises(OpenMethodStudyExecutionError, match="SPLIT_DISJOINTNESS_UNPROVEN"):
            _validate_disjoint_splits(paths)


def test_calibration_does_not_swallow_interrupt(tmp_path, monkeypatch):
    import pytest
    import wmloop.execute.open_method_study_runner as runner

    def interrupt(**kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(runner, "run_method_calibration", interrupt)
    with pytest.raises(KeyboardInterrupt):
        runner._run_calibrations(
            study_path=tmp_path, study={"arms": [{"role": "baseline", "bundle_path": "bundle", "method_id": "method"}]},
            output_root=tmp_path / "out", timeout_seconds=1, runtime_python=Path(sys.executable),
        )


def test_trial_success_requires_schema_and_identity_gates():
    verifier = {"required_validity_gates": ["data_bound"]}
    valid = [
        {"validity_gates": {"data_bound": True, "verifier_process": True, "evaluation_schema": True, "identity_binding": True}},
        {"validity_gates": {"data_bound": True, "verifier_process": True, "evaluation_schema": True, "identity_binding": True}},
    ]
    invalid = [dict(valid[0]), {"validity_gates": {"data_bound": True, "verifier_process": True, "evaluation_schema": False, "identity_binding": True}}]

    assert _evaluation_gates_valid(valid[:1], verifier)
    assert not _evaluation_gates_valid(invalid[1:], verifier)
    truthy = {"validity_gates": {**valid[0]["validity_gates"], "data_bound": "true"}}
    assert not _evaluation_gates_valid([truthy], verifier)


def test_failed_trial_charges_time_spent_under_gpu_lease(tmp_path, monkeypatch):
    import wmloop.execute.open_method_study_runner as runner

    class Lease:
        index = 0
        def to_document(self):
            return {"index": 0}
        def release(self):
            self.released = True

    lease = Lease()
    class Manager:
        def acquire(self, *args, **kwargs):
            return lease

    times = iter([10.0, 25.0])
    monkeypatch.setattr(runner.time, "monotonic", lambda: next(times))
    def fail(**kwargs):
        raise RuntimeError("injected launch failure")
    monkeypatch.setattr(runner, "run_candidate_operation", fail)
    result = runner._run_trial(
        study_path=tmp_path, study={"study_id": "study"},
        arm={"role": "baseline", "bundle_path": "bundle", "estimated_gpu_hours_per_seed": 1},
        method={"method_id": "method", "training": {"mode": "training"}}, seed=1,
        phase="selection", paths={"checkpoint": tmp_path, "train_split": tmp_path},
        verifier_spec={"verifier_id": "verifier"}, output_root=tmp_path / "trial",
        runtime_python=Path(sys.executable), gpu_indices=[0], lease_manager=Manager(), gpu_wait_seconds=0,
    )
    assert result["state"] == "failed"
    assert result["gpu_seconds"] == 15.0
    assert lease.released


def test_input_drift_does_not_publish_execution_receipt(tmp_path, monkeypatch):
    import pytest
    import wmloop.execute.open_method_study_runner as runner

    fixture = _runner_fixture(tmp_path)
    original = runner._write_knowledge

    def drift_after_execution(**kwargs):
        result = original(**kwargs)
        fixture["checkpoint"].write_bytes(b"changed-checkpoint")
        return result

    monkeypatch.setattr(runner, "_write_knowledge", drift_after_execution)
    with pytest.raises(runner.OpenMethodStudyExecutionError, match="BINDING"):
        execute_open_method_study(
            study_root=fixture["study"], checkpoint=fixture["checkpoint"],
            train_split=fixture["train"], selection_split=fixture["selection"],
            confirmation_split=fixture["confirmation"], verifier=fixture["verifier"],
            output_root=tmp_path / "execution", runtime_python=Path(sys.executable),
            gpu_indices=[0], lease_manager=fixture["manager"],
        )
    assert not (tmp_path / "execution" / "execution.json").exists()
    assert not (tmp_path / "execution" / "evidence-index.json").exists()


def test_failed_selection_never_runs_confirmation(tmp_path, monkeypatch):
    import wmloop.execute.open_method_study_runner as runner

    fixture = _runner_fixture(tmp_path)
    phases = []
    monkeypatch.setattr(runner, "_run_calibrations", lambda **kwargs: [
        {"role": role, "state": "passed"} for role in runner.ROLES
    ])

    def fail_trial(**kwargs):
        phases.append(kwargs["phase"])
        return runner._failed_trial(str(kwargs["arm"]["role"]), kwargs["seed"], RuntimeError("failure"))

    monkeypatch.setattr(runner, "_run_trial", fail_trial)
    result = execute_open_method_study(
        study_root=fixture["study"], checkpoint=fixture["checkpoint"],
        train_split=fixture["train"], selection_split=fixture["selection"],
        confirmation_split=fixture["confirmation"], verifier=fixture["verifier"],
        output_root=tmp_path / "execution", runtime_python=Path(sys.executable),
        gpu_indices=[0], lease_manager=fixture["manager"],
    )
    assert phases == ["selection"] * 8
    assert result["confirmation_settlement"]["reason"] == "INCOMPLETE_PAIRED_OBSERVATIONS"
    assert result["knowledge"]["effect_record_count"] == 0
