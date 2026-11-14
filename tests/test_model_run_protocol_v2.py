"""CPU integration coverage for the receipt-bound model-run protocol."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from wmloop.control.model_run import compile_model_run
from wmloop.execute.gpu_lease import GpuLeaseManager
from wmloop.execute.model_runner import ModelRunnerError, execute_model_run
from wmloop.experiments.training_scale import build_training_scale_plan


_PROTOCOL_WORKER = r'''
import hashlib, json, os, sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
stage = sys.argv[2]
output = Path(manifest['output']['root'])
config = json.loads(Path(__file__).with_name('protocol-config.json').read_text())
def write(name, value):
    path = output / name
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, sort_keys=True))
    temporary.replace(path)

if stage == 'train':
    plan = manifest['training']['scale_plan']
    checkpoint = output / 'checkpoints' / 'selected.pt'
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b'cpu-trained-checkpoint')
    support_ids = list(plan['dataset']['selected_train_record_ids'])
    if config.get('bad_binding') == 'support':
        support_ids[0] = 'train-record-99999999-' + ('0' * 64)
    if config.get('bad_binding') == 'steps':
        planned_steps = plan['updates']['planned_steps'] + 1
    else:
        planned_steps = plan['updates']['planned_steps']
    if config.get('bad_binding') == 'seed':
        seed = plan['replication']['seeds'][0] + 1
    else:
        seed = plan['replication']['seeds'][0]
    if config.get('outside_output'):
        checkpoint = Path(config['outside_output'])
        checkpoint.write_bytes(b'outside-checkpoint')
    receipt = {
        'schema_version': 1,
        'artifact_type': 'verdiwm-model-checkpoint-receipt',
        'state': 'complete',
        'model_run_id': manifest['model_run_id'],
        'scale_plan_digest': manifest['training']['scale_plan_digest'],
        'checkpoint_path': str(checkpoint),
        'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'planned_steps': planned_steps,
        'seed': seed,
        'support_record_ids': support_ids,
        'claim_boundary': 'CPU fixture verifies execution provenance only.',
    }
    if config.get('bad_checkpoint_digest'):
        receipt['checkpoint_sha256'] = '0' * 64
    write('checkpoint.json', receipt)
    write('train-observed.json', {
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'physical_gpu_indexes': os.environ.get('VERDIWM_PHYSICAL_GPU_INDEXES'),
        'physical_gpu_uuids': os.environ.get('VERDIWM_PHYSICAL_GPU_UUIDS'),
    })
elif stage == 'evaluate':
    checkpoint = Path(sys.argv[3])
    if os.environ.get('VERDIWM_TRAINED_CHECKPOINT') != str(checkpoint):
        raise SystemExit('checkpoint argument and environment disagree')
    if not checkpoint.is_file():
        raise SystemExit('trained checkpoint missing')
    write('evaluate-observed.json', {
        'checkpoint_path': str(checkpoint),
        'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'physical_gpu_indexes': os.environ.get('VERDIWM_PHYSICAL_GPU_INDEXES'),
        'physical_gpu_uuids': os.environ.get('VERDIWM_PHYSICAL_GPU_UUIDS'),
    })
    evidence = {
        'schema_version': 1,
        'artifact_type': 'verdiwm-heldout-evidence-receipt',
        'state': 'complete',
        'model_run_id': manifest['model_run_id'],
        'checkpoint_path': str(checkpoint),
        'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'claim_boundary': 'CPU fixture verifies handoff provenance only.',
    }
    if config.get('evaluation_wrong_checkpoint'):
        evidence['checkpoint_sha256'] = '0' * 64
    write('evidence.json', evidence)
else:
    raise SystemExit('unknown stage')
'''


def _protocol_v2_fixture(root: Path, *, config: dict[str, object] | None = None):
    model, data = root / "model", root / "data"
    model.mkdir()
    data.mkdir()
    (model / "cpu_protocol.py").write_text(_PROTOCOL_WORKER, encoding="utf-8")
    (model / "protocol-config.json").write_text(
        json.dumps(config or {}), encoding="utf-8"
    )
    evaluator = root / "evaluator.json"
    evaluator.write_text('{"metrics":["score"],"test_fixture":true}', encoding="utf-8")
    for split in ("train", "val"):
        (data / f"{split}.json").write_text(
            json.dumps(
                {
                    "records": [
                        {
                            "episode_id": f"{split}-{index}",
                            "sample_id": f"{split}-{index}:0:150",
                            "horizon_frames": 150,
                        }
                        for index in range(20)
                    ]
                }
            ),
            encoding="utf-8",
        )
    plan = build_training_scale_plan(
        train_manifest=data / "train.json",
        val_manifest=data / "val.json",
        stage="probe",
        requested_seed_count=1,
        seed=17,
    )
    execution = {
        "kind": "pipeline",
        "repo_root": str(model),
        "output_root": str(root / "output"),
        "evaluator_contract": str(evaluator),
        "runtime_python": sys.executable,
        "budget_total_gpu_hours": 1,
        "asset_bindings": {"--checkpoint": str(model)},
        "model_run_adapter": {
            "profile_id": "cpu-protocol-v2",
            "model_family": "cpu-fixture",
            "capability_level": "L2",
            "runner": {
                "requires_training": True,
                "protocol_version": 2,
                "adapter_id": None,
                "train": ["cpu_protocol.py", "{model_run_manifest}", "train"],
                "evaluate": [
                    "cpu_protocol.py",
                    "{model_run_manifest}",
                    "evaluate",
                    "{trained_checkpoint}",
                ],
                "checkpoint_receipt": "checkpoint.json",
                "evaluation_receipt": "evidence.json",
            },
        },
    }
    allocation = {
        "artifact_type": "verdiwm-gpu-resource-allocation",
        "state": "ready",
        "roles": [
            {"role": "autonomous_candidate_evaluation", "gpu_indices": [0]}
        ],
    }
    manifest = compile_model_run(
        campaign_id="cpu-protocol-fixture",
        execution=execution,
        model=model,
        data=data,
        training_scale_plan=plan,
        resource_allocation=allocation,
    )
    output = root / "output"
    output.mkdir()
    (output / "model-run.json").write_text(json.dumps(manifest), encoding="utf-8")
    snapshot = {
        "gpus": [
            {
                "index": 0,
                "uuid": "cpu-fixture-gpu-0",
                "name": "CPU fixture",
                "memory_used_mib": 0,
                "utilization_gpu_percent": 0,
            }
        ],
        "compute_apps": [],
    }
    manager = GpuLeaseManager(
        lock_root=root / "locks", snapshot_provider=lambda: snapshot
    )
    return manifest, manager


def test_protocol_v2_runs_cpu_train_and_evaluate_with_same_checkpoint_and_gpu_lease(
    tmp_path: Path,
):
    manifest, manager = _protocol_v2_fixture(tmp_path)

    result = execute_model_run(manifest, lease_manager=manager)

    output = tmp_path / "output"
    trained = output / "checkpoints" / "selected.pt"
    train_observed = json.loads((output / "train-observed.json").read_text())
    eval_observed = json.loads((output / "evaluate-observed.json").read_text())
    assert result["state"] == "completed"
    assert [row["stage"] for row in result["stages"]] == ["train", "evaluate"]
    assert result["trained_checkpoint_path"] == str(trained)
    assert train_observed == {
        "cuda_visible_devices": "0",
        "physical_gpu_indexes": "0",
        "physical_gpu_uuids": "cpu-fixture-gpu-0",
    }
    assert eval_observed == {
        "checkpoint_path": str(trained),
        "checkpoint_sha256": result["trained_checkpoint_sha256"],
        "cuda_visible_devices": "0",
        "physical_gpu_indexes": "0",
        "physical_gpu_uuids": "cpu-fixture-gpu-0",
    }
    with manager.acquire([0]):
        pass


@pytest.mark.parametrize(
    ("config", "error_code"),
    [
        ({"outside_output": "{outside}"}, "MODEL_RUN_CHECKPOINT_OUTSIDE_OUTPUT"),
        ({"bad_binding": "steps"}, "MODEL_RUN_CHECKPOINT_RECEIPT_BINDING_MISMATCH"),
        ({"bad_binding": "seed"}, "MODEL_RUN_CHECKPOINT_RECEIPT_BINDING_MISMATCH"),
        ({"bad_binding": "support"}, "MODEL_RUN_CHECKPOINT_RECEIPT_BINDING_MISMATCH"),
        ({"bad_checkpoint_digest": True}, "MODEL_RUN_CHECKPOINT_DIGEST_MISMATCH"),
    ],
)
def test_protocol_v2_rejects_checkpoint_receipts_without_launching_evaluator(
    tmp_path: Path, config: dict[str, object], error_code: str
):
    outside = tmp_path / "outside.pt"
    rendered = {
        key: value.replace("{outside}", str(outside)) if isinstance(value, str) else value
        for key, value in config.items()
    }
    manifest, manager = _protocol_v2_fixture(tmp_path, config=rendered)

    with pytest.raises(ModelRunnerError, match=error_code):
        execute_model_run(manifest, lease_manager=manager)

    assert (tmp_path / "output/train-observed.json").is_file()
    assert not (tmp_path / "output/evaluate-observed.json").exists()
    with manager.acquire([0]):
        pass


def test_protocol_v2_rejects_evaluation_receipt_for_a_different_checkpoint(
    tmp_path: Path,
):
    manifest, manager = _protocol_v2_fixture(
        tmp_path, config={"evaluation_wrong_checkpoint": True}
    )

    with pytest.raises(ModelRunnerError, match="MODEL_RUN_EVALUATION_CHECKPOINT_MISMATCH"):
        execute_model_run(manifest, lease_manager=manager)

    assert (tmp_path / "output/evaluate-observed.json").is_file()
    with manager.acquire([0]):
        pass
