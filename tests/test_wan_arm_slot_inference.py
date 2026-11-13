"""The native Wan bridge enforces bounds without loading a model in CI."""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from wmloop.archive.store import ContentAddressedStore
from wmloop.contracts import ContractValidationError, validate_document
from wmloop.execute.budget import BudgetError, BudgetLedger, BudgetPolicy
from wmloop.execute.gpu_lease import GpuLeaseManager
from wmloop.execute.wan_arm_slot import WanArmSlotError, run_inference


def _request(tmp_path: Path, monkeypatch) -> dict:
    source = tmp_path / "source"
    script = source / "scripts/evaluation/rollout_full_episode_sa_wm.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        "import json, os, pathlib, sys\n"
        "args = sys.argv[1:]\n"
        "out = pathlib.Path(args[args.index('--output_dir')+1])\n"
        "out.mkdir()\n"
        "(out/'observed.json').write_text(json.dumps({'args':args,'gpu':os.environ.get('CUDA_VISIBLE_DEVICES')}))\n"
        "value = lambda flag: args[args.index(flag)+1]\n"
        "rows = []\n"
        "for index, selected in enumerate(json.loads(pathlib.Path(value('--selection')).read_text())['selected'][:int(value('--limit'))]):\n"
        "    folder = out/str(index); folder.mkdir()\n"
        "    row = dict(selected, selection_sample_index=index, dir=folder.name, chunks=1, frames=13, fps=float(value('--fps')))\n"
        "    for key in ('gt_video','sa_wm_video','comparison_video'):\n"
        "        video=folder/(key+'.mp4'); video.write_bytes(b'process-contract-placeholder')\n"
        "        row[key]=str(video); row[key+'_file']=video.name\n"
        "    rows.append(row)\n"
        "manifest = {key:value(flag) for key,flag in [('config','--config'),('checkpoint','--checkpoint'),('video_latent_root','--video_latent_root'),('split','--split')]}\n"
        "manifest.update(base_model=False, condition_view='arm_slot_eef_pose', condition_override_root='', samples=rows)\n"
        "manifest.update({key:int(value(flag)) for key,flag in [('seed','--seed'),('inference_steps','--inference_steps'),('max_chunks','--max_chunks')]})\n"
        "(out/'manifest.json').write_text(json.dumps(manifest))\n"
    )
    dependency = source / "sa_wm" / "world_model.py"
    dependency.parent.mkdir()
    dependency.write_text("# native model dependency\n")
    model = tmp_path / "model"
    model.mkdir()
    (model / "dit_model.safetensors").write_bytes(b"fake-weights-for-process-contract-test")
    backbone = tmp_path / "backbone"
    backbone.mkdir()
    (backbone / "config.json").write_text('{}')
    (backbone / "vae.pth").write_bytes(b"vae-process-contract-placeholder")
    data = tmp_path / "data"
    data.mkdir()
    (data / "index.json").write_text('{}')
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"world_model": {"backbone": "wan2.2", "action_schema": "arm_slot", "action_dim": 10, "max_arm_slots": 2,
        "checkpoint": str(backbone), "vae_checkpoint": str(backbone / "vae.pth")},
        "wm_training": {"video_latent_condition_view": "arm_slot_eef_pose", "action_condition_timestep_rate": "raw", "video_latent_required_target_fps": 5.0}}))
    selection = tmp_path / "selection.json"
    selection.write_text('{"selected":[{"episode_id":109,"frame_now":40}]}')
    snapshot = {"gpus": [{"index": 2, "uuid": "test-gpu", "name": "CI", "memory_used_mib": 0, "utilization_gpu_percent": 0}], "compute_apps": []}
    monkeypatch.setattr("wmloop.execute.wan_arm_slot.GpuLeaseManager",
        lambda: GpuLeaseManager(lock_root=tmp_path / "locks", snapshot_provider=lambda: snapshot))
    return dict(source_root=source, checkpoint=model, data_root=data, config=config, selection=selection,
        runtime_python=Path(sys.executable), output_root=tmp_path / "result", budget_db=tmp_path / "budget.db",
        budget_gpu_hours=0.01, gpu_indices=[2], seed=123, timeout_seconds=3.0)


def test_bounded_native_command_has_durable_cas_and_settled_cost(tmp_path, monkeypatch):
    request = _request(tmp_path, monkeypatch)
    result = run_inference(**request)
    assert result["state"] == "passed"
    observed = json.loads((request["output_root"] / "rollouts/observed.json").read_bytes())
    assert observed["gpu"] == "2"
    args = observed["args"]
    for name, value in (("--device", "cuda:0"), ("--max_chunks", "1"), ("--limit", "1"), ("--inference_steps", "25"), ("--seed", "123")):
        assert args[args.index(name) + 1] == value
    cas = ContentAddressedStore(request["output_root"] / "evidence")
    assert cas.read_bytes(result["receipt_ref"]) == (request["output_root"] / "receipt.json").read_bytes()
    for relative, ref in result["artifacts"].items():
        assert cas.read_bytes(ref) == (request["output_root"] / relative).read_bytes()
    ledger = BudgetLedger(request["budget_db"], BudgetPolicy(total_gpu_hours=0.01))
    assert ledger.get(result["trial_id"]).state == "settled"
    assert result["authority"] == {"diagnostic_only": True, "promotion": False}
    assert result["validation"] == {"input_unchanged": True, "errors": []}
    with pytest.raises(WanArmSlotError, match="OUTPUT_EXISTS"):
        run_inference(**request)


@pytest.mark.parametrize("change", ["dimensions", "unbounded", "output_in_source", "symlink_parent"])
def test_invalid_or_incompatible_request_cannot_acquire_gpu(tmp_path, monkeypatch, change):
    request = _request(tmp_path, monkeypatch)
    if change == "dimensions":
        config = json.loads(request["config"].read_bytes())
        config["world_model"]["action_dim"] = 7
        request["config"].write_text(json.dumps(config))
    elif change == "unbounded":
        request["max_chunks"] = 0
    elif change == "output_in_source":
        request["output_root"] = request["source_root"] / "run"
    else:
        link = tmp_path / "linked-parent"
        link.symlink_to(request["source_root"], target_is_directory=True)
        request["source_root"] = link
    monkeypatch.setattr("wmloop.execute.wan_arm_slot.GpuLeaseManager", lambda: pytest.fail("invalid request acquired a GPU"))
    with pytest.raises(WanArmSlotError):
        run_inference(**request)
    assert not request["budget_db"].exists()


def test_timeout_retains_failed_evidence_and_debits_shared_budget(tmp_path, monkeypatch):
    request = _request(tmp_path, monkeypatch)
    script = request["source_root"] / "scripts/evaluation/rollout_full_episode_sa_wm.py"
    script.write_text("import time\ntime.sleep(30)\n")
    request["timeout_seconds"] = 0.1
    request["budget_gpu_hours"] = 0.00004
    result = run_inference(**request)
    assert result["state"] == "failed"
    assert result["process"]["error"] == "TIMEOUT"
    request["output_root"] = tmp_path / "second-attempt"
    with pytest.raises(BudgetError, match="GLOBAL_BUDGET_EXHAUSTED"):
        run_inference(**request)
    assert not request["output_root"].exists()


def test_missing_gpu_releases_unused_reservation(tmp_path, monkeypatch):
    request = _request(tmp_path, monkeypatch)
    request["gpu_indices"] = [3]
    with pytest.raises(RuntimeError, match="GPU_LEASE_UNAVAILABLE"):
        run_inference(**request)
    ledger = BudgetLedger(request["budget_db"], BudgetPolicy(total_gpu_hours=0.01))
    assert ledger.visible_settled_trial_ids() == ()
    request["gpu_indices"] = [2]
    assert run_inference(**request)["state"] == "passed"


def test_native_yaml_accepts_indentless_sequences(tmp_path, monkeypatch):
    request = _request(tmp_path, monkeypatch)
    config = json.loads(request["config"].read_text())
    config["wm_training"]["dataset_roots"] = [str(request["data_root"])]
    request["config"].write_text(yaml.safe_dump(config))
    assert run_inference(**request)["state"] == "passed"


@pytest.mark.parametrize("asset", ["config.json", "vae.pth"])
def test_backbone_and_vae_changes_invalidate_input_binding(tmp_path, monkeypatch, asset):
    request = _request(tmp_path, monkeypatch)
    script = request["source_root"] / "scripts/evaluation/rollout_full_episode_sa_wm.py"
    target = tmp_path / "backbone" / asset
    with script.open("a") as handle:
        handle.write(f"pathlib.Path({str(target)!r}).write_bytes(b'changed-asset')\n")
    result = run_inference(**request)
    assert result["process"]["state"] == "passed"
    assert result["validation"] == {"input_unchanged": False, "errors": ["WAN_ARM_SLOT_INPUT_CHANGED"]}
    assert result["state"] == "failed"
    assert str(target) in result["input_binding"]["model_asset_sha256"]


@pytest.mark.parametrize("fault", ["missing", "symlink", "output_overlap"])
def test_invalid_configured_assets_fail_before_budget_or_gpu(tmp_path, monkeypatch, fault):
    request = _request(tmp_path, monkeypatch)
    asset = tmp_path / "backbone/vae.pth"
    if fault == "missing":
        asset.unlink()
    elif fault == "symlink":
        asset.unlink()
        asset.symlink_to(request["config"])
    else:
        request["output_root"] = tmp_path / "backbone/results"
    monkeypatch.setattr("wmloop.execute.wan_arm_slot.GpuLeaseManager",
                        lambda: pytest.fail("invalid asset request acquired a GPU"))
    with pytest.raises(WanArmSlotError):
        run_inference(**request)
    assert not request["budget_db"].exists()


@pytest.mark.parametrize("fault", ["empty_success", "missing_video", "wrong_seed", "input_mutation"])
def test_successful_process_cannot_hide_missing_output_or_input_drift(tmp_path, monkeypatch, fault):
    request = _request(tmp_path, monkeypatch)
    script = request["source_root"] / "scripts/evaluation/rollout_full_episode_sa_wm.py"
    if fault == "empty_success":
        script.write_text("pass\n")
    elif fault == "missing_video":
        with script.open('a') as handle:
            handle.write("pathlib.Path(rows[0]['gt_video']).unlink()\n")
    elif fault == "wrong_seed":
        with script.open('a') as handle:
            handle.write("manifest['seed']+=1\n(out/'manifest.json').write_text(json.dumps(manifest))\n")
    else:
        with script.open('a') as handle:
            handle.write("pathlib.Path(value('--config')).write_text('{}')\n")
    result = run_inference(**request)
    assert result["process"]["state"] == "passed"
    assert result["state"] == "failed"
    assert result["validation"]["errors"]
    if fault == "input_mutation":
        assert result["validation"] == {"input_unchanged": False, "errors": ["WAN_ARM_SLOT_INPUT_CHANGED"]}
    ledger = BudgetLedger(request["budget_db"], BudgetPolicy(total_gpu_hours=0.01))
    assert ledger.get(result["trial_id"]).state == "settled"


@pytest.mark.parametrize("relative", ["world_model.py", "dataset.py", "trajectory/loader.py"])
def test_dependency_change_invalidates_source_binding_without_entrypoint_change(tmp_path, monkeypatch, relative):
    request = _request(tmp_path, monkeypatch)
    dependency = request["source_root"] / "sa_wm" / relative
    dependency.parent.mkdir(parents=True, exist_ok=True)
    dependency.write_text("# initial dependency\n")
    script = request["source_root"] / "scripts/evaluation/rollout_full_episode_sa_wm.py"
    with script.open("a") as handle:
        handle.write(f"pathlib.Path({('sa_wm/' + relative)!r}).write_text('# modified dependency\\n')\n")
    result = run_inference(**request)
    assert result["process"]["state"] == "passed"
    assert result["state"] == "failed"
    assert result["validation"] == {"input_unchanged": False, "errors": ["WAN_ARM_SLOT_INPUT_CHANGED"]}
    assert result["input_binding"]["source_tree_revision"]["kind"] == "source_tree_sha256"


def test_unbound_source_tree_is_rejected_before_budget_or_gpu(tmp_path, monkeypatch):
    request = _request(tmp_path, monkeypatch)
    monkeypatch.setattr("wmloop.execute.wan_arm_slot.compute_source_tree_revision",
                        lambda source: {"state": "unbound"})
    monkeypatch.setattr("wmloop.execute.wan_arm_slot.GpuLeaseManager",
                        lambda: pytest.fail("unbound source acquired a GPU"))
    with pytest.raises(WanArmSlotError, match="SOURCE_REVISION_UNBOUND"):
        run_inference(**request)
    assert not request["budget_db"].exists()
    assert not request["output_root"].exists()


def test_failure_before_model_launch_cleans_own_output_and_releases_budget(tmp_path, monkeypatch):
    import wmloop.execute.wan_arm_slot as module

    request = _request(tmp_path, monkeypatch)
    write = module.atomic_write
    def fail_write(*args, **kwargs):
        raise OSError("write failed")
    monkeypatch.setattr(module, "atomic_write", fail_write)
    with pytest.raises(OSError, match="write failed"):
        run_inference(**request)
    assert not request["output_root"].exists()
    ledger = BudgetLedger(request["budget_db"], BudgetPolicy(total_gpu_hours=0.01))
    assert ledger.visible_settled_trial_ids() == ()
    monkeypatch.setattr(module, "atomic_write", write)
    assert run_inference(**request)["state"] == "passed"


@pytest.mark.parametrize("field", ["selection_sample_index", "episode_id", "frame_now", "fps"])
def test_manifest_booleans_cannot_masquerade_as_numeric_sample_bindings(tmp_path, monkeypatch, field):
    request = _request(tmp_path, monkeypatch)
    request["selection"].write_text('{"selected":[{"episode_id":1,"frame_now":1}]}')
    config = json.loads(request["config"].read_text())
    config["wm_training"]["video_latent_required_target_fps"] = 1
    request["config"].write_text(json.dumps(config))
    script = request["source_root"] / "scripts/evaluation/rollout_full_episode_sa_wm.py"
    with script.open("a") as handle:
        handle.write(f"rows[0][{field!r}] = {'False' if field == 'selection_sample_index' else 'True'}\n")
        handle.write("(out/'manifest.json').write_text(json.dumps(manifest))\n")
    result = run_inference(**request)
    assert result["process"]["state"] == "passed"
    assert result["state"] == "failed"
    assert result["validation"]["errors"] == ["WAN_ARM_SLOT_SAMPLE_BINDING_MISMATCH"]


def test_receipt_schema_rejects_contradictory_success_and_missing_bindings(tmp_path, monkeypatch):
    request = _request(tmp_path, monkeypatch)
    result = run_inference(**request)
    receipt = json.loads(Path(result["receipt_path"]).read_text())
    validate_document("wan_arm_slot_inference_receipt", receipt)
    for path, value in (
        (("validation", "input_unchanged"), False),
        (("validation", "errors"), ["WAN_ARM_SLOT_INPUT_CHANGED"]),
        (("process", "state"), "failed"),
        (("process", "returncode"), 1),
        (("gpu", "index"), True),
        (("input_binding", "source_tree_revision"), {"state": "unbound"}),
        (("input_binding", "undeclared"), "not a contract field"),
    ):
        modified = deepcopy(receipt)
        modified[path[0]][path[1]] = value
        with pytest.raises(ContractValidationError):
            validate_document("wan_arm_slot_inference_receipt", modified)
