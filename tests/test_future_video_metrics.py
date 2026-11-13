"""The public frozen CPU recipe has stable identity and fails before GPU work."""

from __future__ import annotations

from copy import deepcopy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import imageio.v2 as imageio
import numpy as np
import pytest

from wmloop.contracts import ContractValidationError, validate_document
from wmloop.evaluate import future_video_metrics as video_metrics
from wmloop.evaluate.future_video_metrics import (
    FutureVideoMetricError, freeze_future_video_protocol, future_frame_metrics,
    measure_frozen_future_video, recompute_future_video_metrics,
)
from wmloop.storage import canonical_bytes, sha256_bytes
from wmloop.archive.store import ArchiveStore, ContentAddressedStore, SettledTrialRecord
from wmloop.diagnose.paired_observation import (
    PairedObservationError, build_wan_history_observations, verify_paired_observation,
)
from wmloop.diagnose.wan_native_evidence import wan_invocation_matches
from wmloop.execute import auto_experiment
from wmloop.execute.experiment_scheduler import plan_candidate_batch, run_selected_queue
from wmloop.execute.gpu_lease import GpuLeaseManager
from wmloop.execute.process_lifecycle import LocalProcessCancelled
from wmloop.execute.wan_arm_slot import build_native_execution_evidence, run_inference
from wmloop.retrieve.index import ProbeRetrievalError, index_probe_experience, retrieve_probe_experiences
from wmloop.retrieve.paired_prior import apply_paired_observation_priors


def declared_protocol() -> dict:
    return {
        "evaluator_id": "future-screen-test-v1", "formal_acceptance": False,
        "metric_tie_epsilon": 1e-12, "ranking_primary_metric": "psnr_db",
        "comparison": {
            "primitive": "history_sparse_dilation", "baseline_history_rollout_dilation": 2,
            "allowed_candidate_history_rollout_dilations": [1, 3],
            "config_fields": ["world_model.history_rollout_dilation", "wm_training.history_rollout_dilation"],
            "episode_id": 109, "frame_now": 40, "seed": 123, "split": "val",
            "inference_steps": 25, "max_chunks": 1,
            "identical_ground_truth_bytes_required": True, "other_config_semantics_must_match": True,
            "weights_fixed": True, "data_fixed": True,
        },
    }


@pytest.fixture(scope="module")
def encoded_videos(tmp_path_factory) -> dict[str, bytes]:
    root = tmp_path_factory.mktemp("future-recipe-videos")
    clips = {}
    for label, value in (("ground_truth", 96), ("control", 100), ("candidate", 110)):
        path = root / f"{label}.mp4"
        with imageio.get_writer(path, fps=5, codec="libx264", pixelformat="yuv444p", ffmpeg_params=["-crf", "0"]) as writer:
            for index in range(13):
                writer.append_data(np.full((512, 768, 3), value + index, dtype=np.uint8))
        clips[label] = path.read_bytes()
    return clips


def test_numeric_recipe_excludes_anchor_and_aggregates_mse_before_psnr():
    target = np.zeros((13, 512, 768, 3), dtype=np.uint8)
    prediction = target.copy()
    prediction[0] = 255
    prediction[1:7] = 1
    prediction[7:] = 3
    output = future_frame_metrics(prediction, target)
    assert output["metrics"]["aggregate_mse"] == 5.0
    assert output["metrics"]["psnr_db"] == 10 * math.log10(255 ** 2 / 5)
    assert [row["video_frame_index"] for row in output["per_frame"]] == list(range(1, 13))
    identical = future_frame_metrics(target, target)
    assert identical["metrics"]["psnr_db"] is None
    assert identical["metrics"]["psnr_is_infinite"] is True
    assert identical["metrics"]["global_ssim_rgb"] == 1.0


@pytest.mark.parametrize("shape,dtype", [((12, 512, 768, 3), np.uint8), ((13, 8, 8, 3), np.uint8), ((13, 512, 768, 3), np.float32)])
def test_numeric_recipe_rejects_shape_and_dtype(shape, dtype):
    with pytest.raises(FutureVideoMetricError, match="FRAME_CONTRACT_INVALID"):
        future_frame_metrics(np.zeros(shape, dtype=dtype), np.zeros(shape, dtype=dtype))


def test_public_freeze_measure_real_video_and_no_overwrite(tmp_path, encoded_videos):
    frozen = freeze_future_video_protocol(protocol=declared_protocol(), output_root=tmp_path / "frozen")
    freeze = json.loads((frozen / "freeze-record.json").read_bytes())
    assert set(freeze["files"]) == {"protocol.json", "future_video_recipe.py"}
    assert video_metrics.supported_frozen_algorithm({"future_video_recipe.py": freeze["files"]["future_video_recipe.py"]}, freeze)
    prediction, target = tmp_path / "prediction.mp4", tmp_path / "target.mp4"
    prediction.write_bytes(encoded_videos["control"])
    target.write_bytes(encoded_videos["ground_truth"])
    output = tmp_path / "new-measurements" / "measurement.json"
    measurement = measure_frozen_future_video(frozen_root=frozen, prediction=prediction, ground_truth=target, output=output)
    validate_document("future_frame_measurement", measurement)
    assert json.loads(output.read_bytes()) == measurement
    assert measurement["input_videos_sha256"] == {"prediction": sha256_bytes(prediction.read_bytes()), "ground_truth": sha256_bytes(target.read_bytes())}
    # Lossless YUV encoding still rounds the RGB colour-space conversion.
    decoded_prediction = np.stack(imageio.mimread(prediction, format="ffmpeg"))
    decoded_target = np.stack(imageio.mimread(target, format="ffmpeg"))
    assert measurement["metrics"] == future_frame_metrics(decoded_prediction, decoded_target)["metrics"]
    assert measurement["diagnostic_only"] is True and measurement["promotion_allowed"] is False
    with pytest.raises(FutureVideoMetricError, match="OUTPUT_EXISTS"):
        freeze_future_video_protocol(protocol=declared_protocol(), output_root=frozen)
    with pytest.raises(FutureVideoMetricError, match="OUTPUT_EXISTS"):
        measure_frozen_future_video(frozen_root=frozen, prediction=prediction, ground_truth=target, output=output)


@pytest.mark.parametrize("fault", ["empty_comparison", "missing_config_fields", "boolean_seed", "nonfinite", "formal_authority"])
def test_protocol_failure_creates_no_frozen_output(tmp_path, fault):
    protocol = declared_protocol()
    if fault == "empty_comparison":
        protocol["comparison"] = {}
    elif fault == "missing_config_fields":
        protocol["comparison"].pop("config_fields")
    elif fault == "boolean_seed":
        protocol["comparison"]["seed"] = True
    elif fault == "formal_authority":
        protocol["formal_acceptance"] = True
    else:
        protocol["metric_tie_epsilon"] = float("nan")
    with pytest.raises(FutureVideoMetricError, match="PROTOCOL_INVALID"):
        freeze_future_video_protocol(protocol=protocol, output_root=tmp_path / "frozen")
    assert not (tmp_path / "frozen").exists()


@pytest.mark.parametrize("fault", ["missing_protocol_binding", "algorithm_drift", "malformed_protocol", "symlink"])
def test_measure_rechecks_frozen_evidence_before_decoding(tmp_path, monkeypatch, fault):
    frozen = freeze_future_video_protocol(protocol=declared_protocol(), output_root=tmp_path / "frozen")
    freeze_path = frozen / "freeze-record.json"
    freeze = json.loads(freeze_path.read_bytes())
    if fault == "missing_protocol_binding":
        freeze["files"].pop("protocol.json")
        freeze_path.write_bytes(canonical_bytes(freeze))
    elif fault == "algorithm_drift":
        (frozen / "future_video_recipe.py").write_text("changed algorithm")
    elif fault == "malformed_protocol":
        payload = b"[]"
        (frozen / "protocol.json").write_bytes(payload)
        freeze["files"]["protocol.json"] = sha256_bytes(payload)
        freeze_path.write_bytes(canonical_bytes(freeze))
    else:
        recipe = frozen / "future_video_recipe.py"
        source = tmp_path / "recipe.py"
        source.write_bytes(recipe.read_bytes())
        recipe.unlink()
        recipe.symlink_to(source)
    monkeypatch.setattr(video_metrics, "recompute_future_video_metrics", lambda **kwargs: pytest.fail("unbound freeze reached decoder"))
    with pytest.raises(FutureVideoMetricError):
        measure_frozen_future_video(frozen_root=frozen, prediction=tmp_path / "missing", ground_truth=tmp_path / "missing", output=tmp_path / "measure.json")
    assert not (tmp_path / "measure.json").exists()


def test_worker_stdout_is_exact_numeric_json(encoded_videos):
    measured = recompute_future_video_metrics(prediction=encoded_videos["candidate"], ground_truth=encoded_videos["ground_truth"])
    assert set(measured) == {"metrics", "per_frame"}
    assert measured["metrics"]["aggregate_mse"] == 196.0
    with pytest.raises(FutureVideoMetricError, match="REPLAY_FAILED"):
        recompute_future_video_metrics(prediction=b"not-a-video", ground_truth=encoded_videos["ground_truth"])


@pytest.mark.parametrize("failure", ["timeout", "cancellation"])
def test_timeout_and_cancel_retire_decoder_and_its_descendant(tmp_path, monkeypatch, failure):
    pid_path = tmp_path / "descendant.pid"
    original_popen = subprocess.Popen
    clock = time.monotonic
    worker = (
        "import pathlib, subprocess, sys, time\n"
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'])\n"
        f"pathlib.Path({str(pid_path)!r}).write_text(str(child.pid))\n"
        "time.sleep(30)\n"
    )
    processes = []

    def launch(_command, **kwargs):
        assert kwargs["start_new_session"] is True
        process = original_popen([sys.executable, "-c", worker], **kwargs)
        processes.append(process)
        deadline = clock() + 5
        while not pid_path.exists() and clock() < deadline:
            time.sleep(0.01)
        assert pid_path.exists()
        return process

    ticks = iter([0.0, 100.0])
    monkeypatch.setattr(video_metrics.subprocess, "Popen", launch)
    if failure == "timeout":
        monkeypatch.setattr(video_metrics, "time", SimpleNamespace(monotonic=lambda: next(ticks)))
        expected_error = FutureVideoMetricError
    else:
        class CancelledScope:
            def __enter__(self):
                return self

            def check(self):
                raise LocalProcessCancelled("LOCAL_PROCESS_CANCELLED")

            def __exit__(self, *_args):
                return False

        monkeypatch.setattr(video_metrics, "LocalCommandSignals", CancelledScope)
        expected_error = LocalProcessCancelled
    with pytest.raises(expected_error):
        recompute_future_video_metrics(prediction=b"a", ground_truth=b"b")
    assert processes[0].poll() is not None
    child_pid = int(pid_path.read_text())
    deadline = clock() + 2
    while Path(f"/proc/{child_pid}/stat").exists() and clock() < deadline:
        if Path(f"/proc/{child_pid}/stat").read_text().split()[2] == "Z":
            break
        time.sleep(0.01)
    assert not Path(f"/proc/{child_pid}/stat").exists() or Path(f"/proc/{child_pid}/stat").read_text().split()[2] == "Z"


@pytest.fixture(scope="module")
def paired_evidence(tmp_path_factory, encoded_videos):
    """Real MP4/protocol/process formats; fake model assets and lease, no GPU."""
    root = tmp_path_factory.mktemp("paired-evidence")
    source = root / "model-source"
    script = source / "scripts/evaluation/rollout_full_episode_sa_wm.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        "import json, pathlib, shutil, sys\n"
        "args=sys.argv[1:]\n"
        "value=lambda flag: args[args.index(flag)+1]\n"
        "out=pathlib.Path(value('--output_dir')); out.mkdir()\n"
        "config=json.loads(pathlib.Path(value('--config')).read_bytes())\n"
        "dose=config['world_model']['history_rollout_dilation']\n"
        "fixtures=pathlib.Path(__file__).parents[2]/'fixtures'\n"
        "rows=[]\n"
        "for index, selected in enumerate(json.loads(pathlib.Path(value('--selection')).read_bytes())['selected'][:int(value('--limit'))]):\n"
        "    folder=out/str(index); folder.mkdir()\n"
        "    row=dict(selected,selection_sample_index=index,dir=folder.name,chunks=1,frames=13,fps=float(value('--fps')))\n"
        "    for key,clip in [('gt_video','ground_truth'),('sa_wm_video','control' if dose==2 else 'candidate'),('comparison_video','control')]:\n"
        "        video=folder/(key+'.mp4'); shutil.copyfile(fixtures/(clip+'.mp4'),video)\n"
        "        row[key]=str(video); row[key+'_file']=video.name\n"
        "    rows.append(row)\n"
        "manifest={key:value(flag) for key,flag in [('config','--config'),('checkpoint','--checkpoint'),('video_latent_root','--video_latent_root'),('split','--split')]}\n"
        "manifest.update(base_model=False,condition_view='arm_slot_eef_pose',condition_override_root='',samples=rows)\n"
        "manifest.update({key:int(value(flag)) for key,flag in [('seed','--seed'),('inference_steps','--inference_steps'),('max_chunks','--max_chunks')]})\n"
        "(out/'manifest.json').write_text(json.dumps(manifest))\n"
    )
    (source / "fixtures").mkdir()
    for name, payload in encoded_videos.items():
        (source / "fixtures" / f"{name}.mp4").write_bytes(payload)
    model, backbone, data = root / "model", root / "backbone", root / "data"
    for directory in (model, backbone, data):
        directory.mkdir()
    (model / "dit_model.safetensors").write_bytes(b"CPU process-contract fixture; not model weights")
    (backbone / "config.json").write_text("{}")
    (backbone / "vae.pth").write_bytes(b"CPU fixture")
    (data / "index.json").write_text("{}")
    selection = root / "selection.json"
    selection.write_bytes(canonical_bytes({"selected": [{"episode_id": 109, "frame_now": 40}]}))
    freeze_future_video_protocol(protocol=declared_protocol(), output_root=root / "frozen")
    specification = {
        "schema_version": 1, "primitive": "history_sparse_dilation",
        "protocol": "frozen/protocol.json", "freeze_record": "frozen/freeze-record.json",
        "evaluator_files": {"future_video_recipe.py": "frozen/future_video_recipe.py"},
        "metrics": {"psnr_db": "higher", "global_ssim_rgb": "higher", "aggregate_mse": "lower"},
        "null_tolerance": 1e-12,
    }
    patch = pytest.MonkeyPatch()
    snapshot = {"gpus": [{"index": 2, "uuid": "CPU-contract-test", "name": "CI", "memory_used_mib": 0, "utilization_gpu_percent": 0}], "compute_apps": []}
    patch.setattr("wmloop.execute.wan_arm_slot.GpuLeaseManager", lambda: GpuLeaseManager(lock_root=root / "locks", snapshot_provider=lambda: snapshot))
    native = {}
    try:
        for side, dose in (("control", 2), ("candidate", 1)):
            side_root = root / side
            side_root.mkdir()
            config = side_root / "config.json"
            config.write_bytes(canonical_bytes({
                "world_model": {"backbone": "wan2.2", "action_schema": "arm_slot", "action_dim": 10, "max_arm_slots": 2,
                                "checkpoint": str(backbone), "vae_checkpoint": str(backbone / "vae.pth"), "history_rollout_dilation": dose},
                "wm_training": {"video_latent_condition_view": "arm_slot_eef_pose", "action_condition_timestep_rate": "raw", "video_latent_required_target_fps": 5.0, "history_rollout_dilation": dose},
            }))
            output_root = side_root / "native"
            receipt = run_inference(
                source_root=source, checkpoint=model, config=config, selection=selection, data_root=data,
                runtime_python=Path(sys.executable), output_root=output_root, budget_db=root / "budget.db",
                budget_gpu_hours=0.1, gpu_indices=[2], seed=123, timeout_seconds=3.0,
            )
            assert receipt["state"] == "passed"
            native[side] = build_native_execution_evidence(
                output_root=output_root, input_binding=receipt["input_binding"],
                selection=[{"episode_id": 109, "frame_now": 40}], process=receipt["process"],
            )
            assert native[side]["state"] == "passed"
            (side_root / "execution.json").write_bytes(canonical_bytes(native[side]))
            prediction = output_root / "rollouts/0/sa_wm_video.mp4"
            ground_truth = output_root / "rollouts/0/gt_video.mp4"
            measure_frozen_future_video(frozen_root=root / "frozen", prediction=prediction, ground_truth=ground_truth, output=side_root / "measurement.json")
            specification[side] = {
                "execution": f"{side}/execution.json", "config": f"{side}/config.json", "selection": "selection.json",
                "manifest": f"{side}/native/rollouts/manifest.json", "measurement": f"{side}/measurement.json",
                "prediction": prediction.relative_to(root).as_posix(), "ground_truth": ground_truth.relative_to(root).as_posix(),
            }
    finally:
        patch.undo()
    observations = build_wan_history_observations(evidence_root=root, specification=specification)
    payloads = {name: (root / name).read_bytes() for name in observations[0]["evidence"]["files"]}
    return {"root": root, "specification": specification, "observations": observations, "payloads": payloads, "native": native}


def _paired_result(pairs: list) -> dict:
    return {
        "schema_version": 1, "artifact_type": "verdiwm-diagnostic-probe-result", "state": "ready",
        "probe_id": "history-pair", "model_family": "source", "runtime_capability": "video",
        "failure_signatures": ["history", "temporal"], "primitive": "history_sparse_dilation",
        "metrics": {"psnr_db": pairs[0]["candidate_value"]}, "paired_observations": pairs,
        "claim_boundary": "CPU fixture of ranking-only diagnostic memory, no scientific claim.",
    }


def _archive_pair(root: Path, paired_evidence: dict, *, trial="trial", verdict="PASS", observations=None) -> dict:
    cas = ContentAddressedStore(root / "cas")
    archive_db = root / "archive.db"
    pairs = deepcopy(paired_evidence["observations"] if observations is None else observations)
    result = _paired_result(pairs)
    result_ref = cas.put_bytes(canonical_bytes(result), media_type="application/json")
    artifacts = {name: cas.put_bytes(payload, media_type="application/octet-stream").uri for name, payload in paired_evidence["payloads"].items()}
    artifacts["result.json"] = result_ref.uri
    receipt = {"archive_trial_id": trial, "settlement_state": "settled", "stage": "screen", "verdict": {"verdict": verdict}, "artifact_refs": artifacts}
    receipt_ref = cas.put_bytes(canonical_bytes(receipt), media_type="application/json")
    receipt.update(receipt_ref=receipt_ref.uri, receipt_hash=receipt_ref.sha256)
    ArchiveStore(archive_db).record_settled_trial(SettledTrialRecord(
        trial_id=trial, proposal_id=f"proposal-{trial}", goal_id="goal", library_version="v1",
        failure_context_ref=result_ref.uri, verdict_ref=result_ref.uri, receipt_ref=receipt_ref.uri,
        gpu_hours=0.0, hypothesis_hash="a" * 64, impl_diff_hash="b" * 64,
        evaluator_hash="c" * 64, settlement_state="settled", receipt_hash=receipt_ref.sha256,
    ))
    arguments = {"database_path": root / "index.db", "result": result, "receipt": receipt, "archive_db": archive_db, "cas_root": root / "cas"}
    index_probe_experience(**arguments)
    return arguments


def _matches(stores):
    return retrieve_probe_experiences(
        **{key: stores[key] for key in ("database_path", "archive_db", "cas_root")},
        model_family="source", runtime_capability="video", failure_signatures=("history", "temporal"),
    )


def _batch(query):
    root = Path(__file__).parents[1]
    batch = json.loads((root / "configs/smoke/auto_experiment_candidate_batch_cuda_v1.json").read_bytes())
    batch["scoring"]["retrieval_weight"] = 1.0
    for index, candidate in enumerate(batch["candidates"]):
        candidate.update(candidate_id=f"dose-{1 if index == 0 else 3}", expected_gain=0.6 - index * 0.01,
                         uncertainty=0.2, information_gain=0.2, novelty=0.2)
        candidate["retrieval_keys"] = {"primitive": "history_sparse_dilation", "failure_signatures": ["history", "temporal"], "stage": "screen", "paired_query": deepcopy(query)}
        candidate["retrieval_keys"]["paired_query"]["candidate_parameters"]["history_rollout_dilation"] = 1 if index == 0 else 3
        candidate["stages"] = candidate["stages"][:1]
    return batch


def test_public_producer_cas_index_prior_changes_public_queue(tmp_path, paired_evidence):
    assert {row["outcome"] for row in paired_evidence["observations"]} == {"negative"}
    stores = _archive_pair(tmp_path, paired_evidence)
    matches = _matches(stores)
    assert len(matches) == 2  # Two signatures, one archived trial.
    assert len(matches[0].paired_observations) == 3
    batch = _batch(paired_evidence["observations"][0]["query"])
    cold = apply_paired_observation_priors(batch, [])
    reused = apply_paired_observation_priors(batch, [row.to_dict() for row in matches], archive_db=stores["archive_db"], cas_root=stores["cas_root"])
    assert "paired_observation_prior" not in batch["candidates"][0]
    assert reused["candidates"][0]["paired_observation_prior"] == -0.5
    assert reused["candidates"][0]["paired_observation_retrieval"]["outcomes"] == {"positive": 0, "negative": 1, "null": 0}
    assert reused["candidates"][1]["paired_observation_prior"] == 0.0
    for label, document in (("cold", cold), ("reused", reused)):
        source = tmp_path / f"{label}.json"
        source.write_bytes(canonical_bytes(document))
        queue = plan_candidate_batch(batch_path=source, output_root=tmp_path / label, workspace_root=Path(__file__).parents[1])
        assert queue["selected"][0]["candidate_id"] == ("dose-1" if label == "cold" else "dose-3")


@pytest.mark.parametrize("suffix", [
    ["world_model.history_rollout_dilation=3", "wm_training.history_rollout_dilation=3"],
    ["--confi", "different.yaml"], ["--seed=999"], ["--seed", "999"], ["unexpected"],
])
def test_hidden_native_overrides_cannot_relabel_dose(paired_evidence, suffix):
    native = deepcopy(paired_evidence["native"]["candidate"])
    assert wan_invocation_matches(native["process"], native["input_binding"])
    native["process"]["command"].extend(suffix)
    assert not wan_invocation_matches(native["process"], native["input_binding"])


@pytest.mark.parametrize("fault", ["measurement", "manifest_context", "invocation_override", "config_dose", "extra_config_semantics", "ground_truth", "numeric_recipe"])
def test_paired_proof_rejects_hash_consistent_but_invalid_evidence(paired_evidence, fault):
    observation = deepcopy(paired_evidence["observations"][0])
    payloads = dict(paired_evidence["payloads"])
    spec = observation["evidence"]["specification"]
    if fault == "measurement":
        path = spec["candidate"]["measurement"]
        document = json.loads(payloads[path]); document["metrics"]["psnr_db"] += 0.5
    elif fault == "manifest_context":
        path = spec["candidate"]["manifest"]
        document = json.loads(payloads[path]); document["samples"][0]["frame_now"] += 1
    elif fault == "invocation_override":
        path = spec["candidate"]["execution"]
        document = json.loads(payloads[path]); document["process"]["command"].extend(["world_model.history_rollout_dilation=3", "wm_training.history_rollout_dilation=3"])
    elif fault in {"config_dose", "extra_config_semantics"}:
        path = spec["candidate"]["config"]
        document = json.loads(payloads[path])
        if fault == "config_dose":
            document["world_model"]["history_rollout_dilation"] = 3
        else:
            document["world_model"]["extra_setting"] = True
    elif fault == "numeric_recipe":
        path = spec["evaluator_files"]["future_video_recipe.py"]
        payloads[path] += b"\n# unrecognized algorithm revision\n"
        document = None
    else:
        path = spec["candidate"]["ground_truth"]
        payloads[path] = b"substituted video"
        document = None
    if document is not None:
        payloads[path] = canonical_bytes(document)
    observation["evidence"]["files"][path] = sha256_bytes(payloads[path])
    with pytest.raises((PairedObservationError, ContractValidationError)):
        verify_paired_observation(observation, read_artifact=payloads.__getitem__)


def test_duplicate_query_with_distinct_valid_proofs_is_rejected(tmp_path, paired_evidence):
    first = deepcopy(paired_evidence["observations"][0])
    # Both proofs were independently reconstructed by the public producer from
    # actual video bytes. They have the same query but opposite signed values.
    positive = _signed_variant(tmp_path / "positive-proof", paired_evidence, "positive")
    second = positive["observations"][0]
    assert first["query"] == second["query"]
    assert first["evidence"] != second["evidence"]
    result = {
        "schema_version": 1, "artifact_type": "verdiwm-diagnostic-probe-result", "state": "ready",
        "probe_id": "history", "model_family": "source", "runtime_capability": "video",
        "failure_signatures": ["history"], "metrics": {"score": 0.0},
        "paired_observations": [first, second], "claim_boundary": "Ranking only.",
    }
    with pytest.raises(ContractValidationError, match="PAIRED_QUERY_DUPLICATE"):
        validate_document("diagnostic_probe_result", result)


def test_missing_video_removes_archived_pair_and_abstains(tmp_path, paired_evidence):
    stores = _archive_pair(tmp_path, paired_evidence)
    matches = _matches(stores)
    candidate_path = paired_evidence["specification"]["candidate"]["prediction"]
    cas = ContentAddressedStore(stores["cas_root"])
    ref = cas.put_bytes(paired_evidence["payloads"][candidate_path], media_type="video/mp4")
    ref.path.unlink()
    assert _matches(stores) == ()
    batch = _batch(paired_evidence["observations"][0]["query"])
    reused = apply_paired_observation_priors(batch, [row.to_dict() for row in matches], archive_db=stores["archive_db"], cas_root=stores["cas_root"])
    assert reused["candidates"][0]["paired_observation_prior"] == 0.0
    assert reused["candidates"][0]["paired_observation_retrieval"]["state"] == "abstain"


@pytest.mark.parametrize("field,value", [("seed", 999), ("metric_name", "aggregate_mse"), ("protocol_sha256", "a" * 64), ("context_sha256", "b" * 64)])
def test_incompatible_queries_cannot_reuse_signed_outcome(tmp_path, paired_evidence, field, value):
    stores = _archive_pair(tmp_path, paired_evidence)
    matches = _matches(stores)
    query = deepcopy(paired_evidence["observations"][0]["query"])
    query[field] = value
    batch = _batch(query)
    reused = apply_paired_observation_priors(batch, [row.to_dict() for row in matches], archive_db=stores["archive_db"], cas_root=stores["cas_root"])
    assert reused["candidates"][0]["paired_observation_prior"] == 0.0


def test_void_execution_does_not_add_effect_prior(tmp_path, paired_evidence):
    stores = _archive_pair(tmp_path, paired_evidence, verdict="VOID")
    matches = _matches(stores)
    batch = _batch(paired_evidence["observations"][0]["query"])
    reused = apply_paired_observation_priors(batch, [row.to_dict() for row in matches], archive_db=stores["archive_db"], cas_root=stores["cas_root"])
    assert reused["candidates"][0]["paired_observation_prior"] == 0.0


def _signed_variant(root: Path, paired_evidence: dict, outcome: str) -> dict:
    """Build another valid proof by exchanging actual encoded predictions."""
    payloads = dict(paired_evidence["payloads"])
    spec = paired_evidence["specification"]
    control_prediction = payloads[spec["control"]["prediction"]]
    candidate_prediction = payloads[spec["candidate"]["prediction"]]
    for side in ("control", "candidate"):
        paths = spec[side]
        prediction = candidate_prediction if side == "control" and outcome == "positive" else control_prediction
        payloads[paths["prediction"]] = prediction
        source = json.loads(payloads[paths["execution"]])
        manifest = json.loads(payloads[paths["manifest"]])
        sample = manifest["samples"][0]
        source["artifacts"]["rollouts/" + sample["dir"] + "/" + sample["sa_wm_video_file"]] = "cas://sha256/" + sha256_bytes(prediction)
        payloads[paths["execution"]] = canonical_bytes(source)
        measurement = json.loads(payloads[paths["measurement"]])
        measurement["input_videos_sha256"]["prediction"] = sha256_bytes(prediction)
        measurement.update(recompute_future_video_metrics(prediction=prediction, ground_truth=payloads[paths["ground_truth"]]))
        payloads[paths["measurement"]] = canonical_bytes(measurement)
    for path, payload in payloads.items():
        destination = root / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
    observations = build_wan_history_observations(evidence_root=root, specification=spec)
    assert {row["outcome"] for row in observations} == {outcome}
    return {"observations": observations, "payloads": payloads}


@pytest.mark.parametrize("outcome,expected_candidate", [
    ("negative", "dose-3"), ("positive", "dose-1"), ("null", "dose-1"),
])
def test_public_scheduler_accumulates_signed_memory_for_next_round(
    tmp_path, monkeypatch, paired_evidence, outcome, expected_candidate,
):
    """Exercise public planning/execution/indexing with CPU-generated MP4 proofs."""
    evidence = (paired_evidence if outcome == "negative" else
                _signed_variant(tmp_path / "proof", paired_evidence, outcome))
    calls = []

    def execute(*, scratch, **kwargs):
        calls.append(kwargs["fencing_token"])
        samples = {"samples": [
            {"status": "ready", "gpu_uuid": "CPU-fixture", "phase": "before",
             "memory_used_mib": 0.0, "utilization_gpu_percent": 0.0},
            {"status": "ready", "gpu_uuid": "CPU-fixture", "phase": "during",
             "memory_used_mib": 64.0, "utilization_gpu_percent": 10.0},
        ]}
        payloads = dict(evidence["payloads"])
        payloads["diagnostic.json"] = canonical_bytes(_paired_result(evidence["observations"]))
        payloads["result.json"] = canonical_bytes({
            "schema_version": 1, "artifact_type": "verdiwm-auto-experiment-result",
            "state": "ready", "device": {"type": "cuda", "gpu_uuid": "CPU-fixture"},
            "metrics": {"finite_fraction": 1.0, "max_abs_error": 0.0},
            "claim_boundary": "CPU scheduler fixture with synthetic GPU telemetry.",
        })
        payloads.update({"stdout.log": b"CPU fixture\n", "stderr.log": b"",
                         "gpu-sampling.json": canonical_bytes(samples)})
        for name, payload in payloads.items():
            path = scratch / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        return {"lease": {"uuid": "CPU-fixture"}, "gpu_sampling": samples,
                "timed_out": False, "exit_code": 0, "duration_seconds": 1.0}

    monkeypatch.setattr(auto_experiment, "_execute_trial", execute)
    monkeypatch.setattr(auto_experiment, "_git_state", lambda _: {
        "revision": "cpu-fixed", "implementation_hash": "a" * 64})
    batch = _batch(evidence["observations"][0]["query"])
    for candidate in batch["candidates"]:
        candidate["experiment_kind"] = "diagnostic"
        stage = candidate["stages"][0]
        stage["diagnostic_result_path"] = "diagnostic.json"
        stage["artifacts"] += ["diagnostic.json", *evidence["payloads"]]
    batch_path = tmp_path / "batch.json"
    batch_path.write_bytes(canonical_bytes(batch))
    workspace = Path(__file__).parents[1]
    cold = plan_candidate_batch(batch_path=batch_path, output_root=tmp_path / "round1", workspace_root=workspace)
    assert cold["selected"][0]["candidate_id"] == "dose-1"
    stores = {"archive_db": tmp_path / "archive.db", "cas_root": tmp_path / "cas"}
    arguments = dict(queue_path=tmp_path / "round1/queue.json", workspace_root=workspace,
                     lock_root=tmp_path / "locks", retrieval_db=tmp_path / "index.db", **stores)
    assert run_selected_queue(**arguments)["candidate_states"] == {"dose-1": "completed"}
    matches = _matches({"database_path": arguments["retrieval_db"], **stores})
    assert len(matches) == 2
    assert {pair["outcome"] for row in matches for pair in row.paired_observations} == {outcome}
    assert all(row.verdict == "PASS" for row in matches)
    assert run_selected_queue(**arguments)["candidate_states"] == {"dose-1": "completed"}
    assert len(_matches({"database_path": arguments["retrieval_db"], **stores})) == 2
    assert calls == [1]
    reused = apply_paired_observation_priors(
        batch, [row.to_dict() for row in matches], **stores,
    )
    batch_path.write_bytes(canonical_bytes(reused))
    hot = plan_candidate_batch(batch_path=batch_path, output_root=tmp_path / "round2", workspace_root=workspace)
    assert hot["selected"][0]["candidate_id"] == expected_candidate


@pytest.mark.parametrize("outcome,prior", [("positive", 0.25), ("null", 0.0)])
def test_positive_and_null_are_retained_and_reused_from_cas(tmp_path, paired_evidence, outcome, prior):
    variant = _signed_variant(tmp_path / "pair", paired_evidence, outcome)
    assert variant["observations"][0]["query"] == paired_evidence["observations"][0]["query"]
    stores = _archive_pair(tmp_path / "stores", variant)
    matches = _matches(stores)
    assert {pair["outcome"] for row in matches for pair in row.paired_observations} == {outcome}
    batch = _batch(variant["observations"][0]["query"])
    reused = apply_paired_observation_priors(batch, [row.to_dict() for row in matches] * 2, archive_db=stores["archive_db"], cas_root=stores["cas_root"])
    candidate = reused["candidates"][0]
    assert candidate["paired_observation_prior"] == prior
    assert candidate["paired_observation_retrieval"]["state"] == "applied"
    assert candidate["paired_observation_retrieval"]["outcomes"][outcome] == 1
    assert len(candidate["retrieval_matches"]) == 1


def test_opposing_independent_trials_abstain_without_losing_either_outcome(tmp_path, paired_evidence):
    variant = _signed_variant(tmp_path / "positive-pair", paired_evidence, "positive")
    stores = _archive_pair(tmp_path / "stores", paired_evidence, trial="negative")
    _archive_pair(tmp_path / "stores", variant, trial="positive")
    matches = _matches(stores)
    assert len(matches) == 4
    batch = _batch(variant["observations"][0]["query"])
    reused = apply_paired_observation_priors(batch, [row.to_dict() for row in matches], archive_db=stores["archive_db"], cas_root=stores["cas_root"])
    candidate = reused["candidates"][0]
    assert candidate["paired_observation_prior"] == 0.0
    assert candidate["paired_observation_retrieval"] == {
        "state": "abstain", "reason": "PAIRED_OPPOSING_SIGNS", "matched_trial_ids": ["negative", "positive"],
        "outcomes": {"positive": 1, "negative": 1, "null": 0},
    }
    assert {row["outcome"] for row in candidate["retrieval_matches"]} == {"negative", "positive"}


def test_capsule_keeps_negative_null_positive_and_copies_query(paired_evidence):
    from wmloop.retrieve.evidence_capsule import build_evidence_capsule

    rows = []
    for index, outcome in enumerate(["positive"] * 5 + ["negative", "null"]):
        observation = deepcopy(paired_evidence["observations"][0])
        observation["outcome"] = outcome
        row = {
            "failure_signature": "history", "verdict": "PASS", "primitive": "history_sparse_dilation",
            "metric_outcome": 100.0 - index, "archive_trial_id": f"trial-{index}",
            "receipt_ref": f"cas://receipt/{index}", "result_ref": f"cas://result/{index}", "receipt_hash": "a" * 64,
            "paired_observations": [observation],
        }
        rows.extend([row, dict(row, failure_signature="temporal")])
    capsule = build_evidence_capsule(probe={"failure_signatures": ["history"]}, matches=rows, max_evidence=3)
    selected = capsule["selected_evidence"]
    assert {pair["outcome"] for row in selected for pair in row["paired_observations"]} == {"positive", "negative", "null"}
    assert len({row["evidence"]["archive_trial_id"] for row in selected}) == 3
    original_seed = rows[0]["paired_observations"][0]["query"]["seed"]
    for row in selected:
        row["paired_observations"][0]["query"]["seed"] = 999
    assert all(row["paired_observations"][0]["query"]["seed"] == original_seed for row in rows)
