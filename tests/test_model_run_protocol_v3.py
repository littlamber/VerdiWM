"""CPU integration coverage for the Agibot engineering-diagnostic protocol."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from wmloop.control.model_run import (
    ModelRunError,
    compile_model_run,
    model_run_digest,
    validate_model_run,
)
from wmloop.control.agibot_native_config import (
    AgibotNativeConfigError,
    bind_agibot_native_config,
    validate_agibot_native_config,
)
from wmloop.execute.agibot_support import build_agibot_training_support
from wmloop.execute import agibot_trace_runtime as trace_runtime
from wmloop.execute.gpu_lease import GpuLeaseManager
from wmloop.execute.budget import BudgetPolicy, BudgetLedger
from wmloop.execute.model_runner import ModelRunnerError, execute_model_run
from wmloop.execute._agibot_runtime_isolation import agibot_runtime_context
from wmloop.execute.model_run_protocol_v3 import _validate_agibot_training_progress
from wmloop.execute.model_run_budget import model_run_budget_plan_sha256
from wmloop.experiments.training_scale import build_training_scale_plan


_NATIVE_FILES = (
    "scripts/training/train_world_model.py",
    "sa_wm/world_model/trainer.py",
    "sa_wm/data/video_latent_dataset.py",
    "sa_wm/data/video_latent_reader.py",
    "sa_wm/data/video_latent.py",
    "sa_wm/world_model/action_contract.py",
)
_LOADER = {
    "num_history": 2,
    "num_frames": 2,
    "sample_stride_video_frames": 4,
    "history_selector": "recent",
    "history_offsets": [],
    "history_dilations": [1, 2],
    "history_collapse_prob": 0.15,
    "history_eval_dilation": 2,
    "generated_history_enabled": False,
    "condition_view": "arm_slot_eef_pose",
    "body_conditioning": True,
}
_WORKER = r'''
import hashlib, json, os, sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
stage = sys.argv[2]
config = json.loads(Path(__file__).with_name("protocol-config.json").read_text())
output = Path(manifest["output"]["root"])

if config.get("fail_stage") == stage:
    raise SystemExit(9)

def write(name, value):
    path = output / name
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True))
    temporary.replace(path)

training = manifest["training"]
bindings = training["agibot_receipt_bindings"]
receipt_bindings = {
    key: value for key, value in bindings.items()
    if key != "validation_sample_window_ids"
}

if stage == "train":
    plan = training["scale_plan"]
    planned_updates = plan["updates"]["planned_steps"]
    planned_microsteps = (
        planned_updates * plan["parallelism"]["gradient_accumulation"]
    )
    checkpoint_dir = output / "checkpoints" / f"checkpoint-{planned_microsteps}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir / "dit_model.safetensors"
    checkpoint.write_bytes(b"cpu-trained-agibot-checkpoint")
    native_config = json.loads(json.dumps({}))
    native_yaml_path = Path(training["agibot_native_config"]["path"])
    native_text = native_yaml_path.read_text(encoding="utf-8")
    import yaml
    native_config = yaml.safe_load(native_text)
    meta = {
        "checkpoint_contract_version": 2,
        "step": planned_microsteps,
        "world_size": plan["parallelism"]["world_size"],
        "format": "safetensors+optim_shards",
        "safetensors_exported": True,
        "model_config_dict": {
            "max_train_steps": planned_microsteps,
            "batch_size": plan["parallelism"]["batch_size"],
            "grad_accum_steps": plan["parallelism"]["gradient_accumulation"],
        },
        "training_config_dict": {
            "save_interval": planned_microsteps,
            "val_interval": planned_microsteps,
        },
    }
    if config.get("bad_checkpoint_binding"):
        meta["step"] += 1
    (checkpoint_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    if config.get("forge_checkpoint_receipt"):
        write("checkpoint.json", {"artifact_type": "forged-worker-receipt"})
    write("train-observed.json", {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "physical_gpu_indexes": os.environ.get("VERDIWM_PHYSICAL_GPU_INDEXES"),
        "physical_gpu_uuids": os.environ.get("VERDIWM_PHYSICAL_GPU_UUIDS"),
    })
elif stage == "engineering_diagnostic":
    checkpoint = Path(os.environ["VERDIWM_TRAINED_CHECKPOINT"])
    if os.environ.get("VERDIWM_TRAINED_CHECKPOINT") != str(checkpoint):
        raise SystemExit("checkpoint argument and environment disagree")
    if not checkpoint.is_file():
        raise SystemExit("trained checkpoint missing")
    write("diagnostic-observed.json", {
        "checkpoint_path": str(checkpoint),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "physical_gpu_indexes": os.environ.get("VERDIWM_PHYSICAL_GPU_INDEXES"),
        "physical_gpu_uuids": os.environ.get("VERDIWM_PHYSICAL_GPU_UUIDS"),
    })
    validation = bindings["validation_sample_window_ids"]
    window_digest = bindings["validation_sample_window_ids_sha256"]
    if config.get("bad_diagnostic_binding"):
        window_digest = "0" * 64
        receipt_bindings["validation_sample_window_ids_sha256"] = window_digest
    consumption_rows = [
        {"window_id": window_id, "loss": float(index + 1)}
        for index, window_id in enumerate(validation)
    ]
    if config.get("bad_diagnostic_trace"):
        consumption_rows.reverse()
    consumed_loss = sum(row["loss"] for row in consumption_rows) / len(consumption_rows)
    if config.get("bad_diagnostic_mean"):
        consumed_loss += 1.0
    consumption_trace_sha256 = hashlib.sha256(
        json.dumps(
            consumption_rows,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    raw_trace = {
        "schema_version": 1,
        "artifact_type": "verdiwm-agibot-checkpoint-evaluation-trace",
        "state": "complete",
        "model_run_id": manifest["model_run_id"],
        "checkpoint_identity": {
            "path": str(checkpoint),
            "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        },
        "checkpoint_meta_sha256": hashlib.sha256(
            (checkpoint.parent / "meta.json").read_bytes()
        ).hexdigest(),
        "checkpoint_reload_verified": True,
        "checkpoint_state_key_count": 1,
        "native_config_sha256": training["agibot_native_config"]["sha256"],
        "support_bindings": receipt_bindings,
        "consumption_trace": consumption_rows,
        "native_validation_loss": consumed_loss,
    }
    raw_path = output / "receipts" / "checkpoint-evaluation-raw.json"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps(raw_trace, sort_keys=True), encoding="utf-8")
    if config.get("forge_evidence_receipt"):
        write("evidence.json", {"artifact_type": "forged-worker-evidence"})
else:
    raise SystemExit("unknown stage")
'''


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _agibot_row(split: str, episode_uid: str, shard_prefix: str) -> dict[str, object]:
    def descriptor(component: str) -> dict[str, object]:
        return {
            "shard": f"{shard_prefix}-{component}.bin",
            "offset_bytes": 0,
            "shape": [6, 4],
            "dtype": "float32",
        }

    return {
        "dataset": "agibot",
        "split": split,
        "episode_id": 0 if split == "train" else 1,
        "episode_uid": episode_uid,
        "domain": {"dataset": "agibot", "domain_id": 9},
        "time": {"num_video_frames": 24, "num_latent_frames": 6},
        "latent": descriptor("latent"),
        "signals": descriptor("signals"),
        "model_condition_views": {"arm_slot_eef_pose": descriptor("condition")},
    }


def test_checkpoint_evaluator_help_exits_successfully():
    result = subprocess.run(
        [sys.executable, "-m", "wmloop.execute.agibot_checkpoint_evaluator", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--checkpoint" in result.stdout


def _protocol_v3_fixture(
    root: Path,
    *,
    config: dict[str, object] | None = None,
    world_model_domains: int = 10,
    wm_training_domains: int = 10,
    budget_total_gpu_hours: float = 1.0,
    native_config_supports_history_ref: bool = True,
):
    model = root / "model"
    data = root / "data"
    source = root / "native-source"
    base_model = root / "wan-diffusers-base"
    model.mkdir(parents=True)
    base_model.mkdir()
    (base_model / "config.json").write_text(
        json.dumps({"model_type": "ti2v"}), encoding="utf-8"
    )
    (base_model / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"fixture.weight": "diffusion-part.safetensors"}}),
        encoding="utf-8",
    )
    (base_model / "diffusion-part.safetensors").write_bytes(b"fixture base weights")
    (base_model / "Wan2.2_VAE.pth").write_bytes(b"cpu-fixture-vae")
    (model / "dit_model.safetensors").write_bytes(b"cpu-fixture-warmstart")
    (model / "meta.json").write_text(
        json.dumps({
            "model_config_dict": {
                "backbone": "wan2.2",
                "action_schema": "arm_slot",
                "action_dim": 10,
                "pred_frames": 3,
                "history_length": 5,
            }
        }),
        encoding="utf-8",
    )
    _write_json(model / "protocol-config.json", config or {})
    (model / "cpu_agibot_protocol.py").write_text(_WORKER, encoding="utf-8")

    _write_json(
        data / "manifest.json",
        {
            "production_contract": {
                "dataset_key": "agibot-cpu-protocol-fixture",
                "family": "agibot",
                "condition_view": "arm_slot_eef_pose",
                "body_conditioning": True,
            },
            "condition_schema": {"condition_view": "arm_slot_eef_pose"},
            "condition_schema_v2": {
                "body": {
                    "normalization": "stats/body_control.json schema_version=3 per-slot contract"
                }
            },
        },
    )
    _write_json(data / "stats" / "arm_slot_eef_pose.json", {"schema_version": 1})
    _write_json(
        data / "stats" / "body_control.json",
        {"schema_version": 3, "source": "train_only"},
    )
    train_native = [_agibot_row("train", "episode-train-001", "train")]
    validation_native = [_agibot_row("val", "episode-val-001", "val")]
    for name, rows in (
        ("train.index.jsonl", train_native),
        ("val.index.jsonl", validation_native),
    ):
        (data / name).write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
    for row in train_native + validation_native:
        descriptors = [
            row["latent"],
            row["signals"],
            *row["model_condition_views"].values(),
        ]
        for descriptor in descriptors:
            _write_json(
                data / "shards" / str(descriptor["shard"]), {"fixture": True}
            )
    for relative in _NATIVE_FILES:
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# fixture source: {relative}\n", encoding="utf-8")
    (source / "sa_wm" / "__init__.py").write_text("", encoding="utf-8")
    (source / "sa_wm" / "config.py").write_text(
        "from pathlib import Path\n"
        "import os\n"
        "import yaml\n"
        "\n"
        "class ConfigKeyError(ValueError):\n"
        "    pass\n"
        "\n"
        "def load_config(yaml_path=None, cli_overrides=None):\n"
        "    if cli_overrides != []:\n"
        "        raise ValueError('expected no CLI overrides')\n"
        "    if Path.cwd() != Path(__file__).resolve().parents[1]:\n"
        "        raise RuntimeError('native source root is not the working directory')\n"
        "    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':\n"
        "        raise RuntimeError('native config preflight can see CUDA devices')\n"
        "    config = yaml.safe_load(Path(yaml_path).read_text(encoding='utf-8'))\n"
        "    unsupported = config['world_model'].get('history_ref_enabled') is not None\n"
        f"    if unsupported and {not native_config_supports_history_ref!r}:\n"
        "        raise ConfigKeyError(\"Key 'history_ref_enabled' is not in 'WorldModelConfig'\")\n"
        "    (Path(__file__).resolve().parents[1] / 'native-config-preflight-observed').touch()\n"
        "    return config\n",
        encoding="utf-8",
    )
    preflight_python = root / "fixture-preflight-python"
    preflight_python.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        f"source_root = Path({str(source)!r})\n"
        "if Path.cwd() != source_root or os.environ.get('CUDA_VISIBLE_DEVICES') != '':\n"
        "    print(json.dumps({'error_type': 'RuntimeError', 'reason': 'preflight isolation failed'}))\n"
        "    raise SystemExit(1)\n"
        f"if {not native_config_supports_history_ref!r}:\n"
        "    print(json.dumps({'error_type': 'ConfigKeyError', 'reason': \"Key 'history_ref_enabled' is not in 'WorldModelConfig'\"}))\n"
        "    raise SystemExit(1)\n"
        "(source_root / 'native-config-preflight-observed').touch()\n",
        encoding="utf-8",
    )
    preflight_python.chmod(0o700)

    train_manifest = data / "train.json"
    validation_manifest = data / "validation.json"
    _write_json(
        train_manifest,
        {
            "records": [
                {
                    "episode_id": "training-episode",
                    "sample_id": f"training-episode:{index}:0:150",
                    "horizon_frames": 150,
                }
                for index in range(4)
            ]
        },
    )
    _write_json(
        validation_manifest,
        {
            "records": [
                {
                    "episode_id": "validation-episode",
                    "sample_id": f"validation-episode:{index}:0:150",
                    "horizon_frames": 150,
                }
                for index in range(4)
            ]
        },
    )
    plan = build_training_scale_plan(
        dataset_root=data,
        stage="engineering_diagnostic",
        batch_size=2,
        gradient_accumulation=2,
        requested_seed_count=1,
        seed=17,
    )
    support = build_agibot_training_support(
        data, source, loader=_LOADER, validation_sample_count=3
    )
    evaluator = root / "evaluator.json"
    _write_json(evaluator, {"metrics": ["native_validation_loss"], "fixture": True})
    training_config = root / "wan-training.yaml"
    training_config.write_text(
        "output_dir: /source/checkpoints\n"
        "world_model:\n"
        "  backbone: wan2.2\n"
        "  action_schema: arm_slot\n"
        "  action_dim: 10\n"
        "  pred_frames: 3\n"
        "  history_length: 5\n"
        f"  action_num_domains: {world_model_domains}\n"
        "  fsdp_shard_size: 8\n"
        "  max_train_steps: 0\n"
        "  batch_size: 4\n"
        "  grad_accum_steps: 1\n"
        "  history_ref_enabled: false\n"
        "wm_training:\n"
        f"  action_num_domains: {wm_training_domains}\n"
        "  output_dir: /source/training-output\n"
        "  dataset_roots: [/source/train-data]\n"
        "  dataset_probs: [0.25]\n"
        "  val_dataset_probs: [0.75]\n"
        "  warmup_steps: 100\n",
        encoding="utf-8",
    )
    original_training_config = training_config.read_bytes()
    execution = {
        "kind": "pipeline",
        "repo_root": str(model),
        "source_root": str(source),
        "output_root": str(root / "output"),
        "evaluator_contract": str(evaluator),
        "runtime_python": str(preflight_python),
        "budget_total_gpu_hours": budget_total_gpu_hours,
        "budget_db": str(root / "state" / "budget.db"),
        "cas_root": str(root / "state" / "archive"),
        "budget_max_trial_gpu_hours": 1,
        "asset_bindings": {
            "--base-model": str(base_model),
            "--init_from": str(model),
            "--vae-checkpoint": str(base_model / "Wan2.2_VAE.pth"),
        },
        "stage_timeouts_seconds": {"train": 30, "engineering_diagnostic": 30},
        "agibot_training_support": support,
        "agibot_native_config": str(training_config),
        "model_run_adapter": {
            "profile_id": "cpu-agibot-protocol-v3",
            "model_family": "agibot-fixture",
            "capability_level": "L2",
            "runner": {
                "requires_training": True,
                "protocol_version": 3,
                "adapter_id": None,
                "train": [
                    "{source_root}/scripts/training/train_world_model.py",
                    "--config",
                    "{agibot_native_config}",
                    "--init_from",
                    "{asset:--init_from}",
                ],
                "evaluate": [
                    "cpu_agibot_protocol.py",
                    "{model_run_manifest}",
                    "engineering_diagnostic",
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
        campaign_id="cpu-agibot-protocol-fixture",
        execution=execution,
        model=model,
        data=data,
        training_scale_plan=plan,
        resource_allocation=allocation,
    )
    output = root / "output"
    # Protocol-v3 binds its derived config while compiling, so the output
    # directory already exists when the manifest is returned.
    output.mkdir(exist_ok=True)
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


def _declare_fsdp_enabled(manifest: dict[str, object]) -> Path:
    training = manifest["training"]
    bound = training["agibot_native_config"]
    source_path = Path(bound["source_path"])
    source_text = source_path.read_text(encoding="utf-8")
    source_path.write_text(
        source_text.replace(
            "  fsdp_shard_size: 8\n",
            "  fsdp_enabled: true\n  fsdp_shard_size: 8\n",
            1,
        ),
        encoding="utf-8",
    )
    return source_path


def _fixture_asset_bindings(root: Path) -> dict[str, str]:
    base_model = root / "wan-diffusers-base"
    return {
        "--base-model": str(base_model),
        "--init_from": str(root / "model"),
        "--vae-checkpoint": str(base_model / "Wan2.2_VAE.pth"),
    }


def test_native_config_bind_rejects_fsdp_with_single_gpu_plan(tmp_path: Path):
    manifest, _manager = _protocol_v3_fixture(tmp_path)
    source_path = _declare_fsdp_enabled(manifest)

    with pytest.raises(
        AgibotNativeConfigError,
        match="MODEL_RUN_AGIBOT_FSDP_REQUIRES_MULTI_GPU_PLAN:.*world_size >= 2",
    ):
        bind_agibot_native_config(
            source_path,
            manifest["training"]["scale_plan"],
            output_root=tmp_path / "output",
            dataset_root=tmp_path / "data",
            asset_bindings=_fixture_asset_bindings(tmp_path),
        )


def test_native_config_validate_rejects_fsdp_with_single_gpu_plan(tmp_path: Path):
    manifest, _manager = _protocol_v3_fixture(tmp_path)
    _declare_fsdp_enabled(manifest)
    training = manifest["training"]

    with pytest.raises(
        AgibotNativeConfigError,
        match="MODEL_RUN_AGIBOT_FSDP_REQUIRES_MULTI_GPU_PLAN:.*world_size >= 2",
    ):
        validate_agibot_native_config(
            training["agibot_native_config"],
            training["scale_plan"],
            output_root=tmp_path / "output",
            dataset_root=tmp_path / "data",
            asset_bindings=_fixture_asset_bindings(tmp_path),
        )


def test_protocol_v3_native_config_preflight_runs_before_gpu_lease(tmp_path: Path):
    manifest, manager = _protocol_v3_fixture(tmp_path)
    preflight_observed = Path(manifest["source"]["source_root"]) / (
        "native-config-preflight-observed"
    )
    acquire_many = manager.acquire_many

    def acquire_after_preflight(*args, **kwargs):
        assert preflight_observed.is_file()
        return acquire_many(*args, **kwargs)

    manager.acquire_many = acquire_after_preflight
    result = execute_model_run(
        manifest,
        lease_manager=manager,
        process_runner=_cpu_protocol_process_runner(manifest),
    )

    assert result["state"] == "completed"


def test_protocol_v3_rejects_native_config_before_budget_or_gpu_lease(tmp_path: Path):
    manifest, manager = _protocol_v3_fixture(
        tmp_path, native_config_supports_history_ref=False
    )
    lease_calls = 0

    def reject_if_leased(*_args, **_kwargs):
        nonlocal lease_calls
        lease_calls += 1
        raise AssertionError("invalid native config must fail before GPU lease")

    manager.acquire_many = reject_if_leased

    with pytest.raises(
        ModelRunnerError,
        match=(
            "MODEL_RUN_AGIBOT_NATIVE_CONFIG_PREFLIGHT_FAILED:"
            "ConfigKeyError: Key 'history_ref_enabled'"
        ),
    ):
        execute_model_run(
            manifest,
            lease_manager=manager,
            process_runner=lambda *_args, **_kwargs: pytest.fail(
                "invalid native config must fail before training"
            ),
        )

    assert lease_calls == 0
    assert not Path(manifest["budget_binding"]["ledger_path"]).exists()
    assert not (tmp_path / "output" / "execution-receipt.json").exists()
    assert not (
        Path(manifest["source"]["source_root"])
        / "native-config-preflight-observed"
    ).exists()


def _reseal_model_run(manifest: dict[str, object]) -> None:
    """Recompute public digests so tests exercise semantic budget checks."""

    binding = manifest["budget_binding"]
    policy = binding["policy"]
    plan_document = {
        key: value
        for key, value in manifest.items()
        if key not in {"model_run_id", "budget_binding"}
    }
    plan_sha256 = model_run_budget_plan_sha256(
        plan_document,
        ledger_path=str(binding["ledger_path"]),
        cas_root=str(binding["cas_root"]),
        estimate=float(binding["estimated_gpu_hours"]),
        cost_class=str(binding["cost_class"]),
        human_approved=bool(binding["human_approved"]),
        policy=policy,
    )
    binding["plan_sha256"] = plan_sha256
    binding["trial_id"] = "model-run-trial-" + plan_sha256[:24]
    manifest["model_run_id"] = "model-run-" + model_run_digest(manifest)[:24]


def _set_manifest_world_size(manifest: dict[str, object], world_size: int) -> None:
    training = manifest["training"]
    plan = training["scale_plan"]
    parallelism = plan["parallelism"]
    parallelism["world_size"] = world_size
    parallelism["effective_batch_size"] = (
        parallelism["batch_size"]
        * parallelism["gradient_accumulation"]
        * world_size
    )
    training["scale_plan_digest"] = hashlib.sha256(
        json.dumps(plan, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()

    allocation = manifest["resource_binding"]["allocation"]
    allocation["roles"][0]["gpu_indices"] = list(range(world_size))

    native_config = training["agibot_native_config"]
    config_path = Path(native_config["path"])
    config_path.chmod(0o600)
    import yaml

    derived = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    derived["world_model"]["fsdp_shard_size"] = world_size
    config_path.write_text(
        yaml.safe_dump(derived, sort_keys=False), encoding="utf-8"
    )
    native_config["sha256"] = hashlib.sha256(config_path.read_bytes()).hexdigest()


def test_protocol_v3_over_budget_run_is_blocked_before_output_binding(tmp_path: Path):
    with pytest.raises(
        ModelRunError,
        match=r"MODEL_RUN_BUDGET_ESTIMATE_EXCEEDS_TOTAL:estimated=0.0166667:budget=0.01",
    ):
        _protocol_v3_fixture(tmp_path, budget_total_gpu_hours=0.01)

    assert not (tmp_path / "output" / "model-run.json").exists()
    assert not (tmp_path / "output" / "config" / "agibot-training.yaml").exists()


def test_protocol_v3_recomputes_budget_after_timeout_mutation_and_resealing(
    tmp_path: Path,
):
    manifest, _manager = _protocol_v3_fixture(tmp_path)
    manifest["execution_policy"]["stage_timeouts_seconds"]["train"] = 60.0
    _reseal_model_run(manifest)

    with pytest.raises(ModelRunError, match="MODEL_RUN_BUDGET_ESTIMATE_MISMATCH"):
        validate_model_run(manifest)


def test_protocol_v3_recomputes_budget_from_allocated_training_world_size(
    tmp_path: Path,
):
    manifest, _manager = _protocol_v3_fixture(tmp_path)
    _set_manifest_world_size(manifest, 2)
    _reseal_model_run(manifest)

    with pytest.raises(ModelRunError, match="MODEL_RUN_BUDGET_ESTIMATE_MISMATCH"):
        validate_model_run(manifest)


def test_protocol_v3_rejects_checkpoint_evaluation_beyond_planned_updates(
    tmp_path: Path,
):
    manifest, _manager = _protocol_v3_fixture(tmp_path)
    training = manifest["training"]
    plan = training["scale_plan"]
    plan["updates"]["checkpoint_eval_steps"] = [
        plan["updates"]["planned_steps"] + 1
    ]
    training["scale_plan_digest"] = hashlib.sha256(
        json.dumps(
            plan, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()
    _reseal_model_run(manifest)

    with pytest.raises(
        ModelRunError, match="MODEL_RUN_SCALE_PLAN_CHECKPOINT_SCHEDULE_INVALID"
    ):
        validate_model_run(manifest)


def test_protocol_v3_rejects_resealed_estimate_over_declared_budget(tmp_path: Path):
    manifest, _manager = _protocol_v3_fixture(tmp_path)
    manifest["execution_policy"]["stage_timeouts_seconds"] = {
        "train": 2400.0,
        "engineering_diagnostic": 1200.0,
    }
    manifest["budget_binding"]["estimated_gpu_hours"] = 1.0
    manifest["budget_binding"]["policy"]["total_gpu_hours"] = 0.5
    _reseal_model_run(manifest)

    with pytest.raises(
        ModelRunError,
        match=r"MODEL_RUN_BUDGET_ESTIMATE_EXCEEDS_TOTAL:estimated=1:budget=0.5",
    ):
        validate_model_run(manifest)


def _write_cpu_training_trace(manifest: dict[str, object]) -> None:
    output = Path(manifest["output"]["root"])
    config_path = output / "runtime" / "agibot-trace-config.json"
    config = trace_runtime.load_trace_config(config_path)
    plan = manifest["training"]["scale_plan"]
    updates = plan["updates"]["planned_steps"]
    accumulation = plan["parallelism"]["gradient_accumulation"]
    support = manifest["training"]["agibot_training_support"]
    validation_ids = support["validation_support"]["sample_window_ids"]
    batch_size = config["batch_size"]
    for rank in range(config["world_size"]):
        recorder = trace_runtime._Recorder(
            config, trace_runtime.trace_shard_path(config, rank), rank
        )
        for update in range(1, updates + 1):
            for accumulated_step in range(accumulation):
                microstep = (update - 1) * accumulation + accumulated_step + 1
                recorder.add_train_batch(
                    samples=[
                        {
                            "split_id": "train",
                            "dataset_index": 0,
                            "dataset_key": "agibot-cpu-protocol-fixture",
                            "episode_entry_index": 0,
                            "source_sample_index": microstep * batch_size + slot,
                            "episode_uid": "episode-train-001",
                            "first_future_frame": microstep * 8 + slot,
                            "sample_window_id": (
                                f"episode-train-001:{microstep * 8 + slot:06d}"
                            ),
                        }
                        for slot in range(batch_size)
                    ],
                    microstep=microstep,
                    loss=1.0 / update,
                )
            recorder.commit_optimizer_update(
                expected_microbatches=accumulation,
                learning_rate_min=1e-4,
                learning_rate_max=1e-4,
                parameter_delta_max_abs=1e-6,
                parameter_sample_count=5,
                parameter_changed_sample_count=1,
            )
        recorder.begin_validation_pass()
        for index, window_id in enumerate(validation_ids):
            episode_uid, frame = window_id.rsplit(":", 1)
            recorder.add_validation_batch(
                samples=[{
                    "split_id": "val",
                    "dataset_index": 0,
                    "dataset_key": "agibot-cpu-protocol-fixture",
                    "episode_entry_index": 0,
                    "source_sample_index": index,
                    "episode_uid": episode_uid,
                    "first_future_frame": int(frame),
                    "sample_window_id": window_id,
                }],
                batch_index=index,
                loss=float(index + 1),
            )
        recorder.finalize(expected_optimizer_updates=updates)
    fixture_config = json.loads(
        (Path(manifest["source"]["model_root"]) / "protocol-config.json").read_text(
            encoding="utf-8"
        )
    )
    if fixture_config.get("nonfinite_training_loss"):
        shard_path = trace_runtime.trace_shard_path(config, 0)
        records = [
            json.loads(line)
            for line in shard_path.read_text(encoding="utf-8").splitlines()
        ]
        for record in records:
            if record.get("record_type") == "event" and record.get("phase") == "train":
                record["batch_loss"] = float("nan")
                break
        shard_path.write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
            encoding="utf-8",
        )


def _cpu_protocol_process_runner(manifest: dict[str, object]):
    output = Path(manifest["output"]["root"])
    worker = Path(manifest["source"]["model_root"]) / "cpu_agibot_protocol.py"
    manifest_path = Path(manifest["manifest_path"])

    def run(command, **kwargs):
        if "torch.distributed.run" in command:
            # Keep the production command-construction boundary under test:
            # only replace torchrun/native GPU execution with this CPU trainer.
            assert command[0] == manifest["runtime"]["python"]
            assert command[1:5] == [
                "-m",
                "torch.distributed.run",
                "--nproc_per_node",
                str(manifest["training"]["scale_plan"]["parallelism"]["world_size"]),
            ]
            bridge = next(
                Path(token) for token in command
                if Path(token).name == "agibot_trace_bridge.py"
            )
            assert bridge.name == "agibot_trace_bridge.py"
            trace_option = command.index("--trace-config")
            assert Path(command[trace_option + 1]) == (
                output / "runtime" / "agibot-trace-config.json"
            )
            separator = command.index("--")
            native_command = command[separator + 1 :]
            assert native_command == manifest["runtime"]["train_command"]
            assert native_command[0].endswith(
                "scripts/training/train_world_model.py"
            )
            result = subprocess.run(
                [sys.executable, str(worker), str(manifest_path), "train"],
                **kwargs,
            )
            if result.returncode == 0:
                _write_cpu_training_trace(manifest)
            return subprocess.CompletedProcess(
                command, result.returncode, result.stdout, result.stderr
            )
        checkpoint = Path(manifest["training"]["checkpoint_receipt"])
        checkpoint_receipt = json.loads(checkpoint.read_text(encoding="utf-8"))
        trained_checkpoint = checkpoint_receipt["checkpoint_identity"]["path"]
        assert command[0] == manifest["runtime"]["python"]
        assert "wmloop.execute.agibot_checkpoint_evaluator" in command
        assert command[command.index("--checkpoint") + 1] == trained_checkpoint
        assert command[command.index("--raw-output") + 1] == str(
            output / "receipts" / "checkpoint-evaluation-raw.json"
        )
        assert kwargs["env"]["VERDIWM_TRAINED_CHECKPOINT"] == trained_checkpoint
        result = subprocess.run(
            [sys.executable, str(worker), str(manifest_path), "engineering_diagnostic"],
            **kwargs,
        )
        return subprocess.CompletedProcess(
            command, result.returncode, result.stdout, result.stderr
        )

    return run


def test_protocol_v3_freezes_explicit_native_config_bytes_and_domain_capacity(
    tmp_path: Path,
):
    manifest, _manager = _protocol_v3_fixture(tmp_path)
    bound = manifest["training"]["agibot_native_config"]
    config_path = Path(bound["path"])
    source_path = Path(bound["source_path"])
    import yaml

    source = yaml.safe_load(source_path.read_bytes())
    derived = yaml.safe_load(config_path.read_bytes())
    output = tmp_path / "output"
    assert bound["sha256"] == hashlib.sha256(config_path.read_bytes()).hexdigest()
    assert bound["source_sha256"] == hashlib.sha256(source_path.read_bytes()).hexdigest()
    assert bound["base_model_path"] == str(tmp_path / "wan-diffusers-base")
    assert bound["init_from_path"] == str(tmp_path / "model")
    assert bound["vae_checkpoint_path"] == str(
        tmp_path / "wan-diffusers-base" / "Wan2.2_VAE.pth"
    )
    train_command = manifest["runtime"]["train_command"]
    assert train_command[-2:] == ["--init_from", str(tmp_path / "model")]
    assert bound["world_model_action_num_domains"] == 10
    assert bound["wm_training_action_num_domains"] == 10
    assert source_path.read_bytes() == (tmp_path / "wan-training.yaml").read_bytes()
    assert derived["output_dir"] == str(output / "checkpoints")
    assert derived["world_model"]["checkpoint"] == str(
        tmp_path / "wan-diffusers-base"
    )
    assert derived["world_model"]["vae_checkpoint"] == str(
        tmp_path / "wan-diffusers-base" / "Wan2.2_VAE.pth"
    )
    assert derived["world_model"]["fsdp_shard_size"] == manifest["training"]["scale_plan"]["parallelism"]["world_size"]
    assert derived["wm_training"]["output_dir"] == str(output / "checkpoints")
    assert derived["wm_training"]["dataset_roots"] == [str(tmp_path / "data")]
    assert derived["wm_training"]["dataset_probs"] == [1.0]
    assert derived["wm_training"]["val_dataset_probs"] == [1.0]
    plan = manifest["training"]["scale_plan"]
    optimizer_updates = plan["updates"]["planned_steps"]
    accumulation = plan["parallelism"]["gradient_accumulation"]
    assert derived["world_model"]["max_train_steps"] == optimizer_updates * accumulation
    assert derived["world_model"]["batch_size"] == plan["parallelism"]["batch_size"]
    assert derived["world_model"]["grad_accum_steps"] == accumulation
    assert derived["wm_training"]["warmup_steps"] == 0
    assert source["world_model"]["max_train_steps"] == 0
    assert source["world_model"]["batch_size"] == 4
    assert source["world_model"]["grad_accum_steps"] == 1
    assert source["wm_training"]["dataset_roots"] == ["/source/train-data"]
    assert source["wm_training"]["warmup_steps"] == 100
    assert config_path == output / "config" / "agibot-training.yaml"
    assert output / "checkpoints" != tmp_path / "model"

    config_path.write_text(
        "world_model:\n  action_num_domains: 9\n"
        "wm_training:\n  action_num_domains: 10\n",
        encoding="utf-8",
    )
    with pytest.raises(ModelRunError, match="AGIBOT_NATIVE_CONFIG_BINDING_MISMATCH"):
        validate_model_run(manifest)


@pytest.mark.parametrize(
    "broken_asset",
    [
        "base_shard_missing",
        "base_shard_empty",
        "base_shard_parent_symlink",
        "warmstart_weights_empty",
        "warmstart_metadata",
    ],
)
def test_protocol_v3_rejects_invalid_model_assets_before_budget_or_gpu_lease(
    tmp_path: Path, broken_asset: str
):
    manifest, manager = _protocol_v3_fixture(tmp_path)
    base_model = tmp_path / "wan-diffusers-base"
    warmstart = tmp_path / "model"
    if broken_asset == "base_shard_missing":
        (base_model / "diffusion-part.safetensors").unlink()
        expected = "MODEL_RUN_AGIBOT_BASE_MODEL_SHARD_MISSING"
    elif broken_asset == "base_shard_empty":
        (base_model / "diffusion-part.safetensors").write_bytes(b"")
        expected = "MODEL_RUN_AGIBOT_BASE_MODEL_SHARD_EMPTY"
    elif broken_asset == "base_shard_parent_symlink":
        (base_model / "diffusion-part.safetensors").unlink()
        outside = tmp_path / "outside-shards"
        outside.mkdir()
        (outside / "diffusion-part.safetensors").write_bytes(b"external shard")
        (base_model / "linked-shards").symlink_to(outside, target_is_directory=True)
        index_path = base_model / "diffusion_pytorch_model.safetensors.index.json"
        index_path.write_text(
            json.dumps(
                {"weight_map": {"fixture.weight": "linked-shards/diffusion-part.safetensors"}}
            ),
            encoding="utf-8",
        )
        expected = "MODEL_RUN_AGIBOT_BASE_MODEL_SHARD_MISSING"
    elif broken_asset == "warmstart_weights_empty":
        (warmstart / "dit_model.safetensors").write_bytes(b"")
        expected = "MODEL_RUN_AGIBOT_WARMSTART_WEIGHTS_EMPTY"
    else:
        metadata_path = warmstart / "meta.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["model_config_dict"]["action_dim"] = 11
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        expected = "MODEL_RUN_AGIBOT_WARMSTART_CONFIG_MISMATCH:action_dim"

    lease_calls = 0

    def reject_if_leased(*_args, **_kwargs):
        nonlocal lease_calls
        lease_calls += 1
        raise AssertionError("invalid model assets must fail before GPU lease")

    manager.acquire_many = reject_if_leased
    with pytest.raises(ModelRunnerError, match=expected):
        execute_model_run(
            manifest,
            lease_manager=manager,
            process_runner=lambda *_args, **_kwargs: pytest.fail(
                "invalid model assets must fail before training"
            ),
        )

    assert lease_calls == 0
    assert not Path(manifest["budget_binding"]["ledger_path"]).exists()


def test_protocol_v3_rejects_source_config_mutation_after_compilation(tmp_path: Path):
    manifest, _manager = _protocol_v3_fixture(tmp_path)
    source_path = Path(manifest["training"]["agibot_native_config"]["source_path"])
    source_path.write_text(source_path.read_text() + "# modified\n", encoding="utf-8")

    with pytest.raises(ModelRunError, match="AGIBOT_NATIVE_CONFIG_SOURCE_BINDING_MISMATCH"):
        validate_model_run(manifest)


def test_protocol_v3_rejects_native_config_checkpoint_path_escape(tmp_path: Path):
    manifest, _manager = _protocol_v3_fixture(tmp_path)
    config_path = Path(manifest["training"]["agibot_native_config"]["path"])
    config_path.chmod(0o600)
    config_path.write_text(
        config_path.read_text().replace(str(tmp_path / "output" / "checkpoints"), "/tmp/escape"),
        encoding="utf-8",
    )

    with pytest.raises(ModelRunError, match="AGIBOT_NATIVE_CONFIG_BINDING_MISMATCH"):
        validate_model_run(manifest)


def test_protocol_v3_refuses_native_config_with_too_small_domain_capacity(
    tmp_path: Path,
):
    with pytest.raises(
        ModelRunError, match="AGIBOT_NATIVE_CONFIG_DOMAIN_CAPACITY_TOO_SMALL"
    ):
        _protocol_v3_fixture(
            tmp_path, world_model_domains=9, wm_training_domains=10
        )


def test_protocol_v3_manifest_schema_rejects_unbound_native_config_fields(
    tmp_path: Path,
):
    manifest, _manager = _protocol_v3_fixture(tmp_path)
    manifest["training"]["agibot_native_config"]["unexpected"] = "value"

    with pytest.raises(ModelRunError, match="MODEL_RUN_SCHEMA_INVALID"):
        validate_model_run(manifest)


def test_protocol_v3_runs_cpu_train_and_engineering_diagnostic_with_bound_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    manifest, manager = _protocol_v3_fixture(tmp_path)
    output = tmp_path / "output"
    monkeypatch.setenv("WANDB_API_KEY", "must-not-reach-child")
    monkeypatch.setenv("SWANLAB_API_KEY", "must-not-reach-child")
    monkeypatch.setenv("HF_TOKEN", "must-not-reach-child")
    monkeypatch.setenv("HTTPS_PROXY", "http://must-not-reach-child")
    monkeypatch.setenv("PYTHONPATH", "/caller/private/pythonpath")
    monkeypatch.setenv("CUDA_HOME", "/fixture/cuda")
    monkeypatch.setenv("NCCL_DEBUG", "INFO")
    cpu_runner = _cpu_protocol_process_runner(manifest)
    observed_stages: list[str] = []
    runtime_tmp_aliases: list[Path] = []

    def isolated_runner(command, **kwargs):
        observed_stages.append("train" if "torch.distributed.run" in command else "evaluation")
        cwd = Path(kwargs["cwd"]).resolve()
        env = kwargs["env"]
        assert cwd == output / "runtime" / "workspace"
        assert cwd != Path(manifest["source"]["model_root"])
        assert cwd != Path(manifest["source"]["source_root"])
        for name in ("HOME", "TMPDIR", "XDG_CACHE_HOME", "WANDB_DIR", "SWANLAB_LOG_DIR"):
            assert Path(env[name]).resolve().is_relative_to(output.resolve())
        if os.name != "nt":
            runtime_tmp_aliases.append(Path(env["TMPDIR"]))
            assert Path(env["TMPDIR"]).is_symlink()
            assert len(os.fsencode(env["TMPDIR"])) <= 64
        assert env["PYTHONDONTWRITEBYTECODE"] == "1"
        assert env["PYTHONNOUSERSITE"] == "1"
        assert env["CUDA_HOME"] == "/fixture/cuda"
        assert env["NCCL_DEBUG"] == "INFO"
        assert env["WANDB_MODE"] == "disabled"
        assert env["WANDB_DISABLED"] == "true"
        assert env["SWANLAB_MODE"] == "disabled"
        assert env["HF_HUB_OFFLINE"] == "1"
        assert env["HF_HUB_DISABLE_TELEMETRY"] == "1"
        assert env["TRANSFORMERS_OFFLINE"] == "1"
        assert env["HF_DATASETS_OFFLINE"] == "1"
        assert "/caller/private/pythonpath" not in env["PYTHONPATH"]
        for name in ("WANDB_API_KEY", "SWANLAB_API_KEY", "HF_TOKEN", "HTTPS_PROXY"):
            assert name not in env
        return cpu_runner(command, **kwargs)

    result = execute_model_run(
        manifest,
        lease_manager=manager,
        process_runner=isolated_runner,
    )

    assert result["state"] == "completed"
    assert result["settlement_class"] == "engineering_diagnostic"
    assert result["evidence_class"] == "runtime_only"
    if os.name != "nt":
        assert runtime_tmp_aliases
        assert all(not path.exists() and not path.is_symlink() for path in runtime_tmp_aliases)
    assert result["quality_eligible"] is False
    assert result["agibot_psnr_execution"]["state"] == "unavailable"
    assert result["agibot_psnr_execution"]["quality_eligible"] is False
    assert result["agibot_psnr_execution"]["reason"] == (
        "AGIBOT_PAIRED_PSNR_ROLLOUT_BUILDER_UNAVAILABLE"
    )
    assert [stage["stage"] for stage in result["stages"]] == [
        "train",
        "engineering_diagnostic",
    ]
    assert result["trained_checkpoint_path"] == str(
        output / "checkpoints" / "checkpoint-2" / "dit_model.safetensors"
    )
    assert result["engineering_diagnostic_receipt"] == str(output / "evidence.json")
    assert result["training_trace_path"] == str(
        output / "receipts" / "training-trace" / "merged.json"
    )
    assert result["training_trace_sha256"] == hashlib.sha256(
        (output / "receipts" / "training-trace" / "merged.json").read_bytes()
    ).hexdigest()
    assert result["stages"][0]["command"][1:4] == [
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
    ]
    assert "--log-dir" in result["stages"][0]["command"]
    assert result["stages"][0]["command"][result["stages"][0]["command"].index("--tee") + 1] == "3"
    assert "not a model-effect result" in result["claim_boundary"]
    assert observed_stages == ["train", "evaluation"]
    assert manifest["isolation"] == {
        "model_source_read_only": True,
        "dataset_read_only": True,
        "evaluator_read_only": True,
        "output_root_fresh": True,
        "network_required": False,
        "network_policy": "known_clients_offline_only",
        "os_sandbox_enforced": False,
    }
    expected_gpu_binding = {
        "cuda_visible_devices": "0",
        "physical_gpu_indexes": "0",
        "physical_gpu_uuids": "cpu-fixture-gpu-0",
    }
    train_observed = json.loads((output / "train-observed.json").read_text())
    diagnostic_observed = json.loads(
        (output / "diagnostic-observed.json").read_text()
    )
    assert train_observed == expected_gpu_binding
    assert diagnostic_observed == {
        "checkpoint_path": result["trained_checkpoint_path"],
        **expected_gpu_binding,
    }


@pytest.mark.skipif(os.name == "nt", reason="AF_UNIX listener paths are POSIX-specific")
def test_deep_run_tmp_path_uses_short_multiprocessing_socket_directory(
    tmp_path: Path,
):
    output = tmp_path.joinpath(*(f"nested-{index}" for index in range(12)), "output")
    output.mkdir(parents=True)
    _cwd, env, cleanup = agibot_runtime_context(
        {
            "source": {"source_root": str(tmp_path)},
            "runtime": {"python": str(Path(sys.executable))},
        },
        output,
        lease_environment={},
    )
    try:
        assert Path(env["TMPDIR"]).resolve() == output / "runtime" / "tmp"
        child_env = {**os.environ, "TMPDIR": env["TMPDIR"]}
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from multiprocessing.connection import Listener; "
                "listener=Listener(authkey=b'verdiwm-test', family='AF_UNIX'); "
                "print(listener.address); listener.close()",
            ],
            env=child_env,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        socket_path = result.stdout.strip()
        assert len(os.fsencode(socket_path)) < 104
        assert Path(socket_path).is_relative_to(Path(env["TMPDIR"]))
    finally:
        cleanup()
    assert not Path(env["TMPDIR"]).exists()


def test_protocol_v3_worker_cannot_forge_authoritative_receipts(tmp_path: Path):
    manifest, manager = _protocol_v3_fixture(
        tmp_path,
        config={
            "forge_checkpoint_receipt": True,
            "forge_evidence_receipt": True,
        },
    )

    result = execute_model_run(
        manifest,
        lease_manager=manager,
        process_runner=_cpu_protocol_process_runner(manifest),
    )

    checkpoint_receipt = json.loads(
        (tmp_path / "output" / "checkpoint.json").read_text(encoding="utf-8")
    )
    evidence_receipt = json.loads(
        (tmp_path / "output" / "evidence.json").read_text(encoding="utf-8")
    )
    assert result["state"] == "completed"
    assert checkpoint_receipt["artifact_type"] == (
        "verdiwm-agibot-engineering-diagnostic-checkpoint-receipt"
    )
    assert evidence_receipt["artifact_type"] == (
        "verdiwm-agibot-engineering-diagnostic-evaluation-receipt"
    )
    assert checkpoint_receipt["checkpoint_identity"]["path"] == (
        result["trained_checkpoint_path"]
    )
    assert evidence_receipt["checkpoint_identity"]["path"] == (
        result["trained_checkpoint_path"]
    )


@pytest.mark.parametrize(
    ("config", "error_code"),
    [
        (
            {"bad_checkpoint_binding": True},
            "MODEL_RUN_AGIBOT_CHECKPOINT_METADATA_MISMATCH",
        ),
        (
            {"bad_diagnostic_binding": True},
            "MODEL_RUN_AGIBOT_DIAGNOSTIC_TRACE_BINDING_MISMATCH",
        ),
        (
            {"nonfinite_training_loss": True},
            "MODEL_RUN_AGIBOT_TRACE_INVALID:AGIBOT_TRACE_EVENT_LOSS_INVALID",
        ),
        (
            {"bad_diagnostic_trace": True},
            "MODEL_RUN_AGIBOT_DIAGNOSTIC_TRACE_BINDING_MISMATCH",
        ),
        (
            {"bad_diagnostic_mean": True},
            "MODEL_RUN_AGIBOT_DIAGNOSTIC_TRACE_MEAN_MISMATCH",
        ),
    ],
)
def test_protocol_v3_rejects_receipts_bound_to_other_support(
    tmp_path: Path, config: dict[str, object], error_code: str
):
    manifest, manager = _protocol_v3_fixture(tmp_path, config=config)

    with pytest.raises(ModelRunnerError, match=error_code):
        execute_model_run(
            manifest,
            lease_manager=manager,
            process_runner=_cpu_protocol_process_runner(manifest),
        )

    receipt = json.loads((tmp_path / "output" / "execution-receipt.json").read_text())
    assert receipt["state"] == "failed"
    assert receipt["launch_attempted"] is True
    assert receipt["budget_state"] == "settled"
    assert receipt["quality_eligible"] is not True


@pytest.mark.parametrize(
    ("mutation", "error_code"),
    [
        ("progress_binding", "MODEL_RUN_AGIBOT_TRAINING_PROGRESS_BINDING_MISMATCH"),
        ("nonfinite_loss", "MODEL_RUN_AGIBOT_TRAINING_PROGRESS_INVALID"),
    ],
)
def test_protocol_v3_rejects_invalid_training_progress(
    mutation: str, error_code: str
):
    progress = {
        "seed": 17,
        "planned_optimizer_updates": 1,
        "completed_optimizer_updates": 1,
        "planned_microsteps": 2,
        "completed_microsteps": 2,
        "training_loss_rows": [
            {"optimizer_update": 1, "microstep": 2, "loss": 1.0}
        ],
    }
    if mutation == "progress_binding":
        progress["completed_microsteps"] = 1
    else:
        progress["training_loss_rows"][0]["loss"] = float("nan")

    with pytest.raises(ModelRunnerError, match=error_code):
        _validate_agibot_training_progress(
            progress, seed=17, optimizer_updates=1, microsteps=2
        )


@pytest.mark.parametrize(
    ("failed_stage", "error_code"),
    [
        ("train", "MODEL_RUN_TRAIN_FAILED"),
        ("engineering_diagnostic", "MODEL_RUN_ENGINEERING_DIAGNOSTIC_FAILED"),
    ],
)
def test_protocol_v3_nonzero_stage_exit_never_settles_diagnostic(
    tmp_path: Path, failed_stage: str, error_code: str
):
    manifest, manager = _protocol_v3_fixture(
        tmp_path, config={"fail_stage": failed_stage}
    )
    runtime_tmp_aliases: list[Path] = []
    cpu_runner = _cpu_protocol_process_runner(manifest)

    def record_tmp_alias(command, **kwargs):
        runtime_tmp_aliases.append(Path(kwargs["env"]["TMPDIR"]))
        return cpu_runner(command, **kwargs)

    with pytest.raises(ModelRunnerError, match=error_code):
        execute_model_run(
            manifest,
            lease_manager=manager,
            process_runner=record_tmp_alias,
        )

    receipt = json.loads((tmp_path / "output" / "execution-receipt.json").read_text())
    assert receipt["state"] == "failed"
    assert receipt["launch_attempted"] is True
    assert receipt["budget_state"] == "settled"
    assert not (tmp_path / "output" / "evidence.json").exists()
    if os.name != "nt":
        assert runtime_tmp_aliases
        assert all(not path.exists() and not path.is_symlink() for path in runtime_tmp_aliases)


def test_protocol_v3_failure_receipt_keeps_bounded_output_head_and_tail(
    tmp_path: Path,
):
    manifest, manager = _protocol_v3_fixture(tmp_path)
    stdout = "TRAIN STDOUT START\n" + ("o" * 5000) + "\nTRAIN STDOUT END"
    stderr = (
        "Traceback (most recent call last):\n"
        + ("e" * 5000)
        + "\nRuntimeError: original training failure"
    )

    def fail_with_logs(command, **_kwargs):
        return subprocess.CompletedProcess(command, 9, stdout, stderr)

    with pytest.raises(ModelRunnerError, match="MODEL_RUN_TRAIN_FAILED"):
        execute_model_run(manifest, lease_manager=manager, process_runner=fail_with_logs)

    receipt = json.loads((tmp_path / "output" / "execution-receipt.json").read_text())
    stage = receipt["stages"][0]
    assert stage["stderr_head"].startswith("Traceback (most recent call last):")
    assert stage["stderr_tail"].endswith("RuntimeError: original training failure")
    assert stage["stdout_head"].startswith("TRAIN STDOUT START")
    assert stage["stdout_tail"].endswith("TRAIN STDOUT END")
    assert all(
        len(stage[key]) <= 2000
        for key in ("stdout_head", "stdout_tail", "stderr_head", "stderr_tail")
    )


def test_protocol_v3_failure_receipt_includes_torch_elastic_worker_root_cause(
    tmp_path: Path,
):
    manifest, manager = _protocol_v3_fixture(tmp_path)

    def fail_with_elastic_summary(command, **_kwargs):
        log_dir = Path(command[command.index("--log-dir") + 1])
        worker_stderr = log_dir / "attempt_0" / "0" / "stderr.log"
        worker_stderr.parent.mkdir(parents=True)
        worker_stderr.write_text(
            "Traceback (most recent call last):\n"
            "RuntimeError: worker ran out of CUDA memory\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(
            command,
            1,
            "",
            "torch.distributed.elastic.multiprocessing.errors.ChildFailedError\n"
            "error_file=N/A\n",
        )

    with pytest.raises(ModelRunnerError, match="MODEL_RUN_TRAIN_FAILED"):
        execute_model_run(
            manifest, lease_manager=manager, process_runner=fail_with_elastic_summary
        )

    receipt = json.loads((tmp_path / "output" / "execution-receipt.json").read_text())
    stage = receipt["stages"][0]
    assert "Torch Elastic rank 0: RuntimeError: worker ran out of CUDA memory" in stage["stderr_tail"]
    assert len(stage["stderr_tail"]) <= 2000


def test_protocol_v3_completion_is_recovered_without_retraining(tmp_path: Path):
    manifest, manager = _protocol_v3_fixture(tmp_path)
    runner = _cpu_protocol_process_runner(manifest)
    first = execute_model_run(manifest, lease_manager=manager, process_runner=runner)
    second = execute_model_run(manifest, lease_manager=manager, process_runner=runner)

    assert first["state"] == second["state"] == "completed"
    assert second["budget_receipt_ref"].startswith("cas://sha256/")
    assert len(second["stages"]) == 2


def test_protocol_v3_completed_recovery_revalidates_diagnostic_evidence(
    tmp_path: Path,
):
    manifest, manager = _protocol_v3_fixture(tmp_path)
    execute_model_run(
        manifest,
        lease_manager=manager,
        process_runner=_cpu_protocol_process_runner(manifest),
    )

    evidence_path = tmp_path / "output" / "evidence.json"
    evidence = json.loads(evidence_path.read_text())
    evidence["consumption_trace"]["rows"][0]["loss"] += 10.0
    evidence["consumption_trace"]["sha256"] = hashlib.sha256(
        json.dumps(
            evidence["consumption_trace"]["rows"],
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    evidence_path.write_text(json.dumps(evidence, sort_keys=True), encoding="utf-8")

    def no_process_launch(*_args, **_kwargs):
        raise AssertionError("completed recovery must not launch a stage")

    with pytest.raises(
        ModelRunnerError,
        match="MODEL_RUN_RECOVERY_DIAGNOSTIC_BINDING_MISMATCH",
    ):
        execute_model_run(
            manifest,
            lease_manager=manager,
            process_runner=no_process_launch,
        )

    receipt = json.loads((tmp_path / "output" / "execution-receipt.json").read_text())
    assert receipt["state"] == "completed"
    assert receipt["budget_state"] == "settled"


def test_protocol_v3_completed_recovery_revalidates_checkpoint_progress(
    tmp_path: Path,
):
    manifest, manager = _protocol_v3_fixture(tmp_path)
    execute_model_run(
        manifest,
        lease_manager=manager,
        process_runner=_cpu_protocol_process_runner(manifest),
    )

    checkpoint_receipt_path = tmp_path / "output" / "checkpoint.json"
    checkpoint_receipt = json.loads(checkpoint_receipt_path.read_text())
    checkpoint_receipt["checkpoint_training_progress"]["seed"] += 1
    checkpoint_receipt_path.write_text(
        json.dumps(checkpoint_receipt, sort_keys=True), encoding="utf-8"
    )

    def no_process_launch(*_args, **_kwargs):
        raise AssertionError("completed recovery must not launch a stage")

    with pytest.raises(
        ModelRunnerError,
        match="MODEL_RUN_RECOVERY_CHECKPOINT_BINDING_MISMATCH",
    ):
        execute_model_run(
            manifest,
            lease_manager=manager,
            process_runner=no_process_launch,
        )

    receipt = json.loads((tmp_path / "output" / "execution-receipt.json").read_text())
    assert receipt["state"] == "completed"
    assert receipt["budget_state"] == "settled"


def test_protocol_v3_cancellation_before_gpu_lease_releases_budget(tmp_path: Path):
    manifest, manager = _protocol_v3_fixture(tmp_path)

    result = execute_model_run(
        manifest,
        lease_manager=manager,
        process_runner=_cpu_protocol_process_runner(manifest),
        cancel_requested=lambda: True,
    )

    assert result["state"] == "cancelled_before_launch"
    assert result["launch_attempted"] is False
    assert result["budget_state"] == "released"
    assert not (tmp_path / "output" / "checkpoint.json").exists()
    ledger = BudgetLedger(
        Path(manifest["budget_binding"]["ledger_path"]),
        BudgetPolicy(total_gpu_hours=1.0, max_trial_gpu_hours=1.0),
    )
    released = ledger.get(manifest["budget_binding"]["trial_id"])
    assert released is not None and released.state == "released"
    receipt_path = tmp_path / "output" / "execution-receipt.json"
    receipt_path.unlink()
    with pytest.raises(ModelRunnerError, match="RECOVERY_REQUIRED"):
        execute_model_run(
            manifest,
            lease_manager=manager,
            process_runner=_cpu_protocol_process_runner(manifest),
        )


def test_protocol_v3_process_start_failure_is_settled_and_not_retried(tmp_path: Path):
    manifest, manager = _protocol_v3_fixture(tmp_path)
    calls = 0

    def fail_to_start(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise OSError("injected process-start failure")

    with pytest.raises(ModelRunnerError, match="MODEL_RUN_PROCESS_START_FAILED"):
        execute_model_run(manifest, lease_manager=manager, process_runner=fail_to_start)

    receipt_path = tmp_path / "output" / "execution-receipt.json"
    receipt = json.loads(receipt_path.read_text())
    assert receipt["state"] == "failed"
    assert receipt["launch_attempted"] is True
    assert receipt["budget_state"] == "settled"
    # A terminal reservation with missing or damaged evidence never launches again.
    receipt_path.unlink()
    with pytest.raises(ModelRunnerError, match="RECOVERY_REQUIRED"):
        execute_model_run(manifest, lease_manager=manager, process_runner=fail_to_start)
    assert calls == 1


def test_protocol_v3_timeout_persists_terminal_settlement(tmp_path: Path):
    manifest, manager = _protocol_v3_fixture(tmp_path)
    timeout_output = "TRACEBACK START\n" + ("x" * 5000) + "\nTIMEOUT ROOT CAUSE"
    runtime_tmp_aliases: list[Path] = []

    def timeout(command, **kwargs):
        runtime_tmp_aliases.append(Path(kwargs["env"]["TMPDIR"]))
        raise subprocess.TimeoutExpired(
            command,
            kwargs["timeout"],
            output=timeout_output,
            stderr=timeout_output,
        )

    with pytest.raises(ModelRunnerError, match="MODEL_RUN_TRAIN_TIMEOUT"):
        execute_model_run(manifest, lease_manager=manager, process_runner=timeout)

    receipt = json.loads((tmp_path / "output" / "execution-receipt.json").read_text())
    assert receipt["state"] == "timed_out"
    assert receipt["timed_out_stage"] == "train"
    assert receipt["launch_attempted"] is True
    assert receipt["budget_state"] == "settled"
    timed_out_stage = receipt["stages"][0]
    assert timed_out_stage["stderr_head"].startswith("TRACEBACK START")
    assert timed_out_stage["stderr_tail"].endswith("TIMEOUT ROOT CAUSE")
    assert len(timed_out_stage["stderr_head"]) <= 2000
    assert len(timed_out_stage["stderr_tail"]) <= 2000
    timeout_receipt = json.loads(
        (tmp_path / "output" / "execution-timeout-receipt.json").read_text()
    )
    assert timeout_receipt["state"] == "timed_out"
    if os.name != "nt":
        assert runtime_tmp_aliases
        assert all(not path.exists() and not path.is_symlink() for path in runtime_tmp_aliases)


def test_protocol_v3_cancellation_after_launch_attempt_is_settled(tmp_path: Path):
    manifest, manager = _protocol_v3_fixture(tmp_path)
    cancelled = False
    calls = 0
    runtime_tmp_aliases: list[Path] = []

    def cancel_after_start(command, **kwargs):
        nonlocal cancelled, calls
        calls += 1
        runtime_tmp_aliases.append(Path(kwargs["env"]["TMPDIR"]))
        cancelled = True
        return subprocess.CompletedProcess(command, 0, "", "")

    with pytest.raises(ModelRunnerError, match="MODEL_RUN_CANCELLED"):
        execute_model_run(
            manifest,
            lease_manager=manager,
            process_runner=cancel_after_start,
            cancel_requested=lambda: cancelled,
        )

    receipt = json.loads((tmp_path / "output" / "execution-receipt.json").read_text())
    assert calls == 1
    assert receipt["state"] == "cancelled"
    assert receipt["launch_attempted"] is True
    assert receipt["budget_state"] == "settled"
    assert receipt["stages"][0]["returncode"] == 0
    if os.name != "nt":
        assert runtime_tmp_aliases
        assert all(not path.exists() and not path.is_symlink() for path in runtime_tmp_aliases)
