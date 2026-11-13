"""Bounded inference for existing Wan arm-slot checkpoints, without retraining.

The external SA-WM rollout is the model interface. This runner does not inject
the separate 7D-action/14D-proprio training adapter into a 10D arm-slot model.
Execution receipts are diagnostic evidence; only verify owns promotion.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Sequence

import yaml

from wmloop.archive.store import ContentAddressedStore
from wmloop.contracts import validate_document
from wmloop.control.onboarding import compute_source_tree_revision
from wmloop.diagnose.wan_native_evidence import wan_invocation_matches, wan_manifest_errors
from wmloop.execute.budget import BudgetLedger, BudgetPolicy
from wmloop.execute.gpu_lease import GpuLeaseManager
from wmloop.execute.open_method_runtime import resolve_runtime_python, run_local_command
from wmloop.storage import (
    atomic_write, canonical_bytes, checked_path, exclusive_file_lock,
    sha256_bytes, sha256_file,
)


class WanArmSlotError(ValueError):
    """An incompatible or unbounded inference request cannot run."""


def _path(value: Path, *, directory: bool = False) -> Path:
    result = checked_path(value, code="WAN_ARM_SLOT_PATH_INVALID", error=WanArmSlotError)
    if not (result.is_dir() if directory else result.is_file()):
        raise WanArmSlotError("WAN_ARM_SLOT_INPUT_MISSING")
    return result


def _positive_integer(value: int, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise WanArmSlotError(f"WAN_ARM_SLOT_{name}_INVALID")


def _data_binding(root: Path) -> dict[str, str]:
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise WanArmSlotError("WAN_ARM_SLOT_DATA_SYMLINK")
        if path.is_file():
            result[path.relative_to(root).as_posix()] = sha256_file(path)
    if not result:
        raise WanArmSlotError("WAN_ARM_SLOT_DATA_EMPTY")
    return result


def _model_asset_paths(world: dict, source: Path) -> tuple[Path, ...]:
    paths = []
    for key in ("checkpoint", "vae_checkpoint"):
        value = world.get(key)
        if not isinstance(value, str) or not value.strip():
            raise WanArmSlotError("WAN_ARM_SLOT_MODEL_ASSET_REQUIRED")
        candidate = Path(value).expanduser()
        path = checked_path(candidate if candidate.is_absolute() else source / candidate,
                            code="WAN_ARM_SLOT_MODEL_ASSET_INVALID", error=WanArmSlotError)
        if not (path.is_file() or path.is_dir()):
            raise WanArmSlotError("WAN_ARM_SLOT_MODEL_ASSET_MISSING")
        paths.append(path)
    return tuple(paths)


def _model_asset_binding(paths: tuple[Path, ...]) -> dict[str, str]:
    bindings = {}
    for path in paths:
        if path.is_dir():
            bindings.update({str(path / relative): digest for relative, digest in _data_binding(path).items()})
        elif str(path) not in bindings:
            bindings[str(path)] = sha256_file(path)
    return bindings


def _output_errors(root: Path, binding: dict, selected: list[dict]) -> list[str]:
    """Exit zero needs the native manifest and all three nonempty videos."""
    try:
        manifest = json.loads(_path(root / "manifest.json").read_bytes())
        if not isinstance(manifest, dict):
            raise WanArmSlotError("WAN_ARM_SLOT_MANIFEST_INVALID")
        errors = wan_manifest_errors(manifest, binding, selected)
        if errors:
            return errors
        for row in manifest["samples"]:
            sample_dir = checked_path(root / row["dir"], code="WAN_ARM_SLOT_VIDEO_PATH_INVALID", error=WanArmSlotError)
            if root not in sample_dir.parents:
                raise WanArmSlotError("WAN_ARM_SLOT_VIDEO_PATH_INVALID")
            for key in ("gt_video", "sa_wm_video", "comparison_video"):
                name = row[key + "_file"]
                if not isinstance(name, str) or Path(name).name != name:
                    raise WanArmSlotError("WAN_ARM_SLOT_VIDEO_PATH_INVALID")
                video = _path(sample_dir / name)
                if str(video) != row[key] or video.stat().st_size == 0:
                    raise WanArmSlotError("WAN_ARM_SLOT_VIDEO_MISSING")
    except (OSError, ValueError, KeyError, TypeError):
        return ["WAN_ARM_SLOT_OUTPUT_CONTRACT_FAILED"]
    return []


def inference_inputs_unchanged(binding: dict) -> bool:
    """Recheck the existing inference binding, including all model assets."""
    try:
        source = _path(Path(binding["source_root"]), directory=True)
        config = _path(Path(binding["config"]))
        world = yaml.safe_load(config.read_text(encoding="utf-8"))["world_model"]
        override = binding["condition_override_root"]
        return (
            sha256_file(config) == binding["config_sha256"]
            and sha256_file(_path(Path(binding["selection"]))) == binding["selection_sha256"]
            and sha256_file(_path(Path(binding["checkpoint"]) / "dit_model.safetensors")) == binding["checkpoint_sha256"]
            and _model_asset_binding(_model_asset_paths(world, source)) == binding["model_asset_sha256"]
            and sha256_file(_path(source / "scripts/evaluation/rollout_full_episode_sa_wm.py")) == binding["entrypoint_sha256"]
            and compute_source_tree_revision(source) == binding["source_tree_revision"]
            and _data_binding(_path(Path(binding["data_root"]), directory=True)) == binding["data_sha256"]
            and (_data_binding(_path(Path(override), directory=True)) if override else None) == binding["condition_override_sha256"]
        )
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError):
        return False


def build_native_execution_evidence(
    *, output_root: Path, input_binding: dict, selection: list[dict], process: dict,
) -> dict[str, object]:
    """Validate already admitted native work, without acquiring or settling it.

    The enclosing auto-experiment owns the lease, budget, CAS and terminal
    receipt. This is an execution artifact, with no settlement authority.
    """
    output = _path(output_root, directory=True)
    errors = _output_errors(output / "rollouts", input_binding, selection)
    unchanged = inference_inputs_unchanged(input_binding)
    if not unchanged:
        errors.append("WAN_ARM_SLOT_INPUT_CHANGED")
    if process.get("state") != "passed" or type(process.get("returncode")) is not int or process["returncode"] != 0:
        errors.append("WAN_ARM_SLOT_PROCESS_FAILED")
    if not wan_invocation_matches(process, input_binding):
        errors.append("WAN_ARM_SLOT_INVOCATION_BINDING_MISMATCH")
    artifacts = {}
    for path in sorted((output / "rollouts").rglob("*")):
        checked_path(path, code="WAN_ARM_SLOT_OUTPUT_SYMLINK", error=WanArmSlotError)
        if path.is_file():
            artifacts[path.relative_to(output).as_posix()] = "cas://sha256/" + sha256_file(path)
    evidence = {
        "schema_version": 1, "artifact_type": "verdiwm-wan-arm-slot-native-execution",
        "state": "failed" if errors else "passed", "input_binding": input_binding,
        "process": process, "artifacts": artifacts,
        "validation": {"input_unchanged": unchanged, "errors": errors},
        "authority": {"diagnostic_only": True, "promotion": False},
        "claim_boundary": "Native execution evidence only; the enclosing auto-experiment owns settlement.",
    }
    validate_document("wan_arm_slot_native_execution", evidence)
    return evidence


def run_inference(
    *, source_root: Path, checkpoint: Path, config: Path, selection: Path,
    data_root: Path, runtime_python: Path, output_root: Path, budget_db: Path,
    budget_gpu_hours: float, gpu_indices: Sequence[int], seed: int,
    inference_steps: int = 25, max_chunks: int = 1, limit: int = 1,
    split: str = "val", timeout_seconds: float = 1800.0,
    condition_override_root: Path | None = None,
) -> dict[str, object]:
    """Run one bounded target inference with a lease, budget debit and CAS logs.

    Use the same budget database and total across baseline, probes and candidate
    runs. Timeout is reserved before launch; actual elapsed GPU time is settled
    even for failed model processes. A new output root is required per run.
    """
    for value, name in ((inference_steps, "STEPS"), (max_chunks, "CHUNKS"), (limit, "LIMIT")):
        _positive_integer(value, name)
    if type(seed) is not int or seed < 0 or split not in {"train", "val", "test"}:
        raise WanArmSlotError("WAN_ARM_SLOT_SEED_OR_SPLIT_INVALID")
    if type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise WanArmSlotError("WAN_ARM_SLOT_TIMEOUT_INVALID")
    if type(budget_gpu_hours) not in (int, float) or not math.isfinite(budget_gpu_hours) or budget_gpu_hours <= 0:
        raise WanArmSlotError("WAN_ARM_SLOT_BUDGET_INVALID")
    source = _path(source_root, directory=True)
    model = _path(checkpoint, directory=True)
    data = _path(data_root, directory=True)
    config_path, selection_path = _path(config), _path(selection)
    weights = _path(model / "dit_model.safetensors")
    script = _path(source / "scripts/evaluation/rollout_full_episode_sa_wm.py")
    runtime = resolve_runtime_python(runtime_python)
    output = checked_path(output_root, code="WAN_ARM_SLOT_OUTPUT_INVALID", error=WanArmSlotError)
    ledger_path = checked_path(budget_db, code="WAN_ARM_SLOT_BUDGET_PATH_INVALID", error=WanArmSlotError)
    override = _path(condition_override_root, directory=True) if condition_override_root else None
    if ledger_path == output or output in ledger_path.parents or ledger_path in output.parents:
        raise WanArmSlotError("WAN_ARM_SLOT_BUDGET_OUTPUT_OVERLAP")

    # Native model configurations use full YAML, including PyYAML's default
    # indentless sequences. The checked-in contract subset is not their format.
    try:
        document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise WanArmSlotError("WAN_ARM_SLOT_CONFIG_INVALID") from exc
    if not isinstance(document, dict):
        raise WanArmSlotError("WAN_ARM_SLOT_CONFIG_INVALID")
    world = document.get("world_model", {})
    training = document.get("wm_training", {})
    if (not isinstance(world, dict) or not isinstance(training, dict)
        or world.get("backbone") != "wan2.2" or world.get("action_schema") != "arm_slot"
        or type(world.get("action_dim")) is not int or world["action_dim"] != 10
        or type(world.get("max_arm_slots")) is not int or world["max_arm_slots"] != 2
        or training.get("video_latent_condition_view") != "arm_slot_eef_pose"
        or training.get("action_condition_timestep_rate") != "raw"):
        raise WanArmSlotError("WAN_ARM_SLOT_CONFIG_INCOMPATIBLE")
    asset_paths = _model_asset_paths(world, source)
    inputs = (source, model, data, config_path, selection_path, *asset_paths,
              *((override,) if override else ()))
    for writable in (output, ledger_path):
        if any(writable == path or path in writable.parents or writable in path.parents for path in inputs):
            raise WanArmSlotError("WAN_ARM_SLOT_OUTPUT_OVERLAPS_INPUT")
    try:
        selected = json.loads(selection_path.read_bytes())["selected"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise WanArmSlotError("WAN_ARM_SLOT_SELECTION_INVALID") from exc
    if not isinstance(selected, list) or not selected or any(not isinstance(row, dict) for row in selected):
        raise WanArmSlotError("WAN_ARM_SLOT_SELECTION_INVALID")
    active = selected[:limit]
    sample_indices = [row.get("sample_index", index) for index, row in enumerate(active)]
    if (any(type(row.get("episode_id")) is not int or row["episode_id"] < 0
            or type(row.get("frame_now")) is not int or row["frame_now"] < 0 for row in active)
        or any(type(index) is not int or index < 0 for index in sample_indices)
        or len(set(sample_indices)) != len(sample_indices)):
        raise WanArmSlotError("WAN_ARM_SLOT_SELECTION_INVALID")
    fps = training.get("video_latent_required_target_fps")
    if type(fps) not in (int, float) or not math.isfinite(fps) or fps <= 0:
        raise WanArmSlotError("WAN_ARM_SLOT_FPS_INVALID")
    source_revision = compute_source_tree_revision(source)
    if source_revision.get("state") != "bound":
        raise WanArmSlotError("WAN_ARM_SLOT_SOURCE_REVISION_UNBOUND")

    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with exclusive_file_lock(output.parent / f".{output.name}.lock"):
        if output.exists():
            raise WanArmSlotError("WAN_ARM_SLOT_OUTPUT_EXISTS")
        binding = {
            "config": str(config_path), "selection": str(selection_path),
            "config_sha256": sha256_file(config_path),
            "selection_sha256": sha256_file(selection_path),
            "checkpoint_sha256": sha256_file(weights),
            "model_asset_sha256": _model_asset_binding(asset_paths),
            "entrypoint_sha256": sha256_file(script),
            "source_tree_revision": source_revision,
            "source_root": str(source), "checkpoint": str(model),
            "data_root": str(data), "runtime_python": str(runtime),
            "condition_override_root": str(override) if override else None,
            "data_sha256": _data_binding(data),
            "condition_override_sha256": _data_binding(override) if override else None,
            "seed": seed, "split": split, "inference_steps": inference_steps,
            "max_chunks": max_chunks, "limit": limit, "fps": fps,
            "timeout_seconds": timeout_seconds, "output_root": str(output),
        }
        identity = sha256_bytes(canonical_bytes(binding))
        trial_id = "wan-arm-slot-" + identity[:24]
        ledger = BudgetLedger(ledger_path, BudgetPolicy(total_gpu_hours=budget_gpu_hours))
        reservation = ledger.admit(trial_id, cost_class="low", estimated_gpu_hours=timeout_seconds / 3600.0,
                                   plan_sha256=identity, reject_existing=True)
        launched = False
        output_created = False
        try:
            with GpuLeaseManager().acquire(gpu_indices) as lease:
                output.mkdir(mode=0o700, exist_ok=False)
                output_created = True
                atomic_write(output / "input-binding.json", canonical_bytes(binding) + b"\n")
                command = [
                    str(runtime), str(script), "--config", str(config_path),
                    "--checkpoint", str(model), "--selection", str(selection_path),
                    "--video_latent_root", str(data), "--split", split,
                    "--output_dir", str(output / "rollouts"), "--device", "cuda:0",
                    "--inference_steps", str(inference_steps), "--seed", str(seed),
                    "--fps", str(fps), "--limit", str(limit), "--max_chunks", str(max_chunks),
                    "--decode_chunk_latents", "8",
                ]
                if override is not None:
                    command += ["--condition_override_root", str(override)]
                launched = True
                process = run_local_command(command=command, cwd=source, output_root=output / "process",
                    timeout_seconds=timeout_seconds, runtime_python=runtime, gpu_devices=(lease.index,))
                gpu = lease.to_document()
        except BaseException:
            if not launched:
                ledger.release(trial_id, fencing_token=reservation.fencing_token)
                if output_created:
                    shutil.rmtree(output)
            # A started attempt retains its reservation if interrupted before a
            # durable process receipt. Never refund work whose cost is unknown.
            raise

        errors = _output_errors(output / "rollouts", binding, selected) if process["state"] == "passed" else []
        unchanged = inference_inputs_unchanged(binding)
        if not unchanged:
            errors.append("WAN_ARM_SLOT_INPUT_CHANGED")
        cas = ContentAddressedStore(output / "evidence")
        refs = {}
        for path in sorted(output.rglob("*")):
            if output / "evidence" in path.parents:
                continue
            if path.is_symlink():
                errors.append("WAN_ARM_SLOT_OUTPUT_SYMLINK")
                continue
            if path.is_file():
                refs[path.relative_to(output).as_posix()] = cas.put_file(path, media_type="application/octet-stream").uri
        receipt = {
            "schema_version": 1, "artifact_type": "verdiwm-wan-arm-slot-inference-receipt",
            "trial_id": trial_id, "state": "failed" if errors else process["state"], "input_binding": binding,
            "process": process, "gpu": gpu, "artifacts": refs,
            "validation": {"input_unchanged": unchanged, "errors": errors},
            "actual_gpu_hours": float(process["duration_seconds"]) / 3600.0,
            "authority": {"diagnostic_only": True, "promotion": False},
            "claim_boundary": "Bounded real-model inference only; no formal improvement or transfer verdict.",
        }
        validate_document("wan_arm_slot_inference_receipt", receipt)
        receipt_bytes = canonical_bytes(receipt) + b"\n"
        receipt_ref = cas.put_bytes(receipt_bytes, media_type="application/json")
        atomic_write(output / "receipt.json", receipt_bytes)
        ledger.settle(trial_id, fencing_token=reservation.fencing_token,
                      actual_gpu_hours=receipt["actual_gpu_hours"], receipt_ref=receipt_ref.uri)
        return {**receipt, "receipt_ref": receipt_ref.uri, "receipt_path": str(output / "receipt.json")}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "checkpoint", "config", "selection", "data-root",
                 "runtime-python", "output-root", "budget-db"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--budget-gpu-hours", type=float, required=True)
    parser.add_argument("--gpu", type=int, action="append", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--inference-steps", type=int, default=25)
    parser.add_argument("--max-chunks", type=int, default=1)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--condition-override-root", type=Path)
    options = vars(parser.parse_args(argv))
    options["gpu_indices"] = options.pop("gpu")
    result = run_inference(**options)
    # Existing control-plane stdout JSON protocol; progress stays in logs.
    from wmloop.control.commands.io import emit_json
    emit_json(result)
    return 0 if result["state"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
