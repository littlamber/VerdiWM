"""Training and frozen evaluation smoke components: runtime."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Mapping, Sequence

from wmloop.execute.training_smoke.errors import OrchestratorTrainingEvalSmokeError
from wmloop.execute.training_smoke.io import (
    _canonical_json_bytes,
    _load_json_object,
    _write_bytes_atomic,
)
from wmloop.execute.training_smoke.metrics import _positive_int
from wmloop.runtime_contract import runtime_tree_sha256


def _training_entrypoint_command(
    *,
    runtime: Path,
    worktree: Path,
    run_root: Path,
    config_path: Path,
    checkpoint_path: Path,
) -> tuple[str, ...]:
    wrapper = (
        "import os, runpy, sys; "
        "train_path, run_root = sys.argv[1], sys.argv[2]; "
        "os.chdir(run_root); "
        "sys.argv = [train_path, *sys.argv[3:]]; "
        "namespace = runpy.run_path(train_path, run_name='__main__'); "
        "entrypoint = namespace.get('main'); "
        "state = getattr(entrypoint, '__globals__', namespace); "
        "step = int(state.get('_step', 0)); "
        "checkpoint_dir = state.get('_ckpt_dir', ''); "
        "save = state.get('save_checkpoint'); "
        "model = state.get('_model'); "
        "optimizer = state.get('_optimizer'); "
        "assert step > 0 and checkpoint_dir and callable(save) and model is not None and optimizer is not None; "
        "final_path = os.path.join(checkpoint_dir, f'checkpoint_{step}.pt'); "
        "save(model, optimizer, step, state.get('_epoch', 0), final_path, "
        "wandb_run_id=state.get('_wandb_run_id'))"
    )
    return (
        str(runtime),
        "-c",
        wrapper,
        str(Path(worktree) / "train.py"),
        str(run_root),
        "--config",
        str(config_path),
        "--ckpt_path",
        str(checkpoint_path),
    )


def _copy_training_sidecar(*, source: Path, run_root: Path) -> Path:
    if source.is_symlink() or not source.is_file():
        raise OrchestratorTrainingEvalSmokeError(
            "M3_TRAINING_EVAL_SMOKE_SIDECAR_MISSING"
        )
    target = Path(run_root) / "wmloop_interventions" / source.name
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


def _copy_training_sidecars(
    *, source_root: Path, run_root: Path, primitive_names: Sequence[str]
) -> list[str]:
    copied: list[str] = []
    for primitive in dict.fromkeys(str(name) for name in primitive_names):
        if not primitive or "/" in primitive or "\\" in primitive:
            raise OrchestratorTrainingEvalSmokeError(
                "M3_TRAINING_EVAL_SMOKE_SIDECAR_INVALID"
            )
        target = _copy_training_sidecar(
            source=Path(source_root) / f"{primitive}.json", run_root=run_root
        )
        copied.append(str(target))
    if not copied:
        raise OrchestratorTrainingEvalSmokeError(
            "M3_TRAINING_EVAL_SMOKE_SIDECAR_MISSING"
        )
    return copied


def _copy_candidate_runtime(
    *,
    source: Path,
    destination: Path,
    source_revision: str,
    rendered_primitives: Sequence[Mapping[str, object]],
) -> dict[str, str]:
    source = Path(source)
    destination = Path(destination)
    if (
        source.is_symlink()
        or not source.is_dir()
        or destination.exists()
        or destination.is_symlink()
    ):
        raise OrchestratorTrainingEvalSmokeError(
            "M3_TRAINING_EVAL_SMOKE_CANDIDATE_RUNTIME_INVALID"
        )
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc", "*.pyo"),
    )
    tree_sha256 = runtime_tree_sha256(destination)
    manifest_path = destination / "wmloop-runtime-manifest.json"
    _write_bytes_atomic(
        manifest_path,
        _canonical_json_bytes(
            {
                "schema_version": 1,
                "artifact_type": "wmloop-materialized-candidate-runtime",
                "state": "ready",
                "source_revision": source_revision,
                "runtime_path": str(destination),
                "tree_sha256": tree_sha256,
                "rendered_primitives": [dict(item) for item in rendered_primitives],
            }
        ),
    )
    return {
        "runtime_path": str(destination),
        "manifest_path": str(manifest_path),
        "tree_sha256": tree_sha256,
    }


def _validate_probe_runtime_binding(
    *,
    baseline_runtime_sha256: str,
    candidate_runtime_sha256: str,
    baseline_manifest_paths: Sequence[Path],
    candidate_manifest_paths: Sequence[Path],
) -> dict[str, object]:
    if not baseline_manifest_paths or not candidate_manifest_paths:
        raise OrchestratorTrainingEvalSmokeError(
            "M3_TRAINING_EVAL_SMOKE_RUNTIME_BINDING_EMPTY"
        )
    records: list[dict[str, str]] = []
    for side, expected, paths in (
        ("baseline", baseline_runtime_sha256, baseline_manifest_paths),
        ("candidate", candidate_runtime_sha256, candidate_manifest_paths),
    ):
        for path in paths:
            manifest = _load_json_object(Path(path))
            observed = manifest.get("vendor_runtime_sha256")
            if observed != expected:
                raise OrchestratorTrainingEvalSmokeError(
                    f"M3_TRAINING_EVAL_SMOKE_RUNTIME_BINDING_MISMATCH:{side}:{path}"
                )
            records.append(
                {
                    "side": side,
                    "manifest_path": str(Path(path)),
                    "vendor_root": str(manifest.get("vendor_root") or ""),
                    "runtime_sha256": expected,
                }
            )
    return {
        "state": "ready",
        "baseline_runtime_sha256": baseline_runtime_sha256,
        "candidate_runtime_sha256": candidate_runtime_sha256,
        "records": records,
    }


def _horizon_probe_command(
    *,
    runtime: Path,
    repo_root: Path,
    vendor_root: Path | None = None,
    data_root: Path,
    checkpoint_root: Path,
    checkpoint_path: Path,
    environment: str = "push_cube",
    output_root: Path,
    horizons: Sequence[int],
    max_trajectories: int,
    inference_steps: int,
    gpu_index: int,
    gpu_exclusivity_audit_manifest: Path,
    gpu_exclusivity_max_age_seconds: float | None = 300.0,
    seed: int = 101,
) -> tuple[str, ...]:
    command = [
        str(runtime),
        "-m",
        "wmloop.diagnose.horizon_runtime",
        "run",
        "--repo-root",
        str(repo_root),
        "--data-root",
        str(data_root),
        "--checkpoint-root",
        str(checkpoint_root),
        "--checkpoint-path",
        str(checkpoint_path),
        "--environment",
        environment,
        "--split",
        "ind_test",
        "--output-root",
        str(output_root),
        "--horizons",
        *(str(value) for value in horizons),
        "--max-trajectories",
        str(max_trajectories),
        "--num-inference-steps",
        str(inference_steps),
        "--device",
        "cuda",
        "--seed",
        str(seed),
        "--mode",
        "autoregressive",
        "--max-evidence",
        "1",
        "--gpu-index",
        str(gpu_index),
        "--gpu-exclusivity-audit-manifest",
        str(gpu_exclusivity_audit_manifest),
    ]
    if gpu_exclusivity_max_age_seconds is not None:
        command.extend(
            ["--gpu-exclusivity-max-age-seconds", str(gpu_exclusivity_max_age_seconds)]
        )
    if vendor_root is not None:
        command.extend(["--vendor-root", str(vendor_root)])
    return tuple(command)


def _raw_probe_measure_command(
    *,
    runtime: Path,
    repo_root: Path,
    vendor_root: Path | None = None,
    data_root: Path,
    checkpoint_root: Path,
    checkpoint_path: Path,
    inverse_summary_path: Path,
    environment: str = "push_cube",
    output_root: Path,
    horizons: Sequence[int],
    primary_horizon: int,
    max_trajectories: int,
    inference_steps: int,
    gpu_index: int,
    gpu_exclusivity_audit_manifest: Path,
    gpu_exclusivity_max_age_seconds: float | None = 300.0,
    seed: int = 101,
    action_only: bool = False,
) -> tuple[str, ...]:
    command = [
        str(runtime),
        "-m",
        "wmloop.diagnose.raw_probe_measure_runtime",
        "run",
        "--repo-root",
        str(repo_root),
        "--data-root",
        str(data_root),
        "--checkpoint-root",
        str(checkpoint_root),
        "--checkpoint-path",
        str(checkpoint_path),
        "--inverse-summary",
        str(inverse_summary_path),
        "--environment",
        environment,
        "--output-root",
        str(output_root),
        "--horizons",
        *(str(value) for value in horizons),
        "--primary-horizon",
        str(primary_horizon),
        "--max-trajectories",
        str(max_trajectories),
        "--num-inference-steps",
        str(inference_steps),
        "--device",
        "cuda",
        "--seed",
        str(seed),
        "--mode",
        "autoregressive",
        "--gpu-index",
        str(gpu_index),
        "--gpu-exclusivity-audit-manifest",
        str(gpu_exclusivity_audit_manifest),
    ]
    if gpu_exclusivity_max_age_seconds is not None:
        command.extend(
            ["--gpu-exclusivity-max-age-seconds", str(gpu_exclusivity_max_age_seconds)]
        )
    if vendor_root is not None:
        command.extend(["--vendor-root", str(vendor_root)])
    if action_only:
        command.append("--action-only")
    return tuple(command)


def _checkpoint_step(*, runtime_python: Path, checkpoint_path: Path) -> int:
    if checkpoint_path.is_symlink() or not checkpoint_path.is_file():
        raise OrchestratorTrainingEvalSmokeError(
            "M3_TRAINING_EVAL_SMOKE_OFFICIAL_CHECKPOINT_MISSING"
        )
    code = "import json, sys, torch; p=torch.load(sys.argv[1], map_location='cpu', weights_only=False); print(json.dumps({'step': int(p.get('step', 0))}))"
    completed = subprocess.run(
        [str(runtime_python), "-c", code, str(checkpoint_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if completed.returncode != 0:
        raise OrchestratorTrainingEvalSmokeError(
            "M3_TRAINING_EVAL_SMOKE_CHECKPOINT_STEP_UNREADABLE"
        )
    payload = json.loads(completed.stdout)
    step = payload.get("step")
    if not isinstance(step, int) or step < 0:
        raise OrchestratorTrainingEvalSmokeError(
            "M3_TRAINING_EVAL_SMOKE_CHECKPOINT_STEP_INVALID"
        )
    return step


def _load_yaml_with_pyyaml(path: Path) -> dict[str, object]:
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        raise OrchestratorTrainingEvalSmokeError(
            "M3_TRAINING_EVAL_SMOKE_YAML_UNAVAILABLE"
        ) from exc
    try:
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:  # type: ignore[attr-defined]
        raise OrchestratorTrainingEvalSmokeError(
            "M3_TRAINING_EVAL_SMOKE_CONFIG_INVALID"
        ) from exc
    if not isinstance(payload, dict):
        raise OrchestratorTrainingEvalSmokeError(
            "M3_TRAINING_EVAL_SMOKE_CONFIG_INVALID"
        )
    return payload


def _safe_run_name(proposal_id: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_]+", "_", proposal_id).strip("_")
    if not name:
        raise OrchestratorTrainingEvalSmokeError(
            "M3_TRAINING_EVAL_SMOKE_RUN_NAME_INVALID"
        )
    return name[:96]


def _candidate_checkpoint_path(
    *, training_run_root: Path, run_name: str, training_config: Mapping[str, object]
) -> Path:
    return (
        Path(training_run_root)
        / "checkpoints"
        / f"{run_name}_{_checkpoint_resolution_suffix(training_config)}"
        / "latest.pt"
    )


def _checkpoint_resolution_suffix(training_config: Mapping[str, object]) -> str:
    dataset = training_config.get("dataset")
    if not isinstance(dataset, Mapping):
        raise OrchestratorTrainingEvalSmokeError(
            "M3_TRAINING_EVAL_SMOKE_TRAINING_CONFIG_INVALID"
        )
    obs_shape = dataset.get("obs_shape")
    if not isinstance(obs_shape, list) or len(obs_shape) != 3:
        raise OrchestratorTrainingEvalSmokeError(
            "M3_TRAINING_EVAL_SMOKE_TRAINING_CONFIG_INVALID"
        )
    height = _positive_int(obs_shape[1])
    width = _positive_int(obs_shape[2])
    return f"{height}x{width}"
