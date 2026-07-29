#!/usr/bin/env python3
"""Build the pre-registered ACWM training-seed checkpoint stability queue."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Mapping, Sequence

from scripts.export.acwm_autoloop_queue import _queue_row, _record_primitive_parameters


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/experiments/acwm_cloth_self_forcing_train_seed_replication_v1.json"
DEFAULT_OUT = ROOT / "results/reports/acwm-cloth-self-forcing-training-seed-stability-queue-v1"
DEFAULT_REPORT_ROOT = ROOT / "results/reports"
DEFAULT_LIMITED_GATE = ROOT / "results/reports/limited-campaign-gate-8env-official-current-warning-r1/manifest.json"
DEFAULT_FAILURE_MANIFEST = ROOT / "results/reports/m1-raw-failure-reports-ladder-r1/manifest.json"
DEFAULT_GOAL = ROOT / "configs/goal/g1_long_horizon_ladder_v1.yaml"
DEFAULT_RUNTIME_PYTHON = Path(os.environ.get("VERDIWM_RUNTIME_PYTHON", sys.executable))
DEFAULT_DATA_ROOT = Path(os.environ.get("ACWM_DATA_ROOT", "data/ACWM-Phys"))
DEFAULT_CHECKPOINT_ROOT = Path(os.environ.get("ACWM_CHECKPOINT_ROOT", "checkpoints/ACWM-Phys"))
DEFAULT_DATASET_FREEZE = ROOT / "runs/m0/protocol/dataset-freeze.json"
DEFAULT_HELDOUT_PROTOCOL = ROOT / "runs/m0/protocol/heldout-protocol.json"
DEFAULT_ARCHIVE_DB = ROOT / "results/archive.db"
DEFAULT_CAS_ROOT = ROOT / "results"


class AcwmTrainingSeedStabilityQueueError(RuntimeError):
    """The checkpoint stability queue contract is invalid."""


def build_stability_queue(
    *,
    experiment_config: Path,
    output_root: Path,
    repo_root: Path = ROOT,
    report_root: Path = DEFAULT_REPORT_ROOT,
    limited_gate: Path = DEFAULT_LIMITED_GATE,
    failure_manifest: Path = DEFAULT_FAILURE_MANIFEST,
    goal_config: Path = DEFAULT_GOAL,
    runtime_python: Path = DEFAULT_RUNTIME_PYTHON,
    data_root: Path = DEFAULT_DATA_ROOT,
    checkpoint_root: Path = DEFAULT_CHECKPOINT_ROOT,
    dataset_freeze: Path = DEFAULT_DATASET_FREEZE,
    heldout_protocol: Path = DEFAULT_HELDOUT_PROTOCOL,
    archive_db: Path = DEFAULT_ARCHIVE_DB,
    cas_root: Path = DEFAULT_CAS_ROOT,
    candidate_gpus: Sequence[int] = (0, 1, 2),
    train_batch_size: int = 8,
    screen_campaign_suffix: str = "r1-retry2",
    stability_revision: str = "r1",
) -> dict[str, object]:
    config_path = Path(experiment_config).resolve(strict=True)
    config = _load_json(config_path)
    if config.get("artifact_type") != "wmloop-acwm-targeted-gap-plan" or config.get("state") != "ready":
        raise AcwmTrainingSeedStabilityQueueError("ACWM_TRAINING_SEED_CONFIG_INVALID")
    contract = _mapping(config.get("training_seed_contract"), "ACWM_TRAINING_SEED_CONTRACT_INVALID")
    training_seeds = _seeds(contract.get("seeds"), "ACWM_TRAINING_SEEDS_INVALID")
    eval_seeds = _seeds(contract.get("evaluation_seeds"), "ACWM_EVAL_SEEDS_INVALID")
    screen_steps = _positive_int(contract.get("screen_steps"), "ACWM_SCREEN_STEPS_INVALID")
    confirmation_steps = _positive_int(
        contract.get("confirmation_cap_steps"), "ACWM_CONFIRMATION_STEPS_INVALID"
    )
    if confirmation_steps <= screen_steps:
        raise AcwmTrainingSeedStabilityQueueError("ACWM_STABILITY_LADDER_INVALID")
    checkpoint_steps = [step for step in (800, confirmation_steps) if screen_steps < step <= confirmation_steps]
    if not checkpoint_steps or checkpoint_steps[-1] != confirmation_steps:
        raise AcwmTrainingSeedStabilityQueueError("ACWM_STABILITY_LADDER_INVALID")
    records = config.get("environment_records")
    if not isinstance(records, list) or len(records) != 1 or not isinstance(records[0], Mapping):
        raise AcwmTrainingSeedStabilityQueueError("ACWM_TRAINING_SEED_TARGET_INVALID")
    target = records[0]
    environment = str(target.get("environment") or "")
    primitives = target.get("recommended_existing_primitives")
    if not environment or not isinstance(primitives, list) or len(primitives) != 1:
        raise AcwmTrainingSeedStabilityQueueError("ACWM_TRAINING_SEED_TARGET_INVALID")
    primitive = str(primitives[0])
    if not primitive or not candidate_gpus or train_batch_size < 1:
        raise AcwmTrainingSeedStabilityQueueError("ACWM_TRAINING_SEED_RUNTIME_INVALID")

    repo = Path(repo_root).resolve()
    reports = Path(report_root).resolve()
    parameters = _record_primitive_parameters(target, primitive)
    rows: list[dict[str, object]] = []
    confirmation_roots: dict[int, Path] = {}
    rank = 1
    for training_seed in training_seeds:
        quality_manifests = [
            reports
            / (
                f"acwm-formal-trainseed-gate-{environment}-{primitive}-"
                f"ts{training_seed}-es{eval_seed}-r1"
            )
            / "manifest.json"
            for eval_seed in eval_seeds
        ]
        campaign_id = (
            f"acwm-trainseed-stability-confirm-{environment}-{primitive}-"
            f"ts{training_seed}-t{confirmation_steps}-{stability_revision}"
        )
        row = _queue_row(
            rank=rank,
            phase="confirm_staged",
            environment=environment,
            primitive=primitive,
            seed=training_seed,
            train_steps=confirmation_steps,
            train_batch_size=train_batch_size,
            campaign_id=campaign_id,
            report_root=reports,
            repo_root=repo,
            limited_gate=Path(limited_gate).resolve(),
            failure_manifest=Path(failure_manifest).resolve(),
            goal_config=Path(goal_config).resolve(),
            runtime_python=Path(runtime_python).resolve(),
            data_root=Path(data_root).resolve(),
            checkpoint_root=Path(checkpoint_root).resolve(),
            dataset_freeze=Path(dataset_freeze).resolve(),
            heldout_protocol=Path(heldout_protocol).resolve(),
            archive_db=Path(archive_db).resolve(),
            cas_root=Path(cas_root).resolve(),
            gpus=candidate_gpus,
            primitive_parameters=parameters,
            proposal_routing_plan=config_path,
            requires_official_quality_manifest=str(quality_manifests[0]),
        )
        row["training_seed"] = training_seed
        row["requires_official_quality_manifests"] = [str(path) for path in quality_manifests]
        row["source_screen_manifest"] = str(
            reports
            / f"acwm-autoloop-screen-{environment}-{primitive}-s{training_seed}-t{screen_steps}-{screen_campaign_suffix}"
            / "envs"
            / environment
            / "manifest.json"
        )
        row["checkpoint_ladder_steps"] = [screen_steps, *checkpoint_steps]
        rows.append(row)
        confirmation_roots[training_seed] = Path(str(row["output_root"]))
        rank += 1

    for checkpoint_step in checkpoint_steps:
        for training_seed in training_seeds:
            confirmation_root = confirmation_roots[training_seed]
            for eval_seed in eval_seeds:
                rows.append(
                    _official_gate_row(
                        rank=rank,
                        environment=environment,
                        primitive=primitive,
                        training_seed=training_seed,
                        eval_seed=eval_seed,
                        checkpoint_step=checkpoint_step,
                        confirmation_root=confirmation_root,
                        report_root=reports,
                        repo_root=repo,
                        runtime_python=Path(runtime_python).resolve(),
                        data_root=Path(data_root).resolve(),
                        checkpoint_root=Path(checkpoint_root).resolve(),
                        dataset_freeze=Path(dataset_freeze).resolve(),
                        heldout_protocol=Path(heldout_protocol).resolve(),
                        archive_db=Path(archive_db).resolve(),
                        cas_root=Path(cas_root).resolve(),
                        candidate_gpus=candidate_gpus,
                        revision=stability_revision,
                    )
                )
                rank += 1

    payload = {
        "schema_version": 1,
        "artifact_type": "verdiwm-acwm-training-seed-stability-queue",
        "state": "ready",
        "experiment_config": str(config_path),
        "experiment_id": config.get("experiment_id"),
        "environment": environment,
        "primitive": primitive,
        "training_seeds": training_seeds,
        "eval_seeds": eval_seeds,
        "screen_steps": screen_steps,
        "confirmation_steps": confirmation_steps,
        "checkpoint_ladder_steps": [screen_steps, *checkpoint_steps],
        "training_row_count": len(training_seeds),
        "official_gate_row_count": len(training_seeds) * len(eval_seeds) * len(checkpoint_steps),
        "row_count": len(rows),
        "rows": rows,
        "claim_boundary": (
            "This queue tests post-512 repair checkpoint stability for independent repair-training seeds. "
            "It does not establish independent base-model pretraining or cross-backbone transfer."
        ),
    }
    return _write_bundle(Path(output_root).resolve(), payload)


def _official_gate_row(
    *,
    rank: int,
    environment: str,
    primitive: str,
    training_seed: int,
    eval_seed: int,
    checkpoint_step: int,
    confirmation_root: Path,
    report_root: Path,
    repo_root: Path,
    runtime_python: Path,
    data_root: Path,
    checkpoint_root: Path,
    dataset_freeze: Path,
    heldout_protocol: Path,
    archive_db: Path,
    cas_root: Path,
    candidate_gpus: Sequence[int],
    revision: str,
) -> dict[str, object]:
    campaign_id = (
        f"acwm-trainseed-stability-gate-{environment}-{primitive}-"
        f"ts{training_seed}-es{eval_seed}-step{checkpoint_step}-{revision}"
    )
    output_root = report_root / campaign_id
    retained_root = confirmation_root / "envs" / environment / "retained_training"
    ready_manifest = confirmation_root / "envs" / environment / "manifest.json"
    candidate_checkpoint = retained_root / "checkpoints" / f"relative_step_{checkpoint_step:06d}.pt"
    candidate_runtime = confirmation_root / "envs" / environment / "retained_runtime"
    return {
        "rank": rank,
        "phase": "confirm_official_eval_gate",
        "campaign_id": campaign_id,
        "environment": environment,
        "primitive": primitive,
        "seed": eval_seed,
        "training_seed": training_seed,
        "eval_seed": eval_seed,
        "checkpoint_step": checkpoint_step,
        "train_steps": 0,
        "resource_class": "gpu",
        "output_root": str(output_root),
        "candidate_gpus": [int(gpu) for gpu in candidate_gpus],
        "allow_any_idle_gpu": True,
        "requires_ready_manifest": str(ready_manifest),
        "archive_db": str(archive_db),
        "cas_root": str(cas_root),
        "gpu_audit_root_template": str(
            report_root / f"gpu-exclusivity-audit-{campaign_id}-gpu{{gpu}}-{{attempt_id}}"
        ),
        "launch_argv_template": [
            str(repo_root / ".venv/bin/python3"),
            "-m",
            "scripts.export.acwm_formal_visualization",
            "--output-root",
            str(output_root),
            "--environment",
            environment,
            "--primitive",
            primitive,
            "--seed",
            str(eval_seed),
            "--training-seed",
            str(training_seed),
            "--runtime-python",
            str(runtime_python),
            "--data-root",
            str(data_root),
            "--checkpoint-root",
            str(checkpoint_root),
            "--dataset-freeze",
            str(dataset_freeze),
            "--heldout-protocol",
            str(heldout_protocol),
            "--candidate-checkpoint",
            str(candidate_checkpoint),
            "--candidate-runtime-root",
            str(candidate_runtime),
            "--gpu-index",
            "{gpu}",
            "--steps",
            "50",
            "--split",
            "ind_test",
            "--max-trajs",
            "3",
            "--max-saved-vids",
            "1",
            "--batch-size",
            "1",
            "--num-workers",
            "2",
            "--test-cuts",
            "1",
            "--hard-case-top-k",
            "1",
        ],
    }


def _write_bundle(destination: Path, payload: dict[str, object]) -> dict[str, object]:
    if destination.exists() or destination.is_symlink():
        raise AcwmTrainingSeedStabilityQueueError("ACWM_STABILITY_QUEUE_OUTPUT_EXISTS")
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    try:
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary.mkdir(mode=0o700)
        _write_json(temporary / "queue.json", payload)
        fields = [
            "rank",
            "phase",
            "campaign_id",
            "training_seed",
            "eval_seed",
            "checkpoint_step",
            "train_steps",
            "output_root",
        ]
        with (temporary / "queue.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in payload["rows"]:
                assert isinstance(row, Mapping)
                writer.writerow({key: row.get(key, "") for key in fields})
        manifest = {
            "schema_version": 1,
            "artifact_type": "verdiwm-acwm-training-seed-stability-queue-manifest",
            "state": "ready",
            "queue_path": str(destination / "queue.json"),
            "row_count": payload["row_count"],
            "training_row_count": payload["training_row_count"],
            "official_gate_row_count": payload["official_gate_row_count"],
            "checkpoint_ladder_steps": payload["checkpoint_ladder_steps"],
        }
        _write_json(temporary / "manifest.json", manifest)
        os.replace(temporary, destination)
        return manifest
    except Exception:
        if temporary.is_symlink():
            temporary.unlink()
        elif temporary.exists():
            shutil.rmtree(temporary)
        raise


def _mapping(value: object, error: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise AcwmTrainingSeedStabilityQueueError(error)
    return value


def _seeds(value: object, error: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise AcwmTrainingSeedStabilityQueueError(error)
    seeds: list[int] = []
    for raw in value:
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 1 or raw in seeds:
            raise AcwmTrainingSeedStabilityQueueError(error)
        seeds.append(raw)
    return seeds


def _positive_int(value: object, error: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise AcwmTrainingSeedStabilityQueueError(error)
    return value


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AcwmTrainingSeedStabilityQueueError(f"ACWM_TRAINING_SEED_JSON_INVALID:{path}")
    return payload


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--runtime-python", type=Path, default=DEFAULT_RUNTIME_PYTHON)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--candidate-gpus", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--train-batch-size", type=int, default=8)
    args = parser.parse_args(argv)
    manifest = build_stability_queue(
        experiment_config=args.experiment_config,
        output_root=args.output_root,
        repo_root=args.repo_root,
        report_root=args.report_root,
        runtime_python=args.runtime_python,
        data_root=args.data_root,
        checkpoint_root=args.checkpoint_root,
        candidate_gpus=args.candidate_gpus,
        train_batch_size=args.train_batch_size,
    )
    print(json.dumps(manifest, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
