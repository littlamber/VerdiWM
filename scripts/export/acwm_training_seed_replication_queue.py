#!/usr/bin/env python3
"""Build the frozen ACWM repair-training seed replication gate queue."""

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


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/experiments/acwm_cloth_self_forcing_train_seed_replication_v1.json"
DEFAULT_REPORT_ROOT = ROOT / "results/reports"
DEFAULT_OUT = DEFAULT_REPORT_ROOT / "acwm-cloth-self-forcing-training-seed-factorial-queue-v1"
DEFAULT_RUNTIME_PYTHON = Path(os.environ.get("VERDIWM_RUNTIME_PYTHON", sys.executable))
DEFAULT_DATA_ROOT = Path(os.environ.get("ACWM_DATA_ROOT", "data/ACWM-Phys"))
DEFAULT_CHECKPOINT_ROOT = Path(os.environ.get("ACWM_CHECKPOINT_ROOT", "checkpoints/ACWM-Phys"))


class AcwmTrainingSeedQueueError(RuntimeError):
    """The factorial gate queue contract is invalid."""


def build_queue(
    *,
    experiment_config: Path,
    output_root: Path,
    repo_root: Path = ROOT,
    report_root: Path = DEFAULT_REPORT_ROOT,
    runtime_python: Path = DEFAULT_RUNTIME_PYTHON,
    data_root: Path = DEFAULT_DATA_ROOT,
    checkpoint_root: Path = DEFAULT_CHECKPOINT_ROOT,
    candidate_gpus: Sequence[int] = (0, 1, 2),
    screen_campaign_suffix: str = "r1-retry2",
) -> dict[str, object]:
    config_path = Path(experiment_config).resolve(strict=True)
    config = _load_json(config_path)
    if config.get("artifact_type") != "wmloop-acwm-targeted-gap-plan" or config.get("state") != "ready":
        raise AcwmTrainingSeedQueueError("ACWM_TRAINING_SEED_CONFIG_INVALID")
    contract = _mapping(config.get("training_seed_contract"), "ACWM_TRAINING_SEED_CONTRACT_INVALID")
    training_seeds = _seeds(contract.get("seeds"), "ACWM_TRAINING_SEEDS_INVALID")
    eval_seeds = _seeds(contract.get("evaluation_seeds"), "ACWM_EVAL_SEEDS_INVALID")
    if len(training_seeds) < 2 or len(eval_seeds) < 2:
        raise AcwmTrainingSeedQueueError("ACWM_FACTORIAL_REPLICATION_TOO_SMALL")
    records = config.get("environment_records")
    if not isinstance(records, list) or len(records) != 1 or not isinstance(records[0], Mapping):
        raise AcwmTrainingSeedQueueError("ACWM_TRAINING_SEED_TARGET_INVALID")
    target = records[0]
    environment = str(target.get("environment") or "")
    primitives = target.get("recommended_existing_primitives")
    if not environment or not isinstance(primitives, list) or len(primitives) != 1:
        raise AcwmTrainingSeedQueueError("ACWM_TRAINING_SEED_TARGET_INVALID")
    primitive = str(primitives[0])
    if not primitive or not candidate_gpus or any(int(gpu) < 0 for gpu in candidate_gpus):
        raise AcwmTrainingSeedQueueError("ACWM_TRAINING_SEED_RUNTIME_INVALID")

    repo = Path(repo_root).resolve()
    reports = Path(report_root).resolve()
    rows: list[dict[str, object]] = []
    rank = 1
    for training_seed in training_seeds:
        screen_campaign = (
            f"acwm-autoloop-screen-{environment}-{primitive}-s{training_seed}-t512-{screen_campaign_suffix}"
        )
        screen_root = reports / screen_campaign
        source_manifest = screen_root / "envs" / environment / "manifest.json"
        candidate_checkpoint = screen_root / "envs" / environment / "retained_training" / "latest.pt"
        candidate_runtime = screen_root / "envs" / environment / "retained_runtime"
        for eval_seed in eval_seeds:
            campaign_id = (
                f"acwm-formal-trainseed-gate-{environment}-{primitive}-"
                f"ts{training_seed}-es{eval_seed}-r1"
            )
            gate_root = reports / campaign_id
            rows.append(
                {
                    "rank": rank,
                    "phase": "official_eval_gate",
                    "campaign_id": campaign_id,
                    "environment": environment,
                    "primitive": primitive,
                    "seed": eval_seed,
                    "training_seed": training_seed,
                    "eval_seed": eval_seed,
                    "checkpoint_step": 512,
                    "train_steps": 0,
                    "resource_class": "gpu",
                    "output_root": str(gate_root),
                    "candidate_gpus": [int(gpu) for gpu in candidate_gpus],
                    "allow_any_idle_gpu": True,
                    "requires_positive_manifest": "",
                    "requires_ready_manifest": str(source_manifest),
                    "source_screen_manifest": str(source_manifest),
                    "source_screen_output_root": str(screen_root),
                    "archive_db": str(repo / "results/archive.db"),
                    "cas_root": str(repo / "results"),
                    "gpu_audit_root_template": str(
                        reports / f"gpu-exclusivity-audit-{campaign_id}-gpu{{gpu}}-{{attempt_id}}"
                    ),
                    "launch_argv_template": [
                        str(repo / ".venv/bin/python3"),
                        "-m",
                        "scripts.export.acwm_formal_visualization",
                        "--output-root",
                        str(gate_root),
                        "--environment",
                        environment,
                        "--primitive",
                        primitive,
                        "--seed",
                        str(eval_seed),
                        "--training-seed",
                        str(training_seed),
                        "--runtime-python",
                        str(Path(runtime_python).resolve()),
                        "--data-root",
                        str(Path(data_root).resolve()),
                        "--checkpoint-root",
                        str(Path(checkpoint_root).resolve()),
                        "--dataset-freeze",
                        str(repo / "runs/m0/protocol/dataset-freeze.json"),
                        "--heldout-protocol",
                        str(repo / "runs/m0/protocol/heldout-protocol.json"),
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
            )
            rank += 1

    destination = Path(output_root).resolve()
    payload = {
        "schema_version": 1,
        "artifact_type": "verdiwm-acwm-training-seed-factorial-queue",
        "state": "ready",
        "experiment_config": str(config_path),
        "experiment_id": config.get("experiment_id"),
        "environment": environment,
        "primitive": primitive,
        "training_seeds": training_seeds,
        "eval_seeds": eval_seeds,
        "factorial_shape": [len(training_seeds), len(eval_seeds)],
        "row_count": len(rows),
        "dependency_policy": "all trained checkpoints are gated once their screen manifest is ready, independent of screen sign",
        "rows": rows,
        "claim_boundary": (
            "These are independent repair-fine-tuning seeds, not independent base-model pretraining seeds. "
            "Only settled frozen official-gate receipts support quality claims."
        ),
    }
    return _write_bundle(destination, payload)


def _write_bundle(destination: Path, payload: dict[str, object]) -> dict[str, object]:
    if destination.exists() or destination.is_symlink():
        raise AcwmTrainingSeedQueueError("ACWM_TRAINING_SEED_QUEUE_OUTPUT_EXISTS")
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    try:
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary.mkdir(mode=0o700)
        _write_json(temporary / "queue.json", payload)
        rows = payload["rows"]
        assert isinstance(rows, list)
        fields = [
            "rank", "campaign_id", "environment", "primitive", "training_seed", "eval_seed",
            "checkpoint_step", "output_root", "requires_ready_manifest",
        ]
        with (temporary / "queue.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                assert isinstance(row, Mapping)
                writer.writerow({key: row.get(key, "") for key in fields})
        lines = [
            "# ACWM repair-training seed replication queue",
            "",
            f"Target: `{payload['environment']} + {payload['primitive']}`",
            f"Factorial design: `{payload['factorial_shape'][0]} x {payload['factorial_shape'][1]}`",
            "",
            "| Rank | Training seed | Eval seed | Campaign |",
            "|---:|---:|---:|---|",
        ]
        for row in rows:
            assert isinstance(row, Mapping)
            lines.append(
                f"| {row['rank']} | {row['training_seed']} | {row['eval_seed']} | `{row['campaign_id']}` |"
            )
        (temporary / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        manifest = {
            "schema_version": 1,
            "artifact_type": "verdiwm-acwm-training-seed-factorial-queue-manifest",
            "state": "ready",
            "queue_path": str(destination / "queue.json"),
            "csv_path": str(destination / "queue.csv"),
            "row_count": payload["row_count"],
            "factorial_shape": payload["factorial_shape"],
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
        raise AcwmTrainingSeedQueueError(error)
    return value


def _seeds(value: object, error: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise AcwmTrainingSeedQueueError(error)
    seeds: list[int] = []
    for raw in value:
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 1 or raw in seeds:
            raise AcwmTrainingSeedQueueError(error)
        seeds.append(raw)
    return seeds


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AcwmTrainingSeedQueueError(f"ACWM_TRAINING_SEED_JSON_INVALID:{path}")
    return payload


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--runtime-python", type=Path, default=DEFAULT_RUNTIME_PYTHON)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--screen-campaign-suffix", default="r1-retry2")
    args = parser.parse_args(argv)
    result = build_queue(
        experiment_config=args.experiment_config,
        output_root=args.output_root,
        report_root=args.report_root,
        runtime_python=args.runtime_python,
        data_root=args.data_root,
        checkpoint_root=args.checkpoint_root,
        candidate_gpus=args.gpus,
        screen_campaign_suffix=args.screen_campaign_suffix,
    )
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
