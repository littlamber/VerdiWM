#!/usr/bin/env python3
"""Build paired long-horizon work for selected ACWM training-seed checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path


class AcwmTrainingSeedHorizonQueueError(RuntimeError):
    """The selected-checkpoint horizon contract is invalid."""


def _load_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AcwmTrainingSeedHorizonQueueError(f"TRAINSEED_HORIZON_JSON_INVALID:{path}") from exc
    if not isinstance(payload, dict):
        raise AcwmTrainingSeedHorizonQueueError(f"TRAINSEED_HORIZON_JSON_INVALID:{path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_root(checkpoint: Path) -> Path:
    for ancestor in checkpoint.parents:
        if ancestor.name == "retained_training":
            runtime = ancestor.parent / "retained_runtime"
            if runtime.is_dir():
                return runtime.resolve()
            break
    raise AcwmTrainingSeedHorizonQueueError(
        f"TRAINSEED_HORIZON_RUNTIME_ROOT_MISSING:{checkpoint}"
    )


def _audit_template(report_root: Path, campaign_id: str) -> str:
    return str(report_root / f"gpu-exclusivity-audit-{campaign_id}-gpu{{gpu}}-{{attempt_id}}")


def _base_row(
    *, rank: int, phase: str, campaign_id: str, environment: str, primitive: str,
    seed: int, output_root: Path, dependencies: Sequence[Path], argv: Sequence[str],
    report_root: Path, repo_root: Path, gpus: Sequence[int], resource_class: str,
    training_seed: int | None = None, checkpoint_step: int | None = None,
) -> dict[str, object]:
    return {
        "rank": rank,
        "phase": phase,
        "campaign_id": campaign_id,
        "environment": environment,
        "primitive": primitive,
        "seed": seed,
        "training_seed": training_seed,
        "checkpoint_step": checkpoint_step,
        "train_steps": 0,
        "resource_class": resource_class,
        "output_root": str(output_root),
        "candidate_gpus": list(gpus),
        "allow_any_idle_gpu": True,
        "requires_positive_manifest": "",
        "requires_ready_manifests": [str(path) for path in dependencies],
        "requires_official_quality_manifest": "",
        "archive_db": str(repo_root / "results/archive.db"),
        "cas_root": str(repo_root / "results"),
        "gpu_audit_root_template": _audit_template(report_root, campaign_id),
        "launch_argv_template": list(argv),
    }


def _horizon_argv(
    *, runtime_python: Path, repo_root: Path, data_root: Path, checkpoint_root: Path,
    environment: str, output_root: Path, horizons: Sequence[int], trajectory_seed: int,
    max_trajectories: int, checkpoint: Path | None = None, vendor_root: Path | None = None,
) -> list[str]:
    argv = [
        str(runtime_python), "-m", "wmloop.diagnose.horizon_runtime", "run",
        "--repo-root", str(repo_root), "--data-root", str(data_root),
        "--checkpoint-root", str(checkpoint_root), "--environment", environment,
        "--split", "ind_test", "--output-root", str(output_root), "--horizons",
        *(str(value) for value in horizons), "--max-trajectories", str(max_trajectories),
        "--num-inference-steps", "50", "--device", "cuda", "--seed",
        str(trajectory_seed), "--mode", "autoregressive", "--max-evidence", "1",
        "--max-video-evidence", str(max_trajectories), "--archive-db",
        str(repo_root / "results/archive.db"), "--cas-root", str(repo_root / "results"),
        "--gpu-index", "{gpu}", "--gpu-exclusivity-audit-manifest",
        "{gpu_audit_manifest}", "--gpu-exclusivity-max-age-seconds", "3600",
    ]
    if checkpoint is not None:
        argv[argv.index("--environment"):argv.index("--environment")] = [
            "--checkpoint-path", str(checkpoint)
        ]
    if vendor_root is not None:
        argv[argv.index("--data-root"):argv.index("--data-root")] = [
            "--vendor-root", str(vendor_root)
        ]
    return argv


def _gate_manifests(
    *, report_root: Path, environment: str, primitive: str, training_seed: int,
    checkpoint_step: int, eval_seeds: Sequence[int], revision: str,
) -> list[Path]:
    paths = []
    for eval_seed in eval_seeds:
        if checkpoint_step == 512:
            name = (
                f"acwm-formal-trainseed-gate-{environment}-{primitive}-"
                f"ts{training_seed}-es{eval_seed}-r1"
            )
        else:
            name = (
                f"acwm-trainseed-stability-gate-{environment}-{primitive}-"
                f"ts{training_seed}-es{eval_seed}-step{checkpoint_step}-{revision}"
            )
        path = report_root / name / "manifest.json"
        payload = _load_json(path.resolve(strict=True))
        gate = payload.get("official_quality_gate")
        if payload.get("state") != "ready" or not isinstance(gate, Mapping) or gate.get("pass") is not True:
            raise AcwmTrainingSeedHorizonQueueError(
                f"TRAINSEED_HORIZON_GATE_NOT_PASSING:{training_seed}:{eval_seed}:{checkpoint_step}"
            )
        paths.append(path.resolve())
    return paths


def build_training_seed_horizon_queue(
    *, stability_manifest: Path, output_root: Path, report_root: Path, repo_root: Path,
    runtime_python: Path, data_root: Path, checkpoint_root: Path,
    gpus: Sequence[int] = (0, 1, 2), horizons: Sequence[int] = (16, 32, 48),
    trajectory_seed: int = 101, max_trajectories: int = 3, revision: str = "r1",
) -> dict[str, object]:
    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise AcwmTrainingSeedHorizonQueueError("TRAINSEED_HORIZON_OUTPUT_EXISTS")
    reports = Path(report_root).resolve(strict=True)
    repo = Path(repo_root).resolve(strict=True)
    runtime = Path(runtime_python).resolve(strict=True)
    data = Path(data_root).resolve(strict=True)
    checkpoints = Path(checkpoint_root).resolve(strict=True)
    manifest_path = Path(stability_manifest).resolve(strict=True)
    manifest = _load_json(manifest_path)
    if (
        manifest.get("artifact_type")
        != "verdiwm-acwm-training-seed-checkpoint-stability-summary-manifest"
        or manifest.get("state") != "ready"
    ):
        raise AcwmTrainingSeedHorizonQueueError("TRAINSEED_HORIZON_STABILITY_INVALID")
    summary_path = Path(str(manifest.get("summary_path") or "")).resolve(strict=True)
    summary = _load_json(summary_path)
    environment = str(summary.get("environment") or "")
    primitive = str(summary.get("primitive") or "")
    selected = summary.get("selected_checkpoints")
    eval_seeds = summary.get("eval_seeds")
    if (
        summary.get("artifact_type")
        != "verdiwm-acwm-training-seed-checkpoint-stability-summary"
        or summary.get("state") != "ready"
        or not environment or not primitive or not isinstance(selected, list)
        or len(selected) < 2 or not isinstance(eval_seeds, list) or not eval_seeds
    ):
        raise AcwmTrainingSeedHorizonQueueError("TRAINSEED_HORIZON_STABILITY_INVALID")
    horizon_values = sorted({int(value) for value in horizons})
    if not horizon_values or horizon_values[0] < 2 or not gpus or max_trajectories < 1:
        raise AcwmTrainingSeedHorizonQueueError("TRAINSEED_HORIZON_PROTOCOL_INVALID")

    case = f"{environment}-{primitive}-seed-selected-h{max(horizon_values)}-{revision}"
    baseline_id = f"acwm-trainseed-horizon-baseline-{case}"
    baseline_root = reports / baseline_id
    rows: list[dict[str, object]] = [
        _base_row(
            rank=1, phase="long_horizon_baseline", campaign_id=baseline_id,
            environment=environment, primitive=primitive, seed=trajectory_seed,
            output_root=baseline_root, dependencies=[manifest_path], report_root=reports,
            repo_root=repo, gpus=gpus, resource_class="gpu",
            argv=_horizon_argv(
                runtime_python=runtime, repo_root=repo, data_root=data,
                checkpoint_root=checkpoints, environment=environment,
                output_root=baseline_root, horizons=horizon_values,
                trajectory_seed=trajectory_seed, max_trajectories=max_trajectories,
            ),
        )
    ]
    baseline_manifest = baseline_root / "manifest.json"
    profile_paths: list[Path] = []
    profile_manifests: list[Path] = []
    candidate_rows: list[tuple[Mapping[str, object], Path, int, int]] = []
    rank = 2
    for raw in selected:
        if not isinstance(raw, Mapping):
            raise AcwmTrainingSeedHorizonQueueError("TRAINSEED_HORIZON_SELECTION_INVALID")
        training_seed = int(raw["training_seed"])
        step = int(raw["checkpoint_step"])
        checkpoint = Path(str(raw["checkpoint_path"])).resolve(strict=True)
        if _sha256(checkpoint) != str(raw.get("checkpoint_sha256") or ""):
            raise AcwmTrainingSeedHorizonQueueError(
                f"TRAINSEED_HORIZON_CHECKPOINT_SHA_MISMATCH:{training_seed}"
            )
        runtime_root = _runtime_root(checkpoint)
        gates = _gate_manifests(
            report_root=reports, environment=environment, primitive=primitive,
            training_seed=training_seed, checkpoint_step=step,
            eval_seeds=[int(value) for value in eval_seeds], revision=revision,
        )
        candidate_id = f"acwm-trainseed-horizon-candidate-{environment}-{primitive}-ts{training_seed}-step{step}-{revision}"
        candidate_root = reports / candidate_id
        candidate = _base_row(
            rank=rank, phase="long_horizon_candidate", campaign_id=candidate_id,
            environment=environment, primitive=primitive, seed=trajectory_seed,
            training_seed=training_seed, checkpoint_step=step, output_root=candidate_root,
            dependencies=[manifest_path, baseline_manifest, *gates], report_root=reports,
            repo_root=repo, gpus=gpus, resource_class="gpu",
            argv=_horizon_argv(
                runtime_python=runtime, repo_root=repo, data_root=data,
                checkpoint_root=checkpoints, environment=environment,
                output_root=candidate_root, horizons=horizon_values,
                trajectory_seed=trajectory_seed, max_trajectories=max_trajectories,
                checkpoint=checkpoint, vendor_root=runtime_root,
            ),
        )
        rows.append(candidate)
        candidate_rows.append((raw, candidate_root, training_seed, step))
        rank += 1

    for _, candidate_root, training_seed, step in candidate_rows:
        candidate_manifest = candidate_root / "manifest.json"
        profile_id = f"acwm-trainseed-horizon-effect-profile-{environment}-{primitive}-ts{training_seed}-step{step}-{revision}"
        profile_root = reports / profile_id
        profile = _base_row(
            rank=rank, phase="horizon_effect_profile", campaign_id=profile_id,
            environment=environment, primitive=primitive, seed=trajectory_seed,
            training_seed=training_seed, checkpoint_step=step, output_root=profile_root,
            dependencies=[baseline_manifest, candidate_manifest, manifest_path],
            report_root=reports, repo_root=repo, gpus=gpus, resource_class="cpu",
            argv=[
                str(repo / ".venv/bin/python3"),
                str(repo / "scripts/export/acwm_horizon_effect_profile.py"),
                "--baseline-manifest", str(baseline_manifest),
                "--candidate-manifest", str(candidate_manifest), "--primitive", primitive,
                "--training-seed", str(training_seed), "--failure-report",
                str(reports / "m1-raw-failure-reports-ladder-r2-threshold-aligned/failure_reports" / f"{environment}.json"),
                "--mechanism-cards", str(reports / "primitive-mechanism-cards-r1/mechanism_cards.csv"),
                "--checkpoint-ladder-manifest", str(manifest_path),
                "--output-root", str(profile_root),
            ],
        )
        rows.append(profile)
        profile_paths.append(profile_root / "horizon-effect-profile.json")
        profile_manifests.append(profile_root / "manifest.json")
        rank += 1

        triptych_id = f"acwm-trainseed-horizon-triptych-{environment}-{primitive}-ts{training_seed}-step{step}-{revision}"
        triptych_root = reports / triptych_id
        rows.append(
            _base_row(
                rank=rank, phase="horizon_triptych", campaign_id=triptych_id,
                environment=environment, primitive=primitive, seed=trajectory_seed,
                training_seed=training_seed, checkpoint_step=step, output_root=triptych_root,
                dependencies=[baseline_manifest, candidate_manifest], report_root=reports,
                repo_root=repo, gpus=gpus, resource_class="cpu",
                argv=[str(runtime), str(repo / "scripts/export/acwm_horizon_triptych.py"),
                      "--baseline-manifest", str(baseline_manifest),
                      "--candidate-manifest", str(candidate_manifest),
                      "--output-root", str(triptych_root)],
            )
        )
        rank += 1

    summary_id = f"acwm-trainseed-horizon-stability-summary-{case}"
    summary_root = reports / summary_id
    summary_argv = [str(repo / ".venv/bin/python3"),
                    str(repo / "scripts/export/acwm_training_seed_horizon_summary.py"),
                    "--stability-manifest", str(manifest_path), "--output-root", str(summary_root)]
    for path in profile_paths:
        summary_argv.extend(["--profile", str(path)])
    rows.append(_base_row(
        rank=rank, phase="training_seed_horizon_summary", campaign_id=summary_id,
        environment=environment, primitive=primitive, seed=trajectory_seed,
        output_root=summary_root, dependencies=profile_manifests, report_root=reports,
        repo_root=repo, gpus=gpus, resource_class="cpu", argv=summary_argv,
    ))
    rank += 1
    experience_id = f"acwm-trainseed-horizon-experience-map-{case}"
    experience_root = reports / experience_id
    experience_argv = [str(repo / ".venv/bin/python3"),
                       str(repo / "scripts/export/acwm_horizon_experience_map.py"),
                       "--output-root", str(experience_root)]
    for path in profile_paths:
        experience_argv.extend(["--profile", str(path)])
    rows.append(_base_row(
        rank=rank, phase="horizon_experience_map", campaign_id=experience_id,
        environment=environment, primitive=primitive, seed=trajectory_seed,
        output_root=experience_root, dependencies=profile_manifests, report_root=reports,
        repo_root=repo, gpus=gpus, resource_class="cpu", argv=experience_argv,
    ))

    queue = {
        "schema_version": 1, "artifact_type": "wmloop-acwm-autoloop-queue",
        "state": "ready", "row_count": len(rows), "preferred_gpus": list(gpus),
        "training_seed_horizon_contract": {
            "state": "verified", "environment": environment, "primitive": primitive,
            "selected_checkpoint_count": len(selected), "horizons": horizon_values,
            "trajectory_seed": trajectory_seed, "max_trajectories": max_trajectories,
            "execution_order": "shared_baseline_then_parallel_candidates_then_cpu_evidence",
            "claim_boundary": "Paired long-horizon observation across independent repair-training seeds; not cross-backbone causal transfer.",
        },
        "rows": rows,
    }
    packet_manifest = {
        "schema_version": 1,
        "artifact_type": "verdiwm-acwm-training-seed-horizon-queue-manifest",
        "state": "ready", "environment": environment, "primitive": primitive,
        "row_count": len(rows), "selected_checkpoint_count": len(selected),
        "horizons": horizon_values, "queue_path": str(destination / "autoloop-queue.json"),
        "stability_manifest": str(manifest_path),
    }
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    temporary.mkdir(mode=0o700, parents=True)
    try:
        (temporary / "autoloop-queue.json").write_text(
            json.dumps(queue, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (temporary / "manifest.json").write_text(
            json.dumps(packet_manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return packet_manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stability-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, default=Path("results/reports"))
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--runtime-python", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--horizons", nargs="+", type=int, default=[16, 32, 48])
    parser.add_argument("--trajectory-seed", type=int, default=101)
    parser.add_argument("--max-trajectories", type=int, default=3)
    parser.add_argument("--revision", default="r1")
    args = parser.parse_args(argv)
    result = build_training_seed_horizon_queue(
        stability_manifest=args.stability_manifest, output_root=args.output_root,
        report_root=args.report_root, repo_root=args.repo_root,
        runtime_python=args.runtime_python, data_root=args.data_root,
        checkpoint_root=args.checkpoint_root, gpus=args.gpus, horizons=args.horizons,
        trajectory_seed=args.trajectory_seed, max_trajectories=args.max_trajectories,
        revision=args.revision,
    )
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
