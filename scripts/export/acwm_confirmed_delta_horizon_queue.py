#!/usr/bin/env python3
"""Build long-horizon evidence work for a multi-seed checkpoint transform."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path


class ConfirmedDeltaHorizonQueueError(RuntimeError):
    """The confirmation evidence cannot authorize long-horizon work."""


def _read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfirmedDeltaHorizonQueueError(f"DELTA_HORIZON_JSON_INVALID:{path}") from exc
    if not isinstance(payload, dict):
        raise ConfirmedDeltaHorizonQueueError(f"DELTA_HORIZON_JSON_INVALID:{path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_confirmation_group(
    pass_manifests: Sequence[Path],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if len(pass_manifests) < 2:
        raise ConfirmedDeltaHorizonQueueError("DELTA_HORIZON_TWO_PASSES_REQUIRED")
    paths = [Path(path).resolve(strict=True) for path in pass_manifests]
    if len(set(paths)) != len(paths):
        raise ConfirmedDeltaHorizonQueueError("DELTA_HORIZON_PASS_MANIFEST_DUPLICATE")
    manifests = [_read_json(path) for path in paths]
    identities: set[tuple[str, str, str, str, float]] = set()
    seeds: set[int] = set()
    for manifest in manifests:
        gate = manifest.get("official_quality_gate")
        provenance = manifest.get("checkpoint_transform_provenance")
        seed = manifest.get("eval_seed", manifest.get("seed"))
        alpha = provenance.get("alpha") if isinstance(provenance, Mapping) else None
        if (
            manifest.get("state") != "ready"
            or manifest.get("primitive") != "checkpoint_delta_scaling"
            or not isinstance(gate, Mapping)
            or gate.get("pass") is not True
            or not isinstance(provenance, Mapping)
            or provenance.get("state") != "verified"
            or isinstance(seed, bool)
            or not isinstance(seed, int)
            or isinstance(alpha, bool)
            or not isinstance(alpha, (int, float))
        ):
            raise ConfirmedDeltaHorizonQueueError("DELTA_HORIZON_PASS_MANIFEST_INVALID")
        identity = (
            str(manifest.get("environment") or ""),
            str(provenance.get("source_primitive") or ""),
            str(manifest.get("baseline_checkpoint_sha256") or ""),
            str(manifest.get("candidate_checkpoint_sha256") or ""),
            float(alpha),
        )
        if not all(identity[:4]):
            raise ConfirmedDeltaHorizonQueueError("DELTA_HORIZON_IDENTITY_MISSING")
        identities.add(identity)
        seeds.add(seed)
    if len(identities) != 1:
        raise ConfirmedDeltaHorizonQueueError("DELTA_HORIZON_PASS_IDENTITY_MISMATCH")
    if len(seeds) < 2:
        raise ConfirmedDeltaHorizonQueueError("DELTA_HORIZON_DISTINCT_SEEDS_REQUIRED")
    environment, source_primitive, baseline_sha, candidate_sha, alpha = identities.pop()
    first = manifests[0]
    return manifests, {
        "environment": environment,
        "source_primitive": source_primitive,
        "baseline_checkpoint_sha256": baseline_sha,
        "candidate_checkpoint_sha256": candidate_sha,
        "alpha": alpha,
        "eval_seeds": sorted(seeds),
        "baseline_checkpoint": str(first.get("baseline_checkpoint") or ""),
        "candidate_checkpoint": str(first.get("candidate_checkpoint") or ""),
        "candidate_runtime_root": str(first.get("candidate_runtime_root") or ""),
        "pass_manifest_paths": [str(path) for path in paths],
        "pass_manifest_sha256": [_sha256(path) for path in paths],
    }


def _gpu_audit_template(report_root: Path, campaign_id: str) -> str:
    return str(report_root / f"gpu-exclusivity-audit-{campaign_id}-gpu{{gpu}}-{{attempt_id}}")


def _delta_case_id(
    *, environment: str, source_primitive: str, alpha: float, max_horizon: int, revision: int
) -> str:
    alpha_token = str(alpha).replace("-", "m").replace(".", "p")
    return (
        f"{environment}-checkpoint_delta_scaling-{source_primitive}-"
        f"alpha{alpha_token}-h{max_horizon}-r{revision}"
    )


def _horizon_row(
    *,
    rank: int,
    side: str,
    identity: Mapping[str, object],
    report_root: Path,
    repo_root: Path,
    runtime_python: Path,
    data_root: Path,
    checkpoint_root: Path,
    gpus: Sequence[int],
    horizons: Sequence[int],
    mode: str,
    num_inference_steps: int,
    max_trajectories: int,
    trajectory_seed: int,
    revision: int,
) -> dict[str, object]:
    environment = str(identity["environment"])
    source_primitive = str(identity["source_primitive"])
    case_id = _delta_case_id(
        environment=environment,
        source_primitive=source_primitive,
        alpha=float(identity["alpha"]),
        max_horizon=max(horizons),
        revision=revision,
    )
    campaign_id = f"acwm-long-horizon-{side}-{case_id}"
    output_root = report_root / campaign_id
    checkpoint = Path(str(identity[f"{side}_checkpoint"])).resolve(strict=True)
    argv = [
        str(runtime_python),
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
        str(checkpoint),
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
        str(num_inference_steps),
        "--device",
        "cuda",
        "--seed",
        str(trajectory_seed),
        "--mode",
        mode,
        "--max-evidence",
        "1",
        "--max-video-evidence",
        str(max_trajectories),
        "--archive-db",
        str(repo_root / "results/archive.db"),
        "--cas-root",
        str(repo_root / "results"),
        "--gpu-index",
        "{gpu}",
        "--gpu-exclusivity-audit-manifest",
        "{gpu_audit_manifest}",
        "--gpu-exclusivity-max-age-seconds",
        "3600",
    ]
    if side == "candidate":
        runtime_root = Path(str(identity["candidate_runtime_root"])).resolve(strict=True)
        argv[argv.index("--data-root"):argv.index("--data-root")] = [
            "--vendor-root",
            str(runtime_root),
        ]
    return {
        "rank": rank,
        "phase": f"long_horizon_{side}",
        "campaign_id": campaign_id,
        "environment": environment,
        "primitive": "checkpoint_delta_scaling",
        "source_primitive": source_primitive,
        "checkpoint_delta_alpha": identity["alpha"],
        "seed": trajectory_seed,
        "train_steps": 0,
        "output_root": str(output_root),
        "candidate_gpus": list(gpus),
        "allow_any_idle_gpu": True,
        "requires_positive_manifest": "",
        "requires_ready_manifests": list(identity["pass_manifest_paths"]),
        "requires_official_quality_manifest": "",
        "archive_db": str(repo_root / "results/archive.db"),
        "cas_root": str(repo_root / "results"),
        "gpu_audit_root_template": _gpu_audit_template(report_root, campaign_id),
        "launch_argv_template": argv,
    }


def _dependent_row(
    *,
    rank: int,
    phase: str,
    campaign_id: str,
    environment: str,
    source_primitive: str,
    alpha: float,
    output_root: Path,
    dependencies: Sequence[Path],
    argv: Sequence[str],
    report_root: Path,
    repo_root: Path,
    gpus: Sequence[int],
    trajectory_seed: int,
) -> dict[str, object]:
    return {
        "rank": rank,
        "phase": phase,
        "campaign_id": campaign_id,
        "environment": environment,
        "primitive": "checkpoint_delta_scaling",
        "source_primitive": source_primitive,
        "checkpoint_delta_alpha": alpha,
        "seed": trajectory_seed,
        "train_steps": 0,
        "resource_class": "cpu",
        "output_root": str(output_root),
        "candidate_gpus": list(gpus),
        "allow_any_idle_gpu": True,
        "requires_positive_manifest": "",
        "requires_ready_manifests": [str(path) for path in dependencies],
        "requires_official_quality_manifest": "",
        "archive_db": str(repo_root / "results/archive.db"),
        "cas_root": str(repo_root / "results"),
        "gpu_audit_root_template": _gpu_audit_template(report_root, campaign_id),
        "launch_argv_template": list(argv),
    }


def build_confirmed_delta_horizon_queue(
    *,
    pass_manifests: Sequence[Path],
    output_root: Path,
    report_root: Path,
    repo_root: Path,
    runtime_python: Path,
    data_root: Path,
    checkpoint_root: Path,
    gpus: Sequence[int] = (0, 1, 2),
    horizons: Sequence[int] = (16, 32, 48, 64, 96),
    mode: str = "autoregressive",
    num_inference_steps: int = 50,
    max_trajectories: int = 1,
    trajectory_seed: int = 101,
    factorized_replication: bool = False,
    revision: int = 1,
) -> dict[str, object]:
    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise ConfirmedDeltaHorizonQueueError("DELTA_HORIZON_OUTPUT_EXISTS")
    report_root = Path(report_root).resolve(strict=True)
    repo_root = Path(repo_root).resolve(strict=True)
    runtime_python = Path(runtime_python).resolve(strict=True)
    data_root = Path(data_root).resolve(strict=True)
    checkpoint_root = Path(checkpoint_root).resolve(strict=True)
    if not gpus or any(isinstance(gpu, bool) or gpu < 0 for gpu in gpus):
        raise ConfirmedDeltaHorizonQueueError("DELTA_HORIZON_GPUS_INVALID")
    horizon_values = sorted(set(int(value) for value in horizons))
    if not horizon_values or horizon_values[0] < 2 or mode not in {"parallel", "autoregressive"}:
        raise ConfirmedDeltaHorizonQueueError("DELTA_HORIZON_PROTOCOL_INVALID")
    if (
        num_inference_steps < 1
        or max_trajectories < 1
        or trajectory_seed < 0
        or revision < 1
    ):
        raise ConfirmedDeltaHorizonQueueError("DELTA_HORIZON_PROTOCOL_INVALID")
    _, identity = _validated_confirmation_group(pass_manifests)

    baseline = _horizon_row(
        rank=1,
        side="baseline",
        identity=identity,
        report_root=report_root,
        repo_root=repo_root,
        runtime_python=runtime_python,
        data_root=data_root,
        checkpoint_root=checkpoint_root,
        gpus=gpus,
        horizons=horizon_values,
        mode=mode,
        num_inference_steps=num_inference_steps,
        max_trajectories=max_trajectories,
        trajectory_seed=trajectory_seed,
        revision=revision,
    )
    candidate = _horizon_row(
        rank=2,
        side="candidate",
        identity=identity,
        report_root=report_root,
        repo_root=repo_root,
        runtime_python=runtime_python,
        data_root=data_root,
        checkpoint_root=checkpoint_root,
        gpus=gpus,
        horizons=horizon_values,
        mode=mode,
        num_inference_steps=num_inference_steps,
        max_trajectories=max_trajectories,
        trajectory_seed=trajectory_seed,
        revision=revision,
    )
    environment = str(identity["environment"])
    source_primitive = str(identity["source_primitive"])
    alpha = float(identity["alpha"])
    case_id = _delta_case_id(
        environment=environment,
        source_primitive=source_primitive,
        alpha=alpha,
        max_horizon=max(horizon_values),
        revision=revision,
    )
    baseline_manifest = Path(str(baseline["output_root"])) / "manifest.json"
    candidate_manifest = Path(str(candidate["output_root"])) / "manifest.json"
    candidate["requires_ready_manifests"] = [
        *candidate["requires_ready_manifests"],
        str(baseline_manifest),
    ]
    paired_dependencies = [baseline_manifest, candidate_manifest]
    event_gate = None
    event_gate_args: list[str] = []
    profile_dependencies = list(paired_dependencies)
    if environment == "pour_water":
        event_id = f"acwm-pour-water-event-gate-{case_id}"
        event_root = report_root / event_id
        event_gate = _dependent_row(
            rank=3,
            phase="event_semantic_gate",
            campaign_id=event_id,
            environment=environment,
            source_primitive=source_primitive,
            alpha=alpha,
            output_root=event_root,
            dependencies=paired_dependencies,
            argv=[
                str(runtime_python),
                str(repo_root / "scripts/export/acwm_pour_water_event_gate.py"),
                "--baseline-manifest",
                str(baseline_manifest),
                "--candidate-manifest",
                str(candidate_manifest),
                "--primitive",
                "checkpoint_delta_scaling",
                "--output-root",
                str(event_root),
            ],
            report_root=report_root,
            repo_root=repo_root,
            gpus=gpus,
            trajectory_seed=trajectory_seed,
        )
        event_report = event_root / "event-gate.json"
        profile_dependencies.append(event_root / "manifest.json")
        event_gate_args = ["--event-gate", str(event_report)]
    profile_id = f"acwm-horizon-effect-profile-{case_id}"
    profile_root = report_root / profile_id
    profile = _dependent_row(
        rank=4 if event_gate is not None else 3,
        phase="horizon_effect_profile",
        campaign_id=profile_id,
        environment=environment,
        source_primitive=source_primitive,
        alpha=alpha,
        output_root=profile_root,
        dependencies=profile_dependencies,
        argv=[
            str(repo_root / ".venv/bin/python3"),
            str(repo_root / "scripts/export/acwm_horizon_effect_profile.py"),
            "--baseline-manifest",
            str(baseline_manifest),
            "--candidate-manifest",
            str(candidate_manifest),
            "--primitive",
            "checkpoint_delta_scaling",
            "--mechanism-primitive",
            source_primitive,
            "--failure-report",
            str(report_root / "m1-raw-failure-reports-ladder-r2-threshold-aligned/failure_reports" / f"{environment}.json"),
            "--mechanism-cards",
            str(report_root / "primitive-mechanism-cards-r1/mechanism_cards.csv"),
            *event_gate_args,
            "--output-root",
            str(profile_root),
        ],
        report_root=report_root,
        repo_root=repo_root,
        gpus=gpus,
        trajectory_seed=trajectory_seed,
    )
    triptych_id = f"acwm-horizon-triptych-{case_id}"
    triptych_root = report_root / triptych_id
    triptych = _dependent_row(
        rank=5 if event_gate is not None else 4,
        phase="horizon_triptych",
        campaign_id=triptych_id,
        environment=environment,
        source_primitive=source_primitive,
        alpha=alpha,
        output_root=triptych_root,
        dependencies=paired_dependencies,
        argv=[
            str(runtime_python),
            str(repo_root / "scripts/export/acwm_horizon_triptych.py"),
            "--baseline-manifest",
            str(baseline_manifest),
            "--candidate-manifest",
            str(candidate_manifest),
            "--output-root",
            str(triptych_root),
        ],
        report_root=report_root,
        repo_root=repo_root,
        gpus=gpus,
        trajectory_seed=trajectory_seed,
    )
    experience_id = f"acwm-horizon-experience-map-{case_id}"
    experience_root = report_root / experience_id
    profile_manifest = profile_root / "manifest.json"
    experience = _dependent_row(
        rank=6 if event_gate is not None else 5,
        phase="horizon_experience_map",
        campaign_id=experience_id,
        environment=environment,
        source_primitive=source_primitive,
        alpha=alpha,
        output_root=experience_root,
        dependencies=[profile_manifest],
        argv=[
            str(repo_root / ".venv/bin/python3"),
            str(repo_root / "scripts/export/acwm_horizon_experience_map.py"),
            "--profile",
            str(profile_root / "horizon-effect-profile.json"),
            "--output-root",
            str(experience_root),
        ],
        report_root=report_root,
        repo_root=repo_root,
        gpus=gpus,
        trajectory_seed=trajectory_seed,
    )
    rows = [baseline, candidate]
    if event_gate is not None:
        rows.append(event_gate)
    rows.extend([profile, triptych, experience])
    contract = {
        "state": "verified",
        "kind": "multi_seed_checkpoint_delta_long_horizon_evidence",
        "pass_count": len(identity["pass_manifest_paths"]),
        "pass_manifests": identity["pass_manifest_paths"],
        "pass_manifest_sha256": identity["pass_manifest_sha256"],
        "distinct_eval_seeds": identity["eval_seeds"],
        "candidate_checkpoint_sha256": identity["candidate_checkpoint_sha256"],
        "horizons": horizon_values,
        "mode": mode,
        "trajectory_seed": trajectory_seed,
        "max_trajectories": max_trajectories,
        "evidence_role": (
            "factorized_heldout_replication"
            if factorized_replication
            else "initial_long_horizon_observation"
        ),
        "revision": revision,
        "execution_order": "baseline_then_candidate",
        "claim_boundary": (
            "The frozen 37-frame official gate governs short-horizon pixel quality. This paired "
            "autoregressive evidence governs long-horizon drift and environment-event claims."
        ),
    }
    queue = {
        "schema_version": 1,
        "artifact_type": "wmloop-acwm-autoloop-queue",
        "state": "ready",
        "row_count": len(rows),
        "preferred_gpus": list(gpus),
        "post_confirmation_evidence_contract": contract,
        "rows": rows,
    }
    manifest = {
        "schema_version": 1,
        "artifact_type": "wmloop-confirmed-delta-horizon-queue-manifest",
        "state": "ready",
        "environment": environment,
        "primitive": "checkpoint_delta_scaling",
        "source_primitive": source_primitive,
        "checkpoint_delta_alpha": alpha,
        "max_horizon": max(horizon_values),
        "seconds_at_10fps": max(horizon_values) / 10.0,
        "trajectory_seed": trajectory_seed,
        "max_trajectories": max_trajectories,
        "evidence_role": contract["evidence_role"],
        "revision": revision,
        "queue_path": str(destination / "autoloop-queue.json"),
        "post_confirmation_evidence_contract": contract,
    }
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    temporary.mkdir(mode=0o700, parents=True)
    try:
        (temporary / "autoloop-queue.json").write_text(
            json.dumps(queue, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pass-manifest", action="append", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, default=Path("results/reports"))
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--runtime-python", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--horizons", nargs="+", type=int, default=[16, 32, 48, 64, 96])
    parser.add_argument("--mode", choices=("parallel", "autoregressive"), default="autoregressive")
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--max-trajectories", type=int, default=1)
    parser.add_argument("--trajectory-seed", type=int, default=101)
    parser.add_argument("--factorized-replication", action="store_true")
    parser.add_argument("--revision", type=int, default=1)
    args = parser.parse_args(argv)
    result = build_confirmed_delta_horizon_queue(
        pass_manifests=args.pass_manifest,
        output_root=args.output_root,
        report_root=args.report_root,
        repo_root=args.repo_root,
        runtime_python=args.runtime_python,
        data_root=args.data_root,
        checkpoint_root=args.checkpoint_root,
        gpus=args.gpus,
        horizons=args.horizons,
        mode=args.mode,
        num_inference_steps=args.num_inference_steps,
        max_trajectories=args.max_trajectories,
        trajectory_seed=args.trajectory_seed,
        factorized_replication=args.factorized_replication,
        revision=args.revision,
    )
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
