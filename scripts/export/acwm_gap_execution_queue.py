#!/usr/bin/env python3
"""Create an executable ACWM gap-driven queue from the staging plan."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from wmloop.execute.training_monitor_policy import DEFAULT_CONFIRMATION_STEPS


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GAP_PLAN = ROOT / "results/reports/acwm-gap-driven-staging-plan-r1/acwm-gap-driven-staging-plan.json"
DEFAULT_OUT = ROOT / "results/reports/acwm-gap-execution-queue-r7"
DEFAULT_LIMITED_GATE = ROOT / "results/reports/limited-campaign-gate-8env-official-current-warning-r1/manifest.json"
DEFAULT_FAILURE_MANIFEST = ROOT / "results/reports/m1-raw-failure-reports-ladder-r1/manifest.json"
DEFAULT_GOAL = ROOT / "configs/goal/g1_long_horizon_ladder_v1.yaml"
DEFAULT_RUNTIME_PYTHON = Path(os.environ.get("VERDIWM_RUNTIME_PYTHON", sys.executable))
DEFAULT_DATA_ROOT = Path(os.environ.get("ACWM_DATA_ROOT", "data/ACWM-Phys"))
DEFAULT_CHECKPOINT_ROOT = Path(os.environ.get("ACWM_CHECKPOINT_ROOT", "checkpoints/ACWM-Phys"))
DEFAULT_DATASET_FREEZE = ROOT / "runs/m0/protocol/dataset-freeze.json"
DEFAULT_HELDOUT_PROTOCOL = ROOT / "runs/m0/protocol/heldout-protocol.json"
DEFAULT_GPU_AUDIT_ROOT = ROOT / "results/reports/gpu-exclusivity-audit-acwm-gap-gpu01-r2"
DEFAULT_M4_PHASE_GATE = ROOT / "results/reports/m4-phase-gate-ladder-r3/manifest.json"
DEFAULT_PRIMITIVE_MATERIALIZATION_GATE = ROOT / "results/reports/primitive-materialization-gate-current-r1/manifest.json"
DEFAULT_ARCHIVE_DB = ROOT / "results/archive.db"
DEFAULT_CAS_ROOT = ROOT / "results"
RUNNABLE_PRIMITIVES = {
    "drift_token_trim",
    "history_noise_schedule",
    "latent_motion_prior",
    "mixture_reweight",
    "next_forcing",
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise TypeError(f"JSON object expected: {path}")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_text(argv: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in argv)


def campaign_command(
    *,
    repo_root: Path,
    limited_gate: Path,
    failure_manifest: Path,
    goal_config: Path,
    output_root: Path,
    runtime_python: Path,
    data_root: Path,
    checkpoint_root: Path,
    dataset_freeze: Path,
    heldout_protocol: Path,
    gpu_audit_manifest: Path,
    gpus: list[int],
    environment: str,
    primitive: str,
    campaign_id: str,
    seed: int,
    train_steps: int,
    train_batch_size: int,
    training_timeout_seconds: int,
    formal_publish: bool,
    m4_phase_gate: Path,
    primitive_materialization_gate: Path,
    archive_db: Path,
    cas_root: Path,
) -> list[str]:
    argv = [
        str(ROOT / ".venv/bin/python3"),
        "-m",
        "wmloop.execute.training_eval_limited_campaign",
        "run",
        "--repo-root",
        str(repo_root),
        "--limited-gate-manifest",
        str(limited_gate),
        "--failure-report-manifest",
        str(failure_manifest),
        "--goal-config",
        str(goal_config),
        "--output-root",
        str(output_root),
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
        "--gpu-exclusivity-audit-manifest",
        str(gpu_audit_manifest),
        "--gpus",
        *(str(gpu) for gpu in gpus),
        "--environment",
        environment,
        "--parallel-slots",
        str(min(len(gpus), 1)),
        "--campaign-id",
        campaign_id,
        "--train-steps",
        str(train_steps),
        "--train-batch-size",
        str(train_batch_size),
        "--train-val-batch-size",
        "8",
        "--train-size",
        "32",
        "--train-num-workers",
        "2",
        "--proposal-primitive",
        primitive,
        "--weight",
        "0.2",
        "--history-noise",
        "0.1",
        "--keep-tokens",
        "2",
        "--trial-seed",
        str(seed),
        "--max-accept-trajectories",
        "1",
        "--replication-count",
        "1",
        "--eval-inference-steps",
        "1",
        "--hook-timeout-seconds",
        "60",
        "--training-timeout-seconds",
        str(training_timeout_seconds),
        "--eval-timeout-seconds",
        "1800",
        "--gpu-exclusivity-max-age-seconds",
        "3600",
        "--archive-db",
        str(archive_db),
        "--cas-root",
        str(cas_root),
        "--poll-interval-seconds",
        "5",
    ]
    if formal_publish:
        argv.extend(
            [
                "--m4-phase-gate-manifest",
                str(m4_phase_gate),
                "--primitive-materialization-gate-manifest",
                str(primitive_materialization_gate),
                "--publish-settled-trials",
            ]
        )
    return argv


def gpu_audit_command(*, output_root: Path, gpus: list[int], archive_db: Path, cas_root: Path) -> list[str]:
    return [
        str(ROOT / ".venv/bin/python3"),
        "-m",
        "wmloop.execute.gpu_exclusivity_audit",
        "run",
        "--output-root",
        str(output_root),
        "--gpus",
        *(str(gpu) for gpu in gpus),
        "--archive-db",
        str(archive_db),
        "--cas-root",
        str(cas_root),
    ]


def direct_campaign_specs(
    records: list[dict[str, Any]],
    *,
    seeds: list[int],
    formal_train_steps: int,
    canary_train_steps: int,
    train_batch_size: int,
    formal_training_timeout_seconds: int,
    canary_training_timeout_seconds: int,
) -> list[dict[str, Any]]:
    by_env = {str(record["environment"]): record for record in records}
    specs: list[dict[str, Any]] = []
    robot = by_env.get("robot_arm")
    if robot and robot.get("evidence_level") == "repeated_positive":
        for seed in seeds:
            specs.append(
                {
                    "environment": "robot_arm",
                    "primitive": "latent_motion_prior",
                    "seed": seed,
                    "priority": "P0",
                    "queue_type": "formal_confirmation",
                    "formal_publish": True,
                    "train_steps": formal_train_steps,
                    "train_batch_size": train_batch_size,
                    "training_timeout_seconds": formal_training_timeout_seconds,
                    "rationale": "Confirm the only stable positive cell under the frozen formal M4 verifier.",
                    "claim_boundary": "Confirmation evidence only; no registry, probe, goal, or protocol mutation.",
                }
            )
    pour = by_env.get("pour_water")
    if pour and pour.get("evidence_level") == "candidate_unstable":
        for seed in seeds:
            specs.append(
                {
                    "environment": "pour_water",
                    "primitive": "drift_token_trim",
                    "seed": seed,
                    "priority": "P0",
                    "queue_type": "staging_canary",
                    "formal_publish": False,
                    "train_steps": canary_train_steps,
                    "train_batch_size": train_batch_size,
                    "training_timeout_seconds": canary_training_timeout_seconds,
                    "rationale": "Particle/fluid-regime candidate with high max delta but unstable mean; canary before formal rerun.",
                    "claim_boundary": "Canary evidence only; rerun through formal publication if it repeats.",
                }
            )
    cloth = by_env.get("cloth_move")
    if cloth and cloth.get("evidence_level") == "candidate_unstable":
        for seed in seeds[:1]:
            specs.append(
                {
                    "environment": "cloth_move",
                    "primitive": "latent_motion_prior",
                    "seed": seed,
                    "priority": "P1",
                    "queue_type": "staging_canary",
                    "formal_publish": False,
                    "train_steps": canary_train_steps,
                    "train_batch_size": train_batch_size,
                    "training_timeout_seconds": canary_training_timeout_seconds,
                    "rationale": "Deformable-regime weak candidate; run only after robot_arm/pour_water queue or if spare GPU time exists.",
                    "claim_boundary": "Official-current checkpoint warning remains isolated from official 100k reproduction claims.",
                }
            )
    return specs


def materialization_orders(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    orders: list[dict[str, Any]] = []
    for record in records:
        env = str(record["environment"])
        if record.get("evidence_level") == "repeated_positive":
            continue
        for probe in record.get("diagnostic_probe_candidates", []) or []:
            orders.append(
                {
                    "environment": env,
                    "order_type": "diagnostic_probe_staging",
                    "target_name": probe,
                    "target_role": "diagnostic",
                    "priority": "P0",
                    "allowed_paths": f"wmloop/diagnose/probes/{probe}.py; tests/test_{probe}.py; configs/probes/acwm_v1_staging.json",
                    "forbidden_boundary": "Must not enter verdict_evidence or frozen acwm_v1 verdict registry without a version boundary.",
                    "admission_gates": "schema; unit test; report-field sidecar; no verifier wiring",
                }
            )
        for primitive in record.get("recommended_existing_primitives", []) or []:
            if primitive in RUNNABLE_PRIMITIVES:
                continue
            orders.append(
                {
                    "environment": env,
                    "order_type": "primitive_runtime_materialization",
                    "target_name": primitive,
                    "target_role": "closed_loop_runtime_ready",
                    "priority": "P1",
                    "allowed_paths": f"wmloop/primitives/definitions/{primitive}/; vendor/ACWM-Phys/acwm/wmloop_hooks/; tests/test_primitive_runtime_smoke.py",
                    "forbidden_boundary": "Must not mutate eval.py, splits, goal specs, verdict probes, or formal registry during the active campaign.",
                    "admission_gates": "manifest; clean diff; hook contract; runtime smoke; canary; version boundary before formal registry promotion",
                }
            )
    unique: dict[tuple[str, str, str], dict[str, Any]] = {}
    for order in orders:
        unique[(str(order["environment"]), str(order["order_type"]), str(order["target_name"]))] = order
    return list(unique.values())


def build_queue(
    *,
    gap_plan_path: Path,
    repo_root: Path,
    limited_gate: Path,
    failure_manifest: Path,
    goal_config: Path,
    runtime_python: Path,
    data_root: Path,
    checkpoint_root: Path,
    dataset_freeze: Path,
    heldout_protocol: Path,
    gpu_audit_root: Path,
    m4_phase_gate: Path,
    primitive_materialization_gate: Path,
    archive_db: Path,
    cas_root: Path,
    gpus: list[int],
    seeds: list[int],
    formal_train_steps: int = DEFAULT_CONFIRMATION_STEPS,
    canary_train_steps: int = 512,
    train_batch_size: int = 16,
    formal_training_timeout_seconds: int = 43200,
    canary_training_timeout_seconds: int = 43200,
) -> dict[str, Any]:
    if formal_train_steps < 1 or canary_train_steps < 1 or train_batch_size < 1:
        raise ValueError("train steps and batch size must be positive")
    if formal_training_timeout_seconds < 1 or canary_training_timeout_seconds < 1:
        raise ValueError("training timeouts must be positive")
    gap_plan = read_json(gap_plan_path)
    records = gap_plan.get("environment_records")
    if not isinstance(records, list):
        raise ValueError("gap plan missing environment_records")
    gpu_manifest = gpu_audit_root / "manifest.json"
    gpu_preflight = gpu_audit_command(output_root=gpu_audit_root, gpus=gpus, archive_db=archive_db, cas_root=cas_root)
    direct_specs = direct_campaign_specs(
        records,
        seeds=seeds,
        formal_train_steps=formal_train_steps,
        canary_train_steps=canary_train_steps,
        train_batch_size=train_batch_size,
        formal_training_timeout_seconds=formal_training_timeout_seconds,
        canary_training_timeout_seconds=canary_training_timeout_seconds,
    )
    direct_queue: list[dict[str, Any]] = []
    for index, spec in enumerate(direct_specs, start=1):
        campaign_id = "acwm-gap-{queue_type}-{environment}-{primitive}-s{seed}-t{train_steps}-r1".format(**spec)
        campaign_output = ROOT / "results/reports" / campaign_id
        assigned_gpu = gpus[(index - 1) % len(gpus)]
        item_gpu_audit_root = ROOT / "results/reports" / f"gpu-exclusivity-audit-{campaign_id}-gpu{assigned_gpu}-r1"
        item_gpu_audit_manifest = item_gpu_audit_root / "manifest.json"
        item_gpu_audit = gpu_audit_command(
            output_root=item_gpu_audit_root,
            gpus=[assigned_gpu],
            archive_db=archive_db,
            cas_root=cas_root,
        )
        argv = campaign_command(
            repo_root=repo_root,
            limited_gate=limited_gate,
            failure_manifest=failure_manifest,
            goal_config=goal_config,
            output_root=campaign_output,
            runtime_python=runtime_python,
            data_root=data_root,
            checkpoint_root=checkpoint_root,
            dataset_freeze=dataset_freeze,
            heldout_protocol=heldout_protocol,
            gpu_audit_manifest=item_gpu_audit_manifest,
            gpus=[assigned_gpu],
            environment=str(spec["environment"]),
            primitive=str(spec["primitive"]),
            campaign_id=campaign_id,
            seed=int(spec["seed"]),
            train_steps=int(spec["train_steps"]),
            train_batch_size=int(spec["train_batch_size"]),
            training_timeout_seconds=int(spec["training_timeout_seconds"]),
            formal_publish=bool(spec["formal_publish"]),
            m4_phase_gate=m4_phase_gate,
            primitive_materialization_gate=primitive_materialization_gate,
            archive_db=archive_db,
            cas_root=cas_root,
        )
        direct_queue.append(
            {
                "ordinal": index,
                "campaign_id": campaign_id,
                "output_root": str(campaign_output),
                "assigned_gpu": assigned_gpu,
                "gpu_audit_root": str(item_gpu_audit_root),
                "gpu_audit_manifest": str(item_gpu_audit_manifest),
                "gpu_audit_command": item_gpu_audit,
                "gpu_audit_command_text": command_text(item_gpu_audit),
                "command": argv,
                "command_text": command_text(argv),
                **spec,
            }
        )
    materialization = materialization_orders(records)
    return {
        "schema_version": 1,
        "artifact_type": "wmloop-acwm-gap-execution-queue",
        "state": "ready",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_files": {
            "gap_plan": str(gap_plan_path),
            "gap_plan_sha256": sha256_file(gap_plan_path),
            "limited_gate": str(limited_gate),
            "failure_manifest": str(failure_manifest),
            "goal_config": str(goal_config),
            "gpu_audit_root": str(gpu_audit_root),
            "m4_phase_gate": str(m4_phase_gate),
            "primitive_materialization_gate": str(primitive_materialization_gate),
        },
        "available_gpus_for_queue": gpus,
        "runnable_primitives": sorted(RUNNABLE_PRIMITIVES),
        "global_gpu_preflight": {
            "required": True,
            "output_root": str(gpu_audit_root),
            "manifest_path": str(gpu_manifest),
            "command": gpu_preflight,
            "command_text": command_text(gpu_preflight),
            "freshness_seconds": 3600,
        },
        "direct_campaign_count": len(direct_queue),
        "direct_campaigns": direct_queue,
        "materialization_order_count": len(materialization),
        "materialization_orders": materialization,
        "policy": {
            "formal_publish_default": "only repeated-positive confirmation",
            "candidate_canary_default": "limited pilot first, then formal rerun if repeated positive",
            "verdict_probe_mutation_allowed": False,
            "formal_registry_mutation_allowed": False,
            "diagnostic_probe_staging_allowed": True,
            "formal_train_steps": formal_train_steps,
            "canary_train_steps": canary_train_steps,
            "train_batch_size": train_batch_size,
            "formal_training_timeout_seconds": formal_training_timeout_seconds,
            "canary_training_timeout_seconds": canary_training_timeout_seconds,
            "screen_train_steps": canary_train_steps,
            "confirmation_train_steps": formal_train_steps,
            "screen_to_confirmation_rule": "512-step canary screens only route positives to the 1k confirmation ladder; canary results are not paper-facing final evidence.",
            "checkpoint_eval_ladder": [512, 800, 1000],
            "best_checkpoint_rule": "Evaluate ladder checkpoints with frozen held-out rollouts and pick best held-out primary metric, not final by default.",
        },
    }


def render_markdown(queue: dict[str, Any]) -> str:
    lines = [
        "# ACWM Gap Execution Queue R1",
        "",
        f"Generated at: `{queue['generated_at']}`",
        "",
        "## Launch Preflight",
        "",
        "Optional whole-slot audit before launching queue commands:",
        "",
        "```bash",
        queue["global_gpu_preflight"]["command_text"],
        "```",
        "",
        "Each direct campaign below also has its own per-GPU audit, which is the manifest actually consumed by that command. GPU audit freshness window: `3600` seconds.",
        "",
        "## Direct Campaign Queue",
        "",
        "| # | Type | Env | Primitive | Seed | GPU | Formal publish | Priority |",
        "|--:|:--|:--|:--|--:|--:|:--|:--|",
    ]
    for item in queue["direct_campaigns"]:
        lines.append(
            f"| {item['ordinal']} | {item['queue_type']} | {item['environment']} | {item['primitive']} | {item['seed']} | {item['assigned_gpu']} | {item['formal_publish']} | {item['priority']} |"
        )
    lines.extend(["", "### Commands", ""])
    for item in queue["direct_campaigns"]:
        lines.extend(
            [
                f"#### {item['ordinal']}. `{item['campaign_id']}`",
                "",
                f"- rationale: {item['rationale']}",
                f"- boundary: {item['claim_boundary']}",
                f"- assigned GPU: `{item['assigned_gpu']}`",
                "",
                "GPU audit:",
                "",
                "```bash",
                item["gpu_audit_command_text"],
                "```",
                "",
                "Campaign:",
                "",
                "```bash",
                item["command_text"],
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "## Materialization / Probe Staging Orders",
            "",
            "| Env | Type | Target | Priority | Boundary |",
            "|:--|:--|:--|:--|:--|",
        ]
    )
    for order in queue["materialization_orders"]:
        lines.append(
            f"| {order['environment']} | {order['order_type']} | {order['target_name']} | {order['priority']} | {order['forbidden_boundary']} |"
        )
    lines.extend(
        [
            "",
            "## Policy",
            "",
            "- Verdict probes and the formal registry remain frozen.",
            "- Candidate canaries default to 512 train steps and remain limited-pilot evidence unless rerun through the 1k confirmation ladder.",
            "- Ladder checkpoints should be evaluated at 512/800/1000; select the best held-out checkpoint rather than assuming the final checkpoint is best.",
            "- New probes are diagnostic-only until a version boundary promotes them.",
            "",
        ]
    )
    return "\n".join(lines)


def write_bundle(queue: dict[str, Any], output_root: Path) -> dict[str, Any]:
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"output exists: {output_root}")
    temporary = output_root.parent / f".{output_root.name}.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True, mode=0o700)
    try:
        write_json(temporary / "acwm-gap-execution-queue.json", queue)
        (temporary / "acwm-gap-execution-queue.md").write_text(render_markdown(queue), encoding="utf-8")
        write_csv(
            temporary / "tables/direct-campaigns.csv",
            queue["direct_campaigns"],
            [
                "ordinal",
                "queue_type",
                "environment",
                "primitive",
                "seed",
                "formal_publish",
                "train_steps",
                "train_batch_size",
                "training_timeout_seconds",
                "priority",
                "campaign_id",
                "output_root",
                "assigned_gpu",
                "gpu_audit_root",
                "gpu_audit_manifest",
                "gpu_audit_command_text",
                "rationale",
                "claim_boundary",
                "command_text",
            ],
        )
        write_csv(
            temporary / "tables/materialization-orders.csv",
            queue["materialization_orders"],
            [
                "environment",
                "order_type",
                "target_name",
                "target_role",
                "priority",
                "allowed_paths",
                "forbidden_boundary",
                "admission_gates",
            ],
        )
        commands_path = temporary / "commands/run-direct-queue.sh"
        commands_path.parent.mkdir(parents=True, exist_ok=True)
        commands_path.write_text(
            "\n".join(
                [
                    "#!/usr/bin/env bash",
                    "set -euo pipefail",
                    "",
                    "# Each campaign uses its own per-GPU audit manifest.",
                    "",
                    *(line for item in queue["direct_campaigns"] for line in (item["gpu_audit_command_text"], item["command_text"])),
                    "",
                ]
            ),
            encoding="utf-8",
        )
        manifest = {
            "schema_version": 1,
            "artifact_type": "wmloop-acwm-gap-execution-queue-manifest",
            "state": "ready",
            "output_root": str(output_root),
            "report_path": str(output_root / "acwm-gap-execution-queue.json"),
            "markdown_path": str(output_root / "acwm-gap-execution-queue.md"),
            "direct_campaigns_csv": str(output_root / "tables/direct-campaigns.csv"),
            "materialization_orders_csv": str(output_root / "tables/materialization-orders.csv"),
            "run_script": str(output_root / "commands/run-direct-queue.sh"),
            "direct_campaign_count": queue["direct_campaign_count"],
            "materialization_order_count": queue["materialization_order_count"],
            "global_gpu_preflight": queue["global_gpu_preflight"],
            "policy": queue["policy"],
        }
        write_json(temporary / "manifest.json", manifest)
        os.replace(temporary, output_root)
        return manifest
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gap-plan", type=Path, default=DEFAULT_GAP_PLAN)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--limited-gate", type=Path, default=DEFAULT_LIMITED_GATE)
    parser.add_argument("--failure-manifest", type=Path, default=DEFAULT_FAILURE_MANIFEST)
    parser.add_argument("--goal-config", type=Path, default=DEFAULT_GOAL)
    parser.add_argument("--runtime-python", type=Path, default=DEFAULT_RUNTIME_PYTHON)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--dataset-freeze", type=Path, default=DEFAULT_DATASET_FREEZE)
    parser.add_argument("--heldout-protocol", type=Path, default=DEFAULT_HELDOUT_PROTOCOL)
    parser.add_argument("--gpu-audit-root", type=Path, default=DEFAULT_GPU_AUDIT_ROOT)
    parser.add_argument("--m4-phase-gate", type=Path, default=DEFAULT_M4_PHASE_GATE)
    parser.add_argument("--primitive-materialization-gate", type=Path, default=DEFAULT_PRIMITIVE_MATERIALIZATION_GATE)
    parser.add_argument("--archive-db", type=Path, default=DEFAULT_ARCHIVE_DB)
    parser.add_argument("--cas-root", type=Path, default=DEFAULT_CAS_ROOT)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--seeds", type=int, nargs="+", default=[501, 502, 503])
    parser.add_argument("--formal-train-steps", type=int, default=DEFAULT_CONFIRMATION_STEPS)
    parser.add_argument("--canary-train-steps", type=int, default=512)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--formal-training-timeout-seconds", type=int, default=43200)
    parser.add_argument("--canary-training-timeout-seconds", type=int, default=43200)
    args = parser.parse_args(argv)

    queue = build_queue(
        gap_plan_path=args.gap_plan.resolve(strict=True),
        repo_root=args.repo_root.resolve(strict=True),
        limited_gate=args.limited_gate.resolve(strict=True),
        failure_manifest=args.failure_manifest.resolve(strict=True),
        goal_config=args.goal_config,
        runtime_python=args.runtime_python,
        data_root=args.data_root,
        checkpoint_root=args.checkpoint_root,
        dataset_freeze=args.dataset_freeze,
        heldout_protocol=args.heldout_protocol,
        gpu_audit_root=args.gpu_audit_root,
        m4_phase_gate=args.m4_phase_gate,
        primitive_materialization_gate=args.primitive_materialization_gate,
        archive_db=args.archive_db,
        cas_root=args.cas_root,
        gpus=args.gpus,
        seeds=args.seeds,
        formal_train_steps=args.formal_train_steps,
        canary_train_steps=args.canary_train_steps,
        train_batch_size=args.train_batch_size,
        formal_training_timeout_seconds=args.formal_training_timeout_seconds,
        canary_training_timeout_seconds=args.canary_training_timeout_seconds,
    )
    manifest = write_bundle(queue, args.output_root.resolve())
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
