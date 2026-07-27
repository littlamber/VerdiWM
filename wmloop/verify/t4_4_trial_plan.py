"""Build a read-only T4.4 trial execution plan from convergence blockers."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from wmloop.archive.store import ArchiveStore, ContentAddressedStore


REQUIRED_ARMS = ("prior", "cold_start", "shuffled_prior")
DEFAULT_SEEDS = (101, 202, 303)


class T44TrialPlanError(RuntimeError):
    """T4.4 trial planning failed closed."""


def run_t44_trial_plan(
    *,
    prior_convergence_manifest: Path,
    primitive_materialization_gate_manifest: Path,
    output_root: Path,
    repo_root: Path,
    limited_gate_manifest: Path,
    failure_report_manifest: Path,
    goal_config: Path,
    runtime_python: Path,
    data_root: Path,
    checkpoint_root: Path,
    dataset_freeze: Path,
    heldout_protocol: Path,
    gpu_exclusivity_audit_manifest: Path | None = None,
    m4_phase_gate_manifest: Path | None = None,
    environments: Sequence[str] = ("pour_water",),
    gpus: Sequence[int] = (0,),
    seeds: Sequence[int] = DEFAULT_SEEDS,
    arm_primitives: Mapping[str, str] | None = None,
    trial_output_base: Path | None = None,
    train_steps: int = 16,
    train_batch_size: int = 16,
    train_val_batch_size: int = 8,
    train_size: int = 32,
    train_num_workers: int = 2,
    eval_inference_steps: int = 1,
    max_accept_trajectories: int = 1,
    extend_unreached: bool = False,
    archive_db: Path | None = None,
    cas_root: Path | None = None,
) -> dict[str, object]:
    """Write a T4.4 execution plan without launching training."""

    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise T44TrialPlanError("T44_TRIAL_PLAN_OUTPUT_EXISTS")
    normalized_seeds = _normalize_seeds(seeds)
    normalized_envs = _normalize_strings(environments, "T44_TRIAL_PLAN_ENVIRONMENTS_INVALID")
    normalized_gpus = _normalize_gpus(gpus)
    if train_steps < 1 or train_batch_size < 1 or train_val_batch_size < 1 or train_size < 1 or train_num_workers < 0:
        raise T44TrialPlanError("T44_TRIAL_PLAN_TRAINING_SCALE_INVALID")
    if eval_inference_steps < 1 or max_accept_trajectories < 1:
        raise T44TrialPlanError("T44_TRIAL_PLAN_EVAL_SCALE_INVALID")
    archive = ArchiveStore(archive_db) if archive_db is not None else None
    cas_storage_root = cas_root if cas_root is not None else (Path(archive_db).resolve().parent if archive_db is not None else destination.parent)
    cas = ContentAddressedStore(Path(cas_storage_root))
    convergence = _load_convergence(prior_convergence_manifest, cas=cas, archive=archive)
    materialization = _load_materialization_gate(primitive_materialization_gate_manifest, cas=cas, archive=archive)
    primitive_plan = _normalize_arm_primitives(arm_primitives or {})
    ready_primitives = set(materialization["closed_loop_ready_primitives"])
    blocker_list = _blockers(
        convergence=convergence,
        materialization=materialization,
        arm_primitives=primitive_plan,
        ready_primitives=ready_primitives,
        gpu_exclusivity_audit_manifest=gpu_exclusivity_audit_manifest,
    )
    existing = _existing_coverage(convergence["report"])
    unreached = _unreached_threshold_seeds(convergence["report"]) if extend_unreached else {}
    trial_base = Path(trial_output_base).resolve() if trial_output_base is not None else destination.parent / f"{destination.name}-trial-runs"
    planned = _planned_trials(
        existing=existing,
        unreached=unreached,
        extend_unreached=extend_unreached,
        arm_primitives=primitive_plan,
        ready_primitives=ready_primitives,
        environments=normalized_envs,
        seeds=normalized_seeds,
        gpus=normalized_gpus,
        trial_base=trial_base,
        command_inputs={
            "repo_root": repo_root,
            "limited_gate_manifest": limited_gate_manifest,
            "failure_report_manifest": failure_report_manifest,
            "goal_config": goal_config,
            "runtime_python": runtime_python,
            "data_root": data_root,
            "checkpoint_root": checkpoint_root,
            "dataset_freeze": dataset_freeze,
            "heldout_protocol": heldout_protocol,
            "gpu_exclusivity_audit_manifest": gpu_exclusivity_audit_manifest,
            "m4_phase_gate_manifest": m4_phase_gate_manifest,
            "primitive_materialization_gate_manifest": primitive_materialization_gate_manifest,
            "archive_db": archive_db,
            "cas_root": cas_root,
        },
        train_steps=train_steps,
        train_batch_size=train_batch_size,
        train_val_batch_size=train_val_batch_size,
        train_size=train_size,
        train_num_workers=train_num_workers,
        eval_inference_steps=eval_inference_steps,
        max_accept_trajectories=max_accept_trajectories,
    )
    report = {
        "schema_version": 1,
        "artifact_type": "wmloop-t4-4-trial-plan",
        "state": "ready" if not blocker_list else "blocked",
        "t4_4_trial_plan_ready": not blocker_list,
        "required_arms": list(REQUIRED_ARMS),
        "seeds": list(normalized_seeds),
        "environments": list(normalized_envs),
        "gpus": list(normalized_gpus),
        "target_progress_delta": convergence["report"].get("target_progress_delta"),
        "extend_unreached": extend_unreached,
        "unreached_threshold_seeds": {arm: sorted(values) for arm, values in unreached.items()},
        "existing_coverage": existing,
        "arm_primitives": primitive_plan,
        "closed_loop_ready_primitives": sorted(ready_primitives),
        "planned_trial_count": len(planned),
        "planned_trials": planned,
        "blockers": blocker_list,
        "sources": {
            "prior_convergence": convergence["summary"],
            "primitive_materialization_gate": materialization["summary"],
        },
        "limitations": [
            "This plan is read-only and does not launch GPU work.",
            "Arm labels only become valid evidence when the runner command records --trial-arm and --trial-seed.",
            "The plan refuses to fabricate arm primitive choices; each arm must be explicitly declared and closed-loop eligible.",
        ],
    }
    return _write_report_bundle(report=report, output_root=destination, cas=cas, archive=archive)


def _blockers(
    *,
    convergence: Mapping[str, object],
    materialization: Mapping[str, object],
    arm_primitives: Mapping[str, str],
    ready_primitives: set[str],
    gpu_exclusivity_audit_manifest: Path | None,
) -> list[dict[str, object]]:
    blockers: list[dict[str, object]] = []
    if materialization.get("closed_loop_ready_count", 0) < 3:
        blockers.append(
            {
                "code": "closed_loop_ready_primitive_count_below_t44_minimum",
                "observed": materialization.get("closed_loop_ready_count"),
                "minimum": 3,
            }
        )
    if convergence["report"].get("target_progress_delta") is None:
        blockers.append({"code": "target_progress_delta_missing"})
    if gpu_exclusivity_audit_manifest is None:
        blockers.append({"code": "gpu_exclusivity_audit_missing"})
    for arm in REQUIRED_ARMS:
        primitive = arm_primitives.get(arm)
        if primitive is None:
            blockers.append({"code": "arm_primitive_plan_missing", "arm": arm})
        elif primitive not in ready_primitives:
            blockers.append({"code": "arm_primitive_not_closed_loop_ready", "arm": arm, "primitive": primitive})
    return blockers


def _planned_trials(
    *,
    existing: Mapping[str, object],
    unreached: Mapping[str, set[str]],
    extend_unreached: bool,
    arm_primitives: Mapping[str, str],
    ready_primitives: set[str],
    environments: Sequence[str],
    seeds: Sequence[int],
    gpus: Sequence[int],
    trial_base: Path,
    command_inputs: Mapping[str, Path | None],
    train_steps: int,
    train_batch_size: int,
    train_val_batch_size: int,
    train_size: int,
    train_num_workers: int,
    eval_inference_steps: int,
    max_accept_trajectories: int,
) -> list[dict[str, object]]:
    planned = []
    existing_seeds_by_arm = {
        str(arm): {str(seed) for seed in seeds}
        for arm, seeds in (existing.get("seeds_by_arm") if isinstance(existing.get("seeds_by_arm"), Mapping) else {}).items()
        if isinstance(seeds, list)
    }
    ordinal = 0
    for arm in REQUIRED_ARMS:
        primitive = arm_primitives.get(arm)
        if primitive is None or primitive not in ready_primitives:
            continue
        for seed in seeds:
            seed_text = str(seed)
            seed_exists = seed_text in existing_seeds_by_arm.get(arm, set())
            plan_kind = "missing_coverage"
            if seed_exists:
                if not extend_unreached or seed_text not in unreached.get(arm, set()):
                    continue
                plan_kind = "threshold_extension"
            for environment in environments:
                gpu = int(gpus[ordinal % len(gpus)])
                suffix = "" if plan_kind == "missing_coverage" else f"-extend-t{train_steps}"
                output_root = trial_base / f"{arm}-s{seed}-{_safe_token(environment)}-{_safe_token(primitive)}{suffix}"
                command = _trial_command(
                    command_inputs=command_inputs,
                    output_root=output_root,
                    environment=environment,
                    primitive=primitive,
                    arm=arm,
                    seed=seed,
                    gpu=gpu,
                    train_steps=train_steps,
                    train_batch_size=train_batch_size,
                    train_val_batch_size=train_val_batch_size,
                    train_size=train_size,
                    train_num_workers=train_num_workers,
                    eval_inference_steps=eval_inference_steps,
                    max_accept_trajectories=max_accept_trajectories,
                )
                planned.append(
                    {
                        "plan_kind": plan_kind,
                        "arm": arm,
                        "seed": seed,
                        "environment": environment,
                        "primitive": primitive,
                        "gpu": gpu,
                        "output_root": str(output_root),
                        "command": command,
                    }
                )
                ordinal += 1
    return planned


def _unreached_threshold_seeds(report: Mapping[str, Any]) -> dict[str, set[str]]:
    threshold_budgets = report.get("threshold_budgets")
    if not isinstance(threshold_budgets, Mapping):
        return {}
    output: dict[str, set[str]] = {}
    for arm in REQUIRED_ARMS:
        payload = threshold_budgets.get(arm)
        if not isinstance(payload, Mapping):
            continue
        seed_results = payload.get("seed_results")
        if not isinstance(seed_results, list):
            continue
        for item in seed_results:
            if not isinstance(item, Mapping):
                continue
            seed = item.get("seed")
            if isinstance(seed, (str, int)) and str(seed) and item.get("reached") is False:
                output.setdefault(arm, set()).add(str(seed))
    return output


def _trial_command(
    *,
    command_inputs: Mapping[str, Path | None],
    output_root: Path,
    environment: str,
    primitive: str,
    arm: str,
    seed: int,
    gpu: int,
    train_steps: int,
    train_batch_size: int,
    train_val_batch_size: int,
    train_size: int,
    train_num_workers: int,
    eval_inference_steps: int,
    max_accept_trajectories: int,
) -> list[str]:
    gpu_audit = command_inputs["gpu_exclusivity_audit_manifest"]
    if gpu_audit is None:
        return []
    command = [
        "uv",
        "run",
        "python",
        "-m",
        "wmloop.execute.training_eval_limited_campaign",
        "run",
        "--repo-root",
        str(command_inputs["repo_root"]),
        "--limited-gate-manifest",
        str(command_inputs["limited_gate_manifest"]),
        "--failure-report-manifest",
        str(command_inputs["failure_report_manifest"]),
        "--goal-config",
        str(command_inputs["goal_config"]),
        "--output-root",
        str(output_root),
        "--runtime-python",
        str(command_inputs["runtime_python"]),
        "--data-root",
        str(command_inputs["data_root"]),
        "--checkpoint-root",
        str(command_inputs["checkpoint_root"]),
        "--dataset-freeze",
        str(command_inputs["dataset_freeze"]),
        "--heldout-protocol",
        str(command_inputs["heldout_protocol"]),
        "--gpu-exclusivity-audit-manifest",
        str(gpu_audit),
        "--gpus",
        str(gpu),
        "--parallel-slots",
        "1",
        "--environment",
        environment,
        "--proposal-primitive",
        primitive,
        "--trial-arm",
        arm,
        "--trial-seed",
        str(seed),
        "--train-steps",
        str(train_steps),
        "--train-batch-size",
        str(train_batch_size),
        "--train-val-batch-size",
        str(train_val_batch_size),
        "--train-size",
        str(train_size),
        "--train-num-workers",
        str(train_num_workers),
        "--max-accept-trajectories",
        str(max_accept_trajectories),
        "--replication-count",
        "1",
        "--eval-inference-steps",
        str(eval_inference_steps),
        "--primitive-materialization-gate-manifest",
        str(command_inputs["primitive_materialization_gate_manifest"]),
    ]
    if command_inputs["archive_db"] is not None:
        command.extend(["--archive-db", str(command_inputs["archive_db"])])
    if command_inputs["cas_root"] is not None:
        command.extend(["--cas-root", str(command_inputs["cas_root"])])
    if command_inputs["m4_phase_gate_manifest"] is not None:
        command.extend(["--m4-phase-gate-manifest", str(command_inputs["m4_phase_gate_manifest"])])
        command.append("--publish-settled-trials")
    return command


def _existing_coverage(report: Mapping[str, Any]) -> dict[str, object]:
    seeds_by_arm: dict[str, set[str]] = {arm: set() for arm in REQUIRED_ARMS}
    records = report.get("records")
    if isinstance(records, list):
        for record in records:
            if not isinstance(record, Mapping):
                continue
            arm = record.get("arm")
            seed = record.get("seed")
            if isinstance(arm, str) and arm in seeds_by_arm and isinstance(seed, (str, int)) and str(seed):
                seeds_by_arm[arm].add(str(seed))
    return {
        "seeds_by_arm": {arm: sorted(values) for arm, values in seeds_by_arm.items() if values},
        "seed_counts": {arm: len(values) for arm, values in seeds_by_arm.items() if values},
    }


def _normalize_arm_primitives(values: Mapping[str, str]) -> dict[str, str]:
    output = {}
    for arm, primitive in values.items():
        if arm not in REQUIRED_ARMS:
            raise T44TrialPlanError(f"T44_TRIAL_PLAN_ARM_INVALID:{arm}")
        if not isinstance(primitive, str) or not primitive:
            raise T44TrialPlanError("T44_TRIAL_PLAN_PRIMITIVE_INVALID")
        output[arm] = primitive
    return output


def _load_convergence(path: Path, *, cas: ContentAddressedStore, archive: ArchiveStore | None) -> dict[str, object]:
    manifest = _load_source(path, cas=cas, archive=archive)
    payload = manifest["payload"]
    if payload.get("artifact_type") != "wmloop-t4-4-prior-convergence-manifest":
        raise T44TrialPlanError("T44_TRIAL_PLAN_CONVERGENCE_MANIFEST_INVALID")
    report_path = payload.get("report_path")
    if not isinstance(report_path, str) or not report_path:
        raise T44TrialPlanError("T44_TRIAL_PLAN_CONVERGENCE_REPORT_MISSING")
    report_source = _load_source(Path(report_path), cas=cas, archive=archive)
    report = report_source["payload"]
    if report.get("artifact_type") != "wmloop-t4-4-prior-convergence":
        raise T44TrialPlanError("T44_TRIAL_PLAN_CONVERGENCE_REPORT_INVALID")
    return {
        "summary": {**manifest["summary"], "report_ref": report_source["summary"]["cas_ref"]},
        "report": report,
    }


def _load_materialization_gate(path: Path, *, cas: ContentAddressedStore, archive: ArchiveStore | None) -> dict[str, object]:
    manifest = _load_source(path, cas=cas, archive=archive)
    payload = manifest["payload"]
    if payload.get("artifact_type") != "wmloop-primitive-materialization-gate-manifest":
        raise T44TrialPlanError("T44_TRIAL_PLAN_MATERIALIZATION_GATE_INVALID")
    report_path = payload.get("report_path")
    if not isinstance(report_path, str) or not report_path:
        raise T44TrialPlanError("T44_TRIAL_PLAN_MATERIALIZATION_GATE_REPORT_MISSING")
    report_source = _load_source(Path(report_path), cas=cas, archive=archive)
    report = report_source["payload"]
    if report.get("artifact_type") != "wmloop-primitive-materialization-gate":
        raise T44TrialPlanError("T44_TRIAL_PLAN_MATERIALIZATION_GATE_REPORT_INVALID")
    ready = report.get("closed_loop_ready_primitives")
    if not isinstance(ready, list) or not all(isinstance(item, str) and item for item in ready):
        raise T44TrialPlanError("T44_TRIAL_PLAN_MATERIALIZATION_GATE_REPORT_INVALID")
    return {
        "summary": {**manifest["summary"], "report_ref": report_source["summary"]["cas_ref"]},
        "closed_loop_ready_count": int(report.get("closed_loop_ready_count", 0)),
        "closed_loop_ready_primitives": sorted(set(ready)),
    }


def _load_source(path: Path, *, cas: ContentAddressedStore, archive: ArchiveStore | None) -> dict[str, object]:
    resolved = Path(path).resolve(strict=True)
    try:
        payload_bytes = resolved.read_bytes()
        payload = json.loads(payload_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise T44TrialPlanError(f"T44_TRIAL_PLAN_SOURCE_INVALID:{resolved}") from exc
    if not isinstance(payload, dict):
        raise T44TrialPlanError(f"T44_TRIAL_PLAN_SOURCE_NOT_OBJECT:{resolved}")
    ref = cas.put_bytes(payload_bytes, media_type="application/json").uri
    if archive is not None:
        archive.record_artifact_reference(ref)
    return {
        "payload": payload,
        "summary": {
            "path": str(resolved),
            "artifact_type": payload.get("artifact_type"),
            "state": payload.get("state"),
            "cas_ref": ref,
        },
    }


def _write_report_bundle(
    *,
    report: Mapping[str, object],
    output_root: Path,
    cas: ContentAddressedStore,
    archive: ArchiveStore | None,
) -> dict[str, object]:
    destination = Path(output_root).resolve()
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    try:
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary.mkdir(mode=0o700)
        report_bytes = _canonical_json_bytes(report)
        markdown_bytes = _render_markdown(report).encode("utf-8")
        _write_bytes_atomic(temporary / "t4-4-trial-plan.json", report_bytes)
        _write_bytes_atomic(temporary / "t4-4-trial-plan.md", markdown_bytes)
        report_ref = cas.put_bytes(report_bytes, media_type="application/json").uri
        markdown_ref = cas.put_bytes(markdown_bytes, media_type="text/markdown").uri
        if archive is not None:
            archive.record_artifact_reference(report_ref)
            archive.record_artifact_reference(markdown_ref)
        manifest = {
            "schema_version": 1,
            "artifact_type": "wmloop-t4-4-trial-plan-manifest",
            "state": report["state"],
            "t4_4_trial_plan_ready": report["t4_4_trial_plan_ready"],
            "planned_trial_count": report["planned_trial_count"],
            "blockers": report["blockers"],
            "report_path": str(destination / "t4-4-trial-plan.json"),
            "markdown_path": str(destination / "t4-4-trial-plan.md"),
            "cas_refs": {"t4_4_trial_plan_json": report_ref, "t4_4_trial_plan_markdown": markdown_ref},
        }
        _write_bytes_atomic(temporary / "manifest.json", _canonical_json_bytes(manifest))
        os.replace(temporary, destination)
        return manifest
    except Exception:
        if temporary.is_dir():
            shutil.rmtree(temporary, ignore_errors=True)
        elif temporary.exists():
            temporary.unlink()
        raise


def _render_markdown(report: Mapping[str, object]) -> str:
    lines = [
        "# T4.4 Trial Plan",
        "",
        f"State: `{report['state']}`",
        f"Planned trials: `{report['planned_trial_count']}`",
        f"Closed-loop ready primitives: `{report['closed_loop_ready_primitives']}`",
        f"Target progress delta: `{report['target_progress_delta']}`",
        "",
        "## Blockers",
        "",
    ]
    blockers = report.get("blockers")
    if isinstance(blockers, list) and blockers:
        for blocker in blockers:
            lines.append(f"- `{blocker}`")
    else:
        lines.append("- None")
    lines.extend(["", "## Planned Trials", "", "| Arm | Seed | Env | Primitive | GPU | Output |", "|:--|--:|:--|:--|--:|:--|"])
    for trial in report.get("planned_trials", []):
        if isinstance(trial, Mapping):
            lines.append(
                f"| `{trial['arm']}` | {trial['seed']} | `{trial['environment']}` | "
                f"`{trial['primitive']}` | {trial['gpu']} | `{trial['output_root']}` |"
            )
    return "\n".join(lines) + "\n"


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def _normalize_seeds(values: Sequence[int]) -> tuple[int, ...]:
    seeds = tuple(int(value) for value in values)
    if not seeds or len(set(seeds)) != len(seeds) or any(value < 1 for value in seeds):
        raise T44TrialPlanError("T44_TRIAL_PLAN_SEEDS_INVALID")
    return seeds


def _normalize_gpus(values: Sequence[int]) -> tuple[int, ...]:
    gpus = tuple(int(value) for value in values)
    if not gpus or len(set(gpus)) != len(gpus) or any(value < 0 for value in gpus):
        raise T44TrialPlanError("T44_TRIAL_PLAN_GPUS_INVALID")
    return gpus


def _normalize_strings(values: Sequence[str], code: str) -> tuple[str, ...]:
    output = tuple(str(value) for value in values if str(value))
    if not output or len(set(output)) != len(output):
        raise T44TrialPlanError(code)
    return output


def _safe_token(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value)).strip("_") or "unknown"


def _parse_arm_primitive(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected ARM=PRIMITIVE")
    arm, primitive = value.split("=", 1)
    if arm not in REQUIRED_ARMS or not primitive:
        raise argparse.ArgumentTypeError("expected one of prior,cold_start,shuffled_prior=PRIMITIVE")
    return arm, primitive


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="build a T4.4 trial execution plan")
    run.add_argument("--prior-convergence-manifest", type=Path, required=True)
    run.add_argument("--primitive-materialization-gate-manifest", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--repo-root", type=Path, default=Path("."))
    run.add_argument("--limited-gate-manifest", type=Path, required=True)
    run.add_argument("--failure-report-manifest", type=Path, required=True)
    run.add_argument("--goal-config", type=Path, required=True)
    run.add_argument("--runtime-python", type=Path, required=True)
    run.add_argument("--data-root", type=Path, required=True)
    run.add_argument("--checkpoint-root", type=Path, required=True)
    run.add_argument("--dataset-freeze", type=Path, required=True)
    run.add_argument("--heldout-protocol", type=Path, required=True)
    run.add_argument("--gpu-exclusivity-audit-manifest", type=Path)
    run.add_argument("--m4-phase-gate-manifest", type=Path)
    run.add_argument("--environment", dest="environments", action="append", default=[])
    run.add_argument("--gpu", dest="gpus", type=int, action="append", default=[])
    run.add_argument("--seed", dest="seeds", type=int, action="append", default=[])
    run.add_argument("--arm-primitive", action="append", type=_parse_arm_primitive, default=[])
    run.add_argument("--trial-output-base", type=Path)
    run.add_argument("--train-steps", type=int, default=16)
    run.add_argument("--train-batch-size", type=int, default=16)
    run.add_argument("--train-val-batch-size", type=int, default=8)
    run.add_argument("--train-size", type=int, default=32)
    run.add_argument("--train-num-workers", type=int, default=2)
    run.add_argument("--eval-inference-steps", type=int, default=1)
    run.add_argument("--max-accept-trajectories", type=int, default=1)
    run.add_argument("--extend-unreached", action="store_true")
    run.add_argument("--archive-db", type=Path)
    run.add_argument("--cas-root", type=Path)
    args = parser.parse_args(argv)
    if args.command == "run":
        arm_primitives = {arm: primitive for arm, primitive in args.arm_primitive}
        manifest = run_t44_trial_plan(
            prior_convergence_manifest=args.prior_convergence_manifest,
            primitive_materialization_gate_manifest=args.primitive_materialization_gate_manifest,
            output_root=args.output_root,
            repo_root=args.repo_root,
            limited_gate_manifest=args.limited_gate_manifest,
            failure_report_manifest=args.failure_report_manifest,
            goal_config=args.goal_config,
            runtime_python=args.runtime_python,
            data_root=args.data_root,
            checkpoint_root=args.checkpoint_root,
            dataset_freeze=args.dataset_freeze,
            heldout_protocol=args.heldout_protocol,
            gpu_exclusivity_audit_manifest=args.gpu_exclusivity_audit_manifest,
            m4_phase_gate_manifest=args.m4_phase_gate_manifest,
            environments=tuple(args.environments) if args.environments else ("pour_water",),
            gpus=tuple(args.gpus) if args.gpus else (0,),
            seeds=tuple(args.seeds) if args.seeds else DEFAULT_SEEDS,
            arm_primitives=arm_primitives,
            trial_output_base=args.trial_output_base,
            train_steps=args.train_steps,
            train_batch_size=args.train_batch_size,
            train_val_batch_size=args.train_val_batch_size,
            train_size=args.train_size,
            train_num_workers=args.train_num_workers,
            eval_inference_steps=args.eval_inference_steps,
            max_accept_trajectories=args.max_accept_trajectories,
            extend_unreached=args.extend_unreached,
            archive_db=args.archive_db,
            cas_root=args.cas_root,
        )
        print(json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
