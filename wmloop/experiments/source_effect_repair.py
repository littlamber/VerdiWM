"""Execute a frozen source-effect evidence-repair job."""

from __future__ import annotations

import json
import csv
import io
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from wmloop.experiments.acwm_fingerprint import sha256_file
from wmloop.experiments._artifacts import canonical_json, write_bundle


class SourceEffectRepairError(ValueError):
    """A source-effect repair plan or job violates its frozen contract."""


def load_source_effect_repair_plan(*, config_path: Path) -> dict[str, object]:
    config_file = Path(config_path).resolve()
    repo_root = config_file.parents[2]
    config = _load_json(config_file, "SOURCE_EFFECT_REPAIR_CONFIG_INVALID")
    if (
        config.get("artifact_type") != "verdiwm-source-effect-repair-preregistration"
        or config.get("state") != "frozen_before_execution"
        or not isinstance(config.get("primitive"), str)
    ):
        raise SourceEffectRepairError("SOURCE_EFFECT_REPAIR_CONFIG_CONTRACT_INVALID")
    source_audit = _resolve_under(repo_root, _required_string(config, "source_audit"))
    expected_audit_sha = _required_string(config, "source_audit_sha256")
    if not source_audit.is_file() or sha256_file(source_audit) != expected_audit_sha:
        raise SourceEffectRepairError("SOURCE_EFFECT_REPAIR_SOURCE_AUDIT_SHA_MISMATCH")
    audit = _load_json(source_audit, "SOURCE_EFFECT_REPAIR_SOURCE_AUDIT_INVALID")
    if (
        audit.get("artifact_type") != "verdiwm-source-effect-evidence-audit"
        or audit.get("primitive") != config.get("primitive")
    ):
        raise SourceEffectRepairError("SOURCE_EFFECT_REPAIR_SOURCE_AUDIT_CONTRACT_INVALID")
    runtime = config.get("runtime")
    protocol = config.get("frozen_protocol")
    jobs = config.get("jobs")
    if not isinstance(runtime, Mapping) or not isinstance(protocol, Mapping):
        raise SourceEffectRepairError("SOURCE_EFFECT_REPAIR_RUNTIME_INVALID")
    if not isinstance(jobs, list) or not jobs or any(not isinstance(row, Mapping) for row in jobs):
        raise SourceEffectRepairError("SOURCE_EFFECT_REPAIR_JOBS_INVALID")
    runtime_python = Path(_required_string(runtime, "python")).resolve()
    data_root = Path(_required_string(runtime, "data_root")).resolve()
    checkpoint_root = Path(_required_string(runtime, "checkpoint_root")).resolve()
    gpus = runtime.get("candidate_gpus")
    if (
        not runtime_python.is_file()
        or not data_root.is_dir()
        or not checkpoint_root.is_dir()
        or not isinstance(gpus, list)
        or not gpus
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in gpus)
        or len(set(gpus)) != len(gpus)
    ):
        raise SourceEffectRepairError("SOURCE_EFFECT_REPAIR_RUNTIME_INVALID")
    dataset_freeze = _resolve_under(repo_root, _required_string(protocol, "dataset_freeze"))
    heldout_protocol = _resolve_under(repo_root, _required_string(protocol, "heldout_protocol"))
    if not dataset_freeze.is_file() or not heldout_protocol.is_file():
        raise SourceEffectRepairError("SOURCE_EFFECT_REPAIR_PROTOCOL_INPUT_MISSING")
    job_ids: set[str] = set()
    output_roots: set[Path] = set()
    resolved_jobs: list[dict[str, object]] = []
    for job in jobs:
        job_id = _required_string(job, "job_id")
        if job_id in job_ids:
            raise SourceEffectRepairError("SOURCE_EFFECT_REPAIR_JOB_ID_DUPLICATE")
        job_ids.add(job_id)
        checkpoint = _resolve_under(repo_root, _required_string(job, "candidate_checkpoint"))
        expected_checkpoint_sha = _required_string(job, "candidate_checkpoint_sha256")
        candidate_runtime = _resolve_under(repo_root, _required_string(job, "candidate_runtime_root"))
        output_root = _resolve_under(repo_root, _required_string(job, "output_root"))
        if output_root in output_roots:
            raise SourceEffectRepairError("SOURCE_EFFECT_REPAIR_OUTPUT_DUPLICATE")
        output_roots.add(output_root)
        if not checkpoint.is_file() or sha256_file(checkpoint) != expected_checkpoint_sha:
            raise SourceEffectRepairError(f"SOURCE_EFFECT_REPAIR_CHECKPOINT_SHA_MISMATCH:{job_id}")
        if not candidate_runtime.is_dir():
            raise SourceEffectRepairError(f"SOURCE_EFFECT_REPAIR_CANDIDATE_RUNTIME_MISSING:{job_id}")
        training_seed = _required_int(job, "training_seed")
        eval_seed = _required_int(job, "eval_seed")
        resolved_jobs.append(
            {
                **job,
                "training_seed": training_seed,
                "eval_seed": eval_seed,
                "candidate_checkpoint": str(checkpoint),
                "candidate_runtime_root": str(candidate_runtime),
                "output_root": str(output_root),
            }
        )
    return {
        "config_path": str(config_file),
        "config_sha256": sha256_file(config_file),
        "repo_root": str(repo_root),
        "primitive": config["primitive"],
        "runtime_python": str(runtime_python),
        "data_root": str(data_root),
        "checkpoint_root": str(checkpoint_root),
        "candidate_gpus": list(gpus),
        "dataset_freeze": str(dataset_freeze),
        "heldout_protocol": str(heldout_protocol),
        "protocol": dict(protocol),
        "jobs": resolved_jobs,
    }


def execute_source_effect_repair_job(
    *,
    config_path: Path,
    job_id: str,
    gpu_index: int,
) -> dict[str, object]:
    plan = load_source_effect_repair_plan(config_path=config_path)
    if gpu_index not in plan["candidate_gpus"]:
        raise SourceEffectRepairError("SOURCE_EFFECT_REPAIR_GPU_NOT_ALLOWED")
    matches = [row for row in plan["jobs"] if row["job_id"] == job_id]
    if len(matches) != 1:
        raise SourceEffectRepairError("SOURCE_EFFECT_REPAIR_JOB_UNKNOWN")
    job = matches[0]
    output_root = Path(str(job["output_root"]))
    if output_root.exists() or output_root.is_symlink():
        raise SourceEffectRepairError("SOURCE_EFFECT_REPAIR_OUTPUT_EXISTS")
    protocol = plan["protocol"]
    from scripts.export.acwm_formal_visualization import run_export

    return run_export(
        output_root=output_root,
        environment=str(job["environment"]),
        primitive=str(plan["primitive"]),
        seed=int(job["eval_seed"]),
        training_seed=int(job["training_seed"]),
        runtime_python=Path(str(plan["runtime_python"])),
        data_root=Path(str(plan["data_root"])),
        checkpoint_root=Path(str(plan["checkpoint_root"])),
        dataset_freeze=Path(str(plan["dataset_freeze"])),
        heldout_protocol=Path(str(plan["heldout_protocol"])),
        candidate_checkpoint=Path(str(job["candidate_checkpoint"])),
        candidate_runtime_root=Path(str(job["candidate_runtime_root"])),
        gpu_index=gpu_index,
        steps=int(protocol["inference_steps"]),
        split=str(protocol["split"]),
        max_trajs=int(protocol["max_trajectories"]),
        max_saved_vids=int(protocol["max_saved_videos"]),
        batch_size=int(protocol["batch_size"]),
        num_workers=int(protocol["num_workers"]),
        test_cuts=int(protocol["test_cuts"]),
        hard_case_top_k=1,
    )


def settle_source_effect_repair(
    *,
    config_path: Path,
    output_root: Path,
    archive_db: Path | None = None,
    cas_root: Path | None = None,
) -> dict[str, object]:
    plan = load_source_effect_repair_plan(config_path=config_path)
    rows: list[dict[str, object]] = []
    for job in plan["jobs"]:
        manifest_path = Path(str(job["output_root"])) / "manifest.json"
        manifest = _load_json(manifest_path, "SOURCE_EFFECT_REPAIR_RECEIPT_INVALID")
        gate = manifest.get("official_quality_gate")
        provenance = manifest.get("protocol_provenance")
        checks = gate.get("checks") if isinstance(gate, Mapping) else None
        expected_protocol = {
            "dataset_freeze_sha256": sha256_file(Path(str(plan["dataset_freeze"]))),
            "heldout_protocol_sha256": sha256_file(Path(str(plan["heldout_protocol"]))),
        }
        valid = (
            manifest.get("state") == "ready"
            and manifest.get("environment") == job["environment"]
            and manifest.get("primitive") == plan["primitive"]
            and manifest.get("training_seed") == job["training_seed"]
            and manifest.get("eval_seed") == job["eval_seed"]
            and manifest.get("candidate_checkpoint_sha256") == job["candidate_checkpoint_sha256"]
            and isinstance(gate, Mapping)
            and isinstance(gate.get("pass"), bool)
            and isinstance(checks, Mapping)
            and bool(checks)
            and all(isinstance(value, bool) for value in checks.values())
            and isinstance(provenance, Mapping)
            and all(provenance.get(key) == value for key, value in expected_protocol.items())
        )
        if not valid:
            raise SourceEffectRepairError(f"SOURCE_EFFECT_REPAIR_RECEIPT_CONTRACT_FAILED:{job['job_id']}")
        delta = gate.get("delta_candidate_minus_baseline")
        rows.append(
            {
                "job_id": job["job_id"],
                "environment": job["environment"],
                "primitive": plan["primitive"],
                "training_seed": job["training_seed"],
                "eval_seed": job["eval_seed"],
                "repetition": job["repetition"],
                "candidate_checkpoint_sha256": job["candidate_checkpoint_sha256"],
                "positive": gate["pass"],
                "psnr_delta": delta.get("psnr") if isinstance(delta, Mapping) else None,
                "ssim_delta": delta.get("ssim") if isinstance(delta, Mapping) else None,
                "mse_delta": delta.get("mse") if isinstance(delta, Mapping) else None,
                "masked_mse_delta": delta.get("masked_mse") if isinstance(delta, Mapping) else None,
                "receipt_ref": str(manifest_path.resolve()),
                "receipt_sha256": sha256_file(manifest_path),
            }
        )
    groups = summarize_source_effect_repair(rows=rows)
    reproducible_positive_groups = [
        row
        for row in groups
        if row["classification"] == "consistently_positive"
        and int(row["distinct_eval_seed_count"]) >= 3
    ]
    inconsistent_groups = [
        row for row in groups if row["classification"] == "sign_inconsistent"
    ]
    report = {
        "schema_version": 1,
        "artifact_type": "verdiwm-source-effect-repair-settlement",
        "state": "ready",
        "primitive": plan["primitive"],
        "preregistration": plan["config_path"],
        "preregistration_sha256": plan["config_sha256"],
        "claim_boundary": (
            "This settlement establishes frozen-protocol source-effect reproducibility only. "
            "It does not change the transfer certificate or establish cross-backbone transfer."
        ),
        "receipt_count": len(rows),
        "group_count": len(groups),
        "all_receipts_ready": True,
        "groups": groups,
        "reproducible_positive_groups": reproducible_positive_groups,
        "inconsistent_groups": inconsistent_groups,
        "receipts": rows,
        "decisions": {
            "legacy_cloth_train875_positive_reproduced": _group_positive(groups, "cloth_move", 875),
            "push_rope_train874_positive_reproduced": _group_positive(groups, "push_rope", 874),
            "cloth_train2805_positive_reproduced": _group_positive(groups, "cloth_move", 2805),
            "robot_arm_train2830_positive_reproduced": _group_positive(groups, "robot_arm", 2830),
        },
        "next_action": (
            "Replicate only eval-stable positive groups with independent training seeds. "
            "Retain sign-inconsistent groups as uncertain and exclude them from stable-positive "
            "source support until a separately preregistered repair succeeds."
        ),
    }
    destination = Path(output_root).resolve()
    return write_bundle(
        output_root=destination,
        files={
            "source-effect-repair-settlement.json": canonical_json(report),
            "source-effect-repair-settlement.md": _settlement_markdown(report).encode("utf-8"),
            "tables/groups.csv": _rows_csv(groups).encode("utf-8"),
            "tables/receipts.csv": _rows_csv(rows).encode("utf-8"),
            "input-preregistration.json": Path(str(plan["config_path"])).read_bytes(),
        },
        manifest_fields={
            "artifact_type": "verdiwm-source-effect-repair-settlement-manifest",
            "state": "ready",
            "primitive": plan["primitive"],
            "receipt_count": len(rows),
            "group_count": len(groups),
            "report_path": str(destination / "source-effect-repair-settlement.json"),
        },
        archive_db=archive_db,
        cas_root=cas_root,
    )


def summarize_source_effect_repair(*, rows: list[Mapping[str, Any]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["environment"]),
                int(row["training_seed"]),
                str(row["candidate_checkpoint_sha256"]),
            )
        ].append(row)
    groups: list[dict[str, object]] = []
    for (environment, training_seed, checkpoint_sha), values in sorted(grouped.items()):
        signs = {bool(row["positive"]) for row in values}
        classification = (
            "consistently_positive"
            if signs == {True}
            else "consistently_negative"
            if signs == {False}
            else "sign_inconsistent"
        )
        groups.append(
            {
                "environment": environment,
                "training_seed": training_seed,
                "candidate_checkpoint_sha256": checkpoint_sha,
                "receipt_count": len(values),
                "distinct_eval_seed_count": len({int(row["eval_seed"]) for row in values}),
                "eval_seeds": sorted({int(row["eval_seed"]) for row in values}),
                "positive_count": sum(row["positive"] is True for row in values),
                "negative_count": sum(row["positive"] is False for row in values),
                "classification": classification,
            }
        )
    return groups


def _group_positive(groups: list[Mapping[str, Any]], environment: str, training_seed: int) -> bool:
    matches = [
        row
        for row in groups
        if row["environment"] == environment and row["training_seed"] == training_seed
    ]
    return len(matches) == 1 and matches[0]["classification"] == "consistently_positive"


def _rows_csv(rows: list[Mapping[str, object]]) -> str:
    fields = sorted({key for row in rows for key in row})
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                key: json.dumps(value, sort_keys=True) if isinstance(value, (list, dict)) else value
                for key, value in row.items()
            }
        )
    return output.getvalue()


def _settlement_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Source-Effect Repair Settlement",
        "",
        str(report["claim_boundary"]),
        "",
        "| Environment | Training seed | Eval seeds | Positive | Negative | Verdict |",
        "|---|---:|---|---:|---:|---|",
    ]
    for row in report["groups"]:
        lines.append(
            f"| {row['environment']} | {row['training_seed']} | "
            f"{', '.join(str(value) for value in row['eval_seeds'])} | {row['positive_count']} | "
            f"{row['negative_count']} | `{row['classification']}` |"
        )
    lines.extend(["", "## Next Action", "", str(report["next_action"]), ""])
    return "\n".join(lines)


def _resolve_under(root: Path, value: str) -> Path:
    path = (root / value).resolve() if not Path(value).is_absolute() else Path(value).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise SourceEffectRepairError(f"SOURCE_EFFECT_REPAIR_PATH_ESCAPE:{value}") from exc
    return path


def _required_string(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise SourceEffectRepairError(f"SOURCE_EFFECT_REPAIR_FIELD_INVALID:{key}")
    return value


def _required_int(row: Mapping[str, Any], key: str) -> int:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SourceEffectRepairError(f"SOURCE_EFFECT_REPAIR_FIELD_INVALID:{key}")
    return value


def _load_json(path: Path, code: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SourceEffectRepairError(f"{code}:{path}") from exc
    if not isinstance(payload, Mapping):
        raise SourceEffectRepairError(f"{code}:{path}")
    return payload
