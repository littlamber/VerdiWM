"""Build an executable queue for retained-checkpoint effect-label gates."""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from wmloop.experiments._artifacts import canonical_json, write_bundle
from wmloop.experiments.acwm_fingerprint import sha256_file
from wmloop.execute.acwm_primitive_routes import primitive_execution_role


class EffectLabelQueueError(ValueError):
    """The completion plan cannot be converted into a trustworthy queue."""


def build_effect_label_gate_queue(
    *,
    completion_plan_path: Path,
    reports_root: Path,
    repo_root: Path,
    output_root: Path,
    runtime_python: Path,
    data_root: Path,
    checkpoint_root: Path,
    gpus: Sequence[int],
    archive_db: Path | None = None,
    cas_root: Path | None = None,
) -> dict[str, object]:
    plan = _load_json(Path(completion_plan_path))
    if plan.get("artifact_type") != "verdiwm-acwm-effect-label-completion-plan":
        raise EffectLabelQueueError("EFFECT_LABEL_QUEUE_PLAN_TYPE_INVALID")
    actions = plan.get("actions")
    if not isinstance(actions, list) or any(not isinstance(row, Mapping) for row in actions):
        raise EffectLabelQueueError("EFFECT_LABEL_QUEUE_ACTIONS_INVALID")
    gpu_list = tuple(int(value) for value in gpus)
    if not gpu_list or len(set(gpu_list)) != len(gpu_list) or any(value < 0 for value in gpu_list):
        raise EffectLabelQueueError("EFFECT_LABEL_QUEUE_GPUS_INVALID")

    reports = Path(reports_root).resolve()
    repo = Path(repo_root).resolve()
    runtime = Path(runtime_python).resolve()
    data = Path(data_root).resolve()
    checkpoints = Path(checkpoint_root).resolve()
    required = (reports, repo / "vendor/ACWM-Phys", runtime, data, checkpoints)
    for path in required:
        if path.is_symlink() or not path.exists():
            raise EffectLabelQueueError(f"EFFECT_LABEL_QUEUE_REQUIRED_PATH_MISSING:{path}")

    rows: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    for action in actions:
        if action.get("action") != "official_gate_existing_checkpoint":
            skipped.append(
                {
                    "ordinal": action.get("ordinal"),
                    "environment": action.get("environment"),
                    "primitive": action.get("primitive"),
                    "reason": "NEW_SCREEN_REQUIRED",
                }
            )
            continue
        rows.append(
            _queue_row(
                action=action,
                rank=len(rows) + 1,
                reports_root=reports,
                repo_root=repo,
                runtime_python=runtime,
                data_root=data,
                checkpoint_root=checkpoints,
                gpus=gpu_list,
                archive_db=archive_db,
                cas_root=cas_root,
            )
        )
    report = {
        "schema_version": 1,
        "artifact_type": "verdiwm-acwm-effect-label-gate-queue",
        "state": "ready" if rows else "blocked",
        "completion_plan": str(Path(completion_plan_path).resolve()),
        "claim_boundary": (
            "Queue execution creates settled target-local selector labels only. "
            "It does not promote a primitive or establish cross-backbone transfer."
        ),
        "row_count": len(rows),
        "skipped_action_count": len(skipped),
        "candidate_gpus": list(gpu_list),
        "rows": rows,
        "skipped_actions": skipped,
    }
    destination = Path(output_root).resolve()
    return write_bundle(
        output_root=destination,
        files={
            "autoloop-queue.json": canonical_json(report),
            "effect-label-gate-queue.md": _markdown(report).encode("utf-8"),
            "tables/queue.csv": _csv(rows).encode("utf-8"),
            "input-completion-plan.json": canonical_json(plan),
        },
        manifest_fields={
            "artifact_type": "verdiwm-acwm-effect-label-gate-queue-manifest",
            "state": report["state"],
            "row_count": len(rows),
            "skipped_action_count": len(skipped),
            "queue_path": str(destination / "autoloop-queue.json"),
        },
        archive_db=archive_db,
        cas_root=cas_root,
    )


def _queue_row(
    *,
    action: Mapping[str, Any],
    rank: int,
    reports_root: Path,
    repo_root: Path,
    runtime_python: Path,
    data_root: Path,
    checkpoint_root: Path,
    gpus: Sequence[int],
    archive_db: Path | None,
    cas_root: Path | None,
) -> dict[str, object]:
    environment = _required_string(action, "environment")
    primitive = _required_string(action, "primitive")
    seed = _required_int(action, "seed")
    role = primitive_execution_role(primitive)
    if role not in {"quality_screen", "runtime_only"}:
        raise EffectLabelQueueError(f"EFFECT_LABEL_QUEUE_PRIMITIVE_NOT_ADMISSIBLE:{primitive}:{role}")
    source_manifest = _resolve_under(reports_root, _required_string(action, "source_manifest"))
    candidate_checkpoint = _resolve_under(reports_root, _required_string(action, "checkpoint_ref"))
    if not source_manifest.is_file() or candidate_checkpoint.is_symlink() or not candidate_checkpoint.is_file():
        raise EffectLabelQueueError(f"EFFECT_LABEL_QUEUE_SOURCE_MISSING:{environment}:{primitive}")
    source = _load_json(source_manifest)
    if source.get("state") != "ready" or str(source.get("environment")) != environment:
        raise EffectLabelQueueError(f"EFFECT_LABEL_QUEUE_SOURCE_NOT_READY:{source_manifest}")
    expected_sha = _required_string(action, "checkpoint_sha256")
    retained_manifest = candidate_checkpoint.parent / "manifest.json"
    retained = _load_json(retained_manifest)
    if retained.get("sha256") != expected_sha or Path(str(retained.get("retained_path") or "")).resolve() != candidate_checkpoint:
        raise EffectLabelQueueError(f"EFFECT_LABEL_QUEUE_RETAINED_MANIFEST_MISMATCH:{candidate_checkpoint}")
    observed_sha = sha256_file(candidate_checkpoint)
    if observed_sha != expected_sha:
        raise EffectLabelQueueError(f"EFFECT_LABEL_QUEUE_CHECKPOINT_SHA_MISMATCH:{candidate_checkpoint}")

    vendor_root = repo_root / "vendor/ACWM-Phys"
    runtime_value = source.get("candidate_runtime_root")
    candidate_runtime = Path(str(runtime_value)).resolve() if isinstance(runtime_value, str) and runtime_value else vendor_root
    if candidate_runtime.is_symlink() or not candidate_runtime.is_dir():
        raise EffectLabelQueueError(f"EFFECT_LABEL_QUEUE_RUNTIME_MISSING:{candidate_runtime}")
    if role == "runtime_only" and candidate_runtime == vendor_root:
        raise EffectLabelQueueError(f"EFFECT_LABEL_QUEUE_RUNTIME_ONLY_NOT_MATERIALIZED:{primitive}")

    campaign_id = _campaign_id(reports_root, environment=environment, primitive=primitive, seed=seed)
    gate_root = reports_root / campaign_id
    argv = [
        str(repo_root / ".venv/bin/python3"),
        str(repo_root / "scripts/export/acwm_formal_visualization.py"),
        "--output-root", str(gate_root),
        "--environment", environment,
        "--primitive", primitive,
        "--seed", str(seed),
        "--runtime-python", str(runtime_python),
        "--data-root", str(data_root),
        "--checkpoint-root", str(checkpoint_root),
        "--dataset-freeze", str(repo_root / "runs/m0/protocol/dataset-freeze.json"),
        "--heldout-protocol", str(repo_root / "runs/m0/protocol/heldout-protocol.json"),
        "--candidate-checkpoint", str(candidate_checkpoint),
        "--candidate-runtime-root", str(candidate_runtime),
        "--gpu-index", "{gpu}",
        "--steps", "50",
        "--split", "ind_test",
        "--max-trajs", "3",
        "--max-saved-vids", "3",
        "--batch-size", "1",
        "--num-workers", "2",
        "--test-cuts", "1",
        "--hard-case-top-k", "1",
    ]
    if role == "runtime_only":
        argv.append("--require-candidate-runtime-hook")
    return {
        "rank": rank,
        "phase": "effect_label_official_gate",
        "campaign_id": campaign_id,
        "environment": environment,
        "primitive": primitive,
        "execution_role": role,
        "seed": seed,
        "train_steps": 0,
        "output_root": str(gate_root),
        "candidate_gpus": list(gpus),
        "allow_any_idle_gpu": False,
        "requires_positive_manifest": "",
        "requires_ready_manifest": str(source_manifest),
        "requires_official_quality_manifest": "",
        "source_screen_manifest": str(source_manifest),
        "candidate_checkpoint": str(candidate_checkpoint),
        "candidate_checkpoint_sha256": observed_sha,
        "candidate_runtime_root": str(candidate_runtime),
        "archive_db": str(archive_db or repo_root / "results/archive.db"),
        "cas_root": str(cas_root or repo_root / "results"),
        "gpu_audit_root_template": str(
            reports_root / f"gpu-exclusivity-audit-{campaign_id}-gpu{{gpu}}-{{attempt_id}}"
        ),
        "launch_argv_template": argv,
    }


def _campaign_id(reports_root: Path, *, environment: str, primitive: str, seed: int) -> str:
    stem = f"acwm-effect-label-gate-{environment}-{primitive}-s{seed}"
    revision = 1
    while (reports_root / f"{stem}-r{revision}").exists() or (reports_root / f"{stem}-r{revision}.launching").exists():
        revision += 1
    return f"{stem}-r{revision}"


def _resolve_under(root: Path, value: str) -> Path:
    path = (root / value).resolve() if not Path(value).is_absolute() else Path(value).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise EffectLabelQueueError(f"EFFECT_LABEL_QUEUE_PATH_ESCAPE:{value}") from exc
    return path


def _required_string(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise EffectLabelQueueError(f"EFFECT_LABEL_QUEUE_FIELD_INVALID:{key}")
    return value


def _required_int(row: Mapping[str, Any], key: str) -> int:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise EffectLabelQueueError(f"EFFECT_LABEL_QUEUE_FIELD_INVALID:{key}")
    return value


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EffectLabelQueueError(f"EFFECT_LABEL_QUEUE_JSON_INVALID:{path}") from exc
    if not isinstance(payload, Mapping):
        raise EffectLabelQueueError(f"EFFECT_LABEL_QUEUE_JSON_INVALID:{path}")
    return payload


def _csv(rows: Sequence[Mapping[str, object]]) -> str:
    fields = ["rank", "campaign_id", "environment", "primitive", "execution_role", "seed", "candidate_checkpoint_sha256", "output_root"]
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows({field: row.get(field) for field in fields} for row in rows)
    return output.getvalue()


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# ACWM Effect-Label Gate Queue",
        "",
        str(report["claim_boundary"]),
        "",
        f"State: `{report['state']}`",
        f"Official gate rows: `{report['row_count']}`",
        f"Skipped actions: `{report['skipped_action_count']}`",
        "",
        "| Rank | Environment | Primitive | Seed |",
        "|---:|---|---|---:|",
    ]
    for row in report["rows"]:
        lines.append(f"| {row['rank']} | {row['environment']} | {row['primitive']} | {row['seed']} |")
    return "\n".join(lines) + "\n"
