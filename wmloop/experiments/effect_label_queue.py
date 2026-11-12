"""Build an executable queue for retained-checkpoint effect-label gates."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from wmloop.experiments._artifacts import canonical_json, write_bundle
from wmloop.experiments.acwm_fingerprint import sha256_file
from wmloop.execute.acwm_primitive_routes import primitive_execution_role
from wmloop.storage import checked_path


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
    control_python: Path | None = None,
    vendor_root: Path | None = None,
    dataset_freeze: Path | None = None,
    heldout_protocol: Path | None = None,
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

    reports = checked_path(reports_root, code="EFFECT_LABEL_QUEUE_SOURCE_SYMLINK", error=EffectLabelQueueError)
    repo = Path(repo_root).resolve()
    runtime = Path(os.path.abspath(runtime_python))
    data = Path(data_root).resolve()
    checkpoints = Path(checkpoint_root).resolve()
    # Preserve the venv interpreter path: resolving its symlink changes sys.prefix.
    control = Path(os.path.abspath(control_python or sys.executable))
    vendor = Path(vendor_root or repo / "vendor/ACWM-Phys").resolve()
    freeze = Path(dataset_freeze or repo / "runs/m0/protocol/dataset-freeze.json").resolve()
    heldout = Path(heldout_protocol or repo / "runs/m0/protocol/heldout-protocol.json").resolve()
    if not control.is_file() or not os.access(control, os.X_OK):
        raise EffectLabelQueueError("EFFECT_LABEL_QUEUE_CONTROL_PYTHON_INVALID")
    if not runtime.is_file() or not os.access(runtime, os.X_OK):
        raise EffectLabelQueueError("EFFECT_LABEL_QUEUE_RUNTIME_PYTHON_INVALID")
    required = (reports, vendor, data, checkpoints, freeze, heldout)
    for path in required:
        if not path.exists():
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
                control_python=control,
                vendor_root=vendor,
                dataset_freeze=freeze,
                heldout_protocol=heldout,
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
    destination = checked_path(output_root, code="EFFECT_LABEL_QUEUE_OUTPUT_SYMLINK", error=EffectLabelQueueError)
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
    control_python: Path,
    vendor_root: Path,
    dataset_freeze: Path,
    heldout_protocol: Path,
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

    runtime_value = source.get("candidate_runtime_root")
    candidate_runtime = checked_path(Path(runtime_value), code="EFFECT_LABEL_QUEUE_RUNTIME_SYMLINK", error=EffectLabelQueueError) if isinstance(runtime_value, str) and runtime_value else vendor_root
    if not candidate_runtime.is_dir():
        raise EffectLabelQueueError(f"EFFECT_LABEL_QUEUE_RUNTIME_MISSING:{candidate_runtime}")
    if role == "runtime_only" and candidate_runtime == vendor_root:
        raise EffectLabelQueueError(f"EFFECT_LABEL_QUEUE_RUNTIME_ONLY_NOT_MATERIALIZED:{primitive}")

    campaign_id = _campaign_id(reports_root, environment=environment, primitive=primitive, seed=seed)
    gate_root = reports_root / campaign_id
    argv = [
        str(control_python), "-m", "wmloop.experiments.formal_visualization",
        "--vendor-root", str(vendor_root),
        "--output-root", str(gate_root),
        "--environment", environment,
        "--primitive", primitive,
        "--seed", str(seed),
        "--runtime-python", str(runtime_python),
        "--data-root", str(data_root),
        "--checkpoint-root", str(checkpoint_root),
        "--dataset-freeze", str(dataset_freeze),
        "--heldout-protocol", str(heldout_protocol),
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
    path = checked_path(root / value, code="EFFECT_LABEL_QUEUE_SOURCE_SYMLINK", error=EffectLabelQueueError)
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
    path = checked_path(path, code="EFFECT_LABEL_QUEUE_SOURCE_SYMLINK", error=EffectLabelQueueError)
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completion-plan", type=Path, required=True)
    parser.add_argument("--reports-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--control-python", type=Path)
    parser.add_argument("--vendor-root", type=Path)
    parser.add_argument("--dataset-freeze", type=Path)
    parser.add_argument("--heldout-protocol", type=Path)
    parser.add_argument("--runtime-python", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs="+", required=True)
    parser.add_argument("--archive-db", type=Path)
    parser.add_argument("--cas-root", type=Path)
    args = parser.parse_args(argv)
    manifest = build_effect_label_gate_queue(
        completion_plan_path=args.completion_plan,
        reports_root=args.reports_root,
        repo_root=args.repo_root,
        output_root=args.output_root,
        runtime_python=args.runtime_python,
        control_python=args.control_python,
        vendor_root=args.vendor_root,
        dataset_freeze=args.dataset_freeze,
        heldout_protocol=args.heldout_protocol,
        data_root=args.data_root,
        checkpoint_root=args.checkpoint_root,
        gpus=args.gpus,
        archive_db=args.archive_db,
        cas_root=args.cas_root,
    )
    sys.stdout.write(json.dumps(manifest, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
