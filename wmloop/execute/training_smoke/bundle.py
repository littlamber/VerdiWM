"""Training and frozen evaluation smoke components: bundle."""

from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Any, Mapping

from wmloop.archive.store import ArchiveStore, ContentAddressedStore
from wmloop.execute.training_monitor_policy import checkpoint_eval_ladder
from wmloop.execute.training_smoke.errors import OrchestratorTrainingEvalSmokeError
from wmloop.execute.training_smoke.io import (
    _canonical_json_bytes,
    _load_json_object,
    _write_bytes_atomic,
)
from wmloop.execute.training_smoke.runtime import _safe_run_name
from wmloop.runtime_contract import runtime_tree_sha256
from wmloop.storage import sha256_file


def _write_report_bundle(
    *,
    report: Mapping[str, object],
    output_root: Path,
    cas: ContentAddressedStore,
    archive: ArchiveStore | None,
    budget_db: Path,
) -> dict[str, object]:
    destination = Path(output_root).resolve()
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    try:
        temporary.mkdir(mode=0o700)
        public_report = _bundle_evaluation_files(
            report=report, temporary=temporary, destination=destination
        )
        report_bytes = _canonical_json_bytes(public_report)
        markdown_bytes = _render_markdown(public_report).encode("utf-8")
        _write_bytes_atomic(
            temporary / "orchestrator-training-eval-smoke.json", report_bytes
        )
        _write_bytes_atomic(
            temporary / "orchestrator-training-eval-smoke.md", markdown_bytes
        )
        shutil.copy2(budget_db, temporary / "budget.db")
        report_ref = cas.put_bytes(report_bytes, media_type="application/json").uri
        markdown_ref = cas.put_bytes(markdown_bytes, media_type="text/markdown").uri
        budget_ref = cas.put_bytes(
            budget_db.read_bytes(), media_type="application/x-sqlite3"
        ).uri
        if archive is not None:
            for ref in (report_ref, markdown_ref, budget_ref):
                archive.record_artifact_reference(ref)
        manifest = {
            "schema_version": 1,
            "artifact_type": "wmloop-m3-training-eval-smoke-manifest",
            "state": public_report["state"],
            "environment": public_report["environment"],
            "goal_id": public_report["goal_id"],
            "primary_metric": public_report.get("primary_metric"),
            "proposal_id": public_report["proposal_id"],
            **(
                {"trial_arm": public_report["trial_arm"]}
                if "trial_arm" in public_report
                else {}
            ),
            **({"seed": public_report["seed"]} if "seed" in public_report else {}),
            "verdict": public_report["verdict"],
            "action_following_gate": public_report["action_following_gate"],
            "delta_m_ver": public_report["delta_m_ver"],
            "gpu_exclusivity_audit": public_report.get("gpu_exclusivity_audit"),
            "formal_settled_trial_published": public_report.get(
                "formal_settled_trial_published", False
            ),
            "candidate_checkpoint_retained": public_report.get("receipt", {}).get(
                "candidate_checkpoint_retained", False
            ),
            "candidate_checkpoint_retained_path": public_report.get("receipt", {}).get(
                "candidate_checkpoint_retained_path"
            ),
            "candidate_checkpoint_sha256": public_report.get("receipt", {}).get(
                "candidate_checkpoint_sha256"
            ),
            "candidate_runtime_retained": public_report.get("receipt", {}).get(
                "candidate_runtime_retained", False
            ),
            "candidate_runtime_root": public_report.get("receipt", {}).get(
                "candidate_runtime_root"
            ),
            "candidate_runtime_sha256": public_report.get("receipt", {}).get(
                "candidate_runtime_sha256"
            ),
            "report_path": str(destination / "orchestrator-training-eval-smoke.json"),
            "markdown_path": str(destination / "orchestrator-training-eval-smoke.md"),
            "budget_db_path": str(destination / "budget.db"),
            "evaluation_dir": str(destination / "evaluation"),
            "retained_training_dir": str(destination / "retained_training"),
            "retained_runtime_dir": str(destination / "retained_runtime"),
            "cas_refs": {
                "orchestrator_training_eval_smoke_json": report_ref,
                "orchestrator_training_eval_smoke_markdown": markdown_ref,
                "budget_db": budget_ref,
            },
            "limitations": public_report["limitations"],
        }
        _write_bytes_atomic(
            temporary / "manifest.json", _canonical_json_bytes(manifest)
        )
        os.replace(temporary, destination)
        return manifest
    except Exception:
        if temporary.is_symlink():
            temporary.unlink()
        elif temporary.exists():
            shutil.rmtree(temporary)
        raise


def _bundle_evaluation_files(
    *,
    report: Mapping[str, object],
    temporary: Path,
    destination: Path,
) -> dict[str, Any]:
    public = json.loads(_canonical_json_bytes(report))
    receipt = public.get("receipt")
    if not isinstance(receipt, dict):
        raise OrchestratorTrainingEvalSmokeError(
            "M3_TRAINING_EVAL_SMOKE_REPORT_INVALID"
        )
    evaluation = receipt.get("evaluation")
    if not isinstance(evaluation, dict):
        raise OrchestratorTrainingEvalSmokeError(
            "M3_TRAINING_EVAL_SMOKE_REPORT_INVALID"
        )
    final_dir = destination / "evaluation"
    bundle_dir = temporary / "evaluation"
    bundle_dir.mkdir(mode=0o700, parents=True)
    path_keys = {
        "accept_cohort_view_manifest_path": "accept-cohort-view-manifest.json",
        "baseline_manifest_path": "baseline-horizon-probe-manifest.json",
        "candidate_manifest_path": "candidate-horizon-probe-manifest.json",
        "baseline_metrics_path": "baseline-horizon-probe-metrics.json",
        "candidate_metrics_path": "candidate-horizon-probe-metrics.json",
    }
    for key, name in path_keys.items():
        value = evaluation.get(key)
        if not isinstance(value, str) or not value:
            raise OrchestratorTrainingEvalSmokeError(
                "M3_TRAINING_EVAL_SMOKE_REPORT_INVALID"
            )
        source = Path(value)
        if source.is_symlink() or not source.is_file():
            raise OrchestratorTrainingEvalSmokeError(
                "M3_TRAINING_EVAL_SMOKE_EVALUATION_ARTIFACT_MISSING"
            )
        shutil.copy2(source, bundle_dir / name)
        evaluation[key] = str(final_dir / name)
    replications = evaluation.get("replications")
    if isinstance(replications, list):
        for index, replication in enumerate(replications, start=1):
            if not isinstance(replication, dict):
                raise OrchestratorTrainingEvalSmokeError(
                    "M3_TRAINING_EVAL_SMOKE_REPORT_INVALID"
                )
            label = str(replication.get("label") or f"r{index:02d}")
            safe_label = _safe_run_name(label)
            replicate_bundle_dir = bundle_dir / safe_label
            replicate_final_dir = final_dir / safe_label
            replicate_bundle_dir.mkdir(mode=0o700)
            for key, name in path_keys.items():
                value = replication.get(key)
                if not isinstance(value, str) or not value:
                    raise OrchestratorTrainingEvalSmokeError(
                        "M3_TRAINING_EVAL_SMOKE_REPORT_INVALID"
                    )
                source = Path(value)
                if source.is_symlink() or not source.is_file():
                    raise OrchestratorTrainingEvalSmokeError(
                        "M3_TRAINING_EVAL_SMOKE_EVALUATION_ARTIFACT_MISSING"
                    )
                shutil.copy2(source, replicate_bundle_dir / name)
                replication[key] = str(replicate_final_dir / name)
    action_following = evaluation.get("action_following")
    if isinstance(action_following, dict):
        _bundle_action_following_files(
            action_following=action_following,
            bundle_dir=bundle_dir,
            final_dir=final_dir,
        )
    _bundle_candidate_checkpoint(
        receipt=receipt, temporary=temporary, destination=destination, public=public
    )
    # Older synthetic receipts in the unit suite predate runtime-bound evidence.
    # Real training/eval receipts always carry both runtime fields and therefore
    # still fail closed if the candidate runtime cannot be retained.
    if "candidate_runtime_path" in receipt or "candidate_runtime_sha256" in receipt:
        _bundle_candidate_runtime(
            receipt=receipt, temporary=temporary, destination=destination
        )
    return public


def _bundle_action_following_files(
    *, action_following: dict[str, object], bundle_dir: Path, final_dir: Path
) -> None:
    path_keys = {
        "manifest_path": "manifest.json",
        "measurements_path": "measurements.json",
    }
    for side in ("baseline", "candidate"):
        payload = action_following.get(side)
        if not isinstance(payload, dict):
            raise OrchestratorTrainingEvalSmokeError(
                "M3_TRAINING_EVAL_SMOKE_REPORT_INVALID"
            )
        side_bundle = bundle_dir / "action-following" / side
        side_final = final_dir / "action-following" / side
        side_bundle.mkdir(mode=0o700, parents=True)
        for key, name in path_keys.items():
            value = payload.get(key)
            if not isinstance(value, str) or not value:
                raise OrchestratorTrainingEvalSmokeError(
                    "M3_TRAINING_EVAL_SMOKE_REPORT_INVALID"
                )
            source = Path(value)
            if source.is_symlink() or not source.is_file():
                raise OrchestratorTrainingEvalSmokeError(
                    "M3_TRAINING_EVAL_SMOKE_EVALUATION_ARTIFACT_MISSING"
                )
            shutil.copy2(source, side_bundle / name)
            payload[key] = str(side_final / name)


def _bundle_candidate_checkpoint(
    *,
    receipt: dict[str, object],
    temporary: Path,
    destination: Path,
    public: dict[str, Any],
) -> None:
    source_value = receipt.get("candidate_checkpoint_path")
    if not isinstance(source_value, str) or not source_value:
        raise OrchestratorTrainingEvalSmokeError(
            "M3_TRAINING_EVAL_SMOKE_REPORT_INVALID"
        )
    source = Path(source_value)
    if source.is_symlink() or not source.is_file():
        raise OrchestratorTrainingEvalSmokeError(
            "M3_TRAINING_EVAL_SMOKE_CANDIDATE_CHECKPOINT_MISSING"
        )
    retained_dir = temporary / "retained_training"
    retained_dir.mkdir(mode=0o700, parents=True)
    retained_path = retained_dir / "latest.pt"
    latest_storage_mode = _link_or_copy_checkpoint(source, retained_path)
    digest = _file_sha256(retained_path)
    byte_count = retained_path.stat().st_size
    final_path = destination / "retained_training" / "latest.pt"
    training_scale = receipt.get("training_scale")
    train_steps = (
        training_scale.get("train_steps")
        if isinstance(training_scale, Mapping)
        else None
    )
    ladder = _bundle_checkpoint_ladder(
        source=source,
        retained_dir=retained_dir,
        destination=destination,
        train_steps=train_steps,
    )
    manifest = {
        "schema_version": 1,
        "artifact_type": "wmloop-retained-training-checkpoint",
        "source_path": str(source),
        "retained_path": str(final_path),
        "sha256": digest,
        "bytes": byte_count,
        "storage_mode": latest_storage_mode,
        "checkpoint_ladder": ladder,
    }
    _write_bytes_atomic(retained_dir / "manifest.json", _canonical_json_bytes(manifest))
    receipt["candidate_checkpoint_retained"] = True
    receipt["candidate_checkpoint_retained_path"] = str(final_path)
    receipt["candidate_checkpoint_retained_manifest_path"] = str(
        destination / "retained_training" / "manifest.json"
    )
    receipt["candidate_checkpoint_sha256"] = digest
    receipt["candidate_checkpoint_bytes"] = byte_count
    receipt["candidate_checkpoint_ladder"] = ladder
    limitations = public.get("limitations")
    if isinstance(limitations, list):
        public["limitations"] = [
            item
            for item in limitations
            if not (
                isinstance(item, str)
                and "candidate checkpoint" in item
                and "not retained" in item
            )
        ]
        public["limitations"].append(
            "The final candidate and configured checkpoint ladder are retained under retained_training; formal claims still require frozen official evaluator receipts."
        )


def _bundle_candidate_runtime(
    *, receipt: dict[str, object], temporary: Path, destination: Path
) -> None:
    source_value = receipt.get("candidate_runtime_path")
    expected_sha = receipt.get("candidate_runtime_sha256")
    if (
        not isinstance(source_value, str)
        or not source_value
        or not isinstance(expected_sha, str)
        or not expected_sha
    ):
        raise OrchestratorTrainingEvalSmokeError(
            "M3_TRAINING_EVAL_SMOKE_CANDIDATE_RUNTIME_MISSING"
        )
    source = Path(source_value)
    if source.is_symlink() or not source.is_dir():
        raise OrchestratorTrainingEvalSmokeError(
            "M3_TRAINING_EVAL_SMOKE_CANDIDATE_RUNTIME_MISSING"
        )
    retained = temporary / "retained_runtime"
    shutil.copytree(source, retained)
    actual_sha = runtime_tree_sha256(retained)
    if actual_sha != expected_sha:
        raise OrchestratorTrainingEvalSmokeError(
            "M3_TRAINING_EVAL_SMOKE_CANDIDATE_RUNTIME_HASH_MISMATCH"
        )
    manifest_path = retained / "wmloop-runtime-manifest.json"
    manifest = _load_json_object(manifest_path)
    manifest["runtime_path"] = str(destination / "retained_runtime")
    manifest["tree_sha256"] = actual_sha
    _write_bytes_atomic(manifest_path, _canonical_json_bytes(manifest))
    receipt["candidate_runtime_retained"] = True
    receipt["candidate_runtime_root"] = str(destination / "retained_runtime")
    receipt["candidate_runtime_manifest_path"] = str(
        destination / "retained_runtime" / "wmloop-runtime-manifest.json"
    )
    receipt["candidate_runtime_sha256"] = actual_sha


def _bundle_checkpoint_ladder(
    *,
    source: Path,
    retained_dir: Path,
    destination: Path,
    train_steps: object,
) -> list[dict[str, object]]:
    if (
        isinstance(train_steps, bool)
        or not isinstance(train_steps, int)
        or train_steps < 1
    ):
        return []
    checkpoint_files: dict[int, Path] = {}
    for checkpoint in source.parent.glob("checkpoint_*.pt"):
        match = re.fullmatch(r"checkpoint_(\d+)\.pt", checkpoint.name)
        if match is not None and checkpoint.is_file() and not checkpoint.is_symlink():
            checkpoint_files[int(match.group(1))] = checkpoint
    final_absolute_step = max(checkpoint_files, default=-1)
    base_step = (
        final_absolute_step - train_steps
        if final_absolute_step >= train_steps
        else None
    )
    records: list[dict[str, object]] = []
    ladder_dir = retained_dir / "checkpoints"
    ladder_dir.mkdir(mode=0o700)
    for relative_step in checkpoint_eval_ladder(train_steps):
        absolute_step = base_step + relative_step if base_step is not None else None
        checkpoint_source = (
            checkpoint_files.get(absolute_step) if absolute_step is not None else None
        )
        if checkpoint_source is None and relative_step == train_steps:
            checkpoint_source = source
        if checkpoint_source is None:
            records.append(
                {
                    "relative_step": relative_step,
                    "absolute_step": absolute_step,
                    "state": "checkpoint_missing",
                    "retained_path": None,
                }
            )
            continue
        retained_name = f"relative_step_{relative_step:06d}.pt"
        retained_checkpoint = ladder_dir / retained_name
        storage_mode = _link_or_copy_checkpoint(checkpoint_source, retained_checkpoint)
        records.append(
            {
                "relative_step": relative_step,
                "absolute_step": absolute_step,
                "state": "retained",
                "source_path": str(checkpoint_source),
                "retained_path": str(
                    destination / "retained_training" / "checkpoints" / retained_name
                ),
                "bytes": retained_checkpoint.stat().st_size,
                "storage_mode": storage_mode,
                "sha256": None,
                "hash_state": "deferred_until_official_gate",
            }
        )
    return records


def _link_or_copy_checkpoint(source: Path, destination: Path) -> str:
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def _file_sha256(path: Path) -> str:
    return sha256_file(path)


def _render_markdown(report: Mapping[str, Any]) -> str:
    receipt = report["receipt"]
    evaluation = receipt["evaluation"]
    metric_name = str(evaluation.get("primary_metric") or "auc_psnr_16_64")
    rows = []
    for attempt in receipt["candidate"]["receipts"]:
        rows.append(
            f"| {attempt['label']} | {attempt['passed']} | {attempt['timed_out']} | "
            f"{attempt['exit_code']} | {attempt['duration_seconds']:.3f} |"
        )
    lines = [
        f"# M3 {report['environment']} Training+Eval Smoke",
        "",
        f"State: `{report['state']}`",
        f"Proposal: `{report['proposal_id']}`",
        f"GPU index: `{report['gpu_index']}`",
        f"Verdict: `{report['verdict']}`",
        f"Primary metric: `{metric_name}`",
        f"Delta {metric_name}: `{evaluation.get('delta_primary_metric', evaluation['delta_auc_psnr_16_64'])}`",
        f"Replication count: `{evaluation.get('replication_count', 1)}`",
        f"Replication deltas: `{evaluation.get('replication_deltas', [evaluation['delta_auc_psnr_16_64']])}`",
        f"Action following observed: `{evaluation.get('action_following', {}).get('gate_observed')}`",
        f"Action following threshold: `{evaluation.get('action_following', {}).get('gate_threshold')}`",
        f"Settlement: `{report['settlement_state']}`",
        f"Worktree removed: `{receipt['worktree_removed']}`",
        "",
        "| Check | Passed | Timed out | Exit code | Seconds |",
        "|:--|:--|:--|:--|--:|",
        *rows,
        "",
        "## Limitations",
        "",
    ]
    for item in report["limitations"]:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"
