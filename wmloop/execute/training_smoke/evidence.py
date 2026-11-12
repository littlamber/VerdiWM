"""Training and frozen evaluation smoke components: evidence."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from wmloop.archive.store import ArchiveStore, ContentAddressedStore
from wmloop.diagnose.probes.action_following import (
    no_action_delta_psnr,
    per_frame_inverse_dynamics_accuracy,
)
from wmloop.evaluate.plan import build_baseline_evaluation_plan
from wmloop.execute.training_smoke.errors import OrchestratorTrainingEvalSmokeError
from wmloop.execute.training_smoke.io import (
    _canonical_json_bytes,
    _load_json_object,
    _write_bytes_atomic,
)
from wmloop.execute.training_smoke.metrics import (
    _auc_value,
    _environment_spec,
    _finite_float,
    _has_required_horizons,
    _replication_delta,
    _replication_value,
)
from wmloop.orchestrator_training_smoke import _put_file
from wmloop.verify.judge import VerificationEvidence


def _verification_evidence_from_probe_manifests(
    *,
    proposal_id: str,
    baseline_manifest: Mapping[str, object],
    candidate_manifest: Mapping[str, object],
    diff_audit_passed: bool,
    required_horizons: Sequence[int],
    metric_name: str = "auc_psnr_16_64",
) -> VerificationEvidence:
    baseline_auc = _auc_value(baseline_manifest, horizons=required_horizons)
    candidate_auc = _auc_value(candidate_manifest, horizons=required_horizons)
    delta = candidate_auc - baseline_auc
    extended = _has_required_horizons(
        baseline_manifest, required_horizons
    ) and _has_required_horizons(candidate_manifest, required_horizons)
    complete = (
        baseline_manifest.get("state") == "ready"
        and candidate_manifest.get("state") == "ready"
        and extended
        and math.isfinite(delta)
    )
    return VerificationEvidence(
        proposal_id=proposal_id,
        readonly_evaluator_verified=True,
        accept_split_verified=True,
        extended_horizon_verified=extended,
        diff_audit_passed=diff_audit_passed,
        evidence_complete=complete,
        accept_metric_deltas={metric_name: delta},
        replication_deltas=[delta],
        action_following_observed=0.0,
        action_following_threshold=0.0,
    )


def _verification_evidence_from_probe_replications(
    *,
    proposal_id: str,
    replications: Sequence[Mapping[str, object]],
    diff_audit_passed: bool,
    required_horizons: Sequence[int],
    metric_name: str = "auc_psnr_16_64",
    action_following: Mapping[str, object] | None = None,
    action_following_threshold: float | None = 0.0,
) -> VerificationEvidence:
    if not replications:
        raise OrchestratorTrainingEvalSmokeError(
            "M3_TRAINING_EVAL_SMOKE_REPLICATIONS_EMPTY"
        )
    deltas = [
        _replication_delta(item, metric_name=metric_name) for item in replications
    ]
    mean_delta = sum(deltas) / len(deltas)
    extended = all(
        bool(item.get("required_horizons_complete")) for item in replications
    )
    complete = extended and all(math.isfinite(value) for value in deltas)
    action_enabled = action_following_threshold is not None
    action_observed = (
        _action_following_observed(action_following) if action_enabled else None
    )
    return VerificationEvidence(
        proposal_id=proposal_id,
        readonly_evaluator_verified=True,
        accept_split_verified=True,
        extended_horizon_verified=extended,
        diff_audit_passed=diff_audit_passed,
        evidence_complete=complete,
        accept_metric_deltas={metric_name: mean_delta},
        replication_deltas=deltas,
        action_following_observed=action_observed,
        action_following_threshold=action_following_threshold,
        action_following_gate_enabled=action_enabled,
    )


def _write_accept_cohort_view_inputs(
    *,
    output_root: Path,
    heldout_protocol: Mapping[str, object],
    dataset_freeze: Mapping[str, object],
    environment: str = "push_cube",
    max_accept_trajectories: int,
) -> dict[str, object]:
    _environment_spec(environment)
    plan = build_baseline_evaluation_plan(
        dataset_freeze, heldout_protocol, gpus=(0, 1, 2)
    )
    selection = plan.selection(environment, "ind_accept")
    selected = list(selection.trajectory_ids[:max_accept_trajectories])
    if not selected:
        raise OrchestratorTrainingEvalSmokeError(
            "M3_TRAINING_EVAL_SMOKE_ACCEPT_SELECTION_EMPTY"
        )
    root = Path(output_root) / "accept-cohort"
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    ids_path = root / "trajectory-ids.json"
    _write_bytes_atomic(
        ids_path,
        json.dumps(
            selected,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n",
    )
    return {
        "view_root": root / "view",
        "trajectory_ids_path": ids_path,
        "trajectory_count": len(selected),
        "dataset_freeze_sha256": plan.dataset_freeze_sha256,
        "heldout_protocol_sha256": plan.heldout_protocol_sha256,
    }


def _write_accept_replicate_inputs(
    *,
    output_root: Path,
    heldout_protocol: Mapping[str, object],
    dataset_freeze: Mapping[str, object],
    environment: str = "push_cube",
    max_accept_trajectories: int,
    replication_count: int,
) -> tuple[dict[str, object], ...]:
    _environment_spec(environment)
    plan = build_baseline_evaluation_plan(
        dataset_freeze, heldout_protocol, gpus=(0, 1, 2)
    )
    selection = plan.selection(environment, "ind_accept")
    required = max_accept_trajectories * replication_count
    selected_all = list(selection.trajectory_ids[:required])
    if len(selected_all) != required:
        raise OrchestratorTrainingEvalSmokeError(
            "M3_TRAINING_EVAL_SMOKE_ACCEPT_SELECTION_INSUFFICIENT"
        )
    root = Path(output_root) / "accept-cohorts"
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    replicates: list[dict[str, object]] = []
    for index in range(replication_count):
        label = f"r{index + 1:02d}"
        start = index * max_accept_trajectories
        selected = selected_all[start : start + max_accept_trajectories]
        replicate_root = root / label
        replicate_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        ids_path = replicate_root / "trajectory-ids.json"
        _write_bytes_atomic(
            ids_path,
            json.dumps(
                selected,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            + b"\n",
        )
        replicates.append(
            {
                "index": index + 1,
                "label": label,
                "seed": 101 + index,
                "trajectory_ids": selected,
                "trajectory_ids_path": ids_path,
                "trajectory_count": len(selected),
                "view_root": replicate_root / "view",
                "baseline_probe_root": Path(output_root)
                / f"baseline-horizon-probe-{label}",
                "candidate_probe_root": Path(output_root)
                / f"candidate-horizon-probe-{label}",
                "dataset_freeze_sha256": plan.dataset_freeze_sha256,
                "heldout_protocol_sha256": plan.heldout_protocol_sha256,
            }
        )
    return tuple(replicates)


def _load_probe_replications(
    *,
    replicates: Sequence[Mapping[str, object]],
    required_horizons: Sequence[int],
    metric_name: str = "auc_psnr_16_64",
) -> tuple[dict[str, object], ...]:
    records: list[dict[str, object]] = []
    for replicate in replicates:
        baseline_root = Path(str(replicate["baseline_probe_root"]))
        candidate_root = Path(str(replicate["candidate_probe_root"]))
        baseline_manifest = _load_json_object(baseline_root / "manifest.json")
        candidate_manifest = _load_json_object(candidate_root / "manifest.json")
        baseline_auc = _auc_value(baseline_manifest, horizons=required_horizons)
        candidate_auc = _auc_value(candidate_manifest, horizons=required_horizons)
        delta = candidate_auc - baseline_auc
        records.append(
            {
                "index": int(replicate["index"]),
                "label": str(replicate["label"]),
                "seed": int(replicate["seed"]),
                "trajectory_ids": list(replicate["trajectory_ids"]),  # type: ignore[arg-type]
                "trajectory_count": int(replicate["trajectory_count"]),
                "primary_metric": metric_name,
                "baseline_primary_metric": baseline_auc,
                "candidate_primary_metric": candidate_auc,
                "delta_primary_metric": delta,
                f"baseline_{metric_name}": baseline_auc,
                f"candidate_{metric_name}": candidate_auc,
                f"delta_{metric_name}": delta,
                "baseline_auc_psnr_16_64": baseline_auc,
                "candidate_auc_psnr_16_64": candidate_auc,
                "delta_auc_psnr_16_64": delta,
                "required_horizons_complete": _has_required_horizons(
                    baseline_manifest, required_horizons
                )
                and _has_required_horizons(candidate_manifest, required_horizons),
                "accept_cohort_view_manifest_path": str(
                    Path(str(replicate["view_root"])) / "cohort-view-manifest.json"
                ),
                "baseline_manifest_path": str(baseline_root / "manifest.json"),
                "candidate_manifest_path": str(candidate_root / "manifest.json"),
                "baseline_metrics_path": str(baseline_root / "metrics.json"),
                "candidate_metrics_path": str(candidate_root / "metrics.json"),
            }
        )
    return tuple(records)


def _load_action_following_probe_pair(
    *, baseline_root: Path, candidate_root: Path
) -> dict[str, object]:
    baseline = _load_action_following_measurement(
        Path(baseline_root) / "measurements.json"
    )
    candidate = _load_action_following_measurement(
        Path(candidate_root) / "measurements.json"
    )
    return {
        "baseline": {
            **baseline,
            "manifest_path": str(Path(baseline_root) / "manifest.json"),
            "measurements_path": str(Path(baseline_root) / "measurements.json"),
        },
        "candidate": {
            **candidate,
            "manifest_path": str(Path(candidate_root) / "manifest.json"),
            "measurements_path": str(Path(candidate_root) / "measurements.json"),
        },
        "delta_inv_dyn_acc_perframe": float(candidate["inv_dyn_acc_perframe"])
        - float(baseline["inv_dyn_acc_perframe"]),
        "delta_no_action_delta_psnr": float(candidate["no_action_delta_psnr"])
        - float(baseline["no_action_delta_psnr"]),
    }


def _load_action_following_measurement(path: Path) -> dict[str, object]:
    payload = _load_json_object(path)
    if (
        payload.get("artifact_type")
        not in {
            "wmloop-raw-probe-measurement-input",
            "wmloop-action-following-measurement-input",
        }
        or payload.get("source_kind") != "measured"
    ):
        raise OrchestratorTrainingEvalSmokeError(
            "M3_TRAINING_EVAL_SMOKE_ACTION_MEASURE_INVALID"
        )
    records = payload.get("records")
    if (
        not isinstance(records, list)
        or len(records) != 1
        or not isinstance(records[0], Mapping)
    ):
        raise OrchestratorTrainingEvalSmokeError(
            "M3_TRAINING_EVAL_SMOKE_ACTION_MEASURE_INVALID"
        )
    record = records[0]
    action = record.get("action_following")
    if not isinstance(action, Mapping):
        raise OrchestratorTrainingEvalSmokeError(
            "M3_TRAINING_EVAL_SMOKE_ACTION_MEASURE_INVALID"
        )
    predicted = _action_rows(action.get("predicted_actions"))
    target = _action_rows(action.get("target_actions"))
    tolerance = _finite_float(
        action.get("tolerance"), "M3_TRAINING_EVAL_SMOKE_ACTION_MEASURE_INVALID"
    )
    action_conditioned_psnr = _finite_float(
        action.get("action_conditioned_psnr"),
        "M3_TRAINING_EVAL_SMOKE_ACTION_MEASURE_INVALID",
    )
    no_action_psnr = _finite_float(
        action.get("no_action_psnr"), "M3_TRAINING_EVAL_SMOKE_ACTION_MEASURE_INVALID"
    )
    inverse_dynamics_r2 = _finite_float(
        action.get("inverse_dynamics_r2"),
        "M3_TRAINING_EVAL_SMOKE_ACTION_MEASURE_INVALID",
    )
    low_confidence = action.get("low_confidence")
    if not isinstance(low_confidence, bool):
        raise OrchestratorTrainingEvalSmokeError(
            "M3_TRAINING_EVAL_SMOKE_ACTION_MEASURE_INVALID"
        )
    try:
        accuracy = per_frame_inverse_dynamics_accuracy(
            predicted_actions=predicted,
            target_actions=target,
            tolerance=tolerance,
        )
        delta = no_action_delta_psnr(
            action_conditioned_psnr=action_conditioned_psnr,
            no_action_psnr=no_action_psnr,
        )
    except ValueError as exc:
        raise OrchestratorTrainingEvalSmokeError(
            "M3_TRAINING_EVAL_SMOKE_ACTION_MEASURE_INVALID"
        ) from exc
    return {
        "state": "measured",
        "inv_dyn_acc_perframe": accuracy,
        "no_action_delta_psnr": delta,
        "frame_count": len(target),
        "tolerance": tolerance,
        "action_conditioned_psnr": action_conditioned_psnr,
        "no_action_psnr": no_action_psnr,
        "inverse_dynamics_r2": inverse_dynamics_r2,
        "low_confidence": low_confidence,
    }


def _action_rows(raw: object) -> list[list[float]]:
    if not isinstance(raw, list) or not raw:
        raise OrchestratorTrainingEvalSmokeError(
            "M3_TRAINING_EVAL_SMOKE_ACTION_MEASURE_INVALID"
        )
    rows: list[list[float]] = []
    for row in raw:
        if not isinstance(row, list) or not row:
            raise OrchestratorTrainingEvalSmokeError(
                "M3_TRAINING_EVAL_SMOKE_ACTION_MEASURE_INVALID"
            )
        rows.append(
            [
                _finite_float(value, "M3_TRAINING_EVAL_SMOKE_ACTION_MEASURE_INVALID")
                for value in row
            ]
        )
    return rows


def _action_following_observed(action_following: Mapping[str, object] | None) -> float:
    if action_following is None:
        return 0.0
    candidate = action_following.get("candidate")
    if not isinstance(candidate, Mapping):
        raise OrchestratorTrainingEvalSmokeError(
            "M3_TRAINING_EVAL_SMOKE_ACTION_MEASURE_INVALID"
        )
    return _finite_float(
        candidate.get("inv_dyn_acc_perframe"),
        "M3_TRAINING_EVAL_SMOKE_ACTION_MEASURE_INVALID",
    )


def _replication_records_with_refs(
    *,
    replications: Sequence[Mapping[str, object]],
    cas: ContentAddressedStore,
    archive: ArchiveStore | None,
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    path_media = {
        "accept_cohort_view_manifest_path": "application/json",
        "baseline_manifest_path": "application/json",
        "candidate_manifest_path": "application/json",
        "baseline_metrics_path": "application/json",
        "candidate_metrics_path": "application/json",
    }
    for replication in replications:
        item = dict(replication)
        refs: dict[str, str] = {}
        for key, media_type in path_media.items():
            value = item.get(key)
            if not isinstance(value, str):
                raise OrchestratorTrainingEvalSmokeError(
                    "M3_TRAINING_EVAL_SMOKE_REPLICATION_ARTIFACT_INVALID"
                )
            refs[key.replace("_path", "_ref")] = _put_file(
                cas, Path(value), archive=archive, media_type=media_type
            )
        item["cas_refs"] = refs
        output.append(item)
    return output


def _action_following_records_with_refs(
    *,
    action_following: Mapping[str, object],
    cas: ContentAddressedStore,
    archive: ArchiveStore | None,
) -> dict[str, object]:
    output = json.loads(_canonical_json_bytes(action_following))
    for side in ("baseline", "candidate"):
        payload = output.get(side)
        if not isinstance(payload, dict):
            raise OrchestratorTrainingEvalSmokeError(
                "M3_TRAINING_EVAL_SMOKE_ACTION_MEASURE_INVALID"
            )
        refs: dict[str, str] = {}
        for key in ("manifest_path", "measurements_path"):
            value = payload.get(key)
            if not isinstance(value, str):
                raise OrchestratorTrainingEvalSmokeError(
                    "M3_TRAINING_EVAL_SMOKE_ACTION_MEASURE_INVALID"
                )
            refs[key.replace("_path", "_ref")] = _put_file(
                cas, Path(value), archive=archive, media_type="application/json"
            )
        payload["cas_refs"] = refs
    return output


def _evaluation_summary(
    *,
    replications: Sequence[Mapping[str, object]],
    evidence: VerificationEvidence,
    metric_name: str = "auc_psnr_16_64",
    action_following: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if not replications:
        raise OrchestratorTrainingEvalSmokeError(
            "M3_TRAINING_EVAL_SMOKE_REPLICATIONS_EMPTY"
        )
    baseline_values = [
        _replication_value(item, prefix="baseline", metric_name=metric_name)
        for item in replications
    ]
    candidate_values = [
        _replication_value(item, prefix="candidate", metric_name=metric_name)
        for item in replications
    ]
    deltas = [
        _replication_delta(item, metric_name=metric_name) for item in replications
    ]
    first = replications[0]
    summary: dict[str, object] = {
        "primary_metric": metric_name,
        "baseline_primary_metric": sum(baseline_values) / len(baseline_values),
        "candidate_primary_metric": sum(candidate_values) / len(candidate_values),
        "delta_primary_metric": evidence.accept_metric_deltas[metric_name],
        "baseline_auc_psnr_16_64": sum(baseline_values) / len(baseline_values),
        "candidate_auc_psnr_16_64": sum(candidate_values) / len(candidate_values),
        "delta_auc_psnr_16_64": evidence.accept_metric_deltas[metric_name],
        "replication_deltas": deltas,
        "replication_count": len(replications),
        "required_horizons_complete": all(
            bool(item.get("required_horizons_complete")) for item in replications
        ),
        "replications": [dict(item) for item in replications],
    }
    for key in (
        "accept_cohort_view_manifest_path",
        "baseline_manifest_path",
        "candidate_manifest_path",
        "baseline_metrics_path",
        "candidate_metrics_path",
    ):
        summary[key] = first[key]
    if action_following is not None:
        summary["action_following"] = {
            **dict(action_following),
            "gate_enabled": evidence.action_following_gate_enabled,
            "gate_threshold": evidence.action_following_threshold,
            "gate_observed": evidence.action_following_observed,
        }
    return summary
