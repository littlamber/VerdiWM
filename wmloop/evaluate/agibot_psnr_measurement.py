"""Paired RGB rollout measurement for frozen Agibot validation cohorts.

This service measures already-produced baseline and candidate rollouts. It does
not launch inference or training and grants no settlement, verdict, promotion,
or archive-memory authority.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
import mimetypes
from pathlib import Path
import statistics
from typing import Any

import imageio.v3 as imageio
import numpy as np

from wmloop.archive.store import ArtifactRef, ContentAddressedStore
from wmloop.contracts import ContractValidationError, validate_document
from wmloop.evaluate.future_video_recipe import global_rgb_ssim
from wmloop.storage import canonical_bytes, checked_path, sha256_bytes
from wmloop.verify.statistics import student_t_mean_interval


EVALUATOR_ID = "agibot-paired-rgb-psnr-v1"
_CLAIM_BOUNDARY = (
    "Measurement only: paired rollout metrics and a PSNR threshold check; this artifact "
    "does not grant formal verification, settlement, promotion, or archive-memory authority."
)
_LPIPS_BOUNDARY = "LPIPS is not measured by the current Agibot RGB evaluator."
_GOAL_CLAIM = (
    "This frozen target applies to paired RGB rollout PSNR on the bound dataset, "
    "support and evaluator only. SSIM is report-only; LPIPS is not measured."
)
_EVALUATOR_SPEC = {
    "evaluator_id": EVALUATOR_ID,
    "version": 1,
    "input": "all aligned uint8 RGB frames from one rollout, no resizing or color conversion",
    "psnr": "10*log10(255^2/mean_squared_error_over_all_rgb_values_and_frames)",
    "ssim": "wmloop.evaluate.future_video_recipe.global_rgb_ssim over all aligned RGB values",
    "sample_aggregation": "arithmetic mean of per-sample PSNR and SSIM within each seed",
    "inference": (
        "paired candidate-minus-baseline delta per seed; Student-t two-sided 95% CI across seeds"
    ),
    "threshold": "lower_bound_db > 2.0",
}
class AgibotPSNRMeasurementError(ValueError):
    """A frozen paired rollout cohort could not be measured safely."""


@dataclass(frozen=True)
class MeasurementArtifact:
    """Validated measurement document and its immutable CAS copy."""

    document: dict[str, Any]
    artifact: ArtifactRef


def evaluator_digest() -> str:
    """Bind the numeric recipe and its imported SSIM implementation to source."""

    source_digest = sha256_bytes(Path(__file__).read_bytes())
    ssim_source_path = Path(global_rgb_ssim.__code__.co_filename)
    ssim_source_digest = sha256_bytes(ssim_source_path.read_bytes())
    return sha256_bytes(
        canonical_bytes(
            {
                "recipe": _EVALUATOR_SPEC,
                "source_sha256": source_digest,
                "ssim_implementation_sha256": ssim_source_digest,
            }
        )
    )


def measure_agibot_rollouts(
    cohort: Mapping[str, Any],
    *,
    input_root: Path,
    cas_root: Path,
    schema_root: Path | None = None,
    frozen_goal: Mapping[str, Any] | None = None,
    _allow_frozen_goal_cohort_digest_mismatch: bool = False,
) -> MeasurementArtifact:
    """Measure an already-generated, paired Agibot RGB rollout cohort.

    ``input_root`` resolves relative prediction and ground-truth file paths in
    the cohort. Inputs may be RGB videos supported by ImageIO or ``.npy`` files
    containing ``[time, height, width, 3]`` uint8 frames. Source bytes are
    snapshotted to CAS before decoding, so the metrics and evidence refer to the
    same immutable content.
    """

    normalized = _validate_cohort(cohort, schema_root=schema_root)
    expected_goal = build_agibot_psnr_frozen_goal(normalized)
    if frozen_goal is None:
        bound_goal = expected_goal
    else:
        bound_goal = _validate_frozen_goal(
            frozen_goal,
            expected_goal=expected_goal,
            schema_root=schema_root,
            allow_cohort_digest_mismatch=_allow_frozen_goal_cohort_digest_mismatch,
        )
    base = checked_path(
        Path(input_root),
        code="AGIBOT_PSNR_INPUT_ROOT_INVALID",
        error=AgibotPSNRMeasurementError,
    )
    if not base.is_dir():
        raise AgibotPSNRMeasurementError("AGIBOT_PSNR_INPUT_ROOT_INVALID")
    cas = ContentAddressedStore(Path(cas_root))
    manifest_ref = cas.put_bytes(canonical_bytes(normalized), media_type="application/json")
    goal_bytes = canonical_bytes(bound_goal)
    goal_ref = cas.put_bytes(goal_bytes, media_type="application/json")

    arms = {name: normalized[name] for name in ("baseline", "candidate")}
    references: dict[tuple[str, int, str, str], ArtifactRef] = {}
    rollouts = {
        arm_name: {run["seed"]: run for run in arm["rollouts"]}
        for arm_name, arm in arms.items()
    }

    # Snapshot each distinct file once. Ground-truth bytes are required to be
    # identical across both arms and every seed for a validation sample.
    for arm_name, arm in arms.items():
        for run in arm["rollouts"]:
            seed = run["seed"]
            for sample in run["samples"]:
                sample_id = sample["sample_id"]
                for role in ("prediction", "ground_truth"):
                    key = (arm_name, seed, sample_id, role)
                    source = _source_path(base, sample[f"{role}_frames"])
                    ref = _snapshot_file(
                        cas,
                        source,
                        expected_sha256=sample[f"{role}_sha256"],
                        role=role,
                    )
                    references[key] = ref

    _validate_ground_truth_bindings(
        arms,
        references,
        normalized["seed_cohort"],
        normalized["validation_sample_ids"],
    )

    evidence_bindings: list[dict[str, Any]] = []
    per_seed: list[dict[str, Any]] = []
    for seed in normalized["seed_cohort"]:
        sample_metrics: list[dict[str, Any]] = []
        baseline_psnr: list[float] = []
        candidate_psnr: list[float] = []
        baseline_ssim: list[float] = []
        candidate_ssim: list[float] = []
        for sample_id in normalized["validation_sample_ids"]:
            baseline_sample = _find_sample(rollouts["baseline"][seed], sample_id)
            candidate_sample = _find_sample(rollouts["candidate"][seed], sample_id)
            baseline_prediction = references[("baseline", seed, sample_id, "prediction")]
            candidate_prediction = references[("candidate", seed, sample_id, "prediction")]
            ground_truth = references[("baseline", seed, sample_id, "ground_truth")]
            baseline_values = _decode_frames(
                baseline_prediction.path,
                extension=Path(baseline_sample["prediction_frames"]).suffix,
            )
            candidate_values = _decode_frames(
                candidate_prediction.path,
                extension=Path(candidate_sample["prediction_frames"]).suffix,
            )
            target_values = _decode_frames(
                ground_truth.path,
                extension=Path(baseline_sample["ground_truth_frames"]).suffix,
            )
            baseline_score = _score(baseline_values, target_values)
            candidate_score = _score(candidate_values, target_values)
            delta = candidate_score["psnr_db"] - baseline_score["psnr_db"]
            sample_metrics.append(
                {
                    "sample_id": sample_id,
                    "baseline_psnr_db": baseline_score["psnr_db"],
                    "candidate_psnr_db": candidate_score["psnr_db"],
                    "delta_psnr_db": delta,
                    "baseline_global_ssim_rgb": baseline_score["global_ssim_rgb"],
                    "candidate_global_ssim_rgb": candidate_score["global_ssim_rgb"],
                }
            )
            baseline_psnr.append(baseline_score["psnr_db"])
            candidate_psnr.append(candidate_score["psnr_db"])
            baseline_ssim.append(baseline_score["global_ssim_rgb"])
            candidate_ssim.append(candidate_score["global_ssim_rgb"])
            evidence_bindings.append(
                {
                    "seed": seed,
                    "sample_id": sample_id,
                    "baseline_prediction": _artifact_document(baseline_prediction),
                    "candidate_prediction": _artifact_document(candidate_prediction),
                    "ground_truth": _artifact_document(ground_truth),
                }
            )
        baseline_mean = statistics.fmean(baseline_psnr)
        candidate_mean = statistics.fmean(candidate_psnr)
        per_seed.append(
            {
                "seed": seed,
                "baseline_mean_psnr_db": baseline_mean,
                "candidate_mean_psnr_db": candidate_mean,
                "paired_delta_psnr_db": candidate_mean - baseline_mean,
                "baseline_mean_global_ssim_rgb": statistics.fmean(baseline_ssim),
                "candidate_mean_global_ssim_rgb": statistics.fmean(candidate_ssim),
                "sample_metrics": sample_metrics,
            }
        )

    interval = student_t_mean_interval([row["paired_delta_psnr_db"] for row in per_seed])
    ssim_baseline = statistics.fmean(row["baseline_mean_global_ssim_rgb"] for row in per_seed)
    ssim_candidate = statistics.fmean(row["candidate_mean_global_ssim_rgb"] for row in per_seed)
    rollout_settings_sha256 = sha256_bytes(
        canonical_bytes(arms["baseline"]["rollout_settings"])
    )
    decode_settings_sha256 = sha256_bytes(
        canonical_bytes(arms["baseline"]["decode_settings"])
    )
    threshold_state = (
        "above_threshold" if interval["lower_bound"] > 2.0 else "at_or_below_threshold"
    )
    document: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "verdiwm-agibot-paired-psnr-measurement",
        "claim_boundary": _CLAIM_BOUNDARY,
        "authority": "paired_measurement_only",
        "evaluator_id": EVALUATOR_ID,
        "evaluator_digest": normalized["evaluator_digest"],
        "input_manifest": _artifact_document(manifest_ref),
        "frozen_goal_ref": _artifact_document(goal_ref),
        "frozen_goal_sha256": sha256_bytes(goal_bytes),
        "cohort_sha256": bound_goal["cohort_sha256"],
        "dataset_sha256": normalized["dataset_sha256"],
        "support_sha256": normalized["support_sha256"],
        "validation_sample_ids": list(normalized["validation_sample_ids"]),
        "seed_cohort": list(normalized["seed_cohort"]),
        "rollout_settings_sha256": rollout_settings_sha256,
        "decode_settings_sha256": decode_settings_sha256,
        "checkpoint_sha256": {
            "baseline": arms["baseline"]["checkpoint_sha256"],
            "candidate": arms["candidate"]["checkpoint_sha256"],
        },
        "evidence_bindings": evidence_bindings,
        "per_seed": per_seed,
        "statistics": {
            "metric": "psnr_db",
            "mean_paired_delta_db": interval["mean"],
            "standard_error_db": interval["standard_error"],
            "lower_bound_db": interval["lower_bound"],
            "upper_bound_db": interval["upper_bound"],
            "confidence_interval": "student_t_two_sided_95",
            "sample_count": len(per_seed),
        },
        "threshold_check": {
            "minimum_effect_db": 2.0,
            "rule": "lower_bound_db > minimum_effect_db",
            "state": threshold_state,
        },
        "report_only_metrics": {
            "global_ssim_rgb": {
                "baseline_mean": ssim_baseline,
                "candidate_mean": ssim_candidate,
                "mean_delta": ssim_candidate - ssim_baseline,
            },
            "lpips": {"state": "not_measured", "value": None, "claim_boundary": _LPIPS_BOUNDARY},
        },
    }
    identity_payload = canonical_bytes(document)
    document["measurement_id"] = f"agibot-psnr-measurement-{sha256_bytes(identity_payload)[:24]}"
    _validate_measurement(document, schema_root=schema_root)
    measurement_ref = cas.put_bytes(canonical_bytes(document), media_type="application/json")
    return MeasurementArtifact(document=document, artifact=measurement_ref)


def build_agibot_psnr_frozen_goal(cohort: Mapping[str, Any]) -> dict[str, Any]:
    """Build the immutable PSNR target and cohort contract for a validated cohort.

    The cohort digest deliberately covers the complete canonical manifest. A
    replay verifier may pass this goal back while materializing source files at
    different temporary paths; that private replay path is the only place where
    the digest mismatch is allowed.
    """

    rollout_settings_sha256 = sha256_bytes(
        canonical_bytes(cohort["baseline"]["rollout_settings"])
    )
    decode_settings_sha256 = sha256_bytes(
        canonical_bytes(cohort["baseline"]["decode_settings"])
    )
    return {
        "schema_version": 1,
        "artifact_type": "verdiwm-agibot-psnr-frozen-goal",
        "claim_boundary": _GOAL_CLAIM,
        "primary_metric": "psnr_db",
        "direction": "maximize",
        "minimum_effect_db": 2.0,
        "minimum_paired_seeds": 3,
        "confidence_interval": "student_t_two_sided_95",
        "threshold_rule": "lower_bound_db > minimum_effect_db",
        "dataset_sha256": cohort["dataset_sha256"],
        "support_sha256": cohort["support_sha256"],
        "evaluator_digest": cohort["evaluator_digest"],
        "validation_sample_ids": list(cohort["validation_sample_ids"]),
        "seed_cohort": list(cohort["seed_cohort"]),
        "rollout_settings_sha256": rollout_settings_sha256,
        "decode_settings_sha256": decode_settings_sha256,
        "cohort_sha256": sha256_bytes(canonical_bytes(cohort)),
        "report_only_metrics": {"ssim": "report_only", "lpips": "not_measured"},
    }


def _validate_frozen_goal(
    goal: Mapping[str, Any],
    *,
    expected_goal: Mapping[str, Any],
    schema_root: Path | None,
    allow_cohort_digest_mismatch: bool,
) -> dict[str, Any]:
    if not isinstance(goal, Mapping):
        raise AgibotPSNRMeasurementError("AGIBOT_PSNR_FROZEN_GOAL_INVALID")
    normalized = dict(goal)
    try:
        json.dumps(normalized, allow_nan=False)
        validate_document("agibot_psnr_frozen_goal", normalized, root=schema_root)
    except (TypeError, ValueError, ContractValidationError) as exc:
        raise AgibotPSNRMeasurementError("AGIBOT_PSNR_FROZEN_GOAL_INVALID") from exc
    expected = dict(expected_goal)
    if allow_cohort_digest_mismatch:
        expected.pop("cohort_sha256")
        observed = dict(normalized)
        observed.pop("cohort_sha256", None)
        if observed != expected:
            raise AgibotPSNRMeasurementError("AGIBOT_PSNR_FROZEN_GOAL_BINDING_MISMATCH")
    elif normalized != expected:
        raise AgibotPSNRMeasurementError("AGIBOT_PSNR_FROZEN_GOAL_BINDING_MISMATCH")
    return normalized


def _validate_cohort(cohort: Mapping[str, Any], *, schema_root: Path | None) -> dict[str, Any]:
    if not isinstance(cohort, Mapping):
        raise AgibotPSNRMeasurementError("AGIBOT_PSNR_COHORT_INVALID")
    normalized = dict(cohort)
    try:
        json.dumps(normalized, allow_nan=False)
        validate_document("agibot_psnr_rollout_cohort", normalized, root=schema_root)
    except (TypeError, ValueError, ContractValidationError) as exc:
        raise AgibotPSNRMeasurementError(f"AGIBOT_PSNR_COHORT_INVALID:{exc}") from exc
    expected_digest = evaluator_digest()
    if normalized["evaluator_digest"] != expected_digest:
        raise AgibotPSNRMeasurementError("AGIBOT_PSNR_EVALUATOR_DIGEST_MISMATCH")
    sample_ids = normalized["validation_sample_ids"]
    seeds = normalized["seed_cohort"]
    for arm_name in ("baseline", "candidate"):
        arm = normalized[arm_name]
        if (
            arm["evaluator_digest"] != expected_digest
            or arm["dataset_sha256"] != normalized["dataset_sha256"]
            or arm["support_sha256"] != normalized["support_sha256"]
            or arm["validation_sample_ids"] != sample_ids
            or arm["seed_cohort"] != seeds
        ):
            raise AgibotPSNRMeasurementError(f"AGIBOT_PSNR_{arm_name.upper()}_BINDING_MISMATCH")
        if [row["seed"] for row in arm["rollouts"]] != seeds:
            raise AgibotPSNRMeasurementError(f"AGIBOT_PSNR_{arm_name.upper()}_SEED_COHORT_MISMATCH")
        for rollout in arm["rollouts"]:
            ids = [sample["sample_id"] for sample in rollout["samples"]]
            if len(ids) != len(set(ids)) or set(ids) != set(sample_ids):
                raise AgibotPSNRMeasurementError(
                    f"AGIBOT_PSNR_{arm_name.upper()}_SAMPLE_COHORT_MISMATCH"
                )
    for field in ("rollout_settings", "decode_settings"):
        if normalized["baseline"][field] != normalized["candidate"][field]:
            raise AgibotPSNRMeasurementError(f"AGIBOT_PSNR_{field.upper()}_MISMATCH")
    return normalized


def _source_path(input_root: Path, raw: str) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = input_root / path
    try:
        resolved = checked_path(
            path,
            code="AGIBOT_PSNR_SOURCE_PATH_SYMLINK",
            error=AgibotPSNRMeasurementError,
        )
    except OSError as exc:
        raise AgibotPSNRMeasurementError("AGIBOT_PSNR_SOURCE_PATH_INVALID") from exc
    try:
        resolved.relative_to(input_root)
    except ValueError as exc:
        raise AgibotPSNRMeasurementError("AGIBOT_PSNR_SOURCE_PATH_OUTSIDE_ROOT") from exc
    if not resolved.is_file():
        raise AgibotPSNRMeasurementError("AGIBOT_PSNR_SOURCE_FILE_MISSING")
    return resolved


def _snapshot_file(
    cas: ContentAddressedStore,
    path: Path,
    *,
    expected_sha256: str,
    role: str,
) -> ArtifactRef:
    try:
        ref = cas.put_file(
            path,
            media_type=mimetypes.guess_type(path.name)[0] or "application/octet-stream",
        )
    except (OSError, ValueError) as exc:
        raise AgibotPSNRMeasurementError(f"AGIBOT_PSNR_{role.upper()}_SNAPSHOT_FAILED") from exc
    if ref.sha256 != expected_sha256:
        raise AgibotPSNRMeasurementError(f"AGIBOT_PSNR_{role.upper()}_DIGEST_MISMATCH")
    return ref


def _validate_ground_truth_bindings(
    arms: Mapping[str, Mapping[str, Any]],
    references: Mapping[tuple[str, int, str, str], ArtifactRef],
    seeds: Sequence[int],
    sample_ids: Sequence[str],
) -> None:
    for sample_id in sample_ids:
        expected: str | None = None
        for arm_name in ("baseline", "candidate"):
            for seed in seeds:
                sample = _find_sample(
                    next(run for run in arms[arm_name]["rollouts"] if run["seed"] == seed),
                    sample_id,
                )
                ref = references[(arm_name, seed, sample_id, "ground_truth")]
                if sample["ground_truth_sha256"] != ref.sha256:
                    raise AgibotPSNRMeasurementError("AGIBOT_PSNR_GROUND_TRUTH_DIGEST_MISMATCH")
                if expected is not None and ref.sha256 != expected:
                    raise AgibotPSNRMeasurementError("AGIBOT_PSNR_GROUND_TRUTH_PAIR_MISMATCH")
                expected = ref.sha256


def _find_sample(rollout: Mapping[str, Any], sample_id: str) -> Mapping[str, Any]:
    for sample in rollout["samples"]:
        if sample["sample_id"] == sample_id:
            return sample
    raise AgibotPSNRMeasurementError("AGIBOT_PSNR_SAMPLE_NOT_FOUND")


def _decode_frames(path: Path, *, extension: str) -> np.ndarray:
    try:
        if extension.lower() == ".npy":
            frames = np.load(path, allow_pickle=False)
        else:
            frames = imageio.imread(path, extension=extension or None)
    except Exception as exc:
        raise AgibotPSNRMeasurementError("AGIBOT_PSNR_FRAME_DECODE_FAILED") from exc
    if not isinstance(frames, np.ndarray):
        raise AgibotPSNRMeasurementError("AGIBOT_PSNR_FRAME_ARRAY_INVALID")
    if frames.ndim == 3:
        frames = frames[np.newaxis, ...]
    if (
        frames.ndim != 4
        or frames.shape[0] < 1
        or frames.shape[1] < 1
        or frames.shape[2] < 1
        or frames.shape[3] != 3
        or frames.dtype != np.uint8
    ):
        raise AgibotPSNRMeasurementError("AGIBOT_PSNR_RGB_FRAME_CONTRACT_INVALID")
    return frames


def _score(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    if prediction.shape != target.shape:
        raise AgibotPSNRMeasurementError("AGIBOT_PSNR_PAIRED_FRAME_SHAPE_MISMATCH")
    x = prediction.astype(np.float64)
    y = target.astype(np.float64)
    mse = float(np.mean(np.square(x - y), dtype=np.float64))
    if not math.isfinite(mse):
        raise AgibotPSNRMeasurementError("AGIBOT_PSNR_MSE_INVALID")
    if mse == 0.0:
        raise AgibotPSNRMeasurementError("AGIBOT_PSNR_PERFECT_PREDICTION_NOT_ESTIMABLE")
    psnr = 10.0 * math.log10(255.0**2 / mse)
    ssim = global_rgb_ssim(x, y, data_range=255.0)
    if not math.isfinite(psnr) or not math.isfinite(ssim):
        raise AgibotPSNRMeasurementError("AGIBOT_PSNR_METRIC_NOT_FINITE")
    return {"psnr_db": psnr, "global_ssim_rgb": ssim}


def _artifact_document(ref: ArtifactRef) -> dict[str, Any]:
    return {
        "sha256": ref.sha256,
        "size_bytes": ref.size,
        "media_type": ref.media_type,
        "uri": ref.uri,
    }


def _validate_measurement(document: Mapping[str, Any], *, schema_root: Path | None) -> None:
    try:
        json.dumps(document, allow_nan=False)
        validate_document("agibot_paired_psnr_measurement", document, root=schema_root)
    except (TypeError, ValueError, ContractValidationError) as exc:
        raise AgibotPSNRMeasurementError(f"AGIBOT_PSNR_MEASUREMENT_INVALID:{exc}") from exc
