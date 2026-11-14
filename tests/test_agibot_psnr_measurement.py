"""CPU-only contract tests for paired Agibot rollout measurement."""

from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest

from wmloop.archive.store import ContentAddressedStore
from wmloop.archive.agibot_psnr_settlement import (
    AgibotPSNRSettlementError,
    settle_agibot_psnr_trial,
    verify_agibot_psnr_settled_receipt,
)
from wmloop.contracts import ContractValidationError, validate_document
from wmloop.evaluate import future_video_recipe
from wmloop.evaluate.agibot_psnr_measurement import (
    AgibotPSNRMeasurementError,
    build_agibot_psnr_frozen_goal,
    evaluator_digest,
    measure_agibot_rollouts,
)
from wmloop.storage import canonical_bytes, sha256_bytes
from wmloop.verify.agibot_psnr_measurement import (
    AgibotPSNRVerificationError,
    verify_agibot_psnr_measurement,
)
from wmloop.verify.statistics import student_t_mean_interval


_HASH = "a" * 64


def test_evaluator_digest_binds_the_ssim_formula_source(monkeypatch):
    original_digest = evaluator_digest()
    ssim_source_path = Path(future_video_recipe.__file__)
    original_read_bytes = Path.read_bytes

    def changed_ssim_source(path: Path) -> bytes:
        source = original_read_bytes(path)
        if path == ssim_source_path:
            return source + b"\n# simulated SSIM implementation change\n"
        return source

    monkeypatch.setattr(Path, "read_bytes", changed_ssim_source)

    assert evaluator_digest() != original_digest


def _write_frames(path: Path, value: int, *, width: int = 5) -> str:
    frames = np.full((3, 4, width, 3), value, dtype=np.uint8)
    payload_path = path.with_suffix(".npy")
    np.save(payload_path, frames, allow_pickle=False)
    return sha256_bytes(payload_path.read_bytes())


def _write_video(path: Path, value: int) -> None:
    with imageio.get_writer(
        path,
        fps=5,
        codec="libx264",
        pixelformat="yuv444p",
        ffmpeg_params=["-crf", "0"],
    ) as writer:
        for _ in range(3):
            writer.append_data(np.full((16, 16, 3), value, dtype=np.uint8))


def _cohort(root: Path) -> dict[str, object]:
    sample_ids = ["episode-a:12", "episode-b:24"]
    seeds = [11, 22, 33]
    cohort: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "verdiwm-agibot-psnr-rollout-cohort",
        "claim_boundary": "Frozen validation rollout pairing for measurement only.",
        "evaluator_id": "agibot-paired-rgb-psnr-v1",
        "evaluator_digest": evaluator_digest(),
        "dataset_sha256": _HASH,
        "support_sha256": "b" * 64,
        "validation_sample_ids": sample_ids,
        "seed_cohort": seeds,
        "baseline": _arm(root, "baseline", sample_ids, seeds, base_values=[30, 28, 26]),
        "candidate": _arm(root, "candidate", sample_ids, seeds, base_values=[12, 10, 9]),
    }
    validate_document("agibot_psnr_rollout_cohort", cohort)
    return cohort


def _arm(
    root: Path,
    label: str,
    sample_ids: list[str],
    seeds: list[int],
    *,
    base_values: list[int],
) -> dict[str, object]:
    rollouts = []
    for seed_index, seed in enumerate(seeds):
        samples = []
        for sample_index, sample_id in enumerate(sample_ids):
            target_path = root / f"gt-{sample_index}.npy"
            if not target_path.exists():
                _write_frames(target_path, 100 + sample_index)
            prediction_path = root / f"{label}-{seed}-{sample_index}.npy"
            difference = base_values[seed_index] + sample_index
            prediction_digest = _write_frames(prediction_path, 100 + sample_index + difference)
            samples.append(
                {
                    "sample_id": sample_id,
                    "prediction_frames": prediction_path.name,
                    "prediction_sha256": prediction_digest,
                    "ground_truth_frames": target_path.name,
                    "ground_truth_sha256": sha256_bytes(target_path.read_bytes()),
                }
            )
        rollouts.append({"seed": seed, "samples": samples})
    return {
        "checkpoint_sha256": ("c" if label == "baseline" else "d") * 64,
        "evaluator_digest": evaluator_digest(),
        "dataset_sha256": _HASH,
        "support_sha256": "b" * 64,
        "validation_sample_ids": sample_ids.copy(),
        "seed_cohort": seeds.copy(),
        "rollout_settings": {"prediction_frames": 3, "camera": "front"},
        "decode_settings": {"format": "uint8_rgb", "frame_policy": "all_frames"},
        "rollouts": rollouts,
    }


def test_measures_three_paired_seeds_and_stores_cas_evidence(tmp_path: Path):
    cohort = _cohort(tmp_path)
    result = measure_agibot_rollouts(
        cohort,
        input_root=tmp_path,
        cas_root=tmp_path / "store",
    )
    document = result.document
    validate_document("agibot_paired_psnr_measurement", document)
    assert document["authority"] == "paired_measurement_only"
    assert document["statistics"]["sample_count"] == 3
    expected = student_t_mean_interval(
        [row["paired_delta_psnr_db"] for row in document["per_seed"]]
    )
    assert document["statistics"]["lower_bound_db"] == expected["lower_bound"]
    assert document["threshold_check"]["state"] == (
        "above_threshold" if expected["lower_bound"] > 2.0 else "at_or_below_threshold"
    )
    assert document["report_only_metrics"]["lpips"]["state"] == "not_measured"

    cas = ContentAddressedStore(tmp_path / "store")
    assert cas.read_bytes(document["input_manifest"]["uri"]) == json.dumps(
        cohort, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode()
    assert cas.read_bytes(result.artifact.uri) == json.dumps(
        document, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode()
    assert len(document["evidence_bindings"]) == 6
    for binding in document["evidence_bindings"]:
        for key in ("baseline_prediction", "candidate_prediction", "ground_truth"):
            assert cas.read_bytes(binding[key]["uri"])


def test_measurement_stores_the_experiment_frozen_goal_in_cas(tmp_path: Path):
    cohort = _cohort(tmp_path)
    result = measure_agibot_rollouts(
        cohort,
        input_root=tmp_path,
        cas_root=tmp_path / "store",
    )
    document = result.document
    cas = ContentAddressedStore(tmp_path / "store")
    goal_bytes = cas.read_bytes(document["frozen_goal_ref"]["uri"])
    goal = json.loads(goal_bytes)
    validate_document("agibot_psnr_frozen_goal", goal)
    assert goal == build_agibot_psnr_frozen_goal(cohort)
    assert document["frozen_goal_sha256"] == sha256_bytes(goal_bytes)
    assert document["cohort_sha256"] == goal["cohort_sha256"]
    assert goal["validation_sample_ids"] == cohort["validation_sample_ids"]
    assert goal["seed_cohort"] == cohort["seed_cohort"]


def test_measurement_rejects_a_goal_changed_after_experiment_freeze(tmp_path: Path):
    cohort = _cohort(tmp_path)
    goal = build_agibot_psnr_frozen_goal(cohort)
    goal["validation_sample_ids"] = ["different-sample"]
    with pytest.raises(AgibotPSNRMeasurementError, match="FROZEN_GOAL_BINDING_MISMATCH"):
        measure_agibot_rollouts(
            cohort,
            input_root=tmp_path,
            cas_root=tmp_path / "store",
            frozen_goal=goal,
        )


def test_verifier_rejects_a_measurement_goal_hash_rebinding(tmp_path: Path):
    cohort = _cohort(tmp_path)
    result = measure_agibot_rollouts(
        cohort,
        input_root=tmp_path,
        cas_root=tmp_path / "store",
    )
    tampered = dict(result.document)
    tampered["frozen_goal_sha256"] = "e" * 64
    identity = dict(tampered)
    identity.pop("measurement_id")
    tampered["measurement_id"] = "agibot-psnr-measurement-" + sha256_bytes(canonical_bytes(identity))[:24]
    cas = ContentAddressedStore(tmp_path / "store")
    tampered_ref = cas.put_bytes(canonical_bytes(tampered), media_type="application/json")
    with pytest.raises(AgibotPSNRVerificationError, match="FROZEN_GOAL_HASH_MISMATCH"):
        verify_agibot_psnr_measurement(
            measurement_ref=tampered_ref.uri,
            cas_root=tmp_path / "store",
        )


def test_settlement_carries_the_measurement_goal_without_rederiving_it(tmp_path: Path):
    cohort = _cohort(tmp_path)
    result = measure_agibot_rollouts(
        cohort,
        input_root=tmp_path,
        cas_root=tmp_path / "store",
    )
    settled = settle_agibot_psnr_trial(
        measurement_ref=result.artifact.uri,
        trial_id="goal-binding-trial",
        hypothesis="goal binding",
        method="paired replay",
        implementation_sha256="f" * 64,
        archive_db=tmp_path / "archive.db",
        cas_root=tmp_path / "store",
    )
    assert settled.goal_ref == result.document["frozen_goal_ref"]["uri"]
    verified = verify_agibot_psnr_settled_receipt(
        receipt_ref=settled.receipt_ref,
        archive_db=tmp_path / "archive.db",
        cas_root=tmp_path / "store",
        replay_measurement=True,
    )
    assert verified.goal_ref == result.document["frozen_goal_ref"]["uri"]
    assert verified.goal == json.loads(
        ContentAddressedStore(tmp_path / "store").read_bytes(verified.goal_ref)
    )


def test_settlement_verifier_rejects_symlinked_archive_path(tmp_path: Path):
    cohort = _cohort(tmp_path)
    result = measure_agibot_rollouts(
        cohort,
        input_root=tmp_path,
        cas_root=tmp_path / "store",
    )
    settled = settle_agibot_psnr_trial(
        measurement_ref=result.artifact.uri,
        trial_id="symlink-archive-trial",
        hypothesis="path trust",
        method="paired replay",
        implementation_sha256="f" * 64,
        archive_db=tmp_path / "archive.db",
        cas_root=tmp_path / "store",
    )
    alias = tmp_path / "archive-alias.db"
    alias.symlink_to(tmp_path / "archive.db")
    with pytest.raises(AgibotPSNRSettlementError, match="ARCHIVE_INVALID"):
        verify_agibot_psnr_settled_receipt(
            receipt_ref=settled.receipt_ref,
            archive_db=alias,
            cas_root=tmp_path / "store",
            replay_measurement=False,
        )


def test_measures_rgb_video_inputs_after_cas_snapshot(tmp_path: Path):
    cohort = _cohort(tmp_path)
    sample_ids = cohort["validation_sample_ids"]
    for sample_index, sample_id in enumerate(sample_ids):
        target_value = 100 + sample_index
        target_path = tmp_path / f"gt-video-{sample_index}.mp4"
        _write_video(target_path, target_value)
        target_digest = sha256_bytes(target_path.read_bytes())
        for arm_name in ("baseline", "candidate"):
            for rollout in cohort[arm_name]["rollouts"]:
                sample = next(row for row in rollout["samples"] if row["sample_id"] == sample_id)
                old_values = np.load(tmp_path / sample["prediction_frames"], allow_pickle=False)
                prediction_path = tmp_path / f"{arm_name}-{rollout['seed']}-{sample_index}.mp4"
                _write_video(prediction_path, int(old_values[0, 0, 0, 0]))
                sample["prediction_frames"] = prediction_path.name
                sample["prediction_sha256"] = sha256_bytes(prediction_path.read_bytes())
                sample["ground_truth_frames"] = target_path.name
                sample["ground_truth_sha256"] = target_digest

    result = measure_agibot_rollouts(
        cohort,
        input_root=tmp_path,
        cas_root=tmp_path / "video-store",
    )
    assert result.document["statistics"]["sample_count"] == 3
    assert len(result.document["evidence_bindings"]) == 6


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        (lambda c: c["candidate"].update(dataset_sha256="e" * 64), "CANDIDATE_BINDING_MISMATCH"),
        (
            lambda c: c["candidate"].update(rollout_settings={"prediction_frames": 9}),
            "ROLLOUT_SETTINGS_MISMATCH",
        ),
        (lambda c: c["baseline"]["rollouts"][0]["samples"].pop(), "SAMPLE_COHORT_MISMATCH"),
        (lambda c: c["candidate"].update(evaluator_digest="e" * 64), "CANDIDATE_BINDING_MISMATCH"),
    ],
)
def test_rejects_binding_and_cohort_mismatches(tmp_path: Path, mutate, error: str):
    cohort = _cohort(tmp_path)
    mutate(cohort)
    with pytest.raises(AgibotPSNRMeasurementError, match=error):
        measure_agibot_rollouts(cohort, input_root=tmp_path, cas_root=tmp_path / "store")


def test_rejects_source_digest_drift_and_ground_truth_mismatch(tmp_path: Path):
    cohort = _cohort(tmp_path)
    cohort["baseline"]["rollouts"][0]["samples"][0]["prediction_sha256"] = "e" * 64
    with pytest.raises(AgibotPSNRMeasurementError, match="PREDICTION_DIGEST_MISMATCH"):
        measure_agibot_rollouts(cohort, input_root=tmp_path, cas_root=tmp_path / "store")

    cohort = _cohort(tmp_path)
    cohort["candidate"]["rollouts"][0]["samples"][0]["ground_truth_sha256"] = "e" * 64
    with pytest.raises(AgibotPSNRMeasurementError, match="GROUND_TRUTH_DIGEST_MISMATCH"):
        measure_agibot_rollouts(cohort, input_root=tmp_path, cas_root=tmp_path / "store-2")


def test_rejects_unaligned_frame_shapes(tmp_path: Path):
    cohort = _cohort(tmp_path)
    candidate = cohort["candidate"]["rollouts"][0]["samples"][0]
    candidate_path = tmp_path / candidate["prediction_frames"]
    new_digest = _write_frames(candidate_path.with_name("different-width.npy"), 130, width=6)
    candidate["prediction_frames"] = "different-width.npy"
    candidate["prediction_sha256"] = new_digest
    with pytest.raises(AgibotPSNRMeasurementError, match="PAIRED_FRAME_SHAPE_MISMATCH"):
        measure_agibot_rollouts(cohort, input_root=tmp_path, cas_root=tmp_path / "store")


def test_rejects_relative_path_escape(tmp_path: Path):
    cohort = _cohort(tmp_path)
    escaped = tmp_path.parent / "outside.npy"
    digest = _write_frames(escaped, 131)
    cohort["baseline"]["rollouts"][0]["samples"][0]["prediction_frames"] = "../outside.npy"
    cohort["baseline"]["rollouts"][0]["samples"][0]["prediction_sha256"] = digest
    with pytest.raises(AgibotPSNRMeasurementError, match="SOURCE_PATH_OUTSIDE_ROOT"):
        measure_agibot_rollouts(cohort, input_root=tmp_path, cas_root=tmp_path / "store")


def test_contract_rejects_nonfinite_settings_values(tmp_path: Path):
    cohort = _cohort(tmp_path)
    cohort["baseline"]["rollout_settings"]["temperature"] = float("nan")
    with pytest.raises(ContractValidationError, match="NOT_FINITE_JSON"):
        validate_document("agibot_psnr_rollout_cohort", cohort)


def test_perfect_prediction_fails_closed_without_inventing_finite_psnr(tmp_path: Path):
    cohort = _cohort(tmp_path)
    sample = cohort["baseline"]["rollouts"][0]["samples"][0]
    prediction_path = tmp_path / sample["prediction_frames"]
    target = np.load(tmp_path / sample["ground_truth_frames"], allow_pickle=False)
    np.save(prediction_path, target, allow_pickle=False)
    sample["prediction_sha256"] = sha256_bytes(prediction_path.read_bytes())
    with pytest.raises(AgibotPSNRMeasurementError, match="PERFECT_PREDICTION_NOT_ESTIMABLE"):
        measure_agibot_rollouts(cohort, input_root=tmp_path, cas_root=tmp_path / "store")
