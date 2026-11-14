"""Independent replay verification for Agibot paired PSNR measurement artifacts."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
from typing import Any, Mapping

from wmloop.archive.store import ArchiveInvariantError, ContentAddressedStore
from wmloop.contracts import ContractValidationError, validate_document
from wmloop.evaluate.agibot_psnr_measurement import (
    AgibotPSNRMeasurementError,
    MeasurementArtifact,
    build_agibot_psnr_frozen_goal,
    _validate_cohort,
    measure_agibot_rollouts,
)
from wmloop.storage import canonical_bytes, sha256_bytes
from wmloop.verify.statistics import student_t_mean_interval


class AgibotPSNRVerificationError(ValueError):
    """A paired PSNR artifact failed independent replay or binding checks."""


@dataclass(frozen=True)
class VerifiedAgibotPSNRMeasurement:
    """Measurement and cohort whose numeric fields were recomputed from CAS."""

    reference: str
    document: dict[str, Any]
    cohort: dict[str, Any]
    goal: dict[str, Any]
    goal_ref: str


def verify_agibot_psnr_measurement(
    *, measurement_ref: str, cas_root: Path, schema_root: Path | None = None,
) -> VerifiedAgibotPSNRMeasurement:
    """Recompute every paired rollout metric from the immutable CAS sources.

    A schema-valid JSON claim is not sufficient for effect settlement. This
    verifier restores the declared frames into a private temporary directory,
    runs the frozen measurement implementation, and compares the complete
    result. It therefore rejects altered PSNR statistics even when the attacker
    has also rewritten the measurement's own CAS reference.
    """

    cas = ContentAddressedStore(Path(cas_root))
    try:
        raw_measurement = cas.read_bytes(measurement_ref)
        measurement = _json_object(raw_measurement, "MEASUREMENT_INVALID")
        validate_document("agibot_paired_psnr_measurement", measurement, root=schema_root)
        if canonical_bytes(measurement) != raw_measurement:
            raise AgibotPSNRVerificationError("AGIBOT_PSNR_MEASUREMENT_NOT_CANONICAL")
        _verify_measurement_identity(measurement)

        manifest = measurement["input_manifest"]
        raw_manifest = _read_artifact(manifest, cas, code="MANIFEST_INVALID")
        cohort_value = _json_object(raw_manifest, "MANIFEST_INVALID")
        cohort = _validate_cohort(cohort_value, schema_root=schema_root)
        if canonical_bytes(cohort) != raw_manifest:
            raise AgibotPSNRVerificationError("AGIBOT_PSNR_MANIFEST_NOT_CANONICAL")
        _assert_cohort_projection(measurement, cohort)
        goal_ref_document = measurement["frozen_goal_ref"]
        raw_goal = _read_artifact(goal_ref_document, cas, code="FROZEN_GOAL_INVALID")
        goal = _json_object(raw_goal, "FROZEN_GOAL_INVALID")
        validate_document("agibot_psnr_frozen_goal", goal, root=schema_root)
        if canonical_bytes(goal) != raw_goal:
            raise AgibotPSNRVerificationError("AGIBOT_PSNR_FROZEN_GOAL_NOT_CANONICAL")
        if sha256_bytes(raw_goal) != measurement["frozen_goal_sha256"]:
            raise AgibotPSNRVerificationError("AGIBOT_PSNR_FROZEN_GOAL_HASH_MISMATCH")
        if goal_ref_document["uri"] != f"cas://sha256/{measurement['frozen_goal_sha256']}":
            raise AgibotPSNRVerificationError("AGIBOT_PSNR_FROZEN_GOAL_REF_MISMATCH")
        expected_goal = build_agibot_psnr_frozen_goal(cohort)
        if goal != expected_goal:
            raise AgibotPSNRVerificationError("AGIBOT_PSNR_FROZEN_GOAL_BINDING_MISMATCH")

        evidence = _evidence_map(measurement)
        replay_cohort = deepcopy(cohort)
        with tempfile.TemporaryDirectory(prefix="verdiwm-agibot-psnr-verify-") as temporary:
            root = Path(temporary)
            ordinal = 0
            for arm_name in ("baseline", "candidate"):
                for rollout in replay_cohort[arm_name]["rollouts"]:
                    for sample in rollout["samples"]:
                        sample_id, seed = sample["sample_id"], rollout["seed"]
                        for role, evidence_role in (
                            ("prediction", f"{arm_name}_prediction"),
                            ("ground_truth", "ground_truth"),
                        ):
                            binding = evidence[(seed, sample_id)]
                            artifact = binding[evidence_role]
                            payload = _read_artifact(artifact, cas, code="SOURCE_INVALID")
                            expected_digest = sample[f"{role}_sha256"]
                            if artifact["sha256"] != expected_digest:
                                raise AgibotPSNRVerificationError("AGIBOT_PSNR_SOURCE_BINDING_MISMATCH")
                            suffix = Path(sample[f"{role}_frames"]).suffix
                            ordinal += 1
                            relative = Path("inputs") / f"source-{ordinal:06d}{suffix}"
                            destination = root / relative
                            destination.parent.mkdir(parents=True, exist_ok=True)
                            destination.write_bytes(payload)
                            sample[f"{role}_frames"] = relative.as_posix()
            replay: MeasurementArtifact = measure_agibot_rollouts(
                replay_cohort,
                input_root=root,
                cas_root=root / "replay-cas",
                schema_root=schema_root,
                frozen_goal=goal,
                _allow_frozen_goal_cohort_digest_mismatch=True,
            )
        expected = dict(replay.document)
        actual = dict(measurement)
        # Rebased temporary paths alter only the manifest's content address;
        # all source refs, metric values, statistics and threshold fields must
        # otherwise be byte-for-byte equivalent as JSON values.
        expected.pop("input_manifest", None)
        actual.pop("input_manifest", None)
        expected.pop("measurement_id", None)
        actual.pop("measurement_id", None)
        if expected != actual:
            raise AgibotPSNRVerificationError("AGIBOT_PSNR_REPLAY_MISMATCH")
        return VerifiedAgibotPSNRMeasurement(
            reference=measurement_ref,
            document=measurement,
            cohort=cohort,
            goal=goal,
            goal_ref=goal_ref_document["uri"],
        )
    except AgibotPSNRVerificationError:
        raise
    except (ArchiveInvariantError, AgibotPSNRMeasurementError, ContractValidationError,
            OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise AgibotPSNRVerificationError(
            f"AGIBOT_PSNR_EVIDENCE_INVALID:{type(exc).__name__}"
        ) from exc


def validate_agibot_psnr_measurement_claim(
    measurement: Mapping[str, Any], *, schema_root: Path | None = None,
) -> None:
    """Check the schema and recompute the declared per-seed CI and threshold."""

    try:
        validate_document("agibot_paired_psnr_measurement", measurement, root=schema_root)
        seeds = measurement["seed_cohort"]
        rows = measurement["per_seed"]
        if len(seeds) < 3 or [row.get("seed") for row in rows] != seeds:
            raise AgibotPSNRVerificationError("AGIBOT_PSNR_SEED_COHORT_INVALID")
        interval = student_t_mean_interval([row["paired_delta_psnr_db"] for row in rows])
        statistics = measurement["statistics"]
        expected = {
            "mean_paired_delta_db": interval["mean"],
            "standard_error_db": interval["standard_error"],
            "lower_bound_db": interval["lower_bound"],
            "upper_bound_db": interval["upper_bound"],
        }
        if any(statistics.get(key) != value for key, value in expected.items()):
            raise AgibotPSNRVerificationError("AGIBOT_PSNR_STATISTICS_MISMATCH")
        passed = interval["lower_bound"] > 2.0
        check = measurement["threshold_check"]
        expected_state = "above_threshold" if passed else "at_or_below_threshold"
        if (
            check.get("minimum_effect_db") != 2.0
            or check.get("rule") != "lower_bound_db > minimum_effect_db"
            or check.get("state") != expected_state
            or statistics.get("sample_count") != len(seeds)
            or statistics.get("confidence_interval") != "student_t_two_sided_95"
        ):
            raise AgibotPSNRVerificationError("AGIBOT_PSNR_THRESHOLD_BINDING_INVALID")
        goal_ref = measurement["frozen_goal_ref"]
        if (
            measurement["frozen_goal_sha256"] != goal_ref["sha256"]
            or goal_ref["uri"] != f"cas://sha256/{goal_ref['sha256']}"
            or measurement["cohort_sha256"] != measurement["cohort_sha256"].lower()
        ):
            raise AgibotPSNRVerificationError("AGIBOT_PSNR_FROZEN_GOAL_BINDING_INVALID")
    except AgibotPSNRVerificationError:
        raise
    except (ContractValidationError, KeyError, TypeError, ValueError) as exc:
        raise AgibotPSNRVerificationError("AGIBOT_PSNR_MEASUREMENT_INVALID") from exc


def _verify_measurement_identity(document: Mapping[str, Any]) -> None:
    identity = dict(document)
    observed = identity.pop("measurement_id", None)
    expected = "agibot-psnr-measurement-" + sha256_bytes(canonical_bytes(identity))[:24]
    if observed != expected:
        raise AgibotPSNRVerificationError("AGIBOT_PSNR_MEASUREMENT_ID_MISMATCH")


def _assert_cohort_projection(measurement: Mapping[str, Any], cohort: Mapping[str, Any]) -> None:
    for field in ("dataset_sha256", "support_sha256", "validation_sample_ids", "seed_cohort"):
        if measurement.get(field) != cohort.get(field):
            raise AgibotPSNRVerificationError("AGIBOT_PSNR_COHORT_BINDING_MISMATCH")
    if measurement.get("evaluator_digest") != cohort.get("evaluator_digest"):
        raise AgibotPSNRVerificationError("AGIBOT_PSNR_EVALUATOR_BINDING_MISMATCH")
    expected_cohort_sha256 = sha256_bytes(canonical_bytes(cohort))
    if measurement.get("cohort_sha256") != expected_cohort_sha256:
        raise AgibotPSNRVerificationError("AGIBOT_PSNR_COHORT_DIGEST_MISMATCH")
    for field in ("rollout_settings_sha256", "decode_settings_sha256"):
        expected = sha256_bytes(canonical_bytes(cohort["baseline"][field.removesuffix("_sha256")]))
        if measurement.get(field) != expected:
            raise AgibotPSNRVerificationError("AGIBOT_PSNR_SETTINGS_BINDING_MISMATCH")
    if measurement.get("checkpoint_sha256") != {
        name: cohort[name]["checkpoint_sha256"] for name in ("baseline", "candidate")
    }:
        raise AgibotPSNRVerificationError("AGIBOT_PSNR_CHECKPOINT_BINDING_MISMATCH")


def _evidence_map(measurement: Mapping[str, Any]) -> dict[tuple[int, str], Mapping[str, Any]]:
    result = {}
    for binding in measurement["evidence_bindings"]:
        key = (binding["seed"], binding["sample_id"])
        if key in result:
            raise AgibotPSNRVerificationError("AGIBOT_PSNR_EVIDENCE_BINDING_DUPLICATE")
        result[key] = binding
    expected = {
        (seed, sample_id)
        for seed in measurement["seed_cohort"]
        for sample_id in measurement["validation_sample_ids"]
    }
    if set(result) != expected:
        raise AgibotPSNRVerificationError("AGIBOT_PSNR_EVIDENCE_BINDING_INCOMPLETE")
    return result


def _read_artifact(document: Mapping[str, Any], cas: ContentAddressedStore, *, code: str) -> bytes:
    digest = document.get("sha256")
    uri = document.get("uri")
    size = document.get("size_bytes")
    if (
        not isinstance(digest, str) or len(digest) != 64
        or uri != f"cas://sha256/{digest}"
        or isinstance(size, bool) or not isinstance(size, int) or size < 0
    ):
        raise AgibotPSNRVerificationError(f"AGIBOT_PSNR_{code}")
    payload = cas.read_bytes(uri)
    if len(payload) != size or sha256_bytes(payload) != digest:
        raise AgibotPSNRVerificationError(f"AGIBOT_PSNR_{code}")
    return payload


def _json_object(payload: bytes, code: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AgibotPSNRVerificationError(f"AGIBOT_PSNR_{code}") from exc
    if not isinstance(value, dict):
        raise AgibotPSNRVerificationError(f"AGIBOT_PSNR_{code}")
    return value
