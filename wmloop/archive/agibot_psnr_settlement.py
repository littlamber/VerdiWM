"""Verified Agibot PSNR effect verdicts and append-only Archive settlement."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import sqlite3
from typing import Any, Mapping

from wmloop.archive.settlement_evidence import read_settled_trial_rows
from wmloop.archive.store import (
    ArchiveInvariantError,
    ArchiveStore,
    ContentAddressedStore,
    SettledTrialRecord,
)
from wmloop.contracts import ContractValidationError, validate_document
from wmloop.storage import canonical_bytes, checked_path, sha256_bytes
from wmloop.verify.agibot_psnr_measurement import (
    AgibotPSNRVerificationError,
    VerifiedAgibotPSNRMeasurement,
    validate_agibot_psnr_measurement_claim,
    verify_agibot_psnr_measurement,
)


_GOAL_CLAIM = (
    "This frozen target applies to paired RGB rollout PSNR on the bound dataset, "
    "support and evaluator only. SSIM is report-only; LPIPS is not measured."
)
_VERDICT_CLAIM = (
    "Only independently replayed paired RGB rollout PSNR determines this verdict. "
    "Training loss, a successful training process, or checkpoint reload alone are not effect evidence."
)
_RECEIPT_CLAIM = (
    "This receipt binds an independently replayed paired RGB rollout PSNR verdict "
    "to its frozen target and Archive settlement. Training loss or checkpoint reload is not effect evidence."
)


class AgibotPSNRSettlementError(ValueError):
    """Agibot effect evidence failed settlement or archive replay checks."""


@dataclass(frozen=True)
class AgibotPSNRSettlement:
    trial_id: str
    proposal_id: str
    goal_id: str
    goal_ref: str
    verdict_ref: str
    receipt_ref: str
    measurement_ref: str
    verdict: dict[str, Any]
    receipt: dict[str, Any]
    effect_verified: bool


@dataclass(frozen=True)
class VerifiedAgibotPSNRSettlement:
    trial_id: str
    proposal_id: str
    goal_id: str
    receipt_ref: str
    goal_ref: str
    verdict_ref: str
    measurement_ref: str
    context_ref: str
    verdict: dict[str, Any]
    receipt: dict[str, Any]
    goal: dict[str, Any]
    measurement: dict[str, Any]
    context: dict[str, Any]


def settle_agibot_psnr_trial(
    *,
    measurement_ref: str,
    trial_id: str,
    hypothesis: str,
    method: str,
    implementation_sha256: str,
    archive_db: Path,
    cas_root: Path,
    proposal_id: str | None = None,
    gpu_hours: float = 0.0,
    schema_root: Path | None = None,
) -> AgibotPSNRSettlement:
    """Replay a measurement, issue its effect verdict, and settle it in Archive.

    Archive's existing ``SettledTrialRecord`` is reused with ``cell=None``. The
    PSNR result therefore becomes durable reusable evidence without entering
    M3 intervention-cell/UCB statistics or weakening M3's action-following gate.
    """

    trial_id = _required_text(trial_id, "TRIAL_ID_INVALID", maximum=256)
    proposal_id = _required_text(proposal_id or trial_id, "PROPOSAL_ID_INVALID", maximum=256)
    hypothesis = _required_text(hypothesis, "HYPOTHESIS_INVALID", maximum=4096)
    method = _required_text(method, "METHOD_INVALID", maximum=4096)
    _require_digest(implementation_sha256, "IMPLEMENTATION_DIGEST_INVALID")
    if isinstance(gpu_hours, bool) or not isinstance(gpu_hours, (int, float)) or not math.isfinite(gpu_hours) or gpu_hours < 0:
        raise AgibotPSNRSettlementError("AGIBOT_PSNR_COST_INVALID")

    verified = verify_agibot_psnr_measurement(
        measurement_ref=measurement_ref, cas_root=cas_root, schema_root=schema_root
    )
    measurement = verified.document
    validate_agibot_psnr_measurement_claim(measurement, schema_root=schema_root)
    # The measurement verifier has already read and authenticated the goal
    # artifact created before the experiment. Settlement carries that exact
    # object forward; it must never synthesize a new goal from measured values.
    goal = verified.goal
    goal_ref = verified.goal_ref
    goal_bytes = canonical_bytes(goal)
    goal_sha256 = sha256_bytes(goal_bytes)
    if goal_sha256 != measurement["frozen_goal_sha256"]:
        raise AgibotPSNRSettlementError("AGIBOT_PSNR_FROZEN_GOAL_HASH_MISMATCH")
    goal_id = _goal_id(goal_bytes)

    context = {
        "schema_version": 1,
        "artifact_type": "verdiwm-agibot-psnr-trial-context",
        "claim_boundary": "Hypothesis and method notes are research memory, not an effect verdict.",
        "hypothesis": hypothesis,
        "method": method,
        "goal_id": goal_id,
        "goal_ref": goal_ref,
        "measurement_ref": measurement_ref,
        "dataset_sha256": measurement["dataset_sha256"],
        "support_sha256": measurement["support_sha256"],
        "evaluator_digest": measurement["evaluator_digest"],
        "checkpoint_sha256": measurement["checkpoint_sha256"],
    }
    context_bytes = canonical_bytes(context)
    context_ref = ContentAddressedStore(Path(cas_root)).put_bytes(
        context_bytes, media_type="application/json"
    ).uri
    context_hash = sha256_bytes(context_bytes)

    verdict = _effect_verdict(
        measurement=measurement,
        goal=goal,
        goal_id=goal_id,
        goal_ref=goal_ref,
        goal_sha256=goal_sha256,
        measurement_ref=measurement_ref,
        measurement_sha256=measurement_ref.rsplit("/", 1)[-1],
    )
    verdict_bytes = canonical_bytes(verdict)
    verdict_ref = ContentAddressedStore(Path(cas_root)).put_bytes(
        verdict_bytes, media_type="application/json"
    ).uri
    receipt = _settlement_receipt(
        trial_id=trial_id,
        proposal_id=proposal_id,
        goal_id=goal_id,
        goal_ref=goal_ref,
        goal_sha256=goal_sha256,
        verdict_ref=verdict_ref,
        verdict_sha256=sha256_bytes(verdict_bytes),
        measurement_ref=measurement_ref,
        measurement_sha256=measurement_ref.rsplit("/", 1)[-1],
        context_ref=context_ref,
        context_sha256=context_hash,
        measurement=measurement,
        effect_verified=verdict["verdict"] == "ACCEPT",
    )
    receipt_bytes = canonical_bytes(receipt)
    receipt_ref = ContentAddressedStore(Path(cas_root)).put_bytes(
        receipt_bytes, media_type="application/json"
    ).uri
    receipt_hash = sha256_bytes(receipt_bytes)
    record = SettledTrialRecord(
        trial_id=trial_id,
        proposal_id=proposal_id,
        goal_id=goal_id,
        library_version="agibot-psnr-settlement-v1",
        failure_context_ref=context_ref,
        verdict_ref=verdict_ref,
        receipt_ref=receipt_ref,
        gpu_hours=float(gpu_hours),
        hypothesis_hash=sha256_bytes(canonical_bytes({"hypothesis": hypothesis})),
        impl_diff_hash=implementation_sha256,
        evaluator_hash=measurement["evaluator_digest"],
        settlement_state="settled",
        receipt_hash=receipt_hash,
        cell=None,
    )
    archive = ArchiveStore(Path(archive_db))
    try:
        archive.record_settled_trial(record)
    except ArchiveInvariantError as exc:
        if not archive.has_settled_receipt(trial_id, receipt_ref, receipt_hash):
            raise AgibotPSNRSettlementError("AGIBOT_PSNR_ARCHIVE_SETTLEMENT_FAILED") from exc
    settled = verify_agibot_psnr_settled_receipt(
        receipt_ref=receipt_ref,
        archive_db=archive_db,
        cas_root=cas_root,
        replay_measurement=True,
        schema_root=schema_root,
    )
    return AgibotPSNRSettlement(
        trial_id=trial_id,
        proposal_id=proposal_id,
        goal_id=goal_id,
        goal_ref=goal_ref,
        verdict_ref=verdict_ref,
        receipt_ref=receipt_ref,
        measurement_ref=measurement_ref,
        verdict=settled.verdict,
        receipt=settled.receipt,
        effect_verified=settled.verdict["verdict"] == "ACCEPT",
    )


def verify_agibot_psnr_settled_receipt(
    *,
    receipt_ref: str,
    archive_db: Path,
    cas_root: Path,
    replay_measurement: bool = False,
    schema_root: Path | None = None,
) -> VerifiedAgibotPSNRSettlement:
    """Verify Archive projections, CAS hashes and every verdict binding."""

    try:
        cas = ContentAddressedStore(Path(cas_root))
        # Reject symlinked archive paths before resolving them.  Archive/CAS
        # paths are user supplied state boundaries; resolving first would
        # bypass the repository-wide path trust contract.
        database = checked_path(
            Path(archive_db),
            code="AGIBOT_PSNR_ARCHIVE_INVALID",
            error=AgibotPSNRSettlementError,
        )
        with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as connection:
            connection.row_factory = sqlite3.Row
            rows = read_settled_trial_rows(connection, receipt_ref=receipt_ref, limit=2)
        if len(rows) != 1:
            raise AgibotPSNRSettlementError("AGIBOT_PSNR_SETTLEMENT_NOT_FOUND")
        row = rows[0]
        receipt_bytes = cas.read_bytes(receipt_ref)
        if sha256_bytes(receipt_bytes) != row["receipt_hash"]:
            raise AgibotPSNRSettlementError("AGIBOT_PSNR_RECEIPT_HASH_MISMATCH")
        receipt = _json_object(receipt_bytes, "RECEIPT_INVALID")
        validate_document("agibot_psnr_settlement_receipt", receipt, root=schema_root)
        if canonical_bytes(receipt) != receipt_bytes:
            raise AgibotPSNRSettlementError("AGIBOT_PSNR_RECEIPT_NOT_CANONICAL")

        goal_bytes = cas.read_bytes(receipt["goal_ref"])
        if sha256_bytes(goal_bytes) != receipt["goal_sha256"]:
            raise AgibotPSNRSettlementError("AGIBOT_PSNR_GOAL_HASH_MISMATCH")
        goal = _json_object(goal_bytes, "GOAL_INVALID")
        validate_document("agibot_psnr_frozen_goal", goal, root=schema_root)
        if canonical_bytes(goal) != goal_bytes:
            raise AgibotPSNRSettlementError("AGIBOT_PSNR_GOAL_NOT_CANONICAL")
        if _goal_id(goal_bytes) != receipt["goal_id"]:
            raise AgibotPSNRSettlementError("AGIBOT_PSNR_GOAL_ID_MISMATCH")

        verdict_bytes = cas.read_bytes(receipt["verdict_ref"])
        if sha256_bytes(verdict_bytes) != receipt["verdict_sha256"]:
            raise AgibotPSNRSettlementError("AGIBOT_PSNR_VERDICT_HASH_MISMATCH")
        verdict = _json_object(verdict_bytes, "VERDICT_INVALID")
        validate_document("agibot_psnr_effect_verdict", verdict, root=schema_root)

        measurement_ref = receipt["measurement_ref"]
        measurement_bytes = cas.read_bytes(measurement_ref)
        if sha256_bytes(measurement_bytes) != receipt["measurement_sha256"]:
            raise AgibotPSNRSettlementError("AGIBOT_PSNR_MEASUREMENT_HASH_MISMATCH")
        measurement = _json_object(measurement_bytes, "MEASUREMENT_INVALID")
        validate_agibot_psnr_measurement_claim(measurement, schema_root=schema_root)
        if replay_measurement:
            replayed = verify_agibot_psnr_measurement(
                measurement_ref=measurement_ref, cas_root=cas_root, schema_root=schema_root
            )
            measurement = replayed.document

        context_bytes = cas.read_bytes(receipt["context_ref"])
        if sha256_bytes(context_bytes) != receipt["context_sha256"]:
            raise AgibotPSNRSettlementError("AGIBOT_PSNR_CONTEXT_HASH_MISMATCH")
        context = _json_object(context_bytes, "CONTEXT_INVALID")

        # The receipt, measurement and goal must all point to the same
        # immutable pre-experiment goal artifact. Do not derive an expected
        # goal from the result: that would let a changed result silently change
        # the target used for settlement.
        measurement_goal_ref = measurement["frozen_goal_ref"]
        if (
            measurement_goal_ref["uri"] != receipt["goal_ref"]
            or measurement["frozen_goal_sha256"] != receipt["goal_sha256"]
            or measurement_goal_ref["sha256"] != measurement["frozen_goal_sha256"]
            or measurement_goal_ref["uri"] != f"cas://sha256/{measurement_goal_ref['sha256']}"
        ):
            raise AgibotPSNRSettlementError("AGIBOT_PSNR_FROZEN_GOAL_BINDING_MISMATCH")
        measurement_goal_bytes = cas.read_bytes(measurement_goal_ref["uri"])
        if (
            sha256_bytes(measurement_goal_bytes) != measurement["frozen_goal_sha256"]
            or measurement_goal_bytes != goal_bytes
        ):
            raise AgibotPSNRSettlementError("AGIBOT_PSNR_FROZEN_GOAL_MISMATCH")
        expected_verdict = _effect_verdict(
            measurement=measurement,
            goal=goal,
            goal_id=receipt["goal_id"],
            goal_ref=receipt["goal_ref"],
            goal_sha256=receipt["goal_sha256"],
            measurement_ref=measurement_ref,
            measurement_sha256=receipt["measurement_sha256"],
        )
        if verdict != expected_verdict:
            raise AgibotPSNRSettlementError("AGIBOT_PSNR_VERDICT_BINDING_MISMATCH")
        expected_receipt = _settlement_receipt(
            trial_id=receipt["trial_id"],
            proposal_id=receipt["proposal_id"],
            goal_id=receipt["goal_id"],
            goal_ref=receipt["goal_ref"],
            goal_sha256=receipt["goal_sha256"],
            verdict_ref=receipt["verdict_ref"],
            verdict_sha256=receipt["verdict_sha256"],
            measurement_ref=measurement_ref,
            measurement_sha256=receipt["measurement_sha256"],
            context_ref=receipt["context_ref"],
            context_sha256=receipt["context_sha256"],
            measurement=measurement,
            effect_verified=verdict["verdict"] == "ACCEPT",
        )
        if receipt != expected_receipt:
            raise AgibotPSNRSettlementError("AGIBOT_PSNR_RECEIPT_BINDING_MISMATCH")
        if (
            row["trial_id"] != receipt["trial_id"]
            or row["proposal_id"] != receipt["proposal_id"]
            or row["goal_id"] != receipt["goal_id"]
            or row["failure_context_ref"] != receipt["context_ref"]
            or row["verdict_ref"] != receipt["verdict_ref"]
            or row["receipt_ref"] != receipt_ref
            or row["library_version"] != "agibot-psnr-settlement-v1"
            or json.loads(row["settlement_json"]).get("evidence_scope") != "verified"
        ):
            raise AgibotPSNRSettlementError("AGIBOT_PSNR_ARCHIVE_BINDING_MISMATCH")
        if context.get("artifact_type") != "verdiwm-agibot-psnr-trial-context":
            raise AgibotPSNRSettlementError("AGIBOT_PSNR_CONTEXT_BINDING_MISMATCH")
        if (
            context.get("goal_id") != receipt["goal_id"]
            or context.get("goal_ref") != receipt["goal_ref"]
            or context.get("measurement_ref") != measurement_ref
            or context.get("dataset_sha256") != measurement["dataset_sha256"]
            or context.get("support_sha256") != measurement["support_sha256"]
            or context.get("evaluator_digest") != measurement["evaluator_digest"]
            or context.get("checkpoint_sha256") != measurement["checkpoint_sha256"]
            or not isinstance(context.get("hypothesis"), str)
            or not isinstance(context.get("method"), str)
        ):
            raise AgibotPSNRSettlementError("AGIBOT_PSNR_CONTEXT_BINDING_MISMATCH")
        return VerifiedAgibotPSNRSettlement(
            trial_id=str(row["trial_id"]),
            proposal_id=str(row["proposal_id"]),
            goal_id=str(row["goal_id"]),
            receipt_ref=receipt_ref,
            goal_ref=str(receipt["goal_ref"]),
            verdict_ref=str(receipt["verdict_ref"]),
            measurement_ref=measurement_ref,
            context_ref=str(receipt["context_ref"]),
            verdict=verdict,
            receipt=receipt,
            goal=goal,
            measurement=measurement,
            context=context,
        )
    except AgibotPSNRSettlementError:
        raise
    except (ArchiveInvariantError, ContractValidationError, AgibotPSNRVerificationError,
            OSError, TypeError, ValueError, KeyError, sqlite3.Error, json.JSONDecodeError) as exc:
        raise AgibotPSNRSettlementError(
            f"AGIBOT_PSNR_SETTLEMENT_INVALID:{type(exc).__name__}"
        ) from exc


def _goal_id(goal_bytes: bytes) -> str:
    return "agibot-psnr-" + sha256_bytes(goal_bytes)[:24]


def _effect_verdict(
    *, measurement: Mapping[str, Any], goal: Mapping[str, Any], goal_id: str,
    goal_ref: str, goal_sha256: str, measurement_ref: str, measurement_sha256: str,
) -> dict[str, Any]:
    passed = measurement["statistics"]["lower_bound_db"] > goal["minimum_effect_db"]
    return {
        "schema_version": 1,
        "artifact_type": "verdiwm-agibot-psnr-effect-verdict",
        "claim_boundary": _VERDICT_CLAIM,
        "authority": "measured_paired_rollout_effect",
        "verdict": "ACCEPT" if passed else "REJECT",
        "reason": "PSNR_LOWER_BOUND_ABOVE_2_DB" if passed else "PSNR_LOWER_BOUND_NOT_ABOVE_2_DB",
        "goal_id": goal_id,
        "goal_ref": goal_ref,
        "goal_sha256": goal_sha256,
        "measurement_ref": measurement_ref,
        "measurement_sha256": measurement_sha256,
        "evaluator_digest": measurement["evaluator_digest"],
        "dataset_sha256": measurement["dataset_sha256"],
        "support_sha256": measurement["support_sha256"],
        "validation_sample_ids": list(measurement["validation_sample_ids"]),
        "seed_cohort": list(measurement["seed_cohort"]),
        "checkpoint_sha256": dict(measurement["checkpoint_sha256"]),
        "statistics": dict(measurement["statistics"]),
        "threshold_check": dict(measurement["threshold_check"]),
        "report_only_metrics": {
            "ssim": dict(measurement["report_only_metrics"]["global_ssim_rgb"]),
            "lpips": {"state": "not_measured", "value": None},
        },
    }


def _settlement_receipt(
    *, trial_id: str, proposal_id: str, goal_id: str, goal_ref: str,
    goal_sha256: str, verdict_ref: str, verdict_sha256: str, measurement_ref: str,
    measurement_sha256: str, context_ref: str, context_sha256: str,
    measurement: Mapping[str, Any], effect_verified: bool,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": "verdiwm-agibot-psnr-settlement-receipt",
        "claim_boundary": _RECEIPT_CLAIM,
        "state": "terminal",
        "settlement_state": "settled",
        "trial_id": trial_id,
        "proposal_id": proposal_id,
        "goal_id": goal_id,
        "goal_ref": goal_ref,
        "goal_sha256": goal_sha256,
        "verdict_ref": verdict_ref,
        "verdict_sha256": verdict_sha256,
        "measurement_ref": measurement_ref,
        "measurement_sha256": measurement_sha256,
        "context_ref": context_ref,
        "context_sha256": context_sha256,
        "evaluator_digest": measurement["evaluator_digest"],
        "dataset_sha256": measurement["dataset_sha256"],
        "support_sha256": measurement["support_sha256"],
        "validation_sample_ids": list(measurement["validation_sample_ids"]),
        "seed_cohort": list(measurement["seed_cohort"]),
        "checkpoint_sha256": dict(measurement["checkpoint_sha256"]),
        "effect_verified": effect_verified,
    }


def _required_text(value: object, code: str, *, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum or "\0" in value:
        raise AgibotPSNRSettlementError(f"AGIBOT_PSNR_{code}")
    return value.strip()


def _require_digest(value: object, code: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise AgibotPSNRSettlementError(f"AGIBOT_PSNR_{code}")


def _json_object(payload: bytes, code: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AgibotPSNRSettlementError(f"AGIBOT_PSNR_{code}") from exc
    if not isinstance(value, dict):
        raise AgibotPSNRSettlementError(f"AGIBOT_PSNR_{code}")
    return value
