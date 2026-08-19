"""Frozen, receipt-bound verifier for the Ctrl-World ACWM ladder.

The campaign runner may measure and settle candidates, but it cannot grant a
formal claim.  This module independently reloads every bound input, recomputes
the screen and confirmation gates, and publishes a content-addressed verdict.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from wmloop.archive.store import ArchiveInvariantError, ContentAddressedStore
from wmloop.contracts import ContractValidationError, validate_document
from wmloop.control.acwm_campaign import (
    ACWMCampaignError,
    canonical_json_bytes,
    load_mapping,
    load_measurement_receipt,
    sha256_file,
    validate_acwm_candidate_batch,
)
from wmloop.control.acwm_dual_evaluation import (
    ACWMDualEvaluationError,
    assess_acwm_dual_evaluation,
    validate_acwm_dual_evaluation_contract,
)


class ACWMFrozenVerifierError(ValueError):
    """Frozen ACWM evidence failed a verifier boundary."""


_POLICY_ID = "maximum_confirm_primary_improvement_v1"
_TIE_BREAKER = "candidate_id_ascending"
_TERMINAL_CAMPAIGN_STATE = "settled"


def policy_digest(policy: Mapping[str, object]) -> str:
    """Return the canonical digest of a policy excluding its self digest."""

    payload = {key: value for key, value in policy.items() if key != "policy_digest"}
    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ACWMFrozenVerifierError("ACWM_FROZEN_POLICY_PAYLOAD_INVALID") from exc
    return hashlib.sha256(encoded).hexdigest()


def verifier_implementation_sha256() -> str:
    """Hash the exact verifier implementation executing this decision."""

    try:
        return sha256_file(Path(__file__))
    except ACWMCampaignError as exc:
        raise ACWMFrozenVerifierError("ACWM_FROZEN_VERIFIER_SOURCE_INVALID") from exc


def validate_acwm_frozen_verifier_policy(
    policy: Mapping[str, object],
    contract: Mapping[str, object],
    *,
    root: Path | None = None,
) -> None:
    """Validate the immutable policy, contract binding, and source pin."""

    try:
        validate_document("acwm_frozen_verifier_policy", policy, root=root)
    except ContractValidationError as exc:
        raise ACWMFrozenVerifierError(f"ACWM_FROZEN_POLICY_SCHEMA_INVALID:{exc}") from exc
    try:
        validate_acwm_dual_evaluation_contract(contract, root=root)
    except ACWMDualEvaluationError as exc:
        raise ACWMFrozenVerifierError(f"ACWM_FROZEN_CONTRACT_INVALID:{exc}") from exc
    if policy.get("policy_digest") != policy_digest(policy):
        raise ACWMFrozenVerifierError("ACWM_FROZEN_POLICY_DIGEST_MISMATCH")
    if policy.get("contract_id") != contract.get("contract_id"):
        raise ACWMFrozenVerifierError("ACWM_FROZEN_POLICY_CONTRACT_ID_MISMATCH")
    if policy.get("contract_digest") != contract.get("contract_digest"):
        raise ACWMFrozenVerifierError("ACWM_FROZEN_POLICY_CONTRACT_DIGEST_MISMATCH")
    if policy.get("selection_policy") != _POLICY_ID:
        raise ACWMFrozenVerifierError("ACWM_FROZEN_POLICY_SELECTION_INVALID")
    if policy.get("tie_breaker") != _TIE_BREAKER:
        raise ACWMFrozenVerifierError("ACWM_FROZEN_POLICY_TIE_BREAKER_INVALID")
    if policy.get("implementation_sha256") != verifier_implementation_sha256():
        raise ACWMFrozenVerifierError("ACWM_FROZEN_VERIFIER_IMPLEMENTATION_MISMATCH")


def run_acwm_frozen_verifier(
    *,
    policy_path: Path,
    contract_path: Path,
    screen_root: Path,
    confirm_root: Path,
    output_root: Path,
    project_root: Path | None = None,
) -> dict[str, object]:
    """Verify a complete screen-confirm ladder and publish one CAS verdict."""

    schema_root = Path(project_root).resolve() if project_root is not None else None
    policy_path = _require_regular_file(policy_path, "ACWM_FROZEN_POLICY_FILE_INVALID")
    contract_path = _require_regular_file(contract_path, "ACWM_FROZEN_CONTRACT_FILE_INVALID")
    policy = _load(policy_path, "ACWM_FROZEN_POLICY_FILE_INVALID")
    contract = _load(contract_path, "ACWM_FROZEN_CONTRACT_FILE_INVALID")
    validate_acwm_frozen_verifier_policy(policy, contract, root=schema_root)

    raw_destination = Path(output_root).expanduser()
    if raw_destination.is_symlink():
        raise ACWMFrozenVerifierError("ACWM_FROZEN_OUTPUT_ROOT_INVALID")
    destination = raw_destination.resolve()
    for evidence_root in (screen_root, confirm_root):
        resolved_evidence = Path(evidence_root).expanduser().resolve()
        if destination == resolved_evidence or resolved_evidence in destination.parents:
            raise ACWMFrozenVerifierError("ACWM_FROZEN_OUTPUT_OVERLAPS_EVIDENCE")
    destination.mkdir(mode=0o700, parents=True, exist_ok=True)
    if destination.is_symlink() or not destination.is_dir():
        raise ACWMFrozenVerifierError("ACWM_FROZEN_OUTPUT_ROOT_INVALID")
    try:
        cas = ContentAddressedStore(destination)
        policy_ref = _archive_file(cas, policy_path, "application/json")
        contract_ref = _archive_file(cas, contract_path, "application/json")
        verifier_ref = _archive_file(cas, Path(__file__), "text/x-python")
        screen = _load_campaign(
            stage_root=screen_root,
            expected_stage="screen",
            contract=contract,
            contract_path=contract_path,
            cas=cas,
            schema_root=schema_root,
        )
        confirm = _load_campaign(
            stage_root=confirm_root,
            expected_stage="confirm",
            contract=contract,
            contract_path=contract_path,
            cas=cas,
            schema_root=schema_root,
        )
    except (ArchiveInvariantError, ACWMCampaignError) as exc:
        raise ACWMFrozenVerifierError(f"ACWM_FROZEN_EVIDENCE_INVALID:{exc}") from exc

    primary_metric = _primary_metric_id(contract)
    candidate_rows: list[dict[str, object]] = []
    eligible: list[dict[str, object]] = []
    screen_candidates = screen["candidates"]
    confirm_candidates = confirm["candidates"]
    assert isinstance(screen_candidates, dict)
    assert isinstance(confirm_candidates, dict)
    for candidate_id in sorted(confirm_candidates):
        confirm_row = confirm_candidates[candidate_id]
        screen_row = screen_candidates.get(candidate_id)
        if screen_row is None:
            raise ACWMFrozenVerifierError(
                f"ACWM_FROZEN_SCREEN_ADMISSION_MISSING:{candidate_id}"
            )
        if not bool(screen_row["assessment"]["accepted"]):
            raise ACWMFrozenVerifierError(
                f"ACWM_FROZEN_SCREEN_ADMISSION_REJECTED:{candidate_id}"
            )
        if screen_row["candidate"] != confirm_row["candidate"]:
            raise ACWMFrozenVerifierError(
                f"ACWM_FROZEN_CANDIDATE_DESCRIPTOR_MISMATCH:{candidate_id}"
            )

        accepted = bool(confirm_row["assessment"]["accepted"])
        blockers = list(confirm_row["assessment"]["blockers"])
        row = {
            "candidate": dict(confirm_row["candidate"]),
            "verification_state": "verified_eligible" if accepted else "verified_rejected",
            "screen": dict(screen_row["assessment"]),
            "confirm": dict(confirm_row["assessment"]),
            "evidence_refs": sorted(
                {
                    *screen_row["evidence_refs"],
                    *confirm_row["evidence_refs"],
                }
            ),
            "blockers": blockers,
        }
        candidate_rows.append(row)
        if accepted:
            eligible.append(row)

    selected = _select_candidate(eligible, primary_metric=primary_metric)
    decision = "selected" if selected is not None else "abstained"
    verdict = {
        "schema_version": 1,
        "artifact_type": "verdiwm-acwm-frozen-verdict",
        "verification_state": "verified",
        "verdict_authority": True,
        "decision": decision,
        "contract_id": contract["contract_id"],
        "contract_digest": contract["contract_digest"],
        "contract_ref": contract_ref,
        "policy_id": policy["policy_id"],
        "policy_digest": policy["policy_digest"],
        "policy_ref": policy_ref,
        "verifier_implementation_sha256": policy["implementation_sha256"],
        "verifier_ref": verifier_ref,
        "selection_policy": _POLICY_ID,
        "primary_metric_id": primary_metric,
        "tie_breaker": _TIE_BREAKER,
        "selected_candidate": None if selected is None else dict(selected["candidate"]),
        "candidates": candidate_rows,
        "campaign_refs": {
            "screen_summary": screen["summary_ref"],
            "screen_input_lock": screen["input_lock_ref"],
            "screen_batch": screen["batch_ref"],
            "screen_baseline": screen["baseline_ref"],
            "confirm_summary": confirm["summary_ref"],
            "confirm_input_lock": confirm["input_lock_ref"],
            "confirm_batch": confirm["batch_ref"],
            "confirm_baseline": confirm["baseline_ref"],
        },
        "claim_boundary": (
            "This verdict licenses only the frozen ACWM predictive contract. "
            "It makes no task-success, task-progress, or safety claim."
        ),
    }
    _validate_output("acwm_frozen_verdict", verdict, schema_root=schema_root)
    try:
        verdict_artifact = cas.put_bytes(
            canonical_json_bytes(verdict), media_type="application/json"
        )
    except (ArchiveInvariantError, ACWMCampaignError) as exc:
        raise ACWMFrozenVerifierError("ACWM_FROZEN_VERDICT_ARCHIVE_FAILED") from exc
    manifest = {
        "schema_version": 1,
        "artifact_type": "verdiwm-acwm-verification-manifest",
        "state": "verified",
        "decision": decision,
        "selected_candidate_id": (
            None if selected is None else selected["candidate"]["candidate_id"]
        ),
        "verdict_ref": verdict_artifact.uri,
        "verdict_sha256": verdict_artifact.sha256,
        "policy_digest": policy["policy_digest"],
        "verifier_implementation_sha256": policy["implementation_sha256"],
        "contract_digest": contract["contract_digest"],
        "screen_input_sha256": screen["input_sha256"],
        "confirm_input_sha256": confirm["input_sha256"],
        "verified_candidate_count": len(candidate_rows),
        "eligible_candidate_count": len(eligible),
    }
    _validate_output("acwm_verification_manifest", manifest, schema_root=schema_root)
    _write_json_idempotent(destination / "verification-manifest.json", manifest)
    return manifest


def _load_campaign(
    *,
    stage_root: Path,
    expected_stage: str,
    contract: Mapping[str, object],
    contract_path: Path,
    cas: ContentAddressedStore,
    schema_root: Path | None,
) -> dict[str, object]:
    raw_root = Path(stage_root).expanduser()
    if raw_root.is_symlink():
        raise ACWMFrozenVerifierError(f"ACWM_FROZEN_{expected_stage.upper()}_ROOT_INVALID")
    root = raw_root.resolve()
    if not root.is_dir():
        raise ACWMFrozenVerifierError(f"ACWM_FROZEN_{expected_stage.upper()}_ROOT_INVALID")
    summary_path = _require_regular_file(
        root / "campaign-summary.json",
        f"ACWM_FROZEN_{expected_stage.upper()}_SUMMARY_INVALID",
    )
    input_lock_path = _require_regular_file(
        root / "input-lock.json",
        f"ACWM_FROZEN_{expected_stage.upper()}_INPUT_LOCK_INVALID",
    )
    summary = _load(summary_path, "ACWM_FROZEN_CAMPAIGN_SUMMARY_INVALID")
    input_lock = _load(input_lock_path, "ACWM_FROZEN_CAMPAIGN_INPUT_LOCK_INVALID")
    _validate_campaign_headers(
        summary=summary,
        input_lock=input_lock,
        expected_stage=expected_stage,
        contract=contract,
        contract_path=contract_path,
    )

    batch_path = _path_from_lock(input_lock, "batch_path", "ACWM_FROZEN_BATCH_PATH_INVALID")
    if sha256_file(batch_path) != input_lock.get("batch_file_sha256"):
        raise ACWMFrozenVerifierError("ACWM_FROZEN_BATCH_HASH_MISMATCH")
    batch = _load(batch_path, "ACWM_FROZEN_BATCH_INVALID")
    validate_acwm_candidate_batch(batch, contract, root=schema_root)
    if batch.get("stage") != expected_stage:
        raise ACWMFrozenVerifierError("ACWM_FROZEN_BATCH_STAGE_MISMATCH")
    if batch.get("batch_id") != summary.get("batch_id"):
        raise ACWMFrozenVerifierError("ACWM_FROZEN_BATCH_ID_MISMATCH")
    if batch.get("batch_digest") != summary.get("batch_digest"):
        raise ACWMFrozenVerifierError("ACWM_FROZEN_BATCH_DIGEST_MISMATCH")

    baseline_path = _path_from_lock(
        input_lock, "baseline_path", "ACWM_FROZEN_BASELINE_PATH_INVALID"
    )
    _require_regular_file(
        baseline_path.parent / "manifest.json", "ACWM_FROZEN_BASELINE_MANIFEST_INVALID"
    )
    baseline, baseline_sha256 = load_measurement_receipt(
        baseline_path,
        contract=contract,
        stage=expected_stage,
        root=schema_root,
    )
    if baseline_sha256 != input_lock.get("baseline_sha256"):
        raise ACWMFrozenVerifierError("ACWM_FROZEN_BASELINE_HASH_MISMATCH")
    baseline_ref = _archive_file(cas, baseline_path, "application/json")
    _archive_file(cas, baseline_path.parent / "manifest.json", "application/json")

    summary_rows = summary.get("candidates")
    if not isinstance(summary_rows, list) or not summary_rows:
        raise ACWMFrozenVerifierError("ACWM_FROZEN_CAMPAIGN_CANDIDATES_INVALID")
    rows_by_id: dict[str, Mapping[str, object]] = {}
    for row in summary_rows:
        if not isinstance(row, Mapping):
            raise ACWMFrozenVerifierError("ACWM_FROZEN_CAMPAIGN_CANDIDATES_INVALID")
        candidate_id = str(row.get("candidate_id") or "")
        if not candidate_id or candidate_id in rows_by_id:
            raise ACWMFrozenVerifierError("ACWM_FROZEN_CAMPAIGN_CANDIDATE_DUPLICATE")
        rows_by_id[candidate_id] = row
    batch_candidates = batch.get("candidates")
    assert isinstance(batch_candidates, list)
    descriptors = {
        str(row["candidate_id"]): row
        for row in batch_candidates
        if isinstance(row, Mapping)
    }
    if set(descriptors) != set(rows_by_id):
        raise ACWMFrozenVerifierError("ACWM_FROZEN_CAMPAIGN_CANDIDATE_SET_MISMATCH")
    settlement_ids = {
        path.parent.name
        for path in (root / "candidates").glob("*/settlement.json")
        if path.is_file() and not path.is_symlink()
    }
    if settlement_ids != set(rows_by_id):
        raise ACWMFrozenVerifierError("ACWM_FROZEN_SETTLEMENT_SET_MISMATCH")

    verified: dict[str, dict[str, object]] = {}
    for candidate_id in sorted(rows_by_id):
        verified[candidate_id] = _verify_candidate(
            stage_root=root,
            expected_stage=expected_stage,
            summary_row=rows_by_id[candidate_id],
            candidate=descriptors[candidate_id],
            batch=batch,
            contract=contract,
            baseline=baseline,
            baseline_path=baseline_path,
            baseline_sha256=baseline_sha256,
            input_lock=input_lock,
            cas=cas,
            schema_root=schema_root,
        )
    return {
        "candidates": verified,
        "summary_ref": _archive_file(cas, summary_path, "application/json"),
        "input_lock_ref": _archive_file(cas, input_lock_path, "application/json"),
        "batch_ref": _archive_file(cas, batch_path, "application/json"),
        "baseline_ref": baseline_ref,
        "input_sha256": input_lock["input_sha256"],
    }


def _verify_candidate(
    *,
    stage_root: Path,
    expected_stage: str,
    summary_row: Mapping[str, object],
    candidate: Mapping[str, object],
    batch: Mapping[str, object],
    contract: Mapping[str, object],
    baseline: Mapping[str, object],
    baseline_path: Path,
    baseline_sha256: str,
    input_lock: Mapping[str, object],
    cas: ContentAddressedStore,
    schema_root: Path | None,
) -> dict[str, object]:
    candidate_id = str(candidate["candidate_id"])
    candidate_root = stage_root / "candidates" / candidate_id
    settlement_path = _require_regular_file(
        candidate_root / "settlement.json", "ACWM_FROZEN_SETTLEMENT_INVALID"
    )
    settlement_sha256 = sha256_file(settlement_path)
    if settlement_sha256 != summary_row.get("settlement_sha256"):
        raise ACWMFrozenVerifierError(
            f"ACWM_FROZEN_SETTLEMENT_HASH_MISMATCH:{candidate_id}"
        )
    settlement = _load(settlement_path, "ACWM_FROZEN_SETTLEMENT_INVALID")
    if settlement.get("stage") != expected_stage:
        raise ACWMFrozenVerifierError(f"ACWM_FROZEN_SETTLEMENT_STAGE_MISMATCH:{candidate_id}")
    if settlement.get("contract_id") != contract.get("contract_id"):
        raise ACWMFrozenVerifierError(f"ACWM_FROZEN_SETTLEMENT_CONTRACT_MISMATCH:{candidate_id}")
    if settlement.get("contract_digest") != contract.get("contract_digest"):
        raise ACWMFrozenVerifierError(f"ACWM_FROZEN_SETTLEMENT_CONTRACT_MISMATCH:{candidate_id}")
    if settlement.get("batch_id") != batch.get("batch_id"):
        raise ACWMFrozenVerifierError(f"ACWM_FROZEN_SETTLEMENT_BATCH_MISMATCH:{candidate_id}")
    if settlement.get("batch_digest") != batch.get("batch_digest"):
        raise ACWMFrozenVerifierError(f"ACWM_FROZEN_SETTLEMENT_BATCH_MISMATCH:{candidate_id}")
    if settlement.get("input_sha256") != input_lock.get("input_sha256"):
        raise ACWMFrozenVerifierError(f"ACWM_FROZEN_SETTLEMENT_INPUT_MISMATCH:{candidate_id}")
    if settlement.get("verdict_authority") is not False:
        raise ACWMFrozenVerifierError(f"ACWM_FROZEN_SETTLEMENT_AUTHORITY_INVALID:{candidate_id}")

    if candidate_root.is_symlink() or not candidate_root.is_dir():
        raise ACWMFrozenVerifierError(f"ACWM_FROZEN_CANDIDATE_ROOT_INVALID:{candidate_id}")
    expected_measurement_path = _require_regular_file(
        candidate_root / "measurement" / "measurement.json",
        f"ACWM_FROZEN_MEASUREMENT_INVALID:{candidate_id}",
    )
    _require_regular_file(
        expected_measurement_path.parent / "manifest.json",
        f"ACWM_FROZEN_MEASUREMENT_MANIFEST_INVALID:{candidate_id}",
    )
    candidate_receipt = settlement.get("candidate_receipt")
    baseline_receipt = settlement.get("baseline_receipt")
    if not isinstance(candidate_receipt, Mapping) or not isinstance(baseline_receipt, Mapping):
        raise ACWMFrozenVerifierError(f"ACWM_FROZEN_RECEIPT_BINDING_MISSING:{candidate_id}")
    if Path(str(candidate_receipt.get("path"))).expanduser().resolve() != expected_measurement_path.resolve():
        raise ACWMFrozenVerifierError(f"ACWM_FROZEN_MEASUREMENT_PATH_MISMATCH:{candidate_id}")
    if Path(str(baseline_receipt.get("path"))).expanduser().resolve() != baseline_path.resolve():
        raise ACWMFrozenVerifierError(f"ACWM_FROZEN_BASELINE_PATH_MISMATCH:{candidate_id}")
    if baseline_receipt.get("sha256") != baseline_sha256:
        raise ACWMFrozenVerifierError(f"ACWM_FROZEN_BASELINE_RECEIPT_MISMATCH:{candidate_id}")

    measurement, measurement_sha256 = load_measurement_receipt(
        expected_measurement_path,
        contract=contract,
        stage=expected_stage,
        root=schema_root,
    )
    if candidate_receipt.get("sha256") != measurement_sha256:
        raise ACWMFrozenVerifierError(f"ACWM_FROZEN_MEASUREMENT_HASH_MISMATCH:{candidate_id}")
    descriptor = _candidate_descriptor(candidate)
    if measurement.get("candidate") != descriptor:
        raise ACWMFrozenVerifierError(f"ACWM_FROZEN_MEASUREMENT_CANDIDATE_MISMATCH:{candidate_id}")
    try:
        assessment = assess_acwm_dual_evaluation(
            contract,
            stage=expected_stage,
            baseline=baseline,
            candidate=measurement,
            root=schema_root,
        )
    except ACWMDualEvaluationError as exc:
        raise ACWMFrozenVerifierError(
            f"ACWM_FROZEN_ASSESSMENT_INVALID:{candidate_id}:{exc}"
        ) from exc
    _compare_settlement(settlement, descriptor=descriptor, assessment=assessment, measurement=measurement)
    if summary_row.get("state") != assessment["state"]:
        raise ACWMFrozenVerifierError(f"ACWM_FROZEN_SUMMARY_STATE_MISMATCH:{candidate_id}")

    worker_path = _require_regular_file(
        candidate_root / "worker-receipt.json", "ACWM_FROZEN_WORKER_RECEIPT_INVALID"
    )
    worker = _load(worker_path, "ACWM_FROZEN_WORKER_RECEIPT_INVALID")
    if worker.get("candidate_id") != candidate_id:
        raise ACWMFrozenVerifierError(f"ACWM_FROZEN_WORKER_CANDIDATE_MISMATCH:{candidate_id}")
    if worker.get("state") not in {"completed", "recovered_measurement"}:
        raise ACWMFrozenVerifierError(f"ACWM_FROZEN_WORKER_STATE_INVALID:{candidate_id}")
    if worker.get("state") == "completed" and worker.get("exit_code") != 0:
        raise ACWMFrozenVerifierError(f"ACWM_FROZEN_WORKER_EXIT_INVALID:{candidate_id}")

    evidence_refs = [
        _archive_file(cas, settlement_path, "application/json"),
        _archive_file(cas, expected_measurement_path, "application/json"),
        _archive_file(cas, expected_measurement_path.parent / "manifest.json", "application/json"),
        _archive_file(cas, worker_path, "application/json"),
    ]
    return {
        "candidate": descriptor,
        "assessment": {
            "state": assessment["state"],
            "accepted": assessment["accepted"],
            "metric_deltas": assessment["metric_deltas"],
            "blockers": assessment["blockers"],
        },
        "evidence_refs": evidence_refs,
    }


def _validate_campaign_headers(
    *,
    summary: Mapping[str, object],
    input_lock: Mapping[str, object],
    expected_stage: str,
    contract: Mapping[str, object],
    contract_path: Path,
) -> None:
    if summary.get("artifact_type") != "verdiwm-acwm-campaign-summary":
        raise ACWMFrozenVerifierError("ACWM_FROZEN_CAMPAIGN_SUMMARY_TYPE_INVALID")
    if summary.get("stage") != expected_stage or summary.get("state") != _TERMINAL_CAMPAIGN_STATE:
        raise ACWMFrozenVerifierError("ACWM_FROZEN_CAMPAIGN_STATE_INVALID")
    for field in ("contract_id", "contract_digest", "batch_id", "batch_digest", "input_sha256"):
        if summary.get(field) != input_lock.get(field):
            raise ACWMFrozenVerifierError(f"ACWM_FROZEN_CAMPAIGN_{field.upper()}_MISMATCH")
    if summary.get("contract_id") != contract.get("contract_id"):
        raise ACWMFrozenVerifierError("ACWM_FROZEN_CAMPAIGN_CONTRACT_ID_MISMATCH")
    if summary.get("contract_digest") != contract.get("contract_digest"):
        raise ACWMFrozenVerifierError("ACWM_FROZEN_CAMPAIGN_CONTRACT_DIGEST_MISMATCH")
    if input_lock.get("contract_file_sha256") != sha256_file(contract_path):
        raise ACWMFrozenVerifierError("ACWM_FROZEN_CONTRACT_FILE_HASH_MISMATCH")


def _compare_settlement(
    settlement: Mapping[str, object],
    *,
    descriptor: Mapping[str, object],
    assessment: Mapping[str, object],
    measurement: Mapping[str, object],
) -> None:
    candidate_id = str(descriptor["candidate_id"])
    expected = {
        "candidate": descriptor,
        "state": assessment["state"],
        "accepted": assessment["accepted"],
        "metrics": measurement["metrics"],
        "metric_deltas": assessment["metric_deltas"],
        "blockers": assessment["blockers"],
    }
    for field, value in expected.items():
        if settlement.get(field) != value:
            raise ACWMFrozenVerifierError(
                f"ACWM_FROZEN_SETTLEMENT_RECOMPUTE_MISMATCH:{candidate_id}:{field}"
            )


def _select_candidate(
    eligible: list[dict[str, object]], *, primary_metric: str
) -> dict[str, object] | None:
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda row: (
            -float(row["confirm"]["metric_deltas"][primary_metric]),
            str(row["candidate"]["candidate_id"]),
        ),
    )


def _primary_metric_id(contract: Mapping[str, object]) -> str:
    metrics = contract.get("metrics")
    if not isinstance(metrics, list):
        raise ACWMFrozenVerifierError("ACWM_FROZEN_PRIMARY_METRIC_INVALID")
    primary = [
        str(row["metric_id"])
        for row in metrics
        if isinstance(row, Mapping) and row.get("role") == "primary"
    ]
    if len(primary) != 1:
        raise ACWMFrozenVerifierError("ACWM_FROZEN_PRIMARY_METRIC_INVALID")
    return primary[0]


def _candidate_descriptor(candidate: Mapping[str, object]) -> dict[str, object]:
    return {
        "candidate_id": str(candidate["candidate_id"]),
        "candidate_kind": str(candidate["candidate_kind"]),
        "parameters": {"guidance_scale": float(candidate["guidance_scale"])},
    }


def _archive_file(cas: ContentAddressedStore, path: Path, media_type: str) -> str:
    source = _require_regular_file(path, "ACWM_FROZEN_ARCHIVE_SOURCE_INVALID")
    try:
        return cas.put_bytes(source.read_bytes(), media_type=media_type).uri
    except (OSError, ArchiveInvariantError) as exc:
        raise ACWMFrozenVerifierError("ACWM_FROZEN_ARCHIVE_WRITE_FAILED") from exc


def _path_from_lock(lock: Mapping[str, object], field: str, error_code: str) -> Path:
    value = lock.get(field)
    if not isinstance(value, str) or not value:
        raise ACWMFrozenVerifierError(error_code)
    return _require_regular_file(Path(value), error_code)


def _require_regular_file(path: Path, error_code: str) -> Path:
    raw = Path(path).expanduser()
    if raw.is_symlink():
        raise ACWMFrozenVerifierError(error_code)
    source = raw.resolve()
    if not source.is_file():
        raise ACWMFrozenVerifierError(error_code)
    return source


def _load(path: Path, error_code: str) -> dict[str, object]:
    try:
        return load_mapping(path, error_code=error_code)
    except ACWMCampaignError as exc:
        raise ACWMFrozenVerifierError(error_code) from exc


def _validate_output(
    schema_name: str, payload: Mapping[str, object], *, schema_root: Path | None
) -> None:
    try:
        validate_document(schema_name, payload, root=schema_root)
    except ContractValidationError as exc:
        raise ACWMFrozenVerifierError(f"ACWM_FROZEN_OUTPUT_SCHEMA_INVALID:{exc}") from exc


def _write_json_idempotent(path: Path, payload: Mapping[str, object]) -> None:
    try:
        encoded = canonical_json_bytes(payload)
    except ACWMCampaignError as exc:
        raise ACWMFrozenVerifierError("ACWM_FROZEN_OUTPUT_PAYLOAD_INVALID") from exc
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != encoded:
            raise ACWMFrozenVerifierError("ACWM_FROZEN_IMMUTABLE_WRITE_CONFLICT")
        return
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--screen-root", type=Path, required=True)
    parser.add_argument("--confirm-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = run_acwm_frozen_verifier(
            policy_path=args.policy,
            contract_path=args.contract,
            screen_root=args.screen_root,
            confirm_root=args.confirm_root,
            output_root=args.output_root,
        )
    except ACWMFrozenVerifierError as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
