"""Independent frozen verifier for receipt-bound materialized ACWM methods."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path

from wmloop.archive.store import ArchiveInvariantError, ContentAddressedStore
from wmloop.contracts import ContractValidationError, validate_document
from wmloop.control.acwm_campaign import canonical_json_bytes, load_mapping, sha256_file
from wmloop.control.acwm_dual_evaluation import (
    ACWMDualEvaluationError,
    assess_acwm_dual_evaluation,
    validate_acwm_dual_evaluation_contract,
)
from wmloop.control.acwm_materialized_campaign import (
    candidate_binding_digest,
    load_baseline,
    load_materialized_measurement,
    validate_materialized_candidate_batch,
)


class ACWMMaterializedFrozenVerifierError(ValueError):
    """Frozen evidence failed an integrity or scientific-policy check."""


def policy_digest(policy: Mapping[str, object]) -> str:
    payload = {key: value for key, value in policy.items() if key != "policy_digest"}
    return hashlib.sha256(canonical_json_bytes(payload).rstrip(b"\n")).hexdigest()


def run_materialized_frozen_verifier(
    *,
    policy_path: Path,
    contract_path: Path,
    screen_root: Path,
    confirm_root: Path | None,
    output_root: Path,
    project_root: Path | None = None,
) -> dict[str, object]:
    schema_root = Path(project_root).resolve() if project_root is not None else None
    policy_source = _require_file(policy_path, "ACWM_MATERIALIZED_POLICY_INVALID")
    contract_source = _require_file(contract_path, "ACWM_MATERIALIZED_CONTRACT_INVALID")
    policy = _load(policy_source, "ACWM_MATERIALIZED_POLICY_INVALID")
    contract = _load(contract_source, "ACWM_MATERIALIZED_CONTRACT_INVALID")
    _validate_policy(policy, contract, schema_root=schema_root)
    implementation = Path(__file__).resolve()
    implementation_sha256 = sha256_file(implementation)
    if policy.get("implementation_sha256") != implementation_sha256:
        raise ACWMMaterializedFrozenVerifierError(
            "ACWM_MATERIALIZED_VERIFIER_IMPLEMENTATION_MISMATCH"
        )
    destination = Path(output_root).expanduser().resolve()
    _prepare_destination(destination)
    cas = ContentAddressedStore(destination)
    screen = _verify_stage(
        stage_root=screen_root,
        expected_stage="screen",
        contract=contract,
        cas=cas,
        schema_root=schema_root,
    )
    screen_state = str(screen["settlement"].get("state"))
    screen_accepted = screen["assessment"].get("accepted") is True
    confirm = None
    if screen_state == "failed":
        decision = "operational_failure"
        evidence_stage = "screen"
    elif not screen_accepted:
        decision = "rejected_at_screen"
        evidence_stage = "screen"
        if confirm_root is not None:
            raise ACWMMaterializedFrozenVerifierError(
                "ACWM_MATERIALIZED_CONFIRM_FOR_REJECTED_SCREEN_FORBIDDEN"
            )
    else:
        if confirm_root is None:
            raise ACWMMaterializedFrozenVerifierError(
                "ACWM_MATERIALIZED_CONFIRM_REQUIRED"
            )
        confirm = _verify_stage(
            stage_root=confirm_root,
            expected_stage="confirm",
            contract=contract,
            cas=cas,
            schema_root=schema_root,
        )
        if confirm["candidate"] != screen["candidate"]:
            raise ACWMMaterializedFrozenVerifierError(
                "ACWM_MATERIALIZED_CONFIRM_CANDIDATE_MISMATCH"
            )
        confirm_state = str(confirm["settlement"].get("state"))
        if confirm_state == "failed":
            decision = "operational_failure"
        elif confirm["assessment"].get("accepted") is True:
            decision = "confirmed_positive"
        else:
            decision = "rejected_at_confirm"
        evidence_stage = "confirm"
    contract_ref = _archive(cas, contract_source)
    policy_ref = _archive(cas, policy_source)
    verifier_ref = _archive(cas, implementation)
    refs = {
        str(value)
        for value in [
            contract_ref.uri,
            policy_ref.uri,
            verifier_ref.uri,
            *screen["evidence_refs"],
            *(confirm["evidence_refs"] if confirm is not None else []),
        ]
    }
    verdict = {
        "schema_version": 1,
        "artifact_type": "verdiwm-acwm-materialized-frozen-verdict",
        "verification_state": "verified",
        "verdict_authority": True,
        "decision": decision,
        "evidence_stage": evidence_stage,
        "contract_id": contract["contract_id"],
        "contract_digest": contract["contract_digest"],
        "contract_ref": contract_ref.uri,
        "policy_id": policy["policy_id"],
        "policy_digest": policy["policy_digest"],
        "policy_ref": policy_ref.uri,
        "verifier_implementation_sha256": implementation_sha256,
        "verifier_ref": verifier_ref.uri,
        "candidate": screen["candidate"],
        "screen": _stage_verdict_row(screen),
        "confirm": _stage_verdict_row(confirm) if confirm is not None else None,
        "evidence_refs": sorted(refs),
        "claim_boundary": (
            "This verdict settles only the source-grounded mechanism transfer on Ctrl-World "
            "under the frozen ACWM contexts. A positive does not claim paper reproduction; "
            "a rejection records an applicability boundary rather than erasing the method."
        ),
    }
    _validate_output(
        "acwm_materialized_frozen_verdict", verdict, schema_root=schema_root
    )
    verdict_artifact = cas.put_bytes(
        canonical_json_bytes(verdict), media_type="application/json"
    )
    manifest = {
        "schema_version": 1,
        "artifact_type": "verdiwm-acwm-materialized-verification-manifest",
        "state": "verified",
        "decision": decision,
        "evidence_stage": evidence_stage,
        "candidate_id": screen["candidate"]["candidate_id"],
        "verdict_ref": verdict_artifact.uri,
        "verdict_sha256": verdict_artifact.sha256,
        "policy_digest": policy["policy_digest"],
        "verifier_implementation_sha256": implementation_sha256,
        "contract_digest": contract["contract_digest"],
        "screen_input_sha256": screen["input_sha256"],
        "confirm_input_sha256": confirm["input_sha256"] if confirm is not None else None,
    }
    _validate_output(
        "acwm_materialized_verification_manifest", manifest, schema_root=schema_root
    )
    _write_json_idempotent(destination / "verification-manifest.json", manifest)
    return manifest


def _verify_stage(
    *,
    stage_root: Path,
    expected_stage: str,
    contract: Mapping[str, object],
    cas: ContentAddressedStore,
    schema_root: Path | None,
) -> dict[str, object]:
    root = _require_directory(
        stage_root, f"ACWM_MATERIALIZED_{expected_stage.upper()}_ROOT_INVALID"
    )
    summary_path = _require_file(
        root / "campaign-summary.json",
        f"ACWM_MATERIALIZED_{expected_stage.upper()}_SUMMARY_INVALID",
    )
    input_lock_path = _require_file(
        root / "input-lock.json",
        f"ACWM_MATERIALIZED_{expected_stage.upper()}_INPUT_LOCK_INVALID",
    )
    summary = _load(summary_path, "ACWM_MATERIALIZED_SUMMARY_INVALID")
    input_lock = _load(input_lock_path, "ACWM_MATERIALIZED_INPUT_LOCK_INVALID")
    if (
        summary.get("artifact_type") != "verdiwm-acwm-materialized-campaign-summary"
        or summary.get("state") != "settled"
        or summary.get("stage") != expected_stage
        or summary.get("input_sha256") != input_lock.get("input_sha256")
    ):
        raise ACWMMaterializedFrozenVerifierError(
            "ACWM_MATERIALIZED_STAGE_HEADER_INVALID"
        )
    batch_path = _locked_file(input_lock, "batch", "ACWM_MATERIALIZED_BATCH")
    batch = _load(batch_path, "ACWM_MATERIALIZED_BATCH_INVALID")
    validate_materialized_candidate_batch(batch, contract, root=schema_root)
    if batch.get("stage") != expected_stage or batch.get("batch_id") != summary.get("batch_id"):
        raise ACWMMaterializedFrozenVerifierError(
            "ACWM_MATERIALIZED_STAGE_BATCH_MISMATCH"
        )
    candidates = batch["candidates"]
    assert isinstance(candidates, list)
    if len(candidates) != 1 or not isinstance(candidates[0], Mapping):
        raise ACWMMaterializedFrozenVerifierError(
            "ACWM_MATERIALIZED_STAGE_CANDIDATE_SET_INVALID"
        )
    candidate = dict(candidates[0])
    candidate_id = str(candidate["candidate_id"])
    summary_rows = summary.get("candidates")
    if not isinstance(summary_rows, list) or len(summary_rows) != 1 or not isinstance(summary_rows[0], Mapping):
        raise ACWMMaterializedFrozenVerifierError(
            "ACWM_MATERIALIZED_STAGE_SUMMARY_CANDIDATES_INVALID"
        )
    summary_row = summary_rows[0]
    if summary_row.get("candidate_id") != candidate_id:
        raise ACWMMaterializedFrozenVerifierError(
            "ACWM_MATERIALIZED_STAGE_CANDIDATE_MISMATCH"
        )
    candidate_root = _require_directory(
        root / "candidates" / candidate_id,
        "ACWM_MATERIALIZED_CANDIDATE_ROOT_INVALID",
    )
    candidate_path = _require_file(
        candidate_root / "candidate.json", "ACWM_MATERIALIZED_CANDIDATE_FILE_INVALID"
    )
    if _load(candidate_path, "ACWM_MATERIALIZED_CANDIDATE_FILE_INVALID") != candidate:
        raise ACWMMaterializedFrozenVerifierError(
            "ACWM_MATERIALIZED_CANDIDATE_FILE_MISMATCH"
        )
    settlement_path = _require_file(
        candidate_root / "settlement.json", "ACWM_MATERIALIZED_SETTLEMENT_INVALID"
    )
    if sha256_file(settlement_path) != summary_row.get("settlement_sha256"):
        raise ACWMMaterializedFrozenVerifierError(
            "ACWM_MATERIALIZED_SETTLEMENT_HASH_MISMATCH"
        )
    settlement = _load(settlement_path, "ACWM_MATERIALIZED_SETTLEMENT_INVALID")
    if (
        settlement.get("candidate") != candidate
        or settlement.get("candidate_binding_sha256") != candidate_binding_digest(candidate)
        or settlement.get("batch_digest") != batch.get("batch_digest")
        or settlement.get("input_sha256") != input_lock.get("input_sha256")
        or settlement.get("verdict_authority") is not False
    ):
        raise ACWMMaterializedFrozenVerifierError(
            "ACWM_MATERIALIZED_SETTLEMENT_BINDING_MISMATCH"
        )
    baseline_path = _locked_file(
        input_lock, "baseline", "ACWM_MATERIALIZED_BASELINE", hash_field="baseline_sha256"
    )
    baseline, baseline_sha256 = load_baseline(
        baseline_path, contract=contract, stage=expected_stage, root=schema_root
    )
    if baseline_sha256 != input_lock.get("baseline_sha256"):
        raise ACWMMaterializedFrozenVerifierError(
            "ACWM_MATERIALIZED_BASELINE_HASH_MISMATCH"
        )
    evidence_paths = [
        summary_path,
        input_lock_path,
        batch_path,
        baseline_path,
        baseline_path.parent / "manifest.json",
        candidate_path,
        settlement_path,
        candidate_root / "worker-receipt.json",
    ]
    if settlement.get("state") == "failed":
        assessment = {
            "state": "failed",
            "accepted": False,
            "metric_deltas": {},
            "blockers": list(settlement.get("blockers", [])),
        }
    else:
        measurement_path = _require_file(
            candidate_root / "measurement" / "measurement.json",
            "ACWM_MATERIALIZED_MEASUREMENT_INVALID",
        )
        measurement, measurement_sha256 = load_materialized_measurement(
            measurement_path,
            contract=contract,
            stage=expected_stage,
            candidate=candidate,
            root=schema_root,
        )
        try:
            assessment = assess_acwm_dual_evaluation(
                contract,
                stage=expected_stage,
                baseline=baseline,
                candidate=measurement,
                root=schema_root,
            )
        except ACWMDualEvaluationError as exc:
            raise ACWMMaterializedFrozenVerifierError(
                f"ACWM_MATERIALIZED_REASSESSMENT_INVALID:{exc}"
            ) from exc
        if (
            settlement.get("state") != assessment.get("state")
            or settlement.get("accepted") != assessment.get("accepted")
            or settlement.get("metric_deltas") != assessment.get("metric_deltas")
            or settlement.get("blockers") != assessment.get("blockers")
            or settlement.get("candidate_receipt", {}).get("sha256") != measurement_sha256
        ):
            raise ACWMMaterializedFrozenVerifierError(
                "ACWM_MATERIALIZED_REASSESSMENT_MISMATCH"
            )
        evidence_paths.extend([measurement_path, measurement_path.parent / "manifest.json"])
    provenance = candidate["provenance"]
    assert isinstance(provenance, Mapping)
    for key in (
        "candidate_catalog_path",
        "materialization_receipt_path",
        "descriptor_path",
        "source_assessment_path",
    ):
        evidence_paths.append(Path(str(provenance[key])))
    for row in provenance["required_files"]:
        if isinstance(row, Mapping):
            evidence_paths.append(Path(str(row["path"])))
    refs = [_archive(cas, _require_file(path, "ACWM_MATERIALIZED_EVIDENCE_FILE_INVALID")).uri for path in evidence_paths]
    return {
        "candidate": candidate,
        "settlement": settlement,
        "assessment": assessment,
        "input_sha256": input_lock["input_sha256"],
        "evidence_refs": sorted(set(refs)),
    }


def _stage_verdict_row(stage: Mapping[str, object] | None) -> dict[str, object] | None:
    if stage is None:
        return None
    assessment = stage["assessment"]
    assert isinstance(assessment, Mapping)
    settlement = stage["settlement"]
    assert isinstance(settlement, Mapping)
    return {
        "state": assessment.get("state"),
        "accepted": assessment.get("accepted"),
        "metric_deltas": assessment.get("metric_deltas"),
        "blockers": assessment.get("blockers"),
        "input_sha256": stage["input_sha256"],
        "evidence_refs": stage["evidence_refs"],
        "settlement_state": settlement.get("state"),
    }


def _validate_policy(
    policy: Mapping[str, object],
    contract: Mapping[str, object],
    *,
    schema_root: Path | None,
) -> None:
    _validate_output(
        "acwm_materialized_frozen_verifier_policy", policy, schema_root=schema_root
    )
    validate_acwm_dual_evaluation_contract(contract, root=schema_root)
    if policy.get("policy_digest") != policy_digest(policy):
        raise ACWMMaterializedFrozenVerifierError(
            "ACWM_MATERIALIZED_POLICY_DIGEST_MISMATCH"
        )
    if (
        policy.get("contract_id") != contract.get("contract_id")
        or policy.get("contract_digest") != contract.get("contract_digest")
    ):
        raise ACWMMaterializedFrozenVerifierError(
            "ACWM_MATERIALIZED_POLICY_CONTRACT_MISMATCH"
        )


def _locked_file(
    lock: Mapping[str, object],
    stem: str,
    code: str,
    *,
    hash_field: str | None = None,
) -> Path:
    path = _require_file(Path(str(lock.get(f"{stem}_path", ""))), f"{code}_INVALID")
    expected = lock.get(hash_field or f"{stem}_file_sha256")
    if sha256_file(path) != expected:
        raise ACWMMaterializedFrozenVerifierError(f"{code}_HASH_MISMATCH")
    return path


def _archive(cas: ContentAddressedStore, path: Path):
    media_type = "application/json" if path.suffix in {".json", ".jsonl"} else "text/plain"
    try:
        return cas.put_bytes(path.read_bytes(), media_type=media_type)
    except (OSError, ArchiveInvariantError) as exc:
        raise ACWMMaterializedFrozenVerifierError(
            "ACWM_MATERIALIZED_ARCHIVE_FAILED"
        ) from exc


def _prepare_destination(path: Path) -> None:
    if path.exists():
        if path.is_symlink() or not path.is_dir():
            raise ACWMMaterializedFrozenVerifierError(
                "ACWM_MATERIALIZED_OUTPUT_INVALID"
            )
        return
    path.mkdir(mode=0o700, parents=True, exist_ok=False)


def _validate_output(
    schema: str, payload: Mapping[str, object], *, schema_root: Path | None
) -> None:
    try:
        validate_document(schema, payload, root=schema_root)
    except ContractValidationError as exc:
        raise ACWMMaterializedFrozenVerifierError(
            f"ACWM_MATERIALIZED_SCHEMA_INVALID:{schema}:{exc}"
        ) from exc


def _require_file(path: Path, code: str) -> Path:
    raw = Path(path).expanduser()
    if raw.is_symlink():
        raise ACWMMaterializedFrozenVerifierError(code)
    resolved = raw.resolve()
    if not resolved.is_file():
        raise ACWMMaterializedFrozenVerifierError(code)
    return resolved


def _require_directory(path: Path, code: str) -> Path:
    raw = Path(path).expanduser()
    if raw.is_symlink():
        raise ACWMMaterializedFrozenVerifierError(code)
    resolved = raw.resolve()
    if not resolved.is_dir():
        raise ACWMMaterializedFrozenVerifierError(code)
    return resolved


def _load(path: Path, code: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ACWMMaterializedFrozenVerifierError(code) from exc
    if not isinstance(payload, dict):
        raise ACWMMaterializedFrozenVerifierError(code)
    return payload


def _write_json_idempotent(path: Path, payload: Mapping[str, object]) -> None:
    encoded = canonical_json_bytes(payload)
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != encoded:
            raise ACWMMaterializedFrozenVerifierError(
                "ACWM_MATERIALIZED_IMMUTABLE_WRITE_CONFLICT"
            )
        return
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(encoded)
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--screen-root", type=Path, required=True)
    parser.add_argument("--confirm-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_materialized_frozen_verifier(
        policy_path=args.policy,
        contract_path=args.contract,
        screen_root=args.screen_root,
        confirm_root=args.confirm_root,
        output_root=args.output_root,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
