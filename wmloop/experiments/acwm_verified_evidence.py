"""Project one frozen ACWM verdict into graph-ready verified evidence rows."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from pathlib import Path

from wmloop.archive.store import ArchiveInvariantError, ContentAddressedStore
from wmloop.contracts import ContractValidationError, validate_document
from wmloop.control.acwm_campaign import ACWMCampaignError, canonical_json_bytes, load_mapping


class ACWMVerifiedEvidenceError(ValueError):
    """A frozen verdict could not be projected without changing its meaning."""


def project_acwm_verified_evidence(
    *,
    verifier_root: Path,
    output_path: Path,
    project_root: Path | None = None,
) -> dict[str, object]:
    """Write deterministic JSONL rows whose authority remains the CAS verdict."""

    raw_root = Path(verifier_root).expanduser()
    if raw_root.is_symlink():
        raise ACWMVerifiedEvidenceError("ACWM_VERIFIED_EVIDENCE_ROOT_INVALID")
    root = raw_root.resolve()
    if not root.is_dir():
        raise ACWMVerifiedEvidenceError("ACWM_VERIFIED_EVIDENCE_ROOT_INVALID")
    manifest_path = root / "verification-manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ACWMVerifiedEvidenceError("ACWM_VERIFIED_EVIDENCE_MANIFEST_INVALID")
    try:
        manifest = load_mapping(
            manifest_path, error_code="ACWM_VERIFIED_EVIDENCE_MANIFEST_INVALID"
        )
    except ACWMCampaignError as exc:
        raise ACWMVerifiedEvidenceError("ACWM_VERIFIED_EVIDENCE_MANIFEST_INVALID") from exc
    schema_root = Path(project_root).resolve() if project_root is not None else None
    _validate("acwm_verification_manifest", manifest, schema_root=schema_root)
    verdict_ref = str(manifest["verdict_ref"])
    try:
        verdict_bytes = ContentAddressedStore(root).read_bytes(verdict_ref)
    except ArchiveInvariantError as exc:
        raise ACWMVerifiedEvidenceError("ACWM_VERIFIED_EVIDENCE_VERDICT_INVALID") from exc
    if manifest.get("verdict_sha256") != _digest_from_ref(verdict_ref):
        raise ACWMVerifiedEvidenceError("ACWM_VERIFIED_EVIDENCE_VERDICT_DIGEST_MISMATCH")
    try:
        verdict = json.loads(verdict_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ACWMVerifiedEvidenceError("ACWM_VERIFIED_EVIDENCE_VERDICT_INVALID") from exc
    if not isinstance(verdict, dict):
        raise ACWMVerifiedEvidenceError("ACWM_VERIFIED_EVIDENCE_VERDICT_INVALID")
    _validate("acwm_frozen_verdict", verdict, schema_root=schema_root)
    if verdict.get("verification_state") != "verified" or verdict.get("verdict_authority") is not True:
        raise ACWMVerifiedEvidenceError("ACWM_VERIFIED_EVIDENCE_AUTHORITY_INVALID")
    for field in ("contract_digest", "policy_digest", "verifier_implementation_sha256"):
        if verdict.get(field) != manifest.get(field):
            raise ACWMVerifiedEvidenceError(
                f"ACWM_VERIFIED_EVIDENCE_MANIFEST_BINDING_MISMATCH:{field}"
            )

    selected = verdict.get("selected_candidate")
    selected_id = selected.get("candidate_id") if isinstance(selected, Mapping) else None
    candidate_rows = verdict.get("candidates")
    if not isinstance(candidate_rows, list) or not candidate_rows:
        raise ACWMVerifiedEvidenceError("ACWM_VERIFIED_EVIDENCE_CANDIDATES_INVALID")
    rows = [
        _evidence_row(
            row,
            selected_id=str(selected_id) if selected_id is not None else None,
            verdict=verdict,
            verdict_ref=verdict_ref,
            schema_root=schema_root,
        )
        for row in candidate_rows
        if isinstance(row, Mapping)
    ]
    if len(rows) != len(candidate_rows):
        raise ACWMVerifiedEvidenceError("ACWM_VERIFIED_EVIDENCE_CANDIDATES_INVALID")
    encoded = b"".join(canonical_json_bytes(row) for row in rows)
    _write_bytes_idempotent(Path(output_path).expanduser().resolve(), encoded)
    return {
        "schema_version": 1,
        "artifact_type": "verdiwm-acwm-verified-evidence-projection",
        "state": "ready",
        "verdict_ref": verdict_ref,
        "record_count": len(rows),
        "selected_candidate_id": selected_id,
        "output_path": str(Path(output_path).expanduser().resolve()),
    }


def _evidence_row(
    row: Mapping[str, object],
    *,
    selected_id: str | None,
    verdict: Mapping[str, object],
    verdict_ref: str,
    schema_root: Path | None,
) -> dict[str, object]:
    candidate = row.get("candidate")
    confirm = row.get("confirm")
    if not isinstance(candidate, Mapping) or not isinstance(confirm, Mapping):
        raise ACWMVerifiedEvidenceError("ACWM_VERIFIED_EVIDENCE_CANDIDATE_ROW_INVALID")
    candidate_id = str(candidate.get("candidate_id") or "")
    state = str(row.get("verification_state") or "")
    if state == "verified_rejected":
        outcome = "rejected_harmful"
        promotion_state = "rejected"
    elif state == "verified_eligible" and candidate_id == selected_id:
        outcome = "selected_positive"
        promotion_state = "target_selected"
    elif state == "verified_eligible":
        outcome = "confirmed_positive"
        promotion_state = "verified_not_selected"
    else:
        raise ACWMVerifiedEvidenceError("ACWM_VERIFIED_EVIDENCE_CANDIDATE_STATE_INVALID")
    evidence_refs = row.get("evidence_refs")
    blockers = row.get("blockers")
    if not isinstance(evidence_refs, list) or not isinstance(blockers, list):
        raise ACWMVerifiedEvidenceError("ACWM_VERIFIED_EVIDENCE_CANDIDATE_ROW_INVALID")
    payload = {
        "schema_version": 1,
        "artifact_type": "verdiwm-acwm-verified-evidence",
        "candidate_id": candidate_id,
        "candidate": dict(candidate),
        "model_family": "ctrl-world",
        "stage": "confirm",
        "settlement_state": "settled",
        "verification_state": "verified",
        "verdict_ref": verdict_ref,
        "contract_id": verdict["contract_id"],
        "contract_digest": verdict["contract_digest"],
        "policy_digest": verdict["policy_digest"],
        "outcome": outcome,
        "promotion_state": promotion_state,
        "metric_deltas": confirm["metric_deltas"],
        "blockers": list(blockers),
        "evidence_refs": sorted({verdict_ref, *(str(value) for value in evidence_refs)}),
        "claim_boundary": verdict["claim_boundary"],
    }
    _validate("acwm_verified_evidence", payload, schema_root=schema_root)
    return payload


def _validate(
    schema_name: str,
    payload: Mapping[str, object],
    *,
    schema_root: Path | None,
) -> None:
    try:
        validate_document(schema_name, payload, root=schema_root)
    except ContractValidationError as exc:
        raise ACWMVerifiedEvidenceError(
            f"ACWM_VERIFIED_EVIDENCE_SCHEMA_INVALID:{schema_name}:{exc}"
        ) from exc


def _digest_from_ref(verdict_ref: str) -> str:
    prefix = "cas://sha256/"
    if not verdict_ref.startswith(prefix):
        raise ACWMVerifiedEvidenceError("ACWM_VERIFIED_EVIDENCE_VERDICT_REF_INVALID")
    digest = verdict_ref[len(prefix) :]
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ACWMVerifiedEvidenceError("ACWM_VERIFIED_EVIDENCE_VERDICT_REF_INVALID")
    return digest


def _write_bytes_idempotent(path: Path, payload: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise ACWMVerifiedEvidenceError("ACWM_VERIFIED_EVIDENCE_IMMUTABLE_WRITE_CONFLICT")
        return
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verifier-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = project_acwm_verified_evidence(
            verifier_root=args.verifier_root,
            output_path=args.output,
        )
    except ACWMVerifiedEvidenceError as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
