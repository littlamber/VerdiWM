"""Project a materialized-method frozen verdict into shared graph-ready knowledge."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path

from wmloop.archive.store import ArchiveInvariantError, ContentAddressedStore
from wmloop.contracts import ContractValidationError, validate_document
from wmloop.control.acwm_campaign import canonical_json_bytes, load_mapping


class MaterializedTransferEvidenceError(ValueError):
    """A frozen materialized-method verdict could not be projected faithfully."""


def project_materialized_transfer_evidence(
    *,
    verifier_root: Path,
    output_path: Path,
    project_root: Path | None = None,
) -> dict[str, object]:
    root = Path(verifier_root).expanduser().resolve()
    if root.is_symlink() or not root.is_dir():
        raise MaterializedTransferEvidenceError("MATERIALIZED_EVIDENCE_ROOT_INVALID")
    manifest = load_mapping(
        root / "verification-manifest.json",
        error_code="MATERIALIZED_EVIDENCE_MANIFEST_INVALID",
    )
    schema_root = Path(project_root).resolve() if project_root is not None else None
    _validate("acwm_materialized_verification_manifest", manifest, schema_root)
    verdict_ref = str(manifest["verdict_ref"])
    try:
        verdict = json.loads(ContentAddressedStore(root).read_bytes(verdict_ref))
    except (ArchiveInvariantError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MaterializedTransferEvidenceError("MATERIALIZED_EVIDENCE_VERDICT_INVALID") from exc
    if not isinstance(verdict, dict):
        raise MaterializedTransferEvidenceError("MATERIALIZED_EVIDENCE_VERDICT_INVALID")
    _validate("acwm_materialized_frozen_verdict", verdict, schema_root)
    if (
        verdict.get("verification_state") != "verified"
        or verdict.get("verdict_authority") is not True
        or verdict.get("decision") != manifest.get("decision")
        or verdict.get("policy_digest") != manifest.get("policy_digest")
        or verdict.get("verifier_implementation_sha256")
        != manifest.get("verifier_implementation_sha256")
    ):
        raise MaterializedTransferEvidenceError("MATERIALIZED_EVIDENCE_BINDING_MISMATCH")
    candidate = verdict["candidate"]
    assert isinstance(candidate, Mapping)
    provenance = candidate.get("provenance")
    if not isinstance(provenance, Mapping):
        raise MaterializedTransferEvidenceError("MATERIALIZED_EVIDENCE_PROVENANCE_INVALID")
    stage = str(verdict["evidence_stage"])
    stage_row = verdict["confirm"] if stage == "confirm" else verdict["screen"]
    if not isinstance(stage_row, Mapping):
        raise MaterializedTransferEvidenceError("MATERIALIZED_EVIDENCE_STAGE_INVALID")
    decision = str(verdict["decision"])
    promotion = {
        "confirmed_positive": "verified_target_positive",
        "rejected_at_screen": "verified_negative_boundary",
        "rejected_at_confirm": "verified_negative_boundary",
        "operational_failure": "no_scientific_claim",
    }[decision]
    row = {
        "schema_version": 1,
        "artifact_type": "verdiwm-materialized-transfer-evidence",
        "candidate_id": candidate["candidate_id"],
        "candidate": dict(candidate),
        "model_family": "ctrl-world",
        "stage": stage,
        "settlement_state": "settled",
        "verification_state": "verified",
        "verdict_ref": verdict_ref,
        "contract_id": verdict["contract_id"],
        "contract_digest": verdict["contract_digest"],
        "policy_digest": verdict["policy_digest"],
        "outcome": decision,
        "promotion_state": promotion,
        "source_id": provenance["source_id"],
        "source_digest": provenance["source_digest"],
        "assessment_digest": provenance["assessment_digest"],
        "implementation_revision": provenance["implementation_revision"],
        "metric_deltas": stage_row.get("metric_deltas", {}),
        "blockers": stage_row.get("blockers", []),
        "evidence_refs": sorted(
            {verdict_ref, *(str(value) for value in verdict["evidence_refs"])}
        ),
        "claim_boundary": verdict["claim_boundary"],
    }
    _validate("materialized_transfer_evidence", row, schema_root)
    destination = Path(output_path).expanduser().resolve()
    _write_idempotent(destination, canonical_json_bytes(row))
    return {
        "schema_version": 1,
        "artifact_type": "verdiwm-materialized-transfer-evidence-projection",
        "state": "ready",
        "verdict_ref": verdict_ref,
        "verifier_ref": verdict["verifier_ref"],
        "record_count": 1,
        "candidate_id": candidate["candidate_id"],
        "outcome": decision,
        "output_path": str(destination),
    }


def _validate(schema: str, payload: Mapping[str, object], root: Path | None) -> None:
    try:
        validate_document(schema, payload, root=root)
    except ContractValidationError as exc:
        raise MaterializedTransferEvidenceError(
            f"MATERIALIZED_EVIDENCE_SCHEMA_INVALID:{schema}:{exc}"
        ) from exc


def _write_idempotent(path: Path, payload: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise MaterializedTransferEvidenceError(
                "MATERIALIZED_EVIDENCE_IMMUTABLE_WRITE_CONFLICT"
            )
        return
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(payload)
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verifier-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = project_materialized_transfer_evidence(
        verifier_root=args.verifier_root, output_path=args.output
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
