"""Resumable pipeline state and evidence components: state."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from wmloop.archive.store import ArchiveInvariantError, ContentAddressedStore
from wmloop.contracts import ContractValidationError, validate_document
from wmloop.control.research_state import ResearchJournal
from wmloop.execute.pipeline.io import _load_json, _write_json_atomic
from wmloop.execute.pipeline.types import AutonomousPipelineError
from wmloop.storage import canonical_json_bytes, checked_path, sha256_bytes


def _bind_output_root(
    destination: Path,
    *,
    input_document: Mapping[str, object],
    input_hash: str,
) -> None:
    lock_path = destination / "pipeline-input.lock.json"
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or not destination.is_dir():
            raise AutonomousPipelineError("AUTONOMOUS_PIPELINE_OUTPUT_INVALID")
        lock = _load_json(lock_path, "AUTONOMOUS_PIPELINE_INPUT_LOCK_INVALID")
        if lock.get("input_hash") != input_hash or lock.get("input") != input_document:
            raise AutonomousPipelineError("AUTONOMOUS_PIPELINE_INPUT_MISMATCH")
        return
    destination.mkdir(mode=0o700, parents=True)
    _write_json_atomic(
        lock_path,
        {
            "schema_version": 1,
            "artifact_type": "verdiwm-autonomous-pipeline-input-lock",
            "input_hash": input_hash,
            "input": dict(input_document),
        },
    )


def _settle_pipeline(
    destination: Path,
    *,
    input_hash: str,
    journal: ResearchJournal | None = None,
    state: str,
    verdict: str,
    blocked_stage: str | None,
    candidate_states: Mapping[str, object] | None = None,
    diagnostic_probe: Mapping[str, object] | None = None,
    retrieval: Mapping[str, object] | None = None,
    literature: Mapping[str, object] | None = None,
    literature_methods: Mapping[str, object] | None = None,
    literature_method_prompts: Mapping[str, object] | None = None,
    literature_method_materialization: Mapping[str, object] | None = None,
    method_candidates: Mapping[str, object] | None = None,
    cpbe: Mapping[str, object] | None = None,
    settlement_cas_root: Path | None = None,
) -> dict[str, object]:
    manifest = _manifest(
        destination,
        input_hash=input_hash,
        state=state,
        verdict=verdict,
        blocked_stage=blocked_stage,
        candidate_states=candidate_states,
        diagnostic_probe=diagnostic_probe,
        retrieval=retrieval,
        literature=literature,
        literature_methods=literature_methods,
        literature_method_prompts=literature_method_prompts,
        literature_method_materialization=literature_method_materialization,
        method_candidates=method_candidates,
        cpbe=cpbe,
    )
    _write_json_atomic(destination / "pipeline-manifest.json", manifest)
    if journal is not None:
        refs = list(journal.read()["evidence_refs"])
        if state == "settled":
            if settlement_cas_root is None:
                raise AutonomousPipelineError("AUTONOMOUS_PIPELINE_SETTLEMENT_STORE_REQUIRED")
            cas = ContentAddressedStore(settlement_cas_root)
            execution = _load_json(
                destination / "compiled" / "queue" / "execution.json",
                "AUTONOMOUS_PIPELINE_EXECUTION_INVALID",
            )
            for projection in (manifest, execution):
                ref = cas.put_bytes(canonical_json_bytes(projection), media_type="application/json")
                refs.append(ref.uri)
        journal.advance(
            "settled" if state == "settled" else "blocked",
            f"Pipeline {state}: {blocked_stage or 'execution receipts retained'}.",
            evidence_refs=list(dict.fromkeys(refs)),
        )
    return manifest


def verify_pipeline_settlement(
    *, destination: Path, manifest: Mapping[str, object],
    journal: ResearchJournal, cas_root: Path,
) -> dict[str, object]:
    """Bind both terminal projections to the journal's immutable CAS snapshots.

    The caller must additionally recover every trial through the authoritative
    runner. Snapshot equality is an integrity check, never verdict authority.
    Older unbound settlements require explicit reconciliation.
    """
    execution_path = checked_path(
        destination / "compiled" / "queue" / "execution.json",
        code="AUTONOMOUS_PIPELINE_SETTLEMENT_EVIDENCE_INVALID", error=AutonomousPipelineError,
    )
    execution = _load_json(execution_path, "AUTONOMOUS_PIPELINE_SETTLEMENT_EVIDENCE_INVALID")
    refs = journal.read()["evidence_refs"]
    cas = ContentAddressedStore(cas_root)
    for projection in (manifest, execution):
        payload = canonical_json_bytes(projection)
        ref = "cas://sha256/" + sha256_bytes(payload)
        if ref not in refs:
            raise AutonomousPipelineError("AUTONOMOUS_PIPELINE_SETTLEMENT_PROJECTION_MISMATCH")
        try:
            if cas.read_bytes(ref) != payload:
                raise AutonomousPipelineError("AUTONOMOUS_PIPELINE_SETTLEMENT_EVIDENCE_INVALID")
        except (ArchiveInvariantError, OSError) as exc:
            raise AutonomousPipelineError("AUTONOMOUS_PIPELINE_SETTLEMENT_EVIDENCE_INVALID") from exc
    return execution


def _manifest(
    destination: Path,
    *,
    input_hash: str,
    state: str,
    verdict: str | None,
    blocked_stage: str | None,
    candidate_states: Mapping[str, object] | None = None,
    error: Mapping[str, str] | None = None,
    diagnostic_probe: Mapping[str, object] | None = None,
    retrieval: Mapping[str, object] | None = None,
    literature: Mapping[str, object] | None = None,
    literature_methods: Mapping[str, object] | None = None,
    literature_method_prompts: Mapping[str, object] | None = None,
    literature_method_materialization: Mapping[str, object] | None = None,
    method_candidates: Mapping[str, object] | None = None,
    cpbe: Mapping[str, object] | None = None,
) -> dict[str, object]:
    manifest = {
        "schema_version": 1,
        "artifact_type": "verdiwm-autonomous-pipeline-manifest",
        "state": state,
        "verdict": verdict,
        "input_hash": input_hash,
        "blocked_stage": blocked_stage,
        "candidate_states": dict(candidate_states or {}),
        "diagnostic_probe": (
            dict(diagnostic_probe) if diagnostic_probe is not None else None
        ),
        "retrieval": dict(retrieval) if retrieval is not None else None,
        "literature": dict(literature) if literature is not None else None,
        "literature_methods": (
            dict(literature_methods) if literature_methods is not None else None
        ),
        "literature_method_prompts": (
            dict(literature_method_prompts)
            if literature_method_prompts is not None
            else None
        ),
        "literature_method_materialization": (
            dict(literature_method_materialization)
            if literature_method_materialization is not None
            else None
        ),
        "method_candidates": (
            dict(method_candidates) if method_candidates is not None else None
        ),
        "cpbe": dict(cpbe) if cpbe is not None else None,
        "paths": {
            "research_state": str(destination / "research" / "research-state.json"),
            "onboarding": str(destination / "onboarding" / "manifest.json"),
            "conformance": str(destination / "conformance" / "manifest.json"),
            "compilation": str(destination / "compiled" / "manifest.json"),
            "queue_execution": str(
                destination / "compiled" / "queue" / "execution.json"
            ),
        },
        "error": dict(error) if error is not None else None,
        "claim_boundary": "A PASS proves the declared evaluator completed and settled under its frozen runtime contract; model-quality promotion remains evaluator-specific.",
    }
    try:
        validate_document("autonomous_pipeline_manifest", manifest)
    except ContractValidationError as exc:
        raise AutonomousPipelineError("AUTONOMOUS_PIPELINE_MANIFEST_INVALID") from exc
    return manifest
