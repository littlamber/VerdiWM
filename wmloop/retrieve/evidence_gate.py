"""Fail-closed evidence gates for research claims backed by external retrieval."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


class RetrievalEvidenceError(ValueError):
    """A retrieval manifest cannot support the requested research mode."""


def build_retrieval_evidence_gate(
    manifests: Sequence[Mapping[str, object]],
    *,
    require_network: bool,
    minimum_records: int = 1,
) -> dict[str, object]:
    """Summarize whether source-linked retrieval evidence is actually present."""

    if minimum_records < 1:
        raise RetrievalEvidenceError("RETRIEVAL_EVIDENCE_MINIMUM_INVALID")
    rows: list[dict[str, object]] = []
    for manifest in manifests:
        artifact_type = str(manifest.get("artifact_type") or "")
        if artifact_type == "verdiwm-literature-retrieval-manifest":
            source_state = str(manifest.get("state") or "unknown")
            record_count = _count(manifest.get("record_count"))
        elif artifact_type == "wmloop-mechanism-discovery-manifest":
            source_state = str(manifest.get("retrieval_state") or "unknown")
            record_count = _count(manifest.get("paper_count"))
        else:
            raise RetrievalEvidenceError("RETRIEVAL_EVIDENCE_MANIFEST_UNSUPPORTED")
        rows.append(
            {
                "artifact_type": artifact_type,
                "source_state": source_state,
                "record_count": record_count,
            }
        )
    total = sum(int(row["record_count"]) for row in rows)
    network_observed = any(row["source_state"] == "network" for row in rows)
    blockers = []
    if total < minimum_records:
        blockers.append("external_records_required")
    if require_network and not network_observed:
        blockers.append("live_network_retrieval_required")
    return {
        "schema_version": 1,
        "artifact_type": "verdiwm-retrieval-evidence-gate",
        "state": "blocked" if blockers else "ready",
        "require_network": require_network,
        "minimum_records": minimum_records,
        "record_count": total,
        "network_observed": network_observed,
        "sources": rows,
        "blockers": blockers,
        "claim_boundary": (
            "This gate proves that source-linked retrieval records exist under the "
            "declared source policy. It does not establish novelty, transferability, "
            "implementation correctness, or model improvement."
        ),
    }


def _count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RetrievalEvidenceError("RETRIEVAL_EVIDENCE_COUNT_INVALID")
    return value
