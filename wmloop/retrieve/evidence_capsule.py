"""Small runtime projection of settled retrieval evidence.

The Archive and CAS remain authoritative.  An evidence capsule is only the
bounded input needed by the next routing/compilation step, so the normal loop
does not have to carry a full evidence graph or every retrieval row.
"""

from __future__ import annotations

import json
from copy import deepcopy
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from wmloop.contracts import ContractValidationError, validate_document


class EvidenceCapsuleError(ValueError):
    """A capsule input or persistence invariant failed closed."""


_ROUTES = {"no_diagnostic", "cold_start", "reuse_settled"}


def build_evidence_capsule(
    *,
    probe: Mapping[str, Any] | None,
    matches: Sequence[Mapping[str, Any]] = (),
    max_evidence: int = 3,
) -> dict[str, object]:
    """Build a deterministic, bounded projection for runtime routing.

    ``matches`` must already have passed the receipt/CAS checks in the
    retrieval index.  The capsule copies only scalar routing fields and
    immutable references; it never makes a new scientific claim.
    """

    if max_evidence < 1 or max_evidence > 20:
        raise EvidenceCapsuleError("EVIDENCE_CAPSULE_LIMIT_INVALID")
    if probe is not None and not isinstance(probe, Mapping):
        raise EvidenceCapsuleError("EVIDENCE_CAPSULE_PROBE_INVALID")

    signatures = _signatures(probe)
    if probe is None:
        route = "no_diagnostic"
    elif matches:
        route = "reuse_settled"
    else:
        route = "cold_start"

    selected = [_compact_match(row) for row in matches]
    selected.sort(key=_match_sort_key)
    if any(row.get("paired_observations") for row in selected):
        selected = _select_paired_representatives(selected, max_evidence)
    else:
        selected = selected[:max_evidence]
    capsule: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "verdiwm-evidence-capsule",
        "state": "ready",
        "route": route,
        "query": {
            "model_family": _optional_string(probe, "model_family"),
            "runtime_capability": _optional_string(probe, "runtime_capability"),
            "failure_signatures": signatures,
        },
        "match_count": len(matches),
        "selected_evidence": selected,
        "claim_boundary": (
            "A capsule is a bounded routing projection. Receipt, CAS, Archive, "
            "frozen evaluators, and promotion gates remain authoritative."
        ),
    }
    return capsule


def write_evidence_capsule(path: Path, capsule: Mapping[str, object]) -> Path:
    """Atomically persist a validated capsule and return its resolved path."""

    _validate_capsule(capsule)
    destination = Path(path).expanduser().resolve()
    if destination.exists() and (destination.is_symlink() or not destination.is_file()):
        raise EvidenceCapsuleError("EVIDENCE_CAPSULE_OUTPUT_INVALID")
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(dict(capsule), ensure_ascii=True, sort_keys=True, separators=(",", ":"))
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()
    return destination


def _compact_match(row: Mapping[str, Any]) -> dict[str, object]:
    required = ("failure_signature", "verdict", "archive_trial_id", "receipt_ref", "result_ref", "receipt_hash")
    if any(not isinstance(row.get(key), str) or not str(row[key]) for key in required):
        raise EvidenceCapsuleError("EVIDENCE_CAPSULE_MATCH_INVALID")
    metric = row.get("metric_outcome")
    if metric is not None and (isinstance(metric, bool) or not isinstance(metric, (int, float)) or not math.isfinite(metric)):
        raise EvidenceCapsuleError("EVIDENCE_CAPSULE_METRIC_INVALID")
    primitive = row.get("primitive")
    if primitive is not None and (not isinstance(primitive, str) or not primitive):
        raise EvidenceCapsuleError("EVIDENCE_CAPSULE_PRIMITIVE_INVALID")
    compact = {
        "failure_signature": str(row["failure_signature"]),
        "verdict": str(row["verdict"]),
        "primitive": primitive,
        "metric_outcome": float(metric) if metric is not None else None,
        "evidence": {
            "archive_trial_id": str(row["archive_trial_id"]),
            "receipt_ref": str(row["receipt_ref"]),
            "result_ref": str(row["result_ref"]),
            "receipt_hash": str(row["receipt_hash"]),
        },
    }
    observations = row.get("paired_observations", [])
    try:
        if not isinstance(observations, (list, tuple)):
            raise EvidenceCapsuleError("EVIDENCE_CAPSULE_PAIRED_INVALID")
        for observation in observations:
            validate_document("diagnostic_paired_observation", observation)
    except ContractValidationError as exc:
        raise EvidenceCapsuleError("EVIDENCE_CAPSULE_PAIRED_INVALID") from exc
    if observations:
        compact["paired_observations"] = [
            {key: deepcopy(observation[key]) for key in ("query", "control_value", "candidate_value", "signed_delta", "outcome")}
            for observation in observations
        ]
    return compact


def _select_paired_representatives(rows: list[dict], limit: int) -> list[dict]:
    # Multiple failure signatures are routes to the same trial, not additional
    # experimental support. Preserve a representative of each signed outcome
    # before filling remaining slots; a PASS scalar cannot crowd out negatives.
    unique = []
    identities = set()
    for row in rows:
        evidence = row["evidence"]
        identity = (evidence["archive_trial_id"], evidence["receipt_ref"], evidence["result_ref"])
        if identity not in identities:
            identities.add(identity)
            unique.append(row)
    selected = []
    for outcome in ("negative", "null", "positive"):
        if any(pair["outcome"] == outcome for row in selected for pair in row.get("paired_observations", [])):
            continue
        representative = next((row for row in unique if row not in selected
                               and any(pair["outcome"] == outcome for pair in row.get("paired_observations", []))), None)
        if representative is not None and len(selected) < limit:
            selected.append(representative)
    selected.extend(row for row in unique if row not in selected)
    return selected[:limit]


def _match_sort_key(row: Mapping[str, object]) -> tuple[int, float, str, str]:
    metric = row.get("metric_outcome")
    score = float(metric) if isinstance(metric, (int, float)) else float("-inf")
    return (
        0 if row.get("verdict") == "PASS" else 1,
        -score,
        str(row.get("failure_signature", "")),
        str(row.get("evidence", {}).get("archive_trial_id", ""))
        if isinstance(row.get("evidence"), Mapping)
        else "",
    )


def _signatures(probe: Mapping[str, Any] | None) -> list[str]:
    if probe is None:
        return []
    values = probe.get("failure_signatures", [])
    if not isinstance(values, list) or any(not isinstance(value, str) or not value for value in values):
        raise EvidenceCapsuleError("EVIDENCE_CAPSULE_SIGNATURES_INVALID")
    return sorted(set(values))


def _optional_string(probe: Mapping[str, Any] | None, key: str) -> str | None:
    if probe is None or probe.get(key) is None:
        return None
    value = probe.get(key)
    if not isinstance(value, str) or not value:
        raise EvidenceCapsuleError(f"EVIDENCE_CAPSULE_QUERY_INVALID:{key}")
    return value


def _validate_capsule(capsule: Mapping[str, object]) -> None:
    if capsule.get("schema_version") != 1 or capsule.get("artifact_type") != "verdiwm-evidence-capsule":
        raise EvidenceCapsuleError("EVIDENCE_CAPSULE_CONTRACT_INVALID")
    if capsule.get("state") != "ready" or capsule.get("route") not in _ROUTES:
        raise EvidenceCapsuleError("EVIDENCE_CAPSULE_STATE_INVALID")
    selected = capsule.get("selected_evidence")
    if not isinstance(selected, list) or len(selected) > 20:
        raise EvidenceCapsuleError("EVIDENCE_CAPSULE_ROWS_INVALID")
