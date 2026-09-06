"""Model-agnostic IRG accumulation at the control-plane boundary.

Adapters and runners only emit a normalized observation batch.  This module
settles the batch into an append-only response chart and an accumulation
manifest; it never grants experiment or verifier authority.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from wmloop.archive.store import ArchiveStore, ContentAddressedStore
from wmloop.contracts import ContractValidationError, validate_document
from wmloop.geometry.irg import ResponseChart, estimate_response_chart
from wmloop.geometry.types import GeometryValidationError


class IRGAccumulationError(ValueError):
    """An observation batch cannot be admitted to the IRG ledger."""


_STATES = {"verified", "exploratory", "null", "harmful", "abstained", "disputed"}


def accumulate_irg_observations(
    *,
    batch: Mapping[str, object],
    output_root: Path,
    prior_manifest: Path | None = None,
    archive_db: Path | None = None,
    cas_root: Path | None = None,
) -> dict[str, object]:
    """Append one normalized probe/evidence observation to the IRG ledger.

    ``batch`` is deliberately adapter-neutral.  Required fields are model and
    protocol identity, outcome frame, baseline repeats, dose observations, and
    an evidence reference.  ``state`` is retained verbatim as evidence state;
    all generated artifacts remain diagnostic/ranking-only.
    """

    normalized = _normalize_batch(batch)
    destination = Path(output_root).expanduser().resolve()
    if destination.is_symlink() or (destination.exists() and not destination.is_dir()):
        raise IRGAccumulationError("IRG_ACCUMULATION_OUTPUT_INVALID")
    batch_hash = _digest(normalized)
    if destination.is_dir() and (destination / "manifest.json").is_file() and prior_manifest is None:
        existing = _load_manifest(destination / "manifest.json")
        if existing.get("batch_hash") == batch_hash:
            return {**existing, "manifest_path": str(destination / "manifest.json"), "response_chart_path": str(destination / "response-chart.json")}
        raise IRGAccumulationError("IRG_ACCUMULATION_OUTPUT_EXISTS")
    previous: dict[str, object] | None = None
    if prior_manifest is not None:
        previous = _load_manifest(Path(prior_manifest))
        if previous.get("model_family") != normalized["model_family"]:
            raise IRGAccumulationError("IRG_ACCUMULATION_MODEL_MISMATCH")
        if previous.get("goal_schema") != normalized["goal_schema"]:
            raise IRGAccumulationError("IRG_ACCUMULATION_GOAL_SCHEMA_MISMATCH")

    try:
        chart = estimate_response_chart(
            chart_id=f"irg-chart-{batch_hash[:24]}",
            goal_schema=str(normalized["goal_schema"]),
            outcome_names=normalized["outcome_names"],
            outcome_weights=normalized["outcome_weights"],
            baseline_repeats=normalized["baseline_repeats"],
            dose_observations=normalized["dose_observations"],
        )
    except (GeometryValidationError, ValueError, TypeError) as exc:
        raise IRGAccumulationError(f"IRG_ACCUMULATION_MEASUREMENT_INVALID:{exc}") from exc

    root = Path(cas_root).expanduser().resolve() if cas_root else destination.parent / "cas"
    cas = ContentAddressedStore(root)
    chart_bytes = _canonical(chart.to_dict())
    chart_ref = cas.put_bytes(chart_bytes, media_type="application/json").uri
    if archive_db is not None:
        ArchiveStore(Path(archive_db)).record_artifact_reference(chart_ref)

    sequence = int(previous.get("sequence", 0)) + 1 if previous else 1
    observation_ref = str(normalized["evidence_refs"][0])
    entry = {
        "sequence": sequence,
        "probe_id": normalized["probe_id"],
        "protocol_hash": normalized["protocol_hash"],
        "batch_hash": batch_hash,
        "state": normalized["state"],
        "evidence_refs": list(normalized["evidence_refs"]),
        "chart_ref": chart_ref,
        "observation_ref": observation_ref,
        "failure_signatures": list(normalized["failure_signatures"]),
        "anti_conditions": list(normalized["anti_conditions"]),
        "paired_identity": normalized["paired_identity"],
    }
    history = list(previous.get("observations", [])) if previous else []
    history.append(entry)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "verdiwm-irg-accumulation-manifest",
        "accumulation_id": "irg-accumulation-" + _digest({"model_family": normalized["model_family"], "sequence": sequence, "entry": entry})[:24],
        "sequence": sequence,
        "model_family": normalized["model_family"],
        "capability_class": normalized["capability_class"],
        "goal_schema": normalized["goal_schema"],
        "protocol_hash": normalized["protocol_hash"],
        "state": normalized["state"],
        "routing_authority": "diagnostic_ranking_only",
        "routing_state": (
            "abstain"
            if normalized["state"] in {"abstained", "disputed"}
            else "ready"
        ),
        "active_verdict_unchanged": True,
        "chart_ref": chart_ref,
        "observations": history,
        "failure_signatures": sorted({s for row in history for s in row["failure_signatures"]}),
        "claim_boundary": "IRG accumulation is evidence-conditioned and diagnostic-only; it cannot create a target-side verdict or execution authorization.",
    }
    if previous is not None:
        manifest["parent_accumulation_id"] = previous.get("accumulation_id")
    _write_json_atomic(destination / f"response-chart-{sequence:04d}.json", chart.to_dict())
    # Keep a stable pointer for consumers while retaining each chart revision.
    _write_json_atomic(destination / "response-chart.json", chart.to_dict())
    _write_json_atomic(destination / "manifest.json", manifest)
    return {**manifest, "manifest_path": str(destination / "manifest.json"), "response_chart_path": str(destination / "response-chart.json")}


def _normalize_batch(batch: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(batch, Mapping):
        raise IRGAccumulationError("IRG_ACCUMULATION_BATCH_INVALID")
    try:
        validate_document("irg_observation_batch", batch)
    except ContractValidationError as exc:
        raise IRGAccumulationError(
            f"IRG_ACCUMULATION_BATCH_INVALID:{exc}"
        ) from exc
    text_fields = ("model_family", "capability_class", "goal_schema", "probe_id", "protocol_hash")
    result: dict[str, object] = {}
    for field in text_fields:
        value = batch.get(field)
        if not isinstance(value, str) or not value.strip():
            raise IRGAccumulationError(f"IRG_ACCUMULATION_FIELD_INVALID:{field}")
        result[field] = value.strip()
    state = batch.get("state", "exploratory")
    if state not in _STATES:
        raise IRGAccumulationError("IRG_ACCUMULATION_STATE_INVALID")
    result["state"] = state
    outcomes = batch.get("outcome_names")
    weights = batch.get("outcome_weights")
    if not isinstance(outcomes, Sequence) or isinstance(outcomes, (str, bytes)) or not outcomes:
        raise IRGAccumulationError("IRG_ACCUMULATION_OUTCOME_FRAME_INVALID")
    if not isinstance(weights, Sequence) or len(weights) != len(outcomes):
        raise IRGAccumulationError("IRG_ACCUMULATION_OUTCOME_FRAME_INVALID")
    result["outcome_names"] = tuple(str(v) for v in outcomes)
    result["outcome_weights"] = tuple(float(v) for v in weights)
    baseline = batch.get("baseline_repeats")
    doses = batch.get("dose_observations")
    if not isinstance(baseline, Sequence) or not baseline or not isinstance(doses, Mapping) or not doses:
        raise IRGAccumulationError("IRG_ACCUMULATION_PAIRED_FRAME_INVALID")
    result["baseline_repeats"] = baseline
    normalized_doses: dict[str, dict[float, object]] = {}
    baseline_count = len(baseline)
    for intervention, observations in doses.items():
        if not isinstance(intervention, str) or not intervention or not isinstance(observations, Mapping):
            raise IRGAccumulationError("IRG_ACCUMULATION_PAIRED_FRAME_INVALID")
        if any(not isinstance(repeats, Sequence) or len(repeats) != baseline_count for repeats in observations.values()):
            raise IRGAccumulationError("IRG_ACCUMULATION_PAIRED_IDENTITY_MISMATCH")
        try:
            normalized_doses[intervention] = {
                float(dose): repeats for dose, repeats in observations.items()
            }
        except (TypeError, ValueError) as exc:
            raise IRGAccumulationError("IRG_ACCUMULATION_DOSE_INVALID") from exc
    result["dose_observations"] = normalized_doses
    refs = batch.get("evidence_refs")
    if not isinstance(refs, Sequence) or isinstance(refs, (str, bytes)) or not refs or any(not _content_ref(v) for v in refs):
        raise IRGAccumulationError("IRG_ACCUMULATION_EVIDENCE_REF_INVALID")
    result["evidence_refs"] = tuple(str(v) for v in refs)
    result["failure_signatures"] = tuple(sorted({str(v) for v in batch.get("failure_signatures", ()) if isinstance(v, str) and v}))
    result["anti_conditions"] = tuple(sorted({str(v) for v in batch.get("anti_conditions", ()) if isinstance(v, str) and v}))
    result["paired_identity"] = dict(batch.get("paired_identity", {"repeat_count": baseline_count})) if isinstance(batch.get("paired_identity", {"repeat_count": baseline_count}), Mapping) else {"repeat_count": baseline_count}
    return result


def _content_ref(value: object) -> bool:
    return isinstance(value, str) and (value.startswith("cas://") or value.startswith("urn:") or value.startswith("sha256:"))


def _load_manifest(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IRGAccumulationError("IRG_ACCUMULATION_PARENT_INVALID") from exc
    if not isinstance(payload, dict) or payload.get("artifact_type") != "verdiwm-irg-accumulation-manifest":
        raise IRGAccumulationError("IRG_ACCUMULATION_PARENT_INVALID")
    return payload


def _canonical(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value).rstrip(b"\n")).hexdigest()


def _write_json_atomic(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(_canonical(value))
    os.replace(temporary, path)


__all__ = ["IRGAccumulationError", "accumulate_irg_observations"]
