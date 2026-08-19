"""Compile evidence-bound method records into adapter-executable candidates."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from wmloop.contracts import ContractValidationError, validate_document


class MethodCandidateCompilerError(RuntimeError):
    """A method catalog or its evidence bindings failed closed."""


def compile_method_candidates(
    *,
    batch: dict[str, object],
    catalog_path: Path,
    diagnostic_probe: Mapping[str, object] | None,
    settlement_manifest: Path | None = None,
    literature_methods: Mapping[str, object] | None = None,
    materialization_values: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Append executable catalog entries and return a complete audit report."""

    source = Path(catalog_path).resolve(strict=True)
    catalog = _materialize(
        _load_object(source, "METHOD_CANDIDATE_CATALOG_INVALID"),
        materialization_values or {},
    )
    if not isinstance(catalog, dict):
        raise MethodCandidateCompilerError("METHOD_CANDIDATE_CATALOG_INVALID")
    try:
        validate_document("method_candidate_catalog", catalog)
    except ContractValidationError as exc:
        raise MethodCandidateCompilerError(
            f"METHOD_CANDIDATE_CATALOG_INVALID:{exc}"
        ) from exc

    observed = sorted(
        {
            str(value)
            for value in (diagnostic_probe or {}).get("failure_signatures", [])
            if isinstance(value, str) and value
        }
    )
    historical = _load_settlement_constraints(settlement_manifest)
    literature = _literature_constraints(literature_methods)
    catalog_primitives = {
        row.get("primitive_reference")
        for row in catalog.get("candidates", [])
        if isinstance(row, Mapping) and row.get("primitive_reference") is not None
    }
    existing = {
        str(row.get("candidate_id"))
        for row in batch.get("candidates", [])
        if isinstance(row, Mapping)
    }
    candidates = batch.get("candidates")
    if not isinstance(candidates, list):
        raise MethodCandidateCompilerError("METHOD_CANDIDATE_BATCH_INVALID")

    compiled: list[dict[str, object]] = []
    gaps: list[dict[str, object]] = [
        _catalog_gap(row) for row in catalog.get("capability_gaps", [])
    ]
    gaps.extend(_literature_gaps(literature, catalog_primitives=catalog_primitives))
    for raw in catalog.get("candidates", []):
        if not isinstance(raw, Mapping):
            raise MethodCandidateCompilerError("METHOD_CANDIDATE_CATALOG_INVALID")
        candidate_id = str(raw["candidate_id"])
        provenance = _provenance(raw)
        blockers = _candidate_blockers(
            raw,
            existing=existing,
            observed=observed,
            historical=historical,
        )
        if blockers:
            gaps.append({**provenance, "compilation_status": "blocked", "blockers": blockers})
            continue
        template = raw.get("candidate_template")
        if not isinstance(template, Mapping):
            raise MethodCandidateCompilerError("METHOD_CANDIDATE_TEMPLATE_INVALID")
        candidate = dict(template)
        if candidate.get("candidate_id") != candidate_id:
            raise MethodCandidateCompilerError("METHOD_CANDIDATE_ID_MISMATCH")
        candidate.setdefault("retrieval_prior", _evidence_prior(raw, literature=literature))
        candidates.append(candidate)
        existing.add(candidate_id)
        compiled.append(
            {
                **provenance,
                "compilation_status": "compiled",
                "retrieval_prior": candidate["retrieval_prior"],
                "matched_literature_candidate_ids": _matching_literature_ids(
                    raw, literature
                ),
            }
        )

    report = {
        "schema_version": 1,
        "artifact_type": "verdiwm-method-candidate-compilation",
        "state": "ready" if compiled else "blocked",
        "catalog_id": catalog["catalog_id"],
        "model_family": catalog["model_family"],
        "observed_failure_signatures": observed,
        "compiled_candidate_count": len(compiled),
        "compiled_candidates": compiled,
        "capability_gap_count": len(gaps),
        "capability_gaps": gaps,
        "historical_constraint_count": len(historical),
        "historical_constraints": historical,
        "literature_constraint_count": len(literature),
        "literature_constraints": literature,
        "claim_boundary": (
            "Compilation grants scheduling authority only to recipes whose files, "
            "diagnostic route, and adapter hooks are present. Historical exploratory "
            "settlements constrain ranking but never become promoted transfer claims."
        ),
    }
    try:
        validate_document("method_candidate_compilation", report)
    except ContractValidationError as exc:
        raise MethodCandidateCompilerError(
            f"METHOD_CANDIDATE_COMPILATION_INVALID:{exc}"
        ) from exc
    return report


def _candidate_blockers(
    candidate: Mapping[str, object],
    *,
    existing: set[str],
    observed: Sequence[str],
    historical: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    blockers: list[dict[str, object]] = []
    candidate_id = str(candidate["candidate_id"])
    if candidate_id in existing:
        blockers.append({"code": "CANDIDATE_ID_DUPLICATE"})
    declared = {
        str(value)
        for value in candidate.get("failure_signatures", [])
        if isinstance(value, str)
    }
    if observed and not declared.intersection(observed):
        blockers.append(
            {
                "code": "DIAGNOSTIC_ROUTE_MISMATCH",
                "declared_failure_signatures": sorted(declared),
                "observed_failure_signatures": list(observed),
            }
        )
    historical_ids = {
        str(value)
        for value in candidate.get("historical_candidate_ids", [])
        if isinstance(value, str)
    }
    repeated = [
        dict(row)
        for row in historical
        if str(row.get("candidate_id")) in historical_ids
        and row.get("promotion_authorized") is not True
    ]
    if repeated:
        blockers.append(
            {
                "code": "HISTORICAL_CANDIDATE_NOT_PROMOTED",
                "settlements": repeated,
            }
        )
    for requirement in candidate.get("required_files", []):
        if not isinstance(requirement, Mapping):
            raise MethodCandidateCompilerError("METHOD_CANDIDATE_REQUIREMENT_INVALID")
        path = Path(str(requirement.get("path", "")))
        if not path.is_file() or path.is_symlink():
            blockers.append(
                {
                    "code": "ADAPTER_CAPABILITY_FILE_MISSING",
                    "name": requirement.get("name"),
                    "path": str(path),
                }
            )
            continue
        expected = requirement.get("sha256")
        if expected is not None and _sha256(path.read_bytes()) != expected:
            blockers.append(
                {
                    "code": "ADAPTER_CAPABILITY_FILE_HASH_MISMATCH",
                    "name": requirement.get("name"),
                    "path": str(path),
                }
            )
    blockers.extend(_materialization_receipt_blockers(candidate))
    return blockers


def _materialization_receipt_blockers(
    candidate: Mapping[str, object],
) -> list[dict[str, object]]:
    raw_path = candidate.get("materialization_receipt_path")
    expected = candidate.get("materialization_receipt_sha256")
    if raw_path is None and expected is None:
        return []
    if not isinstance(raw_path, str) or not raw_path or not isinstance(expected, str):
        return [{"code": "MATERIALIZATION_RECEIPT_BINDING_MISSING"}]
    path = Path(raw_path)
    if path.is_symlink() or not path.is_file():
        return [{"code": "MATERIALIZATION_RECEIPT_MISSING", "path": raw_path}]
    payload = path.read_bytes()
    if _sha256(payload) != expected:
        return [{"code": "MATERIALIZATION_RECEIPT_HASH_MISMATCH", "path": raw_path}]
    try:
        receipt = json.loads(payload)
    except json.JSONDecodeError:
        return [{"code": "MATERIALIZATION_RECEIPT_INVALID", "path": raw_path}]
    side_effects = receipt.get("side_effects") if isinstance(receipt, Mapping) else None
    if (
        not isinstance(receipt, Mapping)
        or receipt.get("artifact_type") != "verdiwm-automatic-materialization-receipt"
        or receipt.get("state") != "ready_for_candidate_compilation"
        or receipt.get("candidate_id") != candidate.get("candidate_id")
        or not isinstance(side_effects, Mapping)
        or side_effects.get("candidate_compilation_authority") is not True
    ):
        return [{"code": "MATERIALIZATION_RECEIPT_NOT_ADMITTED", "path": raw_path}]
    return []


def _load_settlement_constraints(path: Path | None) -> list[dict[str, object]]:
    if path is None:
        return []
    source = Path(path).resolve(strict=True)
    manifest = _load_object(source, "METHOD_CANDIDATE_SETTLEMENT_MANIFEST_INVALID")
    if manifest.get("artifact_type") != "verdiwm-ctrl-world-settlement-import-manifest":
        raise MethodCandidateCompilerError(
            "METHOD_CANDIDATE_SETTLEMENT_MANIFEST_INVALID"
        )
    records_root = source.parent / "records"
    constraints: list[dict[str, object]] = []
    for record_path in sorted(records_root.glob("*.json")):
        if record_path.is_symlink() or not record_path.is_file():
            raise MethodCandidateCompilerError("METHOD_CANDIDATE_SETTLEMENT_RECORD_INVALID")
        record = _load_object(record_path, "METHOD_CANDIDATE_SETTLEMENT_RECORD_INVALID")
        if (
            record.get("artifact_type") != "verdiwm-imported-settlement-evidence"
            or record.get("settlement_state") != "settled"
        ):
            raise MethodCandidateCompilerError(
                "METHOD_CANDIDATE_SETTLEMENT_RECORD_INVALID"
            )
        constraints.append(
            {
                "trial_id": record.get("trial_id"),
                "experiment_id": record.get("experiment_id"),
                "candidate_id": record.get("candidate_id"),
                "verdict": record.get("verdict"),
                "evidence_scope": record.get("evidence_scope"),
                "promotion_authorized": record.get("promotion_authorized") is True,
                "source_settlement_sha256": record.get("source_settlement_sha256"),
                "claim_boundary": record.get("claim_boundary"),
            }
        )
    return constraints


def _literature_constraints(
    manifest: Mapping[str, object] | None,
) -> list[dict[str, object]]:
    if manifest is None:
        return []
    rows = manifest.get("records")
    if not isinstance(rows, list):
        raise MethodCandidateCompilerError("METHOD_CANDIDATE_LITERATURE_INVALID")
    constraints = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise MethodCandidateCompilerError("METHOD_CANDIDATE_LITERATURE_INVALID")
        constraints.append(
            {
                "candidate_id": row.get("candidate_id"),
                "primitive_reference": row.get("primitive_reference"),
                "required_hook": row.get("required_hook"),
                "estimated_gpu_hours": row.get("estimated_gpu_hours"),
                "execution_authority": row.get("execution_authority"),
                "state": row.get("state"),
            }
        )
    return constraints


def _matching_literature_ids(
    candidate: Mapping[str, object], literature: Sequence[Mapping[str, object]]
) -> list[str]:
    primitive = candidate.get("primitive_reference")
    return sorted(
        str(row["candidate_id"])
        for row in literature
        if primitive is not None
        and row.get("primitive_reference") == primitive
        and isinstance(row.get("candidate_id"), str)
    )


def _literature_gaps(
    literature: Sequence[Mapping[str, object]], *, catalog_primitives: set[object]
) -> list[dict[str, object]]:
    gaps = []
    for row in literature:
        authority = row.get("execution_authority")
        primitive = row.get("primitive_reference")
        if authority == "ranking_only" and primitive in catalog_primitives:
            continue
        code = (
            "PRIMITIVE_MATERIALIZATION_REQUIRED"
            if authority == "materialization_required"
            else "ADAPTER_RECIPE_MISSING"
        )
        gaps.append(
            {
                "candidate_id": row.get("candidate_id"),
                "source": "literature_method_staging",
                "primitive_reference": primitive,
                "required_hooks": [row.get("required_hook")],
                "estimated_gpu_hours": row.get("estimated_gpu_hours"),
                "compilation_status": "capability_gap",
                "blockers": [{"code": code}],
            }
        )
    return gaps


def _evidence_prior(
    candidate: Mapping[str, object], *, literature: Sequence[Mapping[str, object]]
) -> float:
    matches = _matching_literature_ids(candidate, literature)
    return min(0.9, 0.55 + 0.1 * len(matches))


def _provenance(candidate: Mapping[str, object]) -> dict[str, object]:
    return {
        "candidate_id": candidate["candidate_id"],
        "source": candidate["source"],
        "primitive_reference": candidate.get("primitive_reference"),
        "mechanism_hypothesis": candidate["mechanism_hypothesis"],
        "required_hooks": list(candidate["required_hooks"]),
        "estimated_gpu_hours": candidate["estimated_gpu_hours"],
        "applicability_conditions": list(candidate["applicability_conditions"]),
        "failure_boundaries": list(candidate["failure_boundaries"]),
        "failure_signatures": list(candidate["failure_signatures"]),
    }


def _catalog_gap(raw: object) -> dict[str, object]:
    if not isinstance(raw, Mapping):
        raise MethodCandidateCompilerError("METHOD_CANDIDATE_CATALOG_INVALID")
    return {
        "candidate_id": raw["candidate_id"],
        "source": raw["source"],
        "mechanism_hypothesis": raw["mechanism_hypothesis"],
        "required_hooks": list(raw["required_hooks"]),
        "estimated_gpu_hours": raw["estimated_gpu_hours"],
        "failure_signatures": list(raw["failure_signatures"]),
        "compilation_status": "capability_gap",
        "blockers": [{"code": raw["reason"]}],
    }


def _load_object(path: Path, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MethodCandidateCompilerError(code) from exc
    if not isinstance(value, dict):
        raise MethodCandidateCompilerError(code)
    return value


def _materialize(value: object, values: Mapping[str, str]) -> object:
    if isinstance(value, dict):
        return {str(key): _materialize(child, values) for key, child in value.items()}
    if isinstance(value, list):
        return [_materialize(child, values) for child in value]
    if not isinstance(value, str):
        return value
    result = value
    for placeholder, replacement in values.items():
        result = result.replace(placeholder, replacement)
    remaining = set(re.findall(r"\{[^{}]+\}", result))
    runtime = {
        "{scratch_dir}",
        "{workspace_root}",
        "{output_root}",
        "{gpu_index}",
        "{gpu_uuid}",
    }
    if remaining - runtime:
        raise MethodCandidateCompilerError(
            "METHOD_CANDIDATE_PLACEHOLDER_UNBOUND:"
            + ",".join(sorted(remaining - runtime))
        )
    return result


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()
