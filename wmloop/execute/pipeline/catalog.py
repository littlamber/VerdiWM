"""Resumable pipeline state and evidence components: catalog."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Mapping

from wmloop.contracts import ContractValidationError, validate_document
from wmloop.execute.pipeline.io import _load_json, _sha256, _write_json_atomic
from wmloop.execute.pipeline.types import AutonomousPipelineError


def _candidate_asset_parameters(catalog_path: Path | None) -> tuple[str, ...]:
    if catalog_path is None:
        return ()
    catalog = _load_json(
        catalog_path,
        "AUTONOMOUS_PIPELINE_CANDIDATE_CATALOG_INVALID",
    )
    try:
        validate_document("method_candidate_catalog", catalog)
    except ContractValidationError as exc:
        raise AutonomousPipelineError(
            "AUTONOMOUS_PIPELINE_CANDIDATE_CATALOG_INVALID"
        ) from exc

    parameters: set[str] = set()

    def visit(value: object) -> None:
        if isinstance(value, Mapping):
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
        elif isinstance(value, str):
            parameters.update(
                placeholder[len("{asset:") : -1]
                for placeholder in re.findall(r"\{asset:--[A-Za-z0-9_-]+\}", value)
            )

    visit(catalog)
    return tuple(sorted(parameters))


def _merge_method_catalogs(first: Path, second: Path, destination: Path) -> Path:
    """Append generated literature candidates without rewriting user input."""
    left = _load_json(first, "AUTONOMOUS_PIPELINE_CANDIDATE_CATALOG_INVALID")
    right = _load_json(second, "AUTONOMOUS_PIPELINE_CANDIDATE_CATALOG_INVALID")
    candidates = [
        *[row for row in left.get("candidates", []) if isinstance(row, Mapping)],
        *[row for row in right.get("candidates", []) if isinstance(row, Mapping)],
    ]
    gaps = [
        *[row for row in left.get("capability_gaps", []) if isinstance(row, Mapping)],
        *[row for row in right.get("capability_gaps", []) if isinstance(row, Mapping)],
    ]
    merged = {
        "schema_version": 1,
        "artifact_type": "verdiwm-method-candidate-catalog",
        "catalog_id": f"merged-{_sha256(first.read_bytes() + second.read_bytes())[:24]}",
        "model_family": left.get("model_family")
        or right.get("model_family")
        or "world_model",
        "candidates": candidates,
        "capability_gaps": gaps,
        "claim_boundary": "User-bound candidates and automatically materialized literature surrogates retain independent provenance and receipts.",
    }
    try:
        validate_document("method_candidate_catalog", merged)
    except ContractValidationError as exc:
        raise AutonomousPipelineError(
            "AUTONOMOUS_PIPELINE_CANDIDATE_CATALOG_INVALID"
        ) from exc
    _write_json_atomic(destination, merged)
    return destination.resolve()
