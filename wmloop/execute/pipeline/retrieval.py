"""Resumable pipeline state and evidence components: retrieval."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from wmloop.execute.pipeline.io import _load_json
from wmloop.execute.pipeline.types import AutonomousPipelineError
from wmloop.retrieve.irg_guided_discovery import (
    IRGDiscoveryError,
    build_irg_discovery_request,
)


def _runtime_retrieval_manifest(
    retrieval: Mapping[str, object],
) -> dict[str, object]:
    """Expose the bounded runtime projection in the top-level manifest.

    Compilation still receives the full in-memory retrieval context so exact
    candidate provenance is retained in the compiled batch. The resumable
    pipeline manifest carries the capsule instead of duplicating every row.
    """

    capsule = retrieval.get("capsule")
    if not isinstance(capsule, Mapping):
        return dict(retrieval)
    return {
        "state": retrieval.get("state"),
        "query": (
            dict(capsule.get("query", {}))
            if isinstance(capsule.get("query"), Mapping)
            else {}
        ),
        "match_count": capsule.get("match_count", 0),
        "route": capsule.get("route"),
        "capsule": dict(capsule),
        "capsule_path": retrieval.get("capsule_path"),
        "index_path": retrieval.get("index_path"),
        "irg_guided": retrieval.get("irg_guided"),
    }


def build_irg_context(
    model_irg_path: Path | None,
    protected_metrics: tuple[str, ...],
) -> tuple[dict[str, object] | None, tuple[str, ...], str | None]:
    irg_guided_context = None
    irg_failure_signatures = ()
    irg_model_family = None
    if model_irg_path is not None:
        try:
            irg_payload = _load_json(
                model_irg_path, "AUTONOMOUS_PIPELINE_MODEL_IRG_INVALID"
            )
            irg_request, irg_plan = build_irg_discovery_request(
                irg_payload,
                protected_metrics=protected_metrics,
            )
        except (
            IRGDiscoveryError,
            OSError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            raise AutonomousPipelineError(
                f"AUTONOMOUS_PIPELINE_IRG_DISCOVERY_INVALID:{exc}"
            ) from exc
        irg_guided_context = {
            "state": "planned",
            "model_irg_path": str(model_irg_path),
            "request": {
                "symptom_description": irg_request.symptom_description,
                "failure_signatures": list(irg_request.failure_signatures),
                "target_metrics": list(irg_request.target_metrics),
                "protected_metrics": list(irg_request.protected_metrics),
                "available_hooks": list(irg_request.available_hooks),
                "model_family": irg_request.model_family,
                "cross_domain_lenses": list(irg_request.cross_domain_lenses),
            },
            "plan": irg_plan,
            "authority": "shadow_only",
        }
        irg_failure_signatures = tuple(irg_request.failure_signatures)
        irg_model_family = irg_request.model_family
    return irg_guided_context, irg_failure_signatures, irg_model_family


def build_literature_query(
    *,
    explicit_query: str | None,
    retrieval_context: Mapping[str, object],
    probe_manifest: Mapping[str, object] | None,
    irg_failure_signatures: tuple[str, ...],
    irg_model_family: str | None,
) -> str | None:
    literature_query = explicit_query
    irg_guided = retrieval_context.get("irg_guided")
    if isinstance(irg_guided, Mapping):
        request_payload = irg_guided.get("request")
        if isinstance(request_payload, Mapping):
            lenses = request_payload.get("cross_domain_lenses", ())
            failures = request_payload.get("failure_signatures", ())
            literature_query = " ".join(
                str(value)
                for value in (
                    literature_query,
                    request_payload.get("model_family"),
                    *(lenses if isinstance(lenses, (list, tuple)) else ()),
                    *(failures if isinstance(failures, (list, tuple)) else ()),
                )
                if value
            )
    if (
        literature_query is None
        and probe_manifest is not None
        and retrieval_context.get("state") == "cold_start"
    ):
        probe_signatures = " ".join(
            str(value).replace("_", " ")
            for value in probe_manifest.get("failure_signatures", [])
            if isinstance(value, str)
        )
        irg_signatures = " ".join(
            value.replace("_", " ") for value in irg_failure_signatures
        )
        literature_query = " ".join(
            value
            for value in (
                str(
                    probe_manifest.get("model_family")
                    or irg_model_family
                    or "world model"
                ),
                "world model",
                probe_signatures,
                irg_signatures,
            )
            if value
        )
    return literature_query
