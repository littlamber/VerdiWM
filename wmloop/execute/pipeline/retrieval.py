"""Resumable pipeline state and evidence components: retrieval."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from typing import Mapping

from wmloop.execute.agibot_psnr_contract import build_agibot_psnr_execution_status
from wmloop.execute.pipeline.io import _load_json
from wmloop.execute.pipeline.types import AutonomousPipelineError
from wmloop.retrieve.irg_guided_discovery import (
    IRGDiscoveryError,
    build_irg_discovery_request,
)
from wmloop.retrieve.query_planning import build_literature_query_plan


def build_agibot_psnr_memory_context_for_pipeline(
    *,
    archive_db: Path,
    cas_root: Path,
    limit: int = 5,
) -> dict[str, object]:
    """Read settled Agibot PSNR memory without importing the RGB evaluator cold.

    Most users run the control plane before any Agibot RGB evidence exists.  A
    lightweight Archive probe keeps that path independent from optional image
    decoding dependencies.  Once an Agibot settlement is present, the memory
    reader replays its CAS evidence and surfaces dependency or integrity
    failures as a pipeline error instead of silently treating the round as a
    cold start.
    """

    archive = Path(archive_db)
    cas = Path(cas_root)
    if not archive.is_file() or not (cas / "cas").is_dir():
        return _empty_agibot_psnr_memory("AGIBOT_PSNR_MEMORY_NOT_INITIALIZED")
    try:
        with sqlite3.connect(archive.as_uri() + "?mode=ro", uri=True) as connection:
            row = connection.execute(
                # ArchiveStore keeps settlement state in the canonical
                # ``settlement_json`` projection; ``trials`` has no separate
                # settlement_state column.  The memory reader performs the
                # full binding/CAS replay once an Agibot row is present.
                "SELECT 1 FROM trials WHERE library_version = ? LIMIT 1",
                ("agibot-psnr-settlement-v1",),
            ).fetchone()
    except sqlite3.Error as exc:
        raise AutonomousPipelineError(
            "AUTONOMOUS_PIPELINE_AGIBOT_PSNR_ARCHIVE_INVALID"
        ) from exc
    if row is None:
        return _empty_agibot_psnr_memory("AGIBOT_PSNR_MEMORY_NO_SETTLEMENT")
    try:
        from wmloop.retrieve.agibot_psnr_memory import (
            AgibotPSNRMemoryError,
            build_agibot_psnr_memory_context,
        )

        return build_agibot_psnr_memory_context(
            archive_db=archive,
            cas_root=cas,
            limit=limit,
            replay_measurement=True,
        )
    except ModuleNotFoundError as exc:
        raise AutonomousPipelineError(
            "AUTONOMOUS_PIPELINE_AGIBOT_PSNR_MEMORY_DEPENDENCY_UNAVAILABLE"
        ) from exc
    except AgibotPSNRMemoryError as exc:
        raise AutonomousPipelineError(
            f"AUTONOMOUS_PIPELINE_AGIBOT_PSNR_MEMORY_INVALID:{exc}"
        ) from exc


def _empty_agibot_psnr_memory(reason: str) -> dict[str, object]:
    return {
        "state": "cold_start",
        "query": {
            "dataset_sha256": None,
            "support_sha256": None,
            "evaluator_digest": None,
        },
        "match_count": 0,
        "matches": [],
        "reason": reason,
        "claim_boundary": (
            "No settled Agibot PSNR evidence is available. A cold start does not "
            "establish a model effect."
        ),
    }


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
        "literature_query_plan": retrieval.get("literature_query_plan"),
        "literature_query_plan_path": retrieval.get("literature_query_plan_path"),
        "literature_status": retrieval.get("literature_status"),
        "agibot_psnr_memory": retrieval.get("agibot_psnr_memory"),
        "agibot_psnr_execution": retrieval.get("agibot_psnr_execution"),
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
    return build_literature_query_plan(
        explicit_query=explicit_query,
        retrieval_context=retrieval_context,
        probe_manifest=probe_manifest,
        irg_failure_signatures=irg_failure_signatures,
        irg_model_family=irg_model_family,
    )["query"]
