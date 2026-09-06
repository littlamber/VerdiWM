"""Compose IRG diagnosis, query planning, and mechanism discovery."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from wmloop.retrieve.irg_guided_discovery import (
    IRGDiscoveryError,
    build_irg_discovery_request,
    run_irg_guided_mechanism_discovery,
)
from wmloop.retrieve.mechanism_discovery import (
    MechanismDiscoveryError,
    build_multiview_queries,
)
from wmloop.retrieve.query_policy import QueryPolicyError


class DiscoveryCoordinatorError(ValueError):
    """An IRG-bound discovery stage cannot be planned or resumed."""


@dataclass(frozen=True)
class IRGDiscoveryContext:
    model_family: str
    failure_signatures: tuple[str, ...]
    literature_queries: tuple[str, ...]
    manifest: dict[str, object]


def prepare_irg_discovery(
    *,
    model_irg_path: Path,
    protected_metrics: Sequence[str],
    output_root: Path,
    control_root: Path,
    enable_external_discovery: bool,
    max_results: int,
    timeout_seconds: float,
) -> IRGDiscoveryContext:
    """Build one reusable IRG-driven discovery context and optional live atlas."""

    source = Path(model_irg_path).expanduser().resolve(strict=True)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DiscoveryCoordinatorError("IRG_DISCOVERY_INPUT_INVALID") from exc
    if not isinstance(payload, dict):
        raise DiscoveryCoordinatorError("IRG_DISCOVERY_INPUT_INVALID")
    try:
        request, plan = build_irg_discovery_request(
            payload,
            protected_metrics=protected_metrics,
        )
        queries = tuple(row["query"] for row in build_multiview_queries(request))
    except (IRGDiscoveryError, MechanismDiscoveryError, QueryPolicyError) as exc:
        raise DiscoveryCoordinatorError(f"IRG_DISCOVERY_PLAN_INVALID:{exc}") from exc
    manifest: dict[str, object] = {
        "state": "planned",
        "model_irg_path": str(source),
        "request": {
            "symptom_description": request.symptom_description,
            "failure_signatures": list(request.failure_signatures),
            "target_metrics": list(request.target_metrics),
            "protected_metrics": list(request.protected_metrics),
            "available_hooks": list(request.available_hooks),
            "model_family": request.model_family,
            "cross_domain_lenses": list(request.cross_domain_lenses),
        },
        "query_count": len(queries),
        "plan": plan,
        "authority": "shadow_only",
    }
    if enable_external_discovery:
        try:
            mechanism = run_irg_guided_mechanism_discovery(
                model_irg_path=source,
                protected_metrics=protected_metrics,
                output_root=output_root,
                repo_root=control_root,
                max_papers=min(12, max_results * 2),
                search_results_per_view=min(3, max_results),
                timeout_seconds=timeout_seconds,
            )
        except (
            IRGDiscoveryError,
            MechanismDiscoveryError,
            QueryPolicyError,
            OSError,
            ValueError,
        ) as exc:
            raise DiscoveryCoordinatorError(
                f"IRG_DISCOVERY_EXECUTION_INVALID:{exc}"
            ) from exc
        manifest["state"] = "retrieved"
        manifest["mechanism_discovery"] = mechanism
    return IRGDiscoveryContext(
        model_family=request.model_family,
        failure_signatures=tuple(request.failure_signatures),
        literature_queries=queries,
        manifest=manifest,
    )
