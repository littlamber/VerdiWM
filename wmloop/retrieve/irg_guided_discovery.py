"""Turn a model-conditioned IRG into bounded cross-domain research queries.

An IRG is a local response chart, not a magic global capability oracle.  This
module therefore emits *bottleneck hypotheses* with explicit evidence and a
ceiling boundary, then translates them into a :class:`DiscoveryRequest` for
the existing evidence-bound mechanism discovery pipeline.  Retrieved papers
remain untrusted data until typed local validation and experiment admission.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from wmloop.geometry.model_irg import ModelIRGError, validate_model_irg
from wmloop.retrieve.mechanism_discovery import (
    DiscoveryRequest,
    run_mechanism_discovery,
)
from wmloop.retrieve.query_policy import (
    DiscoveryQueryPolicy,
    QueryPolicyError,
    load_discovery_query_policy,
)


class IRGDiscoveryError(ValueError):
    """An IRG-guided discovery request is malformed or under-specified."""


def derive_irg_bottlenecks(
    irg: Mapping[str, object],
    *,
    top_k: int = 8,
    query_policy: DiscoveryQueryPolicy | None = None,
) -> tuple[dict[str, object], ...]:
    """Extract ranked, evidence-bound local bottleneck hypotheses from an IRG."""

    try:
        validate_model_irg(irg)
    except ModelIRGError as exc:
        raise IRGDiscoveryError(f"IRG_DISCOVERY_IRG_INVALID:{exc}") from exc
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 100:
        raise IRGDiscoveryError("IRG_DISCOVERY_TOP_K_INVALID")
    policy = query_policy or load_discovery_query_policy()
    vector = [float(value) for value in irg["response_vector"]]
    covariance = irg["response_covariance"]
    axes = {str(row["axis"]): row for row in irg["diagnostic_axes"]}
    rows: list[dict[str, object]] = []
    for index, name in enumerate(irg["coordinate_names"]):
        axis = axes.get(str(name))
        if axis is None:
            raise IRGDiscoveryError("IRG_DISCOVERY_AXIS_MISSING")
        response = vector[index]
        variance = float(covariance[index][index])
        uncertainty = math.sqrt(max(variance, 0.0))
        supported = str(axis["support_state"]) == "supported"
        signal_to_noise = abs(response) / (1.0 + uncertainty)
        tags = policy.domain_tags(
            (
                axis.get("probe_id", ""),
                axis.get("outcome", ""),
                axis.get("diagnosis", ""),
            )
        )
        rows.append(
            {
                "axis": str(name),
                "probe_id": str(axis["probe_id"]),
                "outcome": str(axis["outcome"]),
                "response": response,
                "uncertainty": uncertainty,
                "signal_to_noise": signal_to_noise,
                "severity": signal_to_noise if supported else 0.0,
                "support_state": "supported" if supported else "unsupported",
                "domain_tags": list(tags),
                "diagnosis": str(axis["diagnosis"]),
                "evidence_refs": list(axis["evidence_refs"]),
                "limitation_type": (
                    "local_sensitivity_bottleneck" if supported else "evidence_gap"
                ),
                "ceiling_boundary": (
                    "A local IRG response cannot establish a global capability ceiling; "
                    "confirm with a multi-dose or horizon sweep."
                ),
            }
        )
    rows.sort(key=lambda row: (-float(row["severity"]), str(row["axis"])))
    return tuple(rows[:top_k])


def build_irg_discovery_request(
    irg: Mapping[str, object],
    *,
    protected_metrics: Sequence[str],
    top_k: int = 8,
    cross_domain_lenses: Sequence[str] = (),
    query_policy_path: Path | None = None,
) -> tuple[DiscoveryRequest, dict[str, object]]:
    """Compile IRG bottlenecks into a cross-domain mechanism-discovery request."""

    try:
        policy = load_discovery_query_policy(query_policy_path)
    except QueryPolicyError as exc:
        raise IRGDiscoveryError(str(exc)) from exc
    bottlenecks = derive_irg_bottlenecks(
        irg, top_k=top_k, query_policy=policy
    )
    protected = tuple(str(value).strip() for value in protected_metrics if str(value).strip())
    if not protected:
        raise IRGDiscoveryError("IRG_DISCOVERY_PROTECTED_METRICS_REQUIRED")
    portrait = irg["portrait_binding"]
    asset = irg["asset_binding"]
    failure_signatures = tuple(
        "irg_bottleneck:" + _slug(str(row["axis"]))
        for row in bottlenecks
        if row["limitation_type"] == "local_sensitivity_bottleneck"
    )
    if not failure_signatures:
        failure_signatures = ("irg_bottleneck:evidence_gap",)
    domains = {
        tag
        for row in bottlenecks
        for tag in row["domain_tags"]
    }
    lenses = list(policy.lenses_for(sorted(domains)))
    lenses.extend(str(value).strip() for value in cross_domain_lenses if str(value).strip())
    lenses = list(dict.fromkeys(lenses))
    symptom = _symptom_description(bottlenecks)
    request = DiscoveryRequest(
        symptom_description=symptom,
        failure_signatures=failure_signatures,
        target_metrics=tuple(
            dict.fromkeys(str(row["outcome"]) for row in bottlenecks)
        ) or (str(asset["goal_schema"]),),
        protected_metrics=protected,
        available_hooks=_available_hooks(irg),
        model_family=str(portrait["model_family"]),
        cross_domain_lenses=tuple(lenses),
    )
    plan = {
        "schema_version": 1,
        "artifact_type": "verdiwm-irg-guided-discovery-request",
        "model_irg_id": str(irg["irg_id"]),
        "portrait_id": str(portrait["portrait_id"]),
        "asset_id": str(asset["asset_id"]),
        "bottlenecks": list(bottlenecks),
        "failure_signatures": list(failure_signatures),
        "cross_domain_lenses": lenses,
        "query_policy": {
            "policy_id": policy.policy_id,
            "path": str(policy.path),
            "sha256": policy.sha256,
        },
        "request": _request_dict(request),
        "authority": "shadow_only",
        "claim_boundary": (
            "IRG-guided retrieval proposes cross-domain mechanisms from local response "
            "bottleneck hypotheses. It does not establish a global model ceiling, novelty, "
            "or model improvement; every candidate requires target-side typed validation."
        ),
    }
    return request, plan


def run_irg_guided_mechanism_discovery(
    *,
    model_irg_path: Path,
    protected_metrics: Sequence[str],
    output_root: Path,
    repo_root: Path,
    top_k: int = 8,
    cross_domain_lenses: Sequence[str] = (),
    query_policy_path: Path | None = None,
    **kwargs: Any,
) -> dict[str, object]:
    """Run the existing bounded discovery pipeline from an IRG artifact."""

    path = Path(model_irg_path).expanduser().resolve(strict=True)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IRGDiscoveryError("IRG_DISCOVERY_INPUT_INVALID") from exc
    if not isinstance(payload, Mapping):
        raise IRGDiscoveryError("IRG_DISCOVERY_INPUT_INVALID")
    request, plan = build_irg_discovery_request(
        payload,
        protected_metrics=protected_metrics,
        top_k=top_k,
        cross_domain_lenses=cross_domain_lenses,
        query_policy_path=query_policy_path,
    )
    destination = Path(output_root).expanduser().resolve()
    request_path = destination / "irg-guided-request.json"
    manifest_path = destination / "mechanism-discovery" / "manifest.json"
    if destination.exists():
        if destination.is_symlink() or not destination.is_dir():
            raise IRGDiscoveryError("IRG_DISCOVERY_OUTPUT_INVALID")
        try:
            existing = json.loads(request_path.read_text(encoding="utf-8"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise IRGDiscoveryError("IRG_DISCOVERY_RESUME_INVALID") from exc
        if existing != plan or not isinstance(manifest, dict):
            raise IRGDiscoveryError("IRG_DISCOVERY_RESUME_MISMATCH")
        return manifest
    destination.mkdir(mode=0o700, parents=True)
    request_path.write_text(
        json.dumps(plan, sort_keys=True, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return run_mechanism_discovery(
        request=request,
        seed_records=(),
        output_root=destination / "mechanism-discovery",
        repo_root=Path(repo_root),
        query_policy_path=query_policy_path,
        **kwargs,
    )


def _symptom_description(bottlenecks: Sequence[Mapping[str, object]]) -> str:
    parts = []
    for row in bottlenecks[:5]:
        parts.append(
            f"{row['axis']} ({row['outcome']}) response={float(row['response']):.6g}, "
            f"uncertainty={float(row['uncertainty']):.6g}, domains={','.join(row['domain_tags'])}"
        )
    return (
        "IRG-guided local sensitivity hotspots suggest model-specific repair bottlenecks; "
        "search for mechanisms that address these axes without changing protected metrics. "
        + "; ".join(parts)
    )


def _available_hooks(irg: Mapping[str, object]) -> tuple[str, ...]:
    hooks = irg.get("available_hooks")
    if isinstance(hooks, list) and all(isinstance(value, str) and value for value in hooks):
        return tuple(hooks) or ("portrait_bound_hooks_required",)
    return ("portrait_bound_hooks_required",)


def _request_dict(request: DiscoveryRequest) -> dict[str, object]:
    return {
        "symptom_description": request.symptom_description,
        "failure_signatures": list(request.failure_signatures),
        "target_metrics": list(request.target_metrics),
        "protected_metrics": list(request.protected_metrics),
        "available_hooks": list(request.available_hooks),
        "model_family": request.model_family,
        "seed_arxiv_ids": list(request.seed_arxiv_ids),
        "cross_domain_lenses": list(request.cross_domain_lenses),
    }


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "axis"
