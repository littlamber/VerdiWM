"""Discovery stage implementation for the Ctrl-World workflow."""

from __future__ import annotations
import hashlib
from collections.abc import Mapping
from pathlib import Path
from experiments.ctrl_world_research_loop_v2.research_intake import run_research_intake_v2
from experiments.ctrl_world_autonomous_transfer_v1.local_method_intake import run_local_method_intake
from .common import AutonomousTransferWorkflowError, _canonical_bytes, _load, _paths, _write_json_idempotent


def run_discovery(
    config: Mapping[str, object], *, output_root: Path, failure_context: tuple[str, ...]
) -> dict[str, object]:
    paths = _paths(config)
    intake = _load(paths["intake_config"], "AUTONOMOUS_INTAKE_CONFIG_INVALID")
    queries = [str(value) for value in intake.get("queries", [])]
    for value in failure_context:
        for query in _failure_queries(value):
            if query not in queries:
                queries.append(query)
    intake["queries"] = queries[:12]
    intake["policy_digest"] = ""
    intake["policy_digest"] = hashlib.sha256(
        _canonical_bytes(
            {key: value for key, value in intake.items() if key != "policy_digest"}
        )
    ).hexdigest()
    cycle_config = output_root.with_name(output_root.name + "-intake-config.json")
    _write_json_idempotent(cycle_config, intake)
    external_error: str | None = None
    try:
        external = run_research_intake_v2(
            config_path=cycle_config,
            contract_path=paths["contract"],
            output_root=output_root,
            project_root=paths["project_root"],
            failure_context=failure_context,
            source_bundle_path=(
                Path(str(config["source_bundle_path"]))
                if config.get("source_bundle_path")
                else None
            ),
        )
    except Exception as exc:
        # A network/replay outage is an intake failure, not a reason to discard
        # the controller's local reviewed method inventory. Preserve the error
        # in the cycle manifest while keeping the source boundary explicit.
        external = {
            "state": "failed",
            "assessment_paths": [],
            "idea_paths": [],
            "work_order_paths": [],
        }
        external_error = f"{type(exc).__name__}:{str(exc)[:800]}"
    strict_network = _external_research_requires_network(config)
    if strict_network and external_error is None:
        reason = _network_readiness_error(
            external,
            minimum_successful_sources=_external_research_minimum_sources(config),
        )
        if reason is not None:
            external_error = reason
    if strict_network and external_error is not None:
        return {
            "state": "failed",
            "assessment_paths": [],
            "idea_paths": [],
            "work_order_paths": [],
            "external": external,
            "local": {
                "state": "not_run_network_required",
                "local_method_count": 0,
                "assessment_paths": [],
                "idea_paths": [],
                "work_order_paths": [],
            },
            "local_method_count": 0,
            "network_required": True,
            "external_error": external_error,
            "claim_boundary": (
                "External retrieval was required for this fresh deployment. No local reviewed "
                "profile was promoted into an executable discovery queue after the network gate failed."
            ),
        }
    local = run_local_method_intake(
        config=config,
        output_root=output_root / "local-methods",
        project_root=paths["project_root"],
        failure_context=failure_context,
    )
    merged = {
        "state": (
            "ready_for_materialization"
            if local.get("idea_paths") or external.get("idea_paths")
            else ("failed" if external_error else str(external.get("state") or "empty"))
        ),
        "assessment_paths": [
            *[str(value) for value in external.get("assessment_paths", [])],
            *[str(value) for value in local.get("assessment_paths", [])],
        ],
        "idea_paths": [
            *[str(value) for value in external.get("idea_paths", [])],
            *[str(value) for value in local.get("idea_paths", [])],
        ],
        "work_order_paths": [
            *[str(value) for value in external.get("work_order_paths", [])],
            *[str(value) for value in local.get("work_order_paths", [])],
        ],
        "external": external,
        "local": local,
        "local_method_count": int(local.get("local_method_count", 0)),
        "claim_boundary": (
            "External and local method sources share only the immutable intake contract. "
            "Local profiles are not external source claims."
        ),
    }
    if external_error is not None:
        merged["external_error"] = external_error
    return merged


def _failure_query(value: str) -> str:
    return _failure_queries(value)[0]


def _failure_queries(value: str) -> tuple[str, ...]:
    normalized = value.lower().replace(":", " ").replace("_", " ")
    queries: list[str] = []
    if "action conditioning" in normalized:
        queries.extend(
            [
                "vision language action model action conditioning robustness",
                "video generation action conditioned temporal consistency",
            ]
        )
    elif "trajectory fidelity" in normalized:
        queries.extend(
            [
                "video diffusion trajectory fidelity preservation long horizon",
                "world model rollout consistency motion dynamics",
            ]
        )
    elif "horizon drift" in normalized or "long horizon" in normalized:
        queries.extend(
            [
                "long horizon video world model rollout drift memory",
                "autoregressive video generation temporal consistency self forcing",
            ]
        )
    elif "materializer" in normalized or "capability" in normalized:
        queries.extend(
            [
                "modular multimodal world model adapter typed interface",
                "vision language model world model representation transfer",
            ]
        )
    else:
        focus = normalized[:80].strip() or "world model transfer failure"
        queries.append(f"world model {focus} optimization method")
    return tuple(dict.fromkeys(queries))


def _external_research_requires_network(config: Mapping[str, object]) -> bool:
    policy = config.get("external_research")
    return isinstance(policy, Mapping) and policy.get("network_required") is True


def _external_research_minimum_sources(config: Mapping[str, object]) -> int:
    policy = config.get("external_research")
    if not isinstance(policy, Mapping):
        return 1
    value = policy.get("minimum_successful_sources", 1)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise AutonomousTransferWorkflowError("AUTONOMOUS_EXTERNAL_RESEARCH_POLICY_INVALID")
    return value


def _network_readiness_error(
    manifest: Mapping[str, object], *, minimum_successful_sources: int
) -> str | None:
    if manifest.get("retrieval_mode") != "network":
        return "AUTONOMOUS_EXTERNAL_RESEARCH_NETWORK_MODE_REQUIRED"
    retrieval = manifest.get("retrieval")
    if not isinstance(retrieval, list):
        return "AUTONOMOUS_EXTERNAL_RESEARCH_RETRIEVAL_RECEIPT_MISSING"
    successful = {
        str(row.get("source"))
        for row in retrieval
        if isinstance(row, Mapping) and row.get("state") == "fetched"
    }
    if len(successful) < minimum_successful_sources:
        return (
            "AUTONOMOUS_EXTERNAL_RESEARCH_SOURCES_UNAVAILABLE:"
            f"{len(successful)}<{minimum_successful_sources}"
        )
    return None
