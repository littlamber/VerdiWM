"""Build a Ctrl-World autonomous-loop deployment from one automatic revision."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Mapping

from experiments.ctrl_world_autonomous_transfer_v1 import workflow


class CtrlWorldDeploymentError(ValueError):
    """The resource receipt cannot safely produce a controller deployment."""


def compose_fresh_deployment(
    *,
    template_path: Path,
    fresh_root: Path,
    revision: Mapping[str, object],
    gpu_lock_root: Path,
    project_root: Path | None = None,
    require_portrait_loop: bool = True,
) -> dict[str, object]:
    """Compile a new controller namespace with no historical evidence inputs.

    The existing ``compose_deployment`` API remains useful for versioned
    campaigns. This wrapper is the stronger boundary used by the fresh
    Ctrl-World loop: its state must live below ``fresh_root``, its evidence
    roots start empty, and only GPUs 0--5 may be admitted.
    """

    root = (project_root or Path(__file__).resolve().parents[2]).expanduser().resolve()
    namespace = Path(fresh_root).expanduser().resolve()
    state = namespace / "autonomous-controller-v2"
    if namespace == root or root in namespace.parents:
        raise CtrlWorldDeploymentError("CTRL_WORLD_FRESH_NAMESPACE_INSIDE_SOURCE")
    if state == root or root in state.parents:
        raise CtrlWorldDeploymentError("CTRL_WORLD_FRESH_STATE_INSIDE_SOURCE")
    template = _load_mapping(template_path, "CTRL_WORLD_FRESH_TEMPLATE_INVALID")
    required_stages = (
        "portrait_gate",
        "observation_planning",
        "observation_execution",
        "gap_planning",
        "portfolio_planning",
        "closed_loop",
    )
    if require_portrait_loop:
        missing = [name for name in required_stages if not isinstance(template.get(name), Mapping)]
        if missing:
            raise CtrlWorldDeploymentError(
                "CTRL_WORLD_FRESH_PORTRAIT_LOOP_REQUIRED:" + ",".join(missing)
            )
    existing = template.get("existing_evidence_roots", [])
    if not isinstance(existing, list) or existing:
        raise CtrlWorldDeploymentError("CTRL_WORLD_FRESH_HISTORICAL_EVIDENCE_FORBIDDEN")
    if template.get("source_bundle_path"):
        raise CtrlWorldDeploymentError("CTRL_WORLD_FRESH_SOURCE_BUNDLE_FORBIDDEN")
    gpus = revision.get("resource_allocation")
    if not isinstance(gpus, Mapping):
        raise CtrlWorldDeploymentError("CTRL_WORLD_FRESH_ALLOCATION_MISSING")
    role = _candidate_role(gpus)
    if any(index not in range(6) for index in role["gpu_indices"]):
        raise CtrlWorldDeploymentError("CTRL_WORLD_FRESH_GPU_6_7_RESERVED")
    config = json.loads(json.dumps(template))
    paths = config.get("paths")
    if not isinstance(paths, Mapping):
        raise CtrlWorldDeploymentError("CTRL_WORLD_FRESH_PATHS_INVALID")
    intake_path = str(config.get("local_method_discovery", {}).get("intake_config") if isinstance(config.get("local_method_discovery"), Mapping) else paths.get("intake_config") or "")
    config["local_method_discovery"] = {
        "enabled": True,
        "intake_config": intake_path,
        "materializer_profiles": (
            sorted(str(key) for key in config.get("materializers", {}))
            or ["hybrid_relevance_memory"]
        ),
        "max_methods": 8,
    }
    protected_metrics = ["horizon_drift_slope"]
    portfolio = config.get("portfolio_planning")
    if isinstance(portfolio, Mapping) and isinstance(portfolio.get("protected_metrics"), list):
        protected_metrics = sorted({str(value) for value in portfolio["protected_metrics"]}) or protected_metrics
    evaluator_path = Path(str(paths.get("base_evaluator") or paths.get("evaluator") or ""))
    evaluator_binding = "sha256:" + _sha256_file(evaluator_path) if evaluator_path.is_file() else "urn:verdiwm:fresh-evaluator"
    config["metric_evolution"] = {
        "mode": "shadow_only",
        "protected_metric_ids": protected_metrics,
        "candidate_metrics": [
            {
                "schema_version": 1,
                "artifact_type": "wmloop-metric-adequacy",
                "metric_id": "horizon_drift_slope_variance",
                "objective_alignment_claim": "Separates horizon drift slope variability from its protected mean trend.",
                "target_construct": "horizon drift slope variability",
                "direction": "minimize",
                "unit": "normalized slope variance",
                "aggregation": "mean",
                "horizons": [16, 32, 48],
                "splits": ["diagnostic", "heldout"],
                "role": "diagnostic",
                "required_guard_metrics": protected_metrics,
                "falsification_condition": "Reject if the candidate adds no information beyond the protected slope metric.",
                "anti_gaming_concerns": ["short-horizon-only optimization", "variance suppression by collapse"],
                "incremental_information_claim": "Detects instability changes hidden by an unchanged average slope.",
                "evaluator_binding": evaluator_binding,
            }
        ],
    }
    config["external_research"] = {
        "network_required": True,
        "minimum_successful_sources": 2,
    }
    config["existing_evidence_roots"] = []
    return compose_deployment(
        template_path=_write_temporary_template(config, state=state),
        state_root=state,
        revision=revision,
        gpu_lock_root=gpu_lock_root,
        project_root=root,
    )


def _write_temporary_template(config: Mapping[str, object], *, state: Path) -> Path:
    path = state.parent / "deployment-template.json"
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.write_text(json.dumps(config, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return path


def compose_deployment(
    *,
    template_path: Path,
    state_root: Path,
    revision: Mapping[str, object],
    gpu_lock_root: Path,
    project_root: Path | None = None,
) -> dict[str, object]:
    """Persist a revision-bound controller config without exposing config versions.

    The generated config is local execution state.  The user request and its
    immutable automatic revision remain the durable public control-plane API.
    """

    root = (project_root or Path(__file__).resolve().parents[2]).expanduser().resolve()
    state = Path(state_root).expanduser().resolve()
    if state == root or root in state.parents:
        raise CtrlWorldDeploymentError("CTRL_WORLD_DEPLOYMENT_STATE_INSIDE_SOURCE")
    template = _load_mapping(template_path, "CTRL_WORLD_DEPLOYMENT_TEMPLATE_INVALID")
    allocation = revision.get("resource_allocation")
    if not isinstance(allocation, Mapping):
        raise CtrlWorldDeploymentError("CTRL_WORLD_DEPLOYMENT_ALLOCATION_MISSING")
    role = _candidate_role(allocation)
    gpus = role["gpu_indices"]
    parallelism = role["max_parallel_jobs"]
    revision_id = revision.get("revision_id")
    if not isinstance(revision_id, str) or not revision_id:
        raise CtrlWorldDeploymentError("CTRL_WORLD_DEPLOYMENT_REVISION_INVALID")
    config = json.loads(json.dumps(template))
    config["loop_id"] = f"ctrl-world-auto-{revision_id}"
    config["gpu_indices"] = gpus
    config["max_parallel_gpu_jobs"] = parallelism
    config["gpu_lock_root"] = str(Path(gpu_lock_root).expanduser().resolve())
    experiment_scale_plan = (
        root / "experiments/ctrl_world_autonomous_transfer_v1/scale_plan.json"
    )
    conversion_scale_plan = (
        root / "experiments/droid_ctrl_world_conversion_v1/scale_plan.json"
    )
    config["resource_portfolio"] = {
        "policy_id": "ctrl-world-eight-gpu-portfolio-v1",
        "resource_allocation": dict(allocation),
        "experiment_scale_plan": str(experiment_scale_plan),
        "experiment_scale_plan_sha256": _sha256_file(experiment_scale_plan),
        "conversion_scale_plan": str(conversion_scale_plan),
        "conversion_scale_plan_sha256": _sha256_file(conversion_scale_plan),
        "default_confirm_gpu_count": 1,
    }
    # Graph rebuilds reject outputs under the controller state root so SQLite
    # recovery cannot be confused with a graph projection. Keep them siblings.
    graph_prefix = state.parent / f"{state.name}-graphs"
    config["knowledge_graph_root"] = str(graph_prefix / "local-audit")
    config["portable_knowledge_root"] = str(graph_prefix / "portable")
    config["portable_knowledge_records_root"] = str(graph_prefix / "portable-records")
    materializers = config.get("materializers")
    if not isinstance(materializers, dict):
        raise CtrlWorldDeploymentError("CTRL_WORLD_DEPLOYMENT_MATERIALIZERS_INVALID")
    materializers.setdefault(
        "hybrid_relevance_memory",
        {
            "plan_template": str(
                root / "configs/experiments/ctrl_world_hybrid_relevance_memory_materialization_v1.json"
            ),
            "evaluator": str(
                root / "experiments/ctrl_world_hybrid_memory_transfer_v1/evaluate.py"
            ),
        },
    )
    config["config_digest"] = workflow.config_digest(config)
    config_path = state / "deployment" / "controller-config.json"
    _write_json_atomic(config_path, config)
    try:
        workflow.load_and_validate_config(config_path, project_root=root)
    except workflow.AutonomousTransferWorkflowError as exc:
        raise CtrlWorldDeploymentError(str(exc)) from exc
    deployment = {
        "schema_version": 1,
        "artifact_type": "verdiwm-ctrl-world-autonomous-deployment",
        "state": "ready",
        "revision_id": revision_id,
        "revision_digest": revision.get("revision_digest"),
        "controller_config": str(config_path),
        "controller_config_sha256": _sha256_file(config_path),
        "state_root": str(state),
        "resource_allocation": allocation,
        "resource_portfolio_policy": config["resource_portfolio"],
        "candidate_portfolio": {
            "slot_count": parallelism,
            "slots": [
                {
                    "slot": index,
                    "state": "awaiting_source_grounded_materialization",
                    "policy": "independent_one_gpu_candidate_only",
                }
                for index in range(parallelism)
            ],
            "admission_policy": (
                "The controller fills slots only after source grounding, target capability "
                "checks, materialization, and frozen screen/confirm progression."
            ),
        },
        "resume_policy": "restart_the_same_state_root_to_recover_unfinished_work",
        "claim_boundary": (
            "This deployment maps an automatic campaign revision to local controller state. "
            "It neither creates a scientific result nor promotes a candidate."
        ),
    }
    _write_json_atomic(state / "deployment" / "deployment-receipt.json", deployment)
    return deployment


def _candidate_role(allocation: Mapping[str, object]) -> dict[str, object]:
    if allocation.get("state") != "ready":
        raise CtrlWorldDeploymentError("CTRL_WORLD_DEPLOYMENT_GPU_INVENTORY_UNRESOLVED")
    roles = allocation.get("roles")
    if not isinstance(roles, list):
        raise CtrlWorldDeploymentError("CTRL_WORLD_DEPLOYMENT_ROLES_INVALID")
    matches = [
        dict(row)
        for row in roles
        if isinstance(row, Mapping) and row.get("role") == "autonomous_candidate_evaluation"
    ]
    if len(matches) != 1:
        raise CtrlWorldDeploymentError("CTRL_WORLD_DEPLOYMENT_CANDIDATE_ROLE_INVALID")
    role = matches[0]
    gpus = role.get("gpu_indices")
    parallelism = role.get("max_parallel_jobs")
    if (
        not isinstance(gpus, list)
        or not gpus
        or any(not isinstance(index, int) or index < 0 for index in gpus)
        or len(set(gpus)) != len(gpus)
        or not isinstance(parallelism, int)
        or parallelism < 1
        or parallelism > len(gpus)
    ):
        raise CtrlWorldDeploymentError("CTRL_WORLD_DEPLOYMENT_CANDIDATE_ROLE_INVALID")
    return {"gpu_indices": list(gpus), "max_parallel_jobs": parallelism}


def _load_mapping(path: Path, code: str) -> dict[str, object]:
    source = Path(path).expanduser().resolve()
    if not source.is_file() or source.is_symlink():
        raise CtrlWorldDeploymentError(code)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CtrlWorldDeploymentError(code) from exc
    if not isinstance(payload, dict):
        raise CtrlWorldDeploymentError(code)
    return payload


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()
