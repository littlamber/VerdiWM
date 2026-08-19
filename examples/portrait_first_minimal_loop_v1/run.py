#!/usr/bin/env python3
"""Run a deterministic, CPU-only portrait-first planning demonstration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wmloop.control.capability_gap_planner import build_goal_ir, compile_capability_gap_plan
from wmloop.control.intermediate_ir import build_model_capability_ir
from wmloop.control.model_portrait import (
    build_model_portrait,
    build_portrait_readiness_receipt,
)
from wmloop.geometry import build_probe_fingerprint_summary


def build_summary(*, root: Path = ROOT) -> dict[str, object]:
    capabilities = ("action_conditioned", "evaluation", "inference", "rollout")
    report = {
        "schema_version": 1,
        "artifact_type": "wmloop-model-onboarding-report",
        "repo_name": "public-portrait-demo",
        "source_revision": {"kind": "source_tree_sha256", "revision": "1" * 64},
        "capabilities": [
            {"capability": name, "state": "discovered", "evidence": ["public-demo"]}
            for name in capabilities
        ],
        "connector": {
            "entrypoints_by_kind": {name: [name + "-entrypoint"] for name in capabilities},
            "asset_bindings": [{"kind": "model_asset"}],
        },
        "evaluator_contract": {
            "state": "ready",
            "evaluator_id": "public-heldout-evaluator-v1",
            "contract_sha256": "2" * 64,
            "verifier": "public-frozen-verifier-v1",
        },
    }
    capability = build_model_capability_ir(
        report,
        model_family="public-demo",
        hooks=(),
        root=root,
    )
    fingerprint = build_probe_fingerprint_summary(
        model_capability_id=str(capability["capability_id"]),
        model_family="public-demo",
        probe_protocol_id="public-dose-response",
        probe_protocol_version="v1",
        diagnostic_role="diagnostic",
        context_class="public-demo-context",
        split="diagnostic",
        horizons=(1,),
        dose_values=(0.0, 1.0),
        replication_count=1,
        response_dimension=1,
        response_summary="deterministic public control-plane fixture",
        response_digest="sha256:" + "3" * 64,
        uncertainty_summary="one deterministic CPU fixture",
        evidence_refs=("sha256:" + "4" * 64,),
    )
    portrait = build_model_portrait(
        model_capability=capability,
        fingerprints=(fingerprint,),
        root=root,
    )
    goal_binding = "sha256:" + "5" * 64
    readiness = build_portrait_readiness_receipt(
        portrait=portrait,
        goal_binding=goal_binding,
        coverage_policy_id="public-portrait-coverage-v1",
        required_capabilities=("evaluation", "inference", "rollout"),
        required_probe_coverage=(
            {
                "probe_protocol_id": "public-dose-response",
                "probe_protocol_version": "v1",
                "diagnostic_role": "diagnostic",
                "context_class": "public-demo-context",
                "split": "diagnostic",
                "horizons": [1],
                "dose_values": [0.0, 1.0],
                "minimum_replication_count": 1,
            },
        ),
        evaluator_required=True,
        root=root,
    )
    goal = build_goal_ir(
        goal_id="public-portrait-demo-goal",
        goal_binding=goal_binding,
        model_family="public-demo",
        objective="Verify a model's declared inference capability before planning.",
        requirements=(
            {
                "capability": "inference",
                "kind": "observation",
                "dependencies": [],
                "required_model_capabilities": ["evaluation", "inference", "rollout"],
                "required_hooks": [],
                "required_interfaces": [
                    {"kind": "inference", "contract_id": "model-entrypoint:inference:v1"}
                ],
                "required_probe_keys": [],
                "required_data_regimes": [],
                "compatible_model_families": ["public-demo"],
                "manufacturable": False,
                "external_ports": [],
                "source": "public-portrait-demo",
            },
        ),
        root=root,
    )
    gap = compile_capability_gap_plan(
        goal_ir=goal,
        portrait=portrait,
        abi_registry_path=root / "configs/plugins/automatic_module_abis_v1.json",
        maximum_authority_level="L0",
        admitted_abi_ids=(),
        manufacturable_capabilities=(),
        available_data_regimes=(),
        kernel_capabilities=("frozen_evaluation",),
        root=root,
    )
    node = gap["requirement_graph"]["nodes"][0]
    return {
        "schema_version": 1,
        "artifact_type": "verdiwm-public-portrait-first-demo",
        "state": "ready",
        "capability_id": capability["capability_id"],
        "portrait_id": portrait["portrait_id"],
        "readiness_id": readiness["readiness_id"],
        "readiness_state": readiness["state"],
        "gap_plan_id": gap["gap_plan_receipt"]["plan_id"],
        "gap_plan_state": gap["gap_plan_receipt"]["state"],
        "requirement_classification": node["classification"],
        "authority": {
            "gpu_authority": False,
            "module_manufacturing_authority": False,
            "promotion_authority": False,
        },
        "claim_boundary": "This CPU fixture validates control-plane contracts only; it makes no model-quality claim.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = build_summary()
    encoded = json.dumps(payload, sort_keys=True, indent=2) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
