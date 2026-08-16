#!/usr/bin/env python3
"""Execute the frozen Ctrl-World anchor-repair confirmation recipe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from aggregate_ctrl_world_local_fingerprint import aggregate
from evaluate_ctrl_world_anchor_repair_confirm import evaluate
from freeze_ctrl_world_directional_selector import freeze_selector
from run_ctrl_world_anchor_repair_screen import run as run_anchor_screen


class AnchorRepairCandidateError(RuntimeError):
    """The evidence-bound confirmation recipe could not be completed."""


def run(args: argparse.Namespace) -> dict[str, object]:
    output = Path(args.output_root).resolve()
    if output.exists() or output.is_symlink():
        raise AnchorRepairCandidateError("ANCHOR_REPAIR_CANDIDATE_OUTPUT_EXISTS")
    output.mkdir(mode=0o700, parents=True)

    atlas_root = output / "route-atlas"
    aggregate(
        campaign_path=Path(args.route_campaign),
        result_paths=(Path(args.route_result),),
        output_root=atlas_root,
    )
    route_selector = output / "route-selector" / "selector.json"
    freeze_selector(
        atlas_path=atlas_root / "context-local-fingerprint-atlas.json",
        candidate_radius=float(args.candidate_radius),
        confidence_z=float(args.confidence_z),
        minimum_gain_lcb=float(args.minimum_gain_lcb),
        output_path=route_selector,
    )
    route = json.loads(route_selector.read_text(encoding="utf-8"))
    selected_contexts = []
    for row in route.get("contexts", []):
        selected = row.get("selected_candidate")
        context = row.get("context")
        if (
            row.get("selector_action") == "execute"
            and isinstance(selected, dict)
            and selected.get("probe_id") == "first_frame_anchor_blend"
            and float(selected.get("dose", 0.0)) < 0.0
            and isinstance(context, dict)
            and isinstance(context.get("context_id"), str)
        ):
            selected_contexts.append(str(context["context_id"]))
    if not selected_contexts:
        raise AnchorRepairCandidateError("ANCHOR_REPAIR_ROUTE_SELECTOR_ABSTAINS")

    selector = json.loads(Path(args.development_selector).read_text(encoding="utf-8"))
    selected = selector.get("selected_candidate")
    if (
        selector.get("artifact_type")
        != "verdiwm-ctrl-world-frozen-anchor-repair-selector"
        or selector.get("selector_action") != "execute"
        or not isinstance(selected, dict)
        or not isinstance(selected.get("primitive_id"), str)
        or not isinstance(selected.get("strength"), (int, float))
    ):
        raise AnchorRepairCandidateError("ANCHOR_REPAIR_DEVELOPMENT_SELECTOR_INVALID")

    anchor_root = output / "anchor-result"
    screen_args = argparse.Namespace(
        campaign_id=str(args.campaign_id),
        primitive_id=str(selected["primitive_id"]),
        ctrl_world_root=Path(args.ctrl_world_root),
        dataset_root=Path(args.dataset_root),
        data_stat=Path(args.data_stat),
        svd_model_path=Path(args.svd_model_path),
        clip_model_path=Path(args.clip_model_path),
        ckpt_path=Path(args.ckpt_path),
        contexts_json=Path(args.contexts_json),
        strengths=[0.0, float(selected["strength"])],
        seeds=[],
        context_ids=selected_contexts,
        interact_num=int(args.interact_num),
        num_inference_steps=int(args.num_inference_steps),
        zero_identity_tolerance=float(args.zero_identity_tolerance),
        output_root=anchor_root,
    )
    screen = run_anchor_screen(screen_args)
    confirmation_root = output / "confirmation"
    confirmation_manifest = evaluate(
        selector_path=Path(args.development_selector),
        route_selector_path=route_selector,
        result_path=anchor_root / "result.json",
        output_root=confirmation_root,
    )
    confirmation = json.loads(
        (confirmation_root / "confirmation.json").read_text(encoding="utf-8")
    )
    checks = screen.get("zero_identity_checks")
    zero_passed = isinstance(checks, list) and bool(checks) and all(
        isinstance(row, dict) and row.get("state") == "passed" for row in checks
    )
    hook = screen.get("hook_activation")
    metrics = {
        "schema_version": 1,
        "artifact_type": "verdiwm-ctrl-world-anchor-repair-candidate-metrics",
        "state": "ready",
        "metrics": {
            "pipeline_ready": 1.0,
            "route_execute_count": float(len(selected_contexts)),
            "hook_activation_passed": 1.0
            if isinstance(hook, dict) and hook.get("state") == "passed"
            else 0.0,
            "zero_identity_passed": 1.0 if zero_passed else 0.0,
            "scientific_confirm_passed": 1.0
            if confirmation.get("decision") == "passed_independent_context_confirm"
            else 0.0,
            "gain_lcb": float(confirmation.get("gain_lcb", 0.0)),
            "harmful_context_count": float(
                confirmation.get("harmful_context_count", 0)
            ),
        },
        "confirmation": confirmation_manifest,
        "claim_boundary": (
            "The validity gates prove an independently routed confirmation was "
            "executed. The scientific_confirm_passed metric records the effect "
            "without converting a null or harmful result into a runtime failure."
        ),
    }
    (output / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metrics


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--ctrl-world-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--data-stat", type=Path, required=True)
    parser.add_argument("--svd-model-path", type=Path, required=True)
    parser.add_argument("--clip-model-path", type=Path, required=True)
    parser.add_argument("--ckpt-path", type=Path, required=True)
    parser.add_argument("--contexts-json", type=Path, required=True)
    parser.add_argument("--route-campaign", type=Path, required=True)
    parser.add_argument("--route-result", type=Path, required=True)
    parser.add_argument("--development-selector", type=Path, required=True)
    parser.add_argument("--candidate-radius", type=float, default=0.025)
    parser.add_argument("--confidence-z", type=float, default=1.96)
    parser.add_argument("--minimum-gain-lcb", type=float, default=0.0)
    parser.add_argument("--interact-num", type=int, default=4)
    parser.add_argument("--num-inference-steps", type=int, default=2)
    parser.add_argument("--zero-identity-tolerance", type=float, default=1e-6)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    result = run(_parser().parse_args(argv))
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
