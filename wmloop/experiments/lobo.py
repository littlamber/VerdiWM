"""Deterministic leave-one-backbone-out experiment planning."""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Mapping

from wmloop.experiments._artifacts import canonical_json, write_bundle
from wmloop.experiments.spec import ARMS, SELECTORS, backbone_map, load_experiment_spec, stage_map


class LoboPlanError(RuntimeError):
    """A LOBO plan could not be generated."""


def build_lobo_plan(spec: Mapping[str, Any]) -> dict[str, object]:
    """Expand a validated experiment spec into non-leaking trial contracts."""

    backbones = backbone_map(spec)
    stages = stage_map(spec)
    selector_contracts = {str(item["selector"]): str(item["input_contract"]) for item in spec["selectors"]}
    arm_policies = {str(item["arm"]): str(item["candidate_policy"]) for item in spec["arms"]}
    trials: list[dict[str, object]] = []
    sequence_index = 0
    for fold in spec["folds"]:
        fold_id = str(fold["fold_id"])
        target = str(fold["target_backbone"])
        sources = tuple(sorted(str(value) for value in fold["source_backbones"]))
        for scenario in sorted(str(value) for value in fold["scenarios"]):
            for arm in ARMS:
                selectors = SELECTORS if arm == "warm_start" else ("none",)
                for selector in selectors:
                    for seed in sorted(int(value) for value in spec["seeds"]):
                        sequence_index += 1
                        trial_id = _trial_id(fold_id, scenario, arm, selector, seed)
                        trials.append(
                            {
                                "trial_id": trial_id,
                                "sequence_index": sequence_index,
                                "fold_id": fold_id,
                                "target_backbone": target,
                                "source_backbones": list(sources) if arm == "warm_start" else [],
                                "scenario": scenario,
                                "seed": seed,
                                "arm": arm,
                                "selector": selector,
                                "selector_input_contract": selector_contracts.get(selector),
                                "candidate_policy": arm_policies[arm],
                                "source_experience_allowed": arm == "warm_start",
                                "stages": [
                                    {
                                        "stage": stage,
                                        "max_gpu_hours": float(stages[stage]["max_gpu_hours"]),
                                        "formal_evidence": bool(stages[stage]["formal_evidence"]),
                                        "condition": _stage_condition(stage),
                                    }
                                    for stage in stages
                                ],
                                "formal_positive_requires": "settled_confirm_receipt",
                            }
                        )
    blockers = _launch_blockers(spec, backbones)
    return {
        "schema_version": 1,
        "artifact_type": "verdiwm-lobo-experiment-plan",
        "experiment_id": spec["experiment_id"],
        "state": "ready" if not blockers else "blocked",
        "planning_ready": True,
        "launch_ready": not blockers,
        "claim_scope": spec["claim_scope"],
        "metric_contract": dict(spec["metric_contract"]),
        "required_arms": list(ARMS),
        "required_selectors": list(SELECTORS),
        "stage_order": list(stages),
        "fold_count": len(spec["folds"]),
        "planned_trial_count": len(trials),
        "planned_stage_task_count": len(trials) * len(stages),
        "trials": trials,
        "blockers": blockers,
        "invariants": [
            "The target backbone is absent from every fold source set.",
            "Only warm_start may consume cross-backbone experience.",
            "random_search is target-registry sampling and is not shuffled_prior.",
            "Screen is exploratory diagnostics and never a scientific veto; gate may run without screen.",
            "Screen and gate results cannot establish a formal positive result.",
            "A formal positive requires a settled confirm receipt.",
        ],
    }


def run_experiment_plan(
    *,
    spec_path: Path,
    output_root: Path,
    archive_db: Path | None = None,
    cas_root: Path | None = None,
) -> dict[str, object]:
    spec = load_experiment_spec(spec_path)
    report = build_lobo_plan(spec)
    public_spec = {key: value for key, value in spec.items() if not key.startswith("_")}
    files = {
        "experiment-plan.json": canonical_json(report),
        "experiment-plan.md": _render_markdown(report).encode("utf-8"),
        "planned-trials.csv": _render_csv(report).encode("utf-8"),
        "input-spec.json": canonical_json(public_spec),
    }
    destination = Path(output_root).resolve()
    return write_bundle(
        output_root=destination,
        files=files,
        manifest_fields={
            "artifact_type": "verdiwm-lobo-experiment-plan-manifest",
            "state": report["state"],
            "experiment_id": report["experiment_id"],
            "planning_ready": report["planning_ready"],
            "launch_ready": report["launch_ready"],
            "planned_trial_count": report["planned_trial_count"],
            "planned_stage_task_count": report["planned_stage_task_count"],
            "blocker_count": len(report["blockers"]),
            "report_path": str(destination / "experiment-plan.json"),
            "table_path": str(destination / "planned-trials.csv"),
        },
        archive_db=archive_db,
        cas_root=cas_root,
    )


def _launch_blockers(spec: Mapping[str, Any], backbones: Mapping[str, Mapping[str, Any]]) -> list[dict[str, object]]:
    blockers: list[dict[str, object]] = []
    used = sorted(
        {str(fold["target_backbone"]) for fold in spec["folds"]}
        | {str(source) for fold in spec["folds"] for source in fold["source_backbones"]}
    )
    for backbone_id in used:
        status = str(backbones[backbone_id]["status"])
        if status != "ready":
            blockers.append(
                {
                    "code": "backbone_not_ready",
                    "backbone_id": backbone_id,
                    "status": status,
                    "instance_ref": backbones[backbone_id]["instance_ref"],
                }
            )
    return blockers


def _stage_condition(stage: str) -> str:
    return {
        "screen": "always",
        "gate": "always",
        "confirm": "official_gate_positive",
    }[stage]


def _trial_id(fold: str, scenario: str, arm: str, selector: str, seed: int) -> str:
    raw = f"{fold}__{scenario}__{arm}__{selector}__s{seed}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", raw)


def _render_csv(report: Mapping[str, object]) -> str:
    output = io.StringIO(newline="")
    fieldnames = [
        "sequence_index",
        "trial_id",
        "fold_id",
        "target_backbone",
        "source_backbones",
        "scenario",
        "seed",
        "arm",
        "selector",
        "candidate_policy",
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for trial in report["trials"]:  # type: ignore[index]
        row = dict(trial)
        row["source_backbones"] = ";".join(row["source_backbones"])
        writer.writerow({name: row.get(name, "") for name in fieldnames})
    return output.getvalue()


def _render_markdown(report: Mapping[str, object]) -> str:
    lines = [
        "# Cross-Backbone LOBO Plan",
        "",
        f"Experiment: `{report['experiment_id']}`",
        f"State: `{report['state']}`",
        f"Planned trials: `{report['planned_trial_count']}`",
        f"Planned stage tasks: `{report['planned_stage_task_count']}`",
        "",
        "## Claim Boundary",
        "",
        "Only a settled confirm receipt can establish a positive result. Screen is optional exploratory diagnostics; gate is the frozen formal prerequisite for confirm.",
        "",
        "## Blockers",
        "",
    ]
    blockers = report["blockers"]
    if blockers:
        lines.extend(f"- `{item['code']}`: `{json.dumps(item, sort_keys=True)}`" for item in blockers)  # type: ignore[index]
    else:
        lines.append("- None.")
    lines.extend(["", "## Arm Semantics", ""])
    lines.extend(
        [
            "- `warm_start`: source Effect Memory is allowed; selector is explicit.",
            "- `cold_start`: target-local diagnosis/retrieval only.",
            "- `random_search`: uniform target registry sampling; never an alias for shuffled prior.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--archive-db", type=Path)
    parser.add_argument("--cas-root", type=Path)
    args = parser.parse_args(argv)
    manifest = run_experiment_plan(
        spec_path=args.spec,
        output_root=args.output_root,
        archive_db=args.archive_db,
        cas_root=args.cas_root,
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
