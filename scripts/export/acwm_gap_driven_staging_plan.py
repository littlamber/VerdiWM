#!/usr/bin/env python3
"""Build an ACWM-Phys gap-driven staging plan from settled evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from wmloop.execute.acwm_primitive_routes import INVALIDATED_QUALITY_PRIMITIVES


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EFFECTS = ROOT / "results/reports/paper-export-r1/tables/primitive_effects_by_cell.csv"
DEFAULT_GOAL = ROOT / "configs/goal/g1_long_horizon_ladder_v1.yaml"
DEFAULT_PROBES = ROOT / "configs/probes/acwm_v1.json"
DEFAULT_MECHANISM_CARDS = ROOT / "results/reports/primitive-mechanism-cards-r1/mechanism_cards.md"
DEFAULT_PROGRESS = ROOT / "results/reports/progress-generalization-r1/progress-generalization.md"
DEFAULT_OUT = ROOT / "results/reports/acwm-gap-driven-staging-plan-r1"
ACWM_ENVS = (
    "push_cube",
    "stack_cube",
    "push_rope",
    "cloth_move",
    "push_sand",
    "pour_water",
    "robot_arm",
    "reacher",
)


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def as_int(row: dict[str, str], field: str) -> int:
    try:
        return int(float(row.get(field, "0") or 0))
    except ValueError:
        return 0


def as_float(row: dict[str, str], field: str) -> float:
    try:
        return float(row.get(field, "0") or 0)
    except ValueError:
        return 0.0


def classify_cell(row: dict[str, str]) -> str:
    if row.get("primitive") in INVALIDATED_QUALITY_PRIMITIVES:
        return "invalidated_method_evidence"
    trials = as_int(row, "trial_count")
    positives = as_int(row, "positive_trial_count")
    positive_rate = as_float(row, "positive_rate")
    mean_delta = as_float(row, "mean_delta")
    max_delta = as_float(row, "max_delta")
    if trials >= 10 and positives >= 5 and positive_rate >= 0.5 and mean_delta > 0:
        return "repeated_positive"
    if positives > 0 and max_delta > 0:
        return "candidate_unstable"
    if trials >= 5 and mean_delta < 0:
        return "negative_or_avoid"
    return "insufficient_evidence"


def best_cell_for_env(rows: list[dict[str, str]], env: str) -> dict[str, Any]:
    env_rows = [row for row in rows if row.get("environment") == env]
    if not env_rows:
        return {
            "environment": env,
            "best_primitive": "none",
            "trial_count": 0,
            "positive_trial_count": 0,
            "positive_rate": 0.0,
            "mean_delta": 0.0,
            "max_delta": 0.0,
            "evidence_level": "untested",
            "best_manifest": "",
        }
    repeated = [row for row in env_rows if classify_cell(row) == "repeated_positive"]
    pool = repeated or env_rows
    best = sorted(pool, key=lambda row: (as_float(row, "mean_delta"), as_float(row, "positive_rate"), as_int(row, "trial_count")), reverse=True)[0]
    return {
        "environment": env,
        "best_primitive": best.get("primitive", "unknown"),
        "trial_count": as_int(best, "trial_count"),
        "positive_trial_count": as_int(best, "positive_trial_count"),
        "positive_rate": as_float(best, "positive_rate"),
        "mean_delta": as_float(best, "mean_delta"),
        "max_delta": as_float(best, "max_delta"),
        "evidence_level": classify_cell(best),
        "best_manifest": best.get("best_manifest", ""),
    }


def env_gap_family(env: str) -> str:
    if env in {"push_cube", "stack_cube", "push_rope"}:
        return "contact_topology_and_object_interaction"
    if env == "push_sand":
        return "granular_deformation_and_contact"
    if env == "cloth_move":
        return "deformable_surface_dynamics"
    if env == "pour_water":
        return "fluid_transport_and_container_interaction"
    if env == "reacher":
        return "action_binding_or_short_horizon_control"
    if env == "robot_arm":
        return "articulated_motion"
    return "unknown"


def suggested_existing_primitives(env: str) -> list[str]:
    mapping = {
        "push_cube": ["frontier_collection", "mixture_reweight", "inv_dyn_reward_finetune", "next_forcing"],
        "stack_cube": [
            "frontier_collection",
            "mixture_reweight",
            "latent_spatial_memory",
            "inv_dyn_reward_finetune",
            "latent_motion_prior",
            "action_contrastive_finetune",
        ],
        "push_rope": [
            "frontier_collection",
            "mixture_reweight",
            "next_forcing",
            "self_forcing_finetune",
            "latent_motion_prior",
            "action_contrastive_finetune",
        ],
        "cloth_move": ["first_frame_anchor", "latent_spatial_memory", "next_forcing", "self_forcing_finetune"],
        "push_sand": ["frontier_collection", "mixture_reweight", "dino_rep_injection", "next_forcing"],
        "pour_water": [
            "event_window_reweight",
            "self_forcing_finetune",
            "next_forcing",
            "drift_token_trim",
            "history_noise_schedule",
        ],
        "reacher": ["inv_dyn_reward_finetune", "cfg_guidance_schedule", "next_forcing"],
        "robot_arm": ["action_contrastive_finetune", "inv_dyn_reward_finetune", "cfg_guidance_schedule"],
    }
    return mapping.get(env, [])


def diagnostic_probe_candidates(env: str) -> list[str]:
    mapping = {
        "push_cube": ["contact_event_probe", "rigid_pose_slip_probe"],
        "stack_cube": ["support_relation_probe", "contact_instability_probe"],
        "push_rope": ["topology_change_probe", "endpoint_path_probe"],
        "cloth_move": ["surface_fold_probe", "cloth_identity_drift_probe"],
        "push_sand": ["granular_frontier_probe", "mass_redistribution_probe"],
        "pour_water": ["fluid_volume_transport_probe", "container_boundary_leak_probe", "free_surface_probe"],
        "reacher": ["target_conditioning_probe", "inverse_dynamics_confidence_probe"],
        "robot_arm": ["joint_motion_consistency_probe", "articulated_action_binding_probe"],
    }
    return mapping.get(env, [])


def staging_priority(record: dict[str, Any]) -> str:
    level = record["evidence_level"]
    if level == "repeated_positive":
        return "confirm_and_visualize"
    if level == "candidate_unstable":
        return "factorize_then_canary"
    return "diagnose_gap_then_stage"


def load_live_coverage(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("ACWM_STAGING_LIVE_COVERAGE_INVALID") from exc
    rows = payload.get("rows") if isinstance(payload, dict) else None
    if (
        not isinstance(payload, dict)
        or payload.get("artifact_type") != "wmloop-acwm-8env-live-coverage"
        or payload.get("state") != "ready"
        or not isinstance(payload.get("summary"), dict)
        or not isinstance(rows, list)
        or not all(isinstance(row, dict) for row in rows)
    ):
        raise ValueError("ACWM_STAGING_LIVE_COVERAGE_INVALID")
    environments = [row.get("environment") for row in rows]
    if len(environments) != len(set(environments)) or set(environments) != set(ACWM_ENVS):
        raise ValueError("ACWM_STAGING_LIVE_COVERAGE_ENVIRONMENTS_INVALID")
    return payload


def build_report(
    *,
    effects_path: Path,
    goal_path: Path,
    probes_path: Path,
    mechanism_cards_path: Path,
    progress_path: Path,
    live_coverage_path: Path | None = None,
) -> dict[str, Any]:
    effects = read_csv_dicts(effects_path)
    live_coverage = load_live_coverage(live_coverage_path) if live_coverage_path is not None else None
    live_by_environment = {
        str(row["environment"]): row for row in live_coverage["rows"]
    } if live_coverage is not None else {}
    cells = []
    for row in effects:
        enriched = dict(row)
        enriched["evidence_level"] = classify_cell(row)
        cells.append(enriched)

    env_records = []
    for env in ACWM_ENVS:
        record = best_cell_for_env(effects, env)
        record["gap_family"] = env_gap_family(env)
        record["recommended_existing_primitives"] = suggested_existing_primitives(env)
        record["diagnostic_probe_candidates"] = diagnostic_probe_candidates(env)
        live = live_by_environment.get(env)
        if live is not None:
            record.update(
                {
                    "live_coverage_state": str(live.get("coverage_state") or "unknown"),
                    "formally_confirmed": live.get("formally_confirmed") is True,
                    "claim_ready": live.get("claim_ready") is True,
                    "confirmed_methods": str(live.get("confirmed_methods") or ""),
                    "event_semantic_required": live.get("event_semantic_required") is True,
                    "event_semantic_validated": live.get("event_semantic_validated") is True,
                }
            )
            if record["claim_ready"]:
                record["evidence_level"] = "formally_confirmed_positive"
                record["best_primitive"] = record["confirmed_methods"] or record["best_primitive"]
            elif record["live_coverage_state"] == "metric_positive_pending_event_validation":
                record["evidence_level"] = "metric_positive_event_pending"
                record["best_primitive"] = record["confirmed_methods"] or record["best_primitive"]
            record["next_action"] = str(live.get("next_action") or staging_priority(record))
        else:
            record["next_action"] = staging_priority(record)
        env_records.append(record)

    repeated_positive = [
        record
        for record in env_records
        if record.get("claim_ready") is True
        or (live_coverage is None and record["evidence_level"] == "repeated_positive")
    ]
    formally_confirmed = [record for record in env_records if record.get("formally_confirmed") is True]
    unstable = [record for record in env_records if record["evidence_level"] == "candidate_unstable"]
    gap_envs = [record for record in env_records if record not in repeated_positive]
    staging_actions = []
    for record in env_records:
        if record in repeated_positive:
            staging_actions.append(
                {
                    "environment": record["environment"],
                    "work_type": "evidence_retention_and_transfer",
                    "priority": "P0",
                    "action": "Retain confirmed checkpoints, paired evidence, and causal routing experience; do not spend gap-search budget on this environment.",
                    "allowed_layer": "verifier_runs_only",
                    "frozen_boundary": "No probe, registry, goal, or eval-protocol mutation.",
                }
            )
            continue
        staging_actions.append(
            {
                "environment": record["environment"],
                "work_type": "diagnostic_probe_staging",
                "priority": "P0",
                "action": "Stage diagnostic-only probes: " + ", ".join(record["diagnostic_probe_candidates"]),
                "allowed_layer": "diagnostic_signal_only",
                "frozen_boundary": "Diagnostic evidence can route proposals; verdict probes remain frozen.",
            }
        )
        staging_actions.append(
            {
                "environment": record["environment"],
                "work_type": "primitive_candidate_staging",
                "priority": "P1",
                "action": "Prefer existing unexhausted primitives first: " + ", ".join(record["recommended_existing_primitives"]),
                "allowed_layer": "staging_or_frozen_registry_call",
                "frozen_boundary": "New mechanism code must enter staging/admission; it cannot enter the current formal registry.",
            }
        )

    return {
        "schema_version": 1,
        "artifact_type": "wmloop-acwm-gap-driven-staging-plan",
        "state": "ready",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_files": {
            "primitive_effects_by_cell": str(effects_path),
            "primitive_effects_by_cell_sha256": sha256_file(effects_path),
            "goal_spec": str(goal_path),
            "goal_spec_sha256": sha256_file(goal_path),
            "probe_registry": str(probes_path),
            "probe_registry_sha256": sha256_file(probes_path),
            "mechanism_cards": str(mechanism_cards_path),
            "mechanism_cards_sha256": sha256_file(mechanism_cards_path),
            "progress_generalization_report": str(progress_path),
            "progress_generalization_report_sha256": sha256_file(progress_path),
            "live_coverage": str(live_coverage_path) if live_coverage_path is not None else None,
            "live_coverage_sha256": sha256_file(live_coverage_path) if live_coverage_path is not None else None,
        },
        "claim_boundary": {
            "current_scientific_claim": "ACWM-Phys closed-loop execution is proven; broad all-environment improvement is not proven.",
            "formally_confirmed_environment_count": len(formally_confirmed),
            "formally_confirmed_environments": [record["environment"] for record in formally_confirmed],
            "stable_positive_environment_count": len(repeated_positive),
            "stable_positive_environments": [record["environment"] for record in repeated_positive],
            "unstable_candidate_environment_count": len(unstable),
            "gap_environment_count": len(gap_envs),
            "formal_registry_mutation_allowed": False,
            "verdict_probe_mutation_allowed": False,
            "diagnostic_probe_staging_allowed": True,
            "primitive_staging_allowed": True,
        },
        "environment_records": env_records,
        "cell_records": cells,
        "staging_actions": staging_actions,
        "next_project_step": {
            "recommended": "Run gap-driven staging before another broad M4 expansion.",
            "why": "The archive already has enough negative evidence to stop blind exploration and target missing diagnostic/proposal coverage.",
            "p0_sequence": [
                "Retain metric-passing results from invalidated methods as audit-only evidence.",
                "Create diagnostic-only staging probes for every environment without a valid stable positive.",
                "Run diagnosis-matched 512 screens for valid runtime-admitted primitives.",
            ],
            "p1_sequence": [
                "Exhaust existing untested frozen-registry primitives against the diagnosed gaps.",
                "Use literature or Zone B only to create staging candidates, then require admission gates and a new version boundary.",
                "Continue Ctrl-World as an ACWM predictive-quality pilot after paired GT, evaluator adapter, hook adapter, and primitive mapping are ready.",
            ],
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    boundary = report["claim_boundary"]
    lines = [
        "# ACWM Gap-Driven Staging Plan R1",
        "",
        f"Generated at: `{report['generated_at']}`",
        "",
        "## Bottom Line",
        "",
        "- Stable repeated-positive environments under the current valid-method contract: "
        + (", ".join(f"`{env}`" for env in boundary["stable_positive_environments"]) or "none")
        + ".",
        "- This is a closed-loop engineering success, but not an all-environment quality-improvement result.",
        "- Do not run another broad blind expansion before converting the negative cells into diagnostic and staging work.",
        "- Verdict probes and the formal primitive registry remain frozen; only diagnostic probes and staging candidates can evolve.",
        "",
        "## Environment Status",
        "",
        "| Environment | Best current primitive | Evidence level | Trials | Positive rate | Mean delta | Max delta | Next action |",
        "|:--|:--|:--|--:|--:|--:|--:|:--|",
    ]
    for record in report["environment_records"]:
        lines.append(
            "| {environment} | {best_primitive} | {evidence_level} | {trial_count} | {positive_rate:.3f} | {mean_delta:.3f} | {max_delta:.3f} | {next_action} |".format(
                **record
            )
        )
    lines.extend(
        [
            "",
            "## Gap-Driven Work Orders",
            "",
            "| Environment | Work type | Priority | Action | Boundary |",
            "|:--|:--|:--|:--|:--|",
        ]
    )
    for action in report["staging_actions"]:
        lines.append(
            f"| {action['environment']} | {action['work_type']} | {action['priority']} | {action['action']} | {action['frozen_boundary']} |"
        )
    lines.extend(
        [
            "",
            "## Project Priority",
            "",
            "P0 should be ACWM gap-driven staging plus valid-method confirmation:",
            "",
            "1. Retain invalidated-method metric passes as audit-only evidence.",
            "2. Stage diagnostic-only probes for environments without a valid stable positive result.",
            "3. Run only diagnosis-matched, runtime-admitted primitives through the frozen quality gate.",
            "",
            "P1 should be primitive coverage repair:",
            "",
            "1. Exhaust existing untested frozen-registry primitives before inventing new mechanism code.",
            "2. Put any literature/Zone B discovery into staging only.",
            "3. Promote only through schema, clean diff, smoke/canary, and human-approved version boundary.",
            "",
            "P2 should be the Ctrl-World ACWM predictive-quality pilot instance:",
            "",
            "1. Freeze held-out split.",
            "2. Add independent evaluator adapter.",
            "3. Add hook adapter and mapped primitive registry.",
            "4. Run bounded smoke before any formal G2 claim.",
            "",
            "## Source Files",
            "",
        ]
    )
    for name, value in report["source_files"].items():
        if name.endswith("_sha256"):
            continue
        digest = report["source_files"].get(f"{name}_sha256")
        lines.append(f"- `{name}`: `{value}` sha256=`{digest}`")
    lines.append("")
    return "\n".join(lines)


def write_report(report: dict[str, Any], output_root: Path) -> dict[str, Any]:
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"output exists: {output_root}")
    temporary = output_root.parent / f".{output_root.name}.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True, mode=0o700)
    try:
        write_json(temporary / "acwm-gap-driven-staging-plan.json", report)
        (temporary / "acwm-gap-driven-staging-plan.md").write_text(render_markdown(report), encoding="utf-8")
        write_csv(
            temporary / "tables/environment_gap_status.csv",
            report["environment_records"],
            [
                "environment",
                "best_primitive",
                "evidence_level",
                "trial_count",
                "positive_trial_count",
                "positive_rate",
                "mean_delta",
                "max_delta",
                "gap_family",
                "next_action",
                "best_manifest",
            ],
        )
        write_csv(
            temporary / "tables/staging_actions.csv",
            report["staging_actions"],
            ["environment", "work_type", "priority", "action", "allowed_layer", "frozen_boundary"],
        )
        manifest = {
            "schema_version": 1,
            "artifact_type": "wmloop-acwm-gap-driven-staging-plan-manifest",
            "state": "ready",
            "output_root": str(output_root),
            "report_path": str(output_root / "acwm-gap-driven-staging-plan.json"),
            "markdown_path": str(output_root / "acwm-gap-driven-staging-plan.md"),
            "environment_gap_status_path": str(output_root / "tables/environment_gap_status.csv"),
            "staging_actions_path": str(output_root / "tables/staging_actions.csv"),
            "claim_boundary": report["claim_boundary"],
            "next_project_step": report["next_project_step"],
        }
        write_json(temporary / "manifest.json", manifest)
        temporary.replace(output_root)
        return manifest
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--effects", type=Path, default=DEFAULT_EFFECTS)
    parser.add_argument("--goal", type=Path, default=DEFAULT_GOAL)
    parser.add_argument("--probes", type=Path, default=DEFAULT_PROBES)
    parser.add_argument("--mechanism-cards", type=Path, default=DEFAULT_MECHANISM_CARDS)
    parser.add_argument("--progress-report", type=Path, default=DEFAULT_PROGRESS)
    parser.add_argument("--live-coverage", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    report = build_report(
        effects_path=args.effects.resolve(strict=True),
        goal_path=args.goal.resolve(strict=False),
        probes_path=args.probes.resolve(strict=False),
        mechanism_cards_path=args.mechanism_cards.resolve(strict=False),
        progress_path=args.progress_report.resolve(strict=False),
        live_coverage_path=args.live_coverage.resolve(strict=True) if args.live_coverage is not None else None,
    )
    manifest = write_report(report, args.output_root.resolve())
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
