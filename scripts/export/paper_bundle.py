#!/usr/bin/env python3
"""Export paper-facing tables and figures from settled wm-loop artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wmloop.execute.acwm_primitive_routes import INVALIDATED_QUALITY_PRIMITIVES
from scripts.export.acwm_8env_live_coverage import build_live_coverage
from scripts.export.acwm_autoloop_daemon import _collect_quality_gate_inventory

PRIMITIVE_DIR = ROOT / "wmloop" / "primitives" / "definitions"
REPORTS = ROOT / "results" / "reports"


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def safe_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def primitive_names() -> list[str]:
    return sorted(p.name for p in PRIMITIVE_DIR.iterdir() if p.is_dir())


def infer_primitive(proposal_id: str, names: list[str]) -> str:
    for name in names:
        if name in proposal_id:
            return name
    return "unknown"


def write_df(df: pd.DataFrame, csv_path: Path, latex_path: Path | None = None) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    if latex_path is not None:
        with latex_path.open("w", encoding="utf-8") as f:
            f.write(df.to_latex(index=False, escape=False, float_format=lambda x: f"{x:.3f}"))


def load_trial_rows() -> pd.DataFrame:
    names = primitive_names()
    rows: list[dict[str, Any]] = []
    for path in REPORTS.glob("**/envs/*/manifest.json"):
        try:
            data = read_json(path)
        except Exception:
            continue
        delta = (data.get("delta_m_ver") or {}).get("ladder_auc_psnr_envmax")
        if not isinstance(delta, (int, float)):
            continue
        proposal_id = data.get("proposal_id") or ""
        rows.append(
            {
                "environment": data.get("environment") or path.parent.name,
                "primitive": infer_primitive(proposal_id, names),
                "method_invalidated": infer_primitive(proposal_id, names) in INVALIDATED_QUALITY_PRIMITIVES,
                "proposal_id": proposal_id,
                "seed": data.get("seed"),
                "verdict": data.get("verdict"),
                "state": data.get("state"),
                "delta_ladder_auc_psnr_envmax": float(delta),
                "action_following_observed": (data.get("action_following_gate") or {}).get("observed"),
                "goal_id": data.get("goal_id"),
                "primary_metric": data.get("primary_metric"),
                "manifest_path": safe_rel(path),
                "report_path": data.get("report_path"),
                "evaluation_dir": data.get("evaluation_dir"),
            }
        )
    return pd.DataFrame(rows)


def load_official_gate_rows(report_root: Path = REPORTS) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    inventory = _collect_quality_gate_inventory([], report_root=report_root)
    classified_by_manifest = {
        str(Path(str(record["manifest_path"])).resolve()): record
        for record in inventory.get("records", [])
        if isinstance(record, dict) and isinstance(record.get("manifest_path"), str)
    }
    manifests: set[Path] = set()
    for pattern in (
        "acwm-formal-visualization*/manifest.json",
        "acwm-official-gate-*/manifest.json",
        "acwm-autoloop-official-gate-*/manifest.json",
        "acwm-autoloop-confirm-official-gate-*/manifest.json",
    ):
        manifests.update(report_root.glob(pattern))
    for path in sorted(manifests):
        try:
            data = read_json(path)
        except Exception:
            continue
        gate = data.get("official_quality_gate")
        if data.get("state") != "ready" or not isinstance(gate, dict):
            continue
        baseline = gate.get("baseline") or {}
        candidate = gate.get("candidate") or {}
        delta = gate.get("delta_candidate_minus_baseline") or {}
        if not all(isinstance(delta.get(name), (int, float)) for name in ("psnr", "ssim", "mse", "masked_mse")):
            continue
        classified = classified_by_manifest.get(str(path.resolve()), {})
        retention = classified.get("retention") if isinstance(classified, dict) else None
        claim_tier = retention.get("claim_tier") if isinstance(retention, dict) else None
        if not isinstance(claim_tier, str) or not claim_tier:
            claim_tier = (
                "C_gate_pass_method_invalidated_audit_only"
                if data.get("primitive") in INVALIDATED_QUALITY_PRIMITIVES and gate.get("pass") is True
                else "B_gate_pass_confirmation_pending"
                if gate.get("pass") is True
                else "D_gate_failed_audit_only"
            )
        phase = classified.get("phase") if isinstance(classified, dict) else None
        rows.append(
            {
                "stage": "confirm_1k_ladder" if phase == "confirm_official_eval_gate" else "screen_512",
                "environment": data.get("environment"),
                "primitive": data.get("primitive"),
                "seed": data.get("seed"),
                "steps": data.get("steps"),
                "checkpoint_step": classified.get("checkpoint_step") if isinstance(classified, dict) else None,
                "official_pass": gate.get("pass") is True,
                "baseline_psnr": baseline.get("psnr"),
                "candidate_psnr": candidate.get("psnr"),
                "delta_psnr": delta.get("psnr"),
                "baseline_ssim": baseline.get("ssim"),
                "candidate_ssim": candidate.get("ssim"),
                "delta_ssim": delta.get("ssim"),
                "baseline_mse": baseline.get("mse"),
                "candidate_mse": candidate.get("mse"),
                "delta_mse": delta.get("mse"),
                "baseline_masked_mse": baseline.get("masked_mse"),
                "candidate_masked_mse": candidate.get("masked_mse"),
                "delta_masked_mse": delta.get("masked_mse"),
                "candidate_checkpoint_sha256": data.get("candidate_checkpoint_sha256", ""),
                "manifest_path": safe_rel(path),
                "paired_video_count": data.get("paired_video_count", 0),
                "method_invalidated": str(data.get("primitive") or "") in INVALIDATED_QUALITY_PRIMITIVES,
                "claim_tier": claim_tier,
            }
        )
    return pd.DataFrame(rows)


def export_official_quality_gates(gates: pd.DataFrame, out: Path) -> None:
    if gates.empty:
        return
    ordered = gates.sort_values(["official_pass", "delta_psnr"], ascending=[False, False])
    write_df(
        ordered,
        out / "tables" / "official_quality_gates.csv",
        out / "tables" / "official_quality_gates.tex",
    )
    labels = ordered.apply(
        lambda row: f"{row['stage']} | {row['environment']} | {row['primitive']}",
        axis=1,
    )
    metrics = (
        ("delta_psnr", "PSNR delta", True),
        ("delta_ssim", "SSIM delta", True),
        ("delta_mse", "MSE delta", False),
        ("delta_masked_mse", "Masked-MSE delta", False),
    )
    fig_h = max(4.8, 0.32 * len(ordered) + 1.8)
    fig, axes = plt.subplots(2, 2, figsize=(13.0, fig_h * 1.6))
    for ax, (column, title, higher_is_better) in zip(axes.flat, metrics, strict=True):
        values = ordered[column].astype(float)
        good = values > 0 if higher_is_better else values < 0
        colors = ["#1f7a4d" if value else "#b6423c" for value in good]
        ax.barh(labels, values, color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(title)
        ax.invert_yaxis()
    fig.suptitle("Official ACWM 50-step quality gates")
    fig.tight_layout()
    fig.savefig(out / "figures" / "official_quality_gate_deltas.png", dpi=200)
    fig.savefig(out / "figures" / "official_quality_gate_deltas.pdf")
    plt.close(fig)


def load_environment_coverage(report_root: Path = REPORTS) -> tuple[pd.DataFrame, dict[str, Any]]:
    report = build_live_coverage(report_root=report_root)
    rows = report.get("rows")
    if report.get("state") != "ready" or not isinstance(rows, list):
        return pd.DataFrame(), report
    return pd.DataFrame(row for row in rows if isinstance(row, dict)), report


def export_environment_coverage(
    coverage: pd.DataFrame,
    report: dict[str, Any],
    out: Path,
) -> None:
    if coverage.empty:
        return
    write_df(
        coverage,
        out / "tables" / "acwm_8env_live_coverage.csv",
        out / "tables" / "acwm_8env_live_coverage.tex",
    )
    write_json(out / "tables" / "acwm_8env_live_coverage.json", report)

    state_order = {
        "formally_confirmed_positive": 5,
        "confirmation_pending": 4,
        "official_screen_pass_pending_confirmation": 3,
        "internal_positive_pending_official_gate": 2,
        "screen_running": 1,
        "exploring_unconfirmed": 0,
    }
    state_colors = {
        "formally_confirmed_positive": "#1f7a4d",
        "confirmation_pending": "#3976a8",
        "official_screen_pass_pending_confirmation": "#6c8fb3",
        "internal_positive_pending_official_gate": "#b5862d",
        "screen_running": "#8b6cad",
        "exploring_unconfirmed": "#8a8f98",
    }
    ordered = coverage.copy()
    ordered["state_rank"] = ordered["coverage_state"].map(state_order).fillna(-1)
    ordered = ordered.sort_values(["state_rank", "environment"], ascending=[False, True])
    colors = [state_colors.get(str(state), "#8a8f98") for state in ordered["coverage_state"]]

    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    ax.barh(ordered["environment"], [1] * len(ordered), color=colors, height=0.62)
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_xlabel("512 screen -> official gate -> independent 800/1000 confirmation")
    ax.set_title("ACWM Closed-Loop Coverage by Environment")
    ax.invert_yaxis()
    for index, (_, row) in enumerate(ordered.iterrows()):
        methods = str(row.get("confirmed_methods") or "")
        label = f"confirmed: {methods}" if methods else str(row["coverage_state"]).replace("_", " ")
        ax.text(0.03, index, label, va="center", ha="left", fontsize=8, color="white")
    fig.tight_layout()
    fig.savefig(out / "figures" / "acwm_8env_live_coverage.png", dpi=200)
    fig.savefig(out / "figures" / "acwm_8env_live_coverage.pdf")
    plt.close(fig)


def export_primitive_tables(trials: pd.DataFrame, out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "method_invalidated" not in trials.columns:
        trials = trials.copy()
        trials["method_invalidated"] = False
    invalidated = trials[trials["method_invalidated"] == True]  # noqa: E712
    if not invalidated.empty:
        write_df(invalidated, out / "tables" / "invalidated_trial_manifest_rows.csv")
    claimable_trials = trials[trials["method_invalidated"] != True].copy()  # noqa: E712
    grouped = []
    for (env, primitive), g in claimable_trials.groupby(["environment", "primitive"], dropna=False):
        vals = g["delta_ladder_auc_psnr_envmax"].astype(float)
        representative = g.sort_values("delta_ladder_auc_psnr_envmax", ascending=False).iloc[0]
        grouped.append(
            {
                "environment": env,
                "primitive": primitive,
                "trial_count": int(len(g)),
                "positive_trial_count": int((vals > 0).sum()),
                "positive_rate": float((vals > 0).mean()),
                "mean_delta": float(vals.mean()),
                "median_delta": float(vals.median()),
                "std_delta": float(vals.std(ddof=0)) if len(vals) else 0.0,
                "min_delta": float(vals.min()),
                "max_delta": float(vals.max()),
                "best_manifest": representative["manifest_path"],
            }
        )
    effects = pd.DataFrame(grouped).sort_values("mean_delta", ascending=False)
    top = claimable_trials.sort_values("delta_ladder_auc_psnr_envmax", ascending=False).head(25)
    negative = effects[effects["mean_delta"] < 0].sort_values("mean_delta")
    write_df(effects, out / "tables" / "primitive_effects_by_cell.csv", out / "tables" / "primitive_effects_by_cell.tex")
    write_df(top, out / "tables" / "top_positive_trials.csv", out / "tables" / "top_positive_trials.tex")
    write_df(negative, out / "tables" / "negative_cells.csv", out / "tables" / "negative_cells.tex")
    write_df(trials, out / "tables" / "all_trial_manifest_rows.csv")
    return effects, top


def plot_primitive_effects(effects: pd.DataFrame, out: Path) -> None:
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    labels = effects.apply(lambda r: f"{r['environment']} | {r['primitive']}", axis=1)
    colors = ["#1f7a4d" if x >= 0 else "#b6423c" for x in effects["mean_delta"]]
    fig_h = max(4.5, 0.35 * len(effects) + 1.5)
    fig, ax = plt.subplots(figsize=(9.0, fig_h))
    ax.barh(labels, effects["mean_delta"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Mean Δ ladder AUC PSNR")
    ax.set_title("Primitive Effect by Environment Cell")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(fig_dir / "primitive_mean_delta_by_cell.png", dpi=200)
    fig.savefig(fig_dir / "primitive_mean_delta_by_cell.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.0, fig_h))
    clipped = effects["mean_delta"].clip(lower=-120, upper=80)
    ax.barh(labels, clipped, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlim(-120, 80)
    ax.set_xlabel("Mean Δ ladder AUC PSNR, clipped to [-120, 80]")
    ax.set_title("Primitive Effect by Environment Cell (Zoomed)")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(fig_dir / "primitive_mean_delta_by_cell_zoomed.png", dpi=200)
    fig.savefig(fig_dir / "primitive_mean_delta_by_cell_zoomed.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    size = effects["trial_count"].clip(lower=1) * 28
    ax.scatter(effects["mean_delta"], effects["positive_rate"], s=size, alpha=0.70, color="#315c9b", edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Mean Δ ladder AUC PSNR")
    ax.set_ylabel("Positive trial rate")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Effect Size vs Repeatability")
    for _, row in effects.iterrows():
        if row["mean_delta"] > 0 or row["positive_rate"] == 0:
            ax.annotate(str(row["environment"]), (row["mean_delta"], row["positive_rate"]), fontsize=7, xytext=(4, 3), textcoords="offset points")
    fig.tight_layout()
    fig.savefig(fig_dir / "primitive_repeatability_scatter.png", dpi=200)
    fig.savefig(fig_dir / "primitive_repeatability_scatter.pdf")
    plt.close(fig)


def horizon_metric_rows(case_id: str, baseline_path: Path, candidate_path: Path) -> pd.DataFrame:
    base = read_json(baseline_path)["aggregate"]["horizon_metrics"]
    cand = read_json(candidate_path)["aggregate"]["horizon_metrics"]
    rows = []
    for h in sorted(set(base) & set(cand), key=lambda x: int(x)):
        b = base[h]
        c = cand[h]
        rows.append(
            {
                "case_id": case_id,
                "horizon": int(h),
                "baseline_psnr": b["psnr"],
                "candidate_psnr": c["psnr"],
                "delta_psnr": c["psnr"] - b["psnr"],
                "baseline_ssim": b.get("ssim"),
                "candidate_ssim": c.get("ssim"),
                "delta_ssim": (c.get("ssim") - b.get("ssim")) if c.get("ssim") is not None and b.get("ssim") is not None else None,
                "baseline_mmse": b.get("masked_mse"),
                "candidate_mmse": c.get("masked_mse"),
                "delta_mmse": (c.get("masked_mse") - b.get("masked_mse")) if c.get("masked_mse") is not None and b.get("masked_mse") is not None else None,
                "baseline_metrics_path": safe_rel(baseline_path),
                "candidate_metrics_path": safe_rel(candidate_path),
            }
        )
    return pd.DataFrame(rows)


def horizon_effect_profile_rows(profile_path: Path) -> pd.DataFrame:
    profile = read_json(profile_path)
    effects = profile.get("horizon_effects")
    if profile.get("state") != "ready" or not isinstance(effects, list):
        return pd.DataFrame()
    source = profile.get("source") if isinstance(profile.get("source"), dict) else {}
    causal_credit_eligible = profile.get("causal_credit_eligible") is True
    case_id = profile_path.parent.name.removeprefix("acwm-horizon-effect-profile-")
    rows: list[dict[str, Any]] = []
    for effect in effects:
        if not isinstance(effect, dict) or not isinstance(effect.get("horizon"), int):
            continue
        baseline = effect.get("baseline")
        candidate = effect.get("candidate")
        delta = effect.get("delta_candidate_minus_baseline")
        if not isinstance(baseline, dict) or not isinstance(candidate, dict) or not isinstance(delta, dict):
            continue
        rows.append(
            {
                "case_id": case_id,
                "horizon": effect["horizon"],
                "baseline_psnr": baseline.get("psnr"),
                "candidate_psnr": candidate.get("psnr"),
                "delta_psnr": delta.get("psnr"),
                "baseline_ssim": baseline.get("ssim"),
                "candidate_ssim": candidate.get("ssim"),
                "delta_ssim": delta.get("ssim"),
                "baseline_mmse": baseline.get("masked_mse"),
                "candidate_mmse": candidate.get("masked_mse"),
                "delta_mmse": delta.get("masked_mse"),
                "baseline_metrics_path": safe_rel(Path(str(source.get("baseline_manifest") or ""))),
                "candidate_metrics_path": safe_rel(Path(str(source.get("candidate_manifest") or ""))),
                "claim_tier": (
                    "A_factorized_long_horizon_causal"
                    if causal_credit_eligible
                    else "B_paired_long_horizon_routing_prior"
                ),
                "claim_eligible": causal_credit_eligible,
                "effect_scope": (profile.get("effect_classification") or {}).get("effect_scope"),
                "strict_quality_pass": effect.get("strict_quality_pass"),
                "profile_path": safe_rel(profile_path),
            }
        )
    return pd.DataFrame(rows)


def plot_horizon_case(df: pd.DataFrame, title: str, stem: str, out: Path) -> None:
    fig_dir = out / "figures"
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    ax.plot(df["horizon"], df["baseline_psnr"], marker="o", label="Baseline")
    ax.plot(df["horizon"], df["candidate_psnr"], marker="o", label="Candidate")
    ax.set_xlabel("Horizon")
    ax.set_ylabel("PSNR")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / f"{stem}.png", dpi=200)
    fig.savefig(fig_dir / f"{stem}.pdf")
    plt.close(fig)


def export_horizon_cases(out: Path) -> pd.DataFrame:
    cases = [
        (
            "robot_arm_latent_motion_prior_s417",
            "robot_arm + latent_motion_prior (s417, audit only)",
            REPORTS / "m4-formal-campaign-auto-by-diagnosis-8env-train16-s417-r1/envs/robot_arm/evaluation/baseline-horizon-probe-metrics.json",
            REPORTS / "m4-formal-campaign-auto-by-diagnosis-8env-train16-s417-r1/envs/robot_arm/evaluation/candidate-horizon-probe-metrics.json",
        ),
        (
            "pour_water_drift_token_trim_s101",
            "pour_water + drift_token_trim (s101)",
            REPORTS / "t4-4-trial-runs-r3-extend-unreached-gpu12-7200/shuffled_prior-s101-pour_water-drift_token_trim-extend-t64/envs/pour_water/evaluation/baseline-horizon-probe-metrics.json",
            REPORTS / "t4-4-trial-runs-r3-extend-unreached-gpu12-7200/shuffled_prior-s101-pour_water-drift_token_trim-extend-t64/envs/pour_water/evaluation/candidate-horizon-probe-metrics.json",
        ),
    ]
    frames = []
    for case_id, title, baseline_path, candidate_path in cases:
        if baseline_path.exists() and candidate_path.exists():
            df = horizon_metric_rows(case_id, baseline_path, candidate_path)
            if case_id.startswith("robot_arm_latent_motion_prior"):
                df["claim_tier"] = "C_gate_pass_method_invalidated_audit_only"
                df["claim_eligible"] = False
            frames.append(df)
            plot_horizon_case(df, title, f"{case_id}_psnr_curve", out)
    for profile_path in sorted(REPORTS.glob("acwm-horizon-effect-profile-*/horizon-effect-profile.json")):
        try:
            df = horizon_effect_profile_rows(profile_path)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            continue
        if df.empty:
            continue
        frames.append(df)
        profile = read_json(profile_path)
        environment = str(profile.get("environment") or "unknown")
        primitive = str(profile.get("primitive") or "unknown")
        case_id = str(df.iloc[0]["case_id"])
        plot_horizon_case(
            df,
            f"{environment} + {primitive} (paired long horizon)",
            f"{case_id}_psnr_curve",
            out,
        )
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not combined.empty:
        write_df(combined, out / "tables" / "representative_horizon_metrics.csv", out / "tables" / "representative_horizon_metrics.tex")
    return combined


def export_t44(out: Path) -> pd.DataFrame:
    src = REPORTS / "t4-4-prior-convergence-progress-r15-r3-extend-unreached-negative-settled"
    csv_path = src / "prior-convergence.csv"
    if not csv_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    write_df(df, out / "tables" / "t44_prior_convergence_trials.csv", out / "tables" / "t44_prior_convergence_trials.tex")
    summary = (
        df.groupby("arm")
        .agg(
            trial_count=("delta_primary_metric", "count"),
            mean_delta=("delta_primary_metric", "mean"),
            max_delta=("delta_primary_metric", "max"),
            mean_gpu_hours=("gpu_hours", "mean"),
            positive_rate=("delta_primary_metric", lambda x: float((x > 0).mean())),
        )
        .reset_index()
    )
    write_df(summary, out / "tables" / "t44_prior_convergence_summary.csv", out / "tables" / "t44_prior_convergence_summary.tex")

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ordered = summary.sort_values("mean_delta", ascending=False)
    ax.bar(ordered["arm"], ordered["mean_delta"], color=["#1f7a4d" if x >= 0 else "#b6423c" for x in ordered["mean_delta"]])
    for i, arm in enumerate(ordered["arm"]):
        vals = df[df["arm"] == arm]["delta_primary_metric"]
        ax.scatter([i] * len(vals), vals, color="#333333", s=22, zorder=3)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Δ primary metric")
    ax.set_title("T4.4 Three-arm Prior Convergence Outcome")
    fig.tight_layout()
    fig.savefig(out / "figures" / "t44_prior_convergence_delta_by_arm.png", dpi=200)
    fig.savefig(out / "figures" / "t44_prior_convergence_delta_by_arm.pdf")
    plt.close(fig)
    return summary


def export_t45(out: Path) -> pd.DataFrame:
    src = REPORTS / "t4-5-verifier-meta-validation-progress-r25-full-archive-plus-s417-r1" / "verifier-meta-validation.json"
    if not src.exists():
        return pd.DataFrame()
    data = read_json(src)
    matrix = data.get("confusion_matrix") or {}
    rows = []
    for observed, truth_map in matrix.items():
        for truth, count in truth_map.items():
            rows.append({"observed_verdict": observed, "truth_label": truth, "count": count})
    df = pd.DataFrame(rows)
    write_df(df, out / "tables" / "t45_verifier_confusion_matrix.csv", out / "tables" / "t45_verifier_confusion_matrix.tex")
    rates = pd.DataFrame([data.get("rates") or {}])
    write_df(rates, out / "tables" / "t45_verifier_rates.csv", out / "tables" / "t45_verifier_rates.tex")

    if not df.empty:
        pivot = df.pivot(index="observed_verdict", columns="truth_label", values="count").fillna(0)
        fig, ax = plt.subplots(figsize=(7.0, 4.6))
        im = ax.imshow(pivot.values, cmap="Blues")
        ax.set_xticks(range(len(pivot.columns)), labels=pivot.columns, rotation=25, ha="right")
        ax.set_yticks(range(len(pivot.index)), labels=pivot.index)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                ax.text(j, i, int(pivot.iloc[i, j]), ha="center", va="center", fontsize=9)
        ax.set_title("T4.5 Verifier Meta-validation Matrix")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(out / "figures" / "t45_verifier_confusion_matrix.png", dpi=200)
        fig.savefig(out / "figures" / "t45_verifier_confusion_matrix.pdf")
        plt.close(fig)
    return df


def export_auxiliary_tables(out: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {}

    m5_path = REPORTS / "m5-consolidation-r31-after-m4-t52-t55-ready-clean" / "consolidation-report.json"
    if not m5_path.exists():
        m5_path = REPORTS / "m5-consolidation-r30" / "consolidation-report.json"
    if m5_path.exists():
        m5 = read_json(m5_path)
        write_df(pd.DataFrame(m5.get("claim_status") or []), out / "tables" / "m5_claim_status.csv", out / "tables" / "m5_claim_status.tex")
        write_df(pd.DataFrame(m5.get("m5_exports") or []), out / "tables" / "m5_export_status.csv", out / "tables" / "m5_export_status.tex")
        write_df(pd.DataFrame(m5.get("cost_summary") or []), out / "tables" / "m5_cost_summary.csv", out / "tables" / "m5_cost_summary.tex")
        summary["m5"] = {
            "source": safe_rel(m5_path),
            "state": m5.get("state"),
            "m5_export_complete": m5.get("m5_export_complete"),
            "archive_summary": m5.get("archive_summary"),
            "cost_summary": m5.get("cost_summary"),
        }

    adv_path = REPORTS / "adversarial-gate-audit-r2" / "adversarial-gate-audit.json"
    if adv_path.exists():
        adv = read_json(adv_path)
        obs = pd.DataFrame(adv.get("observations") or [])
        write_df(obs, out / "tables" / "t51_adversarial_gate_audit.csv", out / "tables" / "t51_adversarial_gate_audit.tex")
        required = obs[obs["required"] == True] if not obs.empty else obs
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        if not required.empty:
            ax.bar(required["attack_class"], required["intercepted"].astype(int), color="#315c9b")
            ax.set_ylim(0, 1.15)
            ax.set_ylabel("Intercepted")
            ax.set_title("T5.1 Required Attack Interception")
            ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(out / "figures" / "t51_adversarial_gate_interception.png", dpi=200)
        fig.savefig(out / "figures" / "t51_adversarial_gate_interception.pdf")
        plt.close(fig)
        summary["adversarial_gate_audit"] = {
            "source": safe_rel(adv_path),
            "state": adv.get("state"),
            "interception_rate": adv.get("interception_rate"),
            "intercepted_required_attack_count": adv.get("intercepted_required_attack_count"),
            "required_attack_count": adv.get("required_attack_count"),
        }

    surrogate_path = REPORTS / "surrogate-benefit-r1-after-m4" / "surrogate-benefit.json"
    if surrogate_path.exists():
        surr = read_json(surrogate_path)
        rows = pd.DataFrame(surr.get("ranking_rows") or [])
        write_df(rows, out / "tables" / "t52_surrogate_ranking_rows.csv", out / "tables" / "t52_surrogate_ranking_rows.tex")
        write_df(pd.DataFrame([surr.get("benefit_summary") or {}]), out / "tables" / "t52_surrogate_benefit_summary.csv", out / "tables" / "t52_surrogate_benefit_summary.tex")
        if not rows.empty:
            fig, ax = plt.subplots(figsize=(6.5, 4.4))
            ax.scatter(rows["surrogate_rank"], rows["true_mean_verified_gain"], color="#315c9b", s=55)
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_xlabel("Surrogate rank")
            ax.set_ylabel("True mean verified gain")
            ax.set_title("T5.2 Surrogate Sorting vs Measured Gain")
            fig.tight_layout()
            fig.savefig(out / "figures" / "t52_surrogate_rank_vs_true_gain.png", dpi=200)
            fig.savefig(out / "figures" / "t52_surrogate_rank_vs_true_gain.pdf")
            plt.close(fig)
        summary["surrogate_benefit"] = {
            "source": safe_rel(surrogate_path),
            "state": surr.get("state"),
            "benefit_summary": surr.get("benefit_summary"),
        }

    t55_path = REPORTS / "t5-5-defensive-outputs-r1-after-m5" / "defensive-outputs.json"
    if t55_path.exists():
        t55 = read_json(t55_path)
        taxonomy = pd.DataFrame(t55.get("taxonomy_rows") or [])
        sensitivity = pd.DataFrame(t55.get("sensitivity_rows") or [])
        write_df(taxonomy, out / "tables" / "t55_failure_taxonomy_alignment.csv", out / "tables" / "t55_failure_taxonomy_alignment.tex")
        write_df(sensitivity, out / "tables" / "t55_tau_af_sensitivity.csv", out / "tables" / "t55_tau_af_sensitivity.tex")
        if not sensitivity.empty:
            fig, ax = plt.subplots(figsize=(6.4, 4.0))
            ax.plot(sensitivity["tau_multiplier"], sensitivity["static_degradation_interception_rate"], marker="o", label="Static degradation intercepted")
            ax.plot(sensitivity["tau_multiplier"], sensitivity["legitimate_accept_rate"], marker="o", label="Legitimate accept retained")
            ax.set_xlabel("tau_AF multiplier")
            ax.set_ylabel("Rate")
            ax.set_ylim(-0.05, 1.05)
            ax.set_title("T5.5 tau_AF Sensitivity")
            ax.legend()
            ax.grid(True, alpha=0.25)
            fig.tight_layout()
            fig.savefig(out / "figures" / "t55_tau_af_sensitivity.png", dpi=200)
            fig.savefig(out / "figures" / "t55_tau_af_sensitivity.pdf")
            plt.close(fig)
        summary["defensive_outputs"] = {
            "source": safe_rel(t55_path),
            "state": t55.get("state"),
            "taxonomy_row_count": t55.get("taxonomy_row_count"),
            "sensitivity_row_count": t55.get("sensitivity_row_count"),
            "base_tau_af": t55.get("base_tau_af"),
        }
    return summary


def export_visual_assets(out: Path) -> pd.DataFrame:
    rows = []
    for path in sorted((ROOT / "results" / "m1" / "horizon-probe").glob("**/*.png")):
        rows.append({"asset_type": "m1_horizon_probe_png", "exists": path.exists(), "path": safe_rel(path), "note": "existing raw-frame evidence PNG"})
    cloth_root = ROOT / "results" / "recovery" / "cloth_move_continue_11k_to_100k-r1" / "acwm-run" / "eval_videos"
    for path in sorted(cloth_root.glob("**/*.mp4"))[:120]:
        rows.append({"asset_type": "cloth_recovery_eval_mp4", "exists": path.exists(), "path": safe_rel(path), "note": "cloth recovery branch video; not formal M4 side-by-side"})
    formal_roots = _formal_visualization_roots(REPORTS)
    for root in formal_roots:
        for path in sorted((root / "paired_videos").glob("*.mp4")):
            rows.append({"asset_type": "formal_acwm_gt_baseline_candidate_mp4", "exists": path.exists(), "path": safe_rel(path), "note": "retained formal visualization; layout GT|baseline_prediction|candidate_prediction"})
        for path in sorted((root / "baseline_eval").glob("**/*.mp4")):
            rows.append({"asset_type": "formal_acwm_baseline_gt_pred_mp4", "exists": path.exists(), "path": safe_rel(path), "note": "official eval.py --save_videos baseline GT|Pred"})
        for path in sorted((root / "candidate_eval").glob("**/*.mp4")):
            rows.append({"asset_type": "formal_acwm_candidate_gt_pred_mp4", "exists": path.exists(), "path": safe_rel(path), "note": "official eval.py --save_videos candidate GT|Pred"})
    for root in sorted(path for path in REPORTS.glob("acwm-horizon-triptych-*") if path.is_dir()):
        for path in sorted((root / "videos").glob("*.mp4")):
            rows.append(
                {
                    "asset_type": "formal_acwm_long_horizon_gt_baseline_ours_mp4",
                    "exists": path.exists(),
                    "path": safe_rel(path),
                    "note": "paired long-horizon visualization; layout GT|Baseline|Ours",
                }
            )
    for manifest in [
        REPORTS / "m4-formal-campaign-auto-by-diagnosis-8env-train16-s417-r1/envs/robot_arm/evaluation/baseline-horizon-probe-manifest.json",
        REPORTS / "m4-formal-campaign-auto-by-diagnosis-8env-train16-s417-r1/envs/robot_arm/evaluation/candidate-horizon-probe-manifest.json",
        REPORTS / "t4-4-trial-runs-r3-extend-unreached-gpu12-7200/shuffled_prior-s101-pour_water-drift_token_trim-extend-t64/envs/pour_water/evaluation/baseline-horizon-probe-manifest.json",
        REPORTS / "t4-4-trial-runs-r3-extend-unreached-gpu12-7200/shuffled_prior-s101-pour_water-drift_token_trim-extend-t64/envs/pour_water/evaluation/candidate-horizon-probe-manifest.json",
    ]:
        if not manifest.exists():
            continue
        data = read_json(manifest)
        for evidence in data.get("evidence_paths") or []:
            p = Path(evidence)
            rows.append({"asset_type": "formal_m4_evidence_png_reference", "exists": p.exists(), "path": str(p), "note": f"referenced by {safe_rel(manifest)}"})
    df = pd.DataFrame(rows)
    if not df.empty:
        write_df(df, out / "tables" / "visual_asset_manifest.csv", out / "tables" / "visual_asset_manifest.tex")
    return df


def _formal_visualization_roots(report_root: Path) -> list[Path]:
    roots: set[Path] = set()
    for pattern in (
        "acwm-formal-visualization*",
        "acwm-official-gate-*",
        "acwm-autoloop-official-gate-*",
        "acwm-autoloop-confirm-official-gate-*",
    ):
        roots.update(path for path in report_root.glob(pattern) if path.is_dir())
    return sorted(roots)


def write_readme(out: Path, manifest: dict[str, Any]) -> None:
    effects_path = "tables/primitive_effects_by_cell.csv"
    readme = f"""# Paper Export R1

Generated from a frozen VerdiWM evidence archive.

## Claim Boundary

This bundle exports existing closed-loop evidence. Every official-gate pass remains in the tables and retained-media registry even when visual strength is pending. `latent_motion_prior` artifacts remain available as audit evidence but are excluded from method-quality claims because their historical implementation did not update trainable model parameters.

Formal side-by-side videos are exported through `scripts/export/acwm_formal_visualization.py` when a retained candidate checkpoint is available. The `visual_asset_manifest.csv` records retained media and remaining missing references.

## Main Outputs

- Primitive effect table: `{effects_path}`
- Official 50-step quality gates: `tables/official_quality_gates.csv`
- Live 8-environment formal coverage: `tables/acwm_8env_live_coverage.csv`
- All official-gate pass/fail records with claim tiers: `tables/official_quality_gates.csv`
- Invalidated primitive records retained for audit: `tables/invalidated_trial_manifest_rows.csv`
- Representative horizon metrics: `tables/representative_horizon_metrics.csv`
- T4.4 prior convergence: `tables/t44_prior_convergence_summary.csv`
- T4.5 verifier matrix: `tables/t45_verifier_confusion_matrix.csv`
- T5.1 adversarial gate audit: `tables/t51_adversarial_gate_audit.csv`
- T5.2 surrogate sorting benefit: `tables/t52_surrogate_benefit_summary.csv`
- T5.5 taxonomy and tau sweep: `tables/t55_failure_taxonomy_alignment.csv`, `tables/t55_tau_af_sensitivity.csv`
- M5 status/cost: `tables/m5_claim_status.csv`, `tables/m5_cost_summary.csv`
- Figures: `figures/*.png` and `figures/*.pdf`

## Status Summary

- Settled trials: `{manifest.get('status', {}).get('settled_trials')}`
- M5 export complete: `{manifest.get('status', {}).get('m5_export_complete')}`
- Formal M4 mp4 count: `{manifest.get('media', {}).get('formal_m4_mp4_count')}`
- Formal visualization mp4 count: `{manifest.get('media', {}).get('formal_visualization_mp4_count')}`
- Long-horizon triptych mp4 count: `{manifest.get('media', {}).get('long_horizon_triptych_mp4_count')}`
- Existing cloth recovery mp4 count: `{manifest.get('media', {}).get('cloth_recovery_mp4_count')}`
- Tier-A confirmed official-gate rows: `{manifest.get('table_counts', {}).get('official_quality_gate_confirmed_pass_rows')}`
- Tier-B pending-confirmation official-gate rows: `{manifest.get('table_counts', {}).get('official_quality_gate_pending_confirmation_pass_rows')}`
- Formally confirmed ACWM environments: `{manifest.get('status', {}).get('formally_confirmed_environment_count')}/{manifest.get('status', {}).get('environment_count')}`

## Evidence Use

Use `official_quality_gates.csv` for the quantitative fallback set. Use the retained side-by-side videos only for qualitative showcase claims after human inspection; a gate pass alone does not imply a visually obvious improvement.

"""
    (out / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=REPORTS / "paper-export-r1")
    args = parser.parse_args()

    out = args.output
    if out.exists():
        shutil.rmtree(out)
    (out / "tables").mkdir(parents=True)
    (out / "figures").mkdir(parents=True)

    trials = load_trial_rows()
    effects, top = export_primitive_tables(trials, out)
    plot_primitive_effects(effects, out)
    official_gates = load_official_gate_rows()
    export_official_quality_gates(official_gates, out)
    environment_coverage, environment_coverage_report = load_environment_coverage()
    export_environment_coverage(environment_coverage, environment_coverage_report, out)
    horizon = export_horizon_cases(out)
    t44_summary = export_t44(out)
    t45 = export_t45(out)
    aux = export_auxiliary_tables(out)
    media = export_visual_assets(out)

    m4 = read_json(REPORTS / "m4-completion-gate-v35-after-s417-r1" / "manifest.json")
    m5_path = REPORTS / "m5-consolidation-r31-after-m4-t52-t55-ready-clean" / "consolidation-report.json"
    if not m5_path.exists():
        m5_path = REPORTS / "m5-consolidation-r30" / "consolidation-report.json"
    m5 = read_json(m5_path) if m5_path.exists() else {}

    formal_m4_campaign_mp4_count = len(list(REPORTS.glob("*m4-formal-campaign*/**/*.mp4")))
    formal_visualization_mp4_count = sum(
        len(list(root.glob("**/*.mp4"))) for root in _formal_visualization_roots(REPORTS)
    )
    long_horizon_triptych_mp4_count = len(
        list(REPORTS.glob("acwm-horizon-triptych-*/videos/*.mp4"))
    )
    formal_m4_mp4_count = formal_m4_campaign_mp4_count + formal_visualization_mp4_count
    cloth_mp4_count = len(list((ROOT / "results" / "recovery" / "cloth_move_continue_11k_to_100k-r1" / "acwm-run" / "eval_videos").glob("**/*.mp4")))
    output_files = sorted(safe_rel(p) for p in out.glob("**/*") if p.is_file())
    manifest = {
        "artifact_type": "wmloop-paper-export-bundle",
        "schema_version": 1,
        "state": "ready",
        "output_root": safe_rel(out),
        "status": {
            "m4_completion_state": m4.get("state"),
            "m4_completion_allowed": m4.get("m4_completion_allowed"),
            "settled_trial_target": m4.get("settled_trial_target"),
            "settled_trials": (m5.get("archive_summary") or {}).get("settled_trials"),
            "m5_state": m5.get("state"),
            "m5_export_complete": m5.get("m5_export_complete"),
            "environment_count": (environment_coverage_report.get("summary") or {}).get(
                "environment_count"
            ),
            "formally_confirmed_environment_count": (
                environment_coverage_report.get("summary") or {}
            ).get("formally_confirmed_environment_count"),
            "environment_coverage_complete": (environment_coverage_report.get("summary") or {}).get(
                "coverage_complete"
            ),
        },
        "table_counts": {
            "trial_rows": int(len(trials)),
            "primitive_cells": int(len(effects)),
            "horizon_rows": int(len(horizon)),
            "t44_arm_rows": int(len(t44_summary)),
            "t45_matrix_rows": int(len(t45)),
            "visual_asset_rows": int(len(media)),
            "official_quality_gate_rows": int(len(official_gates)),
            "official_quality_gate_pass_rows": int(official_gates["official_pass"].sum()) if not official_gates.empty else 0,
            "official_quality_gate_claimable_pass_rows": int(
                (official_gates["official_pass"] & ~official_gates["method_invalidated"]).sum()
            ) if not official_gates.empty else 0,
            "official_quality_gate_confirmed_pass_rows": int(
                (
                    official_gates["official_pass"]
                    & official_gates["claim_tier"].astype(str).str.startswith("A_")
                ).sum()
            ) if not official_gates.empty else 0,
            "official_quality_gate_pending_confirmation_pass_rows": int(
                (
                    official_gates["official_pass"]
                    & official_gates["claim_tier"].astype(str).str.startswith("B_")
                ).sum()
            ) if not official_gates.empty else 0,
            "official_quality_gate_invalidated_audit_pass_rows": int(
                (official_gates["official_pass"] & official_gates["method_invalidated"]).sum()
            ) if not official_gates.empty else 0,
            "environment_coverage_rows": int(len(environment_coverage)),
        },
        "media": {
            "formal_m4_mp4_count": formal_m4_mp4_count,
            "formal_m4_campaign_mp4_count": formal_m4_campaign_mp4_count,
            "formal_visualization_mp4_count": formal_visualization_mp4_count,
            "long_horizon_triptych_mp4_count": long_horizon_triptych_mp4_count,
            "cloth_recovery_mp4_count": cloth_mp4_count,
            "formal_m4_side_by_side_video_ready": formal_m4_mp4_count > 0,
        },
        "sources": {
            "m4_completion_gate": safe_rel(REPORTS / "m4-completion-gate-v35-after-s417-r1" / "manifest.json"),
            "m5_consolidation": safe_rel(m5_path),
            "effective_primitive_inventory": safe_rel(REPORTS / "effective-primitive-inventory-r1" / "effective-primitive-inventory.md"),
            "t44_prior_convergence": safe_rel(REPORTS / "t4-4-prior-convergence-progress-r15-r3-extend-unreached-negative-settled" / "prior-convergence.json"),
            "t45_verifier_meta_validation": safe_rel(REPORTS / "t4-5-verifier-meta-validation-progress-r25-full-archive-plus-s417-r1" / "verifier-meta-validation.json"),
            "adversarial_gate_audit": safe_rel(REPORTS / "adversarial-gate-audit-r2" / "adversarial-gate-audit.json"),
            "surrogate_benefit": safe_rel(REPORTS / "surrogate-benefit-r1-after-m4" / "surrogate-benefit.json"),
            "defensive_outputs": safe_rel(REPORTS / "t5-5-defensive-outputs-r1-after-m5" / "defensive-outputs.json"),
            "environment_coverage_report_root": safe_rel(REPORTS),
        },
        "auxiliary_summaries": aux,
        "output_files": output_files,
        "limitations": [
            "Most formal M4 training/eval trials are 16-step smoke evidence with INCONCLUSIVE verdicts.",
            "Generated frame evidence referenced by selected M4 manifests lived in temporary worktrees and is no longer present.",
            "Surrogate outputs are proposal-sorting evidence only and are not verifier inputs.",
        ],
    }
    write_json(out / "manifest.json", manifest)
    write_readme(out, manifest)
    print(json.dumps({"state": "ready", "output_root": str(out), "files": len(output_files) + 2}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
