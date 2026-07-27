#!/usr/bin/env python3
"""Summarize ACWM screen campaigns without mutating training outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_CAMPAIGN_GLOBS = ("acwm-screen-v*-*", "acwm-gap-*", "acwm-autoloop-screen-*", "acwm-autoloop-confirm-*")
DEFAULT_VISUAL_GLOBS = ("*visual*", "acwm-screen-v*-visual-*", "acwm-autoloop-visual-*")
MIN_POSITIVE_SCREEN_STEPS = 512


class AcwmScreenSummaryError(RuntimeError):
    """Screen summary export failed closed."""


def run_acwm_screen_summary(
    *,
    report_root: Path,
    output_root: Path,
    campaign_globs: Sequence[str] = DEFAULT_CAMPAIGN_GLOBS,
    visual_globs: Sequence[str] = DEFAULT_VISUAL_GLOBS,
    include_running: bool = True,
) -> dict[str, object]:
    """Export read-only summary tables for ACWM screening campaigns."""

    reports = Path(report_root).resolve()
    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise AcwmScreenSummaryError("ACWM_SCREEN_SUMMARY_OUTPUT_EXISTS")
    campaigns = _discover_campaigns(reports, campaign_globs)
    visual_assets = _discover_visual_assets(reports, visual_globs)
    completed_rows: list[dict[str, object]] = []
    running_rows: list[dict[str, object]] = []
    horizon_rows: list[dict[str, object]] = []

    for campaign in campaigns:
        status_records = _status_records(campaign / "status.json")
        status_by_output = {
            str(Path(str(record.get("output_root", ""))).resolve()): record
            for record in status_records
            if isinstance(record.get("output_root"), str)
        }
        finalized_outputs: set[str] = set()
        for manifest_path in sorted((campaign / "envs").glob("*/manifest.json")):
            if manifest_path.parent.name.startswith("."):
                continue
            row = _completed_row(
                report_root=reports,
                campaign=campaign,
                manifest_path=manifest_path,
                status_record=status_by_output.get(str(manifest_path.parent.resolve())),
                visual_assets=visual_assets,
            )
            finalized_outputs.add(str(manifest_path.parent.resolve()))
            completed_rows.append(row)
            horizon_rows.extend(_horizon_rows(row=row, manifest_path=manifest_path))
        if include_running:
            for record in status_records:
                output_value = record.get("output_root")
                if not isinstance(output_value, str):
                    continue
                output_path = Path(output_value).resolve()
                if str(output_path) in finalized_outputs:
                    continue
                running_rows.append(
                    _running_row(
                        report_root=reports,
                        campaign=campaign,
                        record=record,
                        visual_assets=visual_assets,
                    )
                )

    rows = sorted(
        completed_rows + running_rows,
        key=lambda row: (
            str(row.get("campaign_id", "")),
            str(row.get("environment", "")),
            str(row.get("primitive", "")),
        ),
    )
    env_best_rows = _best_by_environment(completed_rows)
    report = {
        "schema_version": 1,
        "artifact_type": "wmloop-acwm-screen-summary",
        "state": "ready",
        "report_root": str(reports),
        "campaign_globs": list(campaign_globs),
        "visual_globs": list(visual_globs),
        "campaign_count": len(campaigns),
        "row_count": len(rows),
        "completed_row_count": len(completed_rows),
        "running_row_count": len(running_rows),
        "positive_screen_count": sum(1 for row in completed_rows if row.get("screen_decision") in {"promote_to_confirmation", "confirmation_candidate"}),
        "negative_screen_count": sum(1 for row in completed_rows if row.get("screen_decision") == "reject_or_revise"),
        "action_gate_fail_count": sum(1 for row in completed_rows if row.get("screen_decision") == "fail_action_gate"),
        "horizon_row_count": len(horizon_rows),
        "best_by_environment_count": len(env_best_rows),
        "limitations": [
            "This exporter is read-only and does not change frozen verdicts or launch training.",
            "Screen decisions are triage labels. Paper-facing claims still require the frozen evaluator, action gate, and official visualization/eval evidence.",
            f"Runs below {MIN_POSITIVE_SCREEN_STEPS} training steps are treated as startup/health checks and cannot be promoted as positive method signals.",
            "Running rows are derived from campaign status and checkpoint filenames; final metrics only appear after the campaign writes its env manifest.",
        ],
    }

    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    try:
        temporary.mkdir(mode=0o700)
        _write_json(temporary / "screen-summary.json", {**report, "rows": rows, "best_by_environment": env_best_rows})
        _write_csv(temporary / "tables" / "screen-trials.csv", rows, SCREEN_FIELDNAMES)
        _write_csv(temporary / "tables" / "horizon-metrics.csv", horizon_rows, HORIZON_FIELDNAMES)
        _write_csv(temporary / "tables" / "best-by-environment.csv", env_best_rows, BEST_FIELDNAMES)
        _write_markdown(
            temporary / "screen-summary.md",
            report=report,
            rows=rows,
            best_by_environment=env_best_rows,
        )
        manifest = {
            **{key: report[key] for key in report if key != "limitations"},
            "limitations": report["limitations"],
            "summary_path": str(destination / "screen-summary.json"),
            "markdown_path": str(destination / "screen-summary.md"),
            "screen_trials_csv": str(destination / "tables" / "screen-trials.csv"),
            "horizon_metrics_csv": str(destination / "tables" / "horizon-metrics.csv"),
            "best_by_environment_csv": str(destination / "tables" / "best-by-environment.csv"),
        }
        _write_json(temporary / "manifest.json", manifest)
        os.replace(temporary, destination)
        return manifest
    except Exception:
        if temporary.is_symlink():
            temporary.unlink()
        elif temporary.exists():
            shutil.rmtree(temporary)
        raise


SCREEN_FIELDNAMES = [
    "campaign_id",
    "environment",
    "primitive",
    "seed",
    "train_steps",
    "state",
    "verdict",
    "screen_decision",
    "primary_metric",
    "delta_primary_metric",
    "baseline_primary_metric",
    "candidate_primary_metric",
    "action_following_enabled",
    "action_following_pass",
    "action_following_observed",
    "gpu_index",
    "pid",
    "latest_checkpoint_step",
    "latest_checkpoint_path",
    "candidate_checkpoint_retained",
    "candidate_checkpoint_sha256",
    "official_visual_asset_count",
    "official_visual_paths",
    "output_root",
    "manifest_path",
    "report_path",
    "evaluation_dir",
    "log_path",
]


HORIZON_FIELDNAMES = [
    "campaign_id",
    "environment",
    "primitive",
    "seed",
    "train_steps",
    "horizon",
    "baseline_psnr",
    "candidate_psnr",
    "delta_psnr",
    "baseline_ssim",
    "candidate_ssim",
    "delta_ssim",
    "baseline_masked_mse",
    "candidate_masked_mse",
    "delta_masked_mse",
    "baseline_metrics_path",
    "candidate_metrics_path",
]


BEST_FIELDNAMES = [
    "environment",
    "primitive",
    "campaign_id",
    "seed",
    "train_steps",
    "screen_decision",
    "delta_primary_metric",
    "action_following_pass",
    "manifest_path",
    "official_visual_asset_count",
    "official_visual_paths",
]


def _discover_campaigns(report_root: Path, patterns: Sequence[str]) -> list[Path]:
    campaigns: dict[Path, None] = {}
    for pattern in patterns:
        for path in report_root.glob(pattern):
            if path.is_dir() and ((path / "status.json").is_file() or (path / "envs").is_dir()):
                campaigns[path.resolve()] = None
    return sorted(campaigns)


def _discover_visual_assets(report_root: Path, patterns: Sequence[str]) -> list[dict[str, object]]:
    roots: dict[Path, None] = {}
    for pattern in patterns:
        for path in report_root.glob(pattern):
            if path.is_dir():
                roots[path.resolve()] = None
    assets: list[dict[str, object]] = []
    for root in sorted(roots):
        files = [
            path
            for path in sorted(root.rglob("*"))
            if path.is_file() and path.suffix.lower() in {".mp4", ".gif", ".png", ".jpg", ".jpeg", ".csv", ".json"}
        ]
        if not files:
            continue
        assets.append(
            {
                "root": str(root),
                "name": root.name,
                "paths": [str(path) for path in files],
            }
        )
    return assets


def _status_records(status_path: Path) -> list[dict[str, object]]:
    if not status_path.is_file():
        return []
    payload = _load_json(status_path)
    records = payload.get("records")
    if not isinstance(records, list):
        return []
    return [dict(record) for record in records if isinstance(record, Mapping)]


def _completed_row(
    *,
    report_root: Path,
    campaign: Path,
    manifest_path: Path,
    status_record: Mapping[str, object] | None,
    visual_assets: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    manifest = _load_json(manifest_path)
    report = _load_optional_json(Path(str(manifest.get("report_path", ""))))
    proposal_id = str(manifest.get("proposal_id") or report.get("proposal_id") or "")
    primitive = _infer_primitive(proposal_id)
    seed = _first_non_empty(manifest.get("seed"), _parse_seed(proposal_id), _parse_seed(campaign.name))
    primary_metric = str(manifest.get("primary_metric") or report.get("primary_metric") or "")
    delta = _metric_value(manifest.get("delta_m_ver"), primary_metric)
    evaluation = report.get("receipt", {})
    if isinstance(evaluation, Mapping):
        evaluation = evaluation.get("evaluation", {})
    if not isinstance(evaluation, Mapping):
        evaluation = {}
    action_gate = manifest.get("action_following_gate")
    if not isinstance(action_gate, Mapping):
        action_gate = {}
    train_steps = _first_non_empty(
        _command_value(status_record, "--train-steps"),
        _parse_train_steps(campaign.name),
    )
    env = str(manifest.get("environment") or manifest_path.parent.name)
    matched_visuals = _match_visual_paths(
        visual_assets,
        environment=env,
        primitive=primitive,
        seed=seed,
        train_steps=train_steps,
    )
    latest_checkpoint = _latest_checkpoint(campaign=campaign, environment=env)
    return {
        "campaign_id": campaign.name,
        "environment": env,
        "primitive": primitive,
        "seed": seed,
        "train_steps": train_steps,
        "state": manifest.get("state"),
        "verdict": manifest.get("verdict"),
        "screen_decision": _screen_decision(
            state=manifest.get("state"),
            delta=delta,
            action_gate=action_gate,
            train_steps=train_steps,
        ),
        "primary_metric": primary_metric,
        "delta_primary_metric": delta,
        "baseline_primary_metric": _finite_or_none(evaluation.get("baseline_primary_metric")),
        "candidate_primary_metric": _finite_or_none(evaluation.get("candidate_primary_metric")),
        "action_following_enabled": action_gate.get("enabled"),
        "action_following_pass": action_gate.get("pass"),
        "action_following_observed": _finite_or_none(action_gate.get("observed")),
        "gpu_index": _command_value(status_record, "--gpu-index") if status_record is not None else "",
        "pid": status_record.get("pid") if status_record is not None else "",
        "latest_checkpoint_step": latest_checkpoint.get("step"),
        "latest_checkpoint_path": _safe_relative(report_root, latest_checkpoint.get("path")),
        "candidate_checkpoint_retained": manifest.get("candidate_checkpoint_retained"),
        "candidate_checkpoint_sha256": manifest.get("candidate_checkpoint_sha256"),
        "official_visual_asset_count": len(matched_visuals),
        "official_visual_paths": ";".join(_safe_relative(report_root, path) for path in matched_visuals),
        "output_root": _safe_relative(report_root, str(manifest_path.parent)),
        "manifest_path": _safe_relative(report_root, str(manifest_path)),
        "report_path": _safe_relative(report_root, manifest.get("report_path")),
        "evaluation_dir": _safe_relative(report_root, manifest.get("evaluation_dir")),
        "log_path": _safe_relative(report_root, status_record.get("log_path") if status_record is not None else ""),
    }


def _running_row(
    *,
    report_root: Path,
    campaign: Path,
    record: Mapping[str, object],
    visual_assets: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    env = str(record.get("environment") or "")
    primitive = str(record.get("proposal_primitive") or "")
    seed = _first_non_empty(_command_value(record, "--trial-seed"), _parse_seed(campaign.name))
    train_steps = _first_non_empty(_command_value(record, "--train-steps"), _parse_train_steps(campaign.name))
    matched_visuals = _match_visual_paths(
        visual_assets,
        environment=env,
        primitive=primitive,
        seed=seed,
        train_steps=train_steps,
    )
    latest_checkpoint = _latest_checkpoint(campaign=campaign, environment=env)
    return {
        "campaign_id": campaign.name,
        "environment": env,
        "primitive": primitive,
        "seed": seed,
        "train_steps": train_steps,
        "state": record.get("state") or "running",
        "verdict": "",
        "screen_decision": "running",
        "primary_metric": "",
        "delta_primary_metric": "",
        "baseline_primary_metric": "",
        "candidate_primary_metric": "",
        "action_following_enabled": "",
        "action_following_pass": "",
        "action_following_observed": "",
        "gpu_index": record.get("gpu_index"),
        "pid": record.get("pid"),
        "latest_checkpoint_step": latest_checkpoint.get("step"),
        "latest_checkpoint_path": _safe_relative(report_root, latest_checkpoint.get("path")),
        "candidate_checkpoint_retained": "",
        "candidate_checkpoint_sha256": "",
        "official_visual_asset_count": len(matched_visuals),
        "official_visual_paths": ";".join(_safe_relative(report_root, path) for path in matched_visuals),
        "output_root": _safe_relative(report_root, record.get("output_root")),
        "manifest_path": "",
        "report_path": "",
        "evaluation_dir": "",
        "log_path": _safe_relative(report_root, record.get("log_path")),
    }


def _horizon_rows(*, row: Mapping[str, object], manifest_path: Path) -> list[dict[str, object]]:
    evaluation_dir_value = row.get("evaluation_dir")
    if not isinstance(evaluation_dir_value, str) or not evaluation_dir_value:
        return []
    report_root = _common_report_root(manifest_path, evaluation_dir_value)
    evaluation_dir = Path(evaluation_dir_value)
    if not evaluation_dir.is_absolute():
        evaluation_dir = report_root / evaluation_dir
    baseline_path = evaluation_dir / "baseline-horizon-probe-metrics.json"
    candidate_path = evaluation_dir / "candidate-horizon-probe-metrics.json"
    if not baseline_path.is_file() or not candidate_path.is_file():
        return []
    baseline = _horizon_metrics(_load_json(baseline_path))
    candidate = _horizon_metrics(_load_json(candidate_path))
    rows: list[dict[str, object]] = []
    for horizon in sorted(set(baseline) & set(candidate), key=lambda value: int(value)):
        base = baseline[horizon]
        cand = candidate[horizon]
        rows.append(
            {
                "campaign_id": row.get("campaign_id"),
                "environment": row.get("environment"),
                "primitive": row.get("primitive"),
                "seed": row.get("seed"),
                "train_steps": row.get("train_steps"),
                "horizon": int(horizon),
                "baseline_psnr": _finite_or_none(base.get("psnr")),
                "candidate_psnr": _finite_or_none(cand.get("psnr")),
                "delta_psnr": _delta(cand.get("psnr"), base.get("psnr")),
                "baseline_ssim": _finite_or_none(base.get("ssim")),
                "candidate_ssim": _finite_or_none(cand.get("ssim")),
                "delta_ssim": _delta(cand.get("ssim"), base.get("ssim")),
                "baseline_masked_mse": _finite_or_none(base.get("masked_mse")),
                "candidate_masked_mse": _finite_or_none(cand.get("masked_mse")),
                "delta_masked_mse": _delta(cand.get("masked_mse"), base.get("masked_mse")),
                "baseline_metrics_path": str(baseline_path),
                "candidate_metrics_path": str(candidate_path),
            }
        )
    return rows


def _common_report_root(manifest_path: Path, evaluation_dir_value: str) -> Path:
    if Path(evaluation_dir_value).is_absolute():
        return Path("/")
    marker = "results/reports/"
    manifest_text = str(manifest_path.resolve())
    if marker in manifest_text:
        return Path(manifest_text.split(marker, 1)[0]) / marker.rstrip("/")
    return manifest_path.parents[3]


def _horizon_metrics(payload: Mapping[str, object]) -> Mapping[str, Mapping[str, object]]:
    aggregate = payload.get("aggregate")
    if not isinstance(aggregate, Mapping):
        return {}
    metrics = aggregate.get("horizon_metrics")
    if not isinstance(metrics, Mapping):
        return {}
    return {str(key): value for key, value in metrics.items() if isinstance(value, Mapping)}


def _best_by_environment(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    best: dict[str, Mapping[str, object]] = {}
    for row in rows:
        env = str(row.get("environment") or "")
        delta = _finite_or_none(row.get("delta_primary_metric"))
        steps = _int_or_none(row.get("train_steps"))
        if not env or delta is None or steps is None or steps < MIN_POSITIVE_SCREEN_STEPS:
            continue
        current = best.get(env)
        if current is None or float(delta) > float(current.get("delta_primary_metric", float("-inf"))):
            best[env] = row
    return [
        {name: row.get(name, "") for name in BEST_FIELDNAMES}
        for _, row in sorted(best.items(), key=lambda item: item[0])
    ]


def _screen_decision(
    *,
    state: object,
    delta: float | None,
    action_gate: Mapping[str, object],
    train_steps: object,
) -> str:
    if state != "ready":
        return str(state or "not_ready")
    steps = _int_or_none(train_steps)
    if steps is None or steps < MIN_POSITIVE_SCREEN_STEPS:
        return "below_min_screen_budget"
    if action_gate.get("enabled") is True and action_gate.get("pass") is not True:
        return "fail_action_gate"
    if delta is None:
        return "metric_unavailable"
    if delta < 0.0:
        return "reject_or_revise"
    if steps is not None and steps >= 1000:
        return "confirmation_candidate"
    return "promote_to_confirmation"


def _metric_value(raw: object, primary_metric: str) -> float | None:
    if not isinstance(raw, Mapping):
        return None
    candidates = [primary_metric, "ladder_auc_psnr_envmax", "auc_psnr_16_64", "auc_psnr_envmax"]
    candidates.extend(str(key) for key in raw)
    for key in candidates:
        value = _finite_or_none(raw.get(key))
        if value is not None:
            return value
    return None


def _latest_checkpoint(*, campaign: Path, environment: str) -> dict[str, object]:
    candidates = []
    for path in campaign.glob("envs/.*.tmp/training-run/checkpoints/**/checkpoint_*.pt"):
        if environment and environment not in str(path):
            continue
        step = _checkpoint_step_from_name(path.name)
        candidates.append((path.stat().st_mtime, step or -1, path))
    if not candidates:
        for path in campaign.glob("envs/*/retained_training/latest.pt"):
            if environment and environment not in str(path):
                continue
            candidates.append((path.stat().st_mtime, None, path))
    if not candidates:
        return {"step": "", "path": ""}
    _, step, path = max(candidates, key=lambda item: (item[0], item[1] if item[1] is not None else -1))
    return {"step": "" if step is None or step < 0 else step, "path": str(path)}


def _checkpoint_step_from_name(name: str) -> int | None:
    match = re.search(r"checkpoint_(\d+)\.pt$", name)
    if match is None:
        return None
    return int(match.group(1))


def _match_visual_paths(
    assets: Sequence[Mapping[str, object]],
    *,
    environment: str,
    primitive: str,
    seed: object,
    train_steps: object,
) -> list[str]:
    matched: list[str] = []
    seed_token = f"s{seed}" if seed not in {"", None} else ""
    train_token = f"t{train_steps}" if train_steps not in {"", None} else ""
    for asset in assets:
        name = str(asset.get("name") or "")
        if environment and environment not in name:
            continue
        if primitive and primitive not in name:
            continue
        if seed_token and seed_token not in name:
            continue
        has_step_token = re.search(r"(?:^|[-_])t\d+(?:[-_]|$)", name) is not None
        if train_token and has_step_token and train_token not in name:
            continue
        paths = asset.get("paths")
        if isinstance(paths, list):
            matched.extend(str(path) for path in paths)
    return sorted(set(matched))


def _infer_primitive(proposal_id: str) -> str:
    match = re.search(r"training-eval-smoke-(.+?)-unlabeled", proposal_id)
    if match is not None:
        return match.group(1)
    parts = proposal_id.split("-")
    if len(parts) >= 2:
        return parts[-2]
    return "unknown"


def _parse_seed(text: str) -> int | str:
    match = re.search(r"(?:^|[-_])s(\d+)(?:[-_]|$)", text)
    return int(match.group(1)) if match is not None else ""


def _parse_train_steps(text: str) -> int | str:
    match = re.search(r"(?:^|[-_])t(\d+)(?:[-_]|$)", text)
    if match is not None:
        return int(match.group(1))
    match = re.search(r"(?:^|[-_])(\d+)k(?:[-_]|$)", text)
    if match is not None:
        return int(match.group(1)) * 1000
    return ""


def _command_value(record: Mapping[str, object] | None, flag: str) -> object:
    if record is None:
        return ""
    command = record.get("command")
    if not isinstance(command, list):
        return ""
    for index, value in enumerate(command):
        if value == flag and index + 1 < len(command):
            return command[index + 1]
    return ""


def _first_non_empty(*values: object) -> object:
    for value in values:
        if value not in {"", None}:
            parsed = _int_or_none(value)
            return parsed if parsed is not None else value
    return ""


def _int_or_none(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value)
    return None


def _finite_or_none(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _delta(candidate: object, baseline: object) -> float | None:
    cand = _finite_or_none(candidate)
    base = _finite_or_none(baseline)
    if cand is None or base is None:
        return None
    return cand - base


def _safe_relative(root: Path, value: object) -> str:
    if not isinstance(value, str) or not value:
        return ""
    path = Path(value)
    try:
        return str(path.resolve().relative_to(root))
    except (OSError, ValueError):
        return str(path)


def _load_optional_json(path: Path) -> dict[str, object]:
    if not str(path) or not path.is_file():
        return {}
    try:
        return _load_json(path)
    except (OSError, json.JSONDecodeError):
        return {}


def _load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise AcwmScreenSummaryError(f"ACWM_SCREEN_SUMMARY_JSON_NOT_OBJECT:{path}")
    return payload


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")


def _write_csv(path: Path, rows: Iterable[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _write_markdown(
    path: Path,
    *,
    report: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
    best_by_environment: Sequence[Mapping[str, object]],
) -> None:
    lines = [
        "# ACWM Screen Summary",
        "",
        f"State: `{report['state']}`",
        f"Campaign count: `{report['campaign_count']}`",
        f"Completed rows: `{report['completed_row_count']}`",
        f"Running rows: `{report['running_row_count']}`",
        f"Positive screen count: `{report['positive_screen_count']}`",
        f"Negative screen count: `{report['negative_screen_count']}`",
        "",
        "## Best By Environment",
        "",
        "| Env | Primitive | Steps | Decision | Delta | Visuals | Manifest |",
        "|---|---|---:|---|---:|---:|---|",
    ]
    for row in best_by_environment:
        lines.append(
            "| {environment} | {primitive} | {train_steps} | `{screen_decision}` | {delta_primary_metric} | {official_visual_asset_count} | {manifest_path} |".format(
                **{key: row.get(key, "") for key in BEST_FIELDNAMES}
            )
        )
    lines.extend(
        [
            "",
            "## All Rows",
            "",
            "| Campaign | Env | Primitive | Steps | State | Decision | Delta | Latest ckpt |",
            "|---|---|---|---:|---|---|---:|---|",
        ]
    )
    for row in rows:
        lines.append(
            "| {campaign_id} | {environment} | {primitive} | {train_steps} | `{state}` | `{screen_decision}` | {delta_primary_metric} | {latest_checkpoint_step} |".format(
                **{name: row.get(name, "") for name in SCREEN_FIELDNAMES}
            )
        )
    lines.append("")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", nargs="?")
    parser.add_argument("--report-root", type=Path, default=Path("results/reports"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--campaign-glob", action="append", dest="campaign_globs")
    parser.add_argument("--visual-glob", action="append", dest="visual_globs")
    parser.add_argument("--no-running", action="store_true")
    args = parser.parse_args(argv)
    manifest = run_acwm_screen_summary(
        report_root=args.report_root,
        output_root=args.output_root,
        campaign_globs=tuple(args.campaign_globs or DEFAULT_CAMPAIGN_GLOBS),
        visual_globs=tuple(args.visual_globs or DEFAULT_VISUAL_GLOBS),
        include_running=not args.no_running,
    )
    print(json.dumps(manifest, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
