#!/usr/bin/env python3
"""Aggregate multi-seed WAN2.2-DROID confirmations into a reusable receipt."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


METRICS = (
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "photometric_smoothness",
)
METRICS_RECEIPT = "worldarena_metrics_receipt_subject_consistency_background_consistency_motion_smoothness_photometric_smoothness.json"


def _panel_members(root: Path) -> list[tuple[Path, dict[str, Any]]]:
    """Return the frozen panel members for a seed, rejecting silent fallback."""

    panel_path = root / "validation_panel.json"
    if not panel_path.is_file():
        training = json.loads((root / "training_receipt.json").read_text(encoding="utf-8"))
        return [(root, {"sample_id": training.get("sample_id"), "episode_id": str(training.get("sample_id", "")).split(":", 1)[0]})]
    panel = json.loads(panel_path.read_text(encoding="utf-8"))
    rows = panel.get("rows")
    if panel.get("state") != "frozen" or not isinstance(rows, list) or not rows:
        raise ValueError(f"WAN22_CONFIRMATION_VALIDATION_PANEL_INVALID:{root}")
    members: list[tuple[Path, dict[str, Any]]] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or not row.get("run_root"):
            raise ValueError(f"WAN22_CONFIRMATION_VALIDATION_PANEL_ROW_INVALID:{root}")
        sample_id = str(row.get("sample_id") or "")
        episode_id = str(row.get("episode_id") or "")
        if not sample_id or not episode_id or episode_id in seen:
            raise ValueError(f"WAN22_CONFIRMATION_VALIDATION_PANEL_IDENTITY_INVALID:{root}")
        seen.add(episode_id)
        members.append((Path(str(row["run_root"])).expanduser().resolve(strict=True), row))
    return members


def summarize(
    run_roots: list[Path],
    *,
    candidate_id: str,
    visual_bifurcation_range: float = 0.4,
    visual_floor: float = 0.1,
    photometric_floor: float = 0.01,
    capability_gaps: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if len(run_roots) < 2:
        raise ValueError("WAN22_CONFIRMATION_REQUIRES_MULTIPLE_SEEDS")
    rows: list[dict[str, Any]] = []
    panel_rows: list[dict[str, Any]] = []
    configurations: list[dict[str, Any]] = []
    for raw_root in run_roots:
        root = raw_root.expanduser().resolve(strict=True)
        training_path = root / "training_receipt.json"
        training = json.loads(training_path.read_text(encoding="utf-8"))
        configuration = {
            key: training.get(key)
            for key in ("conditioning_mode", "history_decay", "anchor_policy", "anchor_refresh_strength", "branch_count", "branch_selection", "branch_reference_weight", "horizon_frames", "chunk_frames", "steps")
        }
        configurations.append(configuration)
        members = _panel_members(root)
        seed_panel_values: list[dict[str, Any]] = []
        for member_root, panel_row in members:
            metrics_path = member_root / METRICS_RECEIPT
            evaluator = json.loads(metrics_path.read_text(encoding="utf-8"))
            if evaluator.get("state") != "evaluated_partial" or int(evaluator.get("returncode", -1)) != 0:
                raise ValueError(f"WAN22_CONFIRMATION_EVALUATOR_INVALID:{member_root}")
            values = {
                metric: float(evaluator["metrics"][metric]["per_video"][0]["video_results_normalized"])
                for metric in METRICS
            }
            sample_id = str(panel_row.get("sample_id") or evaluator.get("video", {}).get("sample_id") or "")
            episode_id = str(panel_row.get("episode_id") or evaluator.get("video", {}).get("episode_id") or "")
            seed_panel_values.append({"sample_id": sample_id, "episode_id": episode_id, "metrics": values, "metrics_receipt": str(metrics_path)})
            panel_rows.append({"seed": int(training["seed"]), **seed_panel_values[-1]})
        primary = seed_panel_values[0]
        rows.append({"seed": int(training["seed"]), "run_root": str(root), "gpu_hours": float(training["gpu_hours"]), "metrics": primary["metrics"], "validation_panel": seed_panel_values})
    panel_identity = sorted({row["sample_id"] for row in panel_rows})
    expected_panel_size = max((len(row.get("validation_panel", [])) for row in rows), default=1)
    if any(len(row.get("validation_panel", [])) != expected_panel_size for row in rows):
        raise ValueError("WAN22_CONFIRMATION_VALIDATION_PANEL_SIZE_MISMATCH")
    per_panel: dict[str, dict[str, Any]] = {}
    for sample_id in panel_identity:
        members = [row for row in panel_rows if row["sample_id"] == sample_id]
        if len(members) != len(rows):
            raise ValueError("WAN22_CONFIRMATION_VALIDATION_PANEL_INCOMPLETE")
        per_panel[sample_id] = {
            metric: statistics.mean([float(row["metrics"][metric]) for row in members])
            for metric in METRICS
        }
    if any(configuration != configurations[0] for configuration in configurations[1:]):
        raise ValueError("WAN22_CONFIRMATION_CONFIGURATION_MISMATCH")
    aggregate: dict[str, Any] = {}
    for metric in METRICS:
        values = [row["metrics"][metric] for row in rows]
        aggregate[metric] = {
            "values": values,
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "population_stddev": statistics.pstdev(values),
            "range": max(values) - min(values),
        }
    failures: list[str] = []
    if any(aggregate[metric]["range"] >= visual_bifurcation_range for metric in ("subject_consistency", "background_consistency")):
        failures.append("visual_seed_bifurcation")
    if any(aggregate[metric]["median"] < visual_floor for metric in ("subject_consistency", "background_consistency")):
        failures.append("visual_consistency_floor")
    if aggregate["photometric_smoothness"]["median"] <= photometric_floor:
        failures.append("photometric_smoothness_floor")
    return {
        "schema_version": 1,
        "artifact_type": "verdiwm-wan22-droid-multiseed-summary",
        "state": "measured",
        "candidate_id": candidate_id,
        "configuration": configurations[0],
        "seed_count": len(rows),
        "validation_episode_count": len(panel_identity),
        "validation_sample_ids": panel_identity,
        "rows": rows,
        "validation_panel": panel_rows,
        "panel_aggregate": per_panel,
        "aggregate": aggregate,
        "failure_signatures": failures,
        "records": [
            {
                "candidate_id": candidate_id,
                "seed": row["seed"],
                "metrics": row["metrics"],
                "failure_signatures": failures,
                "evidence_refs": [str(Path(row["run_root"]) / METRICS_RECEIPT)],
            }
            for row in rows
        ],
        "diagnostic_thresholds": {
            "visual_bifurcation_range": visual_bifurcation_range,
            "visual_floor": visual_floor,
            "photometric_floor": photometric_floor,
        },
        "total_gpu_hours": sum(row["gpu_hours"] for row in rows),
        "capability_gaps": list(capability_gaps or []),
        "claim_boundary": "This receipt aggregates frozen per-seed metrics; promotion still requires the experiment's declared gates and all required evaluator dimensions.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, action="append", required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--visual-bifurcation-range", type=float, default=0.4)
    parser.add_argument("--visual-floor", type=float, default=0.1)
    parser.add_argument("--photometric-floor", type=float, default=0.01)
    parser.add_argument("--capability-gap", action="append", default=[])
    args = parser.parse_args(argv)
    try:
        gaps = []
        for value in args.capability_gap:
            code, separator, detail = value.partition(":")
            gaps.append({"code": code, "detail": detail if separator else code})
        receipt = summarize(
            args.run_root,
            candidate_id=args.candidate_id,
            visual_bifurcation_range=args.visual_bifurcation_range,
            visual_floor=args.visual_floor,
            photometric_floor=args.photometric_floor,
            capability_gaps=gaps,
        )
        destination = args.output.expanduser().resolve()
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        rendered = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
        destination.write_text(rendered, encoding="utf-8")
        print(rendered, end="")
        return 0
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"state": "blocked", "error": str(exc)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
