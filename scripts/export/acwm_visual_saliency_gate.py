#!/usr/bin/env python3
"""Verify candidate-independent sample selection and visible ACWM improvement."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory
from typing import Mapping, Sequence


class AcwmVisualSaliencyGateError(RuntimeError):
    """The visual admission request or its evidence is invalid."""


def run_gate(*, spec_path: Path, output_path: Path) -> dict[str, object]:
    spec_file = _regular_file(spec_path, "SPEC")
    destination = Path(output_path).resolve()
    if destination.exists() or destination.is_symlink():
        raise AcwmVisualSaliencyGateError(f"VISUAL_GATE_OUTPUT_EXISTS:{destination}")

    spec = _load_json(spec_file)
    environment = _required_text(spec, "environment")
    primitive = _required_text(spec, "primitive")
    source_video = _regular_file(spec.get("source_video"), "SOURCE_VIDEO")
    visual_manifest = _regular_file(spec.get("visual_manifest"), "VISUAL_MANIFEST")
    visual = _load_json(visual_manifest)
    if str(visual.get("environment") or "") != environment:
        raise AcwmVisualSaliencyGateError("VISUAL_GATE_ENVIRONMENT_MISMATCH")

    selection = spec.get("selection")
    if not isinstance(selection, Mapping):
        raise AcwmVisualSaliencyGateError("VISUAL_GATE_SELECTION_MISSING")
    unit_field = str(selection.get("unit_field") or "")
    if unit_field not in {"sample_index", "trajectory_index"}:
        raise AcwmVisualSaliencyGateError("VISUAL_GATE_UNIT_FIELD_INVALID")
    unit_value = selection.get("unit_value")
    if unit_value is None:
        raise AcwmVisualSaliencyGateError("VISUAL_GATE_UNIT_VALUE_MISSING")
    required_rank = int(selection.get("required_baseline_rank") or 1)
    if required_rank < 1:
        raise AcwmVisualSaliencyGateError("VISUAL_GATE_REQUIRED_RANK_INVALID")

    window = spec.get("analysis_window") or {}
    if not isinstance(window, Mapping):
        raise AcwmVisualSaliencyGateError("VISUAL_GATE_WINDOW_INVALID")
    start_frame = int(window.get("start_frame") or 0)
    end_value = window.get("end_frame")
    end_frame = None if end_value is None else int(end_value)
    label_height = int(window.get("label_height") or 30)
    if start_frame < 0 or (end_frame is not None and end_frame <= start_frame) or label_height < 0:
        raise AcwmVisualSaliencyGateError("VISUAL_GATE_WINDOW_INVALID")

    thresholds = _thresholds(spec.get("thresholds"))
    pool = _selection_pool(visual, unit_field=unit_field)
    selected = next((item for item in pool if item["unit_value"] == unit_value), None)
    if selected is None:
        raise AcwmVisualSaliencyGateError("VISUAL_GATE_SELECTED_UNIT_NOT_IN_POOL")
    if Path(str(selected["paired_video_path"])).resolve() != source_video:
        raise AcwmVisualSaliencyGateError("VISUAL_GATE_SOURCE_NOT_SELECTED_UNIT")

    ranking: list[dict[str, object]] = []
    for item in pool:
        video = _regular_file(item["paired_video_path"], "POOL_VIDEO")
        metrics = _paired_mse(
            video,
            left_panel=0,
            right_panel=1,
            label_height=label_height,
            start_frame=0,
            end_frame=None,
        )
        ranking.append({
            "unit_value": item["unit_value"],
            "paired_video_path": str(video),
            "paired_video_sha256": _sha256(video),
            "baseline_to_gt_mean_mse": metrics["mean_mse"],
            "frame_count": metrics["frame_count"],
        })
    ranking.sort(key=lambda item: (-float(item["baseline_to_gt_mean_mse"]), str(item["unit_value"])))
    for rank, item in enumerate(ranking, start=1):
        item["baseline_only_rank"] = rank
    selected_rank = next(
        int(item["baseline_only_rank"])
        for item in ranking
        if item["unit_value"] == unit_value
    )

    baseline = _paired_mse(
        source_video,
        left_panel=0,
        right_panel=1,
        label_height=label_height,
        start_frame=start_frame,
        end_frame=end_frame,
    )
    candidate = _paired_mse(
        source_video,
        left_panel=0,
        right_panel=2,
        label_height=label_height,
        start_frame=start_frame,
        end_frame=end_frame,
    )
    separation = _paired_mse(
        source_video,
        left_panel=1,
        right_panel=2,
        label_height=label_height,
        start_frame=start_frame,
        end_frame=end_frame,
    )
    if not (
        baseline["frame_count"] == candidate["frame_count"] == separation["frame_count"]
    ):
        raise AcwmVisualSaliencyGateError("VISUAL_GATE_FRAME_COUNT_MISMATCH")

    baseline_frames = baseline["frame_mse"]
    candidate_frames = candidate["frame_mse"]
    gains = [left - right for left, right in zip(baseline_frames, candidate_frames)]
    mean_gain = sum(gains) / len(gains)
    improved_fraction = sum(value > 0.0 for value in gains) / len(gains)
    separation_rmse = [math.sqrt(max(0.0, value)) for value in separation["frame_mse"]]
    measurements = {
        "frame_count": len(gains),
        "baseline_to_gt_mean_mse": baseline["mean_mse"],
        "candidate_to_gt_mean_mse": candidate["mean_mse"],
        "mean_mse_gain_baseline_minus_candidate": mean_gain,
        "improved_frame_fraction": improved_fraction,
        "mean_candidate_baseline_rmse": sum(separation_rmse) / len(separation_rmse),
        "peak_candidate_baseline_rmse": max(separation_rmse),
    }
    checks = {
        "baseline_only_rank": selected_rank <= required_rank,
        "minimum_frame_count": len(gains) >= int(thresholds["minimum_frame_count"]),
        "mean_mse_gain": mean_gain >= float(thresholds["minimum_mean_mse_gain"]),
        "improved_frame_fraction": improved_fraction >= float(thresholds["minimum_improved_frame_fraction"]),
        "mean_candidate_baseline_rmse": measurements["mean_candidate_baseline_rmse"]
        >= float(thresholds["minimum_mean_candidate_baseline_rmse"]),
        "peak_candidate_baseline_rmse": measurements["peak_candidate_baseline_rmse"]
        >= float(thresholds["minimum_peak_candidate_baseline_rmse"]),
    }
    passed = all(checks.values())
    report = {
        "schema_version": 1,
        "artifact_type": "wmloop-acwm-visual-saliency-gate",
        "state": "pass" if passed else "rejected",
        "pass": passed,
        "environment": environment,
        "primitive": primitive,
        "source_video": str(source_video),
        "source_video_sha256": _sha256(source_video),
        "visual_manifest": str(visual_manifest),
        "visual_manifest_sha256": _sha256(visual_manifest),
        "selection": {
            "rule": "descending baseline-to-GT full-video MSE; candidate output excluded from ranking",
            "unit_field": unit_field,
            "unit_value": unit_value,
            "required_baseline_rank": required_rank,
            "observed_baseline_only_rank": selected_rank,
            "pool_size": len(ranking),
            "ranking": ranking,
        },
        "analysis_window": {
            "start_frame": start_frame,
            "end_frame": end_frame,
            "label_height": label_height,
        },
        "thresholds": thresholds,
        "measurements": measurements,
        "checks": checks,
        "claim_boundary": (
            "This gate admits a project-page visualization only. Aggregate method claims remain governed by "
            "the frozen official quality gate and independent confirmation."
        ),
        "source_spec": str(spec_file),
        "source_spec_sha256": _sha256(spec_file),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _thresholds(value: object) -> dict[str, float | int]:
    if not isinstance(value, Mapping):
        raise AcwmVisualSaliencyGateError("VISUAL_GATE_THRESHOLDS_MISSING")
    required = {
        "minimum_frame_count": int,
        "minimum_mean_mse_gain": float,
        "minimum_improved_frame_fraction": float,
        "minimum_mean_candidate_baseline_rmse": float,
        "minimum_peak_candidate_baseline_rmse": float,
    }
    result: dict[str, float | int] = {}
    for key, cast in required.items():
        if value.get(key) is None:
            raise AcwmVisualSaliencyGateError(f"VISUAL_GATE_THRESHOLD_MISSING:{key}")
        result[key] = cast(value[key])
    if (
        int(result["minimum_frame_count"]) < 2
        or float(result["minimum_mean_mse_gain"]) <= 0.0
        or not 0.0 < float(result["minimum_improved_frame_fraction"]) <= 1.0
        or float(result["minimum_mean_candidate_baseline_rmse"]) <= 0.0
        or float(result["minimum_peak_candidate_baseline_rmse"]) <= 0.0
    ):
        raise AcwmVisualSaliencyGateError("VISUAL_GATE_THRESHOLD_INVALID")
    return result


def _selection_pool(visual: Mapping[str, object], *, unit_field: str) -> list[dict[str, object]]:
    values = visual.get("paired_videos")
    if not isinstance(values, list) or not values:
        raise AcwmVisualSaliencyGateError("VISUAL_GATE_SELECTION_POOL_MISSING")
    pool: list[dict[str, object]] = []
    seen: set[object] = set()
    for value in values:
        if not isinstance(value, Mapping) or value.get(unit_field) is None:
            raise AcwmVisualSaliencyGateError("VISUAL_GATE_POOL_UNIT_MISSING")
        unit = value[unit_field]
        if unit in seen:
            raise AcwmVisualSaliencyGateError("VISUAL_GATE_POOL_UNIT_DUPLICATE")
        seen.add(unit)
        video = value.get("paired_video_path")
        if not video:
            raise AcwmVisualSaliencyGateError("VISUAL_GATE_POOL_VIDEO_MISSING")
        pool.append({"unit_value": unit, "paired_video_path": str(video)})
    return pool


def _paired_mse(
    video: Path,
    *,
    left_panel: int,
    right_panel: int,
    label_height: int,
    start_frame: int,
    end_frame: int | None,
) -> dict[str, object]:
    media = _probe(video)
    width = int(media["width"])
    height = int(media["height"])
    if width % 3 != 0 or label_height >= height:
        raise AcwmVisualSaliencyGateError(f"VISUAL_GATE_VIDEO_SHAPE_INVALID:{video}")
    total_frames = int(media["frame_count"])
    effective_end = total_frames if end_frame is None else end_frame
    if start_frame >= total_frames or effective_end > total_frames:
        raise AcwmVisualSaliencyGateError(f"VISUAL_GATE_WINDOW_OUT_OF_RANGE:{video}")
    panel_width = width // 3
    trim = f"trim=start_frame={start_frame}:end_frame={effective_end},setpts=PTS-STARTPTS"
    with TemporaryDirectory(prefix="verdiwm-visual-gate-") as temporary:
        stats = Path(temporary) / "psnr.log"
        graph = (
            f"[0:v]split=2[a][b];"
            f"[a]crop={panel_width}:{height - label_height}:{left_panel * panel_width}:{label_height},{trim}[left];"
            f"[b]crop={panel_width}:{height - label_height}:{right_panel * panel_width}:{label_height},{trim}[right];"
            f"[left][right]psnr=stats_file={stats}[out]"
        )
        completed = subprocess.run(
            [
                "ffmpeg", "-nostdin", "-v", "error", "-i", str(video),
                "-filter_complex", graph, "-map", "[out]", "-f", "null", "-",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise AcwmVisualSaliencyGateError(
                f"VISUAL_GATE_PSNR_FAILED:{video}:{completed.stderr.strip()}"
            )
        values: list[float] = []
        for line in stats.read_text(encoding="utf-8").splitlines():
            match = re.search(r"(?:^|\s)mse_avg:([0-9.eE+-]+)", line)
            if match:
                values.append(float(match.group(1)))
    expected = effective_end - start_frame
    if len(values) != expected or not values:
        raise AcwmVisualSaliencyGateError(
            f"VISUAL_GATE_PSNR_FRAME_COUNT_INVALID:{video}:{len(values)}:{expected}"
        )
    return {"frame_count": len(values), "frame_mse": values, "mean_mse": sum(values) / len(values)}


def _probe(video: Path) -> dict[str, int]:
    completed = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
            "stream=width,height,nb_frames", "-of", "json", str(video),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise AcwmVisualSaliencyGateError(f"VISUAL_GATE_PROBE_FAILED:{video}")
    payload = json.loads(completed.stdout)
    streams = payload.get("streams") or []
    if not streams:
        raise AcwmVisualSaliencyGateError(f"VISUAL_GATE_VIDEO_STREAM_MISSING:{video}")
    stream = streams[0]
    result = {key: int(stream.get(key) or 0) for key in ("width", "height", "nb_frames")}
    return {"width": result["width"], "height": result["height"], "frame_count": result["nb_frames"]}


def _required_text(payload: Mapping[str, object], key: str) -> str:
    value = str(payload.get(key) or "").strip()
    if not value:
        raise AcwmVisualSaliencyGateError(f"VISUAL_GATE_FIELD_MISSING:{key}")
    return value


def _regular_file(value: object, label: str) -> Path:
    path = Path(str(value or "")).resolve()
    if path.is_symlink() or not path.is_file():
        raise AcwmVisualSaliencyGateError(f"VISUAL_GATE_{label}_INVALID:{path}")
    return path


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AcwmVisualSaliencyGateError(f"VISUAL_GATE_JSON_INVALID:{path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = run_gate(spec_path=args.spec, output_path=args.output)
    except AcwmVisualSaliencyGateError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["pass"] is True else 3


if __name__ == "__main__":
    raise SystemExit(main())
