#!/usr/bin/env python3
"""Export auditable GT/baseline/ours videos for an ACWM project page."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid
from typing import Mapping, Sequence


class ProjectPageBundleError(RuntimeError):
    """A project-page case does not satisfy its declared evidence contract."""


def run_bundle(*, output_root: Path, spec_path: Path) -> dict[str, object]:
    destination = Path(output_root).resolve()
    spec_file = Path(spec_path).resolve()
    if destination.exists() or destination.is_symlink():
        raise ProjectPageBundleError(f"PROJECTPAGE_OUTPUT_EXISTS:{destination}")
    spec = _load_json(spec_file)
    cases = spec.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ProjectPageBundleError("PROJECTPAGE_CASES_MISSING")
    expected = int(spec.get("expected_case_count") or len(cases))
    if len(cases) != expected:
        raise ProjectPageBundleError("PROJECTPAGE_CASE_COUNT_MISMATCH")
    ids = [str(case.get("id") or "") for case in cases if isinstance(case, Mapping)]
    if len(ids) != len(cases) or any(not value for value in ids) or len(set(ids)) != len(ids):
        raise ProjectPageBundleError("PROJECTPAGE_CASE_IDS_INVALID")
    environments = [
        str(case.get("environment") or "") for case in cases if isinstance(case, Mapping)
    ]
    require_distinct_environments = spec.get("require_distinct_environments") is True
    if require_distinct_environments and len(set(environments)) != len(cases):
        raise ProjectPageBundleError("PROJECTPAGE_ENVIRONMENTS_NOT_DISTINCT")
    require_confirmation = spec.get("require_confirmation") is True
    if require_confirmation and any(
        not isinstance(case, Mapping) or case.get("confirmation_manifest") is None
        for case in cases
    ):
        raise ProjectPageBundleError("PROJECTPAGE_CONFIRMATION_REQUIRED")
    require_visual_saliency = spec.get("require_visual_saliency") is True
    if require_visual_saliency and any(
        not isinstance(case, Mapping) or case.get("visual_saliency_manifest") is None
        for case in cases
    ):
        raise ProjectPageBundleError("PROJECTPAGE_VISUAL_SALIENCY_REQUIRED")

    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    try:
        videos = temporary / "videos"
        posters = temporary / "posters"
        videos.mkdir(parents=True, mode=0o700)
        posters.mkdir(parents=True, mode=0o700)
        records = [
            _export_case(case=case, ordinal=index, destination=destination, temporary=temporary)
            for index, case in enumerate(cases, start=1)
        ]
        report = {
            "schema_version": 1,
            "artifact_type": "wmloop-acwm-projectpage-showcase-bundle",
            "state": "ready",
            "case_count": len(records),
            "distinct_environment_count": len(set(environments)),
            "require_distinct_environments": require_distinct_environments,
            "require_confirmation": require_confirmation,
            "require_visual_saliency": require_visual_saliency,
            "confirmed_case_count": sum(
                record.get("independent_confirmation_pass") is True for record in records
            ),
            "visually_admitted_case_count": sum(
                record.get("visual_saliency_pass") is True for record in records
            ),
            "layout": "labeled_GT|baseline_prediction|ours_prediction",
            "selection_policy": spec.get("selection_policy"),
            "claim_boundary": (
                "Official 50-step gates govern aggregate pixel-metric claims. Selected-trajectory event evidence "
                "is explicitly trajectory-local and does not imply an aggregate environment-level event pass. "
                "No GT pixels are injected into baseline or candidate predictions."
            ),
            "records": records,
            "source_spec": str(spec_file),
            "source_spec_sha256": _sha256(spec_file),
        }
        _write_json(temporary / "manifest.json", report)
        _write_csv(temporary / "metrics.csv", records)
        (temporary / "README.md").write_text(_markdown(records), encoding="utf-8")
        os.replace(temporary, destination)
        return report
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _export_case(
    *, case: Mapping[str, object], ordinal: int, destination: Path, temporary: Path
) -> dict[str, object]:
    case_id = str(case["id"])
    environment = str(case.get("environment") or "")
    primitive = str(case.get("primitive") or "")
    evidence_type = str(case.get("evidence_type") or "")
    if not environment or not primitive:
        raise ProjectPageBundleError(f"PROJECTPAGE_IDENTITY_MISSING:{case_id}")
    if evidence_type not in {"aggregate_official_gate_pass", "selected_trajectory_event_positive"}:
        raise ProjectPageBundleError(f"PROJECTPAGE_EVIDENCE_TYPE_INVALID:{case_id}")

    source_video = _regular_file(case.get("source_video"), "SOURCE_VIDEO", case_id)
    gate_path = _regular_file(case.get("official_gate_manifest"), "GATE_MANIFEST", case_id)
    visual_path = _regular_file(case.get("visual_manifest"), "VISUAL_MANIFEST", case_id)
    gate = _load_json(gate_path)
    visual = _load_json(visual_path)
    _validate_gate(gate, environment=environment, primitive=primitive, case_id=case_id)
    _validate_visual_source(visual, source_video=source_video, environment=environment, case_id=case_id)

    start_frame = _optional_int(case.get("start_frame"))
    end_frame = _optional_int(case.get("end_frame"))
    visual_saliency_evidence = None
    visual_saliency_value = case.get("visual_saliency_manifest")
    if visual_saliency_value is not None:
        visual_saliency_path = _regular_file(
            visual_saliency_value, "VISUAL_SALIENCY_MANIFEST", case_id
        )
        visual_saliency_evidence = _validate_visual_saliency(
            _load_json(visual_saliency_path),
            receipt_path=visual_saliency_path,
            source_video=source_video,
            visual_manifest=visual_path,
            environment=environment,
            primitive=primitive,
            sample_index=case.get("sample_index"),
            trajectory_index=case.get("trajectory_index"),
            start_frame=start_frame,
            end_frame=end_frame,
            case_id=case_id,
        )

    confirmation_evidence = None
    confirmation_value = case.get("confirmation_manifest")
    if confirmation_value is not None:
        confirmation_path = _regular_file(
            confirmation_value, "CONFIRMATION_MANIFEST", case_id
        )
        confirmation = _load_json(confirmation_path)
        _validate_gate(
            confirmation,
            environment=environment,
            primitive=primitive,
            case_id=case_id,
            role="CONFIRMATION",
        )
        _validate_confirmation_match(
            gate, confirmation, case_id=case_id
        )
        confirmation_delta = _gate_delta(confirmation)
        confirmation_evidence = {
            "confirmation_manifest": str(confirmation_path),
            "confirmation_manifest_sha256": _sha256(confirmation_path),
            "eval_seed": confirmation.get("eval_seed"),
            "psnr_delta": float(confirmation_delta["psnr"]),
            "ssim_delta": float(confirmation_delta["ssim"]),
            "mse_delta": float(confirmation_delta["mse"]),
            "masked_mse_delta": float(confirmation_delta["masked_mse"]),
        }

    event_evidence = None
    if evidence_type == "selected_trajectory_event_positive":
        event_path = _regular_file(case.get("event_gate_manifest"), "EVENT_GATE_MANIFEST", case_id)
        trajectory_index = int(case.get("trajectory_index") or -1)
        event_evidence = _validate_event(
            _load_json(event_path),
            event_path=event_path,
            environment=environment,
            primitive=primitive,
            trajectory_index=trajectory_index,
            case_id=case_id,
        )

    safe_id = "".join(char if char.isalnum() or char in "-_" else "_" for char in case_id)
    target = temporary / "videos" / f"{ordinal:02d}_{safe_id}_gt_baseline_ours.mp4"
    _transcode(source_video, target, start_frame=start_frame, end_frame=end_frame)
    media = _probe(target)
    requested_poster_frame = _optional_int(case.get("poster_frame"))
    poster_frame = (
        requested_poster_frame
        if requested_poster_frame is not None
        else max(0, int(media["frame_count"]) - 1)
    )
    if poster_frame < 0 or poster_frame >= int(media["frame_count"]):
        raise ProjectPageBundleError(f"PROJECTPAGE_POSTER_FRAME_INVALID:{case_id}")
    poster = temporary / "posters" / f"{ordinal:02d}_{safe_id}.png"
    _poster(target, poster, frame=poster_frame)

    delta = _gate_delta(gate)
    return {
        "id": case_id,
        "environment": environment,
        "primitive": primitive,
        "evidence_type": evidence_type,
        "aggregate_official_gate_pass": True,
        "aggregate_event_gate_pass": event_evidence.get("aggregate_event_gate_pass") if event_evidence else None,
        "selected_trajectory_event_positive": bool(event_evidence),
        "independent_confirmation_pass": confirmation_evidence is not None,
        "confirmation_evidence": confirmation_evidence,
        "visual_saliency_pass": visual_saliency_evidence is not None,
        "visual_saliency_evidence": visual_saliency_evidence,
        "selection_disclosure": str(case.get("selection_disclosure") or ""),
        "sample_index": case.get("sample_index"),
        "trajectory_index": case.get("trajectory_index"),
        "clip_source_frame_start": start_frame,
        "clip_source_frame_end_exclusive": end_frame,
        "poster_frame": poster_frame,
        "psnr_delta": float(delta["psnr"]),
        "ssim_delta": float(delta["ssim"]),
        "mse_delta": float(delta["mse"]),
        "masked_mse_delta": float(delta["masked_mse"]),
        "event_evidence": event_evidence,
        "video_path": str(destination / "videos" / target.name),
        "video_sha256": _sha256(target),
        "video_size_bytes": target.stat().st_size,
        "video_media": media,
        "poster_path": str(destination / "posters" / poster.name),
        "poster_sha256": _sha256(poster),
        "source_video": str(source_video),
        "source_video_sha256": _sha256(source_video),
        "official_gate_manifest": str(gate_path),
        "official_gate_manifest_sha256": _sha256(gate_path),
        "visual_manifest": str(visual_path),
        "visual_manifest_sha256": _sha256(visual_path),
    }


def _validate_gate(
    gate: Mapping[str, object],
    *,
    environment: str,
    primitive: str,
    case_id: str,
    role: str = "GATE",
) -> None:
    quality = gate.get("official_quality_gate")
    if not isinstance(quality, Mapping) or quality.get("pass") is not True:
        raise ProjectPageBundleError(f"PROJECTPAGE_{role}_NOT_PASSING:{case_id}")
    if str(gate.get("environment") or "") != environment or str(gate.get("primitive") or "") != primitive:
        raise ProjectPageBundleError(f"PROJECTPAGE_{role}_IDENTITY_MISMATCH:{case_id}")
    delta = quality.get("delta_candidate_minus_baseline")
    if not isinstance(delta, Mapping) or any(key not in delta for key in ("psnr", "ssim", "mse", "masked_mse")):
        raise ProjectPageBundleError(f"PROJECTPAGE_{role}_METRICS_MISSING:{case_id}")


def _gate_delta(gate: Mapping[str, object]) -> Mapping[str, object]:
    quality = gate["official_quality_gate"]
    assert isinstance(quality, Mapping)
    delta = quality["delta_candidate_minus_baseline"]
    assert isinstance(delta, Mapping)
    return delta


def _validate_confirmation_match(
    gate: Mapping[str, object], confirmation: Mapping[str, object], *, case_id: str
) -> None:
    for field in ("candidate_checkpoint_sha256", "candidate_runtime_sha256"):
        display_value = gate.get(field)
        confirmation_value = confirmation.get(field)
        if display_value is not None or confirmation_value is not None:
            if str(display_value or "") != str(confirmation_value or ""):
                raise ProjectPageBundleError(
                    f"PROJECTPAGE_CONFIRMATION_ARTIFACT_MISMATCH:{case_id}:{field}"
                )
    display_parameters = gate.get("runtime_parameters")
    confirmation_parameters = confirmation.get("runtime_parameters")
    if display_parameters is not None or confirmation_parameters is not None:
        if display_parameters != confirmation_parameters:
            raise ProjectPageBundleError(
                f"PROJECTPAGE_CONFIRMATION_PARAMETERS_MISMATCH:{case_id}"
            )


def _validate_visual_source(
    visual: Mapping[str, object], *, source_video: Path, environment: str, case_id: str
) -> None:
    if str(visual.get("environment") or "") != environment:
        raise ProjectPageBundleError(f"PROJECTPAGE_VISUAL_ENVIRONMENT_MISMATCH:{case_id}")
    layouts = {str(visual.get("layout") or "")}
    paths: set[Path] = set()

    def walk(value: object) -> None:
        if isinstance(value, Mapping):
            layout = value.get("layout")
            if layout is not None:
                layouts.add(str(layout))
            for key, child in value.items():
                if key in {"paired_video_path", "selected_video_path"} and child:
                    paths.add(Path(str(child)).resolve())
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(visual)
    if source_video not in paths:
        raise ProjectPageBundleError(f"PROJECTPAGE_VIDEO_NOT_IN_VISUAL_MANIFEST:{case_id}")
    if "labeled_GT|baseline_prediction|ours_prediction" not in layouts:
        raise ProjectPageBundleError(f"PROJECTPAGE_LAYOUT_NOT_TRIPTYCH:{case_id}")


def _validate_visual_saliency(
    receipt: Mapping[str, object],
    *,
    receipt_path: Path,
    source_video: Path,
    visual_manifest: Path,
    environment: str,
    primitive: str,
    sample_index: object,
    trajectory_index: object,
    start_frame: int | None,
    end_frame: int | None,
    case_id: str,
) -> dict[str, object]:
    if (
        receipt.get("artifact_type") != "wmloop-acwm-visual-saliency-gate"
        or receipt.get("state") != "pass"
        or receipt.get("pass") is not True
    ):
        raise ProjectPageBundleError(f"PROJECTPAGE_VISUAL_SALIENCY_NOT_PASSING:{case_id}")
    if (
        str(receipt.get("environment") or "") != environment
        or str(receipt.get("primitive") or "") != primitive
    ):
        raise ProjectPageBundleError(f"PROJECTPAGE_VISUAL_SALIENCY_IDENTITY_MISMATCH:{case_id}")
    if (
        Path(str(receipt.get("source_video") or "")).resolve() != source_video
        or str(receipt.get("source_video_sha256") or "") != _sha256(source_video)
    ):
        raise ProjectPageBundleError(f"PROJECTPAGE_VISUAL_SALIENCY_SOURCE_MISMATCH:{case_id}")
    if (
        Path(str(receipt.get("visual_manifest") or "")).resolve() != visual_manifest
        or str(receipt.get("visual_manifest_sha256") or "") != _sha256(visual_manifest)
    ):
        raise ProjectPageBundleError(f"PROJECTPAGE_VISUAL_SALIENCY_POOL_MISMATCH:{case_id}")
    selection = receipt.get("selection")
    if not isinstance(selection, Mapping):
        raise ProjectPageBundleError(f"PROJECTPAGE_VISUAL_SALIENCY_SELECTION_MISSING:{case_id}")
    unit_field = str(selection.get("unit_field") or "")
    expected_unit = sample_index if unit_field == "sample_index" else trajectory_index
    if unit_field not in {"sample_index", "trajectory_index"} or expected_unit is None:
        raise ProjectPageBundleError(f"PROJECTPAGE_VISUAL_SALIENCY_UNIT_INVALID:{case_id}")
    if selection.get("unit_value") != expected_unit:
        raise ProjectPageBundleError(f"PROJECTPAGE_VISUAL_SALIENCY_UNIT_MISMATCH:{case_id}")
    observed_rank = int(selection.get("observed_baseline_only_rank") or 0)
    required_rank = int(selection.get("required_baseline_rank") or 0)
    if observed_rank < 1 or required_rank < 1 or observed_rank > required_rank:
        raise ProjectPageBundleError(f"PROJECTPAGE_VISUAL_SALIENCY_RANK_INVALID:{case_id}")
    window = receipt.get("analysis_window")
    if not isinstance(window, Mapping):
        raise ProjectPageBundleError(f"PROJECTPAGE_VISUAL_SALIENCY_WINDOW_MISSING:{case_id}")
    if int(window.get("start_frame") or 0) != (start_frame or 0):
        raise ProjectPageBundleError(f"PROJECTPAGE_VISUAL_SALIENCY_WINDOW_MISMATCH:{case_id}")
    receipt_end = window.get("end_frame")
    if (None if receipt_end is None else int(receipt_end)) != end_frame:
        raise ProjectPageBundleError(f"PROJECTPAGE_VISUAL_SALIENCY_WINDOW_MISMATCH:{case_id}")
    checks = receipt.get("checks")
    if not isinstance(checks, Mapping) or not checks or not all(value is True for value in checks.values()):
        raise ProjectPageBundleError(f"PROJECTPAGE_VISUAL_SALIENCY_CHECKS_INVALID:{case_id}")
    measurements = receipt.get("measurements")
    if not isinstance(measurements, Mapping):
        raise ProjectPageBundleError(f"PROJECTPAGE_VISUAL_SALIENCY_MEASUREMENTS_MISSING:{case_id}")
    return {
        "visual_saliency_manifest": str(receipt_path),
        "visual_saliency_manifest_sha256": _sha256(receipt_path),
        "baseline_only_rank": observed_rank,
        "selection_pool_size": int(selection.get("pool_size") or 0),
        "frame_count": int(measurements.get("frame_count") or 0),
        "mean_mse_gain_baseline_minus_candidate": float(
            measurements.get("mean_mse_gain_baseline_minus_candidate") or 0.0
        ),
        "improved_frame_fraction": float(measurements.get("improved_frame_fraction") or 0.0),
        "mean_candidate_baseline_rmse": float(
            measurements.get("mean_candidate_baseline_rmse") or 0.0
        ),
        "peak_candidate_baseline_rmse": float(
            measurements.get("peak_candidate_baseline_rmse") or 0.0
        ),
    }


def _validate_event(
    event: Mapping[str, object],
    *,
    event_path: Path,
    environment: str,
    primitive: str,
    trajectory_index: int,
    case_id: str,
) -> dict[str, object]:
    if str(event.get("environment") or "") != environment or str(event.get("primitive") or "") != primitive:
        raise ProjectPageBundleError(f"PROJECTPAGE_EVENT_IDENTITY_MISMATCH:{case_id}")
    results = event.get("trajectory_results")
    if not isinstance(results, list):
        raise ProjectPageBundleError(f"PROJECTPAGE_EVENT_RESULTS_MISSING:{case_id}")
    selected = next(
        (item for item in results if isinstance(item, Mapping) and int(item.get("trajectory_index") or -1) == trajectory_index),
        None,
    )
    if not isinstance(selected, Mapping) or selected.get("candidate_event_pass") is not True:
        raise ProjectPageBundleError(f"PROJECTPAGE_TRAJECTORY_EVENT_NOT_POSITIVE:{case_id}")
    if float(selected.get("completion_uplift") or 0.0) <= 0.0:
        raise ProjectPageBundleError(f"PROJECTPAGE_TRAJECTORY_EVENT_UPLIFT_INVALID:{case_id}")
    return {
        "event_gate_manifest": str(event_path),
        "event_gate_manifest_sha256": _sha256(event_path),
        "aggregate_event_gate_pass": bool(event.get("event_improvement_pass")),
        "aggregate_event_classification": event.get("classification"),
        "trajectory_index": trajectory_index,
        "baseline_event_pass": bool(selected.get("baseline_event_pass")),
        "candidate_event_pass": True,
        "baseline_completion_ratio": float(selected["baseline_completion_ratio"]),
        "candidate_completion_ratio": float(selected["candidate_completion_ratio"]),
        "completion_uplift": float(selected["completion_uplift"]),
    }


def _transcode(source: Path, target: Path, *, start_frame: int | None, end_frame: int | None) -> None:
    if (start_frame is None) != (end_frame is None):
        raise ProjectPageBundleError("PROJECTPAGE_CLIP_RANGE_INCOMPLETE")
    command = ["ffmpeg", "-nostdin", "-y", "-v", "error", "-i", str(source)]
    if start_frame is not None and end_frame is not None:
        if start_frame < 0 or end_frame <= start_frame:
            raise ProjectPageBundleError("PROJECTPAGE_CLIP_RANGE_INVALID")
        command.extend(["-vf", f"trim=start_frame={start_frame}:end_frame={end_frame},setpts=PTS-STARTPTS"])
    command.extend([
        "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(target),
    ])
    _run_media(command, "PROJECTPAGE_VIDEO_TRANSCODE_FAILED")


def _poster(video: Path, target: Path, *, frame: int) -> None:
    _run_media(
        [
            "ffmpeg", "-nostdin", "-y", "-v", "error", "-i", str(video), "-vf",
            f"select=eq(n\\,{frame})", "-frames:v", "1", "-threads", "1", str(target),
        ],
        "PROJECTPAGE_POSTER_EXPORT_FAILED",
    )


def _probe(video: Path) -> dict[str, object]:
    completed = _run_media(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
            "stream=width,height,avg_frame_rate,nb_frames:format=duration", "-of", "json", str(video),
        ],
        "PROJECTPAGE_VIDEO_PROBE_FAILED",
    )
    payload = json.loads(completed.stdout)
    stream = payload["streams"][0]
    frame_count = int(stream.get("nb_frames") or 0)
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    if frame_count < 2 or width < 3 or height < 1 or width % 3 != 0:
        raise ProjectPageBundleError("PROJECTPAGE_VIDEO_SHAPE_INVALID")
    return {
        "width": width,
        "height": height,
        "panel_width": width // 3,
        "frame_count": frame_count,
        "fps": stream.get("avg_frame_rate"),
        "duration_seconds": float(payload["format"]["duration"]),
    }


def _run_media(command: Sequence[str], error: str) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise ProjectPageBundleError(f"{error}:{completed.stderr.strip()}")
    return completed


def _regular_file(value: object, label: str, case_id: str) -> Path:
    path = Path(str(value or "")).resolve()
    if path.is_symlink() or not path.is_file():
        raise ProjectPageBundleError(f"PROJECTPAGE_{label}_INVALID:{case_id}:{path}")
    return path


def _optional_int(value: object) -> int | None:
    return None if value is None else int(value)


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ProjectPageBundleError(f"PROJECTPAGE_JSON_INVALID:{path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    fields = [
        "id", "environment", "primitive", "evidence_type", "sample_index", "trajectory_index",
        "psnr_delta", "ssim_delta", "mse_delta", "masked_mse_delta", "aggregate_event_gate_pass",
        "selected_trajectory_event_positive", "independent_confirmation_pass", "confirmation_psnr_delta",
        "confirmation_ssim_delta", "confirmation_mse_delta", "confirmation_masked_mse_delta",
        "visual_saliency_pass", "baseline_only_rank", "visual_mean_mse_gain",
        "visual_improved_frame_fraction", "visual_mean_candidate_baseline_rmse",
        "video_path", "poster_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = {field: record.get(field) for field in fields}
            confirmation = record.get("confirmation_evidence")
            if isinstance(confirmation, Mapping):
                for metric in ("psnr", "ssim", "mse", "masked_mse"):
                    row[f"confirmation_{metric}_delta"] = confirmation.get(f"{metric}_delta")
            visual = record.get("visual_saliency_evidence")
            if isinstance(visual, Mapping):
                row["baseline_only_rank"] = visual.get("baseline_only_rank")
                row["visual_mean_mse_gain"] = visual.get("mean_mse_gain_baseline_minus_candidate")
                row["visual_improved_frame_fraction"] = visual.get("improved_frame_fraction")
                row["visual_mean_candidate_baseline_rmse"] = visual.get(
                    "mean_candidate_baseline_rmse"
                )
            writer.writerow(row)


def _markdown(records: Sequence[Mapping[str, object]]) -> str:
    lines = [
        "# ACWM Project-Page Showcase Candidates",
        "",
        "All videos use the full-frame `GT | Baseline | Ours` layout with paired trajectories. No GT pixels are injected into predictions.",
        "",
        "| Case | Environment | Primitive | Evidence | PSNR | SSIM | Confirmation | Visual gate | Event note | Video | Poster |",
        "|---|---|---|---|---:|---:|---|---|---|---|---|",
    ]
    for record in records:
        event = record.get("event_evidence")
        note = "aggregate gate" if not isinstance(event, Mapping) else (
            f"trajectory +{float(event['completion_uplift']):.3f}; aggregate event pass={event['aggregate_event_gate_pass']}"
        )
        confirmation = record.get("confirmation_evidence")
        confirmation_note = "not declared" if not isinstance(confirmation, Mapping) else (
            f"pass; PSNR {float(confirmation['psnr_delta']):+.3f}"
        )
        visual = record.get("visual_saliency_evidence")
        visual_note = "not declared" if not isinstance(visual, Mapping) else (
            f"pass; rank {int(visual['baseline_only_rank'])}; "
            f"gain {float(visual['mean_mse_gain_baseline_minus_candidate']):+.1f}; "
            f"frames {float(visual['improved_frame_fraction']):.0%}"
        )
        lines.append(
            f"| {record['id']} | {record['environment']} | {record['primitive']} | {record['evidence_type']} | "
            f"{float(record['psnr_delta']):+.3f} | {float(record['ssim_delta']):+.4f} | {confirmation_note} | "
            f"{visual_note} | {note} | "
            f"{Path(str(record['video_path'])).name} | {Path(str(record['poster_path'])).name} |"
        )
    lines.extend([
        "",
        "## Claim Boundary",
        "",
        "Official 50-step gates govern aggregate pixel-metric claims. A selected event-positive trajectory is a disclosed qualitative case, not an aggregate event-success claim.",
    ])
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run_bundle(output_root=args.output_root, spec_path=args.spec), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
