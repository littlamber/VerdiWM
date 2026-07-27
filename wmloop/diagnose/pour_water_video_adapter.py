"""Turn real GT|prediction pour-water rollouts into diagnostic probe inputs."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from wmloop.contracts import ContractValidationError, validate_document
from wmloop.diagnose.probes.acwm_pour_water_container_boundary_leak_diagnostic_v1 import (
    ContainerBoundaryLeakProbeError,
    measure_container_boundary_leak,
)
from wmloop.diagnose.probes.acwm_pour_water_fluid_volume_transport_diagnostic_v1 import (
    FluidVolumeTransportProbeError,
    measure_fluid_volume_transport,
)
from wmloop.diagnose.probes.acwm_pour_water_free_surface_diagnostic_v1 import (
    PourWaterFreeSurfaceProbeError,
    measure_free_surface,
)


class PourWaterVideoAdapterError(RuntimeError):
    """A rollout could not be converted into defensible fluid measurements."""


def diagnose_paired_rollout_video(*, video_path: Path, trajectory_index: int) -> dict[str, object]:
    """Measure the prediction panel using geometry derived from its paired GT panel."""

    try:
        import cv2
        import imageio.v2 as imageio
        import numpy as np
    except ImportError as exc:
        raise PourWaterVideoAdapterError("POUR_WATER_VIDEO_ADAPTER_DEPENDENCY_MISSING") from exc
    path = Path(video_path).resolve(strict=True)
    if path.is_symlink() or trajectory_index < 0:
        raise PourWaterVideoAdapterError("POUR_WATER_VIDEO_ADAPTER_INPUT_INVALID")
    frames = np.stack(imageio.mimread(path))[:, :, :, :3]
    if len(frames) < 32 or frames.shape[2] < 4:
        raise PourWaterVideoAdapterError("POUR_WATER_VIDEO_ADAPTER_FRAME_COUNT_INSUFFICIENT")
    half = frames.shape[2] // 2
    gt_frames = frames[:, :, :half]
    prediction_frames = frames[:, :, half : half * 2]
    gt_masks = _water_masks(gt_frames, np)
    prediction_masks = _water_masks(prediction_frames, np)
    initial_window = max(2, min(10, len(frames) // 8))
    late_window = max(5, len(frames) // 5)
    initial_occupancy = gt_masks[:initial_window].mean(axis=0)
    late_occupancy = gt_masks[-late_window:].mean(axis=0)
    source_support = initial_occupancy >= 0.15
    target_support = (late_occupancy >= 0.15) & (initial_occupancy < 0.10)
    if int(source_support.sum()) < 100 or int(target_support.sum()) < 100:
        raise PourWaterVideoAdapterError("POUR_WATER_VIDEO_ADAPTER_SUPPORT_INSUFFICIENT")
    source_region = cv2.dilate(source_support.astype("uint8"), np.ones((9, 9), dtype="uint8")) > 0
    target_region = _expanded_bbox_mask(target_support, np=np, margin=12)
    target_y, target_x = _centroid(target_support, np=np)
    late_start = len(frames) - late_window

    fluid_frames: list[dict[str, float]] = []
    leak_frames: list[dict[str, float]] = []
    surface_frames: list[dict[str, float]] = []
    for index, mask in enumerate(prediction_masks):
        water_area = float(mask.sum())
        water_y, water_x = _centroid(mask, np=np, fallback=(target_y, target_x))
        transported = mask & ~source_region
        in_target = transported & target_region
        outside_target = transported & ~target_region
        fluid_frames.append(
            {
                "frame": float(index),
                "water_area": water_area,
                "water_centroid_x": water_x,
                "water_centroid_y": water_y,
                "target_centroid_x": target_x,
                "target_centroid_y": target_y,
                "spill_area": float(outside_target.sum()) if index >= late_start else 0.0,
            }
        )
        if index < late_start:
            continue
        transported_area = int(transported.sum())
        if transported_area > 0:
            leak_frames.append(
                {
                    "frame": float(index),
                    "in_container_area": float(in_target.sum()),
                    "outside_container_area": float(outside_target.sum()),
                }
            )
        surface = _surface_measurement(in_target, frame=index, np=np)
        if surface is not None:
            surface_frames.append(surface)

    outputs = [
        _run_or_incomplete(
            signature="fluid_volume_transport",
            probe_id="acwm_pour_water_fluid_volume_transport_diagnostic_v1",
            callback=lambda: measure_fluid_volume_transport(frames=fluid_frames),
            unavailable_code="fluid_measurement_unavailable",
        ),
        _run_or_incomplete(
            signature="container_boundary_leak",
            probe_id="acwm_pour_water_container_boundary_leak_diagnostic_v1",
            callback=lambda: measure_container_boundary_leak(frames=leak_frames),
            unavailable_code="transported_water_support_unavailable",
        ),
        _run_or_incomplete(
            signature="free_surface",
            probe_id="acwm_pour_water_free_surface_diagnostic_v1",
            callback=lambda: measure_free_surface(frames=surface_frames),
            unavailable_code="target_free_surface_unavailable",
        ),
    ]
    routed = _routed_signatures(outputs)
    return {
        "schema_version": 1,
        "artifact_type": "wmloop-pour-water-video-diagnostic",
        "state": "measured",
        "environment": "pour_water",
        "trajectory_index": trajectory_index,
        "video_path": str(path),
        "video_sha256": _sha256(path),
        "frame_count": int(len(frames)),
        "measurement_contract": {
            "panel_layout": "GT|prediction",
            "target_geometry_source": "paired_gt_late_occupancy",
            "prediction_measurement_source": "prediction_panel_only",
            "late_window_start": late_start,
            "source_support_pixels": int(source_support.sum()),
            "target_support_pixels": int(target_support.sum()),
        },
        "probe_outputs": outputs,
        "routed_failure_families": routed,
        "verdict_exposure_allowed": False,
        "claim_boundary": (
            "These measurements route diagnostic exploration only. Paired GT locates source and target geometry, "
            "but no GT pixels, masks, metrics, or labels are passed into training or candidate generation."
        ),
    }


def build_video_diagnostic_bundle(*, horizon_manifest: Path, output_root: Path) -> dict[str, object]:
    manifest_path = Path(horizon_manifest).resolve(strict=True)
    manifest = _load_horizon_manifest(manifest_path)
    rows = manifest.get("trajectory_results")
    assert isinstance(rows, list)
    diagnostics = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        index = row.get("trajectory_index")
        video = row.get("rollout_video_path")
        if isinstance(index, int) and not isinstance(index, bool) and isinstance(video, str):
            diagnostics.append(diagnose_paired_rollout_video(video_path=Path(video), trajectory_index=index))
    if not diagnostics:
        raise PourWaterVideoAdapterError("POUR_WATER_VIDEO_ADAPTER_NO_TRAJECTORIES")
    routed = list(
        dict.fromkeys(
            signature
            for diagnostic in diagnostics
            for signature in diagnostic["routed_failure_families"]
            if isinstance(signature, str)
        )
    )
    report = {
        "schema_version": 1,
        "artifact_type": "wmloop-pour-water-video-diagnostic-bundle",
        "state": "ready",
        "environment": "pour_water",
        "source_horizon_manifest": str(manifest_path),
        "source_horizon_manifest_sha256": _sha256(manifest_path),
        "trajectory_count": len(diagnostics),
        "routed_failure_families": routed,
        "diagnostics": diagnostics,
        "verdict_exposure_allowed": False,
    }
    return _write_bundle(Path(output_root).resolve(), report)


def _run_or_incomplete(*, signature: str, probe_id: str, callback: Any, unavailable_code: str) -> dict[str, object]:
    try:
        return callback()
    except (FluidVolumeTransportProbeError, ContainerBoundaryLeakProbeError, PourWaterFreeSurfaceProbeError):
        output: dict[str, object] = {
            "schema_version": 1,
            "artifact_type": "wmloop-diagnostic-probe-output",
            "probe_id": probe_id,
            "role": "diagnostic",
            "environment": "pour_water",
            "signature": signature,
            "state": "incomplete",
            "metrics": {"unavailable_code": unavailable_code},
            "flags": [unavailable_code],
            "evidence_refs": [],
            "verdict_exposure_allowed": False,
            "limitations": [
                "The measured prediction did not contain enough support for this probe.",
                "Incomplete diagnostic evidence is not verdict evidence and cannot establish improvement.",
            ],
        }
        try:
            validate_document("diagnostic_probe_output", output)
        except ContractValidationError as exc:
            raise PourWaterVideoAdapterError("POUR_WATER_VIDEO_ADAPTER_INCOMPLETE_CONTRACT_INVALID") from exc
        return output


def _routed_signatures(outputs: Sequence[Mapping[str, object]]) -> list[str]:
    routed = []
    for output in outputs:
        flags = output.get("flags")
        if output.get("state") != "measured" or not isinstance(flags, list):
            continue
        if flags:
            routed.append(str(output["signature"]))
    return list(dict.fromkeys(routed))


def _water_masks(frames: Any, np: Any) -> Any:
    values = frames.astype(np.int16)
    red, green, blue = values[..., 0], values[..., 1], values[..., 2]
    return (blue > 100) & (blue - red > 25) & (green - red > 10) & (blue >= green - 5)


def _expanded_bbox_mask(mask: Any, *, np: Any, margin: int) -> Any:
    ys, xs = np.nonzero(mask)
    if len(xs) < 1:
        raise PourWaterVideoAdapterError("POUR_WATER_VIDEO_ADAPTER_TARGET_EMPTY")
    y0, y1 = max(0, int(ys.min()) - margin), min(mask.shape[0], int(ys.max()) + margin + 1)
    x0, x1 = max(0, int(xs.min()) - margin), min(mask.shape[1], int(xs.max()) + margin + 1)
    output = np.zeros_like(mask, dtype=bool)
    output[y0:y1, x0:x1] = True
    return output


def _centroid(mask: Any, *, np: Any, fallback: tuple[float, float] | None = None) -> tuple[float, float]:
    ys, xs = np.nonzero(mask)
    if len(xs) < 1:
        if fallback is None:
            raise PourWaterVideoAdapterError("POUR_WATER_VIDEO_ADAPTER_CENTROID_EMPTY")
        return fallback
    return float(ys.mean()), float(xs.mean())


def _surface_measurement(mask: Any, *, frame: int, np: Any) -> dict[str, float] | None:
    area = int(mask.sum())
    if area < 20:
        return None
    columns = []
    for x in np.flatnonzero(mask.any(axis=0)):
        columns.append(float(np.flatnonzero(mask[:, x])[0]))
    if len(columns) < 2:
        return None
    surface = np.asarray(columns, dtype=float)
    roughness = float(np.abs(np.diff(surface)).mean() / max(1, mask.shape[0]))
    ys, _ = np.nonzero(mask)
    return {
        "frame": float(frame),
        "surface_area": float(area),
        "surface_centroid_y": float(ys.mean() / max(1, mask.shape[0])),
        "surface_roughness": roughness,
    }


def _load_horizon_manifest(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PourWaterVideoAdapterError("POUR_WATER_VIDEO_ADAPTER_MANIFEST_INVALID") from exc
    if (
        not isinstance(payload, Mapping)
        or payload.get("artifact_type") != "wmloop-horizon-probe-run"
        or payload.get("state") != "ready"
        or payload.get("environment") != "pour_water"
        or not isinstance(payload.get("trajectory_results"), list)
    ):
        raise PourWaterVideoAdapterError("POUR_WATER_VIDEO_ADAPTER_MANIFEST_INVALID")
    return payload


def _write_bundle(destination: Path, report: Mapping[str, object]) -> dict[str, object]:
    if destination.exists() or destination.is_symlink():
        raise PourWaterVideoAdapterError("POUR_WATER_VIDEO_ADAPTER_OUTPUT_EXISTS")
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    try:
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary.mkdir(mode=0o700)
        report_path = temporary / "pour-water-video-diagnostics.json"
        report_path.write_text(
            json.dumps(report, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        manifest = {
            "schema_version": 1,
            "artifact_type": "wmloop-pour-water-video-diagnostic-manifest",
            "state": "ready",
            "environment": "pour_water",
            "trajectory_count": report["trajectory_count"],
            "routed_failure_families": report["routed_failure_families"],
            "report_path": str(destination / report_path.name),
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
        return manifest
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
