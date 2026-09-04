"""WAN2.2-DROID ACWM control-plane primitives.

The module deliberately stops at the interface boundary.  It provides a
deterministic manifest builder and CPU conformance checks for the
``first-frame + action/proprio history -> 150 frame`` contract.  Model
execution is delegated to an explicitly supplied WAN2.2 adapter command; no
Wan2.1 implementation is imported implicitly.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping


class Wan22DroidError(ValueError):
    """The WAN2.2-DROID contract cannot be admitted."""


def required_rollout_source_frames(
    *, horizon_frames: int = 150, chunk_frames: int = 45
) -> int:
    """Return source frames needed by the causal tail contract.

    WAN2.2's shortest causal-VAE tail emits two fewer frames. A 150-frame
    rollout split into 45-frame chunks therefore needs 17 source frames for
    the final latent chunk (15 published frames), i.e. 152 source frames in
    total. Keeping this calculation in the model-family adapter makes the
    requirement explicit without leaking WAN22 commands into the scheduler.
    """

    if horizon_frames < 2 or chunk_frames < 1:
        raise Wan22DroidError("DROID_WINDOW_ARGUMENT_INVALID")
    remainder = horizon_frames % chunk_frames
    return horizon_frames if remainder == 0 else horizon_frames + 2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> Path:
    path = path.expanduser().resolve()
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def build_sample_manifest(
    data_root: Path,
    split: str,
    *,
    horizon_frames: int = 150,
    stride: int = 30,
) -> dict[str, Any]:
    """Build a deterministic window manifest from processed DROID metadata.

    One record is emitted for every valid window start.  The first frame is
    always the observed frame and the following 149 frames are the future
    target.  Episodes shorter than the causal rollout source requirement are
    excluded from the 30-second contract rather than padded.  Every emitted
    window is therefore guaranteed to have enough source frames for the final
    causal tail, not merely enough target frames.
    """

    root = Path(data_root).expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise Wan22DroidError("DROID_DATA_ROOT_INVALID")
    if split not in {"train", "val"}:
        raise Wan22DroidError(f"DROID_SPLIT_INVALID:{split}")
    if horizon_frames < 2 or stride < 1:
        raise Wan22DroidError("DROID_WINDOW_ARGUMENT_INVALID")
    rollout_source_frames = required_rollout_source_frames(
        horizon_frames=horizon_frames
    )
    annotation_root = root / "annotation" / split
    paths = sorted(annotation_root.glob("*.json"))
    if not paths:
        raise Wan22DroidError(f"DROID_ANNOTATION_MISSING:{split}")
    records: list[dict[str, Any]] = []
    excluded = 0
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise Wan22DroidError(f"DROID_ANNOTATION_INVALID:{path.name}") from exc
        if not isinstance(payload, Mapping) or payload.get("split") != split:
            raise Wan22DroidError(f"DROID_SPLIT_MISMATCH:{path.name}")
        episode_id = str(payload.get("episode_id") or path.stem)
        length = int(payload.get("video_length", 0))
        actions = payload.get("action")
        proprio = payload.get("proprio")
        if (
            length < rollout_source_frames
            or not isinstance(actions, list)
            or len(actions) != length
            or not isinstance(proprio, list)
            or len(proprio) != length
            or not actions
            or not isinstance(actions[0], list)
            or len(actions[0]) != 7
            or not isinstance(proprio[0], list)
            or len(proprio[0]) != 14
        ):
            excluded += 1
            continue
        video_rel = str(
            payload.get("video_path") or f"videos/{split}/{episode_id}/wrist.mp4"
        )
        latent_rel = str(
            payload.get("latent_path") or f"latent_videos/{split}/{episode_id}/wrist.pt"
        )
        video = root / video_rel
        latent = root / latent_rel
        if not video.is_file() or video.is_symlink():
            raise Wan22DroidError(f"DROID_VIDEO_MISSING:{episode_id}")
        if not latent.is_file() or latent.is_symlink():
            raise Wan22DroidError(f"DROID_LATENT_MISSING:{episode_id}")
        latent_shape = payload.get("latent_shape")
        latent_channels = (
            int(latent_shape[1])
            if isinstance(latent_shape, list) and len(latent_shape) == 4
            else None
        )
        latent_source = (
            "precomputed_wan22" if latent_channels == 48 else "raw_video_wan22_vae"
        )
        for start in range(0, length - rollout_source_frames + 1, stride):
            records.append(
                {
                    "sample_id": f"{episode_id}:{start:06d}:{horizon_frames}",
                    "episode_id": episode_id,
                    "split": split,
                    "start_frame": start,
                    "horizon_frames": horizon_frames,
                    "source_frame_count": length,
                    "rollout_source_frames_required": rollout_source_frames,
                    "video_path": video_rel,
                    "latent_path": latent_rel,
                    "annotation_path": str(path.relative_to(root)),
                    "action_dim": 7,
                    "proprio_dim": 14,
                    "fps": int(payload.get("processed_fps", 5)),
                    "instruction": str(
                        payload.get("instruction")
                        or "DROID robot action-conditioned future prediction"
                    ),
                    "latent_shape": latent_shape,
                    "precomputed_latent_channels": latent_channels,
                    "latent_source": latent_source,
                }
            )
    if not records:
        raise Wan22DroidError(f"DROID_NO_TARGET_WINDOWS:{split}")
    manifest_digest = hashlib.sha256(
        json.dumps(
            records, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    return {
        "schema_version": 1,
        "artifact_type": "verdiwm-wan22-droid-sample-manifest",
        "split": split,
        "data_root": str(root),
        "horizon_frames": horizon_frames,
        "rollout_source_frames_required": rollout_source_frames,
        "window_stride": stride,
        "record_count": len(records),
        "episode_count": len({str(row["episode_id"]) for row in records}),
        "excluded_short_or_invalid_episodes": excluded,
        "precomputed_latent_compatible_records": sum(
            1 for row in records if row["latent_source"] == "precomputed_wan22"
        ),
        "raw_video_reencode_records": sum(
            1 for row in records if row["latent_source"] == "raw_video_wan22_vae"
        ),
        "manifest_sha256": manifest_digest,
        "records": records,
    }


def write_sample_manifest(
    data_root: Path,
    split: str,
    output: Path,
    *,
    horizon_frames: int = 150,
    stride: int = 30,
) -> dict[str, Any]:
    manifest = build_sample_manifest(
        data_root, split, horizon_frames=horizon_frames, stride=stride
    )
    _write_json(output, manifest)
    return manifest


def validate_contract(
    *,
    train_manifest: Path,
    validation_manifest: Path,
    model: Path,
    source: Path,
    evaluator_contract: Path,
    adapter: Path | None = None,
    horizon_frames: int = 150,
    rollout_source_frames: int | None = None,
) -> dict[str, Any]:
    """Validate immutable paths and tensor dimensions before any GPU launch."""

    blockers: list[str] = []
    required_source_frames = (
        required_rollout_source_frames(horizon_frames=horizon_frames)
        if rollout_source_frames is None
        else int(rollout_source_frames)
    )
    if required_source_frames < horizon_frames:
        raise Wan22DroidError("DROID_ROLLOUT_SOURCE_FRAMES_INVALID")
    manifests: dict[str, Any] = {}
    for name, path in (("train", train_manifest), ("validation", validation_manifest)):
        candidate = Path(path).expanduser().resolve()
        try:
            value = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            blockers.append(f"{name.upper()}_MANIFEST_INVALID")
            continue
        if (
            not isinstance(value, Mapping)
            or value.get("artifact_type") != "verdiwm-wan22-droid-sample-manifest"
        ):
            blockers.append(f"{name.upper()}_MANIFEST_CONTRACT_INVALID")
            continue
        if int(value.get("horizon_frames", 0)) != horizon_frames or not value.get(
            "records"
        ):
            blockers.append(f"{name.upper()}_HORIZON_OR_RECORDS_INVALID")
        if int(value.get("rollout_source_frames_required", 0)) < required_source_frames:
            blockers.append(f"{name.upper()}_ROLLOUT_SOURCE_FRAMES_CONTRACT_INVALID")
        for row in value.get("records", []):
            if row.get("action_dim") != 7 or row.get("proprio_dim") != 14:
                blockers.append(f"{name.upper()}_CONDITIONING_DIM_INVALID")
                break
            if name == "validation" and (
                int(row.get("source_frame_count", 0))
                < int(row.get("start_frame", 0)) + required_source_frames
            ):
                blockers.append(f"{name.upper()}_ROLLOUT_SOURCE_FRAMES_INSUFFICIENT")
                break
        manifests[name] = {
            "path": str(candidate),
            "sha256": _sha256(candidate) if candidate.is_file() else None,
            "records": int(value.get("record_count", 0)),
            "episodes": int(value.get("episode_count", 0)),
        }
    for label, path in (
        ("MODEL", model),
        ("SOURCE", source),
        ("EVALUATOR", evaluator_contract),
    ):
        candidate = Path(path).expanduser().resolve()
        if not candidate.exists() or candidate.is_symlink():
            blockers.append(f"{label}_PATH_INVALID")
    if adapter is None:
        blockers.append("WAN22_ADAPTER_BINDING_REQUIRED")
    else:
        adapter_path = Path(adapter).expanduser().resolve()
        if not adapter_path.is_file() or adapter_path.is_symlink():
            blockers.append("WAN22_ADAPTER_PATH_INVALID")
        elif (
            "wan22" not in adapter_path.name.lower()
            or "droid" not in adapter_path.name.lower()
        ):
            blockers.append("WAN22_ADAPTER_IDENTITY_INVALID")
    # TI2V uses the non-causal WanModel entrypoint; causal model_causal.py is
    # optional. Existence alone is insufficient because a different checkout
    # could otherwise pass conformance and fail only after GPU admission.
    if not (
        Path(source).expanduser().resolve() / "wan" / "modules" / "model.py"
    ).is_file():
        blockers.append("WAN22_MODEL_ENTRYPOINT_MISSING")
    result = {
        "schema_version": 1,
        "artifact_type": "verdiwm-wan22-droid-conformance-report",
        "state": "ready_for_execution" if not blockers else "blocked",
        "model": str(Path(model).expanduser().resolve()),
        "source": str(Path(source).expanduser().resolve()),
        "evaluator_contract": str(Path(evaluator_contract).expanduser().resolve()),
        "adapter": (
            str(Path(adapter).expanduser().resolve()) if adapter is not None else None
        ),
        "horizon_frames": horizon_frames,
        "rollout_source_frames_required": required_source_frames,
        "conditioning": {
            "first_frame": True,
            "action_dim": 7,
            "proprio_dim": 14,
            "history": True,
        },
        "manifests": manifests,
        "blockers": sorted(set(blockers)),
        "claim_boundary": "Conformance is an execution-readiness check; it is not evidence of 30-second quality.",
    }
    return result


def run_external(
    command: list[str],
    *,
    cwd: Path,
    output_root: Path,
    budget_gpu_hours: float,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    """Run an explicitly bound external stage under a wall-clock budget."""

    if budget_gpu_hours <= 0:
        raise Wan22DroidError("GPU_BUDGET_INVALID")
    if not command:
        raise Wan22DroidError("EXTERNAL_COMMAND_REQUIRED")
    if timeout_seconds is None:
        timeout_seconds = max(60.0, float(budget_gpu_hours) * 3600.0)
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(float(timeout_seconds))
        or timeout_seconds <= 0
    ):
        raise Wan22DroidError("EXTERNAL_TIMEOUT_INVALID")
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            command,
            cwd=str(Path(cwd).expanduser().resolve()),
            text=True,
            capture_output=True,
            timeout=float(timeout_seconds),
        )
    except subprocess.TimeoutExpired as exc:
        stdout = (
            exc.stdout.decode()
            if isinstance(exc.stdout, bytes)
            else str(exc.stdout or "")
        )
        stderr = (
            exc.stderr.decode()
            if isinstance(exc.stderr, bytes)
            else str(exc.stderr or "")
        )
        (output_root / "stdout.log").write_text(stdout, encoding="utf-8")
        (output_root / "stderr.log").write_text(stderr, encoding="utf-8")
        _write_json(
            output_root / "timeout_receipt.json",
            {
                "schema_version": 1,
                "artifact_type": "verdiwm-external-stage-timeout-receipt",
                "state": "timed_out",
                "command": command,
                "timeout_seconds": float(timeout_seconds),
                "output_root": str(output_root),
                "claim_boundary": "The external stage exceeded its wall-clock budget; no scientific evidence was accepted.",
            },
        )
        return {
            "state": "timed_out",
            "returncode": None,
            "command": command,
            "output_root": str(output_root),
            "timeout_seconds": float(timeout_seconds),
        }
    (output_root / "stdout.log").write_text(result.stdout, encoding="utf-8")
    (output_root / "stderr.log").write_text(result.stderr, encoding="utf-8")
    return {
        "state": "completed" if result.returncode == 0 else "failed",
        "returncode": result.returncode,
        "command": command,
        "output_root": str(output_root),
        "timeout_seconds": float(timeout_seconds),
    }
