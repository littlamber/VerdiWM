"""Train a lightweight inverse-dynamics head from ACWM GT videos.

This is the M1 training worker for the action-following probe.  It deliberately
keeps the model small and the cache identity explicit: source metadata digest,
environment, architecture, sample counts, R2 and low-confidence state are all
written next to the checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from wmloop.acwm_data import CANONICAL_ACWM_ENVIRONMENTS
from wmloop.archive.store import ArchiveStore, ContentAddressedStore
from wmloop.diagnose.inverse_dynamics import InverseDynamicsCacheRecord
from wmloop.execute.gpu_exclusivity_audit import verify_gpu_exclusivity_ready


class InverseDynamicsTrainingError(RuntimeError):
    """Inverse-dynamics training inputs or outputs were invalid."""


@dataclass(frozen=True)
class TrainingConfig:
    data_root: Path
    environment: str
    split: str
    output_root: Path
    max_trajectories: int = 24
    frames_per_trajectory: int = 8
    image_size: int = 32
    hidden_dim: int = 512
    epochs: int = 20
    batch_size: int = 128
    learning_rate: float = 1e-3
    seed: int = 7
    device: str = "cuda"
    gpu_index: int | None = None
    gpu_exclusivity_audit_manifest: Path | None = None
    gpu_exclusivity_max_age_seconds: float | None = 300.0


def train_inverse_dynamics_head(config: TrainingConfig) -> dict[str, object]:
    _validate_config(config)
    gpu_exclusivity = _verify_gpu_launch_preflight(
        device=config.device,
        gpu_index=config.gpu_index,
        manifest_path=config.gpu_exclusivity_audit_manifest,
        max_age_seconds=config.gpu_exclusivity_max_age_seconds,
    )
    import cv2  # type: ignore[import-not-found]
    import numpy as np  # type: ignore[import-not-found]
    import torch  # type: ignore[import-not-found]
    from torch import nn  # type: ignore[import-not-found]
    from torch.utils.data import DataLoader, TensorDataset  # type: ignore[import-not-found]

    started = time.monotonic()
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    spec = _environment_spec(config.environment)
    split_root = Path(config.data_root).resolve() / spec.dataset_relative_path / config.split
    metadata_path = split_root / "metadata.pt"
    if not metadata_path.is_file() or metadata_path.is_symlink():
        raise InverseDynamicsTrainingError("INVERSE_DYNAMICS_METADATA_MISSING")
    metadata_digest = _sha256_file(metadata_path)
    metadata = torch.load(metadata_path, map_location="cpu", weights_only=False)
    if not isinstance(metadata, list) or not metadata:
        raise InverseDynamicsTrainingError("INVERSE_DYNAMICS_METADATA_INVALID")
    candidate_indices = _candidate_trajectory_indices(len(metadata), config.seed)
    features: list[Any] = []
    targets: list[Any] = []
    skipped: list[dict[str, object]] = []
    used_trajectory_count = 0
    attempted_trajectory_count = 0
    for trajectory_index in candidate_indices:
        if used_trajectory_count >= config.max_trajectories:
            break
        attempted_trajectory_count += 1
        entry = metadata[trajectory_index]
        if not isinstance(entry, Mapping):
            skipped.append({"trajectory_index": trajectory_index, "reason": "metadata_entry_invalid"})
            continue
        video_rel = entry.get("video_path")
        if not isinstance(video_rel, str) or not video_rel:
            skipped.append({"trajectory_index": trajectory_index, "reason": "video_path_missing"})
            continue
        action = _action_array(entry, expected_dim=int(spec_action_dim(config.environment)), np=np)
        if action is None or action.shape[0] < 2:
            skipped.append({"trajectory_index": trajectory_index, "reason": "actions_invalid"})
            continue
        video_path = _resolve_video_path(split_root, video_rel)
        frame_indices = _frame_indices(min(int(action.shape[0]), _entry_length(entry)), config.frames_per_trajectory)
        try:
            gray = _load_gray_frames(video_path, image_size=config.image_size, cv2=cv2, np=np)
        except InverseDynamicsTrainingError as exc:
            skipped.append({"trajectory_index": trajectory_index, "reason": str(exc)})
            continue
        usable = [index for index in frame_indices if index + 1 < gray.shape[0] and index < action.shape[0]]
        if not usable:
            skipped.append({"trajectory_index": trajectory_index, "reason": "no_usable_frame_pairs"})
            continue
        for frame_index in usable:
            before = gray[frame_index]
            after = gray[frame_index + 1]
            feature = np.stack([before, after, after - before], axis=0).astype("float32")
            features.append(feature.reshape(-1))
            targets.append(action[frame_index].astype("float32"))
        used_trajectory_count += 1
    if len(features) < 16:
        raise InverseDynamicsTrainingError("INVERSE_DYNAMICS_SAMPLE_COUNT_TOO_SMALL")
    x_np = np.stack(features).astype("float32")
    y_np = np.stack(targets).astype("float32")
    train_indices, validation_indices = _train_validation_indices(len(x_np), config.seed)
    x_train = torch.from_numpy(x_np[train_indices])
    y_train = torch.from_numpy(y_np[train_indices])
    x_validation = torch.from_numpy(x_np[validation_indices])
    y_validation = torch.from_numpy(y_np[validation_indices])
    y_mean = y_train.mean(dim=0, keepdim=True)
    y_std = y_train.std(dim=0, keepdim=True).clamp_min(1e-6)
    y_train_normalized = (y_train - y_mean) / y_std
    device = torch.device(config.device if config.device != "cuda" or torch.cuda.is_available() else "cpu")
    model = _InverseDynamicsMLP(x_train.shape[1], y_train.shape[1], config.hidden_dim).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count > 5_000_000:
        raise InverseDynamicsTrainingError("INVERSE_DYNAMICS_PARAMETER_CAP_EXCEEDED")
    loader = DataLoader(
        TensorDataset(x_train, y_train_normalized),
        batch_size=min(config.batch_size, len(x_train)),
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    losses: list[float] = []
    model.train()
    for _ in range(config.epochs):
        epoch_losses = []
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        losses.append(sum(epoch_losses) / len(epoch_losses))
    model.eval()
    with torch.no_grad():
        pred = model(x_validation.to(device)).cpu() * y_std + y_mean
    gt_r2 = _r2_score(y_validation, pred)
    destination = Path(config.output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise InverseDynamicsTrainingError("INVERSE_DYNAMICS_OUTPUT_EXISTS")
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    try:
        temporary.mkdir(mode=0o700, parents=True)
        checkpoint_path = temporary / "inverse_dynamics_head.pt"
        torch.save(
            {
                "schema_version": 1,
                "artifact_type": "wmloop-inverse-dynamics-head",
                "environment": config.environment,
                "split": config.split,
                "architecture": _architecture(config),
                "model_state_dict": model.cpu().state_dict(),
                "target_mean": y_mean,
                "target_std": y_std,
                "image_size": config.image_size,
                "metadata_sha256": metadata_digest,
            },
            checkpoint_path,
        )
        record = InverseDynamicsCacheRecord(
            environment=config.environment,
            dataset_digest=metadata_digest,
            architecture=_architecture(config),
            parameter_count=parameter_count,
            gt_r2=gt_r2,
        )
        cache_record_path = record.write(temporary)
        manifest = {
            "schema_version": 1,
            "artifact_type": "wmloop-inverse-dynamics-training-run",
            "state": "ready",
            "environment": config.environment,
            "split": config.split,
            "metadata_path": str(metadata_path),
            "metadata_sha256": metadata_digest,
            "cache_key": record.cache_key,
            "cache_record_path": str(destination / cache_record_path.name),
            "checkpoint_path": str(destination / checkpoint_path.name),
            "architecture": record.architecture,
            "parameter_count": parameter_count,
            "gt_r2": gt_r2,
            "low_confidence": record.low_confidence,
            "sample_count": len(x_np),
            "train_sample_count": len(train_indices),
            "validation_sample_count": len(validation_indices),
            "used_trajectory_count": used_trajectory_count,
            "attempted_trajectory_count": attempted_trajectory_count,
            "skipped_trajectories": skipped,
            "losses": losses,
            "device": str(device),
            "gpu_exclusivity_audit": gpu_exclusivity,
            "duration_seconds": time.monotonic() - started,
        }
        _write_json_atomic(temporary / "manifest.json", manifest)
        os.replace(temporary, destination)
        return manifest
    except Exception:
        if temporary.is_symlink():
            temporary.unlink()
        elif temporary.exists():
            import shutil

            shutil.rmtree(temporary)
        raise


def summarize_inverse_dynamics_runs(
    *,
    runs_root: Path,
    output_path: Path | None = None,
    archive_db: Path | None = None,
    cas_root: Path | None = None,
    environments: Sequence[str] | None = None,
) -> dict[str, object]:
    """Summarize the latest ready inverse-dynamics cache for each environment."""

    root = Path(runs_root).resolve()
    expected = tuple(environments or [spec.environment for spec in CANONICAL_ACWM_ENVIRONMENTS])
    latest: dict[str, Mapping[str, Any]] = {}
    for environment in expected:
        manifests = sorted(root.glob(f"{environment}-r*/manifest.json"), key=lambda item: _run_suffix(item.parent.name))
        for manifest_path in reversed(manifests):
            manifest = _read_json(manifest_path)
            if manifest.get("state") == "ready" and manifest.get("environment") == environment:
                latest[environment] = {**manifest, "manifest_path": str(manifest_path)}
                break
    missing = [environment for environment in expected if environment not in latest]
    archive: ArchiveStore | None = ArchiveStore(archive_db) if archive_db is not None else None
    cas: ContentAddressedStore | None = ContentAddressedStore(cas_root if cas_root is not None else Path(archive_db).resolve().parent) if archive_db is not None else None
    rows: list[dict[str, object]] = []
    for environment in expected:
        manifest = latest.get(environment)
        if manifest is None:
            continue
        row = {
            "environment": environment,
            "manifest_path": manifest["manifest_path"],
            "checkpoint_path": manifest["checkpoint_path"],
            "cache_record_path": manifest["cache_record_path"],
            "metadata_sha256": manifest["metadata_sha256"],
            "cache_key": manifest["cache_key"],
            "architecture": manifest["architecture"],
            "parameter_count": manifest["parameter_count"],
            "gt_r2": manifest["gt_r2"],
            "low_confidence": manifest["low_confidence"],
            "sample_count": manifest["sample_count"],
            "used_trajectory_count": manifest.get("used_trajectory_count", manifest.get("selected_trajectory_count")),
            "attempted_trajectory_count": manifest.get("attempted_trajectory_count", manifest.get("selected_trajectory_count")),
        }
        if archive is not None and cas is not None:
            refs = {
                "manifest_ref": _put_file(cas, Path(str(manifest["manifest_path"]))),
                "checkpoint_ref": _put_file(cas, Path(str(manifest["checkpoint_path"]))),
                "cache_record_ref": _put_file(cas, Path(str(manifest["cache_record_path"]))),
            }
            for uri in refs.values():
                archive.record_artifact_reference(uri)
            row.update(refs)
        rows.append(row)
    summary = {
        "schema_version": 1,
        "artifact_type": "wmloop-inverse-dynamics-cache-summary",
        "state": "ready" if not missing else "incomplete",
        "runs_root": str(root),
        "environment_count": len(rows),
        "missing_environments": missing,
        "low_confidence_count": sum(1 for row in rows if row["low_confidence"]),
        "ready_count": len(rows),
        "records": rows,
    }
    if output_path is not None:
        target = Path(output_path).resolve()
        if target.exists() or target.is_symlink():
            raise InverseDynamicsTrainingError("INVERSE_DYNAMICS_SUMMARY_OUTPUT_EXISTS")
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        _write_json_atomic(target, summary)
    return summary


class _InverseDynamicsMLP:  # placeholder for type checkers; replaced at runtime
    def __new__(cls, input_dim: int, output_dim: int, hidden_dim: int) -> Any:
        import torch  # type: ignore[import-not-found]
        from torch import nn  # type: ignore[import-not-found]

        return nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )


def spec_action_dim(environment: str) -> int:
    return {
        "push_cube": 2,
        "stack_cube": 7,
        "push_rope": 2,
        "cloth_move": 8,
        "push_sand": 7,
        "pour_water": 4,
        "robot_arm": 7,
        "reacher": 2,
    }[environment]


def _environment_spec(environment: str) -> Any:
    for spec in CANONICAL_ACWM_ENVIRONMENTS:
        if spec.environment == environment:
            return spec
    raise InverseDynamicsTrainingError(f"INVERSE_DYNAMICS_ENVIRONMENT_UNKNOWN:{environment}")


def _validate_config(config: TrainingConfig) -> None:
    _environment_spec(config.environment)
    if (
        not config.split
        or config.max_trajectories < 1
        or config.frames_per_trajectory < 1
        or config.image_size < 8
        or config.hidden_dim < 16
        or config.epochs < 1
        or config.batch_size < 1
        or not math.isfinite(config.learning_rate)
        or config.learning_rate <= 0
        or not config.device
    ):
        raise InverseDynamicsTrainingError("INVERSE_DYNAMICS_CONFIG_INVALID")


def _verify_gpu_launch_preflight(
    *,
    device: str,
    gpu_index: int | None,
    manifest_path: Path | None,
    max_age_seconds: float | None,
) -> dict[str, object] | None:
    if not _device_requires_gpu(device):
        return None
    if gpu_index is None:
        raise InverseDynamicsTrainingError("INVERSE_DYNAMICS_GPU_INDEX_REQUIRED")
    if not isinstance(gpu_index, int) or isinstance(gpu_index, bool) or gpu_index < 0:
        raise InverseDynamicsTrainingError("INVERSE_DYNAMICS_GPU_INDEX_INVALID")
    return verify_gpu_exclusivity_ready(
        manifest_path,
        gpu_index=gpu_index,
        max_age_seconds=max_age_seconds,
    )


def _device_requires_gpu(device: str) -> bool:
    return device.strip().lower().startswith("cuda")


def _select_trajectory_indices(total: int, maximum: int, seed: int) -> tuple[int, ...]:
    if total < 1 or maximum < 1:
        raise InverseDynamicsTrainingError("INVERSE_DYNAMICS_SELECTION_INVALID")
    count = min(total, maximum)
    rng = random.Random(seed)
    return tuple(sorted(rng.sample(range(total), count)))


def _candidate_trajectory_indices(total: int, seed: int) -> tuple[int, ...]:
    if total < 1:
        raise InverseDynamicsTrainingError("INVERSE_DYNAMICS_SELECTION_INVALID")
    indices = list(range(total))
    random.Random(seed).shuffle(indices)
    return tuple(indices)


def _frame_indices(length: int, count: int) -> tuple[int, ...]:
    if length < 2 or count < 1:
        return ()
    usable = length - 1
    if count >= usable:
        return tuple(range(usable))
    return tuple(round(index * (usable - 1) / (count - 1)) for index in range(count))


def _train_validation_indices(total: int, seed: int) -> tuple[list[int], list[int]]:
    if total < 16:
        raise InverseDynamicsTrainingError("INVERSE_DYNAMICS_SAMPLE_COUNT_TOO_SMALL")
    indices = list(range(total))
    random.Random(seed).shuffle(indices)
    validation_count = max(4, int(round(total * 0.2)))
    validation = sorted(indices[:validation_count])
    train = sorted(indices[validation_count:])
    if not train or not validation:
        raise InverseDynamicsTrainingError("INVERSE_DYNAMICS_SPLIT_INVALID")
    return train, validation


def _action_array(entry: Mapping[str, Any], *, expected_dim: int, np: Any) -> Any | None:
    raw = entry.get("actions")
    if raw is None:
        raw = entry.get("commands")
    if raw is None:
        return None
    if isinstance(raw, Mapping):
        lin = raw.get("linear_velocity")
        ang = raw.get("angular_velocity")
        if lin is None or ang is None:
            return None
        array = np.stack([_to_numpy(lin, np), _to_numpy(ang, np)], axis=-1)
    else:
        array = _to_numpy(raw, np)
    if len(array.shape) != 2 or array.shape[1] != expected_dim:
        return None
    return array


def _to_numpy(value: Any, np: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype="float32")


def _entry_length(entry: Mapping[str, Any]) -> int:
    value = entry.get("length")
    if isinstance(value, int) and value > 0:
        return value
    action = entry.get("actions")
    if hasattr(action, "shape") and action.shape:
        return int(action.shape[0])
    return 0


def _load_gray_frames(path: Path, *, image_size: int, cv2: Any, np: Any) -> Any:
    if not path.is_file() or path.is_symlink():
        raise InverseDynamicsTrainingError("video_missing")
    capture = cv2.VideoCapture(str(path))
    frames = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (image_size, image_size), interpolation=cv2.INTER_AREA)
            frames.append(gray.astype("float32") / 255.0)
    finally:
        capture.release()
    if len(frames) < 2:
        raise InverseDynamicsTrainingError("video_too_short")
    return np.stack(frames)


def _resolve_video_path(split_root: Path, video_path: str) -> Path:
    raw = Path(video_path)
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
        candidates.append(split_root / raw.name)
    else:
        candidates.append(split_root / raw)
        candidates.append(split_root / raw.name)
    for candidate in candidates:
        if candidate.is_file() and not candidate.is_symlink():
            return candidate
    return candidates[-1]


def _r2_score(target: Any, prediction: Any) -> float:
    import torch  # type: ignore[import-not-found]

    residual = torch.sum((target - prediction) ** 2)
    centered = target - target.mean(dim=0, keepdim=True)
    total = torch.sum(centered**2)
    if float(total) <= 1e-12:
        return 0.0
    value = 1.0 - float(residual / total)
    return value if math.isfinite(value) else 0.0


def _architecture(config: TrainingConfig) -> str:
    return f"frame-diff-mlp-3x{config.hidden_dim}-gray{config.image_size}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InverseDynamicsTrainingError("INVERSE_DYNAMICS_MANIFEST_INVALID") from exc
    if not isinstance(payload, Mapping):
        raise InverseDynamicsTrainingError("INVERSE_DYNAMICS_MANIFEST_INVALID")
    return payload


def _run_suffix(name: str) -> int:
    try:
        return int(name.rsplit("-r", 1)[1])
    except (IndexError, ValueError) as exc:
        raise InverseDynamicsTrainingError("INVERSE_DYNAMICS_RUN_NAME_INVALID") from exc


def _put_file(cas: ContentAddressedStore, path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise InverseDynamicsTrainingError("INVERSE_DYNAMICS_ARTIFACT_INVALID")
    size = path.stat().st_size
    if size <= 0 or size > 256 * 1024 * 1024:
        raise InverseDynamicsTrainingError("INVERSE_DYNAMICS_ARTIFACT_INVALID")
    media_type = "application/json" if path.suffix == ".json" else "application/vnd.pytorch"
    return cas.put_bytes(path.read_bytes(), media_type=media_type).uri


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="train one inverse-dynamics cache head")
    train.add_argument("--data-root", type=Path, required=True)
    train.add_argument("--environment", required=True)
    train.add_argument("--split", default="ind_train")
    train.add_argument("--output-root", type=Path, required=True)
    train.add_argument("--max-trajectories", type=int, default=24)
    train.add_argument("--frames-per-trajectory", type=int, default=8)
    train.add_argument("--image-size", type=int, default=32)
    train.add_argument("--hidden-dim", type=int, default=512)
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--batch-size", type=int, default=128)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--seed", type=int, default=7)
    train.add_argument("--device", default="cuda")
    train.add_argument("--gpu-index", type=int)
    train.add_argument("--gpu-exclusivity-audit-manifest", type=Path)
    train.add_argument("--gpu-exclusivity-max-age-seconds", type=float, default=300.0)
    summarize = commands.add_parser("summarize", help="summarize latest ready inverse-dynamics caches")
    summarize.add_argument("--runs-root", type=Path, required=True)
    summarize.add_argument("--output", type=Path)
    summarize.add_argument("--archive-db", type=Path)
    summarize.add_argument("--cas-root", type=Path)
    args = parser.parse_args(argv)
    if args.command == "train":
        manifest = train_inverse_dynamics_head(
            TrainingConfig(
                data_root=args.data_root,
                environment=args.environment,
                split=args.split,
                output_root=args.output_root,
                max_trajectories=args.max_trajectories,
                frames_per_trajectory=args.frames_per_trajectory,
                image_size=args.image_size,
                hidden_dim=args.hidden_dim,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                seed=args.seed,
                device=args.device,
                gpu_index=args.gpu_index,
                gpu_exclusivity_audit_manifest=args.gpu_exclusivity_audit_manifest,
                gpu_exclusivity_max_age_seconds=args.gpu_exclusivity_max_age_seconds,
            )
        )
        print(json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
        return 0
    if args.command == "summarize":
        summary = summarize_inverse_dynamics_runs(
            runs_root=args.runs_root,
            output_path=args.output,
            archive_db=args.archive_db,
            cas_root=args.cas_root,
        )
        print(json.dumps(summary, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
