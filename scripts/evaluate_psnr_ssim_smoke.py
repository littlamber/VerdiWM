#!/usr/bin/env python3
"""Dependency-light paired PSNR/SSIM evaluator for local Verdi smoke runs.

The evaluator intentionally measures only paired frame fidelity. It never imports a
model, starts CUDA, or writes to an input directory. A baseline receipt may be
provided to classify the measured change as POSITIVE, NULL, or HARMFUL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np

try:
    from wmloop.contracts import ContractValidationError, validate_document
except ImportError:  # pragma: no cover - allows a copied standalone evaluator to explain its error
    ContractValidationError = ValueError  # type: ignore[assignment,misc]
    validate_document = None  # type: ignore[assignment]


class PsnrSsimError(ValueError):
    """The paired evaluator input or output contract is invalid."""


_SUPPORTED_SUFFIXES = {".npy", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
_RECEIPT_ARTIFACT = "verdiwm-psnr-ssim-evidence-receipt"
_EVALUATOR_ID = "wan22-droid-psnr-ssim-smoke-v1"
_CLAIM_BOUNDARY = (
    "Local paired PSNR/SSIM evaluator smoke only; this does not establish "
    "Wan/DROID improvement, minute-level consistency, or transfer validity."
)


def evaluate(
    *,
    predicted_dir: Path,
    ground_truth_dir: Path,
    output_root: Path,
    contract_path: Path | None = None,
    baseline_receipt: Path | None = None,
    data_range: float = 1.0,
    null_threshold: float = 1e-6,
) -> dict[str, object]:
    """Evaluate paired frames and write a deterministic receipt bundle."""

    predicted = _regular_directory(predicted_dir, "PREDICTED_DIR_INVALID")
    ground_truth = _regular_directory(ground_truth_dir, "GROUND_TRUTH_DIR_INVALID")
    if data_range <= 0 or not math.isfinite(data_range):
        raise PsnrSsimError("DATA_RANGE_INVALID")
    if null_threshold < 0 or not math.isfinite(null_threshold):
        raise PsnrSsimError("NULL_THRESHOLD_INVALID")
    destination_input = Path(output_root).expanduser()
    if destination_input.is_symlink():
        raise PsnrSsimError("OUTPUT_ROOT_EXISTS")
    destination = destination_input.resolve()
    _check_output_boundary(destination, (predicted, ground_truth))
    if destination.exists() or destination.is_symlink():
        raise PsnrSsimError("OUTPUT_ROOT_EXISTS")

    pairs = _paired_files(predicted, ground_truth)
    frame_rows: list[dict[str, object]] = []
    for frame_id, prediction_path, target_path in pairs:
        prediction = _load_frame(prediction_path)
        target = _load_frame(target_path)
        if prediction.shape != target.shape:
            raise PsnrSsimError(f"FRAME_SHAPE_MISMATCH:{frame_id}")
        _validate_frame(prediction, frame_id)
        _validate_frame(target, frame_id)
        prediction_float = prediction.astype(np.float64, copy=False)
        target_float = target.astype(np.float64, copy=False)
        mse = float(np.mean(np.square(prediction_float - target_float), dtype=np.float64))
        psnr = math.inf if mse == 0.0 else 10.0 * math.log10((data_range * data_range) / mse)
        ssim = _ssim(prediction_float, target_float, data_range=data_range)
        frame_rows.append(
            {
                "frame_id": frame_id,
                "psnr_db": None if math.isinf(psnr) else round(psnr, 12),
                "psnr_is_infinite": bool(math.isinf(psnr)),
                "ssim": round(ssim, 12),
                "mse": round(mse, 16),
            }
        )

    aggregate_mse = float(np.mean([float(row["mse"]) for row in frame_rows], dtype=np.float64))
    aggregate_psnr = math.inf if aggregate_mse == 0.0 else 10.0 * math.log10((data_range * data_range) / aggregate_mse)
    aggregate_ssim = float(np.mean([float(row["ssim"]) for row in frame_rows]))
    baseline = _load_baseline(baseline_receipt) if baseline_receipt is not None else None
    comparison = _comparison(
        aggregate_psnr=aggregate_psnr,
        aggregate_ssim=aggregate_ssim,
        baseline=baseline,
        threshold=null_threshold,
    )

    contract_digest = _contract_digest(contract_path)
    input_artifacts = {
        "predicted_frames": _directory_binding(predicted),
        "paired_ground_truth_frames": _directory_binding(ground_truth),
    }
    body: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": _RECEIPT_ARTIFACT,
        "evaluator_id": _EVALUATOR_ID,
        "evaluator_contract_sha256": contract_digest,
        "state": "ready",
        "verdict": comparison["verdict"],
        "metrics": {
            "psnr": None if math.isinf(aggregate_psnr) else round(aggregate_psnr, 12),
            "psnr_is_infinite": bool(math.isinf(aggregate_psnr)),
            "ssim": round(aggregate_ssim, 12),
            "frame_count": len(frame_rows),
            "data_range": data_range,
        },
        "per_frame": frame_rows,
        "comparison": comparison,
        "input_artifacts": input_artifacts,
        "baseline_receipt": baseline["ref"] if baseline is not None else None,
        "side_effects": {
            "gpu_execution_started": False,
            "source_modified": False,
            "input_modified": False,
        },
        "claim_boundary": _CLAIM_BOUNDARY,
    }
    body["receipt_id"] = "psnr-ssim-receipt-" + _digest(body)[:24]
    if validate_document is not None:
        try:
            validate_document("psnr_ssim_evidence_receipt", body)
        except ContractValidationError as exc:
            raise PsnrSsimError(f"RECEIPT_SCHEMA_INVALID:{exc}") from exc
    _write_bundle(destination, body)
    return {
        "state": "ready",
        "receipt_id": body["receipt_id"],
        "receipt_path": str(destination / "psnr-ssim-receipt.json"),
        "verdict": body["verdict"],
        "metrics": body["metrics"],
        "evaluator_contract_sha256": contract_digest,
        "gpu_execution_started": False,
        "source_modified": False,
        "claim_boundary": _CLAIM_BOUNDARY,
    }


def verify_receipt(path: Path, *, contract_path: Path | None = None) -> dict[str, object]:
    """Verify schema, receipt identity, and input bindings without executing code."""

    source_input = Path(path).expanduser()
    if source_input.is_symlink():
        raise PsnrSsimError("RECEIPT_MISSING")
    source = source_input.resolve()
    if not source.is_file():
        raise PsnrSsimError("RECEIPT_MISSING")
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PsnrSsimError("RECEIPT_INVALID_JSON") from exc
    if not isinstance(document, dict):
        raise PsnrSsimError("RECEIPT_OBJECT_REQUIRED")
    if validate_document is not None:
        try:
            validate_document("psnr_ssim_evidence_receipt", document)
        except ContractValidationError as exc:
            raise PsnrSsimError(f"RECEIPT_SCHEMA_INVALID:{exc}") from exc
    body = dict(document)
    received = body.pop("receipt_id", None)
    if received != "psnr-ssim-receipt-" + _digest(body)[:24]:
        raise PsnrSsimError("RECEIPT_ID_MISMATCH")
    if contract_path is not None and document.get("evaluator_contract_sha256") != _contract_digest(contract_path):
        raise PsnrSsimError("RECEIPT_EVALUATOR_DRIFT")
    if document.get("side_effects", {}).get("gpu_execution_started") is not False:
        raise PsnrSsimError("RECEIPT_GPU_SIDE_EFFECT")
    if document.get("side_effects", {}).get("source_modified") is not False:
        raise PsnrSsimError("RECEIPT_SOURCE_SIDE_EFFECT")
    for binding in document.get("input_artifacts", {}).values():
        if not isinstance(binding, Mapping):
            raise PsnrSsimError("RECEIPT_INPUT_BINDING_INVALID")
        observed = _directory_binding(Path(str(binding["path"])))
        if observed != dict(binding):
            raise PsnrSsimError("RECEIPT_INPUT_DRIFT")
    return document


def _comparison(*, aggregate_psnr: float, aggregate_ssim: float, baseline: Mapping[str, object] | None, threshold: float) -> dict[str, object]:
    if baseline is None:
        return {"verdict": "MEASURED", "delta_psnr": None, "delta_ssim": None, "threshold": threshold}
    base_metrics = baseline.get("metrics")
    if not isinstance(base_metrics, Mapping):
        raise PsnrSsimError("BASELINE_METRICS_INVALID")
    base_psnr = _metric_value(base_metrics.get("psnr"), base_metrics.get("psnr_is_infinite"))
    base_ssim = float(base_metrics.get("ssim"))
    candidate_psnr = _metric_value(None if math.isinf(aggregate_psnr) else aggregate_psnr, math.isinf(aggregate_psnr))
    if math.isinf(candidate_psnr) and not math.isinf(base_psnr):
        delta_psnr = math.inf
    elif math.isinf(base_psnr) and not math.isinf(candidate_psnr):
        delta_psnr = -math.inf
    elif math.isinf(candidate_psnr) and math.isinf(base_psnr):
        delta_psnr = 0.0
    else:
        delta_psnr = candidate_psnr - base_psnr
    delta_ssim = aggregate_ssim - base_ssim
    if delta_psnr >= threshold and delta_ssim >= threshold:
        verdict = "POSITIVE"
    elif delta_psnr <= -threshold and delta_ssim <= -threshold:
        verdict = "HARMFUL"
    else:
        verdict = "NULL"
    return {
        "verdict": verdict,
        "delta_psnr": None if math.isinf(delta_psnr) else round(delta_psnr, 12),
        "delta_psnr_is_infinite": bool(math.isinf(delta_psnr)),
        "delta_ssim": round(delta_ssim, 12),
        "baseline_receipt_id": baseline.get("receipt_id"),
        "threshold": threshold,
    }


def _metric_value(value: object, infinite: object) -> float:
    if infinite is True:
        return math.inf
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise PsnrSsimError("BASELINE_PSNR_INVALID") from exc
    if not math.isfinite(result):
        raise PsnrSsimError("BASELINE_PSNR_NONFINITE")
    return result


def _load_baseline(path: Path) -> dict[str, object]:
    document = verify_receipt(path)
    return {"ref": "sha256:" + _sha256(path), **document}


def _ssim(prediction: np.ndarray, target: np.ndarray, *, data_range: float) -> float:
    x = prediction.reshape(-1)
    y = target.reshape(-1)
    mean_x = float(np.mean(x))
    mean_y = float(np.mean(y))
    variance_x = float(np.var(x))
    variance_y = float(np.var(y))
    covariance = float(np.mean((x - mean_x) * (y - mean_y)))
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    value = ((2 * mean_x * mean_y + c1) * (2 * covariance + c2)) / ((mean_x * mean_x + mean_y * mean_y + c1) * (variance_x + variance_y + c2))
    return float(max(-1.0, min(1.0, value)))


def _paired_files(predicted: Path, ground_truth: Path) -> list[tuple[str, Path, Path]]:
    left = _index_frames(predicted)
    right = _index_frames(ground_truth)
    if not left or not right:
        raise PsnrSsimError("PAIRED_FRAMES_EMPTY")
    if set(left) != set(right):
        missing_left = sorted(set(right) - set(left))
        missing_right = sorted(set(left) - set(right))
        raise PsnrSsimError(f"PAIRED_FRAMES_MISMATCH:left_missing={missing_left}:right_missing={missing_right}")
    return [(frame_id, left[frame_id], right[frame_id]) for frame_id in sorted(left)]


def _index_frames(directory: Path) -> dict[str, Path]:
    indexed: dict[str, Path] = {}
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.is_symlink() or path.suffix.lower() not in _SUPPORTED_SUFFIXES:
            continue
        frame_id = path.relative_to(directory).with_suffix("").as_posix()
        if frame_id in indexed:
            raise PsnrSsimError(f"FRAME_ID_DUPLICATE:{frame_id}")
        indexed[frame_id] = path.resolve()
    return indexed


def _load_frame(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        try:
            value = np.load(path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise PsnrSsimError(f"FRAME_READ_FAILED:{path.name}") from exc
    else:
        try:
            from PIL import Image

            with Image.open(path) as image:
                value = np.asarray(image)
        except (OSError, ValueError) as exc:
            raise PsnrSsimError(f"FRAME_READ_FAILED:{path.name}") from exc
    if not isinstance(value, np.ndarray) or value.ndim < 2 or value.size == 0:
        raise PsnrSsimError(f"FRAME_ARRAY_INVALID:{path.name}")
    return value


def _validate_frame(value: np.ndarray, frame_id: str) -> None:
    if not np.issubdtype(value.dtype, np.number):
        raise PsnrSsimError(f"FRAME_DTYPE_INVALID:{frame_id}")
    if not np.isfinite(value.astype(np.float64, copy=False)).all():
        raise PsnrSsimError(f"FRAME_NONFINITE:{frame_id}")


def _directory_binding(path: Path) -> dict[str, object]:
    directory = _regular_directory(path, "INPUT_BINDING_DIRECTORY_INVALID")
    digest = hashlib.sha256()
    size = 0
    count = 0
    for member in sorted(directory.rglob("*")):
        if not member.is_file() or member.is_symlink():
            continue
        relative = member.relative_to(directory).as_posix().encode("utf-8")
        payload = member.read_bytes()
        digest.update(relative + b"\0" + hashlib.sha256(payload).digest())
        size += len(payload)
        count += 1
    return {"path": str(directory), "sha256": digest.hexdigest(), "size_bytes": size, "file_count": count}


def _regular_directory(path: Path, code: str) -> Path:
    source = Path(path).expanduser()
    if source.is_symlink():
        raise PsnrSsimError(code)
    resolved = source.resolve()
    if not resolved.is_dir():
        raise PsnrSsimError(code)
    return resolved


def _check_output_boundary(destination: Path, inputs: tuple[Path, ...]) -> None:
    for directory in inputs:
        try:
            destination.relative_to(directory)
        except ValueError:
            pass
        else:
            raise PsnrSsimError("OUTPUT_INPUT_OVERLAP")
        try:
            directory.relative_to(destination)
        except ValueError:
            continue
        raise PsnrSsimError("OUTPUT_INPUT_OVERLAP")


def _contract_digest(path: Path | None) -> str | None:
    if path is None:
        return None
    source_input = Path(path).expanduser()
    if source_input.is_symlink():
        raise PsnrSsimError("EVALUATOR_CONTRACT_INVALID")
    source = source_input.resolve()
    if not source.is_file():
        raise PsnrSsimError("EVALUATOR_CONTRACT_INVALID")
    try:
        contract = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PsnrSsimError("EVALUATOR_CONTRACT_INVALID") from exc
    if not isinstance(contract, Mapping) or contract.get("evaluator_id") != _EVALUATOR_ID or contract.get("metrics") != ["psnr", "ssim"]:
        raise PsnrSsimError("EVALUATOR_CONTRACT_MISMATCH")
    return _sha256(source)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _write_bundle(destination: Path, receipt: Mapping[str, object]) -> None:
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.tmp"
    if temporary.exists() or temporary.is_symlink():
        raise PsnrSsimError("OUTPUT_TEMP_EXISTS")
    try:
        temporary.mkdir(mode=0o700)
        payload = json.dumps(receipt, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False) + "\n"
        (temporary / "psnr-ssim-receipt.json").write_text(payload, encoding="utf-8")
        (temporary / "psnr-ssim-metrics.json").write_text(json.dumps(receipt["metrics"], ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        (temporary / "evaluator_stdout.log").write_text("", encoding="utf-8")
        (temporary / "evaluator_stderr.log").write_text("", encoding="utf-8")
        os.replace(temporary, destination)
    except Exception:
        if temporary.exists() or temporary.is_symlink():
            import shutil

            shutil.rmtree(temporary, ignore_errors=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predicted-dir", type=Path)
    parser.add_argument("--ground-truth-dir", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--baseline-receipt", type=Path)
    parser.add_argument("--data-range", type=float, default=1.0)
    parser.add_argument("--null-threshold", type=float, default=1e-6)
    parser.add_argument("--verify-receipt", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.verify_receipt is not None:
            result = verify_receipt(args.verify_receipt, contract_path=args.contract)
            print(json.dumps({"state": "verified", "receipt_id": result["receipt_id"]}, ensure_ascii=True, sort_keys=True))
            return 0
        if args.predicted_dir is None or args.ground_truth_dir is None or args.output_root is None:
            raise PsnrSsimError("PREDICTED_DIR_GROUND_TRUTH_DIR_OUTPUT_ROOT_REQUIRED")
        result = evaluate(
            predicted_dir=args.predicted_dir,
            ground_truth_dir=args.ground_truth_dir,
            output_root=args.output_root,
            contract_path=args.contract,
            baseline_receipt=args.baseline_receipt,
            data_range=args.data_range,
            null_threshold=args.null_threshold,
        )
        print(json.dumps(result, ensure_ascii=True, sort_keys=True))
        return 0
    except PsnrSsimError as exc:
        print(json.dumps({"state": "blocked", "error": str(exc)}, ensure_ascii=True, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
