"""Intent-preserving Cosmos3 forward-dynamics hook adapters."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any


class Cosmos3HookError(ValueError):
    """A Cosmos3 hook is unavailable or would violate its intent contract."""


HOOK_ANCHORS = {
    "H1": (
        "cosmos_framework/data/vfm/action/datasets/droid_lerobot_dataset.py",
        ("DROIDLeRobotDataset",),
    ),
    "H2": (
        "cosmos_framework/inference/action.py",
        ("build_action_batch", "action_chunk_size"),
    ),
    "H3": (
        "cosmos_framework/model/vfm/vlm_model.py",
        ("init_optimizer_scheduler",),
    ),
    "H4": (
        "cosmos_framework/inference/inference.py",
        ("OmniInference",),
    ),
    "H5": (
        "cosmos_framework/trainer/__init__.py",
        ("optimizer", "scheduler"),
    ),
}


COSMOS3_ACTION_PROBE_DOSE_UNITS = {
    "action_conditioning_scale": "relative_action_input_scale",
    "action_dimension_anisotropy": "relative_action_dimension_contrast_balance",
    "action_dimension_interaction": "radians_action_dimension_coupling",
    "action_embedding_temporal_mix": "temporal_action_input_mix",
    "action_translation_scale": "relative_translation_action_scale",
    "cpbe_residual_63f088b0d5": "relative_action_embedding_phase_dose",
}

ACTION_DIMENSION_INTERACTION_PAIRS = ((0, 3), (1, 4), (2, 5), (6, 7))
ACTION_DIMENSION_INTERACTION_UNCOUPLED = (8, 9)


def audit_cosmos3_forward_dynamics_hooks(cosmos3_root: Path) -> dict[str, object]:
    root = Path(cosmos3_root).resolve(strict=True)
    rows: list[dict[str, object]] = []
    for hook, (relative, tokens) in HOOK_ANCHORS.items():
        path = root / relative
        content = path.read_text(encoding="utf-8") if path.is_file() else ""
        missing = [token for token in tokens if token not in content]
        rows.append(
            {
                "hook": hook,
                "path": str(path),
                "available": path.is_file() and not missing,
                "missing_tokens": missing,
            }
        )
    available = [str(row["hook"]) for row in rows if row["available"] is True]
    return {
        "schema_version": 1,
        "artifact_type": "verdiwm-cosmos3-hook-audit",
        "state": "ready" if len(available) == len(HOOK_ANCHORS) else "blocked",
        "available_hooks": available,
        "rows": rows,
    }


def apply_action_conditioning_scale(actions: Sequence[Sequence[float]], *, dose: float) -> list[list[float]]:
    """Apply a reversible global action-conditioning dose for fingerprinting."""
    value = float(dose)
    if not math.isfinite(value) or abs(value) > 0.1:
        raise Cosmos3HookError("COSMOS3_ACTION_DOSE_OUT_OF_RANGE")
    matrix = _action_matrix(actions)
    return [[entry * (1.0 + value) for entry in row] for row in matrix]


def apply_action_temporal_mix(actions: Sequence[Sequence[float]], *, dose: float) -> list[list[float]]:
    """Mix each action toward its trajectory mean without changing that mean."""
    value = float(dose)
    if not math.isfinite(value) or abs(value) > 0.1:
        raise Cosmos3HookError("COSMOS3_ACTION_DOSE_OUT_OF_RANGE")
    matrix = _action_matrix(actions)
    means = [sum(row[index] for row in matrix) / len(matrix) for index in range(len(matrix[0]))]
    return [
        [entry + value * (means[index] - entry) for index, entry in enumerate(row)]
        for row in matrix
    ]


def apply_action_translation_scale(
    actions: Sequence[Sequence[float]], *, dose: float
) -> list[list[float]]:
    """Scale only the DROID position-delta coordinates in its frozen 10D layout."""
    value = float(dose)
    if not math.isfinite(value) or abs(value) > 0.1:
        raise Cosmos3HookError("COSMOS3_ACTION_DOSE_OUT_OF_RANGE")
    matrix = _action_matrix(actions)
    if len(matrix[0]) != 10:
        raise Cosmos3HookError("COSMOS3_DROID_ACTION_LAYOUT_INVALID")
    return [
        [entry * (1.0 + value) if index < 3 else entry for index, entry in enumerate(row)]
        for row in matrix
    ]


def apply_action_dimension_anisotropy(
    actions: Sequence[Sequence[float]], *, dose: float
) -> list[list[float]]:
    """Dose per-dimension temporal contrast toward or away from balanced energy."""
    value = float(dose)
    if not math.isfinite(value) or abs(value) > 0.1:
        raise Cosmos3HookError("COSMOS3_ACTION_DOSE_OUT_OF_RANGE")
    matrix = _action_matrix(actions)
    if len(matrix[0]) != 10:
        raise Cosmos3HookError("COSMOS3_DROID_ACTION_LAYOUT_INVALID")
    means, gains = _dimension_contrast_parameters(matrix, dose=value)
    return [
        [means[index] + (entry - means[index]) * gains[index] for index, entry in enumerate(row)]
        for row in matrix
    ]


def apply_action_dimension_interaction(
    actions: Sequence[Sequence[float]], *, dose: float
) -> list[list[float]]:
    """Rotate centered DROID action pairs without changing means or energy."""
    value = float(dose)
    if not math.isfinite(value) or abs(value) > 0.1:
        raise Cosmos3HookError("COSMOS3_ACTION_DOSE_OUT_OF_RANGE")
    matrix = _action_matrix(actions)
    if len(matrix[0]) != 10:
        raise Cosmos3HookError("COSMOS3_DROID_ACTION_LAYOUT_INVALID")
    means = [sum(row[index] for row in matrix) / len(matrix) for index in range(10)]
    cosine = math.cos(value)
    sine = math.sin(value)
    transformed: list[list[float]] = []
    for row in matrix:
        output = list(row)
        for left, right in ACTION_DIMENSION_INTERACTION_PAIRS:
            left_centered = row[left] - means[left]
            right_centered = row[right] - means[right]
            output[left] = means[left] + cosine * left_centered - sine * right_centered
            output[right] = means[right] + sine * left_centered + cosine * right_centered
        transformed.append(output)
    return transformed


def apply_action_probe(
    actions: Sequence[Sequence[float]], *, probe_id: str, dose: float
) -> list[list[float]]:
    if probe_id == "action_conditioning_scale":
        return apply_action_conditioning_scale(actions, dose=dose)
    if probe_id == "action_dimension_anisotropy":
        return apply_action_dimension_anisotropy(actions, dose=dose)
    if probe_id == "action_dimension_interaction":
        return apply_action_dimension_interaction(actions, dose=dose)
    if probe_id == "action_embedding_temporal_mix":
        return apply_action_temporal_mix(actions, dose=dose)
    if probe_id == "action_translation_scale":
        return apply_action_translation_scale(actions, dose=dose)
    if probe_id == "cpbe_residual_63f088b0d5":
        value = float(dose)
        if value not in {-0.05, -0.025, 0.0, 0.025, 0.05}:
            raise Cosmos3HookError("COSMOS3_CPBE_DOSE_GRID_MISMATCH")
        return _action_matrix(actions)
    raise Cosmos3HookError("COSMOS3_ACTION_MODE_UNKNOWN")


def cosmos3_probe_dose_unit(probe_id: str) -> str:
    try:
        return COSMOS3_ACTION_PROBE_DOSE_UNITS[probe_id]
    except KeyError as exc:
        raise Cosmos3HookError("COSMOS3_ACTION_MODE_UNKNOWN") from exc


class Cosmos3CPBEActionEmbeddingDeltaDose:
    """Apply the exact r31 phase dose to one packed action trajectory."""

    DOSES = (-0.05, -0.025, 0.0, 0.025, 0.05)

    def __init__(self, model: object, *, dose: float, expected_token_count: int) -> None:
        action2llm = getattr(model, "action2llm", None)
        forward = getattr(action2llm, "forward", None)
        if action2llm is None or not callable(forward):
            raise Cosmos3HookError("COSMOS3_ACTION_EMBEDDING_HOOK_MISSING")
        if float(dose) not in self.DOSES:
            raise Cosmos3HookError("COSMOS3_CPBE_DOSE_GRID_MISMATCH")
        if int(expected_token_count) < 2:
            raise Cosmos3HookError("COSMOS3_CPBE_TOKEN_COUNT_INVALID")
        self.action2llm = action2llm
        self.original = forward
        self.dose = float(dose)
        self.expected_token_count = int(expected_token_count)
        self.audit: dict[str, object] = {
            "invocation_count": 0,
            "maximum_temporal_mean_abs_error": 0.0,
            "maximum_temporal_mean_tolerance": 0.0,
            "expected_token_count": self.expected_token_count,
        }

    def __enter__(self) -> "Cosmos3CPBEActionEmbeddingDeltaDose":
        from types import MethodType

        import torch

        original = self.original
        dose = self.dose
        expected = self.expected_token_count
        audit = self.audit

        def phase_forward(_module: object, *args: object, **kwargs: object) -> object:
            embedding = original(*args, **kwargs)
            if getattr(embedding, "ndim", None) != 2 or int(embedding.shape[0]) != expected:
                raise Cosmos3HookError("COSMOS3_CPBE_PACKED_EMBEDDING_SHAPE_INVALID")
            audit["invocation_count"] = int(audit["invocation_count"]) + 1
            if dose == 0.0:
                return embedding
            delta = embedding[1:] - embedding[:-1]
            event_weight = torch.cat(
                (delta[:1].abs().mean(dim=-1, keepdim=True) * 0.0, delta.abs().mean(dim=-1, keepdim=True)),
                dim=0,
            ).squeeze(-1)
            scale = event_weight.amax()
            event_weight = event_weight / scale.clamp_min(1e-12)
            event_weight = event_weight * (scale > 1e-12).to(event_weight.dtype)
            previous = torch.cat((event_weight[:1] * 0.0, event_weight[:-1]), dim=0)
            following = torch.cat((event_weight[1:], event_weight[:1] * 0.0), dim=0)
            tangent = 0.5 * (following - previous)
            temporal_mean = embedding.mean(dim=0, keepdim=True)
            perturbation = tangent.unsqueeze(-1).to(embedding.dtype) * (embedding - temporal_mean)
            perturbation = perturbation - perturbation.mean(dim=0, keepdim=True)
            output = embedding + dose * perturbation
            error = float((output.mean(dim=0) - temporal_mean.squeeze(0)).abs().max().item())
            mean_scale = max(1.0, float(temporal_mean.abs().amax().item()))
            tolerance = float(torch.finfo(embedding.dtype).eps) * mean_scale
            audit["maximum_temporal_mean_abs_error"] = max(
                float(audit["maximum_temporal_mean_abs_error"]), error
            )
            audit["maximum_temporal_mean_tolerance"] = max(
                float(audit["maximum_temporal_mean_tolerance"]), tolerance
            )
            return output

        self.action2llm.forward = MethodType(phase_forward, self.action2llm)
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        self.action2llm.forward = self.original
        return False


def balance_action_dimensions(
    actions: Sequence[Sequence[float]],
    *,
    blend: float,
    max_gain: float,
    epsilon: float = 1e-6,
) -> tuple[list[list[float]], list[float]]:
    """Rebalance action dimensions by bounded inverse-RMS gains at H2."""
    matrix = _action_matrix(actions)
    blend_value = float(blend)
    max_gain_value = float(max_gain)
    if not 0.0 < blend_value <= 1.0 or max_gain_value < 1.0 or not math.isfinite(max_gain_value):
        raise Cosmos3HookError("COSMOS3_ACTION_BALANCE_CONFIG_INVALID")
    width = len(matrix[0])
    rms = [math.sqrt(sum(row[index] ** 2 for row in matrix) / len(matrix)) for index in range(width)]
    active = [value for value in rms if value > epsilon]
    reference = sorted(active)[len(active) // 2] if active else 1.0
    raw_gains = [reference / max(value, epsilon) for value in rms]
    gains = [1.0 + blend_value * (min(max(gain, 1.0 / max_gain_value), max_gain_value) - 1.0) for gain in raw_gains]
    return ([[entry * gains[index] for index, entry in enumerate(row)] for row in matrix], gains)


def materialize_action_json(
    *,
    source: Path,
    destination: Path,
    mode: str,
    dose: float = 0.0,
    blend: float = 1.0,
    max_gain: float = 4.0,
) -> dict[str, object]:
    source_path = Path(source).resolve(strict=True)
    destination_path = Path(destination).resolve()
    if destination_path.exists() or destination_path.is_symlink():
        raise Cosmos3HookError("COSMOS3_ACTION_OUTPUT_EXISTS")
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if mode in COSMOS3_ACTION_PROBE_DOSE_UNITS:
        transformed = apply_action_probe(payload, probe_id=mode, dose=dose)
        parameters: dict[str, Any] = {"dose": float(dose)}
        gains = (
            _dimension_contrast_parameters(_action_matrix(payload), dose=float(dose))[1]
            if mode == "action_dimension_anisotropy"
            else None
        )
    elif mode == "action_dimension_balancing":
        transformed, gains = balance_action_dimensions(payload, blend=blend, max_gain=max_gain)
        parameters = {"blend": float(blend), "max_gain": float(max_gain)}
    else:
        raise Cosmos3HookError("COSMOS3_ACTION_MODE_UNKNOWN")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    zero_dose = mode in COSMOS3_ACTION_PROBE_DOSE_UNITS and float(dose) == 0.0
    if zero_dose:
        shutil.copyfile(source_path, destination_path)
    else:
        destination_path.write_text(json.dumps(transformed, separators=(",", ":")) + "\n", encoding="utf-8")
    return {
        "schema_version": 1,
        "artifact_type": "verdiwm-cosmos3-action-hook-receipt",
        "mode": mode,
        "parameters": parameters,
        "source_sha256": _sha256(source_path),
        "output_sha256": _sha256(destination_path),
        "shape": [len(transformed), len(transformed[0])],
        "gains": gains,
        "coupling_pairs": (
            [list(pair) for pair in ACTION_DIMENSION_INTERACTION_PAIRS]
            if mode == "action_dimension_interaction"
            else None
        ),
        "dose_unit": COSMOS3_ACTION_PROBE_DOSE_UNITS.get(mode),
        "reversible": mode in COSMOS3_ACTION_PROBE_DOSE_UNITS and float(dose) != 1.0,
        "temporal_mean_max_abs_error": (
            _temporal_mean_max_abs_error(matrix=_action_matrix(payload), transformed=transformed)
            if mode in {
                "action_embedding_temporal_mix",
                "action_dimension_anisotropy",
                "action_dimension_interaction",
            }
            else None
        ),
        "action_dimension_centered_energy_relative_error": (
            _centered_energy_relative_error(
                matrix=_action_matrix(payload), transformed=transformed
            )
            if mode == "action_dimension_interaction"
            else None
        ),
        "unchanged_uncoupled_max_abs_error": (
            _unchanged_indices_max_abs_error(
                matrix=_action_matrix(payload),
                transformed=transformed,
                indices=ACTION_DIMENSION_INTERACTION_UNCOUPLED,
            )
            if mode == "action_dimension_interaction"
            else None
        ),
        "action_dimension_log_energy_spread_before": (
            _temporal_contrast_log_spread(_action_matrix(payload))
            if mode == "action_dimension_anisotropy"
            else None
        ),
        "action_dimension_log_energy_spread_after": (
            _temporal_contrast_log_spread(transformed)
            if mode == "action_dimension_anisotropy"
            else None
        ),
        "unchanged_nontranslation_max_abs_error": (
            _unchanged_columns_max_abs_error(
                matrix=_action_matrix(payload), transformed=transformed, start_index=3
            )
            if mode == "action_translation_scale"
            else None
        ),
        "zero_dose_byte_identity": zero_dose,
    }


def _action_matrix(actions: Sequence[Sequence[float]]) -> list[list[float]]:
    matrix = [[float(value) for value in row] for row in actions]
    if not matrix or not matrix[0] or any(len(row) != len(matrix[0]) for row in matrix):
        raise Cosmos3HookError("COSMOS3_ACTION_SHAPE_INVALID")
    if any(not math.isfinite(value) for row in matrix for value in row):
        raise Cosmos3HookError("COSMOS3_ACTION_NONFINITE")
    return matrix


def _temporal_mean_max_abs_error(
    *, matrix: Sequence[Sequence[float]], transformed: Sequence[Sequence[float]]
) -> float:
    width = len(matrix[0])
    return max(
        abs(
            sum(row[index] for row in matrix) / len(matrix)
            - sum(row[index] for row in transformed) / len(transformed)
        )
        for index in range(width)
    )


def _dimension_contrast_parameters(
    matrix: Sequence[Sequence[float]], *, dose: float, epsilon: float = 1e-8
) -> tuple[list[float], list[float]]:
    width = len(matrix[0])
    means = [sum(row[index] for row in matrix) / len(matrix) for index in range(width)]
    energies = [
        math.sqrt(sum((row[index] - means[index]) ** 2 for row in matrix) / len(matrix))
        for index in range(width)
    ]
    active = [index for index, energy in enumerate(energies) if energy > epsilon]
    scores = [0.0] * width
    if active:
        logs = {index: math.log(energies[index]) for index in active}
        center = sum(logs.values()) / len(logs)
        raw = {index: center - value for index, value in logs.items()}
        scale = max(abs(value) for value in raw.values())
        if scale > epsilon:
            for index, value in raw.items():
                scores[index] = value / scale
    gains = [1.0 + dose * score for score in scores]
    if any(gain <= 0.0 or not math.isfinite(gain) for gain in gains):
        raise Cosmos3HookError("COSMOS3_ACTION_DIMENSION_GAIN_INVALID")
    return means, gains


def _temporal_contrast_log_spread(
    matrix: Sequence[Sequence[float]], *, epsilon: float = 1e-8
) -> float:
    width = len(matrix[0])
    means = [sum(row[index] for row in matrix) / len(matrix) for index in range(width)]
    logs = [
        math.log(energy)
        for index in range(width)
        if (energy := math.sqrt(
            sum((row[index] - means[index]) ** 2 for row in matrix) / len(matrix)
        ))
        > epsilon
    ]
    return max(logs) - min(logs) if len(logs) >= 2 else 0.0


def _centered_energy_relative_error(
    *, matrix: Sequence[Sequence[float]], transformed: Sequence[Sequence[float]]
) -> float:
    width = len(matrix[0])
    source_means = [sum(row[index] for row in matrix) / len(matrix) for index in range(width)]
    transformed_means = [
        sum(row[index] for row in transformed) / len(transformed) for index in range(width)
    ]
    source_energy = sum(
        (row[index] - source_means[index]) ** 2
        for row in matrix
        for index in range(width)
    )
    transformed_energy = sum(
        (row[index] - transformed_means[index]) ** 2
        for row in transformed
        for index in range(width)
    )
    return abs(transformed_energy - source_energy) / max(source_energy, 1e-12)


def _unchanged_indices_max_abs_error(
    *,
    matrix: Sequence[Sequence[float]],
    transformed: Sequence[Sequence[float]],
    indices: Sequence[int],
) -> float:
    return max(
        abs(row[index] - transformed_row[index])
        for row, transformed_row in zip(matrix, transformed, strict=True)
        for index in indices
    )


def _unchanged_columns_max_abs_error(
    *,
    matrix: Sequence[Sequence[float]],
    transformed: Sequence[Sequence[float]],
    start_index: int,
) -> float:
    return max(
        abs(row[index] - transformed_row[index])
        for row, transformed_row in zip(matrix, transformed, strict=True)
        for index in range(start_index, len(row))
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
