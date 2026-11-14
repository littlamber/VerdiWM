"""Shape-only mirror of the native Agibot warm-start compatibility rules."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


Shape = tuple[int, ...]


@dataclass(frozen=True)
class CheckpointArchitectureReport:
    """Bounded result of comparing checkpoint and initialized model tensors."""

    compatible: bool
    missing: tuple[str, ...]
    unexpected: tuple[str, ...]
    shape_mismatches: tuple[str, ...]
    compatible_expansions: tuple[str, ...]
    history_reference_missing: tuple[str, ...]
    history_reference_expected: tuple[str, ...]

    def summary(self, *, limit: int = 5) -> str:
        def group(name: str, keys: Sequence[str]) -> str:
            if not keys:
                return f"{name}=0"
            shown = ",".join(keys[:limit])
            suffix = ",…" if len(keys) > limit else ""
            return f"{name}={len(keys)}[{shown}{suffix}]"

        history = ""
        if self.history_reference_expected:
            missing = len(self.history_reference_missing)
            expected = len(self.history_reference_expected)
            history = f", history_reference_missing={missing}/{expected}"
        return "; ".join(
            (
                group("missing", self.missing),
                group("unexpected", self.unexpected),
                group("shape_mismatch", self.shape_mismatches),
                group("compatible_expansions", self.compatible_expansions),
            )
        ) + history


def compare_agibot_checkpoint_architecture(
    checkpoint_shapes: Mapping[str, Sequence[int]],
    target_shapes: Mapping[str, Sequence[int]],
    *,
    history_ref_enabled: bool,
) -> CheckpointArchitectureReport:
    """Apply the native loader's exact, optional, and narrow expansion rules.

    This does not load tensor values. The native trainer remains authoritative
    and repeats its strict `load_state_dict` checks when training starts.
    """

    checkpoint = {name: tuple(shape) for name, shape in checkpoint_shapes.items()}
    target = {name: tuple(shape) for name, shape in target_shapes.items()}
    unexpected = sorted(checkpoint.keys() - target.keys())
    compatible_expansions: list[str] = []
    mismatched_shapes: list[str] = []
    missing: list[str] = []

    for name, checkpoint_shape in checkpoint.items():
        target_shape = target.get(name)
        if target_shape is None:
            continue
        if checkpoint_shape == target_shape:
            continue
        if _warmstart_expansion_allowed(name, checkpoint_shape, target_shape):
            compatible_expansions.append(name)
            continue
        mismatched_shapes.append(name)
        missing.append(name)

    for name in target.keys() - checkpoint.keys():
        missing.append(name)
    missing.sort()
    mismatched_shapes.sort()
    compatible_expansions.sort()

    history_reference_expected = tuple(
        sorted(name for name in target if name.startswith("history_reference."))
    )
    history_reference_missing = tuple(
        name for name in missing if name.startswith("history_reference.")
    )

    if history_ref_enabled:
        # The native H1 restore accepts only an exact checkpoint, except that
        # the entire new reference branch may be absent. It rejects partial
        # reference restore, all other missing keys, and every expansion.
        reference_is_wholly_absent = (
            bool(history_reference_expected)
            and history_reference_missing == history_reference_expected
            and not any(
                name.startswith("history_reference.")
                for name in mismatched_shapes
            )
        )
        missing_allowed = not missing or (
            reference_is_wholly_absent
            and tuple(missing) == history_reference_expected
        )
        compatible = (
            missing_allowed
            and not unexpected
            and not mismatched_shapes
            and not compatible_expansions
        )
    else:
        compatible = (
            not any(not _warmstart_optional(name) for name in missing)
            and not any(not _warmstart_optional(name) for name in unexpected)
            and not any(not _warmstart_optional(name) for name in mismatched_shapes)
        )

    return CheckpointArchitectureReport(
        compatible=compatible,
        missing=tuple(missing),
        unexpected=tuple(unexpected),
        shape_mismatches=tuple(mismatched_shapes),
        compatible_expansions=tuple(compatible_expansions),
        history_reference_missing=history_reference_missing,
        history_reference_expected=history_reference_expected,
    )


def _warmstart_optional(name: str) -> bool:
    return (
        _action_side(name)
        or name.startswith("history_reference.")
        or name == "camera_id_embedding"
        or "prope_o" in name
    )


def _action_side(name: str) -> bool:
    return (
        "action_encoder" in name
        or "eef_projection_encoder" in name
        or "action_adaln_encoder" in name
        or "action_dense_encoder" in name
        or "action_film" in name
        or "action_injector_" in name
        or "action_scale_shift_layer" in name
        or "action_decoder" in name
        or "body_encoder" in name
        or ".k_action." in name
        or ".v_action." in name
        or "norm_k_action" in name
        or name.startswith("history_reference.")
    )


def _warmstart_expansion_allowed(
    name: str, checkpoint_shape: Shape, target_shape: Shape
) -> bool:
    if name == "patch_embedding.weight":
        return (
            len(checkpoint_shape) == len(target_shape) == 5
            and checkpoint_shape[0] == target_shape[0]
            and checkpoint_shape[2:] == target_shape[2:]
            and checkpoint_shape[1] < target_shape[1]
        )
    return (
        _action_side(name)
        and len(checkpoint_shape) > 0
        and len(checkpoint_shape) == len(target_shape)
        and checkpoint_shape[1:] == target_shape[1:]
        and checkpoint_shape[0] < target_shape[0]
        and ("delta_weight" in name or "delta_bias" in name)
    )
