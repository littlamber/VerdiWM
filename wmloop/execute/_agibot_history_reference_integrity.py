"""Bounded checks for restored Agibot H1 history-reference parameters."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from math import prod


@dataclass(frozen=True)
class HistoryReferenceIntegrityReport:
    """Finite/dead-weight findings from a bounded checkpoint scan."""

    checked_parameters: int
    non_finite: tuple[str, ...]
    dead_weights: tuple[str, ...]

    @property
    def compatible(self) -> bool:
        return not self.non_finite and not self.dead_weights


TensorChunkReader = Callable[[str, int | None, int | None], object]
FiniteCheck = Callable[[object], bool]
NonzeroCheck = Callable[[object], bool]


def should_scan_restored_history_reference(
    *,
    enabled: bool,
    null_enabled: bool,
    expected_names: Sequence[str],
    missing_names: Sequence[str],
) -> bool:
    """Match the trainer gate and distinguish a restored from absent branch."""

    if not enabled or null_enabled:
        return False
    expected = tuple(sorted(expected_names))
    missing = tuple(sorted(missing_names))
    if not expected:
        raise ValueError("enabled history reference has no parameters")
    if missing == expected:
        # Native loading initializes a wholly absent branch after materializing
        # the model; those initialized values are checked by the trainer gate.
        return False
    if missing:
        raise ValueError("history-reference checkpoint branch is only partially restored")
    return True


def inspect_agibot_history_reference_parameters(
    parameter_shapes: Mapping[str, Sequence[int]],
    *,
    read_chunk: TensorChunkReader,
    is_finite: FiniteCheck,
    has_nonzero: NonzeroCheck,
    max_chunk_elements: int = 1_000_000,
) -> HistoryReferenceIntegrityReport:
    """Mirror the native trainer gate while reading tensors in bounded slabs.

    The native gate requires every history-reference parameter to be finite.
    It additionally requires nonzero weights except the residual ``output``
    projections, which intentionally start at zero. Biases may also be zero.
    ``read_chunk`` should return CPU-backed slices and must raise on read errors.
    """

    if max_chunk_elements < 1:
        raise ValueError("max_chunk_elements must be positive")
    if not parameter_shapes:
        raise ValueError("enabled history reference has no parameters")

    non_finite: list[str] = []
    dead_weights: list[str] = []

    for name, raw_shape in sorted(parameter_shapes.items()):
        if not name.startswith("history_reference."):
            raise ValueError("history-reference parameter binding is invalid")
        shape = tuple(int(dimension) for dimension in raw_shape)
        if any(dimension < 0 for dimension in shape):
            raise ValueError("history-reference parameter shape is invalid")

        if not shape:
            chunks = ((None, None),)
        elif 0 in shape:
            # An empty named parameter cannot satisfy the native nonzero gate,
            # and is not a meaningful H1 parameter even when it is a bias.
            if _requires_nonzero_weight(name):
                dead_weights.append(name)
            continue
        else:
            row_elements = prod(shape[1:]) if len(shape) > 1 else 1
            rows_per_chunk = max(1, max_chunk_elements // row_elements)
            chunks = tuple(
                (start, min(shape[0], start + rows_per_chunk))
                for start in range(0, shape[0], rows_per_chunk)
            )

        finite = True
        found_nonzero = False
        for start, stop in chunks:
            values = read_chunk(name, start, stop)
            if not is_finite(values):
                finite = False
            if _requires_nonzero_weight(name) and not found_nonzero:
                found_nonzero = has_nonzero(values)

        if not finite:
            non_finite.append(name)
        if _requires_nonzero_weight(name) and not found_nonzero:
            dead_weights.append(name)

    return HistoryReferenceIntegrityReport(
        checked_parameters=len(parameter_shapes),
        non_finite=tuple(non_finite),
        dead_weights=tuple(dead_weights),
    )


def _requires_nonzero_weight(name: str) -> bool:
    return (
        name.endswith("weight")
        and not name.startswith("history_reference.output.")
    )
