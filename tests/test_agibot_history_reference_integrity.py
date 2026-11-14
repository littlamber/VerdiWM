from __future__ import annotations

import math

import pytest

from wmloop.execute._agibot_history_reference_integrity import (
    inspect_agibot_history_reference_parameters,
    should_scan_restored_history_reference,
)


def _flat(values):
    for value in values:
        if isinstance(value, (list, tuple)):
            yield from _flat(value)
        else:
            yield value


def _scan(parameters, *, chunk_elements=1_000_000):
    calls = []

    def read_chunk(name, start, stop):
        calls.append((name, start, stop))
        shape, flat_values = parameters[name]
        if start is None:
            return flat_values
        width = math.prod(shape[1:]) if len(shape) > 1 else 1
        return flat_values[start * width : stop * width]

    report = inspect_agibot_history_reference_parameters(
        {name: shape for name, (shape, _values) in parameters.items()},
        read_chunk=read_chunk,
        is_finite=lambda values: all(math.isfinite(value) for value in _flat(values)),
        has_nonzero=lambda values: any(value != 0 for value in _flat(values)),
        max_chunk_elements=chunk_elements,
    )
    return report, calls


def test_restored_dead_query_weight_fails_closed():
    report, _calls = _scan(
        {
            "history_reference.query.0.weight": ((4, 4), [0.0] * 16),
            "history_reference.query.0.bias": ((4,), [0.0] * 4),
        }
    )

    assert not report.compatible
    assert report.dead_weights == ("history_reference.query.0.weight",)
    assert report.non_finite == ()


def test_finite_nonzero_qkv_passes_with_legal_zero_bias_and_output_init():
    report, _calls = _scan(
        {
            "history_reference.encoder.weight": ((2, 2), [0.5, -0.5, 1.0, 2.0]),
            "history_reference.position.weight": ((2, 3), [1.0] * 6),
            "history_reference.query.0.weight": ((2, 2), [1.0] * 4),
            "history_reference.query.0.bias": ((2,), [0.0, 0.0]),
            "history_reference.key.0.weight": ((2, 2), [1.0] * 4),
            "history_reference.value.0.weight": ((2, 2), [1.0] * 4),
            "history_reference.output.0.weight": ((2, 2), [0.0] * 4),
        }
    )

    assert report.compatible
    assert report.checked_parameters == 7
    assert report.dead_weights == ()


def test_every_parameter_is_checked_for_finite_values_in_bounded_row_chunks():
    report, calls = _scan(
        {
            "history_reference.query.0.weight": ((3, 4), [1.0] * 11 + [float("nan")]),
            "history_reference.query.0.bias": ((3,), [0.0, 0.0, 0.0]),
            "history_reference.output.0.weight": ((3, 4), [0.0] * 12),
        },
        chunk_elements=4,
    )

    assert not report.compatible
    assert report.non_finite == ("history_reference.query.0.weight",)
    assert calls == [
        ("history_reference.output.0.weight", 0, 1),
        ("history_reference.output.0.weight", 1, 2),
        ("history_reference.output.0.weight", 2, 3),
        ("history_reference.query.0.bias", 0, 3),
        ("history_reference.query.0.weight", 0, 1),
        ("history_reference.query.0.weight", 1, 2),
        ("history_reference.query.0.weight", 2, 3),
    ]


def test_history_reference_scan_matches_native_restore_gate():
    expected = (
        "history_reference.query.0.weight",
        "history_reference.query.0.bias",
    )

    assert not should_scan_restored_history_reference(
        enabled=False,
        null_enabled=False,
        expected_names=expected,
        missing_names=(),
    )
    assert not should_scan_restored_history_reference(
        enabled=True,
        null_enabled=True,
        expected_names=expected,
        missing_names=(),
    )
    assert not should_scan_restored_history_reference(
        enabled=True,
        null_enabled=False,
        expected_names=expected,
        missing_names=expected,
    )
    assert should_scan_restored_history_reference(
        enabled=True,
        null_enabled=False,
        expected_names=expected,
        missing_names=(),
    )


def test_partially_restored_history_reference_fails_closed():
    expected = (
        "history_reference.query.0.weight",
        "history_reference.query.0.bias",
    )

    with pytest.raises(ValueError, match="partially restored"):
        should_scan_restored_history_reference(
            enabled=True,
            null_enabled=False,
            expected_names=expected,
            missing_names=(expected[1],),
        )
