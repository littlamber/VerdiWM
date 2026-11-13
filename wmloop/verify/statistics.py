"""Shared uncertainty estimates for paired verification contrasts."""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence


# Two-sided 95% Student t quantiles. Intermediate degrees of freedom use the
# next smaller entry, making the interval conservative without scipy.
_T95 = ((1, 12.7063), (2, 4.3027), (3, 3.1825), (4, 2.7765),
        (5, 2.5706), (6, 2.4469), (7, 2.3647), (8, 2.3061),
        (9, 2.2622), (10, 2.2282), (12, 2.1789), (15, 2.1315),
        (20, 2.0860), (30, 2.0423), (60, 2.0004), (120, 1.9800))


def student_t_mean_interval(values: Sequence[float]) -> dict[str, float]:
    """Estimate a mean and conservative two-sided 95% Student t interval.

    Values must be independent paired contrasts with an approximately normal
    sampling distribution. Repeated measurements within the same unit must be
    aggregated by the caller first. The interval does not establish provenance,
    independence or a correction for multiple comparisons.
    """
    try:
        if len(values) < 2 or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in values
        ):
            raise ValueError("VERIFICATION_STATISTICS_NUMERIC_INVALID")
        mean = statistics.fmean(values)
        se = statistics.stdev(values) / math.sqrt(len(values))
        critical = next(
            value for degrees, value in reversed(_T95) if degrees <= len(values) - 1
        ) + 0.0001
        lower, upper = mean - critical * se, mean + critical * se
    except (ArithmeticError, ValueError) as exc:
        raise ValueError("VERIFICATION_STATISTICS_NUMERIC_INVALID") from exc
    if not all(math.isfinite(value) for value in (mean, se, lower, upper)):
        raise ValueError("VERIFICATION_STATISTICS_NUMERIC_INVALID")
    return {"mean": mean, "standard_error": se, "lower_bound": lower, "upper_bound": upper}
