#!/usr/bin/env python3
"""Compatibility CLI; implementation lives in the installable VerdiWM package."""

from __future__ import annotations

from wmloop.experiments.formal_visualization import (
    AcwmFormalVisualizationError,
    main,
    rank_baseline_hard_cases,
    run_export,
)

if __name__ == "__main__":
    main()
