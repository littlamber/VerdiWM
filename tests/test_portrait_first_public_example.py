from __future__ import annotations

from pathlib import Path

from scripts.export.validate_portrait_first_public_example import (
    validate_portrait_first_public_example,
)


ROOT = Path(__file__).resolve().parents[1]


def test_public_portrait_first_example_is_cpu_runnable() -> None:
    result = validate_portrait_first_public_example(
        ROOT / "examples" / "portrait_first_minimal_loop_v1"
    )

    assert result["state"] == "ready"
    assert result["readiness_state"] == "ready_for_gap_planning"
    assert result["gap_plan_state"] == "ready_for_portfolio"
    assert result["requirement_classification"] == "satisfied"
    assert result["authority"] == {
        "gpu_authority": False,
        "module_manufacturing_authority": False,
        "promotion_authority": False,
    }
