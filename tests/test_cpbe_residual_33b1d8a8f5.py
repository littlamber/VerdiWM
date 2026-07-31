from __future__ import annotations

from wmloop.diagnose.probes.cpbe_residual_33b1d8a8f5 import measure_cpbe_residual


def test_source_sign_margin_is_schema_valid_and_target_label_free() -> None:
    output = measure_cpbe_residual(
        response_vectors={
            "target": [0.8, 0.2],
            "positive": [1.0, 0.0],
            "negative_a": [0.0, 1.0],
            "negative_b": [0.1, 0.9],
        },
        source_signs={"positive": 1, "negative_a": -1, "negative_b": -1},
        target="target",
    )
    assert output["probe_id"] == 'cpbe_residual_33b1d8a8f5'
    assert output["verdict_exposure_allowed"] is False
    audit = output["metrics"]["fit_audit"]
    assert audit["target_label_used_for_fit"] is False
    assert "target" not in audit["fit_environments"]
