"""Generated CPBE source-sign aggregation for cpbe_residual_33b1d8a8f5."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from wmloop.diagnose.probes.source_sign_margin import measure_source_sign_margin

PROBE_ID = 'cpbe_residual_33b1d8a8f5'
ENVIRONMENT = 'push_cube'
SIGNATURE = 'mixed_source_sign_positive_prediction_vs_negative_target'


def measure_cpbe_residual(
    *,
    response_vectors: Mapping[str, Sequence[float]],
    source_signs: Mapping[str, int],
    target: str,
    evidence_refs: Sequence[str] = (),
) -> dict[str, object]:
    return measure_source_sign_margin(
        probe_id=PROBE_ID,
        environment=ENVIRONMENT,
        signature=SIGNATURE,
        response_vectors=response_vectors,
        source_signs=source_signs,
        target=target,
        evidence_refs=evidence_refs,
    )
