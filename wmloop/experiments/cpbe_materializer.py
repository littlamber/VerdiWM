"""Deterministically materialize supported CPBE work orders into tested code."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from wmloop.experiments._artifacts import canonical_json, write_bundle


class CPBEMaterializerError(ValueError):
    """A work order cannot be lowered without changing its declared intent."""


def publish_cpbe_materialization(
    *,
    plan_path: Path,
    output_root: Path,
) -> dict[str, object]:
    plan_file = Path(plan_path).resolve(strict=True)
    plan = _load_object(plan_file)
    if plan.get("artifact_type") != "verdiwm-cpbe-plan" or plan.get("state") != "ready":
        raise CPBEMaterializerError("CPBE_MATERIALIZER_PLAN_INVALID")
    orders = _mapping_sequence(plan.get("selected_work_orders"), "CPBE_MATERIALIZER_ORDERS_INVALID")
    if not orders:
        raise CPBEMaterializerError("CPBE_MATERIALIZER_ORDERS_EMPTY")

    files: dict[str, bytes] = {"source/cpbe-plan.json": plan_file.read_bytes()}
    rows: list[dict[str, object]] = []
    for order in orders:
        probe_id = _text(order, "probe_id")
        program = _mapping(order, "program")
        _validate_supported(order=order, program=program)
        module_path = f"wmloop/diagnose/probes/{probe_id}.py"
        test_path = f"tests/test_{probe_id}.py"
        descriptor_path = f"configs/probes/staging/{probe_id}.json"
        payloads = {
            module_path: _module_source(order).encode("utf-8"),
            test_path: _test_source(order).encode("utf-8"),
            descriptor_path: canonical_json(_descriptor(order)),
        }
        files.update(payloads)
        rows.append(
            {
                "probe_id": probe_id,
                "materialization_template": _materialization_template(program),
                "program_sha256": hashlib.sha256(canonical_json(program)).hexdigest(),
                "files": {
                    path: hashlib.sha256(payload).hexdigest()
                    for path, payload in payloads.items()
                },
            }
        )
    report = {
        "schema_version": 1,
        "artifact_type": "verdiwm-cpbe-materialization-report",
        "state": "ready",
        "experiment_id": plan["experiment_id"],
        "probe_count": len(rows),
        "probes": rows,
        "claim_boundary": (
            "Generated code implements static and offline diagnostic semantics only. "
            "Runtime canary, selector gain, model quality, and transfer remain unsettled."
        ),
    }
    files["materialization-report.json"] = canonical_json(report)
    return write_bundle(
        output_root=output_root,
        files=files,
        manifest_fields={
            "artifact_type": "verdiwm-cpbe-materialization-manifest",
            "state": "ready",
            "experiment_id": plan["experiment_id"],
            "probe_count": len(rows),
            "report_path": "materialization-report.json",
        },
    )


def _validate_supported(*, order: Mapping[str, Any], program: Mapping[str, Any]) -> None:
    expected = {
        "hook_type": "H2",
        "spatial_mask": "all_action_embedding",
        "diagnostic_only": True,
        "reversible": True,
    }
    mismatched = [field for field, value in expected.items() if program.get(field) != value]
    if mismatched:
        raise CPBEMaterializerError(
            "CPBE_MATERIALIZER_PROGRAM_UNSUPPORTED:" + ",".join(sorted(mismatched))
        )
    template = _materialization_template(program)
    if program.get("temporal_basis") not in {"event_salience", "event_phase_tangent"}:
        raise CPBEMaterializerError("CPBE_MATERIALIZER_TEMPORAL_BASIS_UNSUPPORTED")
    if program.get("contrast_operator") not in {
        "signed_mean_preserving_scale",
        "signed_mean_preserving_phase",
    }:
        raise CPBEMaterializerError("CPBE_MATERIALIZER_CONTRAST_UNSUPPORTED")
    if template == "action_embedding_delta_goal_outcome_v1" and (
        program.get("temporal_basis") != "event_phase_tangent"
        or program.get("contrast_operator") != "signed_mean_preserving_phase"
    ):
        raise CPBEMaterializerError("CPBE_MATERIALIZER_EMBEDDING_DELTA_PROGRAM_UNSUPPORTED")
    parent_ids = program.get("parent_probe_ids")
    rationale = program.get("rationale")
    if not isinstance(parent_ids, list) or len(parent_ids) != 1 or not isinstance(rationale, str):
        raise CPBEMaterializerError("CPBE_MATERIALIZER_PARENT_INVALID")
    if order.get("role") != "diagnostic" or order.get("verdict_exposure_allowed") is not False:
        raise CPBEMaterializerError("CPBE_MATERIALIZER_SCOPE_INVALID")


def _materialization_template(program: Mapping[str, Any]) -> str:
    key = (program.get("signal_source"), program.get("aggregation"))
    templates = {
        ("raw_action_sequence", "source_sign_margin"): "source_sign_margin_v1",
        ("action_embedding_delta", "goal_outcome_vector"): (
            "action_embedding_delta_goal_outcome_v1"
        ),
    }
    template = templates.get(key)
    if template is None:
        mismatched = [
            field
            for field, allowed in {
                "signal_source": {"raw_action_sequence", "action_embedding_delta"},
                "aggregation": {"source_sign_margin", "goal_outcome_vector"},
            }.items()
            if program.get(field) not in allowed
        ]
        raise CPBEMaterializerError(
            "CPBE_MATERIALIZER_PROGRAM_UNSUPPORTED:"
            + ",".join(sorted(mismatched or ("aggregation", "signal_source")))
        )
    return template


def _module_source(order: Mapping[str, Any]) -> str:
    program = _mapping(order, "program")
    if _materialization_template(program) == "action_embedding_delta_goal_outcome_v1":
        return _embedding_delta_module_source(order)
    probe_id = _text(order, "probe_id")
    environment = _text(order, "environment")
    signature = _text(order, "signature")
    return f'''"""Generated CPBE source-sign aggregation for {probe_id}."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from wmloop.diagnose.probes.source_sign_margin import measure_source_sign_margin

PROBE_ID = {probe_id!r}
ENVIRONMENT = {environment!r}
SIGNATURE = {signature!r}


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
'''


def _test_source(order: Mapping[str, Any]) -> str:
    program = _mapping(order, "program")
    if _materialization_template(program) == "action_embedding_delta_goal_outcome_v1":
        return _embedding_delta_test_source(order)
    probe_id = _text(order, "probe_id")
    return f'''from __future__ import annotations

from wmloop.diagnose.probes.{probe_id} import measure_cpbe_residual


def test_source_sign_margin_is_schema_valid_and_target_label_free() -> None:
    output = measure_cpbe_residual(
        response_vectors={{
            "target": [0.8, 0.2],
            "positive": [1.0, 0.0],
            "negative_a": [0.0, 1.0],
            "negative_b": [0.1, 0.9],
        }},
        source_signs={{"positive": 1, "negative_a": -1, "negative_b": -1}},
        target="target",
    )
    assert output["probe_id"] == {probe_id!r}
    assert output["verdict_exposure_allowed"] is False
    audit = output["metrics"]["fit_audit"]
    assert audit["target_label_used_for_fit"] is False
    assert "target" not in audit["fit_environments"]
'''


def _embedding_delta_module_source(order: Mapping[str, Any]) -> str:
    probe_id = _text(order, "probe_id")
    environment = _text(order, "environment")
    signature = _text(order, "signature")
    doses = tuple(float(value) for value in _mapping(order, "program")["dose_schedule"])
    return f'''"""Generated CPBE action-embedding-delta diagnostic for {probe_id}."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path

from wmloop.contracts import ContractValidationError, validate_document

PROBE_ID = {probe_id!r}
ENVIRONMENT = {environment!r}
SIGNATURE = {signature!r}
DOSE_SCHEDULE = {doses!r}


class CPBEEmbeddingDeltaError(ValueError):
    """The embedding-delta fixture or diagnostic output is invalid."""


def embedding_delta_event_weights(
    action_embeddings: Sequence[Sequence[float]],
) -> list[float]:
    rows = _matrix(action_embeddings, "ACTION_EMBEDDINGS")
    if len(rows) < 2:
        raise CPBEEmbeddingDeltaError("CPBE_EMBEDDING_DELTA_SEQUENCE_TOO_SHORT")
    weights = [0.0]
    for previous, current in zip(rows, rows[1:]):
        weights.append(sum(abs(right - left) for left, right in zip(previous, current)) / len(current))
    scale = max(weights)
    return [value / scale if scale > 1e-12 else 0.0 for value in weights]


def measure_cpbe_residual(
    *,
    dose_responses: Mapping[float, Sequence[float]],
    evidence_refs: Sequence[str] = (),
) -> dict[str, object]:
    parsed = {{float(dose): _vector(vector, "GOAL_OUTCOME_VECTOR") for dose, vector in dose_responses.items()}}
    if tuple(sorted(parsed)) != tuple(sorted(DOSE_SCHEDULE)) or 0.0 not in parsed:
        raise CPBEEmbeddingDeltaError("CPBE_EMBEDDING_DELTA_DOSE_SCHEDULE_MISMATCH")
    width = len(parsed[0.0])
    if any(len(vector) != width for vector in parsed.values()):
        raise CPBEEmbeddingDeltaError("CPBE_EMBEDDING_DELTA_OUTCOME_WIDTH_MISMATCH")
    denominator = sum(dose * dose for dose in DOSE_SCHEDULE)
    zero = parsed[0.0]
    response = [
        sum(dose * (parsed[dose][column] - zero[column]) for dose in DOSE_SCHEDULE) / denominator
        for column in range(width)
    ]
    output = {{
        "schema_version": 1,
        "artifact_type": "wmloop-diagnostic-probe-output",
        "probe_id": PROBE_ID,
        "role": "diagnostic",
        "environment": ENVIRONMENT,
        "signature": SIGNATURE,
        "state": "measured",
        "metrics": {{
            "signal_source": "action_embedding_delta",
            "aggregation": "goal_outcome_vector",
            "response_vector": response,
            "target_label_used_for_fit": False,
        }},
        "flags": ["diagnostic_only", "target_label_free"],
        "evidence_refs": list(evidence_refs),
        "verdict_exposure_allowed": False,
        "limitations": [
            "Offline diagnostic fixture only; runtime locality and collision separation remain unsettled."
        ],
    }}
    try:
        validate_document("diagnostic_probe_output", output, root=Path(__file__).resolve().parents[3])
    except ContractValidationError as exc:
        raise CPBEEmbeddingDeltaError(f"CPBE_EMBEDDING_DELTA_OUTPUT_INVALID:{{exc}}") from exc
    return output


def _matrix(value: Sequence[Sequence[float]], code: str) -> list[list[float]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise CPBEEmbeddingDeltaError(code)
    rows = [_vector(row, code) for row in value]
    if not rows or any(len(row) != len(rows[0]) for row in rows):
        raise CPBEEmbeddingDeltaError(code)
    return rows


def _vector(value: Sequence[float], code: str) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise CPBEEmbeddingDeltaError(code)
    parsed = [float(item) for item in value]
    if any(not math.isfinite(item) for item in parsed):
        raise CPBEEmbeddingDeltaError(code)
    return parsed
'''


def _embedding_delta_test_source(order: Mapping[str, Any]) -> str:
    probe_id = _text(order, "probe_id")
    doses = tuple(float(value) for value in _mapping(order, "program")["dose_schedule"])
    responses = {dose: [dose, -2.0 * dose] for dose in doses}
    return f'''from __future__ import annotations

import pytest

from wmloop.diagnose.probes.{probe_id} import (
    embedding_delta_event_weights,
    measure_cpbe_residual,
)


def test_embedding_delta_signal_and_goal_outcome_response_are_exact() -> None:
    weights = embedding_delta_event_weights([[0.0, 0.0], [1.0, 0.0], [1.0, 2.0]])
    assert weights == pytest.approx([0.0, 0.5, 1.0])
    output = measure_cpbe_residual(dose_responses={responses!r})
    assert output["probe_id"] == {probe_id!r}
    assert output["metrics"]["response_vector"] == pytest.approx([1.0, -2.0])
    assert output["metrics"]["target_label_used_for_fit"] is False
    assert output["verdict_exposure_allowed"] is False
'''


def _descriptor(order: Mapping[str, Any]) -> dict[str, object]:
    probe_id = _text(order, "probe_id")
    program = _mapping(order, "program")
    template = _materialization_template(program)
    if template == "source_sign_margin_v1":
        implementation_parameters = {
            "aggregation": "source_sign_margin",
            "fit_scope": "settled_source_environments_only",
            "target_label_used_for_fit": False,
            "projection": "source_centroid_discriminant_in_goal_outcome_frame",
        }
    else:
        implementation_parameters = {
            "signal_source": "action_embedding_delta",
            "aggregation": "goal_outcome_vector",
            "event_weight": "normalized_mean_absolute_embedding_delta",
            "target_label_used_for_fit": False,
        }
    return {
        "schema_version": 1,
        "artifact_type": "wmloop-staged-diagnostic-probe-descriptor",
        "probe_id": probe_id,
        "environment": _text(order, "environment"),
        "signature": _text(order, "signature"),
        "priority": _text(order, "priority"),
        "role": "diagnostic",
        "materialization_state": "offline_fixture_materialized",
        "module": f"wmloop.diagnose.probes.{probe_id}",
        "callable": "measure_cpbe_residual",
        "measurement_callable": "measure_cpbe_residual",
        "program": dict(program),
        "implementation_parameters": implementation_parameters,
        "verdict_exposure_allowed": False,
        "admission_gates_satisfied": [
            "schema_valid_diagnostic_probe_output",
            "offline_fixture_test_passed",
            "no_verdict_evidence_exposure",
        ],
        "admission_gates_pending": [
            "runtime_smoke_on_dev_split",
            "locality_and_nonredundancy_canary_passed",
            "selector_regret_or_coverage_gain_observed",
        ],
        "admission_state": {
            "static": "implemented",
            "offline": "fixture_test_available",
            "canary": "not_run",
            "expanded": "not_run",
        },
        "claim_boundary": (
            "Offline source-sign aggregation only; no runtime, selector-gain, effect, "
            "transfer, or verdict claim is licensed."
        ),
    }


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CPBEMaterializerError(f"CPBE_MATERIALIZER_JSON_INVALID:{path}") from exc
    if not isinstance(value, dict):
        raise CPBEMaterializerError(f"CPBE_MATERIALIZER_JSON_INVALID:{path}")
    return value


def _mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    item = value.get(key)
    if not isinstance(item, Mapping):
        raise CPBEMaterializerError(f"CPBE_MATERIALIZER_MAPPING_INVALID:{key}")
    return item


def _mapping_sequence(value: object, code: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(item, Mapping) for item in value):
        raise CPBEMaterializerError(code)
    return list(value)


def _text(value: Mapping[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise CPBEMaterializerError(f"CPBE_MATERIALIZER_TEXT_INVALID:{key}")
    return item


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    manifest = publish_cpbe_materialization(plan_path=args.plan, output_root=args.output_root)
    print(json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
