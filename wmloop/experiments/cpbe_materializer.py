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
                "materialization_template": "source_sign_margin_v1",
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
        "signal_source": "raw_action_sequence",
        "spatial_mask": "all_action_embedding",
        "aggregation": "source_sign_margin",
        "diagnostic_only": True,
        "reversible": True,
    }
    mismatched = [field for field, value in expected.items() if program.get(field) != value]
    if mismatched:
        raise CPBEMaterializerError(
            "CPBE_MATERIALIZER_PROGRAM_UNSUPPORTED:" + ",".join(sorted(mismatched))
        )
    if program.get("temporal_basis") not in {"event_salience", "event_phase_tangent"}:
        raise CPBEMaterializerError("CPBE_MATERIALIZER_TEMPORAL_BASIS_UNSUPPORTED")
    if program.get("contrast_operator") not in {
        "signed_mean_preserving_scale",
        "signed_mean_preserving_phase",
    }:
        raise CPBEMaterializerError("CPBE_MATERIALIZER_CONTRAST_UNSUPPORTED")
    parent_ids = program.get("parent_probe_ids")
    rationale = program.get("rationale")
    if not isinstance(parent_ids, list) or len(parent_ids) != 1 or not isinstance(rationale, str):
        raise CPBEMaterializerError("CPBE_MATERIALIZER_PARENT_INVALID")
    if order.get("role") != "diagnostic" or order.get("verdict_exposure_allowed") is not False:
        raise CPBEMaterializerError("CPBE_MATERIALIZER_SCOPE_INVALID")


def _module_source(order: Mapping[str, Any]) -> str:
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


def _descriptor(order: Mapping[str, Any]) -> dict[str, object]:
    probe_id = _text(order, "probe_id")
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
        "program": dict(_mapping(order, "program")),
        "implementation_parameters": {
            "aggregation": "source_sign_margin",
            "fit_scope": "settled_source_environments_only",
            "target_label_used_for_fit": False,
            "projection": "source_centroid_discriminant_in_goal_outcome_frame",
        },
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
