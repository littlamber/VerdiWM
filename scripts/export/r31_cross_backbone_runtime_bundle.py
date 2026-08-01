#!/usr/bin/env python3
"""Export path-sanitized r31 Ctrl-World and Cosmos3 runtime canary evidence."""

from __future__ import annotations

import argparse
import csv
import io
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from wmloop.experiments._artifacts import canonical_json, write_bundle


class R31RuntimeBundleError(ValueError):
    """The runtime evidence is incomplete or violates its invariant contract."""


def export_r31_runtime_bundle(
    *,
    ctrl_roots: Sequence[Path],
    cosmos_roots: Sequence[Path],
    ctrl_compile_report: Path,
    cosmos_compile_report: Path,
    reuse_audit: Path,
    output_root: Path,
) -> dict[str, object]:
    ctrl_rows, ctrl_audits = _ctrl_rows(ctrl_roots)
    cosmos_rows, cosmos_audits = _cosmos_rows(cosmos_roots)
    expected_doses = {-0.05, 0.0, 0.05}
    if {row["dose"] for row in ctrl_rows} != expected_doses:
        raise R31RuntimeBundleError("R31_CTRL_DOSE_COVERAGE_INVALID")
    if {row["dose"] for row in cosmos_rows} != expected_doses:
        raise R31RuntimeBundleError("R31_COSMOS_DOSE_COVERAGE_INVALID")

    compile_rows = []
    for backbone, path in (
        ("ctrl_world", ctrl_compile_report),
        ("cosmos3", cosmos_compile_report),
    ):
        report = _mapping(path)
        receipt = report.get("typed_compile_receipt")
        if not isinstance(receipt, Mapping) or receipt.get("compiled") is not True:
            raise R31RuntimeBundleError(f"R31_COMPILE_NOT_READY:{backbone}")
        compile_rows.append(
            {
                "backbone": backbone,
                "compiled": True,
                "missing_required_semantics": list(report.get("missing_required_semantics", [])),
                "semantic_substitution_used": bool(report.get("semantic_substitution_used")),
            }
        )

    reuse = _mapping(reuse_audit)
    bundle = {
        "schema_version": 1,
        "artifact_type": "verdiwm-r31-cross-backbone-runtime-bundle",
        "state": "ready",
        "probe_id": "cpbe_residual_63f088b0d5",
        "semantic_program": {
            "signal_source": "action_embedding_delta",
            "temporal_basis": "event_phase_tangent",
            "contrast_operator": "signed_mean_preserving_phase",
        },
        "compile": compile_rows,
        "runtime": {
            "ctrl_world": {"metrics": ctrl_rows, "hook_audits": ctrl_audits},
            "cosmos3": {"metrics": cosmos_rows, "hook_audits": cosmos_audits},
        },
        "reuse_audit": {
            "state": reuse.get("state"),
            "r31_exact_portability_ready": reuse.get("r31_exact_portability_ready"),
            "cold_vs_warm_identifiable": reuse.get("lobo", {}).get(
                "cold_vs_warm_identifiable"
            ),
            "observed_confirm_count": reuse.get("lobo", {}).get("observed_confirm_count"),
            "formal_chart_missing_backbones": reuse.get("formal_chart_missing_backbones"),
            "gpu_launch_decision": reuse.get("gpu_launch_decision"),
        },
        "interpretation": (
            "Exact r31 semantics compiled and executed on both backbones. The three-dose canaries "
            "show small mixed metric responses and do not establish repair benefit or transfer."
        ),
        "claim_boundary": (
            "Runtime invocation and bounded development response evidence only. This bundle is not "
            "a model-improvement, formal locality, cold-vs-warm, or cross-backbone transfer result."
        ),
    }
    files = {
        "bundle.json": canonical_json(bundle),
        "README.md": _readme(bundle).encode("utf-8"),
        "tables/runtime-canary-metrics.csv": _metrics_csv(ctrl_rows + cosmos_rows),
    }
    return write_bundle(
        output_root=output_root,
        files=files,
        manifest_fields={
            "artifact_type": "verdiwm-r31-cross-backbone-runtime-bundle-manifest",
            "state": "ready",
            "probe_id": bundle["probe_id"],
            "r31_exact_portability_ready": True,
            "cold_vs_warm_identifiable": False,
        },
    )


def _ctrl_rows(roots: Sequence[Path]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    by_dose: dict[float, list[Mapping[str, Any]]] = {}
    audits: list[dict[str, object]] = []
    for root in roots:
        base = Path(root).resolve(strict=True)
        index = _mapping(base / "receipt-index.json")
        audit = _mapping(base / "runtime-probe-audit.json")
        if audit.get("state") != "passed":
            raise R31RuntimeBundleError("R31_CTRL_RUNTIME_AUDIT_FAILED")
        for row in audit["rows"]:
            _validate_hook_audit(row, "R31_CTRL")
            audits.append(dict(row))
        for item in index["rows"]:
            receipt = _mapping(Path(str(item["receipt_ref"])))
            by_dose.setdefault(float(item["dose"]), []).append(receipt["metrics"])
    rows = []
    for dose, metrics in sorted(by_dose.items()):
        keys = sorted(metrics[0])
        rows.append(
            {
                "backbone": "ctrl_world",
                "dose": dose,
                "sample_count": len(metrics),
                **{key: sum(float(row[key]) for row in metrics) / len(metrics) for key in keys},
            }
        )
    return rows, sorted(audits, key=lambda row: float(row["dose"]))


def _cosmos_rows(roots: Sequence[Path]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []
    for root in roots:
        base = Path(root).resolve(strict=True)
        manifest = _mapping(base / "manifest.json")
        if manifest.get("state") != "ready":
            raise R31RuntimeBundleError("R31_COSMOS_CAMPAIGN_NOT_READY")
        for record in manifest["records"]:
            receipt_path = Path(str(record["receipt_ref"])).resolve(strict=True)
            receipt = _mapping(receipt_path)
            intervention_ref = Path(str(receipt["intervention_ref"]))
            hook_path = (
                intervention_ref
                if intervention_ref.is_absolute()
                else receipt_path.parent / intervention_ref
            )
            hook = _mapping(hook_path)
            audit = dict(hook["audit"])
            audit["dose"] = float(record["dose"])
            _validate_hook_audit(audit, "R31_COSMOS")
            audits.append(audit)
            rows.append(
                {
                    "backbone": "cosmos3",
                    "dose": float(record["dose"]),
                    "sample_count": 1,
                    **{key: float(value) for key, value in record["metrics"].items()},
                }
            )
    return sorted(rows, key=lambda row: float(row["dose"])), sorted(
        audits, key=lambda row: float(row["dose"])
    )


def _validate_hook_audit(row: Mapping[str, Any], prefix: str) -> None:
    invocation_count = int(
        row.get("invocation_count", row.get("embedding_hook_invocation_count", 0))
    )
    error = float(row.get("maximum_temporal_mean_abs_error", 0.0))
    tolerance = float(row.get("maximum_temporal_mean_tolerance", 0.0))
    if invocation_count < 1:
        raise R31RuntimeBundleError(f"{prefix}_HOOK_NOT_INVOKED")
    if error > tolerance:
        raise R31RuntimeBundleError(f"{prefix}_MEAN_PRESERVATION_FAILED")


def _metrics_csv(rows: Sequence[Mapping[str, object]]) -> bytes:
    metric_names = sorted(
        {key for row in rows for key in row if key not in {"backbone", "dose", "sample_count"}}
    )
    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=["backbone", "dose", "sample_count", *metric_names],
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def _readme(bundle: Mapping[str, Any]) -> str:
    return (
        "# r31 Cross-Backbone Runtime Canary\n\n"
        "The same typed CPBE program was compiled and invoked at the action-embedding H2 hook "
        "in Ctrl-World and Cosmos3. All three development doses (`-0.05`, `0`, `+0.05`) have "
        "runtime invocation receipts and dtype-aware temporal-mean checks.\n\n"
        "The metric changes are small and mixed. This establishes executable semantic portability, "
        "not repair benefit. Cold-vs-warm LOBO remains unidentifiable because there are no settled "
        "confirmation receipts and the formal response-chart set is incomplete.\n\n"
        f"Claim boundary: {bundle['claim_boundary']}\n"
    )


def _mapping(path: Path) -> Mapping[str, Any]:
    payload = json.loads(Path(path).resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise R31RuntimeBundleError(f"R31_JSON_OBJECT_REQUIRED:{path}")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ctrl-root", type=Path, action="append", required=True)
    parser.add_argument("--cosmos-root", type=Path, action="append", required=True)
    parser.add_argument("--ctrl-compile-report", type=Path, required=True)
    parser.add_argument("--cosmos-compile-report", type=Path, required=True)
    parser.add_argument("--reuse-audit", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = export_r31_runtime_bundle(
        ctrl_roots=args.ctrl_root,
        cosmos_roots=args.cosmos_root,
        ctrl_compile_report=args.ctrl_compile_report,
        cosmos_compile_report=args.cosmos_compile_report,
        reuse_audit=args.reuse_audit,
        output_root=args.output_root,
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
