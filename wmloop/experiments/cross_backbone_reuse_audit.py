"""Audit whether existing evidence can identify cross-backbone LOBO analyses."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from wmloop.experiments._artifacts import canonical_json, write_bundle


class CrossBackboneReuseAuditError(ValueError):
    """A reuse-audit input is missing or has the wrong artifact type."""


def run_cross_backbone_reuse_audit(
    *, config_path: Path, repo_root: Path, output_root: Path
) -> dict[str, object]:
    root = Path(repo_root).resolve(strict=True)
    config = _json(Path(config_path))
    if config.get("artifact_type") != "verdiwm-cross-backbone-reuse-audit-config":
        raise CrossBackboneReuseAuditError("CROSS_BACKBONE_REUSE_CONFIG_INVALID")

    compile_rows = []
    for item in config["probe_compile_reports"]:
        path, report = _source(root, item)
        if report.get("artifact_type") != "verdiwm-probe-semantic-compile-report":
            raise CrossBackboneReuseAuditError("CROSS_BACKBONE_COMPILE_REPORT_INVALID")
        compile_rows.append(
            {
                "backbone": item["backbone"],
                "state": report["state"],
                "compiled": bool(report["typed_compile_receipt"]["compiled"]),
                "missing_required_semantics": report["missing_required_semantics"],
                **_ref(path),
            }
        )

    fingerprint_rows = []
    for item in config["fingerprints"]:
        path, report = _source(root, item)
        locality = report.get("locality_admission")
        if not isinstance(locality, Mapping):
            raise CrossBackboneReuseAuditError("CROSS_BACKBONE_FINGERPRINT_INVALID")
        fingerprint_rows.append(
            {
                "backbone": item["backbone"],
                "split_role": item["split_role"],
                "campaign_id": report.get("campaign_id"),
                "protocol": report.get("protocol"),
                "measurement_count": report.get("measurement_count"),
                "locality_state": locality.get("state"),
                "transfer_eligible": locality.get("cross_backbone_transfer_eligible") is True,
                "path_residuals": locality.get("path_residuals"),
                **_ref(path),
            }
        )

    lobo_path, lobo = _source(root, config["lobo_report"])
    if lobo.get("artifact_type") != "verdiwm-cross-backbone-experiment-report":
        raise CrossBackboneReuseAuditError("CROSS_BACKBONE_LOBO_REPORT_INVALID")
    certificate_path, certificate = _source(root, config["certificate_ablation"])
    if certificate.get("artifact_type") != "verdiwm-transfer-certificate-ablation":
        raise CrossBackboneReuseAuditError("CROSS_BACKBONE_CERTIFICATE_REPORT_INVALID")

    formal_backbones = {
        str(row["backbone"])
        for row in fingerprint_rows
        if row["split_role"] == "formal" and row["transfer_eligible"] is True
    }
    expected_backbones = {str(value) for value in config["expected_backbones"]}
    r31_portable = bool(compile_rows) and all(row["compiled"] for row in compile_rows)
    formal_charts_ready = formal_backbones == expected_backbones
    settled_confirm = int(lobo.get("observed_confirm_count", 0))
    lobo_identifiable = formal_charts_ready and settled_confirm > 0
    minimum_next_work: list[str] = []
    if not r31_portable:
        minimum_next_work.extend(
            [
                "materialize and smoke-test the exact r31 semantic operations for Ctrl-World",
                "materialize an embedding-level H2 adapter and exact r31 operations for Cosmos3",
            ]
        )
    if "ctrl_world" not in formal_backbones:
        minimum_next_work.append("obtain an independent formal-local Ctrl-World chart")
    if "acwm_phys" not in formal_backbones:
        minimum_next_work.append("bind an ACWM-Phys formal response chart to the cross-backbone audit")
    if settled_confirm < 1:
        minimum_next_work.append(
            "run target-compatible repair screen receipts before any full LOBO confirmation grid"
        )
    report = {
        "schema_version": 1,
        "artifact_type": "verdiwm-cross-backbone-reuse-audit",
        "state": "ready" if lobo_identifiable else "blocked",
        "probe_compile_rows": compile_rows,
        "fingerprint_rows": fingerprint_rows,
        "formal_chart_ready_backbones": sorted(formal_backbones),
        "formal_chart_missing_backbones": sorted(expected_backbones - formal_backbones),
        "r31_exact_portability_ready": r31_portable,
        "all_formal_response_charts_ready": formal_charts_ready,
        "lobo": {
            "state": lobo.get("state"),
            "settled_receipt_count": lobo.get("settled_receipt_count"),
            "expected_confirm_count": lobo.get("expected_confirm_count"),
            "observed_confirm_count": settled_confirm,
            "formal_positive_count": lobo.get("formal_positive_count"),
            "cold_vs_warm_identifiable": lobo_identifiable,
            **_ref(lobo_path),
        },
        "certificate_ablation": {
            "state": certificate["state"],
            "scope": certificate["source_scope"],
            "certificate_changed_cell_count": certificate["certificate_changed_cell_count"],
            "certificate_prevented_negative_count": certificate["certificate_prevented_negative_count"],
            "certificate_blocked_positive_count": certificate["certificate_blocked_positive_count"],
            "cross_backbone_result": False,
            **_ref(certificate_path),
        },
        "gpu_launch_decision": (
            "bounded_runtime_canaries_and_repair_screens_only"
            if r31_portable
            else "do_not_launch_r31_sampling_until_exact_runtime_compile_passes"
        ),
        "minimum_next_work": minimum_next_work,
        "claim_boundary": (
            "This audit classifies existing evidence and identifies missing experiments. It does not "
            "convert target-local fingerprints into repair effects, fill absent LOBO receipts, or "
            "promote ACWM leave-one-environment-out replay to cross-backbone transfer evidence."
        ),
    }
    destination = Path(output_root).resolve()
    return write_bundle(
        output_root=destination,
        files={
            "cross-backbone-reuse-audit.json": canonical_json(report),
            "cross-backbone-reuse-audit.md": _markdown(report).encode("utf-8"),
            "input-config.json": canonical_json(config),
        },
        manifest_fields={
            "artifact_type": "verdiwm-cross-backbone-reuse-audit-manifest",
            "state": report["state"],
            "r31_exact_portability_ready": r31_portable,
            "all_formal_response_charts_ready": formal_charts_ready,
            "cold_vs_warm_identifiable": lobo_identifiable,
            "observed_confirm_count": settled_confirm,
            "gpu_execution_started": False,
            "report_path": str(destination / "cross-backbone-reuse-audit.json"),
        },
    )


def _source(root: Path, item: Mapping[str, Any]) -> tuple[Path, Mapping[str, Any]]:
    relative = Path(str(item["path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise CrossBackboneReuseAuditError("CROSS_BACKBONE_SOURCE_PATH_INVALID")
    path = (root / relative).resolve(strict=True)
    return path, _json(path)


def _ref(path: Path) -> dict[str, str]:
    return {"source_ref": str(path), "source_sha256": _sha256(path)}


def _json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise CrossBackboneReuseAuditError(f"CROSS_BACKBONE_JSON_INVALID:{path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Cross-Backbone Evidence Reuse Audit",
        "",
        f"- State: `{report['state']}`",
        f"- Exact r31 portability: `{str(report['r31_exact_portability_ready']).lower()}`",
        f"- Formal response charts complete: `{str(report['all_formal_response_charts_ready']).lower()}`",
        f"- Cold vs warm identifiable: `{str(report['lobo']['cold_vs_warm_identifiable']).lower()}`",
        f"- Observed confirm receipts: `{report['lobo']['observed_confirm_count']}`",
        "",
        "## Minimum Next Work",
        "",
        *(f"- {value}" for value in report["minimum_next_work"]),
        "",
        "## Claim Boundary",
        "",
        str(report["claim_boundary"]),
        "",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = run_cross_backbone_reuse_audit(
        config_path=args.config,
        repo_root=args.repo_root,
        output_root=args.output_root,
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
