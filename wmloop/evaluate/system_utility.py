"""Audit VerdiWM usability and evidence-backed utility without GPU execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from wmloop.contracts import ContractValidationError, validate_document
from wmloop.experiments._artifacts import canonical_json, write_bundle


class SystemUtilityAuditError(ValueError):
    """The system utility audit input or evidence is invalid."""


def load_audit_config(path: Path) -> dict[str, Any]:
    source = Path(path).resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemUtilityAuditError(f"SYSTEM_UTILITY_CONFIG_READ_FAILED:{source}") from exc
    if not isinstance(payload, dict):
        raise SystemUtilityAuditError("SYSTEM_UTILITY_CONFIG_ROOT_OBJECT_REQUIRED")
    try:
        validate_document("system_utility_audit", payload)
    except ContractValidationError as exc:
        raise SystemUtilityAuditError(f"SYSTEM_UTILITY_CONFIG_INVALID:{exc}") from exc
    ids = [str(item["id"]) for item in payload["inputs"]]
    if len(set(ids)) != len(ids):
        raise SystemUtilityAuditError("SYSTEM_UTILITY_INPUT_IDS_NOT_UNIQUE")
    payload["_config_path"] = str(source)
    return payload


def run_system_utility_audit(
    *, config_path: Path, repo_root: Path, output_root: Path
) -> dict[str, object]:
    config = load_audit_config(config_path)
    root = Path(repo_root).resolve(strict=True)
    documents: dict[str, Mapping[str, Any]] = {}
    references: dict[str, dict[str, object]] = {}
    for item in config["inputs"]:
        identifier = str(item["id"])
        relative = _safe_relative(str(item["path"]))
        source = (root / relative).resolve()
        if not source.is_file() or source.is_symlink():
            raise SystemUtilityAuditError(f"SYSTEM_UTILITY_INPUT_MISSING:{identifier}:{relative}")
        try:
            document = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemUtilityAuditError(f"SYSTEM_UTILITY_INPUT_INVALID:{identifier}:{relative}") from exc
        if not isinstance(document, Mapping):
            raise SystemUtilityAuditError(f"SYSTEM_UTILITY_INPUT_OBJECT_REQUIRED:{identifier}")
        expected_type = str(item["artifact_type"])
        if document.get("artifact_type") != expected_type:
            raise SystemUtilityAuditError(f"SYSTEM_UTILITY_ARTIFACT_TYPE_MISMATCH:{identifier}")
        documents[identifier] = document
        references[identifier] = {
            "path": str(relative),
            "sha256": _sha256(source),
            "size_bytes": source.stat().st_size,
        }

    report = build_system_utility_report(config=config, documents=documents, references=references)
    destination = Path(output_root).resolve()
    return write_bundle(
        output_root=destination,
        files={
            "system-utility-audit.json": canonical_json(report),
            "system-utility-audit.md": _markdown(report).encode("utf-8"),
            "input-config.json": canonical_json({key: value for key, value in config.items() if not key.startswith("_")}),
        },
        manifest_fields={
            "artifact_type": "verdiwm-system-utility-audit-manifest",
            "state": report["state"],
            "audit_id": report["audit_id"],
            "operational_state": report["operational_state"],
            "research_effect_state": report["research_effect_state"],
            "passed_gate_count": report["passed_gate_count"],
            "blocked_gate_count": report["blocked_gate_count"],
            "report_path": str(destination / "system-utility-audit.json"),
        },
    )


def build_system_utility_report(
    *,
    config: Mapping[str, Any],
    documents: Mapping[str, Mapping[str, Any]],
    references: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    thresholds = config["thresholds"]
    proof = documents["minimal_loop"]
    efficiency = documents["progressive_fidelity"]
    selector = documents["selector_replay"]
    local_chart = documents["ctrl_world_local_chart"]
    formal_chart = documents["ctrl_world_formal_chart"]
    runtime = documents["cross_backbone_runtime"]
    profiles = documents["ctrl_world_profiles"]

    operational_pass = proof.get("state") == "ready" and proof.get("operational_minimal_loop_pass") is True
    metrics = _mapping(efficiency, "metrics")
    gpu_reduction = _finite(metrics, "gpu_hour_reduction")
    positive_recall = _finite(metrics, "positive_recall")
    efficiency_pass = (
        gpu_reduction >= float(thresholds["minimum_gpu_hour_reduction"])
        and positive_recall >= float(thresholds["minimum_positive_recall"])
    )

    selector_rows = selector.get("selectors")
    if not isinstance(selector_rows, list) or not selector_rows:
        raise SystemUtilityAuditError("SYSTEM_UTILITY_SELECTOR_ROWS_MISSING")
    normalized_selectors = [_selector_row(row) for row in selector_rows]
    best_selector = min(
        normalized_selectors,
        key=lambda row: (float(row["negative_selection"]), -float(row["benefit_sign_accuracy"])),
    )
    worst_negative_selection = max(float(row["negative_selection"]) for row in normalized_selectors)
    divergence_count = int(selector.get("selector_choice_divergence_environment_count", 0))
    selector_pass = (
        float(best_selector["negative_selection"]) <= float(thresholds["maximum_selector_negative_selection"])
        and divergence_count >= int(thresholds["minimum_selector_choice_divergence_environment_count"])
    )

    local_pass = local_chart.get("state") == "settled_admitted" and local_chart.get("cross_backbone_transfer_eligible") is True
    formal_pass = formal_chart.get("state") == "settled_admitted"
    reuse_audit = _mapping(runtime, "reuse_audit")
    runtime_pass = runtime.get("state") == "ready" and reuse_audit.get("r31_exact_portability_ready") is True
    confirm_count = int(reuse_audit.get("observed_confirm_count", 0))
    transfer_pass = formal_pass and confirm_count >= int(thresholds["minimum_confirm_receipts"])

    profile_rows = profiles.get("profiles")
    if not isinstance(profile_rows, Mapping) or not profile_rows:
        raise SystemUtilityAuditError("SYSTEM_UTILITY_MECHANISM_PROFILES_MISSING")
    profile_states = {
        str(state): sum(1 for value in profile_rows.values() if isinstance(value, Mapping) and value.get("state") == state)
        for state in {str(value.get("state")) for value in profile_rows.values() if isinstance(value, Mapping)}
    }

    gates = [
        _gate("operational_minimal_loop", "pass" if operational_pass else "blocked", "The public minimal loop is executable and integrity checked."),
        _gate(
            "progressive_fidelity_efficiency",
            "pass" if efficiency_pass else "blocked",
            f"Observed GPU-hour reduction={gpu_reduction:.6f}, positive recall={positive_recall:.6f}.",
        ),
        _gate(
            "selector_quality",
            "pass" if selector_pass else "blocked",
            f"Best negative selection={float(best_selector['negative_selection']):.6f}, worst={worst_negative_selection:.6f}, selector divergence environments={divergence_count}.",
        ),
        _gate("ctrl_world_local_chart", "pass" if local_pass else "abstain", "Target-local Ctrl-World chart admission."),
        _gate("ctrl_world_formal_chart", "pass" if formal_pass else "abstain", "Formal Ctrl-World split chart admission."),
        _gate("cross_backbone_runtime_portability", "pass" if runtime_pass else "blocked", "Typed runtime canary portability."),
        _gate("cross_backbone_confirm", "pass" if transfer_pass else "blocked", f"Settled cross-backbone confirmation receipts={confirm_count}."),
    ]
    effect_ready = all(gate["state"] == "pass" for gate in gates[2:])
    operational_state = "ready" if operational_pass else "blocked"
    next_work = list(config["next_work"])
    if not selector_pass:
        next_work.insert(0, "separate selector discrimination from source compatibility and reduce negative selection before broad transfer claims")
    if not transfer_pass:
        next_work.insert(0, "run the bounded Ctrl-World warm/cold/random canary and collect settled target-side confirm receipts")
    return {
        "schema_version": 1,
        "artifact_type": "verdiwm-system-utility-audit-report",
        "audit_id": config["audit_id"],
        "state": "ready" if effect_ready else "partial",
        "operational_state": operational_state,
        "research_effect_state": "established" if effect_ready else "not_established",
        "utility_summary": {
            "progressive_fidelity_gpu_hour_reduction": gpu_reduction,
            "progressive_fidelity_positive_recall": positive_recall,
            "best_selector": best_selector,
            "selector_worst_negative_selection": worst_negative_selection,
            "ctrl_world_local_chart_admitted": local_pass,
            "ctrl_world_formal_chart_admitted": formal_pass,
            "cross_backbone_runtime_portable": runtime_pass,
            "cross_backbone_confirm_receipt_count": confirm_count,
            "ctrl_world_mechanism_profile_count": len(profile_rows),
            "ctrl_world_mechanism_profile_states": profile_states,
        },
        "gates": gates,
        "passed_gate_count": sum(gate["state"] == "pass" for gate in gates),
        "blocked_gate_count": sum(gate["state"] == "blocked" for gate in gates),
        "abstained_gate_count": sum(gate["state"] == "abstain" for gate in gates),
        "input_references": dict(references),
        "next_work": list(dict.fromkeys(next_work)),
        "claim_boundary": (
            "This audit proves CPU control-plane usability and summarizes checked-in evidence. "
            "It does not convert target-local charts, runtime portability, or efficiency evidence into "
            "model-improvement or cross-backbone transfer claims. Only settled target-side confirm "
            "receipts may establish those claims."
        ),
    }


def _selector_row(value: object) -> dict[str, object]:
    row = _as_mapping(value, "SYSTEM_UTILITY_SELECTOR_ROW_INVALID")
    return {
        "selector": str(row.get("selector", "")),
        "negative_selection": _finite(row, "negative_selection"),
        "benefit_sign_accuracy": _finite(row, "benefit_sign_accuracy"),
        "top1_positive_hit": _finite(row, "top1_positive_hit"),
    }


def _gate(identifier: str, state: str, detail: str) -> dict[str, str]:
    return {"id": identifier, "state": state, "detail": detail}


def _mapping(value: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    nested = value.get(field)
    if not isinstance(nested, Mapping):
        raise SystemUtilityAuditError(f"SYSTEM_UTILITY_MAPPING_REQUIRED:{field}")
    return nested


def _as_mapping(value: object, error: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SystemUtilityAuditError(error)
    return value


def _finite(mapping: Mapping[str, Any], field: str) -> float:
    try:
        value = float(mapping[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemUtilityAuditError(f"SYSTEM_UTILITY_NUMBER_INVALID:{field}") from exc
    if not math.isfinite(value):
        raise SystemUtilityAuditError(f"SYSTEM_UTILITY_NUMBER_NONFINITE:{field}")
    return value


def _safe_relative(value: str) -> Path:
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise SystemUtilityAuditError(f"SYSTEM_UTILITY_PATH_INVALID:{value}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# VerdiWM System Utility Audit",
        "",
        f"- State: `{report['state']}`",
        f"- Operational state: `{report['operational_state']}`",
        f"- Research effect state: `{report['research_effect_state']}`",
        "",
        "## Gates",
        "",
        "| Gate | State | Detail |",
        "| --- | --- | --- |",
    ]
    lines.extend(f"| `{gate['id']}` | `{gate['state']}` | {gate['detail']} |" for gate in report["gates"])
    lines.extend(["", "## Next Work", ""])
    lines.extend(f"- {item}" for item in report["next_work"])
    lines.extend(["", "## Claim Boundary", "", str(report["claim_boundary"]), ""])
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--require-effect", action="store_true")
    args = parser.parse_args(argv)
    try:
        manifest = run_system_utility_audit(
            config_path=args.config,
            repo_root=args.repo_root,
            output_root=args.output_root,
        )
    except SystemUtilityAuditError as exc:
        print(str(exc), file=__import__("sys").stderr)
        return 2
    print(json.dumps(manifest, ensure_ascii=True, sort_keys=True))
    return 3 if args.require_effect and manifest["research_effect_state"] != "established" else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
