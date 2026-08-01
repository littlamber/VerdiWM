"""Fail-closed compilation of a diagnostic probe onto a backbone instance."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from wmloop.contracts import ContractValidationError, validate_document
from wmloop.experiments._artifacts import canonical_json, write_bundle
from wmloop.geometry import CapabilityProfile, InterventionDescriptor, compile_intervention


class ProbeSemanticCompileError(ValueError):
    """A probe or backbone capability contract is malformed."""


def compile_probe_for_backbone(
    *,
    probe_path: Path,
    instance_path: Path,
    capability_contract_path: Path,
    repo_root: Path,
    output_root: Path,
) -> dict[str, object]:
    """Compile one immutable probe descriptor without semantic substitution."""

    probe = _load_mapping(probe_path)
    instance = _load_mapping(instance_path)
    contract = _load_mapping(capability_contract_path)
    _validate_inputs(probe=probe, instance=instance, contract=contract)
    try:
        validate_document("backbone_probe_capability_contract", contract, root=repo_root)
    except ContractValidationError as exc:
        raise ProbeSemanticCompileError("PROBE_CAPABILITY_CONTRACT_SCHEMA_INVALID") from exc

    surfaces = {
        str(row["surface_id"]): Path(str(row["artifact_ref"]))
        for row in instance["surfaces"]
    }
    surfaces["verdiwm_repo"] = Path(repo_root).resolve(strict=True)
    audited = [
        _audit_capability(row, surfaces=surfaces)
        for row in contract["capabilities"]
    ]
    available = frozenset(
        str(row["name"]) for row in audited if row["available"] is True
    )
    program = probe["program"]
    required = _required_semantics(program)
    invariant_names = tuple(str(value) for value in program["invariants"])
    invariant_checks = {name: name in available for name in invariant_names}
    dose = next(
        (float(value) for value in program["dose_schedule"] if float(value) != 0.0),
        None,
    )
    descriptor = InterventionDescriptor(
        name=str(program["probe_id"]),
        kind="probe_path",
        hook_type=str(program["hook_type"]),
        transformation=(
            f"{program['signal_source']} -> {program['temporal_basis']} -> "
            f"{program['contrast_operator']} -> {program['aggregation']}"
        ),
        scope="diagnostic_inference_only",
        dose_unit="relative_action_embedding_phase_dose",
        schedule="symmetric_finite_difference",
        preconditions=tuple(sorted(required)),
        invariants=invariant_names,
        prediction=str(probe["signature"]),
        required_capabilities=frozenset(required - {str(program["hook_type"])}),
        inference_only=bool(program["diagnostic_only"]),
        reversible=bool(program["reversible"]),
    )
    typed_receipt = compile_intervention(
        descriptor,
        CapabilityProfile(
            backbone_family=str(instance["backbone_family"]),
            capability_class=str(contract["capability_class"]),
            capabilities=available,
            hook_types=frozenset(
                str(row["name"])
                for row in audited
                if row["kind"] == "hook" and row["available"] is True
            ),
        ),
        invariant_checks=invariant_checks,
        dose_direction=dose,
    ).to_dict()
    missing = sorted(required - available)
    report = {
        "schema_version": 1,
        "artifact_type": "verdiwm-probe-semantic-compile-report",
        "state": "compiled" if typed_receipt["compiled"] else "blocked",
        "probe_id": program["probe_id"],
        "probe_descriptor_sha256": _sha256(Path(probe_path)),
        "instance_id": instance["instance_id"],
        "backbone_family": instance["backbone_family"],
        "capability_class": contract["capability_class"],
        "required_semantics": sorted(required),
        "available_required_semantics": sorted(required & available),
        "missing_required_semantics": missing,
        "capability_audit": audited,
        "typed_compile_receipt": typed_receipt,
        "gpu_execution_started": False,
        "semantic_substitution_used": False,
        "claim_boundary": (
            "A compiled receipt proves only exact runtime wiring for this diagnostic probe. "
            "A blocked receipt forbids approximate substitution. Neither state is model-quality, "
            "repair-benefit, or cross-backbone transfer evidence."
        ),
    }
    destination = Path(output_root).resolve()
    return write_bundle(
        output_root=destination,
        files={
            "compile-report.json": canonical_json(report),
            "typed-compile-receipt.json": canonical_json(typed_receipt),
            "compile-report.md": _markdown(report).encode("utf-8"),
            "input-capability-contract.json": canonical_json(contract),
        },
        manifest_fields={
            "artifact_type": "verdiwm-probe-semantic-compile-manifest",
            "state": report["state"],
            "probe_id": report["probe_id"],
            "instance_id": report["instance_id"],
            "compiled": typed_receipt["compiled"],
            "missing_semantic_count": len(missing),
            "gpu_execution_started": False,
            "report_path": str(destination / "compile-report.json"),
        },
    )


def _required_semantics(program: Mapping[str, Any]) -> set[str]:
    return {
        str(program["hook_type"]),
        *(str(value) for value in program["required_capabilities"]),
        f"signal_source:{program['signal_source']}",
        f"temporal_basis:{program['temporal_basis']}",
        f"contrast_operator:{program['contrast_operator']}",
        *(str(value) for value in program["invariants"]),
    }


def _audit_capability(
    row: Mapping[str, Any], *, surfaces: Mapping[str, Path]
) -> dict[str, object]:
    name = str(row["name"])
    kind = str(row["kind"])
    implemented = row.get("implemented") is True
    evidence_rows: list[dict[str, object]] = []
    evidence_pass = True
    for item in row.get("evidence", []):
        surface_id = str(item["surface_id"])
        root = surfaces.get(surface_id)
        relative = Path(str(item["path"]))
        safe_relative = not relative.is_absolute() and ".." not in relative.parts
        path = root / relative if root is not None and safe_relative else None
        exists = path is not None and path.is_file()
        content = path.read_text(encoding="utf-8") if exists else ""
        anchors = tuple(str(value) for value in item["anchors"])
        missing_anchors = [value for value in anchors if value not in content]
        passed = bool(exists and not missing_anchors)
        evidence_pass = evidence_pass and passed
        evidence_rows.append(
            {
                "surface_id": surface_id,
                "path": str(relative),
                "path_exists": bool(exists),
                "anchors": list(anchors),
                "missing_anchors": missing_anchors,
                "sha256": _sha256(path) if exists and path is not None else None,
                "passed": passed,
            }
        )
    if implemented and not evidence_rows:
        evidence_pass = False
    available = bool(implemented and evidence_pass)
    return {
        "name": name,
        "kind": kind,
        "declared_implemented": implemented,
        "available": available,
        "reason": (
            "implementation_and_evidence_verified"
            if available
            else str(row.get("reason", "implementation_or_evidence_missing"))
        ),
        "evidence": evidence_rows,
    }


def _validate_inputs(
    *, probe: Mapping[str, Any], instance: Mapping[str, Any], contract: Mapping[str, Any]
) -> None:
    if probe.get("artifact_type") != "wmloop-staged-diagnostic-probe-descriptor":
        raise ProbeSemanticCompileError("PROBE_DESCRIPTOR_TYPE_INVALID")
    if instance.get("artifact_type") != "wmloop-backbone-instance":
        raise ProbeSemanticCompileError("BACKBONE_INSTANCE_TYPE_INVALID")
    if contract.get("artifact_type") != "verdiwm-backbone-probe-capability-contract":
        raise ProbeSemanticCompileError("PROBE_CAPABILITY_CONTRACT_TYPE_INVALID")
    if contract.get("instance_id") != instance.get("instance_id"):
        raise ProbeSemanticCompileError("PROBE_CAPABILITY_INSTANCE_MISMATCH")
    if contract.get("backbone_family") != instance.get("backbone_family"):
        raise ProbeSemanticCompileError("PROBE_CAPABILITY_BACKBONE_MISMATCH")
    rows = contract.get("capabilities")
    if not isinstance(rows, list) or not rows:
        raise ProbeSemanticCompileError("PROBE_CAPABILITY_ROWS_INVALID")
    names = [str(row.get("name", "")) for row in rows if isinstance(row, Mapping)]
    if len(names) != len(rows) or len(names) != len(set(names)) or any(not name for name in names):
        raise ProbeSemanticCompileError("PROBE_CAPABILITY_NAMES_INVALID")


def _load_mapping(path: Path) -> Mapping[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ProbeSemanticCompileError(f"JSON_OBJECT_REQUIRED:{path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _markdown(report: Mapping[str, Any]) -> str:
    rows = [
        "# Probe Semantic Compile Report",
        "",
        f"- Probe: `{report['probe_id']}`",
        f"- Backbone: `{report['backbone_family']}`",
        f"- Instance: `{report['instance_id']}`",
        f"- State: `{report['state']}`",
        "- Semantic substitution: `false`",
        "- GPU execution: `false`",
        "",
        "## Missing Required Semantics",
        "",
    ]
    missing = report["missing_required_semantics"]
    rows.extend(f"- `{value}`" for value in missing) if missing else rows.append("- None.")
    rows.extend(["", "## Claim Boundary", "", str(report["claim_boundary"]), ""])
    return "\n".join(rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--instance", type=Path, required=True)
    parser.add_argument("--capability-contract", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = compile_probe_for_backbone(
        probe_path=args.probe,
        instance_path=args.instance,
        capability_contract_path=args.capability_contract,
        repo_root=args.repo_root,
        output_root=args.output_root,
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
