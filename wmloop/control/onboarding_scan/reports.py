"""Read-only onboarding scan components: reports."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Mapping

from wmloop.contracts import ContractValidationError, validate_document
from wmloop.control.intermediate_ir import build_model_capability_ir, ir_digest
from wmloop.control.onboarding_scan.io import (
    _bounded as _bounded,
    _canonical_json_bytes as _canonical_json_bytes,
    _safe_size as _safe_size,
    _sha256_file as _sha256_file,
    _write_bytes as _write_bytes,
)
from wmloop.control.onboarding_scan.types import (
    SCHEMA_VERSION,
    OnboardingError,
)


def _write_bundle(
    report: Mapping[str, object],
    destination: Path,
    *,
    repo: Path,
    source_revision: Mapping[str, object],
) -> dict[str, object]:
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=str(destination.parent))
    )
    try:
        report_bytes = _canonical_json_bytes(report)
        capability_ir = build_model_capability_ir(report)
        capability_ir_bytes = _canonical_json_bytes(capability_ir)
        files = {
            "model_manifest.json": {
                "schema_version": SCHEMA_VERSION,
                "artifact_type": "wmloop-model-manifest",
                "repo_root": str(repo),
                "repo_name": repo.name,
                "source_revision": source_revision,
                "entrypoint_count": len(report["entrypoints"]),  # type: ignore[arg-type]
                "asset_count": len(report["assets"]),  # type: ignore[arg-type]
            },
            "runtime_lock.json": report["runtime"],
            "asset_manifest.json": {
                "schema_version": SCHEMA_VERSION,
                "artifact_type": "wmloop-asset-manifest",
                "assets": report["assets"],
            },
            "capability_report.json": {
                "schema_version": SCHEMA_VERSION,
                "artifact_type": "wmloop-capability-report",
                "capabilities": report["capabilities"],
            },
            "model-capability-ir.json": capability_ir,
            "evaluator_contract.json": report["evaluator_contract"],
            "conformance_report.json": report["conformance"],
            "generated_connector/connector.json": report["connector"],
        }
        for relative, payload in files.items():
            _write_bytes(temporary / relative, _canonical_json_bytes(payload))
        _write_bytes(temporary / "onboarding-report.json", report_bytes)
        _write_bytes(
            temporary / "onboarding-report.md", _render_markdown(report).encode("utf-8")
        )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "wmloop-model-onboarding-manifest",
            "repo_root": str(repo),
            "repo_name": repo.name,
            "state": report["state"],
            "optimization_launch_allowed": False,
            "blocker_count": len(report["blockers"]),  # type: ignore[arg-type]
            "report_path": str(destination / "onboarding-report.json"),
            "markdown_path": str(destination / "onboarding-report.md"),
            "source_revision": source_revision,
            "report_sha256": hashlib.sha256(report_bytes).hexdigest(),
            "model_capability_ir_path": str(destination / "model-capability-ir.json"),
            "model_capability_ir_sha256": hashlib.sha256(
                capability_ir_bytes
            ).hexdigest(),
            "model_capability_semantic_digest": ir_digest(capability_ir),
            "sidecar_root": str(destination),
            "source_output_written": False,
        }
        _write_bytes(temporary / "manifest.json", _canonical_json_bytes(manifest))
        os.replace(temporary, destination)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _render_markdown(report: Mapping[str, object]) -> str:
    lines = [
        "# VerdiWM Model Onboarding",
        "",
        f"Repository: `{report['repo_name']}`",
        f"State: `{report['state']}`",
        f"Optimization launch allowed: `{report['optimization_launch_allowed']}`",
        "",
        "## Blockers",
        "",
    ]
    blockers = list(report["blockers"])  # type: ignore[arg-type]
    lines.extend(
        f"- `{item.get('code')}`: {json.dumps(item, ensure_ascii=False, sort_keys=True)}"
        for item in blockers
    )
    if not blockers:
        lines.append("- none")
    lines.extend(
        ["", "## Entrypoints", "", "| Path | Kinds | Main guard |", "|:--|:--|:--|"]
    )
    for item in report["entrypoints"]:  # type: ignore[index]
        lines.append(
            f"| {item['path']} | {', '.join(item['kinds'])} | {item['main_guard']} |"
        )
    if not report["entrypoints"]:  # type: ignore[index]
        lines.append("| none | | |")
    lines.extend(
        [
            "",
            "## Capabilities",
            "",
            "| Capability | State | Evidence |",
            "|:--|:--|:--|",
        ]
    )
    for item in report["capabilities"]:  # type: ignore[index]
        evidence = ", ".join(item["evidence"][:3])
        lines.append(f"| {item['capability']} | {item['state']} | {evidence} |")
    lines.extend(["", "## Side Effects", ""])
    for name, value in report["side_effects"].items():  # type: ignore[union-attr]
        lines.append(f"- `{name}`: `{value}`")
    lines.extend(["", "## Next Actions", ""])
    lines.extend(f"- {item}" for item in report["next_actions"])  # type: ignore[index]
    lines.append("")
    return "\n".join(lines)


def _validate_report(report: Mapping[str, object]) -> None:
    try:
        validate_document("model_onboarding_report", report)
    except ContractValidationError as exc:
        raise OnboardingError("ONBOARDING_REPORT_INVALID") from exc
