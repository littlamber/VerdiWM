"""Generate guarded prompt packets for diagnostic probe materialization."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from wmloop.archive.store import ArchiveStore, ContentAddressedStore
from wmloop.contracts import ContractValidationError, validate_document
from wmloop.control.agent_engineering_policy import engineering_policy, render_engineering_policy


class DiagnosticProbeMaterializationPromptError(RuntimeError):
    """A diagnostic probe prompt batch could not be produced safely."""


def run_diagnostic_probe_materialization_prompt_batch(
    *,
    repo_root: Path,
    failure_signature_bank_manifest: Path,
    output_root: Path,
    probe_ids: Sequence[str] = (),
    archive_db: Path | None = None,
    cas_root: Path | None = None,
) -> dict[str, object]:
    """Write prompt packets for diagnostic-only probe work orders."""

    root = Path(repo_root).resolve()
    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise DiagnosticProbeMaterializationPromptError("DIAGNOSTIC_PROBE_PROMPT_OUTPUT_EXISTS")
    bank_manifest_path = Path(failure_signature_bank_manifest).resolve(strict=True)
    bank_manifest = _load_manifest(bank_manifest_path)
    work_orders = _selected_work_orders(
        bank_manifest,
        probe_ids=probe_ids,
        manifest_root=bank_manifest_path.parent,
    )
    alignment_contract = _alignment_contract()
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    try:
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary.mkdir(mode=0o700)
        prompts_root = temporary / "prompts"
        prompts_root.mkdir(mode=0o700)
        records: list[dict[str, object]] = []
        for probe_id, work_order_path in work_orders:
            work_order_bytes = work_order_path.read_bytes()
            work_order = _load_work_order(work_order_bytes)
            child = _write_child_prompt(
                destination=destination,
                prompts_root=prompts_root,
                probe_id=probe_id,
                work_order=work_order,
                work_order_path=work_order_path,
                work_order_bytes=work_order_bytes,
                alignment_contract=alignment_contract,
            )
            records.append(child)

        report = {
            "schema_version": 1,
            "artifact_type": "wmloop-diagnostic-probe-materialization-prompt-batch",
            "state": "ready",
            "source_failure_signature_bank": {
                "manifest_path": str(bank_manifest_path),
                "state": str(bank_manifest.get("state")),
                "probe_work_order_count": len(
                    _work_order_paths(bank_manifest, manifest_root=bank_manifest_path.parent)
                ),
            },
            "prompt_count": len(records),
            "probe_ids": [str(record["probe_id"]) for record in records],
            "records": records,
            "alignment_contract": alignment_contract,
            "side_effects": {
                "source_code_mutated": False,
                "goal_config_mutated": False,
                "verdict_probe_mutated": False,
                "primitive_registry_mutated": False,
                "gpu_execution_started": False,
                "formal_verdict_mutated": False,
            },
            "limitations": [
                "This batch produces instruction packets only; it does not implement diagnostic probes.",
                "Generated probes remain diagnostic-only until an explicit version boundary promotes them.",
                "No prompt in this batch grants permission to modify frozen verdict evidence or goal configs.",
            ],
        }
        try:
            validate_document("diagnostic_probe_materialization_prompt", report, root=root)
        except ContractValidationError as exc:
            raise DiagnosticProbeMaterializationPromptError(f"DIAGNOSTIC_PROBE_PROMPT_CONTRACT_INVALID:{exc}") from exc

        report_bytes = _canonical_json_bytes(report)
        markdown_bytes = _render_markdown(report).encode("utf-8")
        _write_bytes_atomic(temporary / "diagnostic-probe-materialization-prompt-batch.json", report_bytes)
        _write_bytes_atomic(temporary / "diagnostic-probe-materialization-prompt-batch.md", markdown_bytes)

        cas_refs: dict[str, str] = {}
        archive = ArchiveStore(archive_db) if archive_db is not None else None
        if archive is not None or cas_root is not None:
            root_for_cas = Path(cas_root).resolve() if cas_root is not None else Path(archive_db).resolve().parent  # type: ignore[union-attr]
            cas = ContentAddressedStore(root_for_cas)
            for name, payload, media_type in (
                ("diagnostic_probe_prompt_batch_json", report_bytes, "application/json"),
                ("diagnostic_probe_prompt_batch_markdown", markdown_bytes, "text/markdown"),
                ("failure_signature_bank_manifest", bank_manifest_path.read_bytes(), "application/json"),
            ):
                ref = cas.put_bytes(payload, media_type=media_type).uri
                cas_refs[name] = ref
                if archive is not None:
                    archive.record_artifact_reference(ref)

        manifest = {
            "schema_version": 1,
            "artifact_type": "wmloop-diagnostic-probe-materialization-prompt-batch-manifest",
            "state": report["state"],
            "source_failure_signature_bank": report["source_failure_signature_bank"],
            "prompt_count": report["prompt_count"],
            "probe_ids": report["probe_ids"],
            "records": records,
            "prompt_root": str(destination / "prompts"),
            "report_path": str(destination / "diagnostic-probe-materialization-prompt-batch.json"),
            "markdown_path": str(destination / "diagnostic-probe-materialization-prompt-batch.md"),
            "cas_refs": cas_refs,
            "side_effects": report["side_effects"],
            "limitations": report["limitations"],
        }
        _write_bytes_atomic(temporary / "manifest.json", _canonical_json_bytes(manifest))
        os.replace(temporary, destination)
        return manifest
    except Exception:
        if temporary.is_symlink():
            temporary.unlink()
        elif temporary.exists():
            shutil.rmtree(temporary)
        raise


def _load_manifest(path: Path) -> Mapping[str, Any]:
    payload = _load_json_object(path, "DIAGNOSTIC_PROBE_PROMPT_BANK_MANIFEST_INVALID")
    if payload.get("artifact_type") not in {
        "wmloop-failure-signature-bank-manifest",
        "verdiwm-cpbe-plan-manifest",
    }:
        raise DiagnosticProbeMaterializationPromptError("DIAGNOSTIC_PROBE_PROMPT_BANK_MANIFEST_INVALID")
    if payload.get("state") != "ready":
        raise DiagnosticProbeMaterializationPromptError("DIAGNOSTIC_PROBE_PROMPT_BANK_NOT_READY")
    return payload


def _selected_work_orders(
    bank_manifest: Mapping[str, Any],
    *,
    probe_ids: Sequence[str],
    manifest_root: Path,
) -> list[tuple[str, Path]]:
    available = _work_order_paths(bank_manifest, manifest_root=manifest_root)
    requested = [probe_id for probe_id in probe_ids if probe_id]
    if not requested:
        return sorted(available.items())
    missing = [probe_id for probe_id in requested if probe_id not in available]
    if missing:
        raise DiagnosticProbeMaterializationPromptError(
            "DIAGNOSTIC_PROBE_PROMPT_UNKNOWN_PROBE:" + ",".join(sorted(missing))
        )
    return [(probe_id, available[probe_id]) for probe_id in requested]


def _work_order_paths(bank_manifest: Mapping[str, Any], *, manifest_root: Path) -> dict[str, Path]:
    raw = bank_manifest.get("probe_work_order_paths")
    if not isinstance(raw, Mapping):
        raise DiagnosticProbeMaterializationPromptError("DIAGNOSTIC_PROBE_PROMPT_WORK_ORDERS_INVALID")
    output: dict[str, Path] = {}
    for probe_id, path in raw.items():
        if not isinstance(probe_id, str) or not probe_id or not isinstance(path, str) or not path:
            raise DiagnosticProbeMaterializationPromptError("DIAGNOSTIC_PROBE_PROMPT_WORK_ORDERS_INVALID")
        candidate = Path(path)
        output[probe_id] = (
            candidate.resolve(strict=True)
            if candidate.is_absolute()
            else (manifest_root / candidate).resolve(strict=True)
        )
    return output


def _load_work_order(payload: bytes) -> Mapping[str, Any]:
    try:
        work_order = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise DiagnosticProbeMaterializationPromptError("DIAGNOSTIC_PROBE_PROMPT_WORK_ORDER_INVALID") from exc
    if not isinstance(work_order, Mapping):
        raise DiagnosticProbeMaterializationPromptError("DIAGNOSTIC_PROBE_PROMPT_WORK_ORDER_INVALID")
    if work_order.get("role") != "diagnostic" or work_order.get("verdict_exposure_allowed") is not False:
        raise DiagnosticProbeMaterializationPromptError("DIAGNOSTIC_PROBE_PROMPT_WORK_ORDER_NOT_DIAGNOSTIC")
    if not isinstance(work_order.get("probe_id"), str) or not work_order["probe_id"]:
        raise DiagnosticProbeMaterializationPromptError("DIAGNOSTIC_PROBE_PROMPT_WORK_ORDER_INVALID")
    forbidden = work_order.get("forbidden_surfaces")
    if not isinstance(forbidden, list) or "verdict_evidence" not in forbidden:
        raise DiagnosticProbeMaterializationPromptError("DIAGNOSTIC_PROBE_PROMPT_WORK_ORDER_FORBIDDEN_SURFACES_INVALID")
    return work_order


def _write_child_prompt(
    *,
    destination: Path,
    prompts_root: Path,
    probe_id: str,
    work_order: Mapping[str, Any],
    work_order_path: Path,
    work_order_bytes: bytes,
    alignment_contract: Mapping[str, object],
) -> dict[str, object]:
    if work_order["probe_id"] != probe_id:
        raise DiagnosticProbeMaterializationPromptError("DIAGNOSTIC_PROBE_PROMPT_WORK_ORDER_PROBE_MISMATCH")
    child_root = prompts_root / probe_id
    child_root.mkdir(mode=0o700)
    inputs_root = child_root / "inputs"
    inputs_root.mkdir(mode=0o700)
    prompt_text = _render_prompt(work_order=work_order, alignment_contract=alignment_contract)
    report = {
        "schema_version": 1,
        "artifact_type": "wmloop-diagnostic-probe-materialization-prompt",
        "state": "ready",
        "probe_id": probe_id,
        "environment": work_order["environment"],
        "signature": work_order["signature"],
        "priority": work_order["priority"],
        "source_work_order_path": str(work_order_path),
        "work_order_sha256": hashlib.sha256(work_order_bytes).hexdigest(),
        "prompt_text": prompt_text,
        "alignment_contract": alignment_contract,
        "side_effects": {
            "source_code_mutated": False,
            "goal_config_mutated": False,
            "verdict_probe_mutated": False,
            "gpu_execution_started": False,
        },
    }
    _write_bytes_atomic(child_root / "diagnostic-probe-materialization-prompt.json", _canonical_json_bytes(report))
    _write_bytes_atomic(child_root / "diagnostic-probe-materialization-prompt.txt", prompt_text.encode("utf-8"))
    _write_bytes_atomic(child_root / "diagnostic-probe-materialization-prompt.md", _render_child_markdown(report).encode("utf-8"))
    _write_bytes_atomic(inputs_root / "work-order.json", work_order_bytes)
    return {
        "probe_id": probe_id,
        "environment": str(work_order["environment"]),
        "signature": str(work_order["signature"]),
        "priority": str(work_order["priority"]),
        "work_order_path": str(work_order_path),
        "work_order_sha256": report["work_order_sha256"],
        "prompt_path": str(destination / "prompts" / probe_id / "diagnostic-probe-materialization-prompt.txt"),
        "report_path": str(destination / "prompts" / probe_id / "diagnostic-probe-materialization-prompt.json"),
        "markdown_path": str(destination / "prompts" / probe_id / "diagnostic-probe-materialization-prompt.md"),
    }


def _alignment_contract() -> dict[str, object]:
    return {
        "must_not_change": [
            "configs/goal/",
            "configs/constitution/",
            "configs/eval_frozen.sha256",
            "configs/registry_frozen.sha256",
            "verdict_evidence",
            "frozen evaluator code",
            "primitive registry",
        ],
        "required_runtime_evidence": [
            "schema-valid diagnostic probe output",
            "offline fixture test",
            "runtime smoke on dev split before proposal routing uses the signal",
            "explicit no_verdict_evidence_exposure assertion",
        ],
        "promotion_boundary": (
            "A diagnostic probe may help proposal sorting only after its own admission gates pass. "
            "It may become verdict-facing only through a human-approved version boundary before a new campaign."
        ),
        "engineering_practice_policy": engineering_policy(),
    }


def _render_prompt(*, work_order: Mapping[str, Any], alignment_contract: Mapping[str, object]) -> str:
    lines = [
        "You are implementing one VerdiWM diagnostic probe materialization task.",
        "",
        "Hard contract:",
        "1. Implement only the diagnostic probe named below.",
        "2. The probe is diagnostic-only; do not expose its output to verdict_evidence or the frozen verifier.",
        "3. Mutate only allowed paths. If required logic touches a forbidden surface, stop and write a blocked report.",
        "4. Add fixture-level tests for schema, boundary behavior, and no verdict exposure.",
        "5. Do not change goal specs, constitution manifests, frozen evaluator code, primitive registry, or existing results.",
        "6. Do not start GPU jobs from this prompt; runtime smoke is a separate admission step.",
        "",
        f"Probe id: {work_order['probe_id']}",
        f"Environment: {work_order['environment']}",
        f"Signature: {work_order['signature']}",
        f"Priority: {work_order['priority']}",
        f"Signal contract: {work_order['signal_contract']}",
        "",
        "Allowed mutation paths:",
        *[f"- {path}" for path in _string_list(work_order.get("allowed_mutation_paths"))],
        "",
        "Forbidden surfaces:",
        *[f"- {path}" for path in _string_list(work_order.get("forbidden_surfaces"))],
        "",
        "Admission gates:",
        *[f"- {gate}" for gate in _string_list(work_order.get("admission_gates"))],
        "",
        "Expected implementation shape:",
        "- A pure measurement/aggregation function under wmloop/diagnose/probes/ or a staged adapter.",
        "- A JSON-like diagnostic output with probe_id, environment, signature, state, metrics, evidence_refs, and limitations.",
        "- Tests that fail if the probe output is routed into verdict_evidence.",
        "- A small CLI or callable entrypoint only if it can run offline on fixtures.",
        "",
        render_engineering_policy(alignment_contract.get("engineering_practice_policy")),
        "",
        "Alignment contract:",
        json.dumps(alignment_contract, ensure_ascii=False, sort_keys=True, indent=2),
        "",
        "Expected response shape:",
        "- Implement code and tests if feasible within allowed paths.",
        "- Run the narrow pytest suite and report exact commands.",
        "- If blocked, write the smallest durable blocker artifact and do not edit forbidden files.",
    ]
    return "\n".join(lines) + "\n"


def _render_child_markdown(report: Mapping[str, object]) -> str:
    return "\n".join(
        [
            "# Diagnostic Probe Materialization Prompt",
            "",
            f"Probe: `{report['probe_id']}`",
            f"Environment: `{report['environment']}`",
            f"Signature: `{report['signature']}`",
            "",
            "```text",
            str(report["prompt_text"]).rstrip(),
            "```",
            "",
        ]
    )


def _render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Diagnostic Probe Materialization Prompt Batch",
        "",
        f"State: `{report['state']}`",
        f"Prompt count: `{report['prompt_count']}`",
        "",
        "| Probe | Env | Signature | Priority | Prompt |",
        "|:--|:--|:--|:--|:--|",
    ]
    for record in report["records"]:
        lines.append(
            f"| `{record['probe_id']}` | `{record['environment']}` | `{record['signature']}` | `{record['priority']}` | `{record['prompt_path']}` |"
        )
    lines.extend(["", "## Boundary", "", str(report["alignment_contract"]["promotion_boundary"]), "", "## Side Effects", ""])
    for key, value in report["side_effects"].items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    return "\n".join(lines)


def _load_json_object(path: Path, code: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DiagnosticProbeMaterializationPromptError(code) from exc
    if not isinstance(payload, Mapping):
        raise DiagnosticProbeMaterializationPromptError(code)
    return payload


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str)]


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise DiagnosticProbeMaterializationPromptError("DIAGNOSTIC_PROBE_PROMPT_OUTPUT_EXISTS")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    batch = commands.add_parser("batch", help="generate guarded prompt packets for diagnostic probes")
    batch.add_argument("--repo-root", type=Path, default=Path("."))
    batch.add_argument("--failure-signature-bank-manifest", type=Path, required=True)
    batch.add_argument("--probe-id", action="append", default=[])
    batch.add_argument("--output-root", type=Path, required=True)
    batch.add_argument("--archive-db", type=Path)
    batch.add_argument("--cas-root", type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "batch":
        manifest = run_diagnostic_probe_materialization_prompt_batch(
            repo_root=args.repo_root,
            failure_signature_bank_manifest=args.failure_signature_bank_manifest,
            probe_ids=tuple(args.probe_id),
            output_root=args.output_root,
            archive_db=args.archive_db,
            cas_root=args.cas_root,
        )
        print(json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
