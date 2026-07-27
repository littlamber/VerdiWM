"""Generate a guarded prompt packet for materializing one primitive into code."""

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


class PrimitiveMaterializationPromptError(RuntimeError):
    """A primitive materialization prompt packet could not be produced."""


def run_primitive_materialization_prompt(
    *,
    repo_root: Path,
    work_order: Path,
    output_root: Path,
    intent_compilation_manifest: Path | None = None,
    archive_db: Path | None = None,
    cas_root: Path | None = None,
) -> dict[str, object]:
    """Write a prompt packet that keeps agent code generation inside gates."""

    root = Path(repo_root).resolve()
    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise PrimitiveMaterializationPromptError("PRIMITIVE_MATERIALIZATION_PROMPT_OUTPUT_EXISTS")
    work_order_source = Path(work_order).resolve(strict=True)
    work_order_bytes = work_order_source.read_bytes()
    try:
        packet = json.loads(work_order_bytes)
    except json.JSONDecodeError as exc:
        raise PrimitiveMaterializationPromptError("PRIMITIVE_MATERIALIZATION_WORK_ORDER_INVALID") from exc
    if not isinstance(packet, Mapping) or packet.get("artifact_type") != "wmloop-primitive-materialization-work-order":
        raise PrimitiveMaterializationPromptError("PRIMITIVE_MATERIALIZATION_WORK_ORDER_INVALID")

    intent = _load_intent(intent_compilation_manifest)
    primitive = _string(packet, "primitive")
    alignment_contract = _alignment_contract(packet=packet, intent=intent)
    prompt_text = _render_prompt(packet=packet, intent=intent, alignment_contract=alignment_contract)
    side_effects = {
        "source_code_mutated": False,
        "goal_config_mutated": False,
        "protocol_changed": False,
        "registry_changed": False,
        "gpu_execution_started": False,
        "primitive_promoted": False,
    }
    report = {
        "schema_version": 1,
        "artifact_type": "wmloop-primitive-materialization-prompt",
        "state": "ready",
        "primitive": primitive,
        "source_work_order_path": str(work_order_source),
        "work_order_sha256": hashlib.sha256(work_order_bytes).hexdigest(),
        "intent_compilation": intent,
        "alignment_contract": alignment_contract,
        "prompt_text": prompt_text,
        "side_effects": side_effects,
        "limitations": [
            "This packet is an instruction artifact; it does not edit files or promote the primitive.",
            "A generated implementation remains sidecar-only or hook-only until the materialization gate sees runtime evidence.",
            "The prompt cannot override forbidden paths, frozen evaluator policy, active goal config, or M4 launch gates.",
        ],
    }
    validate_document("primitive_materialization_prompt", report, root=root)

    cas_storage_root = (
        Path(cas_root).resolve()
        if cas_root is not None
        else (Path(archive_db).resolve().parent if archive_db is not None else destination.parent)
    )
    cas = ContentAddressedStore(cas_storage_root)
    archive = ArchiveStore(archive_db) if archive_db is not None else None
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    try:
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary.mkdir(mode=0o700)
        inputs_dir = temporary / "inputs"
        inputs_dir.mkdir(mode=0o700)

        report_bytes = _canonical_json_bytes(report)
        prompt_bytes = prompt_text.encode("utf-8")
        markdown_bytes = _render_markdown(report).encode("utf-8")
        refs = {
            "primitive_materialization_prompt_json": cas.put_bytes(report_bytes, media_type="application/json").uri,
            "primitive_materialization_prompt_text": cas.put_bytes(prompt_bytes, media_type="text/plain").uri,
            "primitive_materialization_prompt_markdown": cas.put_bytes(markdown_bytes, media_type="text/markdown").uri,
            "work_order": cas.put_bytes(work_order_bytes, media_type="application/json").uri,
        }
        if archive is not None:
            for ref in refs.values():
                archive.record_artifact_reference(ref)

        _write_bytes_atomic(temporary / "primitive-materialization-prompt.json", report_bytes)
        _write_bytes_atomic(temporary / "primitive-materialization-prompt.txt", prompt_bytes)
        _write_bytes_atomic(temporary / "primitive-materialization-prompt.md", markdown_bytes)
        _write_bytes_atomic(inputs_dir / "work-order.json", work_order_bytes)

        manifest = {
            "schema_version": 1,
            "artifact_type": "wmloop-primitive-materialization-prompt-manifest",
            "state": "ready",
            "primitive": primitive,
            "source_work_order_path": str(work_order_source),
            "work_order_sha256": report["work_order_sha256"],
            "prompt_path": str(destination / "primitive-materialization-prompt.txt"),
            "report_path": str(destination / "primitive-materialization-prompt.json"),
            "markdown_path": str(destination / "primitive-materialization-prompt.md"),
            "input_snapshot_dir": str(destination / "inputs"),
            "cas_refs": refs,
            "side_effects": side_effects,
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


def run_primitive_materialization_prompt_batch(
    *,
    repo_root: Path,
    materialization_gate_manifest: Path,
    output_root: Path,
    intent_compilation_manifest: Path | None = None,
    primitives: Sequence[str] = (),
    archive_db: Path | None = None,
    cas_root: Path | None = None,
) -> dict[str, object]:
    """Generate guarded prompt packets for work orders emitted by the gate."""

    root = Path(repo_root).resolve()
    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise PrimitiveMaterializationPromptError("PRIMITIVE_MATERIALIZATION_PROMPT_BATCH_OUTPUT_EXISTS")
    gate_path = Path(materialization_gate_manifest).resolve(strict=True)
    gate_bytes = gate_path.read_bytes()
    try:
        gate = json.loads(gate_bytes)
    except json.JSONDecodeError as exc:
        raise PrimitiveMaterializationPromptError("PRIMITIVE_MATERIALIZATION_GATE_MANIFEST_INVALID") from exc
    if not isinstance(gate, Mapping) or gate.get("artifact_type") != "wmloop-primitive-materialization-gate-manifest":
        raise PrimitiveMaterializationPromptError("PRIMITIVE_MATERIALIZATION_GATE_MANIFEST_INVALID")
    work_orders = _selected_work_orders(gate, primitives=primitives)
    intent = _load_intent(intent_compilation_manifest)

    child_records: list[dict[str, object]] = []
    try:
        destination.mkdir(mode=0o700, parents=True)
        prompts_root = destination / "prompts"
        prompts_root.mkdir(mode=0o700)
        for primitive, work_order_path in work_orders:
            child_output = prompts_root / primitive
            child_manifest = run_primitive_materialization_prompt(
                repo_root=root,
                work_order=work_order_path,
                intent_compilation_manifest=intent_compilation_manifest,
                output_root=child_output,
                archive_db=archive_db,
                cas_root=cas_root,
            )
            child_records.append(
                {
                    "primitive": primitive,
                    "state": child_manifest["state"],
                    "work_order_path": str(work_order_path.resolve()),
                    "prompt_manifest_path": str(child_output / "manifest.json"),
                    "prompt_path": child_manifest["prompt_path"],
                    "report_path": child_manifest["report_path"],
                }
            )

        report = {
            "schema_version": 1,
            "artifact_type": "wmloop-primitive-materialization-prompt-batch",
            "state": "ready",
            "source_materialization_gate": {
                "manifest_path": str(gate_path),
                "manifest_sha256": hashlib.sha256(gate_bytes).hexdigest(),
                "state": gate.get("state"),
                "primitive_count": gate.get("primitive_count"),
                "closed_loop_ready_count": gate.get("closed_loop_ready_count"),
                "sidecar_only_count": gate.get("sidecar_only_count"),
                "work_order_count": len(gate.get("work_order_paths", {})) if isinstance(gate.get("work_order_paths"), Mapping) else None,
            },
            "intent_compilation": intent,
            "prompt_count": len(child_records),
            "primitives": [record["primitive"] for record in child_records],
            "records": child_records,
            "side_effects": {
                "source_code_mutated": False,
                "goal_config_mutated": False,
                "protocol_changed": False,
                "registry_changed": False,
                "gpu_execution_started": False,
                "primitive_promoted": False,
            },
            "limitations": [
                "This batch only stages prompt packets from gate work orders; it does not implement code.",
                "Each child prompt inherits the same forbidden-path, WM-Dx routing, and probe-evolution boundaries.",
                "Formal M4 launch remains controlled by phase_gate and is not granted by this batch.",
            ],
        }
        report_bytes = _canonical_json_bytes(report)
        markdown_bytes = _render_batch_markdown(report).encode("utf-8")
        cas_storage_root = (
            Path(cas_root).resolve()
            if cas_root is not None
            else (Path(archive_db).resolve().parent if archive_db is not None else destination.parent)
        )
        cas = ContentAddressedStore(cas_storage_root)
        archive = ArchiveStore(archive_db) if archive_db is not None else None
        refs = {
            "primitive_materialization_prompt_batch_json": cas.put_bytes(report_bytes, media_type="application/json").uri,
            "primitive_materialization_prompt_batch_markdown": cas.put_bytes(markdown_bytes, media_type="text/markdown").uri,
            "materialization_gate_manifest": cas.put_bytes(gate_bytes, media_type="application/json").uri,
        }
        if archive is not None:
            for ref in refs.values():
                archive.record_artifact_reference(ref)
        _write_bytes_atomic(destination / "primitive-materialization-prompt-batch.json", report_bytes)
        _write_bytes_atomic(destination / "primitive-materialization-prompt-batch.md", markdown_bytes)
        manifest = {
            "schema_version": 1,
            "artifact_type": "wmloop-primitive-materialization-prompt-batch-manifest",
            "state": "ready",
            "source_materialization_gate": report["source_materialization_gate"],
            "intent_compilation": intent,
            "prompt_count": len(child_records),
            "primitives": report["primitives"],
            "record_count": len(child_records),
            "report_path": str(destination / "primitive-materialization-prompt-batch.json"),
            "markdown_path": str(destination / "primitive-materialization-prompt-batch.md"),
            "prompt_root": str(prompts_root),
            "records": child_records,
            "cas_refs": refs,
            "side_effects": report["side_effects"],
            "limitations": report["limitations"],
        }
        _write_bytes_atomic(destination / "manifest.json", _canonical_json_bytes(manifest))
        return manifest
    except Exception:
        if destination.is_symlink():
            destination.unlink()
        elif destination.exists():
            shutil.rmtree(destination)
        raise


def _selected_work_orders(
    gate: Mapping[str, Any],
    *,
    primitives: Sequence[str],
) -> list[tuple[str, Path]]:
    raw = gate.get("work_order_paths")
    if not isinstance(raw, Mapping):
        raise PrimitiveMaterializationPromptError("PRIMITIVE_MATERIALIZATION_GATE_WORK_ORDERS_INVALID")
    available: dict[str, Path] = {}
    for primitive, path in raw.items():
        if not isinstance(primitive, str) or not primitive or not isinstance(path, str) or not path:
            raise PrimitiveMaterializationPromptError("PRIMITIVE_MATERIALIZATION_GATE_WORK_ORDERS_INVALID")
        available[primitive] = Path(path).resolve(strict=True)
    requested = [primitive for primitive in primitives if primitive]
    if not requested:
        return sorted(available.items())
    missing = [primitive for primitive in requested if primitive not in available]
    if missing:
        raise PrimitiveMaterializationPromptError(
            "PRIMITIVE_MATERIALIZATION_PROMPT_BATCH_UNKNOWN_PRIMITIVE:" + ",".join(sorted(missing))
        )
    return [(primitive, available[primitive]) for primitive in requested]


def _load_intent(path: Path | None) -> dict[str, object]:
    if path is None:
        return {
            "provided": False,
            "state": "not_provided",
            "intent_binding_ready": None,
            "limited_execution_allowed": None,
            "formal_m4_launch_permission_granted": False,
        }
    manifest_path = Path(path).resolve(strict=True)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("artifact_type") != "wmloop-user-intent-compilation-manifest":
        raise PrimitiveMaterializationPromptError("PRIMITIVE_MATERIALIZATION_INTENT_MANIFEST_INVALID")
    return {
        "provided": True,
        "manifest_path": str(manifest_path),
        "state": payload.get("state"),
        "goal_id": payload.get("goal_id"),
        "compiled_goal_family": payload.get("compiled_goal_family"),
        "compiled_backbone_family": payload.get("compiled_backbone_family"),
        "compiled_environment_scope": payload.get("compiled_environment_scope"),
        "intent_binding_ready": payload.get("intent_binding_ready"),
        "limited_execution_allowed": payload.get("limited_execution_allowed"),
        "formal_m4_launch_permission_granted": payload.get("formal_m4_launch_permission_granted", False),
        "blockers": payload.get("blockers", []),
    }


def _alignment_contract(*, packet: Mapping[str, Any], intent: Mapping[str, object]) -> dict[str, object]:
    forbidden = _strings(packet.get("forbidden_paths"))
    allowed = _strings(packet.get("allowed_mutation_paths"))
    gates = _strings(packet.get("required_gates"))
    practice_policy = engineering_policy()
    return {
        "configuration_sources_of_truth": [
            "primitive work order",
            "frozen primitive registry",
            "active goal_spec or user-intent compilation packet",
            "runtime smoke/campaign manifests",
        ],
        "must_not_change": [
            "eval.py",
            "scripts/eval_all.sh",
            "configs/goal/",
            "runs/m0/protocol/",
            "results/",
            "primitive registry digest",
            "verdict probes or constitutional freeze",
        ],
        "allowed_mutation_paths": allowed,
        "forbidden_paths": forbidden,
        "required_runtime_evidence": [
            "schema-valid rendered primitive sidecar",
            "clean primitive_apply_audit with no forbidden diff",
            "runtime hook smoke for the selected primitive",
            "GPU training/eval smoke before closed-loop promotion",
        ],
        "required_gates": gates,
        "configuration_intent_alignment_rule": (
            "Every generated runtime artifact must expose the primitive name, hook layer, params, "
            "and source work-order hash so reviewers can compare configuration intent with actual execution."
        ),
        "faithful_materialization_rule": (
            "The implementation must faithfully realize the selected primitive's stated method intent. "
            "Implementation difficulty, missing helper APIs, performance concerns, or integration friction "
            "are not acceptable reasons to silently substitute a weaker behavior, sidecar-only stub, "
            "different primitive, disabled hook, or evaluator-side proxy."
        ),
        "compromise_blocker_policy": (
            "If faithful implementation cannot be completed within the allowed mutation paths and runtime "
            "contract, fail closed with a blocker report that names the unmet intent, attempted touchpoints, "
            "required forbidden or missing surfaces, and validation receipts that could not be produced."
        ),
        "required_intent_to_code_receipts": [
            "configuration intent summary mapped to exact code touchpoints",
            "runtime behavior contract covering when the hook activates and what tensors/losses/state it changes",
            "negative check proving a disabled or substituted implementation does not pass admission",
            "validation receipts that compare rendered config intent with patched runtime behavior",
            "declared compromises or blockers, with no silent substitutions",
        ],
        "diagnosis_routing_rule": (
            "Do not choose a new primitive inside this task; the primitive is already selected upstream "
            "from WM-Dx failure_report routing."
        ),
        "probe_evolution_rule": (
            "This task may not modify verdict probes. Diagnostic probe evolution belongs to a separate "
            "Zone-B admission workflow and must not change verifier evidence during a campaign."
        ),
        "engineering_practice_policy": practice_policy,
        "intent_limited_execution_allowed": intent.get("limited_execution_allowed"),
        "intent_formal_m4_launch_permission_granted": intent.get("formal_m4_launch_permission_granted", False),
    }


def _render_prompt(
    *,
    packet: Mapping[str, Any],
    intent: Mapping[str, object],
    alignment_contract: Mapping[str, object],
) -> str:
    primitive = _string(packet, "primitive")
    params_schema = json.dumps(packet.get("params_schema", {}), ensure_ascii=False, sort_keys=True)
    lines = [
        "You are implementing one VerdiWM primitive materialization task.",
        "",
        "Hard contract:",
        "1. Implement only the primitive named below. Do not select or substitute a different primitive.",
        "2. Keep configuration intent and engineering behavior aligned: runtime artifacts must prove the code path used the primitive, hook, params, and work-order hash.",
        "3. No silent implementation compromise: difficulty, missing helpers, speed, or integration friction must not turn the primitive into a weaker proxy, disabled hook, sidecar-only stub, evaluator-side trick, or different method.",
        "4. If the code cannot faithfully realize the configured primitive intent inside the allowed runtime surfaces, fail closed with a blocker report instead of pretending success.",
        "5. Mutate only allowed files. If the implementation requires forbidden paths, stop and return a blocked report instead of writing code.",
        "6. Do not modify goal specs, held-out protocol, frozen evaluator code, verdict probes, primitive registry digest, or results artifacts.",
        "7. Do not claim closed-loop readiness. Readiness is granted only by primitive_materialization_gate after runtime evidence.",
        "8. Prefer a small ACWM hook plus tests over broad refactors.",
        "",
        f"Primitive: {primitive}",
        f"Layer: {packet.get('layer')}",
        f"Hooks: {json.dumps(packet.get('hooks', []), ensure_ascii=False)}",
        f"Targets failures: {json.dumps(packet.get('targets_failures', []), ensure_ascii=False)}",
        f"Params schema: {params_schema}",
        f"Current admission state: {packet.get('current_admission_state')}",
        f"Target admission state: {packet.get('target_admission_state')}",
        "",
        "Allowed mutation paths:",
        *[f"- {path}" for path in _strings(packet.get("allowed_mutation_paths"))],
        "",
        "Forbidden paths:",
        *[f"- {path}" for path in _strings(packet.get("forbidden_paths"))],
        "",
        "Required gates after implementation:",
        *[f"- {gate}" for gate in _strings(packet.get("required_gates"))],
        "",
        "Configuration-intent alignment checks to add:",
        "- Map the primitive's method intent to exact code touchpoints before implementing.",
        "- Sidecar JSON contains primitive name, params, layer/hook, and work-order hash.",
        "- Runtime hook reads the sidecar, validates params, and no-ops safely when disabled.",
        "- Tests fail if rendered config intent differs from patched runtime behavior.",
        "- Tests fail if a stub, weaker substitute, disabled hook, or evaluator-side proxy tries to satisfy the primitive.",
        "- Apply audit must detect attempts to touch frozen evaluator/protocol paths.",
        "- Any unavoidable compromise is recorded as a blocker, not hidden behind a passing implementation.",
        "",
        "WM-Dx routing boundary:",
        "- Upstream diagnosis comes from probe-backed failure_report records.",
        "- This task consumes the selected primitive; it does not change diagnosis thresholds or probe roles.",
        "",
        "Probe-evolution boundary:",
        "- Verdict probes are frozen constitutional surfaces.",
        "- Diagnostic probe candidates need their own schema/diff/runtime/admission workflow before use.",
        "",
        render_engineering_policy(alignment_contract.get("engineering_practice_policy")),
        "",
        "Intent packet summary:",
        json.dumps(intent, ensure_ascii=False, sort_keys=True, indent=2),
        "",
        "Alignment contract:",
        json.dumps(alignment_contract, ensure_ascii=False, sort_keys=True, indent=2),
        "",
        "Expected response shape:",
        "- Implement code and tests if feasible within the allowed paths.",
        "- Run the narrow pytest suite and report exact commands.",
        "- If blocked, write the smallest durable blocker artifact and do not edit forbidden files.",
    ]
    return "\n".join(lines) + "\n"


def _render_markdown(report: Mapping[str, object]) -> str:
    lines = [
        "# Primitive Materialization Prompt",
        "",
        f"State: `{report['state']}`",
        f"Primitive: `{report['primitive']}`",
        f"Work order: `{report['source_work_order_path']}`",
        "",
        "## Side Effects",
        "",
    ]
    side_effects = report.get("side_effects")
    if isinstance(side_effects, Mapping):
        for key, value in side_effects.items():
            lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Prompt", "", "```text", str(report["prompt_text"]).rstrip(), "```", ""])
    return "\n".join(lines)


def _render_batch_markdown(report: Mapping[str, object]) -> str:
    lines = [
        "# Primitive Materialization Prompt Batch",
        "",
        f"State: `{report['state']}`",
        f"Prompt count: `{report['prompt_count']}`",
        "",
        "| Primitive | State | Prompt |",
        "|:--|:--|:--|",
    ]
    records = report.get("records")
    if isinstance(records, list):
        for record in records:
            if isinstance(record, Mapping):
                lines.append(f"| `{record.get('primitive')}` | `{record.get('state')}` | `{record.get('prompt_path')}` |")
    lines.extend(["", "## Limitations", ""])
    limitations = report.get("limitations")
    if isinstance(limitations, list):
        lines.extend(f"- {item}" for item in limitations)
    return "\n".join(lines) + "\n"


def _strings(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str)]


def _string(document: Mapping[str, Any], key: str) -> str:
    value = document.get(key)
    if not isinstance(value, str) or not value:
        raise PrimitiveMaterializationPromptError(f"PRIMITIVE_MATERIALIZATION_WORK_ORDER_FIELD_INVALID:{key}")
    return value


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8") + b"\n"


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise PrimitiveMaterializationPromptError(f"PRIMITIVE_MATERIALIZATION_PROMPT_OUTPUT_EXISTS:{path}")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="generate a guarded primitive materialization prompt packet")
    run.add_argument("--repo-root", type=Path, default=Path("."))
    run.add_argument("--work-order", type=Path, required=True)
    run.add_argument("--intent-compilation-manifest", type=Path)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--archive-db", type=Path)
    run.add_argument("--cas-root", type=Path)
    batch = commands.add_parser("batch", help="generate guarded prompt packets for a materialization gate")
    batch.add_argument("--repo-root", type=Path, default=Path("."))
    batch.add_argument("--materialization-gate-manifest", type=Path, required=True)
    batch.add_argument("--intent-compilation-manifest", type=Path)
    batch.add_argument("--primitive", action="append", default=[])
    batch.add_argument("--output-root", type=Path, required=True)
    batch.add_argument("--archive-db", type=Path)
    batch.add_argument("--cas-root", type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "run":
        manifest = run_primitive_materialization_prompt(
            repo_root=args.repo_root,
            work_order=args.work_order,
            intent_compilation_manifest=args.intent_compilation_manifest,
            output_root=args.output_root,
            archive_db=args.archive_db,
            cas_root=args.cas_root,
        )
        print(json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
        return 0
    if args.command == "batch":
        manifest = run_primitive_materialization_prompt_batch(
            repo_root=args.repo_root,
            materialization_gate_manifest=args.materialization_gate_manifest,
            intent_compilation_manifest=args.intent_compilation_manifest,
            primitives=tuple(args.primitive),
            output_root=args.output_root,
            archive_db=args.archive_db,
            cas_root=args.cas_root,
        )
        print(json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
