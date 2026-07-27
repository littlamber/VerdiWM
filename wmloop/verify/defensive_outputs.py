"""Export T5.5 defensive taxonomy alignment and tau_AF sensitivity tables."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from wmloop.archive.store import ArchiveStore, ContentAddressedStore
from wmloop.verify.judge import VerificationEvidence, judge


class DefensiveOutputsError(RuntimeError):
    """The defensive-output export failed closed."""


TAU_MULTIPLIERS = (0.5, 1.0, 2.0)
FALLBACK_TAU_AF = 0.55


def run_defensive_outputs(
    *,
    output_root: Path,
    goal_spec: Path,
    probe_registry: Path,
    archive_db: Path | None = None,
    cas_root: Path | None = None,
    base_tau_af: float | None = None,
) -> dict[str, object]:
    """Write T5.5 paper-facing defensive outputs without launching evaluation."""

    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise DefensiveOutputsError("DEFENSIVE_OUTPUT_EXISTS")
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    goal = _load_json_mapping(Path(goal_spec).resolve(strict=True), "DEFENSIVE_GOAL_SPEC_INVALID")
    probes = _load_json_mapping(Path(probe_registry).resolve(strict=True), "DEFENSIVE_PROBE_REGISTRY_INVALID")
    tau_value, tau_source = _resolve_tau(goal=goal, override=base_tau_af)
    taxonomy_rows = _taxonomy_rows(goal=goal, probes=probes)
    sensitivity_rows, sensitivity_case_rows = _sensitivity_rows(base_tau=tau_value)
    blockers = _blockers(taxonomy_rows=taxonomy_rows, sensitivity_rows=sensitivity_rows)
    state = "ready" if not blockers else "blocked"
    archive = ArchiveStore(archive_db) if archive_db is not None else None
    cas_storage_root = (
        Path(cas_root).resolve()
        if cas_root is not None
        else (Path(archive_db).resolve().parent if archive_db is not None else destination.parent)
    )
    cas = ContentAddressedStore(cas_storage_root)
    report = {
        "schema_version": 1,
        "artifact_type": "wmloop-t5-5-defensive-outputs-report",
        "state": state,
        "defensive_outputs_complete": state == "ready",
        "goal_spec": str(Path(goal_spec).resolve()),
        "probe_registry": str(Path(probe_registry).resolve()),
        "base_tau_af": tau_value,
        "base_tau_af_source": tau_source,
        "taxonomy_row_count": len(taxonomy_rows),
        "sensitivity_row_count": len(sensitivity_rows),
        "sensitivity_case_count": len(sensitivity_case_rows),
        "blockers": blockers,
        "taxonomy_rows": taxonomy_rows,
        "sensitivity_rows": sensitivity_rows,
        "sensitivity_case_rows": sensitivity_case_rows,
        "limitations": [
            "The taxonomy table is an alignment artifact for paper claims; uncovered modes are explicitly marked as gated extensions.",
            "The tau_AF sweep uses constructed verifier evidence and does not launch model training or evaluation.",
            "Changing tau_AF for the formal campaign remains a version-boundary decision; this export is sensitivity analysis only.",
        ],
    }
    files = {
        "defensive_outputs_json": ("defensive-outputs.json", _canonical_json_bytes(report), "application/json"),
        "defensive_outputs_markdown": (
            "defensive-outputs.md",
            _render_markdown(report).encode("utf-8"),
            "text/markdown",
        ),
        "failure_taxonomy_alignment_csv": (
            "tables/failure-taxonomy-alignment.csv",
            _csv_bytes(taxonomy_rows),
            "text/csv",
        ),
        "tau_af_sensitivity_csv": ("tables/tau-af-sensitivity.csv", _csv_bytes(sensitivity_rows), "text/csv"),
        "tau_af_sensitivity_cases_csv": (
            "tables/tau-af-sensitivity-cases.csv",
            _csv_bytes(sensitivity_case_rows),
            "text/csv",
        ),
        "failure_taxonomy_alignment_latex": (
            "latex/failure-taxonomy-alignment.tex",
            _latex_table(taxonomy_rows, "Failure mode taxonomy alignment").encode("utf-8"),
            "text/x-tex",
        ),
        "tau_af_sensitivity_latex": (
            "latex/tau-af-sensitivity.tex",
            _latex_table(sensitivity_rows, "tau_AF sensitivity").encode("utf-8"),
            "text/x-tex",
        ),
    }
    try:
        temporary.mkdir(mode=0o700)
        cas_refs: dict[str, str] = {}
        for key, (relative, payload, media_type) in files.items():
            _write_bytes_atomic(temporary / relative, payload)
            ref = cas.put_bytes(payload, media_type=media_type).uri
            if archive is not None:
                archive.record_artifact_reference(ref)
            cas_refs[key] = ref
        manifest = {
            "schema_version": 1,
            "artifact_type": "wmloop-t5-5-defensive-outputs-manifest",
            "state": state,
            "defensive_outputs_complete": state == "ready",
            "taxonomy_row_count": len(taxonomy_rows),
            "sensitivity_row_count": len(sensitivity_rows),
            "sensitivity_case_count": len(sensitivity_case_rows),
            "base_tau_af": tau_value,
            "base_tau_af_source": tau_source,
            "blocker_count": len(blockers),
            "blockers": blockers,
            "report_path": str(destination / "defensive-outputs.json"),
            "markdown_path": str(destination / "defensive-outputs.md"),
            "tables_dir": str(destination / "tables"),
            "latex_dir": str(destination / "latex"),
            "cas_refs": cas_refs,
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


def _resolve_tau(*, goal: Mapping[str, Any], override: float | None) -> tuple[float, str]:
    if override is not None:
        if not math.isfinite(override) or override <= 0.0 or override > 1.0:
            raise DefensiveOutputsError("DEFENSIVE_TAU_OVERRIDE_INVALID")
        return override, "cli_override"
    gate = goal.get("action_following_gate")
    if isinstance(gate, Mapping):
        raw = gate.get("tau_af")
        if isinstance(raw, (int, float)) and not isinstance(raw, bool) and math.isfinite(float(raw)) and float(raw) > 0.0:
            return float(raw), "goal_spec.action_following_gate.tau_af"
    return FALLBACK_TAU_AF, "fallback_verifier_smoke_threshold"


def _taxonomy_rows(*, goal: Mapping[str, Any], probes: Mapping[str, Any]) -> list[dict[str, object]]:
    envs = goal.get("envs") if isinstance(goal.get("envs"), list) else []
    env_list = ",".join(str(env) for env in envs)
    _probe_ids(probes)
    rows = [
        _tax_row(
            "long_horizon_compounding_drift",
            "WMBench/GigaWorld-style temporal consistency failure",
            "horizon_curve",
            "verdict",
            env_list,
            "covered",
            "Per-environment horizon ladder measures long-horizon AUC and segment drift.",
        ),
        _tax_row(
            "action_condition_ignoring_static_solution",
            "Action-conditioned world model degeneration",
            "action_following",
            "verdict",
            env_list,
            "covered",
            "AF gate blocks static/action-ignoring solutions from receiving verified metric credit.",
        ),
        _tax_row(
            "appearance_texture_drift",
            "Visual persistence and appearance stability failure",
            "appearance_drift",
            "diagnostic",
            "cloth_move,push_sand,pour_water,push_rope",
            "covered_diagnostic_only",
            "Used for diagnosis and routing; it does not independently certify ACCEPT.",
        ),
        _tax_row(
            "ood_physics_generalization_gap",
            "Out-of-distribution physics or dynamics gap",
            "ood_profile",
            "diagnostic",
            env_list,
            "covered_diagnostic_only",
            "InD-vs-OoD gap informs failure attribution and proposal routing.",
        ),
        _tax_row(
            "contact_dynamics_discontinuity",
            "Rigid/deformable contact and force-transfer failure",
            "horizon_curve;action_following",
            "verdict",
            "push_cube,stack_cube,push_rope,cloth_move,push_sand,pour_water",
            "partially_covered",
            "Covered through trajectory degradation and action-following behavior, not direct force probes.",
        ),
        _tax_row(
            "object_relation_spatial_memory",
            "Object identity, relation, and spatial memory failure",
            "horizon_curve;appearance_drift",
            "mixed",
            "push_cube,stack_cube,robot_arm,reacher",
            "partially_covered",
            "Local protocol observes downstream rollout and appearance symptoms; explicit relation probes are a gated extension.",
        ),
        _tax_row(
            "evaluation_or_split_exploitation",
            "Verifier, metric, or held-out split exploitation",
            "G1_readonly;G2_heldout;G3_audit",
            "gate",
            env_list,
            "covered_by_adversarial_gate",
            "T5.1 adversarial audit covers eval-path tamper, hardcoded metrics, and held-out contamination.",
        ),
        _tax_row(
            "language_or_task_semantic_misalignment",
            "Task/language semantic mismatch in WAM-style systems",
            "gated_extension",
            "extension",
            "none_active_in_G1_ACWM",
            "not_covered_in_current_goal",
            "Documented as a G2/WAM extension path, not claimed for the current ACWM G1 campaign.",
        ),
    ]
    return rows


def _tax_row(
    failure_mode: str,
    literature_category: str,
    local_probe_ids: str,
    probe_role: str,
    local_environments: str,
    coverage_state: str,
    alignment_note: str,
) -> dict[str, object]:
    return {
        "failure_mode": failure_mode,
        "literature_category": literature_category,
        "local_probe_ids": local_probe_ids,
        "probe_role": probe_role,
        "local_environments": local_environments,
        "coverage_state": coverage_state,
        "alignment_note": alignment_note,
    }


def _probe_ids(probes: Mapping[str, Any]) -> set[str]:
    raw = probes.get("probes")
    if not isinstance(raw, list):
        raise DefensiveOutputsError("DEFENSIVE_PROBE_REGISTRY_INVALID")
    ids: set[str] = set()
    for item in raw:
        if not isinstance(item, Mapping) or not isinstance(item.get("id"), str):
            raise DefensiveOutputsError("DEFENSIVE_PROBE_REGISTRY_INVALID")
        ids.add(str(item["id"]))
    return ids


def _sensitivity_rows(*, base_tau: float) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    summary_rows: list[dict[str, object]] = []
    case_rows: list[dict[str, object]] = []
    for multiplier in TAU_MULTIPLIERS:
        unclipped_tau = base_tau * multiplier
        tau = min(1.0, unclipped_tau)
        cases = [
            ("legitimate_accept", 0.61, "legitimate"),
            ("static_degradation", 0.10, "static_degradation"),
        ]
        verdicts = []
        for case_name, observed, case_family in cases:
            result = judge(_evidence(tau=tau, observed=observed, proposal_id=f"t55-{case_name}-{multiplier:g}")).to_dict()
            static_intercepted = (
                case_family == "static_degradation"
                and result["verdict"] != "ACCEPT"
                and result["violation"] == "AF_GATE_FAILED"
                and all(float(value) == 0.0 for value in result["delta_m_ver"].values())
            )
            case_row = {
                "tau_multiplier": multiplier,
                "tau_af_unclipped": unclipped_tau,
                "tau_af": tau,
                "tau_clipped": tau != unclipped_tau,
                "case": case_name,
                "case_family": case_family,
                "action_following_observed": observed,
                "verdict": result["verdict"],
                "violation": result["violation"],
                "action_following_pass": result["action_following_gate"]["pass"],
                "static_degradation_intercepted": static_intercepted,
            }
            verdicts.append(case_row)
            case_rows.append(case_row)
        static_cases = [row for row in verdicts if row["case_family"] == "static_degradation"]
        legitimate_cases = [row for row in verdicts if row["case_family"] == "legitimate"]
        summary_rows.append(
            {
                "tau_multiplier": multiplier,
                "tau_af_unclipped": unclipped_tau,
                "tau_af": tau,
                "tau_clipped": tau != unclipped_tau,
                "case_count": len(verdicts),
                "accept_count": sum(1 for row in verdicts if row["verdict"] == "ACCEPT"),
                "accept_rate": sum(1 for row in verdicts if row["verdict"] == "ACCEPT") / len(verdicts),
                "legitimate_accept_rate": sum(1 for row in legitimate_cases if row["verdict"] == "ACCEPT")
                / len(legitimate_cases),
                "static_degradation_case_count": len(static_cases),
                "static_degradation_intercepted_count": sum(
                    1 for row in static_cases if row["static_degradation_intercepted"] is True
                ),
                "static_degradation_interception_rate": sum(
                    1 for row in static_cases if row["static_degradation_intercepted"] is True
                )
                / len(static_cases),
            }
        )
    return summary_rows, case_rows


def _evidence(*, tau: float, observed: float, proposal_id: str) -> VerificationEvidence:
    return VerificationEvidence(
        proposal_id=proposal_id,
        readonly_evaluator_verified=True,
        accept_split_verified=True,
        extended_horizon_verified=True,
        diff_audit_passed=True,
        evidence_complete=True,
        accept_metric_deltas={"auc_psnr_16_64": 1.0},
        replication_deltas=[0.9, 1.0, 1.1],
        action_following_observed=observed,
        action_following_threshold=tau,
    )


def _blockers(
    *,
    taxonomy_rows: Sequence[Mapping[str, object]],
    sensitivity_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    blockers: list[dict[str, object]] = []
    if len(taxonomy_rows) < 8:
        blockers.append({"code": "DEFENSIVE_TAXONOMY_ROWS_INSUFFICIENT", "observed": len(taxonomy_rows)})
    if len(sensitivity_rows) < 3:
        blockers.append({"code": "DEFENSIVE_TAU_SWEEP_ROWS_INSUFFICIENT", "observed": len(sensitivity_rows)})
    if any(row.get("static_degradation_interception_rate") != 1.0 for row in sensitivity_rows):
        blockers.append({"code": "DEFENSIVE_STATIC_DEGRADATION_NOT_INTERCEPTED"})
    return blockers


def _load_json_mapping(path: Path, code: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DefensiveOutputsError(code) from exc
    if not isinstance(payload, Mapping):
        raise DefensiveOutputsError(code)
    return payload


def _render_markdown(report: Mapping[str, object]) -> str:
    lines = [
        "# T5.5 Defensive Outputs",
        "",
        f"State: `{report['state']}`",
        f"Complete: `{report['defensive_outputs_complete']}`",
        f"Base tau_AF: `{report['base_tau_af']}` ({report['base_tau_af_source']})",
        "",
        "## Taxonomy Alignment",
        "",
        "| Failure Mode | Probe | Role | Coverage |",
        "|:--|:--|:--|:--|",
    ]
    for row in report["taxonomy_rows"]:
        lines.append(
            f"| {row['failure_mode']} | {row['local_probe_ids']} | {row['probe_role']} | {row['coverage_state']} |"
        )
    lines.extend(["", "## tau_AF Sensitivity", "", "| Multiplier | tau_AF | Accept Rate | Static Interception |", "|--:|--:|--:|--:|"])
    for row in report["sensitivity_rows"]:
        lines.append(
            f"| {row['tau_multiplier']} | {row['tau_af']} | {row['accept_rate']} | {row['static_degradation_interception_rate']} |"
        )
    if report["blockers"]:
        lines.extend(["", "## Blockers", ""])
        for blocker in report["blockers"]:
            lines.append(f"- `{json.dumps(blocker, sort_keys=True, ensure_ascii=False)}`")
    lines.extend(["", "## Limitations", ""])
    for item in report["limitations"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def _csv_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    if not rows:
        return b"\n"
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(str(key))
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: "" if row.get(key) is None else row.get(key, "") for key in columns})
    return output.getvalue().encode("utf-8")


def _latex_table(rows: Sequence[Mapping[str, object]], caption: str) -> str:
    if not rows:
        return "% empty table\n"
    columns = [str(key) for key in rows[0].keys()]
    lines = [
        "\\begin{tabular}{" + "l" * len(columns) + "}",
        "\\hline",
        " & ".join(_latex_escape(column) for column in columns) + " \\\\",
        "\\hline",
    ]
    for row in rows:
        lines.append(" & ".join(_latex_escape(str(row.get(column, ""))) for column in columns) + " \\\\")
    lines.extend(["\\hline", "\\end{tabular}", f"% {caption}", ""])
    return "\n".join(lines)


def _latex_escape(value: str) -> str:
    return (
        value.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("_", "\\_")
        .replace("{", "\\{")
        .replace("}", "\\}")
    )


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise DefensiveOutputsError("DEFENSIVE_OUTPUT_EXISTS")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="export T5.5 defensive tables")
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--goal-spec", type=Path, required=True)
    run.add_argument("--probe-registry", type=Path, required=True)
    run.add_argument("--archive-db", type=Path)
    run.add_argument("--cas-root", type=Path)
    run.add_argument("--base-tau-af", type=float)
    args = parser.parse_args(argv)
    if args.command == "run":
        manifest = run_defensive_outputs(
            output_root=args.output_root,
            goal_spec=args.goal_spec,
            probe_registry=args.probe_registry,
            archive_db=args.archive_db,
            cas_root=args.cas_root,
            base_tau_af=args.base_tau_af,
        )
        print(json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
        return 0
    raise DefensiveOutputsError("DEFENSIVE_COMMAND_INVALID")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
