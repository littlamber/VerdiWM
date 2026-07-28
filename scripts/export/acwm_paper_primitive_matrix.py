#!/usr/bin/env python3
"""Export paper-facing ACWM environment-by-primitive evidence tables."""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


ENVIRONMENT_ORDER = (
    "push_cube",
    "stack_cube",
    "push_rope",
    "cloth_move",
    "push_sand",
    "pour_water",
    "robot_arm",
    "reacher",
)
METRICS = ("psnr", "ssim", "mse", "masked_mse")
REPO_ROOT = Path(__file__).resolve().parents[2]


class PaperPrimitiveMatrixError(ValueError):
    """The settled effect-label index cannot support a paper table."""


def export_paper_primitive_matrix(*, effect_label_index: Path, output_root: Path) -> dict[str, object]:
    source = Path(effect_label_index).resolve()
    payload = _load_json(source)
    labels = payload.get("labels")
    if not isinstance(labels, list) or not labels:
        raise PaperPrimitiveMatrixError("PAPER_PRIMITIVE_MATRIX_LABELS_INVALID")

    receipt_rows = [_receipt_row(label) for label in labels if isinstance(label, Mapping)]
    if len(receipt_rows) != len(labels):
        raise PaperPrimitiveMatrixError("PAPER_PRIMITIVE_MATRIX_LABEL_INVALID")

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in receipt_rows:
        grouped[(row["environment"], row["primitive"])].append(row)
    cell_rows = [_cell_row(environment, primitive, rows) for (environment, primitive), rows in grouped.items()]
    cell_rows.sort(key=lambda row: (_environment_rank(row["environment"]), row["primitive"]))
    positive_rows = [row for row in cell_rows if row["verdict"] == "pass"]

    primitives = sorted({row["primitive"] for row in cell_rows})
    destination = Path(output_root).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    _write_csv(destination / "tables" / "all_gate_receipts.csv", receipt_rows)
    _write_csv(destination / "tables" / "all_environment_primitive_cells.csv", cell_rows)
    _write_csv(destination / "tables" / "positive_environment_primitive_cells.csv", positive_rows)
    _write_csv(destination / "tables" / "environment_primitive_matrix.csv", _matrix_rows(cell_rows, primitives))
    (destination / "tables" / "positive_environment_primitive_cells.md").write_text(
        _markdown_table(positive_rows), encoding="utf-8"
    )
    (destination / "tables" / "positive_environment_primitive_cells.tex").write_text(
        _latex_table(positive_rows), encoding="utf-8"
    )
    (destination / "figures").mkdir(parents=True, exist_ok=True)
    (destination / "figures" / "environment_primitive_gate_heatmap.svg").write_text(
        _heatmap_svg(cell_rows, primitives), encoding="utf-8"
    )

    counts = {
        "receipt_count": len(receipt_rows),
        "attempted_cell_count": len(cell_rows),
        "positive_cell_count": len(positive_rows),
        "failed_cell_count": sum(row["verdict"] == "fail" for row in cell_rows),
        "excluded_cell_count": sum(row["verdict"] == "excluded" for row in cell_rows),
        "positive_environment_count": len({row["environment"] for row in positive_rows}),
        "environment_count": len(ENVIRONMENT_ORDER),
        "primitive_count": len(primitives),
    }
    report = {
        "schema_version": 1,
        "artifact_type": "verdiwm-acwm-paper-primitive-matrix",
        "state": "ready",
        "source_effect_label_index": _portable_path(source),
        "counts": counts,
        "environment_order": list(ENVIRONMENT_ORDER),
        "primitive_order": primitives,
        "selection_rule": (
            "A cell passes when at least one settled selector-admissible official 50-step gate passes all four "
            "metric checks. The displayed representative is the passing checkpoint with largest PSNR delta; "
            "failed cells retain the attempt satisfying the most gate checks, then largest PSNR delta."
        ),
        "claim_boundary": (
            "This summarizes current checkpoint-level frozen-gate evidence. It does not imply multi-seed "
            "statistical significance, complete 8x8 coverage, or cross-backbone transfer."
        ),
        "files": [
            "tables/all_gate_receipts.csv",
            "tables/all_environment_primitive_cells.csv",
            "tables/positive_environment_primitive_cells.csv",
            "tables/positive_environment_primitive_cells.md",
            "tables/positive_environment_primitive_cells.tex",
            "tables/environment_primitive_matrix.csv",
            "figures/environment_primitive_gate_heatmap.svg",
        ],
    }
    (destination / "paper-primitive-matrix.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (destination / "README.md").write_text(_readme(counts), encoding="utf-8")
    return report


def _receipt_row(label: Mapping[str, Any]) -> dict[str, Any]:
    delta = label.get("delta_candidate_minus_baseline")
    checks = label.get("official_gate_checks")
    if not isinstance(delta, Mapping) or not all(isinstance(delta.get(metric), (int, float)) for metric in METRICS):
        raise PaperPrimitiveMatrixError("PAPER_PRIMITIVE_MATRIX_DELTA_INVALID")
    if not isinstance(checks, Mapping):
        checks = {}
    evidence_ref = str(label.get("evidence_ref") or "")
    evidence = _load_json(Path(evidence_ref)) if evidence_ref and Path(evidence_ref).is_file() else {}
    positive = label.get("positive")
    admissible = bool(label.get("settled")) and bool(label.get("selector_admissible")) and isinstance(positive, bool)
    return {
        "environment": str(label.get("environment") or ""),
        "primitive": str(label.get("primitive") or ""),
        "verdict": "pass" if admissible and positive else ("fail" if admissible else "excluded"),
        "checkpoint_step": _checkpoint_step(str(label.get("label_id") or ""), evidence),
        "eval_seed": label.get("seed"),
        "delta_psnr": float(delta["psnr"]),
        "delta_ssim": float(delta["ssim"]),
        "delta_mse": float(delta["mse"]),
        "delta_masked_mse": float(delta["masked_mse"]),
        "passed_check_count": sum(value is True for value in checks.values()),
        "candidate_checkpoint_sha256": evidence.get("candidate_checkpoint_sha256"),
        "label_id": str(label.get("label_id") or ""),
        "evidence_ref": _portable_path(Path(evidence_ref)) if evidence_ref else "",
        "exclusion_reason": label.get("selector_exclusion_reason"),
    }


def _cell_row(environment: str, primitive: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row["verdict"] != "excluded"]
    passing = [row for row in valid if row["verdict"] == "pass"]
    if passing:
        representative = max(passing, key=lambda row: row["delta_psnr"])
        verdict = "pass"
    elif valid:
        representative = max(valid, key=lambda row: (row["passed_check_count"], row["delta_psnr"]))
        verdict = "fail"
    else:
        representative = rows[0]
        verdict = "excluded"
    steps = sorted({row["checkpoint_step"] for row in rows if isinstance(row["checkpoint_step"], int)})
    seeds = sorted({row["eval_seed"] for row in valid if isinstance(row["eval_seed"], int)})
    return {
        "environment": environment,
        "primitive": primitive,
        "verdict": verdict,
        "stability": _stability(valid),
        "receipt_count": len(rows),
        "passing_receipt_count": len(passing),
        "distinct_seed_count": len(seeds),
        "evaluated_steps": "/".join(str(step) for step in steps) or "unknown",
        "selected_checkpoint_step": representative["checkpoint_step"],
        "selected_eval_seed": representative["eval_seed"],
        "delta_psnr": representative["delta_psnr"],
        "delta_ssim": representative["delta_ssim"],
        "delta_mse": representative["delta_mse"],
        "delta_masked_mse": representative["delta_masked_mse"],
        "candidate_checkpoint_sha256": representative["candidate_checkpoint_sha256"],
        "evidence_ref": representative["evidence_ref"],
    }


def _stability(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "excluded"
    verdicts = {row["verdict"] for row in rows}
    if verdicts == {"pass"}:
        return "all_observed_checkpoints_pass" if len(rows) > 1 else "single_checkpoint_pass"
    if "pass" in verdicts:
        return "checkpoint_sensitive"
    return "no_passing_checkpoint"


def _checkpoint_step(label_id: str, evidence: Mapping[str, Any]) -> int | None:
    values = (str(evidence.get("candidate_checkpoint") or ""), label_id)
    for value in values:
        for pattern in (r"relative_step_(\d+)", r"step(\d+)", r"-t(\d+)(?:-|/)"):
            match = re.search(pattern, value)
            if match:
                return int(match.group(1))
    return None


def _matrix_rows(cells: list[dict[str, Any]], primitives: list[str]) -> list[dict[str, Any]]:
    indexed = {(row["environment"], row["primitive"]): row for row in cells}
    rows = []
    for environment in ENVIRONMENT_ORDER:
        row: dict[str, Any] = {"environment": environment}
        for primitive in primitives:
            cell = indexed.get((environment, primitive))
            row[primitive] = "untested" if cell is None else f"{cell['verdict']}:{cell['delta_psnr']:+.3f}"
        rows.append(row)
    return rows


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    columns = ("environment", "primitive", "stability", "selected_checkpoint_step", "selected_eval_seed", "delta_psnr", "delta_ssim", "delta_mse", "delta_masked_mse")
    header = "| " + " | ".join(columns) + " |\n"
    rule = "| " + " | ".join("---" for _ in columns) + " |\n"
    body = "".join("| " + " | ".join(_format_value(row[column]) for column in columns) + " |\n" for row in rows)
    return header + rule + body


def _latex_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Current ACWM-Phys environment--primitive cells passing the frozen 50-step gate. Lower is better for MSE metrics.}",
        r"\label{tab:acwm_positive_cells}",
        r"\begin{tabular}{lllr rrrrr}",
        r"\toprule",
        r"Environment & Primitive & Stability & Step & Seed & $\Delta$PSNR$\uparrow$ & $\Delta$SSIM$\uparrow$ & $\Delta$MSE$\downarrow$ & $\Delta$mMSE$\downarrow$ \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            "{} & {} & {} & {} & {} & {:+.3f} & {:+.4f} & {:+.6f} & {:+.6f} \\\\".format(
                _latex_escape(row["environment"]),
                _latex_escape(row["primitive"]),
                _latex_escape(row["stability"]),
                row["selected_checkpoint_step"] or "--",
                row["selected_eval_seed"] or "--",
                row["delta_psnr"],
                row["delta_ssim"],
                row["delta_mse"],
                row["delta_masked_mse"],
            )
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    return "\n".join(lines)


def _heatmap_svg(cells: list[dict[str, Any]], primitives: list[str]) -> str:
    indexed = {(row["environment"], row["primitive"]): row for row in cells}
    left, top, cell_w, cell_h = 170, 210, 145, 58
    width = left + len(primitives) * cell_w + 40
    height = top + len(ENVIRONMENT_ORDER) * cell_h + 105
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#202124;letter-spacing:0}.title{font-size:24px;font-weight:700}.sub{font-size:13px;fill:#5f6368}.label{font-size:13px}.cell{font-size:12px;font-weight:700}.legend{font-size:12px}</style>',
        '<text class="title" x="24" y="36">ACWM-Phys primitive evidence matrix</text>',
        '<text class="sub" x="24" y="62">Frozen 50-step official gate; cell text is selected checkpoint delta PSNR (dB)</text>',
    ]
    for index, primitive in enumerate(primitives):
        x = left + index * cell_w + cell_w / 2
        parts.append(f'<text class="label" x="{x}" y="{top - 16}" text-anchor="start" transform="rotate(-48 {x} {top - 16})">{html.escape(primitive)}</text>')
    colors = {"pass": "#cdebdc", "fail": "#f3d0cc", "excluded": "#f4e2ad", "untested": "#eeeeee"}
    for row_index, environment in enumerate(ENVIRONMENT_ORDER):
        y = top + row_index * cell_h
        parts.append(f'<text class="label" x="{left - 12}" y="{y + 35}" text-anchor="end">{html.escape(environment)}</text>')
        for column_index, primitive in enumerate(primitives):
            x = left + column_index * cell_w
            cell = indexed.get((environment, primitive))
            verdict = "untested" if cell is None else str(cell["verdict"])
            text = "--" if cell is None else ("EXC" if verdict == "excluded" else f"{cell['delta_psnr']:+.2f}")
            parts.append(f'<rect x="{x + 2}" y="{y + 2}" width="{cell_w - 4}" height="{cell_h - 4}" rx="3" fill="{colors[verdict]}" stroke="#ffffff"/>')
            parts.append(f'<text class="cell" x="{x + cell_w / 2}" y="{y + 35}" text-anchor="middle">{html.escape(text)}</text>')
    legend_y = top + len(ENVIRONMENT_ORDER) * cell_h + 48
    for index, (label, color) in enumerate((("Pass", colors["pass"]), ("Fail", colors["fail"]), ("Excluded", colors["excluded"]), ("Untested", colors["untested"]))):
        x = 24 + index * 145
        parts.append(f'<rect x="{x}" y="{legend_y - 15}" width="18" height="18" rx="2" fill="{color}"/>')
        parts.append(f'<text class="legend" x="{x + 26}" y="{legend_y}">{label}</text>')
    parts.append('</svg>\n')
    return "".join(parts)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise PaperPrimitiveMatrixError("PAPER_PRIMITIVE_MATRIX_ROWS_EMPTY")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _readme(counts: Mapping[str, int]) -> str:
    return f"""# ACWM-Phys Primitive Evidence Matrix

This bundle contains paper-facing summaries derived only from the frozen settled effect-label index.

- Gate receipts: {counts['receipt_count']}
- Attempted environment-primitive cells: {counts['attempted_cell_count']}
- Gate-positive cells: {counts['positive_cell_count']}
- Gate-positive environments: {counts['positive_environment_count']}/{counts['environment_count']}
- Failed cells: {counts['failed_cell_count']}
- Excluded/non-comparable cells: {counts['excluded_cell_count']}

Use `tables/positive_environment_primitive_cells.tex` in the main paper. Use the complete cell and receipt CSVs in the appendix. The SVG heatmap encodes the strict four-metric verdict by color; the number in each evaluated cell is selected checkpoint delta PSNR, not a standalone pass criterion.
"""


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:+.6f}"
    return str(value)


def _latex_escape(value: Any) -> str:
    return str(value).replace("_", r"\_")


def _environment_rank(environment: str) -> int:
    try:
        return ENVIRONMENT_ORDER.index(environment)
    except ValueError:
        return len(ENVIRONMENT_ORDER)


def _portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PaperPrimitiveMatrixError(f"PAPER_PRIMITIVE_MATRIX_JSON_INVALID:{path}") from exc
    if not isinstance(payload, dict):
        raise PaperPrimitiveMatrixError(f"PAPER_PRIMITIVE_MATRIX_JSON_INVALID:{path}")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--effect-label-index", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    report = export_paper_primitive_matrix(
        effect_label_index=args.effect_label_index,
        output_root=args.output_root,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
