#!/usr/bin/env python3
"""Export paper-facing ACWM primitive, probe, affinity, and IRG maps."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
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
OUTCOME_NAMES = ("psnr", "ssim", "negative_mse", "negative_masked_mse")
REPO_ROOT = Path(__file__).resolve().parents[2]


class MethodEvidenceMapError(ValueError):
    """The frozen method-evidence inputs are malformed or inconsistent."""


def export_method_evidence_maps(
    *,
    primitive_matrix_root: Path,
    projection_stack_root: Path,
    affinity_path: Path,
    output_root: Path,
) -> dict[str, object]:
    primitive_root = Path(primitive_matrix_root).resolve(strict=True)
    stack_root = Path(projection_stack_root).resolve(strict=True)
    affinity_source = Path(affinity_path).resolve(strict=True)
    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise MethodEvidenceMapError("METHOD_EVIDENCE_OUTPUT_EXISTS")

    primitive_report = _load_json(primitive_root / "paper-primitive-matrix.json")
    primitive_cells = _read_csv(primitive_root / "tables" / "all_environment_primitive_cells.csv")
    stack = _load_json(stack_root / "projection-stack.json")
    affinity = _load_json(affinity_source)
    probe_rows = _probe_response_rows(stack)
    affinity_rows, affinity_matrix = _affinity_rows(
        affinity=affinity,
        primitives=tuple(str(value) for value in primitive_report["primitive_order"]),
        active_probes=tuple(dict.fromkeys(str(row["probe"]) for row in probe_rows)),
    )
    irg_rows = _irg_rows(probe_rows)

    tables = destination / "tables"
    figures = destination / "figures"
    inputs = destination / "inputs"
    tables.mkdir(parents=True)
    figures.mkdir()
    inputs.mkdir()

    portable_primitive_cells = _portableize(primitive_cells)
    _write_csv(tables / "environment_primitive_effects.csv", portable_primitive_cells)
    primitive_matrix_rows = _read_csv(
        primitive_root / "tables" / "environment_primitive_matrix.csv"
    )
    _write_csv(
        tables / "environment_primitive_matrix.csv",
        primitive_matrix_rows,
    )
    shutil.copyfile(
        primitive_root / "tables" / "positive_environment_primitive_cells.tex",
        tables / "positive_environment_primitive_cells.tex",
    )
    _write_csv(tables / "environment_probe_jacobian.csv", probe_rows)
    _write_csv(tables / "primitive_probe_affinity.csv", affinity_rows)
    _write_csv(tables / "primitive_probe_affinity_matrix.csv", affinity_matrix)
    _write_csv(tables / "irg_projection.csv", irg_rows)
    (tables / "environment_probe_jacobian.md").write_text(
        _probe_markdown(probe_rows), encoding="utf-8"
    )
    (tables / "environment_probe_jacobian.tex").write_text(
        _probe_latex(probe_rows), encoding="utf-8"
    )
    (tables / "primitive_probe_affinity.tex").write_text(
        _affinity_latex(affinity_rows), encoding="utf-8"
    )

    shutil.copyfile(
        primitive_root / "figures" / "environment_primitive_gate_heatmap.svg",
        figures / "environment_primitive_gate_heatmap.svg",
    )
    _render_figures(
        figures=figures,
        primitive_cells=primitive_cells,
        probe_rows=probe_rows,
        affinity_rows=affinity_rows,
        irg_rows=irg_rows,
    )

    (inputs / "paper-primitive-matrix.json").write_text(
        _pretty_json(_portableize(primitive_report)), encoding="utf-8"
    )
    (inputs / "projection-stack.json").write_text(
        _pretty_json(_portableize(stack)), encoding="utf-8"
    )
    (inputs / "primitive-probe-affinity.json").write_text(
        _pretty_json(_portableize(affinity)), encoding="utf-8"
    )

    report = {
        "schema_version": 1,
        "artifact_type": "verdiwm-acwm-method-evidence-maps",
        "state": "ready",
        "active_selector": "r20-active-action-temporal-alignment",
        "counts": {
            "environment_count": len(ENVIRONMENT_ORDER),
            "primitive_count": len(primitive_report["primitive_order"]),
            "attempted_primitive_cell_count": int(primitive_report["counts"]["attempted_cell_count"]),
            "positive_primitive_cell_count": int(primitive_report["counts"]["positive_cell_count"]),
            "active_probe_count": len({str(row["probe"]) for row in probe_rows}),
            "probe_measurement_count": len(probe_rows),
            "local_probe_measurement_count": sum(bool(row["locality_pass"]) for row in probe_rows),
            "affinity_relation_count": len(affinity_rows),
        },
        "visual_encodings": {
            "environment_primitive_gate_heatmap": "strict four-metric official-gate state; text is selected checkpoint delta PSNR",
            "environment_probe_psnr_response_heatmap": "signed dPSNR/ddose; crossed cells fail the frozen locality threshold",
            "primitive_probe_affinity_graph": "solid edges are active required paths; dashed edges are missing required paths; dotted edges are staged successor axes",
            "irg_pca": "PCA of per-feature standardized active-r20 response coordinates with unsupported paths masked; color is best current gate-positive primitive",
        },
        "claim_boundary": (
            "The primitive map summarizes settled checkpoint-level official-gate evidence. Probe maps summarize "
            "pilot paired-dose measurements and selector routing coordinates, not model improvements. The PCA is "
            "descriptive and does not establish cross-backbone alignment or causal transfer."
        ),
        "source_artifacts": {
            "primitive_matrix": _portable_path(primitive_root),
            "projection_stack": _portable_path(stack_root),
            "affinity": _portable_path(affinity_source),
        },
    }
    (destination / "bundle.json").write_text(_pretty_json(report), encoding="utf-8")
    (destination / "README.md").write_text(_readme(report), encoding="utf-8")
    _write_manifest(destination)
    return report


def _probe_response_rows(stack: Mapping[str, Any]) -> list[dict[str, object]]:
    sources = stack.get("probe_sources")
    if not isinstance(sources, list) or not sources:
        raise MethodEvidenceMapError("METHOD_EVIDENCE_PROBE_SOURCES_INVALID")
    indexed: dict[tuple[str, str], dict[str, object]] = {}
    for source in sources:
        if not isinstance(source, Mapping):
            raise MethodEvidenceMapError("METHOD_EVIDENCE_PROBE_SOURCE_INVALID")
        projection = Path(str(source.get("projection_path") or "")).resolve(strict=True)
        chart_root = projection.parent / "charts"
        path_order = source.get("path_order")
        if not isinstance(path_order, list) or not path_order:
            raise MethodEvidenceMapError("METHOD_EVIDENCE_PATH_ORDER_INVALID")
        for environment in ENVIRONMENT_ORDER:
            chart = _load_json(chart_root / f"{environment}.json")
            interventions = chart.get("intervention_names")
            jacobian = chart.get("jacobian")
            residuals = chart.get("locality_residuals")
            outcomes = chart.get("outcome_names")
            if outcomes != list(OUTCOME_NAMES) or not isinstance(interventions, list):
                raise MethodEvidenceMapError("METHOD_EVIDENCE_CHART_CONTRACT_INVALID")
            if not isinstance(jacobian, list) or len(jacobian) != len(OUTCOME_NAMES) or not isinstance(residuals, Mapping):
                raise MethodEvidenceMapError("METHOD_EVIDENCE_CHART_VALUES_INVALID")
            for column, probe in enumerate(interventions):
                if probe not in path_order:
                    raise MethodEvidenceMapError("METHOD_EVIDENCE_CHART_PATH_UNDECLARED")
                values = [float(jacobian[row][column]) for row in range(len(OUTCOME_NAMES))]
                residual = float(residuals[probe])
                record = {
                    "environment": environment,
                    "probe": str(probe),
                    "d_psnr_d_dose": values[0],
                    "d_ssim_d_dose": values[1],
                    "d_negative_mse_d_dose": values[2],
                    "d_negative_masked_mse_d_dose": values[3],
                    "response_l2": math.sqrt(sum(value * value for value in values)),
                    "locality_residual": residual,
                    "locality_threshold": 0.5,
                    "locality_pass": residual <= 0.5,
                    "repeat_count": int(chart["repeat_count"]),
                    "chart_id": str(chart["chart_id"]),
                    "source_chart": _portable_path(chart_root / f"{environment}.json"),
                }
                key = (environment, str(probe))
                if key in indexed:
                    raise MethodEvidenceMapError("METHOD_EVIDENCE_PROBE_DUPLICATED")
                indexed[key] = record
    expected = {(environment, probe) for environment in ENVIRONMENT_ORDER for probe in _active_probe_order(stack)}
    if set(indexed) != expected:
        raise MethodEvidenceMapError("METHOD_EVIDENCE_PROBE_COVERAGE_INCOMPLETE")
    return [indexed[(environment, probe)] for environment in ENVIRONMENT_ORDER for probe in _active_probe_order(stack)]


def _active_probe_order(stack: Mapping[str, Any]) -> tuple[str, ...]:
    rows = stack.get("probe_sources")
    return tuple(
        str(probe)
        for source in rows
        if isinstance(source, Mapping)
        for probe in source.get("path_order", [])
    )


def _affinity_rows(
    *, affinity: Mapping[str, Any], primitives: Sequence[str], active_probes: Sequence[str]
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    mappings = affinity.get("primitives")
    if not isinstance(mappings, Mapping):
        raise MethodEvidenceMapError("METHOD_EVIDENCE_AFFINITY_INVALID")
    relations: list[dict[str, object]] = []
    probe_order = list(active_probes)
    for primitive in primitives:
        spec = mappings.get(primitive)
        if not isinstance(spec, Mapping):
            relations.append(
                {
                    "primitive": primitive,
                    "probe": "unmapped",
                    "relation": "unmapped",
                    "coverage_state": "unmapped",
                    "mechanism_rationale": "No active-r20 primitive-probe affinity is declared.",
                }
            )
            continue
        required = spec.get("required_probe_paths")
        if not isinstance(required, list):
            raise MethodEvidenceMapError("METHOD_EVIDENCE_REQUIRED_PROBES_INVALID")
        for probe in required:
            name = str(probe)
            relation = "required_active" if name in active_probes else "required_missing"
            if name not in probe_order:
                probe_order.append(name)
            relations.append(
                {
                    "primitive": primitive,
                    "probe": name,
                    "relation": relation,
                    "coverage_state": str(spec.get("coverage_state") or "unknown"),
                    "mechanism_rationale": str(spec.get("mechanism_rationale") or ""),
                }
            )
        successor = spec.get("successor_probe_axis")
        if isinstance(successor, str) and successor and successor not in required:
            if successor not in probe_order:
                probe_order.append(successor)
            relations.append(
                {
                    "primitive": primitive,
                    "probe": successor,
                    "relation": "candidate_successor",
                    "coverage_state": str(spec.get("coverage_state") or "unknown"),
                    "mechanism_rationale": str(spec.get("mechanism_rationale") or ""),
                }
            )
    lookup = {(str(row["primitive"]), str(row["probe"])): str(row["relation"]) for row in relations}
    matrix = []
    for primitive in primitives:
        row: dict[str, object] = {"primitive": primitive}
        for probe in probe_order:
            row[probe] = lookup.get((primitive, probe), "none")
        matrix.append(row)
    return relations, matrix


def _irg_rows(probe_rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - exercised in the paper environment
        raise MethodEvidenceMapError("METHOD_EVIDENCE_NUMPY_REQUIRED") from exc
    probes = tuple(dict.fromkeys(str(row["probe"]) for row in probe_rows))
    indexed = {(str(row["environment"]), str(row["probe"])): row for row in probe_rows}
    vectors = []
    for environment in ENVIRONMENT_ORDER:
        vector = []
        for probe in probes:
            row = indexed[(environment, probe)]
            supported = bool(row["locality_pass"])
            for key in (
                "d_psnr_d_dose",
                "d_ssim_d_dose",
                "d_negative_mse_d_dose",
                "d_negative_masked_mse_d_dose",
            ):
                vector.append(float(row[key]) if supported else 0.0)
        vectors.append(vector)
    matrix = np.asarray(vectors, dtype=float)
    means = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    usable = std > 1e-12
    standardized = (matrix[:, usable] - means[usable]) / std[usable]
    if standardized.shape[1] < 2:
        raise MethodEvidenceMapError("METHOD_EVIDENCE_PCA_RANK_INVALID")
    u, singular, _vh = np.linalg.svd(standardized, full_matrices=False)
    coordinates = u[:, :2] * singular[:2]
    variance = np.square(singular)
    ratio = variance / variance.sum()
    rows = []
    for index, environment in enumerate(ENVIRONMENT_ORDER):
        rows.append(
            {
                "environment": environment,
                "pc1": float(coordinates[index, 0]),
                "pc2": float(coordinates[index, 1]),
                "pc1_explained_variance_ratio": float(ratio[0]),
                "pc2_explained_variance_ratio": float(ratio[1]),
                "active_probe_count": len(probes),
                "local_probe_count": sum(
                    bool(indexed[(environment, probe)]["locality_pass"]) for probe in probes
                ),
            }
        )
    return rows


def _render_figures(
    *,
    figures: Path,
    primitive_cells: Sequence[Mapping[str, str]],
    probe_rows: Sequence[Mapping[str, object]],
    affinity_rows: Sequence[Mapping[str, object]],
    irg_rows: Sequence[Mapping[str, object]],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
    except ImportError as exc:  # pragma: no cover - exercised in the paper environment
        raise MethodEvidenceMapError("METHOD_EVIDENCE_PLOT_DEPENDENCIES_REQUIRED") from exc

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "svg.fonttype": "none",
        }
    )
    primitive_names = tuple(dict.fromkeys(str(row["primitive"]) for row in primitive_cells))
    cell_index = {(str(row["environment"]), str(row["primitive"])): row for row in primitive_cells}
    state_value = {"untested": 0, "excluded": 1, "fail": 2, "pass": 3}
    primitive_values = np.asarray(
        [
            [
                state_value["untested" if (environment, primitive) not in cell_index else str(cell_index[(environment, primitive)]["verdict"])]
                for primitive in primitive_names
            ]
            for environment in ENVIRONMENT_ORDER
        ]
    )
    fig, ax = plt.subplots(figsize=(12.4, 5.6))
    fig.subplots_adjust(left=0.12, right=0.98, top=0.9, bottom=0.34)
    colors = ["#EEEEEE", "#F4E2AD", "#F3D0CC", "#CDEBDC"]
    ax.imshow(
        primitive_values,
        cmap=ListedColormap(colors),
        norm=BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], len(colors)),
        aspect="auto",
    )
    ax.set_xticks(range(len(primitive_names)), [_display(value) for value in primitive_names], rotation=35, ha="right")
    ax.set_yticks(range(len(ENVIRONMENT_ORDER)), [_display(value) for value in ENVIRONMENT_ORDER])
    ax.set_title("ACWM-Phys environment-by-primitive evidence")
    for y, environment in enumerate(ENVIRONMENT_ORDER):
        for x, primitive in enumerate(primitive_names):
            cell = cell_index.get((environment, primitive))
            if cell is None:
                label = "--"
            elif cell["verdict"] == "excluded":
                label = "EXC"
            else:
                label = f"{float(cell['delta_psnr']):+.2f}"
            ax.text(x, y, label, ha="center", va="center", fontsize=7.5, fontweight="bold")
    legend = [
        Patch(facecolor=color, edgecolor="#CCCCCC", label=label)
        for color, label in zip(colors, ("Untested", "Excluded", "Fail", "Pass"), strict=True)
    ]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, 0.025), ncol=4, frameon=False)
    fig.text(0.5, 0.09, "Cell text is selected-checkpoint delta PSNR; color encodes the strict four-metric official gate.", ha="center", color="#555555")
    _save_figure(fig, figures / "environment_primitive_gate_heatmap")
    plt.close(fig)

    probes = tuple(dict.fromkeys(str(row["probe"]) for row in probe_rows))
    response = {(str(row["environment"]), str(row["probe"])): row for row in probe_rows}
    values = np.asarray(
        [[float(response[(environment, probe)]["d_psnr_d_dose"]) for probe in probes] for environment in ENVIRONMENT_ORDER]
    )
    bound = float(np.quantile(np.abs(values), 0.9)) or 1.0
    fig, ax = plt.subplots(figsize=(12.4, 5.6))
    fig.subplots_adjust(left=0.12, right=0.98, top=0.9, bottom=0.38)
    ax.imshow(values, cmap="RdBu", norm=TwoSlopeNorm(vcenter=0.0, vmin=-bound, vmax=bound), aspect="auto")
    ax.set_xticks(range(len(probes)), [_display(value) for value in probes], rotation=38, ha="right")
    ax.set_yticks(range(len(ENVIRONMENT_ORDER)), [_display(value) for value in ENVIRONMENT_ORDER])
    ax.set_title("Environment-by-probe local response")
    ax.set_xlabel("Active IRG probe direction")
    for y, environment in enumerate(ENVIRONMENT_ORDER):
        for x, probe in enumerate(probes):
            row = response[(environment, probe)]
            if not bool(row["locality_pass"]):
                ax.text(x, y, "x", ha="center", va="center", color="#111111", fontsize=10, fontweight="bold")
    fig.text(
        0.98,
        0.04,
        f"Blue: +dPSNR/ddose   White: 0   Red: -dPSNR/ddose   symmetric clip: +/-{bound:.2f}",
        ha="right",
        color="#555555",
    )
    fig.text(0.12, 0.04, "Crossed cells fail locality residual <= 0.5.", color="#555555")
    _save_figure(fig, figures / "environment_probe_psnr_response_heatmap")
    plt.close(fig)

    primitives = tuple(dict.fromkeys(str(row["primitive"]) for row in affinity_rows))
    affinity_probes = tuple(dict.fromkeys(str(row["probe"]) for row in affinity_rows if row["probe"] != "unmapped"))
    fig_height = max(5.6, 0.42 * max(len(primitives), len(affinity_probes)))
    fig, ax = plt.subplots(figsize=(12.2, fig_height), constrained_layout=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.7, max(len(primitives), len(affinity_probes)) - 0.3)
    ax.axis("off")
    primitive_y = {name: len(primitives) - 1 - index for index, name in enumerate(primitives)}
    probe_y = {name: len(affinity_probes) - 1 - index for index, name in enumerate(affinity_probes)}
    styles = {
        "required_active": ("#0072B2", "-", 2.0),
        "required_missing": ("#D55E00", "--", 2.0),
        "candidate_successor": ("#777777", ":", 1.5),
    }
    for row in affinity_rows:
        relation = str(row["relation"])
        probe = str(row["probe"])
        primitive = str(row["primitive"])
        if relation not in styles or probe not in probe_y:
            continue
        color, linestyle, width = styles[relation]
        ax.plot([0.33, 0.67], [primitive_y[primitive], probe_y[probe]], color=color, linestyle=linestyle, linewidth=width, alpha=0.8, zorder=1)
    for name, y in primitive_y.items():
        ax.text(0.31, y, _display(name), ha="right", va="center", bbox={"boxstyle": "round,pad=0.28", "fc": "#E8F1F8", "ec": "#0072B2", "lw": 0.8}, zorder=2)
    for name, y in probe_y.items():
        ax.text(0.69, y, _display(name), ha="left", va="center", bbox={"boxstyle": "round,pad=0.28", "fc": "#F4F4F4", "ec": "#666666", "lw": 0.8}, zorder=2)
    ax.text(0.31, max(primitive_y.values()) + 0.55, "Repair primitives", ha="right", fontweight="bold")
    ax.text(0.69, max(probe_y.values()) + 0.55, "Diagnostic intervention axes", ha="left", fontweight="bold")
    legend = [Line2D([0], [0], color=color, linestyle=style, linewidth=width, label=_display(name)) for name, (color, style, width) in styles.items()]
    ax.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False)
    ax.set_title("Mechanism contract between repair primitives and probes", pad=18)
    _save_figure(fig, figures / "primitive_probe_affinity_graph")
    plt.close(fig)

    best = _best_positive_primitive(primitive_cells)
    palette = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442"]
    primitive_names = tuple(dict.fromkeys(value for value in best.values() if value is not None))
    colors = {name: palette[index % len(palette)] for index, name in enumerate(primitive_names)}
    fig, ax = plt.subplots(figsize=(7.5, 5.5), constrained_layout=True)
    annotation_offsets = {
        "push_cube": (8, 8),
        "push_sand": (-48, 8),
        "stack_cube": (5, 5),
        "cloth_move": (5, 5),
    }
    for row in irg_rows:
        environment = str(row["environment"])
        primitive = best.get(environment)
        color = colors.get(primitive, "#9A9A9A")
        ax.scatter(float(row["pc1"]), float(row["pc2"]), s=78, color=color, edgecolor="white", linewidth=0.8, zorder=3)
        ax.annotate(
            _display(environment),
            (float(row["pc1"]), float(row["pc2"])),
            xytext=annotation_offsets.get(environment, (5, 5)),
            textcoords="offset points",
            fontsize=8,
        )
    pc1 = 100.0 * float(irg_rows[0]["pc1_explained_variance_ratio"])
    pc2 = 100.0 * float(irg_rows[0]["pc2_explained_variance_ratio"])
    ax.axhline(0, color="#D0D0D0", linewidth=0.7)
    ax.axvline(0, color="#D0D0D0", linewidth=0.7)
    ax.set_xlabel(f"IRG PC1 ({pc1:.1f}% variance)")
    ax.set_ylabel(f"IRG PC2 ({pc2:.1f}% variance)")
    ax.set_title("ACWM-Phys environments in the active-r20 IRG")
    handles = [Patch(facecolor=colors[name], label=_display(name)) for name in primitive_names]
    handles.append(Patch(facecolor="#9A9A9A", label="No gate-positive primitive"))
    ax.legend(handles=handles, loc="best", frameon=False, fontsize=8)
    ax.grid(color="#ECECEC", linewidth=0.6)
    _save_figure(fig, figures / "irg_environment_pca")
    plt.close(fig)


def _best_positive_primitive(cells: Sequence[Mapping[str, str]]) -> dict[str, str | None]:
    best: dict[str, tuple[float, str]] = {}
    for row in cells:
        if row.get("verdict") != "pass":
            continue
        environment = str(row["environment"])
        candidate = (float(row["delta_psnr"]), str(row["primitive"]))
        if environment not in best or candidate[0] > best[environment][0]:
            best[environment] = candidate
    return {environment: best.get(environment, (0.0, None))[1] for environment in ENVIRONMENT_ORDER}


def _save_figure(fig: Any, stem: Path) -> None:
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(stem.with_suffix(f".{suffix}"), dpi=240 if suffix == "png" else None, bbox_inches="tight")
    svg = stem.with_suffix(".svg")
    normalized = "\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n"
    svg.write_text(normalized, encoding="utf-8")


def _probe_markdown(rows: Sequence[Mapping[str, object]]) -> str:
    lines = [
        "| Environment | Probe | dPSNR/ddose | dSSIM/ddose | d(-MSE)/ddose | d(-mMSE)/ddose | Locality | Local? |",
        "|---|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        lines.append(
            "| {environment} | {probe} | {d_psnr_d_dose:+.4f} | {d_ssim_d_dose:+.5f} | "
            "{d_negative_mse_d_dose:+.6f} | {d_negative_masked_mse_d_dose:+.6f} | "
            "{locality_residual:.3f} | {locality_pass} |".format(**row)
        )
    return "\n".join(lines) + "\n"


def _probe_latex(rows: Sequence[Mapping[str, object]]) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering\scriptsize",
        r"\caption{Active-r20 paired-dose response Jacobians. A dagger marks paths failing the frozen locality threshold and therefore excluded from IRG routing.}",
        r"\label{tab:acwm_probe_jacobian}",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Environment & Probe & $\partial$PSNR & $\partial$SSIM & $\partial(-\mathrm{MSE})$ & $\partial(-\mathrm{mMSE})$ & Locality \\",
        r"\midrule",
    ]
    for row in rows:
        mark = "" if bool(row["locality_pass"]) else r"$^\dagger$"
        lines.append(
            "{} & {}{} & {:+.3f} & {:+.4f} & {:+.5f} & {:+.5f} & {:.3f} \\\\".format(
                _latex_escape(row["environment"]),
                _latex_escape(row["probe"]),
                mark,
                float(row["d_psnr_d_dose"]),
                float(row["d_ssim_d_dose"]),
                float(row["d_negative_mse_d_dose"]),
                float(row["d_negative_masked_mse_d_dose"]),
                float(row["locality_residual"]),
            )
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    return "\n".join(lines)


def _affinity_latex(rows: Sequence[Mapping[str, object]]) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering\small",
        r"\caption{Mechanism contract between repair primitives and diagnostic intervention axes.}",
        r"\label{tab:primitive_probe_affinity}",
        r"\begin{tabular}{lll}",
        r"\toprule",
        r"Primitive & Probe axis & Relation \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            "{} & {} & {} \\\\".format(
                _latex_escape(row["primitive"]),
                _latex_escape(row["probe"]),
                _latex_escape(row["relation"]),
            )
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def _readme(report: Mapping[str, Any]) -> str:
    counts = report["counts"]
    return f"""# ACWM-Phys Method Evidence Maps

This bundle links frozen primitive effects, paired-dose probe responses, the
primitive-to-probe mechanism contract, and the active-r20 IRG projection.

## Contents

- `figures/environment_primitive_gate_heatmap.svg`: strict official-gate evidence.
- `figures/environment_probe_psnr_response_heatmap.*`: signed local probe response; crossed cells are nonlocal.
- `figures/primitive_probe_affinity_graph.*`: required, missing, and successor probe relations.
- `figures/irg_environment_pca.*`: descriptive active-r20 response-coordinate projection.
- `tables/`: CSV, Markdown, and LaTeX source tables.

## Snapshot

- Environments: {counts['environment_count']}
- Primitives: {counts['primitive_count']}
- Attempted primitive cells: {counts['attempted_primitive_cell_count']}
- Gate-positive primitive cells: {counts['positive_primitive_cell_count']}
- Active IRG directions: {counts['active_probe_count']}
- Paired-dose environment-probe measurements: {counts['probe_measurement_count']}
- Local environment-probe measurements: {counts['local_probe_measurement_count']}

## Claim Boundary

{report['claim_boundary']}
"""


def _write_manifest(root: Path) -> None:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item.name != "MANIFEST.sha256"):
        rows.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root).as_posix()}")
    (root / "MANIFEST.sha256").write_text("\n".join(rows) + "\n", encoding="utf-8")


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise MethodEvidenceMapError(f"METHOD_EVIDENCE_CSV_INVALID:{path}") from exc
    if not rows:
        raise MethodEvidenceMapError(f"METHOD_EVIDENCE_CSV_EMPTY:{path}")
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise MethodEvidenceMapError("METHOD_EVIDENCE_ROWS_EMPTY")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MethodEvidenceMapError(f"METHOD_EVIDENCE_JSON_INVALID:{path}") from exc
    if not isinstance(payload, Mapping):
        raise MethodEvidenceMapError(f"METHOD_EVIDENCE_JSON_INVALID:{path}")
    return payload


def _pretty_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"


def _portableize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _portableize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_portableize(item) for item in value]
    if isinstance(value, str) and value.startswith(str(REPO_ROOT) + "/"):
        return _portable_path(Path(value))
    return value


def _portable_path(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _display(value: str) -> str:
    return value.replace("_", " ")


def _latex_escape(value: object) -> str:
    return str(value).replace("_", r"\_")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primitive-matrix-root", type=Path, required=True)
    parser.add_argument("--projection-stack-root", type=Path, required=True)
    parser.add_argument("--affinity", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    report = export_method_evidence_maps(
        primitive_matrix_root=args.primitive_matrix_root,
        projection_stack_root=args.projection_stack_root,
        affinity_path=args.affinity,
        output_root=args.output_root,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
