"""Run a preregistered random probe-subset expansion with CPU selector replay."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import fmean
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np

from wmloop.experiments._artifacts import canonical_json, write_bundle
from wmloop.experiments.selector_replay import run_selector_replay


class RandomProbeExpansionError(ValueError):
    """The preregistration or frozen replay inputs are invalid."""


_METRICS = (
    "top1_positive_hit",
    "benefit_sign_accuracy",
    "ranking_kendall_tau",
    "selection_regret",
    "negative_selection",
)


def run_random_probe_expansion(
    *,
    config_path: Path,
    output_root: Path,
    archive_db: Path | None = None,
    cas_root: Path | None = None,
) -> dict[str, object]:
    config_file = Path(config_path).resolve(strict=True)
    config = _load_mapping(config_file)
    if config.get("artifact_type") != "verdiwm-random-probe-expansion-preregistration":
        raise RandomProbeExpansionError("RANDOM_PROBE_CONFIG_TYPE_INVALID")
    if config.get("study_id") != "S4_probe_information_and_collision":
        raise RandomProbeExpansionError("RANDOM_PROBE_STUDY_ID_INVALID")
    if config.get("selector") != "irg":
        raise RandomProbeExpansionError("RANDOM_PROBE_SELECTOR_INVALID")
    if config.get("randomization_algorithm") != "sha256_seed_path_ascending_nested_prefix":
        raise RandomProbeExpansionError("RANDOM_PROBE_ALGORITHM_INVALID")

    repo_root = config_file.parents[2]
    projection_path = _resolve_source(config.get("source_projection_path"), repo_root)
    plan_path = _resolve_source(config.get("selector_plan_path"), repo_root)
    labels_path = _resolve_source(config.get("effect_label_index_path"), repo_root)
    projection_rows = _load_jsonl(projection_path)
    path_pool = _string_list(config.get("path_pool"), "RANDOM_PROBE_PATH_POOL_INVALID")
    if len(path_pool) < 2 or len(set(path_pool)) != len(path_pool):
        raise RandomProbeExpansionError("RANDOM_PROBE_PATH_POOL_INVALID")
    available_paths = _available_irg_paths(projection_rows)
    if set(path_pool) != available_paths:
        raise RandomProbeExpansionError("RANDOM_PROBE_PATH_POOL_MISMATCH")
    subset_sizes = _positive_int_list(config.get("subset_sizes"), "RANDOM_PROBE_SUBSET_SIZES_INVALID")
    if sorted(set(subset_sizes)) != subset_sizes or subset_sizes[-1] != len(path_pool):
        raise RandomProbeExpansionError("RANDOM_PROBE_SUBSET_SIZES_INVALID")
    randomization_seeds = _positive_int_list(
        config.get("randomization_seeds"), "RANDOM_PROBE_SEEDS_INVALID"
    )
    if len(set(randomization_seeds)) != len(randomization_seeds):
        raise RandomProbeExpansionError("RANDOM_PROBE_SEEDS_INVALID")

    execution_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    with TemporaryDirectory(prefix="verdiwm-random-probe-") as temporary:
        temporary_root = Path(temporary)
        for seed in randomization_seeds:
            path_order = deterministic_path_order(path_pool, seed)
            for subset_size in subset_sizes:
                selected_paths = path_order[:subset_size]
                filtered_rows = filter_projection_rows(projection_rows, selected_paths)
                projection_file = temporary_root / f"projection-s{seed}-k{subset_size}.jsonl"
                projection_file.write_bytes(b"".join(canonical_json(row) for row in filtered_rows))
                replay_root = temporary_root / f"replay-s{seed}-k{subset_size}"
                manifest = run_selector_replay(
                    plan_path=plan_path,
                    projection_path=projection_file,
                    effect_label_index=labels_path,
                    output_root=replay_root,
                )
                replay = _load_mapping(replay_root / "selector-replay.json")
                selector_metrics = _selector_metrics(replay, "irg")
                condition_number = _gram_condition_number(filtered_rows)
                execution = {
                    "execution_id": f"random_probe_subset_s{seed}_k{subset_size}",
                    "randomization_seed": seed,
                    "subset_size": subset_size,
                    "selected_paths": list(selected_paths),
                    "projection_sha256": _sha256(projection_file),
                    "state": manifest.get("state"),
                    "evaluated_cell_count": selector_metrics.get("evaluated_cell_count"),
                    "abstained_cell_count": selector_metrics.get("abstained_cell_count"),
                    "gram_condition_number": condition_number,
                    "metrics": {metric: selector_metrics.get(metric) for metric in _METRICS},
                }
                execution_rows.append(execution)
                metric_rows.append(
                    {
                        "selector": "irg_random_subset",
                        "randomization_seed": seed,
                        "subset_size": subset_size,
                        "selected_path_count": len(selected_paths),
                        "evaluated_cell_count": selector_metrics.get("evaluated_cell_count"),
                        "abstained_cell_count": selector_metrics.get("abstained_cell_count"),
                        "probe_gpu_hours": 0.0,
                        "gram_condition_number": condition_number,
                        **{metric: selector_metrics.get(metric) for metric in _METRICS},
                    }
                )

    expected_execution_count = len(randomization_seeds) * len(subset_sizes)
    if len(execution_rows) != expected_execution_count:
        raise RandomProbeExpansionError("RANDOM_PROBE_EXECUTION_COUNT_MISMATCH")
    report = {
        "schema_version": 1,
        "artifact_type": "verdiwm-random-probe-expansion",
        "experiment_id": config.get("experiment_id"),
        "study_id": config["study_id"],
        "state": "ready",
        "selector": "irg",
        "cpu_only": True,
        "randomization_algorithm": config["randomization_algorithm"],
        "nested_subsets": True,
        "path_pool": path_pool,
        "subset_sizes": subset_sizes,
        "randomization_seeds": randomization_seeds,
        "execution_count": len(execution_rows),
        "probe_gpu_hours": 0.0,
        "metric_summary_by_subset_size": _summarize_by_size(metric_rows),
        "executions": execution_rows,
        "source_refs": [
            _source_ref("preregistration", config_file),
            _source_ref("source_projection", projection_path),
            _source_ref("selector_plan", plan_path),
            _source_ref("effect_label_index", labels_path),
        ],
        "claim_boundary": (
            "Deterministic CPU replay of preregistered random IRG path subsets on the frozen "
            "ACWM-Phys leave-one-environment-out split. It is a random-expansion control and "
            "must not be represented as static_probe, new GPU evidence, or repair-quality evidence."
        ),
    }
    files = {
        "random-probe-expansion.json": canonical_json(report),
        "random-probe-expansion.md": _markdown(report).encode("utf-8"),
        "tables/selector-metrics.csv": _csv(metric_rows).encode("utf-8"),
        "tables/subsets.csv": _csv(execution_rows).encode("utf-8"),
        "input-preregistration.json": canonical_json(config),
    }
    destination = Path(output_root).resolve()
    return write_bundle(
        output_root=destination,
        files=files,
        manifest_fields={
            "artifact_type": "verdiwm-random-probe-expansion-manifest",
            "state": "ready",
            "study_id": config["study_id"],
            "execution_count": len(execution_rows),
            "randomization_seed_count": len(randomization_seeds),
            "subset_size_count": len(subset_sizes),
            "evaluated_cell_count": sum(int(row["evaluated_cell_count"] or 0) for row in metric_rows),
            "abstained_cell_count": sum(int(row["abstained_cell_count"] or 0) for row in metric_rows),
            "probe_gpu_hours": 0.0,
            "report_path": str(destination / "random-probe-expansion.json"),
        },
        archive_db=archive_db,
        cas_root=cas_root,
    )


def deterministic_path_order(path_pool: Sequence[str], seed: int) -> tuple[str, ...]:
    """Return a Python-version-independent order for nested random subsets."""

    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise RandomProbeExpansionError("RANDOM_PROBE_SEED_INVALID")
    paths = tuple(str(path) for path in path_pool)
    if not paths or len(set(paths)) != len(paths) or any(not path for path in paths):
        raise RandomProbeExpansionError("RANDOM_PROBE_PATH_POOL_INVALID")
    return tuple(
        sorted(
            paths,
            key=lambda path: (hashlib.sha256(f"{seed}:{path}".encode("utf-8")).hexdigest(), path),
        )
    )


def filter_projection_rows(
    rows: Sequence[Mapping[str, Any]], selected_paths: Sequence[str]
) -> list[dict[str, object]]:
    """Keep non-IRG controls unchanged and retain only selected IRG paths."""

    selected = set(str(path) for path in selected_paths)
    if not selected:
        raise RandomProbeExpansionError("RANDOM_PROBE_SUBSET_EMPTY")
    result: list[dict[str, object]] = []
    for raw in rows:
        row = dict(raw)
        if row.get("selector") != "irg":
            result.append(row)
            continue
        names = row.get("feature_names")
        values = row.get("features")
        if not isinstance(names, list) or not isinstance(values, list) or len(names) != len(values):
            raise RandomProbeExpansionError("RANDOM_PROBE_IRG_ROW_INVALID")
        keep = [index for index, name in enumerate(names) if str(name).rsplit(":", 1)[-1] in selected]
        if not keep:
            raise RandomProbeExpansionError("RANDOM_PROBE_SUBSET_NOT_PRESENT")
        row["feature_names"] = [str(names[index]) for index in keep]
        row["features"] = [float(values[index]) for index in keep]
        row["random_subset_paths"] = sorted(selected)
        row["random_subset_path_count"] = len(selected)
        result.append(row)
    return result


def _available_irg_paths(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    path_sets: list[set[str]] = []
    for row in rows:
        if row.get("selector") != "irg":
            continue
        names = row.get("feature_names")
        if not isinstance(names, list) or not names:
            raise RandomProbeExpansionError("RANDOM_PROBE_IRG_ROW_INVALID")
        path_sets.append({str(name).rsplit(":", 1)[-1] for name in names})
    if not path_sets or any(paths != path_sets[0] for paths in path_sets[1:]):
        raise RandomProbeExpansionError("RANDOM_PROBE_PATH_FRAME_INCONSISTENT")
    return path_sets[0]


def _gram_condition_number(rows: Sequence[Mapping[str, Any]]) -> float | None:
    vectors = [row["features"] for row in rows if row.get("selector") == "irg"]
    if len(vectors) < 2:
        return None
    matrix = np.asarray(vectors, dtype=np.float64)
    matrix -= matrix.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(matrix, compute_uv=False)
    nonzero = singular[singular > max(matrix.shape) * np.finfo(np.float64).eps * singular[0]]
    if nonzero.size < 2:
        return None
    value = float(nonzero[0] / nonzero[-1])
    return value if math.isfinite(value) else None


def _selector_metrics(report: Mapping[str, Any], selector: str) -> Mapping[str, Any]:
    rows = report.get("selectors")
    if not isinstance(rows, list):
        raise RandomProbeExpansionError("RANDOM_PROBE_REPLAY_METRICS_INVALID")
    matches = [row for row in rows if isinstance(row, Mapping) and row.get("selector") == selector]
    if len(matches) != 1:
        raise RandomProbeExpansionError("RANDOM_PROBE_REPLAY_METRICS_INVALID")
    return matches[0]


def _summarize_by_size(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for subset_size in sorted({int(row["subset_size"]) for row in rows}):
        selected = [row for row in rows if int(row["subset_size"]) == subset_size]
        summary: dict[str, object] = {"subset_size": subset_size, "replicate_count": len(selected)}
        for metric in (*_METRICS, "gram_condition_number"):
            values = [float(row[metric]) for row in selected if row.get(metric) is not None]
            summary[metric] = fmean(values) if values else None
            summary[f"{metric}_observed_replicates"] = len(values)
        result.append(summary)
    return result


def _resolve_source(value: object, root: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise RandomProbeExpansionError("RANDOM_PROBE_SOURCE_INVALID")
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve(strict=True)


def _load_mapping(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RandomProbeExpansionError(f"RANDOM_PROBE_JSON_INVALID:{path}") from exc
    if not isinstance(payload, Mapping):
        raise RandomProbeExpansionError(f"RANDOM_PROBE_MAPPING_INVALID:{path}")
    return payload


def _load_jsonl(path: Path) -> list[Mapping[str, Any]]:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise RandomProbeExpansionError(f"RANDOM_PROBE_JSONL_INVALID:{path}") from exc
    if not rows or any(not isinstance(row, Mapping) for row in rows):
        raise RandomProbeExpansionError(f"RANDOM_PROBE_JSONL_INVALID:{path}")
    return rows


def _string_list(value: object, error: str) -> list[str]:
    if not isinstance(value, list) or not value or any(not isinstance(item, str) or not item for item in value):
        raise RandomProbeExpansionError(error)
    return list(value)


def _positive_int_list(value: object, error: str) -> list[int]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, int) or isinstance(item, bool) or item <= 0 for item in value)
    ):
        raise RandomProbeExpansionError(error)
    return list(value)


def _source_ref(role: str, path: Path) -> dict[str, object]:
    return {"role": role, "path": str(path), "sha256": _sha256(path), "size_bytes": path.stat().st_size}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _csv(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return ""
    fields = sorted({key for row in rows for key in row})
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                key: json.dumps(value, sort_keys=True, separators=(",", ":"))
                if isinstance(value, (list, dict))
                else value
                for key, value in row.items()
            }
        )
    return output.getvalue()


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Random Probe Expansion",
        "",
        f"State: `{report['state']}`",
        "",
        f"CPU-only executions: `{report['execution_count']}`",
        "",
        "| Probe paths | Replicates | Sign accuracy | Selection regret | Gram condition |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in report["metric_summary_by_subset_size"]:
        def display(name: str) -> str:
            value = row.get(name)
            return "NA" if value is None else f"{float(value):.6f}"

        lines.append(
            f"| {row['subset_size']} | {row['replicate_count']} | {display('benefit_sign_accuracy')} | "
            f"{display('selection_regret')} | {display('gram_condition_number')} |"
        )
    lines.extend(["", str(report["claim_boundary"]), ""])
    return "\n".join(lines)
