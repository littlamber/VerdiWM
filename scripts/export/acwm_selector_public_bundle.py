#!/usr/bin/env python3
"""Export a path-free public ACWM selector-ablation evidence bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path


class PublicSelectorBundleError(ValueError):
    """Selector evidence is incomplete or unsafe for public export."""


def build_public_selector_bundle(
    *,
    effect_label_index: Path,
    selector_plan: Path,
    selector_replay: Path,
    output_root: Path,
) -> dict[str, object]:
    labels = _load_json(effect_label_index)
    plan = _load_json(selector_plan)
    replay = _load_json(selector_replay)
    environments = tuple(str(value) for value in labels.get("expected_environments", ()))
    if len(environments) != 8 or len(set(environments)) != 8:
        raise PublicSelectorBundleError("PUBLIC_SELECTOR_ENVIRONMENTS_INVALID")
    if plan.get("state") != "ready" or int(plan.get("selector_identifiable_fold_count", 0)) != 8:
        raise PublicSelectorBundleError("PUBLIC_SELECTOR_PLAN_NOT_READY")
    if replay.get("formal_comparison_ready") is not True or int(replay.get("multi_candidate_environment_count", 0)) != 8:
        raise PublicSelectorBundleError("PUBLIC_SELECTOR_REPLAY_NOT_READY")

    consensus, ambiguous = _consensus_summary(labels, environments)
    label_summary = {
        "schema_version": 1,
        "artifact_type": "verdiwm-public-effect-label-summary",
        "environment_count": len(environments),
        "settled_label_count": labels.get("settled_label_count"),
        "settled_positive_count": labels.get("settled_positive_count"),
        "settled_negative_count": labels.get("settled_negative_count"),
        "consensus_primitives_by_environment": consensus,
        "ambiguous_primitives_by_environment": ambiguous,
        "claim_boundary": "Only settled official-gate signs are summarized. Internal paths, checkpoints, and evidence references are excluded.",
    }
    plan_summary = {
        "schema_version": 1,
        "artifact_type": "verdiwm-public-selector-plan-summary",
        "experiment_id": plan.get("experiment_id"),
        "fold_count": plan.get("fold_count"),
        "selector_count": plan.get("selector_count"),
        "seed_count": plan.get("seed_count"),
        "planned_trial_count": plan.get("planned_trial_count"),
        "candidate_supported_fold_count": plan.get("candidate_supported_fold_count"),
        "selector_identifiable_fold_count": plan.get("selector_identifiable_fold_count"),
        "fold_candidate_support": plan.get("fold_candidate_support"),
        "ambiguous_candidates_by_environment": plan.get("ambiguous_candidates_by_environment"),
        "claim_scope": plan.get("claim_scope"),
    }
    replay_public = {
        key: replay.get(key)
        for key in (
            "schema_version",
            "artifact_type",
            "state",
            "claim_boundary",
            "planned_cell_count",
            "evaluated_cell_count",
            "abstained_cell_count",
            "environment_count",
            "evaluated_environment_count",
            "multi_candidate_environment_count",
            "selector_choice_divergence_environment_count",
            "selector_choice_divergence_environments",
            "formal_comparison_ready",
            "selection_discrimination_ready",
            "distance_contract",
            "ranking_contract",
            "target_label_contract",
            "selectors",
            "cells",
        )
    }
    bundle = {
        "schema_version": 1,
        "artifact_type": "verdiwm-public-selector-evidence-bundle",
        "state": "ready",
        "environment_count": 8,
        "replay_cell_count": replay_public["evaluated_cell_count"],
        "formal_comparison_ready": replay_public["formal_comparison_ready"],
        "selection_discrimination_ready": replay_public["selection_discrimination_ready"],
        "claim_boundary": (
            "The 8-environment evidence matrix is complete. Ranking and sign metrics are public evidence; "
            "top-1 selector superiority is not claimed when selection_discrimination_ready is false."
        ),
    }
    files = {
        "README.md": _markdown(bundle, replay_public).encode("utf-8"),
        "bundle.json": _json_bytes(bundle),
        "effect-label-summary.json": _json_bytes(label_summary),
        "selector-plan-summary.json": _json_bytes(plan_summary),
        "selector-replay.json": _json_bytes(replay_public),
        "tables/selector-metrics.csv": _csv(replay_public.get("selectors", [])).encode("utf-8"),
        "tables/selection-cells.csv": _csv(replay_public.get("cells", [])).encode("utf-8"),
    }
    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise PublicSelectorBundleError("PUBLIC_SELECTOR_OUTPUT_EXISTS")
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        for relative, content in files.items():
            path = temporary / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        _write_manifest(temporary)
        _assert_path_free(temporary)
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return bundle


def _consensus_summary(
    payload: Mapping[str, object], environments: Sequence[str]
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    signs: dict[tuple[str, str], list[bool]] = defaultdict(list)
    rows = payload.get("labels")
    for row in rows if isinstance(rows, list) else ():
        if not isinstance(row, Mapping) or row.get("settled") is not True or not isinstance(row.get("positive"), bool):
            continue
        environment = str(row.get("environment") or "")
        primitive = str(row.get("primitive") or "")
        if environment in environments and primitive:
            signs[(environment, primitive)].append(bool(row["positive"]))
    consensus = {environment: [] for environment in environments}
    ambiguous = {environment: [] for environment in environments}
    for (environment, primitive), values in signs.items():
        (consensus if len(set(values)) == 1 else ambiguous)[environment].append(primitive)
    return (
        {key: sorted(value) for key, value in consensus.items()},
        {key: sorted(value) for key, value in ambiguous.items()},
    )


def _markdown(bundle: Mapping[str, object], replay: Mapping[str, object]) -> str:
    lines = [
        "# VerdiWM ACWM Selector Ablation Evidence",
        "",
        str(bundle["claim_boundary"]),
        "",
        f"- Environments: `{bundle['environment_count']}`",
        f"- Evaluated replay cells: `{bundle['replay_cell_count']}`",
        f"- Formal comparison ready: `{str(bundle['formal_comparison_ready']).lower()}`",
        f"- Selection discrimination ready: `{str(bundle['selection_discrimination_ready']).lower()}`",
        "",
        "| Selector | Top-1 positive | Negative selection | Sign accuracy | Kendall tau | Regret |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in replay.get("selectors", []):
        if not isinstance(row, Mapping):
            continue
        lines.append(
            "| {selector} | {top1_positive_hit:.4f} | {negative_selection:.4f} | {benefit_sign_accuracy:.4f} | {ranking_kendall_tau:.4f} | {selection_regret:.4f} |".format(
                **row
            )
        )
    return "\n".join(lines) + "\n"


def _csv(rows: object) -> str:
    values = [dict(row) for row in rows if isinstance(row, Mapping)] if isinstance(rows, list) else []
    if not values:
        return ""
    fieldnames = sorted({key for row in values for key in row if key not in {"source_effects"}})
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for row in values:
        writer.writerow({key: row.get(key, "") for key in fieldnames})
    return output.getvalue()


def _load_json(path: Path) -> Mapping[str, object]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PublicSelectorBundleError(f"PUBLIC_SELECTOR_JSON_INVALID:{path}") from exc
    if not isinstance(payload, Mapping):
        raise PublicSelectorBundleError(f"PUBLIC_SELECTOR_JSON_INVALID:{path}")
    return payload


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _write_manifest(root: Path) -> None:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.sha256":
            rows.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root).as_posix()}")
    (root / "MANIFEST.sha256").write_text("\n".join(rows) + "\n", encoding="utf-8")


def _assert_path_free(root: Path) -> None:
    blocked_prefixes = ("/" + "mnt" + "/", "/" + "root" + "/")
    for path in root.rglob("*"):
        if path.is_file() and path.suffix in {".json", ".md", ".csv"}:
            text = path.read_text(encoding="utf-8")
            if any(prefix in text for prefix in blocked_prefixes):
                raise PublicSelectorBundleError(f"PUBLIC_SELECTOR_MACHINE_PATH:{path.name}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--effect-label-index", type=Path, required=True)
    parser.add_argument("--selector-plan", type=Path, required=True)
    parser.add_argument("--selector-replay", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    report = build_public_selector_bundle(
        effect_label_index=args.effect_label_index,
        selector_plan=args.selector_plan,
        selector_replay=args.selector_replay,
        output_root=args.output_root,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
