"""Plan the matched ACWM reference-instance selector ablation."""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from wmloop.experiments._artifacts import canonical_json, write_bundle
from wmloop.experiments.spec import SELECTORS


class SelectorAblationError(ValueError):
    """The selector ablation contract or its evidence is malformed."""


def build_selector_ablation_plan(
    *,
    config_path: Path,
    fingerprint_atlas_root: Path,
    output_root: Path,
    effect_label_index: Path | None = None,
    archive_db: Path | None = None,
    cas_root: Path | None = None,
) -> dict[str, object]:
    config = _load_json(config_path, "SELECTOR_ABLATION_CONFIG_INVALID")
    _validate_config(config)
    atlas_root = Path(fingerprint_atlas_root).resolve()
    atlas_manifest = _load_json(atlas_root / "manifest.json", "SELECTOR_ATLAS_MANIFEST_INVALID")
    atlas = _load_json(atlas_root / "fingerprint-atlas.json", "SELECTOR_ATLAS_INVALID")
    projection_rows = _load_jsonl(atlas_root / "selector-input-projections.jsonl")
    environments = tuple(str(value) for value in config["environments"])
    selectors = tuple(str(value["selector"]) for value in config["selectors"])
    seeds = tuple(int(value) for value in config["seeds"])
    blockers: list[dict[str, object]] = []
    checklist: list[dict[str, object]] = []

    atlas_ready = (
        atlas_manifest.get("state") == "ready"
        and atlas_manifest.get("environment_count") == len(environments)
        and atlas_manifest.get("measurement_complete_count") == len(environments)
        and atlas_manifest.get("locality_calibrated_count") == len(environments)
    )
    _check(checklist, blockers, "eight_environment_fingerprint_atlas", atlas_ready, "atlas must be 8/8 complete and calibrated")
    projection_pairs = {(str(row.get("environment")), str(row.get("selector"))) for row in projection_rows}
    expected_pairs = {(environment, selector) for environment in environments for selector in selectors}
    projections_ready = projection_pairs == expected_pairs and len(projection_rows) == len(expected_pairs)
    _check(checklist, blockers, "four_selector_projection_rows_per_environment", projections_ready, "exactly one projection per environment and selector")

    labels = _load_effect_labels(effect_label_index) if effect_label_index else []
    label_environments = {str(row["environment"]) for row in labels if row.get("settled") is True}
    labels_ready = set(environments).issubset(label_environments)
    _check(
        checklist,
        blockers,
        "settled_effect_label_index",
        labels_ready,
        f"settled target coverage {len(label_environments & set(environments))}/{len(environments)}",
    )
    settled_candidates, ambiguous_candidates = _settled_consensus_candidates(
        labels=labels,
        environments=environments,
    )
    fold_candidate_support = {
        target: sorted(
            settled_candidates[target]
            & set().union(*(settled_candidates[source] for source in environments if source != target))
        )
        for target in environments
    }
    supported_fold_count = sum(bool(values) for values in fold_candidate_support.values())
    identifiable_fold_count = sum(len(values) >= 2 for values in fold_candidate_support.values())
    candidate_support_ready = identifiable_fold_count == len(environments)
    _check(
        checklist,
        blockers,
        "nonleaking_candidate_pool_support",
        candidate_support_ready,
        (
            f"held-out folds with any source-supported target candidate {supported_fold_count}/{len(environments)}; "
            f"folds with at least two candidates for selector identification {identifiable_fold_count}/{len(environments)}"
        ),
    )

    trials: list[dict[str, object]] = []
    sequence = 0
    for target in environments:
        sources = [environment for environment in environments if environment != target]
        for selector in selectors:
            for seed in seeds:
                sequence += 1
                trials.append(
                    {
                        "sequence_index": sequence,
                        "trial_id": f"holdout_{target}__{selector}__s{seed}",
                        "fold_id": f"holdout_{target}",
                        "target_environment": target,
                        "source_environments": sources,
                        "selector": selector,
                        "seed": seed,
                        "source_supported_target_candidates": fold_candidate_support[target],
                        "candidate_pool_contract": config["matched_trial_contract"]["candidate_pool"],
                        "screen_budget_contract": config["matched_trial_contract"]["screen_budget"],
                        "confirm_budget_contract": config["matched_trial_contract"]["confirm_budget"],
                        "formal_evidence_requires": "settled_official_gate_receipt",
                    }
                )
    report = {
        "schema_version": 1,
        "artifact_type": "verdiwm-reference-selector-ablation-plan",
        "experiment_id": config["experiment_id"],
        "state": "ready" if not blockers else "blocked",
        "cpu_replay_ready": atlas_ready and projections_ready,
        "gpu_confirmation_ready": not blockers,
        "claim_scope": config["claim_scope"],
        "fold_count": len(environments),
        "selector_count": len(selectors),
        "seed_count": len(seeds),
        "planned_trial_count": len(trials),
        "effect_label_count": len(labels),
        "settled_label_environment_count": len(label_environments & set(environments)),
        "candidate_supported_fold_count": supported_fold_count,
        "selector_identifiable_fold_count": identifiable_fold_count,
        "fold_candidate_support": fold_candidate_support,
        "ambiguous_candidates_by_environment": ambiguous_candidates,
        "evidence_checklist": checklist,
        "blockers": blockers,
        "trials": trials,
    }
    destination = Path(output_root).resolve()
    files = {
        "selector-ablation-plan.json": canonical_json(report),
        "selector-ablation-plan.md": _markdown(report).encode("utf-8"),
        "tables/planned-trials.csv": _csv(trials).encode("utf-8"),
        "tables/evidence-checklist.csv": _csv(checklist).encode("utf-8"),
        "input-config.json": canonical_json(config),
        "input-atlas-manifest.json": canonical_json(atlas_manifest),
    }
    if effect_label_index is not None:
        files["input-effect-label-index.json"] = canonical_json({"labels": labels})
    return write_bundle(
        output_root=destination,
        files=files,
        manifest_fields={
            "artifact_type": "verdiwm-reference-selector-ablation-plan-manifest",
            "state": report["state"],
            "experiment_id": config["experiment_id"],
            "fold_count": len(environments),
            "planned_trial_count": len(trials),
            "blocker_count": len(blockers),
            "candidate_supported_fold_count": supported_fold_count,
            "selector_identifiable_fold_count": identifiable_fold_count,
            "cpu_replay_ready": report["cpu_replay_ready"],
            "gpu_confirmation_ready": report["gpu_confirmation_ready"],
            "report_path": str(destination / "selector-ablation-plan.json"),
        },
        archive_db=archive_db,
        cas_root=cas_root,
    )


def _settled_consensus_candidates(
    *,
    labels: list[Mapping[str, Any]],
    environments: tuple[str, ...],
) -> tuple[dict[str, set[str]], dict[str, list[str]]]:
    signs_by_candidate: dict[tuple[str, str], list[bool]] = {}
    for row in labels:
        environment = row.get("environment")
        primitive = row.get("primitive")
        positive = row.get("positive")
        if (
            row.get("settled") is not True
            or environment not in environments
            or not isinstance(primitive, str)
            or not primitive
            or not isinstance(positive, bool)
        ):
            continue
        signs_by_candidate.setdefault((str(environment), primitive), []).append(positive)

    settled_candidates = {environment: set() for environment in environments}
    ambiguous_candidates = {environment: [] for environment in environments}
    for (environment, primitive), signs in signs_by_candidate.items():
        if len(set(signs)) == 1:
            settled_candidates[environment].add(primitive)
        else:
            ambiguous_candidates[environment].append(primitive)
    return settled_candidates, {
        environment: sorted(primitives)
        for environment, primitives in ambiguous_candidates.items()
    }


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("artifact_type") != "verdiwm-reference-selector-ablation":
        raise SelectorAblationError("SELECTOR_ABLATION_CONFIG_TYPE_INVALID")
    environments = tuple(str(value) for value in config.get("environments", ()))
    if len(environments) != 8 or len(set(environments)) != 8:
        raise SelectorAblationError("SELECTOR_ABLATION_ENVIRONMENTS_INVALID")
    selectors = tuple(str(value["selector"]) for value in config.get("selectors", ()))
    if selectors != SELECTORS:
        raise SelectorAblationError("SELECTOR_ABLATION_SELECTORS_INVALID")
    seeds = tuple(int(value) for value in config.get("seeds", ()))
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise SelectorAblationError("SELECTOR_ABLATION_SEEDS_INVALID")
    if config.get("split_contract", {}).get("kind") != "leave_one_environment_out":
        raise SelectorAblationError("SELECTOR_ABLATION_SPLIT_INVALID")


def _load_effect_labels(path: Path) -> list[Mapping[str, Any]]:
    payload = _load_json(Path(path), "SELECTOR_EFFECT_LABEL_INDEX_INVALID")
    labels = payload.get("labels")
    if not isinstance(labels, list) or any(not isinstance(row, Mapping) for row in labels):
        raise SelectorAblationError("SELECTOR_EFFECT_LABEL_INDEX_INVALID")
    return labels


def _check(
    checklist: list[dict[str, object]],
    blockers: list[dict[str, object]],
    evidence_id: str,
    passed: bool,
    detail: str,
) -> None:
    checklist.append({"evidence_id": evidence_id, "passed": passed, "detail": detail})
    if not passed:
        blockers.append({"code": "evidence_missing", "evidence_id": evidence_id, "detail": detail})


def _load_json(path: Path, code: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SelectorAblationError(f"{code}:{path}") from exc
    if not isinstance(payload, Mapping):
        raise SelectorAblationError(f"{code}:{path}")
    return payload


def _load_jsonl(path: Path) -> list[Mapping[str, Any]]:
    try:
        rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise SelectorAblationError(f"SELECTOR_PROJECTIONS_INVALID:{path}") from exc
    if any(not isinstance(row, Mapping) for row in rows):
        raise SelectorAblationError(f"SELECTOR_PROJECTIONS_INVALID:{path}")
    return rows


def _csv(rows: list[Mapping[str, object]]) -> str:
    if not rows:
        return ""
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        normalized = {key: ";".join(str(value) for value in item) if isinstance(item, list) else item for key, item in row.items()}
        writer.writerow(normalized)
    return output.getvalue()


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# ACWM-Phys Selector Ablation Plan",
        "",
        f"State: `{report['state']}`",
        f"CPU replay ready: `{str(report['cpu_replay_ready']).lower()}`",
        f"GPU confirmation ready: `{str(report['gpu_confirmation_ready']).lower()}`",
        f"Planned matched trials: `{report['planned_trial_count']}`",
        "",
        "## Evidence Checklist",
        "",
    ]
    for row in report["evidence_checklist"]:
        marker = "PASS" if row["passed"] else "BLOCKED"
        lines.append(f"- `{marker}` `{row['evidence_id']}`: {row['detail']}")
    lines.extend(
        [
            "",
            "CPU-only selector replay may proceed from the frozen projections. GPU confirmation remains blocked until every held-out environment has a non-leaking settled effect-label pool.",
        ]
    )
    return "\n".join(lines) + "\n"
