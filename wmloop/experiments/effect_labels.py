"""Index settled ACWM official-gate outcomes for selector supervision."""

from __future__ import annotations

import csv
import io
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Mapping

from wmloop.experiments._artifacts import canonical_json, write_bundle
from wmloop.experiments.acwm_fingerprint import sha256_file


class EffectLabelIndexError(ValueError):
    """Historical official-gate evidence cannot be indexed safely."""


def build_effect_label_index(
    *,
    reports_root: Path,
    output_root: Path,
    expected_environments: Sequence[str],
    archive_db: Path | None = None,
    cas_root: Path | None = None,
) -> dict[str, object]:
    root = Path(reports_root).resolve()
    labels: list[dict[str, object]] = []
    manifest_paths = sorted(
        {
            *root.glob("acwm-autoloop-confirm-official-gate-*/manifest.json"),
            *root.glob("acwm-effect-label-gate-*/manifest.json"),
        }
    )
    for manifest_path in manifest_paths:
        payload = _load_json(manifest_path)
        gate = payload.get("official_quality_gate")
        if not isinstance(gate, Mapping):
            continue
        checks = gate.get("checks")
        gate_pass = gate.get("pass")
        settled = (
            payload.get("state") == "ready"
            and gate.get("state") in {"pass", "fail"}
            and isinstance(gate_pass, bool)
            and isinstance(checks, Mapping)
            and bool(checks)
            and all(isinstance(value, bool) for value in checks.values())
            and isinstance(payload.get("environment"), str)
            and isinstance(payload.get("primitive"), str)
            and isinstance(payload.get("seed"), int)
        )
        delta = gate.get("delta_candidate_minus_baseline")
        labels.append(
            {
                "schema_version": 1,
                "artifact_type": "verdiwm-settled-effect-label",
                "label_id": manifest_path.parent.name,
                "environment": payload.get("environment"),
                "primitive": payload.get("primitive"),
                "seed": payload.get("seed"),
                "train_steps": payload.get("candidate_checkpoint_step"),
                "inference_steps": payload.get("steps"),
                "label_source": (
                    "independent_confirmation_gate"
                    if manifest_path.parent.name.startswith("acwm-autoloop-confirm-official-gate-")
                    else "retained_checkpoint_completion_gate"
                ),
                "settled": settled,
                "positive": bool(gate_pass) if settled else None,
                "official_gate_state": gate.get("state"),
                "official_gate_checks": dict(checks) if isinstance(checks, Mapping) else None,
                "delta_candidate_minus_baseline": dict(delta) if isinstance(delta, Mapping) else None,
                "evidence_ref": str(manifest_path),
                "evidence_sha256": sha256_file(manifest_path),
                "claim_boundary": "Target-local settled official-gate label for selector supervision; not cross-backbone causal evidence.",
            }
        )
    expected = tuple(str(value) for value in expected_environments)
    settled_labels = [row for row in labels if row["settled"] is True]
    settled_environments = {str(row["environment"]) for row in settled_labels}
    counts = Counter(str(row["environment"]) for row in settled_labels)
    report = {
        "schema_version": 1,
        "artifact_type": "verdiwm-settled-effect-label-index",
        "state": "ready",
        "claim_boundary": "Only complete official confirmation gates or explicit retained-checkpoint completion gates become settled labels. Screen, hard-case-only, and visual-only evidence are excluded.",
        "expected_environments": list(expected),
        "label_count": len(labels),
        "settled_label_count": len(settled_labels),
        "settled_positive_count": sum(row["positive"] is True for row in settled_labels),
        "settled_negative_count": sum(row["positive"] is False for row in settled_labels),
        "settled_environment_count": len(settled_environments & set(expected)),
        "missing_environments": [environment for environment in expected if environment not in settled_environments],
        "settled_labels_by_environment": {environment: counts.get(environment, 0) for environment in expected},
        "labels": labels,
    }
    destination = Path(output_root).resolve()
    return write_bundle(
        output_root=destination,
        files={
            "effect-label-index.json": canonical_json(report),
            "effect-label-index.md": _markdown(report).encode("utf-8"),
            "tables/effect-labels.csv": _csv(labels).encode("utf-8"),
        },
        manifest_fields={
            "artifact_type": "verdiwm-settled-effect-label-index-manifest",
            "state": "ready",
            "label_count": len(labels),
            "settled_label_count": len(settled_labels),
            "settled_environment_count": len(settled_environments & set(expected)),
            "missing_environment_count": len([environment for environment in expected if environment not in settled_environments]),
            "report_path": str(destination / "effect-label-index.json"),
        },
        archive_db=archive_db,
        cas_root=cas_root,
    )


def build_effect_label_completion_plan(
    *,
    effect_label_index_path: Path,
    screen_summary_path: Path,
    candidate_frontier_path: Path,
    output_root: Path,
    candidates_per_environment: int = 3,
    archive_db: Path | None = None,
    cas_root: Path | None = None,
) -> dict[str, object]:
    if candidates_per_environment < 2:
        raise EffectLabelIndexError("EFFECT_LABEL_COMPLETION_CANDIDATE_COUNT_INVALID")
    label_index = _load_json(Path(effect_label_index_path))
    screen_summary = _load_json(Path(screen_summary_path))
    frontier = _load_json(Path(candidate_frontier_path))
    missing = tuple(str(value) for value in label_index.get("missing_environments", ()))
    screen_rows = screen_summary.get("rows")
    frontier_rows = frontier.get("environments")
    if not isinstance(screen_rows, list) or not isinstance(frontier_rows, list):
        raise EffectLabelIndexError("EFFECT_LABEL_COMPLETION_INPUT_INVALID")
    frontier_by_environment = {
        str(row.get("environment")): row for row in frontier_rows if isinstance(row, Mapping)
    }
    actions: list[dict[str, object]] = []
    environment_plans: list[dict[str, object]] = []
    ordinal = 0
    for environment in missing:
        retained = [
            row
            for row in screen_rows
            if isinstance(row, Mapping)
            and str(row.get("environment")) == environment
            and row.get("candidate_checkpoint_retained") is True
            and isinstance(row.get("latest_checkpoint_path"), str)
            and row.get("latest_checkpoint_path")
        ]
        positive = sorted(
            (row for row in retained if float(row.get("delta_primary_metric", 0.0)) > 0.0),
            key=lambda row: float(row.get("delta_primary_metric", 0.0)),
            reverse=True,
        )
        negative = sorted(
            (row for row in retained if float(row.get("delta_primary_metric", 0.0)) <= 0.0),
            key=lambda row: float(row.get("delta_primary_metric", 0.0)),
        )
        selected: list[dict[str, object]] = []
        for polarity, rows in (("screen_positive", positive), ("screen_nonpositive", negative)):
            if not rows:
                continue
            source = rows[0]
            selected.append(
                {
                    "action": "official_gate_existing_checkpoint",
                    "screen_polarity": polarity,
                    "primitive": str(source.get("primitive")),
                    "parameters": {},
                    "seed": int(source.get("seed")),
                    "train_steps": int(source.get("train_steps") or 0),
                    "source_campaign_id": str(source.get("campaign_id")),
                    "source_manifest": str(source.get("manifest_path")),
                    "checkpoint_ref": str(source.get("latest_checkpoint_path")),
                    "checkpoint_sha256": source.get("candidate_checkpoint_sha256"),
                    "screen_delta": float(source.get("delta_primary_metric", 0.0)),
                }
            )
        frontier_row = frontier_by_environment.get(environment, {})
        open_candidates = frontier_row.get("open_candidates", ()) if isinstance(frontier_row, Mapping) else ()
        seen_primitives = {str(row["primitive"]) for row in selected}
        if isinstance(open_candidates, list):
            ranked = sorted(
                (row for row in open_candidates if isinstance(row, Mapping)),
                key=lambda row: (
                    int(row.get("experience_routing", {}).get("rank_band", 9))
                    if isinstance(row.get("experience_routing"), Mapping)
                    else 9,
                    int(row.get("failure_evidence_penalty", 0)),
                    str(row.get("primitive", "")),
                    json.dumps(row.get("parameters", {}), sort_keys=True),
                ),
            )
            for source in ranked:
                primitive = str(source.get("primitive"))
                if not primitive or primitive in seen_primitives:
                    continue
                selected.append(
                    {
                        "action": "screen_512_then_official_gate_if_positive",
                        "screen_polarity": "unknown",
                        "primitive": primitive,
                        "parameters": dict(source.get("parameters", {}))
                        if isinstance(source.get("parameters"), Mapping)
                        else {},
                        "seed": None,
                        "train_steps": 512,
                        "source_campaign_id": None,
                        "source_manifest": None,
                        "checkpoint_ref": None,
                        "checkpoint_sha256": None,
                        "screen_delta": None,
                    }
                )
                seen_primitives.add(primitive)
                if len(selected) >= candidates_per_environment:
                    break
        selected = selected[:candidates_per_environment]
        if len(selected) < 2:
            raise EffectLabelIndexError(f"EFFECT_LABEL_COMPLETION_CANDIDATES_INSUFFICIENT:{environment}")
        environment_plans.append(
            {
                "environment": environment,
                "current_settled_label_count": int(
                    label_index.get("settled_labels_by_environment", {}).get(environment, 0)
                ),
                "target_new_settled_labels": 2,
                "candidate_count": len(selected),
                "reusable_checkpoint_count": sum(
                    row["action"] == "official_gate_existing_checkpoint" for row in selected
                ),
                "new_screen_candidate_count": sum(
                    row["action"].startswith("screen_512") for row in selected
                ),
                "has_reusable_positive_screen": bool(positive),
                "has_reusable_nonpositive_screen": bool(negative),
                "requires_new_512_screen": any(row["action"].startswith("screen_512") for row in selected),
            }
        )
        for row in selected:
            ordinal += 1
            actions.append(
                {
                    "ordinal": ordinal,
                    "environment": environment,
                    "priority": "P0" if row["action"] == "official_gate_existing_checkpoint" else "P1",
                    **row,
                    "formal_label_requires": "settled frozen 50-step official gate receipt",
                    "claim_boundary": "Queue entry only; screen evidence never becomes a selector label without the official gate.",
                }
            )
    report = {
        "schema_version": 1,
        "artifact_type": "verdiwm-acwm-effect-label-completion-plan",
        "state": "ready",
        "claim_boundary": "This plan minimizes new training by reusing retained screen checkpoints where possible. It does not create labels or positive claims.",
        "missing_environments": list(missing),
        "environment_count": len(missing),
        "action_count": len(actions),
        "reusable_checkpoint_action_count": sum(
            row["action"] == "official_gate_existing_checkpoint" for row in actions
        ),
        "new_screen_action_count": sum(row["action"].startswith("screen_512") for row in actions),
        "environment_plans": environment_plans,
        "actions": actions,
    }
    destination = Path(output_root).resolve()
    return write_bundle(
        output_root=destination,
        files={
            "effect-label-completion-plan.json": canonical_json(report),
            "effect-label-completion-plan.md": _completion_markdown(report).encode("utf-8"),
            "tables/actions.csv": _completion_csv(actions).encode("utf-8"),
        },
        manifest_fields={
            "artifact_type": "verdiwm-acwm-effect-label-completion-plan-manifest",
            "state": "ready",
            "environment_count": len(missing),
            "action_count": len(actions),
            "reusable_checkpoint_action_count": report["reusable_checkpoint_action_count"],
            "new_screen_action_count": report["new_screen_action_count"],
            "report_path": str(destination / "effect-label-completion-plan.json"),
        },
        archive_db=archive_db,
        cas_root=cas_root,
    )


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EffectLabelIndexError(f"EFFECT_LABEL_MANIFEST_INVALID:{path}") from exc
    if not isinstance(payload, Mapping):
        raise EffectLabelIndexError(f"EFFECT_LABEL_MANIFEST_INVALID:{path}")
    return payload


def _csv(rows: list[Mapping[str, object]]) -> str:
    if not rows:
        return ""
    fields = [
        "label_id",
        "environment",
        "primitive",
        "seed",
        "train_steps",
        "inference_steps",
        "settled",
        "positive",
        "official_gate_state",
        "evidence_ref",
        "evidence_sha256",
    ]
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows({field: row.get(field) for field in fields} for row in rows)
    return output.getvalue()


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# ACWM-Phys Settled Effect Labels",
        "",
        str(report["claim_boundary"]),
        "",
        f"Settled labels: `{report['settled_label_count']}` "
        f"(`{report['settled_positive_count']}` positive, `{report['settled_negative_count']}` negative).",
        f"Environment coverage: `{report['settled_environment_count']}/{len(report['expected_environments'])}`.",
        "",
        "| Environment | Settled labels |",
        "|---|---:|",
    ]
    for environment, count in report["settled_labels_by_environment"].items():
        lines.append(f"| {environment} | {count} |")
    lines.extend(["", f"Missing environments: `{', '.join(report['missing_environments']) or 'none'}`."])
    return "\n".join(lines) + "\n"


def _completion_csv(rows: Sequence[Mapping[str, object]]) -> str:
    fields = [
        "ordinal",
        "environment",
        "priority",
        "action",
        "screen_polarity",
        "primitive",
        "parameters",
        "seed",
        "train_steps",
        "source_campaign_id",
        "source_manifest",
        "checkpoint_ref",
        "checkpoint_sha256",
        "screen_delta",
        "formal_label_requires",
    ]
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        projected = {field: row.get(field) for field in fields}
        projected["parameters"] = json.dumps(projected["parameters"], sort_keys=True)
        writer.writerow(projected)
    return output.getvalue()


def _completion_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# ACWM-Phys Effect-Label Completion Plan",
        "",
        str(report["claim_boundary"]),
        "",
        f"Missing environments: `{report['environment_count']}`.",
        f"Reusable checkpoint gates: `{report['reusable_checkpoint_action_count']}`.",
        f"New 512-step screens: `{report['new_screen_action_count']}`.",
        "",
        "| Environment | Current labels | Candidates | Reuse checkpoint | New screen |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in report["environment_plans"]:
        lines.append(
            f"| {row['environment']} | {row['current_settled_label_count']} | {row['candidate_count']} | "
            f"{row['reusable_checkpoint_count']} | {row['new_screen_candidate_count']} |"
        )
    return "\n".join(lines) + "\n"
