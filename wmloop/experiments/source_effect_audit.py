"""Audit source-effect evidence before transfer-certificate recovery."""

from __future__ import annotations

import csv
import io
import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import fmean
from typing import Any

from wmloop.experiments._artifacts import canonical_json, write_bundle
from wmloop.experiments.acwm_fingerprint import sha256_file


class SourceEffectAuditError(ValueError):
    """Source-effect evidence is malformed or cannot be verified."""


_GATE_PATTERNS = (
    "acwm-formal-trainseed-gate-*/manifest.json",
    "acwm-autoloop-confirm-official-gate-*/manifest.json",
    "acwm-effect-label-gate-*/manifest.json",
    "acwm-autoloop-official-gate-*/manifest.json",
    "acwm-official-gate-*/manifest.json",
)


def build_source_effect_audit(
    *,
    effect_label_index_path: Path,
    reports_root: Path,
    additional_receipt_roots: Sequence[Path] = (),
    output_root: Path,
    primitive: str,
    minimum_independent_seeds: int = 3,
    archive_db: Path | None = None,
    cas_root: Path | None = None,
) -> dict[str, object]:
    """Build a protocol-aware audit of one primitive's source-effect signs."""

    if not primitive:
        raise SourceEffectAuditError("SOURCE_EFFECT_AUDIT_PRIMITIVE_INVALID")
    if minimum_independent_seeds < 2:
        raise SourceEffectAuditError("SOURCE_EFFECT_AUDIT_MINIMUM_SEEDS_INVALID")
    index_path = Path(effect_label_index_path).resolve()
    reports = Path(reports_root).resolve()
    label_index = _load_json(index_path, "SOURCE_EFFECT_AUDIT_INDEX_INVALID")
    if label_index.get("artifact_type") != "verdiwm-settled-effect-label-index":
        raise SourceEffectAuditError("SOURCE_EFFECT_AUDIT_INDEX_TYPE_INVALID")
    environments = tuple(str(value) for value in label_index.get("expected_environments", ()))
    if not environments or len(set(environments)) != len(environments):
        raise SourceEffectAuditError("SOURCE_EFFECT_AUDIT_ENVIRONMENTS_INVALID")
    labels = label_index.get("labels")
    if not isinstance(labels, list) or any(not isinstance(row, Mapping) for row in labels):
        raise SourceEffectAuditError("SOURCE_EFFECT_AUDIT_LABELS_INVALID")

    indexed_rows = _verify_indexed_rows(
        labels=labels,
        primitive=primitive,
        reports_root=reports,
    )
    receipt_roots = (reports, *(Path(root).resolve() for root in additional_receipt_roots))
    if any(not root.is_dir() for root in receipt_roots):
        raise SourceEffectAuditError("SOURCE_EFFECT_AUDIT_RECEIPT_ROOT_INVALID")
    receipt_rows = _inventory_receipts(
        receipt_roots=receipt_roots,
        primitive=primitive,
        environments=environments,
    )
    environment_rows = [
        _environment_audit(
            environment=environment,
            indexed=[row for row in indexed_rows if row["environment"] == environment],
            receipts=[row for row in receipt_rows if row["environment"] == environment],
            minimum_independent_seeds=minimum_independent_seeds,
        )
        for environment in environments
    ]
    stable_positive = [
        str(row["environment"])
        for row in environment_rows
        if row["classification"] == "stable_positive"
    ]
    stable_negative = [
        str(row["environment"])
        for row in environment_rows
        if row["classification"] == "stable_negative"
    ]
    underreplicated_positive = [
        str(row["environment"])
        for row in environment_rows
        if row["classification"] == "positive_underreplicated"
    ]
    unstable = [
        str(row["environment"])
        for row in environment_rows
        if row["classification"] in {"same_protocol_reproduction_conflict", "eval_seed_sensitive"}
    ]
    if stable_positive and stable_negative:
        collision_verdict = "stable_cross_environment_mechanism_collision"
    elif stable_negative and underreplicated_positive:
        collision_verdict = "mixed_effects_with_underreplicated_positive_sources"
    elif stable_negative and unstable:
        collision_verdict = "mixed_effects_with_unstable_positive_sources"
    else:
        collision_verdict = "mechanism_collision_not_established"
    work_orders = [
        {
            "priority": (
                "P0"
                if row["classification"]
                in {
                    "positive_underreplicated",
                    "same_protocol_reproduction_conflict",
                    "eval_seed_sensitive",
                }
                else "P2"
                if row["recommended_action"] == "retain_settled_direction"
                else "P1"
            ),
            "environment": row["environment"],
            "primitive": primitive,
            "classification": row["classification"],
            "action": row["recommended_action"],
            "required_new_training_seed_count": row["required_new_training_seed_count"],
            "required_eval_seed_count": row["required_eval_seed_count"],
            "claim_boundary": "Evidence-repair work order only; it cannot change the frozen transfer certificate.",
        }
        for row in environment_rows
        if row["recommended_action"] != "no_action_no_current_evidence"
    ]
    report = {
        "schema_version": 1,
        "artifact_type": "verdiwm-source-effect-evidence-audit",
        "state": "ready",
        "primitive": primitive,
        "minimum_independent_training_seeds": minimum_independent_seeds,
        "claim_boundary": (
            "This audit classifies source-effect evidence quality. It does not alter the frozen "
            "transfer certificate, admit a probe, or establish target-model quality improvement."
        ),
        "input_effect_label_index": str(index_path),
        "input_effect_label_index_sha256": sha256_file(index_path),
        "receipt_roots": [str(root) for root in receipt_roots],
        "indexed_receipt_count": len(indexed_rows),
        "discovered_official_gate_receipt_count": len(receipt_rows),
        "protocol_complete_receipt_count": sum(row["protocol_complete"] is True for row in receipt_rows),
        "protocol_incomplete_receipt_count": sum(row["protocol_complete"] is False for row in receipt_rows),
        "collision_verdict": collision_verdict,
        "stable_positive_environments": stable_positive,
        "stable_negative_environments": stable_negative,
        "underreplicated_positive_environments": underreplicated_positive,
        "unstable_environments": unstable,
        "environment_audits": environment_rows,
        "work_orders": work_orders,
    }
    destination = Path(output_root).resolve()
    return write_bundle(
        output_root=destination,
        files={
            "source-effect-audit.json": canonical_json(report),
            "source-effect-audit.md": _markdown(report).encode("utf-8"),
            "tables/environment-audit.csv": _csv(environment_rows).encode("utf-8"),
            "tables/indexed-labels.csv": _csv(indexed_rows).encode("utf-8"),
            "tables/official-gate-receipts.csv": _csv(receipt_rows).encode("utf-8"),
            "tables/work-orders.csv": _csv(work_orders).encode("utf-8"),
        },
        manifest_fields={
            "artifact_type": "verdiwm-source-effect-evidence-audit-manifest",
            "state": "ready",
            "primitive": primitive,
            "collision_verdict": collision_verdict,
            "environment_count": len(environment_rows),
            "work_order_count": len(work_orders),
            "report_path": str(destination / "source-effect-audit.json"),
        },
        archive_db=archive_db,
        cas_root=cas_root,
    )


def _verify_indexed_rows(
    *,
    labels: Sequence[Mapping[str, Any]],
    primitive: str,
    reports_root: Path,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for label in labels:
        if label.get("primitive") != primitive or label.get("settled") is not True:
            continue
        evidence_ref = label.get("evidence_ref")
        expected_sha = label.get("evidence_sha256")
        if not isinstance(evidence_ref, str) or not isinstance(expected_sha, str):
            raise SourceEffectAuditError("SOURCE_EFFECT_AUDIT_INDEX_PROVENANCE_MISSING")
        evidence_path = Path(evidence_ref).resolve()
        try:
            evidence_path.relative_to(reports_root)
        except ValueError as exc:
            raise SourceEffectAuditError("SOURCE_EFFECT_AUDIT_INDEX_PATH_ESCAPE") from exc
        if not evidence_path.is_file() or sha256_file(evidence_path) != expected_sha:
            raise SourceEffectAuditError(f"SOURCE_EFFECT_AUDIT_INDEX_SHA_MISMATCH:{evidence_path}")
        evidence = _load_json(evidence_path, "SOURCE_EFFECT_AUDIT_EVIDENCE_INVALID")
        for key in ("environment", "primitive", "seed"):
            if evidence.get(key) != label.get(key):
                raise SourceEffectAuditError(f"SOURCE_EFFECT_AUDIT_INDEX_FIELD_MISMATCH:{key}")
        gate = evidence.get("official_quality_gate")
        if not isinstance(gate, Mapping) or gate.get("pass") is not label.get("positive"):
            raise SourceEffectAuditError("SOURCE_EFFECT_AUDIT_INDEX_GATE_MISMATCH")
        delta = gate.get("delta_candidate_minus_baseline")
        rows.append(
            {
                "label_id": label.get("label_id"),
                "environment": label.get("environment"),
                "primitive": primitive,
                "seed": label.get("seed"),
                "training_seed": _training_seed(evidence),
                "eval_seed": evidence.get("eval_seed", evidence.get("seed")),
                "label_source": label.get("label_source"),
                "positive": label.get("positive"),
                "psnr_delta": delta.get("psnr") if isinstance(delta, Mapping) else None,
                "candidate_checkpoint_sha256": evidence.get("candidate_checkpoint_sha256"),
                "evidence_ref": str(evidence_path),
                "evidence_sha256": expected_sha,
            }
        )
    return rows


def _inventory_receipts(
    *,
    receipt_roots: Sequence[Path],
    primitive: str,
    environments: Sequence[str],
) -> list[dict[str, object]]:
    paths_by_receipt_id: dict[str, Path] = {}
    for receipt_root in receipt_roots:
        for pattern in _GATE_PATTERNS:
            for path in sorted(receipt_root.glob(pattern)):
                resolved = path.resolve()
                paths_by_receipt_id.setdefault(resolved.parent.name, resolved)
    rows: list[dict[str, object]] = []
    allowed = set(environments)
    for path in sorted(paths_by_receipt_id.values()):
        payload = _load_json(path, "SOURCE_EFFECT_AUDIT_RECEIPT_INVALID")
        gate = payload.get("official_quality_gate")
        environment = payload.get("environment")
        if (
            payload.get("state") != "ready"
            or payload.get("primitive") != primitive
            or environment not in allowed
            or not isinstance(gate, Mapping)
            or not isinstance(gate.get("pass"), bool)
        ):
            continue
        provenance = payload.get("protocol_provenance")
        provenance = provenance if isinstance(provenance, Mapping) else {}
        required_protocol = {
            "dataset_freeze_sha256": provenance.get("dataset_freeze_sha256"),
            "heldout_protocol_sha256": provenance.get("heldout_protocol_sha256"),
            "eval_script_sha256": provenance.get("eval_script_sha256"),
        }
        protocol_complete = all(isinstance(value, str) and value for value in required_protocol.values())
        delta = gate.get("delta_candidate_minus_baseline")
        checkpoint_sha = payload.get("candidate_checkpoint_sha256")
        rows.append(
            {
                "receipt_id": path.parent.name,
                "environment": environment,
                "primitive": primitive,
                "seed": payload.get("seed"),
                "training_seed": _training_seed(payload),
                "eval_seed": payload.get("eval_seed", payload.get("seed")),
                "candidate_checkpoint_sha256": checkpoint_sha,
                "checkpoint_step": _checkpoint_step(payload),
                "positive": gate.get("pass"),
                "psnr_delta": delta.get("psnr") if isinstance(delta, Mapping) else None,
                "ssim_delta": delta.get("ssim") if isinstance(delta, Mapping) else None,
                "mse_delta": delta.get("mse") if isinstance(delta, Mapping) else None,
                "masked_mse_delta": delta.get("masked_mse") if isinstance(delta, Mapping) else None,
                "inference_steps": payload.get("steps"),
                "split": payload.get("split"),
                "max_trajs": payload.get("max_trajs"),
                "dataset_freeze_sha256": required_protocol["dataset_freeze_sha256"],
                "heldout_protocol_sha256": required_protocol["heldout_protocol_sha256"],
                "eval_script_sha256": required_protocol["eval_script_sha256"],
                "eval_config_sha256": provenance.get("eval_config_sha256"),
                "protocol_complete": protocol_complete,
                "evidence_ref": str(path.resolve()),
                "evidence_sha256": sha256_file(path),
            }
        )
    return rows


def _environment_audit(
    *,
    environment: str,
    indexed: Sequence[Mapping[str, Any]],
    receipts: Sequence[Mapping[str, Any]],
    minimum_independent_seeds: int,
) -> dict[str, object]:
    complete = [row for row in receipts if row["protocol_complete"] is True]
    signs = sorted({"positive" if row["positive"] is True else "negative" for row in complete})
    seeds = sorted(
        {
            int(row["training_seed"])
            for row in complete
            if isinstance(row.get("training_seed"), int)
        }
    )
    checkpoints = sorted(
        {str(row["candidate_checkpoint_sha256"]) for row in complete if row.get("candidate_checkpoint_sha256")}
    )
    training_seed_effects = _training_seed_effects(complete)
    groups: dict[tuple[object, ...], list[Mapping[str, Any]]] = defaultdict(list)
    checkpoint_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in complete:
        key = (
            row.get("candidate_checkpoint_sha256"),
            row.get("eval_seed"),
            row.get("dataset_freeze_sha256"),
            row.get("heldout_protocol_sha256"),
            row.get("eval_script_sha256"),
            row.get("eval_config_sha256"),
            row.get("inference_steps"),
            row.get("split"),
            row.get("max_trajs"),
        )
        groups[key].append(row)
        if row.get("candidate_checkpoint_sha256"):
            checkpoint_groups[str(row["candidate_checkpoint_sha256"])].append(row)
    reproduction_conflicts = [
        sorted(str(row["receipt_id"]) for row in values)
        for values in groups.values()
        if len({bool(row["positive"]) for row in values}) > 1
    ]
    eval_seed_conflicts = [
        checkpoint
        for checkpoint, values in checkpoint_groups.items()
        if len({bool(row["positive"]) for row in values}) > 1
        and len({row.get("eval_seed") for row in values}) > 1
    ]
    if not receipts:
        classification = "no_official_gate_evidence"
        action = "no_action_no_current_evidence"
    elif not complete:
        classification = "protocol_incomplete_only"
        action = "rerun_frozen_official_gate"
    elif reproduction_conflicts:
        classification = "same_protocol_reproduction_conflict"
        action = "rerun_deterministic_official_gate_replication"
    elif eval_seed_conflicts:
        classification = "eval_seed_sensitive"
        action = "run_frozen_multi_eval_seed_replication"
    elif signs == ["positive"] and len(seeds) >= minimum_independent_seeds:
        classification = "stable_positive"
        action = "retain_settled_direction"
    elif signs == ["negative"] and len(seeds) >= minimum_independent_seeds:
        classification = "stable_negative"
        action = "retain_settled_direction"
    elif signs == ["positive"]:
        classification = "positive_underreplicated"
        action = "replicate_independent_training_seed_then_official_gate"
    elif signs == ["negative"]:
        classification = "negative_underreplicated"
        action = "replicate_independent_training_seed_then_official_gate"
    else:
        classification = "mixed_across_checkpoints"
        action = "stratify_checkpoint_and_training_seed_protocol"
    required_training = (
        max(0, minimum_independent_seeds - len(seeds))
        if classification in {"positive_underreplicated", "negative_underreplicated"}
        else 0
    )
    required_eval = 3 if classification in {"same_protocol_reproduction_conflict", "eval_seed_sensitive"} else 0
    return {
        "environment": environment,
        "indexed_label_count": len(indexed),
        "indexed_positive_count": sum(row["positive"] is True for row in indexed),
        "indexed_negative_count": sum(row["positive"] is False for row in indexed),
        "discovered_receipt_count": len(receipts),
        "protocol_complete_receipt_count": len(complete),
        "protocol_incomplete_receipt_count": len(receipts) - len(complete),
        "independent_training_seed_count": len(seeds),
        "training_seeds": seeds,
        "distinct_checkpoint_count": len(checkpoints),
        "training_seed_effects": training_seed_effects,
        "eval_stable_positive_training_seed_checkpoint_count": sum(
            row["classification"] == "eval_seed_stable_positive"
            for row in training_seed_effects
        ),
        "eval_stable_positive_training_seeds": sorted(
            {
                int(row["training_seed"])
                for row in training_seed_effects
                if row["classification"] == "eval_seed_stable_positive"
            }
        ),
        "sign_inconsistent_training_seeds": sorted(
            {
                int(row["training_seed"])
                for row in training_seed_effects
                if row["classification"] == "sign_inconsistent"
            }
        ),
        "observed_signs": signs,
        "mean_psnr_delta": fmean(float(row["psnr_delta"]) for row in complete if isinstance(row.get("psnr_delta"), (int, float))) if any(isinstance(row.get("psnr_delta"), (int, float)) for row in complete) else None,
        "same_protocol_reproduction_conflict_count": len(reproduction_conflicts),
        "same_protocol_reproduction_conflicts": reproduction_conflicts,
        "eval_seed_sensitive_checkpoint_count": len(eval_seed_conflicts),
        "eval_seed_sensitive_checkpoint_sha256": sorted(eval_seed_conflicts),
        "classification": classification,
        "recommended_action": action,
        "required_new_training_seed_count": required_training,
        "required_eval_seed_count": required_eval,
    }


def _training_seed_effects(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, object]]:
    grouped: dict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        training_seed = row.get("training_seed")
        checkpoint_sha = row.get("candidate_checkpoint_sha256")
        if (
            isinstance(training_seed, int)
            and not isinstance(training_seed, bool)
            and isinstance(checkpoint_sha, str)
            and checkpoint_sha
        ):
            grouped[(training_seed, checkpoint_sha)].append(row)
    effects: list[dict[str, object]] = []
    for (training_seed, checkpoint_sha), values in sorted(grouped.items()):
        signs = {bool(row["positive"]) for row in values}
        eval_seed_count = len({row.get("eval_seed") for row in values})
        if signs == {True}:
            classification = (
                "eval_seed_stable_positive"
                if eval_seed_count >= 3
                else "positive_eval_seed_underreplicated"
            )
        elif signs == {False}:
            classification = (
                "eval_seed_stable_negative"
                if eval_seed_count >= 3
                else "negative_eval_seed_underreplicated"
            )
        else:
            classification = "sign_inconsistent"
        effects.append(
            {
                "training_seed": training_seed,
                "candidate_checkpoint_sha256": checkpoint_sha,
                "receipt_count": len(values),
                "distinct_eval_seed_count": eval_seed_count,
                "eval_seeds": sorted({
                    int(row["eval_seed"])
                    for row in values
                    if isinstance(row.get("eval_seed"), int)
                    and not isinstance(row.get("eval_seed"), bool)
                }),
                "positive_count": sum(row["positive"] is True for row in values),
                "negative_count": sum(row["positive"] is False for row in values),
                "classification": classification,
            }
        )
    return effects


def _checkpoint_step(payload: Mapping[str, Any]) -> int | None:
    value = payload.get("candidate_checkpoint_step")
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    checkpoint = payload.get("candidate_checkpoint")
    if not isinstance(checkpoint, str):
        return None
    stem = Path(checkpoint).stem
    for prefix in ("relative_step_", "step_", "checkpoint_"):
        if stem.startswith(prefix) and stem[len(prefix) :].isdigit():
            return int(stem[len(prefix) :])
    return None


def _training_seed(payload: Mapping[str, Any]) -> int | None:
    value = payload.get("training_seed")
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    checkpoint = payload.get("candidate_checkpoint")
    if isinstance(checkpoint, str):
        match = re.search(r"(?:^|[-_/])s(\d+)(?:[-_/]|$)", checkpoint)
        if match:
            return int(match.group(1))
    return None


def _load_json(path: Path, code: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SourceEffectAuditError(f"{code}:{path}") from exc
    if not isinstance(payload, Mapping):
        raise SourceEffectAuditError(f"{code}:{path}")
    return payload


def _csv(rows: Sequence[Mapping[str, object]]) -> str:
    if not rows:
        return "\n"
    fields = sorted({key for row in rows for key in row})
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                key: json.dumps(value, sort_keys=True) if isinstance(value, (list, dict)) else value
                for key, value in row.items()
            }
        )
    return output.getvalue()


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        f"# Source-Effect Audit: `{report['primitive']}`",
        "",
        str(report["claim_boundary"]),
        "",
        f"Collision verdict: `{report['collision_verdict']}`",
        "",
        "| Environment | Indexed labels | Complete receipts | Seeds | Stable + seeds | Signs | Classification | Next action |",
        "|---|---:|---:|---:|---|---|---|---|",
    ]
    for row in report["environment_audits"]:
        lines.append(
            "| {environment} | {indexed_label_count} | {protocol_complete_receipt_count} | "
            "{independent_training_seed_count} | {stable_positive_seeds} | {signs} | `{classification}` | `{recommended_action}` |".format(
                **row,
                signs=", ".join(row["observed_signs"]) or "none",
                stable_positive_seeds=", ".join(
                    str(seed) for seed in row["eval_stable_positive_training_seeds"]
                ) or "none",
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "Stable negative evidence is retained. Positive or inconsistent sources must be repaired "
            "with the listed frozen-protocol work orders before a new collision-disambiguating probe "
            "is admitted. The transfer-certificate thresholds remain unchanged.",
            "",
        ]
    )
    return "\n".join(lines)
