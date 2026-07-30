"""Evaluate evolved probe collision alerts against frozen target-local labels."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from wmloop.experiments._artifacts import canonical_json, write_bundle


class CollisionLabelError(ValueError):
    """Collision labels or replay evidence are malformed or incomparable."""


def evaluate_collision_labels(
    *,
    config_path: Path,
    output_root: Path,
    archive_db: Path | None = None,
    cas_root: Path | None = None,
) -> dict[str, object]:
    config_file = Path(config_path).resolve(strict=True)
    config = _load_mapping(config_file)
    if config.get("artifact_type") != "verdiwm-collision-label-preregistration":
        raise CollisionLabelError("COLLISION_LABEL_CONFIG_TYPE_INVALID")
    if config.get("study_id") != "S4_probe_information_and_collision":
        raise CollisionLabelError("COLLISION_LABEL_STUDY_ID_INVALID")
    if config.get("selector") != "irg":
        raise CollisionLabelError("COLLISION_LABEL_SELECTOR_INVALID")
    if config.get("evolved_replay_selection_rule") != "latest_completed_iteration_not_best_metric":
        raise CollisionLabelError("COLLISION_LABEL_SELECTION_RULE_INVALID")

    repo_root = config_file.parents[2]
    fixed_root = _resolve_source(config.get("fixed_replay_root"), repo_root)
    evolved_root = _resolve_source(config.get("evolved_replay_root"), repo_root)
    labels_path = _resolve_source(config.get("effect_label_index_path"), repo_root)
    fixed_top1 = _top1(fixed_root / "tables" / "candidates.csv", "irg")
    evolved_top1 = _top1(evolved_root / "tables" / "candidates.csv", "irg")
    evolved_states = _fold_states(evolved_root / "tables" / "cells.csv", "irg")
    truth = _effect_truth(labels_path)

    cases: list[dict[str, object]] = []
    for target, fixed in sorted(fixed_top1.items()):
        fixed_key = (target, str(fixed["primitive"]))
        if fixed_key not in truth:
            raise CollisionLabelError(f"COLLISION_LABEL_TARGET_TRUTH_MISSING:{target}:{fixed['primitive']}")
        target_positive = truth[fixed_key]
        ground_truth_collision = (str(fixed["predicted_sign"]) == "positive") is not target_positive
        evolved = evolved_top1.get(target)
        state = evolved_states.get(target)
        evolved_state = str(state["state"]) if state is not None else "missing"
        evolved_abstention_reason = state.get("abstention_reason") if state is not None else "fold_missing"
        changed_signature = bool(
            evolved is not None
            and (
                evolved["primitive"] != fixed["primitive"]
                or evolved["predicted_sign"] != fixed["predicted_sign"]
            )
        )
        predicted_collision = evolved_state != "evaluated" or changed_signature
        evolved_target_positive: bool | None = None
        evolved_correct: bool | None = None
        if evolved is not None:
            evolved_key = (target, str(evolved["primitive"]))
            if evolved_key not in truth:
                raise CollisionLabelError(
                    f"COLLISION_LABEL_EVOLVED_TRUTH_MISSING:{target}:{evolved['primitive']}"
                )
            evolved_target_positive = truth[evolved_key]
            evolved_correct = (str(evolved["predicted_sign"]) == "positive") is evolved_target_positive
        cases.append(
            {
                "target_environment": target,
                "fixed_primitive": fixed["primitive"],
                "fixed_predicted_sign": fixed["predicted_sign"],
                "fixed_target_positive": target_positive,
                "ground_truth_collision": ground_truth_collision,
                "ground_truth_source": "settled_target_local_official_gate_sign",
                "evolved_state": evolved_state,
                "evolved_abstention_reason": evolved_abstention_reason,
                "evolved_primitive": evolved.get("primitive") if evolved else None,
                "evolved_predicted_sign": evolved.get("predicted_sign") if evolved else None,
                "evolved_target_positive": evolved_target_positive,
                "evolved_correct_before_certificate": evolved_correct,
                "evolved_signature_changed": changed_signature,
                "predicted_collision": predicted_collision,
            }
        )

    positive_count = sum(bool(row["ground_truth_collision"]) for row in cases)
    negative_count = len(cases) - positive_count
    if not cases or positive_count == 0 or negative_count == 0:
        raise CollisionLabelError("COLLISION_LABEL_CLASS_COVERAGE_INVALID")
    confusion = _confusion(cases)
    precision = _ratio(confusion["true_positive"], confusion["true_positive"] + confusion["false_positive"])
    recall = _ratio(confusion["true_positive"], confusion["true_positive"] + confusion["false_negative"])
    collision_f1 = (
        None if precision is None or recall is None or precision + recall == 0.0
        else 2.0 * precision * recall / (precision + recall)
    )
    accepted = [row for row in cases if row["evolved_state"] == "evaluated"]
    post_evolution_collision_rate = (
        None
        if not accepted
        else sum(row["evolved_correct_before_certificate"] is False for row in accepted) / len(accepted)
    )
    comparable = [row for row in cases if row["evolved_correct_before_certificate"] is not None]
    pre_certificate_collision_rate = (
        None
        if not comparable
        else sum(row["evolved_correct_before_certificate"] is False for row in comparable) / len(comparable)
    )
    corrected = [
        str(row["target_environment"])
        for row in comparable
        if row["ground_truth_collision"] and row["evolved_correct_before_certificate"] is True
    ]
    regressed = [
        str(row["target_environment"])
        for row in comparable
        if not row["ground_truth_collision"] and row["evolved_correct_before_certificate"] is False
    ]
    report = {
        "schema_version": 1,
        "artifact_type": "verdiwm-collision-label-evaluation",
        "study_id": config["study_id"],
        "state": "ready",
        "label_contract": (
            "A fixed-minimal top-1 case is a collision when its predicted benefit sign disagrees with "
            "the frozen target-local settled official-gate sign. Labels are frozen before and independent "
            "of the evolved replay being evaluated."
        ),
        "alert_contract": (
            "The evolved system alerts on a fixed-region collision when it abstains or changes the top-1 "
            "primitive/sign signature."
        ),
        "evolved_replay_selection_rule": config["evolved_replay_selection_rule"],
        "case_count": len(cases),
        "positive_collision_count": positive_count,
        "negative_collision_count": negative_count,
        "confusion": confusion,
        "collision_detection_precision": precision,
        "collision_detection_recall": recall,
        "collision_detection_f1": collision_f1,
        "accepted_case_count": len(accepted),
        "accepted_coverage": len(accepted) / len(cases),
        "post_evolution_collision_rate": post_evolution_collision_rate,
        "pre_certificate_comparable_case_count": len(comparable),
        "pre_certificate_collision_rate": pre_certificate_collision_rate,
        "corrected_targets_before_certificate": corrected,
        "regressed_targets_before_certificate": regressed,
        "cases": cases,
        "source_refs": [
            _source_ref("preregistration", config_file),
            _source_ref("fixed_candidates", fixed_root / "tables" / "candidates.csv"),
            _source_ref("evolved_candidates", evolved_root / "tables" / "candidates.csv"),
            _source_ref("evolved_cells", evolved_root / "tables" / "cells.csv"),
            _source_ref("effect_label_index", labels_path),
        ],
        "claim_boundary": (
            "This evaluates collision alerting on historical frozen selector evidence. A null post-evolution "
            "collision rate means the evolved selector accepted no folds; it must not be reported as zero risk "
            "or as a repair-quality gain."
        ),
    }
    destination = Path(output_root).resolve()
    return write_bundle(
        output_root=destination,
        files={
            "collision-label-evaluation.json": canonical_json(report),
            "collision-label-evaluation.md": _markdown(report).encode("utf-8"),
            "tables/collision-cases.csv": _csv(cases).encode("utf-8"),
            "input-preregistration.json": canonical_json(config),
        },
        manifest_fields={
            "artifact_type": "verdiwm-collision-label-evaluation-manifest",
            "state": "ready",
            "study_id": config["study_id"],
            "case_count": len(cases),
            "positive_collision_count": positive_count,
            "negative_collision_count": negative_count,
            "collision_detection_f1": collision_f1,
            "accepted_case_count": len(accepted),
            "post_evolution_collision_rate": post_evolution_collision_rate,
            "report_path": str(destination / "collision-label-evaluation.json"),
        },
        archive_db=archive_db,
        cas_root=cas_root,
    )


def _top1(path: Path, selector: str) -> dict[str, dict[str, object]]:
    grouped: dict[str, set[tuple[str, str]]] = {}
    for row in _csv_rows(path):
        if row.get("selector") != selector or row.get("rank") != "1":
            continue
        target = str(row.get("target_environment") or "")
        primitive = str(row.get("primitive") or "")
        predicted = str(row.get("predicted_sign") or "")
        if not target or not primitive or predicted not in {"positive", "negative"}:
            continue
        grouped.setdefault(target, set()).add((primitive, predicted))
    result: dict[str, dict[str, object]] = {}
    for target, values in grouped.items():
        if len(values) != 1:
            raise CollisionLabelError(f"COLLISION_LABEL_TOP1_SEED_DISAGREEMENT:{target}")
        primitive, predicted = next(iter(values))
        result[target] = {"primitive": primitive, "predicted_sign": predicted}
    if not result:
        raise CollisionLabelError(f"COLLISION_LABEL_TOP1_MISSING:{path}")
    return result


def _fold_states(path: Path, selector: str) -> dict[str, dict[str, object]]:
    grouped: dict[str, set[tuple[str, str | None]]] = {}
    for row in _csv_rows(path):
        if row.get("selector") != selector:
            continue
        target = str(row.get("target_environment") or "")
        state = str(row.get("state") or "")
        reason = str(row.get("abstention_reason") or "") or None
        if target and state in {"evaluated", "abstained"}:
            grouped.setdefault(target, set()).add((state, reason))
    result: dict[str, dict[str, object]] = {}
    for target, values in grouped.items():
        if len(values) != 1:
            raise CollisionLabelError(f"COLLISION_LABEL_STATE_SEED_DISAGREEMENT:{target}")
        state, reason = next(iter(values))
        result[target] = {"state": state, "abstention_reason": reason}
    return result


def _effect_truth(path: Path) -> dict[tuple[str, str], bool]:
    payload = _load_mapping(path)
    labels = payload.get("labels")
    if not isinstance(labels, list):
        raise CollisionLabelError("COLLISION_LABEL_EFFECT_INDEX_INVALID")
    grouped: dict[tuple[str, str], set[bool]] = {}
    for row in labels:
        if not isinstance(row, Mapping):
            raise CollisionLabelError("COLLISION_LABEL_EFFECT_INDEX_INVALID")
        positive = row.get("positive")
        if (
            not bool(row.get("settled"))
            or row.get("selector_admissible") is False
            or not isinstance(positive, bool)
        ):
            continue
        key = (str(row.get("environment") or ""), str(row.get("primitive") or ""))
        if not all(key):
            raise CollisionLabelError("COLLISION_LABEL_EFFECT_INDEX_INVALID")
        grouped.setdefault(key, set()).add(positive)
    return {key: next(iter(signs)) for key, signs in grouped.items() if len(signs) == 1}


def _confusion(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        "true_positive": sum(bool(row["ground_truth_collision"]) and bool(row["predicted_collision"]) for row in rows),
        "false_positive": sum(not bool(row["ground_truth_collision"]) and bool(row["predicted_collision"]) for row in rows),
        "true_negative": sum(not bool(row["ground_truth_collision"]) and not bool(row["predicted_collision"]) for row in rows),
        "false_negative": sum(bool(row["ground_truth_collision"]) and not bool(row["predicted_collision"]) for row in rows),
    }


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _resolve_source(value: object, root: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise CollisionLabelError("COLLISION_LABEL_SOURCE_INVALID")
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve(strict=True)


def _csv_rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except OSError as exc:
        raise CollisionLabelError(f"COLLISION_LABEL_CSV_INVALID:{path}") from exc


def _load_mapping(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CollisionLabelError(f"COLLISION_LABEL_JSON_INVALID:{path}") from exc
    if not isinstance(payload, Mapping):
        raise CollisionLabelError(f"COLLISION_LABEL_JSON_INVALID:{path}")
    return payload


def _source_ref(role: str, path: Path) -> dict[str, object]:
    return {"role": role, "path": str(path), "sha256": _sha256(path), "size_bytes": path.stat().st_size}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _csv(rows: Sequence[Mapping[str, Any]]) -> str:
    fields = sorted({key for row in rows for key in row})
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _markdown(report: Mapping[str, Any]) -> str:
    post_rate = report["post_evolution_collision_rate"]
    post_display = "NA (zero accepted coverage)" if post_rate is None else f"{float(post_rate):.6f}"
    lines = [
        "# S4 Collision Label Evaluation",
        "",
        f"State: `{report['state']}`",
        "",
        f"Cases: `{report['case_count']}` ({report['positive_collision_count']} collision, {report['negative_collision_count']} non-collision)",
        f"Collision detection F1: `{float(report['collision_detection_f1']):.6f}`",
        f"Accepted coverage: `{float(report['accepted_coverage']):.6f}`",
        f"Post-evolution collision rate: `{post_display}`",
        f"Pre-certificate comparable collision rate: `{float(report['pre_certificate_collision_rate']):.6f}`",
        "",
        f"Corrected before certificate: `{', '.join(report['corrected_targets_before_certificate']) or 'none'}`",
        f"Regressed before certificate: `{', '.join(report['regressed_targets_before_certificate']) or 'none'}`",
        "",
        str(report["claim_boundary"]),
        "",
    ]
    return "\n".join(lines)
