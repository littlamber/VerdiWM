"""Admit a candidate diagnostic probe path from held-out selector replay evidence."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from wmloop.experiments._artifacts import canonical_json, write_bundle


class SelectorProbeAdmissionError(ValueError):
    """Selector-probe admission inputs are malformed or incomparable."""


_RISK_COUNTS = (
    "transfer_certificate_abstention_count",
    "transfer_work_order_count",
    "probe_evolution_work_order_count",
)


def evaluate_selector_probe_admission(
    *,
    baseline_replay_root: Path,
    candidate_replay_root: Path,
    candidate_atlas_manifest: Path,
    candidate_affinity: Path,
    output_root: Path,
    archive_db: Path | None = None,
    cas_root: Path | None = None,
) -> dict[str, object]:
    baseline_root = Path(baseline_replay_root).resolve()
    candidate_root = Path(candidate_replay_root).resolve()
    baseline = _load_json(baseline_root / "manifest.json")
    candidate = _load_json(candidate_root / "manifest.json")
    atlas = _load_json(Path(candidate_atlas_manifest).resolve())
    affinity = _load_json(Path(candidate_affinity).resolve())
    if baseline.get("artifact_type") != "verdiwm-acwm-selector-cpu-replay-manifest":
        raise SelectorProbeAdmissionError("SELECTOR_PROBE_BASELINE_MANIFEST_INVALID")
    if candidate.get("artifact_type") != "verdiwm-acwm-selector-cpu-replay-manifest":
        raise SelectorProbeAdmissionError("SELECTOR_PROBE_CANDIDATE_MANIFEST_INVALID")
    if atlas.get("artifact_type") != "verdiwm-acwm-fingerprint-atlas-manifest":
        raise SelectorProbeAdmissionError("SELECTOR_PROBE_ATLAS_MANIFEST_INVALID")
    if affinity.get("artifact_type") != "verdiwm-primitive-probe-affinity":
        raise SelectorProbeAdmissionError("SELECTOR_PROBE_AFFINITY_INVALID")

    baseline_top1 = _irg_top1(baseline_root / "tables" / "candidates.csv")
    candidate_top1 = _irg_top1(candidate_root / "tables" / "candidates.csv")
    common_targets = sorted(set(baseline_top1) & set(candidate_top1))
    if not common_targets:
        raise SelectorProbeAdmissionError("SELECTOR_PROBE_COMPARABLE_TARGETS_MISSING")
    corrected = [
        target
        for target in common_targets
        if not baseline_top1[target]["correct"] and candidate_top1[target]["correct"]
    ]
    regressed = [
        target
        for target in common_targets
        if baseline_top1[target]["correct"] and not candidate_top1[target]["correct"]
    ]
    risk_checks = {
        name: int(candidate[name]) <= int(baseline[name])
        for name in _RISK_COUNTS
    }
    atlas_checks = {
        "eight_environment_measurements_complete": int(atlas.get("measurement_complete_count", 0)) == 8,
        "at_least_two_locality_calibrated_environments": int(atlas.get("locality_calibrated_count", 0)) >= 2,
    }
    admission_checks = {
        **atlas_checks,
        **{f"nonincreasing_{name}": passed for name, passed in risk_checks.items()},
        "heldout_top1_correction_present": bool(corrected),
        "heldout_top1_regression_absent": not regressed,
    }
    admitted = all(admission_checks.values())
    report = {
        "schema_version": 1,
        "artifact_type": "verdiwm-selector-probe-admission",
        "state": "ready",
        "admission_state": "admitted" if admitted else "rejected",
        "promote_affinity_allowed": admitted,
        "baseline_replay_root": str(baseline_root),
        "candidate_replay_root": str(candidate_root),
        "candidate_atlas_manifest": str(Path(candidate_atlas_manifest).resolve()),
        "candidate_affinity": str(Path(candidate_affinity).resolve()),
        "candidate_affinity_contract_id": affinity.get("contract_id"),
        "common_heldout_target_count": len(common_targets),
        "corrected_targets": corrected,
        "regressed_targets": regressed,
        "baseline_top1": baseline_top1,
        "candidate_top1": candidate_top1,
        "risk_counts": {
            name: {"baseline": int(baseline[name]), "candidate": int(candidate[name])}
            for name in _RISK_COUNTS
        },
        "atlas_counts": {
            "measurement_complete_count": int(atlas.get("measurement_complete_count", 0)),
            "locality_calibrated_count": int(atlas.get("locality_calibrated_count", 0)),
        },
        "admission_checks": admission_checks,
        "claim_boundary": "Admission authorizes a diagnostic selector-path update only. It is not a primitive quality gain, transfer certificate, or permission to skip the 512-step and official-gate chain.",
    }
    destination = Path(output_root).resolve()
    return write_bundle(
        output_root=destination,
        files={
            "selector-probe-admission.json": canonical_json(report),
            "selector-probe-admission.md": _markdown(report).encode("utf-8"),
        },
        manifest_fields={
            "artifact_type": "verdiwm-selector-probe-admission-manifest",
            "state": "ready",
            "admission_state": report["admission_state"],
            "promote_affinity_allowed": admitted,
            "corrected_target_count": len(corrected),
            "regressed_target_count": len(regressed),
            "report_path": str(destination / "selector-probe-admission.json"),
        },
        archive_db=archive_db,
        cas_root=cas_root,
    )


def _irg_top1(path: Path) -> dict[str, dict[str, object]]:
    grouped: dict[str, set[tuple[str, str, bool]]] = {}
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("selector") != "irg" or row.get("rank") != "1":
                    continue
                target = str(row.get("target_environment") or "")
                predicted_sign = str(row.get("predicted_sign") or "")
                target_positive = _parse_bool(row.get("target_positive"))
                primitive = str(row.get("primitive") or "")
                if not target or predicted_sign not in {"positive", "negative"} or target_positive is None:
                    continue
                grouped.setdefault(target, set()).add((primitive, predicted_sign, target_positive))
    except OSError as exc:
        raise SelectorProbeAdmissionError(f"SELECTOR_PROBE_CANDIDATES_INVALID:{path}") from exc
    result: dict[str, dict[str, object]] = {}
    for target, outcomes in grouped.items():
        if len(outcomes) != 1:
            raise SelectorProbeAdmissionError(f"SELECTOR_PROBE_SEED_DISAGREEMENT:{target}")
        primitive, predicted_sign, target_positive = next(iter(outcomes))
        result[target] = {
            "primitive": primitive,
            "predicted_sign": predicted_sign,
            "target_positive": target_positive,
            "correct": (predicted_sign == "positive") is target_positive,
        }
    return result


def _parse_bool(value: object) -> bool | None:
    if value == "True":
        return True
    if value == "False":
        return False
    return None


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SelectorProbeAdmissionError(f"SELECTOR_PROBE_JSON_INVALID:{path}") from exc
    if not isinstance(payload, Mapping):
        raise SelectorProbeAdmissionError(f"SELECTOR_PROBE_JSON_INVALID:{path}")
    return payload


def _markdown(report: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# Selector Probe Admission",
            "",
            f"Admission: `{report['admission_state']}`",
            f"Corrected targets: `{', '.join(report['corrected_targets']) or 'none'}`",
            f"Regressed targets: `{', '.join(report['regressed_targets']) or 'none'}`",
            f"Locality calibrated: `{report['atlas_counts']['locality_calibrated_count']}/8`",
            "",
            str(report["claim_boundary"]),
            "",
        ]
    )
