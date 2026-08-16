#!/usr/bin/env python3
"""Freeze a Ctrl-World directional selector before wider effect labels exist."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


class DirectionalSelectorFreezeError(ValueError):
    """A narrow atlas cannot produce a preregistered selector."""


def _load(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise DirectionalSelectorFreezeError("DIRECTIONAL_SELECTOR_FREEZE_INPUT_INVALID")
    return payload


def _finite(value: object, code: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise DirectionalSelectorFreezeError(code)
    return float(value)


def freeze_selector(
    *,
    atlas_path: Path,
    candidate_radius: float,
    confidence_z: float,
    minimum_gain_lcb: float,
    output_path: Path,
) -> dict[str, object]:
    source = atlas_path.resolve(strict=True)
    atlas = _load(source)
    if (
        atlas.get("artifact_type") != "verdiwm-ctrl-world-context-local-fingerprint-atlas"
        or atlas.get("state") != "ready"
        or int(atlas.get("context_count", 0)) < 1
    ):
        raise DirectionalSelectorFreezeError("DIRECTIONAL_SELECTOR_FREEZE_ATLAS_INVALID")
    candidate_radius = _finite(candidate_radius, "DIRECTIONAL_SELECTOR_FREEZE_RADIUS_INVALID")
    confidence_z = _finite(confidence_z, "DIRECTIONAL_SELECTOR_FREEZE_CONFIDENCE_INVALID")
    minimum_gain_lcb = _finite(minimum_gain_lcb, "DIRECTIONAL_SELECTOR_FREEZE_THRESHOLD_INVALID")
    if candidate_radius <= 0.0 or confidence_z < 0.0 or output_path.exists() or output_path.is_symlink():
        raise DirectionalSelectorFreezeError("DIRECTIONAL_SELECTOR_FREEZE_ARGUMENT_INVALID")
    outcome_names = tuple(str(value) for value in atlas.get("outcome_names", ()))
    weights = tuple(_finite(value, "DIRECTIONAL_SELECTOR_FREEZE_WEIGHT_INVALID") for value in atlas.get("outcome_weights", ()))
    if not outcome_names or len(weights) != len(outcome_names):
        raise DirectionalSelectorFreezeError("DIRECTIONAL_SELECTOR_FREEZE_OUTCOME_FRAME_INVALID")
    contexts: list[dict[str, object]] = []
    for row in atlas.get("contexts", ()):
        context = row["context"]
        chart = row["chart"]
        locality = row["locality_admission"]
        names = tuple(str(value) for value in chart["intervention_names"])
        jacobian = chart["jacobian"]
        covariance = chart["covariance"]
        repeat_count = int(chart["repeat_count"])
        width = len(outcome_names) * len(names)
        candidates: list[dict[str, object]] = []
        for probe_id in sorted(str(value) for value in locality.get("supported_local_paths", ())):
            path_index = names.index(probe_id)
            coefficients = [0.0] * width
            for outcome_index, weight in enumerate(weights):
                coefficients[outcome_index * len(names) + path_index] = math.sqrt(weight)
            variance = sum(
                coefficients[i] * float(covariance[i][j]) * coefficients[j]
                for i in range(width)
                for j in range(width)
            )
            slope_se = math.sqrt(max(variance, 0.0) / repeat_count)
            slope = sum(weights[index] * float(jacobian[index][path_index]) for index in range(len(weights)))
            for dose in (-candidate_radius, candidate_radius):
                predicted = dose * slope
                standard_error = abs(dose) * slope_se
                lcb = predicted - confidence_z * standard_error
                candidates.append(
                    {
                        "candidate_id": f"{probe_id}:{dose:+g}",
                        "probe_id": probe_id,
                        "dose": dose,
                        "predicted_weighted_gain": predicted,
                        "predicted_weighted_gain_standard_error": standard_error,
                        "predicted_weighted_gain_lcb": lcb,
                    }
                )
        ranked = sorted(
            candidates,
            key=lambda item: (
                -float(item["predicted_weighted_gain_lcb"]),
                -float(item["predicted_weighted_gain"]),
                str(item["candidate_id"]),
            ),
        )
        selected = ranked[0] if ranked and float(ranked[0]["predicted_weighted_gain_lcb"]) > minimum_gain_lcb else None
        contexts.append(
            {
                "context": context,
                "supported_local_paths": sorted(str(value) for value in locality.get("supported_local_paths", ())),
                "selector_action": "execute" if selected is not None else "abstain",
                "selected_candidate": selected,
                "ranked_candidates": ranked,
            }
        )
    document = {
        "schema_version": 1,
        "artifact_type": "verdiwm-ctrl-world-frozen-directional-selector",
        "state": "frozen",
        "fingerprint_campaign_id": atlas.get("campaign_id"),
        "fingerprint_atlas": {"path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()},
        "candidate_radius": candidate_radius,
        "confidence_z": confidence_z,
        "minimum_predicted_gain_lcb": minimum_gain_lcb,
        "abstain_when_no_candidate_passes": True,
        "contexts": contexts,
        "effect_labels_observed": False,
        "claim_boundary": "Frozen selector order and abstention only; no wider-dose effect label was read during construction.",
    }
    output_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    output_path.write_text(json.dumps(document, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "artifact_type": "verdiwm-ctrl-world-frozen-directional-selector-manifest",
        "state": "frozen",
        "selector_path": str(output_path.resolve()),
        "selector_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "context_count": len(contexts),
        "execute_count": sum(row["selector_action"] == "execute" for row in contexts),
        "abstain_count": sum(row["selector_action"] == "abstain" for row in contexts),
    }
    (output_path.parent / "selector-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--candidate-radius", type=float, required=True)
    parser.add_argument("--confidence-z", type=float, required=True)
    parser.add_argument("--minimum-gain-lcb", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = freeze_selector(
        atlas_path=args.atlas,
        candidate_radius=args.candidate_radius,
        confidence_z=args.confidence_z,
        minimum_gain_lcb=args.minimum_gain_lcb,
        output_path=args.output.resolve(),
    )
    print(json.dumps(manifest, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
