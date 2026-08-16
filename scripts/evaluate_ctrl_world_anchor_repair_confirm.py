#!/usr/bin/env python3
"""Evaluate a frozen Ctrl-World anchor-repair choice on confirm contexts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from scripts.settle_ctrl_world_anchor_repair_screen import (
        AnchorRepairSettlementError,
        _load,
        _mean_standard_error,
        _parse_result,
    )
except ModuleNotFoundError:
    from settle_ctrl_world_anchor_repair_screen import (  # type: ignore[no-redef]
        AnchorRepairSettlementError,
        _load,
        _mean_standard_error,
        _parse_result,
    )


class AnchorRepairConfirmError(AnchorRepairSettlementError):
    """Confirm data do not match the frozen repair selector."""


def evaluate(
    *,
    selector_path: Path,
    route_selector_path: Path,
    result_path: Path,
    output_root: Path,
) -> dict[str, object]:
    selector_file = selector_path.resolve(strict=True)
    route_selector_file = route_selector_path.resolve(strict=True)
    result_file = result_path.resolve(strict=True)
    target = output_root.resolve()
    if target.exists() or target.is_symlink():
        raise AnchorRepairConfirmError("CTRL_WORLD_REPAIR_CONFIRM_OUTPUT_EXISTS")
    selector = _load(selector_file)
    if (
        selector.get("artifact_type") != "verdiwm-ctrl-world-frozen-anchor-repair-selector"
        or selector.get("state") != "frozen"
    ):
        raise AnchorRepairConfirmError("CTRL_WORLD_REPAIR_CONFIRM_SELECTOR_INVALID")
    if selector.get("selector_action") != "execute" or not isinstance(selector.get("selected_candidate"), Mapping):
        raise AnchorRepairConfirmError("CTRL_WORLD_REPAIR_CONFIRM_SELECTOR_ABSTAINS")
    selected = selector["selected_candidate"]
    route_selector = _load(route_selector_file)
    if (
        route_selector.get("artifact_type") != "verdiwm-ctrl-world-frozen-directional-selector"
        or route_selector.get("state") != "frozen"
        or route_selector.get("effect_labels_observed") is not False
    ):
        raise AnchorRepairConfirmError("CTRL_WORLD_REPAIR_CONFIRM_ROUTE_SELECTOR_INVALID")
    negative_routes: dict[str, Mapping[str, Any]] = {}
    for route in route_selector.get("contexts", ()):
        if not isinstance(route, Mapping) or not isinstance(route.get("context"), Mapping):
            raise AnchorRepairConfirmError("CTRL_WORLD_REPAIR_CONFIRM_ROUTE_SELECTOR_INVALID")
        context_id = route["context"].get("context_id")
        candidate = route.get("selected_candidate")
        if (
            route.get("selector_action") == "execute"
            and isinstance(candidate, Mapping)
            and candidate.get("probe_id") == "first_frame_anchor_blend"
            and math.isclose(float(candidate.get("dose", 0.0)), -0.025, abs_tol=1e-12)
        ):
            if not isinstance(context_id, str) or not context_id or context_id in negative_routes:
                raise AnchorRepairConfirmError("CTRL_WORLD_REPAIR_CONFIRM_ROUTE_SELECTOR_INVALID")
            negative_routes[context_id] = route
    if not negative_routes:
        raise AnchorRepairConfirmError("CTRL_WORLD_REPAIR_CONFIRM_ROUTE_SELECTOR_ABSTAINS")
    primitive_id, outcome_names, rows, payload = _parse_result(result_file)
    selected_primitive = selected.get("primitive_id")
    selected_strength = selected.get("strength")
    if primitive_id != selected_primitive or not isinstance(selected_strength, (int, float)):
        raise AnchorRepairConfirmError("CTRL_WORLD_REPAIR_CONFIRM_CANDIDATE_MISMATCH")
    strength = float(selected_strength)
    if strength not in rows or set(rows) != {0.0, strength}:
        raise AnchorRepairConfirmError("CTRL_WORLD_REPAIR_CONFIRM_STRENGTH_MISMATCH")
    weights = tuple(float(value) for value in selector.get("outcome_weights", ()))
    if tuple(selector.get("outcome_names", ())) != outcome_names or len(weights) != len(outcome_names):
        raise AnchorRepairConfirmError("CTRL_WORLD_REPAIR_CONFIRM_OUTCOME_FRAME_MISMATCH")
    identities = sorted(rows[0.0])
    result_context_ids = {identity[0] for identity in identities}
    if result_context_ids != set(negative_routes):
        raise AnchorRepairConfirmError("CTRL_WORLD_REPAIR_CONFIRM_ROUTE_CONTEXT_MISMATCH")
    gains = []
    per_identity = []
    for identity in identities:
        gain = sum(
            weight * (after - before)
            for weight, after, before in zip(weights, rows[strength][identity], rows[0.0][identity], strict=True)
        )
        gains.append(gain)
        per_identity.append(
            {
                "identity": {
                    "context_id": identity[0],
                    "episode_id": identity[1],
                    "start_idx": identity[2],
                    "seed": identity[3],
                },
                "weighted_gain": gain,
            }
        )
    mean, standard_error = _mean_standard_error(gains)
    confidence_z = float(selector.get("confidence_z"))
    gain_lcb = mean - confidence_z * standard_error
    context_ids = sorted({identity[0] for identity in identities})
    per_context = []
    for context_id in context_ids:
        values = [
            float(row["weighted_gain"])
            for row in per_identity
            if row["identity"]["context_id"] == context_id  # type: ignore[index]
        ]
        context_mean, context_se = _mean_standard_error(values)
        per_context.append(
            {
                "context_id": context_id,
                "mean_weighted_gain": context_mean,
                "standard_error": context_se,
                "gain_lcb": context_mean - confidence_z * context_se,
                "positive": context_mean > 0.0,
            }
        )
    passed = gain_lcb > 0.0 and all(float(row["mean_weighted_gain"]) > 0.0 for row in per_context)
    report = {
        "schema_version": 1,
        "artifact_type": "verdiwm-ctrl-world-anchor-repair-confirmation",
        "state": "ready",
        "decision": "passed_independent_context_confirm" if passed else "failed_confirm",
        "frozen_selector": {"path": str(selector_file), "sha256": hashlib.sha256(selector_file.read_bytes()).hexdigest()},
        "frozen_route_selector": {
            "path": str(route_selector_file),
            "sha256": hashlib.sha256(route_selector_file.read_bytes()).hexdigest(),
            "negative_route_context_ids": sorted(negative_routes),
        },
        "confirm_result": {"path": str(result_file), "sha256": hashlib.sha256(result_file.read_bytes()).hexdigest()},
        "selected_candidate_id": selected.get("candidate_id"),
        "primitive_id": primitive_id,
        "primitive_type": payload.get("primitive_type"),
        "strength": strength,
        "outcome_names": list(outcome_names),
        "outcome_weights": list(weights),
        "mean_weighted_gain": mean,
        "standard_error": standard_error,
        "gain_lcb": gain_lcb,
        "positive_context_fraction": sum(row["positive"] for row in per_context) / len(per_context),
        "harmful_context_count": sum(not row["positive"] for row in per_context),
        "per_context": per_context,
        "per_identity": per_identity,
        "claim_boundary": (
            "Independent exact-context/seed confirmation of a frozen inference-only repair. "
            "Episodes may have appeared at other replay starts in earlier diagnostic work; this "
            "does not establish training benefit, task success, cross-backbone transfer, or RSI."
        ),
    }
    target.mkdir(mode=0o700, parents=True)
    report_path = target / "confirmation.json"
    report_path.write_text(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "artifact_type": "verdiwm-ctrl-world-anchor-repair-confirmation-manifest",
        "state": "ready",
        "decision": report["decision"],
        "confirmation": {"path": str(report_path), "sha256": hashlib.sha256(report_path.read_bytes()).hexdigest()},
    }
    (target / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selector", type=Path, required=True)
    parser.add_argument("--route-selector", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = evaluate(
        selector_path=args.selector,
        route_selector_path=args.route_selector,
        result_path=args.result,
        output_root=args.output_root,
    )
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
