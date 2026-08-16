#!/usr/bin/env python3
"""Select the widest admitted Ctrl-World radius per context and probe path."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


class ContextLocalSettlementError(ValueError):
    """Candidate atlases cannot be compared under one frozen frame."""


def _load(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContextLocalSettlementError(f"CONTEXT_LOCAL_ATLAS_INVALID:{path}") from exc
    if not isinstance(payload, Mapping):
        raise ContextLocalSettlementError(f"CONTEXT_LOCAL_ATLAS_INVALID:{path}")
    return payload


def _finite(value: object, code: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ContextLocalSettlementError(code)
    return float(value)


def _candidate(root: Path) -> dict[str, object]:
    resolved = root.resolve(strict=True)
    report_path = resolved / "context-local-fingerprint-atlas.json"
    report = _load(report_path)
    if (
        report.get("artifact_type") != "verdiwm-ctrl-world-context-local-fingerprint-atlas"
        or report.get("state") != "ready"
    ):
        raise ContextLocalSettlementError("CONTEXT_LOCAL_ATLAS_NOT_READY")
    receipts = report.get("input_receipts")
    contexts = report.get("contexts")
    if not isinstance(receipts, list) or not receipts or not isinstance(contexts, list) or not contexts:
        raise ContextLocalSettlementError("CONTEXT_LOCAL_ATLAS_FRAME_INVALID")
    radii = {
        max(abs(_finite(value, "CONTEXT_LOCAL_ATLAS_DOSE_INVALID")) for value in row.get("doses", ()))
        for row in receipts
        if isinstance(row, Mapping)
    }
    if len(radii) != 1:
        raise ContextLocalSettlementError("CONTEXT_LOCAL_ATLAS_RADIUS_INVALID")
    radius = radii.pop()
    context_rows: dict[str, dict[str, object]] = {}
    for row in contexts:
        if not isinstance(row, Mapping):
            raise ContextLocalSettlementError("CONTEXT_LOCAL_ATLAS_CONTEXT_INVALID")
        context = row.get("context")
        chart = row.get("chart")
        locality = row.get("locality_admission")
        if not isinstance(context, Mapping) or not isinstance(chart, Mapping) or not isinstance(locality, Mapping):
            raise ContextLocalSettlementError("CONTEXT_LOCAL_ATLAS_CONTEXT_INVALID")
        context_id = context.get("context_id")
        residuals = locality.get("path_residuals")
        if not isinstance(context_id, str) or not context_id or not isinstance(residuals, Mapping):
            raise ContextLocalSettlementError("CONTEXT_LOCAL_ATLAS_CONTEXT_INVALID")
        context_rows[context_id] = {
            "identity": dict(context),
            "outcome_names": list(chart.get("outcome_names", ())),
            "intervention_names": list(chart.get("intervention_names", ())),
            "jacobian": list(chart.get("jacobian", ())),
            "covariance": list(chart.get("covariance", ())),
            "repeat_count": int(chart.get("repeat_count", 0)),
            "threshold": _finite(locality.get("maximum_residual"), "CONTEXT_LOCAL_ATLAS_THRESHOLD_INVALID"),
            "residuals": {str(name): _finite(value, "CONTEXT_LOCAL_ATLAS_RESIDUAL_INVALID") for name, value in residuals.items()},
        }
    return {
        "campaign_id": report.get("campaign_id"),
        "radius": radius,
        "outcome_names": list(report.get("outcome_names", ())),
        "outcome_weights": list(report.get("outcome_weights", ())),
        "contexts": context_rows,
        "report_path": str(report_path),
        "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
    }


def settle(*, atlas_roots: Sequence[Path], output_root: Path) -> dict[str, object]:
    if len(atlas_roots) < 2:
        raise ContextLocalSettlementError("CONTEXT_LOCAL_SETTLEMENT_CANDIDATES_INSUFFICIENT")
    candidates = [_candidate(Path(root)) for root in atlas_roots]
    if len({candidate["campaign_id"] for candidate in candidates}) != len(candidates):
        raise ContextLocalSettlementError("CONTEXT_LOCAL_SETTLEMENT_CAMPAIGN_DUPLICATE")
    if len({tuple(candidate["outcome_names"]) for candidate in candidates}) != 1 or len(
        {tuple(candidate["outcome_weights"]) for candidate in candidates}
    ) != 1:
        raise ContextLocalSettlementError("CONTEXT_LOCAL_SETTLEMENT_OUTCOME_FRAME_MISMATCH")
    context_sets = [set(candidate["contexts"]) for candidate in candidates]
    if any(contexts != context_sets[0] for contexts in context_sets[1:]):
        raise ContextLocalSettlementError("CONTEXT_LOCAL_SETTLEMENT_CONTEXT_FRAME_MISMATCH")
    if output_root.exists() or output_root.is_symlink():
        raise ContextLocalSettlementError("CONTEXT_LOCAL_SETTLEMENT_OUTPUT_EXISTS")

    selections: list[dict[str, object]] = []
    admitted_count = 0
    for context_id in sorted(context_sets[0]):
        frames = [candidate["contexts"][context_id] for candidate in candidates]
        identity = frames[0]["identity"]
        names = tuple(frames[0]["intervention_names"])
        if any(frame["identity"] != identity or tuple(frame["intervention_names"]) != names for frame in frames[1:]):
            raise ContextLocalSettlementError("CONTEXT_LOCAL_SETTLEMENT_CONTEXT_IDENTITY_MISMATCH")
        for probe_id in names:
            probe_candidates: list[dict[str, object]] = []
            for candidate, frame in zip(candidates, frames, strict=True):
                residual = float(frame["residuals"][probe_id])
                probe_candidates.append(
                    {
                        "campaign_id": candidate["campaign_id"],
                        "radius": candidate["radius"],
                        "residual": residual,
                        "threshold": frame["threshold"],
                        "state": "passed" if residual <= float(frame["threshold"]) else "failed",
                        "report_path": candidate["report_path"],
                        "report_sha256": candidate["report_sha256"],
                    }
                )
            passed = [row for row in probe_candidates if row["state"] == "passed"]
            selected = max(passed, key=lambda row: (float(row["radius"]), -float(row["residual"])), default=None)
            if selected is not None:
                admitted_count += 1
            selections.append(
                {
                    "context": identity,
                    "probe_id": probe_id,
                    "state": "admitted" if selected is not None else "abstained",
                    "selected_campaign_id": selected["campaign_id"] if selected else None,
                    "selected_radius": selected["radius"] if selected else None,
                    "selected_residual": selected["residual"] if selected else None,
                    "candidates": sorted(probe_candidates, key=lambda row: float(row["radius"]), reverse=True),
                }
            )
    report = {
        "schema_version": 1,
        "artifact_type": "verdiwm-ctrl-world-context-local-fingerprint-settlement",
        "state": "ready",
        "selection_policy": "widest_locality_admitted_radius_per_context_and_probe_then_lowest_residual",
        "candidate_count": len(candidates),
        "context_count": len(context_sets[0]),
        "probe_count": len(selections) // len(context_sets[0]),
        "admitted_context_probe_count": admitted_count,
        "abstained_context_probe_count": len(selections) - admitted_count,
        "selections": selections,
        "routing_readiness": {
            "state": "not_licensed",
            "reason": "Radius settlement admits diagnostic coordinates only; held-out repair selector gain remains unmeasured.",
        },
        "claim_boundary": "This settles target-local probe radii only. It is not repair-effect, transfer, or self-evolution evidence.",
    }
    output_root.mkdir(mode=0o700, parents=True)
    report_path = output_root / "settlement.json"
    report_path.write_text(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "artifact_type": "verdiwm-ctrl-world-context-local-fingerprint-settlement-manifest",
        "state": "ready",
        "candidate_count": len(candidates),
        "admitted_context_probe_count": admitted_count,
        "abstained_context_probe_count": len(selections) - admitted_count,
        "report_path": str(report_path.resolve()),
        "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas-root", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = settle(atlas_roots=args.atlas_root, output_root=args.output_root.resolve())
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
