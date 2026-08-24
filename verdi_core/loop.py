"""Deterministic CPU control-plane loop."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .adapter import ModelAdapter
from .contracts import (
    CapabilityIR,
    Evidence,
    Goal,
    Portrait,
    ProbeFingerprint,
    canonical_digest,
)
from .knowledge import KnowledgeGraph


def run_loop(adapter: ModelAdapter, *, state_root: Path) -> dict[str, Any]:
    root = Path(state_root)
    root.mkdir(parents=True, exist_ok=True)
    goal = Goal("demo-goal-v1", "improve the declared metric", "quality", ("safety",), "heldout", 0.01)
    report = adapter.inspect()
    capability = CapabilityIR(
        model_id=str(report["model_id"]),
        revision=str(report["revision"]),
        capabilities=tuple(sorted(str(v) for v in report["capabilities"])),
        hooks=tuple(sorted(str(v) for v in report.get("hooks", []))),
        evaluator_id=str(report["evaluator_id"]),
    )
    probe = adapter.probe("action-sensitivity")
    fingerprint = ProbeFingerprint(
        fingerprint_id="fingerprint-" + canonical_digest(probe)[7:31],
        model_id=capability.model_id,
        probe_id="action-sensitivity",
        dimensions=("action", "horizon"),
        response_digest=str(probe["response_digest"]),
        uncertainty=str(probe["uncertainty"]),
    )
    portrait_body = {
        "model_id": capability.model_id,
        "capability_digest": canonical_digest(capability.__dict__),
        "fingerprint_ids": [fingerprint.fingerprint_id],
    }
    portrait = Portrait(
        portrait_id="portrait-" + canonical_digest(portrait_body)[7:31],
        model_id=capability.model_id,
        capability_digest=portrait_body["capability_digest"],
        fingerprint_ids=(fingerprint.fingerprint_id,),
        readiness="ready_for_experiment",
    )
    graph = KnowledgeGraph(root / "knowledge")
    records: list[dict[str, Any]] = []
    for hypothesis_id in ("baseline", "bounded-repair", "overdose-control"):
        intervention = {"hypothesis_id": hypothesis_id, "portrait_id": portrait.portrait_id}
        adapter.intervene(intervention)
        result = adapter.evaluate(intervention, goal.heldout_split)
        outcome = str(result["outcome"])
        evidence = Evidence(
            evidence_id="evidence-" + canonical_digest(result)[7:31],
            experiment_id=f"demo-{hypothesis_id}",
            model_id=capability.model_id,
            hypothesis_id=hypothesis_id,
            outcome=outcome,
            delta=float(result["delta"]),
            protected_ok=bool(result["protected_ok"]),
            verifier_digest=canonical_digest({"evaluator": capability.evaluator_id, "split": goal.heldout_split}),
            claim_boundary="CPU fixture evidence; requires target-side adapter verification.",
        )
        records.append(graph.append(evidence))
    summary = {
        "state": "settled",
        "goal": goal.__dict__,
        "capability": capability.__dict__,
        "fingerprint": fingerprint.__dict__,
        "portrait": portrait.__dict__,
        "evidence_count": len(records),
        "outcomes": [record["outcome"] for record in records],
        "knowledge_path": str(graph.path),
    }
    (root / "loop-summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary

