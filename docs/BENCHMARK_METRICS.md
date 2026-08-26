# Benchmark-Aware Metrics

VerdiWM treats benchmark metrics as versioned scientific instruments, not a flat list of scores. The default catalog contains the action-conditioned WorldArena metric family, but no metric is assumed applicable merely because it exists in that catalog.

## Adapter Manifest

An adapter's `inspect()` report declares only observable facts. It may supply benchmark records discovered from a pinned official paper, code release, or evaluator manifest. It must not claim that an unavailable field exists.

```python
{
    "model_id": "my-world-v1",
    "revision": "model-sha-or-tag",
    "capabilities": ["rollout", "evaluation", "intervention"],
    "available_signals": ["video_ground_truth", "state_ground_truth", "interaction_index"],
    "heldout_split": "worldarena-heldout-v1",
    "benchmark_metrics": [{
        "metric_id": "domain_metric", "benchmark": "worldarena",
        "description": "short definition from the pinned source",
        "direction": "minimize", "role_candidates": ["protected", "diagnostic"],
        "required_signals": ["state_ground_truth"], "required_capabilities": ["rollout"],
        "cost": "medium", "ground_truth": True,
        "evaluator_ref": "official-evaluator@commit",
        "source_refs": ["https://official.example/release"],
        "diagnostic_only": False, "implementation_status": "catalogued",
    }],
}
```

The research path also sends retrieved benchmark documents through the `benchmark_catalog_extractor` role. Extracted records are provenance-tagged, validated as structured catalog rows, and retained only as candidates until their evaluator has passed the gates below.

## Autonomous Selection

The AI sees the model portrait, probe fingerprint, objective, observable signals, capabilities, and candidate catalog. It selects one primary metric, protected metrics, diagnostics, directions, a practical threshold, and an evaluation order. The Kernel independently rejects a selection when a metric needs an unavailable signal or capability, is not permitted for its chosen role, or lacks machine-readable ground truth but is proposed as primary/protected.

VLM, human-preference, and intervention-sensitivity signals may be retained as diagnostics, but cannot by themselves produce a positive scientific verdict.

## Staged Evaluation And Verdicts

The selected plan lists lower-cost pilot metrics and the full promotion set. High-cost long-horizon metrics are deferred until a candidate has a non-harmful pilot. Formal promotion is always evaluated on the frozen held-out split:

```text
primary improves beyond the practical threshold
AND every protected metric stays within its declared regression margin
AND paired, independent replication passes
    => confirmed_positive
otherwise => null, harmful, or abstain
```

A weighted average cannot compensate for a protected regression. The current WorldArena seed catalog uses `rollout_video_l1` as a likely primary candidate, with final-state, multi-view, and (when supported) long-horizon drift metrics available as protected candidates. The AI may choose a different eligible set for a different model; it cannot make a subjective metric a verdict source.

## Evaluator Materialization

When an official metric applies but the adapter has no implementation, the bounded `EngineeringAgent` may create it in an isolated worktree. It is only eligible for formal use after its receipt contains all of:

- passing evaluator contract tests;
- a SHA-256 evaluator revision digest;
- a SHA-256 frozen held-out split digest;
- agreement with pinned benchmark reference fixtures; and
- deterministic repeatability on the same fixtures.

The agent can inspect files, patch only its worktree, run tests, and collect receipts. It cannot write the original checkout, publish code, upload data, or expand its declared filesystem/GPU/time authority. An incomplete receipt is an `abstain`, not a degraded formal score.

## Knowledge Projection

Every selected plan is projected into the graph: model -> metric plan -> metric -> evaluator. The plan retains catalog/source digests, rejected metrics and reasons, role assignment, staged cost policy, and evaluator provenance. This lets a future model query not only which method helped, but which metrics were considered applicable, unavailable, or insufficient for that claim.
