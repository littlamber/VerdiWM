# LLM, Skills, and the VerdiWM Control Plane

The model should help design experiments, but it must not be the component that
decides whether arbitrary code, data, or a GPU process is allowed to run. The
usable architecture is a three-layer boundary:

1. **Research planner (LLM):** reads diagnostics, prior receipts, and public
   training recipes; writes a versioned `verdiwm-research-proposal` containing
   an objective, hypothesis, falsification criterion, evidence references,
   workflow choice, scale request, budget, and expected artifacts.
2. **Skill/workflow compiler:** maps a proposal to bounded workflows such as
   `training_scale_planner`, `checkpoint_ladder_eval`, and
   `long_horizon_rollout_eval`. It binds the proposal to an existing engineering
   manifest and generated scale plan. It does not execute shell, create a
   hidden repository, or promote a result.

The workflow registry is intentionally metadata-only. Each plugin declares its
input contract, output artifact, side-effect boundary, cost model, and whether
it is safe to reuse across models. Each workflow declares its own required
capabilities and version, and a compiled manifest hashes only the selected
capabilities. Adding an unrelated plugin therefore does not invalidate existing
runs. Registration changes metadata, not central Python dispatch code.
3. **Lightweight runtime:** composes selected capability receipts into an
   observable run and owns only lifecycle orchestration. Repository lint,
   scale planning, proposal compilation, and verification validate their own
   versioned contracts; the frozen verifier remains the only promotion
   authority. The runtime has no legacy-protocol branch or central insight
   singleton.

## Contracts

Use:

```bash
verdiwm compile-proposal \
  --proposal experiments/<id>/proposal.json \
  --engineering-manifest experiments/<id>/experiment.json \
  --training-scale-plan experiments/<id>/scale-plan.json \
  --model-capability-ir model-instance/model-capability-ir.json \
  --output experiments/<id>/compiled-manifest.json
```

Compilation fails closed when the proposal omits a falsification criterion,
selects a non-admitted skill, changes the frozen evaluator or data split, or
does not exactly match the manifest-derived scale plan. A blocked scale plan
(for example, insufficient independent episodes) produces a durable `blocked`
manifest and no launch.

The compiled receipt records content hashes, source revision and dirty state,
workflow skills, blockers, and the downstream requirements for a GPU lease,
budget receipt, held-out verifier, and archive receipt. `launch_state` remains
`not_started`; no background training is implied. If a local-validated recipe
was selected, its id and admission status are retained in the receipt; a
paper-only or `shadow_only` recipe remains ranking evidence and is rejected by
the planner/compiler boundary.

The reusable boundary is split into Model Capability IR, Experiment IR, and
Evidence IR. The first describes what a model exposes, the second compiles a
model-neutral experiment against those capabilities, and the third stores a
path-independent intervention effect with explicit claim authority. Local paths
remain only in sidecars and compiled dispatch receipts. See
`docs/INTERMEDIATE_REPRESENTATIONS.md` for the Kernel/plugin ownership and L0,
L1, and L2 authority model.

## Why both LLM and skills are needed

The LLM supplies research judgment: it can explain why a long-horizon failure
suggests explicit memory, propose a discriminating ablation, and choose a
public recipe as a shadow prior. Skills supply repeatable procedural knowledge:
they make repository layout, scale planning, checkpoint ladders, and evidence
receipts consistent. The control plane supplies authority and reproducibility.

Public papers and blogs remain ranking-only until a target-backbone local screen
has a held-out receipt. Their batch size, epochs, and step counts must never
be copied into Ctrl-World as defaults without local validation.

## Credential Boundary

An LLM adapter must use separately provisioned service credentials. It must
never read, copy, log, or persist an interactive Codex session or bridge
credential. Request metadata may record only a provider/model alias and a
prompt-template digest.

Isolated materialization inherits only explicitly named non-sensitive process
variables. Its plan contract fixes `preserve_codex_auth` to `false` and rejects
bridge, Codex, provider, API-key, token, password, and secret variables. A
deployment that wants an LLM adapter provides a separate service credential to
that adapter's process boundary; generated patch code never receives it.
