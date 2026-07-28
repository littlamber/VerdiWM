# Architecture

VerdiWM separates research freedom from claim authority. Proposal and coding
agents may explore within a bounded intervention language, but only a frozen,
independent verifier can accept an effect.

## Planes

### Constitutional plane

The constitutional plane freezes the goal, metric roles, held-out split,
evaluation horizons, acceptance thresholds, budget policy, primitive registry,
and evaluator hashes. Any change that can alter a verdict creates a new
campaign version.

Primary code:

- `wmloop/contracts.py`
- `wmloop/constitution.py`
- `wmloop/control/user_intent_compiler.py`
- `wmloop/verify/round_start_guard.py`

### Diagnostic and geometry plane

Verdict probes answer whether the requested behavior improved. Diagnostic
probes explain where the model fails and route interventions, but cannot enter
the acceptance calculation. Small, reversible perturbations produce local
response charts. Their Jacobians define the Interventional Repair Geometry
(IRG), a repair-oriented coordinate system rather than a representation
similarity score.

Primary code:

- `wmloop/diagnose/`
- `wmloop/experiments/joint_fingerprint.py`
- `wmloop/geometry/irg.py`
- `wmloop/geometry/assets.py`
- `wmloop/geometry/types.py`

Each canonical `IRGAsset` keeps raw finite-dose evidence separate from the
locality-supported effective objects `J_X`, `G_X`, and `r_X`. Response
covariance is composed only inside source groups with a shared paired baseline
frame. Missing cross-group covariance remains explicitly unobserved and forces
transfer abstention without disabling target-local routing.

Joint calibration schedules every semantic probe path against one no-hook
baseline for each target, seed, trajectory batch, evaluator, and generation
mode. Condition-level receipts are atomic and resumable; a frame mismatch
invalidates the chart instead of being normalized away.

### Proposal and compilation plane

The proposer selects a semantic primitive, dose, scope, and falsifiable
prediction. Compilation fails closed unless the target capability profile
supports the required hook and every invariant has a runtime check. The
compiler receipt records declared intent, concrete implementation, source
revision, checks, and blockers.

Primary code:

- `wmloop/propose/`
- `wmloop/primitives/`
- `wmloop/control/agent_engineering_policy.py`
- `wmloop/execute/agent_staging.py`

### Execution and verification plane

Execution occurs in an isolated worktree with budget reservation and fencing.
The verifier consumes settled receipts and frozen evidence only after execution
has reached a terminal state. Progressive fidelity keeps discovery affordable:
cheap screen, official gate, then confirmation.

Primary code:

- `wmloop/orchestrator.py`
- `wmloop/execute/`
- `wmloop/verify/`
- `wmloop/evaluate/`

### Memory and evolution plane

The archive retains raw artifacts and context-local effect records, including
null and harmful interventions. Transfer requires a certificate covering
compile validity, support overlap, effective sample size, chart alignment,
effect-sign agreement, and a calibrated lower bound. Failed terms produce
abstention. Confident opposing effects in nearby IRG regions are repair
collisions; they trigger counterexample-driven probe proposals.

Primary code:

- `wmloop/archive/`
- `wmloop/geometry/memory.py`
- `wmloop/geometry/transfer.py`
- `wmloop/geometry/evolution.py`

## Legal state ordering

One research round follows this order:

```text
freeze check -> diagnose -> propose -> compile -> reserve budget
-> execute -> settle receipt -> verify -> archive -> update routing prior
```

A provider never receives a verdict before settlement. A missing receipt,
changed evaluator, unchecked invariant, exhausted budget, or unavailable hook
cannot be converted into a positive result.

## Backbone instance contract

A portable instance declares:

1. capability profile and hook vocabulary;
2. goal and outcome schemas;
3. dataset and held-out split adapter;
4. verdict and diagnostic probe adapters;
5. evaluator and protocol freeze;
6. intervention compiler and runtime hooks;
7. receipt and artifact collection;
8. frozen regression harness.

`wmloop.control.backbone_instance` audits these surfaces. A draft adapter may be
useful for planning but is not licensed for formal claims.

## Trust boundaries

- Proposal ranking is advisory and cannot alter verifier evidence.
- Diagnostic probes route work but do not vote on acceptance.
- Agent-written patches cannot modify frozen evaluators or held-out data.
- The archive is append-oriented and content addressed.
- Transfer is opt-in and abstains when evidence is insufficient.
