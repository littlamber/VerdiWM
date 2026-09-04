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

The model-conditioned binding layer in `wmloop/geometry/model_irg.py` combines
this measured vector with one immutable Model Portrait. It records coordinate-
level diagnoses, method effects, and collision/evolution references in a single
content-addressed artifact. Neighbor retrieval and collision detection remain
ranking/diagnostic operations; they cannot create a target verdict or bypass
frozen verification.

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
an optional cheap screen for diagnostics, followed by the frozen official gate
and confirmation. A screen result is never a scientific veto; formal-first
candidates may enter at gate.

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

Probe creation is mediated by Counterexample-Guided Probe Basis Expansion
(CPBE), not delegated directly to a language model. CPBE expresses residual,
structured-mutation, atlas-retrieval, and LLM-generated candidates in one
capability-checked Probe DSL. An evidence-conditioned acquisition function
ranks expected regret and coverage gain per unit cost while penalizing
nonlocality and redundancy. Selected probes pass static, offline, canary, and
expanded stages before entering the diagnostic basis. See [CPBE](CPBE.md).

A probe proposal is not an evolved asset until its successor campaign settles.
The settlement binds the counterexample lineage, successor campaign, complete
paired measurement count, unchanged locality threshold, and final admission
state. A failed successor remains in memory and cannot be routed into transfer
or LOBO merely because its code executed successfully.

Primary code:

- `wmloop/archive/`
- `wmloop/geometry/memory.py`
- `wmloop/geometry/transfer.py`
- `wmloop/geometry/evolution.py`
- `wmloop/experiments/cpbe.py`

Cold start has two evidence sources. The receipt-bound retrieval index searches
settled probe/trial experience first. When it has no compatible record, an
optional bounded arXiv lookup stages paper methods as untrusted data-only
records. `wmloop/retrieve/method_staging.py` converts each staged record into a
strict method candidate with a source identity, target failure signatures,
hook, bounded dose, applicability conditions, invariants, falsifiable
prediction, and cost estimates. Explicit matches to the frozen registry can
affect ranking only. Unknown mechanisms become prompt-compatible materialization
work orders with no command or GPU authority. The autonomous runner renders
guarded prompt packets and binds the future patch to `AgentRepairSession`
source-revision, registry-digest, and required-check receipts. It does not run a
coding agent in the active model checkout. A candidate can enter a live queue
only after static, offline, canary, shadow-replay, and next-version approval gates.
The safety language therefore stays fixed while the set of admissible
capabilities can grow across version boundaries.

### Background campaign execution

`wmloop.execute.campaign_daemon` is the durable coordinator for draining one or
more admitted candidate queues. It does not create hypotheses, edit model code,
or bypass conformance. It projects each candidate into a stable isolated worker
queue and delegates execution to `run_selected_queue`, which remains the only
`screen -> gate -> confirm` transition owner.

All workers share the same campaign budget database, Archive, CAS, and GPU lease
namespace. `max_parallel`, the queue's declared GPU-hour ceiling, per-trial cost
caps, and GPU leases are independent limits: a launch must satisfy all of them.
For multiple queues, one daemon-level ledger ceiling is either declared with
`--budget-total-gpu-hours` or derived as the sum of the immutable queue ceilings;
that same policy is passed to every worker and bound in each execution record.
The daemon persists input hashes, candidate attempts, cycle records, and terminal
state using atomic file replacement. A restart therefore resumes the same worker
root and scheduler receipt projection instead of replaying a settled trial.

The daemon periodically calls the executor's proof-checking cleanup routine.
Scratch remains retained unless its marker is terminal and settled, its receipt
is content addressed, every required artifact is present in CAS, and the trial
is visible in Archive. Worker failures are retried only up to the configured
candidate attempt limit; cycle exhaustion and terminal candidate failures are
reported as `blocked`, never converted into evidence.

Failure and capacity are separate states. `GPU_LEASE_UNAVAILABLE` occurs before
a child process starts, so the executor removes the unlaunched scratch attempt,
releases its fenced budget reservation, and raises a scheduling deferral. The
daemon records the deferral count and retries it without incrementing scientific
failure attempts. Commands that launch and then fail still settle as terminal
negative evidence; temporary lack of GPU capacity creates neither a receipt nor
an Archive trial.

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
