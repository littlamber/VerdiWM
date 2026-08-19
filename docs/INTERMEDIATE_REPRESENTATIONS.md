# Intermediate Representations and Plugin Boundary

VerdiWM uses three small intermediate representations to keep the runtime
portable across model families without moving scientific authority into plugin
code. The design rule is **open exploration, closed authorization**: planners
and plugins may propose and compile work, while the Kernel alone owns admission,
receipts, budgets, promotion, revocation, rollback, and immutable evidence.

## Ownership

The privileged Kernel remains intentionally small. It owns:

- constitution and evaluator locks;
- lifecycle orchestration and dispatch;
- GPU and trial budget admission;
- Archive/CAS writes and event facts;
- promotion, revocation, rollback, and resume rules.

Plugins own model- or workflow-specific capability. They may provide adapters,
probes, intervention primitives, compilers, executors, verifier adapters,
memory projections, routing policies, or constitution proposals. A plugin does
not receive arbitrary Kernel authority and is not imported as an unrestricted
Python entrypoint during workflow admission.

`configs/plugins/core_workflows_v1.json` is the admission catalog. Each plugin
declares its version, input contract, output artifact, side-effect class, cost
model, cross-model reuse status, authority level, and required model
capabilities. Each workflow is admitted by `(workflow_id, workflow_version)`.
Adding an unrelated plugin does not change the capability digest of an existing
selected workflow.

## Model Capability IR

`verdiwm-model-capability-ir` is emitted by read-only onboarding. It contains:

- semantic model family and content revision;
- available model capabilities;
- generic training, evaluation, rollout, and inference interfaces;
- optional semantic hook contracts;
- asset classes, not asset locations;
- frozen evaluator identity and authority state.

The IR excludes repository roots, runtime executables, entrypoint paths,
checkpoint paths, and dataset paths. Its identity is derived from its semantic
body, so schema-valid tampering is rejected and the same model content copied to
a different checkout location retains the same capability identity.

Local paths remain in the onboarding sidecar and conformance receipt because
they are execution bindings, not shared knowledge. Conformance binds both the
Capability IR file hash and its semantic digest before any scheduler may use it.

## Experiment IR

`verdiwm-experiment-ir` binds one research proposal to:

- a selected manifest-declared workflow;
- one Model Capability IR digest;
- a frozen dataset identity and split policy;
- a path-independent training-plan digest;
- held-out metrics, horizons, seeds, and evaluator digest;
- explicit budget, artifacts, interventions, and authority policy.

The compiler checks the capabilities required by only the selected plugins.
Missing capabilities produce durable `MODEL_CAPABILITY_MISSING:*` blockers and
leave launch state at `not_started`; they never trigger a fallback
implementation or GPU process. Dataset freeze files are represented by content
hash. Semantic freeze identifiers are hashed as semantic identities. Training
manifest locations are removed from the reusable plan digest while their file
hashes and counts remain bound.

The compiled experiment manifest is a local dispatch receipt and may contain
paths to its exact engineering manifest, scale plan, registry, and Capability
IR. The embedded Experiment IR remains portable. The loader rechecks every
local file hash, registry digest, selected capability digest, Capability IR
digest, and embedded Experiment IR binding before admission.

## Evidence IR

`verdiwm-evidence-ir` is the reusable intervention-effect record. It contains:

- semantic model and task context;
- intervention intent and target hook classes;
- positive, null, harmful, or interaction outcomes;
- validity region and content-addressed evidence references;
- explicit authority bindings and claim scope.

Without both a content-addressed goal binding and evaluator binding, Evidence IR
is `schema_valid` and `ranking_only`. A settled target effect with both bindings
may be `target_confirmed`. A confirmed effect with a licensed transfer
certificate and both bindings may be `transfer_licensed`, but still requires
target-side validation. Paths, filenames, repository bindings, checkpoint
locations, and ordinary URLs are rejected from reusable evidence.

Archive/CAS objects remain authoritative. Evidence IR, retrieval indexes,
graphs, and atlases are deterministic projections that can be rebuilt or
revoked. Shared memory is therefore an intervention-effect atlas, not a graph of
free-form claims tied to one checkout.

## L0, L1, and L2

The authority layers remain separate:

| Layer | May do | May not do |
|---|---|---|
| L0 model evolution | Execute an admitted model intervention and emit receipts | Change evaluator, split, budget, or promotion rules |
| L1 optimizer evolution | Propose workflows, rankings, probes, and bounded search policies | Self-authorize a claim or bypass Kernel admission |
| L2 constitution evolution | Generate versioned shadow constitution candidates | Activate its own candidate or rewrite historical evidence |

An L2 proposal becomes active only through an external authorization path that
creates a new versioned constitution and preserves rollback. No layer rewrites
frozen historical evidence in place.

## Reuse and Caching

Reusable cache keys are semantic digests:

- Model Capability IR digest for model-side capability reuse;
- selected workflow capability digest for plugin composition reuse;
- Experiment IR digest for model-neutral experiment compilation;
- Evidence IR and CAS references for intervention-effect retrieval.

Local compiled receipts additionally bind exact files for resume and execution.
This split lets a new model reuse the platform and shared knowledge after one
capability declaration, adapter or hook mapping, conformance run, and minimal
target validation. It does not require rebuilding the experiment system, and it
does not pretend that evidence from another model proves a target-side effect.

## Current Claim Boundary

The IR, registry, tamper detection, and CPU control-plane contracts are
implemented and tested. This establishes infrastructure reuse and safe
background experiment admission. It does not establish that the current RSI
loop improves model quality, that cross-model transfer is calibrated, or that
autonomous optimizer evolution is effective. Those claims still require
controlled Ctrl-World experiments and frozen-verifier evidence.
