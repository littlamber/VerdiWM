# VerdiWM Clean v0.1.0

VerdiWM is a model-agnostic research control plane for testing world-model
improvement ideas and retaining evidence that can be transferred to another
model. It owns the workflow around a model: contracts, budgets, isolated
worktrees, stage receipts, frozen evaluation gates, retries, and knowledge
projection. The model-specific adapter owns loading, inference, probes,
interventions, and the domain evaluator.

This repository is the lightweight Kernel release. It contains a CPU fixture
adapter so the complete control-plane contract can be verified without a GPU,
model weights, a dataset, or an AI provider. It does not contain Ctrl-World,
model checkpoints, datasets, private credentials, or historical experiment
bundles. Those belong in separate adapter and artifact repositories.

## Start In Five Minutes

Use the following path to verify that the installation is healthy:

```bash
python -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/verdi doctor
.venv/bin/verdi demo --state-root state/demo
.venv/bin/verdi graph --state-root state/demo
```

Expected result: `verdi doctor` reports a model-agnostic kernel, the demo
finishes without network access, and `verdi graph` reports the fixture's
portrait, fingerprint, and evidence records. The state directory is local
runtime data and can be deleted after the smoke test.

To exercise retrieval, idea extraction, scheduling, evaluation, and knowledge
projection as one offline composition test:

```bash
.venv/bin/verdi cycle --offline \
  --state-root state/cycle \
  --objective "improve held-out quality"
```

Run the release checks before opening a pull request or publishing a package:

```bash
./scripts/release_preflight.sh
python -m build                 # optional wheel/sdist build
```

## Choose Your Entry Point

| You have | Start with | What it proves |
| --- | --- | --- |
| No model or GPU | `verdi demo` | Basic contracts and graph projection |
| A model-independent integration | `verdi cycle --offline` | Research composition and restart behavior |
| A real model and evaluator | `verdi campaign ...` | Isolated, restartable experiment stages |
| An existing community graph | `verdi graph-bundle` and `verdi transfer` | Import/export and migration ranking |

## Connect A Real Model

Users do not implement the scheduler, worker, evaluator, retrieval system, or
knowledge graph. Provide a thin adapter implementing the `ModelAdapter`
protocol, plus a data/evaluator manifest:

```text
model SDK or HTTP API + data manifest + objective
    -> adapter and evaluator draft
    -> contract tests and metric adequacy checks
    -> human approval of permissions and assumptions
    -> autonomous campaign with isolated artifacts
```

The adapter should expose `inspect`, `probe`, `intervene`, and `evaluate`.
Adapters may support only the capabilities they can honestly provide; an
unsupported probe is recorded as unsupported rather than silently omitted. See
[`docs/USER_BOUNDARY.md`](docs/USER_BOUNDARY.md),
[`docs/PROTOCOLS.md`](docs/PROTOCOLS.md), and
[`docs/MIGRATION.md`](docs/MIGRATION.md).

An external adapter supplies the stage runner to the Kernel:

```bash
.venv/bin/verdi campaign autonomous-run \
  --state-root state/run \
  --run-id run-001 \
  --model-id my-world-v1 \
  --objective "improve held-out quality" \
  --ideas ideas.json \
  --runner my_adapter:stage_runner \
  --replanner my_adapter:replan \
  --worktree-root state/worktrees \
  --output-root state/artifacts
```

The Kernel never writes the original model checkout, pushes code, uploads
artifacts, escalates privileges, or reads outside the declared run boundary.
Failed stages can be diagnosed and retried in an isolated worktree. A real
scientific claim still requires a fixed held-out evaluator and replicated
evidence; a completed training process alone is not evidence of improvement.

The graph command can export a portable community view. A bundle keeps
the SQLite query snapshot, append-only records, a transfer index, and a
dependency-free interactive HTML graph:

```bash
.venv/bin/verdi graph-bundle \
  --state-root state/demo \
  --output-root state/demo-community-bundle
```

Share the resulting directory as one community artifact. It contains
`knowledge.sqlite3` for local queries, `graph.json` and `knowledge.jsonl` for
interchange, `transfer_index.json` for migration ranking, and `graph.html` for
a dependency-free interactive viewer. Portable exports replace machine-local
paths with content-addressed artifact references. Large logs, videos, and
checkpoints stay outside the database and can be uploaded as separate versioned
artifacts.

The knowledge graph is layered (`L0` ontology, `L1` model portraits and probe
fingerprints, `L2` methods and sources, `L3` experiments and evidence, `L4`
transfer reasoning, `L5` provenance).  Large logs, videos, and checkpoints are
artifact references with content hashes rather than database blobs.  The
transfer index is ranking-only: a target model must still pass conformance and
held-out validation.  A new model can rank prior methods with:

```bash
.venv/bin/verdi transfer --state-root state/demo \
  --target-model-id new-world --diagnostic history_dependence \
  --architecture dit --capability rollout
```

For optional AI autonomy, set `VERDI_AI_BASE_URL`, `VERDI_AI_API_KEY`, and
`VERDI_AI_MODEL`. Any OpenAI-compatible endpoint works; the same provider is
used for planning, both extraction routes, metric selection, and probe
evolution.

To run a restartable campaign with AI-managed repair from one command, use
the `autonomous-run` entry point. The adapter supplies the model-specific
stage runner; the kernel creates isolated worktrees, records AI/tool audits,
and retries actionable failures:

```bash
.venv/bin/verdi campaign autonomous-run \
  --state-root state/run \
  --run-id run-001 \
  --model-id my-world-v1 \
  --objective "improve held-out quality" \
  --ideas ideas.json \
  --runner my_adapter:stage_runner \
  --replanner my_adapter:replan \
  --worktree-root state/worktrees \
  --output-root state/artifacts
```

For an offline control-plane smoke test, replace the provider with
`--offline` and use `adapters.fixture_campaign:runner`. This verifies the
repair/retry and replicated-positive stop gates without claiming model
science.

The optional `EngineeringAgent` adds a bounded tool loop for code inspection,
isolated worktree edits, tests, command execution, failure diagnosis, retry,
and artifact collection. GPT-5.6 is a suitable backend; a Codex backend can be
added later without changing Kernel contracts. The agent cannot write the
original checkout, push or upload externally, escalate privileges, or exceed
the run's declared path, GPU, timeout, and budget policy.

The demo runs: onboarding -> probe fingerprint -> model portrait -> paired
screen -> frozen verification -> positive/null/harmful knowledge projection.
It is a contract test, not a claim about a real model.

## Adapter boundary

An adapter implements `inspect`, `probe`, `evaluate`, and `intervene` methods
behind the small `ModelAdapter` protocol. Capability level L0 only needs
`inspect` and `evaluate`; L1 adds probes; L2 adds interventions; L3 adds
reproduction/export. Adding an adapter must not require editing the kernel.

## Repository And Release Boundary

The release intentionally does not include Ctrl-World, Cosmos, model weights,
datasets, GPU launchers, or historical experiment bundles. Publish the Kernel,
each domain adapter, community knowledge bundles, and large model artifacts as
separate versioned repositories or releases. This keeps installation small and
lets a new user choose the model and artifact versions they actually have
access to.

For the full lifecycle and evidence boundaries, read
[`docs/FULL_LOOP.md`](docs/FULL_LOOP.md). For publishing, use
[`PUBLISHING.md`](PUBLISHING.md) and inspect the release checklist before
creating a GitHub release.
