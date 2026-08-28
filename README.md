# VerdiWM

**A lightweight, model-agnostic control plane for reliable world-model Auto Research.**

English | [简体中文](README_zh.md)

VerdiWM runs the work around a world model: diagnose behavior, schedule bounded
interventions, execute them in isolated worktrees, evaluate frozen held-out
splits, and retain auditable evidence. Your model-specific code stays behind a
small adapter; the Kernel does not depend on a model framework, GPU launcher,
or vendor API.

> **Release status: v0.1.0 Kernel.** The control-plane contracts and offline
> fixture path are tested. A real world-model adapter, domain evaluator, and
> live provider credentials are still required for scientific claims.

## Why VerdiWM?

Most Auto Research systems optimize a score and keep experiment summaries.
VerdiWM is designed to retain the stronger object: a bounded, reviewable claim
with its conditions, evidence, counter-evidence, and provenance.

```text
goal
  -> inspect + diagnostic probes
  -> model portrait and failure fingerprint
  -> retrieve and propose bounded interventions
  -> isolated execution with receipts
  -> frozen held-out verification
  -> positive / null / harmful / abstain evidence
  -> portable graph and bounded transfer candidates
```

The graph is therefore not a recipe replay system. It is a search and
decision surface: known failures are excluded, uncertain boundaries become
experiments, and transfer is only a ranked hypothesis until the target model
passes its own conformance and held-out checks.

## Five-minute local demo

Requires Python 3.10-3.12. The fixture uses no GPU, model weights, network, or
AI provider.

```bash
python -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/verdi doctor
.venv/bin/verdi demo --state-root state/demo
.venv/bin/verdi graph --state-root state/demo
```

The demo runs the complete control-plane path: onboarding, probes, portrait,
fingerprint, frozen verification, and evidence projection. Its metrics are
contract-test fixtures, not results for a real world model.

To exercise retrieval, idea extraction, scheduling, evaluation, and restartable
state as one offline composition:

```bash
.venv/bin/verdi cycle --offline \
  --state-root state/cycle \
  --objective "improve held-out quality"
```

Inspect or export the resulting local knowledge state:

```bash
.venv/bin/verdi knowledge --state-root state/demo
.venv/bin/verdi graph-bundle \
  --state-root state/demo \
  --output-root state/demo-community-bundle
```

## Use it with a real model

The user-facing boundary is intentionally small:

```text
model SDK or HTTP endpoint + data manifest + objective
    -> thin adapter and evaluator
    -> contract tests and metric-adequacy checks
    -> human approval of permissions and assumptions
    -> isolated, restartable campaign
```

Implement the `ModelAdapter` protocol (`inspect`, `probe`, `intervene`, and
`evaluate`) or wrap an existing SDK/HTTP service. Declare only capabilities
that are actually available. Unsupported probes or interventions must become
an explicit `abstain`, never a silently skipped result.

```python
from typing import Any

class MyWorldAdapter:
    adapter_id = "my-world"
    version = "2026-01"

    def inspect(self) -> dict[str, Any]:
        return {
            "model_id": "my-world-v1",
            "revision": "git-sha-or-image-digest",
            "capabilities": ["inference", "rollout", "evaluation", "intervention"],
            "hooks": ["action-conditioning"],
            "evaluator_id": "my-heldout-evaluator-v1",
        }

    def probe(self, probe_id: str) -> dict[str, Any]:
        ...

    def intervene(self, hypothesis: dict[str, Any]) -> dict[str, Any]:
        ...

    def evaluate(self, intervention: dict[str, Any], split: str) -> dict[str, Any]:
        ...
```

The Kernel never edits the original checkout, pushes code, uploads artifacts,
or escalates privileges. Model loading, training, rollouts, data access, and
domain-specific evaluator semantics remain in the adapter package. See
[`docs/USER_BOUNDARY.md`](docs/USER_BOUNDARY.md) and
[`docs/PROTOCOLS.md`](docs/PROTOCOLS.md).

## Run a campaign

Give VerdiWM ideas and an adapter-owned stage runner. The supervisor persists
stage transitions, retries, receipts, and evidence in SQLite, so a stopped run
can be resumed without duplicating settled records.

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

For a deterministic control-plane check, add `--offline` and use the fixture
runner. Other useful commands are:

```bash
.venv/bin/verdi campaign status --state-root state/run --run-id run-001
.venv/bin/verdi campaign resume --state-root state/run --run-id run-001 \
  --model-id my-world-v1 --runner my_adapter:stage_runner
```

The optional AI provider is OpenAI-compatible. Set `VERDI_AI_BASE_URL`,
`VERDI_AI_API_KEY`, and `VERDI_AI_MODEL` to enable planning, source extraction,
metric selection, probe evolution, and bounded engineering repair. Without
those variables, use the deterministic `--offline` path.

## Evidence and evaluation

Every promoted result is tied to a frozen objective, data split, evaluator,
metric direction, seed policy, budget, artifact digest, and claim boundary.
Outcomes include `confirmed_positive`, `null`, `harmful`, and `abstain`.

The benchmark-aware selector can propose WorldArena-style object consistency,
action consistency, and downstream task-success metrics when the adapter
actually exposes the required data and ground truth. A primary improvement
cannot hide a protected-metric regression. Expensive long-horizon checks run
only after cheaper pilot checks pass.

For a scientific claim, the evaluator should additionally provide:

- frozen train/validation/held-out splits and their hashes;
- independent seeds and replicated runs;
- protected metrics and uncertainty intervals;
- downstream task success, not only local probe improvement;
- complete code, configuration, checkpoint, and runtime receipts;
- non-local or cross-model validation when transfer is claimed.

The CPU fixture proves the control-plane contract only. It must not be cited as
evidence that a real world model improved. The current release checklist still
has open gates for a real external adapter/evaluator and live integrations:
[`docs/RELEASE_CHECKLIST.md`](docs/RELEASE_CHECKLIST.md).

## Knowledge graph and transfer

The portable projection is layered:

| Layer | Contents |
| --- | --- |
| L0 | ontology and contracts |
| L1 | model portraits and probe fingerprints |
| L2 | methods, sources, and interventions |
| L3 | experiments and evidence |
| L4 | transfer assessments and boundaries |
| L5 | provenance and integrity receipts |

SQLite is the local query projection. Append-only JSONL, `graph.json`,
`transfer_index.json`, and a dependency-free `graph.html` form the community
bundle. Logs, videos, and checkpoints remain versioned artifact references,
rather than database blobs.

```bash
.venv/bin/verdi transfer --state-root state/demo \
  --target-model-id new-world \
  --diagnostic history_dependence \
  --architecture dit \
  --capability rollout
```

Transfer output is ranking only. The target model must still pass adapter
conformance, frozen held-out evaluation, replication, and protected-metric
gates.

## Mechanism search

`VerdiWM-mechanism-search` is the incubating research layer for a typed
Mechanism DSL and candidate-program search. It supports failure modes,
composable operators, preconditions, anti-conditions, falsifiable predictions,
budget-aware selection, semantic deduplication, and claim revision
(`retain`, `shrink`, `split`, `revoke`).

It is deliberately separate from this Kernel while the interface is being
validated against a real adapter. Only model-independent search and claim
logic should eventually move into the Kernel; model-specific patch materializers,
training commands, weights, datasets, and evaluators stay in external adapter
repositories. The incubator is released as a companion repository so its
experimental interface and dependencies do not expand this Kernel release.

## Repository layout

```text
verdi_core/       model-agnostic contracts, orchestration, evidence, storage
adapters/         CPU fixtures only
tests/            contract and lifecycle tests
docs/             architecture, protocols, evaluation, migration, release
scripts/          release checks
```

Ctrl-World, Cosmos, checkpoints, datasets, GPU launchers, private credentials,
and historical experiment bundles are intentionally outside this repository.
The companion Ctrl-World adapter binds user-owned assets through environment
variables and never modifies the upstream checkout.

## Development and release

```bash
python -m pytest -q
./scripts/release_preflight.sh
python -m build
```

Before publishing, read [`PUBLISHING.md`](PUBLISHING.md) and the release
checklist. Keep changes model-agnostic and dependency-light; add model-specific
logic to an adapter instead of `verdi_core`. See
[`CONTRIBUTING.md`](CONTRIBUTING.md) and [`SECURITY.md`](SECURITY.md).

## Roadmap

The next research milestone is evidence-gated mechanism search: connect typed
candidate programs to the Boundary-Aware Research Evidence Graph, choose
experiments by information gain, and revise claims when counterexamples appear.
The engineering order remains conservative:

1. complete the real external adapter and evaluator validation chain;
2. replicate across seeds, held-out splits, and non-local conditions;
3. validate mechanism search on that adapter;
4. then fold stable model-independent components back into the Kernel.

## License

VerdiWM is released under the [Apache License 2.0](LICENSE).
