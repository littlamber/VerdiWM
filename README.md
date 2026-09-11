# VerdiWM

VerdiWM is an intent-first research workbench for diagnosing and improving
world-model systems. Describe the model, data, and objective once; VerdiWM
resolves the adapter and evaluation contract, runs bounded experiments, and
keeps evidence for every decision.

The user-facing CLI is `verdi`. The existing `verdiwm` command remains as a
backward-compatible alias, so existing scripts continue to work.

This repository is model-agnostic at the control-plane level. It does not
ship model weights, datasets, API keys, or a GPU runtime. Those remain the
responsibility of each deployment.

## Quick start

Requirements: Python 3.10 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/littlamber/VerdiWM.git
cd VerdiWM
python -m pip install uv
uv sync --group dev
uv run verdiwm doctor
```

Run `verdi` with no subcommand in a terminal to open the lightweight interactive
session. Type `/` to show the command palette; prefixes such as `/res` filter
the list. Python's optional standard-library readline integration supplies
history and Tab completion when available. Set `NO_COLOR=1` for plain output:

```text
verdi> /
verdi> /start
Model directory [/path/to/model]:
Data directory [/path/to/data]:
Research goal (for example: improve long-horizon consistency):
verdi> /plan "improve minute-scale consistency"
verdi> /run --plan ./.verdiwm/research-plan.json --confirm
verdi> /status
verdi> /exit
```

On a first run, `/start` or `/setup` discovers conventional `model/` and
`data/` paths and asks only for confirmation and one research goal. Once the
project file exists, `/start` continues by generating a reviewable plan. Common
shortcuts are accepted: `/plan "goal"` expands to `/plan --goal "goal"`, and
`/run PLAN` expands to `/run --plan PLAN`.

Command execution displays a short progress line and then a readable result card
with state, paths, campaign identifiers, and blockers instead of requiring users
to parse raw JSON. Natural-language lines in the session only produce a safe
next-step plan hint; they never import a model or start a GPU campaign silently. `/run` still
requires an explicit `--confirm` and keeps the research-plan, evaluator, and
evidence gates. Non-TTY, CI, and pipeline invocations continue to print normal
help, and the `verdiwm` compatibility alias keeps its existing behavior. Use
`verdi chat` (or `verdi shell`) to force the session explicitly.

`doctor` validates the installed package, schemas, adapter profiles, and
lightweight runtime contracts. A CPU-only installation is enough for the
included control-plane examples:

```bash
uv run python scripts/export/validate_public_example.py \
  examples/acwm_minimal_loop_cloth_next_forcing_v2
uv run python examples/portrait_first_minimal_loop_v1/run.py
```

These examples validate orchestration contracts. They do not make a claim
about model quality.

## First project

Prepare four inputs: the model weights/configuration directory, the executable
model source directory, the dataset path, and one sentence describing the
research objective. The weights and source may be separate checkouts. Model
weights and datasets stay on the user's machine and are never uploaded.

With the conventional layout, put the source in `model/` and the data in
`data/` (or `dataset/`) and run:

```bash
uv run verdiwm setup --goal "improve long-horizon prediction stability"
```

Use explicit locations when your layout is different:

```bash
uv run verdiwm setup \
  --model /path/to/model \
  --source /path/to/model-source \
  --data /path/to/data \
  --goal "improve long-horizon prediction stability"
```

Run the read-only onboarding check next:

```bash
uv run verdiwm check
```

For an unfamiliar model, generate a durable questionnaire for the user or an
agent such as Codex:

```bash
uv run verdiwm guide-model --output ./.verdiwm/onboarding-questions.json
```

The questions come from a read-only scan of entrypoints, weights, runtime, and
evaluation bindings. Codex may inspect source and draft an adapter or config,
but evaluation semantics, metric thresholds, and GPU launch require explicit
confirmation. Never put credentials in the questionnaire or project file.

If a frozen evaluator contract and model Python environment already exist, bind
them during init:

```bash
uv run verdiwm setup \
  --model /path/to/model \
  --data /path/to/data \
  --goal "improve long-horizon prediction stability" \
  --evaluator-contract /path/to/evaluator.json \
  --runtime-python /path/to/model/.venv/bin/python
```

### One-confirmation research flow

For a low-configuration entry point, compile a reviewable plan directly from a
model, dataset, and objective. Planning only scans and hashes inputs; it does
not import the model, call a research service, or allocate a GPU:

```bash
uv run verdiwm research plan \
  --model /path/to/model \
  --data /path/to/data \
  --goal "improve minute-scale long-horizon interaction consistency" \
  --budget 4gpu-hours \
  --mode hybrid \
  --output ./.verdiwm/research-plan.json
```

Review the plan's `state`, `blockers`, evaluator binding, input digest, and
stage order. When the plan is ready, confirm it once:

```bash
uv run verdiwm research run \
  --plan ./.verdiwm/research-plan.json \
  --confirm
```

The confirmation rechecks every bound file, creates an immutable CampaignStore
revision, and enters the available onboarding, IRG/diagnostic, retrieval,
candidate materialization, paired screening, confirmation, and knowledge stages.
Stages that cannot be proven remain gated. Generated text, compilation, interface
calibration, or a single loss decrease never establishes a model improvement;
open methods require target-side four-arm validation with a frozen verifier and
held-out confirmation.

## Run your project

Create a project file next to your model and dataset:

```toml
[project]
model = "./model"
data = "./data"              # ./dataset is also discovered
budget = "1gpu-hour"
state_root = "./.verdiwm/state"
```

After the check has no blockers, launch a campaign. Pass the checkpoint as an
asset, for example:

```bash
uv run verdiwm check
uv run verdiwm run \
  --goal "improve long-horizon action-conditioned prediction" \
  --target-metrics runtime_ready \
  --asset=--ckpt_path=/path/to/checkpoint.pt
```

If `check` or `run` reports a missing evaluator entrypoint, evaluator
contract, runtime, or weight, that is an intentional safety stop. The command
lists the missing information instead of guessing scientific semantics or
allocating a GPU. Existing adapter profiles usually need only the paths; a
completely new model must answer the questionnaire before an isolated launch
configuration can be generated.

The command discovers conventional `model/` and `data/` (or `dataset/`)
directories when a project file is not present. It selects an unambiguous
installed adapter profile, resolves declared evaluator metrics, and creates
an isolated adapter overlay when the model interface requires one. Unknown
metrics, ambiguous profiles, missing scientific assets, and protocol drift
fail closed with an actionable diagnostic.

## Batch heterogeneous models

Compile one immutable request for several model instances, then materialize
each ready row into its own campaign:

```bash
uv run verdiwm batch plan \
  --manifest batch-request.json \
  --output-root ./.verdiwm/batches/my-batch
uv run verdiwm batch run \
  --plan ./.verdiwm/batches/my-batch/plan.json \
  --max-parallel 2
uv run verdiwm batch status \
  --execution ./.verdiwm/batches/my-batch/execution.json
```

The batch shares one budget ledger, Archive, and CAS while preserving separate
campaign revisions and evaluator receipts. Missing frozen evaluators and input
drift stay visible as row-level blockers. `batch-plan` and `batch-run` remain
available as script-friendly aliases.

### Publish community knowledge

After a campaign settles, stage path-free semantic records from one or more local
artifact directories. The exporter is read-only and validates recognized records
before they cross the publication boundary:

```bash
uv run verdiwm community export \
  --source-root ./local-artifacts \
  --source-root ./.verdiwm/semantic-records \
  --execution ./.verdiwm/batches/my-batch/execution.json \
  --output-root ./.verdiwm/community-export
```

It ignores unrelated runtime JSON, rejects invalid recognized semantic documents,
deduplicates by canonical SHA-256, and never imports a model, allocates a GPU, or
copies execution databases and local paths. Publish only the staged `records/`
directory:

```bash
uv run verdiwm community publish \
  --documents-dir ./.verdiwm/community-export/records \
  --execution ./.verdiwm/batches/my-batch/execution.json \
  --output-root ./community-bundle \
  --publisher-id community/example \
  --signing-key ./publisher-private.pem
uv run verdiwm community verify \
  --bundle-root ./community-bundle \
  --public-key ./publisher-public.pem
```

The bundle contains only path-free semantic records, a deterministic graph,
quality audit, member SHA-256 hashes, and an Ed25519 signature. The optional
`execution.json` contributes only a batch identity binding; campaign paths,
budget databases, model paths, and runtime commands are never copied. Retrieved
records remain hypotheses until target-side frozen verification settles them.
See [Community bundles](docs/COMMUNITY_BUNDLES.md) for the publication boundary.

Create an append-only lifecycle record when a community artifact is revoked or
superseded:

```bash
uv run verdiwm community lifecycle \
  --action revocation \
  --subject-kind community_bundle \
  --subject-id verdiwm-bundle-0123456789abcdef01234567 \
  --reason "target-side verifier found an invalid claim" \
  --authority-ref cas://sha256/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa \
  --evidence-ref cas://sha256/bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb \
  --output ./lifecycle/revocation.json
```

Explicit options remain available for CI and reproducibility:

```bash
uv run verdiwm run \
  --model /path/to/model \
  --data /path/to/data \
  --goal "improve long-horizon action-conditioned prediction" \
  --budget 8gpu-hour \
  --mode hybrid
```

Campaign state can be inspected or controlled with:

```bash
uv run verdiwm status CAMPAIGN_ID
uv run verdiwm cancel CAMPAIGN_ID
uv run verdiwm reproduce CAMPAIGN_ID
```

## Local workbench

Start the local interface:

```bash
uv run verdiwm-workbench --port 8765
```

The workbench discovers materialized `graph.json` files below `state_root` by
default. When immutable experiment artifacts are kept elsewhere, bind that
directory explicitly:

```bash
uv run verdiwm-workbench --port 8765 \
  --state-root ./.verdiwm/state \
  --evidence-root /path/to/verdiwm-runs
```

Open <http://127.0.0.1:8765>. The workbench provides project discovery,
quick-start/causal-discovery/hybrid modes, campaign control, task details,
and an interactive evidence graph. It is a local UI; it does not upload
models or data.

### Windows policy errors

Some Windows installations block the generated `.venv\\Scripts\\*.exe`
console launchers with “Application Control policy has blocked this file”.
Run the modules through Python instead:

```powershell
uv run python -m wmloop.cli doctor
uv run python -m wmloop.control.workbench --port 8765
```

If `uv run python --version` is blocked too, the machine's application-control
policy must allow the installed Python/uv binaries; this is an administrator
or IT policy change, not a VerdiWM error. When the repository was extracted
from a downloaded ZIP, use the file Properties dialog's **Unblock** option
before reinstalling.

## What it provides

- Intent-to-contract compilation with typed goals, metrics, probes, trials,
  verdicts, and evidence.
- Adapter/profile onboarding with read-only discovery and conformance checks.
- Bounded execution with progressive-fidelity screens, immutable receipts,
  independent verification, cancellation, and reproduction.
- Evidence graph and effect memory that retain positive, null, and harmful
  outcomes with provenance.
- Research modes and a local workbench for repeated experiments.
- First-contact executor bootstrap for new model families through bounded,
  conformance-checked adapter generation.

The core loop and extension boundaries are documented in
[Architecture](docs/ARCHITECTURE.md), [Onboarding](docs/ONBOARDING.md), and
[Backbone instantiation](docs/BACKBONE_INSTANTIATION.md). The workbench
contains the supported research modes.

## Scope and status

The public release is `1.0.3` (stable). The control plane, schemas, CLI,
examples, workbench, automatic mechanism composition, and first-contact model
bootstrap pass the reproducible release gate. A new model family can acquire a
conformance-checked adapter automatically when a deployment supplies a trusted
base profile and bounded repair provider. Scientific assets and evaluator
semantics cannot be inferred safely: successful orchestration is not evidence
that a repair improves model quality, and quality claims still require the
model's actual runtime, data, and frozen verification protocol.

For release checks and contribution guidance, see
[CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md), and the
repository's machine-readable `RELEASE_AUDIT.json`.

## License

VerdiWM is released under the [Apache License 2.0](LICENSE). External
datasets, model weights, and upstream projects retain their own licenses.
