# VerdiWM

VerdiWM is an intent-first research workbench for diagnosing and improving
world-model systems. Describe the model, data, and objective once; VerdiWM
resolves the adapter and evaluation contract, runs bounded experiments, and
keeps evidence for every decision.

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

Prepare four inputs: the model source directory, a checkpoint or weight file,
the dataset path, and one sentence describing the research objective. Model
weights and datasets stay on the user's machine and are never uploaded.

With the conventional layout, put the source in `model/` and the data in
`data/` (or `dataset/`) and run:

```bash
uv run verdiwm init --goal "improve long-horizon prediction stability"
```

Use explicit locations when your layout is different:

```bash
uv run verdiwm init \
  --model /path/to/model \
  --data /path/to/data \
  --goal "improve long-horizon prediction stability"
```

Run the read-only onboarding check next:

```bash
uv run verdiwm check-model
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
uv run verdiwm init \
  --model /path/to/model \
  --data /path/to/data \
  --goal "improve long-horizon prediction stability" \
  --evaluator-contract /path/to/evaluator.json \
  --runtime-python /path/to/model/.venv/bin/python
```

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
uv run verdiwm check-model
uv run verdiwm run \
  --goal "improve long-horizon action-conditioned prediction" \
  --target-metrics runtime_ready \
  --asset=--ckpt_path=/path/to/checkpoint.pt
```

If `check-model` or `run` reports a missing evaluator entrypoint, evaluator
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
