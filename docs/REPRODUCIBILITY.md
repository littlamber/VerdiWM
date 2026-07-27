# Reproducibility

## CPU control plane

Requirements:

- Python 3.10
- `uv`

```bash
git clone <your-verdiwm-repository>
cd VerdiWM
python -m pip install uv
uv sync --all-groups
bash scripts/ci/check_control_plane.sh
```

The public CI harness covers the geometry modules, contracts, backbone
instantiation surfaces, probe/materialization policies, and integrity of the
checked-in minimal-loop example. It does not download models or launch GPU
training.

The release includes the self-contained public test set. Development-only
integration tests that require the frozen upstream checkout, private run
protocols, historical result archives, or a configured Docker host are not
copied into the GitHub staging tree; those dependencies are unavailable in a
fresh clone and would make an otherwise valid `pytest` invocation fail.

## Validate the evidence bundle

```bash
uv run python scripts/export/validate_public_example.py \
  examples/acwm_minimal_loop_cloth_next_forcing_v2
```

The validator checks:

- artifact paths stay inside the example directory;
- every declared size and SHA-256 digest matches;
- roles and paths are unique;
- the operational chain checks remain true;
- paper-level replication remains false for the shared-seed bundle;
- public JSON and Markdown contain no local host paths.

## ACWM-Phys runtime

ACWM-Phys is an external dependency. This repository does not redistribute its
training data, checkpoints, or source checkout. Obtain those assets from the
upstream authors and provide explicit paths at runtime:

```bash
export VERDIWM_RUNTIME_PYTHON=/path/to/acwm/python
export ACWM_DATA_ROOT=/path/to/ACWM-Phys
export ACWM_CHECKPOINT_ROOT=/path/to/ACWM-Phys-checkpoints
```

Before a GPU campaign, record the upstream commit, checkpoint SHA-256, dataset
freeze, held-out protocol, evaluator hashes, environment package lock, CUDA
runtime, and physical GPU assignment. Import success alone is not a CUDA
runtime test.

## Claim levels

Use these labels consistently:

- **screen candidate:** a cheap paired run produced a positive screening score;
- **official gate pass:** the frozen evaluator passed every required metric;
- **confirmed local effect:** confirmation used the declared independent seeds
  or cohorts and passed the frozen gate;
- **transfer licensed:** every transfer-certificate term passed on held-out
  calibration;
- **abstain:** evidence or capabilities were insufficient.

The bundled example is an operational gate pass with checkpoint confirmation,
but its evaluations share a seed. It must not be relabeled as independent-seed
causal replication.

## Deterministic release construction

The development checkout can rebuild a clean public tree with:

```bash
python scripts/export/build_github_staging.py \
  --source-root . \
  --output-root ../VerdiWM-github-v0.1
```

The builder uses an allowlist, performs path/secret/size/symlink audits, and
writes `MANIFEST.sha256` plus `RELEASE_AUDIT.json`. It refuses to overwrite an
existing destination.
