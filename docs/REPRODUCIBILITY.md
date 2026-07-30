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

The four checked-in Cosmos3 target-local bundles each carry an independent
`MANIFEST.sha256`. Validate them without model assets:

```bash
for example in \
  cosmos3_target_local_irg_wide_v1 \
  cosmos3_target_local_irg_narrow_v1 \
  cosmos3_target_local_irg_temporal_mix_v1 \
  cosmos3_action_dimension_anisotropy_counterexample_v3; do
  (cd "examples/${example}" && sha256sum -c MANIFEST.sha256)
done
```

The release builder repeats this coverage check, validates each indexed video,
and records all four verdicts in `RELEASE_AUDIT.json`.

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

## Ctrl-World ACWM predictive fingerprint

Ctrl-World is evaluated as an action-conditioned world model. The public runner
uses paired ground-truth rollout quality and explicitly excludes downstream task
success from the verdict. It imports the external checkout read-only, applies a
reversible action-embedding dose, preserves frozen episode/seed pairs, and emits
schema-validated receipts.

Run the asset and dataset preflight first:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_ctrl_world_predictive_campaign.py \
  --ctrl-world-root /path/to/Ctrl-World \
  --ctrl-world-model-root /path/to/checkpoint-matched-Ctrl-World-source \
  --campaign configs/experiments/ctrl_world_irg_calibration_pilot_v1.json \
  --heldout-split configs/goal/ctrl_world_heldout_split.json \
  --dataset-freeze configs/backbones/ctrl_world_g2_dataset_freeze.json \
  --dataset-root /path/to/droid_new_setup_full \
  --data-stat /path/to/stat.json \
  --svd-model-path /path/to/stable-video-diffusion-img2vid \
  --clip-model-path /path/to/clip-vit-base-patch32 \
  --ckpt-path /path/to/checkpoint-10000.pt \
  --reward-ckpt /path/to/checkpoint_best.pt \
  --protocol pilot \
  --output-root outputs/ctrl-world-preflight \
  --dry-run
```

Remove `--dry-run` to execute all five configured doses. The runner loads the
world model and reward scorer once, reseeds each frozen episode independently,
and writes the receipt index plus target-local fingerprint under the output
root.

Large checkpoint hashing can be deferred for a bounded pilot because a cold
network-filesystem scan may dominate runtime. Such a receipt remains marked
`pilot_ready_hash_deferred` and cannot support a formal claim. Set
`--hash-large-assets --asset-hash-cache outputs/ctrl-world-asset-hashes.json`
once to compute the digests; later runs may reuse the cache after size and
mtime verification.

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

The builder uses an allowlist, performs path/secret/size/symlink audits,
validates every Cosmos3 fingerprint bundle, and writes `MANIFEST.sha256` plus
`RELEASE_AUDIT.json`. It refuses to overwrite an existing destination.
