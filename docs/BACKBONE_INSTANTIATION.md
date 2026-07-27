# Backbone Instantiation

VerdiWM treats ACWM-Phys as a reference instance, not as a hidden system
assumption. A new backbone is admitted through a versioned instance packet.
The packet separates code that can execute from evidence that can support a
formal claim.

## Required surfaces

Every instance declares these surfaces in
`configs/schemas/backbone_instance.schema.json`:

1. a user goal and metric-role specification;
2. a dataset and held-out split adapter;
3. verdict probes and diagnostic-only probes;
4. an evaluator and immutable protocol freeze;
5. a hook adapter for the intervention vocabulary;
6. an executor that emits settled receipts;
7. an archive adapter for content-addressed evidence;
8. a regression harness that protects the frozen surfaces.

A surface may be `draft`, `missing`, `ready`, or `external_ready`. Declaring a
missing file as ready does not bypass the audit. Draft surfaces may support
planning, but they cannot license a formal verdict.

## Bring-up sequence

### 1. Compile the goal

Create a goal under `configs/goal/` from the user objective. Separate metrics
into three roles:

- **goal:** the behavior to improve;
- **validity:** non-regression constraints required for acceptance;
- **diagnostic:** routing signals that cannot vote on acceptance.

Freeze held-out data, horizons, thresholds, random seeds, and evaluator hashes
before launching a formal campaign. A protocol change creates a new campaign
version; it is not an in-place patch.

### 2. Declare capabilities

Copy the public ACWM packet as a structural reference:

```bash
cp configs/backbones/acwm_phys_g1_long_horizon_ladder_v1.json \
  configs/backbones/<backbone>_<goal>_v1.json
```

Replace every surface with an artifact from the target backbone. Keep
unfinished surfaces explicitly `draft` or `missing`. Do not point at ACWM
adapters merely to make the packet validate.

### 3. Audit the instance

```bash
uv run verdiwm-backbone-audit \
  --instance-config configs/backbones/<backbone>_<goal>_v1.json \
  --output-root results/instance-audit
```

The audit is read-only. It validates schemas and freezes, checks that declared
artifacts exist, and emits explicit blockers. It never starts a GPU job or
grants a phase gate by itself.

### 4. Materialize hooks

Map the target model to the typed hook vocabulary:

| Hook | Contract | Typical interventions |
|---|---|---|
| `H1` | dataset/sample transform | collection and mixture reweighting |
| `H2` | conditioning transform | anchors, external representations, memory |
| `H3` | loss assembly | rollout, contrastive, reward, and distillation losses |
| `H4` | sampler callback | context trimming and guidance schedules |
| `H5` | optimizer/model configuration | bounded scale or capacity changes |

An adapter must preserve the primitive's declared semantic intent. If the
backbone cannot expose the required behavior, compilation returns a blocker;
it must not substitute a convenient sidecar or unrelated hyperparameter.

Generate the eligibility matrix after the hook adapter exists:

```bash
uv run verdiwm-capability-matrix \
  --instance-config configs/backbones/<backbone>_<goal>_v1.json \
  --output-root results/capability-matrix
```

### 5. Run the frozen harness

Before GPU execution, test tensor shape, dtype, device, determinism, patch
scope, artifact collection, timeout settlement, and evaluator immutability.
Record the upstream revision, environment lock, checkpoint digest, and physical
GPU assignment in the trial receipt.

### 6. Promote evidence progressively

Use the same claim ladder for every backbone:

```text
diagnosis -> matched primitive -> cheap screen -> frozen official gate
-> independent confirmation -> context-local memory -> certified transfer
```

A positive screen is a scheduling signal, not an accepted result. Transfer is
licensed only when semantic compilation, support overlap, effective sample
size, chart alignment, sign agreement, and the calibrated effect lower bound
all pass. Otherwise the correct output is `abstain` and a cold-start search.

## Definition of done

A new backbone instance is closed-loop ready when all required execution
surfaces are present and contract-valid. It is formal-verdict ready only when
the held-out protocol, verdict probes, evaluator, and hashes are frozen and the
regression harness passes. It is transfer ready only after measured target
charts calibrate the transfer certificate; code availability alone is not
evidence of transfer.
