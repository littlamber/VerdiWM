# Ctrl-World Fingerprint-Conditioned Signed History Correction

## Status And Claim Boundary

This document defines a trainable research hypothesis and its falsification
plan. It is not an implemented primitive, a trained checkpoint, or evidence of
improvement.

Working name: **Fingerprint-Conditioned Signed History Correction (FSHC)**.

The motivating observation is context-local: a small first-frame anchor dose
helps some Ctrl-World rollout contexts, harms others, and has no reliable
direction in a third group. A global positive or negative anchor is therefore
not a valid repair. The target capability is to learn when old visual evidence
should be retained, attenuated, opposed, or ignored under generated-history
rollout.

## Evidence And Nearest Mechanisms

| Source | Reusable mechanism | What it does not provide |
|---|---|---|
| Self-Forcing, `2506.08009v2` | Train on generated rather than only ground-truth history | No context-dependent history reliability or signed correction |
| Diffusion Forcing, `2407.01392v4` | Heterogeneous temporal noise/reliability | Noise level is not a learned signed old-state correction |
| GameNGen, `2408.14837v2` | Conditioning augmentation stabilizes long autoregressive generation | Fixed augmentation does not handle sign reversals |
| Professor Forcing, `1610.09038v1` | Align teacher and free-running dynamics | Global distribution alignment does not route individual history items |
| KalmanNet, `2107.10043v3` | Learn a correction gain from prediction innovation | It is not a video diffusion history operator |
| Bayesian KalmanNet, `2309.03058v4` | Predict uncertainty for a learned filter | Uncertainty alone does not specify signed visual correction |
| Mamba, `2312.00752v2` | Input-dependent propagation and forgetting | The update is retention-oriented, not probe-calibrated signed correction |
| Neural Turing Machine, `1410.5401v2` | Differentiable external memory access | Generic memory has no Ctrl-World failure-specific training signal |

FSHC is not the union of these methods. Its potentially distinct claim is:

> A generated-history video diffusion model can learn a state-dependent,
> signed temporal correction operator whose direction and abstention are
> calibrated by local causal response rather than fixed age, fixed noise, or
> generic attention.

This remains a hypothesis until the nearest-neighbour ablations below reject
equivalent explanations.

## Target Hook Mapping

Ctrl-World currently exposes three usable training locations:

1. `models/ctrl_world.py`: history latent construction and the existing
   per-history-frame noise augmentation.
2. `models/ctrl_world.py`: action embeddings supplied to U-Net cross-attention.
3. `models/unet_spatio_temporal_condition.py`: frame-flattened U-Net condition
   path, where a compact temporal correction module can affect history inputs
   without replacing the backbone.

The initial implementation should modify the history-conditioning construction
before U-Net input. It should not require replacing the entire SVD U-Net.

## Mechanism

Let `h_i` be a historical latent, `c` the current latent, `a` the action-window
embedding, `t` the diffusion noise level, and `age_i` the history age. Define a
runtime feature vector using only quantities available in unseen contexts:

```text
q_i = [pool(h_i), pool(c), pool(c - h_i), age_i,
       action_delta(a), rollout_consistency, diffusion_noise(t)]
```

A small shared correction network predicts:

```text
s_i = tanh(f_sign(q_i))             # signed correction, [-1, 1]
r_i = sigmoid(f_confidence(q_i))    # confidence, [0, 1]
u_i = sigmoid(f_abstain(q_i))       # use probability, [0, 1]
```

The corrected history is:

```text
h_i' = h_i + u_i * r_i * s_i * phi(c - h_i)
```

`phi` starts as an identity or channelwise low-rank adapter. Positive `s_i`
moves old history toward the current state, negative `s_i` extrapolates away
from a misleading old anchor, and small `u_i*r_i` abstains. The strength must
be norm-bounded so the module cannot silently replace the history latent.

The first version should predict one scalar per history frame plus an optional
small channelwise residual. A full spatial gate is deferred unless the scalar
version fails with spatially localized residual evidence.

## Training Distribution

Training alternates two history sources:

```text
teacher branch:   ground-truth history with heterogeneous corruption
rollout branch:   bounded model-generated history with truncated gradients
```

The generated-history probability is scheduled from low to moderate rather
than immediately replacing teacher history. A single batch may mix teacher and
generated histories per sample. This is the Self-Forcing-derived component,
not the proposed novelty.

## Losses

The total objective is:

```text
L = L_diffusion
  + lambda_roll * L_generated_history
  + lambda_probe * L_local_direction
  + lambda_rank * L_counterfactual_rank
  + lambda_cons * L_teacher_rollout_consistency
  + lambda_sparse * mean(u * r)
```

`L_local_direction` uses only development contexts where the frozen paired
probe produced a reliable direction. It constrains the predicted aggregate
signed response to agree with the probe direction and uses an abstain target
where neither direction was reliable. Raw episode IDs or context IDs are not
inputs.

`L_counterfactual_rank` compares small positive, zero, and negative correction
doses under the same sample and seed. It requires the selected direction to
rank better on the declared prediction objective while preserving action
sensitivity. This prevents the module from learning a sign label without a
corresponding model-quality effect.

`L_teacher_rollout_consistency` follows the Professor-Forcing motivation but
uses feature consistency or a bounded discriminator only if a cheaper feature
loss is insufficient.

`L_sparse` encourages abstention. It must be paired with the main quality loss
so the trivial all-zero correction is not accepted as success.

## Fingerprint Use

The four probes are not concatenated to the runtime gate. They have three
roles:

1. define the causal axes that motivated the module;
2. label a bounded set of local direction/abstention constraints;
3. select held-out contexts that test whether the learned runtime features
   generalize beyond probed episodes.

This prevents the method from memorizing fingerprint values or requiring a
probe execution before every inference call.

## Required Ablations

Every row trains a new checkpoint under the same data, steps, effective batch,
optimizer, evaluator, and seed policy.

| ID | Training condition | Question |
|---|---|---|
| A0 | Official checkpoint, no training | Published baseline |
| A1 | Ordinary teacher-history fine-tuning | Is improvement just more training? |
| A2 | Generated-history exposure only | Self-Forcing contribution |
| A3 | Heterogeneous history noise only | Diffusion/GameNGen contribution |
| A4 | Fixed negative correction | Does the known development direction broadcast? |
| A5 | Fixed positive correction | Symmetric fixed-control check |
| A6 | Unsigned learned retention gate | Is signed correction necessary? |
| A7 | Signed gate without probe-derived losses | Does ordinary end-to-end learning discover the effect? |
| A8 | Signed gate with direction loss but no rollout branch | Is generated-history exposure necessary? |
| A9 | Full FSHC | Combined hypothesis |
| A10 | Full FSHC without abstention regularization | Does abstention prevent harmful routing? |
| A11 | Full FSHC with shuffled probe labels | Does causal calibration contain real information? |

The nearest-method claim fails if A9 cannot beat A2, A3, A6, and A7 on
held-out contexts. The fingerprint claim fails if A11 matches A9.

## Evaluation And Falsification

Primary evaluation must use held-out episode/start contexts that were not used
for probe calibration. Required outputs:

- long-horizon latent/image prediction quality;
- final-interaction quality and horizon degradation slope;
- action-following or counterfactual action sensitivity;
- task success where the existing replay environment supports it;
- harmful-routing rate and abstention calibration;
- per-context correction sign, magnitude, and confidence;
- compute, peak memory, throughput, and checkpoint size.

Reject FSHC if any of the following holds:

1. improvement is explained by ordinary fine-tuning or generated-history
   exposure alone;
2. signed routing does not generalize to held-out contexts;
3. protected action sensitivity or short-horizon quality regresses beyond the
   frozen tolerance;
4. the method primarily selects one global sign;
5. shuffled direction labels perform equivalently;
6. gains appear only under inference-time manual overrides;
7. the added module cannot be removed at inference without losing the claimed
   capability, but its weights were not trained in the checkpoint.

## Implementation Order

1. Build a pure-tensor correction module and unit-test shape, zero-dose
   identity, bounded norm, positive/negative direction, and abstention.
2. Add teacher-history training only and reproduce the original loss when the
   module is disabled.
3. Add bounded generated-history rollout with gradient truncation.
4. Materialize direction, ranking, and abstention losses from frozen
   development probe evidence.
5. Run a one-GPU overfit smoke on a tiny fixed batch.
6. Run an eight-GPU throughput/memory smoke and freeze batch/accumulation.
7. Train A1, A2, A3, A6, A7, and A9 first; admit the remaining ablations only
   if A9 clears the predeclared canary gate.

No eight-GPU formal training should start before steps 1-5 pass and the
held-out context split is frozen.
