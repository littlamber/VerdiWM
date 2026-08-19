# World-Model Training Recipe Research

`configs/retrieval/world_model_training_recipes_v1.json` is a checked-in,
offline-readable registry of public training reports. It is deliberately a
shadow knowledge source: it can rank comparable work and explain what is
known, but it cannot authorize a local GPU run.

## What the public sources actually disclose

| Source | Data | Sequence | Reported optimization | Evidence boundary |
| --- | --- | --- | --- | --- |
| [Genie](https://arxiv.org/abs/2402.15391) | 6.8M 16-second clips, about 30k hours, 10 FPS | 160 frames | dynamics model: batch 512, 125k steps, max LR 3e-5, 5k warmup, 256 TPUv5p | Final foundation-model recipe; checkpoint choice and episode count are not specified |
| [GAIA-1](https://arxiv.org/abs/2309.17080) | 4,700 hours / about 420M images; 400 hours of disjoint validation | component sequences use 7 images; source has 6.25/12.5/25 Hz tasks | world model: batch 128 across 64 A100 80GB, 100k steps, LR 1e-4, 2.5k warmup, cosine decay | Proprietary driving data and separate component recipes |
| [Cosmos](https://arxiv.org/abs/2501.03575) | about 20M raw hours; about 100M pretraining clips and 10M fine-tuning clips | progressive 512px/57 frames to 720px/121 frames | 2.5k warmup is reported; high-quality fine-tuning is `O(10k)` iterations; universal batch and total pretraining steps are not reported | Platform with several model families, not one recipe |
| [DreamerV3](https://arxiv.org/abs/2301.04104) | online interaction across 150+ tasks | task-dependent imagined rollouts | no fixed offline video-pretraining budget | Useful algorithmic prior, not a video recipe |
| [Diffusion Forcing](https://arxiv.org/abs/2407.01392) | Minecraft/DMLab and task datasets | task-dependent frame stacking; rollouts up to 1000 frames | the paper explicitly does not study internet-scale training | Mechanism/ablation reference only |
| [V-JEPA 2](https://arxiv.org/abs/2506.09985), representation pretraining | over 1M hours and 22M videos | 16-frame warmup/main, 64-frame cooldown | 12k warmup + 228k constant + 12k cooldown; batch/LR not stated as one universal value | Self-supervised representation pretraining, not action conditioning |
| [V-JEPA 2-AC](https://arxiv.org/abs/2506.09985), DROID post-training | less than 62 hours of robot video | 4-second, 4 FPS, 16-frame clips | batch 256; 4.5k warmup + 85.5k constant + 4.5k decay; LR peaks at 4.25e-4 | Frozen encoder and a task-specific camera/data convention |

The numbers are not directly comparable. A “step” in a latent video model,
an online-RL update, and a diffusion denoising update do not consume the same
amount of data or compute. Episode diversity, temporal correlation, token
compression, action conditioning, rollout horizon, and validation protocol
must therefore stay explicit in the local receipt.

## How VerdiWM uses the registry

```bash
verdiwm training-recipes
verdiwm training-recipes --recipe-id genie_dynamics_pretrain_v1
```

The output includes source URLs, evidence tier, disclosed fields, and fields
the source did not disclose. All checked-in entries are `shadow_only` and
`ranking_only`.

`verdiwm plan-training --training-recipe ...` only accepts a recipe that has a
local validation receipt and explicit `local_validated` or
`reusable_optimization_memory` status. A paper number cannot silently replace
the manifest-derived dataset count, episode-diversity gate, checkpoint ladder,
or held-out stopping policy. This is the intended path for promoting a source:

1. import the source with its URL/version and evidence claims;
2. run a bounded local screen on the target backbone and dataset;
3. record held-out and long-horizon results, seeds, and compute receipt;
4. explicitly admit only the validated subset of fields into a backbone-specific
   profile.

There is no dedicated external “world-model training skill” installed in this
environment. The registry and planner boundary are the safer equivalent: they
make literature searchable and reusable without turning an unverifiable blog
or a large-lab budget into automatic execution authority.
