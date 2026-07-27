# VerdiWM Minimal Closed-Loop Proof

- Environment: `cloth_move`
- Primitive: `next_forcing`
- Seed: `2802`
- Operational loop: `PASS`
- Paper-level replicated effect: `PENDING`

## Progressive Fidelity

| Stage | Evidence |
|---|---|
| 512-step screen | AUC delta `11.9055` |
| Official 50-step gate | PSNR delta `+0.7900` |
| Checkpoint confirmation | PSNR delta `+0.9400` |
| Best checkpoint | relative step `800` |
| Long horizon | `aggregate_long_horizon_positive` on `[16, 32, 48]` |

## Claim Boundary

A diagnosis-routed primitive was materially compiled, produced a positive 512-step screen, passed the frozen official 50-step pixel gate, survived checkpoint-ladder confirmation, and yielded an aggregate long-horizon routing prior.

This bundle does not by itself establish IRG cross-backbone transfer, a calibrated transfer certificate, or a paper-level replicated causal effect when evaluation seeds are shared.

The initial and confirmation evaluations use the same evaluation seed. This is recorded explicitly rather than being promoted to an unsupported replication claim.

## Files

- `evidence/failure_report.json` (`failure_report`, public `6d95298f022277f59150265d08bd14704218e0add750aff3fe8e917a1122aa20`, source `dac9e5a5c0507302261356025ec6ca7ece265d3e1e6da73db7b7633a3a71104c`)
- `evidence/intervention_receipt.json` (`intervention_receipt`, public `989ab0aad67db1d257fbd0ef5bc8acbac139447badf7a9bba5cd3dfa95dcaa71`, source `989ab0aad67db1d257fbd0ef5bc8acbac139447badf7a9bba5cd3dfa95dcaa71`)
- `evidence/screen_manifest.json` (`screen_manifest`, public `376d2e247e3409959af1b8805c1b0ab6abbbc75b603bb9b2942ef5198d86b130`, source `8dc6b265f6ef22f131d4ea48a51bb42631be7ae309fcfdd865128bacb26c34ba`)
- `evidence/official_gate_manifest.json` (`official_gate_manifest`, public `c34565b3fdd1e595b8995224a9e738a1db16ca62fdd53c9925f3ff677b69a374`, source `9e0735123d95c201ce80f0149e6c5f19ec240c9185d327fbe09bc4b3a53e7ed3`)
- `evidence/confirmation_manifest.json` (`confirmation_manifest`, public `7db333cce6621ddab49e1072ec4b41a8e6167196622e4f5de2a2158e26cf68aa`, source `3b3af2f55f31ca691c893cbd75148b3b21e707e14cfda7bd86b8c2985fa955a0`)
- `evidence/checkpoint_ladder.json` (`checkpoint_ladder`, public `2790c7219c90fd228cc322edd44d63b9f8cc9ad6b9c073d2cba0e2f7b1fb5735`, source `604c1b10546097665fc8ce75dc2db7bc8f90fcd2fca48e79110f15d09dc367f9`)
- `evidence/effect_profile.json` (`effect_profile`, public `2b7aa26110ca95d7fe79de357c3e26f91bb938574047aae6aeed2171c17dd499`, source `3340bd6b74122a19047357f6d520107faea04cf36347aec897893ba924356a3b`)
- `evidence/experience_map.json` (`experience_map`, public `a2cf17a1ff2764e09d36ea05296264b6bf4428e61826cb90c6e093eb92f77024`, source `6305c228c13a29c14b31fc2f7bc55a665fae1d410094153b95064c639568ddb4`)
- `media/showcase_video.mp4` (`showcase_video`, public `ec797b07970c5fcdd3e90bfff38ebb3e6e05498d49898b86d929ab8206388210`, source `ec797b07970c5fcdd3e90bfff38ebb3e6e05498d49898b86d929ab8206388210`)
