# Source-Effect Audit: `self_forcing_finetune`

This audit classifies source-effect evidence quality. It does not alter the frozen transfer certificate, admit a probe, or establish target-model quality improvement.

Collision verdict: `mixed_effects_with_unstable_positive_sources`

| Environment | Indexed labels | Complete receipts | Seeds | Stable + seeds | Signs | Classification | Next action |
|---|---:|---:|---:|---|---|---|---|
| push_cube | 1 | 1 | 1 | none | negative | `negative_underreplicated` | `replicate_independent_training_seed_then_official_gate` |
| stack_cube | 1 | 4 | 3 | none | negative | `stable_negative` | `retain_settled_direction` |
| push_rope | 1 | 6 | 1 | none | negative, positive | `eval_seed_sensitive` | `run_frozen_multi_eval_seed_replication` |
| cloth_move | 2 | 21 | 5 | 2805, 4101, 4202, 4303 | negative, positive | `same_protocol_reproduction_conflict` | `rerun_deterministic_official_gate_replication` |
| push_sand | 0 | 0 | 0 | none | none | `no_official_gate_evidence` | `no_action_no_current_evidence` |
| pour_water | 1 | 5 | 3 | none | negative | `stable_negative` | `retain_settled_direction` |
| robot_arm | 2 | 14 | 3 | 3311 | negative, positive | `eval_seed_sensitive` | `run_frozen_multi_eval_seed_replication` |
| reacher | 1 | 7 | 5 | none | negative | `stable_negative` | `retain_settled_direction` |

## Decision

Stable negative evidence is retained. Positive or inconsistent sources must be repaired with the listed frozen-protocol work orders before a new collision-disambiguating probe is admitted. The transfer-certificate thresholds remain unchanged.
