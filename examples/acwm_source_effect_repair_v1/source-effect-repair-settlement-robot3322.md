# Source-Effect Repair Settlement

This settlement establishes frozen-protocol source-effect reproducibility only. It does not change the transfer certificate or establish cross-backbone transfer.

| Environment | Training seed | Eval seeds | Positive | Negative | Verdict |
|---|---:|---|---:|---:|---|
| robot_arm | 3322 | 101, 202, 303 | 1 | 2 | `sign_inconsistent` |

## Next Action

Replicate only eval-stable positive groups with independent training seeds. Retain sign-inconsistent groups as uncertain and exclude them from stable-positive source support until a separately preregistered repair succeeds.
