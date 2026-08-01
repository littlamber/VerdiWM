# Probe Taxonomy

VerdiWM uses three probe layers. They are related, but they are not
interchangeable.

| Layer | Canonical artifact | Purpose |
|---|---|---|
| Outcome and verdict diagnostics | `acwm_v1.json` | Define observed coordinates such as horizon degradation, action following, appearance drift, and OoD gaps. These coordinates contribute to the outcome vector `z`; they are not intervention directions. |
| Canonical base intervention probes | `irg_base_v1.json` | Define the four cross-backbone semantic perturbation families used to initialize an IRG chart: action scaling, controlled context retention, first-frame anchoring strength, and sampler-noise stress. |
| Admitted IRG intervention paths | `../experiments/acwm_phys_joint_irg_autoregressive_pilot_v1.json` | Define the executable, locality-admitted columns used by one instantiated atlas. A path may split a base family by polarity or be added by counterexample-driven probe evolution. |

The ACWM-Phys v1 atlas has seven executable intervention paths. Those seven
paths are not seven base probes. In particular, action scaling is split into
amplification and attenuation, while additional mechanism-specific paths were
admitted during atlas construction. Controlled context retention and
sampler-noise stress are not yet materialized in that atlas; the existing
first-frame anchor repair hook has not yet been calibrated as a paired IRG
probe.

Code and papers must use the layer-qualified terms `outcome diagnostic`,
`base intervention probe`, and `admitted intervention path`.
