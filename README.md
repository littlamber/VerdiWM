# VerdiWM Clean v0.1.0

VerdiWM is a lightweight, model-agnostic control plane for automatically
testing world-model improvement hypotheses and retaining transferable evidence.
This release is intentionally small. It proves the control-plane loop with a
CPU fixture adapter; model-specific runtimes are external adapters and are not
part of the kernel.

The kernel owns contracts, receipts, budgets, experiment state, verification,
and knowledge projection. An adapter owns model loading, inference, probes, and
interventions. A knowledge record contains semantic identities and evidence
references, never local paths or credentials.

The complete composition layer is included in `verdi_core`: retrieval with
HTML/PDF and human handoff, autonomous search planning, dual-route idea
extraction, AI-assisted metric selection, probe evolution, and a replaceable
scheduler. See `docs/FULL_LOOP.md` for the lifecycle. These modules are
provider-agnostic; only the model SDK and optional AI/search providers are
injected by the user.

## Quick start

```bash
python -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/verdi doctor
.venv/bin/verdi demo --state-root state/demo
.venv/bin/verdi graph --state-root state/demo
```

The demo runs: onboarding -> probe fingerprint -> model portrait -> paired
screen -> frozen verification -> positive/null/harmful knowledge projection.
It is a contract test, not a claim about a real model.

## Adapter boundary

An adapter implements `inspect`, `probe`, `evaluate`, and `intervene` methods
behind the small `ModelAdapter` protocol. Capability level L0 only needs
`inspect` and `evaluate`; L1 adds probes; L2 adds interventions; L3 adds
reproduction/export. Adding an adapter must not require editing the kernel.

## Scope and evidence

The release intentionally does not include Ctrl-World, Cosmos, model weights,
datasets, GPU launchers, or historical experiment bundles. Those belong in
separate adapter/domain repositories after this clean loop passes publication
and restart tests.
