# Release Checklist

- [ ] `python -m pytest -q` passes from a clean checkout.
- [ ] `verdi doctor` reports a model-agnostic kernel.
- [ ] `verdi demo` settles portrait, fingerprint, and three evidence outcomes.
- [ ] Re-running the demo does not duplicate knowledge records.
- [ ] The repository contains no weights, datasets, local paths, credentials,
      or model-specific historical experiment bundles.
- [ ] A second adapter passes the same contract without Kernel changes.
- [ ] Only after this checklist passes may legacy development trees be archived
      or removed.

