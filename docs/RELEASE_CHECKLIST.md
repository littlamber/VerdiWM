# Release Checklist

- [x] `python -m pytest -q` passes from a clean checkout.
- [x] `verdi doctor` reports a model-agnostic kernel.
- [x] `verdi demo` settles portrait, fingerprint, and three evidence outcomes.
- [x] Re-running the demo does not duplicate knowledge records.
- [x] The repository contains no weights, datasets, local paths, credentials,
      or model-specific historical experiment bundles.
- [x] A second adapter passes the same contract without Kernel changes.
- [ ] Only after this checklist passes may legacy development trees be archived
      or removed.
