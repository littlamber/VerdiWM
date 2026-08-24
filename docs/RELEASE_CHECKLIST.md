# Release Checklist

- [x] `python -m pytest -q` passes from a clean checkout.
- [x] `verdi doctor` reports a model-agnostic kernel.
- [x] `verdi demo` settles portrait, fingerprint, and three evidence outcomes.
- [x] Re-running the demo does not duplicate knowledge records.
- [x] The repository contains no weights, datasets, local paths, credentials,
      or model-specific historical experiment bundles.
- [x] A second adapter passes the same contract without Kernel changes.
- [x] SQLite graph projection, query, export, and restart hydration pass.
- [x] HTML/code ingestion and PDF/unsupported-type degradation are explicit.
- [x] Local, Docker, and HTTP worker contracts are implemented.
- [x] Scheduler budget, retry, and SQLite resume paths pass.
- [x] Offline `verdi cycle` runs retrieval, dual extraction, benchmark review,
      scheduling, evaluation, and evidence projection without external APIs.
- [ ] A real external model adapter and real domain evaluator pass held-out
      scientific validation.
- [ ] External network providers pass live integration checks with credentials.
- [ ] Only after this checklist passes may legacy development trees be archived
      or removed.
