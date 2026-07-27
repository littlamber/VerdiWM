# Security Policy

## Reporting

Report vulnerabilities privately to the repository owner rather than opening a
public issue. Include the affected revision, reproduction steps, impact, and a
minimal proof of concept when possible.

## Execution warning

VerdiWM can stage and execute agent-generated training code. A Git worktree is
an audit boundary, not an operating-system sandbox. Run untrusted candidates in
a dedicated container or equivalent isolation with:

- no network unless explicitly required;
- a read-only source and evaluator mount;
- a writable per-trial output directory only;
- dropped Linux capabilities and no host credentials;
- CPU, memory, process, wall-clock, disk, and GPU limits;
- an explicit allowlist for commands and artifact collection.

Never expose production secrets or personal datasets to proposal providers.
Frozen hashes and receipts improve auditability but do not replace process
isolation.

