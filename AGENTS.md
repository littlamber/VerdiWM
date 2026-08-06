# VerdiWM Engineering Rules

These rules apply to all changes in this checkout. They are intentionally
compatible with the public Google Engineering Practices: make changes small,
reviewable, testable, and explicit about ownership and failure modes.

## Design and code

- Prefer a small, single-purpose module and a narrow interface over a global
  framework or a second copy of an existing contract.
- Preserve immutable evidence, frozen evaluators, held-out splits, and the
  distinction between exploratory and claim-authoritative records.
- Validate external input at the boundary. Fail closed with a stable error
  code; never substitute a weaker implementation silently.
- Keep side effects explicit. GPU allocation, process launch, archive writes,
  and cleanup require a receipt or a durable marker.
- Use deterministic ordering, canonical JSON, content hashes, and idempotent
  writes whenever an operation may be resumed.
- Comments should explain a non-obvious invariant or trade-off, not narrate
  ordinary code.

## Tests and verification

- Every new behavior needs focused unit tests. Shared orchestration needs an
  integration test covering success, failure, resume, and cleanup.
- Do not claim CUDA execution from import success. A GPU test must record the
  physical GPU identity, observed activity, exit status, and output artifact.
- Do not claim a formal effect from a screen result. Only the frozen verifier
  may promote a result.
- Run the focused tests, `bash scripts/ci/check_control_plane.sh`, and
  `git diff --check` before reporting completion.

## Experiments and artifacts

- A plan must state the objective, hypothesis, falsification criterion,
  expected cost, selection reason, dependencies, and artifact policy before a
  GPU process is admitted.
- All terminal trials are archived. Only records that pass their declared
  verifier may affect reusable optimization memory.
- Scratch files may be removed only after the durable receipt and every
  declared artifact have been content-addressed. Never delete raw datasets,
  checkpoints, source trees, or another campaign's output.

## Change management

- Keep public APIs backward compatible within a minor version. Add a schema
  version when changing a persisted contract.
- Never rewrite history, existing evidence, or user changes. Use a new
  campaign/release version for protocol changes.
- Add or update a focused design note when a change crosses module ownership,
  alters a persisted schema, or changes GPU admission policy.
