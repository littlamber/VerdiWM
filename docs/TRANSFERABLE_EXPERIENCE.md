# Transferable Experience

VerdiWM now treats experience deposition as a first-class derived view of the
existing `EffectMemory` and `TransferCertificate` contracts. It does not add a
second evidence store.

`build_transferable_experience` preserves the source effect, context, measured
uncertainty, negative boundaries, evidence references, and certificate terms.
The resulting record has three useful states:

- `local_only`: a settled effect that is useful in its original context but has
  not earned reuse authority;
- `licensed_prior`: a confirmed effect whose existing transfer certificate is
  licensed and whose terms all pass;
- `abstained`: transfer was considered but a certificate did not license it.

All states remain retrievable. A `licensed_prior` is permission to schedule a
bounded target-side reuse experiment, not a replacement for target validation.
Rejected and null effects therefore remain valuable negative routing evidence.

```text
EffectRecord + TransferCertificate
        -> transferable experience projection
        -> ranking / bounded reuse experiment
        -> target-side confirmation
```

The projection is deterministic and content-derived through `experience_id`.
Archive/CAS receipts and the full Evidence Graph remain authoritative.

For cross-model reuse, `EffectMemory.portable_experiences()` emits a stricter
derived view. Its knowledge fields are semantic (model family, capability
class, goal/evaluator protocol, dataset regime, horizons, and primitive), and
its evidence references must be `cas://`, `urn:`, or `sha256:` values. Runtime
paths, repository names, checkpoint filenames, and local manifests are rejected
by the portability validator. Those bindings remain in execution receipts and
are never copied into shared knowledge.
