# Interoperability Profile

The Kernel intentionally defines a small Python protocol instead of depending
on a model framework. Mature community conventions can be adopted at the
edges:

- **W3C PROV mapping:** an experiment is a `prov:Activity`, the model portrait,
  intervention, and verifier output are `prov:Entity` objects, and the adapter
  is a `prov:Agent`. `Evidence.verifier_digest` and `claim_boundary` preserve
  provenance and scope without requiring a graph database.
- **OpenTelemetry boundary:** adapters may emit traces using `experiment_id`,
  `model_id`, `hypothesis_id`, `split`, and `outcome` attributes. Telemetry is
  observability only; promotion decisions remain Kernel evidence decisions.
- **RO-Crate-style packaging:** a future export can package the JSONL ledger,
  manifest, contract version, and checksums as portable research artifacts.
- **Evaluator interoperability:** an adapter may wrap HELM, lm-evaluation-
  harness, MLflow, or a domain evaluator, but those tools remain optional and
  outside the Kernel dependency set.

These are compatibility profiles, not runtime requirements. Any integration
must preserve deterministic identities, held-out verification, append-only
evidence, and explicit positive/null/harmful/abstain outcomes.

`verdi_core.providers` includes generic HTTP implementations for an
OpenAI-compatible chat endpoint and a JSON search endpoint. They are configured
by the caller; no API vendor, model family, or search provider is compiled into
the Kernel.
