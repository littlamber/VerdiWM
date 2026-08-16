# Mechanism Discovery Beyond Keyword Matching

The literature pipeline separates candidate collection from mechanism and
novelty assessment. A keyword hit can add a paper to the candidate pool, but a
missing keyword is never evidence that a mechanism is new.

## Pipeline

```text
diagnostic symptom, metrics, hooks, budget
  -> diagnostic, operator, architecture, and cross-domain queries
  -> explicit seed papers plus bounded arXiv search
  -> backward citation expansion from full text
  -> local-source fallback for unavailable papers
  -> evidence-bound mechanism extraction
  -> operator-axis comparison against every registered primitive
  -> unresolved/equivalent/extension/composition/structural-candidate
```

The mechanism representation has eight axes:

1. `state_representation`
2. `conditioning_path`
3. `update_operator`
4. `reliability_routing`
5. `training_distribution`
6. `learning_signal`
7. `gradient_path`
8. `inference_transition`

## Complexity Budget

The default discovery profile is `light`. It allows at most 12 papers, three
search results per view, and the four core diagnostic/operator/architecture
views. Cross-domain query lenses remain available in the `full` profile, but
are not silently mixed into the normal route. The selected budget and whether
query views were truncated are recorded in the atlas report and input hash.

Use `--complexity-budget full` only for a deliberate audit or a new research
campaign. This limits retrieval work; it does not change the evidence or
promotion boundary, and it never turns a paper into executable authority.

Titles and method names do not participate in structural similarity. Reviewed
or model-produced annotations must quote text that exists in the staged
abstract or full text. An annotation with an invented or unavailable quote
fails closed. If fewer than four axes have evidence-supported semantics, the
novelty state remains `unresolved`.

Axis tags are canonicalized through
`configs/retrieval/mechanism_tag_ontology_v1.json` before comparison. This
prevents lexical aliases such as `generated_history` and
`self_generated_history` from being treated as unrelated mechanisms. Unknown
tags remain distinct and therefore cannot silently inherit a known concept.

Comparison can include three reference classes:

1. registered VERDIWM primitives;
2. evidence-supported entries from prior mechanism atlases;
3. settled target-side mechanisms, including candidates that failed promotion.

The CLI accepts repeatable `--reference-atlas` and `--reference-profiles`
arguments. A new paper is not screened as structurally new merely because it
uses different terminology from prior literature or from an already attempted
target-side module.

The comparison label is a research-screening label, not a publication novelty
claim. A candidate may be admitted to implementation only after source,
revision, license, target hook, falsification criterion, and target-side
evaluation are complete.

## Local Source Completion

Network and parser failures produce both `missing-sources.json` and
`MISSING_SOURCES.md`. Each record includes the exact arXiv revision, PDF and
HTML URLs, accepted filenames, destination directory, and failure reason.

The accepted local formats are:

```text
<arxiv-id-with-revision>.txt
<arxiv-id-with-revision>.html
<arxiv-id-with-revision>.pdf
<base-arxiv-id>.txt
<base-arxiv-id>.html
<base-arxiv-id>.pdf
```

TXT and HTML sources are parsed with the standard library. PDF sources require
`pdftotext`; if it is unavailable, the record remains missing with reason
`local_pdf_parser_unavailable`. Every local file's path, size, and SHA256 enter
the discovery input hash and atlas provenance.

For Ctrl-World the shared local source directory is:

```text
/path/to/verdiwm-runs/ctrl-world/literature-sources
```

## Ctrl-World Run

The original history and its reviewed annotations remain available:

```text
configs/retrieval/ctrl_world_mechanism_discovery_v1.json
configs/retrieval/ctrl_world_mechanism_annotations_v1.json
```

The current failure-specific retrieval chain is:

```text
configs/retrieval/ctrl_world_cclvr_risk_calibrated_router_mechanism_discovery_v1.json
configs/retrieval/ctrl_world_cclvr_risk_calibrated_router_mechanism_discovery_v2.json
configs/retrieval/ctrl_world_cclvr_risk_calibrated_router_mechanism_annotations_v1.json
configs/retrieval/ctrl_world_cclvr_risk_calibrated_router_mechanism_annotations_v2.json
```

The immutable atlas outputs are:

```text
/path/to/verdiwm-runs/ctrl-world/mechanism-discovery-v10
/path/to/verdiwm-runs/ctrl-world/mechanism-discovery-v11
/path/to/verdiwm-runs/ctrl-world/mechanism-discovery-v12
/path/to/verdiwm-runs/ctrl-world/mechanism-discovery-v13
```

v11 contains 80 bounded papers from the broad failure-specific search and has
zero missing sources. v12 is the first targeted evidence pass; it deliberately
records one fail-closed annotation-evidence miss. v13 repeats that targeted
pass with exact arXiv revisions and has 13 papers, five newly
evidence-supported structural candidates, and zero missing sources. v12 and
v13 should be read together; neither authorizes training.

The next design-only work order is:

```text
configs/primitives/ctrl_world_conservative_distributional_residual_policy_v1.json
docs/CTRL_WORLD_CONSERVATIVE_DISTRIBUTIONAL_RESIDUAL_POLICY_WORK_ORDER.md
```

Known limitation: citation expansion is backward-only because available
forward-citation APIs have been rate-limited during earlier runs. These atlases
must not be described as complete citation graphs or publication-level novelty
proof.
