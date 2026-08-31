# Release checklist

This checklist is the publication gate for GitHub, ModelScope, and Python
artifacts. It validates packaging and public examples; it does not claim that
VerdiWM has achieved unconstrained recursive self-improvement.

## Before the gate

1. Review every untracked file and add only intended source, configuration,
   tests, and documentation to Git.
2. Commit the release candidate and start from a clean checkout. A local build
   from untracked files is not reproducible.
3. Confirm that model weights, datasets, run outputs, credentials, private
   paths, and vendor checkouts are absent.

## Run the gate

Install `uv`, then run:

```bash
scripts/ci/release_preflight.sh --output-dir dist/verdiwm-release
```

The command checks the control-plane suite against the locked dependencies,
builds the wheel and source distribution, installs the local wheel and its
declared runtime dependencies in a fresh Python 3.10 environment, exercises
the CLI and public APIs, builds an allowlisted ModelScope/GitHub repository,
and audits it for local paths, secrets, unsafe file types, symlinks, and
oversized files.
It also builds the staged repository with its public packaging manifest and
runs the CPU-only portrait-first example, so the upload candidate is usable
without the development-only source tree.

During development only, `--allow-dirty` can validate the current worktree.
Its output is not a publishable release because uncommitted and untracked
inputs cannot be reconstructed from Git.

## Publish

Upload `dist/verdiwm-release/repository/` as the repository content. Publish
the wheel and source archive beside it when the target platform supports
Python artifacts. Preserve `RELEASE_AUDIT.json` and `MANIFEST.sha256`; they are
the integrity and public-surface receipts for the staged tree.

Do not add local deployment configs, state roots, model weights, datasets,
credentials, private paths, or vendor checkouts back into the staged tree.
Public IRs and manifests remain path-independent; a deployment supplies local
asset bindings and credentials outside the published repository.

After upload, clone the public repository into a new directory and run the
same preflight without `--allow-dirty`. This final clone test is the proof that
the release does not depend on local state.
