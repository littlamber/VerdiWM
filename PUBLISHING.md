# Publishing

This directory is the reviewable release candidate for GitHub and ModelScope.
Run `scripts/release_preflight.sh`, inspect the generated wheel with `uv build`,
and commit the clean tree before uploading. No remote push is performed by the
local build.

For GitHub, create an empty repository and push this directory's initial commit.
For ModelScope, create a Code repository and upload the same source tree; keep
weights and datasets in their own versioned artifacts. The README must state
that the fixture demo validates only the control-plane contract, not scientific
improvement on a target model.
