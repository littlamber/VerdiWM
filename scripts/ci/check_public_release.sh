#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
export UV_OFFLINE="${UV_OFFLINE:-1}"

uv run --frozen --no-sync python -m py_compile \
  wmloop/control/model_portrait.py \
  wmloop/control/adaptive_observation.py \
  wmloop/control/capability_gap_planner.py \
  wmloop/control/experiment_portfolio.py \
  wmloop/control/module_manufacturing.py \
  wmloop/control/resource_portfolio.py \
  wmloop/control/model_batch.py \
  wmloop/experiments/community_bundle.py \
  wmloop/experiments/community_export.py \
  experiments/ctrl_world_autonomous_transfer_v1/controller.py \
  experiments/ctrl_world_autonomous_transfer_v1/workflow.py \
  examples/portrait_first_minimal_loop_v1/run.py

uv run --frozen --no-sync pytest -q \
  tests/test_verdiwm_public_release.py \
  tests/test_portrait_first_public_example.py

uv run --frozen --no-sync verdiwm doctor >/dev/null
uv run --frozen --no-sync verdiwm community export --help >/dev/null
uv run --frozen --no-sync verdiwm evidence project --help >/dev/null
uv run --frozen --no-sync verdiwm-workbench --help >/dev/null
uv run --frozen --no-sync verdiwm community publish --help >/dev/null
uv run --frozen --no-sync verdiwm community verify --help >/dev/null
uv run --frozen --no-sync verdiwm-ctrl-world-autonomous-transfer --help >/dev/null
uv run --frozen --no-sync python scripts/export/validate_portrait_first_public_example.py \
  examples/portrait_first_minimal_loop_v1 >/dev/null
