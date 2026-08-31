#!/usr/bin/env bash
set -euo pipefail

build_dir="$(mktemp -d)"
trap 'rm -rf "$build_dir"' EXIT

# Validate the public wheel contract rather than files from the private
# development checkout.
uv build --wheel --out-dir "$build_dir"
wheel_path="$(find "$build_dir" -maxdepth 1 -type f -name 'verdiwm-*.whl' -print -quit)"
test -n "$wheel_path"
python - "$wheel_path" <<'PY'
import sys
import zipfile

names = set(zipfile.ZipFile(sys.argv[1]).namelist())
required = {
    "wmloop/__init__.py",
    "wmloop/cli.py",
    "wmloop/control/workbench.py",
    "wmloop/control/onboarding.py",
    "wmloop/execute/autonomous_pipeline.py",
    "wmloop/execute/experiment_scheduler.py",
    "configs/schemas/goal_spec.schema.json",
    "configs/schemas/adapter_profile.schema.json",
    "configs/retrieval/mechanism_tag_ontology_v1.json",
}
missing = sorted(required - names)
if missing:
    raise SystemExit(f"wheel is missing public files: {missing}")
PY

# Compile every Python file that is actually present in the public tree.
mapfile -d '' python_files < <(find wmloop experiments scripts -type f -name '*.py' -print0)
uv run python -m py_compile "${python_files[@]}"

# Run the tests shipped in this release.
uv run pytest -q tests

uv run python scripts/export/validate_public_example.py \
  examples/acwm_minimal_loop_cloth_next_forcing_v2
uv run python scripts/export/acwm_public_experience_bundle.py validate \
  --output-root examples/acwm_experience_atlas_v1
