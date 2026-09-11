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
    "wmloop/control/research_state.py",
    "wmloop/control/mechanism_hypothesis.py",
    "wmloop/control/method_realization.py",
    "wmloop/control/open_method_study.py",
    "wmloop/control/open_method_generation.py",
    "wmloop/execute/open_method_runtime.py",
    "wmloop/execute/open_method_study_runner.py",
    "wmloop/control/method_calibration.py",
    "configs/schemas/research_state.schema.json",
    "configs/schemas/mechanism_hypothesis.schema.json",
    "configs/schemas/method_implementation_validation.schema.json",
    "configs/schemas/open_method_composition.schema.json",
    "configs/schemas/open_method_evaluation.schema.json",
    "configs/schemas/open_method_study_execution.schema.json",
    "configs/schemas/open_method_verifier.schema.json",
    "wmloop/control/onboarding.py",
    "wmloop/execute/autonomous_pipeline.py",
    "wmloop/execute/experiment_scheduler.py",
    "wmloop/experiments/community_bundle.py",
    "wmloop/experiments/community_export.py",
    "configs/schemas/goal_spec.schema.json",
    "configs/schemas/adapter_profile.schema.json",
    "configs/schemas/model_batch_request.schema.json",
    "configs/schemas/model_batch_plan.schema.json",
    "configs/schemas/model_batch_execution.schema.json",
    "configs/schemas/model_batch_status.schema.json",
    "configs/schemas/community_bundle.schema.json",
    "configs/schemas/community_bundle_signature.schema.json",
    "configs/schemas/community_export.schema.json",
    "configs/schemas/settled_evidence.schema.json",
    "wmloop/archive/community_projection.py",
    "wmloop/control/campaign_repository.py",
    "configs/retrieval/mechanism_tag_ontology_v1.json",
}
missing = sorted(required - names)
if missing:
    raise SystemExit(f"wheel is missing public files: {missing}")
PY

python scripts/ci/check_installed_wheel.py "$wheel_path"

# Compile every Python file that is actually present in the public tree.
mapfile -d '' python_files < <(find wmloop experiments scripts -type f -name '*.py' -print0)
uv run python -m py_compile "${python_files[@]}"

# Run the tests shipped in this release.
uv run pytest -q tests

uv run python scripts/export/validate_public_example.py \
  examples/acwm_minimal_loop_cloth_next_forcing_v2
uv run python scripts/export/acwm_public_experience_bundle.py validate \
  --output-root examples/acwm_experience_atlas_v1
