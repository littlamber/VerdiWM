#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
export UV_OFFLINE="${UV_OFFLINE:-1}"
test -f uv.lock
test -f pyproject.toml
uv lock --check

build_dir="$(mktemp -d)"
trap 'rm -rf "$build_dir"' EXIT

# Keep the source tree predictable before spending time on wheel checks.
uv run --offline --frozen --no-sync python scripts/ci/check_repository_layout.py

# Validate the public wheel contract rather than files from the private
# development checkout.
uv run --frozen --no-sync python -m scripts.export.build_github_staging --source-root . --output-root "$build_dir/source" --include-untracked --report "$build_dir/source_report.json"
uv build --wheel --sdist --out-dir "$build_dir" "$build_dir/source"
wheel_path="$(find "$build_dir" -maxdepth 1 -type f -name 'verdiwm-*.whl' -print -quit)"
test -n "$wheel_path"
sdist_path="$(find "$build_dir" -maxdepth 1 -type f -name 'verdiwm-*.tar.gz' -print -quit)"
test -n "$sdist_path"
uv run --frozen --no-sync python -m scripts.ci.check_release_archives "$wheel_path" "$sdist_path" --report "$build_dir/archive_report.json"

uv run --frozen --no-sync python scripts/ci/check_installed_wheel.py "$wheel_path" --report "$build_dir/install_locked.json"
uv run --frozen --no-sync python scripts/ci/check_installed_wheel.py "$wheel_path" --dependency-mode resolved --report "$build_dir/install_resolved.json"

# Compile every Python file that is actually present in the public tree.
mapfile -d '' python_files < <(find wmloop experiments scripts -type f -name '*.py' -print0)
uv run --frozen --no-sync python -m py_compile "${python_files[@]}"

# Run the tests shipped in this release.
(
  cd "$build_dir/source"
  export UV_PROJECT_ENVIRONMENT="$build_dir/test-env"
  uv sync --frozen --group dev
  uv run --frozen --no-sync pytest -q tests --junitxml="$build_dir/pytest.xml"
)

uv run --frozen --no-sync python scripts/export/validate_portrait_first_public_example.py \
  examples/portrait_first_minimal_loop_v1 >/dev/null

if [[ -n "${VERDIWM_RELEASE_OUTPUT:-}" ]]; then
  uv run --frozen --no-sync python -m scripts.export.build_release_delivery --build-dir "$build_dir" --output-dir "$VERDIWM_RELEASE_OUTPUT"
fi
