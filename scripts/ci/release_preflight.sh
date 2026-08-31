#!/usr/bin/env bash
set -euo pipefail

allow_dirty=false
output_dir=""

usage() {
  echo "usage: $0 [--allow-dirty] [--output-dir PATH]" >&2
}

while (($#)); do
  case "$1" in
    --allow-dirty)
      allow_dirty=true
      shift
      ;;
    --output-dir)
      if (($# < 2)); then
        usage
        exit 2
      fi
      output_dir="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage
      exit 2
      ;;
  esac
done

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

public_release=false
if [[ -f .verdiwm-public-release ]]; then
  public_release=true
fi

command -v git >/dev/null || { echo "release preflight requires git" >&2; exit 2; }
command -v uv >/dev/null || { echo "release preflight requires uv" >&2; exit 2; }

if ! $allow_dirty; then
  status="$(git status --porcelain=v1 --untracked-files=all)"
  if [[ -n "$status" ]]; then
    echo "release preflight requires a clean, fully tracked checkout" >&2
    echo "$status" >&2
    exit 2
  fi
else
  echo "warning: validating a dirty development worktree; do not publish this result" >&2
fi

work_dir="$(mktemp -d)"
cleanup() {
  rm -rf "$work_dir"
}
trap cleanup EXIT

artifact_dir="$work_dir/artifacts"
staging_dir="$work_dir/modelscope-repository"
venv_dir="$work_dir/venv"
mkdir -p "$artifact_dir"

if $public_release; then
  scripts/ci/check_public_release.sh
else
  scripts/ci/check_control_plane.sh
fi
uv build --sdist --wheel --out-dir "$artifact_dir"

wheel_path="$(find "$artifact_dir" -maxdepth 1 -type f -name 'verdiwm-*.whl' -print -quit)"
sdist_path="$(find "$artifact_dir" -maxdepth 1 -type f -name 'verdiwm-*.tar.gz' -print -quit)"
test -n "$wheel_path"
test -n "$sdist_path"

python - "$wheel_path" "$sdist_path" "$public_release" <<'PY'
import sys
import tarfile
import zipfile

wheel_path, sdist_path, public_release_arg = sys.argv[1:]
wheel_required = {
    "wmloop/cli.py",
    "wmloop/geometry/memory.py",
    "wmloop/retrieve/evidence_capsule.py",
    "wmloop/retrieve/mechanism_discovery.py",
    "configs/retrieval/mechanism_tag_ontology_v1.json",
    "configs/retrieval/primitive_mechanism_profiles_v1.json",
}
with zipfile.ZipFile(wheel_path) as archive:
    wheel_names = set(archive.namelist())
missing = sorted(wheel_required - wheel_names)
if missing:
    raise SystemExit(f"wheel is missing release files: {missing}")

sdist_required = {
    "README_zh.md",
    "docs/EVIDENCE_CAPSULE.md",
    "docs/MECHANISM_DISCOVERY.md",
    "docs/RELEASE_CHECKLIST.md",
    "docs/TRANSFERABLE_EXPERIENCE.md",
    "scripts/ci/release_preflight.sh",
    "tests/test_evidence_capsule.py",
    "tests/test_mechanism_discovery.py",
    "tests/test_transferable_experience.py",
}
with tarfile.open(sdist_path, "r:gz") as archive:
    sdist_names = {
        name.split("/", 1)[1]
        for name in archive.getnames()
        if "/" in name
    }
missing = sorted(sdist_required - sdist_names)
if missing:
    raise SystemExit(f"sdist is missing release files: {missing}")

forbidden_exact = {
    ".hypothesis",
    "AGENTS.md",
    "MANIFEST.sha256",
    "RELEASE_AUDIT.json",
}
public_release = public_release_arg == "true"
forbidden_prefixes = (
    (".hypothesis/", "figures/", "ops/")
    if public_release
    else (".hypothesis/", "examples/", "figures/", "ops/")
)
unexpected = sorted(
    name
    for name in sdist_names
    if name in forbidden_exact
    or name.startswith(forbidden_prefixes)
    or name.endswith("_WORK_ORDER.md")
)
if unexpected:
    raise SystemExit(f"sdist contains repository-only files: {unexpected[:20]}")
PY

uv venv --python 3.10 "$venv_dir"
uv pip install --python "$venv_dir/bin/python" "$wheel_path"
"$venv_dir/bin/verdiwm" --help >/dev/null
"$venv_dir/bin/verdiwm" doctor >/dev/null
"$venv_dir/bin/python" - <<'PY'
from wmloop.geometry import build_transferable_experience
from wmloop.retrieve.evidence_capsule import build_evidence_capsule
from wmloop.retrieve.mechanism_discovery import DiscoveryRequest

assert callable(build_transferable_experience)
assert callable(build_evidence_capsule)
assert DiscoveryRequest is not None
PY

uv run python scripts/export/build_github_staging.py \
  --source-root "$repo_root" \
  --output-root "$staging_dir" >/dev/null

uv run python - "$staging_dir/RELEASE_AUDIT.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    audit = json.load(handle)
if audit.get("state") != "ready" or not all(audit.get("checks", {}).values()):
    raise SystemExit("release staging audit is not ready")
PY

staged_artifact_dir="$work_dir/public-tree-artifacts"
mkdir -p "$staged_artifact_dir"
staging_validation_env="$work_dir/public-tree-validation-venv"
(
  cd "$staging_dir"
  uv build --sdist --wheel --out-dir "$staged_artifact_dir"
  UV_PROJECT_ENVIRONMENT="$staging_validation_env" uv run python scripts/export/validate_portrait_first_public_example.py \
    examples/portrait_first_minimal_loop_v1 >/dev/null
)
staged_wheel_path="$(find "$staged_artifact_dir" -maxdepth 1 -type f -name 'verdiwm-*.whl' -print -quit)"
test -n "$staged_wheel_path"
staged_venv_dir="$work_dir/public-tree-venv"
uv venv --python 3.10 "$staged_venv_dir"
uv pip install --python "$staged_venv_dir/bin/python" "$staged_wheel_path"
"$staged_venv_dir/bin/verdiwm" --help >/dev/null
"$staged_venv_dir/bin/verdiwm" doctor >/dev/null
"$staged_venv_dir/bin/verdiwm-ctrl-world-autonomous-transfer" --help >/dev/null
test ! -e "$staging_dir/.venv"
python - "$staging_dir/MANIFEST.sha256" "$staging_dir" <<'PY'
import hashlib
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1]).resolve()
root = Path(sys.argv[2]).resolve()
declared = {}
for line in manifest_path.read_text(encoding="utf-8").splitlines():
    digest, relative = line.split("  ", 1)
    declared[relative] = digest
actual = {
    path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
    for path in root.rglob("*")
    if path.is_file() and path.name != "MANIFEST.sha256"
}
if declared != actual:
    raise SystemExit("release staging manifest does not cover final tree")
PY

if [[ -n "$output_dir" ]]; then
  if [[ -e "$output_dir" ]]; then
    echo "output path already exists: $output_dir" >&2
    exit 2
  fi
  mkdir -p "$output_dir"
  cp "$wheel_path" "$sdist_path" "$output_dir/"
  cp -a "$staging_dir" "$output_dir/repository"
  echo "release artifacts: $output_dir"
fi

echo "release preflight: ready"
