#!/usr/bin/env bash
set -euo pipefail

allow_dirty=false
output_dir=""
usage() { echo "usage: $0 [--allow-dirty] [--output-dir PATH]" >&2; }
while (($#)); do
  case "$1" in
    --allow-dirty) allow_dirty=true; shift ;;
    --output-dir) (($# >= 2)) || { usage; exit 2; }; output_dir="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage; exit 2 ;;
  esac
done

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
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
trap 'rm -rf "$work_dir"' EXIT
artifact_dir="$work_dir/artifacts"
staging_dir="$work_dir/source-repository"
venv_dir="$work_dir/venv"
mkdir -p "$artifact_dir"

scripts/ci/check_public_release.sh
uv build --sdist --wheel --out-dir "$artifact_dir"
wheel_path="$(find "$artifact_dir" -maxdepth 1 -type f -name 'verdiwm-*.whl' -print -quit)"
sdist_path="$(find "$artifact_dir" -maxdepth 1 -type f -name 'verdiwm-*.tar.gz' -print -quit)"
test -n "$wheel_path" -a -n "$sdist_path"

python - "$wheel_path" "$sdist_path" <<'PY'
import sys, tarfile, zipfile
wheel_path, sdist_path = sys.argv[1:]
wheel_required = {
    "wmloop/cli.py", "wmloop/geometry/memory.py",
    "wmloop/retrieve/evidence_capsule.py", "wmloop/retrieve/mechanism_discovery.py",
    "configs/retrieval/mechanism_tag_ontology_v1.json",
    "configs/retrieval/primitive_mechanism_profiles_v1.json",
}
with zipfile.ZipFile(wheel_path) as archive:
    missing = sorted(wheel_required - set(archive.namelist()))
    if missing: raise SystemExit(f"wheel is missing release files: {missing}")
sdist_required = {
    "README_zh.md", "docs/EVIDENCE_CAPSULE.md", "docs/MECHANISM_DISCOVERY.md",
    "docs/RELEASE_CHECKLIST.md", "docs/TRANSFERABLE_EXPERIENCE.md",
    "scripts/ci/release_preflight.sh", "tests/test_evidence_capsule.py",
    "tests/test_mechanism_discovery.py", "tests/test_transferable_experience.py",
}
with tarfile.open(sdist_path, "r:gz") as archive:
    names = {name.split("/", 1)[1] for name in archive.getnames() if "/" in name}
missing = sorted(sdist_required - names)
if missing: raise SystemExit(f"sdist is missing release files: {missing}")
allowed_examples = {
    "examples/portrait_first_minimal_loop_v1/README.md",
    "examples/portrait_first_minimal_loop_v1/run.py",
}
forbidden = {".hypothesis", "AGENTS.md", "MANIFEST.sha256", "RELEASE_AUDIT.json", ".audit-state"}
unexpected = sorted(name for name in names if (
    name in forbidden or name.startswith((".hypothesis/", ".audit-state/", "figures/"))
    or (name.startswith("examples/") and name not in allowed_examples)
    or name.endswith((".mp4", ".png", ".pdf", ".svg", ".pt", ".pth", ".ckpt", ".safetensors"))
))
if unexpected: raise SystemExit(f"sdist contains non-source files: {unexpected[:20]}")
PY

uv venv --python 3.10 "$venv_dir"
uv pip install --python "$venv_dir/bin/python" "$wheel_path"
"$venv_dir/bin/verdiwm" --help >/dev/null
"$venv_dir/bin/verdiwm" doctor >/dev/null

uv run python scripts/export/build_github_staging.py \
  --source-root "$repo_root" --output-root "$staging_dir" > "$work_dir/staging-audit.json"
python - "$work_dir/staging-audit.json" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as handle: audit = json.load(handle)
if audit.get("state") != "ready" or not all(audit.get("checks", {}).values()):
    raise SystemExit("release staging audit is not ready")
PY

staged_artifact_dir="$work_dir/staged-artifacts"
mkdir -p "$staged_artifact_dir"
staging_validation_env="$work_dir/staging-validation-venv"
(
  cd "$staging_dir"
  uv build --sdist --wheel --out-dir "$staged_artifact_dir"
  UV_PROJECT_ENVIRONMENT="$staging_validation_env" \
    uv run python scripts/export/validate_portrait_first_public_example.py \
    examples/portrait_first_minimal_loop_v1 >/dev/null
)
staged_wheel_path="$(find "$staged_artifact_dir" -maxdepth 1 -type f -name 'verdiwm-*.whl' -print -quit)"
test -n "$staged_wheel_path"
test ! -e "$staging_dir/.venv"

python - "$staging_dir" <<'PY'
import sys
from pathlib import Path
root = Path(sys.argv[1])
allowed = {
    "examples/portrait_first_minimal_loop_v1/README.md",
    "examples/portrait_first_minimal_loop_v1/run.py",
}
files = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
examples = {p for p in files if p.startswith("examples/")}
if examples != allowed: raise SystemExit(f"unexpected examples in staging: {sorted(examples - allowed)}")
forbidden = {"figures", ".audit-state", "RELEASE_AUDIT.json", "MANIFEST.sha256", ".verdiwm-public-release"}
if any(name in files or any(path.startswith(name + "/") for path in files) for name in forbidden):
    raise SystemExit("generated release artifacts present in staging")
if any(path.endswith((".mp4", ".png", ".pdf", ".svg", ".pt", ".pth", ".ckpt", ".safetensors")) for path in files):
    raise SystemExit("runtime media or weights present in staging")
PY

if [[ -n "$output_dir" ]]; then
  [[ ! -e "$output_dir" ]] || { echo "output path already exists: $output_dir" >&2; exit 2; }
  mkdir -p "$output_dir"
  cp "$wheel_path" "$sdist_path" "$output_dir/"
  cp -a "$staging_dir" "$output_dir/repository"
  echo "release artifacts: $output_dir"
fi
echo "release preflight: ready"
