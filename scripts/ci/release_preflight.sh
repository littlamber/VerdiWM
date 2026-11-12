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
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
export UV_OFFLINE="${UV_OFFLINE:-1}"
if ! $allow_dirty && [[ -n "$(git status --porcelain=v1 --untracked-files=all)" ]]; then
  echo "release preflight requires a clean, fully tracked checkout; use --allow-dirty for a review snapshot" >&2
  exit 2
fi
if [[ -n "$output_dir" ]]; then
  uv run --frozen --no-sync python - "$output_dir" <<'CHECK'
from pathlib import Path
import sys
from wmloop.storage import checked_path
path = checked_path(Path(sys.argv[1]), code="RELEASE_OUTPUT_SYMLINK")
if path.exists():
    raise SystemExit("RELEASE_OUTPUT_EXISTS")
CHECK
  export VERDIWM_RELEASE_OUTPUT="$output_dir"
fi
exec bash scripts/ci/check_control_plane.sh
