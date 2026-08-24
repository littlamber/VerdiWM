#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
python_bin="${PYTHON:-python3}"

"$python_bin" -m pytest -q
"$python_bin" -m verdi_core.cli doctor
state_dir="$(mktemp -d)"
trap 'rm -rf "$state_dir"' EXIT
"$python_bin" -m verdi_core.cli demo --state-root "$state_dir" >/dev/null
"$python_bin" -m verdi_core.cli demo --state-root "$state_dir" >/dev/null
"$python_bin" -m verdi_core.cli graph --state-root "$state_dir"

if rg -n -i "(/share/project|/home/|[A-Za-z]:\\\\|BEGIN (RSA|OPENSSH|EC) PRIVATE KEY|api[_-]?key\\s*[:=]\\s*[\"']|secret[_-]?key\\s*[:=]\\s*[\"']|password\\s*=\\s*[\"'])" verdi_core adapters tests docs README* CONTRIBUTING.md SECURITY.md; then
  echo "release preflight: forbidden path or secret-like content found" >&2
  exit 1
fi
test "$(wc -l < "$state_dir/knowledge/knowledge.jsonl")" -eq 3
echo "release preflight: PASS"
