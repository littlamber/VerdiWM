#!/usr/bin/env bash
# A daemon socket is not sufficient evidence: this verifies runc can start a
# networkless, read-only, capability-dropped container from the chosen image.
set -Eeuo pipefail

if (($# != 1)); then
  printf '%s\n' 'usage: verify-runtime.sh <sandbox-image>' >&2
  exit 64
fi

readonly image="$1"
readonly socket_path="${WM_LOOP_DOCKER_SOCKET:-/run/wm-loop-docker/docker.sock}"
readonly endpoint="unix://$socket_path"

[[ -S "$socket_path" ]] || { printf '%s\n' 'WM_LOOP_DOCKER_SOCKET_MISSING' >&2; exit 1; }
docker --host "$endpoint" info >/dev/null || { printf '%s\n' 'WM_LOOP_DOCKER_DAEMON_UNREACHABLE' >&2; exit 1; }
docker --host "$endpoint" run --pull=never --rm --network none --read-only \
  --cap-drop ALL --security-opt no-new-privileges --pids-limit 512 \
  --memory 8g --cpus 8 --user 65532:65532 "$image" true
printf '%s\n' "WM_LOOP_DOCKER_RUNTIME_READY image=$image"
