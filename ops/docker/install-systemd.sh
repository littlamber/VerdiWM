#!/usr/bin/env bash
# Install only on a real systemd host with a writable cgroup-v2 hierarchy.
# The agent container intentionally fails this check: a Docker API process
# without cgroup delegation cannot safely start trial containers.
set -Eeuo pipefail

readonly script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly config_dir="/etc/wm-loop/docker"
readonly unit_dir="/etc/systemd/system"
readonly socket_name="wm-loop-docker.socket"
readonly service_name="wm-loop-docker.service"
readonly socket_path="/run/wm-loop-docker/docker.sock"
readonly service_group="wmloop-docker"

mode="check"
if (($# == 1)); then
  mode="$1"
fi
if [[ "$mode" != "check" && "$mode" != "install" ]]; then
  printf '%s\n' 'usage: install-systemd.sh [check|install]' >&2
  exit 64
fi

fail() {
  printf '%s\n' "$1" >&2
  exit 1
}

require_systemd_host() {
  [[ -d /run/systemd/system ]] || fail 'WM_LOOP_DOCKER_SYSTEMD_UNAVAILABLE'
  command -v systemctl >/dev/null || fail 'WM_LOOP_DOCKER_SYSTEMCTL_MISSING'
  [[ "$(cat /proc/1/comm 2>/dev/null || true)" == "systemd" ]] || fail 'WM_LOOP_DOCKER_PID1_NOT_SYSTEMD'
}

require_writable_cgroup_v2() {
  command -v findmnt >/dev/null || fail 'WM_LOOP_DOCKER_FINDMNT_MISSING'
  local fstype options
  fstype="$(findmnt -n -o FSTYPE -T /sys/fs/cgroup)"
  options="$(findmnt -n -o OPTIONS -T /sys/fs/cgroup)"
  [[ "$fstype" == "cgroup2" ]] || fail 'WM_LOOP_DOCKER_CGROUP_V2_REQUIRED'
  [[ ",$options," == *,rw,* ]] || fail 'WM_LOOP_DOCKER_CGROUP_NOT_WRITABLE'
}

require_daemon_prerequisites() {
  command -v dockerd >/dev/null || fail 'WM_LOOP_DOCKERD_MISSING'
  command -v docker >/dev/null || fail 'WM_LOOP_DOCKER_CLIENT_MISSING'
  grep -qx 'nodev\toverlay' /proc/filesystems || fail 'WM_LOOP_DOCKER_OVERLAY2_UNAVAILABLE'
  if systemctl is-active --quiet docker.service || systemctl is-active --quiet docker.socket; then
    fail 'WM_LOOP_DOCKER_DEFAULT_DAEMON_ACTIVE'
  fi
}

validate_assets() {
  dockerd --validate --config-file "$script_dir/daemon.json" >/dev/null
  systemd-analyze verify "$script_dir/$socket_name" "$script_dir/$service_name" >/dev/null
}

require_systemd_host
require_writable_cgroup_v2
require_daemon_prerequisites
validate_assets

if [[ "$mode" == "check" ]]; then
  printf '%s\n' 'WM_LOOP_DOCKER_HOST_READY'
  exit 0
fi

[[ "${EUID}" -eq 0 ]] || fail 'WM_LOOP_DOCKER_ROOT_REQUIRED'
if ! getent group "$service_group" >/dev/null; then
  groupadd --system "$service_group"
fi
install -d -m 0750 "$config_dir" /var/lib/wm-loop/docker
install -m 0644 "$script_dir/daemon.json" "$config_dir/daemon.json"
install -m 0644 "$script_dir/$socket_name" "$unit_dir/$socket_name"
install -m 0644 "$script_dir/$service_name" "$unit_dir/$service_name"
systemctl daemon-reload
systemctl enable --now "$socket_name"
for _ in $(seq 1 50); do
  if docker --host "unix://$socket_path" info >/dev/null 2>&1; then
    printf '%s\n' "WM_LOOP_DOCKER_READY socket=$socket_path"
    exit 0
  fi
  sleep 0.1
done
systemctl status "$socket_name" "$service_name" --no-pager >&2 || true
fail 'WM_LOOP_DOCKER_DAEMON_START_FAILED'
