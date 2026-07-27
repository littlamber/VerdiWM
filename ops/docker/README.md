# wm-loop Docker execution host

This directory provisions a dedicated Docker socket for agent trial execution.
It does not use `/var/run/docker.sock`, does not alter the default Docker
daemon, and disables Docker bridge networking. Trial containers must request
`--network none`; a successful API handshake is never treated as a successful
execution sandbox.

## Host contract

The host must be a real systemd machine with a writable cgroup-v2 hierarchy,
`overlay2` support, and no active default `docker.service` or `docker.socket`.
The service's `Delegate=yes` lets `runc` create a cgroup below the service.
These are host-provisioning requirements, not settings an unprivileged nested
agent can emulate.

Install on such a host as root:

```bash
ops/docker/install-systemd.sh check
ops/docker/install-systemd.sh install
ops/docker/verify-runtime.sh <prebuilt-sandbox-image>
```

The verification image must already be available locally and must provide
`true` for uid/gid `65532`. It is run with no network, a read-only rootfs, all
Linux capabilities dropped, `no-new-privileges`, a 512 PID limit, 8 GiB memory
limit and 8 CPU limit. A passing result is the admission gate for executing an
agent-written trial in a container.

The Python `DockerExecutionBackend` follows the same contract and refuses to
run commands until its runtime probe has passed. This is only an interface
boundary on the current agent host; it does not make the nested container host
capable of running trial containers.

## Current host boundary

The current agent host is itself a container: PID 1 is not systemd and
`/sys/fs/cgroup` is mounted read-only. A temporary VFS/no-network daemon can
answer Docker API requests there, but `runc` cannot create
`/sys/fs/cgroup/docker`; it cannot execute containers. The installer therefore
fails closed before writing service files or attempting to retain a daemon.
