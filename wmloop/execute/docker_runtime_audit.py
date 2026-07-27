"""Audit the Docker execution-sandbox contract before agent code can use it."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from wmloop.archive.store import ArchiveStore, ContentAddressedStore
from wmloop.execute.docker_backend import DockerBackendError, DockerExecutionBackend


class DockerRuntimeAuditError(RuntimeError):
    """Docker runtime audit inputs or outputs failed closed."""


Runner = Callable[..., subprocess.CompletedProcess[bytes]]


def run_docker_runtime_audit(
    *,
    assets_root: Path,
    output_root: Path,
    image: str,
    socket_path: Path = Path("/run/wm-loop-docker/docker.sock"),
    skip_live_probe: bool = False,
    archive_db: Path | None = None,
    cas_root: Path | None = None,
    runner: Runner = subprocess.run,
    require_socket: bool = True,
) -> dict[str, object]:
    """Write a CAS-backed audit for the dedicated Docker sandbox runtime."""

    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise DockerRuntimeAuditError("DOCKER_RUNTIME_AUDIT_OUTPUT_EXISTS")
    assets = Path(assets_root).resolve(strict=True)
    if not assets.is_dir() or assets.is_symlink():
        raise DockerRuntimeAuditError("DOCKER_RUNTIME_AUDIT_ASSETS_INVALID")
    static = _audit_static_assets(assets)
    live = _audit_live_runtime(
        image=image,
        socket_path=Path(socket_path),
        skip_live_probe=skip_live_probe,
        runner=runner,
        require_socket=require_socket,
    )
    blockers = [*static["blockers"], *live["blockers"]]
    execution_sandbox_ready = static["state"] == "ready" and live["state"] == "ready"
    state = "ready" if execution_sandbox_ready else "blocked"
    report = {
        "schema_version": 1,
        "artifact_type": "wmloop-docker-runtime-audit",
        "state": state,
        "assets_root": str(assets),
        "socket_path": str(Path(socket_path)),
        "image": image,
        "static_contract_ready": static["state"] == "ready",
        "live_probe_ready": live["state"] == "ready",
        "execution_sandbox_ready": execution_sandbox_ready,
        "blocker_count": len(blockers),
        "blockers": blockers,
        "static_contract": static,
        "live_probe": live,
        "required_contract": [
            "Use only the dedicated wm-loop Docker socket, never /var/run/docker.sock.",
            "Verify the daemon is reachable before any agent-written command can run.",
            "Actually start a prebuilt local image with --network none, --read-only, --cap-drop ALL, no-new-privileges, PID, memory and CPU limits.",
            "Treat a missing socket, unreachable daemon, or failed probe container as blocked.",
        ],
        "m4_launch_allowed": False,
        "formal_training_allowed": False,
        "limitations": [
            "This audit does not install or start a Docker daemon.",
            "A static asset pass is not sufficient for container execution; the live probe must also pass.",
            "This audit never authorizes M4 training; formal training still requires a ready strict phase gate.",
        ],
    }
    return _write_report_bundle(report=report, output_root=destination, archive_db=archive_db, cas_root=cas_root)


def _audit_static_assets(assets: Path) -> dict[str, object]:
    blockers: list[dict[str, object]] = []
    asset_records = {
        name: _asset_record(assets / name)
        for name in (
            "daemon.json",
            "wm-loop-docker.socket",
            "wm-loop-docker.service",
            "install-systemd.sh",
            "verify-runtime.sh",
        )
    }
    for name, record in asset_records.items():
        if record["state"] != "ready":
            blockers.append({"surface": "asset", "asset": name, "reason": record["state"]})

    daemon = _load_daemon_config(assets / "daemon.json", blockers)
    if daemon is not None:
        _expect_equal(blockers, daemon, "data-root", "/var/lib/wm-loop/docker")
        _expect_equal(blockers, daemon, "exec-root", "/run/wm-loop-docker")
        _expect_equal(blockers, daemon, "storage-driver", "overlay2")
        _expect_equal(blockers, daemon, "bridge", "none")
        _expect_equal(blockers, daemon, "iptables", False)
        _expect_equal(blockers, daemon, "ip6tables", False)
        _expect_equal(blockers, daemon, "ip-forward", False)
        _expect_equal(blockers, daemon, "ip-masq", False)
        _expect_equal(blockers, daemon, "userland-proxy", False)
        _expect_equal(blockers, daemon, "default-cgroupns-mode", "private")
        _expect_equal(blockers, daemon, "log-driver", "local")
        _expect_equal(blockers, daemon, "live-restore", True)

    socket_text = _read_text_or_none(assets / "wm-loop-docker.socket", blockers, "socket_unit")
    service_text = _read_text_or_none(assets / "wm-loop-docker.service", blockers, "service_unit")
    installer_text = _read_text_or_none(assets / "install-systemd.sh", blockers, "installer")
    verifier_text = _read_text_or_none(assets / "verify-runtime.sh", blockers, "runtime_verifier")
    _require_text(blockers, socket_text, "socket_unit", "ListenStream=/run/wm-loop-docker/docker.sock")
    _require_text(blockers, socket_text, "socket_unit", "SocketGroup=wmloop-docker")
    _require_text(blockers, service_text, "service_unit", "--host=fd://")
    _require_text(blockers, service_text, "service_unit", "Delegate=yes")
    _require_text(blockers, service_text, "service_unit", "TasksMax=infinity")
    _forbid_text(blockers, (socket_text or "") + (service_text or ""), "systemd_units", "/var/run/docker.sock")
    _require_text(blockers, installer_text, "installer", "WM_LOOP_DOCKER_PID1_NOT_SYSTEMD")
    _require_text(blockers, installer_text, "installer", "WM_LOOP_DOCKER_CGROUP_NOT_WRITABLE")
    _require_text(blockers, installer_text, "installer", "dockerd --validate")
    _require_text(blockers, installer_text, "installer", "systemd-analyze verify")
    for token in (
        "--network none",
        "--read-only",
        "--cap-drop ALL",
        "no-new-privileges",
        "--pids-limit 512",
        "--memory 8g",
        "--cpus 8",
        "--user 65532:65532",
        "WM_LOOP_DOCKER_SOCKET_MISSING",
        "WM_LOOP_DOCKER_DAEMON_UNREACHABLE",
    ):
        _require_text(blockers, verifier_text, "runtime_verifier", token)

    return {
        "state": "ready" if not blockers else "blocked",
        "asset_count": len(asset_records),
        "assets": asset_records,
        "blocker_count": len(blockers),
        "blockers": blockers,
    }


def _audit_live_runtime(
    *,
    image: str,
    socket_path: Path,
    skip_live_probe: bool,
    runner: Runner,
    require_socket: bool,
) -> dict[str, object]:
    if skip_live_probe:
        return {
            "state": "skipped",
            "socket_path": str(socket_path),
            "image": image,
            "executed_container_probe": False,
            "commands": [],
            "blockers": [{"surface": "live_probe", "reason": "DOCKER_RUNTIME_AUDIT_LIVE_PROBE_SKIPPED"}],
        }
    commands: list[dict[str, object]] = []

    def recording_runner(command: Sequence[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        completed = runner(command, **kwargs)
        commands.append(
            {
                "argv": list(command),
                "returncode": completed.returncode,
                "stdout_size": len(completed.stdout or b""),
                "stderr_size": len(completed.stderr or b""),
                "stderr_tail": _decode_tail(completed.stderr),
            }
        )
        return completed

    backend = DockerExecutionBackend(image=image, socket_path=socket_path)
    try:
        receipt = backend.verify_runtime(runner=recording_runner, require_socket=require_socket)
    except DockerBackendError as exc:
        return {
            "state": "blocked",
            "socket_path": str(socket_path),
            "image": image,
            "executed_container_probe": any("run" in command.get("argv", []) for command in commands),
            "commands": commands,
            "blockers": [{"surface": "live_probe", "reason": str(exc)}],
        }
    return {
        "state": "ready",
        "socket_path": str(socket_path),
        "image": image,
        "executed_container_probe": True,
        "runtime_receipt": receipt.to_document(),
        "commands": commands,
        "blockers": [],
    }


def _asset_record(path: Path) -> dict[str, object]:
    if path.is_symlink():
        return {"path": str(path), "state": "symlink"}
    if not path.is_file():
        return {"path": str(path), "state": "missing"}
    payload = path.read_bytes()
    return {
        "path": str(path),
        "state": "ready",
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _load_daemon_config(path: Path, blockers: list[dict[str, object]]) -> Mapping[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        blockers.append({"surface": "daemon.json", "reason": "DAEMON_CONFIG_UNREADABLE", "error_type": type(exc).__name__})
        return None
    if not isinstance(payload, Mapping):
        blockers.append({"surface": "daemon.json", "reason": "DAEMON_CONFIG_NOT_OBJECT"})
        return None
    return payload


def _read_text_or_none(path: Path, blockers: list[dict[str, object]], surface: str) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        blockers.append({"surface": surface, "reason": "ASSET_UNREADABLE", "error_type": type(exc).__name__})
        return None


def _expect_equal(blockers: list[dict[str, object]], payload: Mapping[str, Any], key: str, expected: object) -> None:
    observed = payload.get(key)
    if observed != expected:
        blockers.append({"surface": "daemon.json", "field": key, "expected": expected, "observed": observed})


def _require_text(blockers: list[dict[str, object]], text: str | None, surface: str, token: str) -> None:
    if text is None:
        return
    if token not in text:
        blockers.append({"surface": surface, "reason": "REQUIRED_TOKEN_MISSING", "token": token})


def _forbid_text(blockers: list[dict[str, object]], text: str, surface: str, token: str) -> None:
    if token in text:
        blockers.append({"surface": surface, "reason": "FORBIDDEN_TOKEN_PRESENT", "token": token})


def _decode_tail(payload: bytes | str | None, limit: int = 500) -> str:
    if payload is None:
        return ""
    if isinstance(payload, str):
        text = payload
    else:
        text = payload.decode("utf-8", errors="replace")
    return text[-limit:]


def _write_report_bundle(
    *,
    report: Mapping[str, object],
    output_root: Path,
    archive_db: Path | None,
    cas_root: Path | None,
) -> dict[str, object]:
    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise DockerRuntimeAuditError("DOCKER_RUNTIME_AUDIT_OUTPUT_EXISTS")
    cas_storage_root = Path(cas_root).resolve() if cas_root is not None else destination.parent
    cas = ContentAddressedStore(cas_storage_root)
    archive = ArchiveStore(archive_db) if archive_db is not None else None
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    try:
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary.mkdir(mode=0o700)
        report_bytes = _canonical_json_bytes(report)
        markdown_bytes = _render_markdown(report).encode("utf-8")
        _write_bytes_atomic(temporary / "docker-runtime-audit.json", report_bytes)
        _write_bytes_atomic(temporary / "docker-runtime-audit.md", markdown_bytes)
        report_ref = cas.put_bytes(report_bytes, media_type="application/json").uri
        markdown_ref = cas.put_bytes(markdown_bytes, media_type="text/markdown").uri
        if archive is not None:
            archive.record_artifact_reference(report_ref)
            archive.record_artifact_reference(markdown_ref)
        manifest = {
            "schema_version": 1,
            "artifact_type": "wmloop-docker-runtime-audit-manifest",
            "state": report["state"],
            "static_contract_ready": report["static_contract_ready"],
            "live_probe_ready": report["live_probe_ready"],
            "execution_sandbox_ready": report["execution_sandbox_ready"],
            "blocker_count": report["blocker_count"],
            "blockers": report["blockers"],
            "socket_path": report["socket_path"],
            "image": report["image"],
            "m4_launch_allowed": False,
            "formal_training_allowed": False,
            "report_path": str(destination / "docker-runtime-audit.json"),
            "markdown_path": str(destination / "docker-runtime-audit.md"),
            "cas_refs": {
                "docker_runtime_audit_json": report_ref,
                "docker_runtime_audit_markdown": markdown_ref,
            },
        }
        _write_bytes_atomic(temporary / "manifest.json", _canonical_json_bytes(manifest))
        os.replace(temporary, destination)
        return manifest
    except Exception:
        if temporary.is_symlink():
            temporary.unlink()
        elif temporary.exists():
            shutil.rmtree(temporary)
        raise


def _render_markdown(report: Mapping[str, object]) -> str:
    lines = [
        "# Docker Runtime Audit",
        "",
        f"- state: `{report['state']}`",
        f"- static_contract_ready: `{report['static_contract_ready']}`",
        f"- live_probe_ready: `{report['live_probe_ready']}`",
        f"- execution_sandbox_ready: `{report['execution_sandbox_ready']}`",
        f"- socket_path: `{report['socket_path']}`",
        f"- image: `{report['image']}`",
        f"- blocker_count: `{report['blocker_count']}`",
        "",
    ]
    blockers = report.get("blockers")
    if isinstance(blockers, list) and blockers:
        lines.append("## Blockers")
        for blocker in blockers:
            if isinstance(blocker, Mapping):
                surface = blocker.get("surface")
                reason = blocker.get("reason", blocker.get("field"))
                lines.append(f"- `{surface}`: `{reason}`")
        lines.append("")
    lines.append("This audit does not authorize M4 training.")
    return "\n".join(lines) + "\n"


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise DockerRuntimeAuditError("DOCKER_RUNTIME_AUDIT_OUTPUT_EXISTS")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="audit the Docker execution-sandbox runtime")
    run.add_argument("--assets-root", type=Path, default=Path("ops/docker"))
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--image", required=True)
    run.add_argument("--socket-path", type=Path, default=Path("/run/wm-loop-docker/docker.sock"))
    run.add_argument("--skip-live-probe", action="store_true")
    run.add_argument("--archive-db", type=Path)
    run.add_argument("--cas-root", type=Path)
    args = parser.parse_args(argv)
    if args.command == "run":
        manifest = run_docker_runtime_audit(
            assets_root=args.assets_root,
            output_root=args.output_root,
            image=args.image,
            socket_path=args.socket_path,
            skip_live_probe=args.skip_live_probe,
            archive_db=args.archive_db,
            cas_root=args.cas_root,
        )
        print(json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
        return 0
    raise DockerRuntimeAuditError("DOCKER_RUNTIME_AUDIT_COMMAND_INVALID")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
