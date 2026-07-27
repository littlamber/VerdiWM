"""Receipt-backed M2 timeout/kill smoke for the local execution backend."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping

from wmloop.archive.store import ArchiveStore, ContentAddressedStore
from wmloop.execute.agent_staging import AgentRepairSession
from wmloop.execute.sandbox import SandboxLease, WorktreeSandbox
from wmloop.primitives.registry import PrimitiveRegistry
from wmloop.vendor import verify_vendor_checkout


class TimeoutKillSmokeError(RuntimeError):
    """The timeout/kill smoke could not produce trustworthy evidence."""


def run_timeout_kill_smoke(
    *,
    repo_root: Path,
    output_root: Path,
    archive_db: Path | None = None,
    cas_root: Path | None = None,
    timeout_seconds: float = 0.5,
) -> dict[str, object]:
    """Run a command that must time out, then seal the failed gate receipt."""

    if timeout_seconds <= 0:
        raise TimeoutKillSmokeError("TIMEOUT_KILL_SMOKE_TIMEOUT_INVALID")
    root = Path(repo_root).resolve()
    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise TimeoutKillSmokeError("TIMEOUT_KILL_SMOKE_OUTPUT_EXISTS")
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    trial_id = "timeout-kill-smoke"
    marker = f"wmloop-timeout-kill-smoke-{uuid.uuid4().hex}"
    lease: SandboxLease | None = None
    sandbox: WorktreeSandbox | None = None
    try:
        temporary.mkdir(mode=0o700)
        source_revision = verify_vendor_checkout(root)
        registry = PrimitiveRegistry.from_root(root)
        sandbox = WorktreeSandbox(vendor_root=root / "vendor" / "ACWM-Phys", runs_root=temporary / "runs")
        lease = sandbox.create(trial_id=trial_id, expected_revision=source_revision)
        marker_path = lease.worktree / "wmloop_timeout_kill_smoke.txt"
        marker_path.write_text(f"{marker}\n", encoding="utf-8")
        session = AgentRepairSession(
            worktree=lease.worktree,
            staging_root=temporary / "staging",
            candidate_id=trial_id,
            source_revision=source_revision,
            registry_digest=registry.digest(),
            required_check_labels=("timeout_gate",),
        )
        receipt = session.run(
            label="timeout_gate",
            argv=(sys.executable, "-c", _timeout_process_group_script(), marker),
            timeout_seconds=timeout_seconds,
        )
        candidate = session.seal()
        if not receipt.timed_out:
            raise TimeoutKillSmokeError("TIMEOUT_KILL_SMOKE_DID_NOT_TIMEOUT")
        if candidate.ready_for_promotion:
            raise TimeoutKillSmokeError("TIMEOUT_KILL_SMOKE_PROMOTED_TIMED_OUT_GATE")
        residual_processes = _marker_processes(marker)
        if residual_processes:
            _kill_processes(residual_processes)
            raise TimeoutKillSmokeError("TIMEOUT_KILL_SMOKE_RESIDUAL_PROCESS")
        worktree = lease.worktree
        sandbox.remove(lease)
        lease = None
        report = {
            "schema_version": 1,
            "artifact_type": "wmloop-m2-timeout-kill-smoke-report",
            "state": "ready",
            "trial_id": trial_id,
            "source_revision": source_revision,
            "registry_digest": registry.digest(),
            "timeout_seconds": timeout_seconds,
            "timeout_marker": marker,
            "worktree": str(worktree),
            "worktree_removed": not worktree.exists(),
            "timed_out": receipt.timed_out,
            "timeout_receipt": receipt.to_document(),
            "ready_for_promotion": candidate.ready_for_promotion,
            "candidate": candidate.to_document(),
            "candidate_manifest_path": str(destination / candidate.manifest_path.relative_to(temporary)),
            "candidate_diff_path": str(destination / candidate.diff_path.relative_to(temporary)),
            "residual_process_check": {
                "method": "ps -eo pid=,pgid=,args= marker scan",
                "pass": True,
                "matches": residual_processes,
            },
            "limitations": [
                "This is a local backend timeout/kill receipt smoke; it does not launch ACWM training.",
            ],
        }
        manifest = _write_report_bundle(
            report=report,
            output_root=destination,
            temporary_root=temporary,
            archive_db=archive_db,
            cas_root=cas_root,
        )
        os.replace(temporary, destination)
        return manifest
    except Exception:
        if lease is not None and sandbox is not None:
            try:
                sandbox.remove(lease)
            except Exception:
                pass
        residual_processes = _marker_processes(marker)
        if residual_processes:
            _kill_processes(residual_processes)
        if temporary.is_symlink():
            temporary.unlink()
        elif temporary.exists():
            shutil.rmtree(temporary)
        raise


def _timeout_process_group_script() -> str:
    return (
        "import subprocess, sys, time; "
        "marker=sys.argv[1]; "
        "subprocess.Popen([sys.executable, '-c', 'import sys, time; time.sleep(30)', marker]); "
        "time.sleep(30)"
    )


def _marker_processes(marker: str) -> list[dict[str, object]]:
    completed = subprocess.run(["ps", "-eo", "pid=,pgid=,args="], check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise TimeoutKillSmokeError("TIMEOUT_KILL_SMOKE_PS_FAILED")
    current_pid = os.getpid()
    matches: list[dict[str, object]] = []
    for line in completed.stdout.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) != 3:
            continue
        try:
            pid = int(parts[0])
            pgid = int(parts[1])
        except ValueError:
            continue
        args = parts[2]
        if pid != current_pid and marker in args:
            matches.append({"pid": pid, "pgid": pgid, "args": args[:500]})
    return matches


def _kill_processes(processes: list[Mapping[str, object]]) -> None:
    for process in processes:
        pid = process.get("pid")
        if isinstance(pid, int) and pid > 1:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def _write_report_bundle(
    *,
    report: Mapping[str, object],
    output_root: Path,
    temporary_root: Path,
    archive_db: Path | None,
    cas_root: Path | None,
) -> dict[str, object]:
    report_bytes = _canonical_json_bytes(report)
    markdown_bytes = _render_markdown(report).encode("utf-8")
    _write_bytes_atomic(temporary_root / "timeout-kill-smoke.json", report_bytes)
    _write_bytes_atomic(temporary_root / "timeout-kill-smoke.md", markdown_bytes)
    candidate_manifest = temporary_root / "staging" / "timeout-kill-smoke" / "candidate.json"
    candidate_diff = temporary_root / "staging" / "timeout-kill-smoke" / "candidate.diff"
    if not candidate_manifest.is_file() or not candidate_diff.is_file():
        raise TimeoutKillSmokeError("TIMEOUT_KILL_SMOKE_CANDIDATE_MISSING")
    cas_refs: dict[str, str] = {}
    archive = ArchiveStore(archive_db) if archive_db is not None else None
    if archive is not None or cas_root is not None:
        root = cas_root if cas_root is not None else Path(archive_db).resolve().parent  # type: ignore[union-attr]
        cas = ContentAddressedStore(Path(root))
        for name, payload, media_type in (
            ("timeout_kill_report_json", report_bytes, "application/json"),
            ("timeout_kill_report_markdown", markdown_bytes, "text/markdown"),
            ("timeout_kill_candidate_json", candidate_manifest.read_bytes(), "application/json"),
            ("timeout_kill_candidate_diff", candidate_diff.read_bytes(), "text/x-diff"),
        ):
            ref = cas.put_bytes(payload, media_type=media_type).uri
            cas_refs[name] = ref
            if archive is not None:
                archive.record_artifact_reference(ref)
    manifest = {
        "schema_version": 1,
        "artifact_type": "wmloop-m2-timeout-kill-smoke-manifest",
        "state": report["state"],
        "trial_id": report["trial_id"],
        "timed_out": report["timed_out"],
        "ready_for_promotion": report["ready_for_promotion"],
        "worktree_removed": report["worktree_removed"],
        "residual_process_check_pass": report["residual_process_check"]["pass"],  # type: ignore[index]
        "report_path": str(output_root / "timeout-kill-smoke.json"),
        "markdown_path": str(output_root / "timeout-kill-smoke.md"),
        "candidate_manifest_path": report["candidate_manifest_path"],
        "candidate_diff_path": report["candidate_diff_path"],
        "cas_refs": cas_refs,
        "limitations": report["limitations"],
    }
    _write_bytes_atomic(temporary_root / "manifest.json", _canonical_json_bytes(manifest))
    return manifest


def _render_markdown(report: Mapping[str, Any]) -> str:
    receipt = report["timeout_receipt"]
    residual = report["residual_process_check"]
    return "\n".join(
        [
            "# M2 Timeout-Kill Smoke",
            "",
            f"State: `{report['state']}`",
            f"Timed out: `{receipt['timed_out']}`",
            f"Exit code: `{receipt['exit_code']}`",
            f"Ready for promotion: `{report['ready_for_promotion']}`",
            f"Worktree removed: `{report['worktree_removed']}`",
            f"Residual process check: `{residual['pass']}`",
            "",
            "## Limitation",
            "",
            "- This is a local backend timeout/kill receipt smoke; it does not launch ACWM training.",
        ]
    ) + "\n"


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise TimeoutKillSmokeError("TIMEOUT_KILL_SMOKE_OUTPUT_EXISTS")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run timeout/kill smoke")
    run.add_argument("--repo-root", type=Path, default=Path("."))
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--archive-db", type=Path)
    run.add_argument("--cas-root", type=Path)
    run.add_argument("--timeout-seconds", type=float, default=0.5)
    args = parser.parse_args(argv)
    if args.command == "run":
        manifest = run_timeout_kill_smoke(
            repo_root=args.repo_root,
            output_root=args.output_root,
            archive_db=args.archive_db,
            cas_root=args.cas_root,
            timeout_seconds=args.timeout_seconds,
        )
        print(json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
