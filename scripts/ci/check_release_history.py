"""Inspect all reachable Git objects without logging credential contents."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import subprocess

from scripts.export.build_github_staging import MAX_TEXT_BYTES, audit_text, is_sensitive_file

logger = logging.getLogger(__name__)


def audit_history(root: Path) -> dict[str, object]:
    command = ["git", "-C", str(root)]
    objects = subprocess.run(command + ["rev-list", "--objects", "--all"], check=True, capture_output=True).stdout
    findings: list[str] = []
    shallow = subprocess.run(command + ["rev-parse", "--is-shallow-repository"], check=True, capture_output=True).stdout.strip()
    if shallow == b"true":
        findings.append("shallow_history_incomplete")
    count = 0
    # Check every historical filename as well as unique object contents; the
    # same blob can have both an innocuous name and a sensitive filename.
    names = subprocess.run(command + ["log", "--all", "--format=", "--name-only", "-z"], check=True, capture_output=True).stdout
    for raw in names.split(b"\0"):
        name = raw.decode("utf-8", errors="replace").strip("\n")
        if name and is_sensitive_file(Path(name)):
            findings.append(f"sensitive_file:{name}")
    with subprocess.Popen(command + ["cat-file", "--batch"], stdin=subprocess.PIPE, stdout=subprocess.PIPE) as process:
        try:
            for line in objects.splitlines():
                oid = line.split(b" ", 1)[0]
                process.stdin.write(oid + b"\n")
                process.stdin.flush()
                header = process.stdout.readline().split()
                if len(header) != 3:
                    raise RuntimeError("HISTORY_OBJECT_UNAVAILABLE")
                size = int(header[2])
                kind = header[1]
                if size > MAX_TEXT_BYTES:
                    remaining = size
                    while remaining:
                        chunk = process.stdout.read(min(remaining, 65536))
                        if not chunk:
                            raise RuntimeError("HISTORY_OBJECT_TRUNCATED")
                        remaining -= len(chunk)
                    if kind in {b"blob", b"commit", b"tag"}:
                        findings.append(f"oversized_object_unscanned:{oid.decode()}")
                else:
                    payload = process.stdout.read(size)
                    if kind in {b"blob", b"commit", b"tag"}:
                        count += 1
                        try:
                            content = payload.decode("utf-8")
                        except UnicodeDecodeError:
                            findings.append(f"binary_object_unscanned:{oid.decode()}")
                        else:
                            if "\0" in content:
                                findings.append(f"binary_object_unscanned:{oid.decode()}")
                            else:
                                findings.extend(audit_text(content, oid.decode()))
                if process.stdout.read(1) != b"\n":
                    raise RuntimeError("HISTORY_OBJECT_FRAMING_INVALID")
        finally:
            process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError("HISTORY_SCAN_FAILED")
    return {
        "status": "blocked" if findings else "passed",
        "scanned_objects": count,
        "findings": sorted(set(findings)),
        "claim_boundary": "Reachable local refs only; known-pattern scan, not proof of secret absence. No object contents are included in this report.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = audit_history(args.repo_root)
    encoded = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.report:
        from wmloop.storage import atomic_write, checked_path

        target = checked_path(args.report, code="HISTORY_REPORT_SYMLINK")
        atomic_write(target, encoded.encode())
    logger.info("%s", encoded)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())
