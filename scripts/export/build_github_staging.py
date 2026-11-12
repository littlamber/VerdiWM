#!/usr/bin/env python3
"""Build a source-only, audited VERDI repository for public distribution."""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Sequence

from scripts.export.validate_portrait_first_public_example import (
    validate_portrait_first_public_example,
)
from wmloop.storage import atomic_write, checked_path, sha256_file

REPO_ROOT = Path(__file__).resolve().parents[2]
logger = logging.getLogger(__name__)


class GithubStagingError(RuntimeError):
    """The source tree cannot be published under the public repository policy."""


ROOT_FILES = {
    ".gitignore",
    "AGENTS.md",
    "CITATION.cff",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "LICENSE",
    "README.md",
    "README_zh.md",
    "SECURITY.md",
    "THIRD_PARTY_NOTICES.md",
    "pyproject.toml",
    "uv.lock",
}
SOURCE_TREES = {
    ".github",
    "configs",
    "docs",
    "experiments",
    "ops",
    "scripts",
    "tests",
    "wmloop",
}
RETAINED_EXAMPLE = Path("examples/portrait_first_minimal_loop_v1")
BLOCKED_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".db",
    ".h5",
    ".hdf5",
    ".mp4",
    ".npy",
    ".npz",
    ".pdf",
    ".png",
    ".pt",
    ".pth",
    ".safetensors",
    ".svg",
}
BLOCKED_PARTS = {
    ".audit-state",
    ".git",
    ".hypothesis",
    ".pytest_cache",
    ".venv",
    "__pycache__",
    "artifacts",
    "cache",
    "figures",
    "logs",
    "outputs",
    "results",
    "runs",
    "vendor",
    "node_modules",
}
PRIVATE_TREES = (Path("docs/plans"), Path("docs/summary"), Path("docs/prompts"), Path("docs/logs"), Path("wmloop/web/ui"))
BLOCKED_ROOT_FILES = {
    ".verdiwm-public-release",
    "MANIFEST.sha256",
    "RELEASE_AUDIT.json",
}
LOCAL_PATH_PATTERN = re.compile(r"(?:^|[\"'\s=,:])(?:/(?:cs|home|mnt|root|Users|share/project)(?:/|\b)|[A-Za-z]:[\\/]+Users[\\/])")
SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----"),
    re.compile(r"\b(?:sk[-_](?:proj-|svcacct-)?|gh[pousr]_|github_pat_)[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
)
MAX_TEXT_BYTES = 5 * 1024 * 1024


def is_sensitive_file(path: Path) -> bool:
    name = path.name.lower()
    return (
        (name.startswith(".env") and name not in {".env.example", ".env.template"})
        or name in {".netrc", ".pypirc", "id_rsa", "id_ed25519", "credentials.json"}
        or path.suffix.lower() in {".pem", ".key", ".p12", ".pfx"}
    )


def audit_text(text: str, label: str) -> list[str]:
    findings = []
    if LOCAL_PATH_PATTERN.search(text):
        findings.append(f"local_path:{label}")
    if any(pattern.search(text) for pattern in SECRET_PATTERNS):
        findings.append(f"secret_pattern:{label}")
    return findings


def build_github_staging(*, source_root: Path, output_root: Path, include_untracked: bool = False) -> dict[str, object]:
    """Copy audited public source; untracked files require an explicit opt-in."""

    source = checked_path(source_root, code="GITHUB_STAGING_SOURCE_SYMLINK", error=GithubStagingError)
    destination = checked_path(output_root, code="GITHUB_STAGING_OUTPUT_SYMLINK", error=GithubStagingError)
    if destination.exists() or destination.is_symlink():
        raise GithubStagingError("GITHUB_STAGING_OUTPUT_EXISTS")
    if not (source / "wmloop" / "geometry" / "irg.py").is_file():
        raise GithubStagingError("GITHUB_STAGING_SOURCE_INVALID")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        copied = 0
        for relative in _source_files(source, include_untracked=include_untracked):
            candidate = checked_path(source / relative, code="GITHUB_STAGING_SOURCE_SYMLINK", error=GithubStagingError)
            # A file scheduled for deletion can remain in Git's index while it
            # is already absent from the working tree. It must not be staged.
            if not candidate.is_file() or candidate.is_symlink():
                continue
            target = temporary / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate, target)
            copied += 1

        portrait_validation = validate_portrait_first_public_example(
            temporary / RETAINED_EXAMPLE
        )
        findings = audit_release_tree(temporary)
        if findings:
            raise GithubStagingError(
                "GITHUB_STAGING_AUDIT_FAILED:" + ";".join(findings[:20])
            )

        audit = {
            "schema_version": 1,
            "artifact_type": "verdiwm-source-release-audit",
            "state": "ready",
            "source_file_count": copied,
            "source_hashes": {p.relative_to(temporary).as_posix(): sha256_file(p) for p in sorted(temporary.rglob("*")) if p.is_file()},
            "checks": {
                "source_only": True,
                "no_generated_artifacts": True,
                "no_detected_local_paths_or_secrets": True,
                "onboarding_demo": True,
            },
            "claim_boundary": "Current source snapshot only; pattern scanning is not proof of secret absence. Git history is not included or approved.",
            "onboarding_demo": {
                "path": RETAINED_EXAMPLE.as_posix(),
                "validation": portrait_validation,
            },
        }
        temporary.rename(destination)
        return audit
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def audit_release_tree(root: Path) -> list[str]:
    """Return publication-policy violations present in a staged repository."""

    release_root = Path(root).resolve(strict=True)
    findings: list[str] = []
    for path in sorted(release_root.rglob("*")):
        relative = path.relative_to(release_root)
        relative_text = relative.as_posix()
        if path.is_symlink():
            findings.append(f"symlink:{relative_text}")
            continue
        if not path.is_file():
            continue
        if any(part in BLOCKED_PARTS for part in relative.parts):
            findings.append(f"blocked_directory:{relative_text}")
        if relative.name in BLOCKED_ROOT_FILES:
            findings.append(f"blocked_artifact:{relative_text}")
        if any(relative == tree or tree in relative.parents for tree in PRIVATE_TREES):
            findings.append(f"private_tree:{relative_text}")
        if is_sensitive_file(relative):
            findings.append(f"sensitive_file:{relative_text}")
        if relative.parts and relative.parts[0] == "examples" and not _is_retained_example(relative):
            findings.append(f"unexpected_example:{relative_text}")
        if path.suffix.lower() in BLOCKED_SUFFIXES:
            findings.append(f"blocked_extension:{relative_text}")
        if path.stat().st_size > MAX_TEXT_BYTES:
            findings.append(f"file_too_large:{relative_text}")
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            findings.append(f"non_text_file:{relative_text}")
            continue
        if "\0" in text:
            findings.append(f"non_text_file:{relative_text}")
            continue
        findings.extend(audit_text(text, relative_text))
    return sorted(set(findings))


def _source_files(source: Path, *, include_untracked: bool = False) -> list[Path]:
    """Return public files tracked by Git, using the working-tree contents."""

    completed = subprocess.run(
        ["git", "-C", str(source), "ls-files", "-z"],
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise GithubStagingError("GITHUB_STAGING_GIT_INDEX_UNAVAILABLE")
    untracked = subprocess.run(
        ["git", "-C", str(source), "ls-files", "--others", "--exclude-standard", "-z"],
        check=True, capture_output=True,
    )
    missing = [Path(raw) for raw in untracked.stdout.decode().split("\0") if raw and _is_public_source_path(Path(raw))]
    if missing and not include_untracked:
        raise GithubStagingError("GITHUB_STAGING_UNTRACKED_SOURCE:" + ";".join(str(path) for path in missing[:20]))
    result: list[Path] = []
    for raw in completed.stdout.decode("utf-8").split("\0"):
        if not raw:
            continue
        relative = Path(raw)
        if _is_public_source_path(relative):
            result.append(relative)
        elif (source / relative).exists() or (source / relative).is_symlink():
            raise GithubStagingError("GITHUB_STAGING_NONPUBLIC_TRACKED_FILE:" + relative.as_posix())
    return sorted(set(result) | set(missing))


def _is_public_source_path(relative: Path) -> bool:
    if is_sensitive_file(relative) or any(relative == tree or tree in relative.parents for tree in PRIVATE_TREES):
        return False
    if relative.name in BLOCKED_ROOT_FILES or any(part in BLOCKED_PARTS for part in relative.parts):
        return False
    if relative.as_posix() in ROOT_FILES:
        return True
    if relative.parts and relative.parts[0] in SOURCE_TREES:
        return True
    return _is_retained_example(relative)


def _is_retained_example(relative: Path) -> bool:
    return relative == RETAINED_EXAMPLE or RETAINED_EXAMPLE in relative.parents


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--include-untracked", action="store_true", help="Include untracked public source in a review snapshot; does not modify Git's index.")
    parser.add_argument("--report", type=Path, help="Optional audit JSON outside the source snapshot.")
    args = parser.parse_args(argv)
    try:
        report = build_github_staging(
            source_root=args.source_root,
            output_root=args.output_root,
            include_untracked=args.include_untracked,
        )
    except GithubStagingError as exc:
        logger.error("%s", exc)
        return 2
    if args.report:
        target = checked_path(args.report, code="GITHUB_REPORT_SYMLINK", error=GithubStagingError)
        if target == args.output_root.resolve() or args.output_root.resolve() in target.parents:
            raise GithubStagingError("GITHUB_REPORT_MUST_BE_OUTSIDE_SOURCE")
        atomic_write(target, (json.dumps(report, indent=2) + "\n").encode())
    logger.info("%s", json.dumps({key: value for key, value in report.items() if key != "source_hashes"}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())
