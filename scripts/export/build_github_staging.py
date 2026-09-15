#!/usr/bin/env python3
"""Build a source-only, audited VERDI repository for public distribution."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.export.validate_portrait_first_public_example import (
    validate_portrait_first_public_example,
)


class GithubStagingError(RuntimeError):
    """The source tree cannot be published under the public repository policy."""


ROOT_FILES = {
    ".gitignore",
    "CONTRIBUTING.md",
    "LICENSE",
    "README.md",
    "README_zh.md",
    "SECURITY.md",
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
}
BLOCKED_ROOT_FILES = {
    ".verdiwm-public-release",
    "MANIFEST.sha256",
    "RELEASE_AUDIT.json",
}
TEXT_SUFFIXES = {
    ".css",
    ".csv",
    ".html",
    ".ini",
    ".json",
    ".jsonl",
    ".js",
    ".md",
    ".py",
    ".service",
    ".sh",
    ".socket",
    ".tex",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
LOCAL_PATH_PATTERN = re.compile(r"(?:^|[\"'\s])/(?:home|mnt|root|Users|share/project)(?:/|\b)")
SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----"),
    re.compile(r"\b(?:sk|ghp)_[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
)


def build_github_staging(*, source_root: Path, output_root: Path) -> dict[str, object]:
    """Copy only tracked source files into a public, clone-ready repository tree."""

    source = Path(source_root).resolve(strict=True)
    destination = Path(output_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise GithubStagingError("GITHUB_STAGING_OUTPUT_EXISTS")
    if not (source / "wmloop" / "geometry" / "irg.py").is_file():
        raise GithubStagingError("GITHUB_STAGING_SOURCE_INVALID")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        copied = 0
        for relative in _source_files(source):
            candidate = source / relative
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
            "checks": {
                "source_only": True,
                "no_generated_artifacts": True,
                "no_local_paths_or_secrets": True,
                "onboarding_demo": True,
            },
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
        if relative.parts and relative.parts[0] == "examples" and not _is_retained_example(relative):
            findings.append(f"unexpected_example:{relative_text}")
        if path.suffix.lower() in BLOCKED_SUFFIXES:
            findings.append(f"blocked_extension:{relative_text}")
        if not _is_textual(path):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            findings.append(f"non_text_file:{relative_text}")
            continue
        if LOCAL_PATH_PATTERN.search(text):
            findings.append(f"local_path:{relative_text}")
        if any(pattern.search(text) for pattern in SECRET_PATTERNS):
            findings.append(f"secret_pattern:{relative_text}")
    return sorted(set(findings))


def _source_files(source: Path) -> list[Path]:
    """Return public files tracked by Git, using the working-tree contents."""

    completed = subprocess.run(
        ["git", "-C", str(source), "ls-files", "-z"],
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise GithubStagingError("GITHUB_STAGING_GIT_INDEX_UNAVAILABLE")
    result: list[Path] = []
    for raw in completed.stdout.decode("utf-8").split("\0"):
        if not raw:
            continue
        relative = Path(raw)
        if _is_public_source_path(relative):
            result.append(relative)
    return sorted(result)


def _is_public_source_path(relative: Path) -> bool:
    if relative.name in BLOCKED_ROOT_FILES or any(part in BLOCKED_PARTS for part in relative.parts):
        return False
    if relative.as_posix() in ROOT_FILES:
        return True
    if relative.parts and relative.parts[0] in SOURCE_TREES:
        return True
    return _is_retained_example(relative)


def _is_retained_example(relative: Path) -> bool:
    return relative == RETAINED_EXAMPLE or RETAINED_EXAMPLE in relative.parents


def _is_textual(path: Path) -> bool:
    return path.suffix.lower() in TEXT_SUFFIXES or path.name in {".gitignore", "LICENSE", "uv.lock"}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = build_github_staging(
            source_root=args.source_root,
            output_root=args.output_root,
        )
    except GithubStagingError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
