#!/usr/bin/env python3
"""Fail CI when new files violate the VerdiWM directory and naming contract."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from wmloop.repository_layout import audit_repository_layout


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args(argv)
    issues = audit_repository_layout(args.repo_root)
    if issues:
        for issue in issues:
            sys.stderr.write(f"{issue.code}: {issue.path}: {issue.detail}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
