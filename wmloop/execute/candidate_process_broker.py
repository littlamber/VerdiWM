#!/usr/bin/env python3
"""Compatibility entrypoint for the Docker-free worktree process backend."""

from wmloop.execute.candidate_local_dev_broker import main


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
