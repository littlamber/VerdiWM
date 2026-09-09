"""Exercise public entrypoints outside the checkout in a clean wheel install."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main() -> None:
    wheel = Path(sys.argv[1]).resolve(strict=True)
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    with tempfile.TemporaryDirectory(prefix="verdiwm-installed-") as temporary:
        root = Path(temporary)
        python = root / "env/bin/python"
        subprocess.run(["uv", "venv", "--python", sys.executable, str(root / "env")], check=True, cwd=root, env=environment)
        subprocess.run(["uv", "pip", "install", "--python", str(python), str(wheel)], check=True, cwd=root, env=environment)
        for name, arguments in (
            ("verdiwm", ["--help"]),
            ("verdiwm", ["evidence", "project", "--help"]),
            ("verdiwm", ["community", "verify", "--help"]),
            ("verdiwm-workbench", ["--help"]),
            ("verdiwm-ctrl-world-autonomous-transfer", ["--help"]),
        ):
            subprocess.run([str(root / "env/bin" / name), *arguments], check=True, cwd=root, env=environment, stdout=subprocess.DEVNULL)
        subprocess.run([str(python), "-I", "-c", "from wmloop.contracts import validate_document; from experiments.ctrl_world_autonomous_transfer_v1 import workflow; from wmloop.archive.community_projection import project_archive_evidence; print('Installed wheel imports and entrypoints passed')"], check=True, cwd=root, env=environment)


if __name__ == "__main__":
    main()
