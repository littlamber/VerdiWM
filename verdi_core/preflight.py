"""Environment and command preflight checks before resource allocation."""

from __future__ import annotations

import shutil
import subprocess
from typing import Iterable


def check_commands(commands: Iterable[str]) -> dict[str, object]:
    checked = list(commands)
    missing = [command for command in checked if shutil.which(command) is None]
    return {"state": "ready" if not missing else "blocked", "missing_commands": missing, "checked_commands": checked}


def run_smoke(command: list[str], *, cwd: str | None = None, timeout: float = 300.0) -> dict[str, object]:
    completed = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False)
    return {"state": "ready" if completed.returncode == 0 else "blocked", "command": command, "returncode": completed.returncode, "stdout": completed.stdout[-4000:], "stderr": completed.stderr[-4000:]}
