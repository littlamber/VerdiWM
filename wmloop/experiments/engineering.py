"""Engineering contracts for reproducible experiment repositories."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping

from wmloop.contracts import ContractValidationError, validate_document


class ExperimentEngineeringError(ValueError):
    """An experiment repository is not ready for admission."""


_EXPERIMENT_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{2,63}$")


def lint_experiment_manifest(
    manifest_path: Path,
    *,
    repo_root: Path | None = None,
    root: Path | None = None,
) -> dict[str, object]:
    """Validate a manifest and the small repository surface it owns."""

    path = Path(manifest_path).expanduser().resolve(strict=True)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentEngineeringError("EXPERIMENT_MANIFEST_INVALID") from exc
    if not isinstance(payload, Mapping):
        raise ExperimentEngineeringError("EXPERIMENT_MANIFEST_OBJECT_REQUIRED")
    try:
        validate_document("experiment_engineering_manifest", payload, root=root)
    except ContractValidationError as exc:
        raise ExperimentEngineeringError(f"EXPERIMENT_MANIFEST_SCHEMA_INVALID:{exc}") from exc
    experiment_id = str(payload["experiment_id"])
    if not _EXPERIMENT_ID.fullmatch(experiment_id):
        raise ExperimentEngineeringError("EXPERIMENT_ID_INVALID")

    base = Path(repo_root or path.parent).expanduser().resolve()
    source = payload["source"]
    assert isinstance(source, Mapping)
    checks: list[dict[str, object]] = []
    for name, relative, required in (
        ("readme", source.get("readme"), True),
        ("entrypoint", source.get("entrypoint"), True),
        ("test_path", source.get("test_path"), True),
        ("scale_plan", payload.get("scale_plan"), True),
    ):
        if not isinstance(relative, str) or not relative.strip():
            checks.append({"name": name, "state": "fail", "detail": "path_missing", "required": required})
            continue
        candidate = (base / relative).resolve()
        inside = candidate == base or base in candidate.parents
        state = "pass" if inside and candidate.is_file() else "fail"
        checks.append({"name": name, "state": state, "detail": str(candidate), "required": required})

    git_revision, dirty_paths = _git_state(base)
    dirty = bool(dirty_paths)
    declared_revision = str(source.get("revision", ""))
    # ``HEAD`` is an explicit dynamic binding used by owned experiment
    # packages whose manifest is versioned in the same commit as the runner.
    # It still requires a real Git revision and a clean checkout below.
    if declared_revision not in {"", "HEAD"} and git_revision and declared_revision != git_revision:
        checks.append({"name": "source_revision", "state": "fail", "detail": f"declared={declared_revision}:observed={git_revision}", "required": True})
    else:
        checks.append({"name": "source_revision", "state": "pass" if git_revision else "fail", "detail": git_revision or "git_unavailable", "required": True})
    dirty_policy = str(source.get("dirty_policy"))
    dirty_allowed = dirty_policy == "allow_with_receipt"
    checks.append({"name": "source_clean", "state": "pass" if not dirty or dirty_allowed else "fail", "detail": "dirty_allowed" if dirty and dirty_allowed else ("dirty" if dirty else "clean"), "required": True})

    blockers = [str(row["name"]) for row in checks if row["required"] and row["state"] == "fail"]
    artifact_policy = payload["artifact_policy"]
    assert isinstance(artifact_policy, Mapping)
    result = {
        "schema_version": 1,
        "artifact_type": "verdiwm-experiment-engineering-lint",
        "state": "ready" if not blockers else "blocked",
        "experiment_id": experiment_id,
        "manifest_path": str(path),
        "repo_root": str(base),
        "source": {
            "revision": git_revision,
            "dirty": dirty,
            "dirty_paths": list(dirty_paths),
            "manifest_sha256": _sha256(path),
        },
        "checks": checks,
        "blockers": blockers,
        "artifact_policy": dict(artifact_policy),
        "claim_boundary": "A passing engineering lint proves repository hygiene and admission metadata, not scientific validity.",
    }
    return result


def _git_state(root: Path) -> tuple[str | None, tuple[str, ...]]:
    try:
        revision = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None, ()
    dirty_paths = tuple(line[3:] for line in status.splitlines() if len(line) >= 4)
    return revision or None, dirty_paths


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
