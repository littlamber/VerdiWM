"""Compile user intent into immutable campaign revisions and GPU role leases.

The public campaign API accepts a small, stable request.  This module owns the
otherwise easy-to-forget bookkeeping needed to make that request reproducible:
content-derived revision identifiers, source/policy/asset fingerprints, and a
non-overlapping GPU allocation.  Revision labels are implementation detail;
callers never need to create ``v1``/``v2`` variants by hand.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence


class AutomaticCampaignError(ValueError):
    """A request cannot be deterministically compiled into a safe revision."""


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def campaign_key(*, model: str, dataset: str, goal: str, budget_gpu_hours: float, adapter: str | None) -> str:
    """Return the stable identity of a user request, excluding implementation state."""

    return digest(
        {
            "model": _normalized_location(model),
            "dataset": _normalized_location(dataset),
            "goal": goal.strip(),
            "budget_gpu_hours": float(budget_gpu_hours),
            "adapter": adapter or "auto",
        }
    )


def automatic_campaign_id(key: str) -> str:
    if len(key) != 64 or any(character not in "0123456789abcdef" for character in key):
        raise AutomaticCampaignError("CAMPAIGN_KEY_INVALID")
    return f"auto-{key[:20]}"


def compile_revision(
    *,
    campaign_id: str,
    campaign_key_value: str,
    goal: str,
    model: str,
    dataset: str,
    budget_gpu_hours: float,
    adapter_profile: str | None,
    constitution_freeze: str | None,
    execution: Mapping[str, object],
    resource_request: Mapping[str, object] | None = None,
    project_root: Path | None = None,
) -> dict[str, object]:
    """Build an immutable revision snapshot from a resolved local execution.

    Large checkpoints and datasets are represented by durable filesystem
    identity rather than copied or fully hashed.  Small policy and source files
    are content-hashed, so code/policy changes yield a different revision.
    """

    if not campaign_id:
        raise AutomaticCampaignError("CAMPAIGN_ID_INVALID")
    if not math.isfinite(float(budget_gpu_hours)) or float(budget_gpu_hours) <= 0:
        raise AutomaticCampaignError("BUDGET_INVALID")
    root = (project_root or Path(__file__).resolve().parents[2]).expanduser().resolve()
    allocation = allocate_gpu_roles(
        resource_request,
        model_family=_model_family(execution, adapter_profile),
    )
    intent = {
        "campaign_key": campaign_key_value,
        "goal": goal.strip(),
        "model": _normalized_location(model),
        "dataset": _normalized_location(dataset),
        "budget_gpu_hours": float(budget_gpu_hours),
        "adapter_profile": adapter_profile,
    }
    policies = {
        "constitution_freeze": _fingerprint_declared_path(constitution_freeze),
        "evaluator_contract": _fingerprint_declared_path(execution.get("evaluator_contract")),
        "probe_contract": _fingerprint_declared_path(execution.get("probe_contract")),
        "candidate_catalog": _fingerprint_declared_path(execution.get("candidate_catalog")),
    }
    snapshot = {
        "intent": intent,
        "project_source": _source_tree_fingerprint(root),
        "model": _fingerprint_declared_path(model),
        "dataset": _fingerprint_declared_path(dataset),
        "policies": policies,
        "execution": _execution_fingerprint(execution),
        "resource_policy": _allocation_semantics(allocation),
    }
    revision_digest = digest(snapshot)
    revision_id = f"r-{revision_digest[:20]}"
    return {
        "schema_version": 1,
        "artifact_type": "verdiwm-automatic-campaign-revision",
        "campaign_id": campaign_id,
        "campaign_key": campaign_key_value,
        "revision_id": revision_id,
        "revision_digest": revision_digest,
        "state": "ready",
        "snapshot": snapshot,
        "resource_allocation": allocation,
        "resume_policy": {
            "same_revision": "resume_in_place_from_durable_stage_receipts",
            "changed_input": "compile_a_new_internal_revision",
            "terminal_negative_evidence": "archive_without_promotion",
        },
        "claim_boundary": (
            "This revision and allocation receipt establish execution provenance only. "
            "They do not establish a model-quality claim or relax frozen evaluation."
        ),
    }


def isolate_execution_for_revision(
    execution: Mapping[str, object], *, revision_id: str
) -> dict[str, object]:
    """Give an automatically generated revision isolated mutable output state."""

    copied = json.loads(json.dumps(execution))
    for field in ("output_root", "state_root", "workspace_root"):
        value = copied.get(field)
        if isinstance(value, str) and value:
            copied[field] = str(Path(value) / revision_id)
    budget_db = copied.get("budget_db")
    if isinstance(budget_db, str) and budget_db:
        copied["budget_db"] = str(Path(budget_db).parent / f"{revision_id}.db")
    return copied


def allocate_gpu_roles(
    request: Mapping[str, object] | None,
    *,
    model_family: str | None = None,
) -> dict[str, object]:
    """Create a deterministic, non-overlapping single-GPU role partition."""

    raw = dict(request or {})
    unknown = set(raw) - {"gpu_indices", "data_preparation_gpus", "candidate_parallelism"}
    if unknown:
        raise AutomaticCampaignError("RESOURCE_REQUEST_UNKNOWN:" + ",".join(sorted(unknown)))
    supplied = raw.get("gpu_indices")
    if supplied is None:
        indices = _detect_gpu_indices()
        inventory_source = "detected" if indices else "unavailable"
    else:
        if not isinstance(supplied, Sequence) or isinstance(supplied, (str, bytes)):
            raise AutomaticCampaignError("RESOURCE_GPU_INDICES_INVALID")
        try:
            indices = tuple(int(value) for value in supplied)
        except (TypeError, ValueError) as exc:
            raise AutomaticCampaignError("RESOURCE_GPU_INDICES_INVALID") from exc
        inventory_source = "declared"
    if any(value < 0 for value in indices) or len(set(indices)) != len(indices):
        raise AutomaticCampaignError("RESOURCE_GPU_INDICES_INVALID")
    data_default = 2 if model_family == "ctrl_world" and len(indices) >= 2 else 0
    data_count = _bounded_nonnegative_int(
        raw.get("data_preparation_gpus", data_default), "RESOURCE_DATA_PREPARATION_INVALID"
    )
    if data_count > len(indices):
        raise AutomaticCampaignError("RESOURCE_DATA_PREPARATION_EXCEEDS_INVENTORY")
    candidate_indices = tuple(indices[: len(indices) - data_count])
    prep_indices = tuple(indices[len(indices) - data_count :]) if data_count else ()
    requested_parallelism = raw.get("candidate_parallelism", len(candidate_indices))
    candidate_parallelism = _bounded_nonnegative_int(
        requested_parallelism, "RESOURCE_CANDIDATE_PARALLELISM_INVALID"
    )
    if candidate_parallelism > len(candidate_indices):
        raise AutomaticCampaignError("RESOURCE_CANDIDATE_PARALLELISM_EXCEEDS_ALLOCATION")
    roles = [
        {
            "role": "autonomous_candidate_evaluation",
            "gpu_indices": list(candidate_indices),
            "per_job_gpus": 1,
            "max_parallel_jobs": candidate_parallelism,
            "work_policy": "independent_candidate_evaluation",
        },
        {
            "role": "droid_data_preparation",
            "gpu_indices": list(prep_indices),
            "per_job_gpus": 1,
            "max_parallel_jobs": len(prep_indices),
            "work_policy": "resumable_shard_conversion",
        },
    ]
    _validate_disjoint_roles(roles)
    payload = {
        "schema_version": 1,
        "artifact_type": "verdiwm-gpu-resource-allocation",
        "state": "ready" if indices else "unresolved",
        "inventory_source": inventory_source,
        "gpu_indices": list(indices),
        "roles": roles,
        "claim_boundary": (
            "Each role grants only exclusive one-GPU process admission. "
            "No role may borrow a GPU assigned to another role."
        ),
    }
    payload["allocation_id"] = digest(payload)
    return payload


def _allocation_semantics(allocation: Mapping[str, object]) -> dict[str, object]:
    return {
        "state": allocation.get("state"),
        "gpu_indices": allocation.get("gpu_indices"),
        "roles": allocation.get("roles"),
    }


def _model_family(execution: Mapping[str, object], adapter_profile: str | None) -> str | None:
    profile = (adapter_profile or "").casefold()
    if "ctrl-world" in profile or "ctrl_world" in profile:
        return "ctrl_world"
    family = execution.get("model_family")
    return str(family) if isinstance(family, str) else None


def _bounded_nonnegative_int(value: object, code: str) -> int:
    if isinstance(value, bool):
        raise AutomaticCampaignError(code)
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise AutomaticCampaignError(code) from exc
    if normalized < 0 or normalized != value:
        raise AutomaticCampaignError(code)
    return normalized


def _validate_disjoint_roles(roles: Sequence[Mapping[str, object]]) -> None:
    seen: set[int] = set()
    for role in roles:
        values = role.get("gpu_indices")
        if not isinstance(values, list):
            raise AutomaticCampaignError("RESOURCE_ROLE_INVALID")
        for value in values:
            if not isinstance(value, int) or value in seen:
                raise AutomaticCampaignError("RESOURCE_ROLE_OVERLAP")
            seen.add(value)


def _detect_gpu_indices() -> tuple[int, ...]:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return ()
    if completed.returncode != 0:
        return ()
    values: list[int] = []
    for line in completed.stdout.splitlines():
        try:
            values.append(int(line.strip()))
        except ValueError:
            return ()
    return tuple(values)


def _normalized_location(value: str) -> str:
    return str(Path(value).expanduser().resolve()) if value else value


def _fingerprint_declared_path(value: object) -> dict[str, object] | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value).expanduser()
    try:
        resolved = path.resolve(strict=True)
        stat = resolved.stat()
    except OSError:
        return {"declared": str(path), "state": "missing"}
    result: dict[str, object] = {
        "declared": str(path),
        "kind": "directory" if resolved.is_dir() else "file",
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if resolved.is_file() and stat.st_size <= 4 * 1024 * 1024:
        result["sha256"] = _sha256_file(resolved)
    elif resolved.is_dir():
        result["directory_marker"] = _directory_marker(resolved)
    return result


def _directory_marker(path: Path) -> str:
    try:
        stat = path.stat()
    except OSError:
        return "unreadable"
    entries: list[tuple[str, int, int]] = []
    try:
        for child in sorted(path.iterdir(), key=lambda item: item.name)[:256]:
            child_stat = child.stat()
            entries.append((child.name, child_stat.st_size, child_stat.st_mtime_ns))
    except OSError:
        return f"stat:{stat.st_ino}:{stat.st_mtime_ns}"
    return digest({"inode": stat.st_ino, "mtime_ns": stat.st_mtime_ns, "entries": entries})


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _source_tree_fingerprint(root: Path) -> dict[str, object]:
    if not root.is_dir():
        return {"state": "missing", "root": str(root)}
    revision = _git(root, ["rev-parse", "HEAD"])
    status = _git(root, ["status", "--porcelain=v1", "--untracked-files=normal"])
    diff = _git_bytes(root, ["diff", "--no-ext-diff", "--binary", "HEAD", "--", "wmloop", "configs", "experiments"])
    untracked = _git(root, ["ls-files", "--others", "--exclude-standard", "--", "wmloop", "configs", "experiments"])
    result: dict[str, object] = {
        "root": str(root),
        "revision": revision.strip() if revision is not None else None,
        "status_sha256": hashlib.sha256((status or "").encode("utf-8")).hexdigest(),
        "diff_sha256": hashlib.sha256(diff or b"").hexdigest(),
        "untracked_sha256": _untracked_source_digest(root, untracked),
    }
    return result


def _untracked_source_digest(root: Path, listing: str | None) -> str:
    rows: list[dict[str, object]] = []
    for relative in sorted((listing or "").splitlines()):
        candidate = root / relative
        try:
            stat = candidate.stat()
        except OSError:
            rows.append({"path": relative, "state": "missing"})
            continue
        if not candidate.is_file() or candidate.is_symlink():
            rows.append({"path": relative, "state": "unsupported"})
            continue
        row: dict[str, object] = {"path": relative, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        if stat.st_size <= 4 * 1024 * 1024:
            row["sha256"] = _sha256_file(candidate)
        rows.append(row)
    return digest(rows)


def _git(root: Path, arguments: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout if completed.returncode == 0 else None


def _git_bytes(root: Path, arguments: list[str]) -> bytes | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout if completed.returncode == 0 else None


def _execution_fingerprint(execution: Mapping[str, object]) -> dict[str, object]:
    fingerprints: dict[str, object] = {}
    for name in sorted(execution):
        value = execution[name]
        if name in {"output_root", "state_root", "workspace_root", "budget_db", "archive_db", "cas_root", "lock_root"}:
            continue
        if name == "asset_bindings" and isinstance(value, Mapping):
            fingerprints[name] = {
                str(parameter): _fingerprint_declared_path(path)
                for parameter, path in sorted(value.items(), key=lambda item: str(item[0]))
            }
        elif name.endswith(("_contract", "_catalog", "_manifest", "_python")):
            fingerprints[name] = _fingerprint_declared_path(value)
        else:
            fingerprints[name] = value
    return fingerprints
