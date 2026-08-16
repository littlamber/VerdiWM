"""Resolve a four-field user request into a versioned adapter execution."""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from wmloop.constitution import ConstitutionalFreezeError, verify_constitutional_freeze
from wmloop.contracts import ContractValidationError, validate_document


class AdapterProfileError(ValueError):
    """A profile or one of its local bindings is invalid."""


_BUDGET_PATTERN = re.compile(
    r"^\s*(?P<value>(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+))\s*"
    r"(?P<unit>gpu(?:[-_ ]?hours?)?|gpu[-_ ]?h|hours?|h)?\s*$",
    re.IGNORECASE,
)
_ENV_CANDIDATE = re.compile(r"^\{env:(?P<name>[A-Z][A-Z0-9_]*)\}$")


@dataclass(frozen=True)
class ResolvedAdapter:
    profile_id: str
    model_family: str
    capability_level: str
    execution: dict[str, Any]
    constitution_freeze: str


def parse_gpu_budget(value: object) -> float:
    """Return a positive GPU-hour value from CLI/API budget syntax."""

    if isinstance(value, Mapping):
        value = value.get("gpu_hours")
    if isinstance(value, bool):
        raise AdapterProfileError("BUDGET_INVALID")
    if isinstance(value, (int, float)):
        hours = float(value)
    elif isinstance(value, str):
        match = _BUDGET_PATTERN.fullmatch(value)
        if match is None:
            raise AdapterProfileError("BUDGET_INVALID")
        hours = float(match.group("value"))
    else:
        raise AdapterProfileError("BUDGET_INVALID")
    if not math.isfinite(hours) or hours <= 0:
        raise AdapterProfileError("BUDGET_INVALID")
    return hours


def compile_adapter_execution(
    *,
    campaign_id: str,
    model: Path,
    data: Path,
    goal: str,
    budget: object,
    campaign_root: Path,
    adapter: str | None = None,
    adapter_profile_path: Path | None = None,
    runtime_python: Path | None = None,
    asset_overrides: Mapping[str, object] | None = None,
    project_root: Path | None = None,
) -> ResolvedAdapter:
    """Compile portable adapter rules into an absolute pipeline contract."""

    root = (project_root or Path(__file__).resolve().parents[2]).resolve()
    model_path = _existing_path(model, directory=True, code="MODEL_PATH_INVALID")
    data_path = _existing_path(data, directory=False, code="DATA_PATH_INVALID")
    normalized_goal = goal.strip()
    if not normalized_goal:
        raise AdapterProfileError("GOAL_REQUIRED")
    profile = _select_profile(
        root=root,
        model=model_path,
        goal=normalized_goal,
        adapter=adapter,
        profile_path=adapter_profile_path,
    )
    constitution_path = _project_file(
        root, str(profile["constitution_freeze"]), "ADAPTER_CONSTITUTION_INVALID"
    )
    try:
        freeze = json.loads(constitution_path.read_text(encoding="utf-8"))
        verify_constitutional_freeze(freeze, root=root)
    except (OSError, json.JSONDecodeError, ConstitutionalFreezeError) as exc:
        raise AdapterProfileError("ADAPTER_CONSTITUTION_INVALID") from exc

    values = {
        "model": str(model_path),
        "model_parent": str(model_path.parent),
        "data": str(data_path),
        "data_parent": str(data_path.parent),
    }
    runtime = (
        _existing_executable(runtime_python, "RUNTIME_PYTHON_INVALID")
        if runtime_python is not None
        else _resolve_candidates(
            profile["runtime_candidates"],
            values=values,
            code="RUNTIME_PYTHON_NOT_FOUND",
            executable=True,
        )
    )
    overrides = {
        _normalize_parameter(str(parameter)): _existing_path(
            Path(str(path)), directory=False, code=f"ASSET_INVALID:{parameter}"
        )
        for parameter, path in (asset_overrides or {}).items()
    }
    assets: dict[str, str] = {}
    for raw_binding in profile["asset_bindings"]:
        binding = dict(raw_binding)
        parameter = _normalize_parameter(str(binding["parameter"]))
        path = overrides.pop(parameter, None)
        if path is None:
            path = _resolve_candidates(
                binding["candidates"],
                values=values,
                code=f"ASSET_NOT_FOUND:{parameter}",
            )
        assets[parameter] = str(path)
    if overrides:
        raise AdapterProfileError(
            f"ASSET_OVERRIDE_UNKNOWN:{','.join(sorted(overrides))}"
        )

    state_root = Path(campaign_root).expanduser().resolve().parent
    execution: dict[str, Any] = {
        "kind": "pipeline",
        "repo_root": str(model_path),
        "output_root": str(state_root / "runs" / campaign_id),
        "evaluator_contract": str(
            _project_file(
                root,
                str(profile["evaluator_contract"]),
                "ADAPTER_EVALUATOR_INVALID",
            )
        ),
        "runtime_python": str(runtime),
        "asset_bindings": assets,
        "probe_imports": bool(profile["probe_imports"]),
        "archive_db": str(state_root / "archive.db"),
        "cas_root": str(state_root / "artifacts"),
        "budget_db": str(state_root / "budgets" / f"{campaign_id}.db"),
        "retrieval_db": str(state_root / "retrieval.db"),
        "budget_total_gpu_hours": parse_gpu_budget(budget),
    }
    probe_contract = profile.get("probe_contract")
    if probe_contract is not None:
        execution["probe_contract"] = str(
            _project_file(root, str(probe_contract), "ADAPTER_PROBE_INVALID")
        )
    candidate_catalog = profile.get("candidate_catalog")
    if candidate_catalog is not None:
        execution["candidate_catalog"] = str(
            _project_file(
                root,
                str(candidate_catalog),
                "ADAPTER_CANDIDATE_CATALOG_INVALID",
            )
        )
    settlement_candidates = profile.get("settlement_manifest_candidates")
    if settlement_candidates is not None:
        execution["settlement_manifest"] = str(
            _resolve_candidates(
                settlement_candidates,
                values=values,
                code="ADAPTER_SETTLEMENT_MANIFEST_NOT_FOUND",
            )
        )
    return ResolvedAdapter(
        profile_id=str(profile["profile_id"]),
        model_family=str(profile["model_family"]),
        capability_level=str(profile["capability_level"]),
        execution=execution,
        constitution_freeze=str(constitution_path),
    )


def _select_profile(
    *,
    root: Path,
    model: Path,
    goal: str,
    adapter: str | None,
    profile_path: Path | None,
) -> dict[str, Any]:
    paths = (
        [Path(profile_path).expanduser().resolve()]
        if profile_path is not None
        else sorted((root / "configs" / "adapters").glob("*.json"))
    )
    profiles = [_load_profile(path, root=root) for path in paths]
    if adapter and adapter != "auto":
        matches = [
            profile
            for profile in profiles
            if adapter == profile["profile_id"] or adapter in profile["aliases"]
        ]
        if len(matches) != 1:
            raise AdapterProfileError("ADAPTER_PROFILE_NOT_FOUND")
        _require_repo_markers(matches[0], model=model)
        return matches[0]
    compatible = [
        profile for profile in profiles if _repo_markers_present(profile, model=model)
    ]
    if not compatible:
        raise AdapterProfileError("ADAPTER_PROFILE_NOT_FOUND")
    lowered_goal = goal.casefold()
    scored = sorted(
        (
            sum(str(keyword).casefold() in lowered_goal for keyword in profile["goal_keywords"]),
            str(profile["profile_id"]),
            profile,
        )
        for profile in compatible
    )
    best_score = scored[-1][0]
    best = [profile for score, _name, profile in scored if score == best_score]
    if len(best) != 1:
        raise AdapterProfileError("ADAPTER_PROFILE_AMBIGUOUS")
    return best[0]


def _load_profile(path: Path, *, root: Path) -> dict[str, Any]:
    try:
        profile = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AdapterProfileError(f"ADAPTER_PROFILE_INVALID:{path}") from exc
    if not isinstance(profile, dict):
        raise AdapterProfileError(f"ADAPTER_PROFILE_INVALID:{path}")
    try:
        validate_document("adapter_profile", profile, root=root)
    except ContractValidationError as exc:
        raise AdapterProfileError(f"ADAPTER_PROFILE_INVALID:{path}") from exc
    parameters = [str(row["parameter"]) for row in profile["asset_bindings"]]
    if len(parameters) != len(set(parameters)):
        raise AdapterProfileError(f"ADAPTER_PROFILE_DUPLICATE_ASSET:{path}")
    return profile


def _require_repo_markers(profile: Mapping[str, Any], *, model: Path) -> None:
    if not _repo_markers_present(profile, model=model):
        raise AdapterProfileError("ADAPTER_MODEL_INCOMPATIBLE")


def _repo_markers_present(profile: Mapping[str, Any], *, model: Path) -> bool:
    return all((model / str(marker)).is_file() for marker in profile["repo_markers"])


def _resolve_candidates(
    candidates: object,
    *,
    values: Mapping[str, str],
    code: str,
    executable: bool = False,
) -> Path:
    if not isinstance(candidates, list):
        raise AdapterProfileError(code)
    for raw in candidates:
        candidate = str(raw)
        env_match = _ENV_CANDIDATE.fullmatch(candidate)
        if env_match is not None:
            candidate = os.environ.get(env_match.group("name"), "")
            if not candidate:
                continue
        else:
            try:
                candidate = candidate.format_map(values)
            except (KeyError, ValueError) as exc:
                raise AdapterProfileError("ADAPTER_PROFILE_TEMPLATE_INVALID") from exc
        path = Path(candidate).expanduser()
        if not path.exists() or (path.is_symlink() and not executable):
            continue
        resolved = path.resolve()
        if executable and (not resolved.is_file() or not os.access(resolved, os.X_OK)):
            continue
        return resolved
    raise AdapterProfileError(code)


def _project_file(root: Path, relative: str, code: str) -> Path:
    path = (root / relative).resolve()
    if root not in path.parents or not path.is_file() or path.is_symlink():
        raise AdapterProfileError(code)
    return path


def _existing_path(path: Path, *, directory: bool, code: str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.exists() or candidate.is_symlink():
        raise AdapterProfileError(code)
    resolved = candidate.resolve()
    if directory and not resolved.is_dir():
        raise AdapterProfileError(code)
    return resolved


def _existing_executable(path: Path, code: str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.exists():
        raise AdapterProfileError(code)
    resolved = candidate.resolve()
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise AdapterProfileError(code)
    return resolved


def _normalize_parameter(value: str) -> str:
    parameter = value if value.startswith("--") else f"--{value}"
    if re.fullmatch(r"--[A-Za-z0-9_-]+", parameter) is None:
        raise AdapterProfileError(f"ASSET_PARAMETER_INVALID:{value}")
    return parameter
