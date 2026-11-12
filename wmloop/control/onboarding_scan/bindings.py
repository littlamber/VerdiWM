"""Read-only onboarding scan components: bindings."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Mapping, Sequence

from wmloop.control.onboarding_scan.types import _MAX_FILES, OnboardingError


def _build_connector(
    repo: Path,
    runtime: Mapping[str, object],
    entrypoints: Sequence[Mapping[str, object]],
    capabilities: Sequence[Mapping[str, object]],
    evaluator_contract: Mapping[str, object],
    asset_bindings: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    grouped: dict[str, list[str]] = {
        "training": [],
        "evaluation": [],
        "rollout": [],
        "inference": [],
    }
    for entrypoint in entrypoints:
        if entrypoint.get("executable") is not True:
            continue
        for kind in entrypoint.get("kinds", []):
            if kind in grouped:
                grouped[kind].append(str(entrypoint["path"]))
    for paths in grouped.values():
        paths.sort()
    return {
        "state": (
            "ready"
            if runtime.get("state") == "ready"
            and evaluator_contract.get("state") == "ready"
            else "binding_required"
        ),
        "kind": "declarative",
        "source_repo": str(repo),
        "runtime_python": runtime.get("selected_python"),
        "entrypoints_by_kind": grouped,
        "capabilities": [
            item["capability"] for item in capabilities if item["state"] == "discovered"
        ],
        "asset_bindings": list(asset_bindings),
        "source_mutation_allowed": False,
        "generated_code": False,
        "manual_bindings": (
            ["evaluator_contract"] if evaluator_contract.get("state") != "ready" else []
        ),
    }


def _discover_asset_bindings(
    entrypoints: Sequence[Mapping[str, object]],
    assets: Sequence[Mapping[str, object]],
    *,
    repo: Path,
    explicit_bindings: Sequence[tuple[str, Path]],
    evaluator_contract: Mapping[str, object],
    additional_parameters: Sequence[str] = (),
) -> list[dict[str, object]]:
    """Resolve input-like CLI paths against discovered repository assets."""

    flags: set[str] = set()
    for entrypoint in entrypoints:
        if entrypoint.get("kinds") == ["unknown"]:
            continue
        flags.update(str(flag) for flag in entrypoint.get("cli_flags", []))
    flags.update(_validate_additional_asset_parameters(additional_parameters))
    required_parameters = _evaluator_asset_parameters(evaluator_contract)
    # Evaluators consume canonical names while adapters may expose a local
    # runner vocabulary.  Bind an evaluator parameter to an explicitly
    # declared adapter asset of the same semantic kind when unambiguous.
    flags.update(required_parameters)
    overrides = _validate_explicit_asset_bindings(
        explicit_bindings,
        flags
        | {
            parameter
            for parameter, _path in explicit_bindings
            if (parameter if parameter.startswith("--") else f"--{parameter}")
            in _validate_additional_asset_parameters(additional_parameters)
        },
    )
    explicit_by_kind: dict[str, list[Path]] = {}
    for parameter, path in overrides.items():
        kind = _binding_kind(parameter.removeprefix("--").replace("-", "_"))
        if kind is not None:
            explicit_by_kind.setdefault(kind, []).append(path)
    rows: list[dict[str, object]] = []
    for flag in sorted(flags):
        field = flag.removeprefix("--").replace("-", "_").lower()
        if any(token in field for token in ("output", "save", "log", "cache", "port")):
            continue
        if not any(
            token in field for token in ("path", "root", "dir", "ckpt", "checkpoint")
        ):
            continue
        kind = _binding_kind(field)
        if kind is None:
            continue
        explicit = overrides.get(flag)
        if explicit is None and flag in required_parameters:
            candidates = explicit_by_kind.get(kind, [])
            if len(candidates) == 1:
                explicit = candidates[0]
        evidence = (
            [str(explicit)]
            if explicit is not None
            else _matching_assets(field, kind, assets)
        )
        resolved = explicit or ((repo / evidence[0]).resolve() if evidence else None)
        rows.append(
            {
                "parameter": flag,
                "kind": kind,
                "state": "discovered" if evidence else "binding_required",
                "evidence": evidence,
                "binding_source": (
                    "explicit" if explicit is not None else "repository_discovery"
                ),
                "resolved_path": str(resolved) if resolved is not None else None,
                "asset_fingerprint": (
                    _asset_fingerprint(resolved) if resolved is not None else None
                ),
                "required_for_evaluator": flag in required_parameters,
            }
        )
    return rows


def _validate_additional_asset_parameters(values: Sequence[str]) -> set[str]:
    parameters: set[str] = set()
    for value in values:
        normalized = value if value.startswith("--") else f"--{value}"
        if not re.fullmatch(r"--[A-Za-z0-9_-]+", normalized):
            raise OnboardingError(f"ASSET_BINDING_PARAMETER_INVALID:{normalized}")
        if normalized in parameters:
            raise OnboardingError(f"ASSET_BINDING_PARAMETER_DUPLICATE:{normalized}")
        parameters.add(normalized)
    return parameters


def _evaluator_asset_parameters(
    evaluator_contract: Mapping[str, object],
) -> set[str]:
    command = evaluator_contract.get("command")
    if evaluator_contract.get("state") != "ready" or not isinstance(command, list):
        return set()
    parameters: set[str] = set()
    for token in command:
        parameters.update(
            placeholder[len("{asset:") : -1]
            for placeholder in re.findall(r"\{asset:--[A-Za-z0-9_-]+\}", str(token))
        )
    return parameters


def _validate_explicit_asset_bindings(
    values: Sequence[tuple[str, Path]], discovered_flags: set[str]
) -> dict[str, Path]:
    bindings: dict[str, Path] = {}
    for parameter, raw_path in values:
        normalized = parameter if parameter.startswith("--") else f"--{parameter}"
        if normalized not in discovered_flags:
            raise OnboardingError(f"ASSET_BINDING_PARAMETER_UNKNOWN:{normalized}")
        if normalized in bindings:
            raise OnboardingError(f"ASSET_BINDING_DUPLICATE:{normalized}")
        raw = Path(raw_path).expanduser()
        if not raw.exists() or raw.is_symlink():
            raise OnboardingError(f"ASSET_BINDING_PATH_INVALID:{normalized}")
        path = raw.resolve()
        bindings[normalized] = path
    return bindings


def _asset_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    entries = (
        [path]
        if path.is_file()
        else sorted(
            item for item in path.rglob("*") if item.is_file() and not item.is_symlink()
        )
    )
    for item in entries[:_MAX_FILES]:
        stat = item.stat()
        relative = item.name if path.is_file() else item.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
    return digest.hexdigest()


def _binding_kind(field: str) -> str | None:
    if "ckpt" in field or "checkpoint" in field:
        return "checkpoint"
    if (
        "dataset" in field
        or "data_" in field
        or field.startswith("data")
        or "droid" in field
        or "subset" in field
    ):
        return "dataset_or_metadata"
    if any(token in field for token in ("model", "policy", "vae", "clip", "svd")):
        return "model_dependency"
    return None


def _matching_assets(
    field: str, kind: str, assets: Sequence[Mapping[str, object]]
) -> list[str]:
    tokens = {
        token
        for token in re.split(r"[^a-z0-9]+", field)
        if token
        not in {
            "path",
            "root",
            "dir",
            "model",
            "checkpoint",
            "ckpt",
            "data",
            "dataset",
            "meta",
            "info",
        }
    }
    candidates = [asset for asset in assets if asset.get("kind") == kind]
    if kind == "checkpoint" and not tokens:
        candidates = sorted(
            candidates,
            key=lambda asset: (-int(asset.get("size_bytes", -1)), str(asset["path"])),
        )
        return [str(asset["path"]) for asset in candidates[:3]]
    matches = [
        str(asset["path"])
        for asset in candidates
        if tokens and any(token in str(asset["path"]).lower() for token in tokens)
    ]
    if kind == "dataset_or_metadata" and not matches and not tokens:
        matches = [str(asset["path"]) for asset in candidates]
    return sorted(matches)[:3]
