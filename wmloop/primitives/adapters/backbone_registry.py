"""Validated mapping from portable primitives to backbone-specific hooks."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wmloop.contracts import ContractValidationError, validate_document
from wmloop.primitives.registry import PrimitiveRegistry, PrimitiveValidationError


class BackbonePrimitiveRegistryError(ValueError):
    """A backbone primitive mapping is malformed or intent-misaligned."""


@dataclass(frozen=True)
class BackbonePrimitiveBinding:
    primitive: str
    source_hooks: tuple[str, ...]
    target_hooks: tuple[str, ...]
    runtime_state: str
    implementation_refs: tuple[str, ...]


@dataclass(frozen=True)
class BackbonePrimitiveRegistry:
    registry_id: str
    backbone_family: str
    goal_family: str
    source_registry_digest: str
    hook_adapter: str
    bindings: tuple[BackbonePrimitiveBinding, ...]
    registry_digest: str

    @property
    def runtime_ready_primitives(self) -> tuple[str, ...]:
        return tuple(binding.primitive for binding in self.bindings if binding.runtime_state == "runtime_ready_external")

    @property
    def materializable_primitives(self) -> tuple[str, ...]:
        return tuple(binding.primitive for binding in self.bindings if binding.runtime_state == "mapped_for_materialization")


def load_backbone_primitive_registry(path: Path, *, root: Path | None = None) -> BackbonePrimitiveRegistry:
    base = (root or Path(__file__).resolve().parents[3]).resolve()
    registry_path = _resolve_inside(base, Path(path))
    try:
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
        validate_document("backbone_primitive_registry", payload, root=base)
    except (OSError, json.JSONDecodeError, ContractValidationError) as exc:
        raise BackbonePrimitiveRegistryError("BACKBONE_PRIMITIVE_REGISTRY_INVALID") from exc
    if not isinstance(payload, Mapping):
        raise BackbonePrimitiveRegistryError("BACKBONE_PRIMITIVE_REGISTRY_INVALID")

    portable = PrimitiveRegistry.from_root(base)
    if str(payload["source_registry_digest"]) != portable.digest():
        raise BackbonePrimitiveRegistryError("BACKBONE_PRIMITIVE_SOURCE_REGISTRY_MISMATCH")

    hook_adapter = _resolve_inside(base, Path(str(payload["hook_adapter"])))
    if not hook_adapter.is_file():
        raise BackbonePrimitiveRegistryError("BACKBONE_PRIMITIVE_HOOK_ADAPTER_MISSING")

    raw_bindings = list(payload["bindings"])
    names = [str(item["primitive"]) for item in raw_bindings]
    if len(names) != len(set(names)):
        raise BackbonePrimitiveRegistryError("BACKBONE_PRIMITIVE_DUPLICATE")

    bindings: list[BackbonePrimitiveBinding] = []
    for item in raw_bindings:
        name = str(item["primitive"])
        try:
            manifest = portable.manifest(name)
        except PrimitiveValidationError as exc:
            raise BackbonePrimitiveRegistryError(f"BACKBONE_PRIMITIVE_UNKNOWN:{name}") from exc
        source_hooks = tuple(str(value) for value in item["source_hooks"])
        target_hooks = tuple(str(value) for value in item["target_hooks"])
        if source_hooks != manifest.hooks:
            raise BackbonePrimitiveRegistryError(f"BACKBONE_PRIMITIVE_SOURCE_HOOK_MISMATCH:{name}")
        if target_hooks != source_hooks:
            raise BackbonePrimitiveRegistryError(f"BACKBONE_PRIMITIVE_TARGET_HOOK_MISMATCH:{name}")
        if item["mechanism_preserved"] is not True:
            raise BackbonePrimitiveRegistryError(f"BACKBONE_PRIMITIVE_INTENT_NOT_PRESERVED:{name}")
        runtime_state = str(item["runtime_state"])
        implementation_refs = tuple(str(value) for value in item["implementation_refs"])
        if runtime_state == "runtime_ready_external" and not item["config_mapping"]:
            raise BackbonePrimitiveRegistryError(f"BACKBONE_PRIMITIVE_RUNTIME_MAPPING_EMPTY:{name}")
        bindings.append(
            BackbonePrimitiveBinding(
                primitive=name,
                source_hooks=source_hooks,
                target_hooks=target_hooks,
                runtime_state=runtime_state,
                implementation_refs=implementation_refs,
            )
        )

    expected_digest = _registry_digest(payload)
    if str(payload["registry_digest"]) != expected_digest:
        raise BackbonePrimitiveRegistryError("BACKBONE_PRIMITIVE_REGISTRY_DIGEST_MISMATCH")
    if not any(binding.runtime_state == "runtime_ready_external" for binding in bindings):
        raise BackbonePrimitiveRegistryError("BACKBONE_PRIMITIVE_RUNTIME_READY_EMPTY")
    return BackbonePrimitiveRegistry(
        registry_id=str(payload["registry_id"]),
        backbone_family=str(payload["backbone_family"]),
        goal_family=str(payload["goal_family"]),
        source_registry_digest=str(payload["source_registry_digest"]),
        hook_adapter=str(payload["hook_adapter"]),
        bindings=tuple(bindings),
        registry_digest=str(payload["registry_digest"]),
    )


def registry_digest(payload: Mapping[str, Any]) -> str:
    """Return the deterministic identity used when authoring a registry."""

    return _registry_digest(payload)


def _registry_digest(payload: Mapping[str, Any]) -> str:
    body = {key: value for key, value in payload.items() if key != "registry_digest"}
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resolve_inside(root: Path, path: Path) -> Path:
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    if resolved != root and root not in resolved.parents:
        raise BackbonePrimitiveRegistryError("BACKBONE_PRIMITIVE_PATH_OUTSIDE_ROOT")
    if not resolved.is_file():
        raise BackbonePrimitiveRegistryError(f"BACKBONE_PRIMITIVE_PATH_MISSING:{path}")
    return resolved
