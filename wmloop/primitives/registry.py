"""Load, filter and validate the finite M2 intervention action space."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from wmloop.contracts import ContractValidationError, load_yaml_document, validate_instance
from wmloop.primitives.hooks import HookContractError, validate_combination_order, validate_manifest_hooks


class PrimitiveValidationError(ValueError):
    """An intervention was outside the registered bounded action space."""


_COST_RANK = {"very_low": 0, "low": 1, "medium": 2, "high": 3}
_LAYERS = {"L1", "L2", "L3", "L4", "L5"}


@dataclass(frozen=True)
class PrimitiveManifest:
    name: str
    layer: str
    family: str
    params_schema: Mapping[str, Any]
    cost_class: str
    literature: tuple[str, ...]
    targets_failures: tuple[str, ...]
    estimated_gpu_hours: float
    conflicts_with: tuple[str, ...]
    hooks: tuple[str, ...]
    hook_order: Mapping[str, int]
    apply_module: str


@dataclass(frozen=True)
class PrimitiveSelection:
    name: str
    layer: str
    params: Mapping[str, Any]
    cost_class: str
    estimated_gpu_hours: float
    hooks: tuple[str, ...]
    hook_order: Mapping[str, int]


class PrimitiveRegistry:
    def __init__(self, manifests: Mapping[str, PrimitiveManifest]) -> None:
        if not manifests:
            raise PrimitiveValidationError("PRIMITIVE_REGISTRY_EMPTY")
        self._manifests = dict(manifests)

    @classmethod
    def from_root(cls, root: Path) -> "PrimitiveRegistry":
        definitions = Path(root) / "wmloop" / "primitives" / "definitions"
        manifests: dict[str, PrimitiveManifest] = {}
        for manifest_path in sorted(definitions.glob("*/manifest.yaml")):
            manifest = _load_manifest(manifest_path)
            if manifest.name in manifests:
                raise PrimitiveValidationError(f"PRIMITIVE_DUPLICATE:{manifest.name}")
            manifests[manifest.name] = manifest
        registry = cls(manifests)
        registry._validate_references()
        return registry

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._manifests))

    def manifest(self, name: str) -> PrimitiveManifest:
        manifest = self._manifests.get(name)
        if manifest is None:
            raise PrimitiveValidationError(f"PRIMITIVE_UNKNOWN:{name}")
        return manifest

    def digest(self) -> str:
        """Return the frozen registry identity to bind every proposal round."""

        payload = [
            {
                "name": item.name,
                "layer": item.layer,
                "family": item.family,
                "params_schema": item.params_schema,
                "cost_class": item.cost_class,
                "literature": item.literature,
                "targets_failures": item.targets_failures,
                "estimated_gpu_hours": item.estimated_gpu_hours,
                "conflicts_with": item.conflicts_with,
                "hooks": item.hooks,
                "hook_order": item.hook_order,
                "apply_module": item.apply_module,
            }
            for item in sorted(self._manifests.values(), key=lambda item: item.name)
        ]
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        ).hexdigest()

    def available_for(
        self,
        *,
        failure: str | None = None,
        layer: str | None = None,
        max_cost_class: str | None = None,
    ) -> tuple[PrimitiveManifest, ...]:
        if layer is not None and layer not in _LAYERS:
            raise PrimitiveValidationError("PRIMITIVE_LAYER_INVALID")
        if max_cost_class is not None and max_cost_class not in _COST_RANK:
            raise PrimitiveValidationError("PRIMITIVE_COST_CLASS_INVALID")
        maximum = _COST_RANK[max_cost_class] if max_cost_class else None
        return tuple(
            manifest
            for manifest in sorted(self._manifests.values(), key=lambda item: item.name)
            if (failure is None or failure in manifest.targets_failures)
            and (layer is None or layer == manifest.layer)
            and (maximum is None or _COST_RANK[manifest.cost_class] <= maximum)
        )

    def validate_selection(self, name: str, params: Mapping[str, Any]) -> PrimitiveSelection:
        manifest = self._manifests.get(name)
        if manifest is None:
            raise PrimitiveValidationError(f"PRIMITIVE_UNKNOWN:{name}")
        try:
            validate_instance(manifest.params_schema, params)
        except ContractValidationError as exc:
            raise PrimitiveValidationError(f"PRIMITIVE_PARAMS_INVALID:{name}:{exc}") from exc
        return PrimitiveSelection(
            name=manifest.name,
            layer=manifest.layer,
            params=dict(params),
            cost_class=manifest.cost_class,
            estimated_gpu_hours=manifest.estimated_gpu_hours,
            hooks=manifest.hooks,
            hook_order=dict(manifest.hook_order),
        )

    def validate_combination(self, selections: list[tuple[str, Mapping[str, Any]]]) -> tuple[PrimitiveSelection, ...]:
        validated = tuple(self.validate_selection(name, params) for name, params in selections)
        if len({selection.name for selection in validated}) != len(validated):
            raise PrimitiveValidationError("PRIMITIVE_DUPLICATE_SELECTION")
        selected_names = {selection.name for selection in validated}
        for selection in validated:
            conflicts = set(self._manifests[selection.name].conflicts_with) & selected_names
            conflicts.discard(selection.name)
            if conflicts:
                raise PrimitiveValidationError(
                    f"PRIMITIVE_CONFLICT:{selection.name}:{','.join(sorted(conflicts))}"
                )
        if sum(selection.layer == "L3" for selection in validated) > 1:
            raise PrimitiveValidationError("PRIMITIVE_DEFAULT_L3_CONFLICT")
        bindings: dict[str, list[int]] = {}
        for selection in validated:
            for hook in selection.hooks:
                bindings.setdefault(hook, []).append(int(selection.hook_order[hook]))
        try:
            validate_combination_order(bindings)
        except HookContractError as exc:
            raise PrimitiveValidationError(str(exc)) from exc
        return validated

    def _validate_references(self) -> None:
        for manifest in self._manifests.values():
            unknown = set(manifest.conflicts_with) - set(self._manifests)
            if unknown:
                raise PrimitiveValidationError(
                    f"PRIMITIVE_CONFLICT_REFERENCE_UNKNOWN:{manifest.name}:{','.join(sorted(unknown))}"
                )


def _load_manifest(path: Path) -> PrimitiveManifest:
    try:
        raw = load_yaml_document(path)
        name = _nonempty(raw, "name")
        layer = _nonempty(raw, "layer")
        family = _nonempty(raw, "family")
        params_schema = raw["params_schema"]
        cost_class = _nonempty(raw, "cost_class")
        literature = _string_tuple(raw, "literature", minimum=1)
        targets_failures = _string_tuple(raw, "targets_failures", minimum=1)
        conflicts_with = _string_tuple(raw, "conflicts_with", minimum=0)
        hooks = _string_tuple(raw, "hooks", minimum=1)
        raw_hook_order = raw["hook_order"]
        if not isinstance(raw_hook_order, Mapping):
            raise ValueError("hook_order")
        hook_order = {str(hook): order for hook, order in raw_hook_order.items()}
        validate_manifest_hooks(hooks, hook_order)
        apply_module = _nonempty(raw, "apply_module")
        estimated_gpu_hours = float(raw["estimated_gpu_hours"])
    except (ContractValidationError, KeyError, TypeError, ValueError) as exc:
        raise PrimitiveValidationError(f"PRIMITIVE_MANIFEST_INVALID:{path}") from exc
    if layer not in _LAYERS or cost_class not in _COST_RANK or estimated_gpu_hours < 0:
        raise PrimitiveValidationError(f"PRIMITIVE_MANIFEST_INVALID:{path}")
    if not isinstance(params_schema, Mapping):
        raise PrimitiveValidationError(f"PRIMITIVE_MANIFEST_INVALID:{path}")
    try:
        validate_instance(params_schema, {})
    except ContractValidationError:
        # Required parameters are expected to reject an empty object; schema parsing
        # is exercised by selection validation with a concrete parameter mapping.
        pass
    return PrimitiveManifest(
        name=name,
        layer=layer,
        family=family,
        params_schema=dict(params_schema),
        cost_class=cost_class,
        literature=literature,
        targets_failures=targets_failures,
        estimated_gpu_hours=estimated_gpu_hours,
        conflicts_with=conflicts_with,
        hooks=hooks,
        hook_order=hook_order,
        apply_module=apply_module,
    )


def _nonempty(raw: Mapping[str, Any], field: str) -> str:
    value = raw[field]
    if not isinstance(value, str) or not value:
        raise ValueError(field)
    return value


def _string_tuple(raw: Mapping[str, Any], field: str, *, minimum: int) -> tuple[str, ...]:
    value = raw.get(field, [])
    if not isinstance(value, list) or len(value) < minimum or any(not isinstance(item, str) or not item for item in value):
        raise ValueError(field)
    return tuple(value)
