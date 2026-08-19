"""Manifest-backed capability catalog for proposal workflow plugins.

Plugins remain declarative capability contracts.  The registry can evolve
without adding branches to the kernel, while execution and scientific
authority stay owned by the existing control plane and frozen verifier.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from wmloop.contracts import ContractValidationError, validate_document


class WorkflowPluginError(ValueError):
    """A workflow plugin registry or selection is outside the action space."""


@dataclass(frozen=True)
class WorkflowPlugin:
    plugin_id: str
    version: str
    role: str
    input_contract: str
    output_artifact: str
    side_effect: str
    cost_model: str
    cross_model_safe: bool
    authority_level: str
    required_model_capabilities: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "plugin_id": self.plugin_id,
            "version": self.version,
            "role": self.role,
            "input_contract": self.input_contract,
            "output_artifact": self.output_artifact,
            "side_effect": self.side_effect,
            "cost_model": self.cost_model,
            "cross_model_safe": self.cross_model_safe,
            "authority_level": self.authority_level,
            "required_model_capabilities": list(self.required_model_capabilities),
        }


@dataclass(frozen=True)
class WorkflowRegistry:
    registry_id: str
    plugins: tuple[WorkflowPlugin, ...]
    workflow_requirements: Mapping[tuple[str, str], frozenset[str]]
    digest: str
    source_path: Path


def _default_registry(root: Path | None = None) -> Path:
    base = (root or Path(__file__).resolve().parents[2]).expanduser().resolve()
    return base / "configs" / "plugins" / "core_workflows_v1.json"


def _load_registry_file(path_text: str, root_text: str) -> WorkflowRegistry:
    path = Path(path_text)
    root = Path(root_text)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("registry object required")
        validate_document("workflow_plugin_registry", payload, root=root)
    except (OSError, json.JSONDecodeError, ValueError, ContractValidationError) as exc:
        raise WorkflowPluginError("WORKFLOW_PLUGIN_REGISTRY_INVALID") from exc

    plugin_rows = payload["plugins"]
    plugins = tuple(
        WorkflowPlugin(
            plugin_id=str(row["plugin_id"]),
            version=str(row["version"]),
            role=str(row["role"]),
            input_contract=str(row["input_contract"]),
            output_artifact=str(row["output_artifact"]),
            side_effect=str(row["side_effect"]),
            cost_model=str(row["cost_model"]),
            cross_model_safe=bool(row["cross_model_safe"]),
            authority_level=str(row["authority_level"]),
            required_model_capabilities=tuple(
                sorted(str(value) for value in row["required_model_capabilities"])
            ),
        )
        for row in plugin_rows
    )
    plugin_ids = [plugin.plugin_id for plugin in plugins]
    if len(plugin_ids) != len(set(plugin_ids)):
        raise WorkflowPluginError("WORKFLOW_PLUGIN_REGISTRY_DUPLICATE_PLUGIN")
    for plugin in plugins:
        capabilities = plugin.required_model_capabilities
        if len(capabilities) != len(set(capabilities)):
            raise WorkflowPluginError(
                "WORKFLOW_PLUGIN_REGISTRY_DUPLICATE_CAPABILITY:"
                + plugin.plugin_id
            )

    requirements: dict[tuple[str, str], frozenset[str]] = {}
    known = set(plugin_ids)
    for row in payload["workflows"]:
        workflow_id = str(row["workflow_id"])
        workflow_version = str(row["workflow_version"])
        workflow_key = (workflow_id, workflow_version)
        if workflow_key in requirements:
            raise WorkflowPluginError("WORKFLOW_PLUGIN_REGISTRY_DUPLICATE_WORKFLOW")
        required = frozenset(str(value) for value in row["required_plugins"])
        unknown = sorted(required - known)
        if unknown:
            raise WorkflowPluginError(
                "WORKFLOW_PLUGIN_REGISTRY_UNKNOWN_REQUIREMENT:" + ",".join(unknown)
            )
        requirements[workflow_key] = required
    digest = hashlib.sha256(_canonical_json(payload)).hexdigest()
    return WorkflowRegistry(
        registry_id=str(payload["registry_id"]),
        plugins=tuple(sorted(plugins, key=lambda plugin: plugin.plugin_id)),
        workflow_requirements=requirements,
        digest=digest,
        source_path=path,
    )


def load_workflow_registry(
    *, registry_path: Path | None = None, root: Path | None = None
) -> WorkflowRegistry:
    """Load one versioned registry without importing plugin implementation code."""

    base = (root or Path(__file__).resolve().parents[2]).expanduser().resolve()
    path = (
        Path(registry_path).expanduser().resolve()
        if registry_path is not None
        else _default_registry(base)
    )
    if not path.is_file() or path.is_symlink():
        raise WorkflowPluginError("WORKFLOW_PLUGIN_REGISTRY_INVALID")
    return _load_registry_file(str(path), str(base))


def workflow_plugins(
    *, registry_path: Path | None = None, root: Path | None = None
) -> tuple[WorkflowPlugin, ...]:
    """Return registered plugin contracts in deterministic order."""

    return load_workflow_registry(registry_path=registry_path, root=root).plugins


def select_workflow_plugins(
    ids: Sequence[object],
    *,
    registry_path: Path | None = None,
    root: Path | None = None,
) -> tuple[WorkflowPlugin, ...]:
    selected = {str(value) for value in ids}
    if len(selected) != len(ids):
        raise WorkflowPluginError("WORKFLOW_PLUGIN_DUPLICATE")
    known = {
        plugin.plugin_id: plugin
        for plugin in workflow_plugins(registry_path=registry_path, root=root)
    }
    unknown = sorted(selected - set(known))
    if unknown:
        raise WorkflowPluginError(
            "WORKFLOW_PLUGIN_NOT_ADMITTED:" + ",".join(unknown)
        )
    return tuple(known[name] for name in sorted(selected))


def require_workflow_plugins(
    workflow_id: object,
    ids: Sequence[object],
    *,
    workflow_version: object | None = None,
    registry_path: Path | None = None,
    root: Path | None = None,
) -> None:
    registry = load_workflow_registry(registry_path=registry_path, root=root)
    workflow_name = str(workflow_id)
    if workflow_version is None:
        matches = [
            required
            for (registered_id, _), required in registry.workflow_requirements.items()
            if registered_id == workflow_name
        ]
        if len(matches) > 1:
            raise WorkflowPluginError(
                f"WORKFLOW_VERSION_REQUIRED:{workflow_name}"
            )
        required = matches[0] if matches else None
    else:
        required = registry.workflow_requirements.get(
            (workflow_name, str(workflow_version))
        )
    if required is None:
        suffix = (
            f":{workflow_version}" if workflow_version is not None else ""
        )
        raise WorkflowPluginError(f"WORKFLOW_NOT_ADMITTED:{workflow_name}{suffix}")
    missing = sorted(required - {str(value) for value in ids})
    if missing:
        raise WorkflowPluginError("WORKFLOW_PLUGIN_REQUIRED:" + ",".join(missing))


def required_model_capabilities(
    plugins: Sequence[WorkflowPlugin],
) -> tuple[str, ...]:
    """Return the model capabilities required by only the selected plugins."""

    return tuple(
        sorted(
            {
                capability
                for plugin in plugins
                for capability in plugin.required_model_capabilities
            }
        )
    )


def workflow_capability_digest(plugins: Sequence[WorkflowPlugin]) -> str:
    payload = [
        plugin.to_dict()
        for plugin in sorted(plugins, key=lambda plugin: plugin.plugin_id)
    ]
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
