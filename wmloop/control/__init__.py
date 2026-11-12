"""Control-plane utilities for receipts, backups, and operational gates.

Public exports are lazy so importing a component does not initialize workflows.
"""

from __future__ import annotations

from importlib import import_module
from types import MappingProxyType
from typing import Any

_EXPORTS = MappingProxyType(
    {
        "MechanismCompositionError": (
            "wmloop.control.mechanism_composition",
            "MechanismCompositionError",
        ),
        "bind_executable_mechanism": (
            "wmloop.control.mechanism_composition",
            "bind_executable_mechanism",
        ),
        "binding_from_embodiment": (
            "wmloop.control.mechanism_composition",
            "binding_from_embodiment",
        ),
        "compile_mechanism_composition": (
            "wmloop.control.mechanism_composition",
            "compile_mechanism_composition",
        ),
        "discover_mechanism_compositions": (
            "wmloop.control.mechanism_composition",
            "discover_mechanism_compositions",
        ),
        "discover_from_memory": (
            "wmloop.control.mechanism_composition",
            "discover_from_memory",
        ),
        "execute_mechanism_composition": (
            "wmloop.control.mechanism_composition",
            "execute_mechanism_composition",
        ),
        "validate_composition_plan": (
            "wmloop.control.mechanism_composition",
            "validate_composition_plan",
        ),
        "write_composition_plan": (
            "wmloop.control.mechanism_composition",
            "write_composition_plan",
        ),
        "ModelExecutorBootstrapError": (
            "wmloop.control.model_executor_bootstrap",
            "ModelExecutorBootstrapError",
        ),
        "bootstrap_request_template": (
            "wmloop.control.model_executor_bootstrap",
            "bootstrap_request_template",
        ),
        "bootstrap_model_executor": (
            "wmloop.control.model_executor_bootstrap",
            "bootstrap_model_executor",
        ),
        "write_bootstrap_manifest": (
            "wmloop.control.model_executor_bootstrap",
            "write_bootstrap_manifest",
        ),
    }
)

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module, attribute = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    return getattr(import_module(module), attribute)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
