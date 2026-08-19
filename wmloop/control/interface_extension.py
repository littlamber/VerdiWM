"""Open, method-scoped interface-extension proposals.

Unlike the legacy ABI registries, this contract does not enumerate method
families or function signatures in advance.  A proposal describes the semantic
surface a Method IR needs.  It remains non-executable until independent
conformance tests and the controller admit a concrete implementation.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from wmloop.contracts import ContractValidationError, validate_document
from wmloop.control.open_method_ir import validate_method_ir
from wmloop.geometry.evidence_ir import reject_runtime_bindings
from wmloop.geometry.types import GeometryValidationError


class MethodInterfaceExtensionError(ValueError):
    """A method-scoped interface proposal crossed a semantic boundary."""


def build_method_interface_extension(
    *,
    method_ir: Mapping[str, object],
    requested_surface: str,
    semantic_role: str,
    typed_inputs: Sequence[Mapping[str, object]],
    typed_outputs: Sequence[Mapping[str, object]],
    side_effect_class: str,
    conformance_tests: Sequence[str],
    negative_tests: Sequence[str],
    root: Path | None = None,
) -> dict[str, object]:
    """Build a new semantic interface proposal without granting execution."""

    validate_method_ir(method_ir, root=root)
    body: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "verdiwm-method-interface-extension",
        "extension_id": "",
        "method_id": method_ir["method_id"],
        "requested_surface": _text(requested_surface, "METHOD_INTERFACE_SURFACE_INVALID"),
        "semantic_role": _text(semantic_role, "METHOD_INTERFACE_ROLE_INVALID"),
        "typed_inputs": [dict(row) for row in typed_inputs],
        "typed_outputs": [dict(row) for row in typed_outputs],
        "side_effect_class": side_effect_class,
        "conformance_tests": [_text(value, "METHOD_INTERFACE_TEST_INVALID") for value in conformance_tests],
        "negative_tests": [_text(value, "METHOD_INTERFACE_TEST_INVALID") for value in negative_tests],
        "authority": {
            "source_mutation": False,
            "evaluator_mutation": False,
            "active_metric_mutation": False,
            "gpu_scheduling": False,
            "promotion": False,
        },
        "state": "proposed",
        "claim_boundary": (
            "This is a method-scoped semantic interface proposal. It grants no source, "
            "evaluator, metric, GPU scheduling, execution, or promotion authority."
        ),
    }
    body["extension_id"] = "method-interface-extension-" + _digest(body, "extension_id")[:24]
    validate_method_interface_extension(body, root=root)
    return body


def validate_method_interface_extension(
    document: Mapping[str, object], *, root: Path | None = None
) -> None:
    try:
        reject_runtime_bindings(document)
        validate_document("method_interface_extension", document, root=root)
    except (ContractValidationError, GeometryValidationError) as exc:
        raise MethodInterfaceExtensionError(f"METHOD_INTERFACE_SCHEMA_INVALID:{exc}") from exc
    expected = "method-interface-extension-" + _digest(document, "extension_id")[:24]
    if document.get("extension_id") != expected:
        raise MethodInterfaceExtensionError("METHOD_INTERFACE_DIGEST_MISMATCH")
    for field in ("typed_inputs", "typed_outputs"):
        rows = document.get(field)
        if not isinstance(rows, list):
            raise MethodInterfaceExtensionError("METHOD_INTERFACE_PORTS_INVALID")
        names = [str(row.get("name")) for row in rows if isinstance(row, Mapping)]
        if len(names) != len(rows) or len(names) != len(set(names)):
            raise MethodInterfaceExtensionError("METHOD_INTERFACE_PORTS_INVALID")


def _text(value: object, code: str) -> str:
    text = str(value).strip()
    if not text:
        raise MethodInterfaceExtensionError(code)
    return text


def _digest(document: Mapping[str, object], excluded: str) -> str:
    payload = {key: value for key, value in document.items() if key != excluded}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode()
    ).hexdigest()
