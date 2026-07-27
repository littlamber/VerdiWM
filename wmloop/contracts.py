"""Versioned, offline-capable contract validation for wm-loop boundaries.

The production dependency set includes :mod:`jsonschema` and PyYAML.  M0 must
also be inspectable on a disconnected shared host, so the narrow YAML and JSON
Schema subset used by the checked-in contracts has a deterministic stdlib
fallback.  An unsupported schema keyword fails closed instead of silently
weakening a contract.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

try:  # Production validation uses the pinned Draft 2020-12 implementation.
    from jsonschema import Draft202012Validator
    from jsonschema.exceptions import SchemaError
except ImportError:  # pragma: no cover - exercised only on intentionally offline bootstrap hosts
    Draft202012Validator = None  # type: ignore[assignment,misc]
    SchemaError = ValueError  # type: ignore[assignment,misc]


class ContractValidationError(ValueError):
    """A document or frozen source failed its versioned contract."""


def _root(root: Path | None) -> Path:
    return (root or Path(__file__).resolve().parents[1]).resolve()


def load_yaml_document(path: Path) -> dict[str, Any]:
    """Load the restricted YAML subset used for checked-in configuration.

    This parser deliberately accepts mappings, block lists and JSON-style
    scalar lists only.  It prevents a missing optional parser dependency from
    converting a configuration error into an implicit default.
    """

    raw_text = path.read_text(encoding="utf-8")
    if raw_text.lstrip().startswith("{"):
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise ContractValidationError(f"YAML_JSON_INVALID:{path}") from exc
        if not isinstance(parsed, dict):
            raise ContractValidationError(f"YAML_ROOT_OBJECT_REQUIRED:{path}")
        return parsed
    lines: list[tuple[int, str]] = []
    for raw in raw_text.splitlines():
        content = _strip_yaml_comment(raw).rstrip()
        if not content.strip():
            continue
        indent = len(content) - len(content.lstrip(" "))
        if "\t" in raw[: len(raw) - len(raw.lstrip())]:
            raise ContractValidationError(f"YAML_TAB_INDENT:{path}")
        lines.append((indent, content.strip()))
    if not lines:
        raise ContractValidationError(f"YAML_EMPTY:{path}")
    value, next_index = _parse_yaml_block(lines, 0, lines[0][0])
    if next_index != len(lines) or not isinstance(value, dict):
        raise ContractValidationError(f"YAML_ROOT_OBJECT_REQUIRED:{path}")
    return value


def _parse_yaml_block(lines: list[tuple[int, str]], index: int, indent: int) -> tuple[Any, int]:
    if index >= len(lines) or lines[index][0] != indent:
        raise ContractValidationError("YAML_INDENTATION")
    is_list = lines[index][1].startswith("- ")
    value: list[Any] | dict[str, Any] = [] if is_list else {}
    while index < len(lines):
        current_indent, content = lines[index]
        if current_indent < indent:
            break
        if current_indent > indent:
            raise ContractValidationError("YAML_UNEXPECTED_INDENT")
        if is_list:
            if not content.startswith("- "):
                raise ContractValidationError("YAML_MIXED_CONTAINER")
            item = content[2:].strip()
            if not item:
                if index + 1 >= len(lines) or lines[index + 1][0] <= indent:
                    raise ContractValidationError("YAML_LIST_VALUE_REQUIRED")
                parsed, index = _parse_yaml_block(lines, index + 1, lines[index + 1][0])
                value.append(parsed)
                continue
            value.append(_yaml_scalar(item))
            index += 1
            continue
        if content.startswith("- ") or ":" not in content:
            raise ContractValidationError("YAML_MAPPING_REQUIRED")
        key, scalar = content.split(":", 1)
        key = key.strip()
        if not key or key in value:
            raise ContractValidationError("YAML_INVALID_OR_DUPLICATE_KEY")
        scalar = scalar.strip()
        if scalar:
            value[key] = _yaml_scalar(scalar)
            index += 1
            continue
        if index + 1 >= len(lines) or lines[index + 1][0] <= indent:
            value[key] = {}
            index += 1
            continue
        parsed, index = _parse_yaml_block(lines, index + 1, lines[index + 1][0])
        value[key] = parsed
    return value, index


def _strip_yaml_comment(raw: str) -> str:
    quote: str | None = None
    escaped = False
    for index, character in enumerate(raw):
        if escaped:
            escaped = False
            continue
        if quote == '"' and character == "\\":
            escaped = True
            continue
        if character in {"'", '"'}:
            if quote is None:
                quote = character
                continue
            if quote == character:
                quote = None
                continue
        if character == "#" and quote is None and (index == 0 or raw[index - 1].isspace()):
            return raw[:index]
    return raw


def _yaml_scalar(value: str) -> Any:
    if value.startswith("[") and value.endswith("]"):
        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:
            raise ContractValidationError("YAML_INLINE_JSON_REQUIRED") from exc
    if value in {"true", "false"}:
        return value == "true"
    if value in {"null", "~"}:
        return None
    if re.fullmatch(r"-?[0-9]+", value):
        return int(value)
    if re.fullmatch(r"-?(?:[0-9]+\.[0-9]*|[0-9]*\.[0-9]+)", value):
        return float(value)
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    return value


def validate_document(schema_name: str, document: Mapping[str, Any], *, root: Path | None = None) -> None:
    schema_path = _root(root) / "configs" / "schemas" / f"{schema_name}.schema.json"
    if not schema_path.is_file():
        raise ContractValidationError(f"SCHEMA_NOT_FOUND:{schema_name}")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    _validate(schema, document, path="$")


def validate_instance(schema: Mapping[str, Any], document: Any) -> None:
    """Validate an in-memory JSON Schema, failing closed on every error.

    The checked-in dependency is authoritative when present.  The restricted
    stdlib validator is retained only so M0 source/freeze checks remain
    inspectable before a disconnected host can install dependencies.
    """

    if Draft202012Validator is None:
        _validate(schema, document, path="$")
        return
    try:
        validator = Draft202012Validator(schema)
    except SchemaError as exc:
        raise ContractValidationError(f"SCHEMA_INVALID:{exc.message}") from exc
    errors = sorted(validator.iter_errors(document), key=lambda error: list(error.absolute_path))
    if errors:
        error = errors[0]
        location = "$" + "".join(
            f"[{part}]" if isinstance(part, int) else f".{part}" for part in error.absolute_path
        )
        raise ContractValidationError(f"{location}: {error.message}")


def _validate(schema: Mapping[str, Any], value: Any, *, path: str) -> None:
    allowed = {
        "$schema",
        "type",
        "additionalProperties",
        "required",
        "properties",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "minItems",
        "items",
        "enum",
        "pattern",
        "minLength",
    }
    unknown = set(schema) - allowed
    if unknown:
        raise ContractValidationError(f"SCHEMA_KEYWORD_UNSUPPORTED:{','.join(sorted(unknown))}")
    expected = schema.get("type")
    if expected is not None and not _matches_type(value, expected):
        raise ContractValidationError(f"{path}: expected {_describe_type(expected)}")
    if "enum" in schema and value not in schema["enum"]:
        raise ContractValidationError(f"{path}: value is not an allowed enum")
    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            raise ContractValidationError(f"{path}: string shorter than minLength")
        if "pattern" in schema and re.search(str(schema["pattern"]), value) is None:
            raise ContractValidationError(f"{path}: string does not match pattern")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            raise ContractValidationError(f"{path}: value below minimum")
        if "maximum" in schema and value > schema["maximum"]:
            raise ContractValidationError(f"{path}: value above maximum")
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            raise ContractValidationError(f"{path}: value must exceed exclusiveMinimum")
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            raise ContractValidationError(f"{path}: fewer than minItems")
        if "items" in schema:
            for index, child in enumerate(value):
                _validate(schema["items"], child, path=f"{path}[{index}]")
    if isinstance(value, Mapping):
        properties = schema.get("properties", {})
        missing = [name for name in schema.get("required", []) if name not in value]
        if missing:
            raise ContractValidationError(f"{path}: missing required {','.join(missing)}")
        if schema.get("additionalProperties", True) is False:
            extras = sorted(set(value) - set(properties))
            if extras:
                raise ContractValidationError(f"{path}: unexpected properties {','.join(extras)}")
        for name, child_schema in properties.items():
            if name in value:
                _validate(child_schema, value[name], path=f"{path}.{name}")


def _matches_type(value: Any, expected: str | Sequence[str]) -> bool:
    names = (expected,) if isinstance(expected, str) else tuple(expected)
    checks = {
        "object": lambda: isinstance(value, Mapping),
        "array": lambda: isinstance(value, list),
        "string": lambda: isinstance(value, str),
        "integer": lambda: isinstance(value, int) and not isinstance(value, bool),
        "number": lambda: isinstance(value, (int, float)) and not isinstance(value, bool),
        "null": lambda: value is None,
        "boolean": lambda: isinstance(value, bool),
    }
    return any(name in checks and checks[name]() for name in names)


def _describe_type(expected: str | Sequence[str]) -> str:
    return expected if isinstance(expected, str) else " or ".join(expected)
