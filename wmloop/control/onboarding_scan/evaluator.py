"""Read-only onboarding scan components: evaluator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from wmloop.control.onboarding_scan.io import _bounded, _sha256_file


def _load_evaluator_contract(path: Path | None) -> dict[str, object]:
    required = [
        "evaluator_id",
        "command",
        "input_artifacts",
        "output_artifacts",
        "metrics",
        "verifier",
    ]
    if path is None:
        return {
            "state": "binding_required",
            "source_path": None,
            "required_fields": required,
            "missing_fields": required,
        }
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        return {
            "state": "blocked",
            "source_path": str(resolved),
            "required_fields": required,
            "missing_fields": required,
            "error": "EVALUATOR_CONTRACT_NOT_FOUND",
        }
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "state": "blocked",
            "source_path": str(resolved),
            "required_fields": required,
            "missing_fields": required,
            "error": _bounded(str(exc)),
        }
    if not isinstance(payload, dict):
        return {
            "state": "blocked",
            "source_path": str(resolved),
            "required_fields": required,
            "missing_fields": required,
            "error": "EVALUATOR_CONTRACT_OBJECT_REQUIRED",
        }
    missing = [field for field in required if field not in payload]
    if missing:
        return {
            "state": "binding_required",
            "source_path": str(resolved),
            "required_fields": required,
            "missing_fields": missing,
            "error": "EVALUATOR_CONTRACT_FIELDS_MISSING",
        }
    invalid = _invalid_evaluator_fields(payload)
    if invalid:
        return {
            "state": "blocked",
            "source_path": str(resolved),
            "required_fields": required,
            "missing_fields": [],
            "error": "EVALUATOR_CONTRACT_FIELDS_INVALID:" + ",".join(invalid),
        }
    contract = {
        "state": "ready",
        "source_path": str(resolved),
        "required_fields": required,
        "missing_fields": [],
        "contract_sha256": _sha256_file(resolved),
        "evaluator_id": str(payload["evaluator_id"]),
        "command": payload["command"],
        "input_artifacts": payload["input_artifacts"],
        "output_artifacts": payload["output_artifacts"],
        "metrics": payload["metrics"],
        "verifier": payload["verifier"],
    }
    working_directory = payload.get("working_directory", ".")
    contract["working_directory"] = working_directory
    contract["conformance_imports"] = list(payload.get("conformance_imports", []))
    entrypoint_probe = payload.get("entrypoint_probe", "help")
    if entrypoint_probe != "help" and entrypoint_probe != "skip":
        return {
            "state": "blocked",
            "source_path": str(resolved),
            "required_fields": required,
            "missing_fields": [],
            "error": "EVALUATOR_ENTRYPOINT_PROBE_INVALID",
        }
    contract["entrypoint_probe"] = entrypoint_probe
    scheduler_template = payload.get("scheduler_template")
    if scheduler_template is not None:
        template_path = Path(str(scheduler_template))
        template_path = (
            template_path.resolve()
            if template_path.is_absolute()
            else (resolved.parent / template_path).resolve()
        )
        if not template_path.is_file() or template_path.is_symlink():
            return {
                "state": "blocked",
                "source_path": str(resolved),
                "required_fields": required,
                "missing_fields": [],
                "error": "EVALUATOR_SCHEDULER_TEMPLATE_INVALID",
            }
        contract["scheduler_template_path"] = str(template_path)
        contract["scheduler_template_sha256"] = _sha256_file(template_path)
    return contract


def _invalid_evaluator_fields(payload: Mapping[str, Any]) -> list[str]:
    invalid: list[str] = []
    if (
        not isinstance(payload.get("evaluator_id"), str)
        or not payload["evaluator_id"].strip()
    ):
        invalid.append("evaluator_id")
    if not _nonempty_string_list(payload.get("command")):
        invalid.append("command")
    for field in ("input_artifacts", "output_artifacts", "metrics"):
        if not _nonempty_string_list(payload.get(field)):
            invalid.append(field)
    if not isinstance(payload.get("verifier"), str) or not payload["verifier"].strip():
        invalid.append("verifier")
    if (
        not isinstance(payload.get("working_directory", "."), str)
        or not str(payload.get("working_directory", ".")).strip()
    ):
        invalid.append("working_directory")
    conformance_imports = payload.get("conformance_imports", [])
    if not isinstance(conformance_imports, list) or not all(
        isinstance(item, str) and item.strip() for item in conformance_imports
    ):
        invalid.append("conformance_imports")
    entrypoint_probe = payload.get("entrypoint_probe", "help")
    if entrypoint_probe not in {"help", "skip"}:
        invalid.append("entrypoint_probe")
    scheduler_template = payload.get("scheduler_template")
    if scheduler_template is not None and (
        not isinstance(scheduler_template, str) or not scheduler_template.strip()
    ):
        invalid.append("scheduler_template")
    return invalid


def _nonempty_string_list(value: object) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, str) and item.strip() for item in value)
    )
