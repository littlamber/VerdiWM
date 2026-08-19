"""Universal process contract for arbitrary open-method candidates."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath

from wmloop.contracts import ContractValidationError, validate_document
from wmloop.control.open_method_ir import validate_method_ir


class CandidateExecutionContractError(ValueError):
    """A candidate process contract was unsafe or inconsistent."""


_PLACEHOLDERS = {
    "{candidate_root}",
    "{input_manifest}",
    "{output_root}",
    "{world_size}",
    "{rank}",
}


def build_candidate_execution_contract(
    *,
    method_ir: Mapping[str, object],
    entrypoints: Mapping[str, object],
    inputs: Sequence[Mapping[str, object]],
    outputs: Sequence[Mapping[str, object]],
    root: Path | None = None,
) -> dict[str, object]:
    validate_method_ir(method_ir, root=root)
    body: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "verdiwm-candidate-execution-contract",
        "execution_id": "",
        "method_id": method_ir["method_id"],
        "protocol_version": 1,
        "entrypoints": {
            "calibrate": _command(entrypoints.get("calibrate"), "CALIBRATE"),
            "train": (
                _command(entrypoints.get("train"), "TRAIN")
                if entrypoints.get("train") is not None
                else None
            ),
            "infer": _command(entrypoints.get("infer"), "INFER"),
        },
        "inputs": [dict(row) for row in inputs],
        "outputs": [dict(row) for row in outputs],
        "isolation": {
            "network_access": False,
            "source_write_access": False,
            "evaluator_access": False,
            "credential_access": False,
            "gpu_lease_required": True,
        },
        "claim_boundary": (
            "This process protocol executes only inside a candidate sandbox and grants no "
            "evaluator, resource-allocation, verdict, or promotion authority."
        ),
    }
    if method_ir["training"]["mode"] == "training" and body["entrypoints"]["train"] is None:
        raise CandidateExecutionContractError("CANDIDATE_EXECUTION_TRAIN_REQUIRED")
    body["execution_id"] = "candidate-execution-" + candidate_execution_digest(body)[:24]
    validate_candidate_execution_contract(body, root=root)
    return body


def validate_candidate_execution_contract(
    document: Mapping[str, object], *, root: Path | None = None
) -> None:
    try:
        validate_document("candidate_execution_contract", document, root=root)
    except ContractValidationError as exc:
        raise CandidateExecutionContractError(f"CANDIDATE_EXECUTION_SCHEMA_INVALID:{exc}") from exc
    if document.get("execution_id") != "candidate-execution-" + candidate_execution_digest(document)[:24]:
        raise CandidateExecutionContractError("CANDIDATE_EXECUTION_DIGEST_MISMATCH")
    entrypoints = document["entrypoints"]
    for name in ("calibrate", "train", "infer"):
        command = entrypoints[name]
        if command is not None:
            _command(command, name.upper())
    for field in ("inputs", "outputs"):
        rows = document[field]
        names = [str(row["name"]) for row in rows]
        if len(names) != len(set(names)):
            raise CandidateExecutionContractError("CANDIDATE_EXECUTION_PORT_DUPLICATE")


def candidate_execution_digest(document: Mapping[str, object]) -> str:
    payload = {key: value for key, value in document.items() if key != "execution_id"}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode()
    ).hexdigest()


def _command(value: object, name: str) -> list[str]:
    if not isinstance(value, list) or not value or any(not isinstance(token, str) or not token for token in value):
        raise CandidateExecutionContractError(f"CANDIDATE_EXECUTION_{name}_INVALID")
    command = [str(token) for token in value]
    for token in command:
        if "\x00" in token or re.search(r"[;&|`\n\r]", token):
            raise CandidateExecutionContractError(f"CANDIDATE_EXECUTION_{name}_INVALID")
        stripped = token
        for placeholder in _PLACEHOLDERS:
            stripped = stripped.replace(placeholder, "placeholder")
        if "{" in stripped or "}" in stripped:
            raise CandidateExecutionContractError("CANDIDATE_EXECUTION_PLACEHOLDER_INVALID")
        pure = PurePosixPath(stripped)
        if pure.is_absolute() or ".." in pure.parts:
            raise CandidateExecutionContractError("CANDIDATE_EXECUTION_PATH_INVALID")
    return command
