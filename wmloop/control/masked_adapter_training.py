"""Receipt and state bindings for frozen-backbone masked adapter training.

The training job is deliberately separate from candidate materialization.  This
module only defines the small, content-addressed contract shared by the trainer,
candidate compiler, and evaluator; it never writes into a source repository.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


class MaskedAdapterTrainingError(ValueError):
    """A trained adapter artifact crossed an evidence boundary."""


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":")) + "\n").encode("utf-8")


def sha256_file(path: Path) -> str:
    source = _require_file(path, "MASKED_ADAPTER_ARTIFACT_MISSING")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def proof_digest(proof: Mapping[str, object]) -> str:
    payload = {key: value for key, value in proof.items() if key != "proof_sha256"}
    return hashlib.sha256(canonical_json_bytes(payload).rstrip(b"\n")).hexdigest()


def training_binding_from_receipt(
    receipt_path: Path,
    *,
    expected_candidate_id: str,
    expected_parameters: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Load a trainer receipt and return the candidate provenance binding."""

    receipt = _load_json(receipt_path, "MASKED_ADAPTER_TRAINING_RECEIPT_INVALID")
    if receipt.get("artifact_type") != "verdiwm-masked-adapter-training-receipt":
        raise MaskedAdapterTrainingError("MASKED_ADAPTER_TRAINING_RECEIPT_INVALID")
    if receipt.get("state") != "ready_for_evaluation":
        raise MaskedAdapterTrainingError("MASKED_ADAPTER_TRAINING_NOT_READY")
    if receipt.get("candidate_id") != expected_candidate_id:
        raise MaskedAdapterTrainingError("MASKED_ADAPTER_TRAINING_CANDIDATE_MISMATCH")
    if expected_parameters is not None and receipt.get("parameters") != dict(expected_parameters):
        raise MaskedAdapterTrainingError("MASKED_ADAPTER_TRAINING_PARAMETERS_MISMATCH")

    state_path = _require_file(
        Path(str(receipt.get("adapter_state_path", ""))),
        "MASKED_ADAPTER_STATE_MISSING",
    )
    state_sha256 = receipt.get("adapter_state_sha256")
    if not isinstance(state_sha256, str) or sha256_file(state_path) != state_sha256:
        raise MaskedAdapterTrainingError("MASKED_ADAPTER_STATE_HASH_MISMATCH")
    split = receipt.get("training_split_fingerprint")
    backbone = receipt.get("backbone_checkpoint_sha256")
    if (
        not isinstance(split, str)
        or not _SHA256_RE.fullmatch(split)
        or not isinstance(backbone, str)
        or not _SHA256_RE.fullmatch(backbone)
    ):
        raise MaskedAdapterTrainingError("MASKED_ADAPTER_TRAINING_DATA_BINDING_INVALID")

    proof = receipt.get("backbone_freeze_proof")
    if not isinstance(proof, Mapping) or proof.get("state") != "frozen":
        raise MaskedAdapterTrainingError("MASKED_ADAPTER_BACKBONE_NOT_FROZEN")
    names = proof.get("trainable_parameter_names")
    if (
        not isinstance(names, list)
        or not names
        or any(
            not isinstance(name, str)
            or not name.startswith("history_corrector.multiscale_side_adapter.")
            for name in names
        )
        or len(names) != len(set(names))
    ):
        raise MaskedAdapterTrainingError("MASKED_ADAPTER_TRAINABLE_SCOPE_INVALID")
    if proof.get("backbone_trainable_parameter_count") != 0:
        raise MaskedAdapterTrainingError("MASKED_ADAPTER_BACKBONE_NOT_FROZEN")
    if proof.get("proof_sha256") != proof_digest(proof):
        raise MaskedAdapterTrainingError("MASKED_ADAPTER_FREEZE_PROOF_HASH_MISMATCH")

    optimizer = receipt.get("optimizer_receipt")
    if not isinstance(optimizer, Mapping):
        raise MaskedAdapterTrainingError("MASKED_ADAPTER_OPTIMIZER_RECEIPT_INVALID")
    steps = optimizer.get("steps")
    learning_rate = optimizer.get("learning_rate")
    if (
        not isinstance(optimizer.get("name"), str)
        or not isinstance(steps, int)
        or isinstance(steps, bool)
        or steps < 1
        or isinstance(learning_rate, bool)
        or not isinstance(learning_rate, (int, float))
        or not math.isfinite(float(learning_rate))
        or float(learning_rate) <= 0.0
    ):
        raise MaskedAdapterTrainingError("MASKED_ADAPTER_OPTIMIZER_RECEIPT_INVALID")

    return {
        "adapter_state_path": str(state_path),
        "adapter_state_sha256": state_sha256,
        "training_receipt_path": str(Path(receipt_path).resolve()),
        "training_receipt_sha256": sha256_file(receipt_path),
        "training_split_fingerprint": split,
        "backbone_checkpoint_sha256": backbone,
        "backbone_freeze_proof": dict(proof),
        "optimizer_receipt": dict(optimizer),
        "adapter_state_format": receipt.get("adapter_state_format", "torch_state_dict"),
    }


def write_training_receipt(path: Path, receipt: Mapping[str, object]) -> Path:
    """Write one immutable receipt atomically and return its resolved path."""

    if receipt.get("artifact_type") != "verdiwm-masked-adapter-training-receipt":
        raise MaskedAdapterTrainingError("MASKED_ADAPTER_TRAINING_RECEIPT_INVALID")
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_bytes(canonical_json_bytes(dict(receipt)))
    temporary.replace(target)
    return target


def bind_training_artifacts(
    candidate: Mapping[str, object],
    receipt_path: Path,
) -> dict[str, object]:
    """Return a candidate with a verified training binding attached."""

    kind = candidate.get("candidate_kind")
    if kind != "materialized_masked_intermediate_action_adapter":
        raise MaskedAdapterTrainingError("MASKED_ADAPTER_CANDIDATE_KIND_INVALID")
    parameters = candidate.get("parameters")
    if not isinstance(parameters, Mapping):
        raise MaskedAdapterTrainingError("MASKED_ADAPTER_PARAMETERS_INVALID")
    binding = training_binding_from_receipt(
        receipt_path,
        expected_candidate_id=str(candidate.get("candidate_id")),
        expected_parameters=parameters,
    )
    provenance = candidate.get("provenance")
    if not isinstance(provenance, Mapping):
        raise MaskedAdapterTrainingError("MASKED_ADAPTER_PROVENANCE_INVALID")
    updated = dict(candidate)
    updated["provenance"] = {**dict(provenance), "training_binding": binding}
    return updated


def _load_json(path: Path, code: str) -> dict[str, object]:
    source = _require_file(path, code)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MaskedAdapterTrainingError(code) from exc
    if not isinstance(payload, dict):
        raise MaskedAdapterTrainingError(code)
    return payload


def _require_file(path: Path, code: str) -> Path:
    raw = Path(path).expanduser()
    if raw.is_symlink():
        raise MaskedAdapterTrainingError(code)
    resolved = raw.resolve()
    if not resolved.is_file():
        raise MaskedAdapterTrainingError(code)
    return resolved


__all__: Sequence[str] = (
    "MaskedAdapterTrainingError",
    "bind_training_artifacts",
    "canonical_json_bytes",
    "proof_digest",
    "sha256_file",
    "training_binding_from_receipt",
    "write_training_receipt",
)
