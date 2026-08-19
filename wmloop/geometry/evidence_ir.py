"""Path-independent evidence representation derived from settled effects."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import re

from wmloop.contracts import ContractValidationError, validate_document
from wmloop.geometry.memory import EffectRecord
from wmloop.geometry.types import GeometryValidationError


_OUTCOME_LABELS = {
    "confirmed": "positive",
    "null": "null",
    "rejected": "harmful",
    "interaction": "interaction",
}
_TRANSFER_STATES = {"local_only", "licensed_prior", "abstained"}
_FORBIDDEN_KEY_TOKENS = {
    "checkpoint",
    "cwd",
    "file",
    "filename",
    "manifest",
    "path",
    "repo",
    "repository",
    "workdir",
}
_ABSOLUTE_PATH = re.compile(r"^(?:/|[A-Za-z]:[\\/]|~[/\\]|file://)")
_FILE_NAME = re.compile(
    r"(?:^|[/\\])[^/\\]+\."
    r"(?:arrow|bin|ckpt|csv|jsonl?|log|md|onnx|parquet|pt|pth|"
    r"safetensors|toml|txt|ya?ml)$",
    re.IGNORECASE,
)
_URI = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")


def build_evidence_ir(
    effect: EffectRecord,
    *,
    transfer_state: str,
    applicability: Mapping[str, object],
    semantic_intent: str | None = None,
    target_hooks: Sequence[str] = (),
    goal_binding: str | None = None,
    evaluator_binding: str | None = None,
    root: Path | None = None,
) -> dict[str, object]:
    """Build one authority-aware Evidence IR without local runtime bindings."""

    if not isinstance(effect, EffectRecord):
        raise GeometryValidationError("EVIDENCE_IR_EFFECT_INVALID")
    if transfer_state not in _TRANSFER_STATES:
        raise GeometryValidationError("EVIDENCE_IR_TRANSFER_STATE_INVALID")
    if not isinstance(applicability, Mapping) or not applicability:
        raise GeometryValidationError("EVIDENCE_IR_VALIDITY_REGION_INVALID")
    if any(not isinstance(value, str) or not value.strip() for value in target_hooks):
        raise GeometryValidationError("EVIDENCE_IR_TARGET_HOOK_INVALID")
    if goal_binding is not None and not is_content_addressed(goal_binding):
        raise GeometryValidationError("EVIDENCE_IR_GOAL_BINDING_INVALID")
    if evaluator_binding is not None and not is_content_addressed(evaluator_binding):
        raise GeometryValidationError("EVIDENCE_IR_EVALUATOR_BINDING_INVALID")

    authority_bound = goal_binding is not None and evaluator_binding is not None
    if transfer_state == "licensed_prior" and effect.status == "confirmed" and authority_bound:
        state = "transfer_licensed"
        claim_scope = "transfer_prior"
        reason = "licensed transfer certificate and frozen authority bindings"
    elif authority_bound:
        state = "target_confirmed"
        claim_scope = "target_local"
        reason = "settled target effect with frozen goal and evaluator bindings"
    else:
        state = "schema_valid"
        claim_scope = "ranking_only"
        reason = "authority bindings are incomplete; retain as non-promotable evidence"

    context = effect.context
    body: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "verdiwm-evidence-ir",
        "context": {
            "model_family": context.backbone_family,
            "capability_class": context.capability_class,
            "goal_protocol": context.goal_schema,
            "outcome_protocol": context.outcome_schema,
            "data_regime": context.data_regime,
            "horizons": list(context.horizons),
        },
        "intervention": {
            "primitive_id": effect.primitive,
            "semantic_intent": semantic_intent or f"bounded intervention: {effect.primitive}",
            "target_hooks": sorted(set(target_hooks)),
        },
        "outcome": {
            "label": _OUTCOME_LABELS[effect.status],
            "mean": effect.mean_effect,
            "standard_error": effect.standard_error,
            "lower_bound": effect.lower_bound,
            "threshold": effect.goal_threshold,
            "replication_count": effect.replication_count,
        },
        "validity_region": dict(applicability),
        "evidence_refs": list(effect.evidence_refs),
        "status": {"state": state, "reason": reason},
        "authority": {
            "goal_binding": goal_binding,
            "evaluator_binding": evaluator_binding,
            "promotion_policy": "verifier_only",
            "claim_scope": claim_scope,
        },
    }
    body["evidence_id"] = "evidence-" + _digest(body)[:24]
    validate_evidence_ir(body, root=root)
    return body


def validate_evidence_ir(
    document: Mapping[str, object], *, root: Path | None = None
) -> None:
    validity = document.get("validity_region")
    if not isinstance(validity, Mapping) or not validity:
        raise GeometryValidationError("EVIDENCE_IR_VALIDITY_REGION_INVALID")
    reject_runtime_bindings(document)
    try:
        validate_document("evidence_ir", document, root=root)
    except ContractValidationError as exc:
        raise GeometryValidationError(f"EVIDENCE_IR_SCHEMA_INVALID:{exc}") from exc
    refs = document.get("evidence_refs")
    if not isinstance(refs, list) or any(
        not isinstance(value, str) or not is_content_addressed(value)
        for value in refs
    ):
        raise GeometryValidationError("EVIDENCE_IR_EVIDENCE_REF_INVALID")
    authority = document.get("authority")
    status = document.get("status")
    if not isinstance(authority, Mapping) or not isinstance(status, Mapping):
        raise GeometryValidationError("EVIDENCE_IR_AUTHORITY_INVALID")
    goal = authority.get("goal_binding")
    evaluator = authority.get("evaluator_binding")
    for binding in (goal, evaluator):
        if isinstance(binding, str) and not is_content_addressed(binding):
            raise GeometryValidationError("EVIDENCE_IR_AUTHORITY_BINDING_INVALID")
    authority_bound = isinstance(goal, str) and isinstance(evaluator, str)
    if authority_bound:
        if status.get("state") == "schema_valid" or authority.get("claim_scope") == "ranking_only":
            raise GeometryValidationError("EVIDENCE_IR_AUTHORITY_STATE_INVALID")
    elif (
        status.get("state") not in {"schema_valid", "shadow", "revoked", "deprecated"}
        or authority.get("claim_scope") != "ranking_only"
    ):
        raise GeometryValidationError("EVIDENCE_IR_AUTHORITY_STATE_INVALID")
    body = dict(document)
    body.pop("evidence_id", None)
    if document.get("evidence_id") != "evidence-" + _digest(body)[:24]:
        raise GeometryValidationError("EVIDENCE_IR_ID_MISMATCH")


def is_content_addressed(value: str) -> bool:
    """Return whether a reference is semantic or content-addressed, not local."""

    if re.fullmatch(r"sha256:[0-9a-f]{64}", value):
        return True
    if value.startswith("cas://"):
        return len(value) > len("cas://") and not any(
            character.isspace() for character in value
        )
    if value.startswith("urn:"):
        return len(value) > len("urn:") and not any(
            character.isspace() for character in value
        )
    return False


def reject_runtime_bindings(value: object) -> None:
    """Reject local filesystem and location bindings in reusable evidence."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            if _has_runtime_binding_key(str(key)):
                raise GeometryValidationError(
                    "PORTABLE_EXPERIENCE_RUNTIME_BINDING_FORBIDDEN:" + str(key)
                )
            reject_runtime_bindings(child)
    elif isinstance(value, list):
        for child in value:
            reject_runtime_bindings(child)
    elif isinstance(value, str) and _looks_like_runtime_path(value):
        raise GeometryValidationError("PORTABLE_EXPERIENCE_RUNTIME_PATH_FORBIDDEN")


def _looks_like_runtime_path(value: str) -> bool:
    if is_content_addressed(value):
        return False
    return bool(
        _ABSOLUTE_PATH.match(value)
        or _URI.match(value)
        or "/" in value
        or "\\" in value
        or _FILE_NAME.search(value)
    )


def _has_runtime_binding_key(key: str) -> bool:
    snake_case = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", key).lower()
    tokens = set(filter(None, re.split(r"[^a-z0-9]+", snake_case)))
    return bool(tokens & _FORBIDDEN_KEY_TOKENS)


def _digest(value: object) -> str:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise GeometryValidationError("EVIDENCE_IR_PAYLOAD_INVALID") from exc
    return hashlib.sha256(payload).hexdigest()
