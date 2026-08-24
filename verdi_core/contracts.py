"""Canonical semantic contracts used by the clean control plane."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def canonical_digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode()).hexdigest()


@dataclass(frozen=True)
class Goal:
    goal_id: str
    objective: str
    primary_metric: str
    protected_metrics: tuple[str, ...]
    heldout_split: str
    budget_gpu_hours: float


@dataclass(frozen=True)
class CapabilityIR:
    model_id: str
    revision: str
    capabilities: tuple[str, ...]
    hooks: tuple[str, ...]
    evaluator_id: str


@dataclass(frozen=True)
class ProbeFingerprint:
    fingerprint_id: str
    model_id: str
    probe_id: str
    dimensions: tuple[str, ...]
    response_digest: str
    uncertainty: str


@dataclass(frozen=True)
class Portrait:
    portrait_id: str
    model_id: str
    capability_digest: str
    fingerprint_ids: tuple[str, ...]
    readiness: str


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    experiment_id: str
    model_id: str
    hypothesis_id: str
    outcome: str
    delta: float
    protected_ok: bool
    verifier_digest: str
    claim_boundary: str


def to_document(value: Any) -> dict[str, Any]:
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"unsupported contract: {type(value)!r}")

