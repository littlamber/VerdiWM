"""Decision state shared by research entry points, with immutable goal binding.

Execution receipts and the budget ledger remain authoritative. This journal
records which research operation is running and the evidence available to the
next decision; it never turns execution completion into a positive effect.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

from wmloop.contracts import ContractValidationError, validate_document
from wmloop.storage import atomic_write, canonical_bytes, checked_path, exclusive_file_lock

_TRANSITIONS = {
    "observe": {"hypothesize", "blocked"},
    "hypothesize": {"select", "observe", "blocked"},
    "select": {"implement", "observe", "blocked", "settled"},
    "implement": {"verify", "blocked"},
    "verify": {"remember", "blocked"},
    "remember": {"refine", "settled", "blocked"},
    "refine": {"observe", "settled", "blocked"},
    "blocked": {"observe", "settled"},
    "settled": set(),
}
_MUTABLE = {"model_portrait_ref", "irg_ref", "hypotheses", "candidate_actions", "evidence_refs", "budget", "stop_conditions"}


class ResearchStateError(ValueError):
    """A research decision would discard its goal, provenance, or evidence."""


def state_digest(document: Mapping[str, object]) -> str:
    return hashlib.sha256(canonical_bytes({k: v for k, v in document.items() if k != "state_id"})).hexdigest()


def _seal(body: dict) -> dict:
    body["state_id"] = "research-state-" + state_digest(body)[:24]
    validate_research_state(body)
    return body


def build_research_state(
    *, goal: str, metrics: Sequence[str], budget_gpu_hours: float,
    trial_count: int | None = None, model_portrait_ref: str | None = None,
    irg_ref: str | None = None, representation_version: str = "response_euclidean_v2",
    hypotheses: Sequence[Mapping[str, object]] = (),
    candidate_actions: Sequence[Mapping[str, object]] = (),
    evidence_refs: Sequence[str] = (), stop_conditions: Sequence[str] = (),
) -> dict:
    if not isinstance(goal, str) or not goal.strip() or type(budget_gpu_hours) not in (int, float) or not math.isfinite(budget_gpu_hours) or budget_gpu_hours < 0:
        raise ResearchStateError("RESEARCH_STATE_INPUT_INVALID")
    if trial_count is not None and (type(trial_count) is not int or trial_count < 0):
        raise ResearchStateError("RESEARCH_STATE_TRIAL_COUNT_INVALID")
    return _seal({
        "schema_version": 1, "artifact_type": "verdiwm-research-state", "state_id": "",
        "phase": "observe", "goal": {"description": goal.strip(), "metrics": list(dict.fromkeys(metrics))},
        "model_portrait_ref": model_portrait_ref, "irg_ref": irg_ref,
        "hypotheses": [dict(x) for x in hypotheses], "candidate_actions": [dict(x) for x in candidate_actions],
        "evidence_refs": list(dict.fromkeys(evidence_refs)),
        "budget": {"gpu_hours_limit": float(budget_gpu_hours), "gpu_hours_remaining": None,
                   "trial_count_remaining": trial_count, "authority": "external_budget_ledger"},
        "representation_version": representation_version, "stop_conditions": list(stop_conditions),
        "last_transition": None, "revision": 0,
        "claim_boundary": "Decision state only. Budget and frozen experiment receipts remain authoritative; settled does not imply improvement.",
    })


def validate_research_state(document: Mapping[str, object], *, root: Path | None = None) -> None:
    try:
        validate_document("research_state", document, root=root)
        expected = "research-state-" + state_digest(document)[:24]
    except (ContractValidationError, ValueError, TypeError) as exc:
        raise ResearchStateError(f"RESEARCH_STATE_INVALID:{exc}") from exc
    if document.get("state_id") != expected:
        raise ResearchStateError("RESEARCH_STATE_DIGEST_MISMATCH")
    for hypothesis in document["hypotheses"]:
        from wmloop.control.mechanism_hypothesis import validate_mechanism_hypothesis
        validate_mechanism_hypothesis(hypothesis, root=root)


def transition_research_state(
    document: Mapping[str, object], *, phase: str, reason: str,
    updates: Mapping[str, object] | None = None, root: Path | None = None,
) -> dict:
    validate_research_state(document, root=root)
    previous = str(document["phase"])
    if (phase != previous and phase not in _TRANSITIONS[previous]) or not reason.strip() or previous == "settled":
        raise ResearchStateError(f"RESEARCH_STATE_TRANSITION_INVALID:{previous}:{phase}")
    changes = dict(updates or {})
    if set(changes) - _MUTABLE:
        raise ResearchStateError("RESEARCH_STATE_IMMUTABLE_BINDING")
    if "budget" in changes:
        raise ResearchStateError("RESEARCH_STATE_BUDGET_REQUIRES_LEDGER_PROJECTION")
    body = json.loads(canonical_bytes(document))
    body.update(changes)
    body["phase"] = phase
    body["revision"] = int(document["revision"]) + 1
    body["last_transition"] = {"from": previous, "to": phase, "reason": reason.strip(), "parent_state_id": document["state_id"]}
    return _seal(body)


def write_research_state(path: Path, document: Mapping[str, object]) -> Path:
    validate_research_state(document)
    destination = checked_path(path, code="RESEARCH_STATE_PATH_INVALID", error=ResearchStateError)
    atomic_write(destination, canonical_bytes(document) + b"\n")
    return destination


class ResearchJournal:
    """Atomic snapshots with a single immutable campaign input binding.

    Reopening preserves the current snapshot. Pipeline retries explicitly
    record a blocked-to-observe transition and resume existing trial receipts.
    """

    def __init__(self, root: Path, *, input_hash: str, initial: Mapping[str, object]):
        self.root = checked_path(root, code="RESEARCH_STATE_PATH_INVALID", error=ResearchStateError)
        self.root.mkdir(parents=True, exist_ok=True)
        self.input_hash = input_hash
        self.path = self.root / "research-state.json"
        self.lock = self.root / ".research-state.lock"
        validate_research_state(initial)
        with exclusive_file_lock(self.lock):
            binding = checked_path(self.root / "research-binding.json", code="RESEARCH_STATE_PATH_INVALID", error=ResearchStateError)
            identity = {"input_hash": input_hash, "initial_state_id": initial["state_id"]}
            if binding.exists():
                if json.loads(binding.read_bytes()) != identity:
                    raise ResearchStateError("RESEARCH_STATE_INPUT_MISMATCH")
            else:
                if self.path.exists() or self.path.is_symlink():
                    raise ResearchStateError("RESEARCH_STATE_BINDING_MISSING")
                atomic_write(binding, canonical_bytes(identity))
            if self.path.exists():
                # Resuming the same bound campaign must preserve its decision
                # state.  A new input hash is rejected above; therefore an
                # existing snapshot is authoritative for this journal.
                self.read()
            else:
                write_research_state(self.path, initial)

    @classmethod
    def open_or_create(cls, root: Path, *, input_hash: str, initial: Mapping[str, object]) -> "ResearchJournal":
        """Open a resumable journal without resetting an existing snapshot."""
        return cls(root, input_hash=input_hash, initial=initial)

    def read(self) -> dict:
        path = checked_path(self.path, code="RESEARCH_STATE_PATH_INVALID", error=ResearchStateError)
        document = json.loads(path.read_bytes())
        validate_research_state(document)
        return document

    def advance(self, phase: str, reason: str, **updates: object) -> dict:
        with exclusive_file_lock(self.lock):
            previous = self.read()
            updated = transition_research_state(previous, phase=phase, reason=reason, updates=updates)
            write_research_state(self.root / "research-history" / (str(previous["state_id"]) + ".json"), previous)
            write_research_state(self.path, updated)
            return updated
