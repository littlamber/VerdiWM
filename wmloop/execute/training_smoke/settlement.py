"""Training and frozen evaluation smoke components: settlement."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from wmloop.archive.store import ArchiveStore, SettledTrialRecord
from wmloop.execute.training_smoke.errors import OrchestratorTrainingEvalSmokeError
from wmloop.execute.training_smoke.io import _canonical_json_bytes
from wmloop.propose.scheduler import InterventionCell


def _record_formal_settled_trial(
    *,
    archive: ArchiveStore,
    proposal: Mapping[str, Any],
    verdict: Mapping[str, Any],
    receipt: Mapping[str, Any],
    receipt_ref: str,
    failure_context_ref: str,
    verdict_ref: str,
    gpu_hours: float,
    settlement_state: str,
    round_start_verification: object,
) -> bool:
    interventions = proposal.get("interventions")
    if (
        not isinstance(interventions, list)
        or not interventions
        or not all(isinstance(item, Mapping) for item in interventions)
    ):
        raise OrchestratorTrainingEvalSmokeError(
            "M3_TRAINING_EVAL_SMOKE_FORMAL_INTERVENTIONS_INVALID"
        )
    single = len(interventions) == 1
    cell = _formal_trial_cell(proposal=proposal, interventions=interventions)
    verdict_label = verdict.get("verdict")
    exploratory = not single or verdict_label == "VOID"
    verified_gain = None if exploratory else _primary_delta(verdict)
    archive.record_settled_trial(
        SettledTrialRecord(
            trial_id=str(proposal["proposal_id"]),
            proposal_id=str(proposal["proposal_id"]),
            goal_id=str(proposal["goal_id"]),
            library_version=str(proposal.get("library_version", "v1.0")),
            failure_context_ref=failure_context_ref,
            verdict_ref=verdict_ref,
            receipt_ref=receipt_ref,
            gpu_hours=float(gpu_hours),
            hypothesis_hash=hashlib.sha256(_canonical_json_bytes(proposal)).hexdigest(),
            impl_diff_hash=_impl_diff_hash(receipt),
            evaluator_hash=_evaluator_hash(round_start_verification),
            settlement_state=settlement_state,
            receipt_hash=_digest_from_cas_ref(receipt_ref),
            cell=cell,
            verified_gain=verified_gain,
            exploratory=exploratory,
        )
    )
    return True


def _formal_trial_cell(
    *, proposal: Mapping[str, Any], interventions: Sequence[Mapping[str, Any]]
) -> InterventionCell:
    if len(interventions) == 1:
        first = interventions[0]
        layer = first.get("layer")
        primitive = first.get("primitive")
        params = first.get("params")
        if (
            not isinstance(layer, str)
            or not isinstance(primitive, str)
            or not isinstance(params, Mapping)
        ):
            raise OrchestratorTrainingEvalSmokeError(
                "M3_TRAINING_EVAL_SMOKE_FORMAL_INTERVENTIONS_INVALID"
            )
        return InterventionCell(
            environment=str(proposal["env"]),
            layer=layer,
            primitive_family=primitive,
            parameter_bucket=json.dumps(
                params, sort_keys=True, separators=(",", ":"), ensure_ascii=True
            ),
        )
    return InterventionCell(
        environment=str(proposal["env"]),
        layer="combo",
        primitive_family="combo",
        parameter_bucket=json.dumps(
            list(interventions),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ),
    )


def _primary_delta(verdict: Mapping[str, Any]) -> float:
    deltas = verdict.get("delta_m_ver")
    if not isinstance(deltas, Mapping) or not deltas:
        raise OrchestratorTrainingEvalSmokeError(
            "M3_TRAINING_EVAL_SMOKE_FORMAL_DELTA_INVALID"
        )
    for value in deltas.values():
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        ):
            return float(value)
    raise OrchestratorTrainingEvalSmokeError(
        "M3_TRAINING_EVAL_SMOKE_FORMAL_DELTA_INVALID"
    )


def _impl_diff_hash(receipt: Mapping[str, Any]) -> str:
    candidate = receipt.get("candidate")
    if isinstance(candidate, Mapping):
        digest = candidate.get("worktree_diff_sha256")
        if isinstance(digest, str) and _is_sha256(digest):
            return digest
    diff_ref = receipt.get("candidate_diff_ref")
    if isinstance(diff_ref, str):
        return _digest_from_cas_ref(diff_ref)
    raise OrchestratorTrainingEvalSmokeError(
        "M3_TRAINING_EVAL_SMOKE_FORMAL_DIFF_HASH_INVALID"
    )


def _evaluator_hash(round_start: object) -> str:
    if isinstance(round_start, Mapping):
        evaluator = round_start.get("evaluator_freeze")
        if isinstance(evaluator, Mapping):
            digest = evaluator.get("sha256")
            if isinstance(digest, str) and _is_sha256(digest):
                return digest
    raise OrchestratorTrainingEvalSmokeError(
        "M3_TRAINING_EVAL_SMOKE_FORMAL_EVALUATOR_HASH_INVALID"
    )


def _digest_from_cas_ref(ref: str) -> str:
    prefix = "cas://sha256/"
    if not isinstance(ref, str) or not ref.startswith(prefix):
        raise OrchestratorTrainingEvalSmokeError(
            "M3_TRAINING_EVAL_SMOKE_FORMAL_CAS_REF_INVALID"
        )
    digest = ref[len(prefix) :]
    if not _is_sha256(digest):
        raise OrchestratorTrainingEvalSmokeError(
            "M3_TRAINING_EVAL_SMOKE_FORMAL_CAS_REF_INVALID"
        )
    return digest


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )
