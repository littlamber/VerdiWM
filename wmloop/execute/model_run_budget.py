"""Budget binding and crash-safe settlement for compiled model runs."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from wmloop.archive.store import ArchiveInvariantError, ContentAddressedStore
from wmloop.execute.budget import (
    BudgetError,
    BudgetLedger,
    BudgetPolicy,
    TrialAdmission,
)
from wmloop.storage import atomic_write, canonical_json_bytes


class ModelRunBudgetError(RuntimeError):
    """A model-run budget could not be admitted, recovered, or settled."""


_COST_CLASSES = {"very_low", "low", "medium", "high"}
_COST_CLASSES_ORDER = {"very_low": 0, "low": 1, "medium": 2, "high": 3}
_TERMINAL_STATES = {
    "completed",
    "failed",
    "cancelled",
    "timed_out",
    "failed_before_launch",
    "cancelled_before_launch",
}


def estimate_model_run_gpu_hours(
    stage_timeouts_seconds: Mapping[str, float], world_size: int
) -> float:
    """Conservatively reserve the sum of frozen stage timeouts per GPU."""

    if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size <= 0:
        raise ModelRunBudgetError("MODEL_RUN_BUDGET_WORLD_SIZE_INVALID")
    if not stage_timeouts_seconds:
        raise ModelRunBudgetError("MODEL_RUN_BUDGET_STAGE_TIMEOUTS_REQUIRED")
    seconds = 0.0
    for timeout in stage_timeouts_seconds.values():
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(float(timeout))
            or float(timeout) <= 0
        ):
            raise ModelRunBudgetError("MODEL_RUN_BUDGET_STAGE_TIMEOUTS_INVALID")
        seconds += float(timeout)
    estimated = seconds * world_size / 3600.0
    if not math.isfinite(estimated) or estimated <= 0:
        raise ModelRunBudgetError("MODEL_RUN_BUDGET_ESTIMATE_INVALID")
    return estimated


def validate_model_run_budget_limit(
    estimated_gpu_hours: object, total_gpu_hours: object
) -> None:
    """Reject a run whose frozen worst-case timeouts exceed its declared cap."""

    estimate = _finite_positive(
        estimated_gpu_hours, "MODEL_RUN_BUDGET_ESTIMATE_INVALID"
    )
    total = _finite_positive(total_gpu_hours, "MODEL_RUN_BUDGET_POLICY_INVALID")
    if estimate > total:
        raise ModelRunBudgetError(
            "MODEL_RUN_BUDGET_ESTIMATE_EXCEEDS_TOTAL:"
            f"estimated={estimate:g}:budget={total:g}"
        )


def validate_model_run_trial_limit(
    estimated_gpu_hours: object, max_trial_gpu_hours: object
) -> None:
    """Reject a run that cannot fit the per-trial cap frozen into its policy."""

    estimate = _finite_positive(
        estimated_gpu_hours, "MODEL_RUN_BUDGET_ESTIMATE_INVALID"
    )
    trial_cap = _finite_positive(
        max_trial_gpu_hours, "MODEL_RUN_BUDGET_POLICY_INVALID"
    )
    if estimate > trial_cap:
        raise ModelRunBudgetError(
            "MODEL_RUN_BUDGET_TRIAL_CAP_EXCEEDED:"
            f"estimated={estimate:g}:cap={trial_cap:g}"
        )


def compile_model_run_budget_binding(
    execution: Mapping[str, object],
    *,
    plan_document: Mapping[str, object],
    estimated_gpu_hours: float,
) -> dict[str, object]:
    """Freeze per-run ledger, CAS and admission policy into the manifest."""

    ledger_path = _required_path(execution.get("budget_db"), "BUDGET_LEDGER_PATH_REQUIRED")
    cas_root = _required_path(execution.get("cas_root"), "BUDGET_CAS_ROOT_REQUIRED")
    total = _finite_positive(
        execution.get("budget_total_gpu_hours"), "BUDGET_TOTAL_INVALID"
    )
    estimate = _finite_positive(estimated_gpu_hours, "BUDGET_ESTIMATE_INVALID")
    validate_model_run_budget_limit(estimate, total)
    max_trial = _finite_positive(
        execution.get("budget_max_trial_gpu_hours", 120.0),
        "BUDGET_TRIAL_CAP_INVALID",
    )
    validate_model_run_trial_limit(estimate, max_trial)
    high_trial_limit = execution.get("budget_high_trial_limit", 2)
    human_approved = execution.get("budget_human_approved", False)
    require_high_approval = execution.get("budget_require_high_cost_approval", True)
    if (
        isinstance(high_trial_limit, bool)
        or not isinstance(high_trial_limit, int)
        or high_trial_limit < 0
        or not isinstance(human_approved, bool)
        or not isinstance(require_high_approval, bool)
    ):
        raise ModelRunBudgetError("MODEL_RUN_BUDGET_POLICY_INVALID")
    cost_class = _cost_class(estimate)
    requested_cost_class = execution.get("budget_cost_class")
    if requested_cost_class is not None:
        if (
            not isinstance(requested_cost_class, str)
            or requested_cost_class not in _COST_CLASSES
            or _COST_CLASSES_ORDER[requested_cost_class]
            < _COST_CLASSES_ORDER[cost_class]
        ):
            raise ModelRunBudgetError("MODEL_RUN_BUDGET_COST_CLASS_INVALID")
        cost_class = requested_cost_class

    frozen_policy: dict[str, object] = {
        "total_gpu_hours": total,
        "high_trial_limit": high_trial_limit,
        "max_trial_gpu_hours": max_trial,
        "require_high_cost_approval": require_high_approval,
    }
    immutable_plan = {
        key: value
        for key, value in plan_document.items()
        if key not in {"model_run_id", "budget_binding"}
    }
    plan_sha256 = model_run_budget_plan_sha256(
        immutable_plan,
        ledger_path=ledger_path,
        cas_root=cas_root,
        estimate=estimate,
        cost_class=str(cost_class),
        human_approved=human_approved,
        policy=frozen_policy,
    )
    return {
        "ledger_path": ledger_path,
        "cas_root": cas_root,
        "policy": frozen_policy,
        "estimated_gpu_hours": estimate,
        "cost_class": str(cost_class),
        "human_approved": human_approved,
        "trial_id": "model-run-trial-" + plan_sha256[:24],
        "plan_sha256": plan_sha256,
    }


def model_run_budget_plan_sha256(
    plan_document: Mapping[str, object],
    *,
    ledger_path: str,
    cas_root: str,
    estimate: float,
    cost_class: str,
    human_approved: bool,
    policy: Mapping[str, object],
) -> str:
    payload = {
        "run_plan": plan_document,
        "ledger_path": ledger_path,
        "cas_root": cas_root,
        "estimated_gpu_hours": estimate,
        "cost_class": cost_class,
        "human_approved": human_approved,
        "policy": dict(policy),
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def validate_model_run_budget_binding(document: Mapping[str, object]) -> None:
    binding = document.get("budget_binding")
    if not isinstance(binding, Mapping):
        raise ModelRunBudgetError("MODEL_RUN_BUDGET_BINDING_REQUIRED")
    ledger_path = _required_path(binding.get("ledger_path"), "MODEL_RUN_BUDGET_LEDGER_PATH_INVALID")
    cas_root = _required_path(binding.get("cas_root"), "MODEL_RUN_BUDGET_CAS_ROOT_INVALID")
    policy = binding.get("policy")
    if not isinstance(policy, Mapping):
        raise ModelRunBudgetError("MODEL_RUN_BUDGET_POLICY_INVALID")
    total = _finite_positive(policy.get("total_gpu_hours"), "MODEL_RUN_BUDGET_POLICY_INVALID")
    max_trial = _finite_positive(policy.get("max_trial_gpu_hours"), "MODEL_RUN_BUDGET_POLICY_INVALID")
    high_trial_limit = policy.get("high_trial_limit")
    require_approval = policy.get("require_high_cost_approval")
    if (
        isinstance(high_trial_limit, bool)
        or not isinstance(high_trial_limit, int)
        or high_trial_limit < 0
        or not isinstance(require_approval, bool)
    ):
        raise ModelRunBudgetError("MODEL_RUN_BUDGET_POLICY_INVALID")
    estimate = _finite_positive(
        binding.get("estimated_gpu_hours"), "MODEL_RUN_BUDGET_ESTIMATE_INVALID"
    )
    cost_class = binding.get("cost_class")
    approved = binding.get("human_approved")
    if cost_class not in _COST_CLASSES or not isinstance(approved, bool):
        raise ModelRunBudgetError("MODEL_RUN_BUDGET_COST_CLASS_INVALID")
    validate_model_run_budget_limit(estimate, total)
    validate_model_run_trial_limit(estimate, max_trial)
    if _COST_CLASSES_ORDER[str(cost_class)] < _COST_CLASSES_ORDER[_cost_class(estimate)]:
        raise ModelRunBudgetError("MODEL_RUN_BUDGET_COST_CLASS_INVALID")
    plan_document = {key: value for key, value in document.items() if key not in {"model_run_id", "budget_binding"}}
    expected = model_run_budget_plan_sha256(
        plan_document,
        ledger_path=ledger_path,
        cas_root=cas_root,
        estimate=estimate,
        cost_class=str(cost_class),
        human_approved=approved,
        policy={
            "total_gpu_hours": total,
            "high_trial_limit": high_trial_limit,
            "max_trial_gpu_hours": max_trial,
            "require_high_cost_approval": require_approval,
        },
    )
    if binding.get("plan_sha256") != expected:
        raise ModelRunBudgetError("MODEL_RUN_BUDGET_PLAN_DIGEST_MISMATCH")
    if binding.get("trial_id") != "model-run-trial-" + expected[:24]:
        raise ModelRunBudgetError("MODEL_RUN_BUDGET_TRIAL_ID_MISMATCH")


@dataclass
class ModelRunBudget:
    binding: Mapping[str, object]
    ledger: BudgetLedger
    cas: ContentAddressedStore
    admission: TrialAdmission
    is_new: bool

    @classmethod
    def admit(cls, manifest: Mapping[str, object]) -> "ModelRunBudget":
        validate_model_run_budget_binding(manifest)
        binding = manifest["budget_binding"]
        policy_document = binding["policy"]
        policy = BudgetPolicy(
            total_gpu_hours=float(policy_document["total_gpu_hours"]),
            high_trial_limit=int(policy_document["high_trial_limit"]),
            max_trial_gpu_hours=float(policy_document["max_trial_gpu_hours"]),
            require_high_cost_approval=bool(policy_document["require_high_cost_approval"]),
        )
        ledger = BudgetLedger(Path(str(binding["ledger_path"])), policy)
        cas = ContentAddressedStore(Path(str(binding["cas_root"])))
        trial_id = str(binding["trial_id"])
        existing = ledger.get(trial_id)
        if existing is not None:
            _require_admission_binding(existing, binding)
            return cls(binding, ledger, cas, existing, False)
        try:
            admission = ledger.admit(
                trial_id,
                cost_class=str(binding["cost_class"]),
                estimated_gpu_hours=float(binding["estimated_gpu_hours"]),
                human_approved=bool(binding["human_approved"]),
                plan_sha256=str(binding["plan_sha256"]),
                reject_existing=True,
            )
        except BudgetError as exc:
            raise ModelRunBudgetError(f"MODEL_RUN_BUDGET_ADMISSION_FAILED:{exc}") from exc
        return cls(binding, ledger, cas, admission, True)

    def release_unlaunched(self) -> None:
        if self.admission.state != "admitted":
            raise ModelRunBudgetError("MODEL_RUN_BUDGET_NOT_ACTIVE")
        try:
            self.ledger.release(
                str(self.binding["trial_id"]),
                fencing_token=self.admission.fencing_token,
            )
        except BudgetError as exc:
            raise ModelRunBudgetError(f"MODEL_RUN_BUDGET_RELEASE_FAILED:{exc}") from exc

    def persist_terminal(self, receipt_path: Path, receipt: Mapping[str, object]) -> str:
        _validate_terminal_binding(receipt, self.binding, self.admission)
        payload = canonical_json_bytes(receipt)
        atomic_write(receipt_path, payload)
        try:
            ref = self.cas.put_bytes(payload, media_type="application/json")
            if receipt.get("budget_state") == "released":
                current = self.ledger.get(str(self.binding["trial_id"]))
                if current is not None:
                    if current.fencing_token != self.admission.fencing_token:
                        raise ModelRunBudgetError("MODEL_RUN_BUDGET_RELEASE_CONFLICT")
                    if current.state == "admitted":
                        self.ledger.release(
                            str(self.binding["trial_id"]),
                            fencing_token=self.admission.fencing_token,
                        )
                    elif current.state != "released":
                        raise ModelRunBudgetError("MODEL_RUN_BUDGET_RELEASE_CONFLICT")
                return ref.uri
            settled = self.ledger.settle(
                str(self.binding["trial_id"]),
                fencing_token=self.admission.fencing_token,
                actual_gpu_hours=float(receipt["allocated_gpu_hours"]),
                receipt_ref=ref.uri,
                plan_sha256=str(self.binding["plan_sha256"]),
            )
        except (ArchiveInvariantError, BudgetError, OSError) as exc:
            raise ModelRunBudgetError(f"MODEL_RUN_BUDGET_SETTLEMENT_INCOMPLETE:{exc}") from exc
        if settled.state != "settled" or settled.receipt_ref != ref.uri:
            raise ModelRunBudgetError("MODEL_RUN_BUDGET_SETTLEMENT_INCOMPLETE")
        return ref.uri

    def recover_terminal(self, receipt_path: Path) -> dict[str, object] | None:
        if not receipt_path.is_file():
            if self.is_new:
                return None
            raise ModelRunBudgetError("MODEL_RUN_BUDGET_RECOVERY_REQUIRED")
        receipt, payload = _read_terminal_receipt(receipt_path)
        _validate_terminal_binding(receipt, self.binding, self.admission)
        budget_state = receipt.get("budget_state")
        if budget_state == "released":
            if self.admission.state == "settled":
                raise ModelRunBudgetError("MODEL_RUN_BUDGET_RECEIPT_LEDGER_CONFLICT")
            try:
                ref = self.cas.put_bytes(payload, media_type="application/json")
                current = self.ledger.get(str(self.binding["trial_id"]))
                if current is not None:
                    if current.fencing_token != self.admission.fencing_token:
                        raise ModelRunBudgetError("MODEL_RUN_BUDGET_RECEIPT_LEDGER_CONFLICT")
                    if current.state == "admitted":
                        self.ledger.release(
                            str(self.binding["trial_id"]),
                            fencing_token=self.admission.fencing_token,
                        )
                    elif current.state != "released":
                        raise ModelRunBudgetError("MODEL_RUN_BUDGET_RECEIPT_LEDGER_CONFLICT")
            except (ArchiveInvariantError, BudgetError, OSError) as exc:
                raise ModelRunBudgetError(f"MODEL_RUN_BUDGET_RECOVERY_INCOMPLETE:{exc}") from exc
            receipt = dict(receipt)
            receipt["budget_receipt_ref"] = ref.uri
            return receipt
        if budget_state != "settled":
            raise ModelRunBudgetError("MODEL_RUN_BUDGET_RECOVERY_REQUIRED")
        actual_gpu_hours = _finite_nonnegative(
            receipt.get("allocated_gpu_hours"), "MODEL_RUN_BUDGET_RECEIPT_INVALID"
        )
        try:
            ref = self.cas.put_bytes(payload, media_type="application/json")
            if self.admission.state == "settled":
                if (
                    self.admission.receipt_ref != ref.uri
                    or self.admission.actual_gpu_hours != actual_gpu_hours
                ):
                    raise ModelRunBudgetError("MODEL_RUN_BUDGET_RECEIPT_LEDGER_CONFLICT")
            else:
                self.ledger.settle(
                    str(self.binding["trial_id"]),
                    fencing_token=self.admission.fencing_token,
                    actual_gpu_hours=actual_gpu_hours,
                    receipt_ref=ref.uri,
                    plan_sha256=str(self.binding["plan_sha256"]),
                )
        except (ArchiveInvariantError, BudgetError, OSError) as exc:
            raise ModelRunBudgetError(f"MODEL_RUN_BUDGET_RECOVERY_INCOMPLETE:{exc}") from exc
        receipt = dict(receipt)
        receipt["budget_receipt_ref"] = ref.uri
        return receipt


def _validate_terminal_binding(
    receipt: Mapping[str, object],
    binding: Mapping[str, object],
    admission: TrialAdmission,
) -> None:
    reservation = receipt.get("budget_reservation")
    if not isinstance(reservation, Mapping):
        raise ModelRunBudgetError("MODEL_RUN_BUDGET_RECEIPT_INVALID")
    if (
        receipt.get("budget_trial_id") != binding.get("trial_id")
        or reservation.get("trial_id") != binding.get("trial_id")
        or reservation.get("fencing_token") != admission.fencing_token
        or reservation.get("plan_sha256") != binding.get("plan_sha256")
        or receipt.get("estimated_gpu_hours") != binding.get("estimated_gpu_hours")
        or receipt.get("state") not in _TERMINAL_STATES
    ):
        raise ModelRunBudgetError("MODEL_RUN_BUDGET_RECEIPT_BINDING_MISMATCH")
    if receipt.get("budget_state") not in {"settled", "released"}:
        raise ModelRunBudgetError("MODEL_RUN_BUDGET_RECEIPT_INVALID")
    if receipt.get("budget_state") == "settled":
        _finite_nonnegative(receipt.get("allocated_gpu_hours"), "MODEL_RUN_BUDGET_RECEIPT_INVALID")
        gpu_count = receipt.get("allocated_gpu_count")
        if (
            isinstance(gpu_count, bool)
            or not isinstance(gpu_count, int)
            or gpu_count <= 0
            or receipt.get("launch_attempted") is not True
        ):
            raise ModelRunBudgetError("MODEL_RUN_BUDGET_RECEIPT_INVALID")
    claim_boundary = receipt.get("claim_boundary")
    if not isinstance(claim_boundary, str) or len(claim_boundary.strip()) < 12:
        raise ModelRunBudgetError("MODEL_RUN_BUDGET_RECEIPT_INVALID")
    if receipt.get("budget_state") == "released" and receipt.get("launch_attempted") is not False:
        raise ModelRunBudgetError("MODEL_RUN_BUDGET_RELEASE_AFTER_LAUNCH")


def _require_admission_binding(
    admission: TrialAdmission, binding: Mapping[str, object]
) -> None:
    if (
        admission.trial_id != binding.get("trial_id")
        or admission.plan_sha256 != binding.get("plan_sha256")
        or admission.cost_class != binding.get("cost_class")
        or admission.estimated_gpu_hours != binding.get("estimated_gpu_hours")
    ):
        raise ModelRunBudgetError("MODEL_RUN_BUDGET_LEDGER_BINDING_MISMATCH")


def _read_terminal_receipt(path: Path) -> tuple[dict[str, object], bytes]:
    try:
        payload = path.read_bytes()
        receipt = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelRunBudgetError("MODEL_RUN_BUDGET_RECEIPT_INVALID") from exc
    if not isinstance(receipt, dict) or canonical_json_bytes(receipt) != payload:
        raise ModelRunBudgetError("MODEL_RUN_BUDGET_RECEIPT_NOT_CANONICAL")
    return receipt, payload


def _required_path(value: object, code: str) -> str:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ModelRunBudgetError(code)
    return str(Path(value).expanduser().resolve())


def _finite_positive(value: object, code: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ModelRunBudgetError(code)
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ModelRunBudgetError(code)
    return result


def _finite_nonnegative(value: object, code: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ModelRunBudgetError(code)
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ModelRunBudgetError(code)
    return result


def _cost_class(estimated_gpu_hours: float) -> str:
    if estimated_gpu_hours <= 0.5:
        return "very_low"
    if estimated_gpu_hours <= 8.0:
        return "low"
    if estimated_gpu_hours <= 48.0:
        return "medium"
    return "high"
