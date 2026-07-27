"""The narrow, claim-licensed control path for one research round.

This module intentionally owns no provider implementation.  It coordinates
the bounded proposal language, the budget/fencing ledger and the independent
judge so a provider cannot observe a verdict before its execution receipt has
been durably settled.
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence

from wmloop.execute.budget import BudgetLedger, TrialAdmission
from wmloop.primitives.registry import PrimitiveRegistry
from wmloop.propose.generator import GeneratedProposal, ProposalContext, ProposalGenerator
from wmloop.verify.judge import JudgeResult, VerificationEvidence, judge


class ProposalExecutor(Protocol):
    """The execution-plane capability exposed to the control plane."""

    def execute(self, proposal: dict[str, object], fencing_token: int) -> "ExecutionOutcome":
        """Run one admitted proposal and return its measured receipt/evidence."""


@dataclass(frozen=True)
class ExecutionOutcome:
    """Execution-plane output that is still untrusted until settlement/judging."""

    actual_gpu_hours: float
    receipt_ref: str
    verification_evidence: VerificationEvidence


@dataclass(frozen=True)
class RoundResult:
    """Replayable summary of a single settled and judged research round."""

    vendor_revision: str
    round_start_verification: object | None
    generated_proposal: GeneratedProposal
    admission: TrialAdmission
    execution: ExecutionOutcome
    settlement: TrialAdmission
    verdict: JudgeResult


class ResearchLoop:
    """Coordinate one round without granting any provider claim authority.

    A vendor verifier is injected so tests can supply a frozen revision while
    production uses :func:`wmloop.vendor.verify_vendor_checkout`.  It runs
    before proposal generation and any executor side effect.
    """

    def __init__(
        self,
        *,
        proposal_generator: ProposalGenerator,
        budget_ledger: BudgetLedger,
        executor: ProposalExecutor,
        vendor_verifier: Callable[[], str],
        m4_launch_guard: Callable[[], object] | None = None,
        round_start_guard: Callable[[ProposalContext], object] | None = None,
        human_approved_high_cost: bool = False,
    ) -> None:
        self._proposal_generator = proposal_generator
        self._budget_ledger = budget_ledger
        self._executor = executor
        self._vendor_verifier = vendor_verifier
        self._m4_launch_guard = m4_launch_guard
        self._round_start_guard = round_start_guard
        self._human_approved_high_cost = human_approved_high_cost

    def run_round(self, context: ProposalContext) -> RoundResult:
        """Execute the only legal state ordering for a proposal round.

        Any exception propagates and therefore produces no verdict.  In
        particular, a provider timeout leaves the admission recoverable by the
        fencing/recovery layer rather than silently treating it as a result.
        """

        if self._m4_launch_guard is not None:
            self._m4_launch_guard()
        round_start_verification = None
        if self._round_start_guard is not None:
            round_start_verification = self._round_start_guard(context)
        vendor_revision = self._vendor_verifier()
        if not isinstance(vendor_revision, str) or not vendor_revision:
            raise RuntimeError("VENDOR_VERIFICATION_RESULT_INVALID")

        generated = self._proposal_generator.generate(context)
        proposal = generated.proposal
        trial_id = _proposal_id(proposal)
        cost_class = _combined_cost_class(context.registry, proposal)
        estimate = _budget_estimate(proposal)
        admission = self._budget_ledger.admit(
            trial_id,
            cost_class=cost_class,
            estimated_gpu_hours=estimate,
            human_approved=self._human_approved_high_cost,
        )
        if admission.state != "admitted":
            raise RuntimeError("ROUND_ALREADY_SETTLED")

        outcome = self._executor.execute(dict(proposal), admission.fencing_token)
        if outcome.verification_evidence.proposal_id != trial_id:
            raise RuntimeError("EXECUTION_EVIDENCE_PROPOSAL_MISMATCH")

        settlement = self._budget_ledger.settle(
            trial_id,
            fencing_token=admission.fencing_token,
            actual_gpu_hours=outcome.actual_gpu_hours,
            receipt_ref=outcome.receipt_ref,
        )
        # The judge only consumes evidence after a terminal receipt exists.
        verdict = judge(outcome.verification_evidence)
        return RoundResult(
            vendor_revision=vendor_revision,
            round_start_verification=round_start_verification,
            generated_proposal=generated,
            admission=admission,
            execution=outcome,
            settlement=settlement,
            verdict=verdict,
        )


class RoundRunner(Protocol):
    def run_round(self, context: ProposalContext) -> RoundResult:
        """Execute one already-authorized round."""


@dataclass(frozen=True)
class CampaignResult:
    completed: tuple[RoundResult, ...]
    paused: bool
    remaining_contexts: tuple[ProposalContext, ...]


class CampaignOrchestrator:
    """Run different environments concurrently while preserving per-env seriality.

    This is deliberately a small dispatch policy rather than a second source
    of trial state.  `ResearchLoop` remains responsible for vendor checks,
    proposal admission, fenced settlement and independent judging.  Pausing
    prevents all future dispatches; currently running work is allowed to reach
    a receipt/settlement boundary instead of being reclassified by the control
    plane.
    """

    def __init__(
        self,
        runner: RoundRunner,
        *,
        parallel_slots: int,
        m4_launch_guard: Callable[[], object] | None = None,
    ) -> None:
        if parallel_slots < 1:
            raise ValueError("CAMPAIGN_PARALLEL_SLOTS_INVALID")
        self._runner = runner
        self._parallel_slots = parallel_slots
        self._m4_launch_guard = m4_launch_guard
        self._pause_requested = False

    def request_pause(self) -> None:
        self._pause_requested = True

    def run(self, contexts: Sequence[ProposalContext], *, pause_after_round: int | None = None) -> CampaignResult:
        return asyncio.run(self.run_async(contexts, pause_after_round=pause_after_round))

    async def run_async(
        self, contexts: Sequence[ProposalContext], *, pause_after_round: int | None = None
    ) -> CampaignResult:
        if self._m4_launch_guard is not None:
            self._m4_launch_guard()
        if pause_after_round is not None and pause_after_round < 1:
            raise ValueError("CAMPAIGN_PAUSE_ROUND_INVALID")
        pending: dict[str, deque[ProposalContext]] = {}
        for context in contexts:
            environment = context.failure_report.get("env")
            if not isinstance(environment, str) or not environment:
                raise ValueError("CAMPAIGN_CONTEXT_ENV_INVALID")
            pending.setdefault(environment, deque()).append(context)
        completed: list[RoundResult] = []
        running: dict[asyncio.Task[RoundResult], str] = {}
        while (pending or running) and not self._pause_requested:
            capacity = self._parallel_slots - len(running)
            if pause_after_round is not None:
                capacity = min(capacity, pause_after_round - len(completed) - len(running))
            if capacity > 0:
                for environment in sorted(tuple(pending)):
                    if capacity == 0:
                        break
                    if environment in running.values():
                        continue
                    context = pending[environment].popleft()
                    if not pending[environment]:
                        del pending[environment]
                    task = asyncio.create_task(asyncio.to_thread(self._runner.run_round, context))
                    running[task] = environment
                    capacity -= 1
            if not running:
                break
            finished, _ = await asyncio.wait(tuple(running), return_when=asyncio.FIRST_COMPLETED)
            for task in finished:
                del running[task]
                completed.append(task.result())
            if pause_after_round is not None and len(completed) >= pause_after_round:
                self._pause_requested = True
        if self._pause_requested and running:
            # Do not abandon a running side effect: collect terminal receipts
            # before returning the checkpoint to a human operator.
            for task in tuple(running):
                completed.append(await task)
        remaining = tuple(context for environment in sorted(pending) for context in pending[environment])
        return CampaignResult(completed=tuple(completed), paused=self._pause_requested, remaining_contexts=remaining)


def _proposal_id(proposal: Mapping[str, Any]) -> str:
    proposal_id = proposal.get("proposal_id")
    if not isinstance(proposal_id, str) or not proposal_id:
        raise RuntimeError("PROPOSAL_ID_INVALID")
    return proposal_id


def _budget_estimate(proposal: Mapping[str, Any]) -> float:
    value = proposal.get("budget_estimate_gpu_hours")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise RuntimeError("PROPOSAL_BUDGET_INVALID")
    return float(value)


def _combined_cost_class(registry: PrimitiveRegistry, proposal: Mapping[str, Any]) -> str:
    interventions = proposal.get("interventions")
    if not isinstance(interventions, list) or not interventions:
        raise RuntimeError("PROPOSAL_INTERVENTIONS_INVALID")
    selections = registry.validate_combination(
        [
            (str(item.get("primitive")), item.get("params"))
            for item in interventions
            if isinstance(item, Mapping) and isinstance(item.get("params"), Mapping)
        ]
    )
    if len(selections) != len(interventions):
        raise RuntimeError("PROPOSAL_INTERVENTIONS_INVALID")
    ranks = {"very_low": 0, "low": 1, "medium": 2, "high": 3}
    primitive_cost_class = max(selections, key=lambda selection: ranks[selection.cost_class]).cost_class
    estimate_cost_class = _cost_class_for_estimate(_budget_estimate(proposal))
    return max((primitive_cost_class, estimate_cost_class), key=lambda cost_class: ranks[cost_class])


def _cost_class_for_estimate(estimated_gpu_hours: float) -> str:
    if estimated_gpu_hours <= 0.5:
        return "very_low"
    if estimated_gpu_hours <= 8.0:
        return "low"
    if estimated_gpu_hours <= 48.0:
        return "medium"
    return "high"
