"""Persistent, restartable campaign orchestration for model adapters."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Callable, Iterable

from .contracts import Evidence, canonical_digest
from .knowledge import KnowledgeGraph
from .storage import SQLiteState


STAGES = ("static_check", "environment_smoke", "gpu_smoke", "short_train", "replicate", "full_train", "heldout_evaluate")
TERMINAL_OUTCOMES = {"confirmed_positive", "positive", "null", "harmful", "abstain"}


@dataclass(frozen=True)
class CampaignPolicy:
    max_ideas: int = 4
    min_replicates: int = 2
    stop_on_first_positive: bool = True
    target_settled_ideas: int | None = None
    human_labels_required_for_model_claim: bool = False
    poll_seconds: float = 30.0
    auto_replan: bool = True
    max_research_rounds: int = 3


StageRunner = Callable[[dict[str, Any], str, dict[str, Any]], dict[str, Any]]
RepairRunner = Callable[[dict[str, Any], str, dict[str, Any], dict[str, Any]], dict[str, Any] | None]
Replanner = Callable[[dict[str, Any], dict[str, Any]], Iterable[dict[str, Any]]]


class CampaignSupervisor:
    """Drive one stage at a time and persist every transition before continuing."""

    def __init__(self, state_root: Path, *, model_id: str, stage_runner: StageRunner, policy: CampaignPolicy | None = None, repair_runner: RepairRunner | None = None, replanner: Replanner | None = None):
        self.state_root = Path(state_root)
        self.state = SQLiteState(self.state_root / "knowledge" / "knowledge.sqlite3")
        self.graph = KnowledgeGraph(self.state_root / "knowledge")
        self.model_id = model_id
        self.stage_runner = stage_runner
        self.policy = policy or CampaignPolicy()
        self.repair_runner = repair_runner
        self.replanner = replanner

    def create(self, *, run_id: str, objective: str, ideas: Iterable[dict[str, Any]], constraints: Iterable[str] = ()) -> dict[str, Any]:
        selected = list(ideas)[: self.policy.max_ideas]
        payload = {
            "run_id": run_id,
            "objective": objective,
            "constraints": list(constraints),
            "model_id": self.model_id,
            "policy": asdict(self.policy),
            "state": "running",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "ideas": {
                str(idea["idea_id"]): {
                    "idea": idea,
                    "state": "queued",
                    "stage_index": 0,
                    "attempts": [],
                    "settlement": None,
                }
                for idea in selected
                if idea.get("idea_id")
            },
            "events": [],
        }
        self._save(payload)
        return payload

    def step(self, run_id: str) -> dict[str, Any]:
        campaign = self.load(run_id)
        if campaign["state"] in {"settled", "stopped"}:
            return campaign
        changed = False
        for idea_id, item in campaign["ideas"].items():
            if item["state"] in {"settled", "stopped", "waiting_human"}:
                continue
            stage_index = int(item["stage_index"])
            if stage_index >= len(STAGES):
                item["state"] = "settled"
                changed = True
                continue
            stage = STAGES[stage_index]
            context = {"run_id": run_id, "model_id": self.model_id, "objective": campaign["objective"], "policy": campaign["policy"], "stage_index": stage_index, "attempt_count": len(item["attempts"])}
            try:
                result = self.stage_runner(item["idea"], stage, context)
            except Exception as error:
                result = {"state": "runtime_failed", "error": f"{type(error).__name__}: {error}"}
            # A model-specific runner may fail because a hook or dependency is
            # missing. Let the engineering agent repair in an isolated scope,
            # then retry the same stage before marking the idea blocked.
            # External resource waits are intentionally different: code repair
            # cannot create an idle GPU or a missing dataset entitlement.
            repairable = result.get("blocker_type") != "resource_unavailable"
            # ``autonomous_campaign`` wraps the stage runner with
            # ``AutonomousStageRunner``, which already owns one repair attempt.
            # Do not invoke the supervisor-level hook a second time for the
            # same failure; duplicate provider calls used to stall persistence
            # for several minutes per idea.
            if repairable and result.get("state") in {"runtime_failed", "blocked", "requires_code_patch"} and self.repair_runner is not None and not context.get("repair_attempted") and not isinstance(result.get("engineering"), dict):
                repaired = self.repair_runner(item["idea"], stage, {**context, "repair_attempted": True}, result)
                if isinstance(repaired, dict):
                    result = repaired
            event = {"idea_id": idea_id, "stage": stage, "result": result, "recorded_at": datetime.now(timezone.utc).isoformat()}
            item["attempts"].append(event)
            campaign["events"].append(event)
            changed = True
            outcome = str(result.get("outcome", ""))
            if result.get("requires_human_labels"):
                item["state"] = "waiting_human"
                item["waiting_reason"] = str(result.get("reason", "human_labels_required"))
                # Preserve the exact request (including an optional batch
                # manifest) so release-time validation can be deterministic.
                item["human_request"] = result
            elif result.get("state") == "abstain" or (
                result.get("state") == "requires_code_patch"
                and isinstance(result.get("engineering"), dict)
                and result["engineering"].get("state") == "abstain"
            ):
                # A non-retryable abstention is a scientific result in its
                # own right. Settle it into Evidence so failed ideas remain
                # discoverable; transient resource blocks stay below.
                settled = {**result, "state": "completed", "outcome": "abstain", "reason": result.get("reason", "abstained")}
                event["result"] = settled
                self._settle_idea(campaign, idea_id, item, settled)
            elif result.get("state") in {"runtime_failed", "blocked", "requires_code_patch"}:
                item["state"] = str(result["state"])
                item["failure"] = result
            elif stage == "full_train" and (result.get("continue_long_train") or result.get("state") == "continue_long_train" or result.get("action") == "continue_long_train"):
                item["state"] = "running"
                item["long_train"] = result
            elif stage == "heldout_evaluate" and outcome in TERMINAL_OUTCOMES:
                independent_replicates = int(result.get("independent_replicates", result.get("replicates", 0)))
                if outcome in {"confirmed_positive", "positive"} and independent_replicates < int(campaign["policy"].get("min_replicates", self.policy.min_replicates)):
                    result = {**result, "outcome": "abstain", "reason": "requires_independent_replicates", "replicates": independent_replicates}
                    # Keep the durable event stream aligned with the result
                    # that was actually settled into the knowledge graph.
                    event["result"] = result
                self._settle_idea(campaign, idea_id, item, result)
            else:
                item["stage_index"] = stage_index + 1
                item["state"] = "running"
            # Persist after every idea stage so a process restart loses no transition.
            self._save(campaign)
            if campaign["state"] == "stopped":
                break
        if changed and campaign["state"] == "running":
            active = [item for item in campaign["ideas"].values() if item["state"] in {"queued", "running"}]
            waiting = [item for item in campaign["ideas"].values() if item["state"] == "waiting_human"]
            failed = [item for item in campaign["ideas"].values() if item["state"] in {"runtime_failed", "blocked"}]
            if not active:
                if waiting:
                    campaign["state"] = "waiting_human"
                elif failed:
                    campaign["state"] = "blocked"
                elif self._maybe_replan(campaign):
                    campaign["state"] = "running"
                else:
                    campaign["state"] = "settled"
            self._save(campaign)
        return campaign

    def _maybe_replan(self, campaign: dict[str, Any]) -> bool:
        """Ask the configured research system for a fresh idea batch.

        Replanning is opt-in by callback and bounded by policy.  A replicated
        positive always stops first, so this path only runs after a complete
        non-positive batch.
        """
        if not self.policy.auto_replan or self.replanner is None:
            return False
        round_index = int(campaign.get("research_round", 0))
        if round_index >= self.policy.max_research_rounds:
            return False
        settled = [item for item in campaign["ideas"].values() if item.get("state") == "settled"]
        if not settled or any(str(item.get("settlement", {}).get("outcome", "")) in {"confirmed_positive", "positive"} for item in settled):
            return False
        new_ideas = list(self.replanner(campaign, {"round": round_index + 1, "settled": settled}))
        accepted = [idea for idea in new_ideas if idea.get("idea_id") and str(idea["idea_id"]) not in campaign["ideas"]]
        if not accepted:
            return False
        campaign["research_round"] = round_index + 1
        for idea in accepted[: self.policy.max_ideas]:
            campaign["ideas"][str(idea["idea_id"])] = {"idea": idea, "state": "queued", "stage_index": 0, "attempts": [], "settlement": None}
        campaign.setdefault("events", []).append({"event": "auto_replan", "round": round_index + 1, "idea_ids": [str(idea["idea_id"]) for idea in accepted]})
        return True

    def run_until_blocked(self, run_id: str, *, max_steps: int | None = None) -> dict[str, Any]:
        steps = 0
        while max_steps is None or steps < max_steps:
            before = self.load(run_id)
            if before["state"] in {"settled", "stopped", "waiting_human", "blocked"}:
                return before
            after = self.step(run_id)
            steps += 1
            if after["state"] in {"settled", "stopped", "waiting_human", "blocked"}:
                return after
            if after == before:
                return after
        return self.load(run_id)

    def resume(self, run_id: str, *, max_steps: int | None = None) -> dict[str, Any]:
        campaign = self.load(run_id)
        if campaign["state"] == "waiting_human":
            # An idea-level wait must be explicitly released by a label ingestion call.
            waiting = [item for item in campaign["ideas"].values() if item["state"] == "waiting_human"]
            if waiting and not any(item.get("human_labels") for item in waiting):
                return campaign
            campaign["state"] = "running"
            self._save(campaign)
        elif campaign["state"] == "blocked":
            # A blocked campaign is recoverable after an external fix (for
            # example an automatically applied dependency patch). Re-enter
            # the scheduler so failed ideas can retry from their current
            # stage; terminal settled ideas remain untouched.
            recoverable = [
                item for item in campaign["ideas"].values()
                if item["state"] in {"runtime_failed", "blocked", "queued", "running"}
            ]
            if recoverable:
                campaign["state"] = "running"
                self._save(campaign)
        return self.run_until_blocked(run_id, max_steps=max_steps)

    def release_human(self, run_id: str, idea_id: str, labels: dict[str, Any]) -> dict[str, Any]:
        campaign = self.load(run_id)
        item = campaign["ideas"].get(idea_id)
        if item is None:
            raise KeyError(idea_id)
        if item["state"] != "waiting_human":
            raise ValueError("idea is not waiting for human labels")
        request = item.get("human_request", {})
        batch = request.get("batch") if isinstance(request, dict) else None
        label_values = labels.get("labels", labels) if isinstance(labels, dict) else labels
        if isinstance(batch, dict) and {"batch_id", "episode_ids", "video_paths", "split"}.issubset(batch):
            from .human_eval import HumanVideoBatch, evaluate_labels

            manifest = HumanVideoBatch(
                batch_id=str(batch["batch_id"]),
                episode_ids=tuple(str(value) for value in batch["episode_ids"]),
                video_paths=tuple(str(value) for value in batch["video_paths"]),
                split=str(batch["split"]),
            )
            evaluation = evaluate_labels(manifest, label_values)
            if evaluation.get("outcome") != "measured":
                raise ValueError(json.dumps(evaluation, sort_keys=True))
            item["human_evaluation"] = evaluation
        item["human_labels"] = label_values
        item["state"] = "running"
        campaign["state"] = "running"
        self._save(campaign)
        return campaign

    def load(self, run_id: str) -> dict[str, Any]:
        rows = [row for row in self.state.list_rows("runs", limit=100000) if row["run_id"] == run_id]
        if not rows:
            raise KeyError(run_id)
        return json.loads(rows[0]["payload_json"])

    def _settle_idea(self, campaign: dict[str, Any], idea_id: str, item: dict[str, Any], result: dict[str, Any]) -> None:
        outcome = str(result.get("outcome", "abstain"))
        item["state"] = "settled"
        item["settlement"] = result
        evidence_payload = {"run_id": campaign["run_id"], "idea_id": idea_id, "result": result}
        evidence = Evidence(
            evidence_id="evidence-" + canonical_digest(evidence_payload)[7:23],
            experiment_id=f"{campaign['run_id']}-{idea_id}",
            model_id=self.model_id,
            hypothesis_id=idea_id,
            outcome=outcome,
            delta=float(result.get("delta", result.get("mean_delta", 0.0))),
            protected_ok=bool(result.get("protected_ok", False)),
            verifier_digest=str(result.get("verifier_digest", canonical_digest({"stage": "heldout_evaluate", "split": result.get("split", "heldout")}))),
            claim_boundary=str(result.get("claim_boundary", "adapter-provided held-out evidence; task success requires domain evaluation.")),
            metric_direction=str(result.get("metric_direction", "maximize")),
            ci95=tuple(result["ci95"]) if isinstance(result.get("ci95"), (list, tuple)) and len(result["ci95"]) == 2 else None,
            split=str(result.get("split", "heldout")),
            artifact_digest=str(result.get("artifact_digest", "")),
        )
        item["evidence"] = self.graph.append(evidence)
        if outcome in {"confirmed_positive", "positive"} and bool(campaign.get("policy", {}).get("stop_on_first_positive", self.policy.stop_on_first_positive)):
            campaign["state"] = "stopped"
            campaign["stop_reason"] = "first_replicated_positive"
        target = campaign.get("policy", {}).get("target_settled_ideas")
        if campaign["state"] == "running" and target is not None:
            settled_count = sum(1 for value in campaign["ideas"].values() if value.get("state") == "settled")
            if settled_count >= int(target):
                campaign["state"] = "stopped"
                campaign["stop_reason"] = "target_settled_ideas"

    def _save(self, payload: dict[str, Any]) -> None:
        self.state._put("runs", "run_id", {"run_id": payload["run_id"], "created_at": payload.get("created_at", "runtime"), "objective": payload["objective"], "state": payload["state"], "payload_json": json.dumps(payload, sort_keys=True)})
